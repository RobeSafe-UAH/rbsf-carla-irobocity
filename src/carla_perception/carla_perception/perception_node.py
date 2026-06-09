import torch
import numpy as np
import cv2
from scipy.spatial import KDTree

from ultralytics import YOLO

import rclpy
import rclpy.duration
import rclpy.time
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import read_points_numpy
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros


CAMERA_FRAME = "cam_front"
LIDAR_FRAME  = "lidar"
EGO_FRAME    = "ego"

# Predefined box dimensions per COCO class id (L × W × H in metres)
BOX_DIMS = {
    0: (0.6,  0.6,  1.8),   # person
    1: (1.8,  0.6,  1.2),   # bicycle
    2: (4.5,  2.0,  1.6),   # car
    3: (2.2,  0.9,  1.4),   # motorcycle
    5: (12.0, 2.9,  3.6),   # bus
    6: (20.0, 3.0,  3.8),   # train
    7: (8.0,  2.5,  3.2),   # truck
}
MIN_INSTANCE_POINTS = 5
SOR_K    = 10
SOR_NSTD = 2.0


def _sor_filter(pts: np.ndarray, k: int, n_std: float) -> np.ndarray:
    """Statistical Outlier Removal: drop points whose mean k-NN distance
    exceeds global_mean + n_std * global_std."""
    if len(pts) <= k:
        return pts
    distances, _ = KDTree(pts).query(pts, k=k + 1)
    mean_dists = distances[:, 1:].mean(axis=1)  # exclude self (col 0)
    thresh = mean_dists.mean() + n_std * mean_dists.std()
    return pts[mean_dists <= thresh]


class PerceptionNode(Node):

    def __init__(self):
        super().__init__("carla_perception")

        # Publishers
        self.segmentation_publisher = self.create_publisher(Image, "segmented_image", 1)
        self.masks_publisher = self.create_publisher(Image, "masks", 1)
        self.painted_cloud_publisher = self.create_publisher(
            PointCloud2, "painted_cloud", 1
        )
        self.projected_lidar_publisher = self.create_publisher(
            Image, "img_projected_lidar", 1
        )
        self.bbox_publisher = self.create_publisher(MarkerArray, "bounding_boxes", 1)

        # TF2 buffer/listener — filled automatically from /tf_static
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Block until intrinsics and extrinsics are published by the agent
        self.get_logger().info("Waiting for camera_info and TF transforms...")
        camera_info     = self._wait_for_camera_info()
        T_ego_cam       = self._wait_for_transform(EGO_FRAME, CAMERA_FRAME)
        T_ego_lidar     = self._wait_for_transform(EGO_FRAME, LIDAR_FRAME)

        # Calibration
        self.K         = np.array(camera_info.k).reshape(3, 3)
        self.img_w     = camera_info.width
        self.img_h     = camera_info.height
        self.lidar2cam = np.linalg.inv(T_ego_cam) @ T_ego_lidar

        self.get_logger().info(
            f"Init complete.\nK:\n{self.K}\nlidar2cam:\n{self.lidar2cam}"
        )

        # Detection model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.segmentation_model = YOLO("yolo26s-seg.pt").to(self.device)
        self.classes        = [0, 1, 2, 3, 5, 6, 7]
        self.conf_threshold = 0.4

        # Mask color map
        self.class_colors = {
            0: (0, 255, 0),                                   # person → green
            **{i: (0, 255, 255) for i in [1, 2, 3, 5, 6, 7]} # vehicles → cyan
        }

        # Subscribers
        self.lidar_subscriber = self.create_subscription(
            PointCloud2, "lidar", self.lidar_callback, 1
        )
        self.camera_subscriber = self.create_subscription(
            Image, "cam_front", self.camera_callback, 1
        )

        # Processing timer. Perception is triggered at 10 Hz.
        self.timer_period = 0.1
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info("PerceptionNode has been started.")

    # ── Init helpers ──────────────────────────────────────────────────────

    def _wait_for_camera_info(self) -> CameraInfo:
        future = rclpy.Future()

        def _cb(msg: CameraInfo):
            if not future.done():
                future.set_result(msg)

        sub = self.create_subscription(CameraInfo, "/cam_front/camera_info", _cb, 1)
        rclpy.spin_until_future_complete(self, future)
        self.destroy_subscription(sub)
        return future.result()

    def _wait_for_transform(self, parent: str, child: str) -> np.ndarray:
        tf_ready = self._tf_buffer.wait_for_transform_async(
            parent, child, rclpy.time.Time()
        )
        rclpy.spin_until_future_complete(self, tf_ready)
        ts = self._tf_buffer.lookup_transform(parent, child, rclpy.time.Time())
        return self._tf_to_matrix(ts)

    @staticmethod
    def _tf_to_matrix(ts) -> np.ndarray:
        t = ts.transform.translation
        q = ts.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
            [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
            [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = [t.x, t.y, t.z]
        return T

    # ── Callbacks ─────────────────────────────────────────────────────────

    def camera_callback(self, msg: Image):
        """Save the latest camera image message for processing in the timer callback."""
        self.last_camera_msg = msg
        self.get_logger().info('Received a new camera image message.')

    def lidar_callback(self, msg: PointCloud2):
        """Save the latest LiDAR message for processing in the timer callback."""
        self.last_lidar_msg = msg
        self.get_logger().info('Received a new LiDAR message.')

    def timer_callback(self):
        """Process the sensor data provided by the simulator.

        Steps:
          1. Run YOLO segmentation on the front camera image.
          2. Merge per-object masks into a single 2D class label image.
          3. Publish the segmented image and the merged mask.
          4. Project the LiDAR point cloud onto the image plane.
          5. Paint and publish each LiDAR point with the class label at its projected pixel.
          6. Save the corrsponding pcl for each instance
        """
        print('Timer callback triggered.')
        import time
        t1 = time.time()
        ros_time = self.get_clock().now().to_msg()
        
        if not hasattr(self, 'last_camera_msg') or not hasattr(self, 'last_lidar_msg'):
            self.get_logger().warning('Waiting for both camera and LiDAR messages...')
            return
        
        # Convert ROS Image to numpy array
        img = np.frombuffer(self.last_camera_msg.data, dtype=np.uint8).reshape(
            (self.last_camera_msg.height, self.last_camera_msg.width, -1)
        )
        # bgr to rgb
        img = img[:, :, ::-1]
        
        # Run segmentation model
        results = self.segmentation_model(
            img,
            classes=self.classes,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )[0]
        
        masks = results.masks.data.cpu().numpy() if results.masks is not None else None
        cls = results.boxes.cls.cpu().numpy() if results.boxes is not None else None
                
        # STEP 2: Merge per-object masks into a single binary mask at the
        # original image resolution. White (255) marks object pixels.
        merged_mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        if masks is not None and cls is not None:
            mask_h, mask_w = masks.shape[1], masks.shape[2]
            temp = np.zeros((mask_h, mask_w), dtype=np.uint8)
            for mask, class_id in zip(masks, cls):
                temp[mask > 0] = 255
            # Resize to the original image size using nearest-neighbour to
            # preserve binary mask values
            merged_mask = cv2.resize(
                temp, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST
            )

        # STEP 3: Publish the segmented image and the merged class mask
        msg = self._create_rgb_img_msg(results.plot(), ros_time)
        self.segmentation_publisher.publish(msg)

        msg = self._create_masks_msg(merged_mask, ros_time)
        self.masks_publisher.publish(msg)
        
        # STEP 4: Project LiDAR points onto the image plane
        lidar_points = read_points_numpy(self.last_lidar_msg, field_names=("x", "y", "z", "intensity"))
        pixel_coords, valid_indices = self.project_lidar_to_image(lidar_points)
        lidar_points = lidar_points[valid_indices]  # keep only points that project into the image

        # STEP 4: Draw projected LiDAR points onto the mask with a jet depth colormap
        if pixel_coords.shape[1] > 0:
            u = pixel_coords[0, :]
            v = pixel_coords[1, :]

            dist = np.sqrt(lidar_points[:, 0]**2 + lidar_points[:, 1]**2).astype(np.float32)
            d_min, d_max = dist.min(), dist.max()
            norm = ((dist - d_min) / (d_max - d_min + 1e-6) * 255).astype(np.uint8)
            # cv2.applyColorMap expects (N, 1, 1); returns (N, 1, 3) BGR
            point_colors = cv2.applyColorMap(norm[:, None, None], cv2.COLORMAP_JET)[:, 0, :]

            img_with_lidar = cv2.cvtColor(merged_mask, cv2.COLOR_GRAY2BGR)
            # Paint a cross-shaped 3-pixel neighbourhood (vectorised, no Python loop per point)
            for dy, dx in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                vv = np.clip(v + dy, 0, self.img_h - 1)
                uu = np.clip(u + dx, 0, self.img_w - 1)
                img_with_lidar[vv, uu] = point_colors

            # Convert BGR → RGB before publishing
            img_with_lidar = cv2.cvtColor(img_with_lidar, cv2.COLOR_BGR2RGB)
            msg = self._create_rgb_img_msg(img_with_lidar, ros_time)
            self.projected_lidar_publisher.publish(msg)
        
        # STEP 5: Gather the per-instance filtered point cloud
        self.instance_clouds = []
        if masks is not None and cls is not None and pixel_coords.shape[1] > 0:
            u_coords = pixel_coords[0, :]
            v_coords = pixel_coords[1, :]
            for mask, class_id in zip(masks, cls):
                instance_mask = cv2.resize(
                    (mask * 255).astype(np.uint8),
                    (self.img_w, self.img_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                in_instance = instance_mask[v_coords, u_coords] > 0
                self.instance_clouds.append({
                    'class_id': int(class_id),
                    'points': lidar_points[in_instance],  # (K, 4) x/y/z/intensity
                })

        # STEP 6: Publish the painted point cloud (1 = inside a detection, 0 = background)
        if pixel_coords.shape[1] > 0:
            u_coords = pixel_coords[0, :]
            v_coords = pixel_coords[1, :]
            occupied = (merged_mask[v_coords, u_coords] > 0).astype(np.uint32)
            msg = self._create_painted_cloud_msg(lidar_points, occupied, ros_time)
            self.painted_cloud_publisher.publish(msg)

        # STEP 7: Filter outliers per instance and fit a predefined bounding box
        marker_array = MarkerArray()
        for i, instance in enumerate(self.instance_clouds):
            pts = instance['points'][:, :3]
            if len(pts) < MIN_INSTANCE_POINTS:
                continue
            pts = _sor_filter(pts, SOR_K, SOR_NSTD)
            if len(pts) < MIN_INSTANCE_POINTS:
                continue
            centroid = pts.mean(axis=0)
            dims = BOX_DIMS.get(instance['class_id'], (2.0, 1.0, 1.5))
            marker_array.markers.append(
                self._make_box_marker(i, centroid, dims, ros_time,
                                      self.last_lidar_msg.header.frame_id)
            )
        self.bbox_publisher.publish(marker_array)

        t2 = time.time()
        self.get_logger().info(f'Perception processing took {t2 - t1:.2f} seconds.')

    # ── Helper functions ─────────────────────────────────────────────────────────

    def _make_box_marker(self, uid: int, centroid: np.ndarray, dims: tuple,
                         ros_time, frame_id: str) -> Marker:
        marker = Marker()
        marker.header.stamp    = ros_time
        marker.header.frame_id = frame_id
        marker.ns      = "bounding_boxes"
        marker.id      = uid
        marker.type    = Marker.LINE_LIST
        marker.action  = Marker.ADD
        marker.scale.x = 0.05
        marker.color   = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
        marker.lifetime = rclpy.duration.Duration(seconds=0.3).to_msg()

        cx, cy, cz = centroid
        dx, dy, dz = dims[0] / 2, dims[1] / 2, dims[2] / 2
        # 8 corners ordered by (sx, sy, sz) in {-1,+1}³
        corners = [
            Point(x=cx + sx * dx, y=cy + sy * dy, z=cz + sz * dz)
            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
        ]
        # 12 edges: 4 along z, 4 along y, 4 along x
        edges = [(0,1),(2,3),(4,5),(6,7),
                 (0,2),(1,3),(4,6),(5,7),
                 (0,4),(1,5),(2,6),(3,7)]
        for a, b in edges:
            marker.points += [corners[a], corners[b]]
        return marker

    def project_lidar_to_image(self, lidar_points: np.ndarray):
        """Project LiDAR points into the front camera image plane.

        Args:
            lidar_points: (N, 4) array with columns [x, y, z, intensity]
                          in the LiDAR coordinate frame.

        Returns:
            pixel_coords: (2, M) integer array of [u, v] pixel coordinates
                          for each valid projected point.
            valid_indices: (M,) indices into the original lidar_points array
                           that correspond to the returned pixels.
        """
        N = lidar_points.shape[0]

        # Build homogeneous coordinates: (N, 4)
        ones = np.ones((N, 1), dtype=np.float32)
        pts_homo = np.hstack((lidar_points[:, :3], ones))

        # Transform from LiDAR frame to camera frame: (4, N)
        pts_cam = self.lidar2cam @ pts_homo.T

        # Keep only points in front of the camera (positive depth)
        front = pts_cam[2, :] > 0
        pts_cam = pts_cam[:, front]
        indices = np.where(front)[0]

        # Apply intrinsic matrix to get image coordinates: (3, M)
        pts_img = self.K @ pts_cam[:3, :]
        pts_img /= pts_img[2, :]   # perspective division

        u = pts_img[0, :].astype(int)
        v = pts_img[1, :].astype(int)

        # Keep only points whose projection falls inside the image
        in_bounds = (u >= 0) & (u < self.img_w) & (v >= 0) & (v < self.img_h)

        return np.vstack((u[in_bounds], v[in_bounds])), indices[in_bounds]
    
    def _create_rgb_img_msg(self, segmented_img, ros_time) -> Image:
        """Create a ROS2 Image message.

        Args:
            segmented_img: (H, W, 3) uint8 RGB array.
            ros_time: ROS timestamp for the message header.
        """
        msg = Image()
        msg.header.stamp = ros_time
        if segmented_img is None:
            return msg
        height, width, channels = segmented_img.shape
        msg.height = height
        msg.width = width
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = width * channels
        msg.data = segmented_img.tobytes()
        return msg
    
    def _create_painted_cloud_msg(self, lidar_points: np.ndarray, labels: np.ndarray, ros_time) -> PointCloud2:
        """Create a painted PointCloud2 message with an extra label field.

        Args:
            lidar_points: (M, 4) float32 array with columns [x, y, z, intensity].
            labels: (M,) uint32 array — 1 if point falls inside a detected instance, 0 otherwise.
            ros_time: ROS timestamp for the message header.
        """
        fields = [
            PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='label',     offset=16, datatype=PointField.UINT32,  count=1),
        ]

        dtype = np.dtype([
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('intensity', '<f4'), ('label', '<u4'),
        ])
        arr = np.zeros(len(lidar_points), dtype=dtype)
        arr['x']         = lidar_points[:, 0]
        arr['y']         = lidar_points[:, 1]
        arr['z']         = lidar_points[:, 2]
        arr['intensity'] = lidar_points[:, 3]
        arr['label']     = labels.astype(np.uint32)

        msg = PointCloud2()
        msg.header.stamp = ros_time
        msg.header.frame_id = self.last_lidar_msg.header.frame_id
        msg.height = 1
        msg.width = len(lidar_points)
        msg.is_bigendian = False
        msg.is_dense = False
        msg.point_step = arr.itemsize
        msg.row_step = arr.nbytes
        msg.fields = fields
        msg.data = arr.tobytes()
        return msg
    
    def _create_masks_msg(self, mask, ros_time) -> Image:
        """Create a ROS2 Image message for the merged 2D class label mask.

        Args:
            mask: (H, W) uint8 array where each pixel contains a class label
                         (0 = background, 1+ = class IDs).
            ros_time: ROS timestamp for the message header.
        """
        msg = Image()
        msg.header.stamp = ros_time
        if mask is None:
            return msg

        height, width = mask.shape
        msg.height = height
        msg.width = width
        msg.encoding = "mono8"
        msg.is_bigendian = False
        msg.step = width
        msg.data = mask.tobytes()
        return msg

# ─────────────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
