"""
perception_node.py — Perception node for the iRobocity course
==============================================================

Subscribed topics (inputs):
    /cam_front                   sensor_msgs/Image          Front RGB camera image
    /cam_front/camera_info       sensor_msgs/CameraInfo     Camera intrinsics (read once at startup)
    /lidar                       sensor_msgs/PointCloud2    LiDAR point cloud
    /tf_static                                              TF tree: map → ego → cam_front / lidar

Published topics (outputs):
    segmented_image              sensor_msgs/Image          Image with YOLO detections overlaid
    img_projected_lidar          sensor_msgs/Image          LiDAR projected onto the image with depth colormap
    painted_cloud                sensor_msgs/PointCloud2    Labelled point cloud (1 = detected object, 0 = background)
    object_centers               visualization_msgs/MarkerArray   3D centroids of each detected instance

Coordinate frames:
    cam_front  — Front camera frame
    lidar      — LiDAR frame
    ego        — Vehicle frame (root of the TF chain)
"""

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
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros


CAMERA_FRAME = "cam_front"
LIDAR_FRAME = "lidar"
EGO_FRAME = "ego"

MIN_INSTANCE_POINTS = 10
SOR_K = 10
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

# https://docs.ros2.org/latest/api/rclpy/api/node.html
class PerceptionNode(Node):
    def __init__(self):
        super().__init__("carla_perception")

        """ Exercise 1 """
        
        # TODO: Create subscribers for the camera and LiDAR topics.
        #       Use camera_callback and lidar_callback respectively.
        #       Queue size 1.
        # Subscribers
        self.lidar_subscriber = self.create_subscription(
            PointCloud2, "lidar", self.lidar_callback, 1
        )
        self.camera_subscriber = self.create_subscription(
            Image, "cam_front", self.camera_callback, 1
        )
        
        # TF2 buffer/listener — filled automatically from /tf_static
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # TODO: Wait for the first camera_info message and the TF transforms 
        # from camera and LiDAR frames to ego frame.
        # Block until intrinsics and extrinsics are published by the agent
        self.get_logger().info("Waiting for camera_info and TF transforms...")
        T_cam_ego = self._wait_for_transform(None, None)
        T_lidar_ego= self._wait_for_transform(None, None)
        # https://docs.ros2.org/latest/api/sensor_msgs/msg/CameraInfo.html
        camera_info = self._wait_for_camera_info()
        self.K = np.array(None).reshape(3, 3)
        self.img_w = None
        self.img_h = None
        self.lidar2cam = np.linalg.inv(T_cam_ego) @ T_lidar_ego

        self.get_logger().info(
            f"Init complete.\nK:\n{self.K}\nlidar2cam:\n{self.lidar2cam}"
        )

        """ Exercise 2 """
        # TODO: config perception timer
        # Processing timer. Perception is triggered at 10 Hz.
        self.timer_period = None
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # TODO: Create publishers for the segmented image, projected LiDAR image,
        # painted point cloud, and center markers.
        # Publishers
        self.segmentation_publisher = self.create_publisher(
            None, None, 1
        )
        self.painted_cloud_publisher = self.create_publisher(
            None, None, 1
        )
        self.projected_lidar_publisher = self.create_publisher(
            None, None, 1
        )
        self.centers_publisher = self.create_publisher(
            None, None, 1
        )

        # ── Other config ──────────────────────────────────────────────────

        # Detection model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.segmentation_model = YOLO("yolo26n-seg.pt").to(self.device)
        # https://gist.github.com/rcland12/dc48e1963268ff98c8b2c4543e7a9be8
        self.classes = [0, 1, 2, 3, 5, 6, 7]    # Pedestrians and vehicles
        self.conf_threshold = 0.4

        # Mask color map
        self.class_colors = {
            0: (0, 255, 0),  # person → green
            **{i: (0, 255, 255) for i in [1, 2, 3, 5, 6, 7]},  # vehicles → cyan
        }

        self.get_logger().info("PerceptionNode has been started.")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def camera_callback(self, msg: Image):
        """Save the latest camera image message for processing in the timer callback."""
        self.last_camera_msg = msg

    def lidar_callback(self, msg: PointCloud2):
        """Save the latest LiDAR message for processing in the timer callback."""
        self.last_lidar_msg = msg

    def timer_callback(self):
        """Perception pipeline running at 10 Hz.

        Steps:
          1. Run YOLO segmentation on the front camera image.
          2. Publish the segmented image.
          3. Project LiDAR points onto the image plane.
          4. Publish the painted point cloud (1 = inside detection, 0 = background).
          5. Assign pcl to each instance using the 2D masks.
          6. Extract the point cloud for each detected instance.
          7. Filter outliers per instance, estimate centroid, publish markers.
        """
        ros_time = self.get_clock().now().to_msg()

        # Check if we have received both camera and LiDAR messages at least once
        if not hasattr(self, "last_camera_msg") or not hasattr(self, "last_lidar_msg"):
            self.get_logger().warning("Waiting for both camera and LiDAR messages...")
            return

        # Convert ROS Image to numpy array (H, W, 3) in RGB
        img = np.frombuffer(self.last_camera_msg.data, dtype=np.uint8).reshape(
            (self.last_camera_msg.height, self.last_camera_msg.width, -1)
        )
        img = img[:, :, ::-1]  # BGR → RGB

        """ Exercise 3 """

        # STEP 1: Run the YOLO segmentation model
        # TODO: fill in the call to self.segmentation_model() with the correct arguments.
        results = self.segmentation_model(
            None,
            classes=None,
            conf=None,
            device=None,
            verbose=False,
        )[0]

        masks = results.masks.data.cpu().numpy() if results.masks is not None else None
        cls   = results.boxes.cls.cpu().numpy()  if results.boxes is not None else None

        # Merge per-object masks into a single binary mask at the
        # original image resolution. White (255) marks object pixels.
        merged_mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        if masks is not None and cls is not None:
            mask_h, mask_w = masks.shape[1], masks.shape[2]
            temp = np.zeros((mask_h, mask_w), dtype=np.uint8)
            for mask, class_id in zip(masks, cls):
                temp[mask > 0] = 255
            # Resize to the original image size using nearest-neighbour to
            # preserve binary mask values
            if temp.shape != (self.img_h, self.img_w):
                merged_mask = cv2.resize(
                    temp, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST
                )
            else:
                merged_mask = temp

        # STEP 2: Publish the segmented image and the merged class mask
        # TODO: complete the method _create_rgb_img_msg().
        msg = self._create_rgb_img_msg(results.plot(), ros_time)
        self.segmentation_publisher.publish(msg)
        
        """ Exercise 4 """

        # STEP 3: Project LiDAR points onto the image plane
        lidar_points = read_points_numpy(
            self.last_lidar_msg, field_names=("x", "y", "z", "intensity")
        )
        # TODO: complete the method project_lidar_to_image() to return pixel 
        # coordinates (2, N) and valid indices (N,).
        pixel_coords, valid_indices = self.project_lidar_to_image(lidar_points)
        lidar_points = lidar_points[
            valid_indices
        ]  # keep only points that project into the image

        # Draw projected LiDAR points with a jet depth colormap and publish.
        if pixel_coords.shape[1] > 0:
            u = pixel_coords[0, :]
            v = pixel_coords[1, :]

            dist = np.sqrt(lidar_points[:, 0] ** 2 + lidar_points[:, 1] ** 2).astype(
                np.float32
            )
            d_min, d_max = dist.min(), dist.max()
            norm = ((dist - d_min) / (d_max - d_min + 1e-6) * 255).astype(np.uint8)
            # cv2.applyColorMap expects (N, 1, 1); returns (N, 1, 3) BGR
            point_colors = cv2.applyColorMap(norm[:, None, None], cv2.COLORMAP_JET)[
                :, 0, :
            ]

            img_with_lidar = cv2.cvtColor(merged_mask, cv2.COLOR_GRAY2BGR)
            # Paint a cross-shaped 3-pixel neighbourhood (vectorised)
            for dy, dx in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                vv = np.clip(v + dy, 0, self.img_h - 1)
                uu = np.clip(u + dx, 0, self.img_w - 1)
                img_with_lidar[vv, uu] = point_colors
            
            # Convert BGR → RGB before publishing
            img_with_lidar = cv2.cvtColor(img_with_lidar, cv2.COLOR_BGR2RGB)

            # TODO: Build an Image message from img_with_lidar using 
            # self._create_rgb_img_msg() and publish it via self.projected_lidar_publisher.
            ...
            

        """ Exercise 5 """

        # STEP 4: Publish the painted point cloud (1 = inside a detection, 0 = background).
        if pixel_coords.shape[1] > 0:
            u_coords = pixel_coords[0, :]
            v_coords = pixel_coords[1, :]
            occupied = (merged_mask[v_coords, u_coords] > 0).astype(np.uint32)
            # TODO: Build a PointCloud2 message with self._create_painted_cloud_msg(
            #       lidar_points, occupied, ros_time) and publish via
            #       self.painted_cloud_publisher.
            msg = self._create_painted_cloud_msg(lidar_points, occupied, ros_time)
            self.painted_cloud_publisher.publish(msg)

        """ Exercise 6 """

        # STEP 5: Collect the point cloud for each detected instance.
        self.instance_clouds = []

        if masks is not None and cls is not None and pixel_coords.shape[1] > 0:
            # Column (u) and row (v) pixel coordinates of every LiDAR point that
            # successfully projected into the image (from STEP 3).
            u_projected = pixel_coords[0, :]
            v_projected = pixel_coords[1, :]

            for mask, class_id in zip(masks, cls):
                # YOLO produces masks at its internal resolution; scale up to the
                # full camera image size so pixel coordinates match.
                mask_full_res = cv2.resize(
                    (mask * 255).astype(np.uint8),
                    (self.img_w, self.img_h),
                    interpolation=cv2.INTER_NEAREST,
                )

                # TODO: Sample the mask at each projected LiDAR pixel location.
                # mask_full_res[v, u] returns 255 if that pixel belongs to this
                # instance and 0 if it belongs to the background.
                mask_values_at_lidar_pixels = None
                in_instance_mask = None  # boolean mask of shape (K,) where K = number of projected LiDAR points

                # Keep only the 3D points whose projection falls inside the mask.
                instance_points = lidar_points[in_instance_mask]  # shape (K, 4): x, y, z, intensity

                self.instance_clouds.append(
                    {
                        "class_id": int(class_id),
                        "points": instance_points,
                    }
                )

        # STEP 6: Filter outliers per instance and estimate each instance's centroid.
        marker_array = MarkerArray()
        for i, instance in enumerate(self.instance_clouds):
            pts = instance["points"][:, :3]
            if len(pts) < MIN_INSTANCE_POINTS:
                continue
            pts = _sor_filter(pts, SOR_K, SOR_NSTD)
            if len(pts) < MIN_INSTANCE_POINTS:
                continue
            centroid = pts.mean(axis=0)
            # TODO: Call self._make_centroid_marker(i, centroid, ros_time,
            #       self.last_lidar_msg.header.frame_id) and append the result
            #       to marker_array.markers.
            ...
            




        # TODO: Publish marker_array via self.bbox_publisher.
        ...
    
    # ── Helper functions ────────────────────────────────────────────────────────
    
    def _create_rgb_img_msg(self, img: np.ndarray, ros_time) -> Image:
        """Build a ROS2 Image message from a (H, W, 3) uint8 RGB array.
        Args:
            img: (H, W, 3) uint8 RGB array.
            ros_time: ROS timestamp for the message header.
        Image message fields (https://docs.ros.org/en/ros2_packages/humble/api/sensor_msgs/msg/Image.html):
            height       — number of pixel rows
            width        — number of pixel columns
            encoding     — pixel format for 3-channel uint8
            is_bigendian — byte order of the pixel data; False for little-endian
            step         — byte length of one row
            data         — raw pixel bytes; call .tobytes() on the numpy array
        """
        msg = Image()
        msg.header.stamp = ros_time
        if img is None:
            return msg
        height, width, channels = img.shape
        # TODO: Fill in the six message fields described in the docstring above.
        msg.height       = ...
        msg.width        = ...
        msg.encoding     = ...
        msg.is_bigendian = False
        msg.step         = ...
        msg.data         = img.tobytes()
        return msg

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
        # TODO: Implement the projection of LiDAR points into the camera image plane.
        N = lidar_points.shape[0]

        # Build homogeneous coordinates: (N, 4)
        ones = None
        pts_homog = None

        # Transform from LiDAR frame to camera frame (4, N)
        # Remember to use the self.lidar2cam matrix and pts_homog
        pts_cam = None

        # Keep only points in front of the camera (positive depth)
        front = pts_cam[2, :] > 0
        pts_cam = pts_cam[:, front]
        indices = np.where(front)[0]

        # Apply intrinsic matrix to get image coordinates: (3, M)
        pts_img = self.K @ pts_cam[:3, :]
        pts_img /= pts_img[2, :]  # perspective division

        u = pts_img[0, :].astype(int)
        v = pts_img[1, :].astype(int)

        # Keep only points whose projection falls inside the image (boolean mask)
        in_bounds = None

        return np.vstack((u[in_bounds], v[in_bounds])), indices[in_bounds]

    def _create_painted_cloud_msg(
        self, lidar_points: np.ndarray, labels: np.ndarray, ros_time
    ) -> PointCloud2:
        """Build a PointCloud2 message with an extra binary label field.
        https://docs.ros.org/en/ros2_packages/humble/api/sensor_msgs/msg/PointCloud2.html
        Args:
            lidar_points: (M, 4) float32 array with columns [x, y, z, intensity].
            labels:       (M,) uint32 array — 1 if the point falls inside a detected
                          instance, 0 if background.
            ros_time:     ROS timestamp for the message header.
        """
        fields = [
            PointField(name="x",         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y",         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z",         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="label",     offset=16, datatype=PointField.UINT32,  count=1),
        ]

        dtype = np.dtype(
            [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"), ("label", "<u4")]
        )
        arr = np.zeros(len(lidar_points), dtype=dtype)
        
        # TODO: Fill in the structured array with the LiDAR points and labels.
        arr["x"] = None
        arr["y"] = None
        arr["z"] = None
        arr["intensity"] = None
        arr["label"] = None

        # TODO: Create a PointCloud2() and fill in the fields below.
        msg = PointCloud2()
        msg.header.stamp    = ...
        msg.header.frame_id = self.last_lidar_msg.header.frame_id
        msg.height      = ...
        msg.width       = ...
        msg.is_bigendian = ...
        msg.is_dense    = ...
        msg.point_step  = arr.itemsize
        msg.row_step    = arr.nbytes
        msg.fields      = fields
        msg.data        = ...
        return msg

    def _make_centroid_marker(
        self, uid: int, centroid: np.ndarray, ros_time, frame_id: str
    ) -> Marker:
        # TODO: Create a Marker() and fill in all fields listed in the docstring.
        # https://docs.ros.org/en/humble/p/visualization_msgs/msg/Marker.html
        marker = Marker()
        marker.header.stamp    = ...
        marker.header.frame_id = ...
        marker.ns     = "centroids"
        marker.id     = ...
        marker.type   = ...
        marker.action = ...
        marker.scale.x = marker.scale.y = marker.scale.z = ...
        marker.color  = ...
        marker.lifetime = ...
        marker.pose.position.x  = ...
        marker.pose.position.y  = ...
        marker.pose.position.z  = ...
        marker.pose.orientation.w = ...
        return marker

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

    def _wait_for_transform(self, destination: str, source: str) -> np.ndarray:
        tf_ready = self._tf_buffer.wait_for_transform_async(
            destination, source, rclpy.time.Time()
        )
        rclpy.spin_until_future_complete(self, tf_ready)
        ts = self._tf_buffer.lookup_transform(destination, source, rclpy.time.Time())
        return self._tf_to_matrix(ts)

    @staticmethod
    def _tf_to_matrix(ts) -> np.ndarray:
        t = ts.transform.translation
        q = ts.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [t.x, t.y, t.z]
        return T

# ──────────────────────────────────────────────────────────────────────────────


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
