"""
agent.py — Simple CARLA agent for course
=========================================
Usage:
    python agent.py                # no traffic
    python agent.py --traffic 50   # with 50 traffic vehicles

Published topics:
    /cam_front                         sensor_msgs/Image
    /cam_front/camera_info             sensor_msgs/CameraInfo   <- intrinsics
    /lidar                             sensor_msgs/PointCloud2
    /tf_static                         TF tree: map->ego->sensors (extrinsics)
"""

import argparse
import random
import math

import carla
import rclpy
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from cv_bridge import CvBridge


# ──────────────────────────────────────────────────────────────────────────────
# SENSOR CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
SENSORS = [
    {
        "id": "cam_front",
        "type": "sensor.camera.rgb",
        "x": 0.0, "y": 0.0, "z": 2.5,
        "roll": -90, "pitch": 0, "yaw": -90,
        "image_size_x": 640, "image_size_y": 480, "fov": 90,
        "sensor_tick": 0.1,  # 10 Hz (matches fixed_delta_seconds for sync)
    },
    {
        "id": "lidar",
        "type": "sensor.lidar.ray_cast",
        "x": 0.0, "y": 0.0, "z": 2.5,
        "roll": 0, "pitch": 0, "yaw": -90,
        "range": 50, "channels": 32,
        "rotation_frequency": 20,    # = 1 / fixed_delta_seconds -> full rotation per tick
        "points_per_second": 320000, # channels x points_per_rotation x rotation_frequency
                                     # 32 x 500 x 20 = 320000 -> 500 pts per channel per rotation
    }
]

# ──────────────────────────────────────────────────────────────────────────────
# ROS 2 Node
# ──────────────────────────────────────────────────────────────────────────────
class CarlaAgentNode(Node):

    def __init__(self, num_traffic: int = 0):
        super().__init__("carla_agent")
        self.bridge = CvBridge()
        self.traffic_actors = []

        # Connect to CARLA
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        bp_lib = self.world.get_blueprint_library()

        # Synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 Hz
        self.world.apply_settings(settings)

        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)

        # Spawn ego vehicle
        vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        spawn_point = self.world.get_map().get_spawn_points()[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.vehicle.set_autopilot(True)
        self.get_logger().info(f"Vehicle spawned (id={self.vehicle.id}) with autopilot")

        # Spawn traffic
        if num_traffic > 0:
            self._spawn_traffic(bp_lib, num_traffic)

        # Third-person spectator
        self.spectator = self.world.get_spectator()
        self.create_timer(0.05, self._tick)

        # TF tree: map -> ego -> each sensor (extrinsics)
        self._publish_tf_tree()

        # Sensors
        self.sensors = []
        self._attach_sensors(bp_lib)

    # ── TF tree (extrinsics) ───────────────────────────────────────────────

    def _publish_tf_tree(self):
        """
        Extrinsics as static TF.
        map -> ego : identity (fixed position; dynamic ego would use dynamic TF)
        ego -> sensor : relative position defined in SENSORS
        """
        broadcaster = StaticTransformBroadcaster(self)
        transforms = [self._make_tf("map", "ego", 0, 0, 0, 0, 0, 0)]

        for spec in SENSORS:
            transforms.append(self._make_tf(
                "ego", spec["id"],
                spec.get("x", 0),
                spec.get("y", 0),
                spec.get("z", 0),
                spec.get("roll",  0),
                spec.get("pitch", 0),
                spec.get("yaw",   0),
            ))

        broadcaster.sendTransform(transforms)
        self.get_logger().info("Extrinsics published on /tf_static")

    def _make_tf(self, parent, child, x, y, z, roll, pitch, yaw) -> TransformStamped:
        tf = TransformStamped()
        tf.header.stamp    = self.get_clock().now().to_msg()
        tf.header.frame_id = parent
        tf.child_frame_id  = child
        tf.transform.translation.x = float(x)
        tf.transform.translation.y = float(y)
        tf.transform.translation.z = float(z)
        cy, sy = math.cos(math.radians(yaw)   / 2), math.sin(math.radians(yaw)   / 2)
        cp, sp = math.cos(math.radians(pitch) / 2), math.sin(math.radians(pitch) / 2)
        cr, sr = math.cos(math.radians(roll)  / 2), math.sin(math.radians(roll)  / 2)
        tf.transform.rotation.w = cr * cp * cy + sr * sp * sy
        tf.transform.rotation.x = sr * cp * cy - cr * sp * sy
        tf.transform.rotation.y = cr * sp * cy + sr * cp * sy
        tf.transform.rotation.z = cr * cp * sy - sr * sp * cy
        return tf

    # ── Camera intrinsics ──────────────────────────────────────────────────

    def _make_camera_info(self, spec: dict) -> CameraInfo:
        """
        Computes the intrinsic matrix K from resolution and FOV and returns
        a CameraInfo message ready to publish.

        For a pinhole camera with horizontal FOV:
            fx = fy = (width / 2) / tan(fov_h / 2)
            cx = width  / 2
            cy = height / 2
        """
        w   = int(spec.get("image_size_x", 640))
        h   = int(spec.get("image_size_y", 480))
        fov = float(spec.get("fov", 90.0))

        f = (w / 2.0) / math.tan(math.radians(fov) / 2.0)
        cx, cy = w / 2.0, h / 2.0

        msg = CameraInfo()
        msg.header.frame_id = spec["id"]
        msg.width  = w
        msg.height = h

        # 3x3 intrinsic matrix flattened row-major
        msg.k = [
            f,   0.0, cx,
            0.0, f,   cy,
            0.0, 0.0, 1.0,
        ]

        # 3x4 projection matrix (no distortion, mono camera)
        msg.p = [
            f,   0.0, cx,  0.0,
            0.0, f,   cy,  0.0,
            0.0, 0.0, 1.0, 0.0,
        ]

        # No distortion (CARLA is an ideal pinhole)
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Rectification matrix = identity
        msg.r = [1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0]

        return msg

    # ── Traffic ────────────────────────────────────────────────────────────
    EXCLUDED_VEHICLES = {"vehicle.tesla.cybertruck"}
    def _spawn_traffic(self, bp_lib, num_traffic: int):
        vehicle_bps = [
            bp for bp in bp_lib.filter("vehicle.*")
            if int(bp.get_attribute("number_of_wheels")) == 4
            and bp.id not in self.EXCLUDED_VEHICLES
        ]
        spawn_points = self.world.get_map().get_spawn_points()[1:]
        random.shuffle(spawn_points)

        num_to_spawn = min(num_traffic, len(spawn_points))
        spawned = 0

        for sp in spawn_points[:num_to_spawn]:
            bp = random.choice(vehicle_bps)
            if bp.has_attribute("color"):
                bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
            actor = self.world.try_spawn_actor(bp, sp)
            if actor is not None:
                actor.set_autopilot(True, self.traffic_manager.get_port())
                self.traffic_actors.append(actor)
                spawned += 1

        self.get_logger().info(f"Traffic: {spawned}/{num_to_spawn} vehicles spawned")

    # ── Sensors ────────────────────────────────────────────────────────────

    def _attach_sensors(self, bp_lib):
        RESERVED = {"id", "type", "x", "y", "z", "roll", "pitch", "yaw"}

        for spec in SENSORS:
            bp = bp_lib.find(spec["type"])
            for k, v in spec.items():
                if k not in RESERVED:
                    bp.set_attribute(str(k), str(v))

            if "camera" in spec["type"]:
                transform = carla.Transform(
                    carla.Location(x=spec.get("x", 0), y=spec.get("y", 0), z=spec.get("z", 0)),
                )

            elif "lidar" in spec["type"]:
                transform = carla.Transform(
                    carla.Location(x=spec.get("x", 0), y=spec.get("y", 0), z=spec.get("z", 0)),
                    carla.Rotation(roll=spec.get("roll", 0), pitch=spec.get("pitch", 0), yaw=spec.get("yaw", 0)),
                )
            actor = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)

            sensor_type = spec["type"]
            sensor_id   = spec["id"]

            if "camera" in sensor_type:
                pub      = self.create_publisher(Image,      sensor_id,                10)
                pub_info = self.create_publisher(CameraInfo, f"{sensor_id}/camera_info", 10)
                cam_info = self._make_camera_info(spec)
                actor.listen(lambda data, p=pub, pi=pub_info, ci=cam_info, sid=sensor_id:
                             self._publish_camera(data, p, pi, ci, sid))
                self.get_logger().info(f"Intrinsics published on /{sensor_id}/camera_info")

            elif "lidar" in sensor_type:
                pub = self.create_publisher(PointCloud2, sensor_id, 10)
                actor.listen(lambda data, p=pub, sid=sensor_id: self._publish_lidar(data, p, sid))

            self.sensors.append(actor)
            self.get_logger().info(f"Sensor '{sensor_id}' attached -> /{sensor_id}")

    # ── Tick ───────────────────────────────────────────────────────────────

    def _tick(self):
        self.world.tick()
        self._update_spectator()

    def _update_spectator(self):
        t = self.vehicle.get_transform()
        self.spectator.set_transform(carla.Transform(
            t.transform(carla.Location(x=-8.0, z=4.5)),
            carla.Rotation(pitch=-15.0, yaw=t.rotation.yaw),
        ))

    # ── Publish callbacks ──────────────────────────────────────────────────

    def _publish_camera(self, image, pub, pub_info, cam_info, frame_id):
        stamp = self.get_clock().now().to_msg()

        # Image
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        msg = self.bridge.cv2_to_imgmsg(array, encoding="bgr8")
        msg.header.stamp    = stamp
        msg.header.frame_id = frame_id
        pub.publish(msg)

        # CameraInfo with same timestamp as the image
        cam_info.header.stamp    = stamp
        cam_info.header.frame_id = frame_id
        pub_info.publish(cam_info)

    def _publish_lidar(self, data, pub, frame_id):
        points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4).copy()
        points[:, 0] = -points[:, 0]  # CARLA left-handed -> ROS right-handed
        msg = PointCloud2()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.height = 1
        msg.width  = len(points)
        msg.fields = [
            PointField(name="x",         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y",         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z",         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step   = 16
        msg.row_step     = msg.point_step * len(points)
        msg.is_dense     = True
        msg.data         = points.tobytes()
        pub.publish(msg)

    # ── Cleanup ────────────────────────────────────────────────────────────

    def destroy_node(self):
        for s in self.sensors:
            s.stop(); s.destroy()
        self.vehicle.destroy()
        for a in self.traffic_actors:
            a.destroy()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        super().destroy_node()


# ──────────────────────────────────────────────────────────────────────────────
def main():
    import sys
    from rclpy.utilities import remove_ros_args

    parser = argparse.ArgumentParser(description="CARLA agent with ROS 2")
    parser.add_argument("--traffic", type=int, default=25, metavar="N",
                        help="Number of traffic vehicles to spawn (default: 25)")
    args = parser.parse_args(remove_ros_args(sys.argv[1:]))

    rclpy.init()
    node = CarlaAgentNode(num_traffic=args.traffic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
