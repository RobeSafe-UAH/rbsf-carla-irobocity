"""
agent_unsolved.py — CARLA + ROS 2 course exercises
===================================================
This is the SAME agent as `agent.py`, but with some pieces of code removed.
Wherever you see three dots `...` next to a `# TODO`, you have to fill it in.

Each exercise is delimited by:

    # >>> TODO Exercise N <<<
    ...
    # >>> end Exercise N <<<

Search the file for "TODO" to jump between them. The complete reference
solution is in `agent.py` (same folder) if you get stuck.

────────────────────────────────────────────────────────────────────────────
 EXERCISES (in the order they appear in the file)
────────────────────────────────────────────────────────────────────────────
 1. Configure the sensors (where they go, resolution, FOV...) -> SENSORS
 2. Connect to the CARLA simulator                            -> __init__
 3. Camera intrinsics: read params + focal length from FOV    -> _make_camera_info
 4. Create the ROS 2 sensor publishers                        -> _attach_sensors
 5. Decode a camera image (BGRA bytes -> BGR)                 -> _publish_camera
 6. Decode a LiDAR scan + fix the coordinate system           -> _publish_lidar
────────────────────────────────────────────────────────────────────────────

How to run (inside the container, with the CARLA server already running):
    ros2 run carla_agent agent_unsolved --traffic 25

Published topics (once everything is filled in):
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
# EXERCISE 1 — Configure the sensors
# ------------------------------------------------------------------------------
# Each sensor is described by a dictionary:
#   - "id"   : the topic name it will publish on
#   - "type" : the CARLA blueprint (camera, lidar, ...)
#   - x/y/z          : WHERE it is attached to the car, in metres
#   - roll/pitch/yaw : HOW it is oriented, in degrees
#   - plus a few sensor-specific settings (resolution, range, ...)
#
# Fill in every `...`. The id, type and orientation (roll/pitch/yaw) are given
# to you: the orientation aligns each sensor with the ROS convention (we will
# explain why on the slides). Suggested values are written next to each line.
# ──────────────────────────────────────────────────────────────────────────────
SENSORS = [
    {
        "id": "cam_front",
        "type": "sensor.camera.rgb",
        "x": ..., "y": ..., "z": ...,            # TODO: position on the car [m]
        "roll": -90, "pitch": 0, "yaw": -90,     # given: ROS camera "optical" frame
        "image_size_x": ..., "image_size_y": ...,  # TODO: image resolution [px]
        "fov": ...,                              # TODO: horizontal field of view [deg]
        "sensor_tick": ...,                      # TODO: seconds between frames
    },
    {
        "id": "lidar",
        "type": "sensor.lidar.ray_cast",
        "x": ..., "y": ..., "z": ...,            # TODO: position on the car [m]
        "roll": 0, "pitch": 0, "yaw": -90,       # given
        "range": ..., "channels": ...,           # TODO: max distance [m] and number of layers
        "rotation_frequency": ...,               # TODO: full turns per second
        "points_per_second": ...,                # TODO: channels x points_per_turn x rotation_frequency
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

        # ──────────────────────────────────────────────────────────────────
        # EXERCISE 2 — Connect to the CARLA simulator
        # ------------------------------------------------------------------
        # A CARLA *client* talks to the simulator over the network.
        # From the client you then get the *world*, which is the running
        # simulation you will spawn actors into.
        # ──────────────────────────────────────────────────────────────────
        # >>> TODO Exercise 2 <<<
        self.client = ...          # TODO 1: create the client
        ...                        # TODO 2: set its timeout to 10 seconds
        self.world = ...           # TODO 3: get the world from the client
        # >>> end Exercise 2 <<<

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
        """
        # ──────────────────────────────────────────────────────────────────
        # EXERCISE 3 — Camera intrinsics
        # ------------------------------------------------------------------
        # 3a) Read the image width, height and FOV from the `spec` dict.
        #     Use spec.get(key, default) so it works even if a key is absent.
        #
        # 3b) Focal length (in pixels) of an ideal pinhole camera.
        #
        # 3c) The principal point (cx, cy) is the image centre.
        # ──────────────────────────────────────────────────────────────────
        # >>> TODO Exercise 3 <<<
        w   = int(spec.get(..., ...))    # TODO 3a: image width
        h   = int(spec.get(..., ...))    # TODO 3a: image height
        fov = float(spec.get(..., ...))  # TODO 3a: horizontal field of view

        f = ...                          # TODO 3b: focal length in pixels
        cx, cy = ..., ...                # TODO 3c: principal point
        # >>> end Exercise 3 <<<

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

            # ──────────────────────────────────────────────────────────────
            # EXERCISE 4 — Create the ROS 2 sensor publishers
            # --------------------------------------------------------------
            # A ROS 2 node publishes data through a *publisher*:
            #
            #     pub = self.create_publisher(MsgType, "topic_name", queue)
            #
            # The CameraInfo publisher below is already written for you, use
            # it as a template.
            #   4a) camera -> message type Image,       topic = sensor_id
            #   4b) lidar  -> message type PointCloud2,  topic = sensor_id
            # Use a queue size of 10 for both.
            # ──────────────────────────────────────────────────────────────
            if "camera" in sensor_type:
                # >>> TODO Exercise 4a (camera) <<<
                pub = ...   # TODO: an Image publisher on topic `sensor_id`
                # >>> end Exercise 4a <<<

                pub_info = self.create_publisher(CameraInfo, f"{sensor_id}/camera_info", 10)
                cam_info = self._make_camera_info(spec)
                actor.listen(lambda data, p=pub, pi=pub_info, ci=cam_info, sid=sensor_id:
                             self._publish_camera(data, p, pi, ci, sid))
                self.get_logger().info(f"Intrinsics published on /{sensor_id}/camera_info")

            elif "lidar" in sensor_type:
                # >>> TODO Exercise 4b (lidar) <<<
                pub = ...   # TODO: a PointCloud2 publisher on topic `sensor_id`
                # >>> end Exercise 4b <<<
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

        # ──────────────────────────────────────────────────────────────────
        # EXERCISE 5 — Decode the camera image
        # ------------------------------------------------------------------
        # CARLA gives the image as a flat array of bytes, 4 values per pixel
        # in B, G, R, A order (A = alpha/transparency, we don't need it).
        # `np.frombuffer(...)` below already gives you that flat array.
        # Reshape it into a proper (H, W, channels) image and drop the alpha.
        # ──────────────────────────────────────────────────────────────────
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        # >>> TODO Exercise 5 <<<
        array = ...   # TODO: reshape to (H, W, 4) and drop the alpha channel
        # >>> end Exercise 5 <<<
        msg = self.bridge.cv2_to_imgmsg(array, encoding="bgr8")
        msg.header.stamp    = stamp
        msg.header.frame_id = frame_id
        pub.publish(msg)

        # CameraInfo with same timestamp as the image
        cam_info.header.stamp    = stamp
        cam_info.header.frame_id = frame_id
        pub_info.publish(cam_info)

    def _publish_lidar(self, data, pub, frame_id):
        # ──────────────────────────────────────────────────────────────────
        # EXERCISE 6 — Decode the LiDAR scan + fix the coordinate system
        # ------------------------------------------------------------------
        # The raw buffer contains float32 values, 4 per point:
        #     x, y, z, intensity  (repeated for every point in the scan)
        #
        # 6) Turn the bytes into a NumPy array with shape (N, 4).
        #     Make sure the result is writable (frombuffer returns a read-only
        #     view into the original buffer).
        # ──────────────────────────────────────────────────────────────────
        # >>> TODO Exercise 6 <<<
        points = ...   # TODO
        # >>> end Exercise 6 <<<
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
