"""
agent.py — Agente CARLA simple para curso
==========================================
Uso:
    python agent.py                # sin tráfico
    python agent.py --traffic 50   # con 50 vehículos de tráfico

Topics publicados:
    /cam_front                         sensor_msgs/Image
    /cam_front/camera_info             sensor_msgs/CameraInfo   ← intrínseca
    /lidar                             sensor_msgs/PointCloud2
    /gnss                              sensor_msgs/NavSatFix
    /tf_static                         Árbol TF: map→ego→sensores (extrínsecas)
"""

import argparse
import random
import math

import carla
import rclpy
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField, NavSatFix, CameraInfo
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURA AQUÍ TUS SENSORES
# ──────────────────────────────────────────────────────────────────────────────
SENSORS = [
    {
        "id": "cam_front",
        "type": "sensor.camera.rgb",
        "x": 2.0, "y": 0.0, "z": 1.4,
        "image_size_x": 640, "image_size_y": 480, "fov": 90,
        "sensor_tick": 0.1,  # 10 Hz (igual que fixed_delta_seconds para sincronía)
    },
    {
        "id": "lidar",
        "type": "sensor.lidar.ray_cast",
        "x": 0.0, "y": 0.0, "z": 2.5,
        "range": 50, "channels": 32,
        "rotation_frequency": 20,    # = 1 / fixed_delta_seconds → vuelta completa por tick
        "points_per_second": 320000, # channels × points_per_rotation × rotation_frequency
                                     # 32 × 500 × 20 = 320000 → 500 pts por canal por vuelta
    },
    {
        "id": "gnss",
        "type": "sensor.other.gnss",
        "x": 0.0, "y": 0.0, "z": 0.0,
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Nodo ROS 2
# ──────────────────────────────────────────────────────────────────────────────
class CarlaAgentNode(Node):

    def __init__(self, num_traffic: int = 0):
        super().__init__("carla_agent")
        self.bridge = CvBridge()
        self.traffic_actors = []

        # Conectar a CARLA
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        bp_lib = self.world.get_blueprint_library()

        # Modo síncrono
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 Hz
        self.world.apply_settings(settings)

        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)

        # Spawnar vehículo ego
        vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        spawn_point = self.world.get_map().get_spawn_points()[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.vehicle.set_autopilot(True)
        self.get_logger().info(f"Vehículo spawneado (id={self.vehicle.id}) con autopilot")

        # Spawnar tráfico
        if num_traffic > 0:
            self._spawn_traffic(bp_lib, num_traffic)

        # Spectator tercera persona
        self.spectator = self.world.get_spectator()
        self.create_timer(0.05, self._tick)

        # Árbol TF: map → ego → cada sensor  (extrínsecas)
        self._publish_tf_tree()

        # Sensores
        self.sensors = []
        self._attach_sensors(bp_lib)

    # ── Árbol TF (extrínsecas) ─────────────────────────────────────────────

    def _publish_tf_tree(self):
        """
        Extrínsecas como TF estático.
        map → ego : identidad (posición fija; para ego dinámico se usaría TF dinámico)
        ego → sensor : posición relativa definida en SENSORS
        """
        broadcaster = StaticTransformBroadcaster(self)
        transforms = [self._make_tf("map", "ego", 0, 0, 0, 0, 0, 0)]

        for spec in SENSORS:
            transforms.append(self._make_tf(
                "ego", spec["id"],
                spec.get("x", 0), spec.get("y", 0), spec.get("z", 0),
                spec.get("roll", 0), spec.get("pitch", 0), spec.get("yaw", 0),
            ))

        broadcaster.sendTransform(transforms)
        self.get_logger().info("Extrínsecas publicadas en /tf_static")

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

    # ── Intrínseca de cámara ───────────────────────────────────────────────

    def _make_camera_info(self, spec: dict) -> CameraInfo:
        """
        Calcula la matriz intrínseca K a partir de la resolución y el FOV
        del sensor y devuelve un mensaje CameraInfo listo para publicar.

        Para una cámara pinhole con FOV horizontal:
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

        # Matriz intrínseca 3×3 aplanada en fila (row-major)
        msg.k = [
            f,   0.0, cx,
            0.0, f,   cy,
            0.0, 0.0, 1.0,
        ]

        # Matriz de proyección 3×4 (sin distorsión, cámara mono)
        msg.p = [
            f,   0.0, cx,  0.0,
            0.0, f,   cy,  0.0,
            0.0, 0.0, 1.0, 0.0,
        ]

        # Sin distorsión (CARLA es pinhole ideal)
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Matriz de rectificación = identidad
        msg.r = [1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0]

        return msg

    # ── Tráfico ────────────────────────────────────────────────────────────

    def _spawn_traffic(self, bp_lib, num_traffic: int):
        vehicle_bps = bp_lib.filter("vehicle.*")
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

        self.get_logger().info(f"Tráfico: {spawned}/{num_to_spawn} vehículos spawneados")

    # ── Sensores ───────────────────────────────────────────────────────────

    def _attach_sensors(self, bp_lib):
        RESERVED = {"id", "type", "x", "y", "z", "roll", "pitch", "yaw"}

        for spec in SENSORS:
            bp = bp_lib.find(spec["type"])
            for k, v in spec.items():
                if k not in RESERVED:
                    bp.set_attribute(str(k), str(v))

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
                self.get_logger().info(f"Intrínseca publicada en /{sensor_id}/camera_info")

            elif "lidar" in sensor_type:
                pub = self.create_publisher(PointCloud2, sensor_id, 10)
                actor.listen(lambda data, p=pub, sid=sensor_id: self._publish_lidar(data, p, sid))

            elif "gnss" in sensor_type:
                pub = self.create_publisher(NavSatFix, sensor_id, 10)
                actor.listen(lambda data, p=pub, sid=sensor_id: self._publish_gnss(data, p, sid))

            self.sensors.append(actor)
            self.get_logger().info(f"Sensor '{sensor_id}' adjuntado → /{sensor_id}")

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

    # ── Callbacks de publicación ───────────────────────────────────────────

    def _publish_camera(self, image, pub, pub_info, cam_info, frame_id):
        stamp = self.get_clock().now().to_msg()

        # Imagen
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        msg = self.bridge.cv2_to_imgmsg(array, encoding="bgr8")
        msg.header.stamp    = stamp
        msg.header.frame_id = frame_id
        pub.publish(msg)

        # CameraInfo con el mismo timestamp que la imagen
        cam_info.header.stamp    = stamp
        cam_info.header.frame_id = frame_id
        pub_info.publish(cam_info)

    def _publish_lidar(self, data, pub, frame_id):
        points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
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

    def _publish_gnss(self, data, pub, frame_id):
        msg = NavSatFix()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.latitude  = data.latitude
        msg.longitude = data.longitude
        msg.altitude  = data.altitude
        pub.publish(msg)

    # ── Limpieza ───────────────────────────────────────────────────────────

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
    parser = argparse.ArgumentParser(description="Agente CARLA con ROS 2")
    parser.add_argument("--traffic", type=int, default=0, metavar="N",
                        help="Número de vehículos de tráfico a generar (default: 0)")
    args = parser.parse_args()

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