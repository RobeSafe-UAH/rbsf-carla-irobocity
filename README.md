# rbsf-carla-irobocity

ROS 2 + CARLA simulation stack developed at RobeSafe (UAH) for the iRobocity summer course.
It spawns an autonomous ego vehicle in CARLA, attaches a front camera and LiDAR, and publishes
sensor data as standard ROS 2 topics alongside a full TF tree and a 3-D car model in RViz.

---

## Requirements

| Dependency | Version |
|---|---|
| Ubuntu | 22.04 |
| Docker | with NVIDIA Container Toolkit |
| NVIDIA GPU | driver compatible with CUDA 12.9 |
| CARLA | 0.9.15 (downloaded by `setup_carla.sh`) |

---

## Installation

### 1. Download CARLA

Run the setup script once to download and unpack CARLA 0.9.15 (and the additional maps):

```bash
bash setup_carla.sh
```

This creates the `CARLA/` directory at the repo root.

### 2. Build the Docker image

```bash
make build
```

The image is based on `nvidia/cuda:12.9.1-devel-ubuntu22.04` and bundles:
- ROS 2 Humble (desktop)
- `uv` for Python environment management
- All Python dependencies from `pyproject.toml` (installed automatically on first shell start via `entrypoint.sh`)

### 3. Run the container

```bash
make run
```

The repo root is mounted at `/workspace` inside the container.
On first start `entrypoint.sh` runs `uv sync` to create the `.venv` and prints GPU/CUDA info.

### 4. Open additional terminals

To attach a new shell to the already-running container (e.g. for RViz, monitoring, etc.):

```bash
make attach
```

---

## Usage

Inside the container all commands are run from `/workspace`.

### Convenience aliases (defined in the image)

| Alias | Expands to |
|---|---|
| `BUILD` | `colcon build && source install/setup.sh` |
| `RUN` | `BUILD` then `ros2 launch stack_launcher carla.launch.py` |
| `RVIZ` | `rviz2 -d /workspace/rviz_cfg.rviz` |

### Step-by-step

**Terminal 1 — start CARLA server in the HOST**

```bash
./CARLA/CarlaUE4.sh -windowed -ResX=1280 -ResY=720
```

**Terminal 2 — build and launch the ROS 2 stack**

```bash
BUILD
ros2 launch stack_launcher carla.launch.py
```

Optional argument to control the number of traffic vehicles (default 25):

```bash
ros2 launch stack_launcher carla.launch.py traffic:=25
```

**Terminal 3 — open RViz** (`make attach` first)

```bash
RVIZ
```

---

## ROS 2 packages

### `carla_agent`

Main CARLA ↔ ROS 2 bridge.

| Topic | Type | Description |
|---|---|---|
| `/cam_front` | `sensor_msgs/Image` | Front RGB camera (640×480, 10 Hz) |
| `/cam_front/camera_info` | `sensor_msgs/CameraInfo` | Camera intrinsics |
| `/lidar` | `sensor_msgs/PointCloud2` | 32-channel LiDAR (50 m range) |
| `/tf_static` | TF | `map → ego → cam_front / lidar` |

The URDF car model (box + cylinder primitives) is installed under
`share/carla_agent/urdf/car.urdf` and loaded by `robot_state_publisher`.

### `carla_perception`

Perception pipeline that subscribes to the sensor topics published by `carla_agent`.

### `stack_launcher`

Launch file that starts all nodes together:
- `carla_agent` — sensor bridge and ego vehicle
- `carla_perception` — perception pipeline
- `robot_state_publisher` — publishes the URDF car model to `/robot_description`

---

## RViz setup

1. Set **Fixed Frame** to `map`.
2. **Add → By display type → RobotModel** — set *Description Topic* to `/robot_description`.
3. **Add → By display type → Image** — set *Image Topic* to `/cam_front`.
4. **Add → By display type → PointCloud2** — set *Topic* to `/lidar`.

The pre-configured `rviz_cfg.rviz` at the repo root already has these displays saved.

---

## Repository layout

```
.
├── CARLA/                  # CARLA 0.9.15 binaries (created by setup_carla.sh)
├── deploy/
│   ├── Dockerfile          # Docker image definition
│   └── entrypoint.sh       # uv sync + env setup run on every shell start
├── src/
│   ├── carla_agent/        # CARLA ↔ ROS 2 bridge + URDF
│   ├── carla_perception/   # Perception pipeline
│   └── stack_launcher/     # Launch files
├── rviz_cfg.rviz           # Pre-configured RViz layout
├── setup_carla.sh          # One-time CARLA download script
├── Makefile                # build_image / run / attach targets
└── pyproject.toml          # Python dependencies (managed by uv)
```

---

## Authors

- Miguel Antunes
- Fabio Sánchez
- Santiago Montiel
- Rodrigo Gutiérrez
