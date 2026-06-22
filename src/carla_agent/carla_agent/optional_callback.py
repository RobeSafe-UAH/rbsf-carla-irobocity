import carla
import numpy as np
import cv2

# ── Depth Camera ──────────────────────────────────────────────────────────────

def depth_camera_callback(image):
    """
    Converts raw CARLA depth image to metric depth in meters.
    CARLA encodes depth in the R, G, B channels as:
        depth = (R + G*256 + B*256^2) / (256^3 - 1) * 1000  (meters)
    """
    # Convert raw image to a numpy array (BGRA)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA

    # Extract R, G, B channels
    B = array[:, :, 0].astype(np.float32)
    G = array[:, :, 1].astype(np.float32)
    R = array[:, :, 2].astype(np.float32)

    # Decode to depth in meters
    depth_meters = (R + G * 256.0 + B * 256.0 ** 2) / (256.0 ** 3 - 1) * 1000.0

    # Optional: normalize for visualization (0–255)
    depth_normalized = cv2.normalize(depth_meters, None, 0, 255, cv2.NORM_MINMAX)
    depth_vis = depth_normalized.astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

    cv2.imshow("Depth Camera", depth_colormap)
    cv2.waitKey(1)

    return depth_meters  # Return raw metric depth if needed downstream


# ── Semantic Segmentation Camera ─────────────────────────────────────────────

# CARLA's CityScapes palette (tag → BGR color)
SEGMENTATION_PALETTE = {
    0:  (0,   0,   0),    # Unlabeled
    1:  (70,  70,  70),   # Building
    2:  (100, 40,  40),   # Fence
    3:  (55,  90,  80),   # Other
    4:  (220, 20,  60),   # Pedestrian
    5:  (153, 153, 153),  # Pole
    6:  (157, 234, 50),   # RoadLine
    7:  (128, 64,  128),  # Road
    8:  (244, 35,  232),  # SideWalk
    9:  (107, 142, 35),   # Vegetation
    10: (0,   0,   142),  # Vehicle
    11: (102, 102, 156),  # Wall
    12: (220, 220, 0),    # TrafficSign
    13: (70,  130, 180),  # Sky
    14: (81,  0,   81),   # Ground
    15: (150, 100, 100),  # Bridge
    16: (230, 150, 140),  # RailTrack
    17: (180, 165, 180),  # GuardRail
    18: (250, 170, 30),   # TrafficLight
    19: (110, 190, 160),  # Static
    20: (170, 120, 50),   # Dynamic
    21: (45,  60,  150),  # Water
    22: (145, 170, 100),  # Terrain
}

def build_palette_lut():
    """Build a 256×3 lookup table for fast colorization."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for tag, color in SEGMENTATION_PALETTE.items():
        lut[tag] = color
    return lut

PALETTE_LUT = build_palette_lut()

def segmentation_camera_callback(image):
    """
    Converts raw CARLA semantic segmentation image.
    The semantic tag is stored in the R channel of each pixel.
    """
    # Convert raw image to numpy array (BGRA)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA

    # Semantic tag lives in the R channel
    tag_map = array[:, :, 2]  # R channel → class tag

    # Colorize using the lookup table
    colorized = PALETTE_LUT[tag_map]  # shape: (H, W, 3), BGR

    cv2.imshow("Segmentation Camera", colorized)
    cv2.waitKey(1)

    return tag_map  # Return raw tag map for downstream use (e.g. loss computation)