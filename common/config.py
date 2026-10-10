"""Configuration helpers and dataclasses shared by the ml-cars-recon scripts."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# Directorios de captura (runtime, excluidos de git)
ISAPI_SNAPS_DIR = "isapi_snaps"
CAPTURES_PREROLL_DIR = "captures_preroll"
CAPTURES_FALLBACK_DIR = "captures_fallback"


def env_value(*names: str, default: str = "") -> str:
    """Return the first non-empty environment variable among ``names``."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


@dataclass
class CameraConfig:
    """Configuración de cámara RTSP."""

    host: str
    user: str
    password: str
    rtsp_channel: str
    snapshot_channel: str
    width: int = 1280
    height: int = 720
    rtsp_transport: str = "udp"


@dataclass
class DetectionConfig:
    """Configuración de detección."""

    model_path: str
    confidence: float = 0.45
    iou_threshold: float = 0.5
    capture_cooldown: float = 0.8
    snapshot_cooldown: float = 1.0
    pre_roll_ms: int = 300
    device: str | None = None
    imgsz: int | None = None
    infer_every_n: int = 1
    half: bool = False


@dataclass
class ROIConfig:
    """Región de interés."""

    width_ratio: float = 0.50
    height_ratio: float = 0.70
    center_x_ratio: float = 0.50
    center_y_ratio: float = 0.40
