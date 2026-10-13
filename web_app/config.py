"""Configuración de la web app: una única fuente de variables de entorno."""

import os
from dataclasses import dataclass

from common.config import ISAPI_SNAPS_DIR


@dataclass(frozen=True)
class Config:
    """Configuración de la web app (una única fuente de variables)."""

    rtsp_url: str = ""
    capture_dir: str = ISAPI_SNAPS_DIR
    latest_limit: int = 3

    model_path: str = "yolov8n.pt"
    yolo_conf: float = 0.45
    infer_every_n: int = 2
    imgsz: int = 640

    roi_w_pct: float = 0.60
    roi_h_pct: float = 0.50
    roi_cy_pct: float = 0.45

    isapi_host: str = "192.168.1.64"
    isapi_user: str = "admin"
    isapi_password: str = ""
    snapshot_channel: str = "101"
    snapshot_cooldown: float = 1.0

    api_token: str = ""


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def load_config() -> Config:
    return Config(
        rtsp_url=os.getenv("RTSP_URL", ""),
        capture_dir=os.getenv("CAPTURE_DIR", "isapi_snaps"),
        latest_limit=_env_int("LATEST_LIMIT", 3),
        model_path=os.getenv("MODEL_PATH", "yolov8n.pt"),
        yolo_conf=_env_float("YOLO_CONF", 0.45),
        infer_every_n=max(1, _env_int("INFER_EVERY_N", 2)),
        imgsz=_env_int("IMG_SIZE", 640),
        roi_w_pct=_env_float("ROI_W_PCT", 0.60),
        roi_h_pct=_env_float("ROI_H_PCT", 0.50),
        roi_cy_pct=_env_float("ROI_CY_PCT", 0.45),
        isapi_host=os.getenv("ISAPI_HOST", "192.168.1.64"),
        isapi_user=os.getenv("ISAPI_USER", "admin"),
        isapi_password=os.getenv("ISAPI_PASSWORD", ""),
        snapshot_channel=os.getenv("SNAPSHOT_CHANNEL", "101"),
        snapshot_cooldown=_env_float("SNAPSHOT_COOLDOWN", 1.0),
        api_token=os.getenv("API_TOKEN", ""),
    )
