"""Shared helpers: FFmpeg low-latency setup, directory handling, JPEG saving."""

import os
from datetime import datetime

import cv2

FFMPEG_LOW_LATENCY_OPTIONS = (
    "rtsp_transport;{transport}|"
    "max_delay;0|"
    "stimeout;5000000|"
    "buffer_size;0|"
    "fflags;nobuffer|"
    "flags;low_delay|"
    "reorder_queue_size;0"
)


def set_ffmpeg_low_latency_env(transport: str = "udp") -> str:
    """Configura OpenCV+FFmpeg para mínima latencia en RTSP. Retorna el string usado."""
    transport = transport.lower().strip()
    if transport not in ("udp", "tcp"):
        transport = "udp"
    options = FFMPEG_LOW_LATENCY_OPTIONS.format(transport=transport)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = options
    return options


def ensure_dir(path) -> None:
    """Crea el directorio (y padres) si no existe."""
    os.makedirs(path, exist_ok=True)


def save_jpeg(frame, folder="captures", prefix="frame", quality=95) -> str:
    """Guarda un frame como JPEG con calidad alta (útil para OCR)."""
    ensure_dir(folder)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    path = os.path.join(folder, f"{timestamp}_{prefix}.jpg")
    cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    print(f"[SAVE] {path}")
    return path


def sharpness(frame) -> float:
    """Varianza de Laplacian como métrica de nitidez del frame."""
    return cv2.Laplacian(frame, cv2.CV_64F).var()