"""Shared helpers: FFmpeg low-latency setup, directory handling, JPEG saving."""

import logging
import os
import uuid
from datetime import datetime

import cv2

logger = logging.getLogger(__name__)

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


def setup_logging(verbose: bool = False) -> None:
    """Configura logging de la app. ``verbose=True`` sube a nivel DEBUG."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_dir(path) -> None:
    """Crea el directorio (y padres) si no existe."""
    os.makedirs(path, exist_ok=True)


def save_jpeg(frame, folder="captures", prefix="frame", quality=95) -> str:
    """Guarda un frame como JPEG con calidad alta (útil para OCR).

    Incluye un sufijo aleatorio corto para evitar colisiones de nombre cuando
    dos capturas ocurren en el mismo milisegundo.
    """
    ensure_dir(folder)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    token = uuid.uuid4().hex[:6]
    path = os.path.join(folder, f"{timestamp}_{token}_{prefix}.jpg")
    cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    logger.info("Imagen guardada: %s", path)
    return path


def sharpness(frame) -> float:
    """Varianza de Laplacian como métrica de nitidez del frame."""
    return cv2.Laplacian(frame, cv2.CV_64F).var()


class StableTrigger:
    """Estabiliza el disparo: exige ``stable_frames`` frames positivos
    consecutivos y se rearma tras ``rearm_frames`` frames negativos.

    Evita capturas espurias por detecciones de un solo frame.
    """

    def __init__(self, stable_frames: int = 3, rearm_frames: int = 8):
        self.stable_frames = max(1, int(stable_frames))
        self.rearm_frames = max(0, int(rearm_frames))
        self._inside = 0
        self._outside = 0
        self.active = False

    def update(self, inside: bool) -> bool:
        """Alimenta con ``inside`` (vehículo dentro del ROI).

        Retorna ``True`` cuando se debe disparar.
        """
        if inside:
            self._inside += 1
            self._outside = 0
        else:
            self._inside = 0
            self._outside += 1
            if self._outside >= self.rearm_frames:
                self.active = False

        if self._inside >= self.stable_frames and not self.active:
            self.active = True
            return True
        return False

    def reset(self) -> None:
        self._inside = 0
        self._outside = 0
        self.active = False
