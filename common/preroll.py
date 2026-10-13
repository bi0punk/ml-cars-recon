"""Núcleo compartido de captura PRE-ROLL.

Centraliza los ajustes, el filtro por carretera y las estrategias de guardado
del frame "un poco ANTES" del trigger que antes duplicaban
``capture_preroll.py`` y ``capture_preroll_lanes.py``.
"""

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from common.config import CAPTURES_FALLBACK_DIR, CAPTURES_PREROLL_DIR, ISAPI_SNAPS_DIR
from common.detector import BoxFilter
from common.frames import FrameGrabberBuffer, FrameGrabberLatest
from common.geometry import parse_polygon, point_inside_polygon
from common.isapi import save_isapi_snapshot
from common.sharpness import best_frame
from common.utils import save_jpeg

logger = logging.getLogger(__name__)


class CameraLike(Protocol):
    """Lo mínimo que la captura pre-roll necesita de una configuración de cámara."""

    host: str
    user: str
    password: str
    snapshot_channel: str


@dataclass(frozen=True)
class CaptureSettings:
    """Ajustes de captura de pre-roll (independientes del CLI)."""

    save_dir: str = CAPTURES_PREROLL_DIR
    pre_roll_ms: int = 300
    sharp_window_ms: int = 200
    sharp_min_frames: int = 1
    sharp_max_frames: int = 9
    fallback_isapi: bool = False


def road_filter_for(
    poly_str: str, width: int, height: int
) -> tuple[np.ndarray | None, BoxFilter | None]:
    """Convierte ``--road-poly`` en (polígono dibujable, box_filter).

    El ``box_filter`` descarta vehículos cuyo centro está fuera del polígono.
    Devuelve ``(None, None)`` si no hay polígono válido.
    """
    poly = parse_polygon(poly_str, width, height)
    if poly is None:
        return None, None

    def _inside_road(box: tuple[int, int, int, int], cls_id: int) -> bool:
        x1, y1, x2, y2 = box
        return point_inside_polygon(((x1 + x2) // 2, (y1 + y2) // 2), poly)

    return poly, _inside_road


def save_sharpest_frame(
    grab_main: FrameGrabberBuffer,
    now: float,
    settings: CaptureSettings,
    camera: CameraLike | None = None,
) -> bool:
    """Guarda el frame más nítido del buffer MAIN, con fallback ISAPI.

    Devuelve ``True`` si se guardó un frame del buffer MAIN.
    """
    target_ts = now - (settings.pre_roll_ms / 1000.0)
    frames = grab_main.get_frames_in_window(target_ts, settings.sharp_window_ms)
    best = best_frame(
        frames,
        target_ts,
        settings.sharp_window_ms,
        min_frames=settings.sharp_min_frames,
        max_frames=settings.sharp_max_frames,
    )
    if best is not None:
        save_jpeg(best, folder=settings.save_dir, prefix="preroll_best")
        return True
    if settings.fallback_isapi and camera is not None:
        logger.warning("Buffer MAIN sin frames suficientes. Fallback ISAPI...")
        save_isapi_snapshot(
            camera.host,
            camera.user,
            camera.password,
            folder=ISAPI_SNAPS_DIR,
            channel=camera.snapshot_channel,
            timeout=4,
        )
    else:
        logger.warning("Buffer MAIN sin frames suficientes.")
    return False


def save_closest_frame(
    grab_main: FrameGrabberBuffer,
    grab_sub: FrameGrabberLatest,
    target_ts: float,
    save_dir: str = CAPTURES_PREROLL_DIR,
    prefix: str = "main",
    fallback_dir: str = CAPTURES_FALLBACK_DIR,
) -> bool:
    """Guarda el frame MAIN más cercano al trigger; fallback al sub-stream.

    Devuelve ``True`` si se guardó un frame del buffer MAIN.
    """
    frame = grab_main.get_closest_frame(target_ts)
    if frame is not None:
        save_jpeg(frame, folder=save_dir, prefix=prefix)
        return True
    logger.warning("Buffer MAIN vacío; usando sub-stream como fallback")
    ok, fallback = grab_sub.read()
    if ok and fallback is not None:
        save_jpeg(fallback, folder=fallback_dir, prefix="sub")
    return False


class CaptureScheduler:
    """Cooldown + executor acotado para capturas asíncronas de pre-roll.

    Evita bloquear el bucle de inferencia: ``maybe_submit`` decide si la
    captura está en cooldown y, si no, la envía al executor.
    """

    def __init__(
        self,
        cooldown: float,
        max_workers: int = 2,
        thread_name_prefix: str = "preroll",
    ):
        self.cooldown = max(0.0, float(cooldown))
        self.last_ts = 0.0
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix=thread_name_prefix
        )

    def is_due(self, now: float | None = None) -> bool:
        """``True`` (y actualiza ``last_ts``) si el cooldown ya expiró."""
        now = time.time() if now is None else now
        if (now - self.last_ts) <= self.cooldown:
            return False
        self.last_ts = now
        return True

    def submit(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        return self._executor.submit(fn, *args, **kwargs)

    def maybe_submit(self, fn: Callable[..., Any], *args, **kwargs) -> bool:
        """Envía ``fn`` al executor solo si el cooldown lo permite."""
        if not self.is_due():
            return False
        self._executor.submit(fn, *args, **kwargs)
        return True

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
