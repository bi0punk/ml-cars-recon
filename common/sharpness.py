"""Métricas de nitidez y selección de frames para captura pre-roll."""

import cv2
import numpy as np

from common.utils import sharpness


def _fast_sharpness(frame: np.ndarray, max_side: int = 640) -> float:
    """Nitidez sobre una miniatura para evitar el coste del Laplacian en 1080p."""
    height, width = frame.shape[:2]
    if max(height, width) > max_side:
        scale = max_side / max(height, width)
        small = cv2.resize(
            frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
        )
    else:
        small = frame
    return sharpness(small)


def best_frame(
    buffer: list[tuple[float, np.ndarray]],
    target_timestamp: float,
    window_ms: float,
    min_frames: int = 1,
    max_frames: int = 9,
) -> np.ndarray | None:
    """Elige el frame más nítido dentro de una ventana temporal.

    - ``buffer``: lista ``(timestamp, frame)`` (ver ``get_frames_in_window``).
    - Se conservan los ``max_frames`` temporalmente más cercanos a
      ``target_timestamp`` y se puntúa su nitidez (downscale previo para
      reducir el coste del Laplacian).
    - Si hay menos de ``min_frames`` candidatos, retorna ``None``.
    """
    if not buffer:
        return None
    window = window_ms / 1000.0
    candidates = [(ts, frame) for ts, frame in buffer if abs(ts - target_timestamp) <= window]
    if not candidates:
        return None
    candidates.sort(key=lambda item: abs(item[0] - target_timestamp))
    candidates = candidates[:max_frames]
    if len(candidates) < min_frames:
        return None
    return max(candidates, key=lambda item: _fast_sharpness(item[1]))[1]
