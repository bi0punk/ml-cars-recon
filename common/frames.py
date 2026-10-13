"""Grabbers de frames RTSP y selección de frame nítido (fachada).

La implementación vive en módulos de responsabilidad única:

- :mod:`common.grabbers` — captura/reconexión en hilo de fondo.
- :mod:`common.sharpness` — selección del frame más nítido.

Este módulo mantiene la API histórica (``FrameGrabberLatest``,
``FrameGrabberBuffer``, ``best_frame``, ...) para no romper a los consumidores.
"""

from common.grabbers import FrameGrabber, FrameGrabberBuffer, FrameGrabberLatest
from common.sharpness import _fast_sharpness, best_frame

__all__ = [
    "FrameGrabber",
    "FrameGrabberBuffer",
    "FrameGrabberLatest",
    "best_frame",
    "_fast_sharpness",
]
