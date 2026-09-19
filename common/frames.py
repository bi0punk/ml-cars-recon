"""Grabbers de frames RTSP en hilo de fondo.

Versión canónica (con reconexión, lock y evento ``ready``) extraída de
``captura_autos_con_carril.py`` y compartida por todos los scripts.
"""

import contextlib
import logging
import threading
import time
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameGrabberLatest:
    """Grabber de sub-stream: publica siempre el frame más reciente."""

    def __init__(
        self,
        rtsp_url: str,
        width: int | None = None,
        height: int | None = None,
        name: str = "sub",
        cap_factory=None,
        stale_frame_threshold: int = 5,
        reconnect_delay: float = 2.0,
        max_reconnect_delay: float = 60.0,
    ):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.name = name
        self.cap = None
        self.frame: np.ndarray | None = None
        self.ok = False
        self.stopped = False
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self._cap_factory = cap_factory or cv2.VideoCapture
        self.stale_frame_threshold = max(1, int(stale_frame_threshold))
        self.reconnect_delay = max(0.1, float(reconnect_delay))
        self.max_reconnect_delay = max(self.reconnect_delay, float(max_reconnect_delay))
        self.reconnect_count = 0

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        logger.info("[%s] Esperando conexión RTSP...", self.name)
        if self.ready.wait(timeout=10):
            logger.info("[%s] FrameGrabberLatest iniciado correctamente", self.name)
        else:
            logger.warning("[%s] Timeout esperando primer frame", self.name)

    def _open_stream(self) -> bool:
        if self.cap is not None:
            with contextlib.suppress(Exception):
                self.cap.release()

        logger.info("[%s] Conectando a: %s", self.name, self.rtsp_url)
        self.cap = self._cap_factory(self.rtsp_url, cv2.CAP_FFMPEG)

        with contextlib.suppress(Exception):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        is_opened = self.cap.isOpened()
        self.reconnect_count += 1
        if is_opened:
            logger.info("[%s] Conexión RTSP exitosa", self.name)
        else:
            logger.error("[%s] Falló conexión RTSP", self.name)
        return is_opened

    def _capture_loop(self) -> None:
        backoff = self.reconnect_delay
        first_frame_received = False
        stale_reads = 0

        while not self.stopped:
            if (self.cap is None or not self.cap.isOpened()) and not self._open_stream():
                time.sleep(backoff)
                backoff = min(backoff * 2, self.max_reconnect_delay)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.ok = False
                stale_reads += 1
                if stale_reads >= self.stale_frame_threshold:
                    logger.warning(
                        "[%s] Stream sin frames (%d intentos seguidos). Reconectando...",
                        self.name,
                        stale_reads,
                    )
                    self._open_stream()
                    stale_reads = 0
                    backoff = min(backoff * 2, self.max_reconnect_delay)
                else:
                    logger.debug("[%s] Frame inválido (%d), reintentando...", self.name, stale_reads)
                    time.sleep(0.05)
                continue

            stale_reads = 0
            backoff = self.reconnect_delay

            if frame.size == 0:
                continue

            if self.width and self.height:
                try:
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
                except Exception:
                    logger.error("[%s] Error redimensionando frame", self.name)
                    continue

            with self.lock:
                self.frame = frame.copy()
                self.ok = True

            if not first_frame_received:
                first_frame_received = True
                self.ready.set()
                logger.info("[%s] Primer frame recibido", self.name)

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Lee el último frame (retorna una copia)."""
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ok, self.frame.copy()

    def release(self) -> None:
        logger.info("[%s] Liberando recursos...", self.name)
        self.stopped = True
        with contextlib.suppress(Exception):
            self.thread.join(timeout=3)
        if self.cap:
            self.cap.release()
        logger.info("[%s] Liberado", self.name)


# Alias para los scripts que solo necesitaban "último frame".
FrameGrabber = FrameGrabberLatest


class FrameGrabberBuffer:
    """Grabber de main-stream con buffer circular para captura pre-roll."""

    def __init__(
        self,
        rtsp_url: str,
        max_seconds: float = 1.5,
        fps_hint: int = 25,
        width: int | None = None,
        height: int | None = None,
        name: str = "main",
        cap_factory=None,
        stale_frame_threshold: int = 5,
        reconnect_delay: float = 2.0,
        max_reconnect_delay: float = 60.0,
    ):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.name = name
        self.max_buffer_size = int(max_seconds * max(fps_hint, 1)) + 5
        self.cap = None
        self.buffer: deque = deque(maxlen=self.max_buffer_size)
        self.ok = False
        self.stopped = False
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self._cap_factory = cap_factory or cv2.VideoCapture
        self.stale_frame_threshold = max(1, int(stale_frame_threshold))
        self.reconnect_delay = max(0.1, float(reconnect_delay))
        self.max_reconnect_delay = max(self.reconnect_delay, float(max_reconnect_delay))
        self.reconnect_count = 0

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        logger.info("[%s] Esperando conexión...", self.name)
        if self.ready.wait(timeout=10):
            logger.info("[%s] Buffer iniciado (size=%s)", self.name, self.max_buffer_size)
        else:
            logger.warning("[%s] Timeout esperando conexión", self.name)

    def _open_stream(self) -> bool:
        if self.cap is not None:
            with contextlib.suppress(Exception):
                self.cap.release()

        logger.info("[%s] Conectando a: %s", self.name, self.rtsp_url)
        self.cap = self._cap_factory(self.rtsp_url, cv2.CAP_FFMPEG)

        with contextlib.suppress(Exception):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        is_opened = self.cap.isOpened()
        self.reconnect_count += 1
        if is_opened:
            logger.info("[%s] Conexión exitosa", self.name)
        else:
            logger.error("[%s] Falló conexión", self.name)
        return is_opened

    def _capture_loop(self) -> None:
        backoff = self.reconnect_delay
        first_frame_received = False
        stale_reads = 0

        while not self.stopped:
            if (self.cap is None or not self.cap.isOpened()) and not self._open_stream():
                time.sleep(backoff)
                backoff = min(backoff * 2, self.max_reconnect_delay)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                self.ok = False
                stale_reads += 1
                if stale_reads >= self.stale_frame_threshold:
                    logger.warning(
                        "[%s] Stream sin frames (%d intentos seguidos). Reconectando...",
                        self.name,
                        stale_reads,
                    )
                    self._open_stream()
                    stale_reads = 0
                    backoff = min(backoff * 2, self.max_reconnect_delay)
                else:
                    logger.debug("[%s] Frame inválido (%d), reintentando...", self.name, stale_reads)
                    time.sleep(0.01)
                continue

            stale_reads = 0
            backoff = self.reconnect_delay

            timestamp = time.time()

            if self.width and self.height:
                try:
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
                except Exception:
                    continue

            with self.lock:
                self.buffer.append((timestamp, frame.copy()))
                self.ok = True

            if not first_frame_received:
                first_frame_received = True
                self.ready.set()
                logger.info("[%s] Buffer recibiendo frames", self.name)

    def get_closest_frame(self, target_timestamp: float) -> np.ndarray | None:
        """Devuelve el frame cuyo timestamp está más cerca de ``target_timestamp``."""
        with self.lock:
            if not self.buffer:
                return None
            closest = min(self.buffer, key=lambda x: abs(x[0] - target_timestamp))
            return closest[1].copy()

    def get_frames_in_window(self, target_timestamp: float, window_ms: float) -> list[tuple[float, np.ndarray]]:
        """Devuelve ``(timestamp, frame)`` dentro de una ventana de tiempo alrededor del target."""
        window = window_ms / 1000.0
        with self.lock:
            return [
                (ts, frame.copy())
                for ts, frame in self.buffer
                if abs(ts - target_timestamp) <= window
            ]

    def get_buffer_info(self) -> dict:
        """Info del buffer."""
        with self.lock:
            if not self.buffer:
                return {"size": 0, "time_span": 0, "max_size": self.max_buffer_size}
            return {
                "size": len(self.buffer),
                "max_size": self.max_buffer_size,
                "time_span": self.buffer[-1][0] - self.buffer[0][0],
            }

    def release(self) -> None:
        logger.info("[%s] Liberando...", self.name)
        self.stopped = True
        with contextlib.suppress(Exception):
            self.thread.join(timeout=3)
        if self.cap:
            self.cap.release()
        logger.info("[%s] Liberado", self.name)