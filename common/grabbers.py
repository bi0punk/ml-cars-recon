"""Grabbers de frames RTSP en hilo de fondo.

`_BaseFrameGrabber` concentra la reconexión robusta, el control de frames
obsoletos y el ciclo de captura; las subclases solo definen cómo almacenan el
frame (`_store_frame`) y qué exponen al consumidor.
"""

import contextlib
import logging
import threading
import time
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class _BaseFrameGrabber:
    """Lógica común de captura/reconexión de los grabbers RTSP."""

    #: Mensajes de ciclo de vida (las subclases los ajustan si lo necesitan).
    _start_message = "Esperando conexión..."
    _ready_message = "Grabber iniciado correctamente"
    _timeout_message = "Timeout esperando conexión"

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
        ready_timeout: float = 10.0,
        invalid_sleep: float = 0.05,
        size_zero_is_invalid: bool = False,
    ):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.name = name
        self.cap: cv2.VideoCapture | None = None
        self.ok = False
        self.stopped = False
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self._cap_factory = cap_factory or cv2.VideoCapture
        self.stale_frame_threshold = max(1, int(stale_frame_threshold))
        self.reconnect_delay = max(0.1, float(reconnect_delay))
        self.max_reconnect_delay = max(self.reconnect_delay, float(max_reconnect_delay))
        self.reconnect_count = 0
        self._invalid_sleep = max(0.0, float(invalid_sleep))
        self._size_zero_is_invalid = size_zero_is_invalid

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        logger.info("[%s] %s", self.name, self._start_message)
        if self.ready.wait(timeout=ready_timeout):
            self._log_ready()
        else:
            self._log_timeout()

    def _log_ready(self) -> None:
        logger.info("[%s] %s", self.name, self._ready_message)

    def _log_timeout(self) -> None:
        logger.warning("[%s] %s", self.name, self._timeout_message)

    def _open_stream(self) -> bool:
        with contextlib.suppress(Exception):
            if self.cap is not None:
                self.cap.release()

        logger.info("[%s] Conectando a: %s", self.name, self.rtsp_url)
        cap = self._cap_factory(self.rtsp_url, cv2.CAP_FFMPEG)

        with contextlib.suppress(Exception):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        is_opened = cap.isOpened()
        self.cap = cap
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
            cap = self.cap
            if (cap is None or not cap.isOpened()) and not self._open_stream():
                time.sleep(backoff)
                backoff = min(backoff * 2, self.max_reconnect_delay)
                continue

            cap = self.cap
            if cap is None:
                time.sleep(backoff)
                continue

            ret, frame = cap.read()
            if not ret or frame is None or (self._size_zero_is_invalid and frame.size == 0):
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
                    logger.debug(
                        "[%s] Frame inválido (%d), reintentando...", self.name, stale_reads
                    )
                    time.sleep(self._invalid_sleep)
                continue

            stale_reads = 0
            backoff = self.reconnect_delay

            if not self._size_zero_is_invalid and frame.size == 0:
                continue

            timestamp = time.time()

            if self.width and self.height:
                try:
                    frame = cv2.resize(
                        frame, (self.width, self.height), interpolation=cv2.INTER_AREA
                    )
                except Exception:
                    logger.error("[%s] Error redimensionando frame", self.name)
                    continue

            with self.lock:
                self._store_frame(timestamp, frame.copy())
                self.ok = True

            if not first_frame_received:
                first_frame_received = True
                self.ready.set()
                logger.info("[%s] Primer frame recibido", self.name)

    def _store_frame(self, timestamp: float, frame: np.ndarray) -> None:
        raise NotImplementedError

    def release(self) -> None:
        logger.info("[%s] Liberando recursos...", self.name)
        self.stopped = True
        with contextlib.suppress(Exception):
            self.thread.join(timeout=3)
        if self.cap:
            self.cap.release()
        logger.info("[%s] Liberado", self.name)


class FrameGrabberLatest(_BaseFrameGrabber):
    """Grabber de sub-stream: publica siempre el frame más reciente."""

    _start_message = "Esperando conexión RTSP..."
    _ready_message = "FrameGrabberLatest iniciado correctamente"
    _timeout_message = "Timeout esperando primer frame"

    def __init__(self, *args, **kwargs):
        self.frame: np.ndarray | None = None
        kwargs.setdefault("invalid_sleep", 0.05)
        kwargs.setdefault("size_zero_is_invalid", False)
        super().__init__(*args, **kwargs)

    def _store_frame(self, timestamp: float, frame: np.ndarray) -> None:
        self.frame = frame

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Lee el último frame (retorna una copia)."""
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ok, self.frame.copy()


# Alias para los scripts que solo necesitaban "último frame".
FrameGrabber = FrameGrabberLatest


class FrameGrabberBuffer(_BaseFrameGrabber):
    """Grabber de main-stream con buffer circular para captura pre-roll."""

    _start_message = "Esperando conexión..."
    _ready_message = "Buffer iniciado"
    _timeout_message = "Timeout esperando conexión"

    def __init__(
        self,
        rtsp_url: str,
        max_seconds: float = 1.5,
        fps_hint: int = 25,
        *args,
        **kwargs,
    ):
        self.max_buffer_size = int(max_seconds * max(fps_hint, 1)) + 5
        self.buffer: deque = deque(maxlen=self.max_buffer_size)
        kwargs.setdefault("invalid_sleep", 0.01)
        kwargs.setdefault("size_zero_is_invalid", True)
        super().__init__(rtsp_url, *args, **kwargs)

    def _log_ready(self) -> None:
        logger.info("[%s] Buffer iniciado (size=%s)", self.name, self.max_buffer_size)

    def _store_frame(self, timestamp: float, frame: np.ndarray) -> None:
        self.buffer.append((timestamp, frame))

    def get_closest_frame(self, target_timestamp: float) -> np.ndarray | None:
        """Devuelve el frame cuyo timestamp está más cerca de ``target_timestamp``.

        READ-ONLY: retorna una referencia al frame almacenado (sin copia). No
        debe mutarse; los consumidores actuales solo lo leen o lo guardan.
        """
        with self.lock:
            if not self.buffer:
                return None
            closest = min(self.buffer, key=lambda x: abs(x[0] - target_timestamp))
            return closest[1]

    def get_frames_in_window(
        self, target_timestamp: float, window_ms: float
    ) -> list[tuple[float, np.ndarray]]:
        """Devuelve ``(timestamp, frame)`` en una ventana alrededor del target.

        READ-ONLY: devuelve referencias a los frames almacenados (sin copias).
        No deben mutarse.
        """
        window = window_ms / 1000.0
        with self.lock:
            return [
                (ts, frame) for ts, frame in self.buffer if abs(ts - target_timestamp) <= window
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
