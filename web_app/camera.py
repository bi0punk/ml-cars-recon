"""Cámara de la web app: adquisición de frames + YOLO + snapshot ISAPI.

La adquisición de frames (con reconexión robusta) la maneja el grabber en su
propio hilo; ``DetectorCamera`` solo hace inferencia y anotación mediante
``VehicleDetector`` sobre el ROI, y dispara snapshots ISAPI en hilo aparte.
"""

import logging
import threading
import time
from typing import Any

import cv2

from common.detector import VehicleDetector
from common.frames import FrameGrabberLatest
from common.geometry import compute_roi
from common.isapi import save_isapi_snapshot as _save_isapi_snapshot
from web_app.config import Config

logger = logging.getLogger(__name__)


def load_model(model_path: str) -> Any:
    """Carga el modelo YOLO (import perezoso para no romper al importar)."""
    from ultralytics import YOLO

    return YOLO(model_path)


class DetectorCamera:
    """Captura frames vía ``FrameGrabberLatest`` y corre YOLO sobre el ROI."""

    def __init__(self, src: str, model, config: Config):
        self.src = src
        self.model = model
        self.config = config
        self.detector = (
            VehicleDetector(
                model,
                conf=config.yolo_conf,
                snapshot_cooldown=config.snapshot_cooldown,
                imgsz=config.imgsz,
                infer_every_n=config.infer_every_n,
            )
            if model is not None
            else None
        )
        self.grabber = FrameGrabberLatest(src, name="web")
        self.frame: Any = None
        self.lock = threading.Lock()
        self.running = True
        self.last_snap_ts = 0.0
        self.stats = {
            "frames_processed": 0,
            "detections": 0,
            "captures": 0,
            "start_time": time.time(),
        }
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _draw_roi(self, draw, roi_rect: tuple[int, int, int, int]) -> None:
        x0, y0, x1, y1 = roi_rect
        cx = draw.shape[1] // 2
        cv2.rectangle(draw, (x0, y0), (x1, y1), (255, 200, 0), 2)
        cv2.line(draw, (cx, y0), (cx, y1), (255, 255, 255), 1)

    def _draw_status(self, draw, roi_rect: tuple[int, int, int, int], detected: bool) -> None:
        height, width = draw.shape[:2]
        x0, y0, x1, y1 = roi_rect

        status_color = (0, 255, 0) if detected else (0, 0, 255)
        status_text = "VEHICULO DETECTADO" if detected else "SIN DETECCION"
        cv2.rectangle(draw, (0, 0), (width, 36), (0, 0, 0), -1)
        cv2.putText(draw, status_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(
            draw,
            f"ROI {x1 - x0}x{y1 - y0}",
            (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

    def _snapshot_in_background(self) -> None:
        _save_isapi_snapshot(
            self.config.isapi_host,
            self.config.isapi_user,
            self.config.isapi_password,
            folder=self.config.capture_dir,
            channel=self.config.snapshot_channel,
            timeout=3,
        )

    def _maybe_take_snapshot(self, trigger_snapshot: bool) -> None:
        if not trigger_snapshot or not self.config.isapi_password:
            return
        now = time.time()
        if (now - self.last_snap_ts) <= self.config.snapshot_cooldown:
            return
        self.last_snap_ts = now
        self.stats["captures"] += 1
        threading.Thread(target=self._snapshot_in_background, daemon=True).start()

    def _loop(self) -> None:
        while self.running:
            ok, frame = self.grabber.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            draw = frame.copy()
            height, width = draw.shape[:2]
            roi_rect = compute_roi(
                width,
                height,
                roi_w=self.config.roi_w_pct,
                roi_h=self.config.roi_h_pct,
                roi_cy=self.config.roi_cy_pct,
            )
            self._draw_roi(draw, roi_rect)

            if self.detector is not None:
                draw, detected, trigger_snapshot = self.detector.detect(draw, roi_rect)
            else:
                detected, trigger_snapshot = False, False
            if detected:
                self.stats["detections"] += 1
            self.stats["frames_processed"] += 1
            self._draw_status(draw, roi_rect, detected)
            self._maybe_take_snapshot(trigger_snapshot)

            with self.lock:
                self.frame = draw

    def read_jpeg(self) -> bytes | None:
        with self.lock:
            if self.frame is None:
                return None
            ok, buf = cv2.imencode(".jpg", self.frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None
        return buf.tobytes()

    def status(self) -> dict:
        runtime = max(time.time() - self.stats["start_time"], 0.001)
        return {
            "connected": self.grabber.ok,
            "has_model": self.model is not None,
            "reconnects": self.grabber.reconnect_count,
            "stats": {
                "frames_processed": self.stats["frames_processed"],
                "detections": self.stats["detections"],
                "captures": self.stats["captures"],
                "fps": round(self.stats["frames_processed"] / runtime, 1),
            },
        }

    def stop(self) -> None:
        self.running = False
        try:
            if self._thread.is_alive():
                self._thread.join(timeout=1.0)
        except Exception:
            pass
        self.grabber.release()
