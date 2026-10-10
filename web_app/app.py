#!/usr/bin/env python3

"""Interfaz web Flask para ml-cars-recon.

Sin efectos colaterales al importar: la cámara y el modelo YOLO se crean de
forma perezosa (``get_camera`` / ``_get_model``), por lo que importar este
módulo no arranca hilos de captura ni carga modelos pesados.
"""

import glob
import logging
import os
import threading
import time

import cv2
from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory

from common.detector import VehicleDetector
from common.frames import FrameGrabberLatest
from common.geometry import compute_roi
from common.isapi import save_isapi_snapshot as _save_isapi_snapshot
from common.utils import set_ffmpeg_low_latency_env

logger = logging.getLogger(__name__)


class Config:
    """Configuración de la web app (una única fuente de variables)."""

    rtsp_url = os.getenv("RTSP_URL", "")
    capture_dir = os.getenv("CAPTURE_DIR", "isapi_snaps")
    latest_limit = int(os.getenv("LATEST_LIMIT", "3"))

    model_path = os.getenv("MODEL_PATH", "yolov8n.pt")
    yolo_conf = float(os.getenv("YOLO_CONF", "0.45"))
    infer_every_n = max(1, int(os.getenv("INFER_EVERY_N", "2")))
    imgsz = int(os.getenv("IMG_SIZE", "640"))

    roi_w_pct = float(os.getenv("ROI_W_PCT", "0.60"))
    roi_h_pct = float(os.getenv("ROI_H_PCT", "0.50"))
    roi_cy_pct = float(os.getenv("ROI_CY_PCT", "0.45"))

    isapi_host = os.getenv("ISAPI_HOST", "192.168.1.64")
    isapi_user = os.getenv("ISAPI_USER", "admin")
    isapi_password = os.getenv("ISAPI_PASSWORD", "")
    snapshot_channel = os.getenv("SNAPSHOT_CHANNEL", "101")
    snapshot_cooldown = float(os.getenv("SNAPSHOT_COOLDOWN", "1.0"))

    api_token = os.getenv("API_TOKEN", "")


_camera = None
_model = None
_model_lock = threading.Lock()


def _get_model():
    """Carga el modelo YOLO una sola vez (perezoso)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    from ultralytics import YOLO

                    _model = YOLO(Config.model_path)
                    logger.info("Modelo YOLO cargado: %s", Config.model_path)
                except Exception as exc:
                    _model = None
                    logger.error("Error al cargar modelo YOLO: %s", exc)
    return _model


def require_auth():
    if not Config.api_token:
        return
    token = request.headers.get("X-API-Token")
    if not token or token != Config.api_token:
        abort(401, description="Unauthorized")


def list_latest_images(limit: int | None = None) -> list[str]:
    limit = Config.latest_limit if limit is None else max(0, int(limit))
    files = glob.glob(os.path.join(Config.capture_dir, "*.jpg"))
    files.sort(key=os.path.getmtime, reverse=True)
    return [os.path.basename(p) for p in files[:limit]]


def save_isapi_snapshot(folder: str | None = None, timeout: int = 3) -> str | None:
    """Descarga snapshot vía ISAPI con Digest (usa la config del módulo)."""
    return _save_isapi_snapshot(
        Config.isapi_host,
        Config.isapi_user,
        Config.isapi_password,
        folder=folder or Config.capture_dir,
        channel=Config.snapshot_channel,
        timeout=timeout,
    )


class DetectorCamera:
    """Captura frames vía ``FrameGrabberLatest`` y corre YOLO sobre el ROI.

    La adquisición de frames (con reconexión robusta) la maneja el grabber en
    su propio hilo; este hilo solo hace inferencia y anotación mediante
    ``VehicleDetector``.
    """

    def __init__(self, src: str, model):
        self.src = src
        self.model = model
        self.detector = VehicleDetector(
            model,
            conf=Config.yolo_conf,
            snapshot_cooldown=Config.snapshot_cooldown,
            imgsz=Config.imgsz,
            infer_every_n=Config.infer_every_n,
        )
        self.grabber = FrameGrabberLatest(src, name="web")
        self.frame = None
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

    def _loop(self) -> None:
        while self.running:
            ok, frame = self.grabber.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            draw = frame.copy()
            H, W = draw.shape[:2]
            roi_rect = compute_roi(
                W, H, roi_w=Config.roi_w_pct, roi_h=Config.roi_h_pct, roi_cy=Config.roi_cy_pct
            )
            x0, y0, x1, y1 = roi_rect
            cx = W // 2

            cv2.rectangle(draw, (x0, y0), (x1, y1), (255, 200, 0), 2)
            cv2.line(draw, (cx, y0), (cx, y1), (255, 255, 255), 1)

            draw, detected, trigger_snapshot = self.detector.detect(draw, roi_rect)
            if detected:
                self.stats["detections"] += 1
            self.stats["frames_processed"] += 1

            status_color = (0, 255, 0) if detected else (0, 0, 255)
            status_text = "VEHICULO DETECTADO" if detected else "SIN DETECCION"
            cv2.rectangle(draw, (0, 0), (W, 36), (0, 0, 0), -1)
            cv2.putText(draw, status_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(
                draw,
                f"ROI {x1 - x0}x{y1 - y0}",
                (10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            if trigger_snapshot and Config.isapi_password:
                now = time.time()
                if (now - self.last_snap_ts) > Config.snapshot_cooldown:
                    self.last_snap_ts = now
                    self.stats["captures"] += 1
                    threading.Thread(target=save_isapi_snapshot, daemon=True).start()

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


def get_camera() -> DetectorCamera:
    """Crea (y cachea) la cámara de forma perezosa."""
    global _camera
    if _camera is None:
        _camera = DetectorCamera(Config.rtsp_url, _get_model())
    return _camera


def reset_camera() -> None:
    """Detiene y elimina la cámara actual (útil en tests/shutdown)."""
    global _camera
    if _camera is not None:
        _camera.stop()
        _camera = None


def create_app() -> Flask:
    set_ffmpeg_low_latency_env("tcp")

    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        require_auth()
        camera = get_camera()

        def gen():
            boundary = b"--frame"
            while True:
                frame = camera.read_jpeg()
                if frame is None:
                    time.sleep(0.05)
                    continue
                yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                time.sleep(0.03)

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/latest_images")
    def api_latest_images():
        require_auth()
        imgs = list_latest_images()
        now = int(time.time())
        data = [{"name": f, "url": f"/captures/{f}?t={now}"} for f in imgs]
        return jsonify(data)

    @app.route("/api/status")
    def api_status():
        require_auth()
        if _camera is None:
            return jsonify({"active": False})
        return jsonify({"active": True, **get_camera().status()})

    @app.route("/captures/<path:filename>")
    def captures(filename):
        safe_dir = os.path.abspath(Config.capture_dir)
        requested = os.path.abspath(os.path.join(Config.capture_dir, filename))
        if not requested.startswith(safe_dir):
            return abort(403)
        if not os.path.exists(requested):
            return abort(404)
        return send_from_directory(Config.capture_dir, filename)

    @app.route("/_shutdown", methods=["POST"])
    def shutdown():
        require_auth()
        reset_camera()
        return "ok"

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=False,
        threaded=True,
    )
