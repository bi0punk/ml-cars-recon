#!/usr/bin/env python3

"""Interfaz web Flask para ml-cars-recon.

Sin efectos colaterales al importar: la cámara y el modelo YOLO se crean de
forma perezosa (``get_camera``), por lo que importar este módulo no arranca
hilos de captura ni carga modelos pesados.
"""

import glob
import logging
import os
import threading
import time

import cv2
from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory

from common.frames import FrameGrabberLatest
from common.geometry import box_inside_roi, compute_roi
from common.isapi import save_isapi_snapshot as _save_isapi_snapshot
from common.utils import set_ffmpeg_low_latency_env

logger = logging.getLogger(__name__)

# ============================
# Configuración por variables
# ============================
RTSP_URL     = os.getenv("RTSP_URL", "")
CAPTURE_DIR  = os.getenv("CAPTURE_DIR", "isapi_snaps")
LATEST_LIMIT = int(os.getenv("LATEST_LIMIT", "3"))

# Detección
MODEL_PATH    = os.getenv("MODEL_PATH", "yolov8n.pt")
YOLO_CONF     = float(os.getenv("YOLO_CONF", "0.45"))
INFER_EVERY_N = int(os.getenv("INFER_EVERY_N", "2"))  # inferir cada N frames (para no bloquear)
IMG_SIZE      = int(os.getenv("IMG_SIZE", "640"))

# ROI (centrado, un poco más alto y extendido hacia abajo)
ROI_W_PCT  = float(os.getenv("ROI_W_PCT", "0.60"))  # % del ancho
ROI_H_PCT  = float(os.getenv("ROI_H_PCT", "0.50"))  # % del alto
ROI_CY_PCT = float(os.getenv("ROI_CY_PCT", "0.45"))  # centro Y (0.50 = centro exacto, 0.45 = 10% arriba)

# ISAPI (para snapshots cuando vehículo está totalmente dentro del ROI)
ISAPI_HOST       = os.getenv("ISAPI_HOST", "192.168.1.64")
ISAPI_USER       = os.getenv("ISAPI_USER", "admin")
ISAPI_PASSWORD   = os.getenv("ISAPI_PASSWORD", "")
SNAPSHOT_CHANNEL = os.getenv("SNAPSHOT_CHANNEL", "101")  # main stream
SNAPSHOT_COOLDOWN = float(os.getenv("SNAPSHOT_COOLDOWN", "1.0"))  # seg entre snapshots

os.makedirs(CAPTURE_DIR, exist_ok=True)

# RTSP baja latencia
set_ffmpeg_low_latency_env("tcp")

API_TOKEN = os.getenv("API_TOKEN", "")

# Clases de vehículos (COCO): car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = [2, 3, 5, 7]
VEHICLE_LABELS = {"2": "car", "3": "motorcycle", "5": "bus", "7": "truck"}

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

                    _model = YOLO(MODEL_PATH)
                    logger.info("Modelo YOLO cargado: %s", MODEL_PATH)
                except Exception as e:
                    _model = None
                    logger.error("Error al cargar modelo YOLO: %s", e)
    return _model


def require_auth():
    token = request.headers.get("X-API-Token")
    if not API_TOKEN:
        return
    if not token or token != API_TOKEN:
        abort(401, description="Unauthorized")


# ============================
# Utilidades
# ============================
def list_latest_images(limit=LATEST_LIMIT):
    pattern = os.path.join(CAPTURE_DIR, "*.jpg")
    files = glob.glob(pattern)
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    files = files[:max(0, int(limit))]
    return [os.path.basename(p) for p in files]


def save_isapi_snapshot(folder=CAPTURE_DIR, timeout=3):
    """Descarga snapshot vía ISAPI con Digest (usa la config del módulo)."""
    return _save_isapi_snapshot(
        ISAPI_HOST,
        ISAPI_USER,
        ISAPI_PASSWORD,
        folder=folder,
        channel=SNAPSHOT_CHANNEL,
        timeout=timeout,
    )


# ============================
# Cámara con detección embebida
# ============================
class DetectorCamera:
    """Captura frames vía ``FrameGrabberLatest`` y corre YOLO sobre el ROI.

    La adquisición de frames (con reconexión robusta) la maneja el grabber
    en su propio hilo; este hilo solo hace inferencia y anotación.
    """

    def __init__(self, src, model):
        self.src = src
        self.model = model
        self.grabber = FrameGrabberLatest(src, name="web")
        self.frame = None  # frame procesado (con anotaciones)
        self.lock = threading.Lock()
        self.running = True
        self.frame_idx = 0
        self.last_snap_ts = 0.0
        self.stats = {
            "frames_processed": 0,
            "detections": 0,
            "captures": 0,
            "start_time": time.time(),
        }
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        while self.running:
            ok, frame = self.grabber.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            self.frame_idx += 1
            draw = frame.copy()
            H, W = draw.shape[:2]

            # ROI centrado (un poco más alto y extendido hacia abajo)
            roi_rect = compute_roi(W, H, roi_w=ROI_W_PCT, roi_h=ROI_H_PCT, roi_cy=ROI_CY_PCT)
            x0, y0, x1, y1 = roi_rect
            cx = W // 2

            # Dibujo ROI
            cv2.rectangle(draw, (x0, y0), (x1, y1), (255, 200, 0), 2)
            cv2.line(draw, (cx, y0), (cx, y1), (255, 255, 255), 1)

            detected = False
            trigger_snapshot = False

            # Inference cada N frames para mantener FPS
            do_infer = self.model is not None and (self.frame_idx % max(1, INFER_EVERY_N) == 0)

            if do_infer and (x1 - x0) > 0 and (y1 - y0) > 0:
                roi_region = draw[y0:y1, x0:x1]
                try:
                    results = self.model.predict(
                        source=roi_region,
                        conf=YOLO_CONF,
                        classes=VEHICLE_CLASSES,
                        verbose=False,
                        imgsz=IMG_SIZE,
                    )
                except Exception as e:
                    logger.error("Predicción YOLO fallida: %s", e)
                    results = []
            else:
                results = []

            for r in results:
                for b in getattr(r, "boxes", []):
                    try:
                        cls_id = int(b.cls[0])
                        conf = float(b.conf[0])
                    except Exception:
                        continue

                    # Coordenadas relativas al ROI → mover a coords globales
                    xA, yA, xB, yB = b.xyxy[0].int().tolist()
                    xA_g, yA_g = xA + x0, yA + y0
                    xB_g, yB_g = xB + x0, yB + y0

                    detected = True
                    cv2.rectangle(draw, (xA_g, yA_g), (xB_g, yB_g), (0, 255, 0), 2)
                    label = VEHICLE_LABELS.get(str(cls_id), str(cls_id))
                    cv2.putText(
                        draw,
                        f"{label} {conf:.2f}",
                        (xA_g, max(yA_g - 5, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                    if box_inside_roi((xA_g, yA_g, xB_g, yB_g), roi_rect):
                        trigger_snapshot = True

            if detected:
                self.stats["detections"] += 1
            self.stats["frames_processed"] += 1

            # Banner superior
            status_color = (0, 255, 0) if detected else (0, 0, 255)
            status_text = "VEHICULO DETECTADO" if detected else "SIN DETECCION"
            cv2.rectangle(draw, (0, 0), (W, 36), (0, 0, 0), -1)
            cv2.putText(draw, status_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            # Info ROI
            cv2.putText(draw, f"ROI {x1 - x0}x{y1 - y0}", (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Snapshot con cooldown
            now = time.time()
            if trigger_snapshot and (now - self.last_snap_ts) > SNAPSHOT_COOLDOWN:
                self.last_snap_ts = now
                self.stats["captures"] += 1
                threading.Thread(target=save_isapi_snapshot, daemon=True).start()

            # Publicamos frame procesado
            with self.lock:
                self.frame = draw

    def read_jpeg(self):
        with self.lock:
            if self.frame is None:
                return None
            ok, buf = cv2.imencode(".jpg", self.frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None
        return buf.tobytes()

    def status(self) -> dict:
        """Estado de la cámara para monitoreo."""
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

    def stop(self):
        self.running = False
        try:
            if self.t.is_alive():
                self.t.join(timeout=1.0)
        except Exception:
            pass
        self.grabber.release()


def get_camera():
    """Crea (y cachea) la cámara de forma perezosa."""
    global _camera
    if _camera is None:
        _camera = DetectorCamera(RTSP_URL, _get_model())
    return _camera


def reset_camera():
    """Detiene y elimina la cámara actual (útil en tests/shutdown)."""
    global _camera
    if _camera is not None:
        _camera.stop()
        _camera = None


# ============================
# Flask
# ============================
def create_app():
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
                time.sleep(0.03)  # ~33 FPS máx; ajusta si quieres

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/latest_images")
    def api_latest_images():
        require_auth()
        imgs = list_latest_images(limit=LATEST_LIMIT)
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
        safe_dir = os.path.abspath(CAPTURE_DIR)
        requested = os.path.abspath(os.path.join(CAPTURE_DIR, filename))
        if not requested.startswith(safe_dir):
            return abort(403)
        if not os.path.exists(requested):
            return abort(404)
        return send_from_directory(CAPTURE_DIR, filename)

    @app.route("/_shutdown", methods=["POST"])
    def _shutdown():
        require_auth()
        reset_camera()
        return "ok"

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")), debug=False, threaded=True)