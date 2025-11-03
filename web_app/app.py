#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import time
import glob
import math
import threading
import requests
from datetime import datetime
from requests.auth import HTTPDigestAuth
from flask import Flask, render_template, Response, jsonify, send_from_directory, abort

# ============================
# Configuración por variables
# ============================
RTSP_URL           = os.getenv("RTSP_URL", "rtsp://admin:9H)p5x84@192.168.1.64:554/Streaming/Channels/102")
CAPTURE_DIR        = os.getenv("CAPTURE_DIR", "isapi_snaps")
LATEST_LIMIT       = int(os.getenv("LATEST_LIMIT", "3"))

# Detección
MODEL_PATH         = os.getenv("MODEL_PATH", "yolov8n.pt")
YOLO_CONF          = float(os.getenv("YOLO_CONF", "0.45"))
INFER_EVERY_N      = int(os.getenv("INFER_EVERY_N", "2"))     # inferir cada N frames (para no bloquear)
IMG_SIZE           = int(os.getenv("IMG_SIZE", "640"))

# ROI (centrado, un poco más alto y extendido hacia abajo)
ROI_W_PCT          = float(os.getenv("ROI_W_PCT", "0.60"))    # % del ancho
ROI_H_PCT          = float(os.getenv("ROI_H_PCT", "0.50"))    # % del alto
ROI_CY_PCT         = float(os.getenv("ROI_CY_PCT", "0.45"))   # centro Y (0.50 = centro exacto, 0.45 = 10% arriba)

# ISAPI (para snapshots cuando vehículo está totalmente dentro del ROI)
ISAPI_HOST         = os.getenv("ISAPI_HOST", "192.168.1.64")
ISAPI_USER         = os.getenv("ISAPI_USER", "admin")
ISAPI_PASSWORD     = os.getenv("ISAPI_PASSWORD", "9H)p5x84")
SNAPSHOT_CHANNEL   = os.getenv("SNAPSHOT_CHANNEL", "101")      # main stream
SNAPSHOT_COOLDOWN  = float(os.getenv("SNAPSHOT_COOLDOWN", "1.0"))  # seg entre snapshots

os.makedirs(CAPTURE_DIR, exist_ok=True)

# RTSP baja latencia (opcional)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|max_delay;0|stimeout;5000000|buffer_size;0"

# ============================
# Flask
# ============================
app = Flask(__name__)

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
    """ Descarga snapshot vía ISAPI con Digest. """
    try:
        os.makedirs(folder, exist_ok=True)
        url = f"http://{ISAPI_HOST}/ISAPI/Streaming/channels/{SNAPSHOT_CHANNEL}/picture"
        resp = requests.get(url, auth=HTTPDigestAuth(ISAPI_USER, ISAPI_PASSWORD), timeout=timeout, stream=True)
        if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith("image"):
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]
            fn = os.path.join(folder, f"{ts}_isapi_{SNAPSHOT_CHANNEL}.jpg")
            with open(fn, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            print(f"[ISAPI] Captura guardada: {fn}")
            return fn
        else:
            print(f"[ISAPI] Error HTTP {resp.status_code}")
    except requests.exceptions.Timeout:
        print("[ISAPI] Timeout snapshot")
    except Exception as e:
        print(f"[ISAPI] Error: {e}")
    return None

def box_inside_roi(box, roi):
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = roi
    return (x1 >= rx1 and y1 >= ry1 and x2 <= rx2 and y2 <= ry2)

# ============================
# Carga de modelo YOLO (una vez)
# ============================
try:
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    print(f"[YOLO] Modelo cargado: {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"[YOLO] ERROR al cargar modelo: {e}")

# Clases de vehículos (COCO): car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = [2, 3, 5, 7]
VEHICLE_LABELS  = {"2": "car", "3": "motorcycle", "5": "bus", "7": "truck"}

# ============================
# Cámara con detección embebida
# ============================
class DetectorCamera:
    def __init__(self, src):
        self.src = src
        self.cap = None
        self.frame = None      # frame procesado (con anotaciones)
        self.raw = None        # frame crudo (opcional)
        self.lock = threading.Lock()
        self.running = True
        self.last_open = 0
        self.reconnect_interval = 5
        self.frame_idx = 0
        self.last_snap_ts = 0.0
        self._open()
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _open(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 15)
        except:
            pass
        self.last_open = time.time()

    def _loop(self):
        empty = 0
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                if time.time() - self.last_open > self.reconnect_interval:
                    print("[RTSP] Reconectando…")
                    self._open()
                time.sleep(0.2)
                continue

            ok, frame = self.cap.read()
            if not ok or frame is None:
                empty += 1
                if empty > 30:
                    print("[RTSP] Muchos frames vacíos. Reabriendo…")
                    self._open()
                    empty = 0
                time.sleep(0.01)
                continue
            empty = 0
            self.frame_idx += 1

            # Guardamos crudo por si se quiere
            draw = frame.copy()
            H, W = draw.shape[:2]

            # ROI centrado (un poco más alto y extendido hacia abajo)
            roi_w = int(W * ROI_W_PCT)
            roi_h = int(H * ROI_H_PCT)
            cx, cy = W // 2, int(H * ROI_CY_PCT)

            x0 = max(0, cx - roi_w // 2)
            y0 = max(0, cy - roi_h // 2)
            x1 = min(W, cx + roi_w // 2)
            y1 = min(H, cy + roi_h // 2)
            roi_rect = (x0, y0, x1, y1)

            # Dibujo ROI
            cv2.rectangle(draw, (x0, y0), (x1, y1), (255, 200, 0), 2)
            cv2.line(draw, (cx, y0), (cx, y1), (255, 255, 255), 1)

            detected = False
            trigger_snapshot = False

            # Inference cada N frames para mantener FPS
            do_infer = (model is not None and (self.frame_idx % max(1, INFER_EVERY_N) == 0))

            if do_infer and (x1 - x0) > 0 and (y1 - y0) > 0:
                roi_region = draw[y0:y1, x0:x1]
                try:
                    results = model.predict(
                        source=roi_region,
                        conf=YOLO_CONF,
                        classes=VEHICLE_CLASSES,
                        verbose=False,
                        imgsz=IMG_SIZE
                    )
                except Exception as e:
                    print(f"[YOLO] Predicción fallida: {e}")
                    results = []
            else:
                results = []

            for r in results:
                for b in getattr(r, "boxes", []):
                    try:
                        cls_id = int(b.cls[0])
                        conf   = float(b.conf[0])
                    except Exception:
                        continue

                    # Coordenadas relativas al ROI
                    xA, yA, xB, yB = b.xyxy[0].int().tolist()
                    # Mover a coords globales
                    xA_g, yA_g = xA + x0, yA + y0
                    xB_g, yB_g = xB + x0, yB + y0

                    detected = True
                    cv2.rectangle(draw, (xA_g, yA_g), (xB_g, yB_g), (0, 255, 0), 2)
                    label = VEHICLE_LABELS.get(str(cls_id), str(cls_id))
                    cv2.putText(draw, f"{label} {conf:.2f}", (xA_g, max(yA_g - 5, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if box_inside_roi((xA_g, yA_g, xB_g, yB_g), roi_rect):
                        trigger_snapshot = True

            # Banner superior
            status_color = (0, 255, 0) if detected else (0, 0, 255)
            status_text  = "VEHICULO DETECTADO" if detected else "SIN DETECCION"
            cv2.rectangle(draw, (0, 0), (W, 36), (0, 0, 0), -1)
            cv2.putText(draw, status_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            # Info ROI
            cv2.putText(draw, f"ROI {x1-x0}x{y1-y0}", (10, H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Snapshot con cooldown
            now = time.time()
            if trigger_snapshot and (now - self.last_snap_ts) > SNAPSHOT_COOLDOWN:
                self.last_snap_ts = now
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

    def stop(self):
        self.running = False
        try:
            if self.t.is_alive():
                self.t.join(timeout=1.0)
        except:
            pass
        if self.cap:
            self.cap.release()

camera = DetectorCamera(RTSP_URL)

# ============================
# Rutas web
# ============================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
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
    imgs = list_latest_images(limit=LATEST_LIMIT)
    now = int(time.time())
    data = [{"name": f, "url": f"/captures/{f}?t={now}"} for f in imgs]
    return jsonify(data)

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
    camera.stop()
    return "ok"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False, threaded=True)
