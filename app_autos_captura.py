#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RTSP + Detección YOLOv8 + Captura ISAPI (baja latencia)
-------------------------------------------------------------
- Detecta vehículos en tiempo real con RTSP de baja latencia (sub-stream).
- Solo toma captura ISAPI si el vehículo está COMPLETO dentro del ROI.
- Captura ISAPI se lanza en hilo separado (no bloquea el loop).
- Reintenta conexión sin crear múltiples ventanas.

Uso:
  python3 cam_isapi_yolo.py --host 192.168.1.64 --user admin --password 'TuClave' \
      --rtsp-channel 102 --snapshot-channel 101 --model yolov8n.pt
"""

import os
import cv2
import time
import argparse
import numpy as np
import requests
import threading
from requests.auth import HTTPDigestAuth
from datetime import datetime
from ultralytics import YOLO

# =========================================
# Opciones FFmpeg para BAJA LATENCIA
# =========================================
# - rtsp_transport=tcp : robusto en LAN
# - max_delay=0         : sin buffer adicional
# - stimeout=5s         : timeout conexión
# - buffer_size=0       : no prebuffer
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|max_delay;0|stimeout;5000000|buffer_size;0"
)

# =====================================================
# UTILIDADES
# =====================================================

def save_isapi_snapshot(host, user, password, folder="captures", channel="101", timeout=4):
    """
    Descarga snapshot vía ISAPI usando autenticación Digest.
    Se recomienda channel=101 (main) para máxima calidad.
    """
    try:
        os.makedirs(folder, exist_ok=True)
        url = f"http://{host}/ISAPI/Streaming/channels/{channel}/picture"
        r = requests.get(url, auth=HTTPDigestAuth(user, password), timeout=timeout, stream=True)
        if r.status_code == 200 and r.headers.get("Content-Type", "").startswith("image"):
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
            filename = os.path.join(folder, f"{ts}_isapi_{channel}.jpg")
            with open(filename, "wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            print(f"[ISAPI] Captura guardada: {filename}")
            return filename
        else:
            print(f"[ISAPI] Error HTTP {r.status_code} / Content-Type={r.headers.get('Content-Type')}")
    except Exception as e:
        print(f"[ISAPI] Error al obtener snapshot: {e}")

def box_inside_roi(box, roi):
    # box=(x1,y1,x2,y2); roi=(x1,y1,x2,y2)
    x1,y1,x2,y2 = box
    rx1,ry1,rx2,ry2 = roi
    return x1 >= rx1 and y1 >= ry1 and x2 <= rx2 and y2 <= ry2

class FrameGrabber:
    """
    Hilo lector que siempre deja disponible el ÚLTIMO frame (descarta los viejos).
    Esto reduce la latencia percibida en RTSP.
    """
    def __init__(self, rtsp_url, width=None, height=None):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.width = width
        self.height = height
        self.ok = False
        self.frame = None
        self.stopped = False
        self.last_open = 0
        self._open()
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _open(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        # Algunos backends respetan buffersize=1
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if self.cap.isOpened():
            self.ok = True
        else:
            self.ok = False

    def _loop(self):
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                # Reintento de reconexión cada 3s
                if time.time() - self.last_open > 3:
                    print("[WARN] RTSP perdido. Reintentando abrir...")
                    self._open()
                    self.last_open = time.time()
                time.sleep(0.2)
                continue
            ok, f = self.cap.read()
            if not ok:
                self.ok = False
                # pequeña pausa antes de reintentar leer
                time.sleep(0.01)
                continue
            if f is not None:
                if self.width and self.height:
                    f = cv2.resize(f, (self.width, self.height), interpolation=cv2.INTER_AREA)
                self.ok, self.frame = True, f

    def read(self):
        return self.ok, self.frame

    def release(self):
        self.stopped = True
        try:
            self.th.join(timeout=1)
        except Exception:
            pass
        if self.cap:
            self.cap.release()

# =====================================================
# PROGRAMA PRINCIPAL
# =====================================================

def main():
    ap = argparse.ArgumentParser(description="RTSP baja latencia + YOLOv8 + captura ISAPI")
    ap.add_argument("--host", default=None, help="IP de la cámara")
    ap.add_argument("--user", default=None, help="Usuario de la cámara")
    ap.add_argument("--password", default=None, help="Contraseña de la cámara (o via env RTSP_PASSWORD)")

    ap.add_argument("--rtsp-channel", default=None, help="Canal RTSP para detección (102=sub recomendado)")
    ap.add_argument("--snapshot-channel", default=None, help="Canal ISAPI para snapshot (101=main recomendado)")

    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--model", default=None)
    ap.add_argument("--conf", type=float, default=0.45, help="Confianza mínima YOLO")
    ap.add_argument("--cooldown", type=float, default=0.8, help="Segundos entre snapshots")
    args = ap.parse_args()

    host = args.host or os.environ.get("RTSP_HOST", "192.168.1.64")
    user = args.user or os.environ.get("RTSP_USER", "admin")
    password = args.password or os.environ.get("RTSP_PASSWORD", "")
    rtsp_channel = args.rtsp_channel or os.environ.get("RTSP_CHANNEL", "102")
    snapshot_channel = args.snapshot_channel or os.environ.get("SNAPSHOT_CHANNEL", "101")
    model_path = args.model or os.environ.get("YOLO_MODEL", "yolov8n.pt")

    if not password:
        print("[ERROR] RTSP_PASSWORD debe estar definida en .env o pasar --password")
        return

    if not os.path.exists(model_path):
        print(f"[ERROR] Modelo no encontrado: {model_path}")
        return

    print(f"[INFO] Cargando modelo: {model_path}")
    model = YOLO(model_path)

    rtsp = f"rtsp://{user}:{password}@{host}:554/ISAPI/Streaming/channels/{rtsp_channel}"

    print(f"[INFO] Conectando RTSP (detección) canal {rtsp_channel}: user={user}, host={host}")
    grab = FrameGrabber(rtsp, width=args.width, height=args.height)

    print("[INFO] Transmisión iniciada. Presiona 'q' para salir.")
    last_capture_ts = 0.0

    try:
        while True:
            ok, frame = grab.read()

            if not ok or frame is None:
                # Ventana de aviso mientras reconecta/lee
                blank = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                cv2.putText(blank, "Reintentando conexión RTSP...", (60, args.height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.imshow("RTSP Live (YOLOv8 - ISAPI)", blank)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time.sleep(0.05)
                continue

            H, W = frame.shape[:2]

            # ROI central (ajusta a tu escena)
            roi_w = int(W * 0.50)
            roi_h = int(H * 0.70)
            cx, cy = W // 2, int(H * 0.40)
            roi_rect = (cx - roi_w // 2, cy - roi_h // 2, cx + roi_w // 2, cy + roi_h // 2)
            (x0, y0, x1, y1) = roi_rect
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 0), 2)

            # Detección YOLO SOLO en el ROI
            roi = frame[y0:y1, x0:x1]
            results = model.predict(source=roi, conf=args.conf, verbose=False)
            detected = False
            trigger_snapshot = False

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names.get(cls_id, str(cls_id))

                    # Filtra vehículos (ajusta etiquetas según tu modelo)
                    if any(k in label.lower() for k in ["car", "vehicle", "truck", "bus", "motorbike"]):
                        detected = True
                        xA, yA, xB, yB = box.xyxy[0].int().tolist()
                        # Reubica a coords globales (frame completo)
                        xA += x0
                        yA += y0
                        xB += x0
                        yB += y0
                        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (xA, max(yA - 5, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                        if box_inside_roi((xA, yA, xB, yB), roi_rect):
                            trigger_snapshot = True

            # Disparo con cooldown y no bloquear el loop
            now = time.time()
            if trigger_snapshot and (now - last_capture_ts) > args.cooldown:
                last_capture_ts = now
                print("[EVENTO] Vehículo COMPLETO en ROI → capturando ISAPI (async)...")
                threading.Thread(
                    target=save_isapi_snapshot,
                    args=(host, user, password),
                    kwargs={"folder": "isapi_snaps", "channel": snapshot_channel, "timeout": 4},
                    daemon=True
                ).start()

            # Banner informativo
            msg = "VEHICULO DETECTADO" if detected else "SIN DETECCION"
            color = (0, 200, 0) if detected else (0, 0, 255)
            cv2.rectangle(frame, (0, 0), (W, 35), (0, 0, 0), -1)
            cv2.putText(frame, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Mostrar ventana única
            cv2.imshow("RTSP Live (YOLOv8 - ISAPI)", frame)

            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        grab.release()
        cv2.destroyAllWindows()
        print("[INFO] Transmisión finalizada correctamente.")

if __name__ == "__main__":
    main()
