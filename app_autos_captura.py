#!/usr/bin/env python3

"""
RTSP + Detección YOLOv8 + Captura ISAPI (baja latencia)
----------------------------------------------------------
- Detecta vehículos en tiempo real con RTSP de baja latencia (sub-stream).
- Solo toma captura ISAPI si el vehículo está COMPLETO dentro del ROI.
- Captura ISAPI se lanza en hilo separado (no bloquea el loop).
- Reintenta conexión sin crear múltiples ventanas.

Uso:
  python3 cam_isapi_yolo.py --host 192.168.1.64 --user admin --password 'TuClave' \
      --rtsp-channel 102 --snapshot-channel 101 --model yolov8n.pt
"""

import argparse
import os
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO

from common.frames import FrameGrabber
from common.geometry import box_inside_roi
from common.isapi import save_isapi_snapshot
from common.rtsp import build_isapi_url
from common.utils import set_ffmpeg_low_latency_env

# Opciones FFmpeg para BAJA LATENCIA (robusto en LAN)
set_ffmpeg_low_latency_env("tcp")


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

    rtsp = build_isapi_url(host, user, password, rtsp_channel)

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
                cv2.putText(blank, "Reintentando conexión RTSP...", (60, args.height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.imshow("RTSP Live (YOLOv8 - ISAPI)", blank)
                if cv2.waitKey(1) & 0xFF == ord("q"):
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
                        cv2.putText(frame, f"{label} {conf:.2f}", (xA, max(yA - 5, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

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
                    daemon=True,
                ).start()

            # Banner informativo
            msg = "VEHICULO DETECTADO" if detected else "SIN DETECCION"
            color = (0, 200, 0) if detected else (0, 0, 255)
            cv2.rectangle(frame, (0, 0), (W, 35), (0, 0, 0), -1)
            cv2.putText(frame, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Mostrar ventana única
            cv2.imshow("RTSP Live (YOLOv8 - ISAPI)", frame)

            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        grab.release()
        cv2.destroyAllWindows()
        print("[INFO] Transmisión finalizada correctamente.")


if __name__ == "__main__":
    main()