#!/usr/bin/env python3

"""
RTSP + Detección YOLOv8 + Captura ISAPI (baja latencia)
----------------------------------------------------------
- Detecta vehículos en tiempo real con RTSP de baja latencia (sub-stream).
- Solo toma captura ISAPI si el vehículo está COMPLETO dentro del ROI.
- Captura ISAPI se lanza en hilo separado (no bloquea el loop).
- Reconexión robusta con backoff (ver common/frames.py).

Uso:
  python3 app_autos_captura.py --host 192.168.1.64 --user admin --password 'TuClave' \
      --rtsp-channel 102 --snapshot-channel 101 --model yolov8n.pt
"""

import argparse
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

from common.detector import VehicleDetector
from common.frames import FrameGrabber
from common.geometry import compute_roi
from common.rtsp import build_isapi_channel_url, build_isapi_url
from common.utils import set_ffmpeg_low_latency_env

WINDOW = "RTSP Live (YOLOv8 - ISAPI)"

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
    ap.add_argument(
        "--substream-suffix",
        action="store_true",
        help="Algunos firmwares exigen el sufijo '01' en el canal (p.ej. 10101)",
    )
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

    url_builder = build_isapi_channel_url if args.substream_suffix else build_isapi_url
    rtsp = url_builder(host, user, password, rtsp_channel)

    print(f"[INFO] Conectando RTSP (detección) canal {rtsp_channel}: user={user}, host={host}")
    grab = FrameGrabber(rtsp, width=args.width, height=args.height)
    detector = VehicleDetector(model, conf=args.conf, cooldown=args.cooldown)

    print("[INFO] Transmisión iniciada. Presiona 'q' para salir.")

    try:
        while True:
            ok, frame = grab.read()

            if not ok or frame is None:
                # Ventana de aviso mientras reconecta/lee
                blank = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                cv2.putText(blank, "Reintentando conexión RTSP...", (60, args.height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.imshow(WINDOW, blank)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                time.sleep(0.05)
                continue

            H, W = frame.shape[:2]

            # ROI central (ajusta a tu escena)
            roi = compute_roi(W, H, roi_w=0.50, roi_h=0.70, roi_cy=0.40)
            (x0, y0, x1, y1) = roi
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 0), 2)

            # Detección YOLO SOLO en el ROI (anota el frame)
            frame, detected, trigger_snapshot = detector.detect(frame, roi)

            # Disparo con cooldown y no bloquear el loop
            if trigger_snapshot:
                detector.trigger_if_ready(
                    host, user, password, snapshot_channel, folder="isapi_snaps", timeout=4
                )

            # Banner informativo
            msg = "VEHICULO DETECTADO" if detected else "SIN DETECCION"
            color = (0, 200, 0) if detected else (0, 0, 255)
            cv2.rectangle(frame, (0, 0), (W, 35), (0, 0, 0), -1)
            cv2.putText(frame, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Mostrar ventana única
            cv2.imshow(WINDOW, frame)

            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        grab.release()
        cv2.destroyAllWindows()
        print("[INFO] Transmisión finalizada correctamente.")


if __name__ == "__main__":
    main()