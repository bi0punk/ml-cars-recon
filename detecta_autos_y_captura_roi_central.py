#!/usr/bin/env python3

"""
RTSP + Detección YOLOv8 + Captura ISAPI (ROI central)
----------------------------------------------------------
- Detecta vehículos en tiempo real con RTSP de baja latencia (sub-stream).
- Solo toma captura ISAPI si el vehículo está COMPLETO dentro del ROI centrado.
- Captura ISAPI se lanza en hilo separado (no bloquea el loop).
- Reconexión robusta con backoff (ver common/frames.py).

Uso:
  python3 detecta_autos_y_captura_roi_central.py --host 192.168.1.64 --user admin \
      --password 'TuClave' --rtsp-channel 102 --snapshot-channel 101 --model yolov8n.pt
"""

import argparse
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

from common.detector import VehicleDetector
from common.frames import FrameGrabber
from common.rtsp import build_isapi_channel_url, build_isapi_url
from common.utils import set_ffmpeg_low_latency_env

WINDOW = "RTSP Live (YOLOv8 - ISAPI)"

# Opciones FFmpeg para BAJA LATENCIA (robusto en LAN)
set_ffmpeg_low_latency_env("tcp")


def main():
    ap = argparse.ArgumentParser(description="RTSP baja latencia + YOLOv8 + captura ISAPI (ROI central)")
    ap.add_argument("--host", default=None, help="IP de la cámara")
    ap.add_argument("--user", default=None, help="Usuario de la cámara")
    ap.add_argument("--password", default=None, help="Contraseña de la cámara (o via env RTSP_PASSWORD)")
    ap.add_argument("--rtsp-channel", default=None, help="Canal RTSP para detección (substream recomendado)")
    ap.add_argument("--snapshot-channel", default=None, help="Canal ISAPI para snapshot (main stream)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--model", default=None)
    ap.add_argument("--conf", type=float, default=0.45, help="Confianza mínima YOLO")
    ap.add_argument("--cooldown", type=float, default=1.0, help="Segundos entre snapshots")
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

    print(f"[INFO] Cargando modelo YOLO: {model_path}")
    model = YOLO(model_path)

    url_builder = build_isapi_channel_url if args.substream_suffix else build_isapi_url
    rtsp_url = url_builder(host, user, password, rtsp_channel)

    # Inicializar grabber
    print(f"[INFO] Conectando a RTSP: user={user}, host={host}, canal={rtsp_channel}")
    grabber = FrameGrabber(rtsp_url, width=args.width, height=args.height)
    detector = VehicleDetector(model, conf=args.conf, cooldown=args.cooldown)

    # Estado
    fps_counter = 0
    fps_time = time.time()

    print("[INFO] Iniciando detección. Presiona 'q' para salir")

    try:
        while True:
            start_time = time.time()
            ok, frame = grabber.read()

            if not ok or frame is None:
                blank = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                cv2.putText(blank, "Conectando...", (50, args.height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(WINDOW, blank)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                time.sleep(0.1)
                continue

            H, W = frame.shape[:2]

            # ================================
            # ROI CENTRADO (RECTÁNGULO FIJO)
            # ================================
            roi_w = int(W * 0.60)  # 60% del ancho
            roi_h = int(H * 0.40)  # 40% de la altura

            # Centro del frame desplazado ligeramente hacia arriba
            cx, cy = W // 2, int(H * 0.30)

            # Extender ROI hacia abajo (manteniendo parte superior más alta)
            x0 = cx - roi_w // 2
            y0 = int(cy - roi_h * 0.6)  # sube un poco el inicio
            x1 = cx + roi_w // 2
            y1 = int(y0 + roi_h * 1.4)  # crece hacia abajo (40% más)

            # Asegurar límites válidos
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W, x1), min(H, y1)

            roi = (x0, y0, x1, y1)

            # Dibujar ROI centrado
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 200, 0), 2)
            cv2.putText(frame, "ROI", (x0 + 5, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            # Línea central de referencia
            cv2.line(frame, (cx, y0), (cx, y1), (255, 255, 255), 1)
            cv2.putText(frame, "Centro", (cx + 5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Detección YOLO dentro del ROI (anota el frame)
            frame, detected, trigger_snapshot = detector.detect(frame, roi)

            # Captura ISAPI (cooldown) en hilo separado
            if trigger_snapshot:
                detector.trigger_if_ready(
                    host, user, password, snapshot_channel, folder="isapi_snaps", timeout=3
                )

            # FPS
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps = fps_counter / (time.time() - fps_time)
                fps_counter = 0
                fps_time = time.time()
                fps_text = f"FPS: {fps:.1f}"
            else:
                fps_text = "FPS: --"

            # UI
            status_color = (0, 255, 0) if detected else (0, 0, 255)
            status_text = "VEHICULO DETECTADO" if detected else "SIN DETECCION"

            cv2.rectangle(frame, (0, 0), (W, 40), (0, 0, 0), -1)
            cv2.putText(frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, fps_text, (W - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            roi_info = f"ROI (centrado): {x1 - x0}x{y1 - y0}"
            cv2.putText(frame, roi_info, (10, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            processing_time = (time.time() - start_time) * 1000
            cv2.putText(frame, f"Proc: {processing_time:.1f}ms", (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Mostrar
            cv2.imshow(WINDOW, frame)

            # Controles
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupción por usuario")
    except Exception as e:
        print(f"[ERROR] Error en el loop principal: {e}")
    finally:
        grabber.release()
        cv2.destroyAllWindows()
        print("[INFO] Programa finalizado correctamente.")


if __name__ == "__main__":
    main()