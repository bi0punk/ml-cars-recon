#!/usr/bin/env python3

"""
RTSP + Detección YOLOv8 + Captura ISAPI (baja latencia)
----------------------------------------------------------
- Detecta vehículos en tiempo real con RTSP de baja latencia (sub-stream).
- Solo toma captura ISAPI si el vehículo está COMPLETO dentro del ROI centrado.
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
from common.rtsp import build_isapi_channel_url
from common.utils import set_ffmpeg_low_latency_env

# Opciones FFmpeg para BAJA LATENCIA (robusto en LAN)
set_ffmpeg_low_latency_env("tcp")

# Clases de vehículos a detectar (COCO dataset)
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
VEHICLE_LABELS = ["car", "motorcycle", "bus", "truck"]


def main():
    ap = argparse.ArgumentParser(description="RTSP baja latencia + YOLOv8 + captura ISAPI")
    ap.add_argument("--host", default=None, help="IP de la cámara")
    ap.add_argument("--user", default=None, help="Usuario de la cámara")
    ap.add_argument("--password", default=None, help="Contraseña de la cámara")
    ap.add_argument("--rtsp-channel", default=None, help="Canal RTSP para detección (substream recomendado)")
    ap.add_argument("--snapshot-channel", default=None, help="Canal ISAPI para snapshot (main stream)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--model", default=None)
    ap.add_argument("--conf", type=float, default=0.45, help="Confianza mínima YOLO")
    ap.add_argument("--cooldown", type=float, default=1.0, help="Segundos entre snapshots")
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

    rtsp_url = build_isapi_channel_url(host, user, password, rtsp_channel)

    # Inicializar grabber
    print(f"[INFO] Conectando a RTSP: user={user}, host={host}, canal={rtsp_channel}")
    grabber = FrameGrabber(rtsp_url, width=args.width, height=args.height)

    # Estado
    last_capture_ts = 0.0
    fps_counter = 0
    fps_time = time.time()
    snapshot_thread = None

    print("[INFO] Iniciando detección. Presiona 'q' para salir")

    try:
        while True:
            start_time = time.time()
            ok, frame = grabber.read()

            if not ok or frame is None:
                blank = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                cv2.putText(blank, "Conectando...", (50, args.height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("RTSP Live (YOLOv8 - ISAPI)", blank)
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

            roi_rect = (x0, y0, x1, y1)

            # Dibujar ROI centrado
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 200, 0), 2)
            cv2.putText(frame, "ROI", (x0 + 5, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            # Línea central de referencia
            cv2.line(frame, (cx, y0), (cx, y1), (255, 255, 255), 1)
            cv2.putText(frame, "Centro", (cx + 5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Detección YOLO dentro del ROI
            roi_region = frame[y0:y1, x0:x1]
            if roi_region.size > 0:
                try:
                    results = model.predict(
                        source=roi_region,
                        conf=args.conf,
                        classes=VEHICLE_CLASSES,
                        verbose=False,
                        imgsz=640,
                    )
                except Exception as e:
                    print(f"[ERROR] Predicción fallida: {e}")
                    results = []
            else:
                results = []

            detected = False
            trigger_snapshot = False

            for r in results:
                for box in getattr(r, "boxes", []):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    # Obtener etiqueta de manera segura
                    try:
                        label = model.names[cls_id]
                    except Exception:
                        label = str(cls_id)

                    if label in VEHICLE_LABELS:
                        detected = True
                        xA, yA, xB, yB = box.xyxy[0].int().tolist()

                        # Convertir a coordenadas globales
                        xA_global, yA_global = xA + x0, yA + y0
                        xB_global, yB_global = xB + x0, yB + y0

                        # Dibujar detección
                        cv2.rectangle(frame, (xA_global, yA_global), (xB_global, yB_global), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (xA_global, max(yA_global - 5, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Verificar si está completamente dentro del ROI
                        if box_inside_roi((xA_global, yA_global, xB_global, yB_global), roi_rect):
                            trigger_snapshot = True

            # Captura ISAPI (cooldown)
            now = time.time()
            if (
                trigger_snapshot
                and (now - last_capture_ts) > args.cooldown
                and (snapshot_thread is None or not snapshot_thread.is_alive())
            ):
                last_capture_ts = now
                print("[EVENTO] Vehículo en ROI → Capturando ISAPI...")
                snapshot_thread = threading.Thread(
                    target=save_isapi_snapshot,
                    args=(host, user, password),
                    kwargs={"folder": "isapi_snaps", "channel": snapshot_channel, "timeout": 3},
                    daemon=True,
                )
                snapshot_thread.start()

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
            cv2.imshow("RTSP Live (YOLOv8 - ISAPI)", frame)

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