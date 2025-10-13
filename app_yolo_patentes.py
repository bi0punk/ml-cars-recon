#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RTSP + Detección YOLOv8 estable (sin crear múltiples ventanas)
"""

import os
import cv2
import time
import argparse
import numpy as np
from ultralytics import YOLO

# Fuerza RTSP por TCP y timeout razonable
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|max_delay;500000|stimeout;5000000"

def open_stream(rtsp_url):
    """Intenta abrir el stream RTSP y retorna el objeto cap."""
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    return cap if cap.isOpened() else None

def main():
    # Argumentos CLI
    ap = argparse.ArgumentParser(description="RTSP en vivo + YOLOv8 (estable)")
    ap.add_argument("--host", default="192.168.1.64")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", required=True)
    ap.add_argument("--channel", default="101")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--model", default="yolov8n.pt")
    args = ap.parse_args()

    # Cargar YOLO
    print(f"[INFO] Cargando modelo: {args.model}")
    model = YOLO(args.model)

    # Construir RTSP
    rtsp = f"rtsp://{args.user}:{args.password}@{args.host}:554/Streaming/Channels/{args.channel}"
    print(f"[INFO] Conectando a: {rtsp}")

    cap = open_stream(rtsp)
    if cap is None:
        print("[ERROR] No se pudo conectar al stream RTSP.")
        return

    print("[INFO] Transmisión iniciada. Presiona 'q' para salir.")
    last_retry = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            # Mostrar mensaje temporal en la ventana (sin cerrarla)
            blank = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(blank, "Reintentando conexión RTSP...", (80, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            cv2.imshow("RTSP Live (YOLOv8 - Estable)", blank)
            cv2.waitKey(1)

            # Reintento cada 3 segundos
            if time.time() - last_retry > 3:
                print("[WARN] RTSP perdido. Reintentando...")
                cap.release()
                cap = open_stream(rtsp)
                last_retry = time.time()
            time.sleep(0.5)
            continue

        # Frame válido → procesamiento normal
        frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
        H, W = frame.shape[:2]

        # ROI central
        roi_w = int(W * 0.50)
        roi_h = int(H * 0.70)
        #cx, cy = W // 2, H // 2
        # Centro del recuadro (mover hacia arriba restando píxeles o porcentaje)
        cx = W // 2
        cy = int(H * 0.40)  # antes era H // 2 → esto lo sube un poco

        roi_rect = (cx - roi_w // 2, cy - roi_h // 2, cx + roi_w // 2, cy + roi_h // 2)
        (x0, y0, x1, y1) = roi_rect
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 0), 2)

        # Detección YOLO en el ROI
        roi = frame[y0:y1, x0:x1]
        results = model.predict(source=roi, conf=0.45, verbose=False)
        detected = False

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names.get(cls_id, str(cls_id))
                if any(k in label.lower() for k in ["car", "plate", "license", "truck", "vehicle"]):
                    detected = True
                    xA, yA, xB, yB = box.xyxy[0].int().tolist()
                    # trasladar a coordenadas globales
                    xA += x0; yA += y0; xB += x0; yB += y0
                    cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (xA, max(yA - 5, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # Banner
        msg = "PLACA DETECTADA" if detected else "SIN DETECCION"
        color = (0, 200, 0) if detected else (0, 0, 255)
        cv2.rectangle(frame, (0, 0), (W, 35), (0, 0, 0), -1)
        cv2.putText(frame, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Mostrar frame único (misma ventana siempre)
        cv2.imshow("RTSP Live (YOLOv8 - Estable)", frame)

        # Tecla 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Transmisión finalizada.")

if __name__ == "__main__":
    main()
