#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RTSP + Detección YOLOv8 + Captura ISAPI y RTSP (mejorado)
-------------------------------------------------------------
- Detecta vehículos en tiempo real.
- Solo toma capturas si el vehículo está completamente dentro del ROI central.
- Guarda:
    1) Captura RTSP (frame actual mostrado)
    2) Captura ISAPI (snapshot nativo de la cámara)
- Reintenta conexión sin crear múltiples ventanas.
"""

import os
import cv2
import time
import argparse
import numpy as np
import requests
from requests.auth import HTTPDigestAuth   # autenticación digest
from datetime import datetime
from ultralytics import YOLO

# Forzar RTSP por TCP y timeout razonable
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|max_delay;500000|stimeout;5000000"
)

# =====================================================
# FUNCIONES AUXILIARES
# =====================================================

def open_stream(rtsp_url):
    """Intenta abrir el stream RTSP y retorna el objeto cap."""
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    return cap if cap.isOpened() else None

def save_rtsp_capture(frame, folder="captures"):
    """Guarda el frame actual mostrado como JPG."""
    os.makedirs(folder, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(folder, f"{ts}_rtsp.jpg")
    cv2.imwrite(filename, frame)
    print(f"[RTSP] Captura guardada: {filename}")

def save_isapi_snapshot(host, user, password, folder="captures", channel="101"):
    """Descarga snapshot vía ISAPI usando autenticación Digest (igual que curl --digest)."""
    os.makedirs(folder, exist_ok=True)
    url = f"http://{host}/ISAPI/Streaming/channels/{channel}/picture"
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(folder, f"{ts}_isapi.jpg")
    try:
        r = requests.get(url, auth=HTTPDigestAuth(user, password), timeout=5, stream=True)
        if r.status_code == 200:
            with open(filename, "wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            print(f"[ISAPI] Captura guardada: {filename}")
        else:
            print(f"[ISAPI] Error HTTP {r.status_code}")
    except Exception as e:
        print(f"[ISAPI] Error al obtener snapshot: {e}")

# =====================================================
# PROGRAMA PRINCIPAL
# =====================================================

def main():
    # Argumentos CLI
    ap = argparse.ArgumentParser(description="RTSP en vivo + YOLOv8 + capturas ISAPI/RTSP")
    ap.add_argument("--host", default="192.168.1.64", help="IP de la cámara")
    ap.add_argument("--user", default="admin", help="Usuario de la cámara")
    ap.add_argument("--password", required=True, help="Contraseña de la cámara")
    ap.add_argument("--channel", default="101", help="Canal (101=main, 102=sub)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--model", default="yolov8n.pt")
    args = ap.parse_args()

    # Cargar modelo YOLO
    print(f"[INFO] Cargando modelo: {args.model}")
    model = YOLO(args.model)

    # Construir URL RTSP
    rtsp = f"rtsp://{args.user}:{args.password}@{args.host}:554/Streaming/Channels/{args.channel}"
    print(f"[INFO] Conectando a: {rtsp}")

    cap = open_stream(rtsp)
    if cap is None:
        print("[ERROR] No se pudo conectar al stream RTSP.")
        return

    print("[INFO] Transmisión iniciada. Presiona 'q' para salir.")
    last_retry = 0
    last_capture = 0  # controla intervalo entre capturas (5s)

    # =====================================================
    # LOOP PRINCIPAL
    # =====================================================
    while True:
        ok, frame = cap.read()

        # Reintento si el stream falla
        if not ok:
            blank = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(blank, "Reintentando conexión RTSP...", (80, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.imshow("RTSP Live (YOLOv8 - Capturas)", blank)
            cv2.waitKey(1)

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

        # ROI central (recuadro visible)
        roi_w = int(W * 0.50)
        roi_h = int(H * 0.70)
        cx, cy = W // 2, int(H * 0.40)  # subido un poco (ajustable)
        roi_rect = (cx - roi_w // 2, cy - roi_h // 2, cx + roi_w // 2, cy + roi_h // 2)
        (x0, y0, x1, y1) = roi_rect
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 0), 2)

        # Detección YOLO dentro del ROI
        roi = frame[y0:y1, x0:x1]
        results = model.predict(source=roi, conf=0.45, verbose=False)
        detected = False

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names.get(cls_id, str(cls_id))

                # Detectar solo vehículos
                if any(k in label.lower() for k in ["car", "vehicle", "truck"]):
                    detected = True
                    xA, yA, xB, yB = box.xyxy[0].int().tolist()
                    xA += x0; yA += y0; xB += x0; yB += y0
                    cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (xA, max(yA - 5, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                    # --- NUEVO: solo capturar si el vehículo está completamente dentro del ROI ---
                    inside_roi = xA >= x0 and yA >= y0 and xB <= x1 and yB <= y1
                    if inside_roi and time.time() - last_capture > 5:
                        last_capture = time.time()
                        print("[EVENTO] Vehículo COMPLETO en ROI → capturando imágenes...")
                        save_rtsp_capture(frame)
                        save_isapi_snapshot(args.host, args.user, args.password, channel=args.channel)

        # Banner informativo
        msg = "VEHICULO DETECTADO" if detected else "SIN DETECCION"
        color = (0, 200, 0) if detected else (0, 0, 255)
        cv2.rectangle(frame, (0, 0), (W, 35), (0, 0, 0), -1)
        cv2.putText(frame, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Mostrar ventana única
        cv2.imshow("RTSP Live (YOLOv8 - Capturas)", frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # =====================================================
    # LIMPIEZA FINAL
    # =====================================================
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Transmisión finalizada correctamente.")

# =====================================================
# PUNTO DE ENTRADA
# =====================================================
if __name__ == "__main__":
    main()
