#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RTSP Live + Detección Simple de Placa por Contornos
---------------------------------------------------
Este script muestra el video en vivo desde una cámara IP y detecta posibles
regiones con forma de placa dentro de una zona central (ROI). La detección
NO reconoce números, únicamente detecta el rectángulo típico de una placa
basándose en contornos y proporciones.

Cómo ejecutarlo:
  python3 detecta_placas_rtsp.py --host 192.168.1.64 --user admin --password "TuClave" --channel 101

Controles:
  q → salir de la ventana

Parámetros útiles:
  --host       IP de la cámara
  --user       Usuario
  --password   Contraseña
  --channel    Canal RTSP (101 = main stream, 102 = sub-stream)
  --width      Ancho del frame mostrado
  --height     Alto del frame mostrado

Resultado:
  Ventana en vivo mostrando:
    - ROI central (zona donde se analiza)
    - Si encuentra regiones tipo placa → texto "PLACA DETECTADA"
    - Si no → "SIN DETECCION"
"""


import os, cv2, argparse
import numpy as np

# Forzar RTSP por TCP para estabilidad
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|max_delay;500000|stimeout;5000000"

def detect_plate_like_regions(frame, roi_rect):
    """Detecta contornos tipo placa solo dentro del ROI."""
    (x0, y0, x1, y1) = roi_rect
    roi = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 60, 180)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    H, W = gray.shape
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w*h
        if area < 400 or area > 15000:
            continue
        aspect = w / float(h)
        if 2.0 <= aspect <= 6.5:  # proporción típica de placa
            boxes.append((x+x0, y+y0, w, h))
    return boxes

def main():
    ap = argparse.ArgumentParser(description="RTSP en vivo + detección en zona central (fluido)")
    ap.add_argument("--host", default="192.168.1.64")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", required=True)
    ap.add_argument("--channel", default="101")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    rtsp = f"rtsp://{args.user}:{args.password}@{args.host}:554/Streaming/Channels/{args.channel}"

    cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    if not cap.isOpened():
        print("No se pudo abrir el RTSP.")
        return

    print("Presiona 'q' para salir.")

    while True:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
            continue

        # Redimensionar para vista fluida
        frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
        H, W = frame.shape[:2]

        # ROI central (ajusta proporción si tu cámara apunta desde arriba o lejos)
        roi_w = int(W * 0.50)
        roi_h = int(H * 0.70)
        cx, cy = W // 2, H // 2
        roi_rect = (
            cx - roi_w // 2,
            cy - roi_h // 2,
            cx + roi_w // 2,
            cy + roi_h // 2
        )

        # Detección dentro del ROI
        boxes = detect_plate_like_regions(frame, roi_rect)
        detected = len(boxes) > 0

        # Dibuja ROI
        (x0, y0, x1, y1) = roi_rect
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 0), 2)

        # Dibuja detecciones
        for (x, y, w, h) in boxes[:2]:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Indicador superior
        msg = "PLACA DETECTADA" if detected else "SIN DETECCION"
        color = (0, 200, 0) if detected else (0, 0, 255)
        cv2.rectangle(frame, (0, 0), (W, 35), (0, 0, 0), -1)
        cv2.putText(frame, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Mostrar
        cv2.imshow("RTSP Live (ROI central)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
