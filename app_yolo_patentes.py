#!/usr/bin/env python3

import argparse
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|max_delay;500000|stimeout;5000000"


def open_stream(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    return cap if cap.isOpened() else None


def main():
    ap = argparse.ArgumentParser(description="RTSP en vivo + YOLOv8 (estable)")
    ap.add_argument("--host", default=None)
    ap.add_argument("--user", default=None)
    ap.add_argument("--password", default=None)
    ap.add_argument("--channel", default=None)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    host = args.host or os.environ.get("RTSP_HOST", "192.168.1.64")
    user = args.user or os.environ.get("RTSP_USER", "admin")
    password = args.password or os.environ.get("RTSP_PASSWORD", "")
    channel = args.channel or os.environ.get("RTSP_CHANNEL", "101")
    model_path = args.model or os.environ.get("YOLO_MODEL", "yolov8n.pt")

    if not password:
        print("[ERROR] RTSP_PASSWORD debe estar definida en .env o pasar --password")
        return

    if not os.path.exists(model_path):
        print(f"[ERROR] Modelo no encontrado: {model_path}")
        return

    print(f"[INFO] Cargando modelo: {model_path}")
    model = YOLO(model_path)

    rtsp = f"rtsp://{user}:{password}@{host}:554/Streaming/Channels/{channel}"
    print(f"[INFO] Conectando a: user={user}, host={host}, channel={channel}")

    cap = open_stream(rtsp)
    if cap is None:
        print("[ERROR] No se pudo conectar al stream RTSP.")
        return

    print("[INFO] Transmisión iniciada. Presiona 'q' para salir.")
    last_retry = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            blank = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(blank, "Reintentando conexion RTSP...", (80, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.imshow("RTSP Live (YOLOv8 - Estable)", blank)
            cv2.waitKey(1)
            if time.time() - last_retry > 3:
                print("[WARN] RTSP perdido. Reintentando...")
                cap.release()
                cap = open_stream(rtsp)
                last_retry = time.time()
            time.sleep(0.5)
            continue

        frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
        H, W = frame.shape[:2]

        roi_w = int(W * 0.50)
        roi_h = int(H * 0.70)
        cx = W // 2
        cy = int(H * 0.40)
        roi_rect = (cx - roi_w // 2, cy - roi_h // 2, cx + roi_w // 2, cy + roi_h // 2)
        x0, y0, x1, y1 = roi_rect
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 0), 2)

        roi = frame[y0:y1, x0:x1]
        results = model.predict(source=roi, conf=0.45, verbose=False)
        detected = False

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                label = model.names.get(cls_id, str(cls_id))
                if any(k in label.lower() for k in ["car", "plate", "license", "truck", "vehicle"]):
                    detected = True
                    xA, yA, xB, yB = box.xyxy[0].int().tolist()
                    xA += x0
                    yA += y0
                    xB += x0
                    yB += y0
                    cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf_val:.2f}", (xA, max(yA - 5, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        msg = "PLACA DETECTADA" if detected else "SIN DETECCION"
        color = (0, 200, 0) if detected else (0, 0, 255)
        cv2.rectangle(frame, (0, 0), (W, 35), (0, 0, 0), -1)
        cv2.putText(frame, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("RTSP Live (YOLOv8 - Estable)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Transmision finalizada.")


if __name__ == "__main__":
    main()
