#!/usr/bin/env python3

"""Detección básica de autos en RTSP con YOLOv8.

Demo mínima que reutiliza common/frames (reconexión robusta) y baja latencia.
Requiere: pip install ultralytics opencv-python
"""

import os
import time

import cv2
from ultralytics import YOLO

from common.config import env_value
from common.frames import FrameGrabber
from common.rtsp import build_streaming_url
from common.utils import set_ffmpeg_low_latency_env

RTSP_USER = os.getenv("RTSP_USER", "admin")
RTSP_PASS = env_value("RTSP_PASSWORD", "RTSP_PASS")
RTSP_HOST = os.getenv("RTSP_HOST", "192.168.1.64")
RTSP_CHANNEL = os.getenv("RTSP_CHANNEL", "101")
CONFIDENCE = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))

set_ffmpeg_low_latency_env("tcp")


def main():
    model = YOLO("yolov8n.pt")  # descarga automática la red pequeña (yolov8n)
    source = build_streaming_url(RTSP_HOST, RTSP_USER, RTSP_PASS, RTSP_CHANNEL)
    grabber = FrameGrabber(source, name="basic")

    fps_time = time.time()
    try:
        while True:
            ok, frame = grabber.read()
            if not ok or frame is None:
                time.sleep(0.1)
                continue

            # inferencia: returns a list con un objeto Result para cada frame
            results = model.predict(frame, imgsz=640, conf=CONFIDENCE, verbose=False)

            r = results[0]
            boxes = getattr(r, "boxes", None)
            names = model.names  # diccionario id->nombre clase (p.ej. 'car')

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls)  # id de clase
                    cls_name = names.get(cls_id, str(cls_id))
                    conf = float(box.conf)
                    if cls_name == "car" and conf >= CONFIDENCE:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"{cls_name} {conf:.2f}",
                            (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )

            # mostrar FPS
            dt = time.time() - fps_time
            fps = 1 / dt if dt > 0 else 0
            fps_time = time.time()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Deteccion Autos", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
                break
    finally:
        grabber.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()