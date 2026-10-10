#!/usr/bin/env python3

"""Captura ISAPI (baja latencia).

Detección de vehículos en tiempo real sobre el sub-stream RTSP y disparo de
snapshot ISAPI (main-stream) solo cuando el vehículo está COMPLETO dentro del
ROI. La captura se lanza en hilo aparte (no bloquea el loop).

Uso:
  python3 capture_isapi.py --host 192.168.1.64 --user admin --password 'TuClave' \
      --rtsp-channel 102 --snapshot-channel 101 --model yolov8n.pt
"""

import argparse
import logging
import os
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

from common import cli
from common.detector import VehicleDetector
from common.frames import FrameGrabberLatest
from common.geometry import compute_roi
from common.rtsp import build_isapi_channel_url, build_isapi_url
from common.utils import set_ffmpeg_low_latency_env, setup_logging

logger = logging.getLogger(__name__)
WINDOW = "RTSP Live (YOLOv8 - ISAPI)"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RTSP baja latencia + YOLOv8 + captura ISAPI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cli.add_camera_arguments(parser)
    cli.add_detection_arguments(parser)
    cli.add_roi_arguments(parser)
    cli.add_verbosity_flag(parser)
    args = parser.parse_args()
    setup_logging(args.verbose)

    camera = cli.camera_config_from_args(args)
    detection = cli.detection_config_from_args(args)
    roi_config = cli.roi_config_from_args(args)

    if not camera.password:
        logger.error("Define RTSP_PASSWORD en .env o pasa --password")
        return 1
    if not os.path.exists(detection.model_path):
        logger.error("Modelo no encontrado: %s", detection.model_path)
        return 1

    set_ffmpeg_low_latency_env(camera.rtsp_transport)

    logger.info("Cargando modelo: %s", detection.model_path)
    model = YOLO(detection.model_path)

    builder = build_isapi_channel_url if args.substream_suffix else build_isapi_url
    rtsp_url = builder(camera.host, camera.user, camera.password, camera.rtsp_channel)

    logger.info(
        "Conectando RTSP canal %s: user=%s, host=%s", camera.rtsp_channel, camera.user, camera.host
    )
    grab = FrameGrabberLatest(rtsp_url, width=camera.width, height=camera.height, name="sub")
    detector = VehicleDetector(
        model,
        conf=detection.confidence,
        iou=detection.iou_threshold,
        cooldown=detection.cooldown,
        device=detection.device,
        imgsz=detection.imgsz,
        infer_every_n=detection.infer_every_n,
    )

    logger.info("Transmisión iniciada. Presiona 'q' para salir.")

    try:
        while True:
            ok, frame = grab.read()
            if not ok or frame is None:
                blank = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
                cv2.putText(
                    blank,
                    "Reintentando conexión RTSP...",
                    (60, camera.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )
                cv2.imshow(WINDOW, blank)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                time.sleep(0.05)
                continue

            H, W = frame.shape[:2]
            roi = compute_roi(
                W,
                H,
                roi_w=roi_config.width_ratio,
                roi_h=roi_config.height_ratio,
                roi_cy=roi_config.center_y_ratio,
            )
            cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 255, 0), 2)

            frame, detected, trigger_snapshot = detector.detect(frame, roi)

            if trigger_snapshot:
                detector.trigger_if_ready(
                    camera.host,
                    camera.user,
                    camera.password,
                    camera.snapshot_channel,
                    folder="isapi_snaps",
                    timeout=4,
                )

            msg = "VEHICULO DETECTADO" if detected else "SIN DETECCION"
            color = (0, 200, 0) if detected else (0, 0, 255)
            cv2.rectangle(frame, (0, 0), (W, 35), (0, 0, 0), -1)
            cv2.putText(frame, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        logger.info("Interrupción por usuario (Ctrl+C)")
    finally:
        grab.release()
        cv2.destroyAllWindows()
        logger.info("Transmisión finalizada correctamente.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
