#!/usr/bin/env python3

"""Captura PRE-ROLL con selección de frame nítido.

Detecta en el sub-stream y guarda desde el main-stream un frame "un poco
ANTES" del trigger (buffer circular). Opciones avanzadas heredadas:

- ``--sharp-window-ms``: elige el frame MÁS NÍTIDO dentro de la ventana.
- ``--stable-frames``/``--rearm-frames``: solo dispara con vehículo ESTABLE
  dentro del ROI (evita disparos espurios de un solo frame).
- ``--road-poly``/``--filter-to-road``: ignora vehículos fuera de la carretera.
- ``--fallback-isapi``: si el buffer MAIN está vacío, intenta snapshot ISAPI.

Uso:
  python3 capture_preroll.py --host 192.168.1.64 --user admin --password 'TuClave' \
      --rtsp-channel 102 --snapshot-channel 101 --model yolov8n.pt --pre-roll-ms 300
"""

import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO

from common import cli
from common.detector import BoxFilter, VehicleDetector
from common.frames import FrameGrabberBuffer, FrameGrabberLatest, best_frame
from common.geometry import draw_polygon, parse_polygon, point_inside_polygon
from common.isapi import save_isapi_snapshot
from common.rtsp import build_isapi_url
from common.utils import StableTrigger, save_jpeg, set_ffmpeg_low_latency_env

WINDOW = "Live (YOLOv8 LPR pre-roll)"


def road_filter_for(
    poly_str: str, width: int, height: int
) -> tuple[np.ndarray | None, BoxFilter | None]:
    """Convierte ``--road-poly`` en (polígono dibujable, box_filter).

    El ``box_filter`` descarta vehículos cuyo centro está fuera del polígono.
    Devuelve ``(None, None)`` si no hay polígono válido.
    """
    poly = parse_polygon(poly_str, width, height)
    if poly is None:
        return None, None

    def _inside_road(box: tuple[int, int, int, int], cls_id: int) -> bool:
        x1, y1, x2, y2 = box
        return point_inside_polygon(((x1 + x2) // 2, (y1 + y2) // 2), poly)

    return poly, _inside_road


def async_capture(
    grab_main: FrameGrabberBuffer,
    now: float,
    args: argparse.Namespace,
    camera,
    detection,
) -> None:
    """Guarda el frame más nítido del buffer, con fallback ISAPI."""
    target_ts = now - (detection.pre_roll_ms / 1000.0)
    frames = grab_main.get_frames_in_window(target_ts, args.sharp_window_ms)
    best = best_frame(
        frames,
        target_ts,
        args.sharp_window_ms,
        min_frames=args.sharp_min_frames,
        max_frames=args.sharp_max_frames,
    )
    if best is not None:
        save_jpeg(best, folder=args.save_dir, prefix="preroll_best")
        return
    if args.fallback_isapi:
        print("[WARN] Buffer MAIN sin frames suficientes. Fallback ISAPI...")
        save_isapi_snapshot(
            camera.host,
            camera.user,
            camera.password,
            folder="isapi_snaps",
            channel=camera.snapshot_channel,
            timeout=4,
        )
    else:
        print("[WARN] Buffer MAIN sin frames suficientes.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RTSP baja latencia + YOLOv8 + PRE-ROLL desde MAIN (frame más nítido)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cli.add_camera_arguments(parser)
    cli.add_detection_arguments(parser)
    cli.add_roi_arguments(parser)

    pr = parser.add_argument_group("Pre-roll")
    pr.add_argument(
        "--pre-roll-ms", type=int, default=300, dest="pre_roll_ms", help="ms ANTES del trigger"
    )
    pr.add_argument("--save-dir", default="captures_preroll", dest="save_dir")
    pr.add_argument("--sharp-window-ms", type=int, default=200, dest="sharp_window_ms")
    pr.add_argument("--sharp-min-frames", type=int, default=1, dest="sharp_min_frames")
    pr.add_argument("--sharp-max-frames", type=int, default=9, dest="sharp_max_frames")
    pr.add_argument("--stable-frames", type=int, default=3, dest="stable_frames")
    pr.add_argument("--rearm-frames", type=int, default=8, dest="rearm_frames")
    pr.add_argument("--inside-margin-px", type=int, default=0, dest="inside_margin_px")
    pr.add_argument(
        "--fallback-isapi", action="store_true", help="ISAPI si el buffer MAIN está vacío"
    )
    pr.add_argument(
        "--road-poly", default="", dest="road_poly", help='Polígono "x,y;x,y;..." (fracciones)'
    )
    pr.add_argument("--filter-to-road", action="store_true", dest="filter_to_road")
    args = parser.parse_args()

    camera = cli.camera_config_from_args(args)
    detection = cli.detection_config_from_args(args)
    roi_config = cli.roi_config_from_args(args)

    if not camera.password:
        print("[ERROR] Define RTSP_PASSWORD en .env o pasa --password")
        return 1
    if not os.path.exists(detection.model_path):
        print(f"[ERROR] Modelo no encontrado: {detection.model_path}")
        return 1

    set_ffmpeg_low_latency_env(camera.rtsp_transport)

    print(f"[INFO] Cargando modelo: {detection.model_path}")
    model = YOLO(detection.model_path)

    sub_url = build_isapi_url(camera.host, camera.user, camera.password, camera.rtsp_channel)
    main_url = build_isapi_url(camera.host, camera.user, camera.password, camera.snapshot_channel)

    grab_sub = FrameGrabberLatest(sub_url, width=camera.width, height=camera.height, name="sub")
    grab_main = FrameGrabberBuffer(
        main_url,
        max_seconds=1.5,
        fps_hint=25,
        width=camera.width,
        height=camera.height,
        name="main",
    )
    detector = VehicleDetector(
        model,
        conf=detection.confidence,
        iou=detection.iou_threshold,
        cooldown=detection.cooldown,
        device=detection.device,
        imgsz=detection.imgsz,
        infer_every_n=detection.infer_every_n,
        margin=args.inside_margin_px,
    )

    road_poly = None
    stable = StableTrigger(stable_frames=args.stable_frames, rearm_frames=args.rearm_frames)
    last_capture_ts = 0.0

    print("[INFO] Transmisión iniciada. Presiona 'q' para salir.")
    try:
        while True:
            ok, frame = grab_sub.read()
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
                time.sleep(0.03)
                continue

            H, W = frame.shape[:2]
            if road_poly is None and args.filter_to_road and args.road_poly:
                road_poly, road_filter = road_filter_for(args.road_poly, W, H)
                if road_filter is not None:
                    detector.box_filter = road_filter
            if road_poly is not None:
                draw_polygon(frame, road_poly)

            cx, cy = W // 2, int(H * roi_config.center_y_ratio)
            rw, rh = int(W * roi_config.width_ratio), int(H * roi_config.height_ratio)
            x0 = max(0, cx - rw // 2)
            y0 = max(0, cy - rh // 2)
            x1 = min(W, x0 + rw)
            y1 = min(H, y0 + rh)
            roi = (x0, y0, x1, y1)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 0), 2)

            frame, detected, _inside = detector.detect(frame, roi)

            if stable.update(detected):
                now = time.time()
                if (now - last_capture_ts) > detection.cooldown:
                    last_capture_ts = now
                    threading.Thread(
                        target=async_capture,
                        args=(grab_main, now, args, camera, detection),
                        daemon=True,
                    ).start()

            msg = "VEHICULO DETECTADO" if detected else "SIN DETECCION"
            color = (0, 200, 0) if detected else (0, 0, 255)
            cv2.rectangle(frame, (0, 0), (W, 35), (0, 0, 0), -1)
            cv2.putText(frame, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\n[INFO] Interrupción por usuario (Ctrl+C)")
    finally:
        grab_sub.release()
        grab_main.release()
        cv2.destroyAllWindows()
        print("[INFO] Transmisión finalizada correctamente.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
