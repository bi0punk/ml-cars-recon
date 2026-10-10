#!/usr/bin/env python3

"""Captura PRE-ROLL con detección de carriles y panel de estadísticas.

Detecta vehículos con YOLO en el sub-stream, dibuja carriles (Canny + Hough)
opcionalmente, y guarda el frame del main-stream ANTERIOR al disparo. Incluye
panel de estadísticas en pantalla (FPS, detecciones, capturas, buffer).

Uso:
  python3 capture_preroll_lanes.py --host 192.168.1.64 --user admin --password 'TuClave' \
      --rtsp-channel 102 --snapshot-channel 101 --model yolov8n.pt
"""

import argparse
import logging
import os
import sys
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO

from common import cli
from common.detector import VehicleDetector
from common.frames import FrameGrabberBuffer, FrameGrabberLatest
from common.geometry import compute_roi
from common.lanes import LaneDetector
from common.rtsp import build_isapi_url
from common.utils import save_jpeg, set_ffmpeg_low_latency_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class PreRollLanesSystem:
    """Detección + carriles + captura pre-roll con estadísticas en vivo."""

    def __init__(self, camera, detection, roi_config, enable_lane_detection: bool = True):
        self.camera = camera
        self.detection = detection
        self.roi_config = roi_config
        self.enable_lane_detection = enable_lane_detection

        logger.info("Cargando modelo YOLO: %s", detection.model_path)
        self.model = YOLO(detection.model_path)
        self.detector = VehicleDetector(
            self.model,
            conf=detection.confidence,
            iou=detection.iou_threshold,
            cooldown=detection.cooldown,
            device=detection.device,
            imgsz=detection.imgsz,
            infer_every_n=detection.infer_every_n,
        )
        self.lane_detector = LaneDetector() if enable_lane_detection else None

        self.grab_sub = FrameGrabberLatest(
            build_isapi_url(camera.host, camera.user, camera.password, camera.rtsp_channel),
            width=camera.width,
            height=camera.height,
            name="sub",
        )
        self.grab_main = FrameGrabberBuffer(
            build_isapi_url(camera.host, camera.user, camera.password, camera.snapshot_channel),
            max_seconds=1.5,
            fps_hint=25,
            width=camera.width,
            height=camera.height,
            name="main",
        )

        self.stats = {
            "frames_processed": 0,
            "detections": 0,
            "captures": 0,
            "start_time": time.time(),
        }
        self._last_capture_ts = 0.0
        logger.info("Sistema inicializado correctamente")

    def _roi(self, w: int, h: int) -> tuple[int, int, int, int]:
        return compute_roi(
            w,
            h,
            roi_w=self.roi_config.width_ratio,
            roi_h=self.roi_config.height_ratio,
            roi_cy=self.roi_config.center_y_ratio,
        )

    def _capture_preroll(self) -> None:
        """Captura con pre-roll desde el main-stream (fallback a sub-stream)."""
        target_ts = time.time() - (self.detection.pre_roll_ms / 1000.0)
        frame = self.grab_main.get_closest_frame(target_ts)
        if frame is not None:
            save_jpeg(frame, folder="captures_preroll", prefix="main")
            self.stats["captures"] += 1
        else:
            logger.warning("Buffer MAIN vacío; usando sub-stream como fallback")
            ok, fallback = self.grab_sub.read()
            if ok and fallback is not None:
                save_jpeg(fallback, folder="captures_fallback", prefix="sub")

    def _maybe_capture(self, trigger_capture: bool) -> bool:
        """Dispara la captura pre-roll SOLO si un vehículo está completo dentro
        del ROI y respeta el cooldown. Retorna ``True`` si se disparó."""
        now = time.time()
        if trigger_capture and (now - self._last_capture_ts) > self.detection.cooldown:
            self._last_capture_ts = now
            threading.Thread(target=self._capture_preroll, daemon=True).start()
            return True
        return False

    def _draw_info_panel(self, frame, detected: bool, fps: float, buffer_info: dict) -> None:
        height, width = frame.shape[:2]
        panel_height = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        status = "VEHICULO DETECTADO" if detected else "SIN DETECCION"
        color = (0, 255, 0) if detected else (100, 100, 100)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        info = f"FPS: {fps:.1f} | Det: {self.stats['detections']} | Cap: {self.stats['captures']}"
        cv2.putText(frame, info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        buffer_text = (
            f"Buffer: {buffer_info.get('size', 0)}/{buffer_info.get('max_size', 0)} "
            f"({buffer_info.get('time_span', 0):.2f}s)"
        )
        cv2.putText(
            frame, buffer_text, (width - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )

    def run(self) -> None:
        window_name = "Sistema Detección Vehículos + Carriles"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        fps_counter = 0
        fps_start = time.time()
        current_fps = 0.0

        logger.info("Sistema iniciado. Presiona 'q' para salir.")
        try:
            while True:
                ok, frame = self.grab_sub.read()
                if not ok or frame is None:
                    logger.warning("Frame no disponible")
                    blank = np.zeros((self.camera.height, self.camera.width, 3), dtype=np.uint8)
                    cv2.putText(
                        blank,
                        "Reconectando...",
                        (60, self.camera.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3,
                    )
                    cv2.imshow(window_name, blank)
                    if cv2.waitKey(100) & 0xFF == ord("q"):
                        break
                    continue

                self.stats["frames_processed"] += 1
                H, W = frame.shape[:2]
                roi = self._roi(W, H)
                cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 255, 0), 2)

                frame, detected, trigger_capture = self.detector.detect(frame, roi)
                if detected:
                    self.stats["detections"] += 1

                if self.lane_detector is not None:
                    try:
                        left_lane, right_lane = self.lane_detector.detect(frame)
                        frame = self.lane_detector.draw_lanes(frame, left_lane, right_lane)
                    except Exception as exc:
                        logger.error("Error en detección de carriles: %s", exc)

                self._maybe_capture(trigger_capture)

                fps_counter += 1
                if fps_counter >= 30:
                    current_fps = fps_counter / (time.time() - fps_start)
                    fps_counter = 0
                    fps_start = time.time()

                self._draw_info_panel(
                    frame, detected, current_fps, self.grab_main.get_buffer_info()
                )
                cv2.imshow(window_name, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Saliendo por solicitud del usuario")
                    break
        except KeyboardInterrupt:
            logger.info("Interrupción de usuario (Ctrl+C)")
        except Exception as exc:
            logger.error("Error inesperado: %s", exc, exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        logger.info("Liberando recursos...")
        self.grab_sub.release()
        self.grab_main.release()
        cv2.destroyAllWindows()

        runtime = max(time.time() - self.stats["start_time"], 0.001)
        logger.info("=" * 60)
        logger.info("ESTADISTICAS FINALES")
        logger.info("=" * 60)
        logger.info(f"Tiempo ejecución: {runtime:.2f}s")
        logger.info(f"Frames procesados: {self.stats['frames_processed']}")
        logger.info(f"Detecciones: {self.stats['detections']}")
        logger.info(f"Capturas: {self.stats['captures']}")
        logger.info(f"FPS promedio: {self.stats['frames_processed'] / runtime:.2f}")
        logger.info("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sistema detección vehículos + carriles + captura pre-roll",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cli.add_camera_arguments(parser)
    cli.add_detection_arguments(parser)
    cli.add_roi_arguments(parser)
    parser.add_argument(
        "--pre-roll-ms", type=int, default=300, dest="pre_roll_ms", help="ms ANTES del trigger"
    )
    parser.add_argument(
        "--disable-lane-detection", action="store_true", dest="disable_lane_detection"
    )
    parser.add_argument("--verbose", action="store_true", help="Log nivel DEBUG")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    camera = cli.camera_config_from_args(args)
    detection = cli.detection_config_from_args(args)
    roi_config = cli.roi_config_from_args(args)

    if not camera.password or not os.path.exists(detection.model_path):
        logger.error(
            "Faltan credenciales (RTSP_PASSWORD/--password) o el modelo %s no existe",
            detection.model_path,
        )
        return 1

    set_ffmpeg_low_latency_env(camera.rtsp_transport)
    system = PreRollLanesSystem(
        camera,
        detection,
        roi_config,
        enable_lane_detection=not args.disable_lane_detection,
    )
    system.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
