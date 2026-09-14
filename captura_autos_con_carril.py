#!/usr/bin/env python3

"""
RTSP + YOLOv8 + Detección de Carretera + Captura PRE-ROLL (v2.1 - BUG FIXES)
-----------------------------------------------------------------------------
Correcciones:
- Fix ventana OpenCV en loop
- Mejor manejo de frames None
- Inicialización robusta de streams
- Timeout de espera mejorado
"""

import argparse
import logging
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO

from common.config import CameraConfig, DetectionConfig, ROIConfig
from common.frames import FrameGrabberBuffer, FrameGrabberLatest
from common.geometry import box_inside_roi
from common.lanes import LaneDetector
from common.rtsp import build_isapi_url
from common.utils import save_jpeg, set_ffmpeg_low_latency_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class VehicleDetectionSystem:
    """Sistema de detección - VERSIÓN CORREGIDA"""

    def __init__(
        self,
        camera_config: CameraConfig,
        detection_config: DetectionConfig,
        roi_config: ROIConfig,
        enable_lane_detection: bool = True,
    ):
        self.camera_config = camera_config
        self.detection_config = detection_config
        self.roi_config = roi_config
        self.enable_lane_detection = enable_lane_detection

        logger.info(f"Cargando modelo YOLO: {detection_config.model_path}")
        self.model = YOLO(detection_config.model_path)

        if self.enable_lane_detection:
            self.lane_detector = LaneDetector()
            logger.info("Detector de carriles inicializado")

        self.sub_rtsp_url = build_isapi_url(
            camera_config.host, camera_config.user, camera_config.password, camera_config.rtsp_channel
        )
        self.main_rtsp_url = build_isapi_url(
            camera_config.host, camera_config.user, camera_config.password, camera_config.snapshot_channel
        )

        logger.info(f"Sub-stream: {self.sub_rtsp_url}")
        logger.info(f"Main-stream: {self.main_rtsp_url}")

        # FIX: Inicializar grabbers de forma síncrona
        logger.info("Inicializando streams...")
        self.grab_sub = FrameGrabberLatest(
            self.sub_rtsp_url,
            width=camera_config.width,
            height=camera_config.height,
            name="sub",
        )

        self.grab_main = FrameGrabberBuffer(
            self.main_rtsp_url,
            max_seconds=1.5,
            fps_hint=25,
            width=camera_config.width,
            height=camera_config.height,
            name="main",
        )

        self.last_capture_time = 0.0
        self.stats = {
            "frames_processed": 0,
            "detections": 0,
            "captures": 0,
            "start_time": time.time(),
        }

        logger.info("Sistema inicializado correctamente")

    def _calculate_roi(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Calcula ROI."""
        roi_w = int(width * self.roi_config.width_ratio)
        roi_h = int(height * self.roi_config.height_ratio)
        cx = int(width * self.roi_config.center_x_ratio)
        cy = int(height * self.roi_config.center_y_ratio)
        return (cx - roi_w // 2, cy - roi_h // 2, cx + roi_w // 2, cy + roi_h // 2)

    def _draw_roi(self, frame: np.ndarray, roi: tuple[int, int, int, int]) -> None:
        """Dibuja ROI."""
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, "ROI", (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    def _detect_vehicles(self, frame: np.ndarray, roi: tuple[int, int, int, int]) -> tuple[bool, bool]:
        """Detecta vehículos."""
        x1, y1, x2, y2 = roi
        roi_frame = frame[y1:y2, x1:x2]

        results = self.model.predict(
            source=roi_frame,
            conf=self.detection_config.confidence,
            iou=self.detection_config.iou_threshold,
            verbose=False,
        )

        detected = False
        trigger_capture = False

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.names.get(cls_id, str(cls_id))

                vehicle_keywords = ["car", "vehicle", "truck", "bus", "motorbike", "motorcycle", "bicycle"]
                if any(keyword in label.lower() for keyword in vehicle_keywords):
                    detected = True

                    xA, yA, xB, yB = box.xyxy[0].int().tolist()
                    xA += x1
                    yA += y1
                    xB += x1
                    yB += y1

                    cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
                    text = f"{label} {conf:.2f}"
                    cv2.putText(frame, text, (xA, max(yA - 5, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                    if box_inside_roi((xA, yA, xB, yB), roi):
                        trigger_capture = True

        if detected:
            self.stats["detections"] += 1

        return detected, trigger_capture

    def _async_capture_preroll(self) -> None:
        """Captura con pre-roll."""
        current_time = time.time()
        target_time = current_time - (self.detection_config.pre_roll_ms / 1000.0)

        frame = self.grab_main.get_closest_frame(target_time)

        if frame is not None:
            save_jpeg(frame, folder="captures_preroll", prefix="main")
            self.stats["captures"] += 1
        else:
            logger.warning("Buffer vacío")
            ok, fallback_frame = self.grab_sub.read()
            if ok and fallback_frame is not None:
                save_jpeg(fallback_frame, folder="captures_fallback", prefix="sub")

    def _draw_info_panel(self, frame: np.ndarray, detected: bool, fps: float, buffer_info: dict) -> None:
        """Dibuja panel de info."""
        height, width = frame.shape[:2]
        panel_height = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        status = "VEHÍCULO DETECTADO" if detected else "SIN DETECCIÓN"
        color = (0, 255, 0) if detected else (100, 100, 100)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        info_text = f"FPS: {fps:.1f} | Det: {self.stats['detections']} | Cap: {self.stats['captures']}"
        cv2.putText(frame, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        buffer_text = (
            f"Buffer: {buffer_info.get('size', 0)}/{buffer_info.get('max_size', 0)} "
            f"({buffer_info.get('time_span', 0):.2f}s)"
        )
        cv2.putText(frame, buffer_text, (width - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def run(self) -> None:
        """Loop principal - VERSIÓN CORREGIDA."""
        logger.info("Sistema iniciado. Presiona 'q' para salir.")

        # FIX: Verificar que ambos streams estén listos
        logger.info("Verificando streams...")
        timeout = 15
        start_wait = time.time()

        while (time.time() - start_wait) < timeout:
            ok_sub, _ = self.grab_sub.read()
            buffer_info = self.grab_main.get_buffer_info()

            if ok_sub and buffer_info.get("size", 0) > 0:
                logger.info("✓ Ambos streams operativos")
                break

            logger.info(f"Esperando streams... ({int(time.time() - start_wait)}s)")
            time.sleep(1)
        else:
            logger.error("Timeout esperando streams. Verifica la configuración RTSP.")
            self.cleanup()
            return

        # FIX: Crear ventana una sola vez ANTES del loop
        window_name = "Sistema Detección Vehículos + Carriles"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0.0

        try:
            while True:
                ok, frame = self.grab_sub.read()

                # FIX: Manejo robusto de frames None
                if not ok or frame is None:
                    logger.warning("Frame no disponible")
                    blank = np.zeros((self.camera_config.height, self.camera_config.width, 3), dtype=np.uint8)
                    cv2.putText(
                        blank,
                        "Reconectando...",
                        (60, self.camera_config.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3,
                    )
                    cv2.imshow(window_name, blank)

                    key = cv2.waitKey(100) & 0xFF  # FIX: Esperar más tiempo
                    if key == ord("q"):
                        break
                    continue

                self.stats["frames_processed"] += 1
                height, width = frame.shape[:2]

                roi = self._calculate_roi(width, height)
                detected, trigger_capture = self._detect_vehicles(frame, roi)

                if self.enable_lane_detection:
                    try:
                        left_lane, right_lane = self.lane_detector.detect(frame)
                        frame = self.lane_detector.draw_lanes(frame, left_lane, right_lane)
                    except Exception as e:
                        logger.error(f"Error en detección de carriles: {e}")

                current_time = time.time()
                if trigger_capture and (current_time - self.last_capture_time) > self.detection_config.cooldown:
                    self.last_capture_time = current_time
                    threading.Thread(target=self._async_capture_preroll, daemon=True).start()

                self._draw_roi(frame, roi)
                buffer_info = self.grab_main.get_buffer_info()

                fps_counter += 1
                if fps_counter >= 30:
                    current_fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()

                self._draw_info_panel(frame, detected, current_fps, buffer_info)

                # FIX: Actualizar ventana existente
                cv2.imshow(window_name, frame)

                # FIX: waitKey más corto para mejor responsividad
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Saliendo por solicitud del usuario")
                    break

        except KeyboardInterrupt:
            logger.info("Interrupción de usuario (Ctrl+C)")
        except Exception as e:
            logger.error(f"Error inesperado: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Limpieza de recursos."""
        logger.info("Liberando recursos...")

        self.grab_sub.release()
        self.grab_main.release()
        cv2.destroyAllWindows()

        runtime = time.time() - self.stats["start_time"]
        logger.info("=" * 60)
        logger.info("ESTADÍSTICAS FINALES")
        logger.info("=" * 60)
        logger.info(f"Tiempo ejecución: {runtime:.2f}s")
        logger.info(f"Frames procesados: {self.stats['frames_processed']}")
        logger.info(f"Detecciones: {self.stats['detections']}")
        logger.info(f"Capturas: {self.stats['captures']}")
        if runtime > 0:
            logger.info(f"FPS promedio: {self.stats['frames_processed'] / runtime:.2f}")
        logger.info("=" * 60)


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    """Parsea argumentos."""
    parser = argparse.ArgumentParser(
        description="Sistema detección vehículos + carriles (v2.1 - Bug Fixes)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    cam_group = parser.add_argument_group("Cámara")
    cam_group.add_argument("--host", default="192.168.1.64", help="IP cámara")
    cam_group.add_argument("--user", default="admin", help="Usuario")
    cam_group.add_argument("--password", required=True, help="Contraseña")
    cam_group.add_argument("--rtsp_channel", default="102", help="Canal sub-stream")
    cam_group.add_argument("--snapshot_channel", default="101", help="Canal main-stream")
    cam_group.add_argument("--width", type=int, default=1280, help="Ancho")
    cam_group.add_argument("--height", type=int, default=720, help="Alto")
    cam_group.add_argument("--rtsp_transport", default="udp", choices=["udp", "tcp"], help="Transporte RTSP")

    det_group = parser.add_argument_group("Detección")
    det_group.add_argument("--model", default="yolov8n.pt", help="Modelo YOLO")
    det_group.add_argument("--conf", type=float, default=0.45, help="Confianza mínima")
    det_group.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    det_group.add_argument("--cooldown", type=float, default=0.8, help="Cooldown (s)")
    det_group.add_argument("--pre_roll_ms", type=int, default=300, help="Pre-roll (ms)")

    roi_group = parser.add_argument_group("ROI")
    roi_group.add_argument("--roi_width_ratio", type=float, default=0.50)
    roi_group.add_argument("--roi_height_ratio", type=float, default=0.70)
    roi_group.add_argument("--roi_center_x_ratio", type=float, default=0.50)
    roi_group.add_argument("--roi_center_y_ratio", type=float, default=0.40)

    opt_group = parser.add_argument_group("Opciones")
    opt_group.add_argument("--disable_lane_detection", action="store_true", help="Deshabilitar detección carriles")
    opt_group.add_argument("--verbose", action="store_true", help="Modo verbose")

    return parser.parse_args()


def main() -> None:
    """Función principal."""
    args = parse_arguments()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    set_ffmpeg_low_latency_env(args.rtsp_transport)

    camera_config = CameraConfig(
        host=args.host,
        user=args.user,
        password=args.password,
        rtsp_channel=args.rtsp_channel,
        snapshot_channel=args.snapshot_channel,
        width=args.width,
        height=args.height,
        rtsp_transport=args.rtsp_transport,
    )

    detection_config = DetectionConfig(
        model_path=args.model,
        confidence=args.conf,
        iou_threshold=args.iou,
        cooldown=args.cooldown,
        pre_roll_ms=args.pre_roll_ms,
    )

    roi_config = ROIConfig(
        width_ratio=args.roi_width_ratio,
        height_ratio=args.roi_height_ratio,
        center_x_ratio=args.roi_center_x_ratio,
        center_y_ratio=args.roi_center_y_ratio,
    )

    system = VehicleDetectionSystem(
        camera_config=camera_config,
        detection_config=detection_config,
        roi_config=roi_config,
        enable_lane_detection=not args.disable_lane_detection,
    )

    system.run()


if __name__ == "__main__":
    main()