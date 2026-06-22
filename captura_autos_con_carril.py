#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RTSP + YOLOv8 + Detección de Carretera + Captura PRE-ROLL (v2.1 - BUG FIXES)
-----------------------------------------------------------------------------
Correcciones:
- Fix ventana OpenCV en loop
- Mejor manejo de frames None
- Inicialización robusta de streams
- Timeout de espera mejorado
"""

import os
import cv2
import time
import argparse
import logging
import numpy as np
import threading
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime
from ultralytics import YOLO

# =============================================================================
# CONFIGURACIÓN DE LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# DATACLASSES PARA CONFIGURACIÓN
# =============================================================================

@dataclass
class CameraConfig:
    """Configuración de cámara RTSP"""
    host: str
    user: str
    password: str
    rtsp_channel: str
    snapshot_channel: str
    width: int = 1280
    height: int = 720
    rtsp_transport: str = "udp"

@dataclass
class DetectionConfig:
    """Configuración de detección"""
    model_path: str
    confidence: float = 0.45
    iou_threshold: float = 0.5
    cooldown: float = 0.8
    pre_roll_ms: int = 300
    
@dataclass
class ROIConfig:
    """Región de interés"""
    width_ratio: float = 0.50
    height_ratio: float = 0.70
    center_x_ratio: float = 0.50
    center_y_ratio: float = 0.40

# =============================================================================
# FFmpeg: CONFIGURACIÓN DE BAJA LATENCIA
# =============================================================================

def configure_ffmpeg_low_latency(transport: str = "udp") -> None:
    """Configura OpenCV+FFmpeg para mínima latencia en RTSP."""
    transport = transport.lower().strip()
    if transport not in ("udp", "tcp"):
        logger.warning(f"Transporte '{transport}' inválido. Usando UDP.")
        transport = "udp"
    
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"rtsp_transport;{transport}|"
        "max_delay;0|"
        "stimeout;5000000|"
        "buffer_size;0|"
        "fflags;nobuffer|"
        "flags;low_delay|"
        "reorder_queue_size;0"
    )
    logger.info(f"FFmpeg configurado: transporte={transport.upper()}")

# =============================================================================
# DETECCIÓN DE CARRILES (LANE DETECTION)
# =============================================================================

class LaneDetector:
    """Detector de carriles usando Canny + Hough Transform"""
    
    def __init__(self, roi_vertices: Optional[np.ndarray] = None):
        self.roi_vertices = roi_vertices
        self.canny_low = 50
        self.canny_high = 150
        self.hough_rho = 2
        self.hough_theta = np.pi / 180
        self.hough_threshold = 50
        self.hough_min_line_len = 40
        self.hough_max_line_gap = 100
        
    def _roi_mask(self, img: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        """Aplica máscara de ROI"""
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, vertices, 255)
        return cv2.bitwise_and(img, mask)
    
    def _calculate_slope_intercept(self, line: np.ndarray) -> Tuple[float, float]:
        """Calcula pendiente e intercepto"""
        x1, y1, x2, y2 = line
        if x2 - x1 == 0:
            return float('inf'), 0
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return slope, intercept
    
    def _separate_lines(self, lines: np.ndarray) -> Tuple[List, List]:
        """Separa líneas izquierda/derecha"""
        left_lines = []
        right_lines = []
        
        if lines is None:
            return left_lines, right_lines
        
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            if abs(y2 - y1) < 10:
                continue
            slope, intercept = self._calculate_slope_intercept(line.reshape(4))
            if abs(slope) < 0.3 or abs(slope) > 3:
                continue
            if slope < 0:
                left_lines.append((slope, intercept))
            else:
                right_lines.append((slope, intercept))
        
        return left_lines, right_lines
    
    def _average_lines(self, lines: List, y1: int, y2: int) -> Optional[np.ndarray]:
        """Promedia líneas"""
        if not lines:
            return None
        slopes = np.array([line[0] for line in lines])
        intercepts = np.array([line[1] for line in lines])
        avg_slope = np.mean(slopes)
        avg_intercept = np.mean(intercepts)
        x1 = int((y1 - avg_intercept) / avg_slope)
        x2 = int((y2 - avg_intercept) / avg_slope)
        return np.array([x1, y1, x2, y2])
    
    def detect(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detecta carriles"""
        height, width = frame.shape[:2]
        
        if self.roi_vertices is None:
            self.roi_vertices = np.array([[
                (int(width * 0.1), height),
                (int(width * 0.45), int(height * 0.6)),
                (int(width * 0.55), int(height * 0.6)),
                (int(width * 0.9), height)
            ]], dtype=np.int32)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)
        roi_edges = self._roi_mask(edges, self.roi_vertices)
        
        lines = cv2.HoughLinesP(
            roi_edges, self.hough_rho, self.hough_theta, self.hough_threshold,
            minLineLength=self.hough_min_line_len, maxLineGap=self.hough_max_line_gap
        )
        
        if lines is None:
            return None, None
        
        left_lines, right_lines = self._separate_lines(lines)
        y1, y2 = height, int(height * 0.6)
        left_lane = self._average_lines(left_lines, y1, y2)
        right_lane = self._average_lines(right_lines, y1, y2)
        
        return left_lane, right_lane
    
    def draw_lanes(self, frame: np.ndarray, left_lane: Optional[np.ndarray], 
                   right_lane: Optional[np.ndarray]) -> np.ndarray:
        """Dibuja carriles"""
        overlay = frame.copy()
        
        if left_lane is not None:
            x1, y1, x2, y2 = left_lane
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 255), 8)
        
        if right_lane is not None:
            x1, y1, x2, y2 = right_lane
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 255), 8)
        
        if left_lane is not None and right_lane is not None:
            x1_l, y1_l, x2_l, y2_l = left_lane
            x1_r, y1_r, x2_r, y2_r = right_lane
            pts = np.array([[x1_l, y1_l], [x2_l, y2_l], [x2_r, y2_r], [x1_r, y1_r]], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        if left_lane is not None:
            x1, y1, x2, y2 = left_lane
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        
        if right_lane is not None:
            x1, y1, x2, y2 = right_lane
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        
        return frame

# =============================================================================
# UTILIDADES
# =============================================================================

def ensure_directory(path: str) -> Path:
    """Crea directorio si no existe"""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def save_frame_jpeg(frame: np.ndarray, folder: str = "captures", 
                    prefix: str = "frame", quality: int = 95) -> str:
    """Guarda frame como JPEG"""
    ensure_directory(folder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filepath = Path(folder) / f"{timestamp}_{prefix}.jpg"
    cv2.imwrite(str(filepath), frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    logger.info(f"Frame guardado: {filepath}")
    return str(filepath)

def is_box_inside_roi(box: Tuple[int, int, int, int], 
                      roi: Tuple[int, int, int, int]) -> bool:
    """Verifica si bounding box está en ROI"""
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = roi
    return x1 >= rx1 and y1 >= ry1 and x2 <= rx2 and y2 <= ry2

# =============================================================================
# FRAME GRABBERS - VERSIÓN CORREGIDA
# =============================================================================

class FrameGrabberLatest:
    """Grabber con último frame - VERSIÓN CORREGIDA"""
    
    def __init__(self, rtsp_url: str, width: Optional[int] = None, 
                 height: Optional[int] = None, name: str = "sub"):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.name = name
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.ok = False
        self.stopped = False
        self.lock = threading.Lock()
        self.ready = threading.Event()  # FIX: Evento para esperar inicialización
        
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        # FIX: Esperar hasta que tengamos el primer frame
        logger.info(f"[{self.name}] Esperando conexión RTSP...")
        if self.ready.wait(timeout=10):  # Timeout de 10 segundos
            logger.info(f"[{self.name}] FrameGrabberLatest iniciado correctamente")
        else:
            logger.warning(f"[{self.name}] Timeout esperando primer frame")
    
    def _open_stream(self) -> bool:
        """Abre conexión RTSP - retorna True si exitoso"""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        
        logger.info(f"[{self.name}] Conectando a: {self.rtsp_url}")
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        
        is_opened = self.cap.isOpened()
        if is_opened:
            logger.info(f"[{self.name}] Conexión RTSP exitosa")
        else:
            logger.error(f"[{self.name}] Falló conexión RTSP")
        
        return is_opened
    
    def _capture_loop(self) -> None:
        """Loop de captura"""
        reconnect_delay = 2.0
        first_frame_received = False
        
        while not self.stopped:
            # Intentar abrir stream
            if self.cap is None or not self.cap.isOpened():
                if not self._open_stream():
                    time.sleep(reconnect_delay)
                    continue
            
            # Leer frame
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                self.ok = False
                logger.warning(f"[{self.name}] Frame inválido, reintentando...")
                time.sleep(0.05)
                continue
            
            # FIX: Validar que el frame tenga contenido
            if frame.size == 0:
                continue
            
            # Redimensionar si es necesario
            if self.width and self.height:
                try:
                    frame = cv2.resize(frame, (self.width, self.height), 
                                       interpolation=cv2.INTER_AREA)
                except Exception as e:
                    logger.error(f"[{self.name}] Error redimensionando: {e}")
                    continue
            
            # Actualizar frame
            with self.lock:
                self.frame = frame.copy()  # FIX: Hacer copia del frame
                self.ok = True
            
            # FIX: Señalar que estamos listos tras primer frame
            if not first_frame_received:
                first_frame_received = True
                self.ready.set()
                logger.info(f"[{self.name}] Primer frame recibido")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Lee el último frame - FIX: retorna copia"""
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ok, self.frame.copy()
    
    def release(self) -> None:
        """Libera recursos"""
        logger.info(f"[{self.name}] Liberando recursos...")
        self.stopped = True
        try:
            self.thread.join(timeout=3)
        except Exception:
            pass
        if self.cap:
            self.cap.release()
        logger.info(f"[{self.name}] Liberado")


class FrameGrabberBuffer:
    """Grabber con buffer circular - VERSIÓN CORREGIDA"""
    
    def __init__(self, rtsp_url: str, max_seconds: float = 1.5, 
                 fps_hint: int = 25, width: Optional[int] = None, 
                 height: Optional[int] = None, name: str = "main"):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.name = name
        self.max_buffer_size = int(max_seconds * max(fps_hint, 1)) + 5
        self.cap: Optional[cv2.VideoCapture] = None
        self.buffer: deque = deque(maxlen=self.max_buffer_size)
        self.ok = False
        self.stopped = False
        self.lock = threading.Lock()
        self.ready = threading.Event()
        
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        logger.info(f"[{self.name}] Esperando conexión...")
        if self.ready.wait(timeout=10):
            logger.info(f"[{self.name}] Buffer iniciado (size={self.max_buffer_size})")
        else:
            logger.warning(f"[{self.name}] Timeout esperando conexión")
    
    def _open_stream(self) -> bool:
        """Abre stream"""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        
        logger.info(f"[{self.name}] Conectando a: {self.rtsp_url}")
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        
        is_opened = self.cap.isOpened()
        if is_opened:
            logger.info(f"[{self.name}] Conexión exitosa")
        else:
            logger.error(f"[{self.name}] Falló conexión")
        
        return is_opened
    
    def _capture_loop(self) -> None:
        """Loop de captura con buffer"""
        reconnect_delay = 2.0
        first_frame_received = False
        
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                if not self._open_stream():
                    time.sleep(reconnect_delay)
                    continue
            
            ret, frame = self.cap.read()
            
            if not ret or frame is None or frame.size == 0:
                self.ok = False
                time.sleep(0.01)
                continue
            
            timestamp = time.time()
            
            if self.width and self.height:
                try:
                    frame = cv2.resize(frame, (self.width, self.height), 
                                       interpolation=cv2.INTER_AREA)
                except Exception:
                    continue
            
            with self.lock:
                self.buffer.append((timestamp, frame.copy()))
                self.ok = True
            
            if not first_frame_received:
                first_frame_received = True
                self.ready.set()
                logger.info(f"[{self.name}] Buffer recibiendo frames")
    
    def get_closest_frame(self, target_timestamp: float) -> Optional[np.ndarray]:
        """Obtiene frame más cercano al timestamp"""
        with self.lock:
            if not self.buffer:
                return None
            closest = min(self.buffer, key=lambda x: abs(x[0] - target_timestamp))
            return closest[1].copy()
    
    def get_buffer_info(self) -> dict:
        """Info del buffer"""
        with self.lock:
            if not self.buffer:
                return {"size": 0, "time_span": 0, "max_size": self.max_buffer_size}
            return {
                "size": len(self.buffer),
                "max_size": self.max_buffer_size,
                "time_span": self.buffer[-1][0] - self.buffer[0][0]
            }
    
    def release(self) -> None:
        """Libera recursos"""
        logger.info(f"[{self.name}] Liberando...")
        self.stopped = True
        try:
            self.thread.join(timeout=3)
        except Exception:
            pass
        if self.cap:
            self.cap.release()
        logger.info(f"[{self.name}] Liberado")

# =============================================================================
# SISTEMA PRINCIPAL - VERSIÓN CORREGIDA
# =============================================================================

class VehicleDetectionSystem:
    """Sistema de detección - VERSIÓN CORREGIDA"""
    
    def __init__(self, camera_config: CameraConfig, 
                 detection_config: DetectionConfig,
                 roi_config: ROIConfig,
                 enable_lane_detection: bool = True,
                 fallback_isapi: bool = False):
        
        self.camera_config = camera_config
        self.detection_config = detection_config
        self.roi_config = roi_config
        self.enable_lane_detection = enable_lane_detection
        self.fallback_isapi = fallback_isapi
        
        logger.info(f"Cargando modelo YOLO: {detection_config.model_path}")
        self.model = YOLO(detection_config.model_path)
        
        if self.enable_lane_detection:
            self.lane_detector = LaneDetector()
            logger.info("Detector de carriles inicializado")
        
        self.sub_rtsp_url = self._build_rtsp_url(camera_config.rtsp_channel)
        self.main_rtsp_url = self._build_rtsp_url(camera_config.snapshot_channel)
        
        logger.info(f"Sub-stream: {self.sub_rtsp_url}")
        logger.info(f"Main-stream: {self.main_rtsp_url}")
        
        # FIX: Inicializar grabbers de forma síncrona
        logger.info("Inicializando streams...")
        self.grab_sub = FrameGrabberLatest(
            self.sub_rtsp_url,
            width=camera_config.width,
            height=camera_config.height,
            name="sub"
        )
        
        self.grab_main = FrameGrabberBuffer(
            self.main_rtsp_url,
            max_seconds=1.5,
            fps_hint=25,
            width=camera_config.width,
            height=camera_config.height,
            name="main"
        )
        
        self.last_capture_time = 0.0
        self.stats = {
            "frames_processed": 0,
            "detections": 0,
            "captures": 0,
            "start_time": time.time()
        }
        
        logger.info("Sistema inicializado correctamente")
    
    def _build_rtsp_url(self, channel: str) -> str:
        """Construye URL RTSP"""
        return (f"rtsp://{self.camera_config.user}:{self.camera_config.password}@"
                f"{self.camera_config.host}:554/ISAPI/Streaming/channels/{channel}")
    
    def _calculate_roi(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Calcula ROI"""
        roi_w = int(width * self.roi_config.width_ratio)
        roi_h = int(height * self.roi_config.height_ratio)
        cx = int(width * self.roi_config.center_x_ratio)
        cy = int(height * self.roi_config.center_y_ratio)
        return (cx - roi_w // 2, cy - roi_h // 2, cx + roi_w // 2, cy + roi_h // 2)
    
    def _draw_roi(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> None:
        """Dibuja ROI"""
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, "ROI", (x1 + 5, y1 + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def _detect_vehicles(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) \
            -> Tuple[bool, bool]:
        """Detecta vehículos"""
        x1, y1, x2, y2 = roi
        roi_frame = frame[y1:y2, x1:x2]
        
        results = self.model.predict(
            source=roi_frame,
            conf=self.detection_config.confidence,
            iou=self.detection_config.iou_threshold,
            verbose=False
        )
        
        detected = False
        trigger_capture = False
        
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.names.get(cls_id, str(cls_id))
                
                vehicle_keywords = ["car", "vehicle", "truck", "bus", "motorbike", 
                                    "motorcycle", "bicycle"]
                if any(keyword in label.lower() for keyword in vehicle_keywords):
                    detected = True
                    
                    xA, yA, xB, yB = box.xyxy[0].int().tolist()
                    xA += x1
                    yA += y1
                    xB += x1
                    yB += y1
                    
                    cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
                    text = f"{label} {conf:.2f}"
                    cv2.putText(frame, text, (xA, max(yA - 5, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    if is_box_inside_roi((xA, yA, xB, yB), roi):
                        trigger_capture = True
        
        if detected:
            self.stats["detections"] += 1
        
        return detected, trigger_capture
    
    def _async_capture_preroll(self) -> None:
        """Captura con pre-roll"""
        current_time = time.time()
        target_time = current_time - (self.detection_config.pre_roll_ms / 1000.0)
        
        frame = self.grab_main.get_closest_frame(target_time)
        
        if frame is not None:
            save_frame_jpeg(frame, folder="captures_preroll", prefix="main")
            self.stats["captures"] += 1
        else:
            logger.warning("Buffer vacío")
            ok, fallback_frame = self.grab_sub.read()
            if ok and fallback_frame is not None:
                save_frame_jpeg(fallback_frame, folder="captures_fallback", prefix="sub")
    
    def _draw_info_panel(self, frame: np.ndarray, detected: bool, 
                         fps: float, buffer_info: dict) -> None:
        """Dibuja panel de info"""
        height, width = frame.shape[:2]
        panel_height = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        status = "VEHÍCULO DETECTADO" if detected else "SIN DETECCIÓN"
        color = (0, 255, 0) if detected else (100, 100, 100)
        cv2.putText(frame, status, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        info_text = (f"FPS: {fps:.1f} | Det: {self.stats['detections']} | "
                     f"Cap: {self.stats['captures']}")
        cv2.putText(frame, info_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        buffer_text = (f"Buffer: {buffer_info.get('size', 0)}/"
                       f"{buffer_info.get('max_size', 0)} "
                       f"({buffer_info.get('time_span', 0):.2f}s)")
        cv2.putText(frame, buffer_text, (width - 300, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def run(self) -> None:
        """Loop principal - VERSIÓN CORREGIDA"""
        logger.info("Sistema iniciado. Presiona 'q' para salir.")
        
        # FIX: Verificar que ambos streams estén listos
        logger.info("Verificando streams...")
        timeout = 15
        start_wait = time.time()
        
        while (time.time() - start_wait) < timeout:
            ok_sub, _ = self.grab_sub.read()
            buffer_info = self.grab_main.get_buffer_info()
            
            if ok_sub and buffer_info.get('size', 0) > 0:
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
                    blank = np.zeros((self.camera_config.height, 
                                      self.camera_config.width, 3), dtype=np.uint8)
                    cv2.putText(blank, "Reconectando...", 
                                (60, self.camera_config.height // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.imshow(window_name, blank)
                    
                    key = cv2.waitKey(100) & 0xFF  # FIX: Esperar más tiempo
                    if key == ord('q'):
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
                if trigger_capture and \
                   (current_time - self.last_capture_time) > self.detection_config.cooldown:
                    self.last_capture_time = current_time
                    threading.Thread(target=self._async_capture_preroll, 
                                     daemon=True).start()
                
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
                if key == ord('q'):
                    logger.info("Saliendo por solicitud del usuario")
                    break
        
        except KeyboardInterrupt:
            logger.info("Interrupción de usuario (Ctrl+C)")
        except Exception as e:
            logger.error(f"Error inesperado: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Limpieza de recursos"""
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
    """Parsea argumentos"""
    parser = argparse.ArgumentParser(
        description="Sistema detección vehículos + carriles (v2.1 - Bug Fixes)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    cam_group = parser.add_argument_group("Cámara")
    cam_group.add_argument("--host", default="192.168.1.64", help="IP cámara")
    cam_group.add_argument("--user", default="admin", help="Usuario")
    cam_group.add_argument("--password", required=True, help="Contraseña")
    cam_group.add_argument("--rtsp_channel", default="102", help="Canal sub-stream")
    cam_group.add_argument("--snapshot_channel", default="101", help="Canal main-stream")
    cam_group.add_argument("--width", type=int, default=1280, help="Ancho")
    cam_group.add_argument("--height", type=int, default=720, help="Alto")
    cam_group.add_argument("--rtsp_transport", default="udp", 
                           choices=["udp", "tcp"], help="Transporte RTSP")
    
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
    opt_group.add_argument("--disable_lane_detection", action="store_true",
                           help="Deshabilitar detección carriles")
    opt_group.add_argument("--fallback_isapi", action="store_true",
                           help="Fallback a ISAPI")
    opt_group.add_argument("--verbose", action="store_true", help="Modo verbose")
    
    return parser.parse_args()

def main():
    """Función principal"""
    args = parse_arguments()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    configure_ffmpeg_low_latency(args.rtsp_transport)
    
    camera_config = CameraConfig(
        host=args.host, user=args.user, password=args.password,
        rtsp_channel=args.rtsp_channel, snapshot_channel=args.snapshot_channel,
        width=args.width, height=args.height, rtsp_transport=args.rtsp_transport
    )
    
    detection_config = DetectionConfig(
        model_path=args.model, confidence=args.conf,
        iou_threshold=args.iou, cooldown=args.cooldown,
        pre_roll_ms=args.pre_roll_ms
    )
    
    roi_config = ROIConfig(
        width_ratio=args.roi_width_ratio, height_ratio=args.roi_height_ratio,
        center_x_ratio=args.roi_center_x_ratio, center_y_ratio=args.roi_center_y_ratio
    )
    
    system = VehicleDetectionSystem(
        camera_config=camera_config, detection_config=detection_config,
        roi_config=roi_config,
        enable_lane_detection=not args.disable_lane_detection,
        fallback_isapi=args.fallback_isapi
    )
    
    system.run()

if __name__ == "__main__":
    main()
