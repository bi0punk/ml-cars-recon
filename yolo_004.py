from ultralytics import YOLO
import cv2
import time
import threading
from queue import Queue
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

class OptimizedYOLOStreaming:
    def __init__(self, rtsp_url, model_path="yolov8n.pt"):
        self.ultima_captura_por_vehiculo = {}
        self.model = YOLO(model_path)
        self.rtsp_url = rtsp_url
        self.vehiculos_interes = ['car', 'bus']
        self.ultima_captura = 0
        
        # Configuración de optimización
        self.skip_frames = 3  # Procesar solo cada 3er frame con YOLO
        self.frame_count = 0
        self.current_detections = []
        
        # Buffer para frames y threading
        self.frame_queue = Queue(maxsize=3)  # Buffer pequeño para evitar delay
        self.running = False
        self.capture_thread = None
        
        # Nuevos atributos para integración web
        self.last_captured_images = []  # Lista de rutas de las últimas imágenes
        self.lock = threading.Lock()    # Lock para acceso seguro a recursos compartidos
        self.frame_web = None            # Frame actual para transmisión web

    def setup_camera(self):
        """Configura la cámara con parámetros optimizados"""
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        # Configuraciones para reducir latencia
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo
        cap.set(cv2.CAP_PROP_FPS, 15)  # Limitar FPS para estabilidad
        
        # Configurar resolución más baja si es necesario
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("❌ No se pudo abrir el stream RTSP")
            return None
        
        return cap
    
    def capture_frames(self, cap):
        """Hilo dedicado para capturar frames"""
        while self.running:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️ Error al capturar frame")
                time.sleep(0.1)
                continue
            
            # Mantener solo el frame más reciente
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                # Descartar frame antiguo y agregar nuevo
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
                self.frame_queue.put(frame)
    
    def process_with_yolo(self, frame):
        """Procesa frame con YOLO de forma optimizada"""
        # Reducir resolución para procesamiento más rápido
        height, width = frame.shape[:2]
        scale_factor = 0.7  # Procesar a 70% del tamaño original
        
        small_frame = cv2.resize(frame, 
                                (int(width * scale_factor), int(height * scale_factor)))
        
        # Procesar con YOLO
        results = self.model(small_frame, verbose=False)[0]
        
        # Escalar detecciones de vuelta al tamaño original
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            
            if label in self.vehiculos_interes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Escalar coordenadas
                x1 = int(x1 / scale_factor)
                y1 = int(y1 / scale_factor)
                x2 = int(x2 / scale_factor)
                y2 = int(y2 / scale_factor)
                
                confidence = float(box.conf[0])
                detections.append({
                    'label': label,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence
                })
        
        return detections
    
    def is_vehicle_in_center_zone(self, cx, cy, zona_central_x, zona_central_y):
        """Verifica si el vehículo está en la zona central"""
        return (zona_central_x[0] <= cx <= zona_central_x[1] and 
                zona_central_y[0] <= cy <= zona_central_y[1])
    
    def get_vehicle_key(self, x1, y1, x2, y2):
        """Genera una clave única para el vehículo basada en su posición"""
        # Usamos el centro del bounding box para identificar el vehículo
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        # Creamos una clave con una tolerancia para el mismo vehículo
        key = f"{cx//50}_{cy//50}"  # Agrupamos por bloques de 50 píxeles
        return key
    
    def draw_detections(self, frame, detections):
        """Dibuja las detecciones en el frame"""
        altura, ancho = frame.shape[:2]
        centro_frame = (ancho // 2, altura // 2)
        margen_x = int(ancho * 0.25)
        margen_y = int(altura * 0.3)
        
        zona_central_x = (centro_frame[0] - margen_x, centro_frame[0] + margen_x)
        zona_central_y = (centro_frame[1] - margen_y, centro_frame[1] + margen_y)
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            label = detection['label']
            confidence = detection['confidence']
            
            # Dibujar bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Texto con confianza
            text = f"{label} ({confidence:.2f})"
            cv2.putText(frame, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Centro del vehículo
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            
            # Verificar si está en zona central
            if self.is_vehicle_in_center_zone(cx, cy, zona_central_x, zona_central_y):
                # Cambiar color del bounding box a amarillo para vehículos en zona central
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                
                # Generar clave única para este vehículo
                vehicle_key = self.get_vehicle_key(x1, y1, x2, y2)
                
                # Verificar si ya capturamos este vehículo recientemente
                tiempo_actual = time.time()
                if (vehicle_key not in self.ultima_captura_por_vehiculo or 
                    tiempo_actual - self.ultima_captura_por_vehiculo[vehicle_key] > 3):
                    
                    self.save_vehicle_image(frame, x1, y1, x2, y2, ancho, altura, label)
                    self.ultima_captura_por_vehiculo[vehicle_key] = tiempo_actual
        
        # Dibujar zona de referencia
        cv2.rectangle(frame,
                     (zona_central_x[0], zona_central_y[0]),
                     (zona_central_x[1], zona_central_y[1]),
                     (0, 255, 255), 2)
        
        return frame
    
    def save_vehicle_image(self, frame, x1, y1, x2, y2, ancho, altura, label):
        """Guarda imagen del vehículo cuando está en zona central"""
        # Expandir un poco el área de recorte para capturar mejor el vehículo
        padding = 20
        x1_clip = max(0, x1 - padding)
        y1_clip = max(0, y1 - padding)
        x2_clip = min(ancho, x2 + padding)
        y2_clip = min(altura, y2 + padding)
        
        auto_recortado = frame[y1_clip:y2_clip, x1_clip:x2_clip]

        carpeta = Path("capturas")
        carpeta.mkdir(parents=True, exist_ok=True)
        
        if auto_recortado.size > 0:
            ahora_chile = datetime.now(ZoneInfo("America/Santiago"))
            hora_str = ahora_chile.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # Incluir milisegundos
            nombre_archivo = f"car_{hora_str}.jpg"
            ruta_completa = carpeta / nombre_archivo
            cv2.imwrite(str(ruta_completa), auto_recortado)
            print(f"[✔] Imagen guardada: {nombre_archivo}")
            
            # Actualizar lista de últimas imágenes (con bloqueo para seguridad en hilos)
            with self.lock:
                self.last_captured_images.insert(0, nombre_archivo)
                # Mantener solo las últimas 3 imágenes
                if len(self.last_captured_images) > 3:
                    self.last_captured_images = self.last_captured_images[:3]
    
    def get_last_captured_images(self):
        """Devuelve las últimas imágenes capturadas"""
        with self.lock:
            return self.last_captured_images.copy()
    
    def get_web_frame(self):
        """Obtiene el frame actual para transmisión web"""
        with self.lock:
            return self.frame_web.copy() if self.frame_web is not None else None
    
    def run(self):
        """Función principal optimizada"""
        cap = self.setup_camera()
        if cap is None:
            return
        
        # Iniciar hilo de captura
        self.running = True
        self.capture_thread = threading.Thread(target=self.capture_frames, args=(cap,))
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        cv2.namedWindow("Cámara IP - Vehículos Optimizada", cv2.WINDOW_NORMAL)
        
        print("🚀 Iniciando streaming optimizado...")
        print("📝 Consejos:")
        print("   - Presiona 'q' para salir")
        print("   - Presiona 's' para saltar procesamiento YOLO por 10 frames")
        print("   - Los vehículos en la zona amarilla se guardarán automáticamente")
        
        skip_yolo_frames = 0
        
        while True:
            try:
                # Obtener frame más reciente
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get_nowait()
                else:
                    time.sleep(0.01)  # Pequeña pausa si no hay frames
                    continue
                
                # Procesar con YOLO solo cada N frames o si no hay detecciones
                if (self.frame_count % self.skip_frames == 0 and 
                    skip_yolo_frames <= 0):
                    self.current_detections = self.process_with_yolo(frame)
                
                # Dibujar detecciones (usar las últimas disponibles)
                frame = self.draw_detections(frame, self.current_detections)
                
                # Actualizar frame para web (con bloqueo para seguridad en hilos)
                with self.lock:
                    # Reducir resolución para transmisión web
                    self.frame_web = cv2.resize(frame, (640, 480))
                
                # Mostrar FPS y contador de vehículos capturados
                fps_text = f"Frame: {self.frame_count} | Capturados: {len(self.ultima_captura_por_vehiculo)}"
                cv2.putText(frame, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow("Cámara IP - Vehículos Optimizada", frame)
                
                self.frame_count += 1
                if skip_yolo_frames > 0:
                    skip_yolo_frames -= 1
                
                # Controles de teclado
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    skip_yolo_frames = 10  # Saltar YOLO por 10 frames
                    print("Saltando procesamiento YOLO por 10 frames")
                
            except Exception as e:
                print(f"Error en el bucle principal: {e}")
                time.sleep(0.1)
        
        # Cleanup
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1)
        cap.release()
        cv2.destroyAllWindows()
        print("🔚 Streaming finalizado")

# Nota: Ahora la ejecución se controla desde app.py
# Este archivo se ejecuta como módulo cuando se usa con Flask