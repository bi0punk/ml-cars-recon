from ultralytics import YOLO
import cv2
import time
import threading
from queue import Queue
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
ahora_chile = datetime.now(ZoneInfo("America/Santiago"))
nombre_archivo = time.strftime("vehiculo_%Y-%m-%d_%H-%M-%S.jpg", time.localtime())


class OptimizedYOLOStreaming:
    def __init__(self, rtsp_url, model_path="yolov8n.pt"):
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
        
    def setup_camera(self):
        """Configura la cámara con parámetros optimizados"""
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        # Configuraciones para reducir latencia
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo
        cap.set(cv2.CAP_PROP_FPS, 15)  # Limitar FPS para estabilidad
        
        # Configurar resolución más baja si es necesario (descomenta si tienes lag)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
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
            if (zona_central_x[0] <= cx <= zona_central_x[1] and 
                zona_central_y[0] <= cy <= zona_central_y[1]):
                
                self.save_vehicle_image(frame, x1, y1, x2, y2, ancho, altura)
        
        # Dibujar zona de referencia
        cv2.rectangle(frame,
                     (zona_central_x[0], zona_central_y[0]),
                     (zona_central_x[1], zona_central_y[1]),
                     (0, 255, 255), 2)
        
        return frame
    
    def save_vehicle_image(self, frame, x1, y1, x2, y2, ancho, altura):
        """Guarda imagen del vehículo si está en zona central"""
        tiempo_actual = time.time()
        if tiempo_actual - self.ultima_captura > 5:
            x1_clip = max(0, x1)
            y1_clip = max(0, y1)
            x2_clip = min(ancho, x2)
            y2_clip = min(altura, y2)
            
            auto_recortado = frame[y1_clip:y2_clip, x1_clip:x2_clip]

            carpeta = Path("capturas")
            carpeta.mkdir(parents=True, exist_ok=True)
            if auto_recortado.size > 0:
                ahora_chile = datetime.now(ZoneInfo("America/Santiago"))
                hora_str = ahora_chile.strftime("%Y-%m-%d_%H-%M-%S")
                nombre_archivo = carpeta / f"vehiculo_{hora_str}.jpg"
                cv2.imwrite(nombre_archivo, auto_recortado)
                print(f"[✔] Imagen guardada: {nombre_archivo}")
                self.ultima_captura = tiempo_actual
    
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
                
                # Mostrar FPS
                fps_text = f"Frame: {self.frame_count}"
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

# Uso del código optimizado
if __name__ == "__main__":
    ip_camera_url = "rtsp://admin:123456@192.168.1.125:554/stream0"
    
    # Crear y ejecutar el streaming optimizado
    streamer = OptimizedYOLOStreaming(ip_camera_url)
    streamer.run()