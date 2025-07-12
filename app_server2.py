from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import cv2
import time
import threading
from queue import Queue
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import base64
import json
import os
import glob

app = Flask(__name__)

class WebOptimizedYOLOStreaming:
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
        
        # Para la interfaz web
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.latest_captures = []  # Lista de las últimas capturas
        
    def setup_camera(self):
        """Configura la cámara con parámetros optimizados"""
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        # Configuraciones para reducir latencia
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo
        cap.set(cv2.CAP_PROP_FPS, 15)  # Limitar FPS para estabilidad
        
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
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
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

        carpeta = Path("static/capturas")
        carpeta.mkdir(parents=True, exist_ok=True)
        
        if auto_recortado.size > 0:
            ahora_chile = datetime.now(ZoneInfo("America/Santiago"))
            hora_str = ahora_chile.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # Incluir milisegundos
            nombre_archivo = carpeta / f"car_{hora_str}.jpg"
            cv2.imwrite(str(nombre_archivo), auto_recortado)
            print(f"[✔] Imagen guardada: {nombre_archivo}")
            
            # Actualizar lista de capturas recientes
            self.update_latest_captures(str(nombre_archivo))
    
    def update_latest_captures(self, new_capture_path):
        """Actualiza la lista de las últimas 3 capturas"""
        # Agregar nueva captura al inicio de la lista
        self.latest_captures.insert(0, {
            'path': new_capture_path.replace('static/', ''),
            'timestamp': datetime.now(ZoneInfo("America/Santiago")).strftime("%H:%M:%S")
        })
        
        # Mantener solo las últimas 3
        if len(self.latest_captures) > 3:
            self.latest_captures = self.latest_captures[:3]
    
    def get_latest_captures(self):
        """Obtiene las últimas 3 capturas"""
        return self.latest_captures
    
    def process_frames(self):
        """Procesa frames para streaming web"""
        cap = self.setup_camera()
        if cap is None:
            return
        
        # Iniciar hilo de captura
        self.running = True
        self.capture_thread = threading.Thread(target=self.capture_frames, args=(cap,))
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        print("🚀 Iniciando streaming web optimizado...")
        
        while self.running:
            try:
                # Obtener frame más reciente
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get_nowait()
                else:
                    time.sleep(0.01)  # Pequeña pausa si no hay frames
                    continue
                
                # Procesar con YOLO solo cada N frames
                if self.frame_count % self.skip_frames == 0:
                    self.current_detections = self.process_with_yolo(frame)
                
                # Dibujar detecciones
                frame = self.draw_detections(frame, self.current_detections)
                
                # Mostrar información en el frame
                fps_text = f"Frame: {self.frame_count} | Capturados: {len(self.ultima_captura_por_vehiculo)}"
                cv2.putText(frame, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Actualizar frame actual para streaming
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                self.frame_count += 1
                
            except Exception as e:
                print(f"Error en el procesamiento: {e}")
                time.sleep(0.1)
        
        # Cleanup
        cap.release()
        print("🔚 Streaming finalizado")
    
    def get_frame(self):
        """Obtiene el frame actual para streaming"""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def start_streaming(self):
        """Inicia el streaming en un hilo separado"""
        if not self.running:
            self.running = True
            streaming_thread = threading.Thread(target=self.process_frames)
            streaming_thread.daemon = True
            streaming_thread.start()
    
    def stop_streaming(self):
        """Detiene el streaming"""
        self.running = False

# Instancia global del streamer
streamer = None

def generate_frames():
    """Generador de frames para Flask"""
    global streamer
    while True:
        if streamer is None:
            time.sleep(0.1)
            continue
            
        frame = streamer.get_frame()
        if frame is not None:
            # Codificar frame a JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.1)

@app.route('/')
def index():
    """Página principal"""
    return render_template('index2.html')

@app.route('/video_feed')
def video_feed():
    """Endpoint para streaming de video"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/latest_captures')
def latest_captures():
    """Endpoint para obtener las últimas capturas"""
    global streamer
    if streamer:
        captures = streamer.get_latest_captures()
        return jsonify(captures)
    return jsonify([])

@app.route('/start_streaming')
def start_streaming():
    """Inicia el streaming"""
    global streamer
    if streamer is None:
        ip_camera_url = "rtsp://admin:123456@192.168.1.125:554/stream0"
        streamer = WebOptimizedYOLOStreaming(ip_camera_url)
        streamer.start_streaming()
        return jsonify({"status": "started"})
    return jsonify({"status": "already running"})

@app.route('/stop_streaming')
def stop_streaming():
    """Detiene el streaming"""
    global streamer
    if streamer:
        streamer.stop_streaming()
        streamer = None
        return jsonify({"status": "stopped"})
    return jsonify({"status": "not running"})

if __name__ == '__main__':
    # Crear carpeta de capturas si no existe
    Path("static/capturas").mkdir(parents=True, exist_ok=True)
    
    print("🌐 Iniciando servidor Flask...")
    print("📱 Accede a: http://localhost:5000")
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)