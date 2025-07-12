from ultralytics import YOLO
import cv2
import time
import threading
from queue import Queue
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import os

class OptimizedYOLOStreaming:
    def __init__(self, rtsp_url, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.rtsp_url = rtsp_url
        self.vehiculos_interes = ['car', 'bus']
        self.frame_queue = Queue(maxsize=3)
        self.current_detections = []
        self.ultima_captura_por_vehiculo = {}
        self.skip_frames = 3
        self.frame_count = 0
        self.running = True
        self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def start(self):
        t = threading.Thread(target=self.capture_loop)
        t.daemon = True
        t.start()

    def capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret: continue
            if self.frame_count % self.skip_frames == 0:
                self.current_detections = self.process_with_yolo(frame)
            frame = self.draw_detections(frame, self.current_detections)
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                try: self.frame_queue.get_nowait()
                except: pass
                self.frame_queue.put(frame)
            self.frame_count += 1

    def process_with_yolo(self, frame):
        height, width = frame.shape[:2]
        scale_factor = 0.7
        small = cv2.resize(frame, (int(width*scale_factor), int(height*scale_factor)))
        results = self.model(small, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            if label in self.vehiculos_interes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = int(x1 / scale_factor)
                y1 = int(y1 / scale_factor)
                x2 = int(x2 / scale_factor)
                y2 = int(y2 / scale_factor)
                detections.append({'label': label, 'bbox': (x1, y1, x2, y2), 'confidence': float(box.conf[0])})
        return detections

    def draw_detections(self, frame, detections):
        h, w = frame.shape[:2]
        cxr = (w // 2 - w//4, w // 2 + w//4)
        cyr = (h // 2 - h//3, h // 2 + h//3)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            conf = det['confidence']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            if cxr[0] <= cx <= cxr[1] and cyr[0] <= cy <= cyr[1]:
                key = f"{cx//50}_{cy//50}"
                if time.time() - self.ultima_captura_por_vehiculo.get(key, 0) > 3:
                    self.save_capture(frame, x1, y1, x2, y2, label)
                    self.ultima_captura_por_vehiculo[key] = time.time()
        cv2.rectangle(frame, (cxr[0], cyr[0]), (cxr[1], cyr[1]), (0,255,255), 2)
        return frame

    def save_capture(self, frame, x1, y1, x2, y2, label):
        padding = 20
        crop = frame[max(0, y1-padding): y2+padding, max(0, x1-padding): x2+padding]
        if crop.size > 0:
            carpeta = Path("static/capturas")
            carpeta.mkdir(parents=True, exist_ok=True)
            ahora = datetime.now(ZoneInfo("America/Santiago"))
            nombre = f"{label}_{ahora.strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]}.jpg"
            cv2.imwrite(str(carpeta / nombre), crop)
