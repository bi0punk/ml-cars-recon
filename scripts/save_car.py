#!/usr/bin/env python3
import cv2
import os
import threading
from datetime import datetime
import torch
import numpy as np
from queue import Queue

# === RUTAS Y CONFIGURACIÓN ===
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(BASE_DIR, "capturas")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONF_DET       = 0.5    # umbral mínima confianza
MAX_PER_OBJ    = 2      # fotos máximas por objeto
IOU_THRESHOLD  = 0.5    # para seguimiento
IP_CAMERA_URL  = "rtsp://admin:123456@192.168.1.125/stream0"
MARGIN         = 0.1    # pad (10%) alrededor de la caja para no cortar partes del objeto

# === CARGAR YOLOv5 con half-precision en GPU si es posible ===
print("[INFO] Cargando YOLOv5...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True).to(device).eval()
if device == "cuda":
    model.half()
model.conf    = CONF_DET       # umbral
model.iou     = IOU_THRESHOLD  # sólovisión interna
model.classes = [2,5]          # coches y buses
names = model.names

# === Frame grabber en background para minimizar lag ===
class CameraReader(threading.Thread):
    def __init__(self, src, queue_size=1):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, queue_size)
        self.queue = Queue(maxsize=queue_size)
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            if self.queue.full():
                try: self.queue.get_nowait()
                except: pass
            self.queue.put(frame)

    def read(self, timeout=1):
        try:
            return True, self.queue.get(timeout=timeout)
        except:
            return False, None

    def stop(self):
        self.running = False
        self.cap.release()

# lanzar reader
reader = CameraReader(IP_CAMERA_URL, queue_size=1)
reader.start()

cv2.namedWindow("Detección", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detección", 1280, 720)

print("[INFO] Streaming iniciado. Presiona 'q' para salir.")

tracked     = {}   # id → {box, count, cls}
next_obj_id = 0

def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = ((a[2] - a[0]) * (a[3] - a[1]) +
             (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0

while True:
    got, frame = reader.read()
    if not got:
        continue

    # inference sobre versión reducida para acelerar
    img = cv2.resize(frame, (640, 640))
    if device == "cuda":
        img = img[:, :, ::-1]  # BGR→RGB
    results = model(img, size=640)
    dets = results.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2,conf,cls]

    h, w = frame.shape[:2]
    seen_ids = set()
    for x1, y1, x2, y2, conf, cls in dets:
        if conf < CONF_DET:
            continue
        cls = int(cls)
        name = names[cls]
        # reescalar coords a tamaño original
        scale_x, scale_y = w / 640, h / 640
        x1o, y1o = int(x1 * scale_x), int(y1 * scale_y)
        x2o, y2o = int(x2 * scale_x), int(y2 * scale_y)

        # aplicar margen
        bw, bh = x2o - x1o, y2o - y1o
        pad_w, pad_h = int(bw * MARGIN), int(bh * MARGIN)
        x1i = max(0, x1o - pad_w)
        y1i = max(0, y1o - pad_h)
        x2i = min(w,   x2o + pad_w)
        y2i = min(h,   y2o + pad_h)
        box = (x1i, y1i, x2i, y2i)

        # seguimiento sencillo
        best_id, best_iou = None, 0
        for oid, data in tracked.items():
            if data['cls'] != name:
                continue
            i = iou(box, data['box'])
            if i > best_iou:
                best_iou, best_id = i, oid

        if best_iou > IOU_THRESHOLD:
            oid = best_id
        else:
            oid = next_obj_id
            tracked[oid] = {'box': box, 'count': 0, 'cls': name}
            next_obj_id += 1

        tracked[oid]['box'] = box
        seen_ids.add(oid)

        # guardar hasta MAX_PER_OBJ fotos
        if tracked[oid]['count'] < MAX_PER_OBJ:
            roi = frame[y1i:y2i, x1i:x2i]
            if roi.size == 0:
                continue

            # nombre con timestamp (solo en archivo)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            fn = f"{name}_{oid}_{ts}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, fn), roi)
            tracked[oid]['count'] += 1
            print(f"[GUARDADO] #{oid} ({name}) foto#{tracked[oid]['count']} → {fn}")

        # dibujar rectángulo con margen
        cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)

    # limpiar objetos que desaparecieron
    for oid in list(tracked):
        if oid not in seen_ids:
            del tracked[oid]

    cv2.imshow("Detección", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

reader.stop()
cv2.destroyAllWindows()
print("[INFO] Finalizado.")
