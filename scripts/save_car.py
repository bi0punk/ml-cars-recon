#!/usr/bin/env python3
import cv2
import os
from datetime import datetime
import torch
import numpy as np

# === RUTAS Y CONFIGURACIÓN ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "capturas")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parámetros
CONF_DET = 0.5            # umbral detección
MAX_PER_OBJ = 2           # fotos por vehículo
IOU_THRESHOLD = 0.5       # para asociar detección a objeto

IP_CAMERA_URL = "rtsp://admin:123456@192.168.1.125/stream0"
# === CARGAR YOLOv5 ===
print("[INFO] Cargando YOLOv5...")
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
model_yolo.classes = [2, 5]                     # 2=car, 5=bus
class_names = model_yolo.names

# === Estructuras de rastreo ===
next_object_id = 0
tracked = {}  # id -> {'box':(x1,y1,x2,y2), 'count':n, 'class':cls_name}

def iou(boxA, boxB):
    # calcula Intersection over Union entre dos cajas
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

# === Conexión a cámara IP ===
cap = cv2.VideoCapture(IP_CAMERA_URL)
if not cap.isOpened():
    print("[ERROR] No se pudo conectar a la cámara IP.")
    exit(1)

# Crear ventana y redimensionarla
cv2.namedWindow("Detección YOLO", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detección YOLO", 1280, 720)   # <--- tamaño 1280×720

print("[INFO] Streaming iniciado. Presiona 'q' para salir.")

# === Bucle principal ===
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model_yolo(frame)
    detections = results.xyxy[0].cpu().numpy()

    updated_ids = set()

    for x1, y1, x2, y2, conf, cls_id in detections:
        if conf < CONF_DET:
            continue

        cls_id = int(cls_id)
        cls_name = class_names.get(cls_id, f"class{cls_id}")
        box = (int(x1), int(y1), int(x2), int(y2))

        # Asociar detección a objeto existente o nuevo
        best_id, best_iou = None, 0
        for obj_id, data in tracked.items():
            if data['class'] != cls_name: continue
            i = iou(box, data['box'])
            if i > best_iou:
                best_iou, best_id = i, obj_id

        if best_iou > IOU_THRESHOLD:
            obj_id = best_id
        else:
            obj_id = next_object_id
            tracked[obj_id] = {'box': box, 'count': 0, 'class': cls_name}
            next_object_id += 1

        tracked[obj_id]['box'] = box
        updated_ids.add(obj_id)

        # Guardar hasta MAX_PER_OBJ fotos por vehículo
        if tracked[obj_id]['count'] < MAX_PER_OBJ:
            x1i, y1i, x2i, y2i = box
            roi = frame[y1i:y2i, x1i:x2i]
            if roi.size == 0:
                continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            cv2.putText(
                roi, timestamp,
                (10, roi.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA
            )

            fname = f"{cls_name}_{obj_id}_{timestamp}.jpg"
            path = os.path.join(OUTPUT_DIR, fname)
            cv2.imwrite(path, roi)
            tracked[obj_id]['count'] += 1
            print(f"[GUARDADO] obj#{obj_id} ({cls_name}) foto #{tracked[obj_id]['count']} → {path}")

        # Dibujar recuadro y etiqueta
        color = (0,255,0) if cls_name=="car" else (255,0,0)
        x1i, y1i, x2i, y2i = box
        cv2.rectangle(frame, (x1i,y1i), (x2i,y2i), color, 2)
        label = f"{cls_name} {conf*100:.0f}%"
        cv2.putText(frame, label, (x1i, y1i-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Detección YOLO", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Finalizado.")
