#!/usr/bin/env python3
import cv2
import os
import csv
import numpy as np
import warnings
from datetime import datetime
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import torch

# === FILTRAR WARNINGS ===
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Could not find ref with POC.*")

# === RUTAS Y CONFIGURACIÓN ===
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "../models/model_by_brand.keras")
CLASSES_DIR  = os.path.join(BASE_DIR, "../data/processed/dataset_ordenado")
NO_RECON_DIR = os.path.join(BASE_DIR, "../data/autos_no_reconocidos")
CSV_PATH     = os.path.join(BASE_DIR, "../resumen_reconocimientos.csv")

CONF_DET  = 0.5    # umbral detección YOLO
CONF_CLAS = 0.5    # umbral clasificación modelo propia

# === RESOLUCIÓN DE SALIDA ===
RESOLUTION = (1280, 720)  # ancho, alto (720p)

# === URL DE LA CÁMARA IP ===
IP_CAMERA_URL = "rtsp://admin:123456@192.168.1.125/stream0"

# === CARGAR MODELOS ===
print("[INFO] Cargando modelo YOLOv5 para detección de autos…")
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
model_yolo.classes = [2]  # COCO clase 2 = car

print("[INFO] Cargando modelo de clasificación de marca/modelo…")
modelo = load_model(MODEL_PATH)
CLASSES = sorted(os.listdir(CLASSES_DIR))

# === PREPARAR SALIDAS Y CONTEO ===
os.makedirs(NO_RECON_DIR, exist_ok=True)
ingreso_conteo = {}
if os.path.exists(CSV_PATH):
    with open(CSV_PATH, newline='') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) == 2:
                ingreso_conteo[row[0]] = int(row[1])
conteo = dict(ingreso_conteo)

# === INICIAR Y CONFIGURAR CÁMARA IP ===
print(f"[INFO] Conectando a cámara IP en {IP_CAMERA_URL}…")
cap = cv2.VideoCapture(IP_CAMERA_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)  # minimizar latencia
if not cap.isOpened():
    print(f"[ERROR] No se pudo abrir el stream IP: {IP_CAMERA_URL}")
    exit(1)

cv2.namedWindow("Detección de autos", cv2.WINDOW_NORMAL)
print("[INFO] Cámara IP iniciada. Presiona 'q' para salir.")

# === BUCLE PRINCIPAL CON CÁLCULO DE FPS ===
prev_time = datetime.now()
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[WARN] No se pudo leer frame de la cámara IP.")
        cv2.waitKey(100)
        continue

    # Asegurar 720p
    frame = cv2.resize(frame, RESOLUTION)

    # Detección YOLOv5
    results = model_yolo(frame)
    detecciones = results.xyxy[0]

    for det in detecciones:
        x1, y1, x2, y2, det_conf, cls = det[:6]
        if float(det_conf) < CONF_DET:
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # Clasificación de marca/modelo
        img = cv2.resize(roi, (224, 224))
        img = img.astype('float32') / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)

        try:
            preds = modelo.predict(img, verbose=0)[0]
        except Exception as e:
            print(f"[ERROR] Clasificación fallida: {e}")
            continue

        idx     = int(np.argmax(preds))
        cls_conf = float(preds[idx])
        label   = CLASSES[idx] if idx < len(CLASSES) else "Desconocido"

        if cls_conf >= CONF_CLAS:
            color = (0, 255, 0)
            txt   = f"{label} ({cls_conf*100:.1f}%)"
            conteo[label] = conteo.get(label, 0) + 1
        else:
            color = (0, 0, 255)
            txt   = f"No recon ({cls_conf*100:.1f}%)"
            fname = os.path.join(
                NO_RECON_DIR,
                f"nr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            )
            cv2.imwrite(fname, roi)
            print(f"[GUARDADO] Imagen no reconocida: {fname}")

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, txt, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Calcular FPS
    now = datetime.now()
    fps = 1.0 / (now - prev_time).total_seconds()
    prev_time = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Detección de autos", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === LIBERAR RECURSOS Y GUARDAR CONTEO ===
cap.release()
cv2.destroyAllWindows()
print("[INFO] Finalizado. Guardando conteo…")

with open(CSV_PATH, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Marca_Modelo", "Cantidad"])
    for marca, cnt in sorted(conteo.items(), key=lambda x: -x[1]):
        writer.writerow([marca, cnt])

print(f"[INFO] Conteo guardado en {CSV_PATH}")
