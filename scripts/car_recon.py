#!/usr/bin/env python3
import cv2
import os
import csv
import numpy as np
from datetime import datetime
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import torch

# === RUTAS Y CONFIGURACIÓN ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/model_by_brand.keras")
CLASSES_DIR = os.path.join(BASE_DIR, "../data/processed/dataset_ordenado")
NO_RECON_DIR = os.path.join(BASE_DIR, "../data/autos_no_reconocidos")
CSV_PATH = os.path.join(BASE_DIR, "../resumen_reconocimientos.csv")
CONF_DET = 0.5      # umbral detección YOLO
CONF_CLAS = 0.5     # umbral clasificación modelo propia

# === CARGAR MODELOS ===
print("[INFO] Cargando modelo YOLOv5 para detección de autos...")
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
model_yolo.classes = [2]  # COCO clase 2 = car

print("[INFO] Cargando modelo de clasificación de marca/modelo...")
modelo = load_model(MODEL_PATH)
CLASSES = sorted(os.listdir(CLASSES_DIR))

# === PREPARAR SALIDAS ===
os.makedirs(NO_RECON_DIR, exist_ok=True)
# Cargar conteo previo si existe
ingreso_conteo = {}
if os.path.exists(CSV_PATH):
    with open(CSV_PATH, newline='') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) == 2:
                ingreso_conteo[row[0]] = int(row[1])
conteo = dict(ingreso_conteo)

# === INICIAR Y VERIFICAR CÁMARA ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[WARN] No se pudo abrir la cámara en índice 0. Intentando índice 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara. Verifica conexión y permisos.")
        exit(1)

cv2.namedWindow("Detección de autos", cv2.WINDOW_NORMAL)
print("[INFO] Cámara iniciada. Presiona 'q' para salir.")

# === BUCLE PRINCIPAL ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] No se pudo leer frame de la cámara.")
        continue  # intenta de nuevo

    results = model_yolo(frame)
    detecciones = results.xyxy[0]

    # Procesar cada detección de auto
    for det in detecciones:
        x1, y1, x2, y2, det_conf, cls = det[:6]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        det_conf = float(det_conf)

        if det_conf < CONF_DET:
            continue

        # Recortar ROI auto
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # Clasificar marca/modelo
        img = cv2.resize(roi, (224, 224))
        img = img.astype('float32') / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)

        preds = modelo.predict(img, verbose=0)[0]
        idx = int(np.argmax(preds))
        cls_conf = float(preds[idx])

        if idx >= len(CLASSES):
            continue
        label = CLASSES[idx]

        # Dibujar y guardar según confianza
        if cls_conf >= CONF_CLAS:
            color = (0, 255, 0)
            txt = f"{label} ({cls_conf*100:.1f}%)"
            conteo[label] = conteo.get(label, 0) + 1
        else:
            color = (0, 0, 255)
            txt = f"No recon ({cls_conf*100:.1f}%)"
            fname = os.path.join(NO_RECON_DIR, f"nr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(fname, roi)
            print(f"[GUARDADO] Imagen no reconocida: {fname}")

        # Dibujar recuadro y texto
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, txt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Mostrar frame único
    cv2.imshow("Detección de autos", frame)
    key = cv2.waitKey(1)
    if key == ord('q') or cv2.getWindowProperty("Detección de autos", cv2.WND_PROP_VISIBLE) < 1:
        break

# === LIBERAR RECURSOS ===
cap.release()
cv2.destroyAllWindows()
print("[INFO] Finalizado. Guardando conteo...")

# Guardar conteo final en CSV
with open(CSV_PATH, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Marca_Modelo", "Cantidad"])
    for marca, cnt in sorted(conteo.items(), key=lambda x: -x[1]):
        writer.writerow([marca, cnt])
print(f"[INFO] Conteo guardado en {CSV_PATH}")
