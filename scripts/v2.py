#!/usr/bin/env python3
import cv2
import os
import csv
import numpy as np
from datetime import datetime
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import torch
import time
import warnings
from collections import defaultdict

# Suprimir avisos redundantes
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")

class CarDetector:
    def __init__(self):
        # Rutas base
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.CONF_DET = 0.5    # Umbral detección YOLOv5
        self.CONF_CLAS = 0.5   # Umbral clasificación
        self.frame_skip = 2    # Clasifica cada 2 frames
        self.csv_interval = 2.0
        self.last_csv = 0
        self.conteo = defaultdict(int)

        # Setup
        self._setup_models()
        self._setup_dirs()
        self._load_previous_counts()
        self._setup_csv()

    def _setup_models(self):
        print("[INFO] Cargando YOLOv5 para detección...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
        self.model_yolo.classes = [2]  # COCO clase 2 = car

        print("[INFO] Cargando modelo de clasificación...")
        model_path = os.path.join(self.script_dir, "../models/modelo_stanfordcars.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo: {model_path}")
        self.model_cl = load_model(model_path)

        classes_dir = os.path.join(self.script_dir, "../data/processed/dataset_ordenado")
        if not os.path.exists(classes_dir):
            raise FileNotFoundError(f"No se encontró clases dir: {classes_dir}")
        self.classes = sorted(os.listdir(classes_dir))

    def _setup_dirs(self):
        self.no_recon_dir = os.path.join(self.script_dir, "../data/autos_no_reconocidos")
        os.makedirs(self.no_recon_dir, exist_ok=True)
        self.csv_path = os.path.join(self.script_dir, "../resumen_reconocimientos.csv")

    def _load_previous_counts(self):
        if os.path.exists(self.csv_path):
            with open(self.csv_path, newline='') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) == 2:
                        self.conteo[row[0]] = int(row[1])

    def _setup_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(["Marca_Modelo","Cantidad"]);

    def _update_csv(self, force=False):
        now = time.time()
        if force or now - self.last_csv >= self.csv_interval:
            try:
                with open(self.csv_path, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(["Marca_Modelo","Cantidad"]);
                    for m, c in sorted(self.conteo.items(), key=lambda x: -x[1]):
                        w.writerow([m, c])
                self.last_csv = now
            except Exception as e:
                print(f"[ERROR] Actualizar CSV: {e}")

    def _preprocess(self, crop):
        img = cv2.resize(crop, (224, 224))
        img = img.astype('float32') / 255.0
        img = img_to_array(img)
        return np.expand_dims(img, axis=0)

    def _classify(self, img):
        preds = self.model_cl.predict(img, verbose=0)[0]
        idx = int(np.argmax(preds))
        if idx >= len(self.classes):
            return None, 0.0
        return self.classes[idx], float(preds[idx])

    def _save_unrecognized(self, crop):
        t = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        fn = os.path.join(self.no_recon_dir, f"nr_{t}.jpg")
        cv2.imwrite(fn, crop)
        print(f"[GUARDADO] {fn}")

    def _init_camera(self):
        print("[INFO] Iniciando cámara...")
        for idx in [0, 1, -1]:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    return cap
                cap.release()
        raise RuntimeError("No se pudo inicializar ninguna cámara.")

    def run(self):
        cap = self._init_camera()
        cv2.namedWindow("Detección de autos", cv2.WINDOW_NORMAL)
        print("[INFO] Presiona 'q' para salir...")

        frame_idx = 0
        last_processed_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_idx += 1
            display_frame = frame.copy()

            # Solo procesar cada N frames
            if frame_idx % self.frame_skip == 0:
                results = self.model_yolo(frame)
                for det in results.xyxy[0].tolist():
                    x1, y1, x2, y2, det_conf, _ = det
                    if det_conf < self.CONF_DET:
                        continue
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue

                    img = self._preprocess(roi)
                    lbl, cls_conf = self._classify(img)

                    if lbl and cls_conf >= self.CONF_CLAS:
                        color = (0, 255, 0)
                        txt = f"{lbl} ({cls_conf*100:.1f}%)"
                        self.conteo[lbl] += 1
                    else:
                        color = (0, 0, 255)
                        txt = f"No recon ({cls_conf*100:.1f}%)"
                        self._save_unrecognized(roi)

                    # Dibujar en display_frame
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, txt, (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                last_processed_frame = display_frame
            else:
                # Usar el último frame procesado para evitar parpadeo
                if last_processed_frame is not None:
                    display_frame = last_processed_frame

            # Mostrar
            cv2.imshow("Detección de autos", display_frame)
            key = cv2.waitKey(1)
            if key == ord('q') or cv2.getWindowProperty("Detección de autos", cv2.WND_PROP_VISIBLE) < 1:
                break

            self._update_csv()

        cap.release()
        cv2.destroyAllWindows()
        self._update_csv(force=True)
        print("[INFO] Finalizado.")

    def get_summary(self):
        total = sum(self.conteo.values())
        print(f"[RESUMEN] Total de autos detectados: {total}")
        for m, c in sorted(self.conteo.items(), key=lambda x: -x[1]):
            print(f"  {m}: {c}")


def main():
    try:
        detector = CarDetector()
        detector.run()
        detector.get_summary()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback; traceback.print_exc()
        
if __name__ == "__main__":
    main()