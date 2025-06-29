# evaluate_model.py
"""
Evalúa el modelo de clasificación de marca/modelo usando Keras.
Requiere que tu dataset procesado esté en carpetas por clase:
  data/processed/dataset_ordenado/<clase>/*.jpg
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

def main():
    # Rutas
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "../models/modelo_stanfordcars.keras")
    DATASET_DIR = os.path.join(BASE_DIR, "../data/processed/dataset_ordenado")

    # Parámetros
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    # Carga modelo
    print(f"[INFO] Cargando modelo desde {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    # Prepara generador (sin augmentations, solo rescale)
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  # importante para evaluar ordenadamente
    )

    # Evalúa
    print("[INFO] Evaluando modelo en todas las clases...")
    loss, acc = model.evaluate(generator, verbose=1)
    print(f"[RESULTADOS] Pérdida (loss): {loss:.4f} | Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
