#!/usr/bin/env python3
# train_by_brand.py
"""
Entrena un modelo de clasificación solo por MARCA.
Partiendo de un dataset organizado en "data/processed/dataset_ordenado/<Marca Modelo Año>/imagenes".
Este script agrupa imágenes por marca, entrena MobileNetV2 y guarda el modelo.
"""
import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.saving import save_model

# Parámetros
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Rutas relativas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "../data/processed/dataset_ordenado")
BRAND_DIR = os.path.join(SCRIPT_DIR, "../data/processed/brand_dataset")
MODEL_H5 = os.path.join(SCRIPT_DIR, "../models/model_by_brand.h5")
MODEL_KERAS = os.path.join(SCRIPT_DIR, "../models/model_by_brand.keras")

# 1. Agrupar por marca
if os.path.exists(BRAND_DIR):
    shutil.rmtree(BRAND_DIR)
os.makedirs(BRAND_DIR, exist_ok=True)

print(f"[INFO] Agrupando imágenes por marca desde {RAW_DIR}...")
for folder in os.listdir(RAW_DIR):
    marca = folder.split()[0]
    src = os.path.join(RAW_DIR, folder)
    dest = os.path.join(BRAND_DIR, marca)
    os.makedirs(dest, exist_ok=True)
    for img in os.listdir(src):
        shutil.copyfile(os.path.join(src, img), os.path.join(dest, img))
print("[INFO] Agrupación por marca completada.")

# 2. Generadores de datos
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1
)

train_gen = datagen.flow_from_directory(
    BRAND_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)
val_gen = datagen.flow_from_directory(
    BRAND_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# 3. Definir modelo
base = MobileNetV2(include_top=False, input_shape=(224,224,3), weights='imagenet')
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)
model = Model(base.input, output)

# Congelar base
for layer in base.layers:
    layer.trainable = False

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Entrenamiento
print(f"[INFO] Entrenando modelo por marca ({train_gen.num_classes} clases)...")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[ModelCheckpoint(MODEL_H5, save_best_only=True, monitor='val_accuracy')]
)
print(f"[INFO] Modelo guardado en HDF5: {MODEL_H5}")

# 5. Convertir a .keras
mejor = load_model(MODEL_H5)
save_model(mejor, MODEL_KERAS)
print(f"[INFO] Modelo convertido a Keras: {MODEL_KERAS}")

