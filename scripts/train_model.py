from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.saving import save_model
import os

# Parámetros
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
DATASET_PATH = "dataset/dataset_ordenado"
MODEL_H5_PATH = "modelo_stanfordcars.h5"
MODEL_KERAS_PATH = "modelo_stanfordcars.keras"

# Preprocesamiento y generadores
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Modelo base
base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights="imagenet")
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(train_gen.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Congelar capas base
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

# Entrenamiento
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[
        ModelCheckpoint(MODEL_H5_PATH, save_best_only=True, monitor='val_accuracy')
    ]
)

print(f"Modelo guardado en formato legacy HDF5: {MODEL_H5_PATH}")

# ✅ Convertir a formato moderno .keras
mejor_modelo = load_model(MODEL_H5_PATH)
save_model(mejor_modelo, MODEL_KERAS_PATH)
print(f"Modelo convertido y guardado en formato Keras moderno: {MODEL_KERAS_PATH}")
