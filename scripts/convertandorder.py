import scipy.io
import pandas as pd
import os
import shutil
from tqdm import tqdm

# Rutas
BASE_PATH = "/home/drbash/Documentos/ml-projects/car-detect/dataset"
DEVKIT = os.path.join(BASE_PATH, "car_devkit", "devkit")
TRAIN_DIR = os.path.join(BASE_PATH, "cars_train", "cars_train")
OUTPUT_DIR = os.path.join(BASE_PATH, "dataset_ordenado")

# Cargar anotaciones y nombres de clases
train_annos = scipy.io.loadmat(os.path.join(DEVKIT, "cars_train_annos.mat"))['annotations'][0]
meta = scipy.io.loadmat(os.path.join(DEVKIT, "cars_meta.mat"))['class_names'][0]

# Crear estructura: {1: "AM General Hummer SUV 2000", ...}
id_to_class = {i+1: meta[i][0] for i in range(len(meta))}

# Crear carpeta base de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Procesar anotaciones
for item in tqdm(train_annos, desc="Copiando imágenes"):
    class_id = int(item[-2][0][0])  # ID clase
    filename = item[-1][0]          # Nombre archivo
    class_name = id_to_class[class_id].replace("/", "-")  # Quitar caracteres problemáticos

    class_path = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_path, exist_ok=True)

    src_img = os.path.join(TRAIN_DIR, filename)
    dst_img = os.path.join(class_path, filename)

    shutil.copyfile(src_img, dst_img)
