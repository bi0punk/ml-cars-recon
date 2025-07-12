import os
import re
from datetime import datetime

# Ruta del directorio con los archivos
directorio = "capturas"  # Cambia si necesitas otro path

# Patrones:
# 1. vehiculo_ o bus_ con milisegundos: bus_2025-07-08_15-43-22-184.jpg
# 2. vehiculo_ o bus_ sin milisegundos: vehiculo_2025-07-07_19-42-00.jpg
patron_completo = re.compile(r"^(vehiculo|bus)_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})(?:-(\d{3}))?\.jpg$")

# Procesar archivos
for nombre_archivo in os.listdir(directorio):
    match = patron_completo.match(nombre_archivo)
    if match:
        _, fecha, hora, ms = match.groups()
        if not ms:
            # Si no tiene milisegundos, se los generamos
            ms = datetime.now().strftime("%f")[:3]
        nuevo_nombre = f"car_{fecha}_{hora}-{ms}.jpg"
        ruta_vieja = os.path.join(directorio, nombre_archivo)
        ruta_nueva = os.path.join(directorio, nuevo_nombre)
        os.rename(ruta_vieja, ruta_nueva)
        print(f"✔ Renombrado: {nombre_archivo} → {nuevo_nombre}")
