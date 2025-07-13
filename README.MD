**Optimized YOLO Streaming**

Este proyecto proporciona una solución optimizada para la captura y detección en tiempo real de vehículos (coches y autobuses) desde una cámara IP mediante un modelo YOLOv8.

---

## Índice

1. [Descripción](#descripción)
2. [Características principales](#características-principales)
3. [Requisitos](#requisitos)
4. [Instalación](#instalación)
5. [Estructura del proyecto](#estructura-del-proyecto)
6. [Configuración](#configuración)
7. [Uso](#uso)
8. [Detalles de implementación](#detalles-de-implementación)

   * Clase `OptimizedYOLOStreaming`
   * Métodos clave
   * Optimización y threading
9. [Controles en tiempo de ejecución](#controles-en-tiempo-de-ejecución)
10. [Mejoras y extensiones](#mejoras-y-extensiones)
11. [Licencia](#licencia)

---

## Descripción

Este script conecta con un stream RTSP de una cámara IP, procesa los frames con un modelo YOLOv8 entrenado para detectar vehículos (`car` y `bus`), y guarda automáticamente recortes de los vehículos que pasan por una zona central definida. Para minimizar la latencia y optimizar el rendimiento, utiliza:

* **Buffer y threading**: captura de frames en segundo plano con un buffer limitado.
* **Salto de frames**: solo procesa cada N-ésimo frame con el modelo.
* **Reducción de resolución temporal** para inferencia más rápida.

---

## Características principales

* Detección en streaming de vehículos (`car`, `bus`).
* Guardado automático de imágenes de vehículos en zona central.
* Configuración ajustable de resolución, FPS y salto de frames.
* Mínima latencia mediante buffer de captura y subprocesos.
* Zona central configurable con alertas visuales (caja amarilla).
* Mensajes en consola para eventos clave.

---

## Requisitos

* Python 3.8+
* [ultralytics](https://pypi.org/project/ultralytics/)
* OpenCV (`cv2`)
* NumPy
* zoneinfo (incluido en Python 3.9+)

Instalación de dependencias:

```bash
pip install ultralytics opencv-python numpy
```

---

## Instalación

1. Clonar o descargar este repositorio.
2. Asegurarse de tener el modelo `yolov8n.pt` en la raíz del proyecto (o indicar otra ruta).
3. Instalar dependencias con `pip`.

---

## Estructura del proyecto

```
├── optimized_yolo_streaming.py    # Script principal
├── yolov8n.pt                     # Modelo YOLOv8 (pesos)
└── capturas/                      # Carpeta donde se guardan imágenes
```

---

## Configuración

* **RTSP URL**: en `if __name__ == "__main__"`, modificar `ip_camera_url` con la dirección de la cámara.
* **Modelo**: por defecto `yolov8n.pt`, ajustar ruta en el constructor de `OptimizedYOLOStreaming`.
* **Salto de frames**: atributo `skip_frames` (por defecto 3).
* **Buffer de captura**: tamaño de `frame_queue` (por defecto 3).
* **FPS**: `cap.set(cv2.CAP_PROP_FPS, 15)`.
* **Zona central**: definida mediante márgenes relativos al centro del frame (`margen_x = 25% ancho`, `margen_y = 30% alto`).
* **Tipos de vehículos**: lista `vehiculos_interes = ['car','bus']`.

---

## Uso

```bash
python optimized_yolo_streaming.py
```

Al iniciar, verá en consola:

```
🚀 Iniciando streaming optimizado...
📝 Consejos:
   - Presiona 'q' para salir
   - Presiona 's' para saltar procesamiento YOLO por 10 frames
   - Los vehículos en la zona amarilla se guardarán automáticamente
```

La ventana mostrará el stream con detecciones y zona central resaltada.

---

## Detalles de implementación

### Clase `OptimizedYOLOStreaming`

Encapsula toda la lógica de:

* Conexión y configuración de la cámara (`setup_camera`).
* Hilo de captura de frames (`capture_frames`).
* Procesamiento optimizado con YOLO (`process_with_yolo`).
* Dibujo de detecciones y gestión de recortes (`draw_detections`, `save_vehicle_image`).
* Identificación de vehículos únicos por bloque de píxeles (`get_vehicle_key`).

#### Métodos clave

* **`setup_camera()`**: configura `cv2.VideoCapture` con buffer mínimo y FPS limitado.
* **`capture_frames(cap)`**: hilo que lee frames y mantiene un buffer FIFO.
* **`process_with_yolo(frame)`**: redimensiona el frame, ejecuta inferencia y filtra resultados.
* **`draw_detections(frame, detections)`**: dibuja cajas, texto, zona central y dispara guardado.
* **`save_vehicle_image(...)`**: recorta, crea carpeta `capturas` y guarda la imagen.
* **`run()`**: orquesta la inicialización, bucle principal, input de teclado y limpieza.

#### Optimización y threading

1. **Thread de captura**: evita bloqueos en la lectura RTSP.
2. **Buffer pequeños**: reduce latencia, siempre trabaja con el frame más reciente.
3. **Skip frames**: solo inferencia cada N cuadros.
4. **Reducción temporal de resolución**: velocidad vs precisión.

---

## Controles en tiempo de ejecución

* **`q`**: salir del streaming.
* **`s`**: saltar inferencia YOLO durante los próximos 10 frames.

---

## Mejoras y extensiones

* Soporte para detección de más clases (añadir a `vehiculos_interes`).
* Configuración vía archivo `.yaml` o parámetros CLI.
* Interfaz web para visualizar stream y capturas.
* Integración con base de datos o notificaciones (Telegram, email).
* Ajuste dinámico de resolución y FPS según carga.

---

## Licencia

Este proyecto se distribuye bajo la **Licencia MIT**. Ver `LICENSE` para más detalles.
