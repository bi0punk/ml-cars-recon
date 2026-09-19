# ml-cars-recon

Computer vision system for vehicle detection and license plate capture from IP camera RTSP streams using YOLO object detection models and OpenCV.

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![CI](https://github.com/bi0punk/ml-cars-recon/actions/workflows/ci.yml/badge.svg)](https://github.com/bi0punk/ml-cars-recon/actions/workflows/ci.yml)

## Tabla de Contenidos

- [Características](#características)
- [Stack](#stack)
- [Arquitectura](#arquitectura)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Tests](#tests)
- [Configuración](#configuración)
- [CI](#ci)
- [Datos](#datos)
- [Limitaciones / Roadmap](#limitaciones--roadmap)
- [Licencia](#licencia)

## Características

- Detección de vehículos (autos, camiones) en streams RTSP con YOLOv8
- Captura y reconocimiento de patentes/license plates
- Conteo de vehículos con ROI central configurable
- Captura por carril con filtro de región de interés
- Interfaz web Flask para visualización (web_app) con endpoint `/api/status`
- Reconexión automática con **backoff exponencial** ante pérdida del stream RTSP
- `web_app` sin efectos colaterales al importar (cámara y modelo perezosos)
- Lógica de detección compartida en `common/detector.py` (sin copiar/pegar)
- Modo estable sin creación múltiple de ventanas

## Stack

- Python 3.11+, Ultralytics YOLOv8, OpenCV, Flask, NumPy

## Arquitectura

```
ml-cars-recon/
├── common/                           # Módulos compartidos (ver detalle abajo)
│   ├── config.py                     # Helpers de env + dataclasses de configuración
│   ├── utils.py                      # FFmpeg low-latency, directorios, JPEG, nitidez
│   ├── rtsp.py                       # Builders de URLs RTSP (flavors streaming/ISAPI)
│   ├── isapi.py                      # Snapshot vía ISAPI (Digest auth)
│   ├── frames.py                     # Grabbers de frames en hilo (latest + buffer pre-roll)
│   ├── detector.py                   # VehicleDetector: inferencia YOLO + snapshot ISAPI
│   ├── lanes.py                      # Detector de carriles
│   └── geometry.py                   # ROI y geometría de bounding boxes
├── app_yolo_patentes.py              # Detección YOLO de patentes
├── detecta_autos_y_captura_roi_central.py  # Conteo con ROI central
├── captura_autos_con_carril.py       # Captura por carril
├── basic_car_detector.py             # Demo básica
├── app_autos_captura.py              # Captura automática RTSP
├── app_autos_captures_ultimo.py      # Captura con pre-roll desde MAIN
├── app_forma_patentes_basico.py      # Forma básica de patentes
├── testapp.py                        # App de prueba
├── web_app/                          # Interfaz web Flask
├── tests/                            # Smoke + tests unitarios de common/
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

Todos los scripts comparten la lógica a través de `common/` (grabbers con
reconexión, builders RTSP, ISAPI, conveniencia y `VehicleDetector`), en lugar
de copiar y pegar el mismo código.

### Reconexión con backoff exponencial

Los grabbers de `common/frames.py` detectan streams muertos: si `read()`
falla `stale_frame_threshold` veces seguidas (por defecto 5), se fuerza la
reapertura del stream. El delay de reconexión crece exponencialmente
(`reconnect_delay` → `max_reconnect_delay`, por defecto 2s → 60s) y se
resetea al recibir un frame válido. Estos parámetros son configurables en el
constructor de `FrameGrabberLatest` / `FrameGrabberBuffer`.

Directorios de captura (runtime, excluidos de git):

```
captures_preroll/    # pre-roll del main-stream (captura por carril / testapp)
captures_fallback/   # frames de respaldo del sub-stream
captures_lpr/        # frames seleccionados por nitidez (LPR)
isapi_snaps/         # snapshots descargados vía ISAPI
```

## Requisitos

- Python 3.11+
- Cámara IP con stream RTSP
- Modelo YOLOv8 (descargado automáticamente por ultralytics)
- GPU recomendada para rendimiento en tiempo real

## Instalación

```bash
git clone https://github.com/bi0punk/ml-cars-recon.git
cd ml-cars-recon
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
# Detección de patentes con YOLO
python app_yolo_patentes.py --user admin --password tu_pass --host 192.168.1.64

# Conteo de autos con ROI central
python detecta_autos_y_captura_roi_central.py

# Captura por carril
python captura_autos_con_carril.py

# Web app (Flask)
python web_app/app.py
```

> **Nota sobre URLs RTSP:** algunos firmwares exigen el sufijo `01` en el
> canal de sub-stream (p.ej. `10101`). Si tu cámara no conecta, usa la
> bandera `--substream-suffix` en los scripts de captura.

## Tests

```bash
pip install pytest ruff
pytest -q
ruff check .
```

## Configuración

Variables de entorno (ver `.env.example`):

| Variable               | Default                        | Descripción                              |
|------------------------|--------------------------------|------------------------------------------|
| `RTSP_USER`            | `admin`                        | Usuario RTSP                             |
| `RTSP_PASSWORD`        | —                              | Contraseña RTSP                          |
| `RTSP_HOST`            | `192.168.1.64`                 | IP de la cámara                          |
| `RTSP_PORT`            | `554`                          | Puerto RTSP                              |
| `RTSP_CHANNEL`         | `101`                          | Canal RTSP (102 = sub-stream)            |
| `SNAPSHOT_CHANNEL`     | `101`                          | Canal ISAPI para snapshots (main)        |
| `YOLO_MODEL`           | `yolov8n.pt`                   | Modelo YOLO                              |
| `YOLO_CONF`            | `0.45`                         | Umbral de confianza YOLO (scripts)       |
| `CONFIDENCE_THRESHOLD` | `0.35`                         | Umbral de confianza (basic_car_detector) |
| `RTSP_URL`             | —                              | URL completa del stream (web app)        |
| `CAPTURE_DIR`          | `isapi_snaps`                  | Directorio de capturas (web app)         |
| `LATEST_LIMIT`         | `3`                            | Imágenes recientes a listar              |
| `MODEL_PATH`           | `yolov8n.pt`                   | Modelo YOLO (web app)                    |
| `INFER_EVERY_N`        | `2`                            | Inferir cada N frames (web app)          |
| `IMG_SIZE`             | `640`                          | Tamaño de entrada YOLO                   |
| `ROI_W_PCT` / `ROI_H_PCT` / `ROI_CY_PCT` | `0.60`/`0.50`/`0.45` | ROI relativo (web app)        |
| `ISAPI_HOST` / `ISAPI_USER` / `ISAPI_PASSWORD` | `192.168.1.64`/`admin`/— | Credenciales ISAPI (web app) |
| `SNAPSHOT_COOLDOWN`    | `1.0`                          | Seg entre snapshots (web app)            |
| `HOST` / `PORT`        | `127.0.0.1`/`5000`             | Bind del servidor Flask                  |
| `API_TOKEN`            | —                              | Token opcional para rutas `/api`/`/video_feed` |

## CI

GitHub Actions ejecuta `ruff check .` y `pytest -q` en cada push y PR. El job
de tests instala las dependencias ligeras necesarias (`numpy`, `opencv-python`,
`requests`, `flask`, `python-dotenv`) — sin `ultralytics`/`torch` para CI
rápida, ya que los tests no importan YOLO.

## Datos

- Los pesos de modelos (`.pt`, `.h5`) y capturas están excluidos del control de versiones
- Descargar `yolov8n.pt` desde Ultralytics o entrenar un modelo personalizado
- Las capturas se guardan en `captures_preroll/`, `captures_fallback/`, `captures_lpr/` e `isapi_snaps/`

## Limitaciones / Roadmap

- [x] Reconexión robusta con backoff exponencial ante pérdida del stream
- [x] Módulo compartido de detección (`common/detector.py`)
- [x] Web app sin efectos colaterales al importar + endpoint `/api/status`
- [ ] Reconocimiento OCR de patentes con EasyOCR/PaddleOCR
- [ ] Seguimiento de vehículos multi-frame (tracking)
- [ ] Dashboard web con estadísticas en tiempo real (base en `/api/status`)
- [ ] Almacenamiento en base de datos de detecciones

## Licencia

MIT
