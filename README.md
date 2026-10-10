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
- Captura de snapshots ISAPI al pasar el vehículo por el ROI
- Captura con **pre-roll** desde el main-stream (frame anterior al disparo)
- Selección del frame **más nítido** en la ventana de pre-roll
- Disparo **estable** (N frames consecutivos) y filtro opcional por polígono de carretera
- Detección de carriles (Canny + Hough) con panel de estadísticas en vivo
- Interfaz web Flask (`web_app`) con `/api/status`, MJPEG y snapshots
- Reconexión automática con **backoff exponencial** ante pérdida del stream RTSP
- `web_app` sin efectos colaterales al importar (cámara y modelo perezosos)
- Toda la lógica compartida en `common/` (sin copiar/pegar)

## Stack

- Python 3.11+, Ultralytics YOLOv8, OpenCV, Flask, NumPy

## Arquitectura

```
ml-cars-recon/
├── common/                           # Módulos compartidos (ver detalle abajo)
│   ├── cli.py                        # Argumentos CLI comunes (cámara/detección/ROI)
│   ├── config.py                     # Helpers de env + dataclasses de configuración
│   ├── utils.py                      # FFmpeg low-latency, JPEG, nitidez, StableTrigger
│   ├── rtsp.py                       # Builders de URLs RTSP (flavors streaming/ISAPI)
│   ├── isapi.py                      # Snapshot vía ISAPI (Digest auth)
│   ├── frames.py                     # Grabbers (latest + buffer pre-roll) + best_frame
│   ├── detector.py                   # VehicleDetector: inferencia YOLO + snapshot ISAPI
│   ├── lanes.py                      # Detector de carriles
│   └── geometry.py                   # ROI, bounding boxes y polígonos
├── capture_isapi.py                  # Snapshot ISAPI cuando el vehículo cruza el ROI
├── capture_preroll.py                # Pre-roll + frame nítido + estabilidad + road filter
├── capture_preroll_lanes.py          # Pre-roll + carriles + estadísticas en vivo
├── web_app/                          # Interfaz web Flask
├── tests/                            # Smoke + unitarios de common/ + web app
├── pyproject.toml                    # Dependencias de proyecto (fuente única)
├── .env.example
└── README.md
```

### Módulos compartidos

| Módulo            | Responsabilidad                                                        |
|-------------------|------------------------------------------------------------------------|
| `common/cli.py`   | Grupos de argparse (cámara/detección/ROI) + resolución a configs        |
| `common/config.py`| `CameraConfig`, `DetectionConfig`, `ROIConfig` + `env_value`            |
| `common/frames.py`| `FrameGrabberLatest`, `FrameGrabberBuffer` (reconexión con backoff), `best_frame` |
| `common/detector.py` | `VehicleDetector`: predicción en ROI, `infer_every_n`, `device`/`imgsz`, margen y `box_filter` |
| `common/geometry.py` | `box_inside_roi`, `compute_roi`, polígonos de carretera              |

### Reconexión con backoff exponencial

Los grabbers de `common/frames.py` detectan streams muertos: si `read()`
falla `stale_frame_threshold` veces seguidas (por defecto 5), se fuerza la
reapertura del stream. El delay de reconexión crece exponencialmente
(`reconnect_delay` → `max_reconnect_delay`, por defecto 2s → 60s) y se
resetea al recibir un frame válido. Parámetros configurables en el constructor.

Directorios de captura (runtime, excluidos de git):

```
captures_preroll/    # pre-roll del main-stream (capture_preroll*)
captures_fallback/   # frames de respaldo del sub-stream
isapi_snaps/         # snapshots descargados vía ISAPI (y web app)
```

## Requisitos

- Python 3.11+
- Cámara IP con stream RTSP
- Modelo YOLOv8 (descargado automáticamente por ultralytics la primera vez)
- GPU recomendada para rendimiento en tiempo real (usa `--device cpu/cuda:0/mps`)

## Instalación

Las dependencias se declaran **solo** en `pyproject.toml`:

```bash
git clone https://github.com/bi0punk/ml-cars-recon.git
cd ml-cars-recon
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

> El modelo `yolov8n.pt` se descarga automáticamente al ejecutar cualquier
> script (o descárgalo con `python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"`).

## Uso

```bash
# 1) Snapshot ISAPI cuando el vehículo esté COMPLETO dentro del ROI
python capture_isapi.py --host 192.168.1.64 --user admin --password tu_pass \
    --rtsp-channel 102 --snapshot-channel 101

# 2) Pre-roll desde el main-stream con el frame más nítido + disparo estable
python capture_preroll.py --host 192.168.1.64 --user admin --password tu_pass \
    --pre-roll-ms 300 --sharp-window-ms 200 --stable-frames 3 --rearm-frames 8 \
    --filter-to-road --road-poly "0.15,0.55;0.85,0.55;0.95,0.98;0.05,0.98"

# 3) Pre-roll + carriles + panel de estadísticas
python capture_preroll_lanes.py --host 192.168.1.64 --user admin --password tu_pass

# Opciones de rendimiento comunes (todas cargadas desde .env si está definida):
#   --device cpu|0|cuda:0|mps   --imgsz 640   --infer-every-n 2
#   YOLO_DEVICE / YOLO_IMGSZ / YOLO_INFER_EVERY_N

# Web app (Flask)
python web_app/app.py
```

Opciones útiles en todos los script: `--help` para ver grupos de argumentos
(cámara, detección, ROI) y variables de entorno en `.env.example` como fallback
para las credenciales.

## Tests

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest -q
```

## Configuración

Variables de entorno (ver `.env.example`):

| Variable               | Default                        | Descripción                              |
|------------------------|--------------------------------|------------------------------------------|
| `RTSP_USER`            | `admin`                        | Usuario RTSP                             |
| `RTSP_PASSWORD`        | —                              | Contraseña RTSP                          |
| `RTSP_HOST`            | `192.168.1.64`                 | IP de la cámara                          |
| `RTSP_CHANNEL`         | `102`                          | Canal sub-stream (detección)             |
| `SNAPSHOT_CHANNEL`     | `101`                          | Canal main-stream (snapshot o buffer)    |
| `YOLO_MODEL`           | `yolov8n.pt`                   | Modelo YOLO (CLIs)                       |
| `YOLO_DEVICE`          | —                              | `cpu`/`cuda:0`/`mps` (CLIs)              |
| `YOLO_IMGSZ`           | —                              | Tamaño de entrada YOLO (CLIs)            |
| `YOLO_INFER_EVERY_N`   | `1`                            | Inferencia cada N frames (CLIs)          |
| `RTSP_URL`             | —                              | URL completa del stream (web app)        |
| `CAPTURE_DIR`          | `isapi_snaps`                  | Directorio de capturas (web app)         |
| `LATEST_LIMIT`         | `3`                            | Imágenes recientes a listar              |
| `MODEL_PATH`           | `yolov8n.pt`                   | Modelo YOLO (web app)                    |
| `INFER_EVERY_N`        | `2`                            | Inferir cada N frames (web app)          |
| `IMG_SIZE`             | `640`                          | Tamaño de entrada YOLO (web app)         |
| `ROI_W_PCT` / `ROI_H_PCT` / `ROI_CY_PCT` | `0.60`/`0.50`/`0.45` | ROI relativo (web app)     |
| `ISAPI_HOST` / `ISAPI_USER` / `ISAPI_PASSWORD` | `192.168.1.64`/`admin`/— | Credenciales ISAPI (web app) |
| `SNAPSHOT_COOLDOWN`    | `1.0`                          | Seg entre snapshots (web app)            |
| `HOST` / `PORT`        | `127.0.0.1`/`5000`             | Bind del servidor Flask                  |
| `API_TOKEN`            | —                              | Token opcional para `/api` y `/video_feed` |

## CI

GitHub Actions ejecuta en cada push y PR:

- `ruff check .`
- `ruff format --check .`
- `pytest -q` (dependencias ligeras: `numpy`, `opencv-python`, `requests`, `flask`, `python-dotenv` — sin `ultralytics`/`torch` para CI rápida).

## Datos

- Los pesos de modelos (`.pt`, `.h5`) y capturas están excluidos del control de versiones
- Descargar `yolov8n.pt` desde Ultralytics o entrenar un modelo personalizado
- Las capturas se guardan en `captures_preroll/`, `captures_fallback/` e `isapi_snaps/`

## Limitaciones / Roadmap

- [x] Reconexión robusta con backoff exponencial ante pérdida del stream
- [x] Detección compartida en `common/detector.py` (con `infer_every_n`, device/imgz)
- [x] CLI consolidado (3 entrypoints, sin scripts duplicados)
- [x] Web app sin efectos colaterales + `/api/status` + auth por token
- [ ] Reconocimiento OCR de patentes con EasyOCR/PaddleOCR
- [ ] Seguimiento de vehículos multi-frame (tracking)
- [ ] Dashboard web con estadísticas en tiempo real
- [ ] Almacenamiento en base de datos de detecciones

## Licencia

MIT