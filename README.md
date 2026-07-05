# ml-cars-recon

Computer vision system for vehicle detection and license plate capture from IP camera RTSP streams using YOLO object detection models and OpenCV.

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![CI](https://github.com/tu-usuario/ml-cars-recon/actions/workflows/ci.yml/badge.svg)](https://github.com/tu-usuario/ml-cars-recon/actions/workflows/ci.yml)

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
- Interfaz web Flask para visualización (web_app)
- Reconexión automática ante pérdida del stream RTSP
- Modo estable sin creación múltiple de ventanas

## Stack

- Python 3.11+, Ultralytics YOLOv8, OpenCV, Flask, NumPy

## Arquitectura

```
ml-cars-recon/
├── app_yolo_patentes.py              # Detección YOLO de patentes
├── detecta_autos_y_captura_roi_central.py  # Conteo con ROI
├── captura_autos_con_carril.py       # Captura por carril
├── basic_car_detector.py             # Demo básica
├── app_autos_captura.py              # Captura automática RTSP
├── app_autos_captures_ultimo.py      # Captura con frame final
├── app_forma_patentes_basico.py      # Forma básica de patentes
├── testapp.py                        # App de prueba
├── captures_isapi/                   # Capturas desde ISAPI
├── captures_preroll/                 # Capturas pre-roll
├── web_app/                          # Interfaz web Flask
├── tests/
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

## Requisitos

- Python 3.11+
- Cámara IP con stream RTSP
- Modelo YOLOv8 (descargado automáticamente por ultralytics)
- GPU recomendada para rendimiento en tiempo real

## Instalación

```bash
git clone https://github.com/tu-usuario/ml-cars-recon.git
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
```

## Tests

```bash
pip install pytest ruff
pytest -q
ruff check .
```

## Configuración

Variables de entorno (ver `.env.example`):

| Variable              | Default                        | Descripción                          |
|-----------------------|--------------------------------|--------------------------------------|
| `RTSP_USER`           | `admin`                        | Usuario RTSP                         |
| `RTSP_PASS`           | —                              | Contraseña RTSP                      |
| `RTSP_HOST`           | `192.168.1.64`                 | IP de la cámara                      |
| `RTSP_PORT`           | `554`                          | Puerto RTSP                          |
| `RTSP_PATH`           | `/Streaming/Channels/101`      | Path del stream RTSP                 |
| `CONFIDENCE_THRESHOLD`| `0.5`                          | Umbral de confianza YOLO             |

## CI

GitHub Actions ejecuta ruff lint + pytest en cada push y PR.

## Datos

- Los pesos de modelos (`.pt`, `.h5`) y capturas están excluidos del control de versiones
- Descargar `yolov8n.pt` desde Ultralytics o entrenar un modelo personalizado
- Las capturas se guardan en `captures_isapi/` y `captures_preroll/`

## Limitaciones / Roadmap

- [ ] Reconocimiento OCR de patentes con EasyOCR/PaddleOCR
- [ ] Seguimiento de vehículos multi-frame (tracking)
- [ ] Dashboard web con estadísticas en tiempo real
- [ ] Almacenamiento en base de datos de detecciones

## Licencia

MIT
