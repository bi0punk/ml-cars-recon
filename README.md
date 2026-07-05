# ml-cars-recon

Computer vision system for vehicle detection and license plate capture from IP camera RTSP streams using YOLO object detection models and OpenCV.

## Stack

Python 3, Ultralytics YOLOv8, OpenCV, Flask

## Scripts

| Script | Purpose |
|---|---|
| `app_yolo_patentes.py` | YOLO-based license plate detection |
| `detecta_autos_y_captura_roi_central.py` | Car counting with ROI |
| `captura_autos_con_carril.py` | Lane-based vehicle capture |
| `basic_car_detector.py` | Basic car detection demo |
| `app_autos_captura.py` | Auto capture from RTSP |
| `web_app/` | Flask web interface |

## Environment Variables

Create a `.env` file (use `.env.example` as template):

```env
RTSP_USER=admin
RTSP_PASS=your_password
RTSP_HOST=192.168.1.64
RTSP_PORT=554
RTSP_PATH=/Streaming/Channels/101
CONFIDENCE_THRESHOLD=0.5
```

## Usage

```bash
pip install -r web_app/requirements.txt
python app_yolo_patentes.py
```

## Tests

```bash
pip install pytest
pytest -q
```

## CI

GitHub Actions: lint (ruff) + pytest on every push.

## Data

Model weights (`.pt`, `.h5`) and captures are excluded from version control.
Download YOLOv8n from Ultralytics or train your own model for the detector scripts.

## License

MIT
