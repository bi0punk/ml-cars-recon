# ml-cars-recon

Computer vision system for vehicle detection and license plate capture from IP camera RTSP streams using YOLO object detection models.

## Stack

Python 3, Ultralytics YOLOv8, OpenCV, Flask

## Scripts

| Script | Purpose |
|---|---|
| `app_yolo_patentes.py` | YOLO-based license plate detection |
| `detecta_autos_y_captura_roi_central.py` | Car counting with ROI |
| `captura_autos_con_carril.py` | Lane-based vehicle capture |
| `basic_car_detector.py` | Basic car detection demo |
| `web_app/` | Flask web interface |

## Environment Variables

Create a `.env` file or export:

```env
RTSP_USER=admin
RTSP_PASS=your_password
RTSP_HOST=192.168.1.64
RTSP_PORT=554
RTSP_PATH=/Streaming/Channels/101
ISAPI_USER=admin
ISAPI_PASSWORD=your_password
```

## Usage

```bash
pip install -r web_app/requirements.txt
python app_yolo_patentes.py
```

## License

MIT
