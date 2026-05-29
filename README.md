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

## Usage

```bash
pip install -r web_app/requiremens.txt
python app_yolo_patentes.py
```

## License

MIT
