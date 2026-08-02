import os

from dotenv import load_dotenv

load_dotenv()

RTSP_USER = os.environ.get("RTSP_USER", "admin")
RTSP_PASSWORD = os.environ.get("RTSP_PASSWORD", "")
RTSP_HOST = os.environ.get("RTSP_HOST", "192.168.1.64")
RTSP_CHANNEL = os.environ.get("RTSP_CHANNEL", "101")
RTSP_TRANSPORT = os.environ.get("RTSP_TRANSPORT", "tcp")
YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8n.pt")
ISAPI_HOST = os.environ.get("ISAPI_HOST", RTSP_HOST)
ISAPI_USER = os.environ.get("ISAPI_USER", RTSP_USER)
ISAPI_PASSWORD = os.environ.get("ISAPI_PASSWORD", RTSP_PASSWORD)
