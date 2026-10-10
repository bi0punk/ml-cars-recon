"""Argparse helpers shared by the CLI entrypoints.

Centralizes camera/detection/ROI argument groups and resolution into the
``common.config`` dataclasses, so the entrypoints no longer duplicate argparse.
"""

import argparse
import os

from common.config import CameraConfig, DetectionConfig, ROIConfig, env_value


def add_camera_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Cámara")
    group.add_argument("--host", default=None, help="IP de la cámara (o env RTSP_HOST)")
    group.add_argument("--user", default=None, help="Usuario (o env RTSP_USER)")
    group.add_argument("--password", default=None, help="Contraseña (o env RTSP_PASSWORD)")
    group.add_argument(
        "--rtsp-channel",
        default=None,
        dest="rtsp_channel",
        help="Canal sub-stream para detección (o env RTSP_CHANNEL)",
    )
    group.add_argument(
        "--snapshot-channel",
        default=None,
        dest="snapshot_channel",
        help="Canal main-stream para snapshot/buffer (o env SNAPSHOT_CHANNEL)",
    )
    group.add_argument("--width", type=int, default=1280, help="Ancho de escala")
    group.add_argument("--height", type=int, default=720, help="Alto de escala")
    group.add_argument(
        "--rtsp-transport",
        default="tcp",
        choices=["udp", "tcp"],
        dest="rtsp_transport",
        help="Transporte RTSP (tcp = menor latencia)",
    )
    group.add_argument(
        "--substream-suffix",
        action="store_true",
        help="Algunos firmwares exigen el sufijo '01' en el canal (p.ej. 10101)",
    )


def add_detection_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Detección")
    group.add_argument("--model", default=None, help="Modelo YOLO (.pt) (o env YOLO_MODEL)")
    group.add_argument("--conf", type=float, default=0.45, help="Confianza mínima YOLO")
    group.add_argument("--iou", type=float, default=0.5, help="Umbral IoU")
    group.add_argument("--cooldown", type=float, default=0.8, help="Segundos entre capturas")
    group.add_argument(
        "--imgsz", type=int, default=None, help="Tamaño de entrada YOLO (por defecto el del modelo)"
    )
    group.add_argument("--device", default=None, help="Dispositivo (cpu, cuda:0, mps)")
    group.add_argument(
        "--infer-every-n",
        type=int,
        default=1,
        dest="infer_every_n",
        help="Correr inferencia solo cada N frames (para no saturar la CPU)",
    )


def add_roi_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("ROI")
    group.add_argument("--roi-width-ratio", type=float, default=0.50, dest="roi_width_ratio")
    group.add_argument("--roi-height-ratio", type=float, default=0.70, dest="roi_height_ratio")
    group.add_argument("--roi-center-x-ratio", type=float, default=0.50, dest="roi_center_x_ratio")
    group.add_argument("--roi-center-y-ratio", type=float, default=0.40, dest="roi_center_y_ratio")


def camera_config_from_args(args: argparse.Namespace) -> CameraConfig:
    return CameraConfig(
        host=args.host or env_value("RTSP_HOST", default="192.168.1.64"),
        user=args.user or env_value("RTSP_USER", default="admin"),
        password=args.password or env_value("RTSP_PASSWORD", default=""),
        rtsp_channel=str(args.rtsp_channel or env_value("RTSP_CHANNEL", default="102")),
        snapshot_channel=str(args.snapshot_channel or env_value("SNAPSHOT_CHANNEL", default="101")),
        width=args.width,
        height=args.height,
        rtsp_transport=args.rtsp_transport if args.rtsp_transport in ("udp", "tcp") else "tcp",
    )


def detection_config_from_args(args: argparse.Namespace) -> DetectionConfig:
    device = getattr(args, "device", None)
    if not device:
        device = os.environ.get("YOLO_DEVICE") or None
    imgsz = getattr(args, "imgsz", None)
    if not imgsz:
        imgsz = int(os.environ["YOLO_IMGSZ"]) if os.environ.get("YOLO_IMGSZ") else None

    env_infer = os.environ.get("YOLO_INFER_EVERY_N")
    infer_every_n = getattr(args, "infer_every_n", env_infer or 1)
    try:
        infer_every_n = int(infer_every_n)
    except (TypeError, ValueError):
        infer_every_n = 1

    return DetectionConfig(
        model_path=args.model or env_value("YOLO_MODEL", default="yolov8n.pt"),
        confidence=args.conf,
        iou_threshold=args.iou,
        cooldown=args.cooldown,
        pre_roll_ms=getattr(args, "pre_roll_ms", 300),
        device=device,
        imgsz=imgsz,
        infer_every_n=max(1, infer_every_n),
    )


def roi_config_from_args(args: argparse.Namespace) -> ROIConfig:
    return ROIConfig(
        width_ratio=args.roi_width_ratio,
        height_ratio=args.roi_height_ratio,
        center_x_ratio=args.roi_center_x_ratio,
        center_y_ratio=args.roi_center_y_ratio,
    )
