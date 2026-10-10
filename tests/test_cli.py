"""Tests del módulo común de argparse (common.cli)."""

import argparse

import pytest

from common import cli
from common.config import CameraConfig, DetectionConfig, ROIConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    cli.add_camera_arguments(parser)
    cli.add_detection_arguments(parser)
    cli.add_roi_arguments(parser)
    return parser


class TestCliCameraConfig:
    def test_defaults(self, monkeypatch):
        for key in ("RTSP_HOST", "RTSP_USER", "RTSP_PASSWORD", "RTSP_CHANNEL", "SNAPSHOT_CHANNEL"):
            monkeypatch.delenv(key, raising=False)
        args = build_parser().parse_args([])
        cfg = cli.camera_config_from_args(args)
        assert isinstance(cfg, CameraConfig)
        assert cfg.host == "192.168.1.64"
        assert cfg.user == "admin"
        assert cfg.password == ""
        assert cfg.rtsp_channel == "102"
        assert cfg.snapshot_channel == "101"
        assert cfg.rtsp_transport == "tcp"

    def test_args_override_env(self, monkeypatch):
        monkeypatch.setenv("RTSP_HOST", "env-host")
        args = build_parser().parse_args(["--host", "cli-host", "--user", "cli-user"])
        cfg = cli.camera_config_from_args(args)
        assert cfg.host == "cli-host"
        assert cfg.user == "cli-user"

    def test_env_fallback(self, monkeypatch):
        monkeypatch.setenv("RTSP_HOST", "env-host")
        args = build_parser().parse_args([])
        cfg = cli.camera_config_from_args(args)
        assert cfg.host == "env-host"


class TestCliDetectionConfig:
    def test_defaults(self, monkeypatch):
        for key in ("YOLO_MODEL", "YOLO_DEVICE", "YOLO_IMGSZ", "YOLO_INFER_EVERY_N"):
            monkeypatch.delenv(key, raising=False)
        args = build_parser().parse_args([])
        cfg = cli.detection_config_from_args(args)
        assert isinstance(cfg, DetectionConfig)
        assert cfg.model_path == "yolov8n.pt"
        assert cfg.confidence == pytest.approx(0.45)
        assert cfg.iou_threshold == pytest.approx(0.5)
        assert cfg.cooldown == pytest.approx(0.8)
        assert cfg.pre_roll_ms == 300
        assert cfg.imgsz is None
        assert cfg.device is None
        assert cfg.infer_every_n == 1

    def test_env_fallbacks(self, monkeypatch):
        monkeypatch.setenv("YOLO_DEVICE", "cuda:0")
        monkeypatch.setenv("YOLO_IMGSZ", "320")
        monkeypatch.setenv("YOLO_INFER_EVERY_N", "3")
        args = build_parser().parse_args([])
        cfg = cli.detection_config_from_args(args)
        assert cfg.device == "cuda:0"
        assert cfg.imgsz == 320
        assert cfg.infer_every_n == 3

    def test_args_beat_env(self, monkeypatch):
        monkeypatch.setenv("YOLO_DEVICE", "cuda:0")
        monkeypatch.setenv("YOLO_IMGSZ", "320")
        args = build_parser().parse_args(["--device", "cpu", "--imgsz", "640"])
        cfg = cli.detection_config_from_args(args)
        assert cfg.device == "cpu"
        assert cfg.imgsz == 640

    def test_invalid_env_imgsz_ignored(self, monkeypatch):
        monkeypatch.setenv("YOLO_IMGSZ", "no-es-numero")
        args = build_parser().parse_args([])
        assert cli.detection_config_from_args(args).imgsz is None

    def test_parses_values(self):
        args = build_parser().parse_args(
            [
                "--model",
                "mi_modelo.pt",
                "--conf",
                "0.3",
                "--iou",
                "0.4",
                "--device",
                "cpu",
                "--imgsz",
                "320",
                "--infer-every-n",
                "4",
            ]
        )
        cfg = cli.detection_config_from_args(args)
        assert cfg.model_path == "mi_modelo.pt"
        assert cfg.confidence == pytest.approx(0.3)
        assert cfg.iou_threshold == pytest.approx(0.4)
        assert cfg.device == "cpu"
        assert cfg.imgsz == 320
        assert cfg.infer_every_n == 4


class TestCliRoiConfig:
    def test_defaults(self):
        args = build_parser().parse_args([])
        cfg = cli.roi_config_from_args(args)
        assert isinstance(cfg, ROIConfig)
        assert cfg.width_ratio == pytest.approx(0.50)
        assert cfg.height_ratio == pytest.approx(0.70)
        assert cfg.center_y_ratio == pytest.approx(0.40)

    def test_custom(self):
        args = build_parser().parse_args(
            ["--roi-width-ratio", "0.6", "--roi-height-ratio", "0.4", "--roi-center-y-ratio", "0.5"]
        )
        cfg = cli.roi_config_from_args(args)
        assert cfg.width_ratio == pytest.approx(0.6)
        assert cfg.height_ratio == pytest.approx(0.4)
        assert cfg.center_y_ratio == pytest.approx(0.5)
