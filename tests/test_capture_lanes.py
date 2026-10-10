"""Tests del gating de captura de ``capture_preroll_lanes``."""

import pytest

pytest.importorskip("cv2")

import capture_preroll_lanes as lanes_mod
from capture_preroll import CaptureSettings
from common.config import CameraConfig, DetectionConfig, ROIConfig


class _FakeModel:
    names = {}

    def __init__(self, *args, **kwargs):
        pass


class _GrabberStub:
    def __init__(self, *args, **kwargs):
        self.ok = False
        self.reconnect_count = 0

    def read(self):
        return False, None

    def get_closest_frame(self, target_timestamp):
        return None

    def get_buffer_info(self):
        return {"size": 0, "max_size": 0, "time_span": 0}

    def release(self):
        pass


def _make_system(monkeypatch) -> lanes_mod.PreRollLanesSystem:
    monkeypatch.setattr(lanes_mod, "YOLO", _FakeModel)
    monkeypatch.setattr(lanes_mod, "FrameGrabberLatest", _GrabberStub)
    monkeypatch.setattr(lanes_mod, "FrameGrabberBuffer", _GrabberStub)
    return lanes_mod.PreRollLanesSystem(
        CameraConfig("host", "user", "pass", "102", "101"),
        DetectionConfig("model.pt"),
        ROIConfig(),
        enable_lane_detection=False,
    )


class TestCaptureSettings:
    def test_defaults(self):
        settings = CaptureSettings()
        assert settings.save_dir == "captures_preroll"
        assert settings.pre_roll_ms == 300
        assert settings.sharp_window_ms == 200
        assert settings.sharp_max_frames == 9
        assert settings.fallback_isapi is False

    def test_is_immutable(self):
        settings = CaptureSettings(save_dir="x")
        with pytest.raises(AttributeError):
            settings.save_dir = "y"


class TestPreRollLanesCaptureGating:
    def test_does_not_capture_without_trigger(self, monkeypatch, tmp_path):
        system = _make_system(monkeypatch)
        assert system._maybe_capture(trigger_capture=False) is False
        assert system.stats["captures"] == 0

    def test_captures_on_trigger(self, monkeypatch, tmp_path):
        system = _make_system(monkeypatch)
        assert system._maybe_capture(trigger_capture=True) is True

    def test_respects_cooldown(self, monkeypatch):
        import time

        system = _make_system(monkeypatch)
        assert system._maybe_capture(True) is True
        system._last_capture_ts = time.time()  # captura reciente -> cooldown activo
        assert system._maybe_capture(True) is False


class TestPrerollCli:
    def test_filter_to_road_requires_poly(self, monkeypatch):
        import sys

        import capture_preroll

        monkeypatch.setattr(
            sys, "argv", ["capture_preroll.py", "--filter-to-road", "--password", "x"]
        )
        with pytest.raises(SystemExit) as exc:
            capture_preroll.main()
        assert exc.value.code == 2
