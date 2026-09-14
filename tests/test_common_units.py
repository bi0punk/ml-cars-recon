"""Tests unitarios sobre los módulos de ``common/``."""

import os
import time
from collections import deque

import numpy as np
import pytest

from common.config import env_value
from common.frames import FrameGrabberBuffer, FrameGrabberLatest
from common.geometry import box_inside_roi, compute_roi
from common.isapi import snapshot_url
from common.rtsp import (
    build_isapi_channel_url,
    build_isapi_url,
    build_streaming_channel_url,
    build_streaming_url,
)
from common.utils import save_jpeg, set_ffmpeg_low_latency_env, sharpness


# ---------------------------------------------------------------------------
# common.geometry
# ---------------------------------------------------------------------------
class TestGeometry:
    def test_box_fully_inside_roi(self):
        roi = (100, 100, 500, 400)
        assert box_inside_roi((150, 150, 450, 350), roi) is True

    def test_box_touching_margin_is_outside(self):
        roi = (100, 100, 500, 400)
        assert box_inside_roi((100, 100, 500, 400), roi, margin=1) is False

    def test_box_outside_roi(self):
        roi = (100, 100, 500, 400)
        assert box_inside_roi((50, 50, 200, 200), roi) is False

    def test_compute_roi_centered(self):
        x1, y1, x2, y2 = compute_roi(1280, 720, roi_w=0.50, roi_h=0.70, roi_cy=0.40)
        cx, cy = 1280 // 2, int(720 * 0.40)
        rw, rh = int(1280 * 0.50), int(720 * 0.70)
        assert x1 == cx - rw // 2
        assert x2 == x1 + rw
        assert y1 == cy - rh // 2
        assert y2 == y1 + rh
        assert (x2 - x1) == rw
        assert (y2 - y1) == rh

    def test_compute_roi_clamped_to_frame(self):
        x1, y1, x2, y2 = compute_roi(100, 100, roi_w=0.90, roi_h=0.90, roi_cy=0.10)
        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= 100
        assert y2 <= 100


# ---------------------------------------------------------------------------
# common.rtsp
# ---------------------------------------------------------------------------
class TestRtspUrls:
    HOST = "192.168.1.64"
    USER = "admin"
    PASS = "clave"
    CH = "101"

    def test_streaming_url(self):
        assert build_streaming_url(self.HOST, self.USER, self.PASS, self.CH) == (
            "rtsp://admin:clave@192.168.1.64:554/Streaming/Channels/101"
        )

    def test_streaming_channel_url_appends_substream(self):
        assert build_streaming_channel_url(self.HOST, self.USER, self.PASS, self.CH) == (
            "rtsp://admin:clave@192.168.1.64:554/Streaming/Channels/10101"
        )

    def test_isapi_url(self):
        assert build_isapi_url(self.HOST, self.USER, self.PASS, self.CH) == (
            "rtsp://admin:clave@192.168.1.64:554/ISAPI/Streaming/channels/101"
        )

    def test_isapi_channel_url_appends_substream(self):
        assert build_isapi_channel_url(self.HOST, self.USER, self.PASS, self.CH) == (
            "rtsp://admin:clave@192.168.1.64:554/ISAPI/Streaming/channels/10101"
        )

    def test_custom_port(self):
        url = build_streaming_url(self.HOST, self.USER, self.PASS, self.CH, port=8554)
        assert url.endswith(":8554/Streaming/Channels/101")


# ---------------------------------------------------------------------------
# common.config
# ---------------------------------------------------------------------------
class TestConfig:
    def test_env_value_first_non_empty(self, monkeypatch):
        monkeypatch.setenv("A", "")
        monkeypatch.setenv("B", "b-value")
        monkeypatch.setenv("C", "c-value")
        assert env_value("A", "B", "C") == "b-value"

    def test_env_value_default(self, monkeypatch):
        monkeypatch.delenv("NO_EXISTE_XYZ", raising=False)
        assert env_value("NO_EXISTE_XYZ", default="fallback") == "fallback"


# ---------------------------------------------------------------------------
# common.utils
# ---------------------------------------------------------------------------
class TestUtils:
    def test_ffmpeg_low_latency_env_tcp(self, monkeypatch):
        monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
        options = set_ffmpeg_low_latency_env("tcp")
        assert "rtsp_transport;tcp" in options
        assert options == os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]

    def test_ffmpeg_low_latency_env_invalid_falls_back_to_udp(self, monkeypatch):
        monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
        options = set_ffmpeg_low_latency_env("udp")
        assert "rtsp_transport;udp" in options

    def test_ffmpeg_low_latency_has_extended_flags(self, monkeypatch):
        options = set_ffmpeg_low_latency_env("tcp")
        for flag in ("max_delay;0", "fflags;nobuffer", "flags;low_delay", "reorder_queue_size;0"):
            assert flag in options

    def test_save_jpeg_writes_file(self, tmp_path):
        pytest.importorskip("cv2")
        frame = np.full((64, 64, 3), 120, dtype=np.uint8)
        path = save_jpeg(frame, folder=str(tmp_path), prefix="test")
        assert os.path.exists(path)

    def test_sharpness_scores_noisy_higher_than_blank(self):
        pytest.importorskip("cv2")
        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        noisy = np.random.default_rng(42).integers(0, 256, size=(64, 64, 3)).astype(np.uint8)
        assert sharpness(noisy) > sharpness(blank)


# ---------------------------------------------------------------------------
# common.isapi
# ---------------------------------------------------------------------------
class TestIsapi:
    def test_snapshot_url(self):
        assert snapshot_url("192.168.1.64", "101") == (
            "http://192.168.1.64/ISAPI/Streaming/channels/101/picture"
        )

    def test_save_isapi_snapshot_ok(self, monkeypatch, tmp_path):
        import common.isapi as isapi_mod

        class _Resp:
            status_code = 200
            headers = {"Content-Type": "image/jpeg"}

            def iter_content(self, chunk_size):
                yield b"abc"
                yield b"def"

        responses = deque([_Resp()])
        monkeypatch.setattr(isapi_mod.requests, "get", lambda *a, **k: responses.popleft())

        path = isapi_mod.save_isapi_snapshot(
            "192.168.1.64", "admin", "pass", folder=str(tmp_path), channel="101"
        )
        assert path is not None
        assert os.path.exists(path)

    def test_save_isapi_snapshot_rejects_non_image(self, monkeypatch, tmp_path):
        import common.isapi as isapi_mod

        class _Resp:
            status_code = 200
            headers = {"Content-Type": "text/html"}

            def iter_content(self, chunk_size):
                yield b""

        monkeypatch.setattr(isapi_mod.requests, "get", lambda *a, **k: _Resp())
        assert isapi_mod.save_isapi_snapshot(
            "192.168.1.64", "admin", "pass", folder=str(tmp_path)
        ) is None
        assert not list(tmp_path.iterdir())

    def test_save_isapi_snapshot_timeout(self, monkeypatch, tmp_path):
        import common.isapi as isapi_mod

        def _boom(**kwargs):
            raise isapi_mod.requests.exceptions.Timeout()

        monkeypatch.setattr(isapi_mod.requests, "get", _boom)
        assert isapi_mod.save_isapi_snapshot(
            "192.168.1.64", "admin", "pass", folder=str(tmp_path)
        ) is None


# ---------------------------------------------------------------------------
# common.frames
# ---------------------------------------------------------------------------
class _FakeCap:
    def __init__(self, frames):
        self._frames = list(frames)

    def isOpened(self):
        return True

    def read(self):
        if self._frames:
            return True, self._frames.pop(0)
        return True, np.zeros((32, 32, 3), dtype=np.uint8)

    def set(self, prop, value):
        return True

    def get(self, prop):
        return 0

    def release(self):
        self._frames.clear()


class TestFrames:
    @staticmethod
    def make_frame(value):
        return np.full((32, 32, 3), value, dtype=np.uint8)

    def test_latest_grabber_reads_last_frame(self):
        frames = [self.make_frame(v) for v in (10, 20, 30)]
        grabber = FrameGrabberLatest(
            "rtsp://fakep", name="test-sub", cap_factory=lambda *a, **k: _FakeCap(frames)
        )
        try:
            ok, frame = grabber.read()
            assert ok is True
            assert frame is not None
            assert frame.shape == (32, 32, 3)
        finally:
            grabber.release()

    def test_buffer_window_returns_tuples(self):
        grabber = FrameGrabberBuffer(
            "rtsp://fakep", name="test-main", cap_factory=lambda *a, **k: _FakeCap([])
        )
        try:
            time.sleep(0.15)
            results = grabber.get_frames_in_window(time.time(), 300)
            assert isinstance(results, list)
            assert len(results) > 0
            ts, frame = results[0]
            assert isinstance(ts, float)
            assert frame.shape == (32, 32, 3)
            info = grabber.get_buffer_info()
            assert info["size"] > 0
            assert info["max_size"] > 0
        finally:
            grabber.release()

    def test_buffer_closest_frame(self):
        grabber = FrameGrabberBuffer(
            "rtsp://fakep", name="test-main", cap_factory=lambda *a, **k: _FakeCap([])
        )
        try:
            time.sleep(0.15)
            closest = grabber.get_closest_frame(time.time())
            assert closest is not None
            assert closest.shape == (32, 32, 3)
        finally:
            grabber.release()