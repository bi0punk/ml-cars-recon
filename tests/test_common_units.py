"""Tests unitarios sobre los módulos de ``common/``."""

import os
import time
from collections import deque

import numpy as np
import pytest

from common.config import env_value
from common.detector import VehicleDetector
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


class _FramesThenFailCap:
    """Entrega ``frames`` y luego devuelve read()==False para siempre."""

    def __init__(self, frames):
        self._frames = list(frames)

    def isOpened(self):
        return True

    def read(self):
        if self._frames:
            return True, self._frames.pop(0)
        return False, None

    def set(self, prop, value):
        return True

    def get(self, prop):
        return 0

    def release(self):
        self._frames.clear()


class _GlitchCap(_FakeCap):
    """Como ``_FakeCap`` pero falla ``glitch`` lecturas seguidas y luego sigue."""

    def __init__(self, frames, glitch):
        super().__init__(frames)
        self._glitch = glitch

    def read(self):
        if self._glitch > 0:
            self._glitch -= 1
            return False, None
        return super().read()


class _CountingFactory:
    """Factory que cuenta cuántas veces se abre un nuevo capture."""

    def __init__(self, make_cap):
        self.calls = 0
        self._make_cap = make_cap
        self.caps = []

    def __call__(self, *args, **kwargs):
        self.calls += 1
        cap = self._make_cap(self.calls)
        self.caps.append(cap)
        return cap


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

    def test_latest_reconnects_on_stale_stream(self):
        factory = _CountingFactory(lambda n: _FramesThenFailCap([self.make_frame(10)] * 3))
        grabber = FrameGrabberLatest(
            "rtsp://fakep",
            name="test-sub",
            cap_factory=factory,
            stale_frame_threshold=2,
            reconnect_delay=0.01,
            max_reconnect_delay=0.05,
        )
        try:
            time.sleep(0.3)
            assert factory.calls >= 2, "debería haber reabierto el stream al quedarse sin frames"
            assert grabber.reconnect_count >= 2
            ok, frame = grabber.read()
            assert frame is not None
            assert ok is False
        finally:
            grabber.release()

    def test_latest_recovers_after_reconnect(self):
        factory = _CountingFactory(
            lambda n: _FramesThenFailCap([self.make_frame(10)] * 3) if n == 1 else _FakeCap([])
        )
        grabber = FrameGrabberLatest(
            "rtsp://fakep",
            name="test-sub",
            cap_factory=factory,
            stale_frame_threshold=2,
            reconnect_delay=0.01,
            max_reconnect_delay=0.05,
        )
        try:
            time.sleep(0.3)
            assert factory.calls >= 2, "la reconexión no ocurrió"
            ok, frame = grabber.read()
            assert ok is True
            assert frame is not None
        finally:
            grabber.release()

    def test_buffer_reconnects_and_fills_again(self):
        factory = _CountingFactory(
            lambda n: _FramesThenFailCap([self.make_frame(10)] * 3) if n == 1 else _FakeCap([])
        )
        grabber = FrameGrabberBuffer(
            "rtsp://fakep",
            name="test-main",
            cap_factory=factory,
            stale_frame_threshold=2,
            reconnect_delay=0.01,
            max_reconnect_delay=0.05,
        )
        try:
            time.sleep(0.3)
            assert factory.calls >= 2, "la reconexión no ocurrió"
            info = grabber.get_buffer_info()
            assert info["size"] > 0, "el buffer debería volver a llenarse tras reconectar"
        finally:
            grabber.release()

    def test_stale_threshold_below_does_not_reopen(self):
        factory = _CountingFactory(lambda n: _GlitchCap([self.make_frame(10)] * 20, glitch=2))
        grabber = FrameGrabberLatest(
            "rtsp://fakep",
            name="test-sub",
            cap_factory=factory,
            stale_frame_threshold=5,
            reconnect_delay=0.01,
            max_reconnect_delay=0.05,
        )
        try:
            time.sleep(0.3)
            assert factory.calls == 1, "un glitch breve no debería forzar reconexión"
            ok, frame = grabber.read()
            assert ok is True
            assert frame is not None
        finally:
            grabber.release()


# ---------------------------------------------------------------------------
# common.detector
# ---------------------------------------------------------------------------
class _FakeBox:
    def __init__(self, x1, y1, x2, y2, cls, conf):
        self.cls = np.array([cls], dtype=float)
        self.conf = np.array([conf], dtype=float)
        self.xyxy = np.array([[x1, y1, x2, y2]], dtype=float)


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeYolo:
    names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def __init__(self, boxes):
        self._boxes = boxes
        self.calls = 0

    def predict(self, source, conf=0.45, classes=None, verbose=False):
        self.calls += 1
        return [_FakeResult(self._boxes)]


class TestVehicleDetector:
    @staticmethod
    def make_frame():
        return np.full((100, 100, 3), 60, dtype=np.uint8)

    def test_detect_marks_vehicle_and_triggers_inside_roi(self):
        roi = (10, 10, 90, 90)
        box = _FakeBox(20, 20, 80, 80, 2, 0.9)  # completamente dentro del ROI
        det = VehicleDetector(_FakeYolo([box]))
        frame, detected, trigger = det.detect(self.make_frame(), roi)
        assert detected is True
        assert trigger is True
        assert frame is not None

    def test_detect_does_not_trigger_when_box_outside_roi(self):
        roi = (10, 10, 90, 90)
        box = _FakeBox(200, 200, 240, 240, 7, 0.8)  # fuera del ROI
        det = VehicleDetector(_FakeYolo([box]))
        frame, detected, trigger = det.detect(self.make_frame(), roi)
        assert detected is True
        assert trigger is False

    def test_detect_tolerates_empty_boxes(self):
        det = VehicleDetector(_FakeYolo([]))
        frame, detected, trigger = det.detect(self.make_frame(), (10, 10, 90, 90))
        assert detected is False
        assert trigger is False

    def test_trigger_if_ready_respects_cooldown(self, monkeypatch):
        import common.detector as detector_mod

        monkeypatch.setattr(detector_mod, "save_isapi_snapshot", lambda *a, **k: None)
        det = VehicleDetector(_FakeYolo([]), cooldown=60.0)
        assert det.trigger_if_ready("h", "u", "p", "101") is True
        assert det.trigger_if_ready("h", "u", "p", "101") is False

    def test_trigger_if_ready_respects_zero_cooldown(self, monkeypatch):
        import common.detector as detector_mod

        monkeypatch.setattr(detector_mod, "save_isapi_snapshot", lambda *a, **k: None)
        det = VehicleDetector(_FakeYolo([]), cooldown=0.0)
        assert det.trigger_if_ready("h", "u", "p", "101") is True
        assert det.trigger_if_ready("h", "u", "p", "101") is True