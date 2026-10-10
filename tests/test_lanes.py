"""Tests de common.lanes (detector de carriles)."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from common.lanes import LaneDetector  # noqa: E402


class TestLaneDetector:
    def test_defaults(self):
        det = LaneDetector()
        assert det.canny_low == 50
        assert det.canny_high == 150
        assert det.hough_threshold == 50
        assert det.roi_vertices is None

    def test_configurable_params(self):
        det = LaneDetector(
            canny_low=30,
            canny_high=90,
            hough_threshold=25,
            min_line_length=20,
            max_line_gap=50,
        )
        assert det.canny_low == 30
        assert det.canny_high == 90
        assert det.hough_threshold == 25
        assert det.hough_min_line_len == 20
        assert det.hough_max_line_gap == 50

    def test_detect_blank_frame_returns_no_lanes(self):
        det = LaneDetector()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        left, right = det.detect(frame)
        assert left is None and right is None
        assert det.roi_vertices is not None  # se calculó el ROI por defecto

    def test_draw_lanes_without_lines_keeps_shape(self):
        det = LaneDetector()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        out = det.draw_lanes(frame, None, None)
        assert out.shape == frame.shape

    def test_draw_lanes_returns_frame(self):
        det = LaneDetector()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        left = np.array([10, 240, 120, 100])
        right = np.array([310, 240, 200, 100])
        out = det.draw_lanes(frame, left, right)
        assert out.shape == frame.shape
