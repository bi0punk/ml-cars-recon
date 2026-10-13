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


class TestLaneDetectorInternals:
    def test_slope_intercept_normal(self):
        det = LaneDetector()
        slope, intercept = det._calculate_slope_intercept(np.array([0, 0, 10, 10]))
        assert slope == pytest.approx(1.0)
        assert intercept == pytest.approx(0.0)

    def test_slope_intercept_vertical_is_infinite(self):
        det = LaneDetector()
        slope, intercept = det._calculate_slope_intercept(np.array([5, 0, 5, 50]))
        assert slope == float("inf")
        assert intercept == 0

    def test_separate_lines_none_returns_empty(self):
        det = LaneDetector()
        assert det._separate_lines(None) == ([], [])

    def test_separate_lines_splits_by_slope(self):
        det = LaneDetector()
        lines = np.array(
            [
                [[10, 100, 50, 0]],  # pendiente negativa -> izquierda
                [[50, 0, 90, 100]],  # pendiente positiva -> derecha
                [[0, 50, 100, 55]],  # casi horizontal -> descartada
                [[0, 0, 100, 1000]],  # pendiente fuera de rango -> descartada
            ]
        )
        left, right = det._separate_lines(lines)
        assert len(left) == 1
        assert len(right) == 1

    def test_average_lines_empty_returns_none(self):
        assert LaneDetector()._average_lines([], 240, 100) is None

    def test_average_lines_averages(self):
        avg = LaneDetector()._average_lines([(-2.0, 200.0), (-2.0, 220.0)], 240, 144)
        assert avg is not None
        assert avg.shape == (4,)

    def test_detect_returns_lanes_on_synthetic_roads(self):
        roi = np.array([[(0, 239), (0, 0), (319, 0), (319, 239)]], dtype=np.int32)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.line(frame, (30, 220), (110, 60), (255, 255, 255), 5)
        cv2.line(frame, (290, 220), (210, 60), (255, 255, 255), 5)
        left, right = LaneDetector(roi_vertices=roi).detect(frame)
        assert left is not None
        assert right is not None

    def test_draw_lanes_left_only(self):
        det = LaneDetector()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        out = det.draw_lanes(frame, np.array([10, 240, 120, 100]), None)
        assert out.shape == frame.shape

    def test_draw_lanes_right_only(self):
        det = LaneDetector()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        out = det.draw_lanes(frame, None, np.array([310, 240, 200, 100]))
        assert out.shape == frame.shape
