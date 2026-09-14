"""Detección de carriles mediante Canny + Hough Transform."""

import cv2
import numpy as np


class LaneDetector:
    """Detector de carriles usando Canny + Hough Transform."""

    def __init__(self, roi_vertices: np.ndarray | None = None):
        self.roi_vertices = roi_vertices
        self.canny_low = 50
        self.canny_high = 150
        self.hough_rho = 2
        self.hough_theta = np.pi / 180
        self.hough_threshold = 50
        self.hough_min_line_len = 40
        self.hough_max_line_gap = 100

    def _roi_mask(self, img: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        """Aplica máscara de ROI."""
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, vertices, 255)
        return cv2.bitwise_and(img, mask)

    def _calculate_slope_intercept(self, line: np.ndarray) -> tuple[float, float]:
        """Calcula pendiente e intercepto."""
        x1, y1, x2, y2 = line
        if x2 - x1 == 0:
            return float("inf"), 0
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return slope, intercept

    def _separate_lines(self, lines: np.ndarray) -> tuple[list, list]:
        """Separa líneas izquierda/derecha."""
        left_lines = []
        right_lines = []

        if lines is None:
            return left_lines, right_lines

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            if abs(y2 - y1) < 10:
                continue
            slope, intercept = self._calculate_slope_intercept(line.reshape(4))
            if abs(slope) < 0.3 or abs(slope) > 3:
                continue
            if slope < 0:
                left_lines.append((slope, intercept))
            else:
                right_lines.append((slope, intercept))

        return left_lines, right_lines

    def _average_lines(self, lines: list, y1: int, y2: int) -> np.ndarray | None:
        """Promedia líneas."""
        if not lines:
            return None
        slopes = np.array([line[0] for line in lines])
        intercepts = np.array([line[1] for line in lines])
        avg_slope = np.mean(slopes)
        avg_intercept = np.mean(intercepts)
        x1 = int((y1 - avg_intercept) / avg_slope)
        x2 = int((y2 - avg_intercept) / avg_slope)
        return np.array([x1, y1, x2, y2])

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Detecta carriles."""
        height, width = frame.shape[:2]

        if self.roi_vertices is None:
            self.roi_vertices = np.array(
                [
                    [
                        (int(width * 0.1), height),
                        (int(width * 0.45), int(height * 0.6)),
                        (int(width * 0.55), int(height * 0.6)),
                        (int(width * 0.9), height),
                    ]
                ],
                dtype=np.int32,
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)
        roi_edges = self._roi_mask(edges, self.roi_vertices)

        lines = cv2.HoughLinesP(
            roi_edges,
            self.hough_rho,
            self.hough_theta,
            self.hough_threshold,
            minLineLength=self.hough_min_line_len,
            maxLineGap=self.hough_max_line_gap,
        )

        if lines is None:
            return None, None

        left_lines, right_lines = self._separate_lines(lines)
        y1, y2 = height, int(height * 0.6)
        left_lane = self._average_lines(left_lines, y1, y2)
        right_lane = self._average_lines(right_lines, y1, y2)

        return left_lane, right_lane

    def draw_lanes(
        self,
        frame: np.ndarray,
        left_lane: np.ndarray | None,
        right_lane: np.ndarray | None,
    ) -> np.ndarray:
        """Dibuja carriles."""
        overlay = frame.copy()

        if left_lane is not None:
            x1, y1, x2, y2 = left_lane
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 255), 8)

        if right_lane is not None:
            x1, y1, x2, y2 = right_lane
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 255), 8)

        if left_lane is not None and right_lane is not None:
            x1_l, y1_l, x2_l, y2_l = left_lane
            x1_r, y1_r, x2_r, y2_r = right_lane
            pts = np.array([[x1_l, y1_l], [x2_l, y2_l], [x2_r, y2_r], [x1_r, y1_r]], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        if left_lane is not None:
            x1, y1, x2, y2 = left_lane
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

        if right_lane is not None:
            x1, y1, x2, y2 = right_lane
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

        return frame