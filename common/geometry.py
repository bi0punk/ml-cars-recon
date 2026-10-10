"""ROI y geometría de bounding boxes + polígonos (road filter)."""

import cv2
import numpy as np


def box_inside_roi(box, roi, margin=0):
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = roi
    return x1 >= rx1 + margin and y1 >= ry1 + margin and x2 <= rx2 - margin and y2 <= ry2 - margin


def compute_roi(frame_w, frame_h, roi_w=0.50, roi_h=0.70, roi_cy=0.40):
    cx, cy = frame_w // 2, int(frame_h * roi_cy)
    rw, rh = int(frame_w * roi_w), int(frame_h * roi_h)
    x1 = max(0, cx - rw // 2)
    y1 = max(0, cy - rh // 2)
    x2 = min(frame_w, x1 + rw)
    y2 = min(frame_h, y1 + rh)
    return x1, y1, x2, y2


def parse_polygon(poly_str: str, width: int, height: int) -> np.ndarray | None:
    """Convierte un polígono en coordenadas relativas a píxeles.

    Acepta formato ``"x1,y1;x2,y2;..."`` (fracciones de ancho/alto) y devuelve
    un array ``np.int32`` listo para ``cv2.fillPoly``, o ``None`` si hay menos
    de 3 vértices válidos.
    """
    if not poly_str:
        return None
    points = []
    for part in poly_str.split(";"):
        part = part.strip()
        if not part:
            continue
        x, y = part.split(",")
        points.append([int(float(x) * width), int(float(y) * height)])
    if len(points) < 3:
        return None
    return np.array(points, dtype=np.int32)


def draw_polygon(
    img: np.ndarray, points: np.ndarray, rgba_fill: tuple = (0, 255, 255)
) -> np.ndarray:
    """Dibuja un polígono semitransparente sobre la imagen y devuelve la copia."""
    overlay = img.copy()
    cv2.fillPoly(overlay, [points], rgba_fill)
    cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)
    cv2.polylines(img, [points], True, rgba_fill, 2)
    return img


def point_inside_polygon(point: tuple[int, int], points: np.ndarray) -> bool:
    """``True`` si el punto está dentro (o sobre) del polígono."""
    return cv2.pointPolygonTest(points, point, False) > 0
