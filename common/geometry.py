def box_inside_roi(box, roi):
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = roi
    return (x1 >= rx1 and y1 >= ry1 and x2 <= rx2 and y2 <= ry2)


def compute_roi(frame_w, frame_h, roi_w=0.50, roi_h=0.70, roi_cy=0.40):
    cx, cy = frame_w // 2, int(frame_h * roi_cy)
    rw, rh = int(frame_w * roi_w), int(frame_h * roi_h)
    x1 = max(0, cx - rw // 2)
    y1 = max(0, cy - rh // 2)
    x2 = min(frame_w, x1 + rw)
    y2 = min(frame_h, y1 + rh)
    return x1, y1, x2, y2
