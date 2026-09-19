#!/usr/bin/env python3

"""
Detección YOLOv8 + Captura desde MAIN con PRE-ROLL + Selector del frame más nítido
+ Solo dispara cuando el vehículo está COMPLETO y ESTABLE dentro del ROI
+ Polígono de carretera opcional (para ignorar autos fuera de calzada)

USO EJEMPLO:
python3 cam_lpr_preroll_sharp.py \
  --host 192.168.1.64 --user admin --password 'TuClave' \
  --rtsp_channel 102 --snapshot_channel 101 \
  --model yolov8n.pt \
  --pre_roll_ms 300 \
  --sharp_window_ms 200 \
  --stable_frames 3 --rearm_frames 8 --inside_margin_px 8 \
  --road_poly "0.15,0.55;0.85,0.55;0.95,0.98;0.05,0.98" --filter_to_road
"""

import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO

from common.frames import FrameGrabberBuffer, FrameGrabberLatest
from common.geometry import box_inside_roi
from common.isapi import save_isapi_snapshot
from common.rtsp import build_isapi_url
from common.utils import save_jpeg, set_ffmpeg_low_latency_env, sharpness


def get_best_frame(buffer, target_ts, window_ms, min_f, max_f):
    """Elige el frame más nítido dentro de la ventana temporal alrededor de target_ts."""
    if not buffer:
        return None
    w = window_ms / 1000.0
    cands = [(abs(ts - target_ts), ts, f) for ts, f in buffer if abs(ts - target_ts) <= w]
    if not cands:
        return None
    cands = sorted(cands)[:max_f]
    scored = [(sharpness(f), f) for _, _, f in cands]
    scored = sorted(scored, key=lambda x: x[0], reverse=True)
    if len(scored) >= min_f:
        return scored[0][1]
    return None


def parse_poly(poly_str, W, H):
    pts = []
    for p in poly_str.split(";"):
        if p.strip():
            x, y = p.split(",")
            pts.append([int(float(x) * W), int(float(y) * H)])
    return np.array(pts, np.int32) if len(pts) >= 3 else None


def draw_poly(img, pts):
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], (0, 255, 255))
    cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)
    cv2.polylines(img, [pts], True, (0, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.1.64")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", required=True)
    ap.add_argument("--rtsp_channel", default="102")
    ap.add_argument("--snapshot_channel", default="101")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--cooldown", type=float, default=0.5, help="Cooldown (s) entre capturas")
    ap.add_argument("--pre_roll_ms", type=int, default=300)
    ap.add_argument("--sharp_window_ms", type=int, default=200)  # NEW
    ap.add_argument("--sharp_min_frames", type=int, default=3)  # NEW
    ap.add_argument("--sharp_max_frames", type=int, default=9)  # NEW
    ap.add_argument("--stable_frames", type=int, default=3)
    ap.add_argument("--rearm_frames", type=int, default=8)
    ap.add_argument("--inside_margin_px", type=int, default=8)
    ap.add_argument("--save_dir", default="captures_lpr")
    ap.add_argument("--rtsp_transport", default="udp")
    ap.add_argument("--fallback_isapi", action="store_true")
    ap.add_argument("--road_poly", default="")
    ap.add_argument("--filter_to_road", action="store_true")
    args = ap.parse_args()

    set_ffmpeg_low_latency_env(args.rtsp_transport)
    model = YOLO(args.model)

    sub = build_isapi_url(args.host, args.user, args.password, args.rtsp_channel)
    main = build_isapi_url(args.host, args.user, args.password, args.snapshot_channel)

    gsub = FrameGrabberLatest(sub, args.width, args.height)
    gmain = FrameGrabberBuffer(main, args.width, args.height)

    inside = 0
    outside = 0
    triggered = False
    last = 0

    road = None
    print("[INFO] Running ... press Q to exit")

    while True:
        ok, frame = gsub.read()
        if not ok:
            cv2.waitKey(1)
            continue

        H, W = frame.shape[:2]
        if road is None and args.road_poly:
            road = parse_poly(args.road_poly, W, H)

        roi = (int(W * 0.25), int(H * 0.1), int(W * 0.75), int(H * 0.8))
        x0, y0, x1, y1 = roi
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 0), 2)
        if road is not None:
            draw_poly(frame, road)

        r = model.predict(frame[y0:y1, x0:x1], conf=args.conf, verbose=False)
        best = None
        bestA = 0

        for rr in r:
            for b in rr.boxes:
                lab = model.names[int(b.cls[0])].lower()
                if not any(k in lab for k in ["car", "vehicle", "truck", "bus", "motor"]):
                    continue
                xA, yA, xB, yB = b.xyxy[0].int().tolist()
                xA += x0
                yA += y0
                xB += x0
                yB += y0
                if args.filter_to_road and road is not None:
                    cx, cy = (xA + xB) // 2, (yA + yB) // 2
                    if cv2.pointPolygonTest(road, (cx, cy), False) <= 0:
                        continue
                A = (xB - xA) * (yB - yA)
                if bestA < A:
                    bestA = A
                    best = (xA, yA, xB, yB)
                cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

        inside_now = False
        if best is not None:
            inside_now = box_inside_roi(best, roi, args.inside_margin_px)

        if inside_now:
            inside += 1
            outside = 0
        else:
            inside = 0
            outside += 1

        do = inside >= args.stable_frames and not triggered

        if outside >= args.rearm_frames:
            triggered = False

        now = time.time()
        if do and (now - last) > args.cooldown:  # cooldown configurable
            triggered = True
            last = now
            ts = now - (args.pre_roll_ms / 1000.0)
            buf = gmain.get_frames_in_window(ts, args.sharp_window_ms)
            bestf = get_best_frame(buf, ts, args.sharp_window_ms, args.sharp_min_frames, args.sharp_max_frames)
            if bestf is not None:
                save_jpeg(bestf, args.save_dir, "best")
            else:
                save_jpeg(frame, args.save_dir, "fallback")
                if args.fallback_isapi:
                    save_isapi_snapshot(args.host, args.user, args.password)

        cv2.imshow("LIVE", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    gsub.release()
    gmain.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()