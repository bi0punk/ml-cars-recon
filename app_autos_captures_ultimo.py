#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RTSP + YOLOv8 + Captura con PRE-ROLL desde MAIN stream (buffer circular)
------------------------------------------------------------------------
- Detecta en SUB-stream (baja latencia).
- Guarda imágenes "un poco ANTES" del trigger desde un buffer del MAIN stream.
- Opcional: fallback a snapshot ISAPI si el buffer está vacío.
- Corrige argparse y URL RTSP.
- Amplía flags FFmpeg para baja latencia.

Uso:
  python3 cam_lpr_preroll.py --host 192.168.1.64 --user admin --password 'TuClave' \
    --rtsp_channel 102 --snapshot_channel 101 --model yolov8n.pt --pre_roll_ms 300
"""

import os
import cv2
import time
import argparse
import numpy as np
import requests
import threading
from collections import deque
from requests.auth import HTTPDigestAuth
from datetime import datetime
from ultralytics import YOLO

# =============================================================================
# FFmpeg: BAJA LATENCIA
# =============================================================================
# [CAMBIO] Se añadieron flags extra (fflags=nobuffer, flags=low_delay, reorder_queue_size=0)
# y opción para elegir UDP/TCP desde CLI (--rtsp_transport).
def set_ffmpeg_low_latency_env(transport: str = "udp"):
    transport = transport.lower().strip()
    if transport not in ("udp", "tcp"):
        transport = "udp"
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"rtsp_transport;{transport}|"
        "max_delay;0|"
        "stimeout;5000000|"      # 5s
        "buffer_size;0|"
        "fflags;nobuffer|"
        "flags;low_delay|"
        "reorder_queue_size;0"
    )

# =============================================================================
# UTILIDADES
# =============================================================================

def ensure_dir(p):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def save_jpeg(frame, folder="captures", prefix="frame"):
    ensure_dir(folder)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    path = os.path.join(folder, f"{ts}_{prefix}.jpg")
    # Calidad alta para OCR
    cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"[SAVE] {path}")
    return path

def save_isapi_snapshot(host, user, password, folder="captures_isapi", channel="101", timeout=4):
    """
    Fallback ISAPI (snapshot "del presente").
    [NOTA] Suele llegar "tarde" vs. substream → por eso ahora se usa buffer del MAIN como principal.
    """
    try:
        ensure_dir(folder)
        url = f"http://{host}/ISAPI/Streaming/channels/{channel}/picture"
        r = requests.get(url, auth=HTTPDigestAuth(user, password), timeout=timeout, stream=True)
        if r.status_code == 200 and r.headers.get("Content-Type", "").startswith("image"):
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
            filename = os.path.join(folder, f"{ts}_isapi_{channel}.jpg")
            with open(filename, "wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            print(f"[ISAPI] Captura guardada: {filename}")
            return filename
        else:
            print(f"[ISAPI] Error HTTP {r.status_code} / Content-Type={r.headers.get('Content-Type')}")
    except Exception as e:
        print(f"[ISAPI] Error al obtener snapshot: {e}")

def box_inside_roi(box, roi):
    # box=(x1,y1,x2,y2); roi=(x1,y1,x2,y2)
    x1,y1,x2,y2 = box
    rx1,ry1,rx2,ry2 = roi
    return x1 >= rx1 and y1 >= ry1 and x2 <= rx2 and y2 <= ry2

# =============================================================================
# GRABBERS
# =============================================================================

class FrameGrabberLatest:
    """
    [CAMBIO] Substream (detección): mantiene SOLO el último frame → baja latencia percibida.
    """
    def __init__(self, rtsp_url, width=None, height=None, name="sub"):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.width = width
        self.height = height
        self.ok = False
        self.frame = None
        self.stopped = False
        self.name = name
        self._open()
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _open(self):
        if self.cap is not None:
            try: self.cap.release()
            except: pass
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        try: self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except: pass
        self.ok = self.cap.isOpened()
        if not self.ok:
            print(f"[{self.name}] No pudo abrir RTSP")

    def _loop(self):
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.2)
                continue
            ok, f = self.cap.read()
            if not ok:
                self.ok = False
                time.sleep(0.01)
                continue
            if f is not None:
                if self.width and self.height:
                    f = cv2.resize(f, (self.width, self.height), interpolation=cv2.INTER_AREA)
                self.ok, self.frame = True, f

    def read(self):
        return self.ok, self.frame

    def release(self):
        self.stopped = True
        try: self.th.join(timeout=1)
        except: pass
        if self.cap:
            self.cap.release()

class FrameGrabberBuffer:
    """
    [CAMBIO] Main stream (calidad): mantiene un BUFFER CIRCULAR (ts, frame).
    Permite "viajar al pasado" unos ms para capturar el frame antes del trigger.
    """
    def __init__(self, rtsp_url, max_seconds=1.5, fps_hint=25, width=None, height=None, name="main"):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.width = width
        self.height = height
        self.name = name
        self.ok = False
        self.buffer = deque(maxlen=int(max_seconds * max(fps_hint, 1)) + 5)
        self.stopped = False
        self._open()
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _open(self):
        if self.cap is not None:
            try: self.cap.release()
            except: pass
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        try: self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except: pass
        self.ok = self.cap.isOpened()
        if not self.ok:
            print(f"[{self.name}] No pudo abrir RTSP")

    def _loop(self):
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.2)
                continue
            ok, f = self.cap.read()
            if not ok:
                self.ok = False
                time.sleep(0.005)
                continue
            ts = time.time()
            if f is not None:
                if self.width and self.height:
                    f = cv2.resize(f, (self.width, self.height), interpolation=cv2.INTER_AREA)
                self.ok = True
                self.buffer.append((ts, f))

    def get_closest(self, target_ts):
        """ Devuelve el frame cuyo timestamp está más cerca de target_ts. """
        if not self.buffer:
            return None
        # Búsqueda lineal (buffer pequeño). Si crece, se puede optimizar con bisect.
        best = min(self.buffer, key=lambda t: abs(t[0] - target_ts))
        return best[1]

    def release(self):
        self.stopped = True
        try: self.th.join(timeout=1)
        except: pass
        if self.cap:
            self.cap.release()

# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="RTSP baja latencia + YOLOv8 + PRE-ROLL desde MAIN")
    ap.add_argument("--host", default="192.168.1.64", help="IP de la cámara")
    ap.add_argument("--user", default="admin", help="Usuario de la cámara")
    ap.add_argument("--password", required=True, help="Contraseña de la cámara")

    # [CAMBIO] argparse con dest=... para nombres Python-friendly (sin guiones)
    ap.add_argument("--rtsp_channel", dest="rtsp_channel", default="102",
                    help="Canal RTSP para detección (SUB). Ej: 102")
    ap.add_argument("--snapshot_channel", dest="snapshot_channel", default="101",
                    help="Canal MAIN (alta calidad). Ej: 101")

    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.45, help="Confianza mínima YOLO")
    ap.add_argument("--cooldown", type=float, default=0.8, help="Segundos entre capturas")
    ap.add_argument("--pre_roll_ms", type=int, default=300, help="Cuánto ANTES del trigger tomar el frame del MAIN")
    ap.add_argument("--save_dir", default="captures_lpr", help="Carpeta de salida")
    ap.add_argument("--rtsp_transport", default="udp", choices=["udp","tcp"],
                    help="Transporte RTSP (udp suele dar menor latencia en LAN)")
    ap.add_argument("--fallback_isapi", action="store_true",
                    help="Si buffer MAIN vacío, intenta snapshot ISAPI como plan B")
    args = ap.parse_args()

    # [CAMBIO] Configurar flags FFmpeg según transporte
    set_ffmpeg_low_latency_env(args.rtsp_transport)

    # Cargar modelo YOLO
    print(f"[INFO] Cargando modelo: {args.model}")
    model = YOLO(args.model)

    # [CAMBIO] Construcción de URLs RTSP consistente (sin guiones en atributos)
    # Nota: Algunas cámaras aceptan /Streaming/Channels/<CH> (sin ISAPI). Ajusta si tu firmware lo requiere.
    sub_rtsp  = f"rtsp://{args.user}:{args.password}@{args.host}:554/ISAPI/Streaming/channels/{args.rtsp_channel}"
    main_rtsp = f"rtsp://{args.user}:{args.password}@{args.host}:554/ISAPI/Streaming/channels/{args.snapshot_channel}"

    print(f"[INFO] Substream (detección): {sub_rtsp}")
    print(f"[INFO] Mainstream (buffer):   {main_rtsp}")

    # [CAMBIO] Dos grabbers: sub = último frame (detección), main = buffer circular (captura)
    grab_sub  = FrameGrabberLatest(sub_rtsp,  width=args.width, height=args.height, name="sub")
    # Tip: si tu MAIN es 2560x1440, puedes dejar width/height=None para no reescalar
    grab_main = FrameGrabberBuffer(main_rtsp, max_seconds=1.5, fps_hint=25,
                                   width=args.width, height=args.height, name="main")

    print("[INFO] Transmisión iniciada. Presiona 'q' para salir.")
    last_capture_ts = 0.0

    try:
        while True:
            ok, frame = grab_sub.read()

            if not ok or frame is None:
                # Ventana de aviso mientras reconecta/lee
                blank = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                cv2.putText(blank, "Reintentando conexión RTSP...", (60, args.height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.imshow("Live (YOLOv8 LPR pre-roll)", blank)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time.sleep(0.03)
                continue

            H, W = frame.shape[:2]

            # ROI central (ajusta a tu escena)
            roi_w = int(W * 0.50)
            roi_h = int(H * 0.70)
            cx, cy = W // 2, int(H * 0.40)
            roi_rect = (cx - roi_w // 2, cy - roi_h // 2, cx + roi_w // 2, cy + roi_h // 2)
            (x0, y0, x1, y1) = roi_rect
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 0), 2)

            # Detección YOLO SOLO en el ROI (substream)
            roi = frame[y0:y1, x0:x1]
            results = model.predict(source=roi, conf=args.conf, verbose=False)
            detected = False
            trigger_snapshot = False

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names.get(cls_id, str(cls_id))

                    # Filtra vehículos (ajusta etiquetas según tu modelo)
                    if any(k in label.lower() for k in ["car", "vehicle", "truck", "bus", "motorbike"]):
                        detected = True
                        xA, yA, xB, yB = box.xyxy[0].int().tolist()
                        # Reubica a coords globales (frame completo)
                        xA += x0; yA += y0; xB += x0; yB += y0
                        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (xA, max(yA - 5, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                        if box_inside_roi((xA, yA, xB, yB), roi_rect):
                            trigger_snapshot = True

            # [CAMBIO] Disparo con PRE-ROLL desde el MAIN stream (buffer), no bloquea el loop
            now = time.time()
            if trigger_snapshot and (now - last_capture_ts) > args.cooldown:
                last_capture_ts = now
                target_ts = now - (args.pre_roll_ms / 1000.0)

                def _async_save():
                    cand = grab_main.get_closest(target_ts)
                    if cand is not None:
                        save_jpeg(cand, folder=args.save_dir, prefix="main_preroll")
                    else:
                        print("[WARN] Buffer MAIN vacío.")
                        if args.fallback_isapi:
                            print("[INFO] Intentando fallback ISAPI...")
                            save_isapi_snapshot(args.host, args.user, args.password,
                                                folder="isapi_snaps", channel=args.snapshot_channel, timeout=4)
                        else:
                            # Último recurso: guardar el sub actual
                            save_jpeg(frame, folder=args.save_dir, prefix="sub_now")
                threading.Thread(target=_async_save, daemon=True).start()

            # Banner informativo
            msg = "VEHICULO DETECTADO" if detected else "SIN DETECCION"
            color = (0, 200, 0) if detected else (0, 0, 255)
            cv2.rectangle(frame, (0, 0), (W, 35), (0, 0, 0), -1)
            cv2.putText(frame, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Mostrar ventana única
            cv2.imshow("Live (YOLOv8 LPR pre-roll)", frame)

            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        grab_sub.release()
        grab_main.release()
        cv2.destroyAllWindows()
        print("[INFO] Transmisión finalizada correctamente.")

if __name__ == "__main__":
    main()
