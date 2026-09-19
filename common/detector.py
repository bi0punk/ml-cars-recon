"""Detección de vehículos compartida para los scripts de captura.

Encapsula la lógica que estaba duplicada en ``app_autos_captura.py`` y
``detecta_autos_y_captura_roi_central.py``: inferencia YOLO sobre el ROI,
filtro por clases de vehículo, remapeo a coordenadas globales, chequeo
"vehículo completo dentro del ROI", cooldown y captura ISAPI en hilo aparte.
"""

import logging
import threading
import time

import cv2
import numpy as np

from common.geometry import box_inside_roi
from common.isapi import save_isapi_snapshot

logger = logging.getLogger(__name__)

# Clases de vehículos COCO: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = [2, 3, 5, 7]


class VehicleDetector:
    """Detección de vehículos con YOLO + disparo de snapshot ISAPI."""

    def __init__(self, model, conf: float = 0.45, cooldown: float = 1.0, vehicle_classes=None):
        self.model = model
        self.conf = conf
        self.cooldown = max(0.0, cooldown)
        self.vehicle_classes = list(vehicle_classes) if vehicle_classes else VEHICLE_CLASSES
        self.last_capture_ts = 0.0

    def detect(self, frame, roi):
        """Corre inferencia en ``roi`` (x0, y0, x1, y1) y anota el frame.

        Retorna ``(frame_anotado, detected, trigger_snapshot)`` donde
        ``trigger_snapshot`` es True cuando un vehículo está COMPLETO dentro
        del ROI.
        """
        x0, y0, x1, y1 = roi
        roi_frame = frame[y0:y1, x0:x1]
        detected = False
        trigger_snapshot = False

        if roi_frame.size > 0:
            try:
                results = self.model.predict(
                    source=roi_frame,
                    conf=self.conf,
                    classes=self.vehicle_classes,
                    verbose=False,
                )
            except Exception as exc:
                logger.error("Predicción YOLO fallida: %s", exc)
                results = []
        else:
            results = []

        for r in results:
            for box in getattr(r, "boxes", []):
                try:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                except Exception:
                    continue

                xA, yA, xB, yB = np.asarray(box.xyxy[0]).astype(int).tolist()
                xA_g, yA_g = xA + x0, yA + y0
                xB_g, yB_g = xB + x0, yB + y0

                detected = True
                cv2.rectangle(frame, (xA_g, yA_g), (xB_g, yB_g), (0, 255, 0), 2)
                label = self.model.names.get(cls_id, str(cls_id))
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (xA_g, max(yA_g - 5, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                if box_inside_roi((xA_g, yA_g, xB_g, yB_g), roi):
                    trigger_snapshot = True

        return frame, detected, trigger_snapshot

    def trigger_if_ready(
        self,
        host: str,
        user: str,
        password: str,
        snapshot_channel: str,
        folder: str = "isapi_snaps",
        timeout: int = 4,
    ) -> bool:
        """Lanza snapshot ISAPI en hilo si el cooldown lo permite.

        Retorna ``True`` si se lanzó la captura, ``False`` si aún está en
        cooldown.
        """
        now = time.time()
        if (now - self.last_capture_ts) < self.cooldown:
            return False
        self.last_capture_ts = now
        threading.Thread(
            target=save_isapi_snapshot,
            args=(host, user, password),
            kwargs={"folder": folder, "channel": snapshot_channel, "timeout": timeout},
            daemon=True,
        ).start()
        return True