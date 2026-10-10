"""Detección de vehículos compartida para los scripts de captura.

Encapsula la lógica que estaba duplicada en varios scripts: inferencia YOLO
sobre el ROI, filtro por clases de vehículo, remapeo a coordenadas globales,
chequeo "vehículo completo dentro del ROI", cooldown y captura ISAPI en hilo
aparte.
"""

import logging
import threading
import time
from collections.abc import Callable

import cv2
import numpy as np

from common.geometry import box_inside_roi
from common.isapi import save_isapi_snapshot

logger = logging.getLogger(__name__)

# Clases de vehículos COCO: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = [2, 3, 5, 7]

# (x1, y1, x2, y2) en coordenadas GLOBALES del frame + id de clase
BoxFilter = Callable[[tuple[int, int, int, int], int], bool]


class VehicleDetector:
    """Detección de vehículos con YOLO + disparo de snapshot ISAPI.

    Parámetros de rendimiento:

    - ``infer_every_n`` corre YOLO solo cada N frames y reutiliza el último
      resultado de detección en el resto (reduce CPU/GPU a costa de refresco).
    - ``imgsz``/``device`` se propagan a ``model.predict()`` para controlar
      resolución de entrada y acelerador.
    """

    def __init__(
        self,
        model,
        conf: float = 0.45,
        iou: float = 0.5,
        cooldown: float = 1.0,
        vehicle_classes=None,
        device: str | None = None,
        imgsz: int | None = None,
        infer_every_n: int = 1,
        margin: int = 0,
        box_filter: BoxFilter | None = None,
    ):
        self.model = model
        self.conf = conf
        self.iou = iou
        self.cooldown = max(0.0, cooldown)
        self.vehicle_classes = list(vehicle_classes) if vehicle_classes else VEHICLE_CLASSES
        self.device = device
        self.imgsz = imgsz
        self.infer_every_n = max(1, int(infer_every_n))
        self.margin = max(0, int(margin))
        self.box_filter = box_filter
        self.last_capture_ts = 0.0
        self._frame_count = 0
        self._last_state = (False, False)
        self._last_boxes: list[tuple[int, int, int, int, str, float]] = []

    def detect(self, frame, roi):
        """Corre inferencia en ``roi`` (x0, y0, x1, y1) y anota el frame.

        Retorna ``(frame_anotado, detected, trigger_snapshot)`` donde
        ``trigger_snapshot`` es True cuando un vehículo está COMPLETO dentro
        del ROI. En los frames que se saltan (``infer_every_n > 1``) se
        devuelve el último estado de detección y se vuelven a dibujar las
        últimas cajas, sin re-ejecutar YOLO.
        """
        self._frame_count += 1
        if (self._frame_count - 1) % self.infer_every_n != 0:
            self._draw_boxes(frame)
            return frame, *self._last_state

        x0, y0, x1, y1 = roi
        roi_frame = frame[y0:y1, x0:x1]
        detected = False
        trigger_snapshot = False
        boxes = []

        if roi_frame.size > 0:
            try:
                results = self.model.predict(
                    source=roi_frame,
                    conf=self.conf,
                    iou=self.iou,
                    classes=self.vehicle_classes,
                    verbose=False,
                    **self._predict_kwargs(),
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

                if self.box_filter is not None and not self.box_filter(
                    (xA_g, yA_g, xB_g, yB_g), cls_id
                ):
                    continue

                detected = True
                label = self.model.names.get(cls_id, str(cls_id))
                boxes.append((xA_g, yA_g, xB_g, yB_g, label, conf))

                if box_inside_roi((xA_g, yA_g, xB_g, yB_g), roi, self.margin):
                    trigger_snapshot = True

        self._draw_boxes(frame, boxes)
        self._last_boxes = boxes
        self._last_state = (detected, trigger_snapshot)
        return frame, detected, trigger_snapshot

    def _draw_boxes(
        self,
        frame,
        boxes: list[tuple[int, int, int, int, str, float]] | None = None,
    ) -> None:
        """Dibuja las cajas (dadas o las últimas guardadas) sobre el frame."""
        for x1, y1, x2, y2, label, conf in boxes if boxes is not None else self._last_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, max(y1 - 5, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    def _predict_kwargs(self) -> dict:
        kwargs: dict = {}
        if self.imgsz is not None:
            kwargs["imgsz"] = self.imgsz
        if self.device is not None:
            kwargs["device"] = self.device
        return kwargs

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
