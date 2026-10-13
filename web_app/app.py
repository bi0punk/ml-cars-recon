#!/usr/bin/env python3

"""Interfaz web Flask para ml-cars-recon.

Sin efectos colaterales al importar: la cámara y el modelo YOLO se crean de
forma perezosa (``get_camera`` / ``_get_model``), por lo que importar este
módulo no arranca hilos de captura ni carga modelos pesados.

La configuración se resuelve una vez en ``CONFIG`` (``load_config``) a partir
de variables de entorno; en tests se puede sustituir por
``Config(...)``/``dataclasses.replace``.

La lógica está dividida en módulos de responsabilidad única:

- :mod:`web_app.config` — Configuración.
- :mod:`web_app.camera` — DetectorCamera (YOLO + ROI + snapshot).
- :mod:`web_app.security` — Token + CSRF.
- :mod:`web_app.storage` — Listado y guardado de capturas.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory

from common.utils import set_ffmpeg_low_latency_env
from web_app.camera import DetectorCamera, load_model
from web_app.config import Config, load_config
from web_app.security import origin_allowed, token_matches
from web_app.storage import list_latest_images as _list_latest_images
from web_app.storage import save_isapi_snapshot as _storage_save_isapi_snapshot

logger = logging.getLogger(__name__)

# API pública del módulo (incluye re-exports usados por los tests).
__all__ = [
    "Config",
    "CONFIG",
    "get_camera",
    "reset_camera",
    "create_app",
    "app",
    "list_latest_images",
    "save_isapi_snapshot",
    "require_auth",
]

CONFIG = load_config()

_camera: DetectorCamera | None = None
_model: Any | None = None
_model_lock = threading.Lock()


def require_auth():
    if not CONFIG.api_token:
        return
    token = request.headers.get("X-API-Token") or ""
    if not token_matches(token, CONFIG.api_token):
        abort(401, description="Unauthorized")


def _origin_allowed() -> bool:
    """Bloquea peticiones cross-origin (CSRF) a rutas de estado."""
    return origin_allowed(request.headers.get("Origin"), request.host_url)


def list_latest_images(limit: int | None = None) -> list[str]:
    limit = CONFIG.latest_limit if limit is None else max(0, int(limit))
    return _list_latest_images(CONFIG.capture_dir, CONFIG.latest_limit, limit)


def save_isapi_snapshot(folder: str | None = None, timeout: int = 3) -> str | None:
    """Descarga snapshot vía ISAPI con Digest (usa la config del módulo)."""
    return _storage_save_isapi_snapshot(
        CONFIG.isapi_host,
        CONFIG.isapi_user,
        CONFIG.isapi_password,
        folder=folder or CONFIG.capture_dir,
        channel=CONFIG.snapshot_channel,
        timeout=timeout,
    )


def _get_model():
    """Carga el modelo YOLO una sola vez (perezoso)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    _model = load_model(CONFIG.model_path)
                    logger.info("Modelo YOLO cargado: %s", CONFIG.model_path)
                except Exception as exc:
                    _model = None
                    logger.error("Error al cargar modelo YOLO: %s", exc)
    return _model


def get_camera() -> DetectorCamera:
    """Crea (y cachea) la cámara de forma perezosa."""
    global _camera
    if _camera is None:
        _camera = DetectorCamera(CONFIG.rtsp_url, _get_model(), CONFIG)
    return _camera


def reset_camera() -> None:
    """Detiene y elimina la cámara actual (útil en tests/shutdown)."""
    global _camera
    if _camera is not None:
        _camera.stop()
        _camera = None


def create_app() -> Flask:
    set_ffmpeg_low_latency_env("tcp")

    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        require_auth()
        camera = get_camera()

        def gen():
            boundary = b"--frame"
            while True:
                frame = camera.read_jpeg()
                if frame is None:
                    time.sleep(0.05)
                    continue
                yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                time.sleep(0.03)

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/latest_images")
    def api_latest_images():
        require_auth()
        imgs = list_latest_images()
        now = int(time.time())
        data = [{"name": f, "url": f"/captures/{f}?t={now}"} for f in imgs]
        return jsonify(data)

    @app.route("/api/status")
    def api_status():
        require_auth()
        if _camera is None:
            return jsonify({"active": False})
        return jsonify({"active": True, **get_camera().status()})

    @app.route("/captures/<path:filename>")
    def captures(filename):
        safe_dir = os.path.abspath(CONFIG.capture_dir)
        requested = os.path.abspath(os.path.join(CONFIG.capture_dir, filename))
        if not requested.startswith(safe_dir):
            return abort(403)
        if not os.path.exists(requested):
            return abort(404)
        return send_from_directory(CONFIG.capture_dir, filename)

    @app.route("/_shutdown", methods=["POST"])
    def shutdown():
        require_auth()
        if not _origin_allowed():
            abort(403, description="Forbidden cross-origin")
        reset_camera()
        return "ok"

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=False,
        threaded=True,
    )
