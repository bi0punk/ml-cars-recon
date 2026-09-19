"""Captura de snapshots vía ISAPI (HTTP Digest) para cámaras IP."""

import logging
import os
from datetime import datetime

import requests
from requests.auth import HTTPDigestAuth

logger = logging.getLogger(__name__)


def snapshot_url(host: str, channel: str = "101") -> str:
    return f"http://{host}/ISAPI/Streaming/channels/{channel}/picture"


def save_isapi_snapshot(
    host: str,
    user: str,
    password: str,
    folder: str = "captures",
    channel: str = "101",
    timeout: int = 4,
) -> str | None:
    """Descarga un snapshot vía ISAPI usando autenticación Digest.

    Retorna la ruta del archivo guardado, o ``None`` si falló.
    """
    try:
        os.makedirs(folder, exist_ok=True)
        response = requests.get(
            snapshot_url(host, channel),
            auth=HTTPDigestAuth(user, password),
            timeout=timeout,
            stream=True,
        )
        if response.status_code == 200 and response.headers.get("Content-Type", "").startswith("image"):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]
            filename = os.path.join(folder, f"{timestamp}_isapi_{channel}.jpg")
            with open(filename, "wb") as f:
                for chunk in response.iter_content(8192):
                    f.write(chunk)
            logger.info("Captura ISAPI guardada: %s", filename)
            return filename
        logger.error(
            "Error HTTP %s / Content-Type=%s",
            response.status_code,
            response.headers.get("Content-Type"),
        )
    except requests.exceptions.Timeout:
        logger.error("Timeout: no se pudo obtener snapshot ISAPI")
    except Exception as exc:
        logger.error("Error capturando snapshot ISAPI: %s", exc)
    return None