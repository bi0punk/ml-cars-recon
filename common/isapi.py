"""Captura de snapshots vía ISAPI (HTTP Digest) para cámaras IP."""

import os
from datetime import datetime

import requests
from requests.auth import HTTPDigestAuth


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
            print(f"[ISAPI] Captura guardada: {filename}")
            return filename
        print(f"[ISAPI] Error HTTP {response.status_code} / Content-Type={response.headers.get('Content-Type')}")
    except requests.exceptions.Timeout:
        print("[ISAPI] Timeout: No se pudo obtener snapshot")
    except Exception as exc:
        print(f"[ISAPI] Error: {exc}")
    return None