"""Persistencia de capturas de la web app (listado y snapshot ISAPI)."""

import glob
import os

from common.isapi import save_isapi_snapshot as _save_isapi_snapshot


def list_latest_images(capture_dir: str, latest_limit: int, limit: int | None = None) -> list[str]:
    """Lista los ``limit`` JPEG más recientes de ``capture_dir`` (por mtime)."""
    limit = latest_limit if limit is None else max(0, int(limit))
    files = glob.glob(os.path.join(capture_dir, "*.jpg"))
    files.sort(key=os.path.getmtime, reverse=True)
    return [os.path.basename(p) for p in files[:limit]]


def save_isapi_snapshot(
    host: str,
    user: str,
    password: str,
    folder: str,
    channel: str,
    timeout: int = 3,
) -> str | None:
    """Descarga un snapshot vía ISAPI con Digest."""
    return _save_isapi_snapshot(
        host, user, password, folder=folder, channel=channel, timeout=timeout
    )
