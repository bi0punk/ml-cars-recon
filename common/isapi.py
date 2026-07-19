import os
import logging
import requests
from requests.auth import HTTPDigestAuth

logger = logging.getLogger(__name__)


def save_isapi_snapshot(host, user, password, folder="captures", channel="101", timeout=3):
    try:
        os.makedirs(folder, exist_ok=True)
        url = f"http://{host}/ISAPI/Streaming/channels/{channel}/picture"
        response = requests.get(url, auth=HTTPDigestAuth(user, password), timeout=timeout, stream=True)
        if response.status_code == 200:
            timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(folder, f"snapshot_{timestamp}.jpg")
            with open(path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            logger.info("Snapshot guardada: %s", path)
            return path
        else:
            logger.warning("HTTP %d al capturar snapshot", response.status_code)
            return None
    except Exception as e:
        logger.error("Error capturando snapshot: %s", e)
        return None
