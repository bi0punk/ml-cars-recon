import logging

logger = logging.getLogger(__name__)


def build_rtsp_url(host, user, password, channel="101", profile="main"):
    return f"rtsp://{user}:{password}@{host}:554/Streaming/Channels/{channel}01"


def build_rtsp_url_isapi(host, user, password, channel="101"):
    return f"rtsp://{user}:{password}@{host}:554/ISAPI/Streaming/channels/{channel}01"


def log_rtsp_connection(host, user):
    logger.info("Conectando a RTSP: user=%s, host=%s", user, host)
