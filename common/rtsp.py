"""RTSP URL builders.

Preserva las variantes de URL realmente usadas en el proyecto, para que las
diferencias de firmware de cada cámara sigan funcionando sin cambios:

- ``/Streaming/Channels/{canal}``        (flavores "streaming")
- ``/ISAPI/Streaming/channels/{canal}``  (flavores "isapi")

Algunos firmwares exigen el sufijo de sub-stream ``01`` (p.ej. ``10101``);
para esos casos usa las variantes ``*_channel_url``.
"""


def build_streaming_url(host: str, user: str, password: str, channel: str = "101", port: int = 554) -> str:
    """URL RTSP tipo ``/Streaming/Channels/{canal}`` (sin sufijo de sub-stream)."""
    return f"rtsp://{user}:{password}@{host}:{port}/Streaming/Channels/{channel}"


def build_streaming_channel_url(host: str, user: str, password: str, channel: str = "101", port: int = 554) -> str:
    """URL RTSP tipo ``/Streaming/Channels/{canal}01`` (con sufijo de sub-stream)."""
    return build_streaming_url(host, user, password, f"{channel}01", port)


def build_isapi_url(host: str, user: str, password: str, channel: str = "101", port: int = 554) -> str:
    """URL RTSP tipo ``/ISAPI/Streaming/channels/{canal}`` (sin sufijo de sub-stream)."""
    return f"rtsp://{user}:{password}@{host}:{port}/ISAPI/Streaming/channels/{channel}"


def build_isapi_channel_url(host: str, user: str, password: str, channel: str = "101", port: int = 554) -> str:
    """URL RTSP tipo ``/ISAPI/Streaming/channels/{canal}01`` (con sufijo de sub-stream)."""
    return build_isapi_url(host, user, password, f"{channel}01", port)