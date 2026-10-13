"""Helpers de seguridad HTTP para la web app (token + CSRF)."""

import hmac


def token_matches(provided: str, expected: str) -> bool:
    """Compara tokens en tiempo constante (evita timing attacks)."""
    return hmac.compare_digest(provided, expected)


def origin_allowed(origin: str | None, host_url: str) -> bool:
    """Bloquea peticiones cross-origin (CSRF) a rutas de estado.

    Sin cabecera ``Origin`` se permite (no es una petición de navegador cruzado).
    """
    if not origin:
        return True
    return origin.rstrip("/") == host_url.rstrip("/")
