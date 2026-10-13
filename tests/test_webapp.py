"""Tests de la web app Flask (rutas, auth y cámara lazy)."""

import pytest

pytest.importorskip("flask")

import web_app.app as web_app


@pytest.fixture(autouse=True)
def _no_camera():
    web_app._camera = None
    yield
    web_app.reset_camera()


class TestWebApp:
    def test_index_ok(self):
        client = web_app.app.test_client()
        assert client.get("/").status_code == 200

    def test_api_status_inactive_without_camera(self):
        client = web_app.app.test_client()
        assert client.get("/api/status").get_json() == {"active": False}

    def test_latest_images_empty(self, monkeypatch, tmp_path):
        monkeypatch.setattr(web_app, "CONFIG", web_app.Config(capture_dir=str(tmp_path)))
        assert web_app.list_latest_images() == []

    def test_latest_images_lists_sorted_files(self, monkeypatch, tmp_path):
        monkeypatch.setattr(web_app, "CONFIG", web_app.Config(capture_dir=str(tmp_path)))
        (tmp_path / "a.jpg").write_bytes(b"a")
        (tmp_path / "b.jpg").write_bytes(b"b")
        images = web_app.list_latest_images(limit=1)
        assert len(images) == 1
        assert images[0] in {"a.jpg", "b.jpg"}

    def test_captures_missing_returns_404(self, monkeypatch, tmp_path):
        monkeypatch.setattr(web_app, "CONFIG", web_app.Config(capture_dir=str(tmp_path)))
        client = web_app.app.test_client()
        assert client.get("/captures/nope.jpg").status_code == 404

    def test_captures_blocks_path_traversal(self, monkeypatch, tmp_path):
        monkeypatch.setattr(web_app, "CONFIG", web_app.Config(capture_dir=str(tmp_path)))
        client = web_app.app.test_client()
        assert client.get("/captures/../../etc/passwd").status_code in (403, 404)

    def test_requires_token_on_video_feed(self, monkeypatch):
        monkeypatch.setattr(web_app, "CONFIG", web_app.Config(api_token="sekret"))
        client = web_app.app.test_client()
        assert client.get("/video_feed").status_code == 401

    def test_requires_token_on_status(self, monkeypatch):
        monkeypatch.setattr(web_app, "CONFIG", web_app.Config(api_token="sekret"))
        client = web_app.app.test_client()
        assert client.get("/api/status").status_code == 401

    def test_token_grants_access(self, monkeypatch):
        monkeypatch.setattr(
            web_app,
            "CONFIG",
            web_app.Config(api_token="sekret", rtsp_url="rtsp://invalid-host"),
        )
        client = web_app.app.test_client()
        resp = client.get("/api/status", headers={"X-API-Token": "sekret"})
        assert resp.status_code == 200
        assert resp.get_json() == {"active": False}

    # --- Seguridad ---
    def test_shutdown_blocked_cross_origin(self, monkeypatch):
        monkeypatch.setattr(web_app, "CONFIG", web_app.Config(api_token="sekret"))
        client = web_app.app.test_client()
        resp = client.post(
            "/_shutdown",
            headers={"X-API-Token": "sekret", "Origin": "https://attacker.example"},
        )
        assert resp.status_code == 403
        assert web_app._camera is None

    def test_shutdown_allowed_same_origin(self, monkeypatch):
        monkeypatch.setattr(web_app, "CONFIG", web_app.Config(api_token="sekret"))
        client = web_app.app.test_client()
        resp = client.post(
            "/_shutdown",
            headers={"X-API-Token": "sekret", "Origin": "http://localhost"},
        )
        assert resp.status_code == 200

    def test_wrong_token_rejected(self, monkeypatch):
        monkeypatch.setattr(web_app, "CONFIG", web_app.Config(api_token="sekret"))
        client = web_app.app.test_client()
        assert client.get("/api/status", headers={"X-API-Token": "nope"}).status_code == 401


class TestWebConfig:
    def test_load_config_env_overrides(self, monkeypatch):
        monkeypatch.setenv("LATEST_LIMIT", "5")
        monkeypatch.setenv("YOLO_CONF", "0.7")
        monkeypatch.setenv("INFER_EVERY_N", "0")
        cfg = web_app.load_config()
        assert cfg.latest_limit == 5
        assert cfg.yolo_conf == pytest.approx(0.7)
        assert cfg.infer_every_n == 1

    def test_env_helpers_fallback_on_invalid(self, monkeypatch):
        from web_app import config as cfg_mod

        monkeypatch.setenv("BAD_INT_XYZ", "no-es-numero")
        monkeypatch.setenv("BAD_FLOAT_XYZ", "no-es-numero")
        assert cfg_mod._env_int("BAD_INT_XYZ", 9) == 9
        assert cfg_mod._env_float("BAD_FLOAT_XYZ", 1.5) == pytest.approx(1.5)


class TestWebSecurity:
    def test_origin_allowed_same_origin(self):
        from web_app.security import origin_allowed

        assert origin_allowed("http://localhost/", "http://localhost") is True

    def test_origin_allowed_cross_origin(self):
        from web_app.security import origin_allowed

        assert origin_allowed("https://evil.example", "http://localhost") is False

    def test_origin_allowed_missing_header(self):
        from web_app.security import origin_allowed

        assert origin_allowed(None, "http://localhost") is True

    def test_token_matches(self):
        from web_app.security import token_matches

        assert token_matches("abc", "abc") is True
        assert token_matches("abc", "abd") is False


class TestWebStorage:
    def test_list_latest_images_limit_zero(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"a")
        from web_app.storage import list_latest_images

        assert list_latest_images(str(tmp_path), 3, 0) == []

    def test_storage_save_isapi_snapshot_delegates(self, monkeypatch, tmp_path):
        import web_app.storage as storage

        captured: dict = {}

        def fake(host, user, password, folder, channel, timeout=3):
            captured.update(host=host, folder=folder, channel=channel, timeout=timeout)
            return "x.jpg"

        monkeypatch.setattr(storage, "_save_isapi_snapshot", fake)
        out = storage.save_isapi_snapshot(
            "h", "u", "p", folder=str(tmp_path), channel="101", timeout=5
        )
        assert out == "x.jpg"
        assert captured["channel"] == "101"
        assert captured["timeout"] == 5

    def test_app_save_isapi_snapshot_uses_config(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            web_app, "CONFIG", web_app.Config(capture_dir=str(tmp_path), isapi_host="cam")
        )
        captured: dict = {}

        def fake(host, user, password, folder, channel, timeout=3):
            captured.update(host=host, folder=folder, timeout=timeout)
            return None

        monkeypatch.setattr(web_app, "_storage_save_isapi_snapshot", fake)
        web_app.save_isapi_snapshot(timeout=9)
        assert captured["host"] == "cam"
        assert captured["folder"] == str(tmp_path)
        assert captured["timeout"] == 9
