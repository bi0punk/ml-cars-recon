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
        monkeypatch.setattr(web_app.Config, "capture_dir", str(tmp_path))
        assert web_app.list_latest_images() == []

    def test_latest_images_lists_sorted_files(self, monkeypatch, tmp_path):
        monkeypatch.setattr(web_app.Config, "capture_dir", str(tmp_path))
        (tmp_path / "a.jpg").write_bytes(b"a")
        (tmp_path / "b.jpg").write_bytes(b"b")
        images = web_app.list_latest_images(limit=1)
        assert len(images) == 1
        assert images[0] in {"a.jpg", "b.jpg"}

    def test_captures_missing_returns_404(self, monkeypatch, tmp_path):
        monkeypatch.setattr(web_app.Config, "capture_dir", str(tmp_path))
        client = web_app.app.test_client()
        assert client.get("/captures/nope.jpg").status_code == 404

    def test_captures_blocks_path_traversal(self, monkeypatch, tmp_path):
        monkeypatch.setattr(web_app.Config, "capture_dir", str(tmp_path))
        client = web_app.app.test_client()
        assert client.get("/captures/../../etc/passwd").status_code in (403, 404)

    def test_requires_token_on_video_feed(self, monkeypatch):
        monkeypatch.setattr(web_app.Config, "api_token", "sekret")
        client = web_app.app.test_client()
        assert client.get("/video_feed").status_code == 401

    def test_requires_token_on_status(self, monkeypatch):
        monkeypatch.setattr(web_app.Config, "api_token", "sekret")
        client = web_app.app.test_client()
        assert client.get("/api/status").status_code == 401

    def test_token_grants_access(self, monkeypatch):
        monkeypatch.setattr(web_app.Config, "api_token", "sekret")
        monkeypatch.setattr(web_app.Config, "rtsp_url", "rtsp://invalid-host")
        client = web_app.app.test_client()
        resp = client.get("/api/status", headers={"X-API-Token": "sekret"})
        assert resp.status_code == 200
        assert resp.get_json() == {"active": False}
