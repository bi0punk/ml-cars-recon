from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_repo_has_readme():
    assert (REPO_ROOT / "README.md").exists()


def test_repo_has_gitignore():
    assert (REPO_ROOT / ".gitignore").exists()


def test_repo_has_license():
    assert (REPO_ROOT / "LICENSE").exists()


def test_no_images_tracked():
    import subprocess
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True
    ).stdout
    imgs = [line for line in tracked.splitlines() if line.lower().endswith((".jpg", ".jpeg", ".png"))]
    assert not imgs, f"Imágenes trackeadas: {imgs}"


def test_no_model_files_tracked():
    import subprocess
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True
    ).stdout
    models = [line for line in tracked.splitlines() if line.endswith((".pt", ".h5", ".keras", ".onnx", ".pkl"))]
    assert not models, f"Modelos trackeados: {models}"


def test_web_app_imports_without_side_effects():
    pytest.importorskip("flask")
    import web_app.app as web_app

    assert web_app._camera is None, "importar web_app no debe crear la cámara"
    assert web_app._model is None, "importar web_app no debe cargar el modelo YOLO"


def test_web_app_create_app_does_not_start_camera():
    pytest.importorskip("flask")
    import web_app.app as web_app

    app = web_app.create_app()
    assert app is not None
    assert web_app._camera is None


def test_web_app_reset_camera_is_safe_when_stopped():
    pytest.importorskip("flask")
    import web_app.app as web_app

    web_app.reset_camera()
    assert web_app._camera is None
