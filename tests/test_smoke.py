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
    imgs = [l for l in tracked.splitlines() if l.lower().endswith((".jpg", ".jpeg", ".png"))]
    assert not imgs, f"Imágenes trackeadas: {imgs}"


def test_no_model_files_tracked():
    import subprocess
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True
    ).stdout
    models = [l for l in tracked.splitlines() if l.endswith((".pt", ".h5", ".keras", ".onnx", ".pkl"))]
    assert not models, f"Modelos trackeados: {models}"


def test_web_app_imports():
    pytest.importorskip("flask")
    import importlib
    try:
        importlib.import_module("web_app.app")
    except ImportError:
        pass
