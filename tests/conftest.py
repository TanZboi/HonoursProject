from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="needs a CUDA GPU")
    skip_scene = pytest.mark.skip(reason="set SEG3D_SCENE=/path/to/rbb335.ply to run")
    has_cuda = _cuda_available()
    has_scene = bool(os.environ.get("SEG3D_SCENE"))
    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)
        if "scene" in item.keywords and not has_scene:
            item.add_marker(skip_scene)


@pytest.fixture(scope="session")
def toy_module():
    """scripts/make_toy_scene.py, imported as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("make_toy_scene", REPO_ROOT / "scripts" / "make_toy_scene.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def toy_ply() -> Path:
    return FIXTURES / "toy.ply"
