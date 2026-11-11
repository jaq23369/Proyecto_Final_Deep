"""Pruebas de rutas (paths.py) y utilidades generales (utils.py)."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

from cupo import paths, utils


# ------------------------------------------------------------------ paths

def test_repo_root_contains_package():
    root = paths.repo_root()
    assert (root / "src" / "cupo" / "paths.py").is_file()
    assert (root / "pyproject.toml").is_file()


def test_data_root_uses_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(tmp_path))
    assert paths.data_root() == tmp_path.resolve()


def test_data_root_env_is_absolute_and_resolved(monkeypatch, tmp_path):
    (tmp_path / "a").mkdir()
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(tmp_path / "a" / ".." / "a"))
    root = paths.data_root()
    assert root.is_absolute()
    assert root == (tmp_path / "a").resolve()


def test_data_root_default_is_sibling_data(monkeypatch):
    monkeypatch.delenv(paths.DATA_ROOT_ENV, raising=False)
    assert paths.data_root() == (paths.repo_root().parent / "Data").resolve()


def test_data_root_empty_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv(paths.DATA_ROOT_ENV, "")
    assert paths.data_root() == (paths.repo_root().parent / "Data").resolve()


def test_derived_paths_follow_data_root(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(tmp_path))
    root = tmp_path.resolve()
    assert paths.metadata_csv() == root / "CNRPark+EXT.csv"
    assert paths.labels_dir() == root / "CNR-EXT" / "LABELS"
    assert paths.patches_dir() == root / "CNR-EXT" / "PATCHES"
    assert paths.split_file("val") == root / "CNR-EXT" / "LABELS" / "val.txt"
    assert paths.cnrpark_dir() == root / "CNRPark"
    assert paths.full_images_dir() == root / "CNR-EXT_FULL" / "FULL_IMAGE_1000x750"
    assert paths.camera_boxes_csv(3) == root / "CNR-EXT_FULL" / "camera3.csv"
    assert paths.camera_boxes_csv("7") == root / "CNR-EXT_FULL" / "camera7.csv"


def test_repo_paths_do_not_depend_on_data_root(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(tmp_path))
    root = paths.repo_root()
    assert paths.configs_dir() == root / "configs"
    assert paths.runs_dir() == root / "runs"
    assert paths.models_dir() == root / "models"
    assert paths.results_dir() == root / "results"


# ------------------------------------------------------------------ utils

def test_set_seed_reproducible_random_numpy_torch(monkeypatch):
    # monkeypatch restaura PYTHONHASHSEED al terminar
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    torch = pytest.importorskip("torch")
    utils.set_seed(123)
    a = (random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist())
    utils.set_seed(123)
    b = (random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist())
    assert a == b
    utils.set_seed(124)
    c = (random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist())
    assert a != c


def test_set_seed_sets_pythonhashseed(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    utils.set_seed(7)
    assert os.environ["PYTHONHASHSEED"] == "7"


def test_ensure_dir_creates_nested(tmp_path):
    target = tmp_path / "x" / "y" / "z"
    out = utils.ensure_dir(target)
    assert out == target and target.is_dir()
    # idempotente
    assert utils.ensure_dir(target) == target


def test_env_info_has_expected_keys():
    info = utils.env_info()
    for key in ("git_commit", "git_branch", "git_dirty", "python", "torch", "torchvision",
                "cuda", "gpu", "numpy", "cpu", "cpu_count", "platform", "date"):
        assert key in info
    assert info["python"].startswith("3.")
    assert info["numpy"] == np.__version__
