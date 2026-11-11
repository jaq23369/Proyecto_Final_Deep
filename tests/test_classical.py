"""Pruebas del baseline B1 (cupo.classical y scripts/train_classical.py).

- HOG sobre imágenes sintéticas (dimensión calculada a mano, paralelo = serie).
- ``import cupo.classical`` no carga torch (ni en el proceso ni en los workers de joblib).
- Prueba de humo (marca ``data``): subset 0.05 con la misma estructura de ``runs/``.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cupo import classical as CL

from run_helpers import (
    CONFIGS, E6_COMMON, E6_HISTORY, E6_NPZ, E6_SUMMARY, HAS_DATA, PYTHON, REPO,
    install_split_recorders, load_strict_json, load_train_classical, tiny_classical_cfg,
)

needs_data = pytest.mark.skipif(not HAS_DATA, reason="dataset no disponible en DATA_ROOT")
# 64x64, celdas 8x8 → 8x8 celdas → bloques 2x2: 7x7 bloques · 2·2·9 = 1,764 features
HOG_DIM = 7 * 7 * 2 * 2 * 9


@pytest.fixture(scope="module")
def synthetic_images(tmp_path_factory) -> list[str]:
    folder = tmp_path_factory.mktemp("hog_imgs")
    rng = np.random.default_rng(0)
    out = []
    for i in range(12):
        arr = rng.integers(0, 256, (150, 150, 3), dtype=np.uint8)
        if i % 2:
            arr[40:110, 30:120] = 20  # "auto" oscuro en la mitad de las imágenes
        p = folder / f"patch_{i}.jpg"
        Image.fromarray(arr).save(p)
        out.append(str(p))
    return out


# ================================================================ unitarias

def test_hog_defaults_match_spec():
    assert CL.HOG_DEFAULTS == {"image_size": 64, "orientations": 9, "pixels_per_cell": 8,
                               "cells_per_block": 2}


def test_hog_one_dimension_by_hand(synthetic_images):
    f = CL.hog_one(synthetic_images[0])
    assert f.shape == (HOG_DIM,) and f.dtype == np.float32
    assert np.isfinite(f).all()


def test_hog_one_matches_skimage_reference(synthetic_images):
    from skimage.feature import hog

    with Image.open(synthetic_images[1]) as im:
        gray = im.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
    ref = hog(np.asarray(gray, dtype=np.float32) / 255.0, orientations=9, pixels_per_cell=(8, 8),
              cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
    np.testing.assert_allclose(CL.hog_one(synthetic_images[1]), ref, rtol=1e-5, atol=1e-6)


def test_hog_features_parallel_equals_serial(synthetic_images):
    serial = np.stack([CL.hog_one(p) for p in synthetic_images])
    par = CL.hog_features(synthetic_images, n_jobs=2, chunk_size=5)
    assert par.shape == (12, HOG_DIM)
    np.testing.assert_array_equal(par, serial)


def test_hog_features_custom_params(synthetic_images):
    f = CL.hog_features(synthetic_images[:3], {"image_size": 32}, n_jobs=1)
    # 32x32 → 4x4 celdas → 3x3 bloques · 36 = 324
    assert f.shape == (3, 324)


def test_import_classical_does_not_load_torch():
    code = ("import sys, cupo.classical, cupo.classical as c; "
            "print('torch' in sys.modules)")
    out = subprocess.run([PYTHON, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_hog_parallel_workers_do_not_load_torch(synthetic_images):
    """Los workers de joblib que extraen HOG no importan torch."""
    code = (
        "import sys\n"
        "from joblib import Parallel, delayed\n"
        "from cupo.classical import hog_features\n"
        f"paths = {synthetic_images!r}\n"
        "f = hog_features(paths, n_jobs=2, chunk_size=3)\n"
        "probe = Parallel(n_jobs=2)(delayed(eval)(\"'torch' in __import__('sys').modules\") for _ in range(4))\n"
        "print(f.shape[0], 'torch' in sys.modules, any(probe))\n"
    )
    out = subprocess.run([PYTHON, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["12", "False", "False"]


def test_models_reexports_hog():
    pytest.importorskip("torch")
    from cupo import models

    assert models.hog_features is CL.hog_features
    assert models.HOG_DEFAULTS is CL.HOG_DEFAULTS


def test_b1_config_matches_spec():
    from cupo.runs import load_yaml

    cfg = load_yaml(CONFIGS / "b1_hog_lr.yaml")
    assert cfg["model"]["C_grid"] == [0.01, 0.1, 1.0]
    assert cfg["model"]["hog"] == CL.HOG_DEFAULTS
    assert cfg["model"]["solver"] in ("lbfgs", "saga")


def test_search_C_picks_max_value_on_synthetic(com):
    """search_C elige el C de mayor valor en val y registra una fila por C."""
    pytest.importorskip("torch")
    mod = load_train_classical()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 6))
    y = (X[:, 0] + 0.3 * X[:, 1] + 0.5 * rng.normal(size=600) > -0.5).astype(int)
    mcfg = {"C_grid": [0.01, 0.1, 1.0], "solver": "lbfgs", "max_iter": 500}
    rows, best = mod.search_C(X[:400], y[:400], X[400:], y[400:], mcfg, 0, com)
    assert [r["C"] for r in rows] == [0.01, 0.1, 1.0]
    assert best["row"]["val_value"] == max(r["val_value"] for r in rows)
    assert best["logits"].shape == (200, 2) and (best["logits"][:, 0] == 0).all()


def test_history_frame_columns():
    pytest.importorskip("torch")
    mod = load_train_classical()
    row = {"train_acc": 0.9, "val_loss": 0.3, "val_acc": 0.88, "val_f1_macro": 0.87,
           "val_auc": 0.95, "val_value": 100.0, "train_nll": 0.25}
    h = mod.history_frame(row, 12.0)
    assert list(h.columns)[:len(E6_HISTORY)] == E6_HISTORY
    assert {"train_nll", "lr_start"} <= set(h.columns)


# ===================================================== invariante en ejecución

@pytest.mark.data
@needs_data
def test_train_classical_only_requests_train_and_val(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    mod = load_train_classical()
    rec, inner = install_split_recorders(monkeypatch)
    mod.run(tiny_classical_cfg(subset=0.005), run_name="inv_b1", runs_root=tmp_path)
    assert rec.calls and rec.names == {"train", "val"}
    assert all(allow is False for _, allow in rec.calls)
    assert set(inner.names) <= {"train", "val"}


# ============================================================ prueba de humo

@pytest.fixture(scope="module")
def b1_smoke(tmp_path_factory):
    if not HAS_DATA:
        pytest.skip("dataset no disponible en DATA_ROOT")
    root = tmp_path_factory.mktemp("runs_b1")
    cmd = [PYTHON, str(REPO / "scripts" / "train_classical.py"), "--config",
           str(CONFIGS / "b1_hog_lr.yaml"), "--subset", "0.05", "--runs-dir", str(root),
           "--run-name", "b1_smoke", "--n-boot", "50"]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=900)
    elapsed = time.perf_counter() - t0
    assert proc.returncode == 0, proc.stderr[-3000:]
    return root / "b1_smoke", elapsed, root


@pytest.mark.data
@needs_data
def test_b1_smoke_structure(b1_smoke):
    run_dir, elapsed, root = b1_smoke
    assert [p.name for p in root.iterdir()] == ["b1_smoke"]
    files = {p.name for p in run_dir.iterdir()}
    missing = set(E6_COMMON + ["model.joblib", "c_search.csv"]) - files
    assert not missing, f"faltan {missing}"
    assert "best.pt" not in files
    assert elapsed < 600


@pytest.mark.data
@needs_data
def test_b1_smoke_contents(b1_smoke):
    from cupo.data import load_split, split_dates, subsample
    from cupo.runs import load_npz, load_yaml

    run_dir = b1_smoke[0]
    cfg = load_yaml(run_dir / "config.yaml")
    assert cfg["data"]["subset"] == 0.05 and cfg["train"]["seed"] == 42
    hist = pd.read_csv(run_dir / "history.csv")
    assert list(hist.columns)[:len(E6_HISTORY)] == E6_HISTORY and len(hist) == 1
    search = pd.read_csv(run_dir / "c_search.csv")
    assert search["C"].tolist() == [0.01, 0.1, 1.0]
    s = load_strict_json(run_dir / "summary.json")
    assert not set(E6_SUMMARY) - set(s)
    assert s["C"] in (0.01, 0.1, 1.0)
    assert s["C"] == search.loc[search["val_value"].idxmax(), "C"]
    assert s["params_total"] == HOG_DIM + 1
    assert s["best_epoch"] is None and s["gpu"] is None
    assert s["train_cost_usd"] == pytest.approx(s["train_time_s"] / 3600 * 0.05, rel=1e-9)
    thr = load_strict_json(run_dir / "thresholds.json")
    for v in thr.values():
        assert 0 <= v["t_L"] <= v["t_H"] <= 1
    preds = load_npz(run_dir / "val_preds.npz")
    assert set(E6_NPZ) <= set(preds)
    assert preds["y"].size == len(subsample(load_split("val"), 0.05, 42))
    assert set(preds["date"].tolist()) <= split_dates("val")
    assert (preds["logits"][:, 0] == 0).all()
    for f in run_dir.glob("*.json"):
        load_strict_json(f)


@pytest.mark.data
@needs_data
def test_b1_smoke_model_reloads_and_predicts(b1_smoke):
    pytest.importorskip("torch")
    from cupo.evaluate import classical_logits, probs_from_logits
    from cupo.runs import load_npz, load_run

    run_dir = b1_smoke[0]
    run = load_run(run_dir)
    assert run.kind == "classical"
    preds = load_npz(run_dir / "val_preds.npz")
    paths = preds["path"][:20].tolist()
    feats = CL.hog_features(paths, run.model["hog"], n_jobs=1)
    p = probs_from_logits(classical_logits(run.model, feats))
    np.testing.assert_allclose(p, preds["p"][:20], atol=1e-6)


# ============================================ carpeta del run y subset efectivo

@pytest.fixture
def captured_classical(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from cupo import paths

    mod = load_train_classical()
    root = tmp_path / "runs"
    monkeypatch.setattr(paths, "runs_dir", lambda: root)
    calls = []

    def fake_run(cfg, run_name=None, runs_root=None):
        calls.append({"cfg": cfg, "runs_root": runs_root})

    monkeypatch.setattr(mod, "run", fake_run)
    return mod, root, calls


def test_classical_main_cli_subset_goes_to_smoke(captured_classical):
    mod, root, calls = captured_classical
    assert mod.main(["--config", str(CONFIGS / "b1_hog_lr.yaml"), "--subset", "0.05"]) == 0
    assert calls[0]["cfg"]["data"]["subset"] == 0.05
    assert calls[0]["runs_root"] == root / "_smoke"


def test_classical_main_full_goes_to_runs(captured_classical):
    mod, root, calls = captured_classical
    assert mod.main(["--config", str(CONFIGS / "b1_hog_lr.yaml")]) == 0
    assert calls[0]["cfg"]["data"]["subset"] == 1.0
    assert calls[0]["runs_root"] == root


def test_classical_main_yaml_subset_and_explicit_dir(captured_classical, tmp_path):
    from cupo.runs import load_yaml, save_yaml

    mod, root, calls = captured_classical
    cfg = load_yaml(CONFIGS / "b1_hog_lr.yaml")
    cfg["data"]["subset"] = 0.2
    yaml_path = tmp_path / "b1_sub.yaml"
    save_yaml(yaml_path, cfg)
    explicit = tmp_path / "mis_runs"
    mod.main(["--config", str(yaml_path)])
    mod.main(["--config", str(yaml_path), "--runs-dir", str(explicit)])
    mod.main(["--config", str(yaml_path), "--subset", "1.0"])
    assert calls[0]["cfg"]["data"]["subset"] == 0.2 and calls[0]["runs_root"] == root / "_smoke"
    assert calls[1]["runs_root"] == explicit
    assert calls[2]["cfg"]["data"]["subset"] == 1.0 and calls[2]["runs_root"] == root
