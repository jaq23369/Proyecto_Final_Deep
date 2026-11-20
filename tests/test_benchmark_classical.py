"""Benchmark del baseline clásico B1 y el camino único de features HOG.

- ``cupo.benchmark`` con un ``.joblib`` (por ``--joblib`` y por ``--onnx`` con extensión
  ``.joblib``): claves del JSON, misma base de costo que ONNX, sin GPU, nada en ``models/``.
- Sin dataset: las rutas de val se sustituyen por patches sintéticos en ``tmp_path``.
- Con dataset (marca ``data``): ``models/b1_hog_lr.joblib`` real con pocas repeticiones; solo
  se pide la partición val.
- ``classical.classical_features`` da lo mismo que el entrenamiento (``hog_one``/``hog_features``)
  y que ``app/inference.py``; con B1 real, la p coincide con ``val_preds.npz`` del run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cupo import benchmark as B
from cupo import classical as CL
from cupo import paths
from cupo.runs import load_json, load_yaml

from run_helpers import HAS_DATA, REPO, SplitRecorder, load_strict_json

MODELS_DIR = REPO / "models"
B1_JOBLIB = MODELS_DIR / "b1_hog_lr.joblib"
B1_CARD = B1_JOBLIB.with_suffix(".json")
HAS_B1 = B1_JOBLIB.is_file()
N_SLOT, N_FRAME = 4, 2

TOP_KEYS = {"model", "input_size", "data", "warmup", "n_slot", "n_frame", "preprocess_ms_per_patch",
            "preprocess_basis", "model_latency_includes", "cpu_1_thread", "cpu_all_threads", "gpu_torch",
            "costo_por_1000_pred_Q", "cost", "volume", "volume_by_scenario", "env", "measured_at"}
COST_KEYS = {"core_seconds_per_1000", "usd", "q", "basis", "pipeline_pred_s", "model_only",
             "estimate_note", "cpu_server_usd_per_hour", "gtq_per_usd"}


def _stamp(folder: Path) -> list[tuple[str, int]]:
    if not folder.is_dir():
        return []
    return sorted((p.name, p.stat().st_mtime_ns) for p in folder.iterdir())


def _synthetic_patches(folder: Path, n: int = 8) -> list[str]:
    """Patches 150x150 RGB en JPG; la mitad con un bloque oscuro ("auto")."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        arr = rng.integers(0, 256, (150, 150, 3), dtype=np.uint8)
        if i % 2:
            arr[40:110, 30:120] = 25
        p = folder / f"patch_{i}.jpg"
        Image.fromarray(arr).save(p, quality=92)
        out.append(str(p))
    return out


def _synthetic_bundle(image_paths: list[str]) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    hog = dict(CL.HOG_DEFAULTS)
    X = CL.hog_features(image_paths, hog, n_jobs=1)
    y = np.array([i % 2 for i in range(len(image_paths))])
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500)).fit(X, y)
    return {"pipeline": pipe, "hog": hog, "C": 0.1}


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """Patches sintéticos + ``hog_tiny.joblib`` con tarjeta en una carpeta temporal."""
    import joblib

    root = tmp_path_factory.mktemp("bench_classical")
    imgs = _synthetic_patches(root / "patches")
    model = root / "models" / "hog_tiny.joblib"
    model.parent.mkdir()
    joblib.dump(_synthetic_bundle(imgs), model)
    card = {"name": "hog_tiny", "run_id": "hog_tiny_s0", "model": "hog_lr", "kind": "classical"}
    model.with_suffix(".json").write_text(json.dumps(card), encoding="utf-8")
    return {"root": root, "imgs": imgs, "model": model}


@pytest.fixture
def fake_val(synth, monkeypatch):
    """Las rutas de val del benchmark son los patches sintéticos (sin dataset)."""
    calls: list[tuple[int, int]] = []

    def fake(n_images: int = B.SLOTS_PER_FRAME, seed: int = B.SAMPLE_SEED) -> list[str]:
        calls.append((n_images, seed))
        return list(synth["imgs"])

    monkeypatch.setattr(B, "sample_val_paths", fake)
    return calls


def _run_main(args: list[str], capsys) -> tuple[int, str, str]:
    rc = B.main(args)
    out = capsys.readouterr()
    return rc, out.out, out.err


# ---------------------------------------------------------------------------
# Unidades
# ---------------------------------------------------------------------------

def test_model_kind_and_path_from_args(tmp_path):
    assert B.model_kind(Path("m/b1.joblib")) == B.KIND_CLASSICAL
    assert B.model_kind(Path("m/B1.JOBLIB")) == B.KIND_CLASSICAL
    assert B.model_kind(Path("m/best.onnx")) == B.KIND_ONNX
    args = B.parse_args([])
    assert args.joblib is None
    assert B.model_path_from_args(args) == (paths.models_dir() / "best.onnx").resolve()
    args = B.parse_args(["--joblib", str(tmp_path / "x.joblib"), "--onnx", str(tmp_path / "y.onnx")])
    assert B.model_path_from_args(args) == (tmp_path / "x.joblib").resolve()


def test_model_info_classical(synth):
    info = B.model_info(synth["model"])
    assert info["kind"] == "classical" and info["file"] == "hog_tiny.joblib"
    assert info["onnx"] is None
    assert info["name"] == "hog_tiny" and info["run_id"] == "hog_tiny_s0" and info["model"] == "hog_lr"
    assert info["size_mb"] == pytest.approx(synth["model"].stat().st_size / 1024**2)


def test_load_classical_bundle_rejects_bad_format(tmp_path):
    import joblib

    bad = tmp_path / "malo.joblib"
    joblib.dump({"pipeline": None}, bad)
    with pytest.raises(ValueError, match="pipeline"):
        B.load_classical_bundle(bad)
    joblib.dump([1, 2, 3], bad)
    with pytest.raises(ValueError):
        B.load_classical_bundle(bad)


def test_environment_classical_has_no_onnxruntime():
    env = B.environment(B.KIND_CLASSICAL)
    assert {"cpu", "cpu_count", "scikit-learn", "scikit-image", "python", "platform"} <= set(env)
    assert "onnxruntime" not in env


def test_bench_classical_cpu_modes(synth):
    bundle = B.load_classical_bundle(synth["model"])
    images = []
    for p in synth["imgs"]:
        with Image.open(p) as im:
            images.append(im.convert("RGB"))
    frame = [images[i % len(images)] for i in range(B.SLOTS_PER_FRAME)]
    res = B.bench_classical_cpu(bundle, images, frame, n=3, n_frame=2, threads=1)
    assert res["threads"] == 1 and "scikit" in res["engine"]
    assert res["per_slot"]["batch"] == 1 and res["per_slot"]["n"] == 3
    assert res["per_frame"]["batch"] == B.SLOTS_PER_FRAME and res["per_frame"]["n"] == 2
    res_all = B.bench_classical_cpu(bundle, images, frame, n=2, n_frame=1, threads=None)
    assert res_all["threads"] == "default"


def test_bench_classical_uses_classical_features(synth, monkeypatch):
    """La latencia del modelo incluye el HOG por la función compartida con la app."""
    bundle = B.load_classical_bundle(synth["model"])
    original = CL.classical_features
    sizes: list[int] = []

    def spy(hog_params, images):
        sizes.append(len(images))
        return original(hog_params, images)

    monkeypatch.setattr(CL, "classical_features", spy)
    with Image.open(synth["imgs"][0]) as im:
        img = im.convert("RGB")
    frame = [img] * B.SLOTS_PER_FRAME
    B.bench_classical_cpu(bundle, [img], frame, n=2, n_frame=1, threads=1)
    # warmup + n por espacio (batch 1) y warmup + n_frame por frame (batch 46)
    assert sizes == [1] * (B.WARMUP + 2) + [B.SLOTS_PER_FRAME] * (B.WARMUP + 1)


# ---------------------------------------------------------------------------
# Corrida corta sin dataset (patches sintéticos)
# ---------------------------------------------------------------------------

def _check_classical_json(res: dict, model_file: str, n_slot: int, n_frame: int) -> None:
    assert TOP_KEYS <= set(res), sorted(TOP_KEYS - set(res))
    info = res["model"]
    assert info["kind"] == "classical" and info["file"] == model_file and info["onnx"] is None
    assert res["gpu_torch"] is None
    assert res["model_latency_includes"] == B.MODEL_LATENCY_CLASSICAL
    assert "HOG" in res["model_latency_includes"]
    assert res["preprocess_basis"] == B.PREPROCESS_BASIS_CLASSICAL
    assert res["input_size"] == 64
    assert res["data"] == {"split": "val", "n_images": res["data"]["n_images"], "seed": B.SAMPLE_SEED}
    assert res["n_slot"] == n_slot and res["n_frame"] == n_frame
    assert res["cpu_1_thread"]["threads"] == 1 and res["cpu_all_threads"]["threads"] == "default"
    for block in ("cpu_1_thread", "cpu_all_threads"):
        for mode, batch, n in (("per_slot", 1, n_slot), ("per_frame", B.SLOTS_PER_FRAME, n_frame)):
            st = res[block][mode]
            assert st["batch"] == batch and st["n"] == n
            assert st["p50_ms"] > 0 and st["p95_ms"] >= st["p50_ms"]
            assert st["throughput_pred_s"] == pytest.approx(batch / (st["mean_ms"] / 1000))
    assert "scikit-learn" in res["env"] and "onnxruntime" not in res["env"]


def _check_cost_basis(res: dict) -> None:
    """Misma base que ONNX: 1 núcleo, modo frame, modelo + preprocesamiento."""
    costs = load_yaml(paths.configs_dir() / "costs.yaml")
    usd_h, gtq = float(costs["cpu_server_usd_per_hour"]), float(costs["gtq_per_usd"])
    cost = res["cost"]
    assert COST_KEYS <= set(cost)
    assert cost["cpu_server_usd_per_hour"] == usd_h and cost["gtq_per_usd"] == gtq
    assert cost["basis"].startswith("1 núcleo (")
    assert cost["basis"].endswith("modo frame, modelo + preprocesamiento")
    slots = res["volume"]["slots_per_frame"]
    assert slots == B.SLOTS_PER_FRAME
    frame_ms = res["cpu_1_thread"]["per_frame"]["mean_ms"]
    pre_ms = res["preprocess_ms_per_patch"]
    pipeline = slots / ((frame_ms + slots * pre_ms) / 1000.0)
    assert cost["pipeline_pred_s"] == pytest.approx(pipeline, rel=1e-9)
    seconds = 1000.0 / pipeline
    assert cost["core_seconds_per_1000"] == pytest.approx(seconds, rel=1e-9)
    assert cost["q"] == pytest.approx(seconds / 3600 * usd_h * gtq, rel=1e-9)
    assert res["costo_por_1000_pred_Q"] == pytest.approx(cost["q"], rel=1e-12)
    thr = res["cpu_1_thread"]["per_frame"]["throughput_pred_s"]
    assert cost["model_only"]["q"] == pytest.approx(1000.0 / thr / 3600 * usd_h * gtq, rel=1e-9)
    frame_s = res["volume"]["seconds_per_frame_1core"]
    assert frame_s == pytest.approx((frame_ms + slots * pre_ms) / 1000.0, rel=1e-9)
    assert res["volume"]["cameras_per_core"] == math.floor(res["volume"]["refresh_s"] / frame_s)


@pytest.mark.parametrize("flag", ["--joblib", "--onnx"])
def test_main_classical_synthetic(synth, fake_val, tmp_path, capsys, flag):
    before_models, before_synth = _stamp(MODELS_DIR), _stamp(synth["model"].parent)
    out = tmp_path / "out"
    rc, stdout, err = _run_main([flag, str(synth["model"]), "--n", str(N_SLOT), "--n-frame", str(N_FRAME),
                                 "--out", str(out), "--no-gpu"], capsys)
    assert rc == 0, err
    assert sorted(p.name for p in out.iterdir()) == ["benchmark.json"]
    res = load_strict_json(out / "benchmark.json")
    _check_classical_json(res, "hog_tiny.joblib", N_SLOT, N_FRAME)
    assert res["data"]["n_images"] == len(synth["imgs"])
    _check_cost_basis(res)
    assert fake_val and all(n == B.SLOTS_PER_FRAME and s == B.SAMPLE_SEED for n, s in fake_val)
    assert "Latencia del modelo" in stdout and "HOG" in stdout
    # No escribe en models/ ni junto al .joblib
    assert _stamp(MODELS_DIR) == before_models
    assert _stamp(synth["model"].parent) == before_synth
    text = (out / "benchmark.json").read_text(encoding="utf-8")
    assert str(synth["root"]) not in text and str(synth["root"]).replace("\\", "/") not in text


def test_main_missing_joblib_fails(tmp_path, capsys):
    out = tmp_path / "out"
    rc, _, err = _run_main(["--joblib", str(tmp_path / "no_existe.joblib"), "--out", str(out)], capsys)
    assert rc == 1
    assert "export_classical" in err
    assert not out.exists()


def test_main_bad_joblib_fails_and_writes_nothing(tmp_path, fake_val, capsys):
    import joblib

    bad = tmp_path / "raro.joblib"
    joblib.dump({"pipeline": "x"}, bad)
    out = tmp_path / "out"
    rc, _, err = _run_main(["--joblib", str(bad), "--n", "2", "--n-frame", "1", "--out", str(out)], capsys)
    assert rc == 1 and "Error" in err
    assert not out.exists()


# ---------------------------------------------------------------------------
# Corrida corta con el dataset y el B1 real
# ---------------------------------------------------------------------------

needs_b1 = pytest.mark.skipif(not (HAS_DATA and HAS_B1), reason="requiere el dataset y models/b1_hog_lr.joblib")


@pytest.fixture(scope="module")
def real_b1(tmp_path_factory):
    if not (HAS_DATA and HAS_B1):
        pytest.skip("requiere el dataset y models/b1_hog_lr.joblib")
    import cupo.data

    out = tmp_path_factory.mktemp("bench_b1") / "out"
    before = _stamp(MODELS_DIR)
    with pytest.MonkeyPatch.context() as mp:
        rec = SplitRecorder(cupo.data.load_split)
        mp.setattr(cupo.data, "load_split", rec)
        rc = B.main(["--onnx", str(B1_JOBLIB), "--n", str(N_SLOT), "--n-frame", str(N_FRAME),
                     "--out", str(out), "--no-gpu"])
    return {"rc": rc, "rec": rec, "out": out, "before": before}


@pytest.mark.data
@needs_b1
def test_real_b1_benchmark_json(real_b1):
    assert real_b1["rc"] == 0
    assert sorted(p.name for p in real_b1["out"].iterdir()) == ["benchmark.json"]
    res = load_strict_json(real_b1["out"] / "benchmark.json")
    _check_classical_json(res, "b1_hog_lr.joblib", N_SLOT, N_FRAME)
    assert res["data"]["n_images"] == B.SLOTS_PER_FRAME
    _check_cost_basis(res)
    if B1_CARD.is_file():
        card = load_json(B1_CARD)
        assert res["model"]["name"] == card["name"] and res["model"]["run_id"] == card["run_id"]


@pytest.mark.data
@needs_b1
def test_real_b1_only_reads_val_and_leaves_models(real_b1):
    assert real_b1["rec"].names == {"val"}
    assert all(not allow for _, allow in real_b1["rec"].calls)
    assert _stamp(MODELS_DIR) == real_b1["before"]


@pytest.mark.data
@needs_b1
def test_real_b1_same_keys_and_basis_as_onnx(real_b1, tmp_path):
    """El JSON de B1 tiene las mismas claves que el de un ONNX medido con el mismo protocolo."""
    pytest.importorskip("onnxruntime")
    onnx = next((p for p in (MODELS_DIR / "b2_malexnet.onnx", MODELS_DIR / "best.onnx") if p.is_file()), None)
    if onnx is None:
        pytest.skip("no hay un .onnx en models/")
    before = _stamp(MODELS_DIR)
    out = tmp_path / "onnx_out"
    assert B.main(["--onnx", str(onnx), "--n", "2", "--n-frame", "1", "--out", str(out), "--no-gpu"]) == 0
    assert _stamp(MODELS_DIR) == before
    a = load_json(real_b1["out"] / "benchmark.json")
    b = load_json(out / "benchmark.json")
    assert set(a) == set(b)
    assert set(a["cost"]) == set(b["cost"]) and set(a["cost"]["model_only"]) == set(b["cost"]["model_only"])
    assert set(a["model"]) == set(b["model"])
    assert b["model"]["kind"] == "onnx" and b["model"]["file"] == b["model"]["onnx"] == onnx.name
    assert b["model_latency_includes"] == B.MODEL_LATENCY_ONNX
    # Misma base de costo, distinto motor
    tail = "modo frame, modelo + preprocesamiento"
    assert a["cost"]["basis"].endswith(tail) and b["cost"]["basis"].endswith(tail)
    assert a["data"]["split"] == b["data"]["split"] == "val" and a["data"]["seed"] == b["data"]["seed"]
    assert a["data"]["n_images"] == b["data"]["n_images"]
    assert a["volume"]["slots_per_frame"] == b["volume"]["slots_per_frame"] == B.SLOTS_PER_FRAME


@pytest.mark.data
@pytest.mark.skipif(not HAS_DATA, reason="requiere el dataset")
def test_same_val_sample_for_all_models():
    """``sample_val_paths`` es determinista: B1 y las CNN se miden con los mismos patches."""
    a = B.sample_val_paths()
    b = B.sample_val_paths()
    assert a == b and len(a) == B.SLOTS_PER_FRAME
    assert all(Path(p).is_file() for p in a)


# ---------------------------------------------------------------------------
# classical_features: mismo camino que el entrenamiento y la app
# ---------------------------------------------------------------------------

def test_classical_features_matches_training_path(synth, tmp_path):
    hog = dict(CL.HOG_DEFAULTS)
    paths_ = list(synth["imgs"])
    # Imagen en gris (modo L) en PNG: misma salida por los dos caminos
    gray = tmp_path / "gris.png"
    Image.fromarray(np.random.default_rng(1).integers(0, 256, (90, 120), dtype=np.uint8), "L").save(gray)
    paths_.append(str(gray))
    images = []
    for p in paths_:
        with Image.open(p) as im:
            im.load()
            images.append(im.copy())
    feats = CL.classical_features(hog, images)
    assert feats.shape == (len(paths_), 1764) and feats.dtype == np.float32
    np.testing.assert_array_equal(feats, np.stack([CL.hog_one(p, **hog) for p in paths_]))
    np.testing.assert_array_equal(feats, CL.hog_features(paths_, hog, n_jobs=1))
    np.testing.assert_array_equal(feats, np.stack([CL.hog_image(im.convert("RGB"), **hog) for im in images]))


def test_classical_features_custom_params(synth):
    hog = {"image_size": 32, "orientations": 9, "pixels_per_cell": 8, "cells_per_block": 2}
    with Image.open(synth["imgs"][0]) as im:
        img = im.convert("RGB")
    feats = CL.classical_features(hog, [img, img])
    # 32x32 → 4x4 celdas → 3x3 bloques · 2·2·9 = 324
    assert feats.shape == (2, 324)
    np.testing.assert_array_equal(feats[0], CL.hog_one(synth["imgs"][0], **hog))


def test_app_classical_features_uses_shared_function(synth, monkeypatch):
    pytest.importorskip("onnxruntime")
    from app import inference as inf

    bundle = B.load_classical_bundle(synth["model"])
    imgs = []
    for p in synth["imgs"][:3]:
        with Image.open(p) as im:
            imgs.append(im.convert("RGB"))
    expected = CL.classical_features(bundle["hog"], imgs)
    calls: list[int] = []
    original = CL.classical_features

    def spy(hog_params, images):
        calls.append(len(images))
        return original(hog_params, images)

    monkeypatch.setattr(CL, "classical_features", spy)
    got = inf._classical_features(bundle, imgs)
    assert calls == [3]
    np.testing.assert_array_equal(got, expected)


@pytest.mark.data
@needs_b1
def test_b1_classical_features_match_val_preds():
    """Con B1 real, sigmoid(z) de classical_features coincide con val_preds.npz y con la app."""
    pytest.importorskip("onnxruntime")
    import joblib

    from app import inference as inf
    from cupo.data import load_split

    if not B1_CARD.is_file():
        pytest.skip("falta models/b1_hog_lr.json")
    run_dir = REPO / "runs" / str(load_json(B1_CARD).get("run_id"))
    if not (run_dir / "val_preds.npz").is_file():
        pytest.skip("no está el run de B1 con val_preds.npz")
    preds = np.load(run_dir / "val_preds.npz", allow_pickle=False)
    val = load_split("val").reset_index(drop=True)
    assert len(val) == len(preds["p"])
    idx = np.sort(np.random.default_rng(7).choice(len(val), size=8, replace=False))
    for i in idx:
        assert Path(str(preds["path"][i])).name == Path(val["path"].iloc[i]).name
    imgs = []
    for i in idx:
        with Image.open(val["path"].iloc[i]) as im:
            imgs.append(im.convert("RGB"))
    bundle = joblib.load(B1_JOBLIB)
    z = CL.classical_logits(bundle, CL.classical_features(bundle["hog"], imgs))
    e = np.exp(z - z.max(axis=1, keepdims=True))
    p = e[:, 1] / e.sum(axis=1)
    np.testing.assert_allclose(p, preds["p"][idx], atol=1e-4)
    # Mismas features que el entrenamiento (desde disco)
    np.testing.assert_array_equal(CL.classical_features(bundle["hog"], imgs),
                                  CL.hog_features(list(val["path"].iloc[idx]), bundle["hog"], n_jobs=1))
    p_app, _ = inf.load_predictor(B1_JOBLIB).predict(imgs)
    np.testing.assert_allclose(p_app, preds["p"][idx], atol=1e-4)
