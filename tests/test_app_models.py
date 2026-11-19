"""Pruebas de la app: selector de modelos, backend clásico (B1), export_classical, fotos
propias, mosaicos de CNRPark y el .gitignore de la galería local.

Igual que en ``test_app.py``: ``MODEL_PATH``, ``FEEDBACK_PATH``, ``OWN_GALLERY_DIR`` y, cuando
hace falta, ``DATA_ROOT`` apuntan a carpetas de ``tmp_path``. Las pruebas que usan el modelo B1
exportado (``models/b1_hog_lr.joblib``) o el run de B1 se saltan si no están; las que leen
patches de validación se saltan si no está el dataset. Nada lee test.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app_helpers import (
    REPO,
    TEST_ONNX_ENV,
    TINY_THRESHOLDS,
    add_tiny_model,
    assert_private,
    fake_data_root,
    gallery_json_items,
    has_gallery_data,
    has_key,
    leaks,
    make_model_dir,
    make_tiny_onnx,
    noise_image,
    png_bytes,
    read_card,
    tinted_image,
)
from fastapi.testclient import TestClient

from app import inference as inf
from app import main as M
from cupo import classical as CL
from cupo import paths
from cupo.transforms import preprocess_numpy

USING_TINY = not os.environ.get(TEST_ONNX_ENV)
HAS_DATA = has_gallery_data()
HAS_VAL = paths.split_file("val").is_file() and paths.patches_dir().is_dir()

MODELS_DIR = REPO / "models"
B1_JOBLIB = MODELS_DIR / "b1_hog_lr.joblib"
HAS_B1 = B1_JOBLIB.is_file() and B1_JOBLIB.with_suffix(".json").is_file()


def _b1_run_dir() -> Path | None:
    if not HAS_B1:
        return None
    run_id = json.loads(B1_JOBLIB.with_suffix(".json").read_text(encoding="utf-8")).get("run_id")
    run_dir = REPO / "runs" / str(run_id)
    return run_dir if (run_dir / "model.joblib").is_file() else None


B1_RUN = _b1_run_dir()

ALT_THRESHOLDS = {"comercial": {"t_L": 0.1, "t_H": 0.2}, "uvg": {"t_L": 0.8, "t_H": 0.9}}
ENTRY_KEYS = {"id", "label", "name", "model", "kind", "is_default", "val_f1", "val_value_comercial",
              "thresholds", "available", "error"}


def rule(p: float, t_low: float, t_high: float) -> str:
    return "libre" if p < t_low else ("ocupado" if p >= t_high else "verificar")


def expected_p(onnx_path: Path, imgs: list[Image.Image]) -> np.ndarray:
    """P(ocupado) calculada aparte: ONNX Runtime + preprocess_numpy + softmax(z / T)."""
    import onnxruntime as ort

    card = read_card(onnx_path)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    batch = np.concatenate([preprocess_numpy(im, int(card["input_size"])) for im in imgs], axis=0)
    logits = sess.run(None, {sess.get_inputs()[0].name: batch})[0].astype(np.float64)
    z = logits / float(card.get("temperature", 1.0))
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e[:, 1] / e.sum(axis=1)


def upload(name: str, data: bytes, ctype: str = "image/png") -> dict:
    return {"file": (name, data, ctype)}


def jpeg_bytes(img: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def first_id(kind: str) -> str:
    return next(it["id"] for it in gallery_json_items() if it["kind"] == kind)


# ================================================================ fixtures

@pytest.fixture(autouse=True)
def own_dir(tmp_path, monkeypatch) -> Path:
    """Galería propia vacía en tmp_path (nunca la carpeta real app/galeria_propia)."""
    folder = tmp_path / "galeria_propia"
    folder.mkdir()
    monkeypatch.setenv("OWN_GALLERY_DIR", str(folder))
    return folder


@pytest.fixture(autouse=True)
def _repo_files_untouched():
    """Ni app/feedback.jsonl ni models/ cambian durante las pruebas."""
    fb = REPO / "app" / "feedback.jsonl"

    def stamp():
        files = sorted(MODELS_DIR.glob("*")) if MODELS_DIR.is_dir() else []
        return ([(p.name, p.stat().st_mtime_ns) for p in files],
                fb.stat().st_mtime_ns if fb.exists() else None)

    before = stamp()
    yield
    assert stamp() == before, "se escribió en models/ o en app/feedback.jsonl"


@pytest.fixture
def model_dir(tmp_path, monkeypatch) -> Path:
    """Carpeta de modelos: best (por defecto), alt (otros umbrales), roto (tarjeta inválida) y
    huerfano.onnx sin tarjeta (no cuenta como modelo)."""
    best = make_model_dir(tmp_path / "models")
    folder = best.parent
    add_tiny_model(folder, "alt", "alt_test", ALT_THRESHOLDS)
    roto = add_tiny_model(folder, "roto", "roto_test")
    card = read_card(roto)
    card.pop("thresholds")
    roto.with_suffix(".json").write_text(json.dumps(card), encoding="utf-8")
    make_tiny_onnx(folder / "huerfano.onnx")
    monkeypatch.setenv("MODEL_PATH", str(best))
    return folder


@pytest.fixture
def feedback_file(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "fb" / "feedback.jsonl"
    monkeypatch.setenv("FEEDBACK_PATH", str(path))
    return path


@pytest.fixture
def client(model_dir, feedback_file, monkeypatch) -> TestClient:
    monkeypatch.delenv("SHOW_GROUND_TRUTH", raising=False)
    return TestClient(M.app, raise_server_exceptions=False)


@pytest.fixture
def fake_root(tmp_path, monkeypatch) -> Path:
    root = fake_data_root(tmp_path / "data", cameras=(1,))
    monkeypatch.setenv("DATA_ROOT", str(root))
    return root


# ================================================================ registro de modelos

def test_api_models_lists_registry(client, model_dir, tmp_path):
    r = client.get("/api/models")
    assert r.status_code == 200
    body = r.json()
    assert body["default"] == "best"
    entries = {e["id"]: e for e in body["models"]}
    assert set(entries) == {"best", "alt", "roto"}  # huerfano.onnx no tiene tarjeta
    for e in entries.values():
        assert set(e) == ENTRY_KEYS
        assert e["kind"] == "onnx"
    assert [e["id"] for e in body["models"] if e["is_default"]] == ["best"]
    assert entries["alt"]["available"] is True and entries["alt"]["error"] is None
    assert entries["alt"]["thresholds"] == ALT_THRESHOLDS
    assert entries["alt"]["name"] == "alt_test" and entries["alt"]["label"] == "alt_test"
    assert entries["best"]["available"] is True
    if USING_TINY:
        assert entries["best"]["thresholds"] == TINY_THRESHOLDS
    assert entries["roto"]["available"] is False
    assert "umbrales" in entries["roto"]["error"]
    assert_private(r, str(tmp_path))


def test_api_models_default_listed_even_if_missing(client, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nada" / "best.onnx"))
    body = client.get("/api/models").json()
    assert body["default"] == "best"
    assert [(e["id"], e["available"]) for e in body["models"]] == [("best", False)]


def test_api_models_with_repo_models_folder(feedback_file, monkeypatch):
    """Con models/ del repo (solo lectura): best, B1 y B2 seleccionables y disponibles."""
    if not (MODELS_DIR / "best.onnx").is_file() or not (MODELS_DIR / "best.json").is_file():
        pytest.skip("no hay modelos exportados en models/")
    monkeypatch.delenv("MODEL_PATH", raising=False)
    client = TestClient(M.app, raise_server_exceptions=False)
    body = client.get("/api/models").json()
    expected = {p.stem for suffix in (".onnx", ".joblib") for p in MODELS_DIR.glob(f"*{suffix}")
                if p.with_suffix(".json").is_file()}
    entries = {e["id"]: e for e in body["models"]}
    assert set(entries) == expected and body["default"] == "best"
    for e in entries.values():
        assert e["available"] is True, (e["id"], e["error"])
        assert set(e["thresholds"]) == {"comercial", "uvg"}
    if "b1_hog_lr" in entries:
        assert entries["b1_hog_lr"]["kind"] == "classical"
        assert entries["b1_hog_lr"]["label"].startswith("B1")
    if "b2_malexnet" in entries:
        assert entries["b2_malexnet"]["kind"] == "onnx"
    assert "benchmark" not in entries
    assert_private(client.get("/api/models"))


def test_api_model_by_id(client):
    alt = client.get("/api/model", params={"model": "alt"}).json()
    assert alt["id"] == "alt" and alt["is_default"] is False
    assert alt["name"] == "alt_test" and alt["thresholds"] == ALT_THRESHOLDS
    assert alt["model_file"] == "alt.onnx"
    default = client.get("/api/model").json()
    assert default["id"] == "best" and default["is_default"] is True
    assert client.get("/api/model", params={"model": ""}).json()["id"] == "best"
    # un modelo listado pero con tarjeta inválida responde 503
    r = client.get("/api/model", params={"model": "roto"})
    assert r.status_code == 503


BAD_MODEL_IDS = ["nope", "../best", "best.onnx", "best.json", "5", "huerfano", "..", "best/../alt",
                 "a" * 200]


@pytest.mark.parametrize("bad", BAD_MODEL_IDS)
def test_unknown_model_gives_400_everywhere(client, fake_root, feedback_file, tmp_path, bad):
    r = client.get("/api/model", params={"model": bad})
    assert r.status_code == 400, r.text
    assert "Modelo desconocido" in r.json()["detail"]
    assert_private(r, str(tmp_path))
    img = png_bytes(noise_image(150, 150))
    frame = png_bytes(noise_image(1000, 750))
    boxes = json.dumps([{"x": 10, "y": 10, "w": 50, "h": 50}])
    responses = [
        client.post("/api/predict/patch", files=upload("p.png", img), data={"model": bad}),
        client.post("/api/predict/patch", json={"gallery_id": first_id("patch"), "model": bad}),
        client.post("/api/predict/frame", files=upload("f.png", frame), data={"camera": "1", "model": bad}),
        client.post("/api/predict/frame", json={"gallery_id": first_id("frame"), "model": bad}),
        client.post("/api/predict/custom", files=upload("p.png", img), data={"boxes": boxes, "model": bad}),
        client.post("/api/feedback", json={"slot_id": "1", "reported_state": "libre", "model": bad}),
    ]
    for resp in responses:
        assert resp.status_code == 400, (resp.request.url, resp.text)
        assert_private(resp, str(tmp_path))
    assert not feedback_file.exists()


@pytest.mark.parametrize("bad", [5, 0, True, ["best"], {"id": "best"}, 1.5])
def test_non_string_model_gives_400(client, feedback_file, bad):
    boxes = [{"x": 10, "y": 10, "w": 50, "h": 50}]
    for path, body in [("/api/predict/frame", {"gallery_id": first_id("frame"), "model": bad}),
                       ("/api/predict/patch", {"gallery_id": first_id("patch"), "model": bad}),
                       ("/api/predict/custom", {"gallery_id": first_id("frame"), "boxes": boxes, "model": bad}),
                       ("/api/feedback", {"slot_id": "1", "reported_state": "libre", "model": bad})]:
        r = client.post(path, json=body)
        assert r.status_code == 400, (path, r.text)
    assert not feedback_file.exists()


def test_prediction_reports_model_and_its_thresholds(client, model_dir, fake_root):
    img = tinted_image(150, 150, (200, 90, 60), seed=7)
    p = float(expected_p(model_dir / "alt.onnx", [img])[0])
    r = client.post("/api/predict/patch", files=upload("p.png", png_bytes(img)),
                    data={"model": "alt", "scenario": "uvg"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == {"id": "alt", "label": "alt_test", "name": "alt_test",
                             "run_id": "alt_test_s0", "kind": "onnx"}
    assert body["thresholds"] == ALT_THRESHOLDS
    slot = body["slots"][0]
    assert abs(slot["p_busy"] - p) <= 6e-5
    for s in ("comercial", "uvg"):
        assert slot["states"][s] == rule(p, ALT_THRESHOLDS[s]["t_L"], ALT_THRESHOLDS[s]["t_H"])
    # Sin model: el modelo por defecto y sus umbrales
    default = client.post("/api/predict/patch", files=upload("p.png", png_bytes(img))).json()
    assert default["model"]["id"] == "best"
    assert default["thresholds"] == read_card(model_dir / "best.onnx")["thresholds"]
    # Los otros modos también reportan el modelo usado
    frame = client.post("/api/predict/frame", files=upload("f.png", png_bytes(noise_image(1000, 750))),
                        data={"camera": "1", "model": "alt"}).json()
    custom = client.post("/api/predict/custom", files=upload("p.png", png_bytes(img)),
                         data={"boxes": json.dumps([{"x": 0, "y": 0, "w": 150, "h": 150}]), "model": "alt"}).json()
    for res in (frame, custom):
        assert res["model"]["id"] == "alt" and res["thresholds"] == ALT_THRESHOLDS


def test_feedback_records_selected_model(client, feedback_file):
    r = client.post("/api/feedback", json={"slot_id": "3", "reported_state": "libre", "model": "alt"})
    assert r.status_code == 200
    rec = json.loads(feedback_file.read_text(encoding="utf-8").splitlines()[0])
    assert rec["model_id"] == "alt" and rec["model"] == "alt_test"
    client.post("/api/feedback", json={"slot_id": "4", "reported_state": "libre"})
    rec2 = json.loads(feedback_file.read_text(encoding="utf-8").splitlines()[1])
    assert rec2["model_id"] == "best"


# ================================================================ backend clásico

def _synthetic_bundle():
    """Baseline sintético con el mismo formato que model.joblib (pipeline + hog)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    hog = dict(CL.HOG_DEFAULTS)
    imgs = [tinted_image(60 + i, 60, ((i * 41) % 256, (i * 73) % 256, (i * 29) % 256), seed=i) for i in range(16)]
    X = np.stack([CL.hog_image(im.convert("RGB"), **hog) for im in imgs])
    y = np.array([i % 2 for i in range(16)])
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500)).fit(X, y)
    return {"pipeline": pipe, "hog": hog, "C": 0.1}


@pytest.fixture
def classical_model(model_dir) -> Path:
    """``hog_tiny.joblib`` + tarjeta en la carpeta de modelos de la prueba."""
    import joblib

    path = model_dir / "hog_tiny.joblib"
    joblib.dump(_synthetic_bundle(), path)
    card = {"name": "hog_tiny", "run_id": "hog_tiny_s0", "model": "hog_lr", "kind": "classical",
            "temperature": 1.0, "hog": dict(CL.HOG_DEFAULTS),
            "thresholds": {"comercial": {"t_L": 0.3, "t_H": 0.7}, "uvg": {"t_L": 0.45, "t_H": 0.55}}}
    path.with_suffix(".json").write_text(json.dumps(card), encoding="utf-8")
    return path


def test_classical_predictor_matches_hand_formula(classical_model):
    import joblib

    pred = inf.load_predictor(classical_model)
    assert pred.kind == "classical" and pred.temperature == 1.0
    imgs = [tinted_image(70, 50, c, seed=i) for i, c in enumerate([(230, 20, 20), (20, 230, 20), (90, 90, 90)])]
    p, _ = pred.predict(imgs)
    bundle = joblib.load(classical_model)
    z = bundle["pipeline"].decision_function(np.stack([CL.hog_image(im, **bundle["hog"]) for im in imgs]))
    np.testing.assert_allclose(p, 1.0 / (1.0 + np.exp(-z)), atol=1e-6)


def test_classical_model_via_api(client, classical_model):
    entry = next(e for e in client.get("/api/models").json()["models"] if e["id"] == "hog_tiny")
    assert entry["kind"] == "classical" and entry["available"] is True
    info = client.get("/api/model", params={"model": "hog_tiny"}).json()
    assert info["kind"] == "classical" and info["hog"] == dict(CL.HOG_DEFAULTS)
    assert info["temperature"] == 1.0 and info["model_file"] == "hog_tiny.joblib"
    img = tinted_image(150, 150, (120, 140, 60), seed=3)
    r = client.post("/api/predict/patch", files=upload("p.png", png_bytes(img)), data={"model": "hog_tiny"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"]["kind"] == "classical" and body["model"]["id"] == "hog_tiny"
    p = float(inf.load_predictor(classical_model).predict([img])[0][0])
    assert abs(body["slots"][0]["p_busy"] - p) <= 6e-5


def test_hog_image_is_the_single_path(classical_model, tmp_path, monkeypatch):
    """hog_one, el predictor de la app y la verificación de export_classical pasan por hog_image."""
    original = CL.hog_image
    calls: list[tuple[int, int]] = []

    def spy(img, **kw):
        calls.append(img.size)
        return original(img, **kw)

    img = tinted_image(80, 60, (150, 60, 30), seed=5)
    path = tmp_path / "patch.png"
    img.save(path)
    monkeypatch.setattr(CL, "hog_image", spy)

    feats = CL.hog_one(str(path))
    assert calls == [(80, 60)]
    with Image.open(path) as im:
        np.testing.assert_array_equal(feats, original(im))

    pred = inf.load_predictor(classical_model)
    p_app, _ = pred.predict([img, img.copy()])
    assert len(calls) == 3

    exp = _load_export_classical()
    p_export = exp.p_like_app(pred.session, [str(path)])
    assert len(calls) == 4
    np.testing.assert_allclose(p_export, p_app[:1], atol=1e-9)


@pytest.mark.data
@pytest.mark.skipif(not (HAS_B1 and B1_RUN and HAS_VAL), reason="requiere models/b1_hog_lr.*, el run de B1 y val")
def test_b1_app_p_matches_val_preds(client, model_dir):
    """La p que da la app con B1 coincide con val_preds.npz del run en algunos patches de val."""
    from cupo.data import load_split

    shutil.copy(B1_JOBLIB, model_dir / B1_JOBLIB.name)
    shutil.copy(B1_JOBLIB.with_suffix(".json"), model_dir / B1_JOBLIB.with_suffix(".json").name)
    preds = np.load(B1_RUN / "val_preds.npz", allow_pickle=False)
    val = load_split("val").reset_index(drop=True)
    assert len(val) == len(preds["p"])
    idx = np.sort(np.random.default_rng(123).choice(len(val), size=6, replace=False))
    for key in ("camera", "date", "time", "slot_id"):
        assert np.array_equal(val[key].to_numpy(str)[idx], preds[key][idx].astype(str)), key

    pred = inf.load_predictor(model_dir / B1_JOBLIB.name)
    imgs = []
    for i in idx:
        with Image.open(val["path"].iloc[i]) as im:
            imgs.append(im.convert("RGB"))
    p_direct, _ = pred.predict(imgs)
    np.testing.assert_allclose(p_direct, preds["p"][idx], atol=1e-4)

    for i in idx:
        data = Path(val["path"].iloc[i]).read_bytes()
        r = client.post("/api/predict/patch", files=upload("v.jpg", data, "image/jpeg"),
                        data={"model": "b1_hog_lr"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model"]["id"] == "b1_hog_lr" and body["model"]["kind"] == "classical"
        assert abs(body["slots"][0]["p_busy"] - float(preds["p"][i])) <= 1e-4


# ================================================================ export_classical

def _load_export_classical():
    spec = importlib.util.spec_from_file_location("export_classical", REPO / "scripts" / "export_classical.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["export_classical"] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop("export_classical", None)
        raise
    return mod


def _minimal_run_copy(dest: Path, with_val_preds: bool = False) -> Path:
    """Copia mínima del run de B1: config.yaml + model.joblib + thresholds.json (+ val_preds.npz)."""
    dest.mkdir(parents=True)
    names = ["config.yaml", "model.joblib", "thresholds.json"]
    if with_val_preds:
        names += ["val_preds.npz", "metrics_val.json", "value_val.json"]
    for name in names:
        if (B1_RUN / name).is_file():
            shutil.copy(B1_RUN / name, dest / name)
    return dest


def _fake_run(dest: Path) -> Path:
    from export_helpers import fake_classical_run

    return fake_classical_run(dest)


def _written(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.rglob("*")) if folder.exists() else []


@pytest.mark.parametrize("source", ["sintetico", "b1"])
def test_export_classical_without_val_preds_writes_nothing(tmp_path, capsys, source):
    if source == "b1" and B1_RUN is None:
        pytest.skip("no está el run de B1 en runs/")
    run_dir = _fake_run(tmp_path / "run") if source == "sintetico" else _minimal_run_copy(tmp_path / "run")
    assert not (run_dir / "val_preds.npz").exists()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rc = _load_export_classical().main(["--run", str(run_dir), "--out", str(out_dir / "b1_hog_lr.joblib")])
    assert rc == 2
    assert "val_preds.npz" in capsys.readouterr().err
    assert _written(out_dir) == []
    # Sin carpeta de salida previa tampoco se crea nada
    rc = _load_export_classical().main(["--run", str(run_dir), "--out", str(tmp_path / "nueva" / "m.joblib")])
    assert rc == 2 and not (tmp_path / "nueva").exists()


def test_export_classical_rejects_bad_out_and_torch_run(tmp_path, capsys):
    run_dir = _fake_run(tmp_path / "run")
    out_dir = tmp_path / "out"
    exp = _load_export_classical()
    assert exp.main(["--run", str(run_dir), "--out", str(out_dir / "m.onnx")]) == 1
    torch_run = tmp_path / "torch_run"
    torch_run.mkdir()
    (torch_run / "config.yaml").write_text("name: x\n", encoding="utf-8")
    (torch_run / "best.pt").write_bytes(b"")
    assert exp.main(["--run", str(torch_run), "--out", str(out_dir / "m.joblib")]) == 1
    assert exp.main(["--run", str(tmp_path / "no_existe"), "--out", str(out_dir / "m.joblib")]) == 1
    assert _written(out_dir) == []
    capsys.readouterr()


@pytest.mark.data
@pytest.mark.skipif(not (B1_RUN and HAS_VAL), reason="requiere el run de B1 y el split de val")
def test_export_classical_mismatch_writes_nothing(tmp_path, capsys, monkeypatch):
    """val_preds.npz alterado: la verificación falla (código 2) y no se escribe nada; solo lee val."""
    import cupo.data
    from run_helpers import SplitRecorder

    run_dir = _minimal_run_copy(tmp_path / "run", with_val_preds=True)
    preds = dict(np.load(run_dir / "val_preds.npz", allow_pickle=False))
    preds["p"] = np.clip(1.0 - preds["p"], 0.0, 1.0)
    np.savez(run_dir / "val_preds.npz", **preds)
    rec = SplitRecorder(cupo.data.load_split)
    monkeypatch.setattr(cupo.data, "load_split", rec)
    out_dir = tmp_path / "out"
    rc = _load_export_classical().main(["--run", str(run_dir), "--out", str(out_dir / "b1.joblib"),
                                        "--n-check", "4"])
    assert rc == 2
    assert _written(out_dir) == []
    assert {name for name, _ in rec.calls} == {"val"}
    capsys.readouterr()


@pytest.mark.data
@pytest.mark.skipif(not (B1_RUN and HAS_VAL), reason="requiere el run de B1 y el split de val")
def test_export_classical_roundtrip(tmp_path, capsys, monkeypatch):
    import cupo.data
    from run_helpers import SplitRecorder

    run_dir = _minimal_run_copy(tmp_path / "b1_copia", with_val_preds=True)
    rec = SplitRecorder(cupo.data.load_split)
    monkeypatch.setattr(cupo.data, "load_split", rec)
    out = tmp_path / "out" / "b1_hog_lr.joblib"
    rc = _load_export_classical().main(["--run", str(run_dir), "--out", str(out), "--n-check", "6"])
    capsys.readouterr()
    assert rc == 0
    assert {name for name, _ in rec.calls} == {"val"}
    assert out.read_bytes() == (run_dir / "model.joblib").read_bytes()
    card_text = out.with_suffix(".json").read_text(encoding="utf-8")
    assert leaks(card_text, str(tmp_path)) is False
    card = json.loads(card_text)
    thr = json.loads((run_dir / "thresholds.json").read_text(encoding="utf-8"))
    assert card["kind"] == "classical" and card["run_id"] == "b1_copia"
    assert card["model_file"] == "b1_hog_lr.joblib" and card["temperature"] == 1.0
    assert card["check"]["ok"] is True and card["check"]["n"] == 6
    for s in ("comercial", "uvg"):
        assert card["thresholds"][s] == {"t_L": float(thr[s]["t_L"]), "t_H": float(thr[s]["t_H"])}
    pred = inf.load_predictor(out)
    assert pred.kind == "classical" and pred.thresholds == card["thresholds"]


# ================================================================ fotos propias

def tiles(n: int = 3) -> tuple[Image.Image, list[dict]]:
    """Imagen 640x480 con n bloques de colores y una caja por bloque."""
    colors = [(230, 30, 30), (30, 220, 40), (120, 120, 200), (200, 200, 40)]
    img = noise_image(640, 480, seed=11)
    boxes = []
    for i in range(n):
        x, y = 40 + i * 150, 150
        img.paste(tinted_image(120, 120, colors[i % len(colors)], seed=i), (x, y))
        boxes.append({"slot_id": f"e{i + 1}", "x": x, "y": y, "w": 120, "h": 120})
    return img, boxes


def write_own(folder: Path, name: str, img: Image.Image, boxes=None, raw: str | None = None) -> Path:
    path = folder / name
    if path.suffix.lower() == ".png":
        img.save(path, "PNG")
    else:
        img.save(path, "JPEG", quality=92)
    if raw is not None:
        path.with_suffix(".json").write_text(raw, encoding="utf-8")
    elif boxes is not None:
        path.with_suffix(".json").write_text(json.dumps({"image": name, "boxes": boxes}), encoding="utf-8")
    return path


def own_entry(client, gid: str) -> dict | None:
    return next((it for it in client.get("/api/gallery").json()["items"] if it["id"] == gid), None)


def test_own_photo_with_boxes_is_listed_and_analyzed(client, model_dir, own_dir, tmp_path, monkeypatch):
    img, boxes = tiles()
    path = write_own(own_dir, "parqueo_uvg.png", img, boxes)
    r = client.get("/api/gallery")
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 77
    it = own_entry(client, "own-parqueo_uvg")
    assert it is not None
    assert it["kind"] == "own" and it["subset"] == "UVG" and it["filename"] == "parqueo_uvg.png"
    assert it["has_boxes"] is True and it["n_boxes"] == 3 and it["available"] is True
    assert it["image_url"] == "/api/gallery/own-parqueo_uvg/image"
    assert not any(k.startswith("_") for k in it)
    assert_private(r, str(tmp_path))

    im = client.get("/api/gallery/own-parqueo_uvg/image")
    assert im.status_code == 200 and im.headers["content-type"] == "image/png"
    assert im.content == path.read_bytes()

    monkeypatch.setenv("SHOW_GROUND_TRUTH", "1")  # las fotos propias no tienen etiquetas
    r = client.post("/api/predict/frame", json={"gallery_id": "own-parqueo_uvg", "model": "alt"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "frame" and body["model"]["id"] == "alt"
    assert body["source"]["type"] == "gallery" and body["source"]["id"] == "own-parqueo_uvg"
    assert body["source"]["kind"] == "own"
    assert [s["slot_id"] for s in body["slots"]] == ["e1", "e2", "e3"]
    assert [s["box"] for s in body["slots"]] == [{k: b[k] for k in ("x", "y", "w", "h")} for b in boxes]
    crops = [img.crop((b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"])) for b in boxes]
    np.testing.assert_allclose([s["p_busy"] for s in body["slots"]],
                               expected_p(model_dir / "alt.onnx", crops), atol=6e-5)
    assert has_key(body, "ground_truth") is False
    assert_private(r, str(tmp_path))

    # En el modo dibujar se puede usar con otras cajas; en el modo patch no
    r = client.post("/api/predict/custom", json={"gallery_id": "own-parqueo_uvg",
                                                 "boxes": [{"x": 0, "y": 0, "w": 100, "h": 100}]})
    assert r.status_code == 200 and len(r.json()["slots"]) == 1
    assert client.post("/api/predict/patch", json={"gallery_id": "own-parqueo_uvg"}).status_code == 400


def test_own_photo_bare_list_json_and_jpeg(client, own_dir):
    img, boxes = tiles(2)
    write_own(own_dir, "Foto 2.jpeg", img, raw=json.dumps(boxes))
    it = own_entry(client, "own-Foto 2")
    assert it is not None and it["has_boxes"] is True and it["n_boxes"] == 2
    im = client.get("/api/gallery/own-Foto 2/image")
    assert im.status_code == 200 and im.headers["content-type"] == "image/jpeg"
    r = client.post("/api/predict/frame", json={"gallery_id": "own-Foto 2"})
    assert r.status_code == 200 and [s["slot_id"] for s in r.json()["slots"]] == ["e1", "e2"]


def test_own_photo_without_boxes_is_drawable_only(client, own_dir, tmp_path):
    img, _ = tiles()
    write_own(own_dir, "sin_cajas.jpg", img)
    it = own_entry(client, "own-sin_cajas")
    assert it is not None and it["has_boxes"] is False and it["n_boxes"] == 0
    assert it["available"] is True
    assert client.get("/api/gallery/own-sin_cajas/image").status_code == 200
    r = client.post("/api/predict/frame", json={"gallery_id": "own-sin_cajas"})
    assert r.status_code == 400 and "sin_cajas.json" in r.json()["detail"]
    assert_private(r, str(tmp_path))
    r = client.post("/api/predict/custom", json={"gallery_id": "own-sin_cajas",
                                                 "boxes": [{"x": 10, "y": 10, "w": 80, "h": 80}]})
    assert r.status_code == 200


def test_own_photos_are_read_on_each_request(client, own_dir):
    assert own_entry(client, "own-nueva") is None
    img, boxes = tiles(1)
    write_own(own_dir, "nueva.png", img, boxes)
    assert own_entry(client, "own-nueva") is not None
    (own_dir / "nueva.png").unlink()
    assert own_entry(client, "own-nueva") is None
    assert client.post("/api/predict/frame", json={"gallery_id": "own-nueva"}).status_code == 404


def test_own_gallery_missing_folder(client, tmp_path, monkeypatch):
    monkeypatch.setenv("OWN_GALLERY_DIR", str(tmp_path / "no_existe"))
    r = client.get("/api/gallery")
    assert r.status_code == 200
    assert not any(it["kind"] == "own" for it in r.json()["items"])


def test_own_gallery_ignores_bad_names_and_other_files(client, own_dir):
    img, boxes = tiles(1)
    write_own(own_dir, "ok.png", img, boxes)
    write_own(own_dir, "a..b.png", img, boxes)
    write_own(own_dir, ".oculta.png", img, boxes)
    (own_dir / "notas.txt").write_text("hola", encoding="utf-8")
    (own_dir / "sub").mkdir()
    write_own(own_dir / "sub", "dentro.png", img, boxes)
    ids = sorted(it["id"] for it in client.get("/api/gallery").json()["items"] if it["kind"] == "own")
    assert ids == ["own-ok"]


@pytest.mark.parametrize("gid", ["own-../secreto", "own-..%2Fsecreto", "own-..%5Csecreto", "own-..\\secreto",
                                 "own-ok.png", "own-ok.json", "own-OK.PNG", "own-sub/dentro", "own-",
                                 "own-../galeria_propia/ok", "own-%2E%2E%2Fsecreto", "own-ok%00"])
def test_own_gallery_path_traversal_gives_404(client, own_dir, tmp_path, gid):
    img, boxes = tiles(1)
    write_own(own_dir, "ok.png", img, boxes)
    write_own(tmp_path, "secreto.png", img, boxes)  # fuera de la carpeta
    (own_dir / "sub").mkdir()
    write_own(own_dir / "sub", "dentro.png", img, boxes)
    r = client.get(f"/api/gallery/{gid}/image")
    assert r.status_code == 404, (gid, r.status_code)
    assert_private(r, str(tmp_path))
    for path, body in [("/api/predict/frame", {"gallery_id": gid}),
                       ("/api/predict/custom", {"gallery_id": gid, "boxes": [{"x": 0, "y": 0, "w": 50, "h": 50}]})]:
        r = client.post(path, json=body)
        assert r.status_code == 404, (gid, path, r.status_code)
        assert_private(r, str(tmp_path))
    r = client.post("/api/feedback", json={"slot_id": "1", "reported_state": "libre", "source": gid})
    assert r.status_code == 400


def test_own_photo_feedback_source(client, own_dir, feedback_file):
    img, boxes = tiles(1)
    write_own(own_dir, "ok.png", img, boxes)
    r = client.post("/api/feedback", json={"slot_id": "e1", "reported_state": "ocupado", "source": "own-ok"})
    assert r.status_code == 200
    assert json.loads(feedback_file.read_text(encoding="utf-8"))["source"] == "own-ok"


def _box(i: int) -> dict:
    return {"x": (i % 20) * 30, "y": (i // 20) * 30, "w": 25, "h": 25}


@pytest.mark.parametrize("name,raw,listed_boxes,detail", [
    ("demasiadas", json.dumps({"boxes": [_box(i) for i in range(301)]}), True, "Demasiadas cajas"),
    ("no_json", "{no json", False, "no tiene cajas"),
    ("no_lista", json.dumps({"boxes": {"x": 1}}), False, "no tiene cajas"),
    ("vacia", json.dumps({"boxes": []}), False, "no tiene cajas"),
    ("texto", json.dumps({"boxes": [{"x": "a", "y": 0, "w": 10, "h": 10}]}), True, "no es válida"),
    ("nan", '{"boxes": [{"x": NaN, "y": 0, "w": 10, "h": 10}]}', True, "finitos"),
    ("fuera", json.dumps({"boxes": [{"x": 5000, "y": 0, "w": 10, "h": 10}]}), True, "fuera de la imagen"),
    ("enorme", json.dumps({"boxes": [_box(1)], "pad": "x" * (300 * 1024)}), False, "no tiene cajas"),
    ("numero", "3", False, "no tiene cajas"),
])
def test_own_boxes_json_invalid_is_rejected(client, own_dir, tmp_path, name, raw, listed_boxes, detail):
    img, _ = tiles(1)
    write_own(own_dir, f"{name}.png", img, raw=raw)
    it = own_entry(client, f"own-{name}")
    assert it is not None and it["has_boxes"] is listed_boxes
    r = client.post("/api/predict/frame", json={"gallery_id": f"own-{name}"})
    assert r.status_code == 400, r.text
    assert detail in r.json()["detail"]
    assert_private(r, str(tmp_path))


def test_own_boxes_at_limit_are_accepted(client, own_dir):
    img, _ = tiles(1)
    write_own(own_dir, "limite.png", img, raw=json.dumps({"boxes": [_box(i) for i in range(300)]}))
    r = client.post("/api/predict/frame", json={"gallery_id": "own-limite"})
    assert r.status_code == 200 and len(r.json()["slots"]) == 300


# ================================================================ mosaicos

def _mosaic_item(gid: str) -> dict:
    return next(it for it in gallery_json_items() if it["id"] == gid)


@pytest.fixture
def mosaic_root(tmp_path, monkeypatch) -> tuple[Path, list[Image.Image]]:
    """DATA_ROOT sintético con las 12 celdas de mosaic-a; la última no mide 150x150."""
    root = tmp_path / "mosaic_data"
    tiles_out: list[Image.Image] = []
    for i, cell in enumerate(_mosaic_item("mosaic-a")["cells"]):
        color = ((i * 61) % 256, (i * 113 + 40) % 256, (i * 29 + 90) % 256)
        size = (120, 100) if i == 11 else (150, 150)
        target = root / cell["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        tinted_image(*size, color, seed=i).save(target, "JPEG", quality=95)
        with Image.open(target) as im:
            tile = im.convert("RGB")
        if tile.size != (150, 150):
            tile = tile.resize((150, 150), Image.Resampling.BILINEAR)
        tiles_out.append(tile)
    monkeypatch.setenv("DATA_ROOT", str(root))
    return root, tiles_out


def test_mosaic_image_is_composed(client, mosaic_root):
    _, tiles_ = mosaic_root
    r = client.get("/api/gallery/mosaic-a/image")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    assert img.size == (4 * 150, 3 * 150)
    arr = np.asarray(img, dtype=np.float64)
    for i, tile in enumerate(tiles_):
        x, y = (i % 4) * 150, (i // 4) * 150
        got = arr[y + 20:y + 130, x + 20:x + 130].mean(axis=(0, 1))
        want = np.asarray(tile, dtype=np.float64)[20:130, 20:130].mean(axis=(0, 1))
        assert np.abs(got - want).max() < 4.0, i
    entry = own_entry(client, "mosaic-a")
    assert entry["available"] is True and entry["n_cells"] == 12


def test_mosaic_prediction_and_ground_truth_flag(client, model_dir, mosaic_root, monkeypatch, tmp_path):
    _, tiles_ = mosaic_root
    item = _mosaic_item("mosaic-a")
    r = client.post("/api/predict/frame", json={"gallery_id": "mosaic-a"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image"] == {"width": 600, "height": 450}
    assert [s["slot_id"] for s in body["slots"]] == [str(i) for i in range(1, 13)]
    for i, s in enumerate(body["slots"]):
        assert s["box"] == {"x": (i % 4) * 150, "y": (i // 4) * 150, "w": 150, "h": 150}
    np.testing.assert_allclose([s["p_busy"] for s in body["slots"]],
                               expected_p(model_dir / "best.onnx", tiles_), atol=6e-5)
    assert body["source"]["kind"] == "mosaic"
    assert has_key(body, "ground_truth") is False
    assert_private(r, str(tmp_path))

    for value in ("0", "true", ""):
        monkeypatch.setenv("SHOW_GROUND_TRUTH", value)
        assert has_key(client.post("/api/predict/frame", json={"gallery_id": "mosaic-a"}).json(),
                       "ground_truth") is False

    monkeypatch.setenv("SHOW_GROUND_TRUTH", "1")
    body = client.post("/api/predict/frame", json={"gallery_id": "mosaic-a"}).json()
    expected = [1 if "/busy/" in c["path"] else 0 for c in item["cells"]]
    assert [s["ground_truth"] for s in body["slots"]] == expected
    assert sum(expected) == 6
    assert body["ground_truth"]["available"] is True
    for s in ("comercial", "uvg"):
        cmp_ = body["ground_truth"]["by_scenario"][s]
        assert cmp_["aciertos"] + cmp_["errores"] + cmp_["por_verificar"] == 12


def test_mosaic_other_modes(client, mosaic_root):
    assert client.post("/api/predict/patch", json={"gallery_id": "mosaic-a"}).status_code == 400
    r = client.post("/api/predict/custom", json={"gallery_id": "mosaic-a",
                                                 "boxes": [{"x": 150, "y": 0, "w": 150, "h": 150}]})
    assert r.status_code == 200 and r.json()["image"] == {"width": 600, "height": 450}


def test_mosaic_missing_cell_gives_404(client, mosaic_root, tmp_path):
    root, _ = mosaic_root
    (root / _mosaic_item("mosaic-a")["cells"][5]["path"]).unlink()
    assert own_entry(client, "mosaic-a")["available"] is False
    r = client.get("/api/gallery/mosaic-a/image")
    assert r.status_code == 404 and "DATA_ROOT" in r.json()["detail"]
    assert_private(r, str(tmp_path))
    r = client.post("/api/predict/frame", json={"gallery_id": "mosaic-a"})
    assert r.status_code == 404
    assert_private(r, str(tmp_path))


@pytest.mark.data
@pytest.mark.skipif(not HAS_DATA, reason="dataset no disponible en DATA_ROOT")
def test_real_mosaics(client, monkeypatch):
    for gid in ("mosaic-a", "mosaic-a2", "mosaic-b", "mosaic-b2"):
        r = client.get(f"/api/gallery/{gid}/image")
        assert r.status_code == 200
        assert Image.open(io.BytesIO(r.content)).size == (600, 450)
        monkeypatch.delenv("SHOW_GROUND_TRUTH", raising=False)
        body = client.post("/api/predict/frame", json={"gallery_id": gid}).json()
        assert len(body["slots"]) == 12 and has_key(body, "ground_truth") is False
        monkeypatch.setenv("SHOW_GROUND_TRUTH", "1")
        body = client.post("/api/predict/frame", json={"gallery_id": gid}).json()
        assert sorted(s["ground_truth"] for s in body["slots"]) == [0] * 6 + [1] * 6


# ================================================================ .gitignore

def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def _git_available() -> bool:
    try:
        return _git("rev-parse", "--is-inside-work-tree").stdout.strip() == "true"
    except OSError:
        return False


@pytest.mark.skipif(not _git_available(), reason="sin git o fuera del repositorio")
def test_own_gallery_is_gitignored_except_readme():
    for rel in ("app/galeria_propia/foto.jpg", "app/galeria_propia/foto.json", "app/galeria_propia/Foto_3.jpeg",
                "app/galeria_propia/x.png", "app/galeria_propia/sub/y.jpeg", "app/galeria_propia/notas.txt"):
        assert _git("check-ignore", "--no-index", "-q", rel).returncode == 0, rel
    assert _git("check-ignore", "--no-index", "-q", "app/galeria_propia/LEEME.md").returncode == 1
    status = _git("status", "--porcelain", "--untracked-files=all", "--", "app/galeria_propia")
    assert status.returncode == 0
    names = [line[3:].strip().strip('"') for line in status.stdout.splitlines() if line.strip()]
    assert all(n == "app/galeria_propia/LEEME.md" for n in names), names
    assert (REPO / "app" / "galeria_propia" / "LEEME.md").is_file()


# ================================================================ tiempo del conductor


def test_time_benefit_matches_iterations_table(tmp_path, monkeypatch):
    import app.main as main_mod

    csv_path = tmp_path / "iteraciones.csv"
    csv_path.write_text("name,val_hours_uvg\nb1_hog_lr,7881\nmnv3_ft_cost,7978\nsin_horas,\n", encoding="utf-8")
    monkeypatch.setattr(main_mod, "ITERATIONS_CSV", csv_path)
    tb = main_mod.time_benefit("mnv3_ft_cost")
    arrivals = 900 * 90  # configs/value/uvg.yaml
    assert tb["escenario"] == "uvg" and tb["t_base_min"] == 7.0
    assert tb["minutos_ahorrados"] == round(7978 * 60 / arrivals, 2)
    assert tb["minutos_con_sistema"] == round(7.0 - 7978 * 60 / arrivals, 2)
    assert tb["horas_semestre"] == 7978
    assert tb["segundos_vs_b1"] == round((7978 - 7881) * 3600 / arrivals, 1)
    assert "segundos_vs_b1" not in main_mod.time_benefit("b1_hog_lr")
    assert main_mod.time_benefit("sin_horas") is None
    assert main_mod.time_benefit("desconocido") is None
    assert main_mod.time_benefit(None) is None
    monkeypatch.setattr(main_mod, "ITERATIONS_CSV", tmp_path / "no_existe.csv")
    assert main_mod.time_benefit("mnv3_ft_cost") is None


def test_api_model_includes_time_benefit_for_real_best_model():
    import json as _json

    card_path = paths.repo_root() / "models" / "best.json"
    if not card_path.is_file():
        pytest.skip("sin models/best.json")
    from fastapi.testclient import TestClient

    import app.main as main_mod

    name = _json.loads(card_path.read_text(encoding="utf-8")).get("name")
    body = TestClient(main_mod.app).get("/api/model", params={"model": "best"}).json()
    assert "time_benefit" in body
    assert body["time_benefit"] == main_mod.time_benefit(name)
