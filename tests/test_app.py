"""Pruebas de la app: inferencia ONNX, endpoints, casos límite, galería y privacidad.

Todo corre en proceso con ``TestClient`` y en CPU. El modelo es un ONNX diminuto creado en
``tmp_path`` (ver ``app_helpers.make_model_dir``) y ``MODEL_PATH`` / ``FEEDBACK_PATH`` apuntan
ahí: las pruebas nunca escriben en ``models/`` ni en ``app/feedback.jsonl``.

Las pruebas marcadas con ``data`` usan el dataset real bajo DATA_ROOT y se saltan si no está.
Con las etiquetas reales (SHOW_GROUND_TRUTH) solo se comprueba si la clave aparece o no.
"""

from __future__ import annotations

import importlib.util
import io
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from app_helpers import (
    REPO,
    TEST_ONNX_ENV,
    TINY_B,
    TINY_THRESHOLDS,
    TINY_W,
    assert_private,
    fake_data_root,
    has_key,
    huge_png_bytes,
    leaks,
    make_model_dir,
    noise_image,
    png_bytes,
    read_card,
    tinted_image,
    write_card,
)
from fastapi.testclient import TestClient

from app import inference as inf
from app import main as M
from cupo import frames, paths
from cupo.transforms import preprocess_numpy

GALLERY_JSON = REPO / "app" / "gallery.json"
REPO_FEEDBACK = REPO / "app" / "feedback.jsonl"
USING_TINY = not os.environ.get(TEST_ONNX_ENV)


def _gallery() -> list[dict]:
    return json.loads(GALLERY_JSON.read_text(encoding="utf-8"))["items"]


def _has_data() -> bool:
    try:
        if not (paths.split_file("train").is_file() and paths.split_file("val").is_file()
                and paths.split_file("test").is_file() and paths.metadata_csv().is_file()
                and paths.full_images_dir().is_dir() and paths.cnrpark_dir().is_dir()):
            return False
        return all((paths.data_root() / it["path"]).is_file() for it in _gallery())
    except Exception:
        return False


HAS_DATA = _has_data()


def real_data(fn):
    fn = pytest.mark.skipif(not HAS_DATA, reason="dataset no disponible en DATA_ROOT")(fn)
    return pytest.mark.data(fn)


def gallery_ids(kind: str) -> list[str]:
    return [it["id"] for it in _gallery() if it["kind"] == kind]


# ================================================================ fixtures

@pytest.fixture(autouse=True)
def _repo_feedback_untouched():
    """Ninguna prueba escribe en app/feedback.jsonl."""
    before = REPO_FEEDBACK.stat().st_mtime_ns if REPO_FEEDBACK.exists() else None
    yield
    after = REPO_FEEDBACK.stat().st_mtime_ns if REPO_FEEDBACK.exists() else None
    assert before == after, "se escribió en app/feedback.jsonl"


@pytest.fixture
def model_onnx(tmp_path, monkeypatch) -> Path:
    onnx_path = make_model_dir(tmp_path / "model")
    monkeypatch.setenv("MODEL_PATH", str(onnx_path))
    return onnx_path


@pytest.fixture
def feedback_file(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "fb" / "feedback.jsonl"
    monkeypatch.setenv("FEEDBACK_PATH", str(path))
    return path


@pytest.fixture
def client(model_onnx, feedback_file, monkeypatch) -> TestClient:
    monkeypatch.delenv("SHOW_GROUND_TRUTH", raising=False)
    return TestClient(M.app, raise_server_exceptions=False)


@pytest.fixture
def fake_root(tmp_path, monkeypatch) -> Path:
    """DATA_ROOT sintético con las cajas de las cámaras 1 y 5."""
    root = fake_data_root(tmp_path / "data", cameras=(1, 5))
    monkeypatch.setenv("DATA_ROOT", str(root))
    return root


@pytest.fixture
def empty_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "vacio"
    root.mkdir()
    monkeypatch.setenv("DATA_ROOT", str(root))
    return root


def upload(name: str, data: bytes, ctype: str = "image/png") -> dict:
    return {"file": (name, data, ctype)}


def tiles_image() -> tuple[Image.Image, list[dict]]:
    """Imagen 600x400 con 24 bloques de colores distintos y una caja por bloque."""
    colors = [(r, g, b) for r in (20, 120, 230) for g in (20, 230) for b in (20, 120, 200, 250)]
    img = Image.new("RGB", (600, 400))
    boxes = []
    for i, color in enumerate(colors):
        x, y = (i % 6) * 100, (i // 6) * 100
        img.paste(tinted_image(100, 100, color, seed=i), (x, y))
        boxes.append({"slot_id": f"t{i}", "x": x + 5, "y": y + 5, "w": 90, "h": 90})
    return img, boxes


def expected_p(onnx_path: Path, imgs: list[Image.Image]) -> np.ndarray:
    """P(ocupado) calculada en la prueba: ONNX Runtime + preprocess_numpy + softmax(z / T)."""
    import onnxruntime as ort

    card = read_card(onnx_path)
    size = int(card.get("input_size", 224))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    batch = np.concatenate([preprocess_numpy(im, size) for im in imgs], axis=0)
    logits = sess.run(None, {sess.get_inputs()[0].name: batch})[0].astype(np.float64)
    z = logits / float(card.get("temperature", 1.0))
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e[:, 1] / e.sum(axis=1)


def rule(p: float, t_low: float, t_high: float) -> str:
    return "libre" if p < t_low else ("ocupado" if p >= t_high else "verificar")


# ================================================================ inferencia

def test_softmax_busy_by_hand():
    ln3, ln2 = math.log(3.0), math.log(2.0)
    logits = np.array([[0.0, ln3], [ln3, 0.0], [2.0, 2.0]])
    np.testing.assert_allclose(inf.softmax_busy(logits, 1.0), [0.75, 0.25, 0.5], atol=1e-12)
    s3 = math.sqrt(3.0)  # T = 2: e^{ln3/2} = sqrt(3)
    np.testing.assert_allclose(inf.softmax_busy(logits, 2.0), [s3 / (1 + s3), 1 / (1 + s3), 0.5], atol=1e-12)
    # T = 0.5: e^{2 ln2} = 4 -> 4/5
    np.testing.assert_allclose(inf.softmax_busy(np.array([[0.0, ln2]]), 0.5), [0.8], atol=1e-12)


def test_softmax_busy_is_stable_with_large_logits():
    p = inf.softmax_busy(np.array([[1000.0, 1000.0 + math.log(3.0)], [-1000.0, 1000.0]]), 1.0)
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p, [0.75, 1.0], atol=1e-12)


@pytest.mark.parametrize("t_low,t_high,cases", [
    (0.3, 0.7, [(0.0, "libre"), (0.29, "libre"), (0.3, "verificar"), (0.69, "verificar"),
                (0.7, "ocupado"), (1.0, "ocupado")]),
    (0.5, 0.5, [(0.0, "libre"), (0.4999, "libre"), (0.5, "ocupado"), (0.9, "ocupado")]),
    (0.1, 0.2, [(0.05, "libre"), (0.1, "verificar"), (0.15, "verificar"), (0.2, "ocupado"),
                (0.5, "ocupado")]),
    (0.0, 1.0, [(0.0, "verificar"), (0.5, "verificar"), (0.9999, "verificar"), (1.0, "ocupado")]),
])
def test_decide_rule(t_low, t_high, cases):
    for p, state in cases:
        assert inf.decide(p, t_low, t_high) == state, (p, t_low, t_high)


def test_equal_thresholds_never_verify():
    ps = np.linspace(0, 1, 101)
    assert all(inf.decide(float(p), 0.42, 0.42) != "verificar" for p in ps)


@pytest.mark.skipif(not USING_TINY, reason="solo con el modelo diminuto (pesos conocidos)")
def test_tiny_model_matches_hand_formula(model_onnx):
    """El modelo diminuto es promedio por canal + lineal: p se puede calcular con numpy."""
    pred = inf.load_predictor(model_onnx)
    imgs = [tinted_image(50, 40, c, seed=i) for i, c in enumerate([(230, 20, 20), (20, 230, 20), (128, 128, 128)])]
    p, _ = pred.predict(imgs)
    pooled = np.concatenate([preprocess_numpy(im, pred.input_size) for im in imgs]).mean(axis=(2, 3))
    logits = pooled.astype(np.float64) @ TINY_W.T.astype(np.float64) + TINY_B
    z = logits / pred.temperature
    hand = np.exp(z[:, 1]) / np.exp(z).sum(axis=1)
    np.testing.assert_allclose(p, hand, atol=1e-5)
    assert p[0] > 0.9 and p[1] < 0.1  # rojo -> ocupado, verde -> libre con estos pesos


def test_load_predictor_reads_card(model_onnx):
    pred = inf.load_predictor(model_onnx)
    card = read_card(model_onnx)
    assert pred.temperature == pytest.approx(float(card["temperature"]))
    assert pred.input_size == int(card["input_size"])
    for s in ("comercial", "uvg"):
        assert pred.thresholds[s] == {"t_L": float(card["thresholds"][s]["t_L"]),
                                      "t_H": float(card["thresholds"][s]["t_H"])}


def test_predict_in_blocks_of_64_equals_single_batch(model_onnx):
    pred = inf.load_predictor(model_onnx)
    crops = [tinted_image(30 + i % 7, 30 + i % 5, ((i * 37) % 256, (i * 91) % 256, (i * 53) % 256), seed=i)
             for i in range(130)]
    sizes: list[int] = []
    real = pred.session

    class Spy:
        def run(self, out, feeds):
            sizes.append(next(iter(feeds.values())).shape[0])
            return real.run(out, feeds)

    pred.session = Spy()
    p_blocks, model_ms = pred.predict(crops)
    assert sizes == [64, 64, 2]
    assert model_ms >= 0.0
    full = np.concatenate([preprocess_numpy(c, pred.input_size) for c in crops], axis=0)
    logits = real.run(None, {pred.input_name: full})[0]
    np.testing.assert_allclose(p_blocks, inf.softmax_busy(np.asarray(logits), pred.temperature), atol=1e-6)
    np.testing.assert_allclose(p_blocks, expected_p(model_onnx, crops), atol=1e-6)


def test_predict_without_crops(model_onnx):
    p, ms = inf.load_predictor(model_onnx).predict([])
    assert p.shape == (0,) and ms == 0.0


def test_image_quality_warnings():
    dark = inf.image_quality(Image.new("RGB", (800, 600), (10, 10, 10)))
    assert dark["dark"] and not dark["blurry"] and dark["warnings"]
    flat = inf.image_quality(Image.new("RGB", (800, 600), (128, 128, 128)))
    assert flat["blurry"] and not flat["dark"] and flat["warnings"]
    sharp = inf.image_quality(noise_image(800, 600))
    assert not sharp["dark"] and not sharp["blurry"] and sharp["warnings"] == []


# ================================================================ endpoints

def test_index_and_static(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "Cupo Seguro" in r.text
    for name in ("app.js", "styles.css"):
        r = client.get(f"/static/{name}")
        assert r.status_code == 200 and len(r.content) > 0
    assert client.get("/static/no-existe.js").status_code == 404


def test_api_model(client, model_onnx, tmp_path):
    r = client.get("/api/model")
    assert r.status_code == 200
    body = r.json()
    card = read_card(model_onnx)
    assert body["name"] == card["name"] and body["run_id"] == card["run_id"]
    assert body["onnx_file"] == "best.onnx"
    assert body["temperature"] == pytest.approx(float(card["temperature"]))
    assert set(body["thresholds"]) == {"comercial", "uvg"}
    for s in ("comercial", "uvg"):
        assert body["thresholds"][s]["t_L"] == pytest.approx(card["thresholds"][s]["t_L"])
        assert body["thresholds"][s]["t_H"] == pytest.approx(card["thresholds"][s]["t_H"])
    if USING_TINY:
        assert body["thresholds"] == TINY_THRESHOLDS
        assert body["benchmark"]["matches_model"] is True
        assert body["benchmark"]["cpu_per_frame_p50_ms"] == 10.0
    assert "rule" in body
    assert_private(r, str(tmp_path))


def test_api_model_benchmark_of_other_model(client, model_onnx):
    bench = model_onnx.parent / "benchmark.json"
    bench.write_text(json.dumps({"model": {"name": "otro", "run_id": "otro_s1"}}), encoding="utf-8")
    assert client.get("/api/model").json()["benchmark"]["matches_model"] is False
    bench.unlink()
    assert client.get("/api/model").json()["benchmark"] is None


def test_api_gallery(client):
    r = client.get("/api/gallery")
    assert r.status_code == 200
    body = r.json()
    items = body["items"]
    assert len(items) == 14
    assert sum(it["kind"] == "frame" for it in items) == 8
    assert sum(it["kind"] == "patch" for it in items) == 6
    assert len({it["id"] for it in items}) == 14
    for it in items:
        assert it["image_url"] == f"/api/gallery/{it['id']}/image"
        assert "path" not in it
        assert isinstance(it["available"], bool)
    assert body["ground_truth_enabled"] is False
    assert_private(r)


def test_api_gallery_without_data_marks_unavailable(client, empty_root):
    r = client.get("/api/gallery")
    assert r.status_code == 200
    assert all(it["available"] is False for it in r.json()["items"])
    assert_private(r, str(empty_root))


def test_gallery_json_paths_are_relative():
    text = GALLERY_JSON.read_text(encoding="utf-8")
    assert leaks(text) is False
    for it in _gallery():
        p = it["path"]
        assert not Path(p).is_absolute() and not p.startswith("/")
        assert ":" not in p and "\\" not in p and ".." not in Path(p).parts


@real_data
def test_gallery_image(client):
    fid, pid = gallery_ids("frame")[0], gallery_ids("patch")[0]
    r = client.get(f"/api/gallery/{fid}/image")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(r.content)).size == frames.FULL_SIZE
    r = client.get(f"/api/gallery/{pid}/image")
    assert r.status_code == 200 and len(r.content) > 0


@real_data
def test_predict_frame_from_gallery(client):
    item = next(it for it in _gallery() if it["kind"] == "frame")
    r = client.post("/api/predict/frame", json={"gallery_id": item["id"], "scenario": "uvg"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "frame" and body["scenario"] == "uvg"
    assert body["source"]["type"] == "gallery" and body["source"]["id"] == item["id"]
    n = len(frames.slot_boxes(item["camera"]))
    assert len(body["slots"]) == n and body["latency_ms"]["n_slots"] == n
    for s in ("comercial", "uvg"):
        summ = body["summary_by_scenario"][s]
        assert summ["libres"] + summ["ocupados"] + summ["por_verificar"] == summ["total"] == n
    th = body["thresholds"]
    for slot in body["slots"]:
        assert set(slot) >= {"slot_id", "box", "p_busy", "state", "states"}
        assert 0.0 <= slot["p_busy"] <= 1.0
        assert slot["state"] == slot["states"]["uvg"]
        for s in ("comercial", "uvg"):
            if min(abs(slot["p_busy"] - th[s]["t_L"]), abs(slot["p_busy"] - th[s]["t_H"])) > 1e-4:
                assert slot["states"][s] == rule(slot["p_busy"], th[s]["t_L"], th[s]["t_H"])


def test_predict_frame_upload_scales_camera_boxes(client, fake_root):
    expected = frames.slot_boxes(1)
    for scale in (1, 2):
        img = noise_image(1000 * scale, 750 * scale, seed=scale)
        r = client.post("/api/predict/frame", files=upload("f.png", png_bytes(img)), data={"camera": "1"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["source"] == {"type": "upload", "filename": "f.png", "camera": 1}
        assert [s["slot_id"] for s in body["slots"]] == list(expected["slot_id"])
        for slot, row in zip(body["slots"], expected.itertuples(index=False)):
            assert slot["box"] == {"x": row.x * scale, "y": row.y * scale, "w": row.w * scale, "h": row.h * scale}
        assert body["warnings"] == []


def test_predict_frame_upload_other_aspect_warns(client, fake_root):
    img = noise_image(640, 360)
    r = client.post("/api/predict/frame", files=upload("f.png", png_bytes(img)), data={"camera": "5"})
    assert r.status_code == 200
    assert any("4:3" in w for w in r.json()["warnings"])


def test_predict_patch_matches_onnx(client, model_onnx):
    img = tinted_image(150, 150, (200, 90, 60), seed=7)
    r = client.post("/api/predict/patch", files=upload("p.png", png_bytes(img)), data={"scenario": "uvg"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "patch" and len(body["slots"]) == 1
    slot = body["slots"][0]
    assert slot["box"] == {"x": 0, "y": 0, "w": 150, "h": 150}
    p = float(expected_p(model_onnx, [img])[0])
    assert abs(slot["p_busy"] - p) <= 6e-5
    card = read_card(model_onnx)
    for s in ("comercial", "uvg"):
        assert slot["states"][s] == rule(p, card["thresholds"][s]["t_L"], card["thresholds"][s]["t_H"])
    assert slot["state"] == slot["states"]["uvg"]
    assert body["source"] == {"type": "upload", "filename": "p.png"}
    assert body["model"]["name"] == card["name"]


def test_predict_patch_large_image_warns(client):
    r = client.post("/api/predict/patch", files=upload("g.png", png_bytes(noise_image(800, 600))))
    assert r.status_code == 200
    assert any("más de un espacio" in w for w in r.json()["warnings"])


def test_predict_custom_upload(client, model_onnx):
    img, boxes = tiles_image()
    boxes = boxes[:3] + [
        {"slot_id": "neg", "x": 395, "y": 195, "w": -90, "h": -90},   # caja dibujada al revés
        {"slot_id": "fuera", "x": 2000, "y": 0, "w": 50, "h": 50},     # fuera de la imagen
        {"x": 590, "y": 390, "w": 50, "h": 50},                        # se recorta al borde
    ]
    r = client.post("/api/predict/custom", files=upload("t.png", png_bytes(img)),
                    data={"boxes": json.dumps(boxes), "scenario": "comercial"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "custom"
    assert [s["slot_id"] for s in body["slots"]] == ["t0", "t1", "t2", "neg", "6"]
    assert body["slots"][3]["box"] == {"x": 305, "y": 105, "w": 90, "h": 90}
    assert body["slots"][4]["box"] == {"x": 590, "y": 390, "w": 10, "h": 10}
    assert any("Se ignoraron 1" in w for w in body["warnings"])
    crops = frames.crop_slots(img, [s["box"] for s in body["slots"]])
    np.testing.assert_allclose([s["p_busy"] for s in body["slots"]], expected_p(model_onnx, crops), atol=6e-5)


@pytest.mark.parametrize("pair", [(0.5, 0.5), (0.1, 0.2), (0.0, 1.0), (0.3, 0.7), (0.6, 0.9)])
def test_api_states_follow_thresholds(client, model_onnx, pair):
    card = read_card(model_onnx)
    card["thresholds"]["comercial"] = {"t_L": pair[0], "t_H": pair[1]}
    write_card(model_onnx, card)
    assert client.get("/api/model").json()["thresholds"]["comercial"] == {"t_L": pair[0], "t_H": pair[1]}
    img, boxes = tiles_image()
    r = client.post("/api/predict/custom", files=upload("t.png", png_bytes(img)), data={"boxes": json.dumps(boxes)})
    assert r.status_code == 200
    body = r.json()
    states = []
    for slot in body["slots"]:
        p = slot["p_busy"]
        states.append(slot["states"]["comercial"])
        if min(abs(p - pair[0]), abs(p - pair[1])) > 1e-4:
            assert slot["states"]["comercial"] == rule(p, *pair)
    if pair[0] == pair[1]:
        assert "verificar" not in states
    summ = body["summary_by_scenario"]["comercial"]
    assert (summ["libres"], summ["ocupados"], summ["por_verificar"]) == (
        states.count("libre"), states.count("ocupado"), states.count("verificar"))


def test_tiles_give_spread_of_probabilities(client, model_onnx):
    """Control de la prueba anterior: los bloques producen p en varias zonas de [0, 1]."""
    img, boxes = tiles_image()
    p = np.array([s["p_busy"] for s in client.post(
        "/api/predict/custom", files=upload("t.png", png_bytes(img)),
        data={"boxes": json.dumps(boxes)}).json()["slots"]])
    if USING_TINY:
        assert p.min() < 0.1 and p.max() > 0.9 and np.any((p > 0.3) & (p < 0.7))


def test_feedback_valid_appends_line(client, feedback_file):
    gid = gallery_ids("frame")[0]
    body = {"slot_id": "12", "reported_state": "ocupado", "predicted_state": "libre",
            "p_busy": 0.2, "scenario": "comercial", "source": gid}
    r = client.post("/api/feedback", json=body)
    assert r.status_code == 200 and r.json()["ok"] is True
    r = client.post("/api/feedback", json={"slot_id": "s-3", "reported_state": "libre", "source": "upload"})
    assert r.status_code == 200
    lines = feedback_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    for k, v in body.items():
        assert rec[k] == v
    assert rec["model"] == read_card(Path(os.environ["MODEL_PATH"]))["name"] and "timestamp" in rec
    rec2 = json.loads(lines[1])
    assert rec2["slot_id"] == "s-3" and rec2["p_busy"] is None and rec2["source"] == "upload"


def test_feedback_without_model_still_records(client, feedback_file, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nada" / "best.onnx"))
    r = client.post("/api/feedback", json={"slot_id": "7", "reported_state": "libre"})
    assert r.status_code == 200
    assert json.loads(feedback_file.read_text(encoding="utf-8").splitlines()[0])["model"] is None


# ================================================================ etiquetas reales

def _gt_flags(client) -> tuple[bool, bool, bool, bool]:
    """(frame trae clave, todos los espacios la traen, patch la trae, galería habilitada)."""
    frame = client.post("/api/predict/frame", json={"gallery_id": gallery_ids("frame")[1]}).json()
    patch = client.post("/api/predict/patch", json={"gallery_id": gallery_ids("patch")[0]}).json()
    enabled = client.get("/api/gallery").json()["ground_truth_enabled"]
    return ("ground_truth" in frame,
            bool(frame["slots"]) and all("ground_truth" in s for s in frame["slots"]),
            "ground_truth" in patch["slots"][0],
            enabled)


@real_data
def test_ground_truth_hidden_by_default(client, monkeypatch):
    for value in (None, "0", "true", ""):
        if value is None:
            monkeypatch.delenv("SHOW_GROUND_TRUTH", raising=False)
        else:
            monkeypatch.setenv("SHOW_GROUND_TRUTH", value)
        frame = client.post("/api/predict/frame", json={"gallery_id": gallery_ids("frame")[1]})
        patch = client.post("/api/predict/patch", json={"gallery_id": gallery_ids("patch")[0]})
        custom = client.post("/api/predict/custom", json={"gallery_id": gallery_ids("frame")[0],
                                                          "boxes": [{"x": 100, "y": 300, "w": 80, "h": 70}]})
        assert (frame.status_code, patch.status_code, custom.status_code) == (200, 200, 200)
        present = [has_key(r.json(), "ground_truth") for r in (frame, patch, custom)]
        assert present == [False, False, False], value
        assert client.get("/api/gallery").json()["ground_truth_enabled"] is False


def test_ground_truth_hidden_in_uploads(client, fake_root, monkeypatch):
    monkeypatch.setenv("SHOW_GROUND_TRUTH", "1")
    r = client.post("/api/predict/frame", files=upload("f.png", png_bytes(noise_image(1000, 750))),
                    data={"camera": "1"})
    p = client.post("/api/predict/patch", files=upload("p.png", png_bytes(noise_image(150, 150))))
    assert r.status_code == 200 and p.status_code == 200
    assert has_key(r.json(), "ground_truth") is False
    assert has_key(p.json(), "ground_truth") is False


@real_data
def test_ground_truth_shown_with_flag(client, monkeypatch):
    monkeypatch.setenv("SHOW_GROUND_TRUTH", "1")
    flags = _gt_flags(client)
    assert flags == (True, True, True, True)


# ================================================================ casos límite

@pytest.mark.parametrize("endpoint,data", [
    ("/api/predict/patch", {}),
    ("/api/predict/frame", {"camera": "1"}),
    ("/api/predict/custom", {"boxes": "[{\"x\": 0, \"y\": 0, \"w\": 10, \"h\": 10}]"}),
])
def test_not_an_image(client, fake_root, endpoint, data):
    for name, content, ctype in [("notas.txt", b"hola, esto no es una imagen", "text/plain"),
                                 ("x.pdf", b"%PDF-1.4 ...", "application/pdf"),
                                 ("roto.png", png_bytes(noise_image(50, 50))[:60], "image/png")]:
        r = client.post(endpoint, files=upload(name, content, ctype), data=data)
        assert r.status_code == 400, (endpoint, name, r.status_code)
        assert name in r.json()["detail"]
        assert_private(r, str(fake_root))


def test_empty_file(client):
    r = client.post("/api/predict/patch", files=upload("v.png", b""))
    assert r.status_code == 400 and "vacío" in r.json()["detail"]
    assert_private(r)


def test_upload_over_15_mb(client):
    r = client.post("/api/predict/patch", files=upload("r.jpg", os.urandom(16 * 1024 * 1024), "image/jpeg"))
    assert r.status_code == 413
    assert_private(r)
    # justo por encima del límite (pasa el filtro de Content-Length, lo detiene la lectura)
    r = client.post("/api/predict/patch", files=upload("r.jpg", b"\0" * (15 * 1024 * 1024 + 1), "image/jpeg"))
    assert r.status_code == 413
    assert_private(r)


def test_image_over_50_megapixels(client, fake_root):
    data = huge_png_bytes(8000, 7000)
    assert len(data) < 2 * 1024 * 1024  # pocos bytes, muchos píxeles
    with Image.open(io.BytesIO(data)) as im:
        assert im.size == (8000, 7000)
    r = client.post("/api/predict/patch", files=upload("big.png", data))
    assert r.status_code == 413 and "megapíxeles" in r.json()["detail"]
    r = client.post("/api/predict/frame", files=upload("big.png", data), data={"camera": "1"})
    assert r.status_code == 413
    assert_private(r)


def test_frame_too_small(client, fake_root):
    r = client.post("/api/predict/frame", files=upload("t.png", png_bytes(noise_image(200, 150))),
                    data={"camera": "1"})
    assert r.status_code == 400 and "320x240" in r.json()["detail"]
    r = client.post("/api/predict/frame", files=upload("t.png", png_bytes(noise_image(320, 240))),
                    data={"camera": "1"})
    assert r.status_code == 200


@pytest.mark.parametrize("camera", ["12", "0", "abc", "-1", "1.5", "", None])
def test_invalid_camera(client, fake_root, camera):
    data = {} if camera is None else {"camera": camera}
    r = client.post("/api/predict/frame", files=upload("f.png", png_bytes(noise_image(1000, 750))), data=data)
    assert r.status_code == 400
    assert_private(r, str(fake_root))


def test_invalid_scenario(client, fake_root):
    img = png_bytes(noise_image(150, 150))
    r = client.post("/api/predict/patch", files=upload("p.png", img), data={"scenario": "aeropuerto"})
    assert r.status_code == 400 and "Escenario" in r.json()["detail"]
    r = client.post("/api/predict/frame", json={"gallery_id": gallery_ids("frame")[0], "scenario": "x"})
    assert r.status_code == 400
    r = client.post("/api/predict/patch", files=upload("p.png", img), data={"scenario": "UVG"})
    assert r.status_code == 200 and r.json()["scenario"] == "uvg"


def test_bad_requests_without_image(client):
    assert client.post("/api/predict/frame", json={}).status_code == 400
    assert client.post("/api/predict/patch", json={}).status_code == 400
    assert client.post("/api/predict/custom", json={}).status_code == 400
    r = client.post("/api/predict/frame", content=b"x", headers={"content-type": "text/plain"})
    assert r.status_code == 400
    r = client.post("/api/predict/frame", content=b"{no json", headers={"content-type": "application/json"})
    assert r.status_code == 400
    r = client.post("/api/predict/frame", content=b"[1, 2]", headers={"content-type": "application/json"})
    assert r.status_code == 400
    # un patch de la galería en el modo frame (y viceversa) -> 400, sin abrir la imagen
    assert client.post("/api/predict/frame", json={"gallery_id": gallery_ids("patch")[0]}).status_code == 400
    assert client.post("/api/predict/patch", json={"gallery_id": gallery_ids("frame")[0]}).status_code == 400


@pytest.mark.parametrize("boxes", [
    "[]",
    None,                                   # sin el campo
    "{no json",
    "{\"x\": 1}",                           # no es lista
    "[{\"x\": NaN, \"y\": 0, \"w\": 10, \"h\": 10}]",
    "[{\"x\": 0, \"y\": 0, \"w\": Infinity, \"h\": 10}]",
    "[{\"x\": \"a\", \"y\": 0, \"w\": 10, \"h\": 10}]",
    "[{\"x\": 0, \"y\": 0}]",
    "[5]",
    "[{\"x\": 2000, \"y\": 0, \"w\": 10, \"h\": 10}]",   # todas fuera de la imagen
    "[{\"x\": 10, \"y\": 10, \"w\": 2, \"h\": 2}]",     # demasiado pequeña
    json.dumps([{"x": 0, "y": 0, "w": 10, "h": 10}] * 301),
])
def test_invalid_boxes(client, boxes):
    data = {} if boxes is None else {"boxes": boxes}
    r = client.post("/api/predict/custom", files=upload("t.png", png_bytes(noise_image(600, 400))), data=data)
    assert r.status_code == 400
    assert_private(r)


def test_invalid_boxes_in_json_body(client):
    gid = gallery_ids("frame")[0]
    raw = ('{"gallery_id": "%s", "boxes": [{"x": NaN, "y": 0, "w": 10, "h": 10}]}' % gid).encode()
    r = client.post("/api/predict/custom", content=raw, headers={"content-type": "application/json"})
    assert r.status_code in (400, 404)  # 404 solo si no están los datos de la galería
    if HAS_DATA:
        assert r.status_code == 400


def test_html_slot_id_is_replaced(client):
    boxes = [{"slot_id": "<img src=x onerror=alert(1)>", "x": 10, "y": 10, "w": 50, "h": 50},
             {"slot_id": "a" * 40, "x": 100, "y": 10, "w": 50, "h": 50},
             {"slot_id": "ok-1", "x": 200, "y": 10, "w": 50, "h": 50},
             {"slot_id": "x\ny", "x": 300, "y": 10, "w": 50, "h": 50}]
    r = client.post("/api/predict/custom", files=upload("t.png", png_bytes(noise_image(600, 400))),
                    data={"boxes": json.dumps(boxes)})
    assert r.status_code == 200
    assert [s["slot_id"] for s in r.json()["slots"]] == ["1", "2", "ok-1", "4"]
    assert "<img" not in r.text and "onerror" not in r.text


def test_box_slot_id_with_trailing_newline_is_replaced(client):
    boxes = [{"slot_id": "12\n", "x": 10, "y": 10, "w": 50, "h": 50}]
    r = client.post("/api/predict/custom", files=upload("t.png", png_bytes(noise_image(600, 400))),
                    data={"boxes": json.dumps(boxes)})
    assert r.status_code == 200
    assert r.json()["slots"][0]["slot_id"] == "1"


@pytest.mark.parametrize("body", [
    {"slot_id": "12", "reported_state": "quizas"},
    {"slot_id": "12", "reported_state": "verificar"},
    {"reported_state": "ocupado"},
    {"slot_id": "12"},
    {"slot_id": "12", "reported_state": "ocupado", "p_busy": 1.5},
    {"slot_id": "12", "reported_state": "ocupado", "p_busy": -0.1},
    {"slot_id": "12", "reported_state": "ocupado", "p_busy": "0.3"},
    {"slot_id": "12", "reported_state": "ocupado", "p_busy": True},
    {"slot_id": "12", "reported_state": "ocupado", "scenario": "x"},
    {"slot_id": "12", "reported_state": "ocupado", "predicted_state": "quizas"},
    {"slot_id": "12", "reported_state": "ocupado", "source": "../../etc"},
    {"slot_id": "a b", "reported_state": "ocupado"},
    {"slot_id": "x" * 33, "reported_state": "ocupado"},
    {"slot_id": "1\n2", "reported_state": "ocupado"},
    {"slot_id": "<b>", "reported_state": "ocupado"},
    {"slot_id": True, "reported_state": "ocupado"},
    {"slot_id": ["12"], "reported_state": "ocupado"},
    {"slot_id": "12", "reported_state": "ocupado", "pad": "x" * 3000},
])
def test_feedback_invalid(client, feedback_file, body):
    r = client.post("/api/feedback", json=body)
    assert r.status_code == 400
    assert_private(r)
    assert not feedback_file.exists()


@pytest.mark.parametrize("raw,ctype", [
    (b'{"slot_id": "12", "reported_state": "ocupado", "p_busy": NaN}', "application/json"),
    (b'{"slot_id": "12", "reported_state": "ocupado", "p_busy": Infinity}', "application/json"),
    (b"{no json", "application/json"),
    (b"[1, 2]", "application/json"),
    (b"slot_id=12&reported_state=ocupado", "application/x-www-form-urlencoded"),
])
def test_feedback_invalid_raw(client, feedback_file, raw, ctype):
    r = client.post("/api/feedback", content=raw, headers={"content-type": ctype})
    assert r.status_code == 400
    assert not feedback_file.exists()


def test_feedback_slot_id_with_trailing_newline(client, feedback_file):
    r = client.post("/api/feedback", json={"slot_id": "12\n", "reported_state": "ocupado"})
    assert r.status_code == 400
    assert not feedback_file.exists()


def test_no_model_gives_503(client, tmp_path, monkeypatch):
    missing = tmp_path / "sin_modelo" / "best.onnx"
    monkeypatch.setenv("MODEL_PATH", str(missing))
    r = client.get("/api/model")
    assert r.status_code == 503
    assert "export_onnx" in r.json()["detail"]
    assert_private(r, str(tmp_path))
    r = client.post("/api/predict/patch", files=upload("p.png", png_bytes(noise_image(150, 150))))
    assert r.status_code == 503
    assert_private(r, str(tmp_path))
    h = client.get("/api/health").json()
    assert h["model"] is False and leaks(json.dumps(h), str(tmp_path)) is False


def test_missing_card_gives_503(client, model_onnx, tmp_path):
    model_onnx.with_suffix(".json").unlink()
    r = client.get("/api/model")
    assert r.status_code == 503 and "best.json" in r.json()["detail"]
    assert_private(r, str(tmp_path))


def _set(path, value):
    def mod(card):
        node = card
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value
    return mod


@pytest.mark.parametrize("name,mod", [
    ("t_L > t_H", _set(["thresholds", "uvg"], {"t_L": 0.7, "t_H": 0.6})),
    ("t_H > 1", _set(["thresholds", "comercial", "t_H"], 1.2)),
    ("t_L < 0", _set(["thresholds", "comercial", "t_L"], -0.1)),
    ("t_L NaN", _set(["thresholds", "comercial", "t_L"], float("nan"))),
    ("t_L texto", _set(["thresholds", "comercial", "t_L"], "bajo")),
    ("sin uvg", lambda c: c["thresholds"].pop("uvg")),
    ("sin umbrales", lambda c: c.pop("thresholds")),
    ("T = 0", _set(["temperature"], 0)),
    ("T < 0", _set(["temperature"], -1.0)),
    ("T NaN", _set(["temperature"], float("nan"))),
    ("T texto", _set(["temperature"], "alta")),
])
def test_invalid_card_gives_503(client, model_onnx, tmp_path, name, mod):
    card = read_card(model_onnx)
    mod(card)
    write_card(model_onnx, card)
    r = client.get("/api/model")
    assert r.status_code == 503, name
    assert_private(r, str(tmp_path))
    r = client.post("/api/predict/patch", files=upload("p.png", png_bytes(noise_image(150, 150))))
    assert r.status_code == 503, name


def test_broken_card_json_gives_503(client, model_onnx, tmp_path):
    model_onnx.with_suffix(".json").write_text("{no json", encoding="utf-8")
    r = client.get("/api/model")
    assert r.status_code == 503
    assert_private(r, str(tmp_path))


@pytest.mark.parametrize("content", ["[1, 2]", "3", "null", "\"texto\""])
def test_card_not_an_object_gives_503(client, model_onnx, content):
    model_onnx.with_suffix(".json").write_text(content, encoding="utf-8")
    r = client.get("/api/model")
    assert r.status_code == 503


def test_corrupt_onnx_gives_503(client, model_onnx, tmp_path):
    model_onnx.write_bytes(b"no es onnx")
    r = client.get("/api/model")
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert "best.onnx" in detail and "export_onnx" in detail
    assert_private(r, str(tmp_path))


def test_no_camera_boxes_gives_503(client, empty_root):
    r = client.post("/api/predict/frame", files=upload("f.png", png_bytes(noise_image(1000, 750))),
                    data={"camera": "5"})
    assert r.status_code == 503 and "cámara 5" in r.json()["detail"]
    assert_private(r, str(empty_root))


def test_gallery_404_without_data_root_path(client, empty_root):
    gid = gallery_ids("frame")[0]
    r = client.get(f"/api/gallery/{gid}/image")
    assert r.status_code == 404 and "DATA_ROOT" in r.json()["detail"]
    assert_private(r, str(empty_root))
    r = client.post("/api/predict/frame", json={"gallery_id": gid})
    assert r.status_code == 404
    assert_private(r, str(empty_root))
    r = client.get("/api/gallery/no-existe/image")
    assert r.status_code == 404
    r = client.post("/api/predict/frame", json={"gallery_id": "zzz"})
    assert r.status_code == 404
    assert_private(r, str(empty_root))


def test_unexpected_error_gives_generic_500(client, monkeypatch):
    def boom(img):
        raise RuntimeError("fallo interno en C:/" + "Us" + "ers/secreto/ruta")

    monkeypatch.setattr(M, "image_quality", boom)
    r = client.post("/api/predict/patch", files=upload("p.png", png_bytes(noise_image(150, 150))))
    assert r.status_code == 500
    detail = r.json()["detail"]
    assert "Error interno" in detail
    assert "secreto" not in r.text and "RuntimeError" not in r.text and "Traceback" not in r.text
    assert_private(r)


def test_hot_reload_when_model_is_replaced(client, model_onnx, tmp_path, monkeypatch):
    first = client.get("/api/model").json()
    card = read_card(model_onnx)
    card["name"] = "modelo_nuevo_reemplazado"
    card["run_id"] = "nuevo_s1"
    card["thresholds"]["uvg"] = {"t_L": 0.11, "t_H": 0.22}
    write_card(model_onnx, card)
    second = client.get("/api/model").json()
    assert first["name"] != second["name"] == "modelo_nuevo_reemplazado"
    assert second["thresholds"]["uvg"] == {"t_L": 0.11, "t_H": 0.22}
    # se rompe el .onnx -> 503; se restaura -> vuelve a funcionar sin reiniciar
    good = model_onnx.read_bytes()
    model_onnx.write_bytes(b"no es onnx")
    assert client.get("/api/model").status_code == 503
    model_onnx.write_bytes(good)
    assert client.get("/api/model").status_code == 200
    # cambiar MODEL_PATH a otra carpeta también cambia el modelo
    other = make_model_dir(tmp_path / "otro")
    c2 = read_card(other)
    c2["name"] = "modelo_en_otra_carpeta"
    write_card(other, c2)
    monkeypatch.setenv("MODEL_PATH", str(other))
    assert client.get("/api/model").json()["name"] == "modelo_en_otra_carpeta"


# ================================================================ galería (datos)

def _load_build_gallery():
    spec = importlib.util.spec_from_file_location("build_gallery", REPO / "scripts" / "build_gallery.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_gallery"] = mod  # dataclasses necesita el módulo registrado
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop("build_gallery", None)
        raise
    return mod


@real_data
def test_build_gallery_reproduces_committed_json(tmp_path, capsys):
    out = tmp_path / "gallery.json"
    _load_build_gallery().main(["--out", str(out)])
    capsys.readouterr()
    assert out.read_text(encoding="utf-8") == GALLERY_JSON.read_text(encoding="utf-8")


def _split_dates(name: str) -> set[str]:
    rel = pd.read_csv(paths.split_file(name), sep=" ", header=None, usecols=[0], dtype=str)[0]
    return set(rel.str.split("/", n=2).str[1])


@real_data
def test_gallery_frames_are_from_unseen_dates():
    seen = _split_dates("train") | _split_dates("val")
    frames_items = [it for it in _gallery() if it["kind"] == "frame"]
    assert len(frames_items) == 8
    assert len({it["camera"] for it in frames_items}) == 8
    for it in frames_items:
        assert it["date"] not in seen
        assert it["subset"] == "CNR-EXT"
        assert it["path"].startswith("CNR-EXT_FULL/FULL_IMAGE_1000x750/")
        assert f"/{it['date']}/camera{it['camera']}/" in it["path"]
        assert (paths.data_root() / it["path"]).is_file()


def test_gallery_patches_are_cnrpark_ab():
    patches = [it for it in _gallery() if it["kind"] == "patch"]
    assert len(patches) == 6
    assert sorted(it["camera"] for it in patches) == ["A", "A", "A", "B", "B", "B"]
    for it in patches:
        parts = Path(it["path"]).parts
        assert it["subset"] == "CNRPark" and parts[0] == "CNRPark" and parts[1] == it["camera"]
        assert "CNR-EXT" not in it["path"]
    for cam in ("A", "B"):
        folders = {Path(it["path"]).parts[2] for it in patches if it["camera"] == cam}
        assert folders == {"busy", "free"}


@real_data
def test_gallery_patches_exist():
    for it in _gallery():
        if it["kind"] == "patch":
            assert (paths.data_root() / it["path"]).is_file()
