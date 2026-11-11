"""Pruebas de data.py: particiones, guarda de test, submuestreo, Dataset y DataLoader.

Las pruebas marcadas con ``data`` usan el dataset real bajo DATA_ROOT y se saltan
si no está disponible.
"""

from __future__ import annotations

import ast
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from cupo import data, paths  # noqa: E402
from cupo.transforms import build_transforms  # noqa: E402

HAS_DATA = paths.split_file("train").is_file() and paths.metadata_csv().is_file()
needs_data = [
    pytest.mark.data,
    pytest.mark.skipif(not HAS_DATA, reason="dataset no disponible en DATA_ROOT"),
]


def _mark(*marks):
    def deco(fn):
        for m in reversed(marks):
            fn = m(fn)
        return fn
    return deco


real_data = _mark(*needs_data)


# ================================================================ unitarias

# ------------------------------------------------------ dataset sintético

def _write_images(folder: Path, n: int, mode: str = "RGB", size: int = 20) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        shape = (size, size) if mode == "L" else (size, size, 3)
        arr = rng.integers(0, 256, shape, dtype=np.uint8)
        p = folder / f"img_{i}.png"
        Image.fromarray(arr, mode=mode).save(p)
        out.append(str(p))
    return out


@pytest.fixture
def image_df(tmp_path) -> pd.DataFrame:
    files = _write_images(tmp_path / "rgb", 6) + _write_images(tmp_path / "gray", 4, mode="L")
    return pd.DataFrame({"path": files, "label": [0, 1] * 5})


def test_parking_dataset_returns_rgb_and_label(image_df):
    ds = data.ParkingDataset(image_df)
    assert len(ds) == 10
    for i in range(len(ds)):
        img, label = ds[i]
        assert isinstance(img, Image.Image) and img.mode == "RGB"  # gris → RGB
        assert label == image_df["label"].iloc[i]
        assert isinstance(label, int)


def test_parking_dataset_with_transform(image_df):
    ds = data.ParkingDataset(image_df, build_transforms(16, train=False, augment=False))
    x, y = ds[7]  # imagen en escala de grises
    assert tuple(x.shape) == (3, 16, 16) and x.dtype == torch.float32
    assert y in (0, 1)


def test_parking_dataset_is_picklable(image_df):
    ds = data.ParkingDataset(image_df, build_transforms(16, train=True, augment=True))
    clone = pickle.loads(pickle.dumps(ds))
    assert clone.paths == ds.paths and clone.labels == ds.labels
    x, y = clone[0]
    assert tuple(x.shape) == (3, 16, 16) and y == ds.labels[0]
    # Solo guarda listas de Python, no el DataFrame
    assert not any(isinstance(v, pd.DataFrame) for v in vars(ds).values())


def test_make_loader_batch_shape_synthetic(image_df):
    loader = data.make_loader(image_df, build_transforms(24, False, False), batch_size=4,
                              shuffle=False, num_workers=0)
    x, y = next(iter(loader))
    assert tuple(x.shape) == (4, 3, 24, 24) and x.dtype == torch.float32
    assert tuple(y.shape) == (4,) and y.dtype == torch.int64
    assert y.tolist() == image_df["label"].iloc[:4].tolist()
    assert sum(len(b[1]) for b in loader) == len(image_df)


def test_make_loader_with_worker_processes(image_df):
    # En Windows los workers usan spawn: el Dataset y la transformación se serializan
    # Un reintento: arrancar procesos en Windows puede fallar por carga del sistema;
    # un error real de serialización falla en ambos intentos.
    for attempt in range(2):
        loader = data.make_loader(image_df, build_transforms(16, train=True, augment=True),
                                  batch_size=5, shuffle=True, num_workers=2, seed=0)
        try:
            batches = list(loader)
            break
        except RuntimeError:
            if attempt == 1:
                raise
    assert [tuple(x.shape) for x, _ in batches] == [(5, 3, 16, 16), (5, 3, 16, 16)]
    assert sorted(sum((y.tolist() for _, y in batches), [])) == sorted(image_df["label"].tolist())


def test_make_loader_shuffle_reproducible(image_df):
    def order(seed):
        loader = data.make_loader(image_df, build_transforms(8, False, False), batch_size=10,
                                  shuffle=True, num_workers=0, seed=seed)
        x, _ = next(iter(loader))
        return x[:, 0, 0, 0].tolist()

    assert order(5) == order(5)


# --------------------------------------------------------------- subsample

def _toy_df(n_pos: int = 700, n_neg: int = 300) -> pd.DataFrame:
    labels = np.array([1] * n_pos + [0] * n_neg)
    rng = np.random.default_rng(0)
    rng.shuffle(labels)
    return pd.DataFrame({"path": [f"p{i}.jpg" for i in range(len(labels))], "label": labels})


def test_subsample_is_stratified():
    df = _toy_df(700, 300)
    sub = data.subsample(df, 0.1, seed=0)
    counts = sub["label"].value_counts().to_dict()
    assert counts == {1: 70, 0: 30}


def test_subsample_rounding_and_min_one_per_class():
    df = _toy_df(995, 5)
    sub = data.subsample(df, 0.02, seed=0)
    counts = sub["label"].value_counts().to_dict()
    # 0.02 * 995 = 19.9 → 20 ; 0.02 * 5 = 0.1 → mínimo 1
    assert counts == {1: 20, 0: 1}


def test_subsample_is_reproducible_and_depends_on_seed():
    df = _toy_df()
    a = data.subsample(df, 0.05, seed=42)
    b = data.subsample(df, 0.05, seed=42)
    c = data.subsample(df, 0.05, seed=43)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["path"]) != set(c["path"])


def test_subsample_keeps_rows_and_order():
    df = _toy_df()
    sub = data.subsample(df, 0.2, seed=1)
    assert list(sub.index) == list(range(len(sub)))
    # Las filas existen en el original con su misma etiqueta
    original = dict(zip(df["path"], df["label"]))
    assert all(original[p] == lbl for p, lbl in zip(sub["path"], sub["label"]))
    # Se conserva el orden relativo original
    pos = [int(p[1:-4]) for p in sub["path"]]
    assert pos == sorted(pos)
    assert sub["path"].is_unique


def test_subsample_respects_shuffled_index():
    df = _toy_df()
    rng = np.random.default_rng(3)
    shuffled = df.copy()
    shuffled.index = rng.permutation(np.arange(10_000, 10_000 + len(df)))  # índice desordenado
    sub_a = data.subsample(df, 0.1, seed=7)
    sub_b = data.subsample(shuffled, 0.1, seed=7)
    # El resultado depende del orden de las filas, no de las etiquetas del índice
    pd.testing.assert_frame_equal(sub_a, sub_b)
    original = dict(zip(df["path"], df["label"]))
    assert all(original[p] == lbl for p, lbl in zip(sub_b["path"], sub_b["label"]))
    assert sub_b["label"].value_counts().to_dict() == {1: 70, 0: 30}


def test_subsample_with_duplicate_index_labels():
    df = _toy_df(40, 60)
    df.index = [0] * len(df)  # índice repetido
    sub = data.subsample(df, 0.5, seed=0)
    assert sub["label"].value_counts().to_dict() == {1: 20, 0: 30}
    assert sub["path"].is_unique


def test_subsample_full_fraction_returns_copy():
    df = _toy_df()
    sub = data.subsample(df, 1.0)
    pd.testing.assert_frame_equal(sub, df.reset_index(drop=True))
    sub.loc[0, "label"] = 99
    assert df["label"].iloc[0] != 99


@pytest.mark.parametrize("frac", [0, -0.1, 1.5])
def test_subsample_rejects_invalid_fraction(frac):
    with pytest.raises(ValueError):
        data.subsample(_toy_df(), frac)


def test_subsample_does_not_modify_input():
    df = _toy_df()
    df.index = df.index[::-1]
    before = df.copy()
    data.subsample(df, 0.3, seed=0)
    pd.testing.assert_frame_equal(df, before)


# ------------------------------------------------ guarda de test (sin datos)

@pytest.mark.parametrize("name", ["test", "ood"])
def test_load_protected_split_raises_without_touching_disk(monkeypatch, tmp_path, name):
    # DATA_ROOT vacío: la guarda debe saltar antes de leer cualquier archivo
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(tmp_path))
    with pytest.raises(PermissionError):
        data.load_split(name)
    with pytest.raises(PermissionError):
        data.load_split(name, allow_test=False)


def test_load_split_unknown_name():
    with pytest.raises(ValueError):
        data.load_split("training")


def test_public_api_does_not_expose_read_split_file():
    assert not hasattr(data, "read_split_file")
    public = [n for n in dir(data) if not n.startswith("_")]
    assert not any("read_split" in n for n in public)


def test_protected_splits_constant():
    assert data.PROTECTED_SPLITS == frozenset({"test", "ood"})
    assert set(data.SPLITS) == {"train", "val", "test", "ood"}
    assert data.COLUMNS == ["path", "label", "camera", "weather", "date", "time", "slot_id", "subset"]


# ------------------------------------------- parseo con un DATA_ROOT falso

@pytest.fixture
def fake_root(monkeypatch, tmp_path) -> Path:
    labels = tmp_path / "CNR-EXT" / "LABELS"
    labels.mkdir(parents=True)
    (labels / "train.txt").write_text(
        "SUNNY/2015-11-12/camera8/S_2015-11-12_12.26_C08_280.jpg 1\n"
        "OVERCAST/2015-11-25/camera1/O_2015-11-25_07.05_C01_101.jpg 0\n"
        "RAINY/2016-01-09/camera9/R_2016-01-09_16.47_C09_007.jpg 1\n",
        encoding="utf-8",
    )
    (labels / "val.txt").write_text(
        "SUNNY/2015-12-01/camera2/S_2015-12-01_09.00_C02_5.jpg 0\n", encoding="utf-8"
    )
    (labels / "test.txt").write_text(
        "SUNNY/2016-02-01/camera3/S_2016-02-01_10.00_C03_9.jpg 1\n", encoding="utf-8"
    )
    pd.DataFrame({
        "camera": ["A", "B", "01"],
        "datetime": ["20150703_0805", "20150716_1458", "2015-11-12_12.26"],
        "day": ["3", "16", "12"], "hour": ["8", "14", "12"],
        "image_url": ["CNRPark/A/free/20150703_0805_1.jpg", "CNRPark/B/busy/20150716_1458_22.jpg",
                      "CNR-EXT/PATCHES/SUNNY/2015-11-12/camera1/x.jpg"],
        "minute": ["5", "58", "26"], "month": ["7", "7", "11"],
        "occupancy": ["0", "1", "1"], "slot_id": ["1", "22", "3"],
        "weather": ["S", "O", "S"], "year": ["2015"] * 3, "occupant_changed": ["", "", ""],
    }).to_csv(tmp_path / "CNRPark+EXT.csv", index=False)
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(tmp_path))
    return tmp_path.resolve()


def test_load_split_parses_ext_paths(fake_root):
    df = data.load_split("train")
    assert list(df.columns) == data.COLUMNS
    row = df.iloc[0]
    expected_path = str(fake_root / "CNR-EXT" / "PATCHES" / "SUNNY" / "2015-11-12" / "camera8"
                        / "S_2015-11-12_12.26_C08_280.jpg")
    assert row["path"] == expected_path
    assert row["label"] == 1 and row["camera"] == "8" and row["weather"] == "SUNNY"
    assert row["date"] == "2015-11-12" and row["time"] == "12:26"
    assert row["slot_id"] == "280" and row["subset"] == "EXT"
    assert df["camera"].tolist() == ["8", "1", "9"]
    assert df["time"].tolist() == ["12:26", "07:05", "16:47"]
    assert df["slot_id"].tolist() == ["280", "101", "7"]  # sin ceros a la izquierda
    assert df["weather"].tolist() == ["SUNNY", "OVERCAST", "RAINY"]
    assert df["label"].tolist() == [1, 0, 1]


def test_load_split_returns_independent_copy(fake_root):
    a = data.load_split("train")
    a.loc[0, "label"] = 7
    b = data.load_split("train")
    assert b.loc[0, "label"] == 1


def test_load_split_rejects_malformed_path(monkeypatch, tmp_path):
    labels = tmp_path / "CNR-EXT" / "LABELS"
    labels.mkdir(parents=True)
    (labels / "train.txt").write_text("SUNNY/2015-11-12/camera8/raro.jpg 1\n", encoding="utf-8")
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(tmp_path))
    with pytest.raises(ValueError, match="formato inesperado"):
        data.load_split("train")


def test_load_ood_from_csv(fake_root):
    df = data.load_split("ood", allow_test=True)
    assert list(df.columns) == data.COLUMNS
    assert len(df) == 2  # solo cámaras A y B
    assert df["camera"].tolist() == ["A", "B"]
    assert df["subset"].eq("CNRPark").all()
    assert df["date"].tolist() == ["2015-07-03", "2015-07-16"]
    assert df["time"].tolist() == ["08:05", "14:58"]
    assert df["weather"].tolist() == ["SUNNY", "OVERCAST"]
    assert df["label"].tolist() == [0, 1]
    assert df["slot_id"].tolist() == ["1", "22"]
    assert df["path"].iloc[0] == str(fake_root / "CNRPark" / "A" / "free" / "20150703_0805_1.jpg")


def test_split_dates_fake(fake_root):
    assert data.split_dates("train") == {"2015-11-12", "2015-11-25", "2016-01-09"}
    assert data.split_dates("test") == {"2016-02-01"}


# ------------------------------------------------ invariante de train.py

def test_training_only_loads_train_and_val():
    """train.py no carga test ni ood ni usa allow_test."""
    src = (paths.repo_root() / "src" / "cupo" / "train.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name in ("load_split", "_load_split_cached", "_read_split_file", "_load_ood"):
                calls.append(node)
            for kw in node.keywords:
                assert kw.arg != "allow_test", "train.py no debe usar allow_test"
    assert calls, "train.py debería cargar particiones con load_split"
    for call in calls:
        args = [a.value for a in call.args if isinstance(a, ast.Constant)]
        assert args and set(args) <= {"train", "val"}, f"carga no permitida: {ast.dump(call)}"
    assert "CNRPark" not in src


# ============================================================ integración

@pytest.fixture(scope="module")
def splits():
    if not HAS_DATA:
        pytest.skip("dataset no disponible en DATA_ROOT")
    # test y ood se cargan solo para contar filas y validar el esquema (no se evalúa nada)
    return {
        "train": data.load_split("train"),
        "val": data.load_split("val"),
        "test": data.load_split("test", allow_test=True),
        "ood": data.load_split("ood", allow_test=True),
    }


@real_data
def test_split_counts_match_plan(splits):
    counts = {k: len(v) for k, v in splits.items()}
    assert counts == {"train": 94_493, "val": 18_647, "test": 31_825, "ood": 12_584}
    assert counts["train"] + counts["val"] + counts["test"] == 144_965


@real_data
def test_split_busy_percentages_match_plan(splits):
    busy = {k: round(100 * splits[k]["label"].mean(), 1) for k in ("train", "val", "test")}
    assert busy == {"train": 50.4, "val": 71.9, "test": 57.4}
    ext = pd.concat([splits[k] for k in ("train", "val", "test")])
    assert round(100 * ext["label"].mean(), 1) == 54.7


@real_data
@pytest.mark.parametrize("name", ["train", "val", "test", "ood"])
def test_split_schema_without_nan(splits, name):
    df = splits[name]
    assert list(df.columns) == data.COLUMNS
    assert not df.isna().any().any()
    assert set(df["label"].unique()) <= {0, 1}
    assert pd.api.types.is_integer_dtype(df["label"])
    for col in ("path", "camera", "weather", "date", "time", "slot_id", "subset"):
        assert df[col].map(type).eq(str).all(), col
    assert df["path"].map(os.path.isabs).all()
    assert set(df["weather"].unique()) <= {"SUNNY", "OVERCAST", "RAINY"}
    assert df["date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()
    assert df["time"].str.fullmatch(r"([01]\d|2[0-3]):[0-5]\d").all()
    assert df["slot_id"].str.fullmatch(r"[1-9]\d*").all()
    if name == "ood":
        assert set(df["camera"].unique()) == {"A", "B"}
        assert df["subset"].eq("CNRPark").all()
    else:
        assert set(df["camera"].unique()) <= {str(k) for k in range(1, 10)}
        assert df["subset"].eq("EXT").all()


@real_data
@pytest.mark.parametrize("name", ["train", "val", "test", "ood"])
def test_split_paths_exist_sample(splits, name):
    df = splits[name]
    sample = df.sample(n=50, random_state=0)
    assert all(Path(p).is_file() for p in sample["path"])


@real_data
def test_val_has_no_rain(splits):
    assert "RAINY" not in set(splits["val"]["weather"])


@real_data
def test_protected_splits_raise_with_real_data():
    for name in ("test", "ood"):
        with pytest.raises(PermissionError):
            data.load_split(name)


@real_data
def test_split_dates_are_disjoint():
    dates = {n: data.split_dates(n) for n in ("train", "val", "test")}
    assert all(dates.values())
    assert not dates["train"] & dates["val"]
    assert not dates["train"] & dates["test"]
    assert not dates["val"] & dates["test"]
    assert not dates["train"] & dates["val"] & dates["test"]


@real_data
def test_split_dates_agree_with_loaded_dates(splits):
    for name in ("train", "val", "test"):
        assert data.split_dates(name) == set(splits[name]["date"])


@real_data
def test_ood_dates_do_not_overlap_ext(splits):
    ext_dates = set().union(*(set(splits[n]["date"]) for n in ("train", "val", "test")))
    assert not set(splits["ood"]["date"]) & ext_dates


@real_data
def test_paths_are_unique_across_splits(splits):
    seen: set[str] = set()
    for name in ("train", "val", "test", "ood"):
        p = set(splits[name]["path"])
        assert len(p) == len(splits[name])
        assert not p & seen
        seen |= p


@real_data
def test_make_loader_real_batch():
    df = data.subsample(data.load_split("train"), 0.001, seed=0)
    loader = data.make_loader(df, build_transforms(64, train=True, augment=True), batch_size=8,
                              shuffle=True, num_workers=0, seed=0)
    x, y = next(iter(loader))
    assert tuple(x.shape) == (8, 3, 64, 64) and x.dtype == torch.float32
    assert tuple(y.shape) == (8,) and set(y.tolist()) <= {0, 1}


@real_data
def test_subsample_two_percent_real(splits):
    df = splits["train"]
    sub = data.subsample(df, 0.02, seed=42)
    assert abs(len(sub) - 0.02 * len(df)) <= 2
    assert abs(sub["label"].mean() - df["label"].mean()) < 1e-3
    pd.testing.assert_frame_equal(sub, data.subsample(df, 0.02, seed=42))


def test_regex_matches_doc_example():
    assert re.match(data._EXT_PATH_RE, "SUNNY/2015-11-12/camera8/S_2015-11-12_12.26_C08_280.jpg")
