"""Pruebas de frames.py: listado de frames, cajas, recortes y etiquetas por frame.

Las pruebas marcadas con ``data`` usan el dataset real bajo DATA_ROOT y se saltan
si no está disponible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cupo import frames, paths

HAS_DATA = (
    paths.split_file("test").is_file()
    and paths.metadata_csv().is_file()
    and paths.full_images_dir().is_dir()
)


def real_data(fn):
    fn = pytest.mark.skipif(not HAS_DATA, reason="dataset no disponible en DATA_ROOT")(fn)
    return pytest.mark.data(fn)


# ================================================================ unitarias

# ----------------------------------------------------------- crop_slots

@pytest.fixture
def grid_image() -> tuple[Image.Image, np.ndarray]:
    """Imagen 40x30 donde cada píxel tiene un valor distinto por canal."""
    h, w = 30, 40
    yy, xx = np.mgrid[0:h, 0:w]
    arr = np.stack([xx * 6, yy * 8, (xx + yy) % 256], axis=-1).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB"), arr


BOXES = [(0, 0, 10, 5), (12, 7, 8, 20), (30, 25, 10, 5)]


def _check_crops(crops, arr):
    assert len(crops) == len(BOXES)
    for crop, (x, y, w, h) in zip(crops, BOXES):
        assert crop.mode == "RGB"
        assert crop.size == (w, h)
        np.testing.assert_array_equal(np.asarray(crop), arr[y:y + h, x:x + w])


def test_crop_slots_with_dataframe(grid_image):
    img, arr = grid_image
    boxes = pd.DataFrame(BOXES, columns=["x", "y", "w", "h"])
    boxes.insert(0, "slot_id", ["1", "2", "3"])
    _check_crops(frames.crop_slots(img, boxes), arr)


def test_crop_slots_with_dataframe_shuffled_columns_and_index(grid_image):
    img, arr = grid_image
    boxes = pd.DataFrame(BOXES, columns=["x", "y", "w", "h"])[["h", "w", "y", "x"]]
    boxes.index = [10, 5, 7]
    _check_crops(frames.crop_slots(img, boxes), arr)


def test_crop_slots_with_dicts(grid_image):
    img, arr = grid_image
    boxes = [{"x": x, "y": y, "w": w, "h": h, "slot_id": str(i)} for i, (x, y, w, h) in enumerate(BOXES)]
    _check_crops(frames.crop_slots(img, boxes), arr)


def test_crop_slots_with_tuples_and_lists(grid_image):
    img, arr = grid_image
    _check_crops(frames.crop_slots(img, BOXES), arr)
    _check_crops(frames.crop_slots(img, [list(b) for b in BOXES]), arr)
    _check_crops(frames.crop_slots(img, iter(BOXES)), arr)  # generador


def test_crop_slots_float_coordinates_truncate(grid_image):
    img, arr = grid_image
    crops = frames.crop_slots(img, [(1.9, 2.2, 5.7, 3.1)])
    assert crops[0].size == (5, 3)
    np.testing.assert_array_equal(np.asarray(crops[0]), arr[2:5, 1:6])


def test_crop_slots_gray_input_returns_rgb():
    img = Image.new("L", (20, 20), 77)
    crops = frames.crop_slots(img, [(0, 0, 4, 4)])
    assert crops[0].mode == "RGB"
    assert np.asarray(crops[0]).tolist() == [[[77, 77, 77]] * 4] * 4


def test_crop_slots_empty():
    assert frames.crop_slots(Image.new("RGB", (5, 5)), []) == []
    assert frames.crop_slots(Image.new("RGB", (5, 5)), pd.DataFrame(columns=["x", "y", "w", "h"])) == []


# ---------------------------------------------------- utilidades privadas

@pytest.mark.parametrize("text,expected", [("12:26", 746), ("1226", 746), ("12.26", 746),
                                           ("07:05", 425), ("00:00", 0), ("23:59", 1439)])
def test_to_minutes(text, expected):
    assert frames._to_minutes(text) == expected


# ------------------------------------------------- DATA_ROOT sintético

@pytest.fixture
def fake_root(monkeypatch, tmp_path) -> Path:
    # LABELS/test.txt marca 2016-02-01 como fecha de test
    labels = tmp_path / "CNR-EXT" / "LABELS"
    labels.mkdir(parents=True)
    (labels / "test.txt").write_text(
        "SUNNY/2016-02-01/camera3/S_2016-02-01_10.00_C03_9.jpg 1\n", encoding="utf-8"
    )
    full = tmp_path / "CNR-EXT_FULL"
    imgs = full / "FULL_IMAGE_1000x750"
    for rel in [
        "SUNNY/2015-11-12/camera1/2015-11-12_0709.jpg",
        "RAINY/2016-01-09/camera9/2016-01-09_1647.jpg",
        "OVERCAST/2016-02-01/camera3/2016-02-01_1000.jpg",   # fecha de test
        "SUNNY/2015-11-12/camera1/no_es_frame.jpg",          # nombre que no cumple el patrón
    ]:
        p = imgs / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6)).save(p)
    # Cajas en coordenadas 2592x1944
    (full / "camera1.csv").write_text(
        "SlotId,X,Y,W,H\n"
        "7,1296,972,1296,972\n"
        "8,2500,1900,400,200\n"
        "9,100,50,380,373\n",
        encoding="utf-8",
    )
    rows = [
        # camera, datetime, hour, minute, slot, occupancy
        ("01", "2015-11-12_12.25", 12, 25, 3, 1),
        ("01", "2015-11-12_12.27", 12, 27, 3, 0),   # empate en distancia con 12:25
        ("01", "2015-11-12_12.29", 12, 29, 4, 1),   # a 3 min: fuera con tol=2
        ("01", "2015-11-12_12.26", 12, 26, 5, 0),
        ("01", "2015-11-12_12.24", 12, 24, 6, 1),   # a 2 min: dentro
        ("01", "2015-11-12_12.26", 12, 26, 7, 1),
        ("01", "2015-11-12_12.27", 12, 27, 7, 0),   # el de 12:26 es más cercano
        ("02", "2015-11-12_12.26", 12, 26, 3, 0),   # otra cámara
        ("01", "2015-11-13_12.26", 12, 26, 3, 0),   # otra fecha
        ("03", "2016-02-01_10.00", 10, 0, 9, 1),    # fecha de test
        ("A", "20150703_0805", 8, 5, 1, 0),         # CNRPark: se ignora
    ]
    pd.DataFrame({
        "camera": [r[0] for r in rows],
        "datetime": [r[1] for r in rows],
        "day": ["1"] * len(rows),
        "hour": [str(r[2]) for r in rows],
        "image_url": [f"x/{i}.jpg" for i in range(len(rows))],
        "minute": [str(r[3]) for r in rows],
        "month": ["11"] * len(rows),
        "occupancy": [str(r[5]) for r in rows],
        "slot_id": [str(r[4]) for r in rows],
        "weather": ["S"] * len(rows),
        "year": ["2015"] * len(rows),
        "occupant_changed": [""] * len(rows),
    }).to_csv(tmp_path / "CNRPark+EXT.csv", index=False)
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(tmp_path))
    return tmp_path.resolve()


def test_list_frames_fake_excludes_test_dates(fake_root):
    df = frames.list_frames()
    assert list(df.columns) == frames.FRAME_COLUMNS
    assert len(df) == 2
    assert set(df["date"]) == {"2015-11-12", "2016-01-09"}
    assert not df["is_test_date"].any()
    row = df[df["camera"] == 1].iloc[0]
    assert row["time"] == "07:09" and row["weather"] == "SUNNY" and row["date"] == "2015-11-12"
    assert Path(row["path"]).is_file()
    assert pd.api.types.is_integer_dtype(df["camera"])
    assert list(df.index) == list(range(len(df)))


def test_list_frames_fake_allow_test(fake_root):
    df = frames.list_frames(allow_test=True)
    assert len(df) == 3
    flags = dict(zip(df["date"], df["is_test_date"]))
    assert flags == {"2015-11-12": False, "2016-01-09": False, "2016-02-01": True}
    assert set(df["camera"]) == {1, 3, 9}


def test_slot_boxes_scaling_hand_computed(fake_root):
    boxes = frames.slot_boxes(1)
    assert list(boxes.columns) == ["slot_id", "x", "y", "w", "h"]
    assert boxes["slot_id"].tolist() == ["7", "8", "9"]
    # Factor 1000/2592 = 750/1944 = 0.3858...
    # 7: (1296, 972, 1296, 972) → (500, 375) a (1000, 750)
    # 8: x0=round(964.5)=965, x1=min(1000, 1118.8)=1000; y0=round(733.0)=733, y1=min(750, 810.2)=750
    # 9: x0=round(38.6)=39, x1=round(185.2)=185; y0=round(19.3)=19, y1=round(163.2)=163
    expected = [(500, 375, 500, 375), (965, 733, 35, 17), (39, 19, 146, 144)]
    got = list(boxes[["x", "y", "w", "h"]].itertuples(index=False, name=None))
    assert got == expected


def test_ground_truth_hand_computed(fake_root):
    gt = frames.ground_truth(1, "2015-11-12", "12:26")
    # 3: empate 12:25/12:27 → gana el más temprano (1); 4 queda fuera; 5 exacto (0);
    # 6 a 2 min (1); 7: 12:26 (1) le gana a 12:27
    assert gt == {"3": 1, "5": 0, "6": 1, "7": 1}
    assert all(isinstance(k, str) for k in gt) and all(isinstance(v, int) for v in gt.values())


def test_ground_truth_time_formats_and_tolerance(fake_root):
    base = frames.ground_truth(1, "2015-11-12", "12:26")
    assert frames.ground_truth(1, "2015-11-12", "1226") == base
    assert frames.ground_truth(1, "2015-11-12", "12.26") == base
    wider = frames.ground_truth(1, "2015-11-12", "12:26", tol_min=3)
    assert wider["4"] == 1
    assert frames.ground_truth(1, "2015-11-12", "12:26", tol_min=0) == {"5": 0, "7": 1}
    assert frames.ground_truth(1, "2015-11-12", "15:00") == {}
    assert frames.ground_truth(5, "2015-11-12", "12:26") == {}


def test_ground_truth_test_date_guard_fake(fake_root):
    with pytest.raises(PermissionError):
        frames.ground_truth(3, "2016-02-01", "10:00")
    assert frames.ground_truth(3, "2016-02-01", "10:00", allow_test=True) == {"9": 1}


def test_frames_module_does_not_import_torch():
    src = (paths.repo_root() / "src" / "cupo" / "frames.py").read_text(encoding="utf-8")
    assert "import torch" not in src and "from torch" not in src


# ============================================================ integración

@pytest.fixture(scope="module")
def real_frames():
    if not HAS_DATA:
        pytest.skip("dataset no disponible en DATA_ROOT")
    return frames.list_frames()


@pytest.fixture(scope="module")
def test_dates():
    if not HAS_DATA:
        pytest.skip("dataset no disponible en DATA_ROOT")
    from cupo import data

    return data.split_dates("test")


@real_data
def test_list_frames_excludes_test_dates_real(real_frames, test_dates):
    assert len(real_frames) > 0
    assert list(real_frames.columns) == frames.FRAME_COLUMNS
    assert not real_frames["is_test_date"].any()
    assert not set(real_frames["date"]) & test_dates
    assert not real_frames.isna().any().any()


@real_data
def test_list_frames_allow_test_real(real_frames, test_dates):
    everything = frames.list_frames(allow_test=True)
    assert len(everything) == 4_278
    marked = set(everything.loc[everything["is_test_date"], "date"])
    assert marked and marked <= test_dates
    assert len(everything) - everything["is_test_date"].sum() == len(real_frames)
    assert set(everything["camera"]) == set(range(1, 10))


@real_data
def test_ground_truth_test_date_raises_real(test_dates):
    everything = frames.list_frames(allow_test=True)
    row = everything[everything["is_test_date"]].iloc[0]
    assert row["date"] in test_dates
    with pytest.raises(PermissionError):
        frames.ground_truth(int(row["camera"]), row["date"], row["time"])


@real_data
def test_ground_truth_train_date_real(real_frames):
    from cupo import data

    train_dates = data.split_dates("train")
    train_df = data.load_split("train")
    candidates = real_frames[real_frames["date"].isin(train_dates)].sample(10, random_state=0)
    checked = 0
    for _, row in candidates.iterrows():
        cam = int(row["camera"])
        gt = frames.ground_truth(cam, row["date"], row["time"])
        if not gt:
            continue
        checked += 1
        assert set(gt.values()) <= {0, 1}
        assert set(gt) <= set(frames.slot_boxes(cam)["slot_id"])
        # Coincide con las etiquetas de los patches de train del mismo instante
        p = train_df[(train_df["camera"] == str(cam)) & (train_df["date"] == row["date"])
                     & (train_df["time"] == row["time"])]
        for slot, label in zip(p["slot_id"], p["label"]):
            assert gt.get(slot) == label
    assert checked >= 5


@real_data
@pytest.mark.parametrize("camera", range(1, 10))
def test_slot_boxes_inside_frame_real(camera):
    boxes = frames.slot_boxes(camera)
    assert len(boxes) > 0
    assert list(boxes.columns) == ["slot_id", "x", "y", "w", "h"]
    assert boxes["slot_id"].is_unique
    assert (boxes["x"] >= 0).all() and (boxes["y"] >= 0).all()
    assert (boxes["w"] > 0).all() and (boxes["h"] > 0).all()
    assert (boxes["x"] + boxes["w"] <= frames.FULL_SIZE[0]).all()
    assert (boxes["y"] + boxes["h"] <= frames.FULL_SIZE[1]).all()


@real_data
def test_crop_real_frame(real_frames):
    row = real_frames.iloc[0]
    boxes = frames.slot_boxes(int(row["camera"]))
    with Image.open(row["path"]) as im:
        assert im.size == frames.FULL_SIZE
        crops = frames.crop_slots(im, boxes)
    assert len(crops) == len(boxes)
    for crop, (w, h) in zip(crops, zip(boxes["w"], boxes["h"])):
        assert crop.size == (w, h) and crop.mode == "RGB"
