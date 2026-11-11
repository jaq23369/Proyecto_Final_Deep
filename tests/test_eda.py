"""Pruebas de scripts/eda.py y de su salida versionada en results/eda/.

- Unitarias: funciones puras del script con valores calculados a mano (no necesitan datos).
- Integración (marca ``data``): corre el script con una muestra pequeña sobre el dataset real
  y revisa archivos, números clave, reproducibilidad y coherencia con results/eda/.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cupo import paths

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "eda.py"
VERSIONED = REPO / "results" / "eda"

EXPECTED_PNGS = (
    "01_particiones.png",
    "02_balance_camara.png",
    "03_balance_clima.png",
    "04_fechas.png",
    "05_ocupacion_hora.png",
    "06_ocupacion_slot.png",
    "07_brillo_contraste.png",
    "08_brillo_hora.png",
    "09_atipicas.png",
    "10_mosaico_ejemplos.png",
    "11_frame_cajas.png",
)
# Conteos esperados de las particiones oficiales y de CNRPark A/B
EXPECTED_SIZES = {"train": 94493, "val": 18647, "test": 31825, "ood": 12584}
EXPECTED_BUSY_PCT = {"train": 50.4, "val": 71.9, "test": 57.4}
EXPECTED_LABEL_MISMATCH = 26

# Claves del JSON que se calculan con TODAS las filas (no dependen de la muestra ni de --full-hash)
FULL_DATA_KEYS = (
    "sizes", "val_rainy", "ood_weathers", "date_overlap_total", "shortcuts_val", "slot_extremes",
    "slot_rate_corr_train_val", "labels_nulls", "csv_used_nulls", "csv_unused_nulls", "label_mismatch",
    "label_mismatch_pct", "labels_missing_in_csv", "time_range",
)
# Campos de tiempo de ejecución que no deben contar al comparar corridas
TIMING_KEYS = {"elapsed", "elapsed_s", "runtime", "duration", "seconds", "timestamp"}

# Rutas locales del equipo que no deben aparecer en los textos generados
FORBIDDEN_CASE = re.compile(r"[A-Za-z]:[\\/]Us" r"ers[\\/]")
FORBIDDEN_NOCASE = re.compile(r"/ho" r"me/[^/\s]+/", re.IGNORECASE)

HAS_DATA = paths.split_file("train").is_file() and paths.metadata_csv().is_file()


def _real_data(fn):
    fn = pytest.mark.skipif(not HAS_DATA, reason="dataset no disponible en DATA_ROOT")(fn)
    return pytest.mark.data(fn)


def _load_eda():
    spec = importlib.util.spec_from_file_location("eda_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass necesita el módulo registrado
    spec.loader.exec_module(module)
    return module


eda = _load_eda()


def _forbidden_hits(text: str) -> list[str]:
    return FORBIDDEN_CASE.findall(text) + FORBIDDEN_NOCASE.findall(text)


def _png_text_hits(path: Path) -> list[str]:
    """Busca menciones prohibidas en los metadatos de texto del PNG (los píxeles comprimidos no cuentan)."""
    with Image.open(path) as im:
        meta = " ".join(f"{k} {v}" for k, v in {**im.info, **getattr(im, "text", {})}.items()
                        if isinstance(v, str))
    return _forbidden_hits(meta)


# ---------------------------------------------------------------------------
# Importación sin efectos secundarios
# ---------------------------------------------------------------------------

def test_import_does_not_touch_data_or_write_files(tmp_path):
    """Importar el script no lee datos, no imprime y no crea archivos (solo configura matplotlib y sys.path)."""
    env = dict(os.environ, DATA_ROOT=str(tmp_path / "no_existe"), MPLBACKEND="")
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('m', r'{SCRIPT}')\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['m'] = m; spec.loader.exec_module(m)\n"
        "import matplotlib; assert matplotlib.get_backend().lower() == 'agg'\n"
    )
    before = sorted(p.name for p in tmp_path.iterdir())
    res = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True,
                         timeout=120)
    assert res.returncode == 0, res.stderr
    assert res.stdout == ""
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert not (tmp_path / "no_existe").exists()


def test_script_only_imports_data_modules():
    """El EDA no carga modelos, entrenamiento ni evaluación: no influye en la selección."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    cupo_imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("cupo"):
            cupo_imports |= {f"{node.module}.{a.name}" for a in node.names}
        if isinstance(node, ast.Import):
            for a in node.names:
                assert not a.name.startswith("torch"), "el EDA no debería importar torch"
    assert cupo_imports == {"cupo.data", "cupo.frames", "cupo.paths",
                            "cupo.transforms.IMAGENET_MEAN", "cupo.transforms.IMAGENET_STD"}


def test_auc_is_only_computed_on_validation():
    """Toda llamada a roc_auc_score del script usa datos de validación."""
    src = SCRIPT.read_text(encoding="utf-8")
    calls = [ln for ln in src.splitlines() if "roc_auc_score(" in ln and "import" not in ln]
    assert calls, "se esperaban AUC de atajos/señales en val"
    for ln in calls:
        assert "val" in ln, ln


def test_frame_example_uses_train_dates_only():
    """El frame de ejemplo se elige sin fechas de test (list_frames con allow_test=False)."""
    src = SCRIPT.read_text(encoding="utf-8")
    assert "frames.list_frames(allow_test=False)" in src
    assert "allow_test=True" not in src.split("def pick_frame")[1].split("def frame_with_boxes")[0]


# ---------------------------------------------------------------------------
# Unitarias: formato y tablas
# ---------------------------------------------------------------------------

def test_fmt_values():
    assert eda._fmt(None) == ""
    assert eda._fmt(float("nan")) == ""
    assert eda._fmt(pd.NA) == ""
    assert eda._fmt(1234.56) == "1,234.6"
    assert eda._fmt(np.float32(0.25)) == "0.2"  # un decimal
    assert eda._fmt(np.int64(1234567)) == "1,234,567"
    assert eda._fmt(7) == "7"
    assert eda._fmt("SUNNY") == "SUNNY"


def test_md_table_exact():
    df = pd.DataFrame({"n": [1000, 5], "pct": [12.34, float("nan")]}, index=pd.Index(["a", "b"], name="k"))
    expected = "| k | n | pct |\n|---|---|---|\n| a | 1,000 | 12.3 |\n| b | 5 |  |"
    assert eda.md_table(df) == expected
    expected_noidx = "| n | pct |\n|---|---|\n| 1,000 | 12.3 |\n| 5 |  |"
    assert eda.md_table(df, index=False) == expected_noidx


def test_report_add_strips_and_ends_with_newline():
    rep = eda.Report()
    rep.add("\n\n## Sección\n\ntexto  \n\n")
    assert rep.sections == ["## Sección\n\ntexto\n"]
    assert rep.stats == {}


def test_hour_of():
    df = pd.DataFrame({"time": ["07:05", "18:12", "09:59"]})
    assert eda.hour_of(df).tolist() == [7, 18, 9]


def test_range():
    assert eda._range(pd.Series([1.4, 9.6, 5.0])) == "1-10"


def test_json_default():
    assert eda._json_default(np.int64(3)) == 3
    assert isinstance(eda._json_default(np.int64(3)), int)
    assert eda._json_default(np.float32(1.234567)) == pytest.approx(1.2346)
    assert eda._json_default(Path("a")) == "a"


def test_parse_args_defaults_and_values(tmp_path):
    a = eda.parse_args([])
    assert (a.sample_per_split, a.seed, a.out, a.full_hash) == (3000, 0, None, False)
    b = eda.parse_args(["--sample-per-split", "100", "--seed", "7", "--out", str(tmp_path), "--full-hash"])
    assert (b.sample_per_split, b.seed, b.out, b.full_hash) == (100, 7, tmp_path, True)


@pytest.mark.parametrize("n", ["0", "-5"])
def test_main_rejects_non_positive_sample_before_reading_data(n, tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "no_existe"))
    out = tmp_path / "out"
    with pytest.raises(SystemExit):
        eda.main(["--sample-per-split", n, "--out", str(out)])
    assert not out.exists()


# ---------------------------------------------------------------------------
# Unitarias: balance, espacios y atajos
# ---------------------------------------------------------------------------

def test_balance_hand_computed():
    df = pd.DataFrame({"camera": ["1", "1", "1", "1", "2", "2"], "label": [1, 1, 1, 0, 0, 0]})
    g = eda._balance(df, "camera")
    assert g.loc["1", "n"] == 4 and g.loc["1", "ocupados"] == 3
    assert g.loc["1", "% ocupado"] == pytest.approx(75.0)
    assert g.loc["2", "n"] == 2 and g.loc["2", "ocupados"] == 0
    assert g.loc["2", "% ocupado"] == pytest.approx(0.0)


def _slots_df(spec: dict[str, tuple[int, int]]) -> pd.DataFrame:
    """spec: slot -> (n, ocupados)."""
    rows = []
    for slot, (n, busy) in spec.items():
        rows += [{"slot_id": slot, "label": int(i < busy)} for i in range(n)]
    return pd.DataFrame(rows)


def test_slot_rates_and_extreme_summary_hand_computed():
    # A siempre ocupado, B siempre libre, C 85 % ocupado, D excluido (n < 20), E 50 %
    df = _slots_df({"A": (20, 20), "B": (20, 0), "C": (20, 17), "D": (10, 10), "E": (20, 10)})
    rates = eda._slot_rates(df)
    assert rates.loc["C", "rate"] == pytest.approx(0.85)
    assert rates.loc["D", "n"] == 10
    s = eda._extreme_summary(df)
    assert s["espacios (n≥20)"] == 4
    assert s["ocupados ≥95%"] == 1 and s["libres ≥95%"] == 1
    assert s["% patches (umbral 95%)"] == pytest.approx(100 * 40 / 80)
    assert s["ocupados ≥80%"] == 2 and s["libres ≥80%"] == 1
    assert s["% patches (umbral 80%)"] == pytest.approx(100 * 60 / 80)


@pytest.mark.xfail(strict=True, reason="1 - 0.8 en coma flotante es 0.19999...: un espacio libre exactamente "
                                       "el 80 % del tiempo no cuenta como 'libre ≥80%' (eda.py:340)")
def test_extreme_summary_is_symmetric_at_the_threshold():
    # Ocupado exactamente 80 % (cuenta) frente a libre exactamente 80 % (debería contar igual)
    df = _slots_df({"busy80": (20, 16), "free80": (20, 4)})
    s = eda._extreme_summary(df)
    assert s["ocupados ≥80%"] == 1
    assert s["libres ≥80%"] == 1


def test_prior_falls_back_to_global_train_rate():
    train = pd.DataFrame({"slot_id": ["a", "a", "b", "b"], "label": [1, 1, 0, 1]})
    target = pd.DataFrame({"slot_id": ["a", "b", "z"]}, index=[10, 11, 12])
    prior = eda._prior(train, target, ["slot_id"])
    assert prior.tolist() == pytest.approx([1.0, 0.5, 0.75])
    assert list(prior.index) == [10, 11, 12]


def test_shortcut_table_hand_computed():
    # Espacio s1: 3/4 ocupado; s2: 1/4. Hora y cámara constantes (tasa 0.5 para todas las filas).
    train = pd.DataFrame({
        "slot_id": ["s1"] * 4 + ["s2"] * 4,
        "hour": [10] * 8,
        "camera": ["1"] * 8,
        "label": [1, 1, 1, 0, 0, 0, 0, 1],
    })
    val = pd.DataFrame({"slot_id": ["s1", "s1", "s2", "s2"], "hour": [10] * 4, "camera": ["1"] * 4,
                        "label": [1, 0, 0, 0]})
    tab = eda.shortcut_table(train, val).set_index("predictor (tasa de train por…)")
    # Puntajes [.75, .75, .25, .25] vs [1, 0, 0, 0]: AUC = (0.5 + 1 + 1) / 3
    assert tab.loc["espacio", "AUC val"] == pytest.approx(2.5 / 3)
    assert tab.loc["espacio", "accuracy val (umbral 0.5)"] == pytest.approx(75.0)
    assert tab.loc["hora", "AUC val"] == pytest.approx(0.5)
    # Puntaje constante 0.5 >= 0.5 -> predice ocupado siempre: acierta 1 de 4
    assert tab.loc["hora", "accuracy val (umbral 0.5)"] == pytest.approx(25.0)
    assert tab.loc["cámara", "AUC val"] == pytest.approx(0.5)
    assert tab.loc["espacio + hora", "AUC val"] == pytest.approx(2.5 / 3)
    major = "siempre 'ocupado' (mayoritaria de train)"
    assert tab.loc[major, "AUC val"] == 0.5
    assert tab.loc[major, "accuracy val (umbral 0.5)"] == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Unitarias: muestreo, lectura de imágenes y atípicas
# ---------------------------------------------------------------------------

def _strat_df() -> pd.DataFrame:
    rows = []
    for w, lab, n in (("SUNNY", 0, 30), ("SUNNY", 1, 30), ("RAINY", 0, 20), ("RAINY", 1, 20), ("OVERCAST", 1, 2)):
        rows += [{"weather": w, "label": lab, "id": f"{w}{lab}_{i}"} for i in range(n)]
    return pd.DataFrame(rows)


def test_stratified_sample_returns_everything_when_n_is_large():
    df = _strat_df()
    out = eda.stratified_sample(df, len(df), seed=0)
    assert len(out) == len(df)
    assert list(out.index) == list(range(len(df)))


def test_stratified_sample_proportional_and_reproducible():
    df = _strat_df()  # 102 filas
    out = eda.stratified_sample(df, 10, seed=0)
    counts = out.groupby(["weather", "label"]).size().to_dict()
    # frac = 10/102: 30 -> round(2.94) = 3, 20 -> round(1.96) = 2, 2 -> max(1, round(0.196)) = 1
    assert counts == {("SUNNY", 0): 3, ("SUNNY", 1): 3, ("RAINY", 0): 2, ("RAINY", 1): 2, ("OVERCAST", 1): 1}
    again = eda.stratified_sample(df, 10, seed=0)
    pd.testing.assert_frame_equal(out, again)
    other = eda.stratified_sample(df, 10, seed=1)
    assert set(other["id"]) != set(out["id"])
    assert out["id"].is_unique


def test_image_record_values_hand_computed(tmp_path):
    gray = tmp_path / "gris.png"
    Image.new("RGB", (150, 150), (100, 100, 100)).save(gray)
    rec = eda.image_record(str(gray))
    assert rec["ok"] is True and rec["error"] == ""
    assert (rec["width"], rec["height"]) == (150, 150)
    assert rec["brightness"] == pytest.approx(100.0)
    assert rec["contrast"] == pytest.approx(0.0)

    # Mitad negra y mitad blanca: media 127.5, desviación 127.5
    arr = np.zeros((20, 30, 3), dtype=np.uint8)
    arr[:, 15:] = 255
    half = tmp_path / "mitad.png"
    Image.fromarray(arr).save(half)
    rec2 = eda.image_record(str(half))
    assert (rec2["width"], rec2["height"]) == (30, 20)
    assert rec2["brightness"] == pytest.approx(127.5)
    assert rec2["contrast"] == pytest.approx(127.5)


def test_image_record_hashes(tmp_path):
    a, b, c = tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.bmp"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(a)
    b.write_bytes(a.read_bytes())
    Image.new("RGB", (8, 8), (10, 20, 30)).save(c)  # mismos píxeles, otro formato de archivo
    ra, rb, rc = (eda.image_record(str(p)) for p in (a, b, c))
    assert ra["md5_file"] == rb["md5_file"]
    assert ra["md5_pixels"] == rb["md5_pixels"] == rc["md5_pixels"]
    assert ra["md5_file"] != rc["md5_file"]


def test_image_record_corrupt_and_missing(tmp_path):
    bad = tmp_path / "roto.jpg"
    bad.write_bytes(b"esto no es una imagen")
    rec = eda.image_record(str(bad))
    assert rec["ok"] is False
    assert rec["md5_file"] is not None and rec["md5_pixels"] is None
    assert rec["error"] and np.isnan(rec["brightness"])

    missing = eda.image_record(str(tmp_path / "no_existe.jpg"))
    assert missing["ok"] is False and missing["md5_file"] is None
    assert missing["error"].startswith("FileNotFoundError")


def test_flag_outliers_hand_computed():
    bright = [98, 99, 100, 101, 102, 250]
    rows = [{"split": "train", "weather": "SUNNY", "brightness": float(b), "ok": True} for b in bright]
    rows += [{"split": "val", "weather": "SUNNY", "brightness": v, "ok": True} for v in (5.0, 30.0, 30.0, 30.0)]
    rows.append({"split": "train", "weather": "SUNNY", "brightness": np.nan, "ok": False})
    out = eda.flag_outliers(pd.DataFrame(rows))

    # train/SUNNY: mediana 100.5, MAD 1.5 -> z(250) = 0.6745 * 149.5 / 1.5
    assert out.loc[5, "robust_z"] == pytest.approx(0.6745 * 149.5 / 1.5)
    assert out.loc[0, "robust_z"] == pytest.approx(0.6745 * -2.5 / 1.5)
    assert out["atypical"].tolist()[:6] == [False] * 5 + [True]
    assert bool(out.loc[5, "overexposed"]) and not out.loc[:4, "overexposed"].any()

    # val/SUNNY: mediana 30, MAD 0 -> z indefinido (NaN), no se marca atípica
    val = out[out["split"] == "val"]
    assert val["robust_z"].isna().all() and not val["atypical"].any()
    assert val["dark"].tolist() == [True, True, True, True]
    assert val["black"].tolist() == [True, False, False, False]

    # Fila ilegible: sin z y sin marcas
    last = out.iloc[-1]
    assert np.isnan(last["robust_z"])
    assert not (last["dark"] or last["black"] or last["overexposed"] or last["atypical"])


# ---------------------------------------------------------------------------
# Unitarias: textos generados
# ---------------------------------------------------------------------------

def test_bright_text_cases():
    assert eda._bright_text(pd.DataFrame(columns=["subset", "label", "time"])) == "ninguna."
    over = pd.DataFrame({
        "subset": ["EXT", "EXT", "CNRPark"],
        "label": [1, 1, 0],
        "time": ["13:00", "12:00", "08:30"],
    })
    txt = eda._bright_text(over)
    assert "EXT: 2 (100 % ocupadas; horas 12:00, 13:00)" in txt
    assert "autos claros" in txt
    assert "ood: 1 (0 % ocupadas; horas 08:30)" in txt
    assert "pavimento libre" in txt
    assert txt.endswith(".")


def test_shared_text_black_contradictory_hand_computed():
    rep = eda.Report()
    shared = pd.DataFrame([{"md5": "abc", "archivos train": 30, "archivos test": 8, "ocupados": 3, "brillo": 0.0}])
    txt = eda._shared_text(shared, rep)
    assert rep.stats["dup_full_files"] == 38
    assert rep.stats["dup_full_busy"] == 3
    assert rep.stats["dup_full_contradictory"] == 1
    assert "completamente negros" in txt
    assert "Los 38 archivos idénticos" in txt
    assert "(3 ocupados y 35 libres)" in txt


def test_shared_text_non_black_is_flagged_as_possible_leak():
    rep = eda.Report()
    shared = pd.DataFrame([
        {"md5": "a", "archivos train": 1, "archivos test": 1, "ocupados": 2, "brillo": 120.0},
        {"md5": "b", "archivos train": 2, "archivos test": 1, "ocupados": 0, "brillo": 3.0},
    ])
    txt = eda._shared_text(shared, rep)
    assert "1 de 2 contenidos repetidos son patches negros" in txt
    assert "posible fuga" in txt
    assert rep.stats["dup_full_contradictory"] == 0


# ---------------------------------------------------------------------------
# Código fuente sin menciones prohibidas
# ---------------------------------------------------------------------------

def test_script_source_has_no_forbidden_mentions():
    hits = _forbidden_hits(SCRIPT.read_text(encoding="utf-8"))
    assert hits == [], hits


# ---------------------------------------------------------------------------
# Salida versionada en results/eda/ (no necesita datos)
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_versioned_outputs_complete():
    names = {p.name for p in VERSIONED.iterdir() if p.is_file()}
    assert set(EXPECTED_PNGS) <= names
    assert {"eda_resumen.md", "eda_stats.json"} <= names
    assert sorted(n for n in names if n.endswith(".png")) == sorted(EXPECTED_PNGS)
    for name in EXPECTED_PNGS:
        with Image.open(VERSIONED / name) as im:
            im.verify()
        with Image.open(VERSIONED / name) as im:
            assert im.width >= 300 and im.height >= 200


def test_versioned_json_matches_plan_counts():
    s = _read_json(VERSIONED / "eda_stats.json")
    assert {k: v["n"] for k, v in s["sizes"].items()} == EXPECTED_SIZES
    for split, pct in EXPECTED_BUSY_PCT.items():
        assert s["sizes"][split]["busy_pct"] == pytest.approx(pct, abs=0.05)
    assert s["val_rainy"] == 0
    assert s["ood_weathers"] == ["SUNNY"]
    assert s["date_overlap_total"] == 0
    assert s["label_mismatch"] == EXPECTED_LABEL_MISMATCH
    assert s["labels_nulls"] == 0 and s["csv_used_nulls"] == 0
    assert s["labels_missing_in_csv"] == 0
    # 164 espacios únicos de EXT
    for split in ("train", "val", "test"):
        assert s["slot_extremes"][split]["espacios (n≥20)"] == 164


def test_versioned_json_consistent_with_its_command_line():
    md = (VERSIONED / "eda_resumen.md").read_text(encoding="utf-8")
    s = _read_json(VERSIONED / "eda_stats.json")
    m = re.search(r"--sample-per-split (\d+) --seed (\d+)( --full-hash)?", md)
    assert m, "el resumen debe documentar el comando con que se generó"
    n = int(m.group(1))
    assert all(v["muestra"] <= n for v in s["image_flags"].values())
    assert sum(v["muestra"] for v in s["image_flags"].values()) >= 4 * n - 8  # redondeo del estratificado
    assert ("dup_train_test_full" in s) == bool(m.group(3))


def test_versioned_markdown_consistent_with_json():
    md = (VERSIONED / "eda_resumen.md").read_text(encoding="utf-8")
    s = _read_json(VERSIONED / "eda_stats.json")
    for split, v in s["sizes"].items():
        assert f"{split} {v['n']:,} ({v['busy_pct']:.1f} % ocupado)" in md
    assert f"Val no tiene lluvia ({s['val_rainy']} patches con lluvia)" in md
    assert f"Fechas compartidas entre particiones: {s['date_overlap_total']}" in md
    assert f"{s['label_mismatch']} patches ({s['label_mismatch_pct']:.3f} %)" in md
    assert f"de {s['time_range'][0]} a {s['time_range'][1]}" in md
    for name in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md):
        assert (VERSIONED / name).is_file(), name
    assert set(re.findall(r"!\[[^\]]*\]\(([^)]+\.png)\)", md)) == set(EXPECTED_PNGS)


def test_versioned_markdown_has_key_sections():
    _check_sections((VERSIONED / "eda_resumen.md").read_text(encoding="utf-8"))


def test_versioned_outputs_have_no_forbidden_mentions():
    for name in ("eda_resumen.md", "eda_stats.json"):
        hits = _forbidden_hits((VERSIONED / name).read_text(encoding="utf-8"))
        assert hits == [], (name, hits)
    for name in EXPECTED_PNGS:
        assert _png_text_hits(VERSIONED / name) == [], name


@pytest.mark.xfail(strict=True, reason="eda_stats.json escribe NaN (no es JSON estándar) en "
                                       "'% ocupado entre las oscuras' cuando no hay oscuras (eda.py:562 y 1102)")
def test_versioned_json_is_strict_json():
    def reject(token):
        raise ValueError(f"constante no estándar en JSON: {token}")

    json.loads((VERSIONED / "eda_stats.json").read_text(encoding="utf-8"), parse_constant=reject)


def _check_sections(md: str) -> None:
    assert md.startswith("# EDA de CNRPark+EXT")
    assert "## Hallazgos clave" in md
    assert "### Valores nulos y consistencia de etiquetas" in md
    assert "Nulos en LABELS:" in md
    assert "## 8. Representatividad" in md
    assert "## 9. Preprocesamiento derivado del EDA" in md
    assert "## 10. Licencia y privacidad" in md
    assert "Open Database License (ODbL) v1.0" in md
    assert "Amato" in md  # citas [1] y [2]
    for heading in ("## 1. Particiones", "## 2. Balance de clases por cámara y por clima",
                    "## 3. Distribución temporal", "## 5. Estadísticas de imagen",
                    "## 6. Calidad de los archivos", "## 7. Ejemplos"):
        assert heading in md, heading
    assert "val no tiene lluvia" in md.lower()
    assert "test y ood se usan solo para estadística descriptiva" in md


# ---------------------------------------------------------------------------
# Integración con el dataset real: prueba de humo y reproducibilidad
# ---------------------------------------------------------------------------

def _run_eda(out: Path, seed: int = 0, n: int = 100) -> subprocess.CompletedProcess:
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "--sample-per-split", str(n), "--seed", str(seed), "--out", str(out)],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
    )
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    return res


@pytest.fixture(scope="module")
def eda_runs(tmp_path_factory):
    if not HAS_DATA:
        pytest.skip("dataset no disponible en DATA_ROOT")
    first = tmp_path_factory.mktemp("eda_a")
    second = tmp_path_factory.mktemp("eda_b")
    log = _run_eda(first)
    _run_eda(second)
    return first, second, log


@_real_data
def test_smoke_generates_all_outputs(eda_runs):
    out, _, log = eda_runs
    names = {p.name for p in out.iterdir()}
    assert names == set(EXPECTED_PNGS) | {"eda_resumen.md", "eda_stats.json"}
    for name in EXPECTED_PNGS:
        assert (out / name).stat().st_size > 5_000, name
        with Image.open(out / name) as im:
            im.verify()
    assert "Particiones cargadas" in log.stdout
    # La salida va solo a --out: no toca results/eda versionado
    assert not (out / "results").exists()


@_real_data
def test_smoke_key_numbers(eda_runs):
    s = _read_json(eda_runs[0] / "eda_stats.json")
    assert {k: v["n"] for k, v in s["sizes"].items()} == EXPECTED_SIZES
    for split, pct in EXPECTED_BUSY_PCT.items():
        assert s["sizes"][split]["busy_pct"] == pytest.approx(pct, abs=0.05)
    assert s["val_rainy"] == 0
    assert s["ood_weathers"] == ["SUNNY"]
    assert s["date_overlap_total"] == 0
    assert s["label_mismatch"] == EXPECTED_LABEL_MISMATCH
    assert s["labels_nulls"] == 0
    assert s["csv_used_nulls"] == 0
    assert s["labels_missing_in_csv"] == 0
    assert s["corrupt_in_sample"] == 0
    assert "dup_train_test_full" not in s  # sin --full-hash
    for v in s["image_flags"].values():
        assert 90 <= v["muestra"] <= 100


@_real_data
def test_smoke_markdown_sections(eda_runs):
    md = (eda_runs[0] / "eda_resumen.md").read_text(encoding="utf-8")
    _check_sections(md)
    assert "--sample-per-split 100 --seed 0`" in md
    for name in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md):
        assert (eda_runs[0] / name).is_file(), name


@_real_data
def test_smoke_outputs_have_no_forbidden_mentions(eda_runs):
    out = eda_runs[0]
    for name in ("eda_resumen.md", "eda_stats.json"):
        hits = _forbidden_hits((out / name).read_text(encoding="utf-8"))
        assert hits == [], (name, hits)
    for name in EXPECTED_PNGS:
        assert _png_text_hits(out / name) == [], name


def _strip_timing(stats: dict) -> dict:
    return {k: v for k, v in stats.items() if k not in TIMING_KEYS}


def _strip_md_timing(md: str) -> str:
    return re.sub(r"\(tiempo de ejecución: [^)]*\)", "(tiempo)", md)


@_real_data
def test_reproducible_with_same_seed(eda_runs):
    a, b, _ = eda_runs
    ja = _strip_timing(json.loads((a / "eda_stats.json").read_text(encoding="utf-8")))
    jb = _strip_timing(json.loads((b / "eda_stats.json").read_text(encoding="utf-8")))
    # Comparación por texto: NaN != NaN rompería la igualdad de diccionarios
    assert json.dumps(ja, sort_keys=True) == json.dumps(jb, sort_keys=True)
    ma = _strip_md_timing((a / "eda_resumen.md").read_text(encoding="utf-8"))
    mb = _strip_md_timing((b / "eda_resumen.md").read_text(encoding="utf-8"))
    assert ma == mb


@_real_data
def test_reproducible_figures_with_same_seed(eda_runs):
    a, b, _ = eda_runs
    for name in EXPECTED_PNGS:
        with Image.open(a / name) as ia, Image.open(b / name) as ib:
            assert ia.size == ib.size, name
            assert np.array_equal(np.asarray(ia), np.asarray(ib)), name


@_real_data
def test_versioned_json_coherent_with_fresh_run(eda_runs):
    """Los números calculados con todas las filas coinciden entre results/eda/ y una corrida nueva."""
    fresh = _read_json(eda_runs[0] / "eda_stats.json")
    versioned = _read_json(VERSIONED / "eda_stats.json")
    for key in FULL_DATA_KEYS:
        assert versioned[key] == fresh[key], key
