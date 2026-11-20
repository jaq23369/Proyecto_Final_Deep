"""Pruebas de src/cupo/explain.py: decisión, tabla de un split, selección de instancias,
oclusión, fidelidad, concentración en el centro, muestra estratificada y mapa HOG del B1.

Casi todo usa datos sintéticos calculados a mano. Las pruebas marcadas ``data`` usan el run
real del B1 y una imagen de validación (nunca test ni CNRPark).
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from cupo import explain as E  # noqa: E402
from cupo import paths  # noqa: E402
from cupo.runs import LoadedRun, save_npz  # noqa: E402

from run_helpers import HAS_DATA  # noqa: E402

MOTIVOS = [E.HIT_BUSY, E.HIT_FREE, E.ERR_BUSY_AS_FREE, E.NEAR_THRESHOLD, E.RAIN_HIT,
           E.ERR_FREE_AS_BUSY]
OUT_COLUMNS = ["motivo", "row", "path", "y", "p", "decision", "camera", "weather", "date",
               "time", "slot_id"]
TABLE_COLUMNS = ["path", "y", "p", "camera", "weather", "date", "time", "slot_id"]


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _run(tmp_path: Path, config=None, thresholds=None, temperature=1.0,
         calibrated=False) -> LoadedRun:
    return LoadedRun(run_dir=tmp_path, config=config or {}, kind="torch", model=None,
                     input_size=224, temperature=temperature,
                     thresholds=thresholds if thresholds is not None else
                     {"comercial": {"t_L": 0.2, "t_H": 0.8}},
                     calibrated=calibrated)


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------

def test_decision_three_states_and_exact_borders():
    p = np.array([0.0, 0.1999, 0.2, 0.5, 0.7999, 0.8, 1.0])
    got = E.decision(p, 0.2, 0.8)
    assert got.tolist() == ["libre", "libre", "por verificar", "por verificar",
                            "por verificar", "ocupado", "ocupado"]


def test_decision_accepts_list_and_equal_thresholds():
    # con t_L == t_H no hay zona "por verificar"
    assert E.decision([0.3, 0.5, 0.7], 0.5, 0.5).tolist() == ["libre", "ocupado", "ocupado"]


# ---------------------------------------------------------------------------
# operating_point
# ---------------------------------------------------------------------------

THR = {"comercial": {"t_L": 0.1, "t_H": 0.9}, "uvg": {"t_L": 0.3, "t_H": 0.6}}


def test_operating_point_uses_config_scenario(tmp_path):
    run = _run(tmp_path, config={"select": {"scenario": "uvg"}}, thresholds=THR, temperature=1.7)
    op = E.operating_point(run)
    assert op == E.Operating("uvg", 0.3, 0.6, 1.7)
    assert all(isinstance(v, float) for v in (op.t_L, op.t_H, op.temperature))


def test_operating_point_default_is_comercial(tmp_path):
    op = E.operating_point(_run(tmp_path, config={}, thresholds=THR))
    assert (op.scenario, op.t_L, op.t_H, op.temperature) == ("comercial", 0.1, 0.9, 1.0)


def test_operating_point_falls_back_to_first_scenario(tmp_path):
    thr = {"uvg": {"t_L": 0.3, "t_H": 0.6}, "comercial": {"t_L": 0.1, "t_H": 0.9}}
    op = E.operating_point(_run(tmp_path, config={"select": {"scenario": "inexistente"}},
                                thresholds=thr))
    assert (op.scenario, op.t_L, op.t_H) == ("uvg", 0.3, 0.6)
    # sin escenario en config y sin "comercial" en los umbrales: también el primero
    op2 = E.operating_point(_run(tmp_path, config={}, thresholds={"uvg": thr["uvg"]}))
    assert op2.scenario == "uvg"


def test_operating_point_casts_strings_to_float(tmp_path):
    run = _run(tmp_path, thresholds={"comercial": {"t_L": "0.25", "t_H": 1}}, temperature=2)
    op = E.operating_point(run)
    assert (op.t_L, op.t_H, op.temperature) == (0.25, 1.0, 2.0)
    assert isinstance(op.t_H, float) and isinstance(op.temperature, float)


# ---------------------------------------------------------------------------
# load_split_table
# ---------------------------------------------------------------------------

def _npz_arrays(with_logits: bool):
    arrays = {
        "y": np.array([1, 0, 1], dtype=np.int8),
        "p": np.array([0.11, 0.22, 0.33]),  # p guardada (sin calibrar)
        "camera": np.array([1, 2, 3]),
        "weather": np.array(["SUNNY", "RAINY", "OVERCAST"]),
        "date": np.array(["2015-11-12", "2015-11-13", "2015-11-14"]),
        "time": np.array(["09.00", "10.30", "12.47"]),
        "slot_id": np.array([10, 20, 30]),
        "path": np.array(["a.jpg", "b.jpg", "c.jpg"]),
    }
    if with_logits:
        arrays["logits"] = np.array([[0.0, 2.0], [1.0, 1.0], [3.0, -1.0]], dtype=np.float32)
    return arrays


def test_load_split_table_recomputes_p_with_temperature(tmp_path):
    save_npz(tmp_path / "val_preds.npz", **_npz_arrays(with_logits=True))
    run = _run(tmp_path, temperature=2.0, calibrated=True)
    table = E.load_split_table(run, "val")
    # T = 2: z = (l1 - l0) / 2 → (1, 0, -2)
    expected = [_sigmoid(1.0), 0.5, _sigmoid(-2.0)]
    np.testing.assert_allclose(table.p.to_numpy(), expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(expected, [0.7310585786300049, 0.5, 0.11920292202211755])
    # la p guardada se ignora cuando hay logits
    assert not np.allclose(table.p.to_numpy(), [0.11, 0.22, 0.33])


def test_load_split_table_temperature_one_without_calibration(tmp_path):
    save_npz(tmp_path / "ood_preds.npz", **_npz_arrays(with_logits=True))
    table = E.load_split_table(_run(tmp_path, temperature=1.0), "ood")
    np.testing.assert_allclose(table.p, [_sigmoid(2.0), 0.5, _sigmoid(-4.0)], atol=1e-12)


def test_load_split_table_without_logits_uses_stored_p(tmp_path):
    save_npz(tmp_path / "val_preds.npz", **_npz_arrays(with_logits=False))
    table = E.load_split_table(_run(tmp_path, calibrated=False), "val")
    np.testing.assert_allclose(table.p, [0.11, 0.22, 0.33])


def test_load_split_table_without_logits_calibrated_fails(tmp_path):
    save_npz(tmp_path / "val_preds.npz", **_npz_arrays(with_logits=False))
    with pytest.raises(ValueError, match="logits"):
        E.load_split_table(_run(tmp_path, temperature=1.5, calibrated=True), "val")


def test_load_split_table_columns_and_types(tmp_path):
    save_npz(tmp_path / "val_preds.npz", **_npz_arrays(with_logits=True))
    table = E.load_split_table(_run(tmp_path, temperature=2.0, calibrated=True), "val")
    assert list(table.columns) == TABLE_COLUMNS
    assert len(table) == 3
    assert pd.api.types.is_integer_dtype(table.y)
    assert table.y.tolist() == [1, 0, 1]
    assert pd.api.types.is_float_dtype(table.p)
    assert table.path.tolist() == ["a.jpg", "b.jpg", "c.jpg"]
    assert table.weather.tolist() == ["SUNNY", "RAINY", "OVERCAST"]
    assert table.camera.tolist() == [1, 2, 3]
    assert table.date.tolist() == ["2015-11-12", "2015-11-13", "2015-11-14"]
    assert table.time.tolist() == ["09.00", "10.30", "12.47"]
    assert table.slot_id.tolist() == [10, 20, 30]


def test_load_split_table_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        E.load_split_table(_run(tmp_path), "val")


# ---------------------------------------------------------------------------
# select_instances
# ---------------------------------------------------------------------------

T_L, T_H = 0.2, 0.8

# (clave, y, p, cámara, clima). Cada regla tiene candidatos conocidos y distractores.
ROWS = [
    # regla 1: y=1, p >= 0.999, SUNNY
    ("busy_a", 1, 0.9995, 1, "SUNNY"),
    ("busy_b", 1, 0.999, 2, "SUNNY"),       # borde exacto: sí es candidato
    ("busy_c", 1, 0.99999, 3, "SUNNY"),
    ("busy_low", 1, 0.9989, 1, "SUNNY"),    # distractor: p por debajo
    ("busy_cloud", 1, 0.9999, 2, "OVERCAST"),  # distractor: no soleado ni lluvia
    # regla 2: y=0, p < 0.002, SUNNY
    ("free_a", 0, 0.001, 1, "SUNNY"),
    ("free_b", 0, 0.0001, 4, "SUNNY"),
    ("free_border", 0, 0.002, 2, "SUNNY"),  # distractor: borde exacto, no cumple p < 0.002
    ("free_cloud", 0, 0.0005, 3, "OVERCAST"),  # distractor: no soleado
    # regla 3: y=1 con decisión libre; gana la menor p
    ("err_bf_1", 1, 0.15, 4, "SUNNY"),
    ("err_bf_min", 1, 0.01, 5, "OVERCAST"),
    ("err_bf_3", 1, 0.19, 6, "RAINY"),
    ("busy_at_tL", 1, 0.2, 1, "SUNNY"),     # distractor: p == t_L → por verificar
    # regla 4: el más cercano a t_H en log-odds (0.70); en escala lineal ganaría 0.88
    ("near_lo", 0, 0.70, 2, "OVERCAST"),
    ("near_hi", 0, 0.88, 3, "OVERCAST"),
    # regla 5: lluvia, y=1, p >= 0.999, cámara distinta a la del error (5)
    ("rain_a", 1, 0.9999, 6, "RAINY"),
    ("rain_b", 1, 0.9995, 7, "RAINY"),
    ("rain_same_cam", 1, 0.99999, 5, "RAINY"),  # distractor: misma cámara que el error
    ("rain_low", 1, 0.99, 6, "RAINY"),          # distractor: p por debajo
    # regla 6: y=0 con decisión ocupado; gana la mayor p
    ("err_fb_1", 0, 0.95, 1, "OVERCAST"),
    ("err_fb_rain", 0, 0.9999, 6, "RAINY"),
    ("err_fb_max", 0, 0.99995, 2, "SUNNY"),
]
# fila extra para el borde p == t_H (no está en la tabla base: sería la más cercana a t_H)
EXTRA = ("free_at_tH", 0, 0.8, 3, "SUNNY")
ROWS_ALL = ROWS + [EXTRA]
KEYS = [r[0] for r in ROWS]
# índices no consecutivos para comprobar que ``row`` es la etiqueta original
INDEX = [1000 + 7 * i for i in range(len(ROWS_ALL))]
LABEL = dict(zip([r[0] for r in ROWS_ALL], INDEX))


def _table(keys=None) -> pd.DataFrame:
    if keys is None:
        keys = KEYS
    rows = [r for r in ROWS_ALL if r[0] in keys]
    idx = [LABEL[r[0]] for r in rows]
    return pd.DataFrame({
        "path": [f"{r[0]}.jpg" for r in rows],
        "y": np.array([r[1] for r in rows], dtype=int),
        "p": np.array([r[2] for r in rows], dtype=float),
        "camera": np.array([r[3] for r in rows], dtype=int),
        "weather": [r[4] for r in rows],
        "date": ["2015-11-20"] * len(rows),
        "time": ["10.00"] * len(rows),
        "slot_id": np.arange(len(rows)),
    }, index=idx)


def _without(*drop) -> pd.DataFrame:
    return _table([k for k in KEYS if k not in drop])


def _logit(p):
    return math.log(p / (1 - p))


def test_synthetic_table_near_threshold_is_by_log_odds():
    # comprobación a mano del diseño de la tabla
    gap_lo = abs(_logit(0.70) - _logit(T_H))   # 0.539
    gap_hi = abs(_logit(0.88) - _logit(T_H))   # 0.606
    assert gap_lo < gap_hi and abs(0.88 - T_H) < abs(0.70 - T_H)
    gaps = {k: abs(_logit(p) - _logit(T_H)) for k, _, p, _, _ in ROWS}
    assert min(gaps, key=gaps.get) == "near_lo"


def test_select_instances_order_columns_and_rules():
    table = _table()
    out = E.select_instances(table, T_L, T_H, seed=42)
    assert list(out.columns) == OUT_COLUMNS
    assert out.motivo.tolist() == MOTIVOS
    by = out.set_index("motivo")

    # cada fila coincide con la fila original de la tabla
    for _, r in out.iterrows():
        orig = table.loc[r.row]
        for col in TABLE_COLUMNS:
            assert r[col] == orig[col], (r.motivo, col)
        assert r.decision == E.decision([orig.p], T_L, T_H)[0]

    r1 = by.loc[E.HIT_BUSY]
    assert r1.row in {LABEL["busy_a"], LABEL["busy_b"], LABEL["busy_c"]}
    assert r1.y == 1 and r1.p >= 0.999 and r1.weather == "SUNNY" and r1.decision == "ocupado"

    r2 = by.loc[E.HIT_FREE]
    assert r2.row in {LABEL["free_a"], LABEL["free_b"]}
    assert r2.y == 0 and r2.p < 0.002 and r2.weather == "SUNNY" and r2.decision == "libre"

    r3 = by.loc[E.ERR_BUSY_AS_FREE]
    assert r3.row == LABEL["err_bf_min"]
    assert r3.y == 1 and r3.decision == "libre" and r3.p == 0.01 and r3.camera == 5

    r4 = by.loc[E.NEAR_THRESHOLD]
    assert r4.row == LABEL["near_lo"] and r4.p == 0.70

    r5 = by.loc[E.RAIN_HIT]
    assert r5.row in {LABEL["rain_a"], LABEL["rain_b"]}
    assert r5.weather == "RAINY" and r5.y == 1 and r5.p >= 0.999 and r5.camera != r3.camera

    r6 = by.loc[E.ERR_FREE_AS_BUSY]
    assert r6.row == LABEL["err_fb_max"]
    assert r6.y == 0 and r6.decision == "ocupado" and r6.p == 0.99995


def test_select_instances_same_seed_same_result():
    a = E.select_instances(_table(), T_L, T_H, seed=7)
    b = E.select_instances(_table(), T_L, T_H, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_select_instances_seed_changes_only_random_rules():
    picks = {m: set() for m in MOTIVOS}
    for seed in range(30):
        out = E.select_instances(_table(), T_L, T_H, seed=seed)
        for m, row in zip(out.motivo, out.row):
            picks[m].add(int(row))
    # reglas con azar: con 30 semillas aparecen varios candidatos (y solo candidatos)
    assert len(picks[E.HIT_BUSY]) > 1
    assert picks[E.HIT_BUSY] <= {LABEL["busy_a"], LABEL["busy_b"], LABEL["busy_c"]}
    assert len(picks[E.HIT_FREE]) > 1
    assert picks[E.HIT_FREE] <= {LABEL["free_a"], LABEL["free_b"]}
    assert picks[E.RAIN_HIT] == {LABEL["rain_a"], LABEL["rain_b"]}
    # reglas deterministas: siempre la misma fila
    assert picks[E.ERR_BUSY_AS_FREE] == {LABEL["err_bf_min"]}
    assert picks[E.NEAR_THRESHOLD] == {LABEL["near_lo"]}
    assert picks[E.ERR_FREE_AS_BUSY] == {LABEL["err_fb_max"]}


def test_select_instances_rain_excludes_error_camera():
    # si el único caso de lluvia está en la cámara del error, no hay candidatos
    with pytest.raises(ValueError):
        E.select_instances(_without("rain_a", "rain_b"), T_L, T_H)
    # si cambia el error (otra cámara), el caso de lluvia de la cámara 5 vuelve a valer
    table = _without("rain_a", "rain_b", "err_bf_min")
    out = E.select_instances(table, T_L, T_H).set_index("motivo")
    assert out.loc[E.ERR_BUSY_AS_FREE].row == LABEL["err_bf_1"]  # p = 0.15, cámara 4
    assert out.loc[E.RAIN_HIT].row == LABEL["rain_same_cam"]


def test_select_instances_p_equal_t_h_counts_as_busy_decision():
    # sin otros errores libre→ocupado, el caso p == t_H (decisión "ocupado") es el error;
    # además, con gap 0 en log-odds, es el más cercano al umbral
    keys = [k for k in KEYS if k not in ("err_fb_1", "err_fb_rain", "err_fb_max", "near_hi")]
    out = E.select_instances(_table(keys + ["free_at_tH"]), T_L, T_H).set_index("motivo")
    assert out.loc[E.ERR_FREE_AS_BUSY].row == LABEL["free_at_tH"]
    assert out.loc[E.ERR_FREE_AS_BUSY].decision == "ocupado"
    assert out.loc[E.NEAR_THRESHOLD].row == LABEL["free_at_tH"]


def test_select_instances_does_not_modify_input():
    table = _table()
    before = table.copy()
    E.select_instances(table, T_L, T_H)
    pd.testing.assert_frame_equal(table, before)


@pytest.mark.parametrize("drop", [
    ("busy_a", "busy_b", "busy_c"),            # regla 1
    ("free_a", "free_b"),                      # regla 2
    ("rain_a", "rain_b", "rain_same_cam"),     # regla 5
])
def test_select_instances_rule_without_candidates_raises(drop):
    with pytest.raises(ValueError):
        E.select_instances(_without(*drop), T_L, T_H)


@pytest.mark.xfail(strict=True, raises=IndexError,
                   reason="explain.py:118 y :126 usan .iloc[0]/.iloc[-1] sin revisar si hay "
                          "candidatos: sin errores de un tipo lanzan IndexError, no ValueError")
@pytest.mark.parametrize("drop", [
    ("err_bf_1", "err_bf_min", "err_bf_3"),                    # regla 3
    ("err_fb_1", "err_fb_rain", "err_fb_max", "free_at_tH", "near_hi"),  # regla 6
])
def test_select_instances_error_rules_without_candidates_raise_value_error(drop):
    with pytest.raises(ValueError):
        E.select_instances(_without(*drop), T_L, T_H)


# ---------------------------------------------------------------------------
# margin_score
# ---------------------------------------------------------------------------

def test_margin_score_sign_by_class():
    logits = torch.tensor([[1.0, 4.0], [2.0, -1.0], [0.5, 0.5]])
    np.testing.assert_allclose(E.margin_score(logits, [1, 1, 1]).numpy(), [3.0, -3.0, 0.0])
    np.testing.assert_allclose(E.margin_score(logits, [0, 0, 0]).numpy(), [-3.0, 3.0, 0.0])
    np.testing.assert_allclose(E.margin_score(logits, np.array([1, 0, 1])).numpy(),
                               [3.0, 3.0, 0.0])


def test_margin_score_scalar_target():
    logits = torch.tensor([[1.0, 4.0]])
    out1 = E.margin_score(logits, 1)
    out0 = E.margin_score(logits, 0)
    assert out1.shape == (1,) and out1.item() == 3.0
    assert out0.item() == -3.0
    assert out1.dtype == logits.dtype


# ---------------------------------------------------------------------------
# Modelos sintéticos para oclusión y fidelidad
# ---------------------------------------------------------------------------

class ConstantModel(nn.Module):
    """Ignora la entrada: siempre los mismos logits."""

    def forward(self, x):
        return torch.tensor([[0.3, 1.2]]).repeat(x.shape[0], 1)


class RegionModel(nn.Module):
    """Logits ``[0, s]`` con ``s`` = suma del canal 0 en la región ``[top:bottom, left:right]``.

    El margen de "ocupado" depende solo de esa región.
    """

    def __init__(self, top: int, bottom: int, left: int, right: int) -> None:
        super().__init__()
        self.box = (top, bottom, left, right)

    def forward(self, x):
        t, b, l, r = self.box
        s = x[:, 0, t:b, l:r].sum(dim=(1, 2))
        return torch.stack([torch.zeros_like(s), s], dim=1)


# ---------------------------------------------------------------------------
# occlusion_map
# ---------------------------------------------------------------------------

def test_occlusion_shape_and_constant_model_gives_zeros():
    x = torch.randn(1, 3, 20, 24, generator=torch.Generator().manual_seed(0))
    m = E.occlusion_map(ConstantModel(), x, target=1, patch=8, stride=4)
    assert m.shape == (20, 24)
    assert np.all(m == 0)


def test_occlusion_by_hand_on_region_model():
    # imagen 8x8 de unos, región = cuadrante superior izquierdo (4x4), ventana 4, paso 2
    x = torch.ones(1, 3, 8, 8)
    model = RegionModel(0, 4, 0, 4)
    m = E.occlusion_map(model, x, target=1, patch=4, stride=2)
    assert m.shape == (8, 8)
    # (0,0): solo la ventana (0,0), que tapa los 16 píxeles de la región → 16
    assert m[0, 0] == pytest.approx(16.0)
    # (2,2): ventanas (0,0),(0,2),(2,0),(2,2) tapan 16, 8, 8, 4 → promedio 9
    assert m[2, 2] == pytest.approx(9.0)
    # (6,6): solo la ventana (4,4), que no toca la región → 0
    assert m[6, 6] == pytest.approx(0.0)
    # el máximo cae dentro de la región
    i, j = np.unravel_index(np.argmax(m), m.shape)
    assert i < 4 and j < 4
    assert m.max() == pytest.approx(16.0)
    # explicar "libre" invierte el signo
    m0 = E.occlusion_map(model, x, target=0, patch=4, stride=2)
    np.testing.assert_allclose(m0, -m, atol=1e-6)


def test_occlusion_region_in_other_quadrant_and_batch_size():
    x = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(1)) + 0.5
    model = RegionModel(16, 32, 16, 32)  # cuadrante inferior derecho
    m = E.occlusion_map(model, x, target=1, patch=8, stride=4, batch_size=64)
    i, j = np.unravel_index(np.argmax(m), m.shape)
    assert i >= 16 and j >= 16
    assert np.all(m[:12, :] == 0) and np.all(m[:, :12] == 0)
    # el tamaño de lote no cambia el resultado
    m1 = E.occlusion_map(model, x, target=1, patch=8, stride=4, batch_size=1)
    np.testing.assert_allclose(m, m1, atol=1e-5)


def test_occlusion_uncovered_border_has_no_nan():
    # 10x10 con ventana 4 y paso 4: inicios 0 y 4; filas/columnas 8 y 9 sin cubrir
    x = torch.ones(1, 3, 10, 10)
    m = E.occlusion_map(RegionModel(0, 10, 0, 10), x, target=1, patch=4, stride=4)
    assert m.shape == (10, 10)
    assert not np.isnan(m).any()
    assert np.all(m[8:, :] == 0) and np.all(m[:, 8:] == 0)
    # cada ventana 4x4 tapa 16 unos → caída 16 en las zonas cubiertas
    np.testing.assert_allclose(m[:8, :8], 16.0)


def test_occlusion_baseline_value():
    # con baseline = 1 sobre una imagen de unos, tapar no cambia nada
    x = torch.ones(1, 3, 8, 8)
    m = E.occlusion_map(RegionModel(0, 4, 0, 4), x, target=1, patch=4, stride=2, baseline=1.0)
    assert np.all(m == 0)


def test_occlusion_does_not_modify_input():
    x = torch.ones(1, 3, 8, 8)
    E.occlusion_map(RegionModel(0, 4, 0, 4), x, target=1, patch=4, stride=2)
    assert torch.all(x == 1)


# ---------------------------------------------------------------------------
# normalize_map
# ---------------------------------------------------------------------------

def test_normalize_map_by_hand():
    m = np.array([[-2.0, 0.0], [1.0, 4.0]])
    np.testing.assert_allclose(E.normalize_map(m), [[0.0, 0.0], [0.25, 1.0]])


def test_normalize_map_zeros_and_all_negative_stay_zero():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        z = E.normalize_map(np.zeros((3, 3)))
        n = E.normalize_map(-np.ones((3, 3)))
    assert np.all(z == 0) and np.all(n == 0)
    assert not np.isnan(z).any() and not np.isnan(n).any()


def test_normalize_map_does_not_modify_input():
    m = np.array([[-1.0, 2.0]])
    E.normalize_map(m)
    np.testing.assert_array_equal(m, [[-1.0, 2.0]])


# ---------------------------------------------------------------------------
# deletion_fidelity
# ---------------------------------------------------------------------------

FID_COLUMNS = ["margen", "caida_top", "caida_bottom", "caida_azar"]


def test_deletion_fidelity_by_hand_on_region_model():
    # 16x16 de unos; región = cuadrante superior izquierdo (8x8 = 64 píxeles)
    # fraction 0.2 → área round(51.2) = 51 píxeles; cuadrado al azar de lado round(√51) = 7
    n = 3
    batch = torch.ones(n, 3, 16, 16)
    cam = np.zeros((16, 16))
    cam[:8, :8] = 1.0
    cams = np.stack([cam] * n)
    model = RegionModel(0, 8, 0, 8)
    out = E.deletion_fidelity(model, batch, cams, np.ones(n, dtype=int), fraction=0.2,
                              n_random=5, seed=0)
    assert list(out.columns) == FID_COLUMNS
    assert len(out) == n
    np.testing.assert_allclose(out.margen, 64.0)
    np.testing.assert_allclose(out.caida_top, 51.0)     # los 51 píxeles tapados están en la región
    np.testing.assert_allclose(out.caida_bottom, 0.0)   # los 51 de menor valor, fuera de la región
    # un cuadrado 7x7 tapa como mucho 49 píxeles de la región
    assert np.all(out.caida_azar <= 49.0) and np.all(out.caida_azar >= 0.0)
    assert np.all(out.caida_top > out.caida_azar)
    assert np.all(out.caida_top > out.caida_bottom)


def test_deletion_fidelity_known_region_random_image():
    g = torch.Generator().manual_seed(3)
    batch = torch.rand(4, 3, 24, 24, generator=g) + 0.5
    model = RegionModel(12, 24, 0, 12)  # cuadrante inferior izquierdo
    cams = np.zeros((4, 24, 24))
    cams[:, 12:, :12] = np.random.default_rng(0).uniform(0.5, 1.0, size=(4, 12, 12))
    out = E.deletion_fidelity(model, batch, cams, np.ones(4, dtype=int), seed=1)
    assert len(out) == 4
    assert np.all(out.caida_top > out.caida_azar)
    assert np.all(out.caida_top > out.caida_bottom)
    np.testing.assert_allclose(out.caida_bottom, 0.0, atol=1e-5)


def test_deletion_fidelity_target_zero_flips_margin():
    batch = torch.ones(1, 3, 16, 16)
    cam = np.zeros((1, 16, 16))
    cam[0, :8, :8] = 1.0
    out = E.deletion_fidelity(RegionModel(0, 8, 0, 8), batch, cam, np.array([0]))
    assert out.margen.iloc[0] == pytest.approx(-64.0)
    assert out.caida_top.iloc[0] == pytest.approx(-51.0)


def test_deletion_fidelity_is_deterministic_with_seed():
    g = torch.Generator().manual_seed(5)
    batch = torch.rand(3, 3, 20, 20, generator=g)
    model = RegionModel(0, 10, 0, 10)
    cams = np.random.default_rng(2).random((3, 20, 20))
    tgt = np.array([1, 0, 1])
    a = E.deletion_fidelity(model, batch, cams, tgt, seed=11)
    b = E.deletion_fidelity(model, batch, cams, tgt, seed=11)
    pd.testing.assert_frame_equal(a, b)
    # otra semilla cambia las posiciones al azar (y solo esa columna)
    c = E.deletion_fidelity(model, batch, cams, tgt, seed=12)
    pd.testing.assert_frame_equal(a[FID_COLUMNS[:3]], c[FID_COLUMNS[:3]])
    assert not np.allclose(a.caida_azar, c.caida_azar)


def test_deletion_fidelity_does_not_modify_batch():
    batch = torch.ones(2, 3, 16, 16)
    cams = np.random.default_rng(0).random((2, 16, 16))
    E.deletion_fidelity(RegionModel(0, 8, 0, 8), batch, cams, np.array([1, 1]))
    assert torch.all(batch == 1)


@pytest.mark.xfail(strict=True,
                   reason="explain.py:209/217: si round(fraction*H*W) == 0, order[-0:] toma "
                          "TODOS los píxeles y 'top' tapa la imagen entera")
def test_deletion_fidelity_zero_area_covers_nothing():
    batch = torch.ones(1, 3, 8, 8)
    cam = np.random.default_rng(0).random((1, 8, 8))
    out = E.deletion_fidelity(RegionModel(0, 8, 0, 8), batch, cam, np.array([1]),
                              fraction=0.001)  # 0.001 * 64 = 0.064 → área 0
    assert out.caida_top.iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# center_share
# ---------------------------------------------------------------------------

def test_center_share_uniform_is_area():
    cams = np.ones((2, 8, 8))
    np.testing.assert_allclose(E.center_share(cams, center_frac=0.5), [0.25, 0.25])


def test_center_share_concentrated_and_zero_maps():
    cams = np.zeros((3, 8, 8))
    cams[0, 2:6, 2:6] = 1.0         # todo en el cuadrado central (lado 4, desde 2)
    cams[1, 0, 0] = 1.0             # todo en una esquina
    # cams[2] queda en ceros
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        share = E.center_share(cams, center_frac=0.5)
    np.testing.assert_allclose(share, [1.0, 0.0, 0.0])
    assert not np.isnan(share).any()


def test_center_share_mixed_by_hand():
    cam = np.zeros((1, 8, 8))
    cam[0, 3, 3] = 3.0   # dentro del centro
    cam[0, 0, 7] = 1.0   # fuera
    np.testing.assert_allclose(E.center_share(cam), [0.75])
    # center_frac = 1: todo el parche es el centro
    np.testing.assert_allclose(E.center_share(cam, center_frac=1.0), [1.0])


# ---------------------------------------------------------------------------
# stratified_sample
# ---------------------------------------------------------------------------

def _strata_table() -> pd.DataFrame:
    sizes = {(0, "SUNNY"): 10, (0, "RAINY"): 2, (1, "SUNNY"): 7, (1, "OVERCAST"): 5}
    rows = []
    for (y, w), k in sizes.items():
        rows += [{"y": y, "weather": w, "path": f"{y}_{w}_{i}.jpg"} for i in range(k)]
    df = pd.DataFrame(rows)
    df.index = 500 + 3 * np.arange(len(df))  # índice no consecutivo
    return df


def test_stratified_sample_counts_and_columns():
    table = _strata_table()
    out = E.stratified_sample(table, per_stratum=4, seed=0)
    assert "row" in out.columns
    assert list(out.columns) == ["row", "y", "weather", "path"]
    counts = out.groupby(["y", "weather"]).size().to_dict()
    assert counts == {(0, "RAINY"): 2, (0, "SUNNY"): 4, (1, "OVERCAST"): 4, (1, "SUNNY"): 4}
    # sin reemplazo y cada fila apunta a la original
    assert out.row.is_unique
    for _, r in out.iterrows():
        orig = table.loc[r.row]
        assert (orig.y, orig.weather, orig.path) == (r.y, r.weather, r.path)


def test_stratified_sample_larger_than_strata_returns_all():
    table = _strata_table()
    out = E.stratified_sample(table, per_stratum=100)
    assert len(out) == len(table)
    assert sorted(out.row) == sorted(table.index)


def test_stratified_sample_is_deterministic_with_seed():
    table = _strata_table()
    a = E.stratified_sample(table, 3, seed=9)
    b = E.stratified_sample(table, 3, seed=9)
    pd.testing.assert_frame_equal(a, b)
    rows = {tuple(E.stratified_sample(table, 3, seed=s).row) for s in range(10)}
    assert len(rows) > 1


# ---------------------------------------------------------------------------
# hog_contribution_map con el B1 real (datos reales, solo validación)
# ---------------------------------------------------------------------------

def _b1_run_dir() -> Path | None:
    found = sorted(paths.runs_dir().glob("b1_*/model.joblib"))
    return found[-1].parent if found else None


@pytest.fixture(scope="module")
def b1_val_case():
    run_dir = _b1_run_dir()
    if run_dir is None or not HAS_DATA:
        pytest.skip("no hay run B1 en runs/ o el dataset no está en DATA_ROOT")
    import joblib

    from cupo.data import split_dates
    from cupo.runs import load_npz

    model = joblib.load(run_dir / "model.joblib")
    preds = load_npz(run_dir / "val_preds.npz")
    val_dates = split_dates("val")
    for k, path in enumerate(preds["path"]):
        if str(preds["date"][k]) in val_dates and Path(str(path)).is_file():
            return model, str(path), preds, k
    pytest.skip("no se encontró una imagen de validación en disco")


def test_b1_model_structure_matches_explain():
    run_dir = _b1_run_dir()
    if run_dir is None:
        pytest.skip("no hay run B1 en runs/")
    import joblib

    model = joblib.load(run_dir / "model.joblib")
    assert isinstance(model, dict) and {"hog", "pipeline"} <= set(model)
    assert set(model["hog"]) == {"image_size", "orientations", "pixels_per_cell",
                                 "cells_per_block"}
    steps = model["pipeline"].steps
    scaler, clf = steps[0][1], steps[-1][1]
    assert hasattr(scaler, "mean_") and hasattr(scaler, "scale_")
    assert hasattr(clf, "coef_") and clf.coef_.shape[0] == 1


@pytest.mark.data
def test_hog_contribution_map_sums_to_decision_function(b1_val_case):
    from cupo.classical import hog_one

    model, path, preds, k = b1_val_case
    cfg = model["hog"]
    cmap = E.hog_contribution_map(model, path)
    blocks = cfg["image_size"] // cfg["pixels_per_cell"] - cfg["cells_per_block"] + 1
    assert cmap.shape == (blocks, blocks)  # 64/8 - 2 + 1 = 7 con la config del B1
    assert np.isfinite(cmap).all()

    feats = hog_one(path, **cfg)
    clf = model["pipeline"].steps[-1][1]
    z = float(model["pipeline"].decision_function(feats[None, :])[0])
    assert float(cmap.sum()) + float(clf.intercept_[0]) == pytest.approx(z, abs=1e-6)
    # coincide con el logit guardado del run para la misma imagen de validación
    if "logits" in preds:
        stored = float(preds["logits"][k, 1] - preds["logits"][k, 0])
        assert z == pytest.approx(stored, abs=1e-3)


@pytest.mark.data
def test_hog_contribution_map_is_deterministic(b1_val_case):
    model, path, _, _ = b1_val_case
    np.testing.assert_array_equal(E.hog_contribution_map(model, path),
                                  E.hog_contribution_map(model, path))
