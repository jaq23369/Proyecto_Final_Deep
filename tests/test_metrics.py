"""Pruebas de cupo.metrics con valores calculados a mano (positivo = ocupado)."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from cupo import metrics

# Ejemplo pequeño: 3 ocupados (p = .5, .7, .3) y 3 libres (p = .5, .2, .9).
# Con thr = 0.5 (ocupado si p >= 0.5): pred = [1, 1, 1, 0, 0, 1]
#   TP = 2 (idx 0, 1), FN = 1 (idx 4), FP = 2 (idx 2, 5), TN = 1 (idx 3)
Y_SMALL = np.array([1, 1, 0, 0, 1, 0])
P_SMALL = np.array([0.5, 0.7, 0.5, 0.2, 0.3, 0.9])


class TestBinaryMetrics:
    def test_confusion_and_counts(self):
        m = metrics.binary_metrics(Y_SMALL, P_SMALL, thr=0.5)
        assert m["confusion"] == [[1, 2], [1, 2]]  # [[TN, FP], [FN, TP]]
        assert m["n"] == 6 and m["n_busy"] == 3 and m["n_free"] == 3
        assert m["threshold"] == 0.5

    def test_rates_by_hand(self):
        m = metrics.binary_metrics(Y_SMALL, P_SMALL, thr=0.5)
        assert m["accuracy"] == pytest.approx(3 / 6)
        # P(pred libre | ocupado) = FN / (TP + FN) = 1/3
        assert m["rate_busy_as_free"] == pytest.approx(1 / 3)
        # P(pred ocupado | libre) = FP / (TN + FP) = 2/3
        assert m["rate_free_as_busy"] == pytest.approx(2 / 3)
        assert m["precision_busy"] == pytest.approx(2 / 4)
        assert m["recall_busy"] == pytest.approx(2 / 3)
        assert m["f1_busy"] == pytest.approx(4 / 7)  # 2TP / (2TP + FP + FN)
        assert m["precision_free"] == pytest.approx(1 / 2)
        assert m["recall_free"] == pytest.approx(1 / 3)
        assert m["f1_free"] == pytest.approx(2 / 5)  # 2TN / (2TN + FP + FN)
        assert m["f1_macro"] == pytest.approx((4 / 7 + 2 / 5) / 2)

    def test_auc_and_log_loss_by_hand(self):
        m = metrics.binary_metrics(Y_SMALL, P_SMALL, thr=0.5)
        # Pares (ocupado, libre) ordenados correctamente, empates = 0.5: 4.5 de 9.
        assert m["auc"] == pytest.approx(4.5 / 9)
        ll = -np.mean([math.log(0.5), math.log(0.7), math.log(0.3),
                       math.log(0.5), math.log(0.8), math.log(0.1)])
        assert m["log_loss"] == pytest.approx(ll)

    def test_positive_is_busy(self):
        # Todo ocupado y predicho con p alto: sin falsos libres.
        m = metrics.binary_metrics([1, 1, 0], [0.9, 0.8, 0.1])
        assert m["confusion"] == [[1, 0], [0, 2]]
        assert m["rate_busy_as_free"] == 0.0
        assert m["recall_busy"] == 1.0
        # Un ocupado con p bajo es un ocupado mostrado libre (el error caro).
        m2 = metrics.binary_metrics([1, 1, 0], [0.1, 0.8, 0.1])
        assert m2["confusion"] == [[1, 0], [1, 1]]
        assert m2["rate_busy_as_free"] == pytest.approx(0.5)

    def test_p_equal_threshold_is_busy(self):
        m = metrics.binary_metrics([1, 0], [0.5, 0.5], thr=0.5)
        assert m["confusion"] == [[0, 1], [0, 1]]

    def test_single_class_auc_nan(self):
        m = metrics.binary_metrics([1, 1, 1], [0.2, 0.6, 0.9])
        assert math.isnan(m["auc"])
        assert math.isnan(m["rate_free_as_busy"])  # sin libres
        assert m["rate_busy_as_free"] == pytest.approx(1 / 3)

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            metrics.binary_metrics([1, 0, 1], [0.2, 0.3])

    def test_json_serializable(self, good_scores):
        y, p = good_scores
        json.dumps(metrics.binary_metrics(y, p))


class TestThreeStateRates:
    def test_by_hand(self):
        r = metrics.three_state_rates(Y_SMALL, P_SMALL, t_L=0.3, t_H=0.7)
        # ocupados p = [.5, .7, .3]: ninguno < .3; .7 >= .7; .5 y .3 en verificar
        assert r["q_OL"] == pytest.approx(0.0)
        assert r["q_OO"] == pytest.approx(1 / 3)
        assert r["q_OV"] == pytest.approx(2 / 3)
        # libres p = [.5, .2, .9]: .2 libre, .9 ocupado, .5 verificar
        assert r["q_LL"] == pytest.approx(1 / 3)
        assert r["q_LO"] == pytest.approx(1 / 3)
        assert r["q_LV"] == pytest.approx(1 / 3)
        assert r["verify_rate"] == pytest.approx(3 / 6)
        assert r["n_busy"] == 3 and r["n_free"] == 3

    @pytest.mark.parametrize("t_L,t_H", [(0.0, 1.0), (0.1, 0.9), (0.3, 0.7), (0.5, 0.5), (0.25, 0.8)])
    def test_rates_sum_to_one(self, good_scores, t_L, t_H):
        y, p = good_scores
        r = metrics.three_state_rates(y, p, t_L, t_H)
        assert r["q_OL"] + r["q_OV"] + r["q_OO"] == pytest.approx(1.0)
        assert r["q_LL"] + r["q_LV"] + r["q_LO"] == pytest.approx(1.0)
        for k in ("q_OL", "q_OV", "q_OO", "q_LL", "q_LV", "q_LO", "verify_rate"):
            assert 0.0 <= r[k] <= 1.0

    def test_verify_rate_is_global(self, good_scores):
        y, p = good_scores
        r = metrics.three_state_rates(y, p, 0.2, 0.8)
        expected = np.mean((p >= 0.2) & (p < 0.8))
        assert r["verify_rate"] == pytest.approx(expected)

    def test_equal_thresholds_match_binary(self):
        # p redondeado a 2 decimales: hay muchos p exactamente iguales al umbral.
        rng = np.random.default_rng(0)
        y = rng.binomial(1, 0.6, 2000)
        p = np.round(rng.random(2000), 2)
        for thr in (0.3, 0.5, 0.71):
            assert np.any(p == thr)
            ts = metrics.three_state_rates(y, p, thr, thr)
            bm = metrics.binary_metrics(y, p, thr)
            assert ts["q_OV"] == 0.0 and ts["q_LV"] == 0.0 and ts["verify_rate"] == 0.0
            assert ts["q_OL"] == pytest.approx(bm["rate_busy_as_free"], abs=0)
            assert ts["q_LO"] == pytest.approx(bm["rate_free_as_busy"], abs=0)
            (tn, fp), (fn, tp) = bm["confusion"]
            assert ts["q_OO"] == pytest.approx(tp / (tp + fn))
            assert ts["q_LL"] == pytest.approx(tn / (tn + fp))

    def test_p_equal_to_threshold_small(self):
        # p == t con t_L == t_H: va a OCUPADO, igual que binary_metrics.
        ts = metrics.three_state_rates([1, 0], [0.5, 0.5], 0.5, 0.5)
        assert ts["q_OO"] == 1.0 and ts["q_LO"] == 1.0
        # p == t_L con t_L < t_H: va a VERIFICAR; p == t_H va a OCUPADO.
        ts2 = metrics.three_state_rates([1, 0], [0.3, 0.7], 0.3, 0.7)
        assert ts2["q_OV"] == 1.0 and ts2["q_LO"] == 1.0

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError):
            metrics.three_state_rates([1, 0], [0.4, 0.6], 0.7, 0.3)


class TestBootstrapCI:
    @staticmethod
    def _acc(y, p):
        return metrics.binary_metrics(y, p)["accuracy"]

    def test_contains_estimate(self, good_scores):
        y, p = good_scores[0][:3000], good_scores[1][:3000]
        est = self._acc(y, p)
        lo, hi = metrics.bootstrap_ci(self._acc, y, p, n_boot=100, seed=0)
        assert lo <= est <= hi
        assert lo < hi

    def test_contains_estimate_f1_macro(self, good_scores):
        y, p = good_scores[0][:3000], good_scores[1][:3000]

        def f1m(a, b):
            return metrics.binary_metrics(a, b)["f1_macro"]

        lo, hi = metrics.bootstrap_ci(f1m, y, p, n_boot=50, seed=3)
        assert lo <= f1m(y, p) <= hi

    def test_reproducible_with_seed(self, good_scores):
        y, p = good_scores[0][:2000], good_scores[1][:2000]
        a = metrics.bootstrap_ci(self._acc, y, p, n_boot=60, seed=7)
        b = metrics.bootstrap_ci(self._acc, y, p, n_boot=60, seed=7)
        c = metrics.bootstrap_ci(self._acc, y, p, n_boot=60, seed=8)
        assert a == b
        assert a != c
        assert all(isinstance(v, float) for v in a)

    def test_nan_replicates_ignored(self):
        # AUC con una sola clase da NaN: el intervalo también es NaN.
        lo, hi = metrics.bootstrap_ci(
            lambda a, b: metrics.binary_metrics(a, b)["auc"], [1, 1, 1], [0.2, 0.5, 0.9], n_boot=20
        )
        assert math.isnan(lo) and math.isnan(hi)


class TestBreakdown:
    def test_groups_match_binary_metrics(self, good_scores):
        y, p = good_scores
        rng = np.random.default_rng(5)
        df = pd.DataFrame({"y": y, "p": p, "weather": rng.choice(["SUNNY", "OVERCAST", "RAINY"], y.size),
                           "camera": rng.integers(1, 10, y.size)})
        out = metrics.breakdown(df, by="weather")
        assert set(out) == {"SUNNY", "OVERCAST", "RAINY"}
        sub = df[df.weather == "RAINY"]
        assert out["RAINY"] == metrics.binary_metrics(sub.y.to_numpy(), sub.p.to_numpy())
        by_cam = metrics.breakdown(df, by="camera")
        assert set(by_cam) == {str(c) for c in range(1, 10)}
        assert sum(v["n"] for v in by_cam.values()) == y.size
        json.dumps(by_cam)

    def test_missing_column(self):
        with pytest.raises(KeyError):
            metrics.breakdown(pd.DataFrame({"y": [1], "p": [0.5]}), by="camera")
