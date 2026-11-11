"""Modelo de valor de negocio de Cupo Seguro.

Traduce el desempeño del clasificador (tasas de la decisión de tres estados)
a minutos ahorrados por conductor y a quetzales para cada escenario.

Entradas
--------
- ``rates``: salida de ``metrics.three_state_rates``. Son obligatorias las seis
  tasas ``q_OL, q_OV, q_OO`` (condicionadas en ocupado) y ``q_LL, q_LV, q_LO``
  (condicionadas en libre); L = mostrado libre, V = por verificar, O = mostrado
  ocupado.
- ``scenario_cfg``: escenario cargado con ``load_scenario`` (configs/value/*.yaml).
  Cada parámetro del YAML tiene la forma ``{value, source, confidence}``.

Las tasas se reponderan con la ocupación del escenario ``π``; no se usa la
proporción de clases del conjunto de validación.

Franja "por verificar"
----------------------
Los espacios en amarillo los revisa un operador en la cámara. La revisión
acierta con probabilidad ``a = precision_verificacion``: los libres bien
verificados pasan a verde y los ocupados bien verificados a rojo; con
probabilidad ``1 − a`` el operador se equivoca (los casos en amarillo son los
más ambiguos: sombras, oclusión). Para el conductor, las tasas efectivas de
verde son

    q_LL_eff = q_LL + a·q_LV          libres que terminan en verde
    q_OL_eff = q_OL + (1 − a)·q_OV    ocupados que terminan en verde (error caro)

y son las que entran en ``e``, ``P0`` y el recorrido. Con ``a = 1`` la
verificación es perfecta (``q_OL_eff = q_OL``). El precio de la franja es el
costo de revisión.

Capacidad oculta y recorrido
----------------------------
Los libres que no se muestran en verde (``q_LO`` y la parte de ``q_LV`` que no
termina en verde) sí cuestan: el conductor tiene menos opciones visibles y
recorre más hasta el libre verde más cercano.

Supuesto: los libres están repartidos de forma uniforme a lo largo de un
trayecto lineal que toma ``L = t_recorrido_total`` minutos recorrer completo.
Con ``k`` libres visibles, el más cercano está en promedio a ``L/(k + 1)``
(estadística de orden de uniformes). El número de libres visibles es
``K ~ Bin(F, q)`` con ``q = q_LL_eff``, y su esperanza exacta es

    E[1/(K + 1)] = (1 − (1 − q)^(F+1)) / ((F + 1)·q)      (límite 1 cuando q → 0)

En la rama guiada (``K ≥ 1``, probabilidad ``1 − P0``) se condiciona:

    frac_recorrido = (E[1/(K + 1)] − P0) / (1 − P0)

con límite 1/2 cuando ``P0 → 1`` (un solo verde visible). Si se muestran todos
los libres (``q = 1``) ``frac_recorrido = 1/(F + 1)``. ``factor_recorrido =
frac_recorrido·(F + 1)`` vale 1 en ese caso y crece al ocultar libres;
``k_vis = F·q`` se reporta solo como referencia. Así el ``t_H`` óptimo ya no
queda fijo en 0.5 por construcción: subir ``t_H`` manda a verificar libres que
estaban ocultos en rojo y acorta el recorrido, a cambio del costo de revisión.

Robustez: el castigo a la capacidad oculta escala con ``L``. Con ``L`` menor a
~1.2-1.5 min un modelo degenerado que oculta casi todos los libres (pero casi
nunca muestra un ocupado en verde) puede volver a competir con uno bueno; con
los valores por defecto, un degenerado con q_OL = 0 y q_LL = 0.24 supera a uno
con q_OL = 0.02 y q_LL = 0.97 si L < 1.42 min (comercial) o L < 1.19 min (UVG).
El análisis de sensibilidad (tornado) de ``t_recorrido_total_min`` lo muestra.

Limitaciones conocidas
----------------------
1. Con ocupación alta (π ≈ 0.85-0.90, F ≈ 16-25), ``P0`` ≈ 0 aunque se oculten
   muchos libres; la capacidad oculta solo pesa a través del recorrido, que
   supone libres repartidos uniformemente y un recorrido lineal.
2. La verificación se supone inmediata, con una precisión fija ``a`` igual para
   libres y ocupados.
3. Competencia entre conductores: con ~1.3 llegadas/min, refresco de 30 s y
   pocos verdes, varios conductores pueden ir al mismo espacio y el ``e``
   efectivo sube. No se modela.
4. El personal extra se modela como continuo (tiempo parcial fraccionario), no
   por turnos completos.
5. Los umbrales óptimos pueden caer en las colas (p. ej. 0.02-0.10) y dependen
   de la calibración del modelo. La validación no tiene lluvia, así que test se
   reporta por clima y OOD, siempre con los umbrales elegidos en validación.
6. La verificación tiende a llenar la capacidad gratuita del encargado
   (``n_amber`` ≈ 6 espacios con los valores por defecto). Para separar el
   aporte del modelo del trabajo humano se reporta también el valor sin franja
   (2 estados) y ``operadores_extra``.

Paso 1: probabilidades
----------------------
    N, π     = espacios, ocupacion
    F        = max(1, round((1 − π)·N))            espacios realmente libres (redondeo .5 hacia arriba)
    a        = precision_verificacion
    q_LL_eff = q_LL + a·q_LV                        libres en verde después de verificar
    q_OL_eff = q_OL + (1 − a)·q_OV                  ocupados en verde después de verificar
    e        = π·q_OL_eff / (π·q_OL_eff + (1 − π)·q_LL_eff)   P(un verde esté ocupado); si el denominador es 0, e = 1
    e_c      = min(e, 0.95)                        tope numérico
    P0       = (1 − q_LL_eff)^F                    ningún libre real se muestra en verde → se busca como hoy
    k_vis    = F · q_LL_eff                         libres visibles en verde esperados (informativo)

Paso 2: tiempo y ahorro (minutos por llegada)
---------------------------------------------
    frac_recorrido   = (E[1/(K + 1)] − P0) / (1 − P0)   ver "Capacidad oculta y recorrido"
    factor_recorrido = frac_recorrido · (F + 1)        1 si se ven todos los libres
    t_directo_eff    = min(t_acceso + L · frac_recorrido, t_base)
    T_guiado         = t_directo_eff + (e_c/(1 − e_c))·t_redireccion
    T_sys            = min((1 − P0)·T_guiado + P0·t_base, t_base + t_redireccion)
    A                = t_base − T_sys                  ahorro por llegada (puede ser negativo)

``t_acceso`` es entrar al parqueo y estacionar (no depende de dónde esté el
espacio) y ``L = t_recorrido_total`` es recorrer todos los pasillos a velocidad
de búsqueda. Con un clasificador perfecto ``T_sys = t_acceso + L/(F + 1)``
(≈ 1.00 min en comercial con F = 25 y ≈ 1.09 min en UVG con F = 16).

El tope de ``t_directo_eff`` en ``t_base`` refleja que, si el verde visible más
cercano está más lejos de lo que costaría buscar como hoy, el conductor busca
como hoy. El tope de ``T_sys`` refleja que, tras un intento fallido, el
conductor deja de confiar en la app y busca como hoy: lo peor que pasa es
perder una redirección y luego buscar con el método actual. ``t_sys_capped``
indica si el tope se activó.

Paso 3: abandono y volúmenes (por periodo: mes o semestre)
----------------------------------------------------------
    p_ab(t)   = clip(abandono_base · t / t_base, 0, 1)
    Δab       = p_ab(t_base) − p_ab(T_sys)
    llegadas  = llegadas_dia · dias_periodo
    pred      = N · (3600/refresh_s) · horas_operacion_dia · dias_periodo
    n_amber   = N · (π·q_OV + (1 − π)·q_LV)    espacios en amarillo en promedio

Paso 4: costos y valor
----------------------
    carga_h          = n_amber · revisiones_por_hora_amber · segundos_por_revision / 3600   horas-operador necesarias por hora
    disponible_h     = operadores_incluidos · fraccion_tiempo_operador                      capacidad sin costo incremental
    operadores_extra = max(0, carga_h − disponible_h)                                       continuo (tiempo parcial)
    costo_verif  = operadores_extra · salario_operador_Q_h · horas_operacion_dia · dias_periodo
    costo_infra  = [(n_camaras·costo_camara_Q)/meses_amortizacion + costo_servidor_mes_Q] · meses_periodo
    costo_inf    = pred/1000 · costo_por_1000_pred_Q
    costos       = costo_infra + costo_inf + costo_verif
    ingreso_recuperado = llegadas · Δab · ingreso_por_visita_Q
    valor_operador     = ingreso_recuperado − costos
    horas_ahorradas    = llegadas · A / 60
    valor_usuarios_Q   = horas_ahorradas · valor_tiempo_Q_h
    valor_neto_Q       = valor_usuarios_Q − costos
    costo_sensores     = [(N·costo_sensor_instalado_Q)/meses_amortizacion + mantenimiento_sensores_mes_Q] · meses_periodo
    ahorro_vs_sensores = costo_sensores − costos          (comparativa, no entra al objetivo)

El costo de verificación se modela por capacidad del personal. El parqueo ya
tiene un encargado (``operadores_incluidos``) que dedica una fracción de su
tiempo a revisar cámaras sin costo incremental. Con un modelo bueno pocos
espacios quedan en amarillo y ese encargado los atiende sin costo extra; con un
modelo débil la carga supera su capacidad y hay que contratar personal
(``operadores_extra``, a tiempo parcial si es fraccionario). En el escenario
comercial esto impide que un modelo débil "compre" precisión barata mandando
casi todo a verificar. En UVG, donde el minuto ahorrado vale más, un modelo
débil sí puede justificar contratar personal; por eso la tabla de resultados
reporta también el valor con 2 estados y ``operadores_extra``.

``costo_por_1000_pred_Q`` tiene su valor por defecto en ``configs/costs.yaml``;
un escenario puede sobrescribirlo de forma opcional.

Escenarios
----------
- Comercial (periodo = mes, ``meses_periodo = 1`` fijo, aunque varíe ``dias_mes``):
  objetivo ``valor_operador_mes``.
- UVG (periodo = semestre, ``dias_semestre`` en lugar de ``dias_mes``, sin abandono):
  ``valor_tiempo_Q_h = (1 − frac_personal)·valor_tiempo_estudiante + frac_personal·valor_tiempo_personal``;
  objetivo ``valor_neto_semestre_Q = horas_ahorradas_semestre·valor_tiempo − costos_semestre``.
  ``meses_periodo = meses_semestre`` (meses calendario en que corre la infraestructura).
- Status quo (B0, sin sistema): ``T_sys = t_base``, ``A = 0``, sin costos → valor 0.

Los nombres de los componentes llevan el sufijo del periodo, por ejemplo
``costo_infra_mes`` o ``horas_ahorradas_semestre``.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Raíz del repositorio: src/cupo/../../
REPO_ROOT = Path(__file__).resolve().parents[2]
VALUE_DIR = REPO_ROOT / "configs" / "value"
COSTS_PATH = REPO_ROOT / "configs" / "costs.yaml"

E_CAP = 0.95  # tope numérico de e para que e/(1−e) no explote
MONTH_DAYS = 30.0  # días por mes cuando el YAML no define los meses del periodo
COST_PER_1000_KEY = "costo_por_1000_pred_Q"

RATE_KEYS = ("q_OL", "q_OV", "q_OO", "q_LL", "q_LV", "q_LO")

# Parámetros que son probabilidades: el análisis de sensibilidad los recorta a estos rangos.
_PROB_BOUNDS: dict[str, tuple[float, float]] = {
    "ocupacion": (0.0, 0.97),
    "abandono_base": (0.0, 1.0),
    "frac_personal": (0.0, 1.0),
    "fraccion_tiempo_operador": (0.0, 1.0),
    "precision_verificacion": (0.0, 1.0),
}

_REQUIRED_PARAMS = (
    "espacios", "ocupacion", "llegadas_dia", "horas_operacion_dia", "refresh_s",
    "t_base_min", "t_acceso_min", "t_recorrido_total_min", "t_redireccion_min",
    "n_camaras", "costo_camara_Q", "costo_servidor_mes_Q", "meses_amortizacion",
    "costo_sensor_instalado_Q", "mantenimiento_sensores_mes_Q",
    "segundos_por_revision", "revisiones_por_hora_amber", "fraccion_tiempo_operador",
    "operadores_incluidos", "salario_operador_Q_h", "precision_verificacion",
)


# ---------------------------------------------------------------------------
# Carga de configuración
# ---------------------------------------------------------------------------

def flatten_params(raw: Mapping[str, Any]) -> dict[str, float]:
    """Aplana ``{nombre: {value, source, confidence}}`` a ``{nombre: value}``.

    Acepta también parámetros que ya son números (se devuelven tal cual).
    """
    flat: dict[str, float] = {}
    for key, item in raw.items():
        flat[key] = item["value"] if isinstance(item, Mapping) else item
    return flat


def load_costs(path: str | Path = COSTS_PATH) -> dict[str, float]:
    """Carga ``configs/costs.yaml`` (tipo de cambio, cómputo y costo por predicción)."""
    with Path(path).open(encoding="utf-8") as fh:
        return flatten_params(yaml.safe_load(fh))


def default_cost_per_1000() -> float:
    """Costo por 1000 predicciones por defecto (``configs/costs.yaml``)."""
    return float(load_costs()[COST_PER_1000_KEY])


def _resolve_scenario_path(name: str | Path) -> Path:
    """Acepta el nombre del escenario ('comercial', 'uvg') o una ruta a un YAML."""
    candidate = Path(name)
    if candidate.suffix in {".yaml", ".yml"} and candidate.exists():
        return candidate
    path = VALUE_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No existe el escenario '{name}' en {VALUE_DIR}")
    return path


def _param_sources(raw_params: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Fuente y nivel de confianza de cada parámetro del YAML."""
    return {
        k: {"source": v.get("source", ""), "confidence": v.get("confidence", "")}
        for k, v in raw_params.items()
        if isinstance(v, Mapping)
    }


def load_scenario(name: str | Path) -> dict[str, Any]:
    """Carga un escenario de ``configs/value/<name>.yaml``.

    Devuelve un dict con ``name``, ``description``, ``objective``, ``periodo``,
    ``params`` (valores aplanados), ``sources`` (fuente y confianza de cada
    parámetro) y ``path``. Si el escenario no define ``costo_por_1000_pred_Q``
    se toma el valor por defecto de ``configs/costs.yaml``.
    """
    path = _resolve_scenario_path(name)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    for key in ("name", "objective", "periodo", "params"):
        if key not in raw:
            raise KeyError(f"El escenario {path} no define '{key}'")
    params = flatten_params(raw["params"])
    sources = _param_sources(raw["params"])
    if COST_PER_1000_KEY not in params:
        params[COST_PER_1000_KEY] = default_cost_per_1000()
        sources[COST_PER_1000_KEY] = {
            "source": "Valor por defecto de configs/costs.yaml; se reemplaza con el resultado de benchmark",
            "confidence": "estimado",
        }
    cfg = {
        "name": raw["name"],
        "description": raw.get("description", ""),
        "objective": raw["objective"],
        "periodo": raw["periodo"],
        "params": params,
        "sources": sources,
        "path": str(path),
    }
    _validate_params(cfg)
    return cfg


def scenario_params(scenario_cfg: Mapping[str, Any]) -> dict[str, float]:
    """Parámetros numéricos del escenario, aplanados."""
    return flatten_params(scenario_cfg["params"])


def _validate_params(scenario_cfg: Mapping[str, Any]) -> None:
    """Verifica que el escenario tenga todos los parámetros necesarios."""
    prm = scenario_params(scenario_cfg)
    sfx = scenario_cfg["periodo"]
    missing = [k for k in (*_REQUIRED_PARAMS, f"dias_{sfx}") if k not in prm]
    if "valor_tiempo_Q_h" not in prm and not _has_mixed_time_value(prm):
        missing.append("valor_tiempo_Q_h")
    if missing:
        raise KeyError(f"Faltan parámetros en el escenario '{scenario_cfg['name']}': {missing}")


# ---------------------------------------------------------------------------
# Piezas del cálculo (aceptan escalares o arreglos de numpy que se difunden)
# ---------------------------------------------------------------------------

def _has_mixed_time_value(prm: Mapping[str, float]) -> bool:
    keys = ("valor_tiempo_estudiante_Q_h", "valor_tiempo_personal_Q_h", "frac_personal")
    return all(k in prm for k in keys)


def time_value_q_h(prm: Mapping[str, float]) -> float:
    """Valor del tiempo en Q/h; en UVG es la mezcla estudiante/personal."""
    if _has_mixed_time_value(prm):
        frac = prm["frac_personal"]
        return (1 - frac) * prm["valor_tiempo_estudiante_Q_h"] + frac * prm["valor_tiempo_personal_Q_h"]
    return prm["valor_tiempo_Q_h"]


def verification_load(n_amber: Any, prm: Mapping[str, float]) -> dict[str, Any]:
    """Carga de verificación frente a la capacidad del personal (por hora de operación).

    carga_h          = n_amber · revisiones_por_hora_amber · segundos_por_revision / 3600
    disponible_h     = operadores_incluidos · fraccion_tiempo_operador
    operadores_extra = max(0, carga_h − disponible_h)
    """
    load = n_amber * prm["revisiones_por_hora_amber"] * prm["segundos_por_revision"] / 3600.0
    available = prm["operadores_incluidos"] * prm["fraccion_tiempo_operador"]
    return {
        "carga_h": load,
        "disponible_h": available,
        "operadores_extra": np.maximum(0.0, load - available),
    }


def _cost_per_1000(prm: Mapping[str, float]) -> float:
    """Costo por 1000 predicciones: el del escenario o el de costs.yaml."""
    if COST_PER_1000_KEY in prm:
        return float(prm[COST_PER_1000_KEY])
    return default_cost_per_1000()


def _period(scenario_cfg: Mapping[str, Any], prm: Mapping[str, float]) -> tuple[str, float, float]:
    """Sufijo del periodo, días de operación y meses calendario del periodo.

    Con periodo mensual los meses son 1 fijo, así variar ``dias_mes`` no escala
    los costos mensuales de infraestructura.
    """
    sfx = scenario_cfg["periodo"]
    days = float(prm[f"dias_{sfx}"])
    if sfx == "mes":
        return sfx, days, 1.0
    months = float(prm.get(f"meses_{sfx}", days / MONTH_DAYS))
    return sfx, days, months


def free_spaces(prm: Mapping[str, float]) -> int:
    """F = max(1, round((1 − π)·N)), con redondeo .5 hacia arriba."""
    return max(1, int(np.floor((1 - prm["ocupacion"]) * prm["espacios"] + 0.5)))


def _check_rates(rates: Mapping[str, Any]) -> None:
    """Exige las seis tasas; no se asume 0 en silencio."""
    missing = [k for k in RATE_KEYS if k not in rates]
    if missing:
        raise ValueError(f"Faltan tasas en rates: {missing}; se requieren {list(RATE_KEYS)}")


def _probabilities(rates: Mapping[str, Any], prm: Mapping[str, float]) -> dict[str, Any]:
    """Paso 1: F, tasas efectivas tras verificar, e, e_c y P0."""
    pi = prm["ocupacion"]
    a = prm["precision_verificacion"]
    q_OL_eff = np.asarray(rates["q_OL"], dtype=float) + (1 - a) * np.asarray(rates["q_OV"], dtype=float)
    q_LL_eff = np.asarray(rates["q_LL"], dtype=float) + a * np.asarray(rates["q_LV"], dtype=float)
    num = pi * q_OL_eff
    den = num + (1 - pi) * q_LL_eff
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0)
    F = free_spaces(prm)
    return {
        "F": F,
        "q_OL_eff": q_OL_eff,
        "q_LL_eff": q_LL_eff,
        "e": e,
        "e_c": np.minimum(e, E_CAP),
        "P0": np.clip(1 - q_LL_eff, 0.0, 1.0) ** F,
    }


_P0_EPS = 1e-8  # si 1 − P0 es menor, se usa el límite de un solo verde visible


def _one_minus_pow(q: np.ndarray, n: int) -> np.ndarray:
    """1 − (1 − q)^n calculado de forma estable para q pequeño (q ∈ [0, 1])."""
    inner = np.log1p(-np.minimum(q, 0.5)) * n
    stable = -np.expm1(inner)
    direct = 1.0 - (1.0 - q) ** n
    return np.where(q < 0.5, stable, direct)


def expected_inv_k_plus_1(q: Any, F: int) -> Any:
    """E[1/(K + 1)] con K ~ Bin(F, q): (1 − (1 − q)^(F+1)) / ((F + 1)·q); límite 1 si q → 0."""
    q = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
    safe_q = np.where(q > 0, q, 1.0)
    return np.where(q > 0, _one_minus_pow(q, F + 1) / ((F + 1) * safe_q), 1.0)


def travel_fraction(q: Any, F: int) -> Any:
    """E[1/(K + 1) | K ≥ 1]: fracción esperada de L hasta el verde más cercano.

    frac = (E[1/(K + 1)] − P0) / (1 − P0), con P0 = (1 − q)^F; si 1 − P0 < 1e-8
    se usa el límite 1/2 (un solo verde visible).
    """
    q = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
    p0 = (1.0 - q) ** F
    not_p0 = _one_minus_pow(q, F)
    ok = not_p0 >= _P0_EPS
    frac = (expected_inv_k_plus_1(q, F) - p0) / np.where(ok, not_p0, 1.0)
    return np.where(ok, frac, 0.5)


def _travel(prob: Mapping[str, Any], prm: Mapping[str, float]) -> dict[str, Any]:
    """Recorrido hasta el libre verde más cercano (estadística de orden exacta)."""
    F = prob["F"]
    q = np.clip(prob["q_LL_eff"], 0.0, 1.0)
    frac = travel_fraction(q, F)
    t_direct = np.minimum(prm["t_acceso_min"] + prm["t_recorrido_total_min"] * frac,
                          prm["t_base_min"])
    return {
        "k_vis": F * q,
        "frac_recorrido": frac,
        "factor_recorrido": frac * (F + 1),
        "t_directo_eff": t_direct,
    }


def _time_with_system(
    prob: Mapping[str, Any], travel: Mapping[str, Any], prm: Mapping[str, float]
) -> tuple[Any, Any]:
    """Paso 2: T_sys con tope en t_base + t_redireccion, y si el tope se activó."""
    e_c, P0 = prob["e_c"], prob["P0"]
    guided = travel["t_directo_eff"] + (e_c / (1 - e_c)) * prm["t_redireccion_min"]
    t_mixed = (1 - P0) * guided + P0 * prm["t_base_min"]
    cap = prm["t_base_min"] + prm["t_redireccion_min"]
    return np.minimum(t_mixed, cap), t_mixed > cap


def _abandonment(t: Any, prm: Mapping[str, float]) -> Any:
    """p_ab(t) = clip(abandono_base · t / t_base, 0, 1)."""
    return np.clip(prm.get("abandono_base", 0.0) * t / prm["t_base_min"], 0.0, 1.0)


def _amber_spaces(rates: Mapping[str, Any], prm: Mapping[str, float]) -> Any:
    """n_amber = N·(π·q_OV + (1 − π)·q_LV)."""
    pi = prm["ocupacion"]
    q_OV = np.asarray(rates["q_OV"], dtype=float)
    q_LV = np.asarray(rates["q_LV"], dtype=float)
    return prm["espacios"] * (pi * q_OV + (1 - pi) * q_LV)


def _system_costs(prm: Mapping[str, float], days: float, months: float, extra_ops: Any) -> dict[str, Any]:
    """Paso 4 (costos): verificación (personal extra), infraestructura e inferencia del periodo."""
    hours = prm["horas_operacion_dia"]
    pred = prm["espacios"] * (3600.0 / prm["refresh_s"]) * hours * days
    infra_month = (prm["n_camaras"] * prm["costo_camara_Q"]) / prm["meses_amortizacion"] \
        + prm["costo_servidor_mes_Q"]
    return {
        "pred": pred,
        "costo_verif": extra_ops * prm["salario_operador_Q_h"] * hours * days,
        "costo_infra": infra_month * months,
        "costo_inf": pred / 1000.0 * _cost_per_1000(prm),
    }


def _sensor_costs(prm: Mapping[str, float], months: float) -> float:
    """Costo de la alternativa con un sensor por espacio en el periodo (comparativa)."""
    monthly = (prm["espacios"] * prm["costo_sensor_instalado_Q"]) / prm["meses_amortizacion"] \
        + prm["mantenimiento_sensores_mes_Q"]
    return monthly * months


def _assemble(
    scenario_cfg: Mapping[str, Any],
    prm: Mapping[str, float],
    T_sys: Any,
    n_amber: Any,
    with_system: bool,
) -> dict[str, Any]:
    """Pasos 2-4 a partir de T_sys: ahorro, abandono, volúmenes, costos y valor."""
    sfx, days, months = _period(scenario_cfg, prm)
    t_base = prm["t_base_min"]
    A = t_base - T_sys
    p_ab_base = _abandonment(t_base, prm)
    p_ab_sys = _abandonment(T_sys, prm)
    delta_ab = p_ab_base - p_ab_sys
    arrivals = prm["llegadas_dia"] * days

    staff = verification_load(n_amber, prm)
    costs = _system_costs(prm, days, months, staff["operadores_extra"])
    if not with_system:
        costs = {k: 0.0 for k in costs}
    total_costs = costs["costo_infra"] + costs["costo_inf"] + costs["costo_verif"]
    income = arrivals * delta_ab * prm.get("ingreso_por_visita_Q", 0.0)
    hours_saved = arrivals * A / 60.0
    vt = time_value_q_h(prm)
    users_value = hours_saved * vt
    sensors = _sensor_costs(prm, months)

    out = {
        "T_sys": T_sys,
        "A": A,
        "minutos_ahorrados_por_llegada": A,
        "p_ab_base": p_ab_base,
        "p_ab_sys": p_ab_sys,
        "delta_abandono": delta_ab,
        "n_amber": n_amber,
        **staff,
        "costo_por_1000_pred_Q": _cost_per_1000(prm),
        "valor_tiempo_Q_h": vt,
        "meses_periodo": months,
        f"llegadas_{sfx}": arrivals,
        f"pred_{sfx}": costs["pred"],
        f"costo_verif_{sfx}": costs["costo_verif"],
        f"costo_infra_{sfx}": costs["costo_infra"],
        f"costo_inf_{sfx}": costs["costo_inf"],
        f"costos_{sfx}": total_costs,
        f"ingreso_recuperado_{sfx}": income,
        f"valor_operador_{sfx}": income - total_costs,
        f"horas_ahorradas_{sfx}": hours_saved,
        f"valor_usuarios_{sfx}_Q": users_value,
        f"valor_neto_{sfx}_Q": users_value - total_costs,
        f"costo_sensores_{sfx}": sensors,
        f"ahorro_vs_sensores_{sfx}": (sensors - total_costs) if with_system else 0.0,
    }
    objective = scenario_cfg["objective"]
    if objective not in out:
        raise KeyError(f"El objetivo '{objective}' no es un componente válido: {sorted(out)}")
    out["objective"] = out[objective]
    return out


def _to_native(values: dict[str, Any]) -> dict[str, Any]:
    """Convierte escalares de numpy a bool/float; deja los arreglos como arreglos."""
    native: dict[str, Any] = {}
    for key, val in values.items():
        if isinstance(val, (str, bool, int)):
            native[key] = val
        elif np.ndim(val) == 0:
            native[key] = bool(val) if np.asarray(val).dtype == bool else float(val)
        else:
            native[key] = np.asarray(val)
    return native


def json_safe(values: Mapping[str, Any]) -> dict[str, Any]:
    """Copia lista para JSON estricto: NaN/inf → None y arreglos → listas."""
    def convert(val: Any) -> Any:
        if isinstance(val, np.ndarray):
            return convert(val.tolist())
        if isinstance(val, list):
            return [convert(v) for v in val]
        if isinstance(val, float) and not np.isfinite(val):
            return None
        return val

    return {k: convert(v) for k, v in values.items()}


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def compute_value(rates: Mapping[str, Any], scenario_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Calcula todos los componentes de valor y ``objective`` para un escenario.

    ``rates`` debe traer las seis tasas ``q_OL, q_OV, q_OO, q_LL, q_LV, q_LO``
    (``ValueError`` si falta alguna). Pueden ser escalares (salida de
    ``three_state_rates``) o arreglos de numpy que se difunden entre sí; en ese
    caso cada componente es un arreglo (lo usa la búsqueda vectorizada de umbrales).
    """
    _check_rates(rates)
    prm = scenario_params(scenario_cfg)
    prob = _probabilities(rates, prm)
    travel = _travel(prob, prm)
    T_sys, capped = _time_with_system(prob, travel, prm)
    n_amber = _amber_spaces(rates, prm)
    out = {
        "scenario": scenario_cfg["name"],
        "objective_name": scenario_cfg["objective"],
        **prob,
        **travel,
        "t_sys_capped": capped,
        **_assemble(scenario_cfg, prm, T_sys, n_amber, with_system=True),
    }
    return _to_native(out)


def status_quo(scenario_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Valor sin sistema (B0): T_sys = t_base, A = 0 y sin costos → objetivo 0.

    No hay recomendación, así que ``e``, ``k_vis`` y ``t_directo_eff`` se
    reportan 0 y ``frac_recorrido`` y ``factor_recorrido`` son NaN (no aplican;
    ``json_safe`` los convierte a null); ``P0`` = 1.
    """
    prm = scenario_params(scenario_cfg)
    out = {
        "scenario": scenario_cfg["name"],
        "objective_name": scenario_cfg["objective"],
        "F": free_spaces(prm),
        "q_OL_eff": 0.0,
        "q_LL_eff": 0.0,
        "e": 0.0,
        "e_c": 0.0,
        "P0": 1.0,
        "k_vis": 0.0,
        "frac_recorrido": float("nan"),
        "factor_recorrido": float("nan"),
        "t_directo_eff": 0.0,
        "t_sys_capped": False,
        **_assemble(scenario_cfg, prm, float(prm["t_base_min"]), 0.0, with_system=False),
    }
    return _to_native(out)


def with_param(scenario_cfg: Mapping[str, Any], name: str, value: float) -> dict[str, Any]:
    """Copia del escenario con un parámetro reemplazado (recortado si es probabilidad)."""
    cfg = copy.deepcopy(dict(scenario_cfg))
    cfg["params"] = scenario_params(cfg)
    if name not in cfg["params"]:
        raise KeyError(f"Parámetro desconocido: {name}")
    lo, hi = _PROB_BOUNDS.get(name, (-np.inf, np.inf))
    cfg["params"][name] = float(np.clip(value, lo, hi))
    return cfg


def _default_sensitivity_params(scenario_cfg: Mapping[str, Any]) -> list[str]:
    """Todos los parámetros numéricos distintos de cero (escalar 0 no cambia nada)."""
    return [k for k, v in scenario_params(scenario_cfg).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v != 0]


def _evaluate_point(
    rates: Mapping[str, Any] | None,
    scenario_cfg: Mapping[str, Any],
    reoptimize: bool,
    y: Any,
    p: Any,
) -> tuple[float, float | None, float | None]:
    """Objetivo con tasas fijas o re-optimizando los umbrales."""
    if reoptimize:
        from cupo.thresholds import optimize  # importación local para evitar ciclo

        best = optimize(y, p, scenario_cfg)
        return best["value"], best["t_L"], best["t_H"]
    res = compute_value(rates, scenario_cfg)
    return res["objective"], rates.get("t_L"), rates.get("t_H")


def sensitivity(
    rates: Mapping[str, Any] | None,
    scenario_cfg: Mapping[str, Any],
    params: Iterable[str] | None = None,
    factors: Iterable[float] = (0.5, 0.8, 1.2, 1.5),
    reoptimize: bool = False,
    y: Any = None,
    p: Any = None,
) -> pd.DataFrame:
    """Tabla para el gráfico tornado: cada parámetro multiplicado por cada factor.

    Con ``reoptimize=False`` se mantienen las tasas ``rates`` (umbrales fijos);
    con ``reoptimize=True`` se re-optimizan ``t_L`` y ``t_H`` con ``y`` y ``p``
    para cada variación. Las probabilidades se recortan a su rango válido
    (``ocupacion`` ≤ 0.97) y esas filas se marcan con ``clipped=True``.
    Columnas: param, factor, base_param_value, param_value, clipped, objective,
    base_objective, delta, t_L, t_H.
    """
    if reoptimize and (y is None or p is None):
        raise ValueError("reoptimize=True requiere y y p")
    if not reoptimize and rates is None:
        raise ValueError("reoptimize=False requiere rates")
    names = list(params) if params is not None else _default_sensitivity_params(scenario_cfg)
    base_prm = scenario_params(scenario_cfg)
    base_obj, _, _ = _evaluate_point(rates, scenario_cfg, reoptimize, y, p)

    rows = []
    for name in names:
        for factor in factors:
            target = base_prm[name] * factor
            cfg = with_param(scenario_cfg, name, target)
            obj, t_L, t_H = _evaluate_point(rates, cfg, reoptimize, y, p)
            rows.append({
                "param": name,
                "factor": float(factor),
                "base_param_value": float(base_prm[name]),
                "param_value": float(cfg["params"][name]),
                "clipped": bool(not np.isclose(cfg["params"][name], target)),
                "objective": float(obj),
                "base_objective": float(base_obj),
                "delta": float(obj - base_obj),
                "t_L": t_L,
                "t_H": t_H,
            })
    return pd.DataFrame(rows)


def value_curve(y: Any, p: Any, scenario_cfg: Mapping[str, Any], grid_step: float = 0.01) -> dict[str, pd.DataFrame]:
    """Curvas de valor de un modelo.

    - ``by_t_H``: valor en función de ``t_H`` ∈ [0, 1] con el ``t_L`` óptimo (``t_L ≤ t_H``) para cada
      ``t_H``, junto con q_OL, q_LO y la tasa de verificación en ese punto.
    - ``single_threshold``: decisión de dos estados (``t_L = t_H = t``); valor
      frente a ``t`` junto con accuracy, F1 macro y la tasa ocupado→libre, para
      graficar valor frente a una métrica técnica.
    """
    from cupo.thresholds import curve_by_t_H, curve_single_threshold  # evita ciclo

    return {
        "by_t_H": curve_by_t_H(y, p, scenario_cfg, grid_step),
        "single_threshold": curve_single_threshold(y, p, scenario_cfg, grid_step),
    }
