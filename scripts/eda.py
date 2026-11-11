"""Análisis exploratorio de datos (EDA) de CNR-EXT y CNRPark A/B.

Uso:
    python scripts/eda.py [--sample-per-split N] [--seed 0] [--out RUTA] [--full-hash]

Genera en ``results/eda/`` figuras PNG y ``eda_resumen.md`` (tablas y hallazgos para
la sección 4.2 del informe), además de ``eda_stats.json`` con los números clave.

Nota sobre test y ood: este script carga esas particiones con ``allow_test=True``
únicamente para DESCRIBIR los datos (conteos, distribuciones, fechas, brillo y
calidad de archivos). No carga modelos ni calcula métricas de desempeño sobre
ellas, así que no influye en la selección de modelos, umbrales ni hiperparámetros.
Las únicas comparaciones tipo "predicción" se miden solo en validación: los atajos sin
imagen (tasa de train por espacio, hora o cámara) y las AUC del contraste y del brillo.
La revisión de consistencia de etiquetas lee ``test.txt`` solo para comparar sus etiquetas
con las del CSV (calidad de datos), no para evaluar.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cupo import data, frames, paths  # noqa: E402
from cupo.transforms import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

SPLITS: tuple[str, ...] = ("train", "val", "test", "ood")
WEATHERS: tuple[str, ...] = ("SUNNY", "OVERCAST", "RAINY")
WEATHER_ES: dict[str, str] = {"SUNNY": "soleado", "OVERCAST": "nublado", "RAINY": "lluvia"}
MONTHS_ES: tuple[str, ...] = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
                             "septiembre", "octubre", "noviembre", "diciembre")
LABEL_ES: dict[int, str] = {0: "libre", 1: "ocupado"}
LABEL_COLOR: dict[int, str] = {0: "#2ca02c", 1: "#d62728"}
SPLIT_COLOR: dict[str, str] = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#9467bd", "ood": "#8c564b"}
PATCH_SIZE: tuple[int, int] = (150, 150)

# Umbrales para marcar imágenes atípicas (escala de grises 0-255)
DARK_THRESHOLD = 40.0        # brillo medio por debajo de esto = muy oscura
BLACK_THRESHOLD = 10.0       # brillo medio por debajo de esto = prácticamente negra
BRIGHT_THRESHOLD = 200.0     # brillo medio por encima de esto = sobreexpuesta (reflejos)
ROBUST_Z = 3.5               # |z| robusto (mediana/MAD) dentro de partición y clima
# Un espacio se considera "casi siempre" ocupado/libre si su tasa pasa estos límites
EXTREME_RATES: tuple[float, ...] = (0.95, 0.80)
MIN_SLOT_SAMPLES = 20
# Columnas de CNRPark+EXT.csv que leen cupo.data y cupo.frames
CSV_USED_COLUMNS: tuple[str, ...] = (
    "camera", "datetime", "hour", "minute", "image_url", "occupancy", "slot_id", "weather",
)
DPI = 130


# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------

@dataclass
class Report:
    """Acumula secciones del markdown y números clave para el JSON."""

    sections: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add(self, text: str) -> None:
        self.sections.append(text.strip() + "\n")


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    """Convierte un DataFrame en tabla markdown (sin dependencias extra)."""
    frame = df.reset_index() if index else df
    cols = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def _fmt(value: object) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{value:,.1f}"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return str(value)


def save_fig(fig: plt.Figure, out_dir: Path, name: str, dpi: int = DPI) -> str:
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=dpi)
    plt.close(fig)
    return name


def hour_of(df: pd.DataFrame) -> pd.Series:
    return df["time"].str[:2].astype(int)


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def load_all() -> dict[str, pd.DataFrame]:
    """Carga las cuatro particiones.

    test y ood se cargan con allow_test=True SOLO para estadística descriptiva
    (EDA); no se evalúa ningún modelo sobre ellas.
    """
    out: dict[str, pd.DataFrame] = {}
    for name in SPLITS:
        df = data.load_split(name, allow_test=name in data.PROTECTED_SPLITS)
        df["hour"] = hour_of(df)
        out[name] = df
    return out


# ---------------------------------------------------------------------------
# 1. Tamaño de particiones
# ---------------------------------------------------------------------------

def split_sizes(splits: dict[str, pd.DataFrame], out_dir: Path, rep: Report) -> None:
    rows = []
    for name, df in splits.items():
        rows.append({
            "partición": name,
            "patches": len(df),
            "ocupados": int(df["label"].sum()),
            "libres": int((df["label"] == 0).sum()),
            "% ocupado": 100 * df["label"].mean(),
            "cámaras": df["camera"].nunique(),
            "fechas": df["date"].nunique(),
            "espacios": df["slot_id"].nunique(),
        })
    table = pd.DataFrame(rows).set_index("partición")
    rep.stats["sizes"] = {r["partición"]: {"n": r["patches"], "busy_pct": round(r["% ocupado"], 2)} for r in rows}

    fig, ax = plt.subplots(figsize=(7, 4.2))
    names = list(table.index)
    ax.bar(names, table["libres"], color=LABEL_COLOR[0], label="libre")
    ax.bar(names, table["ocupados"], bottom=table["libres"], color=LABEL_COLOR[1], label="ocupado")
    for i, name in enumerate(names):
        ax.text(i, table.loc[name, "patches"], f"{table.loc[name, 'patches']:,}\n{table.loc[name, '% ocupado']:.1f} % ocup.",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("número de patches")
    ax.set_title("Tamaño de cada partición y proporción de ocupados")
    ax.set_ylim(0, table["patches"].max() * 1.18)
    ax.legend()
    fig_name = save_fig(fig, out_dir, "01_particiones.png")

    rep.add(f"""
## 1. Particiones

{md_table(table)}

![Particiones]({fig_name})
""")


# ---------------------------------------------------------------------------
# 2. Balance por cámara y clima
# ---------------------------------------------------------------------------

def _balance(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by)["label"].agg(n="size", ocupados="sum")
    g["% ocupado"] = 100 * g["ocupados"] / g["n"]
    return g


def _plot_balance(splits: dict[str, pd.DataFrame], by: str, order: list[str], title: str,
                  out_dir: Path, name: str, label_map: dict[str, str] | None = None) -> str:
    fig, axes = plt.subplots(1, len(splits), figsize=(4.2 * len(splits), 4), sharey=False)
    for ax, (split, df) in zip(axes, splits.items()):
        g = _balance(df, by).reindex([k for k in order if k in set(df[by])])
        labels = [label_map.get(k, k) if label_map else k for k in g.index]
        free = g["n"] - g["ocupados"]
        ax.bar(labels, free, color=LABEL_COLOR[0], label="libre")
        ax.bar(labels, g["ocupados"], bottom=free, color=LABEL_COLOR[1], label="ocupado")
        for i, (n, p) in enumerate(zip(g["n"], g["% ocupado"])):
            ax.text(i, n, f"{p:.0f}%", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{split} (n={len(df):,})")
        ax.set_ylim(0, g["n"].max() * 1.15)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("número de patches")
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    return save_fig(fig, out_dir, name)


def class_balance(splits: dict[str, pd.DataFrame], out_dir: Path, rep: Report) -> None:
    cameras = sorted({c for df in splits.values() for c in df["camera"]}, key=lambda c: (not c.isdigit(), c.zfill(2)))
    cam_fig = _plot_balance(splits, "camera", cameras, "Balance de clases por cámara (% ocupado sobre cada barra)",
                            out_dir, "02_balance_camara.png")
    w_fig = _plot_balance(splits, "weather", list(WEATHERS), "Balance de clases por clima (% ocupado sobre cada barra)",
                          out_dir, "03_balance_clima.png", WEATHER_ES)

    cam_tab = pd.concat({s: _balance(df, "camera")["% ocupado"] for s, df in splits.items()}, axis=1)
    cam_n = pd.concat({s: _balance(df, "camera")["n"] for s, df in splits.items()}, axis=1)
    cam_tab = cam_tab.reindex(cameras)
    cam_n = cam_n.reindex(cameras).astype("Int64")
    cam_out = pd.DataFrame({f"{s} n": cam_n[s] for s in splits} | {f"{s} % ocup.": cam_tab[s] for s in splits})
    cam_out.index.name = "cámara"

    w_rows = []
    for s, df in splits.items():
        g = _balance(df, "weather").reindex(list(WEATHERS))
        for w in WEATHERS:
            n = g.loc[w, "n"]
            w_rows.append({"partición": s, "clima": WEATHER_ES[w],
                           "n": 0 if pd.isna(n) else int(n),
                           "% de la partición": 0.0 if pd.isna(n) else 100 * n / len(df),
                           "% ocupado": g.loc[w, "% ocupado"]})
    w_tab = pd.DataFrame(w_rows)

    val_rain = int((splits["val"]["weather"] == "RAINY").sum())
    ood_weathers = sorted(splits["ood"]["weather"].unique())
    rep.stats["val_rainy"] = val_rain
    rep.stats["ood_weathers"] = ood_weathers
    rep.add(f"""
## 2. Balance de clases por cámara y por clima

**Verificación:** val tiene {val_rain} patches con lluvia ({'confirmado: val no tiene lluvia' if val_rain == 0 else 'ATENCIÓN: sí hay lluvia'}).
Climas presentes en ood: {', '.join(WEATHER_ES[w] for w in ood_weathers)}
({'confirmado: ood es todo soleado' if ood_weathers == ['SUNNY'] else 'ATENCIÓN: ood no es solo soleado'}).

### Por clima
{md_table(w_tab, index=False)}

![Balance por clima]({w_fig})

### Por cámara
{md_table(cam_out)}

![Balance por cámara]({cam_fig})
""")


# ---------------------------------------------------------------------------
# 3. Distribución temporal
# ---------------------------------------------------------------------------

def temporal(splits: dict[str, pd.DataFrame], out_dir: Path, rep: Report) -> None:
    dates = {s: set(df["date"]) for s, df in splits.items()}
    overlap_rows = []
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap_rows.append({"par": f"{a} ∩ {b}", "fechas compartidas": len(dates[a] & dates[b])})
    overlap = pd.DataFrame(overlap_rows)
    no_leak = all(r["fechas compartidas"] == 0 for r in overlap_rows)
    rep.stats["date_overlap_total"] = int(overlap["fechas compartidas"].sum())

    date_tab = pd.DataFrame([{
        "partición": s, "fechas": len(dates[s]), "desde": min(dates[s]), "hasta": max(dates[s]),
        "lista": ", ".join(sorted(dates[s])),
    } for s in names])

    # Línea de tiempo: un punto por fecha, color según clima predominante
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for y, s in enumerate(names):
        per_date = splits[s].groupby("date").agg(n=("label", "size"), weather=("weather", lambda x: x.mode().iloc[0]))
        x = pd.to_datetime(per_date.index)
        colors = per_date["weather"].map({"SUNNY": "#f2b701", "OVERCAST": "#7f7f7f", "RAINY": "#1f77b4"})
        ax.scatter(x, [y] * len(x), s=20 + per_date["n"] / per_date["n"].max() * 160, c=colors, edgecolors="k")
    ax.set_yticks(range(len(names)), names)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_title("Fechas de cada partición (tamaño = patches del día, color = clima)")
    ax.legend(handles=[Patch(color="#f2b701", label="soleado"), Patch(color="#7f7f7f", label="nublado"),
                       Patch(color="#1f77b4", label="lluvia")], loc="upper center", ncol=3, fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fechas_fig = save_fig(fig, out_dir, "04_fechas.png")

    # Ocupación por hora del día
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    hour_rows = {}
    for s, df in splits.items():
        g = df.groupby("hour")["label"].agg(["mean", "size"])
        hour_rows[s] = 100 * g["mean"]
        ax1.plot(g.index, 100 * g["mean"], marker="o", color=SPLIT_COLOR[s], label=s)
        ax2.plot(g.index, g["size"], marker="o", color=SPLIT_COLOR[s], label=s)
    ax1.set_title("% ocupado por hora del día")
    ax1.set_xlabel("hora")
    ax1.set_ylabel("% ocupado")
    ax2.set_title("Patches por hora del día")
    ax2.set_xlabel("hora")
    ax2.set_ylabel("número de patches")
    ax1.legend()
    hora_fig = save_fig(fig, out_dir, "05_ocupacion_hora.png")
    hour_tab = pd.DataFrame(hour_rows)
    hour_tab.index.name = "hora"

    rep.add(f"""
## 3. Distribución temporal

{md_table(date_tab, index=False)}

{md_table(overlap, index=False)}

**{'Sin fechas compartidas entre particiones: no hay fuga por días repetidos.' if no_leak else 'ATENCIÓN: hay fechas compartidas entre particiones.'}**

![Fechas]({fechas_fig})

### % ocupado por hora
{md_table(hour_tab)}

![Ocupación por hora]({hora_fig})
""")


# ---------------------------------------------------------------------------
# 4. Ocupación por espacio (slot)
# ---------------------------------------------------------------------------

def _slot_rates(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("slot_id")["label"].agg(rate="mean", n="size")


def _extreme_summary(df: pd.DataFrame) -> dict[str, float]:
    rates = _slot_rates(df)
    rates = rates[rates["n"] >= MIN_SLOT_SAMPLES]
    out: dict[str, float] = {"espacios (n≥20)": len(rates)}
    for hi in EXTREME_RATES:
        busy = rates[rates["rate"] >= hi]
        free = rates[rates["rate"] <= 1 - hi]
        out[f"ocupados ≥{hi:.0%}"] = len(busy)
        out[f"libres ≥{hi:.0%}"] = len(free)
        out[f"% patches (umbral {hi:.0%})"] = 100 * (busy["n"].sum() + free["n"].sum()) / rates["n"].sum()
    return out


def _prior(train: pd.DataFrame, target: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Tasa de ocupación de train por ``keys`` asignada a cada fila de ``target`` (sin mirar la imagen).

    Si la combinación no existe en train se usa la tasa global de train.
    """
    rates = train.groupby(keys)["label"].mean().rename("prior")
    merged = target[keys].merge(rates, left_on=keys, right_index=True, how="left")
    return merged["prior"].fillna(train["label"].mean())


def shortcut_table(train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    """AUC y accuracy en VAL de predictores triviales construidos solo con metadatos de train."""
    rows = []
    for name, keys in (("espacio", ["slot_id"]), ("hora", ["hour"]), ("cámara", ["camera"]),
                       ("espacio + hora", ["slot_id", "hour"])):
        score = _prior(train, val, keys)
        rows.append({"predictor (tasa de train por…)": name,
                     "AUC val": roc_auc_score(val["label"], score),
                     "accuracy val (umbral 0.5)": 100 * float(((score >= 0.5).astype(int) == val["label"]).mean())})
    major = int(train["label"].mean() >= 0.5)
    rows.append({"predictor (tasa de train por…)": f"siempre '{LABEL_ES[major]}' (mayoritaria de train)",
                 "AUC val": 0.5, "accuracy val (umbral 0.5)": 100 * float((val["label"] == major).mean())})
    return pd.DataFrame(rows)


def slot_occupancy(splits: dict[str, pd.DataFrame], out_dir: Path, rep: Report) -> None:
    # La identidad del espacio solo es comparable dentro de EXT (CNRPark tiene otra numeración)
    summary = pd.DataFrame({s: _extreme_summary(df) for s, df in splits.items()}).T
    summary.index.name = "partición"
    count_cols = [c for c in summary.columns if not c.startswith("%")]
    summary[count_cols] = summary[count_cols].astype(int)

    train_rates = _slot_rates(splits["train"])
    val_rates = _slot_rates(splits["val"])

    # Atajos medidos SOLO en val: predictores que no miran la imagen
    shortcuts = shortcut_table(splits["train"], splits["val"])
    rep.stats["shortcuts_val"] = shortcuts.set_index("predictor (tasa de train por…)").round(3).to_dict(orient="index")
    rep.stats["slot_extremes"] = summary.to_dict(orient="index")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = np.linspace(0, 1, 21)
    for s in ("train", "val", "test"):
        r = _slot_rates(splits[s])
        r = r[r["n"] >= MIN_SLOT_SAMPLES]
        ax1.hist(r["rate"], bins=bins, histtype="step", linewidth=2, color=SPLIT_COLOR[s], label=f"{s} ({len(r)} espacios)")
    ax1.set_title("Distribución de la tasa de ocupación por espacio")
    ax1.set_xlabel("fracción del tiempo ocupado")
    ax1.set_ylabel("número de espacios")
    ax1.legend(fontsize=8)
    both = train_rates.add_suffix("_train").join(val_rates.add_suffix("_val"), how="inner")
    both = both[(both["n_train"] >= MIN_SLOT_SAMPLES) & (both["n_val"] >= MIN_SLOT_SAMPLES)]
    ax2.scatter(both["rate_train"], both["rate_val"], s=18, alpha=0.7)
    ax2.plot([0, 1], [0, 1], "k--", linewidth=1)
    corr = float(both["rate_train"].corr(both["rate_val"])) if len(both) > 2 else float("nan")
    rep.stats["slot_rate_corr_train_val"] = round(corr, 3)
    ax2.set_title(f"Ocupación por espacio: train vs val (r = {corr:.2f})")
    ax2.set_xlabel("tasa en train")
    ax2.set_ylabel("tasa en val")
    slot_fig = save_fig(fig, out_dir, "06_ocupacion_slot.png")

    top = train_rates[train_rates["n"] >= MIN_SLOT_SAMPLES].sort_values("rate")
    extremes = pd.concat([top.head(5), top.tail(5)])
    extremes["rate"] = 100 * extremes["rate"]
    extremes = extremes.rename(columns={"rate": "% ocupado (train)", "n": "patches (train)"})
    sc = shortcuts.set_index("predictor (tasa de train por…)")["AUC val"]
    val_hour = 100 * splits["val"].groupby("hour")["label"].mean()

    rep.add(f"""
## 4. Ocupación por espacio y otros atajos posibles

Espacios con ocupación extrema (solo espacios con al menos {MIN_SLOT_SAMPLES} patches):

{md_table(summary)}

Espacios más extremos en train:

{md_table(extremes)}

**Atajos medidos en validación** (la "predicción" es la tasa de ocupación de train para ese espacio, hora o
cámara; no se mira la imagen):

{md_table(shortcuts.assign(**{"AUC val": shortcuts["AUC val"].map("{:.3f}".format)}), index=False)}

- Correlación de la tasa de ocupación por espacio entre train y val: r = {corr:.2f} ({len(both)} espacios en común).
- El espacio por sí solo predice poco (AUC {sc['espacio']:.2f}): no hay un atajo fuerte por posición, porque casi
  ningún espacio está siempre ocupado o siempre libre en train. La **hora del día** predice mucho más
  (AUC {sc['hora']:.2f}; {sc['espacio + hora']:.2f} combinada con el espacio). El modelo no recibe la hora; solo podría
  inferirla por la iluminación, y el brillo por sí solo casi no separa las clases (sección 5). Aun así conviene
  revisarlo con Grad-CAM, con el desglose por clima y con ood (espacios y cámaras nuevos).
- La ocupación cambia mucho con la hora (en val: {val_hour.min():.0f} % a las {val_hour.idxmin()}h y
  {val_hour.max():.0f} % a las {val_hour.idxmax()}h): respalda usar una ocupación de escenario `π` (hora pico) en el
  modelo de valor en lugar de la proporción de una partición.

![Ocupación por espacio]({slot_fig})
""")


# ---------------------------------------------------------------------------
# 5 y 6. Muestra estratificada: estadísticas de imagen y calidad
# ---------------------------------------------------------------------------

def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Muestra proporcional por (clima, etiqueta) con semilla fija."""
    if n >= len(df):
        return df.reset_index(drop=True)
    frac = n / len(df)
    parts = [g.sample(n=max(1, int(round(frac * len(g)))), random_state=seed)
             for _, g in df.groupby(["weather", "label"], sort=True)]
    return pd.concat(parts).reset_index(drop=True)


def image_record(path: str) -> dict[str, object]:
    """Lee una imagen y devuelve hash de archivo y de píxeles, tamaño, brillo y contraste."""
    rec: dict[str, object] = {"ok": False, "md5_file": None, "md5_pixels": None,
                              "width": np.nan, "height": np.nan, "brightness": np.nan, "contrast": np.nan,
                              "error": ""}
    try:
        raw = Path(path).read_bytes()
        rec["md5_file"] = hashlib.md5(raw).hexdigest()
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            rec["width"], rec["height"] = im.size
            rgb = np.asarray(im.convert("RGB"))
        gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.float32)
        rec["md5_pixels"] = hashlib.md5(rgb.tobytes()).hexdigest()
        rec["brightness"] = float(gray.mean())
        rec["contrast"] = float(gray.std())
        rec["ok"] = True
    except Exception as exc:  # noqa: BLE001 - cualquier fallo de lectura cuenta como archivo corrupto
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def build_sample(splits: dict[str, pd.DataFrame], n: int, seed: int) -> pd.DataFrame:
    parts = []
    for s, df in splits.items():
        smp = stratified_sample(df, n, seed)
        recs = pd.DataFrame([image_record(p) for p in smp["path"]])
        parts.append(pd.concat([smp.assign(split=s), recs], axis=1))
    sample = pd.concat(parts, ignore_index=True)
    return flag_outliers(sample)


def flag_outliers(sample: pd.DataFrame) -> pd.DataFrame:
    """Marca imágenes muy oscuras o atípicas en brillo (z robusto por partición y clima).

    El contraste bajo no se marca: es lo normal en un espacio libre (asfalto uniforme).
    """
    ok = sample["ok"]
    med = sample[ok].groupby(["split", "weather"])["brightness"].transform("median")
    mad = sample[ok].groupby(["split", "weather"])["brightness"].transform(lambda x: (x - x.median()).abs().median())
    z = 0.6745 * (sample.loc[ok, "brightness"] - med) / mad.replace(0, np.nan)
    sample["robust_z"] = z.reindex(sample.index)
    sample["dark"] = sample["brightness"] < DARK_THRESHOLD
    sample["black"] = sample["brightness"] < BLACK_THRESHOLD
    sample["overexposed"] = sample["brightness"] > BRIGHT_THRESHOLD
    sample["atypical"] = sample["robust_z"].abs() > ROBUST_Z
    return sample


def image_stats(sample: pd.DataFrame, out_dir: Path, rep: Report, n: int, seed: int) -> None:
    ok = sample[sample["ok"]]
    stats = (ok.groupby(["split", "weather", "label"])[["brightness", "contrast"]]
             .agg(["mean", "std"]).round(1))
    stats.columns = ["brillo medio", "brillo desv.", "contraste medio", "contraste desv."]
    stats = stats.reset_index()
    stats["weather"] = stats["weather"].map(WEATHER_ES)
    stats["label"] = stats["label"].map(LABEL_ES)
    stats = stats.rename(columns={"split": "partición", "weather": "clima", "label": "clase"})
    stats["partición"] = pd.Categorical(stats["partición"], SPLITS, ordered=True)
    stats = stats.sort_values(["partición", "clima", "clase"])

    fig, axes = plt.subplots(2, len(SPLITS), figsize=(4.2 * len(SPLITS), 7), sharey="row")
    for j, s in enumerate(SPLITS):
        sub = ok[ok["split"] == s]
        for i, (metric, name) in enumerate((("brightness", "brillo medio"), ("contrast", "contraste (desv. est.)"))):
            ax = axes[i, j]
            groups, labels, colors = [], [], []
            for w in WEATHERS:
                for lab in (0, 1):
                    vals = sub[(sub["weather"] == w) & (sub["label"] == lab)][metric]
                    if len(vals):
                        groups.append(vals.to_numpy())
                        labels.append(f"{WEATHER_ES[w]}\n{LABEL_ES[lab]}")
                        colors.append(LABEL_COLOR[lab])
            bp = ax.boxplot(groups, patch_artist=True, showfliers=False)
            ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=7)
            for box, c in zip(bp["boxes"], colors):
                box.set_facecolor(c)
                box.set_alpha(0.5)
            if i == 0:
                ax.set_title(f"{s} (muestra n={len(sub):,})")
            if j == 0:
                ax.set_ylabel(f"{name} (0-255)")
    fig.suptitle("Brillo y contraste de los patches por clima y clase (escala de grises)")
    box_fig = save_fig(fig, out_dir, "07_brillo_contraste.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    for w in WEATHERS:
        sub = ok[ok["weather"] == w]
        if len(sub):
            g = sub.groupby("hour")["brightness"].mean()
            ax.plot(g.index, g.values, marker="o", label=WEATHER_ES[w])
    ax.axhline(DARK_THRESHOLD, color="k", linestyle="--", linewidth=1, label=f"umbral oscuro ({DARK_THRESHOLD:.0f})")
    ax.set_title("Brillo medio por hora del día y clima (muestra de todas las particiones)")
    ax.set_xlabel("hora")
    ax.set_ylabel("brillo medio (0-255)")
    ax.legend()
    hour_fig = save_fig(fig, out_dir, "08_brillo_hora.png")

    flags = (ok.groupby("split")[["dark", "black", "overexposed", "atypical"]].sum()
             .reindex(list(SPLITS)).astype(int))
    flags.insert(0, "muestra", ok.groupby("split").size().reindex(list(SPLITS)))
    busy_dark = ok[ok["dark"]].groupby("split")["label"].mean().reindex(list(SPLITS)) * 100
    flags["% ocupado entre las oscuras"] = busy_dark
    flags.columns = ["muestra", f"muy oscuras (<{DARK_THRESHOLD:.0f})", f"casi negras (<{BLACK_THRESHOLD:.0f})",
                     f"sobreexpuestas (>{BRIGHT_THRESHOLD:.0f})", f"atípicas en brillo (z robusto > {ROBUST_Z})",
                     "% ocupado entre las oscuras"]
    flags.index.name = "partición"
    dark = ok[ok["dark"]]
    dark_hours = dark["hour"].value_counts().sort_index()
    rep.stats["image_flags"] = flags.to_dict(orient="index")
    rep.stats["dark_busy_pct"] = round(100 * float(dark["label"].mean()), 1) if len(dark) else None
    rep.stats["black_in_sample"] = int(ok["black"].sum())
    rep.stats["black_busy_in_sample"] = int(ok.loc[ok["black"], "label"].sum())

    # Señales simples de imagen, medidas SOLO en la muestra de val
    val_s = ok[ok["split"] == "val"]
    cue_auc = {
        "contraste": roc_auc_score(val_s["label"], val_s["contrast"]),
        "brillo": roc_auc_score(val_s["label"], val_s["brightness"]),
    }
    rep.stats["image_cue_auc_val"] = {k: round(float(v), 3) for k, v in cue_auc.items()}
    rep.stats["image_cue_auc_val_sym"] = {k: round(float(max(v, 1 - v)), 3) for k, v in cue_auc.items()}
    outlier_fig = plot_outliers(ok, out_dir)
    texts = image_texts(ok)
    rep.stats["dark_profile"] = texts["dark_short"]

    rep.add(f"""
## 5. Estadísticas de imagen (muestra estratificada)

Muestra proporcional por clima y clase: hasta {n:,} patches por partición, semilla {seed}.
Brillo = media de la imagen en escala de grises; contraste = desviación estándar.

{md_table(stats, index=False)}

![Brillo y contraste]({box_fig})

![Brillo por hora]({hour_fig})

- **El contraste separa mucho las clases:** {texts['contrast']} Usado solo como puntaje da
  AUC = {cue_auc['contraste']:.3f} en la muestra de val. Esto explica por qué un clasificador clásico (HOG) ya
  funciona bien, y sugiere que los errores se concentrarán en sombras fuertes (soleado) y en autos oscuros o de
  poco contraste.
- **El brillo casi no separa las clases:** AUC = {cue_auc['brillo']:.3f} en val
  ({'relación inversa leve: los ocupados son un poco más oscuros' if cue_auc['brillo'] < 0.5 else 'relación directa leve'};
  max(AUC, 1 − AUC) = {max(cue_auc['brillo'], 1 - cue_auc['brillo']):.3f}).
- {texts['ood']}

### Imágenes oscuras o atípicas
{md_table(flags)}

Horas de las imágenes muy oscuras: {', '.join(f'{h}h ({c})' for h, c in dark_hours.items()) or 'ninguna'}.
En esta corrida: {texts['dark']} Entre ellas hay
{int(ok["black"].sum())} prácticamente negras (brillo <{BLACK_THRESHOLD:.0f}), de las cuales
{int(ok.loc[ok["black"], "label"].sum())} están etiquetadas como ocupadas: con esos píxeles la clase no se puede
decidir, así que son error irreducible y el caso natural para "por verificar".

Sobreexpuestas (brillo >{BRIGHT_THRESHOLD:.0f}) en esta corrida: {texts['bright']}

![Atípicas]({outlier_fig})
""")


def _range(values: pd.Series) -> str:
    return f"{values.min():.0f}-{values.max():.0f}"


def image_texts(ok: pd.DataFrame) -> dict[str, str]:
    """Frases de la sección 5 calculadas con la muestra (nada escrito a mano)."""
    ext = ok[ok["subset"] == "EXT"]
    ood = ok[ok["subset"] == "CNRPark"]
    by = ext.groupby(["split", "weather", "label"])
    contrast = by["contrast"].mean()
    bright = by["brightness"].mean()
    out: dict[str, str] = {}
    out["contrast"] = (
        f"un espacio libre es asfalto casi uniforme (contraste medio {_range(contrast.xs(0, level='label'))} según "
        f"partición y clima) y un auto agrega bordes (contraste medio {_range(contrast.xs(1, level='label'))})."
    )
    ood_free = ood.loc[ood["label"] == 0, "brightness"].mean()
    ext_free = bright.xs(0, level="label")
    lighter = ood_free > ext_free.max()
    out["ood"] = (
        f"En ood (CNRPark) el brillo medio de los espacios libres es {ood_free:.0f}, frente a {_range(ext_free)} en EXT"
        + (": el pavimento se ve más claro, hay cambio de dominio en la apariencia." if lighter
           else "; no hay un cambio claro de brillo en los libres.")
    )

    dark = ok[ok["dark"]]
    if len(dark):
        early = 100 * float((dark["hour"] <= 8).mean())
        late = 100 * float((dark["hour"] >= 16).mean())
        weather = ", ".join(f"{WEATHER_ES[w]} {c}" for w, c in dark["weather"].value_counts().items())
        out["dark_short"] = f"{early:.0f} % de 7-8h y {late:.0f} % de 16h en adelante"
        out["dark"] = f"el {early:.0f} % es de las 7-8h y el {late:.0f} % de las 16h en adelante (por clima: {weather})."
    else:
        out["dark_short"] = "sin imágenes muy oscuras"
        out["dark"] = "no hay imágenes muy oscuras en la muestra."

    out["bright"] = _bright_text(ok[ok["overexposed"]])
    return out


def _bright_text(over: pd.DataFrame) -> str:
    """Describe las sobreexpuestas de EXT y de ood por separado (clase y hora)."""
    if over.empty:
        return "ninguna."
    parts = []
    for subset, name in (("EXT", "EXT"), ("CNRPark", "ood")):
        sub = over[over["subset"] == subset]
        if sub.empty:
            continue
        busy = 100 * float(sub["label"].mean())
        times = ", ".join(sorted(sub["time"].unique()))
        desc = f"{name}: {len(sub)} ({busy:.0f} % ocupadas; horas {times})"
        if busy == 100:
            desc += (f". Son autos claros o con destello solar (ver 09_atipicas.png). Con solo {len(sub)} casos es una "
                     "observación débil, pero sugiere que el brillo extremo aparece sobre todo con 'ocupado'")
        elif busy == 0:
            desc += ". Todas son pavimento libre muy claro con el sol bajo (ver 09_atipicas.png)"
        parts.append(desc)
    return "; ".join(parts) + "."


def plot_outliers(ok: pd.DataFrame, out_dir: Path) -> str:
    """Mosaico de las imágenes más oscuras y de las más claras de la muestra."""
    darkest = ok.nsmallest(8, "brightness")
    brightest = ok.nlargest(8, "brightness")
    fig, axes = plt.subplots(2, 8, figsize=(14, 4.8))
    for row, (title, sel) in enumerate((("más oscuras", darkest), ("más claras", brightest))):
        for ax, rec in zip(axes[row], sel.itertuples()):
            with Image.open(rec.path) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_title(f"{rec.split} {rec.time}\n{LABEL_ES[rec.label]} b={rec.brightness:.0f}", fontsize=7)
            ax.axis("off")
        axes[row, 0].text(-0.15, 0.5, title, transform=axes[row, 0].transAxes, rotation=90, va="center", ha="right")
    fig.suptitle("Patches más oscuros y más claros de la muestra")
    return save_fig(fig, out_dir, "09_atipicas.png", dpi=100)


def quality(sample: pd.DataFrame, splits: dict[str, pd.DataFrame], full_hash: bool, rep: Report) -> None:
    corrupt = sample[~sample["ok"]]
    ok = sample[sample["ok"]]
    bad_size = ok[(ok["width"] != PATCH_SIZE[0]) | (ok["height"] != PATCH_SIZE[1])]
    sizes = bad_size["width"].astype(int).astype(str) + "x" + bad_size["height"].astype(int).astype(str)
    size_counts = (bad_size.assign(tam=sizes)
                   .groupby("split").agg(n=("tam", "size"), tamaños=("tam", lambda x: ", ".join(sorted(set(x)))))
                   .reindex([x for x in SPLITS if x in set(bad_size["split"])]))
    size_counts.index.name = "partición"

    rows = []
    for s in SPLITS:
        sub = ok[ok["split"] == s]
        rows.append({
            "partición": s, "muestra": len(sample[sample["split"] == s]),
            "corruptas": int((~sample[sample["split"] == s]["ok"]).sum()),
            "tamaño ≠ 150x150": int(len(bad_size[bad_size["split"] == s])),
            "duplicados internos (archivo)": int(sub["md5_file"].duplicated().sum()),
            "duplicados internos (píxeles)": int(sub["md5_pixels"].duplicated().sum()),
        })
    qtab = pd.DataFrame(rows)

    def cross(a: str, b: str, col: str) -> int:
        ha = set(ok[ok["split"] == a][col])
        hb = set(ok[ok["split"] == b][col])
        return len(ha & hb)

    cross_rows = [{"par": f"{a} ∩ {b}", "archivo idéntico": cross(a, b, "md5_file"), "píxeles idénticos": cross(a, b, "md5_pixels")}
                  for a, b in (("train", "test"), ("train", "val"), ("val", "test"), ("train", "ood"))]
    ctab = pd.DataFrame(cross_rows)
    rep.stats["corrupt_in_sample"] = int(len(corrupt))
    rep.stats["non_150_in_sample"] = int(len(bad_size))
    rep.stats["dup_train_test_sample"] = int(cross_rows[0]["archivo idéntico"])

    full_text = ""
    if full_hash:
        shared = full_train_test_hash(splits)
        rep.stats["dup_train_test_full"] = len(shared)
        full_text = (f"\nHash completo (no solo la muestra): {len(splits['train']):,} archivos de train y "
                     f"{len(splits['test']):,} de test; **{len(shared)} contenidos de archivo compartidos**.\n")
        if len(shared):
            full_text += "\n" + md_table(shared, index=False) + "\n\n" + _shared_text(shared, rep) + "\n"

    errors = "\n".join(f"- `{Path(p).name}` ({s}): {e}" for p, s, e in corrupt[["path", "split", "error"]].itertuples(index=False))
    rep.add(f"""
## 6. Calidad de los archivos (muestra)

{md_table(qtab, index=False)}

Duplicados exactos entre particiones (hash MD5 de la muestra; posible fuga si > 0):

{md_table(ctab, index=False)}
{full_text}
{('Tamaños distintos de 150x150 (patches recortados en el borde del frame; el Resize del preprocesamiento '
  'los lleva al tamaño de entrada):' + chr(10) + chr(10) + md_table(size_counts)) if len(size_counts) else 'Todas las imágenes leídas de la muestra miden 150x150.'}

{('Archivos ilegibles:' + chr(10) + errors) if len(corrupt) else 'No se encontraron archivos corruptos o ilegibles en la muestra.'}
""")


def _shared_text(shared: pd.DataFrame, rep: Report) -> str:
    """Interpreta los contenidos repetidos entre train y test (negros y etiquetas contradictorias)."""
    total = shared["archivos train"] + shared["archivos test"]
    black = shared["brillo"] < BLACK_THRESHOLD
    contradictory = (shared["ocupados"] > 0) & (shared["ocupados"] < total)
    rep.stats["dup_full_files"] = int(total.sum())
    rep.stats["dup_full_busy"] = int(shared["ocupados"].sum())
    rep.stats["dup_full_contradictory"] = int(contradictory.sum())
    lines = []
    if black.all():
        lines.append("Todos los contenidos repetidos son patches completamente negros (cámara sin luz), no fotos "
                     "reales repetidas: no es una fuga de información entre train y test.")
    else:
        lines.append(f"{int(black.sum())} de {len(shared)} contenidos repetidos son patches negros; el resto "
                     "debe revisarse como posible fuga.")
    if contradictory.any():
        lines.append(f"Los {int(total[contradictory].sum())} archivos idénticos tienen etiquetas contradictorias "
                     f"({int(shared.loc[contradictory, 'ocupados'].sum())} ocupados y "
                     f"{int((total - shared['ocupados'])[contradictory].sum())} libres): es **ruido de etiqueta**. "
                     "Ningún modelo puede acertar todos; aportan un piso de error y deberían caer en \"por verificar\".")
    return " ".join(lines)


def label_consistency(rep: Report) -> None:
    """Valores nulos en LABELS y en el CSV, y acuerdo entre las etiquetas de LABELS y la columna occupancy.

    Lee también ``test.txt`` SOLO para esta revisión de calidad de datos (no evalúa nada).
    """
    # Se omite data.load_split (guarda de test) a propósito: solo se cuentan nulos y se compara con el CSV;
    # nada de esto llega a modelos ni umbrales.
    # Lectura cruda (como texto) para detectar nulos o etiquetas inválidas sin conversiones previas
    labels = pd.concat({
        s: pd.read_csv(paths.split_file(s), sep=" ", header=None, names=["rel", "label"], dtype=str)
        for s in ("train", "val", "test")
    }, names=["split"]).reset_index(0).reset_index(drop=True)
    csv = pd.read_csv(paths.metadata_csv(), dtype=str)
    used = set(CSV_USED_COLUMNS)

    lab_rows = [{"archivo": f"LABELS/{s}.txt", "filas": len(g), "nulos": int(g[["rel", "label"]].isna().sum().sum()),
                 "etiquetas fuera de {0,1}": int((~g["label"].isin(["0", "1"])).sum())}
                for s, g in labels.groupby("split", sort=False)]
    labels = labels[labels["label"].isin(["0", "1"])].assign(label=lambda d: d["label"].astype(int))
    csv_rows = [{"columna del CSV": c, "nulos": int(csv[c].isna().sum()), "se usa": "sí" if c in used else "no"}
                for c in csv.columns]

    prefix = f"{paths.EXT_DIRNAME}/PATCHES/"
    ext = csv[csv["image_url"].str.startswith(prefix)]
    ext = pd.DataFrame({"rel": ext["image_url"].str[len(prefix):], "occupancy": ext["occupancy"].astype(int)})
    merged = labels.merge(ext, on="rel", how="left")
    missing = merged["occupancy"].isna()
    diff = (~missing) & (merged["occupancy"] != merged["label"])
    by_split = pd.DataFrame({
        "patches": merged.groupby("split", sort=False).size(),
        "sin fila en el CSV": missing.groupby(merged["split"], sort=False).sum().astype(int),
        "etiqueta distinta": diff.groupby(merged["split"], sort=False).sum().astype(int),
    })
    by_split["% distinta"] = (100 * by_split["etiqueta distinta"] / by_split["patches"]).map("{:.3f}".format)
    by_split.index.name = "partición"

    n_diff, n_total = int(diff.sum()), len(merged)
    csv_used_nulls = int(sum(r["nulos"] for r in csv_rows if r["se usa"] == "sí"))
    unused_nulls = {r["columna del CSV"]: r["nulos"] for r in csv_rows if r["se usa"] == "no" and r["nulos"]}
    rep.stats["labels_nulls"] = int(sum(r["nulos"] for r in lab_rows))
    rep.stats["csv_used_nulls"] = csv_used_nulls
    rep.stats["csv_unused_nulls"] = unused_nulls
    rep.stats["label_mismatch"] = n_diff
    rep.stats["label_mismatch_pct"] = round(100 * n_diff / n_total, 3)
    rep.stats["labels_missing_in_csv"] = int(missing.sum())

    unused_txt = (", ".join(f"`{c}` tiene {n:,} nulos" for c, n in unused_nulls.items())
                  + " pero no se usa en el proyecto.") if unused_nulls else "Ninguna columna sin usar tiene nulos."
    rep.add(f"""
### Valores nulos y consistencia de etiquetas (todas las filas, no la muestra)

Esta subsección se calcula sobre todas las filas de LABELS/{{train,val,test}}.txt y de CNRPark+EXT.csv.

{md_table(pd.DataFrame(lab_rows), index=False)}

{md_table(pd.DataFrame(csv_rows), index=False)}

- Nulos en LABELS: {rep.stats['labels_nulls']}. Nulos en las columnas del CSV que se usan: {csv_used_nulls}. {unused_txt}
- Acuerdo entre LABELS y la columna `occupancy` del CSV (mismo archivo):

{md_table(by_split)}

- En total, {n_diff} de {n_total:,} patches ({100 * n_diff / n_total:.3f} %) tienen etiqueta distinta en LABELS y en el CSV,
  y {int(missing.sum())} no aparecen en el CSV. El proyecto usa siempre las etiquetas de LABELS (particiones oficiales);
  el CSV solo aporta metadatos de ood y las etiquetas de los frames completos de la demo.
""")


def full_train_test_hash(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Hash MD5 de TODOS los archivos de train y test (más lento; solo con --full-hash).

    Devuelve una fila por contenido compartido entre train y test, con cuántos archivos
    lo repiten, sus etiquetas y el brillo de la imagen.
    """
    frames_ = []
    for name in ("train", "test"):
        df = splits[name][["path", "label"]].copy()
        df["split"] = name
        df["md5"] = [hashlib.md5(Path(p).read_bytes()).hexdigest() for p in df["path"]]
        frames_.append(df)
    allh = pd.concat(frames_, ignore_index=True)
    in_both = allh.groupby("md5")["split"].nunique()
    dup = allh[allh["md5"].isin(in_both[in_both > 1].index)]
    rows = []
    for md5, g in dup.groupby("md5"):
        with Image.open(g["path"].iloc[0]) as im:
            bright = float(np.asarray(im.convert("L"), dtype=np.float32).mean())
        rows.append({"md5": md5[:10], "archivos train": int((g["split"] == "train").sum()),
                     "archivos test": int((g["split"] == "test").sum()),
                     "ocupados": int(g["label"].sum()), "brillo": bright})
    return pd.DataFrame(rows, columns=["md5", "archivos train", "archivos test", "ocupados", "brillo"])


# ---------------------------------------------------------------------------
# 7. Ejemplos visuales
# ---------------------------------------------------------------------------

def example_mosaic(train: pd.DataFrame, out_dir: Path, seed: int, n_cols: int = 6) -> str:
    """Patches de train: una fila por (clima, clase)."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(WEATHERS) * 2, n_cols, figsize=(1.7 * n_cols, 1.85 * len(WEATHERS) * 2))
    for i, (w, lab) in enumerate((w, lab) for w in WEATHERS for lab in (0, 1)):
        pool = train[(train["weather"] == w) & (train["label"] == lab)]
        picks = pool.iloc[rng.choice(len(pool), size=min(n_cols, len(pool)), replace=False)]
        for j, ax in enumerate(axes[i]):
            ax.axis("off")
            if j < len(picks):
                rec = picks.iloc[j]
                with Image.open(rec["path"]) as im:
                    ax.imshow(im.convert("RGB"))
                ax.set_title(f"cám {rec['camera']} {rec['time']}", fontsize=7)
        axes[i, 0].text(-0.1, 0.5, f"{WEATHER_ES[w]}\n{LABEL_ES[lab]}", transform=axes[i, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=9, color=LABEL_COLOR[lab])
    fig.suptitle("Ejemplos de train: libre y ocupado por clima")
    return save_fig(fig, out_dir, "10_mosaico_ejemplos.png", dpi=100)


def pick_frame(train_dates: set[str], seed: int, n_candidates: int = 40) -> pd.Series | None:
    """Elige un frame soleado de una fecha de train con ocupación mezclada (muchos libres y muchos ocupados)."""
    fr = frames.list_frames(allow_test=False)
    fr = fr[fr["date"].isin(train_dates)]
    if fr.empty:
        return None
    sunny = fr[fr["weather"] == "SUNNY"]
    fr = sunny if len(sunny) else fr
    hours = fr["time"].str[:2].astype(int)
    daytime = (hours >= 9) & (hours <= 16)
    fr = fr[daytime] if daytime.any() else fr
    cands = fr.sample(n=min(n_candidates, len(fr)), random_state=seed)
    best, best_score = None, -1
    for _, rec in cands.iterrows():
        gt = frames.ground_truth(int(rec["camera"]), rec["date"], rec["time"])
        busy = sum(gt.values())
        score = min(busy, len(gt) - busy)  # premia frames con ambas clases
        if score > best_score:
            best, best_score = rec, score
    return best


def frame_with_boxes(train: pd.DataFrame, out_dir: Path, seed: int) -> tuple[str | None, str]:
    rec = pick_frame(set(train["date"]), seed)
    if rec is None:
        return None, "No hay frames completos de fechas de train."
    boxes = frames.slot_boxes(int(rec["camera"]))
    gt = frames.ground_truth(int(rec["camera"]), rec["date"], rec["time"])
    fig, ax = plt.subplots(figsize=(10, 7.5))
    with Image.open(rec["path"]) as im:
        ax.imshow(im.convert("RGB"))
    for b in boxes.itertuples(index=False):
        lab = gt.get(b.slot_id)
        color = LABEL_COLOR[lab] if lab is not None else "#bbbbbb"
        ax.add_patch(Rectangle((b.x, b.y), b.w, b.h, fill=False, edgecolor=color, linewidth=1.8))
        ax.text(b.x + 2, b.y + 10, b.slot_id, color="white", fontsize=6,
                bbox={"facecolor": color, "alpha": 0.7, "pad": 0.5, "linewidth": 0})
    ax.legend(handles=[Patch(color=LABEL_COLOR[0], label="libre"), Patch(color=LABEL_COLOR[1], label="ocupado"),
                       Patch(color="#bbbbbb", label="sin etiqueta")], loc="lower right")
    ax.set_title(f"Frame completo con los espacios: cámara {rec['camera']}, {rec['date']} {rec['time']} "
                 f"({WEATHER_ES[rec['weather']]})")
    ax.axis("off")
    name = save_fig(fig, out_dir, "11_frame_cajas.png", dpi=100)
    info = (f"Cámara {rec['camera']}, {rec['date']} {rec['time']} ({WEATHER_ES[rec['weather']]}): "
            f"{len(boxes)} cajas, {len(gt)} con etiqueta ({sum(gt.values())} ocupadas).")
    return name, info


def examples(splits: dict[str, pd.DataFrame], out_dir: Path, rep: Report, seed: int) -> None:
    mosaic = example_mosaic(splits["train"], out_dir, seed)
    frame_fig, info = frame_with_boxes(splits["train"], out_dir, seed)
    frame_md = f"![Frame con cajas]({frame_fig})" if frame_fig else ""
    rep.add(f"""
## 7. Ejemplos

![Mosaico de ejemplos]({mosaic})

{info}

{frame_md}
""")


# ---------------------------------------------------------------------------
# 8. Representatividad, 9. Preprocesamiento, 10. Licencia y hallazgos clave
# ---------------------------------------------------------------------------

def representativeness(splits: dict[str, pd.DataFrame], rep: Report) -> None:
    """Cobertura horaria y diferencias esperables con un parqueo guatemalteco."""
    rows = []
    for s, df in splits.items():
        rows.append({"partición": s, "primera hora": df["time"].min(), "última hora": df["time"].max(),
                     "patches 18h o después": int((df["hour"] >= 18).sum()),
                     "% 18h o después": 100 * float((df["hour"] >= 18).mean())})
    cov = pd.DataFrame(rows)
    first, last = min(r["primera hora"] for r in rows), max(r["última hora"] for r in rows)
    rep.stats["time_range"] = [first, last]
    rep.add(f"""
## 8. Representatividad

{md_table(cov, index=False)}

- **Solo hay imágenes de día:** de {first} a {last}. No hay imágenes nocturnas, y las 18h casi no tienen
  cobertura (ver tabla). El modelo no está validado de noche; en la app eso debe ir a "por verificar" o declararse
  fuera de alcance.
- **Un solo tipo de parqueo:** CNR-EXT es el parqueo de un campus de investigación en Pisa (Italia), con cámaras
  elevadas en edificios y vistas en diagonal. Diferencias esperables con un parqueo guatemalteco (UVG o comercial):
  - sol más alto e intenso todo el año, con sombras duras y reflejos;
  - lluvias fuertes por la tarde en la época lluviosa (mayo-octubre), y val no tiene lluvia para medir ese caso;
  - otra mezcla de vehículos: más pickups, camionetas y motos, que pueden ocupar el espacio solo en parte;
  - cámaras a otra altura o ángulo, techos o pérgolas, y líneas de parqueo menos visibles;
  - horarios de ocupación distintos (clases y jornadas laborales).
- Por eso el resultado en ood (otro parqueo y otras cámaras) es la mejor señal disponible de cuánto se degrada el
  modelo al instalarlo en un lugar nuevo; para la UVG haría falta una calibración con fotos propias.
""")


def preprocessing_note(rep: Report) -> None:
    """Preprocesamiento elegido y cómo responde a lo visto en el EDA."""
    s = rep.stats
    flags = s["image_flags"]
    dark = sum(int(v[next(k for k in v if k.startswith("muy oscuras"))]) for v in flags.values())
    bright = sum(int(v[next(k for k in v if k.startswith("sobreexpuestas"))]) for v in flags.values())
    total = sum(int(v["muestra"]) for v in flags.values())
    rep.add(f"""
## 9. Preprocesamiento derivado del EDA

- **Normalización fija de ImageNet** (media {IMAGENET_MEAN}, desviación {IMAGENET_STD}): no se calculan
  estadísticas con los datos, así que no hay fuga de val ni de test hacia el preprocesamiento.
- **Resize** de los patches (150x150) al tamaño de entrada de cada modelo. El mismo paso absorbe los
  {s['non_150_in_sample']} patches de la muestra que no miden 150x150 (recortes de borde, sección 6).
- **ColorJitter** (brillo, contraste, saturación y tono) solo en entrenamiento: en la muestra hay {dark} imágenes muy
  oscuras y {bright} sobreexpuestas de {total:,} ({100 * (dark + bright) / total:.1f} %), y el brillo cambia con la
  hora y el clima (sección 5). El aumento de datos reduce la dependencia del modelo respecto de la iluminación.
""")


def license_note(rep: Report) -> None:
    rep.add("""
## 10. Licencia y privacidad

- **Licencia:** CNRPark+EXT se distribuye bajo la *Open Data Commons Open Database License (ODbL) v1.0*
  (según cnrpark.it). Permite usar, adaptar y redistribuir, incluso con fines comerciales, con atribución
  y compartiendo bajo la misma licencia las bases de datos derivadas.
- **Privacidad:** los autores redujeron los frames completos de 2592x1944 a 1000x750 "por motivos de privacidad".
  Los patches de 150x150 recortan solo el espacio de parqueo. No usamos datos personales adicionales y la galería
  de la app solo muestra imágenes del dataset, con atribución.
- **Citas:**
  - [1] G. Amato, F. Carrara, F. Falchi, C. Gennaro, C. Meghini, C. Vairo, "Deep learning for decentralized parking
    lot occupancy detection", *Expert Systems with Applications*, 72, 327-334, 2017.
  - [2] G. Amato, F. Carrara, F. Falchi, C. Gennaro, C. Vairo, "Car parking occupancy detection using smart camera
    networks and deep learning", *IEEE ISCC 2016*, 1212-1217.
""")


def key_findings(splits: dict[str, pd.DataFrame], rep: Report) -> str:
    s = rep.stats
    sizes = s["sizes"]
    ext = s["slot_extremes"]["train"]
    sc = s["shortcuts_val"]
    flags = s["image_flags"]
    dark_key = next(k for k in flags["train"] if k.startswith("muy oscuras"))
    cue = s["image_cue_auc_val"]
    ood = splits["ood"]
    months = sorted(ood["date"].str[:7].unique())
    ood_when = " y ".join(f"{MONTHS_ES[int(m[5:7]) - 1]} de {m[:4]}" for m in months)
    n_cams = ood["camera"].nunique()
    lines = [
        "## Hallazgos clave",
        "",
        "- Tamaños: " + ", ".join(f"{k} {v['n']:,} ({v['busy_pct']:.1f} % ocupado)" for k, v in sizes.items()) + ".",
        f"- El balance cambia entre particiones: val ({sizes['val']['busy_pct']:.1f} % ocupado) está más cargada "
        f"hacia ocupado que train ({sizes['train']['busy_pct']:.1f} %); por eso el modelo de valor repondera con la "
        "ocupación del escenario y no usa la proporción de val.",
        f"- Val no tiene lluvia ({s['val_rainy']} patches con lluvia): el desempeño con lluvia solo se verá en test. "
        "Se reportará por clima y se declara como limitación.",
        f"- Ood (CNRPark A/B) es {'todo soleado' if s['ood_weathers'] == ['SUNNY'] else 'de climas ' + ', '.join(s['ood_weathers'])}"
        f", de {ood_when} y con {n_cams} cámaras distintas a las de EXT: mide el cambio de parqueo, no el de clima.",
        f"- Fechas compartidas entre particiones: {s['date_overlap_total']} (separación por fecha confirmada, sin fuga por días).",
        f"- Pocos espacios son extremos en train: {int(ext['ocupados ≥95%'])} ocupado(s) y {int(ext['libres ≥95%'])} libre(s) "
        f"≥95 % del tiempo; {int(ext['ocupados ≥80%'])} y {int(ext['libres ≥80%'])} con umbral 80 % "
        f"(de {int(ext['espacios (n≥20)'])} espacios; {ext['% patches (umbral 80%)']:.1f} % de los patches).",
        f"- Atajos sin mirar la imagen (AUC en val): espacio {sc['espacio']['AUC val']:.3f}, cámara "
        f"{sc['cámara']['AUC val']:.3f}, hora {sc['hora']['AUC val']:.3f}, espacio + hora {sc['espacio + hora']['AUC val']:.3f}. "
        "No hay atajo fuerte por posición; la hora sí predice, pero el modelo solo la vería a través de la iluminación.",
        f"- Señales simples de imagen (AUC en la muestra de val): contraste {cue['contraste']:.3f}; brillo {cue['brillo']:.3f} "
        f"(relación {'inversa' if cue['brillo'] < 0.5 else 'directa'} leve, max(AUC, 1 − AUC) = "
        f"{s['image_cue_auc_val_sym']['brillo']:.3f}).",
        f"- Imágenes muy oscuras (<{DARK_THRESHOLD:.0f}) en la muestra: "
        + ", ".join(f"{k} {int(v[dark_key])}" for k, v in flags.items())
        + f"; {s['dark_profile']} ({s['dark_busy_pct']} % ocupadas). {s['black_in_sample']} son prácticamente "
        f"negras y {s['black_busy_in_sample']} de ellas están etiquetadas como ocupadas: error irreducible, candidatas a 'por verificar'.",
        f"- Calidad (muestra): {s['corrupt_in_sample']} archivos corruptos, {s['non_150_in_sample']} con tamaño distinto "
        f"de 150x150 (recortes de borde) y {s['dup_train_test_sample']} duplicados exactos entre train y test.",
        f"- Nulos: {s['labels_nulls']} en LABELS y {s['csv_used_nulls']} en las columnas usadas del CSV"
        + "".join(f"; `{c}` tiene {n:,} nulos pero no se usa" for c, n in s["csv_unused_nulls"].items())
        + f". {s['label_mismatch']} patches ({s['label_mismatch_pct']:.3f} %) tienen etiqueta distinta en LABELS y en "
        "`occupancy` del CSV; se usan las de LABELS.",
        f"- Solo hay imágenes de día ({s['time_range'][0]} a {s['time_range'][1]}), de un parqueo italiano con cámaras "
        "elevadas: la transferencia a un parqueo guatemalteco no está garantizada (sección 8).",
    ]
    if "dup_train_test_full" in s:
        line = f"- Hash completo train/test: {s['dup_train_test_full']} contenido(s) de archivo compartido(s)"
        if s.get("dup_full_contradictory"):
            line += (f": {s['dup_full_files']} patches negros idénticos con etiquetas contradictorias "
                     f"({s['dup_full_busy']} ocupados, {s['dup_full_files'] - s['dup_full_busy']} libres), "
                     "ruido de etiqueta y no fuga")
        lines.append(line + " (sección 6).")
    return "\n".join(lines) + "\n"


def write_outputs(rep: Report, splits: dict[str, pd.DataFrame], out_dir: Path, args: argparse.Namespace,
                  elapsed: float) -> Path:
    header = (
        "# EDA de CNRPark+EXT\n\n"
        "Generado con `python scripts/eda.py "
        f"--sample-per-split {args.sample_per_split} --seed {args.seed}"
        f"{' --full-hash' if args.full_hash else ''}` (tiempo de ejecución: {elapsed:.0f} s).\n\n"
        "test y ood se usan solo para estadística descriptiva; no se evalúa ningún modelo sobre ellos.\n"
    )
    body = header + "\n" + key_findings(splits, rep) + "\n" + "\n".join(rep.sections)
    md_path = out_dir / "eda_resumen.md"
    md_path.write_text(body, encoding="utf-8")
    (out_dir / "eda_stats.json").write_text(json.dumps(rep.stats, indent=2, ensure_ascii=False, default=_json_default),
                                            encoding="utf-8")
    return md_path


def _json_default(obj: object) -> object:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return round(float(obj), 4)
    return str(obj)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EDA de CNR-EXT y CNRPark A/B (figuras + resumen en markdown).")
    p.add_argument("--sample-per-split", type=int, default=3000,
                   help="patches por partición para estadísticas de imagen y calidad (default: 3000)")
    p.add_argument("--seed", type=int, default=0, help="semilla del muestreo (default: 0)")
    p.add_argument("--out", type=Path, default=None, help="carpeta de salida (default: results/eda)")
    p.add_argument("--full-hash", action="store_true",
                   help="además, hash de TODOS los archivos de train y test para buscar duplicados (más lento)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.sample_per_split <= 0:
        raise SystemExit("--sample-per-split debe ser positivo")
    start = time.perf_counter()
    out_dir = (args.out or paths.results_dir() / "eda").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = Report()

    print(f"DATA_ROOT = {paths.data_root()}")
    splits = load_all()
    print("Particiones cargadas:", {k: len(v) for k, v in splits.items()})

    split_sizes(splits, out_dir, rep)
    class_balance(splits, out_dir, rep)
    temporal(splits, out_dir, rep)
    slot_occupancy(splits, out_dir, rep)
    print(f"Leyendo muestra de {args.sample_per_split} patches por particion...")
    sample = build_sample(splits, args.sample_per_split, args.seed)
    image_stats(sample, out_dir, rep, args.sample_per_split, args.seed)
    quality(sample, splits, args.full_hash, rep)
    label_consistency(rep)
    examples(splits, out_dir, rep, args.seed)
    representativeness(splits, rep)
    preprocessing_note(rep)
    license_note(rep)

    elapsed = time.perf_counter() - start
    md_path = write_outputs(rep, splits, out_dir, args, elapsed)
    print(f"Listo en {elapsed:.1f} s: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
