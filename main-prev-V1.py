from __future__ import annotations

import io
import math
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox, breaks_cusumolsresid, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ardl import ARDL, UECM
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, coint, grangercausalitytests, kpss, zivot_andrews
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_coint_rank

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    Prophet = None
    HAS_PROPHET = False

try:
    from neuralprophet import NeuralProphet, set_log_level as np_set_log_level
    HAS_NEURALPROPHET = True
except Exception:
    NeuralProphet = None
    np_set_log_level = None
    HAS_NEURALPROPHET = False

try:
    from arch.unitroot import DFGLS, PhillipsPerron
    HAS_ARCH = True
except Exception:
    DFGLS = PhillipsPerron = None
    HAS_ARCH = False


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RAMA Econometrics Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

RF = {
    "deep": "#0A463B",
    "dark": "#11594B",
    "green": "#018849",
    "bright": "#00A759",
    "lime": "#95C14E",
    "ink": "#17352D",
    "muted": "#63756E",
    "line": "#D6E8DF",
    "soft": "#EDF7F1",
    "paper": "#F5FAF7",
    "white": "#FFFFFF",
    "gold": "#D7B43C",
    "danger": "#B33A3A",
}

PLOT_COLORS = [RF["deep"], RF["bright"], RF["lime"], "#4A8B7B", "#B7A14A", "#6F7D76"]


# -----------------------------------------------------------------------------
# STYLE v2.2 — réseau inspiré du projet Shiny, sans bandeau supérieur ni hamburger
# -----------------------------------------------------------------------------
STYLE = f"""
<style>
:root {{
  --rf-deep:{RF['deep']}; --rf-dark:{RF['dark']}; --rf-green:{RF['green']};
  --rf-bright:{RF['bright']}; --rf-lime:{RF['lime']}; --rf-paper:{RF['paper']};
  --rf-ink:{RF['ink']}; --rf-muted:{RF['muted']}; --rf-line:{RF['line']};
  --rf-soft:{RF['soft']}; --rf-glass:rgba(255,255,255,.88);
  --rf-shadow:0 18px 60px rgba(17,89,75,.12);
  --rf-font:"EB Garamond","Garamond","Palatino Linotype","Book Antiqua",Palatino,serif;
  --rf-gutter:clamp(18px,2.6vw,46px);
}}

html, body, [class*="css"], .stApp {{ font-family:var(--rf-font)!important; }}
.stApp {{
  background:
    radial-gradient(circle at 85% 9%, rgba(149,193,78,.12), transparent 22%),
    linear-gradient(180deg, rgba(245,250,247,.88), rgba(250,253,251,.94));
  color:var(--rf-ink);
}}
[data-testid="stMainBlockContainer"], .main .block-container {{max-width:none; width:100%; padding:1rem var(--rf-gutter) 5rem; position:relative; z-index:2;}}

/* Réseau animé : points + liaisons, inspiré du projet Shiny fourni */
.rf-network {{position:fixed;inset:0;width:100%;height:100%;z-index:0;pointer-events:none;opacity:.72;overflow:visible;}}
.rf-net-lines line {{stroke:rgba(0,167,89,.20);stroke-width:1.15;vector-effect:non-scaling-stroke;}}
.rf-net-lines .lime {{stroke:rgba(149,193,78,.22);}}
.rf-net-dots circle {{fill:rgba(0,167,89,.54);filter:drop-shadow(0 0 4px rgba(0,167,89,.26));}}
.rf-net-dots circle.lime {{fill:rgba(149,193,78,.68);filter:drop-shadow(0 0 5px rgba(149,193,78,.30));}}
.rf-net-layer-a {{transform-origin:50% 50%;animation:rfDriftA 26s ease-in-out infinite alternate;}}
.rf-net-layer-b {{transform-origin:50% 50%;animation:rfDriftB 34s ease-in-out infinite alternate;}}
.rf-net-layer-c {{transform-origin:50% 50%;animation:rfDriftC 42s ease-in-out infinite alternate;}}
@keyframes rfDriftA {{from{{transform:translate(-8px,4px)}}to{{transform:translate(30px,-18px)}}}}
@keyframes rfDriftB {{from{{transform:translate(8px,-7px)}}to{{transform:translate(-34px,24px)}}}}
@keyframes rfDriftC {{from{{transform:translate(-3px,-2px)}}to{{transform:translate(20px,28px)}}}}
.rf-spark {{position:fixed;width:5px;height:5px;background:rgba(149,193,78,.72);transform:rotate(45deg);box-shadow:0 0 11px rgba(149,193,78,.62);animation:rfTwinkle 3.8s ease-in-out infinite;z-index:0;pointer-events:none;}}
.rf-s1{{left:9%;top:39%;animation-delay:.3s}}.rf-s2{{left:32%;top:12%;animation-delay:1.2s}}.rf-s3{{left:59%;top:78%;animation-delay:2.2s}}.rf-s4{{left:70%;top:18%;animation-delay:.9s}}.rf-s5{{left:93%;top:44%;animation-delay:1.8s}}.rf-s6{{left:43%;top:66%;animation-delay:2.7s}}
@keyframes rfTwinkle {{0%,100%{{opacity:.12;transform:scale(.55) rotate(45deg)}}50%{{opacity:.85;transform:scale(1.45) rotate(45deg)}}}}
/* Sidebar */
section[data-testid="stSidebar"] {{background:rgba(255,255,255,.73)!important;backdrop-filter:blur(16px) saturate(120%);border-right:1px solid rgba(214,232,223,.95);}}
section[data-testid="stSidebar"] > div {{padding-top:1.3rem;}}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{color:var(--rf-dark)!important;}}

.rf-side-title{{font-weight:900;color:var(--rf-dark);font-size:1.30rem;line-height:1.1;margin:.15rem 0 .10rem;letter-spacing:-.02em;}}
.rf-side-subtitle{{color:var(--rf-muted);font-size:.82rem;line-height:1.25;margin:0 0 .85rem;}}
section[data-testid="stSidebar"] .stButton>button{{justify-content:flex-start;text-align:left;border-color:transparent!important;background:transparent!important;color:var(--rf-muted)!important;box-shadow:none!important;border-radius:12px!important;padding:.62rem .72rem!important;}}
section[data-testid="stSidebar"] .stButton>button:hover{{background:linear-gradient(90deg,rgba(0,167,89,.10),rgba(149,193,78,.07))!important;color:var(--rf-dark)!important;transform:translateX(2px)!important;box-shadow:none!important;}}
section[data-testid="stSidebar"] .stButton>button[kind="primary"]{{background:linear-gradient(90deg,rgba(0,167,89,.13),rgba(149,193,78,.10))!important;color:var(--rf-dark)!important;border:1px solid rgba(0,167,89,.16)!important;box-shadow:0 5px 16px rgba(17,89,75,.06)!important;}}
/* Ne jamais appliquer Garamond aux glyphes Material de Streamlit : évite l'affichage de keyboard_double_arrow_left/right en texte. */
[data-testid="stIconMaterial"], [data-testid="collapsedControl"] span, [data-testid="stSidebarCollapseButton"] span,
.material-symbols-rounded, .material-symbols-outlined, .material-icons, [class*="material-symbols"]{{
  font-family:"Material Symbols Rounded","Material Symbols Outlined","Material Icons"!important;
  font-weight:normal!important;font-style:normal!important;letter-spacing:normal!important;text-transform:none!important;
  white-space:nowrap!important;word-wrap:normal!important;direction:ltr!important;-webkit-font-feature-settings:"liga";font-feature-settings:"liga";
}}

/* Cartes / métriques */
.rf-card {{background:var(--rf-glass);backdrop-filter:blur(13px) saturate(116%);border:1px solid rgba(214,232,223,.82);border-radius:22px;
  padding:20px 22px;box-shadow:var(--rf-shadow);margin:10px 0 18px;}}
.rf-soft {{background:linear-gradient(145deg,rgba(237,247,241,.96),rgba(255,255,255,.95));}}
.rf-kicker {{display:inline-flex;align-items:center;gap:8px;color:var(--rf-green);font-size:.76rem;font-weight:900;letter-spacing:.10em;text-transform:uppercase;margin-bottom:7px;}}
.rf-kicker::before {{content:"";width:26px;height:3px;border-radius:999px;background:linear-gradient(90deg,var(--rf-bright),var(--rf-lime));}}
.rf-note {{border:1px solid #BFE3CF;background:rgba(241,250,245,.94);border-left:5px solid var(--rf-bright);padding:13px 15px;border-radius:14px;margin:10px 0;box-shadow:0 6px 18px rgba(17,89,75,.04);}}
[data-testid="stMetric"] {{background:rgba(255,255,255,.86);border:1px solid var(--rf-line);border-radius:18px;padding:16px 18px;box-shadow:0 8px 28px rgba(17,89,75,.065);}}
[data-testid="stMetricValue"] {{color:var(--rf-dark);font-weight:850;}}

/* Contrôles */
.stButton>button, .stDownloadButton>button {{border-radius:12px!important;border:1px solid var(--rf-green)!important;color:var(--rf-green)!important;
  background:rgba(255,255,255,.88)!important;font-weight:800!important;transition:.18s ease!important;min-height:43px;}}
.stButton>button:hover, .stDownloadButton>button:hover {{transform:translateY(-1px);background:var(--rf-green)!important;color:#fff!important;box-shadow:0 9px 22px rgba(1,136,73,.19)!important;}}
.stButton>button[kind="primary"] {{background:linear-gradient(105deg,var(--rf-green),var(--rf-bright))!important;color:#fff!important;border-color:var(--rf-green)!important;box-shadow:0 8px 20px rgba(1,136,73,.18)!important;}}
.stButton>button[kind="primary"]:hover {{background:linear-gradient(105deg,var(--rf-dark),var(--rf-green))!important;}}
[data-baseweb="select"] > div, [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input, [data-testid="stDateInput"] input {{border-radius:12px!important;border-color:#C5DBD0!important;background:rgba(255,255,255,.94)!important;}}
[data-baseweb="select"] > div:focus-within, [data-testid="stNumberInput"] input:focus {{box-shadow:0 0 0 .20rem rgba(0,167,89,.12)!important;border-color:var(--rf-bright)!important;}}
.stTabs [data-baseweb="tab-list"] {{gap:8px;background:rgba(237,247,241,.70);padding:5px;border-radius:14px;}}
.stTabs [data-baseweb="tab"] {{border-radius:10px;color:var(--rf-muted);font-weight:750;padding:10px 14px;}}
.stTabs [aria-selected="true"] {{background:#fff!important;color:var(--rf-dark)!important;box-shadow:0 4px 14px rgba(17,89,75,.08);}}
[data-testid="stFileUploader"] {{background:rgba(255,255,255,.72);border:1.5px dashed #B8D8C8;border-radius:18px;padding:10px;}}
[data-testid="stExpander"] {{background:rgba(255,255,255,.78);border:1px solid var(--rf-line);border-radius:16px;overflow:hidden;}}

h1,h2,h3,h4,h5,h6 {{font-family:var(--rf-font)!important;color:var(--rf-dark);font-weight:850!important;letter-spacing:-.02em;}}
p, label, div, input, textarea, button {{font-family:var(--rf-font)!important;}}

.rf-model-pill {{display:inline-flex;margin:3px 5px 3px 0;padding:.30rem .62rem;border-radius:999px;background:var(--rf-soft);border:1px solid #CDE5D7;color:var(--rf-dark);font-size:.78rem;font-weight:800;}}
.custom-footer {{position:fixed;left:50%;bottom:10px;transform:translateX(-50%);z-index:1001;background:rgba(255,255,255,.78);border:1px solid rgba(214,232,223,.92);border-radius:13px;padding:8px 12px;display:flex;align-items:center;gap:12px;-webkit-backdrop-filter:blur(9px);backdrop-filter:blur(9px);box-shadow:0 6px 20px rgba(17,89,75,.08);}}
.custom-footer .footnote{{margin:0;color:#2C2C2C;font-size:13px;text-align:center;}}
.custom-footer .social{{display:flex;align-items:center;gap:8px;}}
.custom-footer .social img{{height:18px;width:18px;filter:grayscale(100%);opacity:.85;transition:opacity .2s;}}
.custom-footer .social img:hover{{opacity:1;}}
@media(max-width:640px){{.custom-footer{{width:calc(100% - 24px);padding:8px 10px;bottom:8px;gap:10px;flex-wrap:wrap;justify-content:center;}}}}

@media(prefers-reduced-motion:reduce) {{.rf-net-layer-a,.rf-net-layer-b,.rf-net-layer-c,.rf-spark{{animation:none!important}}}}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

NETWORK_HTML = """
<svg class="rf-network" viewBox="0 0 1600 900" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
  <g class="rf-net-layer-a">
    <g class="rf-net-lines">
      <line x1="90" y1="170" x2="250" y2="115"/><line x1="250" y1="115" x2="420" y2="210"/><line x1="420" y1="210" x2="565" y2="135"/>
      <line x1="565" y1="135" x2="730" y2="245" class="lime"/><line x1="730" y1="245" x2="905" y2="155"/><line x1="905" y1="155" x2="1080" y2="235"/>
      <line x1="1080" y1="235" x2="1245" y2="125"/><line x1="1245" y1="125" x2="1485" y2="205" class="lime"/>
      <line x1="250" y1="115" x2="310" y2="330"/><line x1="420" y1="210" x2="520" y2="385"/><line x1="730" y1="245" x2="690" y2="430"/>
      <line x1="905" y1="155" x2="980" y2="365"/><line x1="1245" y1="125" x2="1295" y2="355"/>
    </g>
    <g class="rf-net-dots">
      <circle cx="90" cy="170" r="5.4"/><circle cx="250" cy="115" r="6.5" class="lime"/><circle cx="420" cy="210" r="4.8"/><circle cx="565" cy="135" r="6.1"/>
      <circle cx="730" cy="245" r="4.8" class="lime"/><circle cx="905" cy="155" r="5.8"/><circle cx="1080" cy="235" r="4.8"/><circle cx="1245" cy="125" r="6.2"/><circle cx="1485" cy="205" r="6.5" class="lime"/>
      <circle cx="310" cy="330" r="4.6"/><circle cx="520" cy="385" r="4.8" class="lime"/><circle cx="690" cy="430" r="5.2"/><circle cx="980" cy="365" r="4.7"/><circle cx="1295" cy="355" r="5.5"/>
    </g>
  </g>
  <g class="rf-net-layer-b">
    <g class="rf-net-lines">
      <line x1="60" y1="610" x2="245" y2="520"/><line x1="245" y1="520" x2="405" y2="650"/><line x1="405" y1="650" x2="600" y2="555" class="lime"/>
      <line x1="600" y1="555" x2="790" y2="700"/><line x1="790" y1="700" x2="965" y2="585"/><line x1="965" y1="585" x2="1175" y2="690"/>
      <line x1="1175" y1="690" x2="1365" y2="545" class="lime"/><line x1="1365" y1="545" x2="1550" y2="650"/>
      <line x1="245" y1="520" x2="330" y2="770"/><line x1="600" y1="555" x2="560" y2="800"/><line x1="965" y1="585" x2="1005" y2="805"/><line x1="1365" y1="545" x2="1425" y2="785"/>
    </g>
    <g class="rf-net-dots">
      <circle cx="60" cy="610" r="4.8"/><circle cx="245" cy="520" r="6.3"/><circle cx="405" cy="650" r="4.8" class="lime"/><circle cx="600" cy="555" r="5.6"/>
      <circle cx="790" cy="700" r="4.9"/><circle cx="965" cy="585" r="6.5" class="lime"/><circle cx="1175" cy="690" r="5.3"/><circle cx="1365" cy="545" r="6.1"/><circle cx="1550" cy="650" r="4.8" class="lime"/>
      <circle cx="330" cy="770" r="4.5"/><circle cx="560" cy="800" r="5.3"/><circle cx="1005" cy="805" r="4.9"/><circle cx="1425" cy="785" r="5.4"/>
    </g>
  </g>
  <g class="rf-net-layer-c">
    <g class="rf-net-lines"><line x1="190" y1="410" x2="385" y2="455"/><line x1="385" y1="455" x2="820" y2="470"/><line x1="820" y1="470" x2="1130" y2="440"/><line x1="1130" y1="440" x2="1510" y2="430"/></g>
    <g class="rf-net-dots"><circle cx="190" cy="410" r="4.2"/><circle cx="385" cy="455" r="4.8" class="lime"/><circle cx="820" cy="470" r="5.1"/><circle cx="1130" cy="440" r="4.4"/><circle cx="1510" cy="430" r="4.8" class="lime"/></g>
  </g>
</svg>
<span class="rf-spark rf-s1"></span><span class="rf-spark rf-s2"></span><span class="rf-spark rf-s3"></span>
<span class="rf-spark rf-s4"></span><span class="rf-spark rf-s5"></span><span class="rf-spark rf-s6"></span>
"""
st.markdown(NETWORK_HTML, unsafe_allow_html=True)



# -----------------------------------------------------------------------------
# UTILITAIRES
# -----------------------------------------------------------------------------
@dataclass
class ForecastResult:
    forecast: pd.Series
    lower: Optional[pd.Series] = None
    upper: Optional[pd.Series] = None
    model: Any = None
    model_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    fitted: Optional[pd.Series] = None
    residuals: Optional[pd.Series] = None


def format_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    value = float(value)
    av = abs(value)
    if av >= 1_000_000_000:
        return f"{value / 1_000_000_000:,.2f} Md".replace(",", " ").replace(".", ",")
    if av >= 1_000_000:
        return f"{value / 1_000_000:,.2f} M".replace(",", " ").replace(".", ",")
    if av >= 1_000:
        return f"{value / 1_000:,.2f} k".replace(",", " ").replace(".", ",")
    return f"{value:,.2f}".replace(",", " ").replace(".", ",")


def smape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    a, b = np.asarray(y_true, float), np.asarray(y_pred, float)
    den = np.abs(a) + np.abs(b)
    mask = den > 1e-12
    return float(np.mean(2 * np.abs(b[mask] - a[mask]) / den[mask])) if mask.any() else np.nan


def mase(y_true: Sequence[float], y_pred: Sequence[float], y_train: Sequence[float], seasonality: int = 1) -> float:
    y_train = np.asarray(y_train, float)
    m = max(1, int(seasonality))
    if len(y_train) <= m:
        return np.nan
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.nan
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / scale)


def metrics_table(y_true, y_pred, y_train=None, seasonality=1) -> pd.DataFrame:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return pd.DataFrame(columns=["Mesure", "Valeur"])
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    values = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse,
        "sMAPE": smape(y_true, y_pred),
        "R²": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
    }
    if y_train is not None:
        values["MASE"] = mase(y_true, y_pred, y_train, seasonality)
    return pd.DataFrame({"Mesure": list(values.keys()), "Valeur": list(values.values())})


def apply_plot_style(fig: go.Figure, title: Optional[str] = None, height: int = 500) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        font=dict(family="Garamond, EB Garamond, serif", color=RF["ink"], size=15),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,.72)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=25, r=20, t=65, b=35),
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor=RF["line"], mirror=False, gridcolor="rgba(214,232,223,.55)", zeroline=False)
    fig.update_yaxes(showline=True, linewidth=1, linecolor=RF["line"], mirror=False, gridcolor="rgba(214,232,223,.55)", zeroline=False)
    return fig


PLOT_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "height": 900, "width": 1500, "scale": 2},
}


def infer_frequency(dates: pd.Series) -> Tuple[str, int, str]:
    idx = pd.DatetimeIndex(pd.to_datetime(dates).dropna().sort_values().unique())
    if len(idx) < 3:
        return "MS", 12, "Mensuelle (supposée)"
    try:
        f = pd.infer_freq(idx)
    except Exception:
        f = None
    if f:
        fu = f.upper()
        if fu.startswith(("M", "BM")):
            return "MS", 12, "Mensuelle"
        if fu.startswith("Q"):
            return "QS", 4, "Trimestrielle"
        if fu.startswith(("A", "Y")):
            return "YS", 1, "Annuelle"
        if fu.startswith("W"):
            return "W", 52, "Hebdomadaire"
        if fu.startswith("D"):
            return "D", 7, "Quotidienne"
    diffs = np.diff(idx.view("i8")) / (86400 * 1e9)
    med = float(np.median(diffs)) if len(diffs) else 30
    if 27 <= med <= 32:
        return "MS", 12, "Mensuelle"
    if 80 <= med <= 100:
        return "QS", 4, "Trimestrielle"
    if 350 <= med <= 380:
        return "YS", 1, "Annuelle"
    if 6 <= med <= 8:
        return "W", 52, "Hebdomadaire"
    return "D", 7, "Quotidienne / irrégulière"


def future_dates_from_df(df: pd.DataFrame, periods: int) -> pd.DatetimeIndex:
    freq, _, _ = infer_frequency(df["Date"])
    last = pd.to_datetime(df["Date"]).max()
    return pd.date_range(start=last, periods=periods + 1, freq=freq)[1:]


def make_unique_columns(columns: Sequence[Any]) -> List[str]:
    """Retourne des noms de colonnes non vides et uniques, même après transposition."""
    seen: Dict[str, int] = {}
    unique: List[str] = []
    for i, col in enumerate(columns):
        base = str(col).strip()
        if not base or base.lower() in {"nan", "none"}:
            base = f"Variable_{i+1}"
        count = seen.get(base, 0)
        unique.append(base if count == 0 else f"{base}__{count+1}")
        seen[base] = count + 1
    return unique


def _coerce_date_series(series: pd.Series) -> pd.Series:
    """Parse les dates sans inverser les dates ISO (YYYY-MM-DD) et les dates mensuelles."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    s = series.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    iso = s.str.match(r"^\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?$", na=False)
    if iso.any():
        out.loc[iso] = pd.to_datetime(s.loc[iso], errors="coerce", yearfirst=True, dayfirst=False)
    rest = ~iso
    if rest.any():
        out.loc[rest] = pd.to_datetime(s.loc[rest], errors="coerce", dayfirst=True)
    return out


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype(str).str.strip().str.replace("\u00a0", "", regex=False).str.replace(" ", "", regex=False)
    # Formats français : 1 234,56 ; et formats standards : 1234.56.
    both = s.str.contains(",", regex=False) & s.str.contains(".", regex=False)
    fr_like = both & (s.str.rfind(",") > s.str.rfind("."))
    s.loc[fr_like] = s.loc[fr_like].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    comma_only = s.str.contains(",", regex=False) & ~s.str.contains(".", regex=False)
    s.loc[comma_only] = s.loc[comma_only].str.replace(",", ".", regex=False)
    s = s.str.replace("%", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise dates et variables sans supposer que les libellés de colonnes sont uniques."""
    out = df.copy()
    if out.shape[1] == 0:
        return pd.DataFrame(columns=["Date"])
    out.columns = make_unique_columns(out.columns)
    if "Date" not in out.columns:
        aliases = [c for c in out.columns if re.search(r"^(date|dates|period|periode|période|time|year|annee|année)$", str(c), flags=re.I)]
        date_col = aliases[0] if aliases else out.columns[0]
        out = out.rename(columns={date_col: "Date"})
        out.columns = make_unique_columns(out.columns)
    out["Date"] = _coerce_date_series(out["Date"])
    for pos, c in enumerate(list(out.columns)):
        if c == "Date":
            continue
        # Utiliser iloc pour lire garantit une Series ; l'affectation par nom force ensuite un dtype numérique.
        out[c] = _coerce_numeric_series(out.iloc[:, pos])
    out = out.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    numeric_cols = [c for c in out.columns if c != "Date" and pd.api.types.is_numeric_dtype(out[c])]
    return out[["Date"] + numeric_cols]


def detect_orientation(raw: pd.DataFrame) -> str:
    if raw.empty:
        return "rows"
    row = raw.iloc[0, 1:].astype(str).tolist() if raw.shape[1] > 1 else []
    col = raw.iloc[1:, 0].astype(str).tolist() if raw.shape[0] > 1 else []
    def score(vals):
        return sum(pd.notna(pd.to_datetime(v, errors="coerce", dayfirst=True)) for v in vals)
    return "columns" if score(row) > score(col) else "rows"


def read_uploaded_file(uploaded) -> pd.DataFrame:
    if uploaded.name.lower().endswith(".csv"):
        try:
            return pd.read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            return pd.read_csv(uploaded, sep=";")
    return pd.read_excel(uploaded)


def transpose_if_needed(raw: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "Auto":
        orientation = detect_orientation(raw)
    else:
        orientation = "columns" if mode == "Variables en lignes / dates en colonnes" else "rows"
    if orientation == "columns":
        first = raw.columns[0]
        out = raw.set_index(first).T.reset_index().rename(columns={"index": "Date"})
        out.columns.name = None
        out.columns = make_unique_columns(out.columns)
        return out
    out = raw.copy()
    out.columns = make_unique_columns(out.columns)
    return out


def clean_model_data(df: pd.DataFrame, vars_needed: List[str]) -> pd.DataFrame:
    cols = ["Date"] + list(dict.fromkeys(vars_needed))
    d = df[cols].copy().replace([np.inf, -np.inf], np.nan).dropna()
    return d.sort_values("Date").reset_index(drop=True)


# -----------------------------------------------------------------------------
# SCÉNARIOS EXOGÈNES
# -----------------------------------------------------------------------------
def project_one_series(series: pd.Series, periods: int, method: str, rolling_window: int = 12) -> np.ndarray:
    s = pd.Series(series).dropna().astype(float)
    if len(s) == 0:
        return np.zeros(periods)
    if method == "Dernière valeur":
        return np.repeat(s.iloc[-1], periods)
    if method == "Moyenne récente":
        return np.repeat(s.tail(min(rolling_window, len(s))).mean(), periods)
    if method == "Tendance linéaire":
        n = min(max(8, rolling_window), len(s))
        y = s.tail(n).values
        x = np.arange(n).reshape(-1, 1)
        mdl = LinearRegression().fit(x, y)
        return mdl.predict(np.arange(n, n + periods).reshape(-1, 1))
    if method == "Croissance moyenne":
        pct = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna().tail(min(rolling_window, len(s) - 1))
        g = float(pct.mean()) if len(pct) else 0.0
        vals = []
        cur = float(s.iloc[-1])
        for _ in range(periods):
            cur = cur * (1 + g)
            vals.append(cur)
        return np.asarray(vals)
    return np.repeat(s.iloc[-1], periods)


def build_future_exog(df: pd.DataFrame, exog_vars: List[str], periods: int, method: str, rolling_window: int = 12) -> pd.DataFrame:
    dates = future_dates_from_df(df, periods)
    out = pd.DataFrame(index=dates)
    for v in exog_vars:
        out[v] = project_one_series(df[v], periods, method, rolling_window)
    out.index.name = "Date"
    return out


def validate_future_exog(future: pd.DataFrame, exog_vars: List[str], periods: int) -> pd.DataFrame:
    out = future.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.set_index("Date")
    missing = [c for c in exog_vars if c not in out.columns]
    if missing:
        raise ValueError(f"Variables exogènes absentes du scénario : {', '.join(missing)}")
    out = out[exog_vars].apply(pd.to_numeric, errors="coerce")
    if len(out) < periods:
        raise ValueError(f"Le scénario exogène doit contenir au moins {periods} lignes.")
    out = out.iloc[:periods]
    if out.isna().any().any():
        raise ValueError("Le scénario exogène contient des valeurs manquantes/non numériques.")
    return out


# -----------------------------------------------------------------------------
# FEATURES ML MULTIVARIÉES
# -----------------------------------------------------------------------------
def make_supervised(
    df: pd.DataFrame,
    target: str,
    exog_vars: List[str],
    target_lags: List[int],
    exog_lags: Dict[str, List[int]],
) -> Tuple[pd.DataFrame, pd.Series]:
    base = df.set_index("Date")[[target] + exog_vars].copy()
    X = pd.DataFrame(index=base.index)
    for lag in sorted(set(target_lags)):
        X[f"{target}_L{lag}"] = base[target].shift(lag)
    for v in exog_vars:
        for lag in sorted(set(exog_lags.get(v, [0]))):
            X[f"{v}_L{lag}"] = base[v].shift(lag)
    y = base[target].rename("y")
    both = pd.concat([X, y], axis=1).dropna()
    return both.drop(columns="y"), both["y"]


def recursive_ml_forecast(
    df: pd.DataFrame,
    target: str,
    exog_vars: List[str],
    future_exog: pd.DataFrame,
    periods: int,
    target_lags: List[int],
    exog_lags: Dict[str, List[int]],
    estimator: Any,
) -> Tuple[np.ndarray, Any, pd.Series, pd.Series, List[str]]:
    X, y = make_supervised(df, target, exog_vars, target_lags, exog_lags)
    if len(X) < max(15, X.shape[1] + 5):
        raise ValueError(f"Échantillon d'apprentissage trop court ({len(X)} observations utilisables pour {X.shape[1]} caractéristiques).")
    estimator.fit(X, y)
    fitted = pd.Series(estimator.predict(X), index=y.index, name="fitted")
    residuals = y - fitted

    hist = df.set_index("Date")[[target] + exog_vars].copy()
    combined = pd.concat([hist, future_exog], axis=0, sort=False)
    forecasts: List[float] = []
    future_dates = list(future_exog.index[:periods])
    for dt in future_dates:
        row: Dict[str, float] = {}
        pos = combined.index.get_loc(dt)
        for lag in sorted(set(target_lags)):
            src_pos = pos - lag
            if src_pos < 0:
                raise ValueError("Lag de la cible trop long pour l'historique disponible.")
            row[f"{target}_L{lag}"] = float(combined.iloc[src_pos][target])
        for v in exog_vars:
            for lag in sorted(set(exog_lags.get(v, [0]))):
                src_pos = pos - lag
                if src_pos < 0:
                    raise ValueError(f"Lag {lag} de {v} trop long.")
                row[f"{v}_L{lag}"] = float(combined.iloc[src_pos][v])
        xrow = pd.DataFrame([row]).reindex(columns=X.columns)
        pred = float(estimator.predict(xrow)[0])
        combined.loc[dt, target] = pred
        forecasts.append(pred)
    return np.asarray(forecasts), estimator, fitted, residuals, list(X.columns)


# -----------------------------------------------------------------------------
# MOTEUR DE PRÉVISION
# -----------------------------------------------------------------------------
def forecast_model(
    df: pd.DataFrame,
    target: str,
    periods: int,
    model_name: str,
    params: Dict[str, Any],
    exog_vars: Optional[List[str]] = None,
    future_exog: Optional[pd.DataFrame] = None,
) -> ForecastResult:
    exog_vars = list(dict.fromkeys(exog_vars or []))

    system_model = model_name in {"VAR", "VECM"}

    # Variables endogènes du système
    if system_model:
        system_vars = list(
            dict.fromkeys(
                params.get("system_vars", [target])
            )
        )

        if target not in system_vars:
            system_vars = [target] + system_vars

    else:
        system_vars = [target]

    # Toutes les variables nécessaires à l'estimation
    needed = list(
        dict.fromkeys(
            system_vars + exog_vars
        )
    )

    d = clean_model_data(df, needed)

    if len(d) < 10:
        raise ValueError(
            "Au moins 10 observations complètes sont requises."
        )

    dates = future_dates_from_df(d, periods)

    y = (
        d.set_index("Date")[target]
        .astype(float)
    )

    # -------------------------------------------------------------------------
    # VARIABLES EXOGÈNES
    # -------------------------------------------------------------------------

    # Pour VAR standard : aucune variable exogène externe n'est prévue ici.
    # Les variables du système sont toutes endogènes.
    if model_name == "VAR" and exog_vars:
        raise ValueError(
            "Le VAR standard de cette application traite les variables "
            "sélectionnées comme endogènes. "
            "Retirez les variables exogènes externes."
        )

    # X représente uniquement de vraies variables exogènes.
    X = (
        d.set_index("Date")[exog_vars].astype(float)
        if exog_vars
        else None
    )

    # Un scénario futur est nécessaire uniquement lorsqu'il existe
    # de vraies variables exogènes.
    if exog_vars:

        if future_exog is None:
            raise ValueError(
                "Un scénario futur des variables exogènes est requis "
                f"pour le modèle {model_name}."
            )

        Xf = validate_future_exog(
            future_exog,
            exog_vars,
            periods
        )

        # Alignement exact avec les dates de prévision
        Xf.index = dates

    else:
        Xf = None

    # 1. Benchmark
    if model_name == "Naïf (dernière valeur)":
        vals = np.repeat(float(y.iloc[-1]), periods)
        fitted = y.shift(1)
        return ForecastResult(pd.Series(vals, dates), model_name=model_name, fitted=fitted, residuals=y - fitted)

    # 2. SARIMAX / ARIMAX
    if model_name == "SARIMAX / ARIMAX":
        order = tuple(params.get("order", (1, 1, 1)))
        seasonal_order = tuple(params.get("seasonal_order", (0, 0, 0, 0)))
        trend = params.get("trend", "c")
        mdl = SARIMAX(
            y,
            exog=X,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=params.get("enforce_stationarity", False),
            enforce_invertibility=params.get("enforce_invertibility", False),
        )
        fit = mdl.fit(disp=False, maxiter=int(params.get("maxiter", 200)))
        pr = fit.get_forecast(steps=periods, exog=Xf)
        ci = pr.conf_int(alpha=float(params.get("alpha", .05)))
        fitted = pd.Series(fit.fittedvalues, index=y.index)
        resid = pd.Series(fit.resid, index=y.index)
        return ForecastResult(
            pd.Series(np.asarray(pr.predicted_mean), dates),
            pd.Series(ci.iloc[:, 0].values, dates),
            pd.Series(ci.iloc[:, 1].values, dates),
            model=fit, model_name=model_name, fitted=fitted, residuals=resid,
            metadata={"AIC": fit.aic, "BIC": fit.bic, "HQIC": getattr(fit, "hqic", np.nan), "order": order, "seasonal_order": seasonal_order},
        )

    # 3. ARDL
    if model_name == "ARDL":
        lags = params.get("target_lags", 1)
        exog_order = params.get("exog_order", None)
        if exog_vars:
            order = {v: exog_order.get(v, 0) for v in exog_vars} if isinstance(exog_order, dict) else exog_order
            mdl = ARDL(y, lags=lags, exog=X, order=order, trend=params.get("trend", "c"), causal=params.get("causal", False), missing="drop")
        else:
            mdl = ARDL(y, lags=lags, trend=params.get("trend", "c"), missing="drop")
        fit = mdl.fit(cov_type=params.get("cov_type", "nonrobust"))
        pred = fit.forecast(steps=periods, exog=Xf)
        fitted_raw = fit.fittedvalues
        fitted = pd.Series(fitted_raw.values, index=fitted_raw.index if hasattr(fitted_raw, "index") else y.index[-len(fitted_raw):])
        resid_raw = fit.resid
        resid = pd.Series(np.asarray(resid_raw), index=fitted.index[-len(resid_raw):])
        return ForecastResult(
            pd.Series(np.asarray(pred), dates), model=fit, model_name=model_name,
            fitted=fitted, residuals=resid,
            metadata={"AIC": fit.aic, "BIC": fit.bic, "HQIC": fit.hqic, "target_lags": lags, "exog_order": exog_order},
        )

    # 4. VAR
    if model_name == "VAR":
        system_vars = params.get("system_vars", [target] + exog_vars)
        if target not in system_vars:
            system_vars = [target] + system_vars
        vdf = clean_model_data(df, system_vars).set_index("Date")[system_vars]
        lag = int(params.get("lag_order", 1))
        auto_ic = params.get("auto_ic", "Manuel")
        mdl = VAR(vdf)
        if auto_ic != "Manuel":
            selected = mdl.select_order(maxlags=int(params.get("maxlags", min(12, max(1, len(vdf)//5)))))
            lag = int(getattr(selected, auto_ic.lower()) or 1)
            lag = max(1, lag)
        fit = mdl.fit(lag, trend=params.get("trend", "c"))
        vals = fit.forecast(vdf.values[-fit.k_ar:], steps=periods)
        idx = system_vars.index(target)
        fitted = pd.Series(fit.fittedvalues[target], index=fit.fittedvalues.index)
        resid = pd.Series(fit.resid[target], index=fit.resid.index)
        return ForecastResult(
            pd.Series(vals[:, idx], dates), model=fit, model_name=model_name,
            fitted=fitted, residuals=resid,
            metadata={"AIC": fit.aic, "BIC": fit.bic, "HQIC": fit.hqic, "lag_order": fit.k_ar, "system_vars": system_vars, "stable": fit.is_stable()},
        )

    # 5. VECM
    if model_name == "VECM":
        system_vars = params.get("system_vars", [target] + exog_vars)
        if target not in system_vars:
            system_vars = [target] + system_vars
        vdf = clean_model_data(df, system_vars).set_index("Date")[system_vars]
        fit = VECM(
            vdf,
            k_ar_diff=int(params.get("k_ar_diff", 1)),
            coint_rank=int(params.get("coint_rank", 1)),
            deterministic=params.get("deterministic", "co"),
            seasons=int(params.get("seasons", 0)),
        ).fit()
        vals = fit.predict(steps=periods)
        idx = system_vars.index(target)
        resid = pd.Series(fit.resid[:, idx], index=vdf.index[-len(fit.resid):])
        return ForecastResult(
            pd.Series(vals[:, idx], dates), model=fit, model_name=model_name,
            residuals=resid,
            metadata={"k_ar_diff": int(params.get("k_ar_diff", 1)), "coint_rank": int(params.get("coint_rank", 1)), "system_vars": system_vars},
        )

    # 6. Theta / ETS (univarié par construction)
    if model_name == "Theta / ETS":
        trend = params.get("ets_trend", "add")
        seasonal = params.get("ets_seasonal", None)
        sp = int(params.get("ets_period", 0) or 0)
        if seasonal is not None and (sp <= 1 or len(y) < 2 * sp):
            seasonal = None
        mdl = ExponentialSmoothing(
            y, trend=trend, damped_trend=bool(params.get("damped_trend", False)),
            seasonal=seasonal, seasonal_periods=sp if seasonal is not None else None,
            initialization_method="estimated",
        )
        fit = mdl.fit(optimized=True, use_brute=True)
        pred = fit.forecast(periods)
        fitted = pd.Series(fit.fittedvalues, index=y.index)
        resid = y - fitted
        return ForecastResult(
            pd.Series(np.asarray(pred), dates), model=fit, model_name=model_name, fitted=fitted, residuals=resid,
            metadata={"AIC": getattr(fit, "aic", np.nan), "BIC": getattr(fit, "bic", np.nan), "seasonal_period": sp if seasonal else 0},
        )

    # 7. Prophet avec régresseurs
    if model_name == "Prophet + régresseurs":
        if not HAS_PROPHET:
            raise RuntimeError("Prophet n'est pas installé dans cet environnement.")
        hist = d[["Date", target] + exog_vars].rename(columns={"Date": "ds", target: "y"})
        m = Prophet(
            changepoint_prior_scale=float(params.get("changepoint_prior_scale", .05)),
            seasonality_prior_scale=float(params.get("seasonality_prior_scale", 10.0)),
            seasonality_mode=params.get("seasonality_mode", "additive"),
            yearly_seasonality=params.get("yearly_seasonality", "auto"),
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=1 - float(params.get("alpha", .05)),
        )
        for v in exog_vars:
            m.add_regressor(v, standardize="auto", mode=params.get("regressor_mode", "additive"))
        m.fit(hist)
        future = pd.DataFrame({"ds": dates})
        if exog_vars:
            for v in exog_vars:
                future[v] = Xf[v].values
        pred = m.predict(future)
        hist_pred = m.predict(hist[["ds"] + exog_vars])
        fitted = pd.Series(hist_pred["yhat"].values, index=pd.DatetimeIndex(hist["ds"]))
        resid = y.reindex(fitted.index) - fitted
        return ForecastResult(
            pd.Series(pred["yhat"].values, dates), pd.Series(pred["yhat_lower"].values, dates), pd.Series(pred["yhat_upper"].values, dates),
            model=m, model_name=model_name, fitted=fitted, residuals=resid,
        )

    # 8. NeuralProphet avec régresseurs futurs
    if model_name == "NeuralProphet + régresseurs":
        if not HAS_NEURALPROPHET:
            raise RuntimeError("NeuralProphet n'est pas installé dans cet environnement.")
        if np_set_log_level is not None:
            try:
                np_set_log_level("ERROR")
            except Exception:
                pass
        hist = d[["Date", target] + exog_vars].rename(columns={"Date": "ds", target: "y"}).copy()
        n_lags = int(params.get("np_n_lags", min(12, max(1, len(hist)//6))))
        epochs = int(params.get("np_epochs", 150))
        m = NeuralProphet(
            n_lags=n_lags, n_forecasts=periods, epochs=epochs,
            yearly_seasonality=params.get("np_yearly", "auto"),
            weekly_seasonality=False, daily_seasonality=False,
            seasonality_mode=params.get("np_seasonality_mode", "additive"),
        )
        for v in exog_vars:
            m = m.add_future_regressor(v, mode=params.get("np_regressor_mode", "additive"))
        freq_code, _, _ = infer_frequency(d["Date"])
        try:
            m.fit(hist, freq=freq_code, progress="none")
        except TypeError:
            m.fit(hist, freq=freq_code)
        regressors_df = None
        if exog_vars:
            regressors_df = Xf.reset_index().rename(columns={"index": "ds", "Date": "ds"})
            if "ds" not in regressors_df.columns:
                regressors_df.insert(0, "ds", dates)
        future = m.make_future_dataframe(
            hist, regressors_df=regressors_df, periods=periods, n_historic_predictions=False
        )
        pred = m.predict(future)
        fp = pred[pred["ds"].isin(pd.DatetimeIndex(dates))].reset_index(drop=True)
        vals = []
        yhat_cols = [c for c in pred.columns if re.fullmatch(r"yhat\d+", str(c))]
        for i in range(periods):
            row = fp.iloc[i] if i < len(fp) else pred.iloc[-periods + i]
            preferred = f"yhat{i+1}"
            val = row.get(preferred, np.nan) if hasattr(row, "get") else np.nan
            if pd.isna(val):
                available = [row.get(c) for c in yhat_cols if pd.notna(row.get(c))]
                val = available[0] if available else np.nan
            vals.append(float(val) if pd.notna(val) else np.nan)
        # fitted yhat1 sur l'historique pour diagnostics
        hist_pred = m.predict(hist)
        fitted = pd.Series(hist_pred.get("yhat1", pd.Series(index=hist_pred.index, dtype=float)).values, index=pd.DatetimeIndex(hist_pred["ds"]))
        resid = y.reindex(fitted.index) - fitted
        return ForecastResult(
            pd.Series(vals, dates), model=m, model_name=model_name, fitted=fitted, residuals=resid,
            metadata={"n_lags": n_lags, "epochs": epochs, "n_regressors": len(exog_vars)},
        )

    # 9. ML dynamique
    if model_name in {"Régression dynamique", "Ridge dynamique", "Elastic Net dynamique", "Random Forest", "Gradient Boosting", "XGBoost", "MLP"}:
        target_lags = params.get("target_lags_list", [1, 2, 3])
        exog_lags = params.get("exog_lags", {v: [0, 1] for v in exog_vars})
        if not target_lags and not exog_vars:
            raise ValueError("Sélectionnez au moins un lag de la cible ou une variable exogène.")
        if model_name == "Régression dynamique":
            estimator = LinearRegression()
        elif model_name == "Ridge dynamique":
            estimator = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=float(params.get("alpha_ridge", 1.0))))])
        elif model_name == "Elastic Net dynamique":
            estimator = Pipeline([("scale", StandardScaler()), ("model", ElasticNet(alpha=float(params.get("alpha_en", .05)), l1_ratio=float(params.get("l1_ratio", .5)), max_iter=10000, random_state=42))])
        elif model_name == "Random Forest":
            estimator = RandomForestRegressor(
                n_estimators=int(params.get("n_estimators", 400)), max_depth=params.get("max_depth", None),
                min_samples_leaf=int(params.get("min_samples_leaf", 2)), max_features=params.get("max_features", "sqrt"),
                random_state=42, n_jobs=-1,
            )
        elif model_name == "Gradient Boosting":
            estimator = GradientBoostingRegressor(
                n_estimators=int(params.get("n_estimators", 250)), learning_rate=float(params.get("learning_rate", .05)),
                max_depth=int(params.get("max_depth_gb", 3)), loss=params.get("loss", "huber"), random_state=42,
            )
        elif model_name == "XGBoost":
            if not HAS_XGB:
                raise RuntimeError("XGBoost n'est pas installé.")
            estimator = xgb.XGBRegressor(
                n_estimators=int(params.get("n_estimators", 400)), max_depth=int(params.get("max_depth_xgb", 4)),
                learning_rate=float(params.get("learning_rate", .04)), subsample=float(params.get("subsample", .85)),
                colsample_bytree=float(params.get("colsample_bytree", .85)), reg_alpha=float(params.get("reg_alpha", .0)),
                reg_lambda=float(params.get("reg_lambda", 1.0)), objective="reg:squarederror", random_state=42, n_jobs=-1,
            )
        else:
            layers = tuple(params.get("hidden_layers", [64, 32]))
            estimator = Pipeline([("scale", StandardScaler()), ("model", MLPRegressor(
                hidden_layer_sizes=layers, alpha=float(params.get("mlp_alpha", .001)), learning_rate_init=float(params.get("mlp_lr", .001)),
                max_iter=int(params.get("max_iter", 1200)), early_stopping=True, validation_fraction=.15, random_state=42,
            ))])
        if future_exog is None:
            future_exog = pd.DataFrame(index=dates)
        else:
            future_exog = validate_future_exog(future_exog, exog_vars, periods) if exog_vars else pd.DataFrame(index=dates)
            future_exog.index = dates
        vals, fit, fitted, resid, features = recursive_ml_forecast(
            d, target, exog_vars, future_exog, periods, target_lags, exog_lags, estimator
        )
        return ForecastResult(
            pd.Series(vals, dates), model=fit, model_name=model_name, fitted=fitted, residuals=resid,
            metadata={"features": features, "n_features": len(features)},
        )

    raise ValueError(f"Modèle non pris en charge : {model_name}")


# -----------------------------------------------------------------------------
# BACKTEST CHRONOLOGIQUE
# -----------------------------------------------------------------------------
def backtest_holdout(
    df: pd.DataFrame,
    target: str,
    model_name: str,
    params: Dict[str, Any],
    exog_vars: List[str],
    test_size: int,
    scenario_method: str,
    exog_evaluation_mode: str = "Exogènes observées (conditionnel)",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    needed = [target] + exog_vars
    d = clean_model_data(df, needed)
    if len(d) <= test_size + 12:
        raise ValueError("Historique insuffisant pour le backtest demandé.")
    train = d.iloc[:-test_size].copy()
    test = d.iloc[-test_size:].copy()
    if exog_vars:
        if exog_evaluation_mode.startswith("Exogènes observées"):
            future_x = test.set_index("Date")[exog_vars]
        else:
            future_x = build_future_exog(train, exog_vars, test_size, scenario_method)
    else:
        future_x = None
    res = forecast_model(train, target, test_size, model_name, params, exog_vars, future_x)
    bt = pd.DataFrame({"Date": test["Date"].values, "Observé": test[target].values, "Prévu": res.forecast.values})
    _, s, _ = infer_frequency(train["Date"])
    mt = metrics_table(bt["Observé"], bt["Prévu"], train[target].values, seasonality=s)
    return bt, mt


def rolling_origin_backtest(
    df: pd.DataFrame,
    target: str,
    model_name: str,
    params: Dict[str, Any],
    exog_vars: List[str],
    horizon: int,
    folds: int,
    scenario_method: str,
    exog_evaluation_mode: str = "Exogènes observées (conditionnel)",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validation expanding-window : chaque fold est strictement postérieur à son échantillon d'estimation."""
    needed = [target] + exog_vars
    d = clean_model_data(df, needed)
    horizon, folds = int(horizon), int(folds)
    min_train = max(18, 3 * horizon)
    max_folds = max(1, (len(d) - min_train) // horizon)
    folds = min(folds, max_folds)
    if len(d) < min_train + horizon:
        raise ValueError("Historique insuffisant pour une validation rolling-origin.")
    rows = []
    per_fold = []
    first_train_end = len(d) - folds * horizon
    for f in range(folds):
        train_end = first_train_end + f * horizon
        train = d.iloc[:train_end].copy()
        test = d.iloc[train_end:train_end + horizon].copy()
        if len(test) < horizon:
            continue
        if exog_vars:
            if exog_evaluation_mode.startswith("Exogènes observées"):
                future_x = test.set_index("Date")[exog_vars]
            else:
                future_x = build_future_exog(train, exog_vars, horizon, scenario_method)
        else:
            future_x = None
        res = forecast_model(train, target, horizon, model_name, params, exog_vars, future_x)
        fold_df = pd.DataFrame({
            "Fold": f + 1, "Date": test["Date"].values, "Observé": test[target].values, "Prévu": res.forecast.values
        })
        rows.append(fold_df)
        _, seas, _ = infer_frequency(train["Date"])
        mt = metrics_table(fold_df["Observé"], fold_df["Prévu"], train[target].values, seasonality=seas)
        item = {"Fold": f + 1, "Train jusqu'à": train["Date"].max(), "Test début": test["Date"].min(), "Test fin": test["Date"].max()}
        item.update({r["Mesure"]: r["Valeur"] for _, r in mt.iterrows()})
        per_fold.append(item)
    if not rows:
        raise ValueError("Aucun fold valide n'a pu être estimé.")
    all_bt = pd.concat(rows, ignore_index=True)
    train_ref = d.iloc[:first_train_end][target].values
    _, seas, _ = infer_frequency(d["Date"])
    overall = metrics_table(all_bt["Observé"], all_bt["Prévu"], train_ref, seasonality=seas)
    return all_bt, overall, pd.DataFrame(per_fold)


# -----------------------------------------------------------------------------
# TESTS ÉCONOMÉTRIQUES
# -----------------------------------------------------------------------------
def stationarity_tests(series: pd.Series) -> pd.DataFrame:
    s = pd.Series(series).dropna().astype(float)
    rows = []
    if len(s) < 12:
        return pd.DataFrame([{"Test": "Information", "Statistique": np.nan, "p-value": np.nan, "Conclusion (5%)": "Échantillon trop court"}])
    try:
        stat, p, usedlag, nobs, *_ = adfuller(s, autolag="AIC")
        rows.append({"Test": "ADF", "Statistique": stat, "p-value": p, "Conclusion (5%)": "Stationnaire" if p < .05 else "Racine unitaire non rejetée", "Lags": usedlag})
    except Exception as e:
        rows.append({"Test": "ADF", "Erreur": str(e)})
    try:
        stat, p, lags, _ = kpss(s, regression="c", nlags="auto")
        rows.append({"Test": "KPSS", "Statistique": stat, "p-value": p, "Conclusion (5%)": "Stationnarité non rejetée" if p >= .05 else "Non-stationnaire", "Lags": lags})
    except Exception as e:
        rows.append({"Test": "KPSS", "Erreur": str(e)})
    try:
        stat, p, crit, usedlag, bp = zivot_andrews(s, regression="c", autolag="AIC")
        rows.append({"Test": "Zivot-Andrews", "Statistique": stat, "p-value": p, "Conclusion (5%)": "Racine unitaire rejetée avec rupture" if p < .05 else "Racine unitaire non rejetée", "Lags": usedlag, "Rupture (index)": bp})
    except Exception as e:
        rows.append({"Test": "Zivot-Andrews", "Erreur": str(e)})
    if HAS_ARCH:
        try:
            pp = PhillipsPerron(s)
            rows.append({"Test": "Phillips-Perron", "Statistique": pp.stat, "p-value": pp.pvalue, "Conclusion (5%)": "Stationnaire" if pp.pvalue < .05 else "Racine unitaire non rejetée", "Lags": pp.lags})
        except Exception as e:
            rows.append({"Test": "Phillips-Perron", "Erreur": str(e)})
        try:
            dg = DFGLS(s)
            rows.append({"Test": "DF-GLS", "Statistique": dg.stat, "p-value": dg.pvalue, "Conclusion (5%)": "Stationnaire" if dg.pvalue < .05 else "Racine unitaire non rejetée", "Lags": dg.lags})
        except Exception as e:
            rows.append({"Test": "DF-GLS", "Erreur": str(e)})
    return pd.DataFrame(rows)


def residual_tests(resid: pd.Series, lags: int = 12) -> pd.DataFrame:
    r = pd.Series(resid).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    rows = []
    if len(r) < 10:
        return pd.DataFrame([{"Test": "Information", "Conclusion": "Résidus insuffisants"}])
    use_lag = min(max(1, lags), max(1, len(r)//4))
    try:
        lb = acorr_ljungbox(r, lags=[use_lag], return_df=True).iloc[0]
        rows.append({"Test": f"Ljung-Box (lag {use_lag})", "Statistique": lb["lb_stat"], "p-value": lb["lb_pvalue"], "Conclusion": "Pas d'autocorrélation détectée" if lb["lb_pvalue"] >= .05 else "Autocorrélation résiduelle"})
    except Exception as e:
        rows.append({"Test": "Ljung-Box", "Erreur": str(e)})
    try:
        jb, p, skew, kurt = jarque_bera(r)
        rows.append({"Test": "Jarque-Bera", "Statistique": jb, "p-value": p, "Conclusion": "Normalité non rejetée" if p >= .05 else "Normalité rejetée"})
    except Exception as e:
        rows.append({"Test": "Jarque-Bera", "Erreur": str(e)})
    try:
        lm, p, f, fp = het_arch(r, nlags=use_lag)
        rows.append({"Test": f"ARCH-LM (lag {use_lag})", "Statistique": lm, "p-value": p, "Conclusion": "Pas d'effet ARCH détecté" if p >= .05 else "Hétéroscédasticité ARCH"})
    except Exception as e:
        rows.append({"Test": "ARCH-LM", "Erreur": str(e)})
    try:
        stat_c, p_c, _ = breaks_cusumolsresid(r, ddof=1)
        rows.append({"Test": "CUSUM stabilité", "Statistique": stat_c, "p-value": p_c, "Conclusion": "Stabilité non rejetée" if p_c >= .05 else "Rupture/instabilité possible"})
    except Exception as e:
        rows.append({"Test": "CUSUM stabilité", "Erreur": str(e)})
    return pd.DataFrame(rows)


def engle_granger_table(df: pd.DataFrame, target: str, exog_vars: List[str]) -> pd.DataFrame:
    rows = []
    for v in exog_vars:
        d = df[[target, v]].dropna()
        if len(d) < 20:
            continue
        try:
            stat, p, _ = coint(d[target], d[v])
            rows.append({"Paire": f"{target} ~ {v}", "Statistique": stat, "p-value": p, "Conclusion (5%)": "Cointégration" if p < .05 else "Cointégration non détectée"})
        except Exception as e:
            rows.append({"Paire": f"{target} ~ {v}", "Erreur": str(e)})
    return pd.DataFrame(rows)


def johansen_summary(df: pd.DataFrame, vars_: List[str], det_order: int = 0, k_ar_diff: int = 1) -> pd.DataFrame:
    d = df[vars_].dropna()
    j = coint_johansen(d, det_order=det_order, k_ar_diff=k_ar_diff)
    rows = []
    for r in range(len(vars_)):
        rows.append({
            "Rang H0": f"r ≤ {r}",
            "Trace": j.lr1[r], "CV 90%": j.cvt[r, 0], "CV 95%": j.cvt[r, 1], "CV 99%": j.cvt[r, 2],
            "Rejet 5%": bool(j.lr1[r] > j.cvt[r, 1]),
        })
    return pd.DataFrame(rows)


def pairwise_granger(df: pd.DataFrame, target: str, causes: List[str], maxlag: int) -> pd.DataFrame:
    rows = []
    for v in causes:
        d = df[[target, v]].dropna()
        if len(d) <= maxlag + 10:
            continue
        try:
            res = grangercausalitytests(d[[target, v]], maxlag=maxlag, verbose=False)
            pvals = [res[l][0]["ssr_ftest"][1] for l in range(1, maxlag + 1)]
            best = int(np.argmin(pvals) + 1)
            rows.append({"Cause candidate": v, "Cible": target, "p-value min": min(pvals), "Lag associé": best, "Conclusion (5%)": "Granger-causal" if min(pvals) < .05 else "Non détectée"})
        except Exception as e:
            rows.append({"Cause candidate": v, "Cible": target, "Erreur": str(e)})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# DIAGNOSTICS MODÈLE / IMPORTANCE
# -----------------------------------------------------------------------------
def feature_importance_table(result: ForecastResult) -> Optional[pd.DataFrame]:
    model = result.model
    features = result.metadata.get("features", [])
    if not features or model is None:
        return None
    raw = model
    if isinstance(model, Pipeline):
        raw = model.named_steps.get("model", model)
    if hasattr(raw, "feature_importances_"):
        imp = np.asarray(raw.feature_importances_)
    elif hasattr(raw, "coef_"):
        imp = np.abs(np.ravel(raw.coef_))
    else:
        return None
    if len(imp) != len(features):
        return None
    out = pd.DataFrame({"Variable/lag": features, "Importance": imp}).sort_values("Importance", ascending=False)
    return out.reset_index(drop=True)


def ardl_bounds_from_result(result: ForecastResult, case: int = 3) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    if result.model_name != "ARDL" or result.model is None:
        return None
    try:
        uecm = UECM.from_ardl(result.model.model).fit()
        b = uecm.bounds_test(case=case)
        summary = pd.DataFrame({"Statistique F": [b.stat], "p-value borne basse": [b.p_values.iloc[0] if hasattr(b, 'p_values') else b.pvalue.iloc[0]], "p-value borne haute": [b.p_values.iloc[1] if hasattr(b, 'p_values') else b.pvalue.iloc[1]]})
        return summary, b.crit_vals if hasattr(b, "crit_vals") else b.critical_values
    except Exception:
        try:
            uecm = UECM.from_ardl(result.model.model).fit()
            b = uecm.bounds_test(case=case)
            return pd.DataFrame({"Statistique F": [b.statistic], "p-value borne basse": [b.pvalue.iloc[0]], "p-value borne haute": [b.pvalue.iloc[1]]}), b.critical_values
        except Exception:
            return None


# -----------------------------------------------------------------------------
# UI HELPERS
# -----------------------------------------------------------------------------
def card_start(kicker: str, title: str, text: str = ""):
    st.markdown(f'<div class="rf-card"><div class="rf-kicker">{kicker}</div><h3>{title}</h3>{f"<p>{text}</p>" if text else ""}', unsafe_allow_html=True)


def card_end():
    st.markdown("</div>", unsafe_allow_html=True)


def model_description(name: str) -> str:
    desc = {
        "Naïf (dernière valeur)": "Benchmark minimal. Sert de référence pour vérifier qu'un modèle complexe apporte réellement un gain prédictif.",
        "SARIMAX / ARIMAX": "Dynamique ARIMA/SARIMA de la cible + plusieurs variables exogènes contemporaines. Pertinent si la dynamique temporelle et la saisonnalité sont structurées.",
        "ARDL": "Retards distribués de la cible et de plusieurs explicatives. Chaque exogène peut recevoir son propre ordre de lag.",
        "VAR": "Système multivarié endogène : toutes les variables sélectionnées sont expliquées par leurs propres retards et ceux des autres variables. Idéal pour IRF/FEVD/Granger.",
        "VECM": "Version à correction d'erreur pour variables I(1) cointégrées. Permet de séparer équilibre de long terme et ajustements de court terme.",
        "Theta / ETS": "Lissage exponentiel tendance/saisonnalité. Modèle volontairement univarié servant de benchmark robuste de niveau, tendance et saisonnalité.",
        "Prophet + régresseurs": "Tendance et saisonnalité flexibles avec variables exogènes additionnelles ; les valeurs futures des régresseurs doivent être fournies/scénarisées.",
        "NeuralProphet + régresseurs": "Autoregression neuronale et composantes de tendance/saisonnalité avec régresseurs futurs scénarisés. Plus flexible, mais plus exigeant en données.",
        "Régression dynamique": "Régression linéaire sur lags de la cible et lags propres de chaque variable exogène.",
        "Ridge dynamique": "Régression dynamique régularisée L2, utile lorsque les lags sont nombreux et corrélés.",
        "Elastic Net dynamique": "Combinaison L1/L2 pour régularisation et sélection partielle de variables/lags.",
        "Random Forest": "Forêt d'arbres sur caractéristiques retardées multivariées ; robuste aux non-linéarités et interactions.",
        "Gradient Boosting": "Boosting séquentiel sur lags multivariés, performant sur petits/moyens échantillons tabulaires.",
        "XGBoost": "Boosting régularisé sur lags multivariés avec sous-échantillonnage et pénalisation.",
        "MLP": "Réseau neuronal feed-forward sur lags multivariés standardisés ; à utiliser avec prudence sur petits échantillons.",
    }
    return desc.get(name, "")


MODEL_LIST = [
    "Naïf (dernière valeur)", "SARIMAX / ARIMAX", "ARDL", "Theta / ETS", "VAR", "VECM",
    "Prophet + régresseurs", "NeuralProphet + régresseurs",
    "Régression dynamique", "Ridge dynamique", "Elastic Net dynamique",
    "Random Forest", "Gradient Boosting", "XGBoost", "MLP",
]


# -----------------------------------------------------------------------------
# STATE
# -----------------------------------------------------------------------------
if "data_uploaded" not in st.session_state:
    st.session_state.data_uploaded = False
if "source_data" not in st.session_state:
    st.session_state.source_data = None
if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = None
if "forecast_context" not in st.session_state:
    st.session_state.forecast_context = None


# Profil / logo : conserver le comportement de l'application d'origine
LOGO_DATA_URI = "https://img.icons8.com/?size=100&id=3tC9EQumUAuq&format=png&color=000000"
ICON_DATA_URI = "https://img.icons8.com/?size=100&id=3tC9EQumUAuq&format=png&color=000000"
st.logo(
    image=LOGO_DATA_URI,
    link="https://ramanambonona.github.io/",
    icon_image=ICON_DATA_URI,
    size="large",
)

# En-tête sobre : aucun bandeau coloré en haut.
st.markdown(
    '<div style="margin:.15rem 0 1rem 0"><h1 style="margin:0">RAMA Forecast Lab</h1>'
    '<div style="color:#63756E;font-size:.98rem">Econometrics · Forecasting · Machine learning</div></div>',
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# SIDEBAR / NAVIGATION
# -----------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Données"

with st.sidebar:
    st.markdown('<div class="rf-side-title">RAMA Forecast</div><div class="rf-side-subtitle">Econometrics · Forecasting · Machine Learning</div>', unsafe_allow_html=True)
    st.divider()
    nav_items = [
        ("Données", "💾 Données"),
        ("Exploration", "🎯Exploration"),
        ("Prévisions", "⏱️Prévisions"),
        ("Tests économétriques", "🛠️   Tests"),
        ("IRF & dynamique", "⏳Impulse"),
    ]
    for nav_key, nav_label in nav_items:
        if st.button(nav_label, key=f"nav_{nav_key}", width="stretch", type="primary" if st.session_state.page == nav_key else "secondary"):
            if st.session_state.page != nav_key:
                st.session_state.page = nav_key
                st.rerun()
    page = st.session_state.page
    st.divider()
    if st.session_state.data_uploaded:
        df0 = st.session_state.source_data
        freq_code, season, freq_label = infer_frequency(df0["Date"])
        st.success("Données actives")
        st.caption(f"{len(df0)} observations · {len(df0.columns)-1} variables")
        st.caption(f"Fréquence : {freq_label}")
        st.caption(f"{df0['Date'].min():%d/%m/%Y} → {df0['Date'].max():%d/%m/%Y}")
    else:
        st.info("Importez des données pour activer les modules.")
    st.divider()
    st.markdown('<span class="rf-model-pill">ADF</span><span class="rf-model-pill">KPSS</span><span class="rf-model-pill">Johansen</span><span class="rf-model-pill">IRF</span><span class="rf-model-pill">FEVD</span><span class="rf-model-pill">ML</span>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE DONNÉES
# -----------------------------------------------------------------------------
if page == "Données":
    card_start("01 · Data", "Importer et structurer les séries", "L'application accepte Excel/CSV et tente de reconnaître automatiquement l'orientation des dates.")
    uploaded = st.file_uploader("Fichier de données", type=["xlsx", "xls", "csv"])
    if uploaded:
        try:
            raw = read_uploaded_file(uploaded)
            with st.expander("Aperçu brut", expanded=False):
                st.dataframe(raw.head(20), width="stretch")
            c1, c2 = st.columns([1, 2])
            with c1:
                orientation = st.selectbox("Orientation", ["Auto", "Dates en lignes / variables en colonnes", "Variables en lignes / dates en colonnes"])
            with c2:
                st.caption("L'orientation automatique compare la présence de dates dans la première ligne et la première colonne.")
            prepared = transpose_if_needed(raw, orientation)
            df = normalize_data(prepared)
            if len(df.columns) < 2:
                st.error("Aucune variable numérique exploitable n'a été détectée.")
            else:
                freq_code, season, freq_label = infer_frequency(df["Date"])
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Observations", len(df))
                m2.metric("Variables", len(df.columns)-1)
                m3.metric("Fréquence", freq_label)
                m4.metric("Valeurs manquantes", int(df.drop(columns="Date").isna().sum().sum()))
                st.dataframe(df.head(20), width="stretch", height=360)
                if st.button("Valider les données", type="primary", width="stretch"):
                    st.session_state.source_data = df
                    st.session_state.data_uploaded = True
                    st.session_state.forecast_result = None
                    st.session_state.forecast_context = None
                    st.success("Données validées. Les modules d'analyse sont maintenant disponibles.")
                    st.balloons()
        except Exception as e:
            st.error(f"Impossible de traiter le fichier : {e}")
    card_end()

    if st.session_state.data_uploaded:
        df = st.session_state.source_data
        card_start("Dataset actif", "Contrôle de qualité")
        miss = df.drop(columns="Date").isna().sum().rename("Manquantes").to_frame()
        miss["%"] = 100 * miss["Manquantes"] / len(df)
        stats_df = df.drop(columns="Date").describe().T[["count", "mean", "std", "min", "max"]]
        t1, t2 = st.tabs(["Valeurs manquantes", "Statistiques descriptives"])
        with t1:
            st.dataframe(miss, width="stretch")
        with t2:
            st.dataframe(stats_df, width="stretch")
        card_end()


# -----------------------------------------------------------------------------
# GUARD
# -----------------------------------------------------------------------------
if page != "Données" and not st.session_state.data_uploaded:
    st.warning("Importez et validez d'abord un fichier dans l'onglet **Données**.")
    st.stop()


# -----------------------------------------------------------------------------
# PAGE EXPLORATION
# -----------------------------------------------------------------------------
if page == "Exploration":
    df = st.session_state.source_data.copy()
    vars_all = list(df.columns.drop("Date"))
    card_start("02 · Explore", "Exploration multivariée", "Visualiser les trajectoires, corrélations, distributions et composantes temporelles avant de modéliser.")
    selected = st.multiselect("Variables", vars_all, default=vars_all[:min(4, len(vars_all))])
    if selected:
        fig = go.Figure()
        for i, v in enumerate(selected):
            fig.add_trace(go.Scatter(x=df["Date"], y=df[v], name=v, mode="lines", line=dict(color=PLOT_COLORS[i % len(PLOT_COLORS)], width=2.2)))
        st.plotly_chart(apply_plot_style(fig, "Évolution des séries", 540), width="stretch", config=PLOT_CONFIG)

        tab1, tab2, tab3 = st.tabs(["Corrélations", "Distributions", "Décomposition"])
        with tab1:
            corr = df[selected].corr()
            heat = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale=[[0, "#E8F4ED"], [.5, "#95C14E"], [1, "#0A463B"]], zmin=-1, zmax=1, text=np.round(corr.values, 2), texttemplate="%{text}"))
            st.plotly_chart(apply_plot_style(heat, "Matrice de corrélation", 520), width="stretch", config=PLOT_CONFIG)
        with tab2:
            v = st.selectbox("Variable", selected, key="dist_var")
            hist = go.Figure(go.Histogram(x=df[v].dropna(), nbinsx=25, marker_color=RF["green"], opacity=.82))
            st.plotly_chart(apply_plot_style(hist, f"Distribution de {v}", 440), width="stretch", config=PLOT_CONFIG)
        with tab3:
            v = st.selectbox("Série à décomposer", selected, key="decomp_var")
            _, season, freq_label = infer_frequency(df["Date"])
            s = df.set_index("Date")[v].dropna()
            if season > 1 and len(s) >= 2 * season:
                dec = seasonal_decompose(s, period=season, model="additive", extrapolate_trend="freq")
                figd = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=["Observé", "Tendance", "Saisonnalité", "Résidu"], vertical_spacing=.06)
                for r, vals, col in [(1, dec.observed, RF["deep"]), (2, dec.trend, RF["green"]), (3, dec.seasonal, RF["lime"]), (4, dec.resid, "#6F7D76")]:
                    figd.add_trace(go.Scatter(x=s.index, y=vals, mode="lines", line=dict(color=col, width=1.8), showlegend=False), row=r, col=1)
                st.plotly_chart(apply_plot_style(figd, f"Décomposition additive · {v} · {freq_label}", 760), width="stretch", config=PLOT_CONFIG)
            else:
                st.info("La série est trop courte ou la fréquence ne permet pas une décomposition saisonnière fiable.")
    card_end()


# -----------------------------------------------------------------------------
# PAGE PRÉVISIONS
# -----------------------------------------------------------------------------
if page == "Prévisions":
    df = st.session_state.source_data.copy()
    vars_all = list(df.columns.drop("Date"))
    freq_code, default_season, freq_label = infer_frequency(df["Date"])

    card_start("03 · Forecast", "Prévision multivariée configurable", "Choisissez la cible, les variables explicatives, leurs retards et les paramètres propres à chaque famille de modèles.")
    top1, top2, top3 = st.columns([1.5, 1.5, 1])
    with top1:
        target = st.selectbox("Variable à prévoir", vars_all)
    with top2:
        model_name = st.selectbox("Modèle", MODEL_LIST)
    with top3:
        periods = st.number_input("Horizon", min_value=1, max_value=120, value=min(12, max(1, default_season)), step=1)
    st.markdown(f'<div class="rf-note"><strong>{model_name}</strong><br>{model_description(model_name)}</div>', unsafe_allow_html=True)

    params: Dict[str, Any] = {}
    exog_vars: List[str] = []
    system_model = model_name in {"VAR", "VECM"}
    exog_capable = model_name in {"SARIMAX / ARIMAX", "ARDL", "Prophet + régresseurs", "NeuralProphet + régresseurs", "Régression dynamique", "Ridge dynamique", "Elastic Net dynamique", "Random Forest", "Gradient Boosting", "XGBoost", "MLP"}

    if system_model:
        candidates = [v for v in vars_all if v != target]
        default_sys = [target] + candidates[:min(2, len(candidates))]
        system_vars = st.multiselect("Variables du système", vars_all, default=default_sys)
        if target not in system_vars:
            system_vars = [target] + system_vars
        params["system_vars"] = list(dict.fromkeys(system_vars))
        exog_vars = [v for v in params["system_vars"] if v != target]
    elif exog_capable:
        exog_vars = st.multiselect("Variables exogènes / explicatives", [v for v in vars_all if v != target], help="Vous pouvez sélectionner une ou plusieurs variables.")

    with st.expander("Paramètres du modèle", expanded=True):
        if model_name == "SARIMAX / ARIMAX":
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Composante non saisonnière**")
                p = st.number_input("p · AR", 0, 12, 1)
                d = st.number_input("d · différenciation", 0, 2, 1)
                q = st.number_input("q · MA", 0, 12, 1)
                params["order"] = (p, d, q)
                params["trend"] = st.selectbox("Tendance déterministe", ["n", "c", "t", "ct"], index=1)
            with c2:
                st.markdown("**Composante saisonnière**")
                P = st.number_input("P · AR saisonnier", 0, 4, 0)
                D = st.number_input("D · différenciation saisonnière", 0, 2, 0)
                Q = st.number_input("Q · MA saisonnier", 0, 4, 0)
                s = st.number_input("s · période saisonnière", 0, 60, int(default_season if default_season > 1 else 0))
                params["seasonal_order"] = (P, D, Q, s) if s > 1 else (0, 0, 0, 0)
                params["enforce_stationarity"] = st.checkbox("Imposer la stationnarité", value=False)
                params["enforce_invertibility"] = st.checkbox("Imposer l'inversibilité", value=False)
            params["alpha"] = 1 - st.slider("Niveau de confiance", .80, .99, .95, .01)

        elif model_name == "ARDL":
            params["target_lags"] = st.slider("Lag maximal de la cible", 1, min(24, max(1, len(df)//5)), min(default_season, 3) if default_season > 1 else 1)
            params["trend"] = st.selectbox("Tendance", ["n", "c", "ct", "ctt"], index=1)
            params["cov_type"] = st.selectbox("Covariance", ["nonrobust", "HC0", "HC1", "HC2", "HC3"], index=0)
            params["causal"] = st.checkbox("ARDL causal (exclure le lag 0 des exogènes)", value=False)
            orders = {}
            if exog_vars:
                st.markdown("**Lag maximal propre à chaque exogène**")
                cols = st.columns(min(3, len(exog_vars)))
                for i, v in enumerate(exog_vars):
                    with cols[i % len(cols)]:
                        orders[v] = st.slider(f"{v}", 0 if not params["causal"] else 1, min(24, max(1, len(df)//5)), min(2, default_season), key=f"ardl_{v}")
            params["exog_order"] = orders

        elif model_name == "VAR":
            c1, c2 = st.columns(2)
            with c1:
                params["auto_ic"] = st.selectbox("Sélection du lag", ["Manuel", "AIC", "BIC", "HQIC", "FPE"])
                params["lag_order"] = st.slider("Lag manuel", 1, min(18, max(1, len(df)//6)), 1)
            with c2:
                params["maxlags"] = st.slider("Lag max pour sélection automatique", 2, min(24, max(2, len(df)//5)), min(12, max(2, len(df)//8)))
                params["trend"] = st.selectbox("Déterministes", ["n", "c", "ct", "ctt"], index=1)

        elif model_name == "VECM":
            nsys = max(2, len(params.get("system_vars", [])))
            c1, c2, c3 = st.columns(3)
            with c1:
                params["k_ar_diff"] = st.slider("Retards des différences", 1, min(12, max(1, len(df)//8)), 1)
            with c2:
                params["coint_rank"] = st.slider("Rang de cointégration", 1, max(1, nsys-1), 1)
            with c3:
                params["deterministic"] = st.selectbox("Déterministes", ["n", "co", "ci", "lo", "li"], index=1)
            params["seasons"] = st.number_input("Nombre de saisons (0 = aucune)", 0, 60, int(default_season if default_season in [4, 12] else 0))

        elif model_name == "Theta / ETS":
            c1, c2, c3 = st.columns(3)
            with c1:
                params["ets_trend"] = st.selectbox("Tendance ETS", [None, "add", "mul"], index=1)
            with c2:
                params["ets_seasonal"] = st.selectbox("Saisonnalité ETS", [None, "add", "mul"], index=1 if default_season > 1 else 0)
            with c3:
                params["ets_period"] = st.number_input("Période saisonnière", 0, 60, int(default_season if default_season > 1 else 0))
            params["damped_trend"] = st.checkbox("Tendance amortie", value=False)

        elif model_name == "Prophet + régresseurs":
            c1, c2 = st.columns(2)
            with c1:
                params["changepoint_prior_scale"] = st.slider("Flexibilité de tendance", .001, .50, .05, .001)
                params["seasonality_prior_scale"] = st.slider("Flexibilité saisonnière", .1, 30.0, 10.0, .1)
            with c2:
                params["seasonality_mode"] = st.selectbox("Mode saisonnier", ["additive", "multiplicative"])
                params["regressor_mode"] = st.selectbox("Effet des régresseurs", ["additive", "multiplicative"])
            params["alpha"] = 1 - st.slider("Niveau de confiance", .80, .99, .95, .01, key="prophet_conf")

        elif model_name == "NeuralProphet + régresseurs":
            c1, c2, c3 = st.columns(3)
            with c1: params["np_n_lags"] = st.slider("Lags autoregressifs", 1, min(36, max(1, len(df)//5)), min(12, max(1, len(df)//8)))
            with c2: params["np_epochs"] = st.slider("Époques", 50, 800, 150, 25)
            with c3: params["np_seasonality_mode"] = st.selectbox("Mode saisonnier", ["additive", "multiplicative"], key="np_season")
            params["np_regressor_mode"] = st.selectbox("Mode des régresseurs", ["additive", "multiplicative"], key="np_reg_mode")
            st.caption("NeuralProphet utilise ici les exogènes comme régresseurs futurs : leur scénario doit couvrir tout l'horizon.")

        elif model_name in {"Régression dynamique", "Ridge dynamique", "Elastic Net dynamique", "Random Forest", "Gradient Boosting", "XGBoost", "MLP"}:
            max_lag = min(24, max(2, len(df)//5))
            params["target_lags_list"] = st.multiselect("Lags de la cible", list(range(1, max_lag+1)), default=list(range(1, min(3, max_lag)+1)))
            exog_lags = {}
            if exog_vars:
                st.markdown("**Lags propres aux variables exogènes (0 = effet contemporain/scénarisé)**")
                for v in exog_vars:
                    exog_lags[v] = st.multiselect(f"Lags · {v}", list(range(0, max_lag+1)), default=[0, 1] if max_lag >= 1 else [0], key=f"ml_lags_{v}")
            params["exog_lags"] = exog_lags
            if model_name == "Ridge dynamique":
                params["alpha_ridge"] = st.number_input("Pénalisation Ridge α", min_value=.0001, max_value=1000.0, value=1.0, format="%.4f")
            elif model_name == "Elastic Net dynamique":
                c1, c2 = st.columns(2)
                with c1: params["alpha_en"] = st.number_input("α", min_value=.0001, max_value=10.0, value=.05, format="%.4f")
                with c2: params["l1_ratio"] = st.slider("Part L1", 0.0, 1.0, .5, .05)
            elif model_name == "Random Forest":
                c1, c2, c3 = st.columns(3)
                with c1: params["n_estimators"] = st.slider("Arbres", 100, 1200, 400, 50)
                with c2: params["max_depth"] = st.select_slider("Profondeur", options=[None, 3, 5, 7, 10, 15, 20], value=None)
                with c3: params["min_samples_leaf"] = st.slider("Feuilles min.", 1, 10, 2)
            elif model_name == "Gradient Boosting":
                c1, c2, c3 = st.columns(3)
                with c1: params["n_estimators"] = st.slider("Estimateurs", 50, 800, 250, 25)
                with c2: params["learning_rate"] = st.slider("Learning rate", .005, .30, .05, .005)
                with c3: params["max_depth_gb"] = st.slider("Profondeur", 1, 8, 3)
                params["loss"] = st.selectbox("Loss", ["huber", "squared_error", "absolute_error"])
            elif model_name == "XGBoost":
                c1, c2, c3 = st.columns(3)
                with c1: params["n_estimators"] = st.slider("Arbres", 100, 1500, 400, 50, key="xgb_n")
                with c2: params["max_depth_xgb"] = st.slider("Profondeur", 2, 12, 4)
                with c3: params["learning_rate"] = st.slider("Learning rate", .005, .30, .04, .005, key="xgb_lr")
                c4, c5, c6, c7 = st.columns(4)
                with c4: params["subsample"] = st.slider("Subsample", .50, 1.0, .85, .05)
                with c5: params["colsample_bytree"] = st.slider("Colonnes", .50, 1.0, .85, .05)
                with c6: params["reg_alpha"] = st.number_input("L1", 0.0, 10.0, 0.0)
                with c7: params["reg_lambda"] = st.number_input("L2", 0.0, 20.0, 1.0)
            elif model_name == "MLP":
                c1, c2, c3 = st.columns(3)
                with c1:
                    layers_txt = st.text_input("Couches cachées", "64,32")
                    try: params["hidden_layers"] = [int(x.strip()) for x in layers_txt.split(",") if x.strip()]
                    except Exception: params["hidden_layers"] = [64, 32]
                with c2: params["mlp_alpha"] = st.number_input("Régularisation", .00001, 1.0, .001, format="%.5f")
                with c3: params["mlp_lr"] = st.number_input("Learning rate", .00001, .1, .001, format="%.5f")
                params["max_iter"] = st.slider("Itérations max.", 300, 3000, 1200, 100)

    # Scénario exogène
    future_exog = None
    scenario_method = "Dernière valeur"
    if exog_vars and not system_model:
        with st.expander("Scénario futur des variables exogènes", expanded=True):
            c1, c2 = st.columns([1, 1])
            with c1:
                scenario_method = st.selectbox("Méthode de projection initiale", ["Dernière valeur", "Moyenne récente", "Tendance linéaire", "Croissance moyenne"])
            with c2:
                rolling_window = st.slider("Fenêtre de calcul", 3, min(36, max(3, len(df)//2)), min(12, max(3, len(df)//2)))
            auto_future = build_future_exog(df, exog_vars, int(periods), scenario_method, rolling_window)
            editable = auto_future.reset_index()
            future_edit = st.data_editor(editable, width="stretch", hide_index=True, num_rows="fixed", key=f"future_{target}_{model_name}_{periods}")
            try:
                future_exog = validate_future_exog(future_edit, exog_vars, int(periods))
            except Exception as e:
                st.error(str(e))

    with st.expander("Validation prédictive", expanded=False):
        max_bt_h = max(1, min(24, max(1, len(df)//8)))
        b1, b2, b3 = st.columns(3)
        with b1:
            bt_horizon = st.slider("Horizon par fold", 1, max_bt_h, min(default_season if default_season > 1 else 3, max_bt_h))
        with b2:
            bt_folds = st.slider("Nombre de folds", 1, 5, min(3, max(1, (len(df)-18)//max(1, bt_horizon))))
        with b3:
            bt_exog_mode = st.selectbox("Exogènes en backtest", ["Exogènes observées (conditionnel)", "Exogènes projetées (pseudo-réel)"])
        st.caption("La fenêtre d'estimation s'agrandit dans le temps ; aucune observation future de la cible n'entre dans l'apprentissage.")

    c_run, c_bt = st.columns([1, 1])
    with c_run:
        run_forecast = st.button("Lancer la prévision", type="primary", width="stretch")
    with c_bt:
        run_backtest = st.button("Backtest rolling-origin", width="stretch")

    if run_forecast:
        try:
            with st.spinner("Estimation du modèle et génération du scénario..."):
                result = forecast_model(df, target, int(periods), model_name, params, exog_vars, future_exog)
                st.session_state.forecast_result = result
                st.session_state.forecast_context = {"target": target, "model_name": model_name, "params": params, "exog_vars": exog_vars, "future_exog": future_exog, "periods": int(periods), "scenario_method": scenario_method}
            st.success("Prévision calculée.")
        except Exception as e:
            st.error(f"Échec de l'estimation : {e}")

    if run_backtest:
        try:
            with st.spinner("Validation rolling-origin en cours..."):
                bt, mt, fold_mt = rolling_origin_backtest(
                    df, target, model_name, params, exog_vars, bt_horizon, bt_folds, scenario_method, bt_exog_mode
                )
            st.session_state.backtest = bt
            st.session_state.backtest_metrics = mt
            st.session_state.backtest_fold_metrics = fold_mt
        except Exception as e:
            st.error(f"Backtest impossible : {e}")

    card_end()

    result: ForecastResult = st.session_state.forecast_result
    ctx = st.session_state.forecast_context
    if result is not None and ctx is not None:
        target_r = ctx["target"]
        hist = df[["Date", target_r]].dropna()
        card_start("Résultats", f"Prévision · {target_r}", f"Modèle : {result.model_name}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["Date"], y=hist[target_r], name="Historique", mode="lines", line=dict(color=RF["deep"], width=2.1)))
        xfc = [hist["Date"].iloc[-1]] + list(result.forecast.index)
        yfc = [hist[target_r].iloc[-1]] + list(result.forecast.values)
        if result.lower is not None and result.upper is not None:
            fig.add_trace(go.Scatter(x=list(result.upper.index)+list(result.lower.index[::-1]), y=list(result.upper.values)+list(result.lower.values[::-1]), fill="toself", fillcolor="rgba(149,193,78,.18)", line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", name="Intervalle"))
        fig.add_trace(go.Scatter(x=xfc, y=yfc, name="Prévision", mode="lines+markers", line=dict(color=RF["bright"], width=2.8), marker=dict(size=7)))
        fig.add_vline(x=hist["Date"].iloc[-1], line_dash="dot", line_color=RF["muted"])
        st.plotly_chart(apply_plot_style(fig, f"{target_r} · historique et prévision", 560), width="stretch", config=PLOT_CONFIG)

        if result.metadata:
            md = result.metadata
            cols = st.columns(min(4, max(1, len([k for k in ["AIC", "BIC", "HQIC", "lag_order", "n_features"] if k in md]))))
            j = 0
            for k in ["AIC", "BIC", "HQIC", "lag_order", "n_features"]:
                if k in md:
                    val = md[k]
                    cols[j % len(cols)].metric(k, f"{val:.3f}" if isinstance(val, (int, float, np.floating)) and np.isfinite(val) else str(val))
                    j += 1

        imp = feature_importance_table(result)
        if imp is not None and not imp.empty:
            with st.expander("Importance des variables et lags", expanded=False):
                topimp = imp.head(25).sort_values("Importance")
                fimp = go.Figure(go.Bar(x=topimp["Importance"], y=topimp["Variable/lag"], orientation="h", marker_color=RF["green"]))
                st.plotly_chart(apply_plot_style(fimp, "Importance des caractéristiques", max(420, 24*len(topimp))), width="stretch", config=PLOT_CONFIG)

        out = pd.DataFrame({"Date": result.forecast.index, "Prévision": result.forecast.values})
        if result.lower is not None: out["Borne_inf"] = result.lower.values
        if result.upper is not None: out["Borne_sup"] = result.upper.values
        with io.BytesIO() as buffer:
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                out.to_excel(writer, index=False, sheet_name="Forecast")
                if ctx.get("future_exog") is not None:
                    ctx["future_exog"].reset_index().to_excel(writer, index=False, sheet_name="Scenario_exogene")
                pd.DataFrame({"clé": list(result.metadata.keys()), "valeur": [str(v) for v in result.metadata.values()]}).to_excel(writer, index=False, sheet_name="Metadata")
            st.download_button("Exporter prévision + scénario (Excel)", buffer.getvalue(), file_name=f"rama_forecast_{target_r}_{result.model_name.replace(' ','_')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        card_end()

    if "backtest" in st.session_state:
        card_start("Validation temporelle", "Backtest chronologique", "Les dernières observations sont tenues hors estimation. Pour les modèles avec exogènes, ce backtest est conditionnel aux valeurs observées des régresseurs.")
        bt = st.session_state.backtest
        mt = st.session_state.backtest_metrics
        figbt = go.Figure()
        figbt.add_trace(go.Scatter(x=bt["Date"], y=bt["Observé"], name="Observé", mode="lines+markers", line=dict(color=RF["deep"], width=2.4)))
        figbt.add_trace(go.Scatter(x=bt["Date"], y=bt["Prévu"], name="Prévu", mode="lines+markers", line=dict(color=RF["bright"], width=2.4)))
        st.plotly_chart(apply_plot_style(figbt, "Backtest rolling-origin · observations hors estimation", 470), width="stretch", config=PLOT_CONFIG)
        cols = st.columns(len(mt))
        for i, row in mt.reset_index(drop=True).iterrows():
            val = row["Valeur"]
            cols[i].metric(row["Mesure"], f"{val:.4f}" if pd.notna(val) else "N/A")
        if "backtest_fold_metrics" in st.session_state:
            with st.expander("Résultats par fold"):
                st.dataframe(st.session_state.backtest_fold_metrics, width="stretch")
        card_end()


# -----------------------------------------------------------------------------
# PAGE TESTS ÉCONOMÉTRIQUES
# -----------------------------------------------------------------------------
if page == "Tests économétriques":
    df = st.session_state.source_data.copy()
    vars_all = list(df.columns.drop("Date"))
    card_start("04 · Diagnostics", "Tests économétriques avancés", "Stationnarité, ruptures, cointégration, causalité de Granger et diagnostics résiduels.")
    tabs = st.tabs(["Stationnarité", "Cointégration", "Granger", "Résidus", "Bounds ARDL"])

    with tabs[0]:
        v = st.selectbox("Variable", vars_all, key="stat_var")
        st.dataframe(stationarity_tests(df[v]), width="stretch")
        if not HAS_ARCH:
            st.caption("Phillips-Perron et DF-GLS deviennent disponibles lorsque le paquet `arch` est installé.")

    with tabs[1]:
        target_c = st.selectbox("Variable de référence", vars_all, key="coint_target")
        ex_c = st.multiselect("Autres variables", [v for v in vars_all if v != target_c], default=[v for v in vars_all if v != target_c][:min(3, len(vars_all)-1)], key="coint_exog")
        if ex_c:
            st.markdown("**Engle-Granger bilatéral**")
            st.dataframe(engle_granger_table(df, target_c, ex_c), width="stretch")
            if len(ex_c) >= 1:
                st.markdown("**Johansen multivarié**")
                joh_vars = [target_c] + ex_c
                c1, c2 = st.columns(2)
                with c1: det = st.selectbox("Terme déterministe Johansen", [-1, 0, 1], index=1, help="-1 aucun, 0 constante, 1 tendance linéaire")
                with c2: kdiff = st.slider("Retards en différences", 1, min(12, max(1, len(df)//10)), 1)
                try:
                    st.dataframe(johansen_summary(df, joh_vars, det, kdiff), width="stretch")
                except Exception as e:
                    st.warning(f"Johansen non estimable avec cette configuration : {e}")

    with tabs[2]:
        target_g = st.selectbox("Cible", vars_all, key="gr_target")
        causes = st.multiselect("Causes candidates", [v for v in vars_all if v != target_g], default=[v for v in vars_all if v != target_g][:min(3, len(vars_all)-1)], key="gr_causes")
        maxlag = st.slider("Lag maximal", 1, min(12, max(1, len(df)//8)), 4)
        if causes:
            st.dataframe(pairwise_granger(df, target_g, causes, maxlag), width="stretch")
            st.caption("Une causalité de Granger signifie un contenu prédictif marginal conditionnel aux retards, et non une causalité structurelle au sens économique.")

    with tabs[3]:
        result = st.session_state.forecast_result
        if result is None or result.residuals is None:
            st.info("Lancez d'abord une prévision afin de diagnostiquer ses résidus.")
        else:
            lagr = st.slider("Lag diagnostic", 1, min(24, max(1, len(result.residuals)//4)), min(12, max(1, len(result.residuals)//5)))
            st.dataframe(residual_tests(result.residuals, lagr), width="stretch")
            r = pd.Series(result.residuals).dropna()
            f = go.Figure()
            f.add_trace(go.Scatter(x=r.index, y=r.values, mode="lines", name="Résidus", line=dict(color=RF["deep"], width=1.8)))
            f.add_hline(y=0, line_dash="dot", line_color=RF["muted"])
            st.plotly_chart(apply_plot_style(f, f"Résidus · {result.model_name}", 400), width="stretch", config=PLOT_CONFIG)

    with tabs[4]:
        result = st.session_state.forecast_result
        if result is None or result.model_name != "ARDL":
            st.info("Estimez un modèle ARDL avec au moins une variable exogène pour lancer le Bounds test de Pesaran-Shin-Smith.")
        else:
            case = st.selectbox("Cas PSS", [1, 2, 3, 4, 5], index=2)
            b = ardl_bounds_from_result(result, int(case))
            if b is None:
                st.warning("Le modèle ARDL courant ne satisfait pas les contraintes nécessaires à sa conversion en UECM (lags contigus/positifs notamment).")
            else:
                sm, cv = b
                st.dataframe(sm, width="stretch")
                st.dataframe(cv, width="stretch")
    card_end()


# -----------------------------------------------------------------------------
# PAGE IRF & DYNAMIQUE
# -----------------------------------------------------------------------------
if page == "IRF & dynamique":
    df = st.session_state.source_data.copy()
    vars_all = list(df.columns.drop("Date"))
    card_start("05 · Dynamic analysis", "Impulsions, IRF, FEVD et stabilité", "Ces outils sont estimés dans un système VAR/VECM, où les interactions dynamiques entre variables sont explicitement modélisées.")

    sys_vars = st.multiselect("Variables du système", vars_all, default=vars_all[:min(3, len(vars_all))], key="dyn_vars")
    if len(sys_vars) >= 2:
        c1, c2, c3 = st.columns(3)
        with c1:
            family = st.selectbox("Famille", ["VAR", "VECM"], key="dyn_family")
        with c2:
            horizon = st.slider("Horizon IRF", 4, 48, 12)
        with c3:
            orth = st.checkbox("IRF orthogonalisée (Cholesky)", value=True)

        if family == "VAR":
            c4, c5 = st.columns(2)
            with c4:
                lag_dyn = st.slider("Lag VAR", 1, min(12, max(1, len(df)//8)), 1, key="dyn_lag")
            with c5:
                trend_dyn = st.selectbox("Déterministes", ["n", "c", "ct", "ctt"], index=1, key="dyn_trend")
            if st.button("Estimer le système dynamique", type="primary"):
                try:
                    d = clean_model_data(df, sys_vars).set_index("Date")[sys_vars]
                    fit = VAR(d).fit(lag_dyn, trend=trend_dyn)
                    st.session_state.dynamic_fit = ("VAR", fit, sys_vars)
                except Exception as e:
                    st.error(str(e))
        else:
            c4, c5, c6 = st.columns(3)
            with c4: kdiff = st.slider("Retards différences", 1, min(10, max(1, len(df)//10)), 1, key="vec_kdiff")
            with c5: rank = st.slider("Rang de cointégration", 1, max(1, len(sys_vars)-1), 1, key="vec_rank")
            with c6: det = st.selectbox("Déterministes", ["n", "co", "ci", "lo", "li"], index=1, key="vec_det")
            if st.button("Estimer le système dynamique", type="primary"):
                try:
                    d = clean_model_data(df, sys_vars).set_index("Date")[sys_vars]
                    fit = VECM(d, k_ar_diff=kdiff, coint_rank=rank, deterministic=det).fit()
                    st.session_state.dynamic_fit = ("VECM", fit, sys_vars)
                except Exception as e:
                    st.error(str(e))

        if "dynamic_fit" in st.session_state:
            fam, fit, fitted_vars = st.session_state.dynamic_fit
            if fitted_vars != sys_vars:
                st.info("Le système affiché correspond à la dernière estimation. Relancez l'estimation après modification des variables.")
            else:
                shock = st.selectbox("Choc sur", sys_vars, key="shock_var")
                response = st.selectbox("Réponse de", sys_vars, key="response_var")
                tabs = st.tabs(["IRF", "IRF cumulée", "FEVD", "Causalité & diagnostics", "Racines / stabilité"])
                with tabs[0]:
                    irf = fit.irf(horizon)
                    arr = irf.orth_irfs if (orth and fam == "VAR") else irf.irfs
                    i, j = sys_vars.index(response), sys_vars.index(shock)
                    vals = arr[:, i, j]
                    fig = go.Figure(go.Scatter(x=np.arange(len(vals)), y=vals, mode="lines+markers", line=dict(color=RF["bright"], width=2.8), marker=dict(size=6), name=f"{shock} → {response}"))
                    fig.add_hline(y=0, line_dash="dot", line_color=RF["muted"])
                    st.plotly_chart(apply_plot_style(fig, f"IRF · choc {shock} → réponse {response}", 500), width="stretch", config=PLOT_CONFIG)
                    st.caption("Avec Cholesky, l'ordre des variables du système affecte l'identification contemporaine des chocs.")
                with tabs[1]:
                    irf = fit.irf(horizon)
                    arr = irf.orth_irfs if (orth and fam == "VAR") else irf.irfs
                    i, j = sys_vars.index(response), sys_vars.index(shock)
                    vals = np.cumsum(arr[:, i, j])
                    fig = go.Figure(go.Scatter(x=np.arange(len(vals)), y=vals, mode="lines+markers", line=dict(color=RF["deep"], width=2.8), marker=dict(size=6)))
                    fig.add_hline(y=0, line_dash="dot", line_color=RF["muted"])
                    st.plotly_chart(apply_plot_style(fig, f"IRF cumulée · {shock} → {response}", 500), width="stretch", config=PLOT_CONFIG)
                with tabs[2]:
                    if fam == "VAR":
                        try:
                            fevd = fit.fevd(horizon)
                            ridx = sys_vars.index(response)
                            # statsmodels: decomp[variable, horizon, shock]
                            mat = fevd.decomp[ridx]
                            fevd_df = pd.DataFrame(mat, columns=sys_vars, index=np.arange(1, mat.shape[0]+1))
                            fig = go.Figure()
                            for k, v in enumerate(sys_vars):
                                fig.add_trace(go.Bar(x=fevd_df.index, y=fevd_df[v], name=v, marker_color=PLOT_COLORS[k % len(PLOT_COLORS)]))
                            fig.update_layout(barmode="stack")
                            st.plotly_chart(apply_plot_style(fig, f"FEVD de {response}", 520), width="stretch", config=PLOT_CONFIG)
                        except Exception as e:
                            st.warning(f"FEVD non disponible : {e}")
                    else:
                        st.info("La FEVD est présentée ici pour le VAR. Pour VECM, privilégiez les IRF et la structure de cointégration.")
                with tabs[3]:
                    if fam == "VAR":
                        rows = []
                        for cause in sys_vars:
                            if cause == response: continue
                            try:
                                t = fit.test_causality(response, [cause], kind="f")
                                rows.append({"Cause": cause, "Réponse": response, "Statistique": t.test_statistic, "p-value": t.pvalue, "Conclusion": "Granger-causal" if t.pvalue < .05 else "Non détectée"})
                            except Exception as e:
                                rows.append({"Cause": cause, "Erreur": str(e)})
                        st.dataframe(pd.DataFrame(rows), width="stretch")
                        d1, d2 = st.columns(2)
                        with d1:
                            try:
                                w = fit.test_whiteness(nlags=max(fit.k_ar+1, min(12, fit.k_ar+6)), adjusted=True)
                                st.metric("Whiteness p-value", f"{w.pvalue:.4f}")
                                st.caption("p ≥ 0,05 : absence d'autocorrélation résiduelle non rejetée.")
                            except Exception as e: st.caption(str(e))
                        with d2:
                            try:
                                n = fit.test_normality()
                                st.metric("Normalité p-value", f"{n.pvalue:.4f}")
                                st.caption("p ≥ 0,05 : normalité multivariée non rejetée.")
                            except Exception as e: st.caption(str(e))
                    else:
                        st.info("Les tests de causalité intégrés affichés ici sont ceux du VAR ; le VECM doit être interprété conjointement avec les coefficients de correction d'erreur et la cointégration.")
                with tabs[4]:
                    if fam == "VAR":
                        roots = fit.roots
                        root_df = pd.DataFrame({"Racine": roots, "Module": np.abs(roots)})
                        st.metric("Système stable", "Oui" if fit.is_stable() else "Non")
                        st.dataframe(root_df, width="stretch")
                        st.caption("Dans la convention de statsmodels, la stabilité du VAR est évaluée via les racines du polynôme caractéristique.")
                    else:
                        st.info("Dans un VECM, la présence de racines unitaires est inhérente à la représentation cointégrée ; la stabilité se lit différemment d'un VAR stationnaire.")
    else:
        st.info("Sélectionnez au moins deux variables.")
    card_end()


# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="custom-footer">
  <p class="footnote">Ramanambonona Ambinintsoa, Ph.D</p>
  <div class="social">
    <a href="mailto:ambinintsoa.uat.ead2@gmail.com" aria-label="Mail">
      <img src="https://img.icons8.com/?size=100&id=86875&format=png&color=000000" alt="Mail">
    </a>
    <a href="https://github.com/ramanambonona" target="_blank" rel="noopener" aria-label="GitHub">
      <img src="https://img.icons8.com/?size=100&id=3tC9EQumUAuq&format=png&color=000000" alt="GitHub">
    </a>
    <a href="https://www.linkedin.com/in/ambinintsoa-ramanambonona" target="_blank" rel="noopener" aria-label="LinkedIn">
      <img src="https://img.icons8.com/?size=100&id=8808&format=png&color=000000" alt="LinkedIn">
    </a>
  </div>
</div>
""", unsafe_allow_html=True)
