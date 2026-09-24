from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    ARTIFACTS_DIR, FAILURE_EVENTS, PROCESSED_DIR,
    ANALOGUE_SENSORS, DIGITAL_SENSORS
)

st.set_page_config(
    page_title="RailGuard · Equipment Health Analytics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ─────────────────────────────────────────────────────────────────────
DARK = {
    "bg": "#0D1117", "bg2": "#161B22", "panel": "#1C2333", "panel2": "#21283A",
    "line": "#30384A", "text": "#E6EDF3", "muted": "#8B949E",
    "cyan": "#39C5CF", "blue": "#58A6FF", "lime": "#7EE787",
    "amber": "#F0883E", "red": "#F85149", "purple": "#BC8CFF",
    "grid": "#21283A", "shadow": "rgba(0,0,0,.4)",
    "tab_bg": "#161B22", "tab_active": "#1C2333",
}
LIGHT = {
    "bg": "#F6F8FA", "bg2": "#EAEEF2", "panel": "#FFFFFF", "panel2": "#F6F8FA",
    "line": "#D0D7DE", "text": "#1F2328", "muted": "#57606A",
    "cyan": "#0969DA", "blue": "#0550AE", "lime": "#1A7F37",
    "amber": "#BF5100", "red": "#CF222E", "purple": "#6639BA",
    "grid": "#EAEEF2", "shadow": "rgba(0,0,0,.08)",
    "tab_bg": "#EAEEF2", "tab_active": "#FFFFFF",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='display:flex;gap:10px;align-items:center;padding:4px 0 14px'>"
        "<span style='font-size:1.5rem'>◈</span>"
        "<div><div style='font-weight:800;font-size:.95rem;letter-spacing:.04em'>RAILGUARD</div>"
        "<div style='font-size:.6rem;letter-spacing:.12em;opacity:.55'>EQUIPMENT HEALTH ANALYTICS</div></div>"
        "</div>", unsafe_allow_html=True
    )
    st.divider()
    theme_choice = st.radio("Theme", ["Dark", "Light"], index=0, horizontal=True)

C = DARK if theme_choice == "Dark" else LIGHT

# ── CSS injection ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
:root {{
  --bg:{C['bg']}; --bg2:{C['bg2']}; --panel:{C['panel']}; --panel2:{C['panel2']};
  --line:{C['line']}; --text:{C['text']}; --muted:{C['muted']};
  --cyan:{C['cyan']}; --blue:{C['blue']}; --lime:{C['lime']};
  --amber:{C['amber']}; --red:{C['red']}; --purple:{C['purple']};
  --grid:{C['grid']}; --shadow:{C['shadow']};
}}
*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"] {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 14px;
}}
.stApp {{ background: var(--bg) !important; color: var(--text); }}
[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stSidebar"] {{
  background: var(--panel) !important;
  border-right: 1px solid var(--line) !important;
}}
[data-testid="stSidebar"] {{
  background: var(--panel) !important;
  border-right: 1px solid var(--line) !important;
  min-width: 300px !important;
}}
[data-testid="stSidebar"] * {{ color: var(--text) !important; }}
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] .stMarkdown p {{ color: var(--text) !important; }}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
  color: var(--text) !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] button {{
  background: var(--panel2) !important;
  color: var(--text) !important;
  border-color: var(--line) !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-baseweb="input"] * {{
  color: var(--text) !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] svg {{ fill: var(--muted) !important; color: var(--muted) !important; }}
[data-testid="stSidebar"] hr {{ border-color: var(--line) !important; opacity: 1 !important; margin: 18px 0 !important; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{ color: var(--text) !important; }}
.block-container {{
  max-width: 1680px !important;
  padding: 1.35rem clamp(1.15rem, 2.6vw, 3rem) 4.5rem !important;
}}
.stMarkdown, .stMarkdown p {{ color: var(--text); }}

/* KPI cards */
.kpi-grid {{ display:grid; gap:18px; width:100%; align-items:stretch; margin: 0 0 6px; }}
.kpi-grid.c4 {{ grid-template-columns: repeat(4, minmax(0,1fr)); }}
.kpi-grid.c5 {{ grid-template-columns: repeat(4, minmax(0,1fr)); }}
.kpi-grid.c3 {{ grid-template-columns: repeat(3, minmax(0,1fr)); }}
.kpi-grid.c2 {{ grid-template-columns: repeat(2, minmax(0,1fr)); }}
.kpi-card {{
  min-width: 0; min-height: 150px; height: 100%;
  background: linear-gradient(145deg, var(--panel), var(--panel2));
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 20px 20px 18px;
  display: flex; flex-direction: column; gap: 5px;
  box-shadow: 0 7px 22px var(--shadow);
  overflow: hidden;
}}
.kpi-label {{ font-size: .66rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: .09em; color: var(--muted); line-height: 1.45; min-height: 1.9em; }}
.kpi-value {{ font-size: clamp(1.55rem, 2.1vw, 1.95rem); font-weight: 800; line-height: 1.12; letter-spacing: -.025em; overflow-wrap:anywhere; word-break:break-word; }}
.kpi-unit  {{ font-size: .72rem; font-weight: 600; color: var(--muted); margin-left: 4px; white-space: nowrap; }}
.kpi-desc  {{ font-size: .69rem; color: var(--muted); line-height: 1.5; margin-top: auto; overflow-wrap:anywhere; }}
.kpi-accent-bar {{ height: 4px; border-radius: 3px; margin-bottom: 10px; width: 36px; flex:0 0 auto; }}

/* Section headers */
.sec-head {{
  display:flex; justify-content:space-between; align-items:flex-end;
  border-bottom: 1px solid var(--line); padding-bottom: 12px; margin: 38px 0 18px;
}}
.sec-title {{
  font-size: 1.2rem; font-weight: 800; color: var(--text); letter-spacing: -.025em; line-height:1.25;
}}
.sec-title .tag {{
  font-size: .62rem; font-weight: 700; font-family:'JetBrains Mono',monospace;
  letter-spacing: .1em; color: var(--cyan); margin-right: 8px; opacity:.85;
}}
.sec-sub {{ font-size: .68rem; color: var(--muted); font-weight: 500; text-align:right; line-height:1.5; }}

/* Hero banner */
.hero-wrap {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 28px 32px 26px;
  margin-bottom: 24px;
  box-shadow: 0 2px 12px var(--shadow);
}}
.hero-eyebrow {{
  font-size: .62rem; font-weight: 700; letter-spacing: .14em;
  text-transform: uppercase; color: var(--cyan); margin-bottom: 6px;
  font-family: 'JetBrains Mono', monospace;
}}
.hero-title {{
  font-size: clamp(1.5rem, 2.8vw, 2.2rem); font-weight: 800;
  line-height: 1.15; letter-spacing: -.04em; color: var(--text); margin-bottom: 6px;
}}
.hero-sub {{
  font-size: .82rem; color: var(--muted); line-height: 1.65; max-width: 820px;
}}
.hero-chips {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }}
.chip {{
  font-size: .62rem; font-weight: 600; letter-spacing:.06em;
  font-family:'JetBrains Mono',monospace;
  border: 1px solid var(--line); border-radius: 999px;
  padding: 5px 10px; color: var(--muted); background: var(--bg2);
  white-space: nowrap;
}}

/* Panel container */
.panel {{
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 14px; padding: 20px 22px;
  box-shadow: 0 2px 8px var(--shadow);
}}
.panel-title {{ font-size: .82rem; font-weight: 700; color: var(--text); margin-bottom: 4px; }}
.panel-sub   {{ font-size: .68rem; color: var(--muted); line-height:1.5; margin-bottom: 10px; }}

/* Insight banners */
.insight {{
  border-left: 3px solid var(--cyan); border-radius: 6px;
  background: var(--panel2); padding: 10px 14px;
  font-size: .74rem; color: var(--muted); line-height: 1.6; margin: 6px 0;
}}
.insight b {{ color: var(--text); font-weight: 700; }}
.insight.warn  {{ border-left-color: var(--amber); }}
.insight.danger{{ border-left-color: var(--red); }}
.insight.good  {{ border-left-color: var(--lime); }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
  background: var(--panel) !important; border-radius: 12px !important;
  padding: 6px !important; gap: 6px !important; border: 1px solid var(--line) !important;
  flex-wrap: nowrap !important; overflow-x: auto !important; scrollbar-width: thin;
  margin-top: 24px !important; margin-bottom: 8px !important;
}}
.stTabs [data-baseweb="tab"] {{
  border-radius: 9px !important; color: var(--muted) !important;
  font-size: .72rem !important; font-weight: 700 !important;
  padding: 10px 15px !important; min-height: 42px !important;
  white-space: nowrap !important; letter-spacing: .02em !important;
}}
.stTabs [data-baseweb="tab"] p {{ color: inherit !important; font-size: inherit !important; font-weight: inherit !important; margin: 0 !important; }}
.stTabs [aria-selected="true"] {{
  background: var(--panel) !important; color: var(--cyan) !important;
  box-shadow: 0 1px 4px var(--shadow) !important;
}}
.stTabs [data-baseweb="tab-highlight"] {{ display:none !important; }}

/* Widgets */
[data-testid='stWidgetLabel'] p {{
  color: var(--muted) !important; font-size: .66rem !important;
  font-weight: 600 !important; text-transform: uppercase !important;
  letter-spacing: .08em !important;
}}
div[data-baseweb="select"] > div {{
  background: var(--panel2) !important; border: 1px solid var(--line) !important;
  color: var(--text) !important; border-radius: 9px !important; min-height: 42px !important;
}}
[data-baseweb='select'] *, [data-baseweb='popover'] * {{ color: var(--text) !important; }}
[data-baseweb='select'] svg {{ fill: var(--muted) !important; color: var(--muted) !important; }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--line); border-radius: 10px; overflow:hidden; }}
[data-testid="stDateInput"] input,
.stDateInput input {{ background: var(--panel2) !important; color: var(--text) !important; -webkit-text-fill-color: var(--text) !important; border-color: var(--line) !important; opacity:1 !important; }}
[data-testid="stDateInput"] [data-baseweb="input"] {{ background: var(--panel2) !important; border:1px solid var(--line) !important; }}
div[data-baseweb="input"] {{ background: var(--panel2) !important; border-color: var(--line) !important; }}
.stButton > button {{
  background: var(--panel2) !important; color: var(--text) !important;
  border: 1px solid var(--line) !important; border-radius: 7px !important;
  font-weight: 600 !important; font-size: .72rem !important;
}}
.stButton > button:hover {{ border-color: var(--cyan) !important; color: var(--cyan) !important; }}
[data-testid="stToggleLabel"] {{ color: var(--muted) !important; font-size: .66rem !important; }}

/* Table */
.rg-table {{ width:100%; border-collapse:collapse; font-size:.74rem; }}
.rg-table th {{
  text-align:left; padding:9px 12px; border-bottom:1px solid var(--line);
  color:var(--muted); font:600 .62rem 'JetBrains Mono',monospace;
  text-transform:uppercase; letter-spacing:.08em; background:var(--panel2);
}}
.rg-table td {{ padding:9px 12px; border-bottom:1px solid var(--line); color:var(--text); }}
.rg-table tbody tr:last-child td {{ border-bottom:0; }}
.rg-table tbody tr:hover {{ background: color-mix(in srgb, var(--cyan) 5%, transparent); }}
.rg-table-wrap {{ border:1px solid var(--line); border-radius:8px; overflow:auto; background:var(--panel); }}

/* Footer */
.rg-footer {{
  margin-top:40px; padding-top:14px; border-top:1px solid var(--line);
  display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px;
  font:500 .62rem 'JetBrains Mono',monospace; color:var(--muted);
  letter-spacing:.06em;
}}

/* Responsive */
@media(max-width:1500px){{
  .kpi-grid.c5 {{ grid-template-columns:repeat(4,minmax(0,1fr)); }}
  .block-container {{ padding-left:1.25rem !important; padding-right:1.25rem !important; }}
}}
@media(max-width:1200px){{
  .kpi-grid.c5 {{ grid-template-columns:repeat(3,minmax(0,1fr)); }}
  .kpi-grid.c4 {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
}}
@media(max-width:800px){{
  .kpi-grid.c5,.kpi-grid.c4,.kpi-grid.c3 {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
}}
@media(max-width:500px){{
  .kpi-grid.c5,.kpi-grid.c4,.kpi-grid.c3,.kpi-grid.c2 {{ grid-template-columns:1fr; }}
}}
</style>
""", unsafe_allow_html=True)

# ── Paths and constants ───────────────────────────────────────────────────────
PROC        = PROCESSED_DIR / 'processed_1min.parquet'
ANOM        = ARTIFACTS_DIR / 'anomaly_scores.parquet'
RISK        = ARTIFACTS_DIR / 'risk_scores.parquet'
SCORE_STATS = ARTIFACTS_DIR / 'anomaly_score_stats.json'
PRED        = ARTIFACTS_DIR / 'predictive_results.json'
SHAP_FILE   = ARTIFACTS_DIR / 'shap_feature_importance.json'
EDA_STATS   = PROCESSED_DIR / 'eda_sensor_stats.csv'
FAIL_CSV    = PROCESSED_DIR / 'failure_event_summary.csv'
QUALITY     = PROCESSED_DIR / 'quality_report.json'

ANALOGUE = ANALOGUE_SENSORS  # from config
DIGITAL  = DIGITAL_SENSORS   # from config
UNITS = {
    'TP2': 'bar', 'TP3': 'bar', 'H1': 'bar', 'DV_pressure': 'bar',
    'Reservoirs': 'bar', 'Oil_temperature': '°C', 'Motor_current': 'A',
    'load_fraction': 'fraction', 'anomaly_score': 'score',
}
SENSOR_DESC = {
    'TP2': 'Compressor intake pressure',
    'TP3': 'Pneumatic panel pressure',
    'H1':  'Cyclonic separator discharge',
    'DV_pressure': 'Air-dryer pressure drop',
    'Reservoirs': 'Downstream reservoir',
    'Oil_temperature': 'Compressor oil temperature',
    'Motor_current': 'Motor phase current',
}
DIGITAL_DESC = {
    'COMP': 'Air intake valve (active = off/offloaded)',
    'DV_eletric': 'Outlet valve (active = under load)',
    'Towers': 'Tower selector (0=tower1, 1=tower2)',
    'MPG': 'Load-start signal (<8.2 bar trigger)',
    'LPS': 'Low pressure switch (<7 bar)',
    'Pressure_switch': 'Tower discharge detection',
    'Oil_level': 'Oil level alarm (active = low)',
    'Caudal_impulses': 'Air flow pulse counter',
}
COLORS = [
    '#39C5CF','#58A6FF','#7EE787','#F0883E','#BC8CFF',
    '#F85149','#E3B341','#79C0FF',
]

# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner='Loading RailGuard data…')
def load_all():
    df = pd.read_parquet(PROC)
    if ANOM.exists():
        df = df.join(pd.read_parquet(ANOM)[['anomaly_score']], how='left')
    if RISK.exists():
        df = df.join(pd.read_parquet(RISK)[['risk_prob']], how='left')
    return df

@st.cache_data
def load_json(p):
    if not Path(p).exists():
        return {}
    return json.loads(Path(p).read_text(encoding='utf8'))

@st.cache_data
def load_eda_stats():
    if EDA_STATS.exists():
        return pd.read_csv(EDA_STATS, index_col=0)
    return pd.DataFrame()

@st.cache_data
def load_fail_summary():
    if FAIL_CSV.exists():
        return pd.read_csv(FAIL_CSV)
    return pd.DataFrame()

def clean(x):
    return x.loc[~x['is_gap'].fillna(False)] if 'is_gap' in x.columns else x

def resamp(x, col, rule='1h'):
    if col not in x:
        return pd.DataFrame()
    y = x[[col]].dropna()
    return y.resample(rule).mean() if rule else y

def score_status(s, sts):
    lo = sts.get('alert_threshold_low', 0.835)
    hi = sts.get('alert_threshold_high', 0.988)
    if pd.isna(s):
        return 'UNKNOWN', C['muted']
    if s >= hi:
        return 'INVESTIGATE', C['red']
    if s >= lo:
        return 'MONITOR', C['amber']
    return 'NORMAL', C['lime']

def fmt(v, decimals=2, fallback='—'):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return fallback
    return f'{v:.{decimals}f}'

# ── Chart theme helper ────────────────────────────────────────────────────────
def apply_chart_theme(fig, height=360, title='', xtitle='', ytitle=''):
    fig.update_layout(
        height=height,
        title=dict(text=title, font=dict(size=13, color=C['text'], family='Inter, sans-serif'), x=0.01, xanchor='left'),
        margin=dict(l=72, r=28, t=92 if title else 24, b=68),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color=C['text'], size=11),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=C['text'], size=10), orientation='h',
                     yanchor='bottom', y=1.06, xanchor='left', x=0, tracegroupgap=8),
        hoverlabel=dict(bgcolor=C['panel'], font_color=C['text'], bordercolor=C['line']),
        hovermode='x unified',
    )
    fig.update_xaxes(
        gridcolor=C['grid'], zeroline=False, linecolor=C['line'],
        tickfont=dict(color=C['muted'], size=10),
        title_font=dict(color=C['muted'], size=10),
        title_text=xtitle, automargin=True, title_standoff=10,
    )
    fig.update_yaxes(
        gridcolor=C['grid'], zeroline=False, linecolor=C['line'],
        tickfont=dict(color=C['muted'], size=10),
        title_font=dict(color=C['muted'], size=10),
        title_text=ytitle, automargin=True, title_standoff=10,
    )
    return fig

def pc(fig, height=360, **kw):
    """Shorthand: apply theme and render."""
    return apply_chart_theme(fig, height=height, **kw)

# ── HTML helpers ─────────────────────────────────────────────────────────────
def section(title, sub=''):
    st.markdown(
        f"<div class='sec-head'>"
        f"<div class='sec-title'><span class='tag'>//</span>{title}</div>"
        f"<div class='sec-sub'>{sub}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

def kpi_grid(cards, cols=4):
    """cards = list of (label, value, unit, desc, color)"""
    html = f"<div class='kpi-grid c{cols}'>"
    for label, value, unit, desc, color in cards:
        html += (
            f"<div class='kpi-card'>"
            f"<div class='kpi-accent-bar' style='background:{color}'></div>"
            f"<div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value' style='color:{color}'>{value}"
            f"<span class='kpi-unit'>{unit}</span></div>"
            f"<div class='kpi-desc'>{desc}</div>"
            f"</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def html_table(frame):
    if frame.empty:
        st.markdown("<div class='panel'><div class='panel-title'>No data available</div></div>", unsafe_allow_html=True)
        return
    headers = ''.join(f"<th>{c}</th>" for c in frame.columns)
    rows = ''
    for _, row in frame.iterrows():
        rows += '<tr>' + ''.join(f"<td>{row[c]}</td>" for c in frame.columns) + '</tr>'
    st.markdown(
        f"<div class='rg-table-wrap'><table class='rg-table'>"
        f"<thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table></div>",
        unsafe_allow_html=True,
    )

# ── Load data ─────────────────────────────────────────────────────────────────
df0      = load_all()
stats    = load_json(SCORE_STATS)
pred     = load_json(PRED)
shap_imp = load_json(SHAP_FILE)
eda_st   = load_eda_stats()
fail_sum = load_fail_summary()
qual     = load_json(QUALITY)

if len(df0):
    dmin, dmax = df0.index.min().date(), df0.index.max().date()
else:
    dmin, dmax = pd.Timestamp('2020-02-01').date(), pd.Timestamp('2020-09-01').date()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='font-size:.65rem;font-weight:700;letter-spacing:.1em;color:var(--muted,#8B949E);text-transform:uppercase;margin-bottom:6px'>ANALYSIS WINDOW</div>", unsafe_allow_html=True)
    dates = st.date_input('', (dmin, dmax), min_value=dmin, max_value=dmax, label_visibility='collapsed')
    if isinstance(dates, tuple) and len(dates) == 2:
        start, end = dates
    else:
        start, end = dmin, dmax

    st.divider()
    st.markdown("<div style='font-size:.65rem;font-weight:700;letter-spacing:.1em;color:var(--muted,#8B949E);text-transform:uppercase;margin-bottom:6px'>MOTOR STATE FILTER</div>", unsafe_allow_html=True)
    motor_states_avail = ['All states', 'Off (≈0A)', 'Offloaded (≈4A)', 'Under Load (≈7A)', 'Starting (≈9A)']
    motor_filter = st.selectbox('', motor_states_avail, label_visibility='collapsed')

    st.divider()
    st.markdown("<div style='font-size:.65rem;font-weight:700;letter-spacing:.1em;color:var(--muted,#8B949E);text-transform:uppercase;margin-bottom:6px'>FAILURE EVENT</div>", unsafe_allow_html=True)
    event_labels = ['All events'] + [e['id'] for e in FAILURE_EVENTS]
    sidebar_event = st.selectbox('', event_labels, label_visibility='collapsed')

    st.divider()
    total_rows = qual.get('structure', {}).get('n_rows', len(df0))
    raw_start = '2020-02-01'
    raw_end   = '2020-09-01'
    st.markdown(
        f"<div style='font-size:.68rem;line-height:1.9;color:#8B949E'>"
        f"DATASET<br><b style='color:{C['text']}'>UCI MetroPT-3</b><br>"
        f"ASSET<br><b style='color:{C['text']}'>APU Compressor</b><br>"
        f"RAW ROWS<br><b style='color:{C['text']}'>{total_rows:,}</b><br>"
        f"EVENTS<br><b style='color:{C['red']}'>4 documented</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ── Filter data ───────────────────────────────────────────────────────────────
df = df0[
    (df0.index >= pd.Timestamp(start)) &
    (df0.index < pd.Timestamp(end) + pd.Timedelta(days=1))
] if len(df0) else df0
ng = clean(df)

# Motor state filter
def get_motor_state_mask(df_in, mf):
    mc = df_in.get('Motor_current', pd.Series(dtype=float))
    if mf == 'Off (≈0A)':
        return mc < 1.0
    elif mf == 'Offloaded (≈4A)':
        return (mc >= 1.0) & (mc < 5.5)
    elif mf == 'Under Load (≈7A)':
        return (mc >= 5.5) & (mc < 8.0)
    elif mf == 'Starting (≈9A)':
        return mc >= 8.0
    return pd.Series(True, index=df_in.index)

if motor_filter != 'All states' and len(ng):
    ng = ng[get_motor_state_mask(ng, motor_filter)]

# Derived analytics
score  = float(ng['anomaly_score'].dropna().iloc[-1]) if len(ng) and 'anomaly_score' in ng and ng['anomaly_score'].notna().any() else np.nan
risk   = float(ng['risk_prob'].dropna().iloc[-1])     if len(ng) and 'risk_prob' in ng and ng['risk_prob'].notna().any() else np.nan
status, status_color = score_status(score, stats)
latest = ng.index[-1] if len(ng) else None

failure_id_series = ng.get('failure_id', pd.Series('', index=ng.index)).fillna('') if len(ng) else pd.Series(dtype=str)
failure_rows = int((failure_id_series != '').sum())
investigate  = int((ng['anomaly_score'] >= stats.get('alert_threshold_high', 0.988)).sum()) if 'anomaly_score' in ng else 0

n_records = len(ng)
coverage_days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
expected_1min = coverage_days * 24 * 60
coverage_pct  = min(100.0, n_records / max(expected_1min, 1) * 100)

avg_oil  = float(ng['Oil_temperature'].mean()) if len(ng) and 'Oil_temperature' in ng else np.nan
avg_curr = float(ng['Motor_current'].mean())   if len(ng) and 'Motor_current' in ng else np.nan
avg_tp2  = float(ng['TP2'].mean())             if len(ng) and 'TP2' in ng else np.nan
avg_h1   = float(ng['H1'].mean())              if len(ng) and 'H1' in ng else np.nan
avg_lf   = float(ng['load_fraction'].mean())   if len(ng) and 'load_fraction' in ng else np.nan

# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"<div class='hero-wrap'>"
    f"<div class='hero-eyebrow'>MetroPT-3 · Air Production Unit · IBM SkillsBuild Data Analytics</div>"
    f"<div class='hero-title'>RailGuard — Equipment Health Analytics</div>"
    f"<div class='hero-sub'>Data-driven analysis of MetroPT-3 Air Production Unit compressor operations. "
    f"Explore sensor performance, operational trends, failure event signatures, and predictive model insights.</div>"
    f"<div class='hero-chips'>"
    f"<span class='chip'>● {start} → {end}</span>"
    f"<span class='chip'>1-MIN PROCESSED LAYER</span>"
    f"<span class='chip'>4 DOCUMENTED FAILURE EVENTS</span>"
    f"<span class='chip'>7 ANALOGUE + 8 DIGITAL SENSORS</span>"
    f"<span class='chip'>MODEL STATE: {status}</span>"
    f"</div></div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
section('Executive Overview', f'Window: {start} → {end}')
kpi_grid([
    ('Total Observations',     f'{n_records:,}',          '',    '1-min records in selected window',            C['cyan']),
    ('Data Coverage',          f'{coverage_pct:.1f}',     '%',   f'{n_records:,} of ~{expected_1min:,} expected min', C['blue']),
    ('Avg Oil Temperature',    fmt(avg_oil, 1),            '°C',  'Mean over selected window',                   C['amber']),
    ('Avg Motor Current',      fmt(avg_curr, 2),           ' A',  'Mean phase current',                          C['purple']),
    ('Avg TP2 Pressure',       fmt(avg_tp2, 3),            ' bar','Compressor intake mean',                      C['lime']),
], cols=5)

kpi_grid([
    ('Avg H1 Pressure',        fmt(avg_h1, 3),             ' bar','Cyclonic separator discharge mean',            C['cyan']),
    ('Loaded Operation',       f'{avg_lf*100:.1f}' if not np.isnan(avg_lf) else '—', '%', 'Fraction of time under load', C['blue']),
    ('Failure Overlap',        f'{failure_rows:,}',        '',    'Rows inside documented failure windows',       C['red'] if failure_rows else C['lime']),
    ('Anomaly: Investigate',   f'{investigate:,}',         '',    f'Score ≥ {stats.get("alert_threshold_high", 0.988):.3f}', C['red'] if investigate else C['lime']),
    ('Model State',            status,                     '',    f'Score {fmt(score,4)} · Risk {fmt(risk*100 if not np.isnan(risk) else np.nan,1)}%', status_color),
], cols=5)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    '📊 Sensor Analytics',
    '📈 Operational Trends',
    '⚙️ Operating States',
    '🔌 Digital Signals',
    '🔗 Sensor Relationships',
    '⚠️ Failure Events',
    '🔍 Anomaly Analytics',
    '🤖 Predictive Model',
    '💡 Maintenance Insights',
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SENSOR ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    section('Sensor Performance Analytics', 'Average values across selected window')

    # Avg bar chart
    ana = [s for s in ANALOGUE if s in ng]
    avgs = {s: float(ng[s].mean()) for s in ana}

    fig_avg = go.Figure(go.Bar(
        x=list(avgs.keys()),
        y=list(avgs.values()),
        marker_color=[C['cyan'], C['blue'], C['purple'], C['amber'], C['lime'], C['red'], C['blue'][:7]],
        text=[f"{v:.3f}" for v in avgs.values()],
        textposition='outside',
        hovertemplate='%{x}<br>Mean: %{y:.4f}<extra></extra>',
    ))
    fig_avg.update_layout(showlegend=False)
    st.plotly_chart(pc(fig_avg, height=340, title='Average Value by Sensor (selected window)', ytitle='Value'), width="stretch", config={'displayModeBar': False})

    section('Descriptive Statistics', 'Min · Max · Mean · Median · Std Dev')
    # Stats table from eda_stats or calculated
    stat_rows = []
    for s in ana:
        col_data = ng[s].dropna()
        if len(col_data) == 0:
            continue
        stat_rows.append({
            'Sensor': s,
            'Unit': UNITS.get(s, ''),
            'Min':    fmt(float(col_data.min()), 4),
            'Max':    fmt(float(col_data.max()), 4),
            'Mean':   fmt(float(col_data.mean()), 4),
            'Median': fmt(float(col_data.median()), 4),
            'Std Dev': fmt(float(col_data.std()), 4),
            'Description': SENSOR_DESC.get(s, ''),
        })
    html_table(pd.DataFrame(stat_rows))

    section('Sensor Value Distributions', 'Select sensor — histogram of observed values')
    dist_col1, dist_col2 = st.columns([1, 3])
    with dist_col1:
        dist_sensor = st.selectbox('Sensor', ana, key='dist_sensor_sel', index=ana.index('H1') if 'H1' in ana else 0)
        show_failure_overlay = st.toggle('Overlay failure window', value=True, key='dist_fail_overlay')
    with dist_col2:
        normal_mask = failure_id_series == '' if len(failure_id_series) else pd.Series(True, index=ng.index)
        fail_mask   = failure_id_series != '' if len(failure_id_series) else pd.Series(False, index=ng.index)
        normal_data = ng.loc[normal_mask, dist_sensor].dropna() if dist_sensor in ng else pd.Series()
        fail_data   = ng.loc[fail_mask, dist_sensor].dropna()   if dist_sensor in ng else pd.Series()

        fig_hist = go.Figure()
        if len(normal_data):
            fig_hist.add_trace(go.Histogram(x=normal_data, name='Normal', nbinsx=80,
                marker_color=C['blue'], opacity=0.70))
        if show_failure_overlay and len(fail_data):
            fig_hist.add_trace(go.Histogram(x=fail_data, name='Failure window', nbinsx=50,
                marker_color=C['red'], opacity=0.75))
        fig_hist.update_layout(barmode='overlay', showlegend=True)
        st.plotly_chart(pc(fig_hist, height=310, title=f'{dist_sensor} Distribution ({UNITS.get(dist_sensor,"")})', xtitle=f'{dist_sensor} ({UNITS.get(dist_sensor,"")})', ytitle='Count'), width="stretch", config={'displayModeBar': False})

    # 4-panel histograms
    section('Distribution Overview', 'H1 · TP2 · Oil Temperature · Motor Current')
    col_a, col_b = st.columns(2)
    for i, (sc, col_ctx) in enumerate([('H1', col_a), ('TP2', col_b), ('Oil_temperature', col_a), ('Motor_current', col_b)]):
        if sc not in ng:
            continue
        data = ng[sc].dropna()
        fh = go.Figure(go.Histogram(x=data, nbinsx=70, marker_color=COLORS[i % len(COLORS)], opacity=0.80))
        with col_ctx:
            st.plotly_chart(pc(fh, height=260, title=f'{sc} ({UNITS.get(sc,"")}) — Histogram', xtitle=UNITS.get(sc,''), ytitle='Count'), width="stretch", config={'displayModeBar': False})

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — OPERATIONAL TRENDS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    section('Operational Trends', 'Time-series analysis of sensor behaviour')

    t1c1, t1c2, t1c3 = st.columns(3)
    with t1c1:
        ts_sensor = st.selectbox('Sensor', ana, index=ana.index('H1') if 'H1' in ana else 0, key='ts_sensor')
    with t1c2:
        ts_res_label = st.selectbox('Resolution', ['1 minute', '10 minutes', '1 hour', '6 hours', 'Daily'], index=2, key='ts_res')
    with t1c3:
        ts_fail_overlay = st.toggle('Mark failure windows', value=True, key='ts_fail')

    ts_rule = {'1 minute': None, '10 minutes': '10min', '1 hour': '1h', '6 hours': '6h', 'Daily': '1D'}[ts_res_label]
    ts_p = resamp(ng, ts_sensor, ts_rule)
    fig_ts = go.Figure()
    if len(ts_p):
        fig_ts.add_trace(go.Scatter(
            x=ts_p.index, y=ts_p[ts_sensor],
            mode='lines', name=ts_sensor,
            line=dict(color=C['cyan'], width=2),
            fill='tozeroy', fillcolor=f"rgba({int(C['cyan'][1:3],16)},{int(C['cyan'][3:5],16)},{int(C['cyan'][5:7],16)},0.06)",
        ))
    if ts_fail_overlay:
        for ev in FAILURE_EVENTS:
            fig_ts.add_vrect(x0=ev['start'], x1=ev['end'],
                fillcolor=C['red'], opacity=0.13, line_width=0,
                annotation_text=ev['id'], annotation_font_color=C['red'],
                annotation_font_size=10,
            )
    st.plotly_chart(pc(fig_ts, height=400, title=f'{ts_sensor} over time ({ts_res_label})', xtitle='Time', ytitle=f'{ts_sensor} ({UNITS.get(ts_sensor,"")})'), width="stretch", config={'displayModeBar': False})

    section('Multi-Sensor Trend Comparison', 'Standardised (z-score) signal overlay')
    signals = [x for x in ['H1', 'TP2', 'Oil_temperature', 'Motor_current', 'load_fraction'] if x in ng]
    fig_multi = go.Figure()
    for i, s in enumerate(signals):
        p = resamp(ng, s, '6h')
        if len(p):
            z = (p[s] - p[s].median()) / (p[s].std() or 1)
            fig_multi.add_trace(go.Scatter(
                x=p.index, y=z, mode='lines', name=s,
                line=dict(color=COLORS[i % len(COLORS)], width=1.6),
            ))
    for ev in FAILURE_EVENTS:
        fig_multi.add_vrect(x0=ev['start'], x1=ev['end'], fillcolor=C['red'], opacity=0.10, line_width=0)
    st.plotly_chart(pc(fig_multi, height=360, title='Normalised sensor signals (z-score, 6h resolution)', ytitle='Z-score'), width="stretch", config={'displayModeBar': False})

    # Side-by-side recent trends
    section('Key Sensor Pair Trends', 'H1 pressure and Motor current — hourly')
    tl, tr = st.columns(2)
    for (sc, color), col_ctx in [
        (('H1', C['cyan']), tl),
        (('Motor_current', C['purple']), tr),
    ]:
        if sc not in ng: continue
        pp = resamp(ng, sc, '1h')
        fig_pair = go.Figure()
        if len(pp):
            fig_pair.add_trace(go.Scatter(x=pp.index, y=pp[sc], mode='lines', line=dict(color=color, width=2), name=sc))
        for ev in FAILURE_EVENTS:
            fig_pair.add_vrect(x0=ev['start'], x1=ev['end'], fillcolor=C['red'], opacity=0.12, line_width=0)
        with col_ctx:
            st.plotly_chart(pc(fig_pair, height=280, title=f'{sc} — hourly', ytitle=UNITS.get(sc,'')), width="stretch", config={'displayModeBar': False})

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — OPERATING STATES
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    section('Motor Operating State Analytics', 'Distribution and current by state')

    if 'Motor_current' in ng and len(ng):
        mc = ng['Motor_current'].dropna()
        # Classify states
        def classify_state(v):
            if v < 1.0:   return 'Off (≈0A)'
            if v < 5.5:   return 'Offloaded (≈4A)'
            if v < 8.0:   return 'Under Load (≈7A)'
            return 'Starting (≈9A)'
        states = mc.apply(classify_state)
        state_counts = states.value_counts()
        state_pct    = states.value_counts(normalize=True) * 100

        col_pie, col_bar = st.columns(2)
        with col_pie:
            fig_pie = go.Figure(go.Pie(
                labels=state_counts.index.tolist(),
                values=state_counts.values.tolist(),
                hole=0.48,
                marker_colors=[C['lime'], C['blue'], C['amber'], C['red']],
                textinfo='label+percent',
                textfont=dict(size=11),
            ))
            fig_pie.update_layout(showlegend=False)
            st.plotly_chart(pc(fig_pie, height=320, title='Motor State Distribution (share of records)'), width="stretch", config={'displayModeBar': False})

        with col_bar:
            state_avg_curr = mc.groupby(states).mean().sort_values()
            fig_scurr = go.Figure(go.Bar(
                x=state_avg_curr.index.tolist(),
                y=state_avg_curr.values.tolist(),
                marker_color=[C['lime'], C['blue'], C['amber'], C['red']],
                text=[f'{v:.2f} A' for v in state_avg_curr.values],
                textposition='outside',
            ))
            fig_scurr.update_layout(showlegend=False)
            st.plotly_chart(pc(fig_scurr, height=320, title='Average Motor Current by Operating State', ytitle='Motor Current (A)'), width="stretch", config={'displayModeBar': False})

        # Table summary
        section('State Summary Table', 'Observed distribution of operating states')
        tbl_rows = []
        for st_name in state_counts.index:
            mask_s = states == st_name
            subset = mc[mask_s]
            tbl_rows.append({
                'State': st_name,
                'Count': f'{len(subset):,}',
                'Pct of Records': f'{state_pct[st_name]:.1f}%',
                'Min Current (A)': fmt(float(subset.min()), 3),
                'Max Current (A)': fmt(float(subset.max()), 3),
                'Mean Current (A)': fmt(float(subset.mean()), 3),
            })
        html_table(pd.DataFrame(tbl_rows))

        # load_fraction donut
        if 'load_fraction' in ng:
            section('Load Fraction Distribution', 'Histogram of load fraction across records')
            fig_lf = go.Figure(go.Histogram(x=ng['load_fraction'].dropna(), nbinsx=50, marker_color=C['cyan'], opacity=0.78))
            st.plotly_chart(pc(fig_lf, height=260, title='Load Fraction — Distribution', xtitle='Load Fraction', ytitle='Count'), width="stretch", config={'displayModeBar': False})
    else:
        st.info('Motor current data not available in selected window.')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DIGITAL SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    section('Digital Signal Activity', 'Activation percentage across available digital sensors')

    digs = [s for s in DIGITAL if s in ng]
    if digs and len(ng):
        pct_active = {}
        for d in digs:
            pct_active[d] = float(ng[d].mean() * 100)

        fig_dig = go.Figure(go.Bar(
            x=list(pct_active.keys()),
            y=list(pct_active.values()),
            marker_color=[COLORS[i % len(COLORS)] for i in range(len(pct_active))],
            text=[f'{v:.1f}%' for v in pct_active.values()],
            textposition='outside',
            hovertemplate='%{x}<br>Active: %{y:.2f}%<extra></extra>',
        ))
        fig_dig.update_layout(showlegend=False)
        fig_dig.update_yaxes(range=[0, 110])
        st.plotly_chart(pc(fig_dig, height=360, title='Digital Signal — % of observations where signal is active (1)', ytitle='Active %'), width="stretch", config={'displayModeBar': False})

        section('Digital Signal Summary Table', 'Activity rate and description for each digital sensor')
        dig_rows = []
        for d in digs:
            n_active = int(ng[d].sum())
            n_total  = int(ng[d].notna().sum())
            dig_rows.append({
                'Signal': d,
                'Active (count)': f'{n_active:,}',
                'Total (count)': f'{n_total:,}',
                'Active %': f'{pct_active[d]:.2f}%',
                'Description': DIGITAL_DESC.get(d, ''),
            })
        html_table(pd.DataFrame(dig_rows))

        # Pie for binary split of a selected signal
        section('Signal Detail', 'Active vs inactive breakdown for selected signal')
        dc1, dc2 = st.columns([1, 2])
        with dc1:
            sel_dig = st.selectbox('Digital signal', digs, key='dig_detail')
        with dc2:
            n1 = int(ng[sel_dig].sum())
            n0 = int((ng[sel_dig] == 0).sum())
            fig_dpie = go.Figure(go.Pie(
                labels=['Active (1)', 'Inactive (0)'],
                values=[n1, n0],
                hole=0.45,
                marker_colors=[C['cyan'], C['grid']],
                textinfo='label+percent',
            ))
            fig_dpie.update_layout(showlegend=False)
            st.plotly_chart(pc(fig_dpie, height=280, title=f'{sel_dig} — Active vs Inactive'), width="stretch", config={'displayModeBar': False})
    else:
        st.info('No digital signal data in selected window.')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SENSOR RELATIONSHIPS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    section('Sensor Relationships', 'Correlations and scatter plots')

    nums = [s for s in ANALOGUE if s in ng]
    if nums and len(ng) > 2:
        # Correlation heatmap
        corr = ng[nums].corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale=[[0, C['red']], [0.5, C['panel2']], [1, C['cyan']]],
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            textfont=dict(size=10),
            hovertemplate='%{y} × %{x}<br>r = %{z:.3f}<extra></extra>',
            colorbar=dict(tickfont=dict(color=C['muted'], size=9), len=0.8),
        ))
        fig_corr.update_layout(height=420)
        st.plotly_chart(pc(fig_corr, height=420, title='Analogue Sensor Correlation Matrix (Pearson r)'), width="stretch", config={'displayModeBar': False})

        # Computed observations
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        max_pair = upper.stack().idxmax()
        min_pair = upper.stack().idxmin()
        max_r    = corr.loc[max_pair[0], max_pair[1]]
        min_r    = corr.loc[min_pair[0], min_pair[1]]
        st.markdown(
            f"<div class='insight good'><b>Strongest positive:</b> {max_pair[0]} × {max_pair[1]} — r = {max_r:.3f}. "
            f"Note: correlation indicates a statistical relationship; it does not establish physical causation.</div>"
            f"<div class='insight warn'><b>Strongest negative:</b> {min_pair[0]} × {min_pair[1]} — r = {min_r:.3f}. "
            f"This reflects the operational behaviour of the compressor cycle.</div>",
            unsafe_allow_html=True,
        )

        section('Scatter Plots', 'TP2 vs H1 · Motor Current vs Oil Temperature')
        sc1, sc2 = st.columns(2)
        with sc1:
            if 'TP2' in ng and 'H1' in ng:
                sample = ng[['TP2', 'H1']].dropna().sample(min(6000, len(ng)), random_state=42)
                fig_s1 = go.Figure(go.Scatter(
                    x=sample['TP2'], y=sample['H1'],
                    mode='markers',
                    marker=dict(color=C['cyan'], size=3, opacity=0.35),
                    hovertemplate='TP2=%{x:.3f}<br>H1=%{y:.3f}<extra></extra>',
                ))
                st.plotly_chart(pc(fig_s1, height=320, title='TP2 vs H1 (sample)', xtitle='TP2 (bar)', ytitle='H1 (bar)'), width="stretch", config={'displayModeBar': False})
        with sc2:
            if 'Motor_current' in ng and 'Oil_temperature' in ng:
                sample2 = ng[['Motor_current', 'Oil_temperature']].dropna().sample(min(6000, len(ng)), random_state=42)
                fig_s2 = go.Figure(go.Scatter(
                    x=sample2['Motor_current'], y=sample2['Oil_temperature'],
                    mode='markers',
                    marker=dict(color=C['amber'], size=3, opacity=0.35),
                    hovertemplate='Motor=%{x:.3f} A<br>Oil=%{y:.1f}°C<extra></extra>',
                ))
                st.plotly_chart(pc(fig_s2, height=320, title='Motor Current vs Oil Temperature (sample)', xtitle='Motor Current (A)', ytitle='Oil Temp (°C)'), width="stretch", config={'displayModeBar': False})
    else:
        st.info('Insufficient data for correlation analysis.')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — FAILURE EVENTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    section('Failure Event Analytics', 'Documented events — MetroPT-3 maintenance records')

    # Summary table from precomputed file
    if not fail_sum.empty:
        fe_display_cols = ['failure_id', 'period', 'n_rows', 'duration_h',
                           'H1_mean', 'TP2_mean', 'Oil_temperature_mean', 'Motor_current_mean', 'LPS_pct_active']
        fe_display_cols = [c for c in fe_display_cols if c in fail_sum.columns]
        fs_display = fail_sum[fe_display_cols].copy()
        for col in fs_display.select_dtypes(include='float').columns:
            fs_display[col] = fs_display[col].map(lambda v: fmt(v, 3))
        html_table(fs_display.rename(columns={
            'failure_id': 'Event', 'period': 'Period', 'n_rows': 'Records',
            'duration_h': 'Duration (h)',
            'H1_mean': 'H1 mean', 'TP2_mean': 'TP2 mean',
            'Oil_temperature_mean': 'Oil Temp mean', 'Motor_current_mean': 'Motor mean',
            'LPS_pct_active': 'LPS active %',
        }))

    # Event selector
    section('Event Deep-Dive', 'Select an event to explore pre/during/post comparisons')
    ev_labels = [e['id'] for e in FAILURE_EVENTS]
    chosen_ev = st.selectbox('Event', ev_labels, key='fail_event_sel')
    ev = next(x for x in FAILURE_EVENTS if x['id'] == chosen_ev)
    es, ee = pd.Timestamp(ev['start']), pd.Timestamp(ev['end'])
    ev_dur_h = (ee - es).total_seconds() / 3600

    # KPI strip for this event
    kpi_grid([
        ('Event ID',          ev['id'],              '',  f"Type: {ev['type']}", C['red']),
        ('Duration',          f'{ev_dur_h:.1f}',     ' h', ev['start'], C['amber']),
        ('Severity',          ev.get('severity','—'),'',  ev['end'], C['purple']),
        ('Maintenance',       str(ev.get('maintenance','—'))[:10], '', 'Documented maintenance date', C['lime']),
    ], cols=4)

    # Before / During / After comparison charts
    if not fail_sum.empty:
        ev_rows = fail_sum[fail_sum['failure_id'] == chosen_ev]
        if not ev_rows.empty:
            periods = {r['period']: r for _, r in ev_rows.iterrows()}
            compare_sensors = ['H1', 'TP2', 'Oil_temperature', 'Motor_current']
            cols_compare = ['before', 'during', 'after'] if all(p in periods for p in ['pre','failure','post']) else list(periods.keys())
            bda_keys = ['pre', 'failure', 'post']
            bda_labels = ['Before (72h)', 'During', 'After (48h)']
            bda_colors = [C['blue'], C['red'], C['lime']]

            section('Before / During / After Comparison', f'{chosen_ev} — sensor mean values')
            c_l, c_r = st.columns(2)
            for i, sc in enumerate(compare_sensors):
                mean_col = f'{sc}_mean'
                if mean_col not in fail_sum.columns:
                    continue
                vals  = [periods[k][mean_col] if k in periods else np.nan for k in bda_keys]
                fig_bda = go.Figure(go.Bar(
                    x=bda_labels, y=vals, marker_color=bda_colors,
                    text=[f'{v:.3f}' if not np.isnan(v) else '—' for v in vals],
                    textposition='outside',
                ))
                fig_bda.update_layout(showlegend=False)
                col_ctx = c_l if i % 2 == 0 else c_r
                with col_ctx:
                    st.plotly_chart(pc(fig_bda, height=260, title=f'{sc} — Before/During/After mean', ytitle=UNITS.get(sc,'')), width="stretch", config={'displayModeBar': False})

    # Signature time-series
    section('Failure Pressure Signature', f'{chosen_ev} — H1 and TP2 around event')
    window_data = df0[
        (df0.index >= es - pd.Timedelta(hours=72)) &
        (df0.index <= ee + pd.Timedelta(hours=48))
    ]
    fig_sig = go.Figure()
    for sc, col_color in [('H1', C['cyan']), ('TP2', C['purple'])]:
        if sc not in window_data:
            continue
        pp = resamp(window_data, sc, '30min')
        if len(pp):
            fig_sig.add_trace(go.Scatter(
                x=pp.index, y=pp[sc], mode='lines', name=sc,
                line=dict(color=col_color, width=2),
            ))
    fig_sig.add_vrect(
        x0=ev['start'], x1=ev['end'],
        fillcolor=C['red'], opacity=0.18, line_width=0,
        annotation_text='FAILURE WINDOW',
        annotation_font_color=C['red'], annotation_font_size=11,
    )
    st.plotly_chart(pc(fig_sig, height=400, title=f'{chosen_ev} — Pressure Signature (H1 + TP2, 30-min resolution)', xtitle='Time', ytitle='Pressure (bar)'), width="stretch", config={'displayModeBar': False})

    # Evidence table inside failure window
    section('Signal Evidence Inside Failure Window', 'Median · Min · Max for documented window')
    inside = df0[(df0.index >= es) & (df0.index <= ee)]
    ev_rows_tbl = []
    for sc in ['H1', 'TP2', 'Oil_temperature', 'Motor_current', 'load_fraction', 'anomaly_score']:
        if sc not in inside or len(inside) == 0:
            continue
        d = inside[sc].dropna()
        ev_rows_tbl.append({
            'Signal': sc, 'Unit': UNITS.get(sc, ''),
            'Min': fmt(float(d.min()), 4), 'Median': fmt(float(d.median()), 4),
            'Max': fmt(float(d.max()), 4), 'Mean': fmt(float(d.mean()), 4),
        })
    html_table(pd.DataFrame(ev_rows_tbl))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — ANOMALY ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    section('Anomaly Detection Analytics', 'Isolation Forest — trained on clean Feb–Mar 2020 baseline')

    lo_thr = stats.get('alert_threshold_low', 0.835)
    hi_thr = stats.get('alert_threshold_high', 0.988)

    # KPI strip
    anom_series = df0['anomaly_score'].dropna() if 'anomaly_score' in df0 else pd.Series(dtype=float)
    n_monitor     = int((anom_series >= lo_thr).sum())
    n_investigate = int((anom_series >= hi_thr).sum())
    kpi_grid([
        ('Monitor Threshold',       fmt(lo_thr, 3),  '', '95th pct of clean baseline',       C['amber']),
        ('Investigate Threshold',   fmt(hi_thr, 3),  '', '99th pct of clean baseline',       C['red']),
        ('Records ≥ Monitor',        f'{n_monitor:,}', '', 'Across full dataset',             C['amber']),
        ('Records ≥ Investigate',    f'{n_investigate:,}', '', 'Across full dataset',        C['red']),
    ], cols=4)

    # Score distribution + thresholds
    section('Anomaly Score Distribution', 'Score histogram across full processed dataset')
    fig_anom_hist = go.Figure(go.Histogram(x=anom_series, nbinsx=80, marker_color=C['cyan'], opacity=0.75))
    fig_anom_hist.add_vline(x=lo_thr, line_dash='dot', line_color=C['amber'],
        annotation_text='MONITOR', annotation_font_color=C['amber'], annotation_font_size=10)
    fig_anom_hist.add_vline(x=hi_thr, line_dash='dot', line_color=C['red'],
        annotation_text='INVESTIGATE', annotation_font_color=C['red'], annotation_font_size=10)
    st.plotly_chart(pc(fig_anom_hist, height=340, title='Anomaly Score Distribution — full dataset', xtitle='Anomaly Score', ytitle='Count'), width="stretch", config={'displayModeBar': False})

    st.markdown(
        f"<div class='insight warn'><b>MONITOR ≥ {lo_thr:.3f}</b> — 95th percentile of clean Feb–Mar baseline scores.</div>"
        f"<div class='insight danger'><b>INVESTIGATE ≥ {hi_thr:.3f}</b> — 99th percentile of clean baseline scores. All four documented failure windows reached this region.</div>"
        f"<div class='insight'><b>Interpretation guidance:</b> An elevated anomaly score indicates deviation from the clean baseline. "
        f"It is an <em>observed anomaly signal</em>, not a confirmed failure. Human engineering review is required before any maintenance action.</div>",
        unsafe_allow_html=True,
    )

    # Score over time with failure overlays
    section('Anomaly Score Over Time', 'Hourly average — with documented failure windows marked')
    asc, bsc = st.columns([3, 1])
    with asc:
        ap = resamp(ng, 'anomaly_score', '1h')
        fig_anom_ts = go.Figure()
        if len(ap):
            fig_anom_ts.add_trace(go.Scatter(
                x=ap.index, y=ap['anomaly_score'], mode='lines', name='Anomaly Score',
                line=dict(color=C['cyan'], width=2),
                fill='tozeroy', fillcolor=f"rgba(57,197,207,0.06)",
            ))
        fig_anom_ts.add_hline(y=lo_thr, line_dash='dot', line_color=C['amber'],
            annotation_text='MONITOR', annotation_font_color=C['amber'], annotation_font_size=10)
        fig_anom_ts.add_hline(y=hi_thr, line_dash='dot', line_color=C['red'],
            annotation_text='INVESTIGATE', annotation_font_color=C['red'], annotation_font_size=10)
        for ev in FAILURE_EVENTS:
            fig_anom_ts.add_vrect(x0=ev['start'], x1=ev['end'],
                fillcolor=C['red'], opacity=0.13, line_width=0,
                annotation_text=ev['id'], annotation_font_color=C['red'], annotation_font_size=10)
        st.plotly_chart(pc(fig_anom_ts, height=380, title='Anomaly Score (hourly avg) — with documented failure windows', ytitle='Score'), width="stretch", config={'displayModeBar': False})
    with bsc:
        st.markdown(
            f"<div class='panel' style='margin-top:0'>"
            f"<div class='panel-title'>Current State</div>"
            f"<div style='font-size:1.6rem;font-weight:800;color:{status_color};margin:10px 0 6px'>{status}</div>"
            f"<div style='font-size:.68rem;color:{C['muted']};font-family:JetBrains Mono,monospace;line-height:1.9'>"
            f"SCORE<br><b style='color:{C['text']}'>{fmt(score, 4)}</b><br>"
            f"RISK<br><b style='color:{C['text']}'>{fmt(risk*100 if not np.isnan(risk) else np.nan, 2)}%</b><br>"
            f"AS OF<br><b style='color:{C['text']}'>{str(latest)[:16] if latest else '—'}</b>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    # Normal vs anomalous
    section('Normal vs Anomalous Observations', 'Observations classified by anomaly threshold')
    if 'anomaly_score' in ng and len(ng):
        n_norm = int((ng['anomaly_score'] < lo_thr).sum())
        n_mon  = int(((ng['anomaly_score'] >= lo_thr) & (ng['anomaly_score'] < hi_thr)).sum())
        n_inv  = int((ng['anomaly_score'] >= hi_thr).sum())
        fig_pie_anom = go.Figure(go.Pie(
            labels=['Normal', 'Monitor', 'Investigate'],
            values=[n_norm, n_mon, n_inv],
            hole=0.45,
            marker_colors=[C['lime'], C['amber'], C['red']],
            textinfo='label+percent+value',
            textfont=dict(size=11),
        ))
        fig_pie_anom.update_layout(showlegend=False)
        st.plotly_chart(pc(fig_pie_anom, height=300, title='Anomaly Score Classification — selected window'), width="stretch", config={'displayModeBar': False})

# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — PREDICTIVE MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    section('Predictive Model Analytics', 'Supervised models trained on chronological split')

    # Model description
    st.markdown(
        f"<div class='insight'><b>Primary model:</b> Isolation Forest (unsupervised) — scores deviation from clean Feb–Mar 2020 baseline. "
        f"Trained on 64,207 records. Thresholds set at 95th/99th percentile of training scores.</div>"
        f"<div class='insight warn'><b>Secondary models:</b> Logistic Regression and HistGradientBoosting trained on chronological split. "
        f"<b>Important limitation:</b> only 4 documented failure events exist; the test period contains very limited failure examples, "
        f"which leads to degenerate supervised metrics on the test set. Validation metrics are more informative.</div>",
        unsafe_allow_html=True,
    )

    # Model results table
    if isinstance(pred, dict):
        lr_v  = pred.get('lr', {}).get('val_metrics', {})
        lr_t  = pred.get('lr', {}).get('test_metrics', {})
        gbt_v = pred.get('gbt', {}).get('val_metrics', {})
        gbt_t = pred.get('gbt', {}).get('test_metrics', {})

        section('Model Performance', 'Validation and test metrics')
        rows_metrics = [
            {'Model': 'Logistic Regression', 'Split': 'Validation',
             'PR-AUC': fmt(lr_v.get('pr_auc', np.nan), 4),
             'ROC-AUC': fmt(lr_v.get('roc_auc', np.nan), 4),
             'Precision': fmt(lr_v.get('precision', np.nan), 4),
             'Recall': fmt(lr_v.get('recall', np.nan), 4),
             'F1': fmt(lr_v.get('f1', np.nan), 4)},
            {'Model': 'Logistic Regression', 'Split': 'Test (limited events)',
             'PR-AUC': fmt(lr_t.get('pr_auc', np.nan), 4),
             'ROC-AUC': fmt(lr_t.get('roc_auc', np.nan), 4),
             'Precision': fmt(lr_t.get('precision', np.nan), 4),
             'Recall': fmt(lr_t.get('recall', np.nan), 4),
             'F1': fmt(lr_t.get('f1', np.nan), 4)},
            {'Model': 'HistGradientBoosting', 'Split': 'Validation',
             'PR-AUC': fmt(gbt_v.get('pr_auc', np.nan), 4),
             'ROC-AUC': fmt(gbt_v.get('roc_auc', np.nan), 4),
             'Precision': fmt(gbt_v.get('precision', np.nan), 4),
             'Recall': fmt(gbt_v.get('recall', np.nan), 4),
             'F1': fmt(gbt_v.get('f1', np.nan), 4)},
            {'Model': 'HistGradientBoosting', 'Split': 'Test (limited events)',
             'PR-AUC': fmt(gbt_t.get('pr_auc', np.nan), 4),
             'ROC-AUC': fmt(gbt_t.get('roc_auc', np.nan), 4),
             'Precision': fmt(gbt_t.get('precision', np.nan), 4),
             'Recall': fmt(gbt_t.get('recall', np.nan), 4),
             'F1': fmt(gbt_t.get('f1', np.nan), 4)},
        ]
        html_table(pd.DataFrame(rows_metrics))

        # Dataset split sizes
        if 'train_size' in pred:
            section('Dataset Split Sizes', 'Chronological train / val / test split')
            kpi_grid([
                ('Train Size',    f"{pred.get('train_size', '—'):,}",  '', f"Positive rate: {pred.get('train_pos_rate', 0)*100:.2f}%",  C['cyan']),
                ('Val Size',      f"{pred.get('val_size', '—'):,}",    '', f"Positive rate: {pred.get('val_pos_rate', 0)*100:.2f}%",    C['blue']),
                ('Test Size',     f"{pred.get('test_size', '—'):,}",   '', f"Positive rate: {pred.get('test_pos_rate', 0)*100:.2f}%",   C['purple']),
            ], cols=3)

        # Confusion matrix images
        cm_lr  = PROCESSED_DIR / 'figures' / 'cm_lr.png'
        cm_gbt = PROCESSED_DIR / 'figures' / 'cm_gbt.png'
        if cm_lr.exists() or cm_gbt.exists():
            section('Confusion Matrices', 'Test set results')
            cm1, cm2 = st.columns(2)
            if cm_lr.exists():
                with cm1:
                    st.image(str(cm_lr), caption='Logistic Regression — Test Set', width="stretch")
            if cm_gbt.exists():
                with cm2:
                    st.image(str(cm_gbt), caption='HistGradientBoosting — Test Set', width="stretch")

        # PR curve
        pr_curve = PROCESSED_DIR / 'figures' / 'pr_curve.png'
        if pr_curve.exists():
            section('Precision–Recall Curve', 'Validation set')
            st.image(str(pr_curve), width="stretch")

    # SHAP Explainability
    section('Model Explainability — SHAP Feature Importance', 'Top features influencing the HistGradientBoosting model')
    if isinstance(shap_imp, list) and shap_imp:
        shap_df = pd.DataFrame(shap_imp)
        feat_col = shap_df.columns[0]
        val_col  = shap_df.columns[-1]
        shap_top = shap_df[shap_df[val_col] > 0].sort_values(val_col, ascending=True).tail(15)

        fig_shap = go.Figure(go.Bar(
            x=shap_top[val_col],
            y=shap_top[feat_col],
            orientation='h',
            marker_color=C['cyan'],
            text=[f'{v:.4f}' for v in shap_top[val_col]],
            textposition='outside',
        ))
        fig_shap.update_layout(showlegend=False)
        st.plotly_chart(pc(fig_shap, height=460, title='Top 15 Features — Mean Absolute SHAP Value', xtitle='Mean |SHAP|'), width="stretch", config={'displayModeBar': False})

        st.markdown(
            f"<div class='insight'><b>Top feature:</b> {shap_top.iloc[-1][feat_col]} "
            f"(mean |SHAP| = {shap_top.iloc[-1][val_col]:.4f}). "
            f"This is the rolling 30-minute maximum of H1 pressure — the cyclonic separator discharge signal. "
            f"Short-term pressure extremes are the strongest discriminative signal.</div>"
            f"<div class='insight warn'><b>Important:</b> SHAP values reflect feature contributions within this model. "
            f"They do not establish physical causation or replace engineering expertise.</div>",
            unsafe_allow_html=True,
        )

        # Pre-generated SHAP figure
        shap_fig_path = PROCESSED_DIR / 'figures' / 'feature_importance_shap.png'
        if shap_fig_path.exists():
            with st.expander('View pre-generated SHAP chart'):
                st.image(str(shap_fig_path), width="stretch")

    if isinstance(pred, dict) and 'limitation' in pred:
        st.markdown(
            f"<div class='insight danger'><b>Documented limitation:</b> {pred['limitation']}</div>",
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — MAINTENANCE INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    section('Maintenance Insights', 'Decision-support summary — not automated diagnosis')

    st.markdown(
        f"<div class='insight'><b>Scope:</b> These insights are analytical decision-support outputs based on sensor data, "
        f"anomaly model scores, and documented failure events. They are not automated maintenance commands or equipment diagnosis. "
        f"Qualified engineering review is required before any action.</div>",
        unsafe_allow_html=True,
    )

    # Current state panel
    kpi_grid([
        ('Current Model State',  status,                  '', f'Anomaly score: {fmt(score,4)}',          status_color),
        ('Risk Probability',     f'{fmt(risk*100 if not np.isnan(risk) else np.nan,1)}', '%', 'GBT secondary model output', C['purple']),
        ('Last Data Point',      str(latest)[:16] if latest else '—', '', 'Most recent 1-min record',   C['muted']),
        ('Documented Events',    '4',                     '', 'All type: Air Leak — High Stress',        C['red']),
    ], cols=4)

    # What needs attention
    section('Signals Requiring Attention', 'Based on latest window values vs documented failure signatures')
    attention_items = []
    if len(ng):
        h1_latest = float(ng['H1'].iloc[-1]) if 'H1' in ng else np.nan
        tp2_latest = float(ng['TP2'].iloc[-1]) if 'TP2' in ng else np.nan
        lps_pct = float(ng['LPS'].mean() * 100) if 'LPS' in ng else np.nan
        h1_fail_mean = 0.10  # from failure event summary (approx across events)

        if not np.isnan(h1_latest) and h1_latest < 2.0:
            attention_items.append(('danger', 'H1 pressure is very low', f'Latest H1 = {h1_latest:.3f} bar. During documented failures H1 dropped near 0 bar. This pattern warrants investigation.'))
        elif not np.isnan(h1_latest) and h1_latest < 6.0:
            attention_items.append(('warn', 'H1 pressure is below typical operating range', f'Latest H1 = {h1_latest:.3f} bar. Normal operating range is approximately 8–10 bar.'))
        else:
            attention_items.append(('good', 'H1 pressure appears normal', f'Latest H1 = {h1_latest:.3f} bar — within documented normal operating range.'))

        if not np.isnan(lps_pct) and lps_pct > 1.0:
            attention_items.append(('danger', 'LPS activity is elevated', f'{lps_pct:.2f}% of selected window. LPS activates below 7 bar. Elevated LPS rate was observed in failure F4 (32.96%).'))
        else:
            attention_items.append(('good', 'LPS activity is normal', f'{lps_pct:.2f}% — low-pressure switch rarely active, consistent with normal pressure levels.'))

        if status == 'INVESTIGATE':
            attention_items.append(('danger', 'Anomaly model: INVESTIGATE state', f'Score {fmt(score,4)} ≥ investigate threshold {fmt(hi_thr,3)}. This score level was observed in all four documented failure windows.'))
        elif status == 'MONITOR':
            attention_items.append(('warn', 'Anomaly model: MONITOR state', f'Score {fmt(score,4)} — above monitor threshold. Continued observation recommended.'))
        else:
            attention_items.append(('good', 'Anomaly model: NORMAL state', f'Score {fmt(score,4)} — within baseline distribution.'))

    for cls, title_txt, body_txt in attention_items:
        st.markdown(f"<div class='insight {cls}'><b>{title_txt}</b><br>{body_txt}</div>", unsafe_allow_html=True)

    # Recommended inspection sequence
    section('Recommended Inspection Sequence', 'Model-evidence → engineering context')
    kpi_grid([
        ('Step 01', 'Pressure Signature',   '', 'Review H1 and TP2 time series around any flagged intervals.',            C['cyan']),
        ('Step 02', 'Compressor Load',       '', 'Check motor current, load fraction, and LPS activation rate.',           C['blue']),
        ('Step 03', 'Failure Event Match',   '', 'Compare sensor behaviour against documented failure F1–F4 signatures.',  C['amber']),
        ('Step 04', 'Human Engineering Review', '', 'Model output is decision-support evidence only, not diagnosis.',      C['lime']),
    ], cols=4)

    # Recent sensor behaviour
    section('Recent Sensor Behaviour', 'Last 6 hours — H1 and Motor Current')
    recent_6h = ng.loc[ng.index >= ng.index.max() - pd.Timedelta(hours=6)] if len(ng) else ng
    rc1, rc2 = st.columns(2)
    for (sc, color, col_ctx) in [('H1', C['cyan'], rc1), ('Motor_current', C['purple'], rc2)]:
        if sc not in recent_6h or len(recent_6h) == 0:
            continue
        pp = resamp(recent_6h, sc, '15min')
        fig_rec = go.Figure()
        if len(pp):
            fig_rec.add_trace(go.Scatter(x=pp.index, y=pp[sc], mode='lines',
                line=dict(color=color, width=2), name=sc))
        with col_ctx:
            st.plotly_chart(pc(fig_rec, height=260, title=f'{sc} — last 6h', ytitle=UNITS.get(sc,'')), width="stretch", config={'displayModeBar': False})

    # Pipeline method note
    with st.expander('Analytics Pipeline & Data Provenance'):
        steps = [
            ('1. Raw Data',       'UCI MetroPT-3 dataset — real operational sensor data from metro APU compressor (1,516,948 rows, ~7 months).'),
            ('2. Quality Check',  'Timestamp validation, gap detection (331 gaps >50s), sensor range checks, and failure event alignment.'),
            ('3. Processing',     'Resampled to 1-minute records. Gap rows flagged with is_gap. Digital signals preserved.'),
            ('4. Feature Eng.',   'Rolling statistics (30-min, 2h, 6h) for key sensors. Load fraction, pressure differentials.'),
            ('5. Anomaly Model',  'Isolation Forest trained on clean Feb–Mar 2020 baseline. Thresholds at 95th/99th percentile.'),
            ('6. Supervised',     'Logistic Regression and HistGradientBoosting on chronological split with 24h pre-failure labelling window.'),
            ('7. Explainability', 'SHAP values computed on GBT model to identify top contributing features.'),
            ('8. This Dashboard', 'All visualisations use computed values from the processed parquet and artifact files. No values are fabricated.'),
        ]
        for title_s, desc_s in steps:
            st.markdown(f"<div class='insight'><b>{title_s}</b><br>{desc_s}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='insight'><b>Dataset DOI:</b> 10.24432/C5VW3R · "
            f"<a href='https://archive.ics.uci.edu/dataset/791/metropt%2B3%2Bdataset' "
            f"target='_blank' style='color:{C['cyan']}'>UCI MetroPT-3 Dataset →</a>"
            f" · License: CC BY 4.0</div>",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"<div class='rg-footer'>"
    f"<span>RAILGUARD · EQUIPMENT HEALTH ANALYTICS</span>"
    f"<span>IBM SKILLSBUILD DATA ANALYTICS PROJECT · MetroPT-3 APU Compressor</span>"
    f"<span>MODEL OUTPUT IS DECISION SUPPORT · HUMAN ENGINEERING REVIEW REQUIRED</span>"
    f"<span>THEME: {theme_choice.upper()}</span>"
    f"</div>",
    unsafe_allow_html=True,
)

