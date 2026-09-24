"""
src/dashboard/app.py — RailGuard Predictive Maintenance Dashboard

Industrial-grade Streamlit application for the MetroPT-3 air compressor
predictive maintenance system. All data is loaded from pre-computed
artifacts — no model training occurs at dashboard startup.

Sections:
  A. Executive Overview
  B. Sensor Analytics
  C. Anomaly Monitoring
  D. Failure Analysis
  E. Predictive Risk (supervised model — with stated limitations)
  F. Maintenance Insights
  G. About / Methodology

Run:
    streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

# ── Path setup (works whether launched from root or src/) ─────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RailGuard — Predictive Maintenance",
    page_icon="🛤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
from src.config import ARTIFACTS_DIR, FAILURE_EVENTS, PROCESSED_DIR

PROC_PARQUET   = PROCESSED_DIR / "processed_1min.parquet"
ANOMALY_SCORES = ARTIFACTS_DIR / "anomaly_scores.parquet"
RISK_SCORES    = ARTIFACTS_DIR / "risk_scores.parquet"
SCORE_STATS    = ARTIFACTS_DIR / "anomaly_score_stats.json"
PRED_RESULTS   = ARTIFACTS_DIR / "predictive_results.json"
SHAP_IMP       = ARTIFACTS_DIR / "shap_feature_importance.json"
BASELINE_STATS = ARTIFACTS_DIR / "baseline_stats.json"

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = {
    "normal":      "#2A9D8F",
    "monitor":     "#E9C46A",
    "investigate": "#E63946",
    "primary":     "#1B3A5C",
    "accent":      "#457B9D",
    "muted":       "#6B7280",
    "bg":          "#F8FAFC",
    "surface":     "#FFFFFF",
    "border":      "#E2E8F0",
}

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {PALETTE['bg']}; }}
    .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}
    h1 {{ color: {PALETTE['primary']}; font-weight: 700; }}
    h2 {{ color: {PALETTE['primary']}; font-weight: 600; }}
    h3 {{ color: {PALETTE['primary']}; font-weight: 600; }}
    .metric-card {{
        background: {PALETTE['surface']};
        border: 1px solid {PALETTE['border']};
        border-radius: 8px;
        padding: 1rem 1.2rem;
        text-align: center;
    }}
    .status-normal   {{ color: {PALETTE['normal']};      font-weight: 700; font-size: 1.1rem; }}
    .status-monitor  {{ color: {PALETTE['monitor']};     font-weight: 700; font-size: 1.1rem; }}
    .status-investigate {{ color: {PALETTE['investigate']}; font-weight: 700; font-size: 1.1rem; }}
    .disclaimer {{
        background: #FEF9C3; border-left: 4px solid #CA8A04;
        padding: 0.75rem 1rem; border-radius: 4px; font-size: 0.85rem;
        color: #78350F;
    }}
    .finding-box {{
        background: {PALETTE['surface']}; border: 1px solid {PALETTE['border']};
        border-radius: 6px; padding: 0.8rem 1rem; margin: 0.4rem 0;
        font-size: 0.9rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading sensor data…")
def load_processed() -> pd.DataFrame:
    return pd.read_parquet(PROC_PARQUET)


@st.cache_data(show_spinner="Loading anomaly scores…")
def load_anomaly_scores() -> pd.Series:
    if not ANOMALY_SCORES.exists():
        return pd.Series(dtype="float32", name="anomaly_score")
    return pd.read_parquet(ANOMALY_SCORES)["anomaly_score"]


@st.cache_data(show_spinner="Loading risk scores…")
def load_risk_scores() -> pd.Series:
    if not RISK_SCORES.exists():
        return pd.Series(dtype="float32", name="risk_prob")
    return pd.read_parquet(RISK_SCORES)["risk_prob"]


@st.cache_data
def load_score_stats() -> dict:
    if not SCORE_STATS.exists():
        return {}
    with open(SCORE_STATS) as f:
        return json.load(f)


@st.cache_data
def load_pred_results() -> dict:
    if not PRED_RESULTS.exists():
        return {}
    with open(PRED_RESULTS) as f:
        return json.load(f)


@st.cache_data
def load_shap_importance() -> list:
    if not SHAP_IMP.exists():
        return []
    with open(SHAP_IMP) as f:
        return json.load(f)


@st.cache_data
def load_baseline_stats() -> dict:
    if not BASELINE_STATS.exists():
        return {}
    with open(BASELINE_STATS) as f:
        return json.load(f)


@st.cache_data
def get_combined_df() -> pd.DataFrame:
    """Merge processed data with anomaly/risk scores."""
    df = load_processed()
    a_scores = load_anomaly_scores()
    r_scores = load_risk_scores()
    if len(a_scores):
        df = df.join(a_scores, how="left")
    if len(r_scores):
        df = df.join(r_scores, how="left")
    return df


@st.cache_data
def get_decision_categories() -> pd.Series:
    """Derive decision categories for the full dataset."""
    stats = load_score_stats()
    low_t  = stats.get("alert_threshold_low",  0.45)
    high_t = stats.get("alert_threshold_high", 0.55)

    a_scores = load_anomaly_scores()
    r_scores = load_risk_scores()

    pred_results = load_pred_results()
    risk_t = pred_results.get("gbt", {}).get("threshold", 0.5)

    def _cat(a: float, r: float) -> str:
        if pd.isna(a):
            return "Normal"
        if a >= high_t or (not pd.isna(r) and r >= risk_t):
            return "Investigate"
        elif a >= low_t:
            return "Monitor"
        return "Normal"

    if len(a_scores) == 0:
        return pd.Series(dtype=str)

    idx = a_scores.index
    r_aligned = r_scores.reindex(idx) if len(r_scores) else pd.Series(0.0, index=idx)
    cats = pd.Series(
        [_cat(float(a), float(r)) for a, r in zip(a_scores.values, r_aligned.values)],
        index=idx,
    )
    return cats


# ─────────────────────────────────────────────────────────────────────────────
# Reusable components
# ─────────────────────────────────────────────────────────────────────────────

def kpi_card(col, label: str, value: str, delta: str = "", status: str = "normal"):
    colour_map = {"normal": PALETTE["normal"], "monitor": PALETTE["monitor"],
                  "investigate": PALETTE["investigate"], "muted": PALETTE["muted"]}
    colour = colour_map.get(status, PALETTE["accent"])
    with col:
        st.markdown(
            f"""<div class="metric-card">
            <div style="font-size:0.75rem;color:{PALETTE['muted']};text-transform:uppercase;
                        letter-spacing:0.05em">{label}</div>
            <div style="font-size:1.8rem;font-weight:700;color:{colour};line-height:1.3">{value}</div>
            {"<div style='font-size:0.8rem;color:" + PALETTE['muted'] + "'>" + delta + "</div>" if delta else ""}
            </div>""",
            unsafe_allow_html=True,
        )


def failure_event_table() -> pd.DataFrame:
    rows = []
    for ev in FAILURE_EVENTS:
        rows.append({
            "ID": ev["id"],
            "Start": ev["start"],
            "End": ev["end"],
            "Type": ev["type"],
            "Severity": ev["severity"],
            "Maintenance": ev["maintenance"],
        })
    return pd.DataFrame(rows)


def _plotly_sensor_timeseries(
    df: pd.DataFrame,
    sensor: str,
    title: str = "",
    height: int = 300,
    mark_failures: bool = True,
) -> go.Figure:
    non_gap = df[~df["is_gap"]] if "is_gap" in df.columns else df
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=non_gap.index, y=non_gap[sensor],
        mode="lines", line=dict(width=1.2, color=PALETTE["accent"]),
        name=sensor,
    ))
    if mark_failures and "failure_id" in df.columns:
        for ev in FAILURE_EVENTS:
            fmask = (non_gap["failure_id"] == ev["id"]) if "failure_id" in non_gap.columns else None
            if fmask is not None and fmask.any():
                fig.add_vrect(
                    x0=ev["start"], x1=ev["end"],
                    fillcolor=PALETTE["investigate"], opacity=0.2,
                    annotation_text=ev["id"], annotation_position="top left",
                    line_width=0,
                )
    fig.update_layout(
        title=title, height=height,
        margin=dict(l=50, r=20, t=40, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        showlegend=False,
    )
    return fig


SENSOR_UNITS = {
    "TP2": "bar", "TP3": "bar", "H1": "bar",
    "DV_pressure": "bar", "Reservoirs": "bar",
    "Oil_temperature": "°C", "Motor_current": "A",
    "load_fraction": "fraction (0–1)",
    "anomaly_score": "score (0–1)",
    "risk_prob": "probability (0–1)",
}

ANALOGUE_SENSORS = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs",
                    "Oil_temperature", "Motor_current"]
DIGITAL_SENSORS  = ["COMP", "DV_eletric", "Towers", "MPG", "LPS",
                    "Pressure_switch", "Oil_level", "Caudal_impulses"]


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.markdown(
    f"<h2 style='color:{PALETTE['primary']};margin-bottom:0.2rem'>🛤️ RailGuard</h2>"
    f"<div style='font-size:0.8rem;color:{PALETTE['muted']}'>Predictive Maintenance System</div>"
    "<hr style='margin:0.5rem 0'>",
    unsafe_allow_html=True,
)

SECTIONS = [
    "A — Executive Overview",
    "B — Sensor Analytics",
    "C — Anomaly Monitoring",
    "D — Failure Analysis",
    "E — Predictive Risk",
    "F — Maintenance Insights",
    "G — About & Methodology",
]
section = st.sidebar.radio("Navigate", SECTIONS, label_visibility="collapsed")

# Global date filter
st.sidebar.markdown("---")
st.sidebar.markdown("**Date range filter**")
try:
    df_meta = load_processed()
    date_min = df_meta.index.min().date()
    date_max = df_meta.index.max().date()
except Exception:
    import datetime
    date_min = datetime.date(2020, 2, 1)
    date_max = datetime.date(2020, 9, 1)

date_start, date_end = st.sidebar.date_input(
    "Select period",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max,
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<div style='font-size:0.75rem;color:{PALETTE['muted']}'>"
    "MetroPT-3 Dataset · Porto Metro · 2020<br>"
    "All model outputs are decision support only.</div>",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: apply date filter
# ─────────────────────────────────────────────────────────────────────────────

def apply_date_filter(df: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp(date_start)
    end   = pd.Timestamp(date_end) + pd.Timedelta(days=1)
    return df[(df.index >= start) & (df.index < end)]


# ─────────────────────────────────────────────────────────────────────────────
# A — Executive Overview
# ─────────────────────────────────────────────────────────────────────────────

if section == "A — Executive Overview":
    st.title("Equipment Health Overview")
    st.markdown(
        "Real-time equipment health summary for the Porto Metro air compressor system. "
        "Anomaly scores and risk indicators are derived from models trained on historical operating data."
    )

    df_full = get_combined_df()
    categories = get_decision_categories()
    score_stats = load_score_stats()

    df_filt  = apply_date_filter(df_full)
    cats_filt = categories.reindex(df_filt.index)

    non_gap  = df_filt[~df_filt["is_gap"]] if "is_gap" in df_filt.columns else df_filt

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    # Latest anomaly score
    latest_score = float(non_gap["anomaly_score"].dropna().iloc[-1]) if "anomaly_score" in non_gap.columns and len(non_gap["anomaly_score"].dropna()) else float("nan")
    latest_cat   = categories.loc[non_gap.index[-1]] if len(non_gap) and len(categories) else "Normal"
    low_t  = score_stats.get("alert_threshold_low",  0.835)
    high_t = score_stats.get("alert_threshold_high", 0.988)
    status_map = {"Normal": "normal", "Monitor": "monitor", "Investigate": "investigate"}

    kpi_card(c1, "Current Status",
             latest_cat if not pd.isna(latest_score) else "No data",
             f"Score: {latest_score:.3f}" if not pd.isna(latest_score) else "",
             status=status_map.get(latest_cat, "normal"))

    # Failure events in period
    n_failures_in_period = int((non_gap.get("failure_id", pd.Series(dtype=str)) != "").sum()) if "failure_id" in non_gap.columns else 0
    kpi_card(c2, "Failure Rows in Period", f"{n_failures_in_period:,}",
             f"{100*n_failures_in_period/max(len(non_gap),1):.1f}% of non-gap rows",
             status="investigate" if n_failures_in_period > 0 else "normal")

    # Investigate events count
    n_invest = int((cats_filt == "Investigate").sum())
    kpi_card(c3, "Investigate Flags",
             f"{n_invest:,}",
             f"{100*n_invest/max(len(cats_filt),1):.1f}% of period",
             status="investigate" if n_invest > 0 else "normal")

    # Mean oil temperature
    mean_ot = float(non_gap["Oil_temperature"].mean()) if "Oil_temperature" in non_gap.columns else float("nan")
    kpi_card(c4, "Avg Oil Temp", f"{mean_ot:.1f} °C" if not pd.isna(mean_ot) else "N/A",
             "Normal range: 55–70 °C", status="normal")

    # Load fraction
    mean_lf = float(non_gap["load_fraction"].mean()) if "load_fraction" in non_gap.columns else float("nan")
    kpi_card(c5, "Avg Load Fraction", f"{mean_lf:.1%}" if not pd.isna(mean_lf) else "N/A",
             "Normal: ~15–25%",
             status="investigate" if not pd.isna(mean_lf) and mean_lf > 0.7 else "normal")

    st.markdown("")

    # ── Anomaly score timeline ─────────────────────────────────────────────────
    st.subheader("Anomaly Score Timeline")
    st.markdown(
        "The anomaly score (0–1) represents deviation from the Feb–Mar 2020 normal baseline. "
        f"Thresholds: **Monitor** > {low_t:.3f} · **Investigate** > {high_t:.3f} "
        "(derived from 95th/99th percentile of training scores, not manually set)."
    )

    if "anomaly_score" in df_filt.columns and len(df_filt["anomaly_score"].dropna()):
        df_plot = df_filt[~df_filt["is_gap"]].resample("1H").agg(
            anomaly_score=("anomaly_score", "mean")
        ) if len(df_filt) > 5000 else df_filt[~df_filt["is_gap"]]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["anomaly_score"],
            mode="lines", line=dict(width=1, color="#94A3B8"), name="Anomaly score",
            fill="tozeroy", fillcolor="rgba(148,163,184,0.15)",
        ))
        # Threshold lines
        fig.add_hline(y=low_t,  line_dash="dot", line_color=PALETTE["monitor"],
                      annotation_text="Monitor threshold", annotation_position="bottom right")
        fig.add_hline(y=high_t, line_dash="dot", line_color=PALETTE["investigate"],
                      annotation_text="Investigate threshold", annotation_position="top right")
        # Failure rectangles
        for ev in FAILURE_EVENTS:
            fig.add_vrect(x0=ev["start"], x1=ev["end"],
                          fillcolor=PALETTE["investigate"], opacity=0.15,
                          annotation_text=ev["id"], annotation_position="top left",
                          line_width=0)
        fig.update_layout(
            height=350, margin=dict(l=50, r=20, t=40, b=40),
            yaxis_range=[0, 1.05], plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="Anomaly score"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Anomaly scores not yet computed. Run `python scripts/train_models.py` first.")

    # ── Category distribution ──────────────────────────────────────────────────
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("Decision Category Distribution")
        if len(cats_filt):
            cat_counts = cats_filt.value_counts()
            fig_pie = px.pie(
                names=cat_counts.index,
                values=cat_counts.values,
                color=cat_counts.index,
                color_discrete_map={
                    "Normal": PALETTE["normal"],
                    "Monitor": PALETTE["monitor"],
                    "Investigate": PALETTE["investigate"],
                },
                hole=0.4,
            )
            fig_pie.update_layout(height=280, margin=dict(l=0, r=0, t=20, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.subheader("Failure Events")
        ev_df = failure_event_table()
        st.dataframe(ev_df[["ID", "Start", "End", "Type"]], use_container_width=True, hide_index=True)

    # ── Quick sensor summary ──────────────────────────────────────────────────
    st.subheader("Sensor Summary (Selected Period)")
    summary_rows = []
    for sensor in ANALOGUE_SENSORS:
        if sensor in non_gap.columns:
            s = non_gap[sensor].dropna()
            summary_rows.append({
                "Sensor": sensor,
                "Unit": SENSOR_UNITS.get(sensor, ""),
                "Mean": f"{s.mean():.3f}",
                "Std":  f"{s.std():.3f}",
                "Min":  f"{s.min():.3f}",
                "Max":  f"{s.max():.3f}",
            })
    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# B — Sensor Analytics
# ─────────────────────────────────────────────────────────────────────────────

elif section == "B — Sensor Analytics":
    st.title("Sensor Analytics")

    df_full = apply_date_filter(get_combined_df())
    non_gap = df_full[~df_full["is_gap"]] if "is_gap" in df_full.columns else df_full

    col_left, col_right = st.columns([1, 3])

    with col_left:
        st.subheader("Configuration")
        selected_sensor = st.selectbox(
            "Primary sensor",
            ANALOGUE_SENSORS,
            index=ANALOGUE_SENSORS.index("H1"),
        )
        resample_opt = st.selectbox("Time resolution", ["1 min (raw)", "10 min", "1 hour"], index=1)
        show_state = st.checkbox("Shade failure windows", value=True)
        show_dist  = st.checkbox("Show distribution", value=True)
        show_corr  = st.checkbox("Show correlation heatmap", value=False)

    resample_map = {"1 min (raw)": None, "10 min": "10min", "1 hour": "1H"}
    res = resample_map[resample_opt]

    with col_right:
        st.subheader(f"{selected_sensor} — Time Series")
        unit = SENSOR_UNITS.get(selected_sensor, "")

        plot_df = non_gap.resample(res).mean() if res else non_gap

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df[selected_sensor],
            mode="lines", line=dict(width=1.2, color=PALETTE["accent"]),
            name=selected_sensor,
        ))
        if show_state and "failure_id" in non_gap.columns:
            for ev in FAILURE_EVENTS:
                fig.add_vrect(
                    x0=ev["start"], x1=ev["end"],
                    fillcolor=PALETTE["investigate"], opacity=0.2,
                    annotation_text=ev["id"], annotation_position="top left",
                    line_width=0,
                )
        fig.update_layout(
            height=300, margin=dict(l=50, r=20, t=30, b=40),
            yaxis_title=f"{selected_sensor} ({unit})",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        )
        st.plotly_chart(fig, use_container_width=True)

        if show_dist:
            st.subheader("Distribution — Normal vs Failure")
            normal_s  = non_gap.loc[non_gap["failure_id"] == "", selected_sensor].dropna() if "failure_id" in non_gap.columns else non_gap[selected_sensor].dropna()
            failure_s = non_gap.loc[non_gap["failure_id"] != "", selected_sensor].dropna() if "failure_id" in non_gap.columns else pd.Series(dtype="float32")

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=normal_s, name="Normal",
                marker_color=PALETTE["normal"], opacity=0.65,
                histnorm="probability density", nbinsx=80,
            ))
            if len(failure_s):
                fig_dist.add_trace(go.Histogram(
                    x=failure_s, name="Failure",
                    marker_color=PALETTE["investigate"], opacity=0.65,
                    histnorm="probability density", nbinsx=40,
                ))
            fig_dist.update_layout(
                barmode="overlay", height=260,
                margin=dict(l=50, r=20, t=30, b=40),
                xaxis_title=f"{selected_sensor} ({unit})",
                yaxis_title="Density",
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", y=1.02),
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        if show_corr:
            st.subheader("Correlation Matrix (Non-gap Rows)")
            corr = non_gap[ANALOGUE_SENSORS].corr()
            fig_corr = px.imshow(
                corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1, aspect="auto",
            )
            fig_corr.update_layout(height=400, margin=dict(l=60, r=20, t=30, b=30))
            st.plotly_chart(fig_corr, use_container_width=True)

    # Operational state summary
    st.subheader("Operational State Summary")
    col1, col2 = st.columns(2)
    with col1:
        if "motor_state" in non_gap.columns:
            state_counts = non_gap["motor_state"].value_counts()
            fig_state = px.bar(
                x=state_counts.index.astype(str), y=state_counts.values,
                labels={"x": "Motor state", "y": "Rows (1-min)"},
                color=state_counts.index.astype(str),
                color_discrete_map={
                    "off": "#94A3B8", "offloaded": "#60A5FA",
                    "loaded": "#F59E0B", "starting": "#EF4444",
                },
                title="Motor State Distribution",
            )
            fig_state.update_layout(showlegend=False, height=280,
                                    plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig_state, use_container_width=True)
    with col2:
        if "load_fraction" in non_gap.columns:
            fig_lf = go.Figure(go.Histogram(
                x=non_gap["load_fraction"].dropna(),
                marker_color=PALETTE["accent"], opacity=0.75, nbinsx=50,
            ))
            fig_lf.update_layout(
                title="Load Fraction Distribution",
                xaxis_title="Load fraction", yaxis_title="Count",
                height=280, margin=dict(l=50, r=20, t=40, b=40),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig_lf, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# C — Anomaly Monitoring
# ─────────────────────────────────────────────────────────────────────────────

elif section == "C — Anomaly Monitoring":
    st.title("Anomaly Monitoring")

    st.markdown(
        """
        The anomaly model is an **Isolation Forest** trained on Feb–Mar 2020 confirmed-normal data.
        It detects sensor patterns that deviate from the learned normal baseline — without requiring failure labels.

        **Score interpretation:** 0 = maximally normal · 1 = maximally anomalous  
        Thresholds are data-derived (95th / 99th percentile of training distribution).
        """
    )

    df_full = get_combined_df()
    cats    = get_decision_categories()
    stats   = load_score_stats()

    df_filt  = apply_date_filter(df_full)
    non_gap  = df_filt[~df_filt["is_gap"]] if "is_gap" in df_filt.columns else df_filt
    cats_filt = cats.reindex(df_filt.index)

    low_t  = stats.get("alert_threshold_low",  0.835)
    high_t = stats.get("alert_threshold_high", 0.988)

    if "anomaly_score" not in non_gap.columns or non_gap["anomaly_score"].isna().all():
        st.warning("Anomaly scores not found. Run `python scripts/train_models.py` first.")
    else:
        # ── Score timeline ─────────────────────────────────────────────────────
        st.subheader("Anomaly Score Over Time")
        resample_c = st.selectbox("Resolution", ["1 min", "15 min", "1 hour"], index=1,
                                  key="anomaly_res")
        res_map = {"1 min": None, "15 min": "15min", "1 hour": "1H"}
        res = res_map[resample_c]
        plot_df = non_gap.resample(res).mean() if res else non_gap

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df["anomaly_score"],
            mode="lines", line=dict(width=1, color="#94A3B8"),
            fill="tozeroy", fillcolor="rgba(148,163,184,0.15)",
            name="Anomaly score",
        ))
        fig.add_hline(y=low_t, line_dash="dash", line_color=PALETTE["monitor"],
                      annotation_text=f"Monitor (p95={low_t:.3f})", annotation_position="bottom right")
        fig.add_hline(y=high_t, line_dash="dash", line_color=PALETTE["investigate"],
                      annotation_text=f"Investigate (p99={high_t:.3f})", annotation_position="top right")
        for ev in FAILURE_EVENTS:
            fig.add_vrect(x0=ev["start"], x1=ev["end"],
                          fillcolor=PALETTE["investigate"], opacity=0.18,
                          annotation_text=ev["id"], annotation_position="top left",
                          line_width=0)
        fig.update_layout(
            height=380, yaxis_range=[0, 1.05],
            margin=dict(l=50, r=20, t=30, b=40),
            yaxis_title="Anomaly score",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Anomaly event summary ──────────────────────────────────────────────
        st.subheader("Anomaly Events")
        col_t, col_s = st.columns([2, 1])
        with col_s:
            thresh_sel = st.slider(
                "Detection threshold", 0.4, 1.0, float(high_t), 0.01,
                help="Slide to adjust what counts as an anomaly event.",
            )
            min_dur = st.number_input("Min event duration (min)", 1, 120, 10)

        from src.models.anomaly import compute_anomaly_events
        a_scores_filt = df_filt["anomaly_score"].dropna()
        events_df = compute_anomaly_events(a_scores_filt, threshold=thresh_sel, min_duration_m=int(min_dur))

        with col_t:
            if len(events_df):
                st.dataframe(
                    events_df.assign(
                        start=events_df["start"].astype(str),
                        end=events_df["end"].astype(str),
                        max_score=events_df["max_score"].round(4),
                        mean_score=events_df["mean_score"].round(4),
                    ),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info(f"No anomaly events above threshold {thresh_sel:.3f} in selected period.")

        # ── Score distribution split ────────────────────────────────────────────
        st.subheader("Score Distribution: Normal vs Failure Windows")
        if "failure_id" in non_gap.columns:
            norm_scores  = non_gap.loc[non_gap["failure_id"] == "",  "anomaly_score"].dropna()
            fail_scores  = non_gap.loc[non_gap["failure_id"] != "",  "anomaly_score"].dropna()
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=norm_scores, name="Normal", histnorm="probability density",
                                            marker_color=PALETTE["normal"], opacity=0.6, nbinsx=80))
            if len(fail_scores):
                fig_dist.add_trace(go.Histogram(x=fail_scores, name="Failure", histnorm="probability density",
                                                marker_color=PALETTE["investigate"], opacity=0.7, nbinsx=40))
            fig_dist.add_vline(x=low_t, line_dash="dot", line_color=PALETTE["monitor"],
                               annotation_text="Monitor")
            fig_dist.add_vline(x=high_t, line_dash="dot", line_color=PALETTE["investigate"],
                               annotation_text="Investigate")
            fig_dist.update_layout(
                barmode="overlay", height=300,
                margin=dict(l=50, r=20, t=30, b=40),
                xaxis_title="Anomaly score", yaxis_title="Density",
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", y=1.02),
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            col_n, col_f = st.columns(2)
            col_n.metric("Normal rows mean score", f"{norm_scores.mean():.4f}")
            col_f.metric("Failure rows mean score", f"{fail_scores.mean():.4f}" if len(fail_scores) else "N/A")


# ─────────────────────────────────────────────────────────────────────────────
# D — Failure Analysis
# ─────────────────────────────────────────────────────────────────────────────

elif section == "D — Failure Analysis":
    st.title("Failure Event Analysis")
    st.markdown(
        "Deep analysis of the four documented air-leak events. "
        "Data is sourced from company maintenance records and confirmed in sensor data."
    )

    df_full = get_combined_df()

    # Event selector
    event_ids = [ev["id"] for ev in FAILURE_EVENTS]
    sel_event_id = st.selectbox("Select failure event", event_ids)
    sel_event = next(ev for ev in FAILURE_EVENTS if ev["id"] == sel_event_id)

    # Window extraction
    fs = pd.Timestamp(sel_event["start"])
    fe = pd.Timestamp(sel_event["end"])
    pre_start = fs - pd.Timedelta(hours=72)
    post_end  = fe + pd.Timedelta(hours=48)
    plot_df   = df_full[(df_full.index >= pre_start) & (df_full.index <= post_end)]
    non_gap_plot = plot_df[~plot_df["is_gap"]] if "is_gap" in plot_df.columns else plot_df

    # Event metadata
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Event", sel_event["id"])
    col_b.metric("Type", sel_event["type"])
    col_c.metric("Start", sel_event["start"][:16])
    col_d.metric("Maintenance", sel_event["maintenance"][:10])

    # ── Pre/during/post stats ──────────────────────────────────────────────────
    st.subheader("Sensor Statistics: Pre / During / Post")
    pre_df   = non_gap_plot[non_gap_plot.index < fs]
    fail_df  = non_gap_plot[(non_gap_plot.index >= fs) & (non_gap_plot.index <= fe)]
    post_df  = non_gap_plot[non_gap_plot.index > fe]

    stat_rows = []
    for sensor in ["H1", "TP2", "Oil_temperature", "Motor_current", "DV_pressure"]:
        if sensor in non_gap_plot.columns:
            for period_name, period_df in [("Pre (72h)", pre_df), ("Failure", fail_df), ("Post (48h)", post_df)]:
                s = period_df[sensor].dropna()
                if len(s):
                    stat_rows.append({
                        "Sensor": sensor,
                        "Period": period_name,
                        "Mean": f"{s.mean():.3f}",
                        "Std":  f"{s.std():.3f}",
                        "Min":  f"{s.min():.3f}",
                        "Max":  f"{s.max():.3f}",
                        "Unit": SENSOR_UNITS.get(sensor, ""),
                    })
    if stat_rows:
        st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

    # ── Multi-sensor panel ─────────────────────────────────────────────────────
    st.subheader("Sensor Panels (72h pre + failure + 48h post)")
    sensors_to_plot = st.multiselect(
        "Sensors to display",
        ANALOGUE_SENSORS,
        default=["H1", "TP2", "Oil_temperature", "Motor_current"],
        key="fail_sensors",
    )
    if sensors_to_plot:
        n = len(sensors_to_plot)
        fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.04)
        for i, sensor in enumerate(sensors_to_plot, 1):
            fig.add_trace(
                go.Scatter(
                    x=non_gap_plot.index, y=non_gap_plot[sensor],
                    mode="lines", line=dict(width=1, color=PALETTE["accent"]),
                    name=sensor,
                ),
                row=i, col=1,
            )
            fig.add_vrect(x0=str(fs), x1=str(fe),
                          fillcolor=PALETTE["investigate"], opacity=0.2,
                          line_width=0, row=i, col=1)
            fig.update_yaxes(title_text=f"{sensor}<br>({SENSOR_UNITS.get(sensor,'')})", row=i, col=1)
        fig.update_layout(
            height=150 * n + 60, showlegend=False,
            margin=dict(l=80, r=20, t=30, b=40),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── LPS analysis ───────────────────────────────────────────────────────────
    st.subheader("LPS (Low Pressure Switch) Activity")
    if "LPS" in non_gap_plot.columns:
        lps_pct_pre  = float(100 * pre_df["LPS"].mean())  if len(pre_df)  else 0.0
        lps_pct_fail = float(100 * fail_df["LPS"].mean()) if len(fail_df) else 0.0
        lps_pct_post = float(100 * post_df["LPS"].mean()) if len(post_df) else 0.0
        c1, c2, c3 = st.columns(3)
        c1.metric("LPS active %  (pre)",     f"{lps_pct_pre:.1f}%")
        c2.metric("LPS active %  (failure)", f"{lps_pct_fail:.1f}%")
        c3.metric("LPS active %  (post)",    f"{lps_pct_post:.1f}%")

    # ── Anomaly score around event ─────────────────────────────────────────────
    if "anomaly_score" in non_gap_plot.columns and not non_gap_plot["anomaly_score"].isna().all():
        st.subheader("Anomaly Score Around Event")
        stats   = load_score_stats()
        high_t  = stats.get("alert_threshold_high", 0.988)
        fig_as  = go.Figure()
        fig_as.add_trace(go.Scatter(
            x=non_gap_plot.index, y=non_gap_plot["anomaly_score"],
            mode="lines", line=dict(width=1.5, color="#94A3B8"), fill="tozeroy",
            fillcolor="rgba(148,163,184,0.15)", name="Anomaly score",
        ))
        fig_as.add_vrect(x0=str(fs), x1=str(fe),
                         fillcolor=PALETTE["investigate"], opacity=0.2, line_width=0)
        fig_as.add_hline(y=high_t, line_dash="dash", line_color=PALETTE["investigate"],
                         annotation_text="Investigate threshold")
        fig_as.update_layout(height=250, margin=dict(l=50, r=20, t=30, b=40),
                             yaxis_range=[0, 1.05], yaxis_title="Anomaly score",
                             plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_as, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# E — Predictive Risk
# ─────────────────────────────────────────────────────────────────────────────

elif section == "E — Predictive Risk":
    st.title("Predictive Risk Model")

    st.markdown(
        """<div class="disclaimer">
        ⚠️ <strong>Important limitation:</strong> Only 4 documented failure events exist in the dataset.
        The supervised models below were trained on these 4 events using strict chronological splits.
        With so few events, supervised model performance is <em>indicative only</em> — the models
        may overfit to specific characteristics of these 4 events and not generalise to novel failures.
        <br><br>
        <strong>The Isolation Forest anomaly model (section C) is the primary predictive component.</strong>
        The supervised models here are included for transparency and to show what is learnable from the available labels.
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("")

    pred_results = load_pred_results()

    if not pred_results:
        st.warning("Model results not found. Run `python scripts/train_models.py` first.")
    else:
        # ── Split info ──────────────────────────────────────────────────────────
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Train set", f"{pred_results.get('train_size',0):,} rows",
                     f"Pos: {pred_results.get('train_pos_rate',0):.1%}")
        col_b.metric("Validation set", f"{pred_results.get('val_size',0):,} rows",
                     f"Pos: {pred_results.get('val_pos_rate',0):.1%}")
        col_c.metric("Test set (F4)", f"{pred_results.get('test_size',0):,} rows",
                     f"Pos: {pred_results.get('test_pos_rate',0):.1%}")

        st.markdown(
            "**Train:** Feb–Apr 2020 (includes F1) · "
            "**Validate:** May–Jun 2020 (includes F2, F3) · "
            "**Test:** Jul–Aug 2020 (includes F4, held out)"
        )

        # ── Metrics table ────────────────────────────────────────────────────────
        st.subheader("Model Evaluation Metrics")
        st.markdown(
            "Primary metric: **Precision-Recall AUC (PR-AUC)**. "
            f"Random classifier baseline PR-AUC ≈ {pred_results.get('lr',{}).get('test_metrics',{}).get('base_rate',0.018):.4f} (class proportion)."
        )

        metric_rows = []
        for model_key, model_label in [("lr", "Logistic Regression"), ("gbt", "Gradient Boosting")]:
            for split_key, split_label in [("val_metrics", "Validation"), ("test_metrics", "Test")]:
                m = pred_results.get(model_key, {}).get(split_key, {})
                if m:
                    metric_rows.append({
                        "Model": model_label,
                        "Split": split_label,
                        "PR-AUC": f"{m.get('pr_auc',0):.4f}",
                        "Precision": f"{m.get('precision',0):.3f}",
                        "Recall": f"{m.get('recall',0):.3f}",
                        "F1": f"{m.get('f1',0):.3f}",
                        "Threshold": f"{m.get('threshold',0.5):.3f}",
                        "TP": m.get("tp", 0),
                        "FP": m.get("fp", 0),
                        "TN": m.get("tn", 0),
                        "FN": m.get("fn", 0),
                    })
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

        # ── PR curve ────────────────────────────────────────────────────────────
        pr_fig_path = PROCESSED_DIR / "figures" / "pr_curve.png"
        if pr_fig_path.exists():
            st.subheader("Precision-Recall Curve (Test Set)")
            st.image(str(pr_fig_path), use_container_width=True)

        # ── Confusion matrices ───────────────────────────────────────────────────
        cm_col1, cm_col2 = st.columns(2)
        for col, model_name, cm_path in [
            (cm_col1, "Logistic Regression", PROCESSED_DIR / "figures" / "cm_lr.png"),
            (cm_col2, "Gradient Boosting",   PROCESSED_DIR / "figures" / "cm_gbt.png"),
        ]:
            if cm_path.exists():
                col.subheader(model_name)
                col.image(str(cm_path), use_container_width=True)

        # ── Risk score timeline ──────────────────────────────────────────────────
        df_full = apply_date_filter(get_combined_df())
        non_gap = df_full[~df_full["is_gap"]] if "is_gap" in df_full.columns else df_full
        if "risk_prob" in non_gap.columns and not non_gap["risk_prob"].isna().all():
            st.subheader("GBT Risk Score Timeline")
            gbt_thresh = pred_results.get("gbt", {}).get("threshold", 0.5)
            plot_df = non_gap.resample("1H").mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=plot_df.index, y=plot_df["risk_prob"],
                mode="lines", line=dict(width=1, color=PALETTE["investigate"]),
                fill="tozeroy", fillcolor="rgba(230,57,70,0.1)", name="Risk probability",
            ))
            fig.add_hline(y=gbt_thresh, line_dash="dash", line_color=PALETTE["investigate"],
                          annotation_text=f"Operating threshold ({gbt_thresh:.3f})")
            for ev in FAILURE_EVENTS:
                fig.add_vrect(x0=ev["start"], x1=ev["end"],
                              fillcolor=PALETTE["investigate"], opacity=0.15, line_width=0,
                              annotation_text=ev["id"])
            fig.update_layout(
                height=300, yaxis_range=[0, 1.05],
                margin=dict(l=50, r=20, t=30, b=40),
                yaxis_title="Risk probability",
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Interpretation ────────────────────────────────────────────────────────
        st.subheader("Interpretation")
        st.markdown(
            """
            **What do these results mean?**

            - **GBT test PR-AUC > base rate**: The model learns *some* pattern in the data that distinguishes
              failure from normal. However, the improvement over a random classifier is modest, reflecting
              the very small number of failure events.

            - **Low recall on test set**: The model correctly flags very few positive rows (low recall).
              This is expected: F4 is a single event that the model has never seen, trained only on F1.
              The features learned from F1 partially generalise but not reliably.

            - **High anomaly model scores during failures**: The Isolation Forest anomaly model (section C)
              shows dramatically elevated scores during all 4 failure events — confirming that anomaly
              detection is more robust here than supervised learning with 4 events.

            **What this system does NOT do:**
            - It cannot predict the exact time of a future failure
            - It cannot detect failure types other than air leaks
            - The supervised model cannot be considered validated with 4 events
            """
        )


# ─────────────────────────────────────────────────────────────────────────────
# F — Maintenance Insights
# ─────────────────────────────────────────────────────────────────────────────

elif section == "F — Maintenance Insights":
    st.title("Maintenance Insights")

    st.markdown(
        """<div class="disclaimer">
        ⚠️ This section provides <strong>decision support only</strong> — not automated maintenance decisions.
        All findings must be reviewed by a qualified maintenance engineer before any action is taken.
        Thresholds are derived from the model's training data distribution, not from external engineering standards.
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("")

    df_full = get_combined_df()
    df_filt = apply_date_filter(df_full)
    non_gap = df_filt[~df_filt["is_gap"]] if "is_gap" in df_filt.columns else df_filt
    cats    = get_decision_categories().reindex(df_filt.index)
    stats   = load_score_stats()
    shap_imp = load_shap_importance()
    baseline = load_baseline_stats()

    low_t  = stats.get("alert_threshold_low",  0.835)
    high_t = stats.get("alert_threshold_high", 0.988)

    # ── Current status assessment ──────────────────────────────────────────────
    st.subheader("Current Equipment Status")

    latest_row = non_gap.iloc[-1] if len(non_gap) else None
    latest_score = float(non_gap["anomaly_score"].dropna().iloc[-1]) if (latest_row is not None and "anomaly_score" in non_gap.columns) else float("nan")
    latest_risk  = float(non_gap["risk_prob"].dropna().iloc[-1])     if (latest_row is not None and "risk_prob" in non_gap.columns) else 0.0
    latest_cat   = cats.iloc[-1] if len(cats) else "Normal"
    latest_ts    = non_gap.index[-1] if len(non_gap) else None

    cat_colour = {"Normal": PALETTE["normal"], "Monitor": PALETTE["monitor"], "Investigate": PALETTE["investigate"]}
    cat_icon   = {"Normal": "✅", "Monitor": "⚠️", "Investigate": "🔴"}

    st.markdown(
        f"""<div style="background:{cat_colour.get(latest_cat,'#ddd')};border-radius:8px;
            padding:1rem 1.5rem;color:white;font-size:1.1rem;margin-bottom:1rem">
        {cat_icon.get(latest_cat,'ℹ️')} &nbsp;
        <strong>Status: {latest_cat}</strong> &nbsp;·&nbsp;
        Anomaly score: {latest_score:.4f} &nbsp;·&nbsp;
        Risk probability: {latest_risk:.3f} &nbsp;·&nbsp;
        As of: {str(latest_ts)[:16] if latest_ts else 'N/A'}
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Evidence ──────────────────────────────────────────────────────────────
    if shap_imp and latest_row is not None and len(baseline):
        st.subheader("Contributing Sensor Signals")
        st.markdown("Features ranked by model importance (SHAP), with current value vs normal baseline.")

        evidence_rows = []
        for item in shap_imp[:15]:
            feat = item["feature"]
            if feat not in latest_row.index:
                continue
            val = float(latest_row.get(feat, float("nan")))
            if pd.isna(val):
                continue
            b = baseline.get(feat, {})
            b_mean = b.get("mean", float("nan"))
            b_std  = b.get("std",  float("nan"))
            if not pd.isna(b_mean) and not pd.isna(b_std) and b_std > 0:
                z = (val - b_mean) / b_std
                dev_pct = 100 * (val - b_mean) / (abs(b_mean) + 1e-9)
                status = "Elevated ↑" if z > 2 else "Depressed ↓" if z < -2 else "Normal"
            else:
                z, dev_pct, status = float("nan"), float("nan"), "N/A"
            evidence_rows.append({
                "Feature": feat.replace("_", " "),
                "Current": f"{val:.4f}",
                "Baseline mean": f"{b_mean:.4f}" if not pd.isna(b_mean) else "N/A",
                "Z-score": f"{z:.2f}" if not pd.isna(z) else "N/A",
                "Deviation %": f"{dev_pct:+.1f}%" if not pd.isna(dev_pct) else "N/A",
                "Status": status,
                "SHAP importance": f"{item['mean_abs_shap']:.5f}",
            })
            if len(evidence_rows) >= 10:
                break
        if evidence_rows:
            st.dataframe(pd.DataFrame(evidence_rows), use_container_width=True, hide_index=True)

    # ── Recent trend ─────────────────────────────────────────────────────────
    st.subheader("Recent Sensor Trends (last 6 hours)")
    last_6h = non_gap.last("6H") if len(non_gap) else pd.DataFrame()
    if len(last_6h):
        trend_sensors = ["H1", "load_fraction", "Oil_temperature", "Motor_current"]
        fig_t = make_subplots(rows=2, cols=2, shared_xaxes=False,
                              subplot_titles=trend_sensors)
        positions = [(1,1),(1,2),(2,1),(2,2)]
        for (r, c), sensor in zip(positions, trend_sensors):
            if sensor in last_6h.columns:
                fig_t.add_trace(
                    go.Scatter(x=last_6h.index, y=last_6h[sensor],
                               mode="lines", line=dict(width=2, color=PALETTE["accent"]),
                               name=sensor),
                    row=r, col=c,
                )
                # Add baseline mean line
                b_mean = baseline.get(sensor, {}).get("mean")
                if b_mean:
                    fig_t.add_hline(y=b_mean, line_dash="dot", line_color="#94A3B8",
                                    annotation_text=f"baseline {b_mean:.2f}", row=r, col=c)
        fig_t.update_layout(
            height=400, showlegend=False,
            margin=dict(l=50, r=20, t=50, b=40),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_t, use_container_width=True)

    # ── Feature importance ─────────────────────────────────────────────────────
    fi_path = PROCESSED_DIR / "figures" / "feature_importance_shap.png"
    if fi_path.exists():
        st.subheader("Feature Importance (SHAP — GBT Model, Test Set)")
        st.markdown(
            "Mean absolute SHAP values indicate which features the model used most "
            "when distinguishing failure from normal patterns. "
            "**This does not prove causation** — it shows model associations."
        )
        st.image(str(fi_path), use_container_width=True)
    elif shap_imp:
        st.subheader("Feature Importance (SHAP)")
        top_n_feats = pd.DataFrame(shap_imp[:20])
        fig_fi = px.bar(
            top_n_feats.sort_values("mean_abs_shap"),
            x="mean_abs_shap", y="feature",
            orientation="h", title="Mean |SHAP| value",
            color_discrete_sequence=[PALETTE["accent"]],
        )
        fig_fi.update_layout(height=500, yaxis_title="", xaxis_title="Mean |SHAP|",
                             plot_bgcolor="white", paper_bgcolor="white",
                             margin=dict(l=200, r=20, t=40, b=40))
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Guidance ──────────────────────────────────────────────────────────────
    st.subheader("Decision Guidance")
    st.markdown(
        f"""
        | Category | Anomaly Score | Recommended Action |
        |----------|--------------|-------------------|
        | ✅ Normal    | < {low_t:.3f} (95th pct) | No action required. Continue normal monitoring. |
        | ⚠️ Monitor  | {low_t:.3f}–{high_t:.3f} | Log the observation. Review sensor trends. Increase monitoring frequency. |
        | 🔴 Investigate | > {high_t:.3f} (99th pct) | A qualified engineer should inspect the equipment. Check H1, DV_eletric, and Oil_temperature sensors for the specific signatures described in the failure analysis. |

        **Note:** These thresholds represent the 95th and 99th percentile of the confirmed-normal training baseline
        (Feb–Mar 2020). At the 95th percentile threshold, approximately 5% of normal operating rows will
        be flagged as Monitor — this is a known false-positive rate, not an error.
        """
    )


# ─────────────────────────────────────────────────────────────────────────────
# G — About & Methodology
# ─────────────────────────────────────────────────────────────────────────────

elif section == "G — About & Methodology":
    st.title("About RailGuard")

    st.markdown(
        """
        ## Project Overview

        **RailGuard** is a predictive maintenance analytics system for the Porto Metro air compressor,
        built on the publicly available MetroPT-3 dataset. It demonstrates an end-to-end pipeline from
        raw sensor ingestion through anomaly detection, supervised risk modelling, explainability,
        and decision support.

        ---

        ## Dataset

        | Property | Value |
        |----------|-------|
        | Source | MetroPT-3 (UC Irvine ML Repository) |
        | Equipment | Porto Metro APU Air Compressor |
        | Period | Feb 2020 – Sep 2020 (213 days) |
        | Raw rows | 1,516,948 at ~10s effective resolution |
        | Processed rows | 306,960 at 1-minute resolution |
        | Analogue sensors | TP2, TP3, H1, DV_pressure, Reservoirs, Oil_temperature, Motor_current |
        | Digital signals | COMP, DV_eletric, Towers, MPG, LPS, Pressure_switch, Oil_level, Caudal_impulses |
        | Documented failures | 4 air-leak events (F1–F4) |

        ---

        ## Architecture

        ```
        Raw CSV (1.5M rows)
             ↓  Preprocessing (1-min aggregation, gap detection)
        Processed Parquet (307K rows, 32 cols)
             ↓  Feature Engineering (rolling 30min/2h/6h windows, gap-aware)
        Feature Matrix (307K × 86 features)
             ↓
         ┌──────────────────────────────────────────────────────┐
         │                                                      │
         │  Anomaly Detection (Isolation Forest)                │
         │  Trained on Feb–Mar 2020 clean normal baseline       │
         │  → Anomaly score [0,1] for each minute               │
         │                                                      │
         │  Supervised Models (LR + GBT)                        │
         │  Trained on F1, validated on F2/F3, tested on F4     │
         │  → Risk probability [0,1]  (indicative only)         │
         │                                                      │
         │  SHAP Explainability (GBT, test set)                 │
         │  → Feature importance ranking                        │
         └──────────────────────────────────────────────────────┘
             ↓
        Decision Support (Normal / Monitor / Investigate)
             ↓
        Streamlit Dashboard
        ```

        ---

        ## Anomaly Detection

        **Method:** Isolation Forest (sklearn.ensemble.IsolationForest)  
        **Training baseline:** Feb–Mar 2020 (73,638 rows of confirmed-normal operation)  
        **Rationale:** Unsupervised; does not require failure labels; effective on high-dimensional
        tabular data without distributional assumptions.

        The anomaly score is a sigmoid-calibrated transformation of the Isolation Forest
        `decision_function` output. Higher = more anomalous. Thresholds are derived from the
        95th (Monitor) and 99th (Investigate) percentiles of training-set scores — not manually set.

        **Performance on known failures:** All 4 failure events show anomaly scores > 0.98 during
        the documented failure windows. Pre-failure degradation is visible in F3 and F4 (scores
        begin rising several hours before failure onset).

        ---

        ## Supervised Models

        **Method:** Logistic Regression (baseline) and HistGradientBoostingClassifier  
        **Label strategy:**
        - Positive: documented failure windows + 24h pre-failure at-risk windows
        - Negative: all other non-gap, non-excluded rows
        - Class imbalance: ~2–4% positive → handled via class_weight='balanced' (LR)
          and sample_weight (GBT)

        **Train/validate/test split (strictly chronological, no shuffling):**
        - Train: Feb–Apr 2020 (F1 at end)
        - Validate: May–Jun 2020 (F2, F3)
        - Test: Jul–Aug 2020 (F4, held out)

        **Primary metric:** Precision-Recall AUC (informative under class imbalance)

        ---

        ## Limitations

        1. **4 labeled failure events is very few** for supervised learning. Results are
           indicative only and should not be used for production decisions without more data.

        2. **All failures are the same type** (air leak). The models are calibrated only
           for this failure mode. Other failure types are not represented.

        3. **Anomaly detection is the primary component.** The supervised models confirm
           the anomaly patterns but do not add reliable generalisation.

        4. **17.7% of 1-minute slots have no data** (maintenance shutdowns, system restarts).
           The largest gaps are ~5.76 hours.

        5. **Correlation ≠ causation.** SHAP values show which features the model uses.
           The physical mechanism (air leak → continuous load → downstream effects) is
           documented in the dataset description, but the model does not prove causation.

        6. **This is a decision support tool**, not an autonomous maintenance decision system.
           All outputs should be reviewed by a qualified engineer.

        ---

        ## Dataset Attribution

        Veloso, B., Ribeiro, J., Pereira, P., & Gama, J. (2022).  
        *MetroPT: A Benchmark Dataset for Predictive Maintenance.*  
        Applied Sciences, 12(12), 5897. https://doi.org/10.3390/app12125897

        Dataset available at the UCI Machine Learning Repository.
        """
    )

    # ── Technology stack ───────────────────────────────────────────────────────
    st.subheader("Technology Stack")
    tech_df = pd.DataFrame([
        {"Component": "Data processing", "Library": "pandas, numpy, pyarrow"},
        {"Component": "Anomaly detection", "Library": "scikit-learn (IsolationForest)"},
        {"Component": "Supervised models", "Library": "scikit-learn (LR, HistGBT)"},
        {"Component": "Explainability", "Library": "shap (TreeExplainer)"},
        {"Component": "Visualisation (static)", "Library": "matplotlib, seaborn"},
        {"Component": "Dashboard", "Library": "streamlit, plotly"},
        {"Component": "Configuration", "Library": "pyyaml"},
        {"Component": "Testing", "Library": "pytest"},
    ])
    st.dataframe(tech_df, use_container_width=True, hide_index=True)
