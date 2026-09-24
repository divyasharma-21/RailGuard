"""
src/analysis/eda.py — Exploratory Data Analysis for RailGuard.

Produces a set of reproducible, publication-quality figures saved to
data/processed/figures/eda/.

Analyses:
  1. Sensor distributions (histograms + KDE)
  2. Time-series overview of all analogue sensors
  3. Operational state characterisation (motor states, load fraction)
  4. Sensor correlation heatmap
  5. Temporal patterns: hour-of-day and day-of-week aggregations
  6. Failure-period overlays on sensor time series (for key sensors)
  7. Normal vs failure state distributions
  8. TP3 vs Reservoirs scatter (should be ~1:1 in healthy operation)

All findings are based on the processed 1-minute DataFrame. No raw-data
statistics are hardcoded — every plot and statistic is derived at runtime.

Usage:
    python -m src.analysis.eda
    from src.analysis.eda import run_eda
    run_eda(df)
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    ANALOGUE_SENSORS,
    DIGITAL_SENSORS,
    FAILURE_EVENTS,
    PROCESSED_DIR,
)

logger = logging.getLogger(__name__)

FIG_DIR = PROCESSED_DIR / "figures" / "eda"

# Consistent colour palette
_PALETTE = sns.color_palette("tab10")
_FAILURE_COLOUR = "#d62728"   # red
_NORMAL_COLOUR  = "#1f77b4"   # blue
_STYLE = "seaborn-v0_8-whitegrid"


def _save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path


def _failure_spans(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Shade documented failure windows on an axis."""
    for event in FAILURE_EVENTS:
        start = pd.Timestamp(event["start"])
        end   = pd.Timestamp(event["end"])
        ax.axvspan(start, end, color=_FAILURE_COLOUR, alpha=0.18, zorder=0)


# ── 1. Sensor Distributions ──────────────────────────────────────────────────

def plot_sensor_distributions(df: pd.DataFrame) -> Path:
    """
    Histogram + KDE for every analogue sensor.
    Normal (no failure) rows shown in blue; failure rows in red.
    """
    normal  = df[df["failure_id"] == ""]
    failure = df[df["failure_id"] != ""]

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(4, 2, figsize=(13, 16))
        axes = axes.flatten()

        for i, col in enumerate(ANALOGUE_SENSORS):
            ax = axes[i]
            normal_vals  = normal[col].dropna()
            failure_vals = failure[col].dropna()

            ax.hist(normal_vals,  bins=80, density=True, alpha=0.55,
                    color=_NORMAL_COLOUR,  label=f"Normal (n={len(normal_vals):,})")
            ax.hist(failure_vals, bins=40, density=True, alpha=0.70,
                    color=_FAILURE_COLOUR, label=f"Failure (n={len(failure_vals):,})")

            unit_map = {
                "TP2": "bar", "TP3": "bar", "H1": "bar",
                "DV_pressure": "bar", "Reservoirs": "bar",
                "Oil_temperature": "°C", "Motor_current": "A",
            }
            ax.set_xlabel(f"{col} ({unit_map.get(col, '')})", fontsize=10)
            ax.set_ylabel("Density", fontsize=9)
            ax.set_title(col, fontsize=11, fontweight="bold")
            ax.legend(fontsize=8)

        # Hide unused subplot
        for j in range(len(ANALOGUE_SENSORS), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            "Analogue Sensor Distributions — Normal vs Failure (1-min buckets)",
            fontsize=13, fontweight="bold", y=1.01,
        )
        fig.tight_layout()
    return _save(fig, "01_sensor_distributions")


# ── 2. Time-Series Overview ──────────────────────────────────────────────────

def plot_sensor_timeseries(df: pd.DataFrame) -> Path:
    """
    Full time-series of all analogue sensors with failure windows shaded.
    Uses a subset (non-gap rows only) for clarity.
    """
    plot_df = df[~df["is_gap"]].copy()

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(len(ANALOGUE_SENSORS), 1,
                                 figsize=(16, 2.8 * len(ANALOGUE_SENSORS)),
                                 sharex=True)

        unit_map = {
            "TP2": "bar", "TP3": "bar", "H1": "bar",
            "DV_pressure": "bar", "Reservoirs": "bar",
            "Oil_temperature": "°C", "Motor_current": "A",
        }

        for ax, col in zip(axes, ANALOGUE_SENSORS):
            ax.plot(plot_df.index, plot_df[col], linewidth=0.5,
                    color=_NORMAL_COLOUR, alpha=0.8)
            _failure_spans(ax, plot_df)
            ax.set_ylabel(f"{col}\n({unit_map.get(col, '')})", fontsize=9)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

        axes[0].set_title(
            "Analogue Sensor Time Series — Feb to Sep 2020\n(red bands = documented failure windows)",
            fontsize=12, fontweight="bold",
        )
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")

        fig.tight_layout()
    return _save(fig, "02_sensor_timeseries")


# ── 3. Operational State Characterisation ────────────────────────────────────

def plot_operational_states(df: pd.DataFrame) -> Path:
    """
    (a) Distribution of motor states as a bar chart.
    (b) Load fraction distribution over time (rolling 24h mean).
    (c) Digital signal activation rates.
    """
    non_gap = df[~df["is_gap"]]

    with plt.style.context(_STYLE):
        fig = plt.figure(figsize=(15, 11))
        gs  = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

        # (a) Motor state distribution
        ax_ms = fig.add_subplot(gs[0, 0])
        state_counts = non_gap["motor_state"].value_counts()
        bars = ax_ms.bar(state_counts.index, state_counts.values,
                         color=_PALETTE[:len(state_counts)])
        ax_ms.set_title("Motor State Distribution", fontweight="bold")
        ax_ms.set_xlabel("State")
        ax_ms.set_ylabel("Count (1-min rows)")
        for bar in bars:
            ax_ms.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{bar.get_height()/len(non_gap)*100:.1f}%",
                ha="center", va="bottom", fontsize=9,
            )

        # (b) Load fraction rolling 24h
        ax_lf = fig.add_subplot(gs[0, 1])
        roll_lf = non_gap["load_fraction"].rolling("24h").mean()
        ax_lf.plot(roll_lf.index, roll_lf.values,
                   linewidth=0.8, color=_PALETTE[1])
        _failure_spans(ax_lf, non_gap)
        ax_lf.set_title("24h Rolling Load Fraction (DV_eletric active rate)",
                         fontweight="bold")
        ax_lf.set_xlabel("Date")
        ax_lf.set_ylabel("Fraction (0–1)")
        ax_lf.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax_lf.xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
        plt.setp(ax_lf.xaxis.get_majorticklabels(), rotation=30, ha="right")

        # (c) Digital signal activation rates (bar)
        ax_dig = fig.add_subplot(gs[1, :])
        dig_rates = {
            col: 100 * non_gap[col].mean()
            for col in DIGITAL_SENSORS
        }
        ax_dig.bar(dig_rates.keys(), dig_rates.values(),
                   color=_PALETTE[:len(dig_rates)])
        ax_dig.set_title("Digital Signal Activation Rates (% time active)",
                          fontweight="bold")
        ax_dig.set_ylabel("% Active")
        ax_dig.set_ylim(0, 110)
        for i, (col, v) in enumerate(dig_rates.items()):
            ax_dig.text(i, v + 1.5, f"{v:.1f}%", ha="center",
                        va="bottom", fontsize=9)

        fig.suptitle("Operational State Characterisation", fontsize=13,
                     fontweight="bold", y=1.01)
    return _save(fig, "03_operational_states")


# ── 4. Correlation Heatmap ───────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    """Pearson correlation matrix for all analogue sensors (non-gap rows)."""
    non_gap = df[~df["is_gap"]][ANALOGUE_SENSORS].dropna()

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(9, 7))
        corr = non_gap.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # show lower triangle
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(
            "Analogue Sensor Correlation Matrix\n(Pearson, 1-min resolution, non-gap rows)",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout()
    return _save(fig, "04_correlation_heatmap")


# ── 5. Temporal Patterns ─────────────────────────────────────────────────────

def plot_temporal_patterns(df: pd.DataFrame) -> Path:
    """
    Hour-of-day and day-of-week aggregations for load fraction and
    Oil_temperature to expose operational schedule effects.
    """
    non_gap = df[(df["failure_id"] == "") & (~df["is_gap"])].copy()
    non_gap["hour"]    = non_gap.index.hour
    non_gap["weekday"] = non_gap.index.dayofweek

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        # Load fraction by hour
        axes[0, 0].plot(
            non_gap.groupby("hour")["load_fraction"].mean().index,
            non_gap.groupby("hour")["load_fraction"].mean().values,
            marker="o", color=_PALETTE[0],
        )
        axes[0, 0].set_title("Avg Load Fraction by Hour of Day\n(normal periods only)")
        axes[0, 0].set_xlabel("Hour")
        axes[0, 0].set_ylabel("Mean load fraction")
        axes[0, 0].set_xticks(range(24))

        # Oil temperature by hour
        ot_hour = non_gap.groupby("hour")["Oil_temperature"].mean()
        axes[0, 1].plot(ot_hour.index, ot_hour.values, marker="o", color=_PALETTE[1])
        axes[0, 1].set_title("Avg Oil Temperature by Hour of Day\n(normal periods only)")
        axes[0, 1].set_xlabel("Hour")
        axes[0, 1].set_ylabel("Oil Temperature (°C)")
        axes[0, 1].set_xticks(range(24))

        # Load fraction by day of week
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        lf_dow = non_gap.groupby("weekday")["load_fraction"].mean()
        axes[1, 0].bar(
            [day_labels[d] for d in lf_dow.index], lf_dow.values,
            color=_PALETTE[2],
        )
        axes[1, 0].set_title("Avg Load Fraction by Day of Week\n(normal periods only)")
        axes[1, 0].set_ylabel("Mean load fraction")

        # Motor current by hour
        mc_hour = non_gap.groupby("hour")["Motor_current"].mean()
        axes[1, 1].plot(mc_hour.index, mc_hour.values, marker="o", color=_PALETTE[3])
        axes[1, 1].set_title("Avg Motor Current by Hour of Day\n(normal periods only)")
        axes[1, 1].set_xlabel("Hour")
        axes[1, 1].set_ylabel("Motor Current (A)")
        axes[1, 1].set_xticks(range(24))

        fig.suptitle("Temporal Operational Patterns", fontsize=13,
                     fontweight="bold", y=1.01)
        fig.tight_layout()
    return _save(fig, "05_temporal_patterns")


# ── 6. Normal vs Failure — Key Sensor Boxplots ───────────────────────────────

def plot_normal_vs_failure_boxplots(df: pd.DataFrame) -> Path:
    """
    Side-by-side boxplots of key analogue sensors comparing normal,
    pre-failure (24h before each event), and failure rows.
    """
    df2 = df[~df["is_gap"]].copy()

    # Assign period labels
    df2["period"] = "Normal"
    for event in FAILURE_EVENTS:
        fs = pd.Timestamp(event["start"])
        fe = pd.Timestamp(event["end"])
        pre_start = fs - pd.Timedelta(hours=24)
        df2.loc[(df2.index >= pre_start) & (df2.index < fs), "period"] = "Pre-failure (24h)"
        df2.loc[(df2.index >= fs) & (df2.index <= fe), "period"] = f"Failure ({event['id']})"

    # Normalise failure labels for grouped comparison
    df2["period_group"] = df2["period"].apply(
        lambda x: "Failure" if x.startswith("Failure") else x
    )

    order = ["Normal", "Pre-failure (24h)", "Failure"]
    palette = {
        "Normal":             _NORMAL_COLOUR,
        "Pre-failure (24h)":  "#ff7f0e",
        "Failure":            _FAILURE_COLOUR,
    }

    key_sensors = ["H1", "TP2", "Oil_temperature", "Motor_current", "DV_pressure", "TP3"]

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes = axes.flatten()

        unit_map = {
            "H1": "bar", "TP2": "bar", "TP3": "bar",
            "DV_pressure": "bar", "Oil_temperature": "°C", "Motor_current": "A",
        }

        for i, col in enumerate(key_sensors):
            ax = axes[i]
            plot_data = df2[["period_group", col]].dropna()
            sns.boxplot(
                data=plot_data,
                x="period_group",
                y=col,
                hue="period_group",
                order=order,
                palette=palette,
                legend=False,
                ax=ax,
                width=0.5,
                flierprops={"marker": ".", "markersize": 3, "alpha": 0.3},
            )
            ax.set_title(col, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel(f"{col} ({unit_map.get(col, '')})")

        fig.suptitle(
            "Key Sensor Values — Normal vs Pre-failure vs Failure\n"
            "(pre-failure = 24h before each documented event)",
            fontsize=12, fontweight="bold", y=1.01,
        )
        fig.tight_layout()
    return _save(fig, "06_normal_vs_failure_boxplots")


# ── 7. TP3 vs Reservoirs Scatter ─────────────────────────────────────────────

def plot_tp3_reservoirs(df: pd.DataFrame) -> Path:
    """
    Scatter of TP3 vs Reservoirs coloured by failure/normal.
    They should be near-identical in healthy operation (1:1 line).
    Divergence may indicate a problem.
    """
    non_gap = df[~df["is_gap"]].copy()
    normal  = non_gap[non_gap["failure_id"] == ""]
    failure = non_gap[non_gap["failure_id"] != ""]

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 7))

        ax.scatter(normal["TP3"], normal["Reservoirs"],
                   s=1, alpha=0.15, color=_NORMAL_COLOUR, label="Normal", rasterized=True)
        ax.scatter(failure["TP3"], failure["Reservoirs"],
                   s=4, alpha=0.6, color=_FAILURE_COLOUR, label="Failure", rasterized=True)

        # 1:1 reference line
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="1:1 line")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_xlabel("TP3 — Pneumatic panel pressure (bar)", fontsize=11)
        ax.set_ylabel("Reservoirs — Downstream pressure (bar)", fontsize=11)
        ax.set_title(
            "TP3 vs Reservoirs Pressure\n(should follow 1:1 in healthy operation)",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=10, markerscale=5)
        fig.tight_layout()
    return _save(fig, "07_tp3_vs_reservoirs")


# ── 8. LPS and Oil_level event timeline ──────────────────────────────────────

def plot_digital_event_timeline(df: pd.DataFrame) -> Path:
    """
    Plot digital signal event timelines for LPS, Oil_level, Caudal_impulses,
    and DV_eletric over the full observation period.
    """
    non_gap = df[~df["is_gap"]]
    signals = ["LPS", "Oil_level", "Caudal_impulses", "DV_eletric", "COMP"]

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(len(signals), 1, figsize=(16, 9), sharex=True)

        for ax, col in zip(axes, signals):
            # Plot as a filled area
            ax.fill_between(
                non_gap.index, non_gap[col],
                alpha=0.7,
                color=_PALETTE[signals.index(col) % len(_PALETTE)],
            )
            _failure_spans(ax, non_gap)
            ax.set_ylabel(col, fontsize=9)
            ax.set_yticks([0, 1])
            ax.set_ylim(-0.05, 1.1)

        axes[0].set_title(
            "Digital Signal Timeline — Feb to Sep 2020\n(red bands = documented failure windows)",
            fontsize=12, fontweight="bold",
        )
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")
        fig.tight_layout()
    return _save(fig, "08_digital_event_timeline")


# ── Main orchestrator ────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame | None = None) -> dict[str, Path]:
    """
    Run the full EDA suite and return a dict mapping analysis name → figure path.

    Parameters
    ----------
    df: Processed 1-minute DataFrame. If None, loads from data/processed/.
    """
    if df is None:
        from src.data.loader import load_parquet
        df = load_parquet(PROCESSED_DIR / "processed_1min.parquet")

    logger.info("Running EDA on %d rows …", len(df))

    figures = {}
    figures["sensor_distributions"]    = plot_sensor_distributions(df)
    figures["sensor_timeseries"]        = plot_sensor_timeseries(df)
    figures["operational_states"]       = plot_operational_states(df)
    figures["correlation_heatmap"]      = plot_correlation_heatmap(df)
    figures["temporal_patterns"]        = plot_temporal_patterns(df)
    figures["normal_vs_failure"]        = plot_normal_vs_failure_boxplots(df)
    figures["tp3_vs_reservoirs"]        = plot_tp3_reservoirs(df)
    figures["digital_event_timeline"]   = plot_digital_event_timeline(df)

    # Compute and save a summary statistics table
    non_gap = df[~df["is_gap"]]
    stats_path = PROCESSED_DIR / "eda_sensor_stats.csv"
    non_gap[ANALOGUE_SENSORS].describe().round(4).to_csv(stats_path)
    logger.info("Sensor statistics saved to %s", stats_path)

    logger.info("EDA complete. %d figures saved to %s", len(figures), FIG_DIR)
    return figures


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_eda()
