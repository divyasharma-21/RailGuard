"""
src/analysis/failure_analysis.py — Per-failure-event deep analysis.

For each of the four documented Air Leak — High Stress events (F1–F4),
this module:
  1. Extracts sensor data for three windows:
       - pre_window:    72h before failure start
       - failure:       documented failure window
       - post_window:   48h after failure end
  2. Computes per-window summary statistics (measured from data, not hardcoded)
  3. Analyses LPS activity and other key digital signals
  4. Plots aligned per-event sensor panels
  5. Generates a comparable cross-event summary table
  6. Saves all outputs to data/processed/figures/failure_analysis/

Key design choices:
  - All statistics are computed from the processed 1-min DataFrame passed in.
  - No statistical conclusion is pre-assumed; results are always printed from
    the measured data so the analysis is reproducible.
  - Pre-failure window is set to 72h to capture slow thermal/pressure drift
    that may precede a leak event.

Usage:
    python -m src.analysis.failure_analysis
    from src.analysis.failure_analysis import run_failure_analysis
    summary = run_failure_analysis(df)
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from src.config import (
    ANALOGUE_SENSORS,
    DIGITAL_SENSORS,
    FAILURE_EVENTS,
    PROCESSED_DIR,
)

logger = logging.getLogger(__name__)

FIG_DIR = PROCESSED_DIR / "figures" / "failure_analysis"

_STYLE          = "seaborn-v0_8-whitegrid"
_COLOUR_PRE     = "#2196F3"   # blue
_COLOUR_FAILURE = "#F44336"   # red
_COLOUR_POST    = "#4CAF50"   # green
_COLOUR_NORMAL  = "#9E9E9E"   # grey


# Window lengths (hours) — chosen to capture pre-failure drift and recovery
PRE_WINDOW_H  = 72
POST_WINDOW_H = 48

# Sensors to feature prominently in per-event plots (confirmed key indicators)
KEY_ANALOGUE  = ["H1", "TP2", "Oil_temperature", "Motor_current", "DV_pressure", "Reservoirs"]
KEY_DIGITAL   = ["DV_eletric", "COMP", "LPS", "Caudal_impulses"]


def _save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path


def _extract_windows(
    df: pd.DataFrame, event: dict
) -> dict[str, pd.DataFrame]:
    """
    Extract pre / failure / post DataFrames for one event.

    Returns dict with keys 'pre', 'failure', 'post'.
    Rows where is_gap=True are excluded from statistics but retained
    in plot data so gaps are visible.
    """
    fs = pd.Timestamp(event["start"])
    fe = pd.Timestamp(event["end"])

    pre_start  = fs - pd.Timedelta(hours=PRE_WINDOW_H)
    post_end   = fe + pd.Timedelta(hours=POST_WINDOW_H)

    windows = {
        "pre":     df[(df.index >= pre_start) & (df.index < fs)],
        "failure": df[(df.index >= fs) & (df.index <= fe)],
        "post":    df[(df.index > fe) & (df.index <= post_end)],
    }
    return windows


def _window_stats(
    win: pd.DataFrame, label: str
) -> dict:
    """Compute summary statistics for a single window DataFrame."""
    non_gap = win[~win["is_gap"]] if "is_gap" in win.columns else win
    stats: dict = {
        "label":    label,
        "n_rows":   len(non_gap),
        "duration_h": round(
            (win.index[-1] - win.index[0]).total_seconds() / 3600, 2
        ) if len(win) > 0 else 0,
    }

    for col in KEY_ANALOGUE:
        if col in non_gap.columns:
            s = non_gap[col].dropna()
            stats[f"{col}_mean"] = float(round(s.mean(), 4)) if len(s) else None
            stats[f"{col}_std"]  = float(round(s.std(), 4))  if len(s) else None
            stats[f"{col}_min"]  = float(round(s.min(), 4))  if len(s) else None
            stats[f"{col}_max"]  = float(round(s.max(), 4))  if len(s) else None

    for col in KEY_DIGITAL:
        if col in non_gap.columns:
            n = len(non_gap)
            pct = 100 * non_gap[col].sum() / n if n > 0 else 0
            stats[f"{col}_pct_active"] = round(float(pct), 2)

    return stats


def _lps_event_summary(win: pd.DataFrame) -> dict:
    """
    Count and characterise LPS=1 runs (consecutive active periods) within a window.

    Returns a dict with count, total_duration_s, max_duration_s,
    and a list of (start, end, duration_s) tuples.
    """
    if "LPS" not in win.columns or len(win) == 0:
        return {"n_events": 0, "events": []}

    s = win["LPS"]
    # Identify transitions
    transitions = s.diff().fillna(0)
    starts = win.index[transitions == 1].tolist()
    ends   = win.index[transitions == -1].tolist()

    # Handle edge cases: LPS=1 at window start or end
    if s.iloc[0] == 1:
        starts = [win.index[0]] + starts
    if s.iloc[-1] == 1:
        ends = ends + [win.index[-1]]

    events = []
    for start, end in zip(starts, ends):
        dur = (end - start).total_seconds()
        events.append({"start": str(start), "end": str(end), "duration_s": dur})

    total_s = sum(e["duration_s"] for e in events)
    max_s   = max((e["duration_s"] for e in events), default=0)

    return {
        "n_events":        len(events),
        "total_active_s":  total_s,
        "max_duration_s":  max_s,
        "events":          events,
    }


def plot_event_panel(
    df: pd.DataFrame,
    event: dict,
    windows: dict[str, pd.DataFrame],
) -> Path:
    """
    Create a detailed multi-panel plot for one failure event.

    Subplots:
      Row 0: H1 and TP2 (pressure indicators)
      Row 1: Oil_temperature and Motor_current
      Row 2: DV_pressure and Reservoirs
      Row 3: Digital signals (DV_eletric, COMP, LPS, Caudal_impulses)

    The pre/failure/post windows are shaded differently.
    The full 72h pre + failure + 48h post context is shown.
    """
    fs = pd.Timestamp(event["start"])
    fe = pd.Timestamp(event["end"])
    pre_start = fs - pd.Timedelta(hours=PRE_WINDOW_H)
    post_end  = fe + pd.Timedelta(hours=POST_WINDOW_H)

    # Full window for plotting
    plot_df = df[(df.index >= pre_start) & (df.index <= post_end) & ~df["is_gap"]]

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True)

        analogue_pairs = [
            ("H1", "TP2"),
            ("Oil_temperature", "Motor_current"),
            ("DV_pressure", "Reservoirs"),
        ]
        unit_map = {
            "H1": "bar", "TP2": "bar", "TP3": "bar",
            "DV_pressure": "bar", "Reservoirs": "bar",
            "Oil_temperature": "°C", "Motor_current": "A",
        }

        for row, (col_a, col_b) in enumerate(analogue_pairs):
            for ax, col in zip(axes[row], [col_a, col_b]):
                ax.plot(plot_df.index, plot_df[col], lw=0.8,
                        color="#1f77b4", alpha=0.9)
                # Shade failure window
                ax.axvspan(fs, fe, color=_COLOUR_FAILURE, alpha=0.2, zorder=0,
                           label="Failure window")
                ax.set_ylabel(f"{col}\n({unit_map.get(col, '')})", fontsize=9)
                ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        # Row 3: Digital signals stacked
        digital_colours = {
            "DV_eletric":      _COLOUR_FAILURE,
            "COMP":            "#2196F3",
            "LPS":             "#FF9800",
            "Caudal_impulses": "#9C27B0",
        }
        for j, (col, colour) in enumerate(digital_colours.items()):
            ax = axes[3, 0] if j < 2 else axes[3, 1]
            other_col = list(digital_colours.keys())[j + 1 if j % 2 == 0 else j - 1]
            if j % 2 == 0:
                ax.cla()  # clear before first signal in this subplot

        # Plot each digital signal in its own panel (axes[3,0] and axes[3,1])
        # Left: DV_eletric and COMP
        for ax_dig, (col_a, col_b) in [
            (axes[3, 0], ("DV_eletric", "COMP")),
            (axes[3, 1], ("LPS",        "Caudal_impulses")),
        ]:
            ax_dig.fill_between(plot_df.index, plot_df[col_a],
                                alpha=0.6, color=digital_colours[col_a],
                                label=col_a)
            ax_dig.fill_between(plot_df.index, plot_df[col_b],
                                alpha=0.4, color=digital_colours[col_b],
                                label=col_b)
            ax_dig.axvspan(fs, fe, color=_COLOUR_FAILURE, alpha=0.15, zorder=0)
            ax_dig.set_ylim(-0.05, 1.15)
            ax_dig.set_yticks([0, 1])
            ax_dig.legend(fontsize=8, loc="upper left")
            ax_dig.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
            ax_dig.xaxis.set_major_locator(mdates.HourLocator(interval=12))
            plt.setp(ax_dig.xaxis.get_majorticklabels(), rotation=30, ha="right")

        fid   = event["id"]
        ftype = event["type"]
        maint = event.get("maintenance", "")
        fig.suptitle(
            f"Failure Event {fid} — {ftype} ({event['severity']})\n"
            f"Window: {fs} → {fe}  |  Maintenance: {maint}\n"
            f"(showing {PRE_WINDOW_H}h pre, failure period, {POST_WINDOW_H}h post)",
            fontsize=11, fontweight="bold", y=1.01,
        )
        fig.tight_layout()

    return _save(fig, f"failure_event_{fid.lower()}")


def plot_cross_event_comparison(all_stats: list[dict]) -> Path:
    """
    Bar chart comparing mean values of key sensors across all four failure events.
    Shows pre / during / post side-by-side for H1, TP2, Oil_temperature, Motor_current.
    """
    compare_cols = [
        "H1_mean", "TP2_mean", "Oil_temperature_mean", "Motor_current_mean",
        "DV_eletric_pct_active", "LPS_pct_active",
    ]
    labels = {
        "H1_mean": "H1 mean (bar)",
        "TP2_mean": "TP2 mean (bar)",
        "Oil_temperature_mean": "Oil Temp mean (°C)",
        "Motor_current_mean": "Motor Current mean (A)",
        "DV_eletric_pct_active": "DV_eletric active (%)",
        "LPS_pct_active": "LPS active (%)",
    }

    # Pivot: rows = failure event × period, cols = metrics
    rows = []
    for s in all_stats:
        rows.append(s)
    stat_df = pd.DataFrame(rows).set_index(["failure_id", "period"])

    n_cols = len(compare_cols)
    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = axes.flatten()

        events = ["F1", "F2", "F3", "F4"]
        period_order  = ["pre", "failure", "post"]
        period_colours = {
            "pre":     _COLOUR_PRE,
            "failure": _COLOUR_FAILURE,
            "post":    _COLOUR_POST,
        }

        for ax, metric in zip(axes, compare_cols):
            x = np.arange(len(events))
            width = 0.25
            for i, period in enumerate(period_order):
                vals = []
                for fid in events:
                    try:
                        v = stat_df.loc[(fid, period), metric]
                        vals.append(float(v) if v is not None else 0)
                    except KeyError:
                        vals.append(0)
                ax.bar(
                    x + (i - 1) * width,
                    vals,
                    width=width,
                    label=period.capitalize(),
                    color=period_colours[period],
                    alpha=0.85,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(events)
            ax.set_title(labels.get(metric, metric), fontweight="bold", fontsize=10)
            ax.set_xlabel("Failure event")
            ax.legend(fontsize=8)

        fig.suptitle(
            "Cross-Event Comparison — Key Sensor Stats (Pre / During / Post)\n"
            "All values computed from processed 1-min data",
            fontsize=12, fontweight="bold", y=1.01,
        )
        fig.tight_layout()
    return _save(fig, "cross_event_comparison")


def run_failure_analysis(
    df: pd.DataFrame | None = None,
) -> list[dict]:
    """
    Run the full failure analysis pipeline.

    Parameters
    ----------
    df: Processed 1-minute DataFrame. If None, loads from data/processed/.

    Returns
    -------
    List of per-window stat dicts for all events (pre/failure/post).
    """
    if df is None:
        from src.data.loader import load_parquet
        df = load_parquet(PROCESSED_DIR / "processed_1min.parquet")

    logger.info("Running failure analysis on %d rows …", len(df))

    all_stats: list[dict] = []

    for event in FAILURE_EVENTS:
        fid = event["id"]
        logger.info("Analysing failure event %s …", fid)

        windows = _extract_windows(df, event)

        for period, win_df in windows.items():
            stats = _window_stats(win_df, label=period)
            stats["failure_id"] = fid
            stats["period"] = period
            if period == "failure":
                stats["lps_analysis"] = _lps_event_summary(win_df)
            all_stats.append(stats)
            logger.info(
                "  %s / %s: %d rows, %s h",
                fid, period, stats["n_rows"], stats["duration_h"],
            )

        # Per-event plot
        plot_event_panel(df, event, windows)

    # Cross-event comparison plot
    plot_cross_event_comparison(all_stats)

    # Save summary table (excluding nested lps_analysis dicts)
    flat_stats = [
        {k: v for k, v in s.items() if k != "lps_analysis"}
        for s in all_stats
    ]
    summary_df = pd.DataFrame(flat_stats)
    summary_path = PROCESSED_DIR / "failure_event_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Failure event summary saved to %s", summary_path)

    _print_failure_summary(all_stats)
    return all_stats


def _print_failure_summary(all_stats: list[dict]) -> None:
    """Print a concise cross-event summary to stdout."""
    print("\n" + "=" * 70)
    print("  RailGuard — Failure Event Analysis Summary")
    print("=" * 70)
    print(f"  {'Event':<6} {'Period':<12} {'H1 mean':>9} {'TP2 mean':>9} "
          f"{'OilT mean':>10} {'Curr mean':>10} {'DV_el%':>7} {'LPS%':>6}")
    print("  " + "-" * 66)
    for s in all_stats:
        fid    = s.get("failure_id", "?")
        period = s.get("label", "?")
        h1     = s.get("H1_mean");     h1_s     = f"{h1:.2f}"     if h1     is not None else "  N/A"
        tp2    = s.get("TP2_mean");    tp2_s    = f"{tp2:.2f}"    if tp2    is not None else "  N/A"
        ot     = s.get("Oil_temperature_mean"); ot_s = f"{ot:.2f}" if ot is not None else "  N/A"
        mc     = s.get("Motor_current_mean");   mc_s = f"{mc:.2f}" if mc is not None else "  N/A"
        dve    = s.get("DV_eletric_pct_active", 0)
        lps    = s.get("LPS_pct_active", 0)
        print(f"  {fid:<6} {period:<12} {h1_s:>9} {tp2_s:>9} {ot_s:>10} {mc_s:>10} "
              f"{dve:>6.1f}% {lps:>5.1f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_failure_analysis()
