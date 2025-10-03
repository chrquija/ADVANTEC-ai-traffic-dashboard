# cycle_length_recommendations.py
# Grant-ready Cycle Length Calculator (ADVANTEC)
# -----------------------------------------------------------
# Adds Webster seeding + constraints, benefits, safety proxies,
# explainability, and a grant narrative download.
#
# Inputs required: 'local_datetime', 'total_volume'
# Optional inputs (if present): 'intersection_name', 'direction',
# 'average_traveltime', 'average_delay' (used in KPIs/benefits)
#
# Author: ADVANTEC dashboard

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple, Dict


# -------------------------
# Time-period filter (AM/MD/PM/ALL)
# -------------------------
@st.cache_data(show_spinner=False)
def filter_by_period(df: pd.DataFrame, time_col: str, period: str) -> pd.DataFrame:
    """Filter dataframe by time period (AM 05–10, MD 11–15, PM 16–20, ALL)."""
    if time_col not in df.columns or df.empty:
        return df
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")

    if period == "AM":
        return d[(d[time_col].dt.hour >= 5) & (d[time_col].dt.hour <= 10)]
    if period == "MD":
        return d[(d[time_col].dt.hour >= 11) & (d[time_col].dt.hour <= 15)]
    if period == "PM":
        return d[(d[time_col].dt.hour >= 16) & (d[time_col].dt.hour <= 20)]
    return d


# -------------------------
# Rule-based thresholds (kept for transparency)
# -------------------------
@st.cache_data(show_spinner=False)
def get_hourly_cycle_length_by_threshold(volume: float) -> str:
    """
    Return cycle length by simple thresholds (your original logic):
    ≥2400 → 140 sec, ≥1500 → 130 sec, ≥600 → 120 sec, ≥300 → 110 sec, else Free mode
    """
    if pd.isna(volume) or volume <= 0:
        return "Free mode"
    if volume >= 2400:
        return "140 sec"
    if volume >= 1500:
        return "130 sec"
    if volume >= 600:
        return "120 sec"
    if volume >= 300:
        return "110 sec"
    return "Free mode"


def _get_status(recommended: str, current: str) -> str:
    """Compare recommended cycle vs current and return status label."""
    if recommended == current:
        return "🟢 OPTIMAL"
    if recommended == "Free mode" and current != "Free mode":
        return "🔽 REDUCE"
    if recommended != "Free mode" and current == "Free mode":
        return "⬆️ INCREASE"
    rec_val = int(recommended.split()[0]) if recommended != "Free mode" else 0
    cur_val = int(current.split()[0]) if current != "Free mode" else 0
    if rec_val > cur_val:
        return "⬆️ INCREASE"
    if rec_val < cur_val:
        return "🔽 REDUCE"
    return "🟢 OPTIMAL"


# -------------------------
# Visual helpers (legend + colors) — theme-able & colorblind-safe
# -------------------------
CYCLE_ORDER = ["Free mode", "110 sec", "120 sec", "130 sec", "140 sec"]
THRESHOLD_TEXT = {
    "140 sec": "≥ 2400 vph",
    "130 sec": "≥ 1500 vph",
    "120 sec": "≥ 600 vph",
    "110 sec": "≥ 300 vph",
    "Free mode": "< 300 vph",
}

def _get_palettes(theme: str):
    """
    Returns (cycle_colors, status_colors, pattern_map) for the selected theme.
    Default is Okabe–Ito colorblind-safe palette.
    """
    if theme == "High Contrast":
        cycle_colors = {"Free mode": "#808080","110 sec": "#1B9E77","120 sec": "#386CB0","130 sec": "#FDC827","140 sec": "#D62728"}
        status_colors = {"🟢 OPTIMAL": "#1B9E77", "⬆️ INCREASE": "#D62728", "🔽 REDUCE": "#386CB0"}
    elif theme == "Greens → Red":
        cycle_colors = {"Free mode": "#9E9E9E","110 sec": "#2ECC71","120 sec": "#27AE60","130 sec": "#E67E22","140 sec": "#E74C3C"}
        status_colors = {"🟢 OPTIMAL": "#27AE60", "⬆️ INCREASE": "#E74C3C", "🔽 REDUCE": "#2E86C1"}
    elif theme == "Monochrome + Accents":
        cycle_colors = {"Free mode": "#95A5A6","110 sec": "#34495E","120 sec": "#2C3E50","130 sec": "#8E44AD","140 sec": "#E74C3C"}
        status_colors = {"🟢 OPTIMAL": "#2ECC71", "⬆️ INCREASE": "#E74C3C", "🔽 REDUCE": "#8E44AD"}
    else:  # Colorblind Safe
        cycle_colors = {"Free mode": "#8C8C8C","110 sec": "#009E73","120 sec": "#0072B2","130 sec": "#E69F00","140 sec": "#D55E00"}
        status_colors = {"🟢 OPTIMAL": "#009E73", "⬆️ INCREASE": "#D55E00", "🔽 REDUCE": "#0072B2"}

    pattern_map = {"Free mode": "", "110 sec": "/", "120 sec": "\\", "130 sec": "x", "140 sec": "."}
    return cycle_colors, status_colors, pattern_map


def _inject_kpi_css():
    """Theme-aware CSS for legend and KPI cards (robust dark-mode support)."""
    st.markdown(
        """
<style>
:root{
  --legend-bg: rgba(15,47,82,.06);
  --legend-border: rgba(79,172,254,.28);
  --legend-title: #0f2f52;
  --kpi-bg: linear-gradient(135deg, rgba(79,172,254,.06), rgba(0,242,254,.04));
  --kpi-border: rgba(79,172,254,.28);
  --kpi-text: #0f2f52;
  --kpi-title: #0f2f52;
  --kpi-muted: rgba(15,47,82,.78);
  --kpi-shadow: 0 8px 20px rgba(79,172,254,.10);
  --kpi-pill: rgba(255,255,255,.65);
}
html.dark,[data-theme="dark"],[Prediction-theme="dark"]{
  --legend-bg: rgba(255,255,255,.08);
  --legend-border: rgba(255,255,255,.18);
  --legend-title: #ffffff;
  --kpi-bg: rgba(255,255,255,.07);
  --kpi-border: rgba(255,255,255,.22);
  --kpi-text: #ffffff;
  --kpi-title: #ffffff;
  --kpi-muted: rgba(255,255,255,.82);
  --kpi-shadow: 0 10px 26px rgba(0,0,0,.35);
  --kpi-pill: rgba(255,255,255,.10);
}
.cvag-legend{border:1px solid var(--legend-border);background:var(--legend-bg);border-radius:12px;padding:.6rem 1rem;box-shadow:0 8px 24px rgba(0,0,0,.10);margin:.25rem 0 .5rem}
.cvag-legend-title{font-weight:800;color:var(--legend-title)!important;margin-bottom:.35rem}
.cvag-kpi-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:4px 0 10px}
@media (max-width:1500px){.cvag-kpi-grid{grid-template-columns:repeat(3,1fr)}}
@media (max-width:900px){.cvag-kpi-grid{grid-template-columns:repeat(2,1fr)}}
@media (max-width:600px){.cvag-kpi-grid{grid-template-columns:1fr}}
.cvag-kpi-card{border-radius:16px;padding:14px 16px;background:var(--kpi-bg);border:1px solid var(--kpi-border);box-shadow:var(--kpi-shadow);color:var(--kpi-text)}
.cvag-kpi-title{font-weight:800;font-size:.95rem;letter-spacing:.2px;color:var(--kpi-title)!important}
.cvag-kpi-value{font-size:2.0rem;line-height:1.05;font-weight:800;margin-top:.25rem;letter-spacing:.3px;color:var(--kpi-title)!important}
.cvag-kpi-delta{margin-top:.15rem;font-size:.95rem;font-weight:700;color:var(--kpi-muted)}
.cvag-kpi-delta.good{color:#2ECC71}.cvag-kpi-delta.warn{color:#F39C12}.cvag-kpi-delta.bad{color:#E74C3C}
.cvag-kpi-foot{margin-top:.35rem;font-size:.85rem;color:var(--kpi-muted)}
.cvag-pill{display:inline-flex;align-items:center;gap:.4rem;padding:.25rem .55rem;border-radius:999px;background:var(--kpi-pill);font-weight:700;font-size:.82rem}
</style>
        """,
        unsafe_allow_html=True,
    )


def _legend_html(cycle_colors: Dict[str, str]) -> str:
    """HTML legend for cycle length thresholds, generated from active palette."""
    items = []
    for label in ["140 sec", "130 sec", "120 sec", "110 sec", "Free mode"]:
        color = cycle_colors.get(label, "#9A9A9A")
        text = THRESHOLD_TEXT[label]
        items.append(
            f'<span style="display:inline-flex;align-items:center;margin:.25rem .5rem;'
            f'padding:.3rem .6rem;border-radius:999px;background:{color};color:#fff;'
            f'font-weight:800;font-size:.85rem;">{label}</span>'
            f'<span style="margin-right:1rem;opacity:.85;font-size:.9rem">{text}</span>'
        )
    return '<div class="cvag-legend"><div class="cvag-legend-title">Cycle Length Thresholds</div>' + "".join(items) + "</div>"


def _sec_value(label: str) -> int:
    """Map label to numeric seconds for sorting/plotting."""
    return int(label.split()[0]) if label != "Free mode" else 0


# -------------------------
# Help / onboarding (unchanged)
# -------------------------
def render_howto_sidebar() -> None:
    with st.sidebar.expander("ℹ️ How to use Cycle Length Calculator (4 steps)", expanded=False):
        st.markdown(
            """
**Step 1. Select Intersection** — choose the corridor/intersection.

**Step 2. Date Range → Custom** — pick a **single day** for clean hourly patterns.

**Step 3. Granularity → Direction** — select **NB** or **SB**.

**Step 4. Search** — results below include cycle recommendations, safety/benefit estimates, and a grant-ready summary.
            """
        )


def _build_header_html(
    intersection_label: str,
    direction_label: str,
    start_label: str,
    end_label: str,
    time_period_label: str,
    current_cycle: str,
) -> str:
    return f"""
<div style="background: linear-gradient(135deg, #2b77e5 0%, #19c3e6 100%);
            border-radius: 16px; padding: 22px 24px 20px; color: #fff;
            box-shadow: 0 10px 26px rgba(25,115,210,.25); margin: 8px 0 16px;">
  <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
    <div style="display:flex; align-items:center; gap:12px;">
      <div style="width:40px; height:40px; border-radius:10px; background: rgba(255,255,255,.18);
                  display:flex; align-items:center; justify-content:center;
                  box-shadow: inset 0 0 0 1px rgba(255,255,255,.15);">
        <span style="font-size:20px;">🔁</span>
      </div>
      <div style="font-size:2.1rem; font-weight:800; letter-spacing:.2px; line-height:1.1;">
        Cycle Length Recommendations for CVAG
      </div>
    </div>
    <span title="Current cycle used for comparison"
          style="display:inline-flex; align-items:center; gap:6px; padding:8px 12px;
                 border-radius:999px; background: rgba(255,255,255,.18);
                 font-weight:800; font-size:.95rem;
                 box-shadow: inset 0 0 0 1px rgba(255,255,255,.18);">
      ⚙️ Current Cycle: {current_cycle}
    </span>
  </div>

  <div style="display:flex; flex-wrap:wrap; gap:8px 10px; margin:12px 0 6px;">
    <span style="display:inline-flex; align-items:center; gap:6px; padding:6px 10px; border-radius:999px;
                 background: rgba(255,255,255,.16); font-weight:700; font-size:.95rem;
                 box-shadow: inset 0 0 0 1px rgba(255,255,255,.18);">
      <span style="opacity:.9;">Intersection:</span><span style="opacity:1;">{intersection_label}</span>
    </span>
    <span style="display:inline-flex; align-items:center; gap:6px; padding:6px 10px; border-radius:999px;
                 background: rgba(255,255,255,.16); font-weight:700; font-size:.95rem;
                 box-shadow: inset 0 0 0 1px rgba(255,255,255,.18);">
      <span style="opacity:.9;">Direction:</span><span style="opacity:1;">{direction_label}</span>
    </span>
    <span style="display:inline-flex; align-items:center; gap:6px; padding:6px 10px; border-radius:999px;
                 background: rgba(255,255,255,.16); font-weight:700; font-size:.95rem;
                 box-shadow: inset 0 0 0 1px rgba(255,255,255,.18);">
      <span style="opacity:.9;">Time Period:</span><span style="opacity:1;">{time_period_label}</span>
    </span>
    <span style="display:inline-flex; align-items:center; gap:6px; padding:6px 10px; border-radius:999px;
                 background: rgba(255,255,255,.16); font-weight:700; font-size:.95rem;
                 box-shadow: inset 0 0 0 1px rgba(255,255,255,.18);">
      <span style="opacity:.9;">Study Type:</span><span style="opacity:1;">Hourly Analysis</span>
    </span>
  </div>

  <div style="display:flex; align-items:center; gap:8px; margin-top:2px;">
    <span style="width:24px; height:24px; border-radius:8px; background: rgba(255,255,255,.18);
                 display:inline-flex; align-items:center; justify-content:center; font-size:13px;
                 box-shadow: inset 0 0 0 1px rgba(255,255,255,.16);">📅</span>
    <span style="font-size:1.05rem; font-weight:600; opacity:.95;">{start_label} — {end_label}</span>
  </div>
</div>
"""


# -------------------------
# Engineering core: Webster cycle + benefits
# -------------------------
def _bound(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _webster_cycle(
    volume_vph: float,
    sat_flow_vphpl: float,
    effective_lanes: float,
    n_phases: int,
    lost_time_per_phase: float,
    ped_min_per_cycle: float,
    min_cycle: float,
    max_cycle: float,
) -> float:
    """
    Compute Webster seeded cycle.
    Y = sum critical v/s ~ (volume) / (sat_flow * lanes)
    L = total lost time per cycle (n_phases * lost_time_per_phase)
    C = (1.5L + 5) / (1 - Y), bounded by ped_min_per_cycle and [min,max]
    Notes: Uses aggregate demand → approximation suited for quick planning.
    """
    sat_total = max(1.0, sat_flow_vphpl * effective_lanes)  # veh/h
    Y = min(0.95, volume_vph / sat_total) if volume_vph > 0 else 0.0  # guard
    L = n_phases * max(0.0, lost_time_per_phase)
    if Y >= 0.95:  # near saturation → push to max
        C = max_cycle
    else:
        C = (1.5 * L + 5.0) / max(1e-3, (1.0 - Y))
    C = max(ped_min_per_cycle, C)              # meet ped minimum
    C = _bound(C, min_cycle, max_cycle)        # enforce bounds
    return float(C)


def _hcm_uniform_delay(
    cycle_s: float,
    g_ratio: float,
    volume_vph: float,
    sat_flow_vphpl: float,
    effective_lanes: float,
) -> float:
    """
    Very simplified HCM uniform delay surrogate (seconds/veh).
    d = 0.5 * C * (1 - g/C)^2 / (1 - min(x, 1.2) * g/C)
    Where x = v / (s * g/C). We approximate g_ratio for major movement.
    """
    C = max(20.0, cycle_s)
    s_total = max(1.0, sat_flow_vphpl * effective_lanes)
    g = max(1e-3, g_ratio * C)
    cap = s_total * (g / C)            # veh/h effective capacity
    x = volume_vph / max(1.0, cap)
    x = min(x, 1.2)                     # cap x to avoid blowup
    term = (1 - g / C)
    d = 0.5 * C * (term ** 2) / max(1e-3, (1 - x * g / C))
    return float(max(0.0, d))


def _format_cycle(sec: float) -> str:
    return "Free mode" if sec < 30 else f"{int(round(sec))} sec"


def _confidence_score(hourly: pd.DataFrame) -> int:
    """
    Lightweight confidence: coverage + variability + smoothness.
    0–100, higher is better.
    """
    if hourly.empty:
        return 0
    # Coverage over a 6-hour window max (AM/MD/PM)
    coverage = min(1.0, len(hourly) / 6.0)
    # Variability: CV between 0.1 and 0.6 is healthy
    v = hourly["Volume"].astype(float)
    cv = float(np.nanstd(v) / max(1e-6, np.nanmean(v))) if v.notna().any() else 0
    cv_score = 1.0 - min(1.0, abs(cv - 0.35) / 0.35)   # peak at ~0.35
    # Smoothness: penalize extreme peaks
    peak_ratio = float(np.nanmax(v) / max(1e-6, np.nanmean(v))) if v.notna().any() else 1.0
    smooth = 1.0 - min(1.0, (peak_ratio - 1.5) / 1.5)  # good if <= ~2.0x
    s = 100 * max(0.0, min(1.0, 0.5 * coverage + 0.3 * cv_score + 0.2 * smooth))
    return int(round(s))


# -------------------------
# KPI-card HTML
# -------------------------
def _kpi_card(title: str, value_html: str, delta_text: str, tone: str = "neutral",
              foot1: Optional[str] = None, foot2: Optional[str] = None) -> str:
    tone = tone if tone in {"good", "warn", "bad", "neutral"} else "neutral"
    foot1_html = f'<div class="cvag-kpi-foot">{foot1}</div>' if foot1 else ""
    foot2_html = f'<div class="cvag-kpi-foot">{foot2}</div>' if foot2 else ""
    tone_class = f" {tone}" if tone != "neutral" else " neutral"
    return f"""
    <div class="cvag-kpi-card">
      <div class="cvag-kpi-title">{title}</div>
      <div class="cvag-kpi-value">{value_html}</div>
      <div class="cvag-kpi-delta{tone_class}">{delta_text}</div>
      {foot1_html}{foot2_html}
    </div>
    """


# -------------------------
# Main renderer
# -------------------------
def render_cycle_length_section(raw: pd.DataFrame, key_prefix: str = "cycle") -> None:
    """Render the Cycle Length Recommendations section with theme-aware styles + grant outputs."""
    render_howto_sidebar()

    if raw is None or raw.empty:
        st.info("No hourly volume data available for cycle length recommendations.")
        return
    if "local_datetime" not in raw.columns or "total_volume" not in raw.columns:
        st.info("Required columns not found: 'local_datetime', 'total_volume'.")
        return

    _inject_kpi_css()

    # ---- Context header bits ----
    d0 = raw.copy()
    d0["local_datetime"] = pd.to_datetime(d0["local_datetime"], errors="coerce")

    start_dt = d0["local_datetime"].min()
    end_dt = d0["local_datetime"].max()
    start_label = start_dt.strftime("%A, %b %d, %Y") if pd.notnull(start_dt) else "N/A"
    end_label = end_dt.strftime("%A, %b %d, %Y") if pd.notnull(end_dt) else "N/A"

    intersections = sorted(d0["intersection_name"].dropna().unique().tolist()) if "intersection_name" in d0 else []
    intersection_label = intersections[0] if len(intersections) == 1 else ("All Intersections" if len(intersections) > 1 else "N/A")

    directions = sorted(d0["direction"].dropna().unique().tolist()) if "direction" in d0 else []
    direction_label = directions[0] if len(directions) == 1 else ("All Directions" if len(directions) > 1 else "N/A")

    header_slot = st.empty()

    # -------------------------
    # Controls (engine + visuals)
    # -------------------------
    c1, c2, c3 = st.columns([2.1, 1.4, 1.3])
    with c1:
        time_period = st.selectbox(
            "🕐 Time Period",
            ["AM (05:00-10:00)", "MD (11:00-15:00)", "PM (16:00-20:00)", "All Day"],
            index=0,
            help="Filter to AM, Midday, PM, or all hours.",
            key=f"{key_prefix}_period",
        )
    with c2:
        current_cycle_label = st.selectbox(
            "⚙️ Current System Cycle",
            CYCLE_ORDER[::-1],  # 140, 130, 120, 110, Free
            index=0,
            help="Current field cycle length used for comparison.",
            key=f"{key_prefix}_current",
        )
    with c3:
        theme_choice = st.selectbox(
            "🎨 Color Theme",
            ["Colorblind Safe", "High Contrast", "Greens → Red", "Monochrome + Accents"],
            index=0,
            help="Pick a palette for readability.",
            key=f"{key_prefix}_theme",
        )

    with st.expander("🔧 Engineering Constraints (used in Webster seeding)", expanded=False):
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            n_phases = st.number_input("Number of Phases", 2, 8, value=4, step=1, key=f"{key_prefix}_phases")
            lost_time_per_phase = st.number_input("Lost Time per Phase (s)", 2.0, 10.0, value=5.0, step=0.5, key=f"{key_prefix}_lost")
            ped_min_per_cycle = st.number_input("Critical Ped Crossing (s)", 0.0, 60.0, value=28.0, step=1.0, help="WALK+FDW for longest crosswalk.", key=f"{key_prefix}_ped")
        with ec2:
            sat_flow_vphpl = st.number_input("Saturation Flow (vph per lane)", 1500, 2200, value=1900, step=50, key=f"{key_prefix}_sat")
            eff_lanes = st.number_input("Effective Lanes (critical mov.)", 1.0, 6.0, value=2.0, step=0.5, key=f"{key_prefix}_lanes")
            g_ratio = st.slider("Assumed Green Ratio (major)", 0.30, 0.70, value=0.50, step=0.01, help="Used for delay estimation.", key=f"{key_prefix}_gr")
        with ec3:
            min_cycle = st.number_input("Min Cycle (s)", 50, 200, value=80, step=5, key=f"{key_prefix}_cmin")
            max_cycle = st.number_input("Max Cycle (s)", 60, 240, value=160, step=5, key=f"{key_prefix}_cmax")
            monet_per_hour = st.number_input("Value of Time ($/hr)", 10.0, 80.0, value=22.0, step=1.0, key=f"{key_prefix}_vot")

    CYCLE_COLORS, STATUS_COLORS, PATTERN_MAP = _get_palettes(theme_choice)
    st.markdown(_legend_html(CYCLE_COLORS), unsafe_allow_html=True)

    # Filter to period
    period_map = {"AM (05:00-10:00)": "AM", "MD (11:00-15:00)": "MD", "PM (16:00-20:00)": "PM", "All Day": "ALL"}
    selected_period = period_map[time_period]
    d = d0 if selected_period == "ALL" else filter_by_period(d0, "local_datetime", selected_period)
    if d.empty:
        header_slot.markdown(_build_header_html(intersection_label, direction_label, start_label, end_label, time_period, current_cycle_label), unsafe_allow_html=True)
        st.warning("⚠️ No data available for the selected time period.")
        return

    # Render header with current cycle
    header_slot.markdown(_build_header_html(intersection_label, direction_label, start_label, end_label, time_period, current_cycle_label), unsafe_allow_html=True)

    # Hourly aggregation
    d["hour"] = d["local_datetime"].dt.hour
    hourly = d.groupby("hour", as_index=False)["total_volume"].mean()
    hourly["Volume"] = hourly["total_volume"].fillna(0).round().astype(int)
    hourly["Hour"] = hourly["hour"].apply(lambda x: f"{x:02d}:00")

    # Recommendations
    # 1) Threshold rule (for transparency + continuity)
    hourly["Rule Rec"] = hourly["Volume"].apply(get_hourly_cycle_length_by_threshold)
    hourly["Rule Rec (sec)"] = hourly["Rule Rec"].apply(_sec_value)

    # 2) Webster seeding with constraints
    webster_sec = []
    for v in hourly["Volume"].tolist():
        c = _webster_cycle(v, sat_flow_vphpl, eff_lanes, n_phases, lost_time_per_phase, ped_min_per_cycle, min_cycle, max_cycle)
        webster_sec.append(c)
    hourly["Webster (sec)"] = webster_sec
    hourly["Webster Rec"] = hourly["Webster (sec)"].apply(_format_cycle)

    # 3) Final rec = max(ped min, max(rule, webster)) bounded in [min,max]
    final_sec = []
    for r_sec, w_sec in zip(hourly["Rule Rec (sec)"].tolist(), hourly["Webster (sec)"].tolist()):
        base = max(float(r_sec), float(w_sec))
        base = max(base, ped_min_per_cycle)
        final_sec.append(_bound(base, float(min_cycle), float(max_cycle)))
    hourly["Final (sec)"] = final_sec
    hourly["CVAG Recommendation"] = hourly["Final (sec)"].apply(_format_cycle)

    # Status vs current cycle
    hourly["System Current Cycle"] = current_cycle_label
    hourly["Status"] = hourly["CVAG Recommendation"].apply(lambda rec: _get_status(rec, current_cycle_label))

    # Benefits estimation (Δ delay -> hours saved/day, $)
    current_sec = _sec_value(current_cycle_label) if current_cycle_label != "Free mode" else 0
    cur_cycle_for_math = max(current_sec, float(min_cycle)) if current_sec > 0 else float(min_cycle)

    hourly["Delay_Current_s_veh"] = hourly.apply(
        lambda r: _hcm_uniform_delay(cycle_s=cur_cycle_for_math, g_ratio=g_ratio, volume_vph=r["Volume"],
                                     sat_flow_vphpl=sat_flow_vphpl, effective_lanes=eff_lanes), axis=1
    )
    hourly["Delay_Rec_s_veh"] = hourly.apply(
        lambda r: _hcm_uniform_delay(cycle_s=max(float(r["Final (sec)"]), float(min_cycle)), g_ratio=g_ratio, volume_vph=r["Volume"],
                                     sat_flow_vphpl=sat_flow_vphpl, effective_lanes=eff_lanes), axis=1
    )
    hourly["ΔDelay_s_veh"] = hourly["Delay_Current_s_veh"] - hourly["Delay_Rec_s_veh"]
    hourly["ΔDelay_Positive"] = hourly["ΔDelay_s_veh"].apply(lambda x: max(0.0, x))

    # Exposure / risk proxies
    HIGH_VOLUME_THRESHOLD_VPH = 1200
    d["total_volume"] = pd.to_numeric(d["total_volume"], errors="coerce")
    high_rows = d.loc[d["total_volume"] > HIGH_VOLUME_THRESHOLD_VPH]
    exposure_hours = len(high_rows)        # count of hourly rows > threshold

    # Rollup KPIs
    total_hours = len(hourly)
    optimal_hours = int((hourly["Status"] == "🟢 OPTIMAL").sum())
    changes_needed = total_hours - optimal_hours
    hours_window_str = {"AM": "05:00–10:00", "MD": "11:00–15:00", "PM": "16:00–20:00", "ALL": "00:00–23:00"}.get(selected_period, "—")

    # Benefits per day (sketch): seconds saved * volume -> veh-sec -> hours
    veh_seconds_saved = float(np.nansum(hourly["ΔDelay_Positive"] * hourly["Volume"]))
    hours_saved_per_day = veh_seconds_saved / 3600.0
    dollar_value_per_day = hours_saved_per_day * float(monet_per_hour)
    # very rough emissions factor ~ 0.9 kg CO2e per veh-hour idling (sketch)
    emissions_kg_per_day = hours_saved_per_day * 0.9

    # Confidence score
    conf = _confidence_score(hourly)

    # -------------------------
    # KPI cards
    # -------------------------
    tone_eff = "good" if (optimal_hours / total_hours * 100 if total_hours else 0) >= 80 else ("warn" if optimal_hours >= total_hours * 0.6 else "bad")
    tone_changes = "good" if changes_needed == 0 else ("warn" if changes_needed <= (total_hours * 0.4) else "bad")
    tone_benefit = "good" if hours_saved_per_day >= 1.0 else ("warn" if hours_saved_per_day >= 0.2 else "neutral")
    tone_conf = "good" if conf >= 75 else ("warn" if conf >= 50 else "bad")

    cards_html = f"""
    <div class="cvag-kpi-grid">
      {_kpi_card("📅 Hours Analyzed", f"{total_hours}", hours_window_str, "neutral")}
      {_kpi_card("✅ Optimal Hours", f"{optimal_hours}", f"{(optimal_hours/total_hours*100 if total_hours else 0):.0f}% efficiency", tone_eff)}
      {_kpi_card("🔧 Changes Needed", f"{changes_needed}", "Hours that should change cycle", tone_changes)}
      {_kpi_card("⏱️ Hours Saved / Day", f"{hours_saved_per_day:.2f}", f"${dollar_value_per_day:,.0f}/day value", tone_benefit, foot1="Sketch from Δdelay")}
      {_kpi_card("🛡️ Data Confidence", f"{conf}/100", "Coverage • Variability • Smoothness", tone_conf)}
    </div>
    """
    st.markdown(cards_html, unsafe_allow_html=True)

    # -------------------------
    # Charts
    # -------------------------
    left, right = st.columns([2.2, 1.8])

    with left:
        # Bar chart by final recommendation + status markers
        fig = px.bar(
            hourly.sort_values("hour"),
            x="Hour",
            y="Volume",
            color="CVAG Recommendation",
            color_discrete_map=CYCLE_COLORS,
            category_orders={"CVAG Recommendation": CYCLE_ORDER, "Hour": [f"{h:02d}:00" for h in range(24)]},
            title="Hourly Volume with Recommended Cycle Length (Final)",
            labels={"Volume": "Avg Volume (vph)", "Hour": "Hour of Day"},
            template="simple_white",
        )
        for tr in fig.data:
            tr.update(marker_line_color="rgba(0,0,0,0.30)", marker_line_width=0.7)
            if tr.name in PATTERN_MAP:
                tr.update(marker_pattern=dict(shape=PATTERN_MAP[tr.name], size=4, solidity=0.25, fillmode="overlay"))

        status_symbols = {"🟢 OPTIMAL": "circle", "⬆️ INCREASE": "triangle-up", "🔽 REDUCE": "triangle-down"}
        fig.add_trace(
            go.Scatter(
                x=hourly["Hour"],
                y=hourly["Volume"],
                mode="markers",
                marker=dict(
                    size=11,
                    color=[STATUS_COLORS[s] for s in hourly["Status"]],
                    symbol=[status_symbols[s] for s in hourly["Status"]],
                    line=dict(width=1, color="white"),
                ),
                name="Status",
                hovertemplate="Hour=%{x}<br>Volume=%{y:.0f}<extra></extra>",
            )
        )
        fig.update_layout(height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=10, r=10, t=50, b=10), bargap=0.15)
        fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor="rgba(0,0,0,0.08)")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        # Hours by status (pie)
        status_counts = hourly["Status"].value_counts().reindex(["🟢 OPTIMAL", "⬆️ INCREASE", "🔽 REDUCE"], fill_value=0)
        pie = px.pie(
            names=status_counts.index,
            values=status_counts.values,
            title="Hours by Status",
            color=status_counts.index,
            color_discrete_map={"🟢 OPTIMAL": STATUS_COLORS["🟢 OPTIMAL"], "⬆️ INCREASE": STATUS_COLORS["⬆️ INCREASE"], "🔽 REDUCE": STATUS_COLORS["🔽 REDUCE"]},
            hole=0.35,
            template="simple_white",
        )
        pie.update_traces(textposition="inside", textinfo="label+percent")
        pie.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(pie, use_container_width=True)

    # -------------------------
    # Table (with your requested tweaks)
    # -------------------------
    hourly["Volume (vehicles)"] = hourly["Volume"].apply(lambda v: f"{int(v):,} vehicles")
    display = hourly[[
        "Hour",
        "Volume (vehicles)",
        "Rule Rec",
        "Webster Rec",
        "CVAG Recommendation",
        "System Current Cycle",
        "Status",
        "Delay_Current_s_veh",
        "Delay_Rec_s_veh",
        "ΔDelay_s_veh",
    ]].rename(columns={
        "Volume (vehicles)": "Total Vehicle Volume (Throughs, lefts, and rights)",
        "Rule Rec": "Rule (legacy)",
        "Webster Rec": "Webster (seeded)",
        "CVAG Recommendation": "Cycle Length Recommendation For CVAG",
        "System Current Cycle": "System Current Cycle",
        "Status": "Cycle Length Status",
        "Delay_Current_s_veh": "Current Delay (s/veh)",
        "Delay_Rec_s_veh": "Rec Delay (s/veh)",
        "ΔDelay_s_veh": "ΔDelay (s/veh)",
    })

    st.dataframe(
        display,
        use_container_width=True,
        column_config={
            "Hour": st.column_config.TextColumn("Hour", width="small"),
            "Total Vehicle Volume (Throughs, lefts, and rights)": st.column_config.TextColumn("Total Vehicle Volume (Throughs, lefts, and rights)"),
            "Rule (legacy)": st.column_config.TextColumn("Rule (legacy)", width="small"),
            "Webster (seeded)": st.column_config.TextColumn("Webster (seeded)", width="small"),
            "Cycle Length Recommendation For CVAG": st.column_config.TextColumn("Cycle Length Recommendation For CVAG", width="medium"),
            "System Current Cycle": st.column_config.TextColumn("System Current Cycle", width="medium"),
            "Cycle Length Status": st.column_config.TextColumn("Cycle Length Status", width="medium"),
            "Current Delay (s/veh)": st.column_config.NumberColumn("Current Delay (s/veh)", format="%.1f"),
            "Rec Delay (s/veh)": st.column_config.NumberColumn("Rec Delay (s/veh)", format="%.1f"),
            "ΔDelay (s/veh)": st.column_config.NumberColumn("ΔDelay (s/veh)", format="%.1f"),
        },
    )

    # -------------------------
    # Insights + grant narrative
    # -------------------------
    peak_vol = int(hourly["Volume"].max()) if len(hourly) else 0
    peak_hr = hourly.loc[hourly["Volume"].idxmax(), "Hour"] if len(hourly) else "—"
    peds_msg = f"Ped minimum enforced at ≥ {ped_min_per_cycle:.0f}s; no recommendation drops below ADA timing."

    st.markdown(
        f"""
        <div class="insight-box" style="margin-top:.6rem;">
          <h4>💡 Optimization Insights</h4>
          <ul style="margin-top:.2rem;">
            <li><b>System Efficiency:</b> {optimal_hours}/{total_hours} hours optimal ({(optimal_hours/total_hours*100 if total_hours else 0):.0f}%).</li>
            <li><b>Volume Profile:</b> Peak {peak_vol:,} vph at {peak_hr}; exposure above {HIGH_VOLUME_THRESHOLD_VPH:,} vph observed for {exposure_hours} hours.</li>
            <li><b>Benefit (sketch):</b> ~{hours_saved_per_day:.2f} hours saved/day (≈ ${dollar_value_per_day:,.0f}/day; ~{emissions_kg_per_day:,.0f} kg CO₂e/day avoided).</li>
            <li><b>Pedestrian Service:</b> {peds_msg}</li>
            <li><b>Confidence:</b> {conf}/100 (coverage, variability, smoothness).</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Downloadables ----------
    # (1) CSV – include numeric columns + strings
    download_df = display.merge(
        hourly[["Hour", "Volume", "Final (sec)"]].rename(columns={"Volume": "Total_Volume_vph", "Final (sec)": "Final_Cycle_sec"}),
        on="Hour", how="left"
    )
    st.download_button(
        "⬇️ Download Cycle Length Analysis (CSV)",
        data=download_df.to_csv(index=False).encode("utf-8"),
        file_name=f"cycle_len_analysis_{selected_period.lower()}.csv",
        mime="text/csv",
        key=f"{key_prefix}_download_csv",
    )

    # (2) Grant-ready summary (txt)
    intersection_for_text = intersection_label.replace("\n", " ")
    grant_text = f"""ADVANTEC Cycle Length Optimization — Grant-Ready Summary
Intersection: {intersection_for_text} | Direction: {direction_label}
Study Window: {start_label} — {end_label} | Period: {time_period}

Problem / Need
• Exposure above high-volume threshold (> {HIGH_VOLUME_THRESHOLD_VPH:,} vph): {exposure_hours} hours in study window.
• Current cycle: {current_cycle_label}; Optimal hours: {optimal_hours}/{total_hours} ({(optimal_hours/total_hours*100 if total_hours else 0):.0f}%).

Recommended Action
• Implement {int(round(np.nanmedian(hourly['Final (sec)'])))}-second cycle for the selected period with phase splits honoring pedestrian min time ({ped_min_per_cycle:.0f}s).
• Update plans for hours flagged “INCREASE”/“REDUCE”; coordinate corridor offsets during peak to protect platoons.

Expected Safety & Mobility Benefits (sketch, planning-level)
• ~{hours_saved_per_day:.2f} hours of vehicle delay saved per day (≈ ${dollar_value_per_day:,.0f}/day user benefit).
• ~{emissions_kg_per_day:,.0f} kg CO₂e/day reduction from idling.
• Pedestrian service maintained: WALK/FDW ≥ {ped_min_per_cycle:.0f}s equivalent; no recommended cycle violates ADA timing.

Method (transparent)
• Seed cycles using Webster: C = (1.5L + 5)/(1 - Y), with L = {n_phases} phases × {lost_time_per_phase:.1f}s lost/phase, Y ≈ demand / ({sat_flow_vphpl}×{eff_lanes:.1f}) v/s.
• Final cycle = max(ped min, rule-based threshold, Webster), bounded [{min_cycle},{max_cycle}] s.
• Benefits estimated via uniform-delay surrogate (HCM-style), then aggregated to hours/day and monetized at ${monet_per_hour:.0f}/hr.
• Data Confidence Score: {conf}/100 (coverage, variability, smoothness).

Implementation & Verification
• Deploy as AM/MD/PM time-of-day plans; monitor travel time/delay for 2 weeks.
• KPIs: 95th TT, arrivals-on-green proxy, hours above threshold, ped wait.
• Adjust splits/offsets iteratively to guard against spillback and improve progression.

Equity & Safety Emphasis
• Minimizes high-delay tails and idling exposure; preserves pedestrian timing.
• Ready to align with SS4A/HSIP narratives (problem → countermeasure → benefit → M&V).
"""
    st.download_button(
        "⬇️ Download Grant-Ready Summary (TXT)",
        data=grant_text.encode("utf-8"),
        file_name=f"grant_summary_{selected_period.lower()}.txt",
        mime="text/plain",
        key=f"{key_prefix}_download_txt",
    )

    # Friendly disclaimer
    st.caption("Notes: Benefits are planning-level estimates using standard approximations and your hourly volumes. "
               "Where detailed turning counts, per-approach saturation flows, and offsets are available, "
               "the engine will refine cycles/splits and coordination scoring for submittal-grade analysis.")
