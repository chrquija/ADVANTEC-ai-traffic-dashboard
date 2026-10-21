# cycle_length_recommendations.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional
from datetime import datetime, timedelta

# -------------------------
# Time-period filter (AM/MD/PM/ALL)
# -------------------------
@st.cache_data
def filter_by_period(df: pd.DataFrame, time_col: str, period: str) -> pd.DataFrame:
    """Filter dataframe by time period (AM 05–10, MD 11–15, PM 16–20, ALL)."""
    if time_col not in df.columns or df.empty:
        return df
    df_copy = df.copy()
    df_copy[time_col] = pd.to_datetime(df_copy[time_col], errors="coerce")

    if period == "AM":
        return df_copy[(df_copy[time_col].dt.hour >= 5) & (df_copy[time_col].dt.hour <= 10)]
    if period == "MD":
        return df_copy[(df_copy[time_col].dt.hour >= 11) & (df_copy[time_col].dt.hour <= 15)]
    if period == "PM":
        return df_copy[(df_copy[time_col].dt.hour >= 16) & (df_copy[time_col].dt.hour <= 20)]
    return df_copy


# -------------------------
# Cycle length thresholds
# -------------------------
@st.cache_data
def get_hourly_cycle_length(volume):
    """
    Return recommended cycle length:
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
CYCLE_ORDER = ["Free mode", "110 sec", "120 sec", "130 sec", "140 sec", "150 sec"]
THRESHOLD_TEXT = {
    "140 sec": "≥ 2400 vehicles",
    "130 sec": "≥ 1500 vehicles",
    "120 sec": "≥ 600 vehicles",
    "110 sec": "≥ 300 vehicles",
    "Free mode": "< 300 vehicles",
}

def _get_palettes(theme: str):
    """
    Returns (cycle_colors, status_colors, pattern_map) for the selected theme.
    Default is Okabe–Ito colorblind-safe palette.
    """
    if theme == "High Contrast":
        cycle_colors = {
            "Free mode": "#808080",
            "110 sec": "#1B9E77",
            "120 sec": "#386CB0",
            "130 sec": "#FDC827",
            "140 sec": "#D62728",
            "150 sec": "#7570B3",
        }
        status_colors = {"🟢 OPTIMAL": "#1B9E77", "⬆️ INCREASE": "#D62728", "🔽 REDUCE": "#386CB0"}
    elif theme == "Greens → Red":
        cycle_colors = {
            "Free mode": "#9E9E9E",
            "110 sec": "#2ECC71",
            "120 sec": "#27AE60",
            "130 sec": "#E67E22",
            "140 sec": "#E74C3C",
            "150 sec": "#C0392B",
        }
        status_colors = {"🟢 OPTIMAL": "#27AE60", "⬆️ INCREASE": "#E74C3C", "🔽 REDUCE": "#2E86C1"}
    elif theme == "Monochrome + Accents":
        cycle_colors = {
            "Free mode": "#95A5A6",
            "110 sec": "#34495E",
            "120 sec": "#2C3E50",
            "130 sec": "#8E44AD",
            "140 sec": "#E74C3C",
            "150 sec": "#943126",
        }
        status_colors = {"🟢 OPTIMAL": "#2ECC71", "⬆️ INCREASE": "#E74C3C", "🔽 REDUCE": "#8E44AD"}
    else:  # "Colorblind Safe" (Okabe–Ito)
        cycle_colors = {
            "Free mode": "#8C8C8C",   # gray
            "110 sec": "#009E73",     # bluish green
            "120 sec": "#0072B2",     # blue
            "130 sec": "#E69F00",     # orange
            "140 sec": "#D55E00",     # vermillion
            "150 sec": "#CC79A7",     # reddish purple
        }
        status_colors = {"🟢 OPTIMAL": "#009E73", "⬆️ INCREASE": "#D55E00", "🔽 REDUCE": "#0072B2"}

    pattern_map = {
        "Free mode": "",
        "110 sec": "/",
        "120 sec": "\\",
        "130 sec": "x",
        "140 sec": ".",
        "150 sec": "+",
    }
    return cycle_colors, status_colors, pattern_map


def _inject_kpi_css():
    """Theme-aware CSS for legend and KPI cards (robust dark-mode support)."""
    st.markdown(
        """
<style>
/* ---------- Light defaults ---------- */
:root{
  /* Legend */
  --legend-bg: rgba(15,47,82,.06);
  --legend-border: rgba(79,172,254,.28);
  --legend-title: #0f2f52;

  /* KPI tiles */
  --kpi-bg: linear-gradient(135deg, rgba(79,172,254,.06), rgba(0,242,254,.04));
  --kpi-border: rgba(79,172,254,.28);
  --kpi-text: #0f2f52;
  --kpi-title: #0f2f52;   /* explicit */
  --kpi-muted: rgba(15,47,82,.78);
  --kpi-shadow: 0 8px 20px rgba(79,172,254,.10);
  --kpi-good: #2ecc71;
  --kpi-warn: #f39c12;
  --kpi-bad: #e74c3c;
  --kpi-pill: rgba(255,255,255,.65);
}

/* ---------- Dark mode overrides ---------- */
html.dark, [Prediction-theme="dark"], [Prediction-base-theme="dark"], body[Prediction-theme="dark"]{
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

/* ---------- Control labels styling (Black font color) ---------- */
.stSelectbox label, .stSelectbox label p, div[data-testid="stSelectbox"] label p,
div[data-testid="stSelectbox"] label div, div[data-testid="stSelectbox"] label span {
  color: #000000 !important;
  font-weight: 700 !important;
  font-size: 0.95rem !important;
}

/* ---------- Executive Header Banner ---------- */
.cvag-header-banner {
  background: linear-gradient(135deg, #0f2b5c 0%, #1a4f8b 45%, #0284c7 100%);
  border-radius: 16px;
  padding: 22px 26px 20px;
  color: #ffffff;
  box-shadow: 0 10px 28px rgba(15, 43, 92, 0.22);
  margin: 8px 0 18px;
  border: 1px solid rgba(255, 255, 255, 0.18);
}
.cvag-header-top {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
}
.cvag-header-title {
  font-size: 1.85rem;
  font-weight: 800;
  letter-spacing: -0.01em;
  line-height: 1.18;
  color: #ffffff !important;
}
.cvag-header-subtitle {
  font-size: 0.90rem;
  font-weight: 500;
  color: rgba(255, 255, 255, 0.88);
  margin-top: 4px;
}
.cvag-header-meta-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 14px;
}
@media (max-width: 900px) {
  .cvag-header-meta-grid { grid-template-columns: 1fr; }
}
.cvag-meta-card {
  background: rgba(255, 255, 255, 0.12);
  border: 1px solid rgba(255, 255, 255, 0.24);
  border-radius: 12px;
  padding: 10px 16px;
  backdrop-filter: blur(8px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  transition: transform 0.15s ease, background 0.15s ease;
}
.cvag-cycle-card {
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.20) 0%, rgba(255, 255, 255, 0.10) 100%);
  border: 1px solid rgba(255, 255, 255, 0.38);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12), inset 0 0 12px rgba(255, 255, 255, 0.08);
}
.cvag-meta-card-label {
  font-size: 0.72rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #93c5fd;
  margin-bottom: 2px;
}
.cvag-meta-card-val {
  font-size: 1.15rem;
  font-weight: 700;
  color: #ffffff !important;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.cvag-header-chips-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}
.cvag-sub-chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 5px 12px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.09);
  border: 1px solid rgba(255, 255, 255, 0.16);
  font-size: 0.82rem;
  color: rgba(255, 255, 255, 0.95);
}
.cvag-sub-chip strong {
  color: #bae6fd;
  font-weight: 700;
}

/* ---------- Tooltip styles ---------- */
.cvag-tooltip-trigger {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.20);
  border: 1px solid rgba(255, 255, 255, 0.35);
  color: #ffffff;
  font-size: 12px;
  font-weight: 800;
  cursor: help;
  position: relative;
  user-select: none;
  flex-shrink: 0;
  transition: all 0.2s ease;
}
.cvag-tooltip-trigger:hover {
  background: rgba(255, 255, 255, 0.35);
  color: #ffffff;
}
.cvag-kpi-card .cvag-tooltip-trigger {
  background: rgba(15,47,82,.14);
  border: none;
  color: #0f2f52;
  width: 19px;
  height: 19px;
  font-size: 11px;
}
.cvag-kpi-card .cvag-tooltip-trigger:hover {
  background: #2b77e5;
  color: #ffffff;
}
.cvag-tooltip-box {
  visibility: hidden;
  opacity: 0;
  width: 250px;
  background-color: #0f172a;
  color: #ffffff !important;
  text-align: left;
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 0.80rem;
  font-weight: 500;
  line-height: 1.35;
  position: absolute;
  z-index: 9999;
  bottom: 130%;
  left: 50%;
  transform: translateX(-50%);
  box-shadow: 0 6px 20px rgba(0,0,0,0.30);
  border: 1px solid rgba(255,255,255,0.15);
  transition: opacity 0.2s, visibility 0.2s;
  pointer-events: none;
}
.cvag-tooltip-box::after {
  content: "";
  position: absolute;
  top: 100%;
  left: 50%;
  margin-left: -5px;
  border-width: 5px;
  border-style: solid;
  border-color: #0f172a transparent transparent transparent;
}
.cvag-tooltip-trigger:hover .cvag-tooltip-box {
  visibility: visible;
  opacity: 1;
}

/* ---------- Chart Card / Container Styling (like VantageLive) ---------- */
[data-testid="stVerticalBlockBordered"] {
  background: var(--secondary-background-color, white) !important;
  border: 1px solid rgba(128, 128, 128, 0.22) !important;
  border-radius: 14px !important;
  padding: 20px 24px !important;
  box-shadow: 0 4px 14px rgba(0,0,0,0.04) !important;
  margin-bottom: 20px !important;
}

/* ---------- Legend block ---------- */
.cvag-legend{
  border:1px solid var(--legend-border);
  background: var(--legend-bg);
  border-radius:12px;
  padding:.6rem 1rem;
  box-shadow:0 8px 24px rgba(0,0,0,.10);
  margin-top:.25rem;
}
.cvag-legend-title{
  font-weight:800;
  color:#000000 !important;
  margin-bottom:.35rem;
}
.cvag-legend-text{
  color:#000000 !important;
  font-weight:700 !important;
  font-size:.92rem !important;
  margin-right:1rem;
}

/* ---------- KPI grid/cards ---------- */
.cvag-kpi-grid{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; margin:4px 0 10px; }
@media (max-width:1500px){ .cvag-kpi-grid{ grid-template-columns:repeat(3,1fr);} }
@media (max-width:900px){ .cvag-kpi-grid{ grid-template-columns:repeat(2,1fr);} }
@media (max-width:600px){ .cvag-kpi-grid{ grid-template-columns:1fr;} }

.cvag-kpi-card{
  border-radius:16px; padding:14px 16px;
  background:var(--kpi-bg);
  border:1px solid var(--kpi-border);
  box-shadow:var(--kpi-shadow);
  color:var(--kpi-text);
}
.cvag-kpi-title-row{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
  margin-bottom: 2px;
}
.cvag-kpi-title{
  font-weight:800; font-size:.95rem; letter-spacing:.2px;
  color: var(--kpi-title) !important;
}
.cvag-kpi-value{
  font-size:2.0rem; line-height:1.05; font-weight:800; margin-top:.25rem; letter-spacing:.3px;
  color: var(--kpi-title) !important;
}
.cvag-kpi-delta{ margin-top:.15rem; font-size:.95rem; font-weight:700; color:var(--kpi-muted); }
.cvag-kpi-delta.good{ color:#2ECC71; } .cvag-kpi-delta.warn{ color:#F39C12; } .cvag-kpi-delta.bad{ color:#E74C3C; }
.cvag-kpi-foot{ margin-top:.35rem; font-size:.85rem; color:var(--kpi-muted); }
.cvag-pill{ display:inline-flex; align-items:center; gap:.4rem; padding:.25rem .55rem; border-radius:999px; background:var(--kpi-pill); font-weight:700; font-size:.82rem; }
</style>
        """,
        unsafe_allow_html=True,
    )


def _legend_html(cycle_colors: dict) -> str:
    """HTML legend for cycle length thresholds, generated from active palette."""
    pill_items = []
    for label in ["140 sec", "130 sec", "120 sec", "110 sec", "Free mode"]:
        color = cycle_colors.get(label, "#9A9A9A")
        text = THRESHOLD_TEXT[label]
        pill_items.append(
            f'<span style="display:inline-flex;align-items:center;margin:.25rem .5rem;'
            f'padding:.3rem .6rem;border-radius:999px;background:{color};color:#fff;'
            f'font-weight:800;font-size:.85rem;">{label}</span>'
            f'<span class="cvag-legend-text" style="margin-right:1rem;color:#000000 !important;font-weight:700;font-size:.92rem;">{text}</span>'
        )
    return (
        '<div class="cvag-legend">'
        '<div class="cvag-legend-title" style="color:#000000 !important;font-weight:800;">Cycle Length Thresholds</div>'
        + "".join(pill_items) +
        '</div>'
    )


def _sec_value(label: str) -> int:
    """Map label to numeric seconds for sorting/plotting."""
    return int(label.split()[0]) if label != "Free mode" else 0


# -------------------------
# Help / onboarding
# -------------------------
def render_howto_sidebar() -> None:
    """Sidebar expander with the exact workflow steps you provided."""
    with st.sidebar.expander("How to Use Cycle Length Calculator (4 Steps)", expanded=False):
        st.markdown(
            """
**Step 1. Select Intersection**  
Pick the corridor/intersection you want to analyze.

**Step 2. Date Range → Custom**  
Choose a **single specific day**.

**Step 3. Granularity → Direction**  
Select **Northbound** or **Southbound**.

**Step 4. Hit Generate**  
The calculator below will use your selections.
            """
        )


def _format_hour_ranges(hour_ids: list) -> str:
    """Group consecutive hour integers into clean ranges (e.g. '5:00 AM–10:00 AM' or '7:00 AM, 4:00 PM–6:00 PM')."""
    if not hour_ids:
        return "None"
    sorted_hrs = sorted(set(int(h) for h in hour_ids))
    ranges = []
    start = sorted_hrs[0]
    prev = sorted_hrs[0]

    for h in sorted_hrs[1:]:
        if h == prev + 1:
            prev = h
        else:
            if start == prev:
                ranges.append((datetime.min + timedelta(hours=start)).strftime("%I:%M %p").lstrip("0"))
            else:
                s_str = (datetime.min + timedelta(hours=start)).strftime("%I:%M %p").lstrip("0")
                e_str = (datetime.min + timedelta(hours=prev)).strftime("%I:%M %p").lstrip("0")
                ranges.append(f"{s_str}–{e_str}")
            start = h
            prev = h

    if start == prev:
        ranges.append((datetime.min + timedelta(hours=start)).strftime("%I:%M %p").lstrip("0"))
    else:
        s_str = (datetime.min + timedelta(hours=start)).strftime("%I:%M %p").lstrip("0")
        e_str = (datetime.min + timedelta(hours=prev)).strftime("%I:%M %p").lstrip("0")
        ranges.append(f"{s_str}–{e_str}")

    return ", ".join(ranges)


def _build_header_html(
    intersection_label: str,
    direction_label: str,
    start_label: str,
    end_label: str,
    time_period_label: str,
    current_cycle: str,
) -> str:
    """Header with executive navy-to-cyan gradient and high-contrast structured metadata cards."""
    date_display = start_label if start_label == end_label else f"{start_label} — {end_label}"
    return (
        f'<div class="cvag-header-banner">'
        f'<div class="cvag-header-top">'
        f'<div>'
        f'<div class="cvag-header-title">Cycle Length Recommendations for CVAG</div>'
        f'<div class="cvag-header-subtitle">Traffic Signal Timing & Corridor Coordination Optimization</div>'
        f'</div>'
        f'<div class="cvag-tooltip-trigger" title="Cycle Length Optimization analyzes hourly traffic volume against regional signal timing thresholds (110s–140s / Free Mode) to determine optimal coordination plans for CVAG corridors.">?'
        f'<span class="cvag-tooltip-box" style="width:280px; bottom:130%;">Cycle Length Optimization analyzes hourly traffic volume against regional signal timing thresholds (110s–140s / Free Mode) to determine optimal coordination plans for CVAG corridors.</span>'
        f'</div>'
        f'</div>'
        f'<div class="cvag-header-meta-grid">'
        f'<div class="cvag-meta-card cvag-cycle-card">'
        f'<div class="cvag-meta-card-label">Current Cycle</div>'
        f'<div class="cvag-meta-card-val">{current_cycle}</div>'
        f'</div>'
        f'<div class="cvag-meta-card">'
        f'<div class="cvag-meta-card-label">Intersection</div>'
        f'<div class="cvag-meta-card-val" title="{intersection_label}">{intersection_label}</div>'
        f'</div>'
        f'<div class="cvag-meta-card">'
        f'<div class="cvag-meta-card-label">Date</div>'
        f'<div class="cvag-meta-card-val" title="{date_display}">{date_display}</div>'
        f'</div>'
        f'</div>'
        f'<div class="cvag-header-chips-row">'
        f'<span class="cvag-sub-chip"><strong>Direction:</strong> {direction_label}</span>'
        f'<span class="cvag-sub-chip"><strong>Time Period:</strong> {time_period_label}</span>'
        f'<span class="cvag-sub-chip"><strong>Study Type:</strong> Hourly Analysis</span>'
        f'</div>'
        f'</div>'
    )


# -------------------------
# KPI-card HTML
# -------------------------
def _kpi_card(title: str, value_html: str, delta_text: str, tone: str = "neutral",
              foot1: Optional[str] = None, foot2: Optional[str] = None,
              tooltip: Optional[str] = None) -> str:
    tone = tone if tone in {"good", "warn", "bad", "neutral"} else "neutral"
    foot1_html = f'<div class="cvag-kpi-foot">{foot1}</div>' if foot1 else ""
    foot2_html = f'<div class="cvag-kpi-foot">{foot2}</div>' if foot2 else ""
    tone_class = f" {tone}" if tone != "neutral" else " neutral"
    tooltip_html = ""
    if tooltip:
        tooltip_html = (
            f'<div class="cvag-tooltip-trigger" title="{tooltip}">'
            f'?<span class="cvag-tooltip-box">{tooltip}</span>'
            f'</div>'
        )
    return (
        f'<div class="cvag-kpi-card">'
        f'<div class="cvag-kpi-title-row">'
        f'<div class="cvag-kpi-title">{title}</div>'
        f'{tooltip_html}'
        f'</div>'
        f'<div class="cvag-kpi-value">{value_html}</div>'
        f'<div class="cvag-kpi-delta{tone_class}">{delta_text}</div>'
        f'{foot1_html}{foot2_html}'
        f'</div>'
    )


# -------------------------
# Main renderer
# -------------------------
def render_cycle_length_section(raw: pd.DataFrame, key_prefix: str = "cycle") -> None:
    """Render the Cycle Length Recommendations section with theme-aware styles."""
    # Sidebar "how to" (safe to call even if parent already has sidebar items)
    render_howto_sidebar()

    if raw is None or raw.empty:
        st.info("No hourly volume Prediction available for cycle length recommendations.")
        return
    if "local_datetime" not in raw.columns or "total_volume" not in raw.columns:
        st.info("Required columns not found: 'local_datetime', 'total_volume'.")
        return

    # Make styles available BEFORE any HTML so colors work immediately
    _inject_kpi_css()

    # ---- Context values for header (static bits) ----
    raw = raw.copy()
    raw["local_datetime"] = pd.to_datetime(raw["local_datetime"], errors="coerce")
    raw["total_volume"] = pd.to_numeric(raw.get("total_volume", 0), errors="coerce").fillna(0)

    start_dt = raw["local_datetime"].min()
    end_dt = raw["local_datetime"].max()
    start_label = start_dt.strftime("%A, %b %d, %Y") if pd.notnull(start_dt) else "N/A"
    end_label = end_dt.strftime("%A, %b %d, %Y") if pd.notnull(end_dt) else "N/A"

    intersections = sorted(raw["intersection_name"].dropna().unique().tolist()) if "intersection_name" in raw else []
    if len(intersections) > 1:
        st.warning(f"**Note:** Cycle Length Recommendations are currently only supported for one intersection at a time to ensure accuracy. Analyzing **{intersections[0]}** only.")
        raw = raw[raw["intersection_name"] == intersections[0]].copy()
        intersections = [intersections[0]]

    if len(intersections) == 1:
        intersection_label = intersections[0]
    else:
        intersection_label = "N/A"

    directions = sorted(raw["direction"].dropna().unique().tolist()) if "direction" in raw else []
    
    dir_map_full = {
        "NB": "Northbound", "SB": "Southbound", "EB": "Eastbound", "WB": "Westbound",
        "NORTHBOUND": "Northbound", "SOUTHBOUND": "Southbound", "EASTBOUND": "Eastbound", "WESTBOUND": "Westbound",
        "ALL APPROACHES": "All Approaches"
    }

    if len(directions) == 1:
        direction_label = directions[0]
    elif len(directions) > 1:
        direction_label = "All Directions"
    else:
        direction_label = "N/A"
    
    # Corridor name next to intersection
    corridor_name = ""
    if "corridor_id" in raw.columns and not raw["corridor_id"].dropna().empty:
        corridor_name = str(raw["corridor_id"].iloc[0]).strip()
        if corridor_name.lower().endswith(" corridor"):
            corridor_name = corridor_name[:-9].strip()

    # Placeholder so we can render the header AFTER we know current_cycle/time period
    header_slot = st.empty()

    # -------------------------
    # Controls (user selections)
    # -------------------------
    # Initialize session state for widgets if not present to avoid warnings/collisions
    if f"{key_prefix}_period" not in st.session_state:
        st.session_state[f"{key_prefix}_period"] = "AM (05:00-10:00)"
    if f"{key_prefix}_dir_selector" not in st.session_state:
        st.session_state[f"{key_prefix}_dir_selector"] = "All Approaches"
    if f"{key_prefix}_current" not in st.session_state:
        st.session_state[f"{key_prefix}_current"] = "140 sec"
    if f"{key_prefix}_theme" not in st.session_state:
        st.session_state[f"{key_prefix}_theme"] = "Colorblind Safe"

    c1, c2, c3, c4 = st.columns([1.5, 1.5, 1.2, 1.3])
    with c1:
        time_period = st.selectbox(
            "Time Period",
            ["AM (05:00-10:00)", "MD (11:00-15:00)", "PM (16:00-20:00)", "All Day"],
            help="Filter cycle recommendations to AM Peak (05:00–10:00), Midday MD (11:00–15:00), PM Peak (16:00–20:00), or All Day (00:00–23:00).",
            key=f"{key_prefix}_period",
        )
    with c2:
        # Get available directions from the raw data
        avail_dirs = sorted(raw["direction"].dropna().unique().tolist()) if "direction" in raw.columns else []
        dir_options = ["All Approaches"] + avail_dirs
        selected_dir = st.selectbox(
            "Direction",
            dir_options,
            help="Filter volume data to a specific approach direction (e.g., Northbound, Southbound) or analyze All Approaches combined.",
            key=f"{key_prefix}_dir_selector",
        )
    with c3:
        current_cycle = st.selectbox(
            "Current Cycle",
            CYCLE_ORDER[::-1],  # 150, 140, 130, 120, 110, Free
            help="Sets the corridor's active field cycle length used as a benchmark for comparison (e.g., 140 sec, 130 sec, 120 sec, 110 sec, Free mode).",
            key=f"{key_prefix}_current",
        )
    with c4:
        theme_choice = st.selectbox(
            "Theme",
            ["Colorblind Safe", "High Contrast", "Greens → Red", "Monochrome + Accents"],
            help="Select visual color palette for charts and status indicators (Colorblind Safe Okabe–Ito, High Contrast, Greens to Red, Monochrome).",
            key=f"{key_prefix}_theme",
        )

    # Resolve palettes & patterns from theme
    CYCLE_COLORS, STATUS_COLORS, PATTERN_MAP = _get_palettes(theme_choice)

    # Legend (uses theme variables, not hard-coded colors)
    st.markdown(_legend_html(CYCLE_COLORS), unsafe_allow_html=True)

    # Apply direction and time period filtering
    if selected_dir != "All Approaches" and "direction" in raw.columns:
        work_df = raw[raw["direction"] == selected_dir].copy()
    else:
        work_df = raw.copy()

    period_map = {"AM (05:00-10:00)": "AM", "MD (11:00-15:00)": "MD", "PM (16:00-20:00)": "PM", "All Day": "ALL"}
    selected_period = period_map[time_period]
    period_data = work_df if selected_period == "ALL" else filter_by_period(work_df, "local_datetime", selected_period)
    if period_data.empty:
        # Render the header (with current cycle bubble) even if no data to show
        sel_dir_full = dir_map_full.get(selected_dir.upper()) if selected_dir != "All Approaches" else "All Approaches"
        if not sel_dir_full:
            sel_dir_full = selected_dir

        if corridor_name:
            int_header_label = f"{corridor_name} Corridor -- {intersection_label}"
        else:
            int_header_label = intersection_label

        header_slot.markdown(
            _build_header_html(
                int_header_label, sel_dir_full, start_label, end_label, time_period, current_cycle
            ),
            unsafe_allow_html=True,
        )
        st.warning("No data available for the selected time period. Check if Hourly Data is Available")
        return

    # Now that we know the user's choices, render the header with the bubble
    sel_dir_full = dir_map_full.get(selected_dir.upper()) if selected_dir != "All Approaches" else "All Approaches"
    if not sel_dir_full:
        sel_dir_full = selected_dir

    if corridor_name:
        int_header_label = f"{corridor_name} Corridor -- {intersection_label}"
    else:
        int_header_label = intersection_label

    header_slot.markdown(
        _build_header_html(
            int_header_label, sel_dir_full, start_label, end_label, time_period, current_cycle
        ),
        unsafe_allow_html=True,
    )

    # Hour window label for KPIs
    period_windows = {"AM": "05:00–10:00", "MD": "11:00–15:00", "PM": "16:00–20:00", "ALL": "00:00–23:00"}
    hours_window_str = period_windows.get(selected_period, "—")

    # Hourly aggregation
    # First sum volumes across directions/approaches for each unique timestamp to get the "Total Volume" for that intersection at that hour.
    t_sum = period_data.groupby("local_datetime", as_index=False)["total_volume"].sum()
    t_sum["hour"] = t_sum["local_datetime"].dt.hour
    
    # Then average those hourly totals across all days in the range to get a representative hourly pattern.
    hourly = t_sum.groupby("hour", as_index=False)["total_volume"].mean()
    hourly["Volume"] = hourly["total_volume"].fillna(0).round().astype(int)

    # Recommendations + Status
    hourly["CVAG Recommendation"] = hourly["Volume"].apply(get_hourly_cycle_length)
    hourly["Status"] = hourly["CVAG Recommendation"].apply(lambda rec: _get_status(rec, current_cycle))
    # Normal time format (12-hour) instead of military
    hourly["Hour"] = hourly["hour"].apply(lambda h: (datetime.min + timedelta(hours=h)).strftime("%I:%M %p").lstrip("0"))
    hourly["Rec (sec)"] = hourly["CVAG Recommendation"].apply(_sec_value)

    # ---------- NEW: add current cycle as its own column ----------
    hourly["System Current Cycle"] = current_cycle

    # --- KPI calculations ---
    total_hours = len(hourly)
    optimal_hours = int((hourly["Status"] == "🟢 OPTIMAL").sum())
    changes_needed = total_hours - optimal_hours

    # Hour integers for increases and reductions
    inc_hour_ids = hourly.loc[hourly["Status"] == "⬆️ INCREASE", "hour"].tolist()
    red_hour_ids = hourly.loc[hourly["Status"] == "🔽 REDUCE", "hour"].tolist()

    # Peak hour & volume in selected time period
    if len(hourly) > 0:
        peak_idx = hourly["Volume"].idxmax()
        peak_volume = int(hourly.loc[peak_idx, "Volume"])
        peak_hour = hourly.loc[peak_idx, "Hour"]
    else:
        peak_volume, peak_hour = 0, "N/A"

    # High-volume threshold KPI (based on raw rows in selected period)
    HIGH_VOLUME_THRESHOLD_VPH = 1200
    total_rows = int(period_data["total_volume"].count())
    high_rows = period_data.loc[period_data["total_volume"] > HIGH_VOLUME_THRESHOLD_VPH]
    high_hours = int(len(high_rows)) if total_rows > 0 else 0
    high_share = (high_hours / total_rows * 100) if total_rows > 0 else 0.0

    # Unique hour-of-day integer IDs that exceeded threshold
    exceed_hour_ids = sorted(high_rows["local_datetime"].dt.hour.unique().tolist()) if len(high_rows) else []

    # Peak capacity utilization
    INTERSECTION_CAPACITY_VPH = 1800
    peak_volume_pd = float(period_data["total_volume"].max()) if total_rows > 0 else 0.0
    peak_capacity_util = (peak_volume_pd / INTERSECTION_CAPACITY_VPH * 100) if INTERSECTION_CAPACITY_VPH else 0.0

    # -------------------------
    # BOXED KPIs (card grid)
    # -------------------------
    tone_peak = "warn" if peak_volume >= 1500 else ("good" if peak_volume >= 600 else "neutral")
    tone_changes = "good" if changes_needed == 0 else ("warn" if changes_needed <= (total_hours * 0.4) else "bad")
    tone_high = "bad" if high_share > 25 else ("warn" if high_share > 10 else "good")
    tone_util = "bad" if peak_capacity_util > 90 else ("warn" if peak_capacity_util > 75 else "good")

    # Retiming footer text organized cleanly
    if changes_needed > 0:
        if inc_hour_ids and red_hour_ids:
            retime_foot1 = f"Increase: {_format_hour_ranges(inc_hour_ids)}"
            retime_foot2 = f"Reduce: {_format_hour_ranges(red_hour_ids)}"
        elif inc_hour_ids:
            retime_foot1 = f"Increase: {_format_hour_ranges(inc_hour_ids)}"
            retime_foot2 = None
        else:
            retime_foot1 = f"Reduce: {_format_hour_ranges(red_hour_ids)}"
            retime_foot2 = None
    else:
        retime_foot1 = "All hours operating at optimal cycle"
        retime_foot2 = None

    card1 = _kpi_card(
        "Time Period",
        f"{total_hours} Hours",
        hours_window_str,
        "neutral",
        foot1=f"Active Window: {time_period}",
        tooltip="Total operational hours evaluated within the selected time window (AM, MD, PM, or All Day)."
    )
    card2 = _kpi_card(
        "Peak Hour",
        peak_hour,
        f"{peak_volume:,} Total Vehicles",
        tone_peak,
        foot1=f"Recommended: {get_hourly_cycle_length(peak_volume)}",
        foot2=f"Max demand in {time_period.split()[0]} window",
        tooltip="Peak traffic demand hour within the selected time period and its corresponding total vehicular volume."
    )
    card3 = _kpi_card(
        "Hours That Need Retiming",
        f"{changes_needed}",
        f"Increase: {len(inc_hour_ids)} | Reduce: {len(red_hour_ids)}",
        tone_changes,
        foot1=retime_foot1,
        foot2=retime_foot2,
        tooltip="Hours where vehicular demand warrants either an increase (longer cycle) or reduction (shorter cycle/Free mode) relative to current timing."
    )
    card4 = _kpi_card(
        "High Volume Hours above 1,200 vehicles",
        f"{high_hours}",
        f"{high_share:.1f}% of time",
        tone_high,
        foot1=f"Threshold: > {HIGH_VOLUME_THRESHOLD_VPH:,} vehicles",
        foot2=f"Exceeded at: {_format_hour_ranges(exceed_hour_ids)}",
        tooltip="Hours where total hourly volume exceeds 1,200 vehicles per hour (VPH), indicating high congestion and heavy coordination requirements."
    )
    card5 = _kpi_card(
        "Peak Capacity Utilization",
        f"{peak_capacity_util:.0f}%",
        f"Peak {int(peak_volume_pd):,} vehicles",
        tone_util,
        foot1=f"Capacity Benchmark: {INTERSECTION_CAPACITY_VPH:,} vehicles",
        foot2=f"Demand Ratio: {peak_volume_pd / INTERSECTION_CAPACITY_VPH:.2f}x benchmark" if INTERSECTION_CAPACITY_VPH else "",
        tooltip="Peak hourly volume as a percentage of nominal intersection capacity (1,800 VPH threshold)."
    )

    cards_html = f'<div class="cvag-kpi-grid">{card1}{card2}{card3}{card4}{card5}</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    # -------------------------
    # Chart (Boxed Card Layout)
    # -------------------------
    with st.container(border=True):
        # Bar chart colored by recommended cycle (palette + patterns + outlines)
        if start_dt == end_dt:
            date_display = start_dt.strftime('%B %d, %Y').replace(" 0", " ")
        else:
            date_display = f"{start_dt.strftime('%b %d')} to {end_dt.strftime('%b %d, %Y')}"
        
        chart_title = f"{sel_dir_full} Total Vehicle Volume Per Hour w/ Recommended Cycle Length (Bar Chart) — {date_display}"
        
        if corridor_name:
            int_subtitle = f"{corridor_name} Corridor -- {intersection_label}"
        else:
            int_subtitle = intersection_label

        # 12-hour format hour list for category_orders
        hour_labels_12h = [(datetime.min + timedelta(hours=h)).strftime("%I:%M %p").lstrip("0") for h in range(24)]

        # 3) Cycle length under the intersection name
        full_title = f"{chart_title}<br><span style='font-size:14px; color:#666;'>{int_subtitle}<br>Current Cycle: {current_cycle}</span>"

        fig = px.bar(
            hourly.sort_values("hour"),
            x="Hour",
            y="Volume",
            color="CVAG Recommendation",
            color_discrete_map=CYCLE_COLORS,
            category_orders={"CVAG Recommendation": CYCLE_ORDER, "Hour": hour_labels_12h},
            title=full_title,
            labels={"Volume": "Total Volume (vehicles)", "Hour": "Hour of Day"},
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
                    color=[_ for _ in (STATUS_COLORS[s] for s in hourly["Status"])],
                    symbol=[status_symbols[s] for s in hourly["Status"]],
                    line=dict(width=1, color="white"),
                ),
                name="Status",
                hovertemplate="<b>%{x}</b><br>Total Volume: %{y:,.0f} vehicles<extra></extra>",
            )
        )

        fig.update_layout(
            height=460,
            title=dict(text=full_title, x=0, xanchor="left", font=dict(color="#000000")),
            xaxis=dict(
                title=dict(text="Hour of Day", font=dict(size=15, weight="bold", color="#000000")),
                tickfont=dict(size=13, color="#000000"),
                tickangle=-45
            ),
            yaxis=dict(
                title=dict(text="Total Volume (vehicles)", font=dict(size=15, weight="bold", color="#000000")),
                tickfont=dict(size=13, color="#000000")
            ),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1.0,
                xanchor="left",
                x=1.01,
                font=dict(size=12, color="#000000"),
                bgcolor="rgba(255,255,255,0.90)",
                bordercolor="#cccccc",
                borderwidth=1,
                title=dict(text="Legend", font=dict(size=13, weight="bold", color="#000000"))
            ),
            margin=dict(l=10, r=140, t=110, b=80),
            bargap=0.18, 
            bargroupgap=0.05,
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="rgba(0,0,0,0.08)")
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # TABLE (with requested tweaks)
    # -------------------------
    # 1) Show "X vehicles" (string) for the total volume
    hourly["Volume (vehicles)"] = hourly["Volume"].apply(lambda v: f"{int(v):,} vehicles")

    # 2) Build display table with the new "System Current Cycle" column
    hourly_display = hourly[[
        "Hour",
        "Volume (vehicles)",                         # formatted text column
        "CVAG Recommendation",
        "System Current Cycle",                      # NEW column
        "Status"
    ]].rename(columns={
        "Volume (vehicles)": "Total Vehicle Volume (Throughs, lefts, and rights)",
        "CVAG Recommendation": "Cycle Length Recommendation For CVAG",
        "System Current Cycle": "System Current Cycle",
        "Status": "Cycle Length Status",
    })

    st.dataframe(
        hourly_display,
        use_container_width=True,
        column_config={
            "Hour": st.column_config.TextColumn("Hour", width="small"),
            # Text column because we include the word "vehicles"
            "Total Vehicle Volume (Throughs, lefts, and rights)": st.column_config.TextColumn(
                "Total Vehicle Volume (Throughs, lefts, and rights)"
            ),
            "Cycle Length Recommendation For CVAG": st.column_config.TextColumn(
                "Cycle Length Recommendation For CVAG", width="medium"
            ),
            "System Current Cycle": st.column_config.TextColumn(
                "System Current Cycle", width="medium"
            ),
            "Cycle Length Status": st.column_config.TextColumn(
                "Cycle Length Status", width="medium"
            ),
        },
    )

    # -------------------------
    # Insights + download
    # -------------------------
    if len(hourly):
        peak_volume = int(hourly["Volume"].max())
        peak_hour = hourly.loc[hourly["Volume"].idxmax(), "Hour"]
    else:
        peak_volume, peak_hour = 0, "—"

    st.markdown(
        f'<div class="insight-box" style="margin-top:.5rem;">'
        f'<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">'
        f'<h4 style="margin:0; font-weight:800; color:#0f2f52;">Cycle Length Optimization Insights</h4>'
        f'<div class="cvag-tooltip-trigger" title="Automated summary of signal timing performance, efficiency, peak volume demand, and retiming recommendations.">?'
        f'<span class="cvag-tooltip-box" style="width:260px; right:0; left:auto; transform:none;">Summary of signal timing optimization results and recommendations for CVAG retiming studies.</span>'
        f'</div>'
        f'</div>'
        f'<p><strong>System Efficiency:</strong> {optimal_hours}/{total_hours} hours optimal ({(optimal_hours/total_hours*100 if total_hours else 0):.0f}%) matching active {current_cycle} cycle.</p>'
        f'<p><strong>Volume Profile:</strong> Peak {peak_volume:,} vehicles at {peak_hour} • High-volume threshold exceedance: {high_hours} hours ({high_share:.1f}% of evaluated period).</p>'
        f'<p><strong>Actions Required:</strong> {len(inc_hour_ids)} hours warrant longer cycles • {len(red_hour_ids)} hours warrant shorter cycles or Free mode.</p>'
        f'<p><strong>Capacity Benchmark:</strong> Peak utilization {peak_capacity_util:.0f}% of intersection benchmark capacity ({INTERSECTION_CAPACITY_VPH:,} vehicles/hr).</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # CSV download: keep a clean numeric column for analysis along with display strings
    download_df = hourly[["Hour", "Volume"]].rename(columns={"Volume": "Total_Volume_vph"}).merge(
        hourly_display, on="Hour", how="right"
    )
    st.download_button(
        "Download Cycle Length Analysis (CSV)",
        data=download_df.to_csv(index=False).encode("utf-8"),
        file_name=f"cycle_length_recommendations_{selected_period.lower()}.csv",
        mime="text/csv",
        key=f"{key_prefix}_download",
    )
