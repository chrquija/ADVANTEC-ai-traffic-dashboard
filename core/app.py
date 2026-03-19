import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import streamlit.components.v1 as components
import contextlib

# --- make sure both /core and the project root are importable ---
import sys, pathlib

_THIS_DIR = pathlib.Path(__file__).resolve().parent  # .../core
_PROJECT_ROOT = _THIS_DIR.parent  # repo root
# prepend so they’re searched first
sys.path[:0] = [str(_THIS_DIR), str(_PROJECT_ROOT)]

# Plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === External project functions ===
from sidebar_functions import (
    load_traffic_data,
    load_volume_data,
    process_traffic_data,
    get_corridor_df,
    get_volume_df,
    get_acyclica_df,
    get_performance_rating,
    performance_chart,
    date_range_preset_controls,
    compute_perf_kpis_interpretable,
    render_badge,
    create_5min_data,
    compute_data_availability,
)

# Cycle length section
from cycle_length_recommendations import render_cycle_length_section

# Map builders (return Plotly figures)
from Map import build_corridor_map, build_intersection_map, build_intersections_overview, get_node_coordinates, \
    INTERSECTION_TO_NODE

# --- Acyclica (Tab 3) ---
try:
    from acyclica_traveltime_analysis import render_tab3_analysis  # sibling import
except ModuleNotFoundError:
    from core.acyclica_traveltime_analysis import render_tab3_analysis  # package import

# AI Prediction Models (Tab 3 sub-sections)
from Prediction.peak_hour_prediction import render_peak_hour_section
from Prediction.incident_detection import render_incident_detection_section
from Prediction.event_impact_analysis import render_event_impact_section

# --- VantageLive (Tab 4) ---
try:
    from VantageLivetab import render_vantage_tab  # sibling import
except ModuleNotFoundError:
    from core.VantageLivetab import render_vantage_tab  # package import

# --- Bosch (Tab 5) ---
try:
    from Boschtab import render_bosch_tab  # sibling import
except ModuleNotFoundError:
    from core.Boschtab import render_bosch_tab  # package import

# --- Shared UI utils for scoped loader and highlights ---
try:
    from ui_utils import cad_loader as scoped_cad_loader, set_active_search_tab, is_active_tab, get_dynamic_xaxis_params
except ModuleNotFoundError:
    from core.ui_utils import cad_loader as scoped_cad_loader, set_active_search_tab, is_active_tab, get_dynamic_xaxis_params

# -----------------------------
# URL state hydration utilities
# -----------------------------
from datetime import date as _date_type


def _str_to_date(s: str | None) -> _date_type | None:
    try:
        return _date_type.fromisoformat(s) if s else None
    except Exception:
        return None


# Hydrate session_state from URL query params on first load
if "url_hydrated" not in st.session_state:
    try:
        qp = st.query_params

        # Tab 1 (Performance & Prediction)
        if qp.get("t1_ready") == "1":
            t1_params = {
                "od_mode": qp.get("t1_od") == "1",
                "origin": qp.get("t1_origin") or None,
                "destination": qp.get("t1_destination") or None,
                "date_range": (
                    _str_to_date(qp.get("t1_date_start")),
                    _str_to_date(qp.get("t1_date_end")),
                ) if qp.get("t1_date_start") and qp.get("t1_date_end") else None,
                "granularity": qp.get("t1_granularity") or "Hourly",
                "time_filter": qp.get("t1_time_filter") or None,
                "start_hour": int(qp.get("t1_start_hour")) if qp.get("t1_start_hour") else None,
                "end_hour": int(qp.get("t1_end_hour")) if qp.get("t1_end_hour") else None,
            }
            st.session_state["t1_params"] = t1_params
            st.session_state["t1_ready"] = True

        # Tab 2 (Volume)
        if qp.get("t2_ready") == "1":
            t2_params = {
                "corridor": qp.get("t2_corridor") or "All Corridors",
                "intersection": qp.get("t2_intersection") or "All Intersections",
                "date_range_vol": (
                    _str_to_date(qp.get("t2_date_start")),
                    _str_to_date(qp.get("t2_date_end")),
                ) if qp.get("t2_date_start") and qp.get("t2_date_end") else None,
                "granularity_vol": qp.get("t2_granularity") or "Hourly",
                "direction_filter": qp.get("t2_direction") or "All Directions",
            }
            st.session_state["t2_params"] = t2_params
            st.session_state["t2_ready"] = True

        # Tab 3 (Acyclica)
        if qp.get("t3_ready") == "1":
            t3_params = {
                "corridor": qp.get("t3_corridor") or "All Corridors",
                "origin": qp.get("t3_origin") or "SELECT",
                "destination": qp.get("t3_destination") or "SELECT",
                "date_range": (
                    _str_to_date(qp.get("t3_date_start")),
                    _str_to_date(qp.get("t3_date_end")),
                ) if qp.get("t3_date_start") and qp.get("t3_date_end") else None,
                "granularity": qp.get("t3_granularity") or "Hourly",
                "direction_filter": qp.get("t3_direction") or "All Directions",
                "time_filter": qp.get("t3_time_filter") or None,
                "start_hour": int(qp.get("t3_start_hour")) if qp.get("t3_start_hour") else None,
                "end_hour": int(qp.get("t3_end_hour")) if qp.get("t3_end_hour") else None,
            }
            st.session_state["t3_params"] = t3_params
            st.session_state["t3_ready"] = True

        # Tab 4 (VantageLive)
        if qp.get("t4_ready") == "1":
            t4_params = {
                "mode": qp.get("t4_mode") or "Vehicles",
                "intersection": qp.get("t4_intersection") or "All Intersections",
                "date_range": (
                    _str_to_date(qp.get("t4_date_start")),
                    _str_to_date(qp.get("t4_date_end")),
                ) if qp.get("t4_date_start") and qp.get("t4_date_end") else None,
                "granularity": qp.get("t4_granularity") or "Daily",
                "direction_filter": qp.get("t4_direction") or "All Directions",
                "turn_filter": qp.get("t4_turn") or "All Turns",
                "chart_type": qp.get("t4_chart") or "Trend (Line)",
            }
            st.session_state["vantage_params"] = t4_params
            st.session_state["vantage_ready"] = True

        # Tab 5 (Bosch)
        if qp.get("t5_ready") == "1":
            # modes could be comma-joined
            modes_s = qp.get("t5_modes") or ""
            modes = tuple([m for m in modes_s.split(",") if m]) if modes_s else ()
            t5_params = {
                "corridor": qp.get("t5_corridor") or "SELECT",
                "date_range": (
                    _str_to_date(qp.get("t5_date_start")),
                    _str_to_date(qp.get("t5_date_end")),
                ) if qp.get("t5_date_start") and qp.get("t5_date_end") else None,
                "granularity": qp.get("t5_granularity") or "Hourly",
                "mode_filter": modes,
            }
            st.session_state["bosch_params"] = t5_params
            st.session_state["bosch_ready"] = True

        # Active tab to open
        last_tab = qp.get("last_tab")
        if last_tab:
            st.session_state["last_active_tab"] = last_tab

    except Exception:
        # Fail-safe: do nothing on hydration errors
        pass
    finally:
        st.session_state["url_hydrated"] = True

# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="Active Transportation & Operations Management Dashboard",
    page_icon="🛣️",
    layout="wide",  # Keep wide layout enabled throughout the app (including after sign-in)
    initial_sidebar_state="expanded",
)

# --- Authentication gate ---
try:
    from auth import require_company_login, render_auth_sidebar_footer
except ModuleNotFoundError:
    from core.auth import require_company_login, render_auth_sidebar_footer

# Only allow @advantec-usa.com emails
if not require_company_login("advantec-usa.com"):
    st.stop()


# With the app configured to wide layout globally, no CSS hack is needed here.

# -------- CAD-style loader (progress bar + step text) --------
@contextlib.contextmanager
def cad_loader(title: str = "Processing…"):
    """Progress loader that logs steps like CAD/Civil tools."""
    # Create placeholders that we can clear later
    title_placeholder = st.empty()
    log_placeholder = st.empty()
    bar_placeholder = st.empty()

    # Show initial state
    with title_placeholder:
        st.markdown(f"### {title}")

    log_container = log_placeholder.container()
    progress_bar = bar_placeholder.progress(0)

    def step(msg: str, pct: int | float):
        with log_container:
            st.write(f"• {msg}")
        progress_bar.progress(int(max(0, min(100, pct))))

    try:
        yield step
        progress_bar.progress(100)
        with log_container:
            st.success("✔️ Done")

        # Wait briefly to show completion, then clear everything
        time.sleep(0.5)
        title_placeholder.empty()
        log_placeholder.empty()
        bar_placeholder.empty()

    except Exception as e:
        with log_container:
            st.error(f"❌ {e}")
        raise


# Plotly UI tweaks + default map height
PLOTLY_CONFIG = {
    "displaylogo": False,
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "toggleSpikelines"]
}
MAP_HEIGHT = 1100  # default map height (px) for the right rail

# =========================
# Constants / Config
# =========================
THEORETICAL_LINK_CAPACITY_VPH = 1800
HIGH_VOLUME_THRESHOLD_VPH = 1200
CRITICAL_DELAY_SEC = 120
HIGH_DELAY_SEC = 1

DIRECTION_COLORS = {
    "EB": "#1f77b4",  # Blue
    "NB": "#9edae5",  # Light Blue
    "SB": "#d62728",  # Red
    "WB": "#ff9896",  # Pink
    "Eastbound": "#1f77b4",
    "Northbound": "#9edae5",
    "Southbound": "#d62728",
    "Westbound": "#ff9896",
}

# Canonical bottom → top node order (ensure labels match your dataset exactly)
DESIRED_NODE_ORDER_BOTTOM_UP = [
    "Avenue 52",
    "Calle Tampico",
    "Village Shopping Ctr",
    "Avenue 50",
    "Sagebrush Ave",
    "Eisenhower Dr",
    "Avenue 48",
    "Avenue 47",
    "Point Happy Simon",
    "Hwy 111",
    # New northern intersections (extend corridor order)
    "Channel Drive",
    "Miles Avenue",
    "Via Sevilla",
    "Fred Waring Drive",
    "Palm Royale Drive",
    "Avenue of the States",
    "Avenue 42",
    "Avenue 41",
    "Harris Lane",
    "Country Club Drive",
    "I-10 Interchange",
    "Varner Road",
    "Market Pl",
    "Del Webb",
]


# Build ordered node list from segment_name like "A → B"
def _build_node_order(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty or "segment_name" not in df.columns:
        return []
    segs = df["segment_name"].dropna().tolist()
    order: list[str] = []
    for s in segs:
        parts = [p.strip() for p in s.split("→")]
        if len(parts) != 2:
            continue
        a, b = parts[0], parts[1]
        if not order:
            order.append(a)
            order.append(b)
        else:
            if order[-1] == a:
                order.append(b)
            elif a not in order and b not in order:
                order.append(a)
                order.append(b)
    # de-duplicate preserving order
    seen, out = set(), []
    for n in order:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


# -------- Canonical helpers (used for robust O-D path building) --------
def _nodes_present_in_data(df: pd.DataFrame) -> set:
    """All node labels that appear in any 'A → B' segment_name."""
    if "segment_name" not in df.columns or df.empty:
        return set()
    parts = df["segment_name"].dropna().str.split("→")
    left = parts.apply(lambda x: x[0].strip() if isinstance(x, list) and len(x) == 2 else None)
    right = parts.apply(lambda x: x[1].strip() if isinstance(x, list) and len(x) == 2 else None)
    return set(pd.concat([left, right], ignore_index=True).dropna().unique())


def _canonical_order_in_data(df: pd.DataFrame) -> list[str]:
    """Canonical corridor order, restricted to nodes that actually exist in the Prediction."""
    present = _nodes_present_in_data(df)
    return [n for n in DESIRED_NODE_ORDER_BOTTOM_UP if n in present]


# =========================
# Robust direction normalization (string-only)
# =========================
def normalize_dir(s: pd.Series) -> pd.Series:
    """
    Vectorized normalizer returning only 'nb', 'sb', or 'unk' (dtype=object).
    Safe for mixed dtype inputs; never returns NaN.
    """
    ser = s.astype(str).str.lower().str.strip()
    ser = ser.str.replace(r"[\s\-\(\)_/\\]+", " ", regex=True)
    nb_mask = ser.str.contains(r"\b(?:nb|north|northbound)\b")
    sb_mask = ser.str.contains(r"\b(?:sb|south|southbound)\b")
    return pd.Series(
        np.where(nb_mask, "nb", np.where(sb_mask, "sb", "unk")),
        index=ser.index,
        dtype="object",
    )


def normalize_dir_value(v) -> str:
    """Scalar helper if ever needed; string-only returns."""
    if v is None:
        return "unk"
    try:
        s = str(v).lower().strip()
    except Exception:
        return "unk"
    s = " ".join([tok for tok in s.replace("-", " ").replace("_", " ").split()])
    if any(t in s for t in [" nb", "nb ", " northbound", " north "]):
        return "nb"
    if any(t in s for t in [" sb", "sb ", " southbound", " south "]):
        return "sb"
    return "unk"


# =========================
# Extra CSS (includes a robust sticky-right-rail implementation)
# =========================
st.markdown("""
<style>
    /* Cards / layout polish */
    .main-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 15px; padding: 2rem; margin: 1rem 0; color: white;
        box-shadow: 0 8px 32px rgba(30, 60, 114, 0.3);
    }
    .context-header {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem; border-radius: 15px; margin: 1rem 0 2rem; color: white; text-align: center;
        box-shadow: 0 8px 32px rgba(79, 172, 254, 0.3); backdrop-filter: blur(10px);
    }
    .context-header h2 { margin: 0; font-size: 2rem; font-weight: 700; }
    .context-header p  { margin: 1rem 0 0; font-size: 1.1rem; opacity: 0.9; font-weight: 300; }

    .insight-box {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.15) 0%, rgba(0, 242, 254, 0.15) 100%);
        border-left: 5px solid #4facfe; border-radius: 12px; padding: 1.25rem 1.5rem; margin: 1.25rem 0;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.1);
    }
    .insight-box h4 { color: #1e3c72; margin-top: 0; font-weight: 600; }

    .performance-badge { display: inline-block; padding: 0.35rem 0.9rem; border-radius: 25px; font-size: 0.85rem;
        font-weight: 600; margin: 0.2rem; border: 2px solid transparent; transition: all 0.3s ease; }
    .performance-badge:hover { transform: scale(1.05); border-color: rgba(255,255,255,0.25); }
    .badge-excellent { background: linear-gradient(45deg, #2ecc71, #27ae60); color: white; }
    .badge-good      { background: linear-gradient(45deg, #3498db, #2980b9); color: white; }
    .badge-fair      { background: linear-gradient(45deg, #f39c12, #e67e22); color: white; }
    .badge-poor      { background: linear-gradient(45deg, #e74c3c, #8e44ad); color: white; }
    .badge-critical  { background: linear-gradient(45deg, #e74c3c, #8e44ad); animation: pulse 2s infinite; }
    @keyframes pulse { 0% {opacity:1} 50% {opacity:.7} 100% {opacity:1} }

    .stTabs [Prediction-baseweb="tab-list"] { gap: 16px; }
    .stTabs [Prediction-baseweb="tab"] { height: 56px; padding: 0 18px; border-radius: 12px;
        background: rgba(79, 172, 254, 0.1); border: 1px solid rgba(79, 172, 254, 0.2); }

    /* ==========================================================
       Sticky Right Rail that actually works with Streamlit.
       ========================================================== */
    :root { --cvag-rail-top: 5.6rem; } /* top offset (enough to clear headers) */

    [Prediction-testid="column"]:has(#od-map-anchor),
    [Prediction-testid="column"]:has(#vol-map-anchor) {
        position: sticky;
        top: var(--cvag-rail-top);
        align-self: flex-start;
        z-index: 1;
    }

    .cvag-map-card {
        background: rgba(79,172,254,0.06);
        border: 1px solid rgba(79,172,254,0.18);
        border-radius: 12px;
        padding: 10px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    }

    @media (max-width: 1100px) {
        [Prediction-testid="column"]:has(#od-map-anchor),
        [Prediction-testid="column"]:has(#vol-map-anchor) {
            position: static;
            top: auto;
        }
    }
</style>
""", unsafe_allow_html=True)


# =========================
# Session state helpers for "Search to commit"
# =========================
def _init_state():
    ss = st.session_state
    ss.setdefault("t1_ready", False)  # Tab 1: results committed?
    ss.setdefault("t2_ready", False)  # Tab 2: results committed?
    ss.setdefault("t1_params", {})  # Tab 1: committed parameters
    ss.setdefault("t2_params", {})  # Tab 2: committed parameters
    ss.setdefault("t1_current", {})  # Tab 1: current (uncommitted) values
    ss.setdefault("t2_current", {})  # Tab 2: current (uncommitted) values


def _freeze_params(d: dict) -> dict:
    """Normalize params for lightweight equality checks (esp. date_range)."""
    if not isinstance(d, dict):
        return {}
    out = dict(d)
    if "date_range" in out and isinstance(out["date_range"], (list, tuple)) and len(out["date_range"]) == 2:
        out["date_range"] = (str(out["date_range"][0]), str(out["date_range"][1]))
    if "date_range_vol" in out and isinstance(out["date_range_vol"], (list, tuple)) and len(out["date_range_vol"]) == 2:
        out["date_range_vol"] = (str(out["date_range_vol"][0]), str(out["date_range_vol"][1]))
    if "date_range_comp" in out and isinstance(out["date_range_comp"], (list, tuple)) and len(out["date_range_comp"]) == 2:
        out["date_range_comp"] = (str(out["date_range_comp"][0]), str(out["date_range_comp"][1]))
    return out


_init_state()

# =========================
# Title / Intro
# =========================
st.markdown("""
<div class="main-container">
    <h1 style="text-align:center; margin:0; font-size:2.5rem; font-weight:800;">
         ADVANTEC CLOUD ANALYTICS
    </h1>
    <p style="text-align:center; margin-top:1rem; font-size:1.1rem; opacity:0.9;">
        Powered By Data. Driven By You. 
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    font-size: 1.05rem; font-weight: 400; color: var(--text-color);
    background: linear-gradient(135deg, rgba(79, 172, 254, 0.1), rgba(0, 242, 254, 0.05));
    padding: 1.5rem; border-radius: 18px; box-shadow: 0 8px 32px rgba(79,172,254,0.08);
    margin: 1.25rem 0; line-height: 1.7; border: 1px solid rgba(79,172,254,0.2); backdrop-filter: blur(8px);
">
    <div style="text-align:center; margin-bottom: 0.8rem;">
            <strong style="font-size: 1.2rem; color: #2980b9;">🌎 Active Transportation & Operations Management Dashboard</strong>
        </div>
        <p>Synthesizes intelligence from <strong>Iteris ClearGuide, Q-Free Kinetic Mobility, FLIR Acyclica, Iteris VantageLive, and Bosch Cloud Analytics</strong>.</p>
        <p>Leveraging millions of data points trained on advanced <strong>Machine Learning algorithms</strong>, this platform optimizes traffic flow, reduces travel time, minimizes fuel consumption, and decreases greenhouse gas emissions across the transportation network.</p>
        <p style="margin-top: 0.8rem; font-size: 0.95rem; opacity: 0.8;"><strong>Key Technologies:</strong> Real-time Anomaly Detection • Intelligent Cycle Length Optimization • Predictive Traffic Modeling • Performance Analytics • Dashboard Generator</p>
    </div>
    """, unsafe_allow_html=True)

# --- Professional "Research Questions" expander (custom-styled, blue-gradient theme) ---
# Scoped CSS for the next expander only
st.markdown(
    """
    <style>
      /* Wrap-scoped styles: apply only inside .exp-pro container */
      .exp-pro [data-testid="stExpander"] > details {
        border: 1px solid rgba(41,128,185,.35);
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 8px 26px rgba(41,128,185,.18);
        background: linear-gradient(135deg, rgba(79,172,254,0.08), rgba(0,242,254,0.06));
      }
      .exp-pro [data-testid="stExpander"] summary {
        list-style: none;
        padding: 14px 16px;
        font-weight: 700;
        color: #0b2538;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-bottom: 1px solid rgba(255,255,255,.25);
      }
      .exp-pro [data-testid="stExpander"] summary::-webkit-details-marker { display: none; }
      .exp-pro [data-testid="stExpander"] summary:hover {
        filter: brightness(0.98);
      }
      .exp-pro [data-testid="stExpander"] > details > div {
        padding: 1rem 1.25rem 1.25rem 1.25rem;
        background: linear-gradient(180deg, rgba(255,255,255,.85), rgba(255,255,255,.95));
      }
      @media (prefers-color-scheme: dark) {
        .exp-pro [data-testid="stExpander"] > details {
          border-color: rgba(79,172,254,.45);
          background: linear-gradient(135deg, rgba(79,172,254,0.12), rgba(0,242,254,0.10));
        }
        .exp-pro [data-testid="stExpander"] summary {
          color: #e6f2ff;
          border-bottom-color: rgba(255,255,255,.18);
        }
        .exp-pro [data-testid="stExpander"] > details > div {
          background: linear-gradient(180deg, rgba(10,20,35,.85), rgba(10,20,35,.90));
        }
      }
    </style>
    <div class="exp-pro">
    """,
    unsafe_allow_html=True,
)

with st.expander("Top 10 Questions This Dashboard Answers", expanded=False):
    st.markdown("Use these questions to guide your analysis across the specific tabs.")

    q_col1, q_col2 = st.columns(2, gap="medium")

    with q_col1:
        st.markdown("#### Operations & Signal Timing")
        st.markdown("""
            1. **Is the current cycle length sufficient for the actual volume?**  
               *Go to: Tab 2 (Kinetic) → Cycle Length Recommendations*

            2. **Which intersections are operating near capacity (>1,800 vehicles per hour)?**  
               *Go to: Tab 2 (Kinetic) → Peak Capacity Utilization*

            3. **Do we need distinct signal plans for AM vs. PM peak?**  
               *Go to: Tab 2 (Kinetic) → Hourly Volume Heatmap*
            """)

        st.markdown("#### Capital Improvement & Prioritization")
        st.markdown("""
            4. **Which intersections have the highest 'Risk Score' for failure?**  
               *Go to: Tab 2 (Kinetic) → Risk Analysis Table*

            5. **Where are the most severe bottlenecks?**  
               *Go to: Tab 1 (Iteris) → Bottleneck Analysis*
            """)

    with q_col2:
        st.markdown("#### System Performance")
        st.markdown("""
            6. **How much 'Buffer Time' is required for reliability?**  
               *Go to: Tab 1 (Iteris) → Buffer Time KPI*

            7. **Which direction (NB vs. SB) is the primary constraint?**  
               *Go to: Tab 3 (Acyclica) → Corridor Performance Analysis*

            8. **Are we seeing non-recurrent congestion (incidents)?**  
               *Go to: Tab 3 (Acyclica) → Incident Detection*
            """)

        st.markdown("#### Data QA/QC")
        st.markdown("""
            9. **Do we have full coverage for the selected date ranges (days/hours), or are there gaps large enough to bias conclusions?**  
               *No, we do not have full coverage. Whenever you select a corridor, the system runs a compute_data_availability check and lists any interruptions in the dataset under the "Missing Data" Caption in the sidebar.*
               *Visual "Shaded Bands" on charts can also detect time intervals where data is missing.*

            10. **Is there an Microsoft Excel Matrix detailing CVAG Phase 1 data availability by intersection?**  
                 *Download/View: [CVAG Phase 1 Data Matrix](https://advantecusa.sharepoint.com/:x:/s/IRVINEOFFICE/IQCr0Fw9pZbhSpNLLAUTn8FoAdqTg-lppit5kGB1jbbH_4g?e=Fycame&nav=MTVfezVDNTJGMzg2LUY2OTctNEQ3OS1BQUZGLUUyNDc4QTM5RDJBNn0)*
                """)

# Close the scoped wrapper div for the styled Research Questions expander
st.markdown("</div>", unsafe_allow_html=True)

# 📘 Project Deliverables & Resources (Separate Expander)
with st.expander("Resources & Deliverables", expanded=False):
    st.markdown("Access the official project documentation and presentation materials.")
    r_col1, r_col2 = st.columns(2, gap="medium")
    with r_col1:
        st.link_button(
            "📊 View Microsoft PowerPoint Presentation",
            "https://advantecusa-my.sharepoint.com/:b:/g/personal/cquijano_advantec-usa_com/IQBrgXkvXkbXRqxjtO9NKhtWARWfPQpBrL5Zi-btVldXCMQ?e=DYZHOg",
            use_container_width=True
        )
    with r_col2:
        st.link_button(
            "📄 View Full Project Report (PDF)",
            "https://advantecusa-my.sharepoint.com/:b:/g/personal/cquijano_advantec-usa_com/IQC1wK1jTP6yRoQ2Ui3pwQ4hARv8aWBNiqxeDY6YA8gdUZU?e=QLmSYi",
            use_container_width=True
        )

# =========================
# --------- NEW TAB 2 HELPERS (aggregation-aware) ----------
# =========================

AGG_META = {
    "Hourly": {"unit": "vehicles", "bucket": "H", "label": "hour", "fixed_hours": 1},
    "Daily": {"unit": "vehicles", "bucket": "D", "label": "day", "fixed_hours": 24},
    "Weekly": {"unit": "vehicles", "bucket": "W", "label": "week", "fixed_hours": 24 * 7},
    "Monthly": {"unit": "vehicles", "bucket": "M", "label": "month", "fixed_hours": None},  # varies by month
}


def _prep_bucket(df: pd.DataFrame, granularity: str, group_cols: list = None) -> pd.DataFrame:
    """
    Aggregate hourly records to the selected bucket (sum of hourly volumes).
    Returns: df with columns [local_datetime, ..., total_volume, bucket_hours].
    """
    if df.empty:
        return df.copy()

    if group_cols is None:
        group_cols = ["intersection_name"]

    g = granularity
    meta = AGG_META[g]
    d = df.copy()
    d["local_datetime"] = pd.to_datetime(d["local_datetime"])

    if g == "Hourly":
        d["bucket"] = d["local_datetime"].dt.floor("H")
    elif g == "Daily":
        d["bucket"] = d["local_datetime"].dt.floor("D")
    elif g == "Weekly":
        d["bucket"] = d["local_datetime"].dt.to_period("W").dt.start_time
    else:  # Monthly
        d["bucket"] = d["local_datetime"].dt.to_period("M").dt.start_time

    agg = (
        d.groupby(["bucket"] + group_cols, as_index=False)
        .agg(total_volume=("total_volume", "sum"))
        .rename(columns={"bucket": "local_datetime"})
    )

    # Hours in the bucket (for capacity/threshold scaling)
    if g == "Monthly":
        agg["bucket_hours"] = pd.to_datetime(agg["local_datetime"]).dt.days_in_month * 24
    else:
        agg["bucket_hours"] = meta["fixed_hours"]
    return agg


def _cap_series_for_x(x_df: pd.DataFrame, cap_vph: float, high_vph: float) -> pd.DataFrame:
    """Build capacity/threshold series aligned to unique time buckets only.
    Robust against duplicated rows from multiple directions/approaches.
    """
    if x_df.empty:
        return pd.DataFrame(columns=["local_datetime", "bucket_hours", "capacity", "high"]) 

    # Ensure we have a single row per unique timestamp. If multiple rows share the
    # same `local_datetime` (e.g., different directions), take the first bucket_hours
    # which is constant per bucket granularity.
    xs = (
        x_df[["local_datetime", "bucket_hours"]]
        .copy()
        .sort_values("local_datetime")
    )
    try:
        xs = xs.groupby("local_datetime", as_index=False)["bucket_hours"].first()
    except Exception:
        # Fallback to previous behavior if groupby fails for any reason
        xs = xs.drop_duplicates(subset=["local_datetime", "bucket_hours"]) 

    xs["capacity"] = xs["bucket_hours"].astype(float) * float(cap_vph)
    xs["high"] = xs["bucket_hours"].astype(float) * float(high_vph)
    return xs


def _fmt_period(ts: pd.Timestamp, granularity: str) -> str:
    ts = pd.to_datetime(ts)
    if granularity == "Hourly":
        return ts.strftime("%b %d, %Y %H:%M")
    if granularity == "Daily":
        return ts.strftime("%b %d, %Y")
    if granularity == "Weekly":
        wk = ts.to_period("W")
        return f"Week of {wk.start_time.strftime('%b %d, %Y')}"
    return ts.strftime("%b %Y")


def improved_volume_charts_for_tab2(
        raw_hourly_df: pd.DataFrame,
        granularity: str,
        cap_vph: float,
        high_vph: float,
        direction_label: str = "",
        intersections: str = "",
        date_range: tuple = None,
        top_k: int = 15,
        all_directions_df: pd.DataFrame = None,
        show_all_approaches: bool = False,
        shade_periods: bool = False,
        comp_hourly_df: pd.DataFrame = None,
        comp_date_range: tuple = None,
        comp_all_directions_df: pd.DataFrame = None,
):
    """
    Returns (fig_trend, fig_box, fig_matrix)
    - fig_trend: Time series per intersection with scaled capacity/threshold overlays.
    - fig_box:   Distribution of bucket totals by intersection.
    - fig_matrix: Average bucket total by intersection (ranking).
    """
    if raw_hourly_df.empty:
        return None, None, None

    # Aggregate to the selected bucket
    if show_all_approaches and all_directions_df is not None:
        # Use all directions data for comparison
        agg = _prep_bucket(all_directions_df, granularity, group_cols=["intersection_name", "direction"])
        # If comparing multiple intersections, collapse directions so each intersection has a single series
        if agg.get("intersection_name") is not None and agg["intersection_name"].nunique() > 1:
            agg = (
                agg.groupby(["local_datetime", "intersection_name"], as_index=False)
                   .agg(total_volume=("total_volume", "sum"),
                        bucket_hours=("bucket_hours", "first"))
            )
    else:
        agg = _prep_bucket(raw_hourly_df, granularity)

    agg_comp = None
    if comp_hourly_df is not None and not comp_hourly_df.empty:
        if show_all_approaches and comp_all_directions_df is not None:
            agg_comp = _prep_bucket(comp_all_directions_df, granularity, group_cols=["intersection_name", "direction"])
            if agg_comp.get("intersection_name") is not None and agg_comp["intersection_name"].nunique() > 1:
                agg_comp = (
                    agg_comp.groupby(["local_datetime", "intersection_name"], as_index=False)
                       .agg(total_volume=("total_volume", "sum"),
                            bucket_hours=("bucket_hours", "first"))
                )
        else:
            agg_comp = _prep_bucket(comp_hourly_df, granularity)

    if agg.empty and (agg_comp is None or agg_comp.empty):
        return None, None, None

    unit = AGG_META[granularity]["unit"]
    label = AGG_META[granularity]["label"]
    # Aggregation prefix for titles: Hourly -> "Hourly", Daily -> "Daily", etc.
    agg_prefix = granularity

    # Limit to top intersections by mean demand
    if "direction" in agg.columns:
        # Ranking based on total volume across all directions for each intersection
        ranking_df = agg.groupby(["local_datetime", "intersection_name"])["total_volume"].sum().reset_index()
        order = ranking_df.groupby("intersection_name")["total_volume"].mean().sort_values(ascending=False)
    else:
        order = agg.groupby("intersection_name")["total_volume"].mean().sort_values(ascending=False)
        
    keep = order.index[:max(1, min(top_k, len(order)))]

    plot_df = agg[agg["intersection_name"].isin(keep)].copy().sort_values("local_datetime")
    plot_df_comp = None
    if agg_comp is not None and not agg_comp.empty:
        plot_df_comp = agg_comp[agg_comp["intersection_name"].isin(keep)].copy().sort_values("local_datetime")

    # ---------- Trend ----------
    fig_trend = go.Figure()

    # Calculate offset for overlay if comparison is active
    offset = None
    p_dates_short = ""
    c_dates_short = ""
    if date_range and len(date_range) == 2:
        p_dates_short = f"{date_range[0].strftime('%b %d')} - {date_range[1].strftime('%b %d')}"
    
    if plot_df_comp is not None and date_range and comp_date_range:
        c_dates_short = f"{comp_date_range[0].strftime('%b %d')} - {comp_date_range[1].strftime('%b %d')}"
        try:
            offset = pd.to_datetime(date_range[0]) - pd.to_datetime(comp_date_range[0])
        except Exception:
            offset = None

    # Shade Time Periods (only for 1-day Hourly charts)
    if shade_periods and granularity == "Hourly" and date_range and len(date_range) == 2 and date_range[0] == date_range[1]:
        d_str = date_range[0].strftime("%Y-%m-%d")
        # AM (05:00-10:00)
        fig_trend.add_vrect(
            x0=f"{d_str} 05:00", x1=f"{d_str} 10:00",
            fillcolor="orange", opacity=0.1, layer="below", line_width=0,
            annotation_text="AM", annotation_position="top left"
        )
        # MD (11:00-15:00)
        fig_trend.add_vrect(
            x0=f"{d_str} 11:00", x1=f"{d_str} 15:00",
            fillcolor="green", opacity=0.1, layer="below", line_width=0,
            annotation_text="MD", annotation_position="top left"
        )
        # PM (16:00-20:00)
        fig_trend.add_vrect(
            x0=f"{d_str} 16:00", x1=f"{d_str} 20:00",
            fillcolor="red", opacity=0.1, layer="below", line_width=0,
            annotation_text="PM", annotation_position="top left"
        )

    mode = "lines+markers"
    xfmt = "%Y-%m-%d %I:%M %p" if granularity == "Hourly" else "%Y-%m-%d"

    if "direction" in plot_df.columns:
        for (name, dr), g in plot_df.groupby(["intersection_name", "direction"]):
            trace_name = f"{name} ({dr})" if len(keep) > 1 else dr
            if plot_df_comp is not None:
                trace_name += f" ({p_dates_short})"
            
            color = DIRECTION_COLORS.get(dr) or DIRECTION_COLORS.get(dr.upper())
            fig_trend.add_trace(
                go.Scatter(
                    x=g["local_datetime"],
                    y=g["total_volume"],
                    mode=mode,
                    name=trace_name,
                    marker=dict(size=6),
                    line=dict(color=color) if color else None,
                    hovertemplate=(
                        f"<b>%{{fullData.name}}</b><br>%{{x|{xfmt}}}<br>Volume: %{{y:,.0f}} {unit}<extra></extra>"),
                )
            )
        
        if plot_df_comp is not None and "direction" in plot_df_comp.columns:
            for (name, dr), g in plot_df_comp.groupby(["intersection_name", "direction"]):
                trace_name = f"{name} ({dr})" if len(keep) > 1 else dr
                trace_name += f" ({c_dates_short})"
                
                color = DIRECTION_COLORS.get(dr) or DIRECTION_COLORS.get(dr.upper())
                
                x_vals = g["local_datetime"]
                if offset is not None:
                    x_vals = x_vals + offset
                
                fig_trend.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=g["total_volume"],
                        mode=mode,
                        name=trace_name,
                        marker=dict(size=6, symbol="diamond-open"),
                        line=dict(color=color, dash="dash") if color else dict(dash="dash"),
                        customdata=g["local_datetime"].dt.strftime(xfmt),
                        hovertemplate=(
                            f"<b>%{{fullData.name}}</b><br>Actual: %{{customdata}}<br>Volume: %{{y:,.0f}} {unit}<extra></extra>"),
                    )
                )
    else:
        for name, g in plot_df.groupby("intersection_name"):
            trace_name = name
            if plot_df_comp is not None:
                trace_name += f" ({p_dates_short})"
            fig_trend.add_trace(
                go.Scatter(
                    x=g["local_datetime"],
                    y=g["total_volume"],
                    mode=mode,
                    name=trace_name,
                    marker=dict(size=6),
                    hovertemplate=(
                        f"<b>%{{fullData.name}}</b><br>%{{x|{xfmt}}}<br>Volume: %{{y:,.0f}} {unit}<extra></extra>"),
                )
            )
        
        if plot_df_comp is not None:
            for name, g in plot_df_comp.groupby("intersection_name"):
                x_vals = g["local_datetime"]
                if offset is not None:
                    x_vals = x_vals + offset
                
                fig_trend.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=g["total_volume"],
                        mode=mode,
                        name=f"{name} ({c_dates_short})",
                        marker=dict(size=6, symbol="diamond-open"),
                        line=dict(dash="dash"),
                        customdata=g["local_datetime"].dt.strftime(xfmt),
                        hovertemplate=(
                            f"<b>%{{fullData.name}}</b><br>Actual: %{{customdata}}<br>Volume: %{{y:,.0f}} {unit}<extra></extra>"),
                    )
                )

    xs = _cap_series_for_x(plot_df, cap_vph, high_vph)
    fig_trend.add_trace(
        go.Scatter(
            x=xs["local_datetime"], y=xs["capacity"],
            name=f"Theoretical Capacity ({cap_vph:,.0f} {unit})", mode="lines",
            line=dict(dash="dash"),
            hovertemplate=(f"%{{x|{xfmt}}}<br>Capacity: %{{y:,.0f}} {unit}<extra></extra>"),
        )
    )
    fig_trend.add_trace(
        go.Scatter(
            x=xs["local_datetime"], y=xs["high"],
            name=f"High Volume Threshold ({high_vph:,.0f} {unit})", mode="lines",
            line=dict(dash="dot"),
            hovertemplate=(f"%{{x|{xfmt}}}<br>Threshold: %{{y:,.0f}} {unit}<extra></extra>"),
        )
    )

    dir_title = "All Approaches" if show_all_approaches else direction_label
    if len(keep) == 1:
        if show_all_approaches:
            main_chart_title = f"All Approaches Vehicle Volume Per {label.capitalize()} by Approach (Line Chart)".strip()
        else:
            main_chart_title = f"{direction_label} Vehicle Volume Per {label.capitalize()} (Line Chart)".strip()
    else:
        main_chart_title = f"{dir_title} Vehicle Volume Per {label.capitalize()} by Intersection (Line Chart)".strip()

    date_subtitle = ""
    if date_range and len(date_range) == 2:
        date_subtitle = f"{date_range[0].strftime('%b %d, %Y')} to {date_range[1].strftime('%b %d, %Y')}"
    
    if comp_date_range and len(comp_date_range) == 2:
        date_subtitle += f" ({comp_date_range[0].strftime('%b %d, %Y')} to {comp_date_range[1].strftime('%b %d, %Y')})"
    
    chart_title_with_sub = f"<b><span style='color:black;'>{main_chart_title}</span></b>"
    if date_subtitle:
        chart_title_with_sub += f"<br><span style='font-size:15px; color:black;'>{date_subtitle}</span>"
    
    if intersections:
        # Use a secondary line for intersections if provided
        chart_title_with_sub += f"<br><span style='font-size:14px; color:#666;'>{intersections}</span>"

    # X-axis configuration
    xaxis_config = dict(
        title=dict(text="Date/Time", font=dict(size=16, weight="bold", color="black")),
        tickfont=dict(size=14, color="black"),
        type="date",
    )
    if granularity == "Hourly":
        # Calculate hourly ticks at 4-hour intervals that always include midnight
        # This allows us to show the date/day at midnight and just time otherwise
        if not plot_df.empty:
            min_ts = plot_df["local_datetime"].min()
            max_ts = plot_df["local_datetime"].max()
            # Start from the beginning of the first day to align midnight ticks
            start_range = min_ts.floor("D")
            end_range = max_ts.ceil("H")
            tick_vals = pd.date_range(start=start_range, end=end_range, freq="4H")
            # Filter ticks to only those within the data range (plus some padding if needed)
            tick_vals = [t for t in tick_vals if t >= min_ts - pd.Timedelta(hours=1) and t <= max_ts + pd.Timedelta(hours=1)]
            
            tick_text = []
            for t in tick_vals:
                if t.hour == 0 and t.minute == 0:
                    tick_text.append(t.strftime("<b>%A</b>\n<b>%b %d</b>"))
                else:
                    tick_text.append(t.strftime("%I:%M %p"))
            
            xaxis_config["tickvals"] = tick_vals
            xaxis_config["ticktext"] = tick_text
        else:
            xaxis_config["dtick"] = 14400000  # 4 hours in ms
            xaxis_config["tickformat"] = "%I:%M %p"
    elif granularity == "Daily":
        if not plot_df.empty:
            min_ts = plot_df["local_datetime"].min()
            max_ts = plot_df["local_datetime"].max()
            tick_vals = pd.date_range(start=min_ts.floor("D"), end=max_ts.ceil("D"), freq="D")
            tick_text = [t.strftime("<b>%A</b>\n<b>%b %d</b>\n<b>%Y</b>") for t in tick_vals]
            xaxis_config["tickvals"] = tick_vals
            xaxis_config["ticktext"] = tick_text
        else:
            xaxis_config["dtick"] = 86400000  # 1 day in ms
            xaxis_config["tickformat"] = "%A\n%b %d\n%Y"
    elif granularity == "Weekly":
        if not plot_df.empty:
            min_ts = plot_df["local_datetime"].min()
            max_ts = plot_df["local_datetime"].max()
            # Align to start of week (Sunday usually)
            start_range = min_ts - pd.Timedelta(days=min_ts.weekday() + 1 if min_ts.weekday() != 6 else 0)
            tick_vals = pd.date_range(start=start_range.floor("D"), end=max_ts.ceil("D"), freq="7D")
            tick_text = [t.strftime("<b>%b %d</b>\n<b>%Y</b>") for t in tick_vals]
            xaxis_config["tickvals"] = tick_vals
            xaxis_config["ticktext"] = tick_text
        else:
            xaxis_config["dtick"] = 7 * 86400000  # 7 days in ms
            xaxis_config["tickformat"] = "%b %d\n%Y"
    elif granularity == "Monthly":
        if not plot_df.empty:
            min_ts = plot_df["local_datetime"].min()
            max_ts = plot_df["local_datetime"].max()
            tick_vals = pd.date_range(start=min_ts.replace(day=1).floor("D"), end=max_ts.ceil("D"), freq="MS")
            tick_text = [t.strftime("<b>%b %Y</b>") for t in tick_vals]
            xaxis_config["tickvals"] = tick_vals
            xaxis_config["ticktext"] = tick_text
        else:
            xaxis_config["dtick"] = "M1"
            xaxis_config["tickformat"] = "%b %Y"

    fig_trend.update_layout(
        title=dict(text=chart_title_with_sub, font=dict(size=20, color="black"), x=0, xanchor="left"),
        xaxis=xaxis_config,
        yaxis=dict(title=dict(text="Vehicle Volume Counts", font=dict(size=16, weight="bold", color="black")), tickfont=dict(size=14, color="black")),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.18,
            xanchor="left",
            x=1.05,
            font=dict(size=13),
            title=dict(text="Legend", font=dict(size=14)),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cccccc",
            borderwidth=1,
        ),
        margin=dict(l=10, r=260, t=100, b=10),
    )
    fig_trend.update_xaxes(patch=xaxis_config, overwrite=True)

    # ---------- Volume by Approach (replacing Box) ----------
    cat_order = order[order.index.isin(keep)].index.tolist()
    
    # Use all directions data for the approach volume chart
    approach_source_df = all_directions_df if all_directions_df is not None else raw_hourly_df
    
    if not approach_source_df.empty and "direction" in approach_source_df.columns:
        # Standardize direction names for the chart
        dir_map = {"NB": "Northbound", "SB": "Southbound", "EB": "Eastbound", "WB": "Westbound"}
        
        # Filter for the selected intersections ('keep')
        df_approach = approach_source_df[approach_source_df["intersection_name"].isin(keep)].copy()
        
        # Aggregate total volume by intersection and direction across the entire range
        df_approach_agg = df_approach.groupby(["intersection_name", "direction"], as_index=False)["total_volume"].sum()
        
        # Map abbreviations to full names
        df_approach_agg["Direction"] = df_approach_agg["direction"].map(lambda x: dir_map.get(str(x).upper(), str(x)))
        
        # Sort directions for consistent legend order
        dir_order = ["Northbound", "Southbound", "Eastbound", "Westbound"]
        df_approach_agg["Direction"] = pd.Categorical(df_approach_agg["Direction"], categories=dir_order, ordered=True)
        df_approach_agg = df_approach_agg.sort_values(["intersection_name", "Direction"])

        # Colors for directions
        color_map = {
            "Northbound": "#3498db", # Blue
            "Southbound": "#e67e22", # Orange
            "Eastbound": "#2ecc71",  # Green
            "Westbound": "#e74c3c"   # Red
        }

        fig_box = px.bar(
            df_approach_agg, x="intersection_name", y="total_volume",
            color="Direction",
            barmode="group",
            category_orders={"intersection_name": cat_order, "Direction": dir_order},
            color_discrete_map=color_map,
            title=f"Total Volume by Approach — {granularity}"
        )
        
        fig_box.update_traces(
            texttemplate="<b>%{y:,.0f}</b>",
            textposition="outside",
            cliponaxis=False,
            textfont=dict(size=12, color="black")
        )
    else:
        # Fallback to the original box plot if directional data is missing
        if plot_df_comp is not None:
            # For comparison, we use Period as color
            label_p = p_dates_short
            label_c = c_dates_short
            plot_df["Period"] = label_p
            plot_df_comp["Period"] = label_c
            box_df = pd.concat([plot_df, plot_df_comp], ignore_index=True)
            fig_box = px.box(
                box_df, x="intersection_name", y="total_volume",
                color="Period",
                category_orders={"intersection_name": cat_order, "Period": [label_p, label_c]},
                color_discrete_sequence=["#3498db", "#e67e22"],
                points=False, title=f"Volume Distribution Analysis — {granularity}"
            )
        else:
            fig_box = px.box(
                plot_df, x="intersection_name", y="total_volume",
                color="direction" if "direction" in plot_df.columns else None,
                category_orders={"intersection_name": cat_order},
                points=False, title=f"Volume Distribution by Intersection — {granularity}"
            )

    fig_box.update_layout(
        xaxis_title="Intersection",
        yaxis_title=f"Total Volume ({unit})",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.05,
            xanchor="left",
            x=1.05,
            font=dict(size=13),
            title=dict(text="Legend", font=dict(size=14)),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cccccc",
            borderwidth=1,
        ),
        margin=dict(l=10, r=200, t=100, b=10),
        bargap=0.15,
        bargroupgap=0.05
    )

    # ---------- Matrix or Heatmap ----------
    date_range_str = ""
    if date_range and len(date_range) == 2:
        date_range_str = f" — {date_range[0].strftime('%b %d, %Y')} to {date_range[1].strftime('%b %d, %Y')}"
    
    comp_date_range_str = ""
    if comp_date_range and len(comp_date_range) == 2:
        comp_date_range_str = f" (Comp: {comp_date_range[0].strftime('%b %d, %Y')} to {comp_date_range[1].strftime('%b %d, %Y')})"

    if len(keep) == 1:
        # Single intersection: Show heatmap of directions vs time
        intersection_name = keep[0]
        # Use all directions data for the heatmap so every approach is visible
        # regardless of the direction filter applied in the sidebar settings.
        heat_source_df = all_directions_df if all_directions_df is not None else raw_hourly_df
        df_heat = heat_source_df[heat_source_df["intersection_name"] == intersection_name].copy()
        
        if not df_heat.empty and "direction" in df_heat.columns:
            # Drop any invalid direction labels that might have slipped through
            df_heat = df_heat[~df_heat["direction"].astype(str).isin(["-", "", "nan", "None"])]
            # Aggregate if there are multiple entries for same (time, direction)
            df_heat = df_heat.groupby(["local_datetime", "direction"], as_index=False)["total_volume"].sum()
            
            # Pivot to wide for heatmap
            heat_pivot = df_heat.pivot(index="direction", columns="local_datetime", values="total_volume").sort_index()
            
            fig_matrix = go.Figure(data=go.Heatmap(
                z=heat_pivot.values,
                x=heat_pivot.columns,
                y=heat_pivot.index,
                colorscale="Blues",
                hovertemplate="<b>%{y}</b><br>Time: %{x}<br>Volume: %{z:,.0f} " + unit + "<extra></extra>",
                showscale=True,
                colorbar=dict(title=f"Volume ({unit})")
            ))
            
            # Use the same title style as fig_trend
            heat_main_title = f"All Approaches Vehicle Volume Per {label.capitalize()} by Approach (Heatmap Chart)".strip()
            
            heat_title = f"<b>{heat_main_title}</b>"
            if date_range_str:
                heat_title += f"<br><span style='font-size:15px; color:#444;'>{date_range_str.strip(' — ')}</span>"
            if intersections:
                heat_title += f"<br><span style='font-size:14px; color:#666;'>{intersections}</span>"

            # X-axis configuration for heatmap
            heat_xaxis = dict(
                title=dict(text="Date/Time", font=dict(size=16, weight="bold")),
                tickfont=dict(size=14)
            )
            if granularity == "Hourly":
                if date_range and len(date_range) == 2:
                    params = get_dynamic_xaxis_params(date_range[0], date_range[1])
                    heat_xaxis["dtick"] = params["dtick"]
                    heat_xaxis["tickformat"] = params["tickformat"]
                else:
                    heat_xaxis["dtick"] = 21600000  # 6 hours fallback
                    heat_xaxis["tickformat"] = "%b %d\n%I:%M %p"

            fig_matrix.update_layout(
                title=dict(text=heat_title, font=dict(size=20), x=0, xanchor="left"),
                xaxis=heat_xaxis,
                yaxis=dict(title=dict(text="Direction", font=dict(size=16, weight="bold")), tickfont=dict(size=14)),
                margin=dict(l=10, r=10, t=100, b=10),
                height=400
            )
        else:
            # Fallback if no direction data or empty
            fig_matrix = px.bar(
                title=f"{label.capitalize()} Vehicle Volume by Intersection"
            )
            fig_matrix.add_annotation(text="No direction data available for heatmap", showarrow=False)
    else:
        # Multiple intersections: Show ranking bar chart
        mat_primary = (
            plot_df.groupby("intersection_name", as_index=False)["total_volume"]
            .mean()
            .rename(columns={"total_volume": "avg_volume"})
        )
        label_p = p_dates_short if plot_df_comp is not None else "Volume"
        mat_primary["Period"] = label_p
        
        if plot_df_comp is not None:
            mat_comp = (
                plot_df_comp.groupby("intersection_name", as_index=False)["total_volume"]
                .mean()
                .rename(columns={"total_volume": "avg_volume"})
            )
            label_c = c_dates_short
            mat_comp["Period"] = label_c
            mat = pd.concat([mat_primary, mat_comp], ignore_index=True)
        else:
            mat = mat_primary

        mat["Rank"] = mat.groupby("Period")["avg_volume"].rank(ascending=False, method="dense").astype(int)
        
        # Sort by primary rank
        rank_order = mat_primary.sort_values("avg_volume", ascending=False)["intersection_name"].tolist()
        
        # Build dynamic title for Bar Chart
        main_bar_title = f"{direction_label} Vehicle Volume Per {label.capitalize()} by Intersection (Bar Chart)".strip()
        bar_title = f"<b>{main_bar_title}</b>"
        if date_subtitle:
            bar_title += f"<br><span style='font-size:15px; color:#444;'>{date_subtitle}</span>"
        if intersections:
            bar_title += f"<br><span style='font-size:14px; color:#666;'>{intersections}</span>"

        fig_matrix = px.bar(
            mat, y="intersection_name", x="avg_volume",
            orientation="h", text="avg_volume",
            color="Period" if plot_df_comp is not None else "intersection_name",
            barmode="group",
            color_discrete_sequence=["#3498db", "#e67e22"] if plot_df_comp is not None else px.colors.sequential.Blues_r,
            category_orders={"intersection_name": rank_order, "Period": [label_p, label_c] if plot_df_comp is not None else None},
            title=bar_title
        )
        
        # Tooltip enhancement: include date range
        if plot_df_comp is not None:
            hover_text = "<b>%{y}</b> (%{fullData.name})<br>" + f"{label.capitalize()} Volume: %{{x:,.0f}} {unit}<extra></extra>"
        else:
            hover_text = f"<b>%{{y}}</b><br>{label.capitalize()} Volume: %{{x:,.0f}} {unit}<br>Period: {date_subtitle}<extra></extra>"
        
        fig_matrix.update_traces(
            texttemplate="<b>%{text:,.0f}</b>", 
            textposition="outside", 
            cliponaxis=False,
            hovertemplate=hover_text,
            textfont=dict(size=13, color="black") # Increased from 11 to 13
        )
        
        fig_matrix.update_layout(
            title=dict(text=bar_title, font=dict(size=20), x=0, xanchor="left"),
            xaxis=dict(title=dict(text=f"{agg_prefix} Volume ({unit})", font=dict(size=16, weight="bold")), tickfont=dict(size=14)),
            yaxis=dict(title=dict(text="Intersection", font=dict(size=16, weight="bold")), tickfont=dict(size=14)), # Increased from 13 to 14
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1.05,
                xanchor="left",
                x=1.05,
                font=dict(size=13), # Increased from 12 to 13
                title=dict(text="Legend", font=dict(size=14)), # Increased from 13 to 14
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#cccccc",
                borderwidth=1,
            ),
            margin=dict(l=10, r=260, t=100, b=10),
            height=max(500, 150 + 60 * len(keep)), # Adjusted height to make bars thicker relative to plot area
            bargap=0.15, # Increased gap slightly to match the look in Screenshot 1
            bargroupgap=0.05,
            uniformtext=dict(minsize=12, mode='show') # Increased from 10 to 12
        )
    return fig_trend, fig_box, fig_matrix


# =========================
# Flip callback for Tab 1 O-D controls (Streamlit-safe)
# =========================
def _flip_od_state():
    """Swap od_origin and od_destination via callback to satisfy Streamlit's widget state rules."""
    try:
        o = st.session_state.get("od_origin")
        d = st.session_state.get("od_destination")
        if o is None or d is None:
            return
        # Record flip snapshot
        st.session_state["od_origin_flip"] = d
        st.session_state["od_destination_flip"] = o
        # Perform swap using temp var (avoid tuple assignment)
        st.session_state["od_origin"] = d
        st.session_state["od_destination"] = o
    except Exception:
        pass


# =========================
# Tabs
# =========================
st.markdown("## Choose Data Source")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Pg.1 ITERIS CLEARGUIDE", "Pg.2 KINETIC MOBILITY", "Pg.3 ACYCLICA", "Pg.4 ITERIS VANTAGE LIVE",
     "Pg.5 BOSCH CLOUD ANALYTICS"])

# -------------------------
# TAB 1: Performance / Travel Time (Search-gated, NO forms)
# -------------------------
with tab1:
    # Load data once to populate controls (safe to load; results stay blank until Search)
    corridor_df = get_corridor_df()

    # -------- Sidebar controls (commit on Search) --------
    with st.sidebar:
        st.image("Logos/ACE-logo-HiRes.jpg", width=210)
        st.image("Logos/CV Sync__.jpg", width=205)

        with st.expander("⚙️ Pg.1 ITERIS CLEARGUIDE SETTINGS", expanded=False):
            st.caption("Select Route and Date Range")
            st.caption("Data: Vehicle Speed, Delay, and Travel Time")
            st.markdown("## 🗺️ Select Route")

            od_mode = st.checkbox(
                "Analysis Pro",
                value=True,
                help="Unlocks advanced analysis (tables, downloads) and removes the 60-day cap.",
                key="od_mode_perf",
            )

            origin, destination = None, None
            if not corridor_df.empty:
                nodes_in_data = _canonical_order_in_data(corridor_df)
                node_list = nodes_in_data if len(nodes_in_data) >= 2 else _build_node_order(corridor_df)

                if len(node_list) >= 2:
                    # Add "SELECT" as the first option for both origin and destination
                    origin_options = ["SELECT"] + node_list
                    destination_options = ["SELECT"] + node_list

                    # Initialize session state defaults for Tab 1 O-D if missing
                    if "od_origin" not in st.session_state:
                        st.session_state["od_origin"] = "SELECT"
                    if "od_destination" not in st.session_state:
                        st.session_state["od_destination"] = "SELECT"

                    # 3-column layout: Origin | Flip | Destination
                    cA, flip_col, cB = st.columns([5, 2, 5])

                    with cA:
                        origin = st.selectbox("Origin", origin_options, key="od_origin")
                    with flip_col:
                        # Center the flip button visually
                        st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
                        # Only show flip button when both selections are made
                        if origin != "SELECT" and destination != "SELECT":
                            st.button(
                                "🔄",
                                key="od_flip_button",
                                help="Flip origin and destination",
                                use_container_width=True,
                                on_click=_flip_od_state,
                            )
                    with cB:
                        destination = st.selectbox("Destination", destination_options, key="od_destination")

                    # Route direction indicator - only show when both are selected
                    if origin != "SELECT" and destination != "SELECT":
                        try:
                            oi = node_list.index(origin)
                            di = node_list.index(destination)
                            if oi < di:
                                arrow = "→"
                                bound = "Northbound"
                                route_text = f"{origin} {arrow} {destination} ({bound})"
                            elif oi > di:
                                arrow = "←"
                                bound = "Southbound"
                                route_text = f"{origin} {arrow} {destination} ({bound})"
                            else:
                                route_text = f"{origin}"
                        except Exception:
                            route_text = None

                        if route_text:
                            st.markdown(
                                """
                                <style>
                                  .od-route-chip {
                                    margin-top: 0.25rem; padding: 6px 10px; border-radius: 8px; font-weight: 700;
                                    background: rgba(79,172,254,0.08); border: 1px solid rgba(79,172,254,0.25); color: #0b2538;
                                  }
                                  @media (prefers-color-scheme: dark) {
                                    .od-route-chip {
                                      background: rgba(79,172,254,0.12);
                                      border-color: rgba(79,172,254,0.45);
                                      color: #e6f2ff; /* high-contrast for dark mode */
                                    }
                                  }
                                </style>
                                """,
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f'<div class="od-route-chip">{route_text}</div>',
                                unsafe_allow_html=True,
                            )
                else:
                    st.info("Not enough nodes found to build O-D options.")

            # ---- Loading animation when O-D changes ----
            try:
                od_pair = (st.session_state.get("od_origin"), st.session_state.get("od_destination"))
                prev_od_pair = st.session_state.get("od_pair_prev")
                if prev_od_pair != od_pair:
                    st.session_state["od_pair_prev"] = od_pair
                    # only show when both are selected and distinct and not "SELECT"
                    if (od_pair[0] and od_pair[1] and
                            od_pair[0] != "SELECT" and od_pair[1] != "SELECT" and
                            od_pair[0] != od_pair[1]):
                        pb = st.progress(0, text="Loading Data availability info...")
                        for i in range(0, 101, 10):
                            time.sleep(0.02)
                            pb.progress(i, text="Loading Data availability info...")
                        pb.empty()
            except Exception:
                pass

            # ---- Availability preview (always show) ----
            try:
                base_df = corridor_df if corridor_df is not None else pd.DataFrame()
                header_label = "Available Data"
                dfx = base_df.copy()

                if (origin and destination and origin != "SELECT" and destination != "SELECT"):
                    path_mask = None
                    if origin != destination and dfx is not None and not dfx.empty:
                        # Build path segments between origin and destination using canonical forward naming
                        nodes = _canonical_order_in_data(dfx)
                        if origin in nodes and destination in nodes:
                            oi = nodes.index(origin)
                            di = nodes.index(destination)
                            imin, imax = (oi, di) if oi < di else (di, oi)
                            segs = [f"{nodes[i]} → {nodes[i + 1]}" for i in range(imin, imax)]
                            dir_target = "nb" if oi < di else "sb"
                            # Filter by segment names present (always forward-named)
                            if "segment_name" in dfx.columns:
                                path_mask = dfx["segment_name"].isin(segs)
                            # Apply path filter first
                            if path_mask is not None:
                                dfx_path = dfx[path_mask]
                            else:
                                dfx_path = dfx
                            # Filter by direction when available; if it drops everything, keep path-only
                            if "direction" in dfx_path.columns:
                                try:
                                    dir_norm = normalize_dir(dfx_path["direction"])
                                    dir_mask = dir_norm == dir_target
                                    dfx_dir = dfx_path[dir_mask]
                                    dfx = dfx_dir if not dfx_dir.empty else dfx_path
                                except Exception:
                                    dfx = dfx_path
                            else:
                                dfx = dfx_path
                            # Dynamic header text
                            arrow = "→" if oi < di else "←"
                            header_label = f"Available Data for {origin} {arrow} {destination}"
                        else:
                            header_label = "Available Data for this Corridor"
                    elif origin == destination and origin not in (None, "", "SELECT"):
                        header_label = "Available Data"
                    else:
                        header_label = "Available Data"
                else:
                    # Initial state (no full O-D yet): summarize entire corridor dataset
                    header_label = "Available Data"

                avail = compute_data_availability(
                    dfx if dfx is not None else pd.DataFrame(),
                    datetime_col="local_datetime",
                    max_gaps=3,
                    current_date=datetime.now(),
                )
                if avail.get("start") and avail.get("end"):
                    start_str = avail["start"].strftime("%b %d, %Y %I:%M %p")
                    end_str = avail["end"].strftime("%b %d, %Y %I:%M %p")
                    mb = avail.get("size_mb", 0.0)
                    size_str = f"({mb:.1f} MB)" if mb > 0 else ""
                    st.caption(header_label)
                    st.caption(f"• Date Range: {start_str} → {end_str} {size_str}")
                    gaps = avail.get("gaps") or []
                    if len(gaps) == 0:
                        st.caption("• Missing Data: None")
                    else:
                        st.caption("• Missing Data: " + "; ".join(gaps))
            except Exception:
                # keep sidebar resilient
                pass

            # Progressive disclosure: only show the rest after both selections are made
            valid_od = bool(origin and destination and
                            origin != "SELECT" and destination != "SELECT" and
                            origin != destination)

            if valid_od:
                # Analysis Period
                if corridor_df.empty or "local_datetime" not in corridor_df.columns:
                    min_date = datetime.today().date() - timedelta(days=7)
                    max_date = datetime.today().date()
                else:
                    min_date = corridor_df["local_datetime"].dt.date.min()
                    max_date = corridor_df["local_datetime"].dt.date.max()

                # If Analysis Pro is OFF, restrict selectable dates to the last 60 days
                if not od_mode:
                    cap_start = max_date - timedelta(days=60)
                    # Ensure within dataset bounds
                    min_date = max(min_date, cap_start)
                    # max_date remains the same (today/data max)

                st.markdown("## 📅 Date And Time")
                if not od_mode:
                    st.info("Analysis Pro is OFF: Date range limited to the last 60 days.")
                date_range = date_range_preset_controls(min_date, max_date, key_prefix="perf")

                # Analysis Settings
                st.markdown("## Granularity")
                granularity = st.selectbox(
                    "Data Aggregation",
                    ["Hourly", "Daily", "Weekly", "Monthly"],
                    index=0,
                    key="granularity_perf",
                    help="Higher aggregation smooths trends but may hide peaks",
                )

                time_filter, start_hour, end_hour = None, None, None
                if granularity == "Hourly":
                    time_filter = st.selectbox(
                        "Time Period Focus",
                        [
                            "All Hours",
                            "Peak Hours (7–9 AM, 4–6 PM)",
                            "AM Peak (7–9 AM)",
                            "PM Peak (4–6 PM)",
                            "Off-Peak",
                            "Custom Range",
                        ],
                        key="time_period_focus_perf",
                    )
                    if time_filter == "Custom Range":
                        c1, c2 = st.columns(2)
                        with c1:
                            start_hour = st.number_input("Start Hour (0–23)", 0, 23, 7, step=1, key="start_hour_perf")
                        with c2:
                            end_hour = st.number_input("End Hour (1–24)", 1, 24, 18, step=1, key="end_hour_perf")

                # track uncommitted controls
                t1_current = {
                    "od_mode": od_mode,
                    "origin": origin,
                    "destination": destination,
                    "date_range": tuple(date_range) if date_range else None,
                    "granularity": granularity,
                    "time_filter": time_filter if granularity == "Hourly" else None,
                    "start_hour": start_hour if (granularity == "Hourly" and time_filter == "Custom Range") else None,
                    "end_hour": end_hour if (granularity == "Hourly" and time_filter == "Custom Range") else None,
                }
                st.session_state["t1_current"] = t1_current

                if st.button("🔍 **Generate**", key="search_tab1", type="primary", use_container_width=True):
                    st.session_state["t1_params"] = t1_current
                    st.session_state["t1_ready"] = True
                    set_active_search_tab("t1")
                    st.session_state["last_active_tab"] = "t1"
                    # Persist committed search to URL
                    # --- SAVE TO URL (The new logic) ---
                    try:
                        ds = t1_current["date_range"][0].isoformat() if t1_current.get("date_range") else ""
                        de = t1_current["date_range"][1].isoformat() if t1_current.get("date_range") else ""

                        st.query_params.update(
                            t1_ready="1",
                            t1_od="1" if t1_current.get("od_mode") else "0",
                            t1_origin=t1_current.get("origin") or "",
                            t1_destination=t1_current.get("destination") or "",
                            t1_date_start=ds or "",
                            t1_date_end=de or "",
                            t1_granularity=t1_current.get("granularity") or "Hourly",
                            t1_time_filter=t1_current.get("time_filter") or "",
                            t1_start_hour=str(t1_current.get("start_hour") or ""),
                            t1_end_hour=str(t1_current.get("end_hour") or ""),
                            last_tab="t1",
                        )
                    except Exception:
                        pass
            else:
                # Reset current to minimal to avoid stale diffs
                st.session_state["t1_current"] = {"origin": origin, "destination": destination, "od_mode": od_mode}

    # -------- Main content area (render only when "Generate" committed) --------
    t1_ready = st.session_state.get("t1_ready", False)
    t1_params = st.session_state.get("t1_params", {})
    t1_pending = t1_ready and _freeze_params(t1_params) != _freeze_params(st.session_state.get("t1_current", {}))

    if not t1_ready:
        st.info("Choose your Route and Date Range in the settings to the left.")
    else:
        if t1_pending:
            st.warning(" Press **Generate** to refresh.")

        # ... existing code continues unchanged from here ...

        try:
            base_df = corridor_df.copy() if not corridor_df.empty else pd.DataFrame()
            if base_df.empty:
                st.error("❌ Failed to load corridor Prediction. Please check your Prediction sources.")
            else:
                # Unpack committed params
                od_mode = t1_params.get("od_mode", True)
                origin = t1_params.get("origin")
                destination = t1_params.get("destination")
                date_range = t1_params.get("date_range")
                granularity = t1_params.get("granularity", "Hourly")
                time_filter = t1_params.get("time_filter")
                start_hour = t1_params.get("start_hour")
                end_hour = t1_params.get("end_hour")

                # --- Prepare working set / O-D path subset ---
                working_df = base_df.copy()
                route_label = "All Segments"

                # Enforce 60-day date cap when Analysis Pro is OFF even if previous params had a wider range
                if date_range and not od_mode:
                    try:
                        # Determine data bounds
                        if not base_df.empty and "local_datetime" in base_df.columns:
                            data_min = base_df["local_datetime"].dt.date.min()
                            data_max = base_df["local_datetime"].dt.date.max()
                        else:
                            data_min = datetime.today().date() - timedelta(days=60)
                            data_max = datetime.today().date()
                        cap_start = max(data_min, data_max - timedelta(days=60))
                        # Clamp the incoming range
                        start, end = date_range
                        start = max(start, cap_start)
                        end = min(end, data_max)
                        date_range = (start, end)
                    except Exception:
                        pass

                # ensure numeric types early
                for c in ["average_traveltime", "average_delay", "average_speed"]:
                    if c in working_df.columns:
                        working_df[c] = pd.to_numeric(working_df[c], errors="coerce")

                desired_dir: str | None = None

                if od_mode and origin and destination:
                    canonical = _canonical_order_in_data(base_df)
                    if len(canonical) < 2:
                        canonical = _build_node_order(base_df)

                    if origin in canonical and destination in canonical:
                        i0, i1 = canonical.index(origin), canonical.index(destination)
                        if i0 < i1:
                            desired_dir = "nb"
                        elif i0 > i1:
                            desired_dir = "sb"
                        else:
                            desired_dir = None

                        imin, imax = (i0, i1) if i0 < i1 else (i1, i0)
                        candidate_segments = [f"{canonical[j]} → {canonical[j + 1]}" for j in range(imin, imax)]
                        seg_names_in_data = set(base_df["segment_name"].dropna().unique().tolist())
                        path_segments = [s for s in candidate_segments if s in seg_names_in_data]

                        seg_df = pd.DataFrame()
                        used_fallback_direct = False

                        if path_segments:
                            seg_df = base_df[base_df["segment_name"].isin(path_segments)].copy()
                        else:
                            # Fallback: use a single aggregated segment if it exists (e.g., "Avenue 41 → Country Club Drive")
                            direct_name = f"{origin} → {destination}"
                            if direct_name in seg_names_in_data:
                                seg_df = base_df[base_df["segment_name"] == direct_name].copy()
                                used_fallback_direct = True

                        if not seg_df.empty:
                            if "direction" in seg_df.columns and desired_dir is not None:
                                dnorm = normalize_dir(seg_df["direction"])
                                seg_df = seg_df.loc[dnorm == desired_dir].copy()

                            if seg_df.empty:
                                st.info("No data found in the selected direction for this O-D.")
                            else:
                                working_df = seg_df.copy()
                                route_label = f"{origin} → {destination}"
                                if used_fallback_direct:
                                    st.caption(
                                        "Using combined segment data for this O-D (intermediate subsegments unavailable).")
                        else:
                            st.info("No matching segments found for the selected O-D on the canonical path.")

                # ---------- Layout: wide content + sticky right rail ----------
                main_col_t1, right_col_t1 = st.columns([7, 3.5], gap="large")

                # Right rail (map code)
                with right_col_t1:
                    st.markdown('<div id="od-map-anchor"></div>', unsafe_allow_html=True)
                    st.markdown("##### Corridor Map", help="Stays visible while you scroll the analysis on the left.")

                    # Corridor map remains visible regardless of Analysis Pro mode
                    fig_od = None
                    if origin and destination and origin != destination:
                        satellite_t1 = st.toggle("🛰️ Satellite View", key="satellite_t1", value=False)
                        try:
                            fig_od = build_corridor_map(origin, destination, satellite=satellite_t1)
                        except Exception:
                            fig_od = None

                    if fig_od:
                        try:
                            fig_od.update_layout(height=MAP_HEIGHT, margin=dict(l=0, r=0, t=32, b=0))
                        except Exception:
                            pass
                        st.markdown(f'<div class="cvag-map-card">', unsafe_allow_html=True)
                        st.plotly_chart(fig_od, use_container_width=True, config=PLOTLY_CONFIG)
                        st.caption(f"Corridor Segment: **{origin} → {destination}**")
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="cvag-map-card">', unsafe_allow_html=True)
                        st.info("Select an **Origin** and **Destination** to display the corridor map.")
                        st.markdown("</div>", unsafe_allow_html=True)

                # Left/main content
                with main_col_t1:
                    if not date_range or len(date_range) != 2:
                        st.warning("⚠️ Please select both start and end dates to proceed.")
                    else:
                        # ---- CAD-style loader starts here ----
                        with scoped_cad_loader("Fetching Data...", tab_id="t1") as step:
                            step("Filtering by date range & aggregating", 20)
                            filtered_data = process_traffic_data(
                                working_df,
                                date_range,
                                granularity,
                                time_filter if granularity == "Hourly" else None,
                                start_hour,
                                end_hour,
                            )

                            if filtered_data.empty:
                                step("No Prediction found for selected filters", 100)
                                st.warning("⚠️ No Prediction available for the selected filters.")
                            else:
                                total_records = len(filtered_data)
                                data_span = (date_range[1] - date_range[0]).days + 1
                                time_context = f" • {time_filter}" if (granularity == "Hourly" and time_filter) else ""
                                dir_display = "Northbound" if desired_dir == "nb" else "Southbound" if desired_dir == "sb" else "All Directions"

                                step("Preparing summary context", 35)
                                st.markdown(
                                    f"""
                                                                    <div style="
                                                                        background: linear-gradient(135deg, #2b77e5 0%, #19c3e6 100%);
                                                                        border-radius:16px; padding:18px 20px; color:#fff; margin:8px 0 14px;
                                                                        box-shadow:0 10px 26px rgba(25,115,210,.25); text-align:left;
                                                                        font-family: inherit;">
                                                                      <div style="display:flex; align-items:center; gap:10px;">
                                                                        <div>
                                                                            <div style="font-size:1.7rem; font-weight:800; letter-spacing:-0.01em; margin-bottom:2px;">
                                                                              <span style="color: #ffffff;">2025 BNP PARIBUS OPEN INDIAN WELLS DASHBOARD</span><span style="color: rgba(255,255,255,0.7);">: ITERIS CLEARGUIDE TRAVEL TIME ANALYSIS</span>
                                                                            </div>
                                                                            <div style="font-size:1.1rem;font-weight:600;opacity:0.9; display:flex; flex-wrap:wrap; gap:15px; margin-top:4px; align-items:center;">
                                                                              <div style="background:rgba(255,255,255,0.15); padding:4px 12px; border-radius:8px; font-size:1.2rem; border:1px solid rgba(255,255,255,0.3); font-weight:700;">
                                                                                📅 {date_range[0].strftime('%b %d, %Y')} to {date_range[1].strftime('%b %d, %Y')} <span style="font-weight:400; opacity:0.8;">({data_span} days)</span>
                                                                              </div>
                                                                              <div style="background:rgba(255,255,255,0.15); padding:4px 12px; border-radius:8px; font-size:1.2rem; border:1px solid rgba(255,255,255,0.3);">
                                                                                <span style="opacity:0.9; font-weight:400;">Corridor Segment:</span> <span style="font-weight:800;">{route_label}</span> <span style="background:rgba(255,255,255,0.2); padding:2px 8px; border-radius:4px; margin-left:8px; font-size:0.9rem; font-weight:700; color:#fff;">{dir_display}</span>
                                                                              </div>
                                                                            </div>
                                                                            <div style="font-size:0.95rem; opacity:0.85; display:flex; flex-wrap:wrap; gap:15px; margin-top:10px; align-items:center;">
                                                                              <div style="display:flex; align-items:center; gap:15px;">
                                                                                <div><span style="opacity:0.8; font-weight:400;">Region:</span> Coachella Valley</div>
                                                                                <div style="border-left: 1px solid rgba(255,255,255,0.3); padding-left: 15px;"><span style="opacity:0.8; font-weight:400;">City:</span> Indian Wells / La Quinta</div>
                                                                                <div style="border-left: 1px solid rgba(255,255,255,0.3); padding-left: 15px;"><span style="opacity:0.8; font-weight:400;">Corridor:</span> Washington Street</div>
                                                                                <div style="border-left: 1px solid rgba(255,255,255,0.3); padding-left: 15px;">📊 {granularity} Aggregation{time_context}</div>
                                                                                <div style="border-left: 1px solid rgba(255,255,255,0.3); padding-left: 15px;">✅ {total_records:,} Prediction points</div>
                                                                              </div>
                                                                            </div>
                                                                        </div>
                                                                      </div>
                                                                    </div>
                                                                    """,
                                    unsafe_allow_html=True,
                                )

                                step("Building hourly O-D series", 55)
                                od_hourly = process_traffic_data(
                                    working_df,
                                    date_range,
                                    "Hourly",
                                    time_filter,
                                    start_hour,
                                    end_hour,
                                )

                                if not od_hourly.empty:
                                    if "direction" in od_hourly.columns and desired_dir is not None:
                                        dnorm2 = normalize_dir(od_hourly["direction"])
                                        od_hourly = od_hourly.loc[dnorm2 == desired_dir].copy()

                                    # Ensure numeric types for aggregations
                                    for c in ["average_traveltime", "average_delay", "average_speed"]:
                                        if c in od_hourly.columns:
                                            od_hourly[c] = pd.to_numeric(od_hourly[c], errors="coerce")

                                    if "segment_name" in od_hourly.columns and "local_datetime" in od_hourly.columns:
                                        # First, average metrics per segment per timestamp
                                        agg_lvl1 = {"average_traveltime": "mean", "average_delay": "mean"}
                                        if "average_speed" in od_hourly.columns:
                                            agg_lvl1["average_speed"] = "mean"
                                        od_hourly = (
                                            od_hourly.groupby(["local_datetime", "segment_name"], as_index=False)
                                            .agg(agg_lvl1)
                                        )

                                    # Then, aggregate across segments for each timestamp:
                                    agg_lvl2 = {"average_traveltime": "sum", "average_delay": "sum"}
                                    if "average_speed" in od_hourly.columns:
                                        # For speed, use mean across segments for the hour
                                        agg_lvl2["average_speed"] = "mean"
                                    od_series = (
                                        od_hourly.groupby("local_datetime", as_index=False)
                                        .agg(agg_lvl2)
                                    )
                                    raw_data = od_series.copy()
                                else:
                                    od_series = pd.DataFrame()
                                    raw_data = filtered_data.copy()

                                if not raw_data.empty:
                                    for col in ["average_delay", "average_traveltime", "average_speed"]:
                                        if col in raw_data.columns:
                                            raw_data[col] = pd.to_numeric(raw_data[col], errors="coerce")

                                if raw_data.empty:
                                    step("No Prediction in window after processing", 100)
                                    st.info("No Prediction in this window.")
                                else:
                                    step("Computing KPIs", 70)
                                    st.subheader(" KPI's (Key Performance Indicators)")
                                    k = compute_perf_kpis_interpretable(raw_data, HIGH_DELAY_SEC)

                                    # Calculate LOS based on Average Speed
                                    avg_tt_val = k["avg_tt"]["value"]
                                    p95_tt_val = k["planning_time"]["value"]
                                    avg_speed_val = float(np.nanmean(raw_data["average_speed"])) if "average_speed" in raw_data.columns and raw_data["average_speed"].notna().any() else 0.0

                                    # Average LOS
                                    if avg_speed_val > 35: los_letter = "A"
                                    elif avg_speed_val >= 28: los_letter = "B"
                                    elif avg_speed_val >= 22: los_letter = "C"
                                    elif avg_speed_val >= 17: los_letter = "D"
                                    elif avg_speed_val >= 13: los_letter = "E"
                                    elif avg_speed_val > 0: los_letter = "F"
                                    else: los_letter = "N/A"

                                    # Worst-case LOS based on Planning Time
                                    worst_speed_val = avg_speed_val * (avg_tt_val / p95_tt_val) if p95_tt_val > 0 else 0.0
                                    if worst_speed_val > 35: los_letter_worst = "A"
                                    elif worst_speed_val >= 28: los_letter_worst = "B"
                                    elif worst_speed_val >= 22: los_letter_worst = "C"
                                    elif worst_speed_val >= 17: los_letter_worst = "D"
                                    elif worst_speed_val >= 13: los_letter_worst = "E"
                                    elif worst_speed_val > 0: los_letter_worst = "F"
                                    else: los_letter_worst = "N/A"

                                    buffer_minutes = max(0.0, k["planning_time"]["value"] - k["avg_tt"]["value"])
                                    buffer_help = (
                                        "Extra minutes to leave earlier so you arrive on time 95% of the time.\n"
                                        "Formula: Planning Time (95th) − Average Travel Time."
                                    )

                                    c1, c2, c3, c4, c5 = st.columns(5)
                                    with c1:
                                        st.metric(
                                            "🎯 Reliability Index",
                                            f"{k['reliability']['value']:.0f}{k['reliability']['unit']}",
                                            help=k['reliability']['help'],
                                        )
                                        st.markdown(render_badge(k['reliability']['score']), unsafe_allow_html=True)
                                    with c2:
                                        st.metric(
                                            "⚠️ Congestion Frequency",
                                            f"{k['congestion_freq']['value']:.1f}{k['congestion_freq']['unit']}",
                                            help=k['congestion_freq']['help'],
                                        )
                                        st.markdown(render_badge(k['congestion_freq']['score']), unsafe_allow_html=True)
                                        st.caption(k['congestion_freq'].get('extra', ''))
                                    with c3:
                                        st.metric(
                                            "⏱️ Average Travel Time",
                                            f"{k['avg_tt']['value']:.1f} {k['avg_tt']['unit']}",
                                            help="Average travel time across the selected period. Used to estimate corridor Level of Service (LOS) per HCM 6th Edition (TRB, 2016) urban arterial standards based on average travel speed. LOS A = over 35 mph (free flow), LOS B = 28 to 35 mph (minor delay), LOS C = 22 to 28 mph (stable flow), LOS D = 17 to 22 mph (approaching capacity), LOS E = 13 to 17 mph (unstable flow), LOS F = under 13 mph (breakdown). Source: Highway Capacity Manual, 6th Edition, Transportation Research Board. https://www.trb.org/Main/Blurbs/175169.aspx",
                                        )
                                        st.markdown(render_badge(k['avg_tt']['score']), unsafe_allow_html=True)
                                        st.caption(f"Estimated Corridor LOS: {los_letter}")
                                    with c4:
                                        st.metric(
                                            "📈 Planning Time (95th Percentile)",
                                            f"{k['planning_time']['value']:.1f} {k['planning_time']['unit']}",
                                            help="95th percentile travel time — only 5% of trips are slower than this. Used to estimate worst-case corridor Level of Service (LOS) per HCM 6th Edition (TRB, 2016). This represents the reliability ceiling: LOS A = over 35 mph, LOS B = 28 to 35 mph, LOS C = 22 to 28 mph, LOS D = 17 to 22 mph, LOS E = 13 to 17 mph, LOS F = under 13 mph. If this LOS is significantly worse than the Average Travel Time LOS, the corridor suffers from unreliable, variable conditions. Source: Highway Capacity Manual, 6th Edition, TRB. https://www.trb.org/Main/Blurbs/175169.aspx",
                                        )
                                        st.markdown(render_badge(k['planning_time']['score']), unsafe_allow_html=True)
                                        st.caption(f"Worst-case Corridor LOS: {los_letter_worst}")
                                    with c5:
                                        st.metric(
                                            "🧭 Buffer Time (leave this much earlier)",
                                            f"{buffer_minutes:.1f} min",
                                            help=buffer_help,
                                        )
                                        st.markdown(render_badge(k['buffer_index']['score']), unsafe_allow_html=True)

                                    if len(filtered_data) > 1:
                                        step("Rendering trend charts", 85)
                                        st.subheader("📈 Performance Trends")
                                        v1, v2 = st.columns(2)

                                        trends_df = od_series if 'od_series' in locals() and not od_series.empty else filtered_data

                                        if 'od_series' in locals() and not od_series.empty and granularity in ("Daily",
                                                                                                               "Weekly",
                                                                                                               "Monthly"):
                                            tmp = od_series.copy()
                                            tmp["local_datetime"] = pd.to_datetime(tmp["local_datetime"])
                                            if granularity == "Daily":
                                                tmp["date_group"] = tmp["local_datetime"].dt.date
                                                trends_df = (
                                                    tmp.groupby("date_group", as_index=False)
                                                    .agg({"average_traveltime": "mean", "average_delay": "mean"})
                                                    .rename(columns={"date_group": "local_datetime"})
                                                )
                                                trends_df["local_datetime"] = pd.to_datetime(
                                                    trends_df["local_datetime"])
                                            elif granularity == "Weekly":
                                                tmp["week_group"] = tmp["local_datetime"].dt.to_period(
                                                    "W").dt.start_time
                                                trends_df = (
                                                    tmp.groupby("week_group", as_index=False)
                                                    .agg({"average_traveltime": "mean", "average_delay": "mean"})
                                                    .rename(columns={"week_group": "local_datetime"})
                                                )
                                            elif granularity == "Monthly":
                                                tmp["month_group"] = tmp["local_datetime"].dt.to_period(
                                                    "M").dt.start_time
                                                trends_df = (
                                                    tmp.groupby("month_group", as_index=False)
                                                    .agg({"average_traveltime": "mean", "average_delay": "mean"})
                                                    .rename(columns={"month_group": "local_datetime"})
                                                )

                                        with v1:
                                            dc = performance_chart(trends_df, "delay", direction_label=dir_display)
                                            if dc:
                                                st.plotly_chart(dc, use_container_width=True, config=PLOTLY_CONFIG)
                                        with v2:
                                            tc = performance_chart(trends_df, "travel", direction_label=dir_display)
                                            if tc:
                                                st.plotly_chart(tc, use_container_width=True, config=PLOTLY_CONFIG)

                                        if od_mode and 'od_series' in locals() and not od_series.empty:
                                            st.subheader("🔍Which Dates/Times have the highest Travel Time and Delay?")

                                            # Apply granularity aggregation to the display data
                                            display_od_series = od_series.copy()
                                            display_od_series["local_datetime"] = pd.to_datetime(
                                                display_od_series["local_datetime"])

                                            if granularity == "Daily":
                                                display_od_series["date_group"] = display_od_series[
                                                    "local_datetime"].dt.date
                                                display_od_series = (
                                                    display_od_series.groupby("date_group", as_index=False)
                                                    .agg({
                                                        "average_traveltime": "mean",
                                                        "average_delay": "mean",
                                                        "average_speed": "mean",
                                                    })
                                                    .rename(columns={"date_group": "local_datetime"})
                                                )
                                                display_od_series["local_datetime"] = pd.to_datetime(
                                                    display_od_series["local_datetime"])
                                            elif granularity == "Weekly":
                                                display_od_series["week_group"] = display_od_series[
                                                    "local_datetime"].dt.to_period("W").dt.start_time
                                                display_od_series = (
                                                    display_od_series.groupby("week_group", as_index=False)
                                                    .agg({
                                                        "average_traveltime": "mean",
                                                        "average_delay": "mean",
                                                        "average_speed": "mean",
                                                    })
                                                    .rename(columns={"week_group": "local_datetime"})
                                                )
                                            elif granularity == "Monthly":
                                                display_od_series["month_group"] = display_od_series[
                                                    "local_datetime"].dt.to_period("M").dt.start_time
                                                display_od_series = (
                                                    display_od_series.groupby("month_group", as_index=False)
                                                    .agg({
                                                        "average_traveltime": "mean",
                                                        "average_delay": "mean",
                                                        "average_speed": "mean",
                                                    })
                                                    .rename(columns={"month_group": "local_datetime"})
                                                )


                                            # For "Hourly", no additional aggregation needed

                                            # Format the display dataframe with units and granularity-aware timestamps
                                            def format_minutes_display(val):
                                                return f"{val:.2f} minutes" if pd.notna(val) else "N/A"


                                            def format_mph_display(val):
                                                return f"{val:.1f} mph" if pd.notna(val) else "N/A"


                                            def format_timestamp_display(timestamp, granularity):
                                                """Format timestamp based on granularity"""
                                                if granularity == "Hourly":
                                                    return timestamp.strftime("%b %d, %Y %I:%M %p")
                                                elif granularity == "Daily":
                                                    return timestamp.strftime("%b %d, %Y")
                                                elif granularity == "Weekly":
                                                    # Show week start date
                                                    week_start = timestamp
                                                    week_end = week_start + pd.Timedelta(days=6)
                                                    return f"Week of {week_start.strftime('%b %d, %Y')} - {week_end.strftime('%b %d, %Y')}"
                                                elif granularity == "Monthly":
                                                    return timestamp.strftime("%B %Y")
                                                else:
                                                    return timestamp.strftime("%b %d, %Y %I:%M %p")


                                            # Apply formatting functions
                                            display_od_series["Formatted Timestamp"] = display_od_series[
                                                "local_datetime"].apply(
                                                lambda x: format_timestamp_display(x, granularity)
                                            )
                                            display_od_series["O-D Travel Time (min)"] = display_od_series[
                                                "average_traveltime"].apply(format_minutes_display)
                                            display_od_series["O-D Delay (min)"] = display_od_series[
                                                "average_delay"].apply(format_minutes_display)
                                            # Add O-D Speed column from available average_speed
                                            if "average_speed" in display_od_series.columns:
                                                display_od_series["O-D Speed (mph)"] = display_od_series[
                                                    "average_speed"].apply(format_mph_display)
                                            else:
                                                # In case hourly (no aggregation) or missing speed, attempt to map from od_series
                                                display_od_series["O-D Speed (mph)"] = np.nan

                                            # Select and rename columns for display
                                            final_display = display_od_series[
                                                ["Formatted Timestamp", "O-D Travel Time (min)",
                                                 "O-D Delay (min)", "O-D Speed (mph)"]].rename(
                                                columns={"Formatted Timestamp": f"Timestamp ({granularity})"}
                                            )

                                            # Sort by travel time descending to show highest first
                                            final_display_sorted = final_display.loc[
                                                display_od_series["average_traveltime"].sort_values(
                                                    ascending=False).index
                                            ].reset_index(drop=True)

                                            st.dataframe(
                                                final_display_sorted,
                                                use_container_width=True,
                                                column_config={
                                                    f"Timestamp ({granularity})": st.column_config.TextColumn(
                                                        f"Timestamp ({granularity})",
                                                        help=f"Date and time aggregated at {granularity.lower()} level"
                                                    ),
                                                    "O-D Travel Time (min)": st.column_config.TextColumn(
                                                        f"O-D Travel Time ({granularity})",
                                                        help=f"Average travel time from origin to destination aggregated by {granularity.lower()}"
                                                    ),
                                                    "O-D Delay (min)": st.column_config.TextColumn(
                                                        f"O-D Delay ({granularity})",
                                                        help=f"Average delay experienced from origin to destination aggregated by {granularity.lower()}"
                                                    ),
                                                    "O-D Speed (mph)": st.column_config.TextColumn(
                                                        "O-D Speed (mph)",
                                                        help=f"Average speed from origin to destination aggregated by {granularity.lower()}"
                                                    ),
                                                },
                                            )

                                    # =========================
                                    # 🚨 Comprehensive Bottleneck Analysis
                                    # =========================
                                    if od_mode:
                                        step("Running bottleneck analysis", 95)
                                        st.subheader("🚨 Comprehensive Bottleneck Analysis")
                                    # Add legend for Performance Rating and Impact Score
                                    if od_mode:
                                        with st.expander("📊 **Rating Methodology & Legend**", expanded=False):
                                            st.markdown("""
                                                                            ### Impact Score Calculation
                                                                            The **Impact Score** (0-100) is a weighted composite metric that identifies bottlenecks:
                                                                            - **45%** weight: Peak Delay (maximum delay observed)
                                                                            - **35%** weight: Average Delay (mean delay across observations)  
                                                                            - **20%** weight: Peak Travel Time (maximum travel time recorded)

                                                                            Higher scores indicate worse performance and more severe bottlenecks.

                                                                            ### Performance Rating Categories
                                                                            """)

                                            # Create a visual legend with colored badges (kept inside expander)
                                            legend_col1, legend_col2, legend_col3, legend_col4, legend_col5 = st.columns(
                                                5)
                                            with legend_col1:
                                                st.markdown(
                                                    '<span class="performance-badge badge-excellent">🟢 Excellent</span>',
                                                    unsafe_allow_html=True)
                                                st.caption("Score: 0-20")
                                                st.caption("Optimal flow")
                                            with legend_col2:
                                                st.markdown('<span class="performance-badge badge-good">🔵 Good</span>',
                                                            unsafe_allow_html=True)
                                                st.caption("Score: 20-40")
                                                st.caption("Minor delays")
                                            with legend_col3:
                                                st.markdown('<span class="performance-badge badge-fair">🟡 Fair</span>',
                                                            unsafe_allow_html=True)
                                                st.caption("Score: 40-60")
                                                st.caption("Moderate congestion")
                                            with legend_col4:
                                                st.markdown('<span class="performance-badge badge-poor">🟠 Poor</span>',
                                                            unsafe_allow_html=True)
                                                st.caption("Score: 60-80")
                                                st.caption("Significant delays")
                                            with legend_col5:
                                                st.markdown(
                                                    '<span class="performance-badge badge-critical">🔴 Critical</span>',
                                                    unsafe_allow_html=True)
                                                st.caption("Score: 80-100")
                                                st.caption("Severe bottleneck")

                                    if od_mode and 'raw_data' in locals() and not raw_data.empty and "segment_name" in working_df.columns:
                                        try:
                                            analysis_df = working_df[
                                                (working_df["local_datetime"].dt.date >= date_range[0])
                                                & (working_df["local_datetime"].dt.date <= date_range[1])
                                                ].copy()

                                            if "direction" in analysis_df.columns:
                                                analysis_df["dir_norm"] = normalize_dir(analysis_df["direction"])
                                            else:
                                                analysis_df["dir_norm"] = "unk"

                                            if od_mode and desired_dir is not None:
                                                analysis_df = analysis_df.loc[
                                                    analysis_df["dir_norm"] == desired_dir].copy()
                                                st.caption(f"Filtered to O-D direction: **{desired_dir.upper()}**")

                                            g = analysis_df.groupby(["segment_name", "dir_norm"]).agg(
                                                average_delay_mean=("average_delay", "mean"),
                                                average_delay_max=("average_delay", "max"),
                                                average_traveltime_mean=("average_traveltime", "mean"),
                                                average_traveltime_max=("average_traveltime", "max"),
                                                average_speed_mean=("average_speed", "mean"),
                                                average_speed_min=("average_speed", "min"),
                                                n=("average_delay", "count"),
                                            ).reset_index()

                                            arrow_map = {"nb": "↑ NB", "sb": "↓ SB", "unk": "• UNK"}
                                            g["Segment (by Dir)"] = g.apply(
                                                lambda
                                                    r: f"{r['segment_name']} ({arrow_map.get(r['dir_norm'], '• UNK')})",
                                                axis=1
                                            )


                                            def _norm(s):
                                                s = s.astype(float)
                                                mn, mx = np.nanmin(s), np.nanmax(s)
                                                if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
                                                    return (s - mn) / (mx - mn)
                                                return pd.Series(np.zeros(len(s)), index=s.index)


                                            score = (
                                                            0.45 * _norm(g["average_delay_max"])
                                                            + 0.35 * _norm(g["average_delay_mean"])
                                                            + 0.20 * _norm(g["average_traveltime_max"])
                                                    ) * 100
                                            g["Bottleneck_Score"] = score.round(1)

                                            bins = [-0.1, 20, 40, 60, 80, 200]
                                            labels = ["🟢 Excellent", "🔵 Good", "🟡 Fair", "🟠 Poor", "🔴 Critical"]
                                            g["🎯 Performance Rating"] = pd.cut(g["Bottleneck_Score"], bins=bins,
                                                                               labels=labels)

                                            final = g[
                                                [
                                                    "Segment (by Dir)",
                                                    "dir_norm",
                                                    "🎯 Performance Rating",
                                                    "Bottleneck_Score",
                                                    "average_delay_max",  # Peak Delay - 45% weight (1st component)
                                                    "average_delay_mean",  # Avg Delay - 35% weight (2nd component)
                                                    "average_traveltime_max",
                                                    # Peak Travel Time - 20% weight (3rd component)
                                                    "average_traveltime_mean",  # Additional context
                                                    "average_speed_mean",  # Additional context
                                                    "average_speed_min",  # Additional context
                                                    "n",
                                                ]
                                            ].rename(
                                                columns={
                                                    "dir_norm": "Dir",
                                                    "average_delay_max": "Peak Delay",
                                                    "average_delay_mean": "Avg Delay",
                                                    "average_traveltime_max": "Peak Time",
                                                    "average_traveltime_mean": "Avg Time",
                                                    "average_speed_mean": "Avg Speed",
                                                    "average_speed_min": "Min Speed",
                                                    "n": "Obs",
                                                }
                                            ).sort_values("Bottleneck_Score", ascending=False)


                                            # Add unit suffixes to the appropriate columns
                                            def format_minutes(val):
                                                return f"{val:.3f} min" if pd.notna(val) else "N/A"


                                            def format_mph(val):
                                                return f"{val:.1f} mph" if pd.notna(val) else "N/A"


                                            # Apply formatting to time columns
                                            for col in ["Peak Delay", "Avg Delay", "Peak Time", "Avg Time"]:
                                                if col in final.columns:
                                                    final[col] = final[col].apply(format_minutes)

                                            # Apply formatting to speed columns
                                            for col in ["Avg Speed", "Min Speed"]:
                                                if col in final.columns:
                                                    final[col] = final[col].apply(format_mph)

                                            st.dataframe(
                                                final.head(15),
                                                use_container_width=True,
                                                column_config={
                                                    "Bottleneck_Score": st.column_config.NumberColumn(
                                                        "🚨 Impact Score",
                                                        help="Composite (0–100); higher ⇒ worse. Based on Peak Delay (45%) + Avg Delay (35%) + Peak Time (20%)",
                                                        format="%.1f",
                                                    ),
                                                    "Dir": st.column_config.TextColumn("Dir"),
                                                    "Peak Delay": st.column_config.TextColumn("Peak Delay",
                                                                                              help="Maximum delay observed (45% weight in Impact Score)"),
                                                    "Avg Delay": st.column_config.TextColumn("Avg Delay",
                                                                                             help="Average delay across observations (35% weight in Impact Score)"),
                                                    "Peak Time": st.column_config.TextColumn("Peak Time",
                                                                                             help="Maximum travel time recorded (20% weight in Impact Score)"),
                                                    "Avg Time": st.column_config.TextColumn("Avg Time",
                                                                                            help="Average travel time across observations"),
                                                    "Avg Speed": st.column_config.TextColumn("Avg Speed",
                                                                                             help="Average speed across observations"),
                                                    "Min Speed": st.column_config.TextColumn("Min Speed",
                                                                                             help="Minimum speed recorded"),
                                                },
                                            )

                                            if od_mode:
                                                # Premium title block for downloads (buttons and file names unchanged)
                                                st.markdown(
                                                    """
                                                    <div style="
                                                        background: linear-gradient(135deg, #0f172a 0%, #111827 100%);
                                                        border: 1px solid rgba(148,163,184,0.25);
                                                        padding: 14px 16px; border-radius: 14px; margin: 8px 0 12px;
                                                        box-shadow: 0 6px 20px rgba(2,6,23,0.35), inset 0 0 0 1px rgba(255,255,255,0.02);
                                                    ">
                                                      <div style="display:flex; align-items:center; gap:12px;">
                                                        <div style="width:36px;height:36px;border-radius:10px; display:flex; align-items:center; justify-content:center;
                                                                    background: linear-gradient(135deg,#fde047,#f59e0b);
                                                                    color:#111; font-weight:900; box-shadow: 0 2px 8px rgba(245,158,11,.45);">⬇️</div>
                                                        <div>
                                                          <div style="font-size:1.05rem; font-weight:800; letter-spacing:.2px; color:#e5e7eb;">Premium Data Exports</div>
                                                          <div style="font-size:.85rem; color:#94a3b8; margin-top:2px;">Download production‑ready CSVs for offline analysis and reporting.</div>
                                                        </div>
                                                      </div>
                                                    </div>
                                                    """,
                                                    unsafe_allow_html=True,
                                                )

                                                st.download_button(
                                                    "⬇️ Download Bottleneck Table (CSV)",
                                                    data=final.to_csv(index=False).encode("utf-8"),
                                                    file_name="CVSYNC_bottlenecks_cquijano.csv",
                                                    mime="text/csv",
                                                )
                                                st.download_button(
                                                    "⬇️ Download Raw CSV (selected filters)",
                                                    data=filtered_data.to_csv(index=False).encode("utf-8"),
                                                    file_name="Filtered_CVSYNCDashboard_RAWDATA_cquijano.csv",
                                                    mime="text/csv",
                                                )
                                                # Add your new 5-minute download button
                                                st.download_button(
                                                    "⬇️ Download Raw CSV (5-minute)",
                                                    data=create_5min_data(filtered_data).to_csv(index=False).encode(
                                                        "utf-8"),
                                                    file_name="5-minute_CVSYNCDashboard_RAWDATA_cquijano.csv",
                                                    mime="text/csv",
                                                )
                                            else:
                                                st.info("Analysis Pro Required to download data files.")
                                        except Exception as e:
                                            st.error(f"❌ Error in performance analysis: {e}")

        except Exception as e:
            st.error(f"❌ Error processing traffic Prediction: {e}")

# -------------------------
# TAB 2: Volume / Capacity (Search-gated, NO forms)
# -------------------------
with tab2:
    # Load Prediction once to populate controls; results stay blank until Search
    volume_df = get_volume_df()

    with st.sidebar:
        with st.expander("⚙️ Pg.2 KINETIC MOBILITY SETTINGS", expanded=False):

            st.caption("Select Intersection(s) and Date Range")
            st.caption("Data: Vehicle Volume")

            # Build corridor options
            if not volume_df.empty and "corridor_id" in volume_df.columns:
                corridors = ["All Corridors"] + sorted(volume_df["corridor_id"].dropna().unique().tolist())
            else:
                corridors = ["All Corridors"]

            # Predefine ordered labels for Kinetic Mobility intersections
            washington_labels = [
                "Avenue 52",
                "Calle Tampico",
                "Village Shopping Center",
                "Avenue 50",
                "Sagebrush Avenue",
                "Eisenhower",
                "Avenue 48",
                "Avenue 47",
                "Channel Drive",
                "Miles Avenue",
                "Via Sevilla",
                "Avenue 42",
                "Harris Lane",
                "Country Club Drive",
                "Varner Road",
            ]

            hwy111_labels = [
                "Park View Drive",
                "Highway 74",
                "Jackalope Trail",
                "Shields Road",
                "Oasis Street",
                "Smurr Street",
                "Jackson Street",
                "Golf Center Parkway",
                "Indio Blvd",
            ]

            epalm_canyon_labels = [
                "Canyon Plaza Drive",
                "Perez Road",
                "Auto Park Drive",
                "Bankside Drive",
                "Cathedral Canyon Drive",
                "Buddy Rogers Avenue",
                "Van Fleet Street",
                "Date Palm Drive",
                "Sun Gate Way",
                "Officer Jermain Gibson",
            ]

            ordered_labels = washington_labels + hwy111_labels + epalm_canyon_labels

            # --- Hydrate from URL query params (once) ---
            if not st.session_state.get("t2_qp_hydrated", False):
                qp = st.query_params
                try:
                    qp_corr = qp.get("t2_corridor")
                    if qp_corr and qp_corr in corridors and "corridor_vol" not in st.session_state:
                        st.session_state["corridor_vol"] = qp_corr

                    # Intersection depends on corridor; construct valid list for hydration
                    corr_for_list = st.session_state.get("corridor_vol", corridors[0]) if corridors else "All Corridors"
                    corr_df = volume_df if corr_for_list == "All Corridors" else volume_df[
                        volume_df["corridor_id"] == corr_for_list]
                    avail = (corr_df["intersection_name"].dropna().unique().tolist() if not corr_df.empty else [])

                    if corr_for_list == "Highway 111":
                        current_ordered_h = hwy111_labels
                    elif corr_for_list == "Highway 111 - E Palm Canyon Drive":
                        current_ordered_h = epalm_canyon_labels
                    elif corr_for_list == "Washington Street":
                        current_ordered_h = washington_labels
                    else:
                        current_ordered_h = ordered_labels

                    intersections_ordered = [lbl for lbl in current_ordered_h if lbl in avail]
                    intersections_pre = ["All Intersections"] + intersections_ordered

                    qp_inter_all = qp.get_all("t2_intersection")
                    if qp_inter_all:
                        # Filter to only those that exist in our ordered list
                        valid_qp = [i for i in qp_inter_all if i in intersections_ordered]
                        if valid_qp and "intersection_vol" not in st.session_state:
                            st.session_state["intersection_vol"] = valid_qp
                        elif "All Intersections" in qp_inter_all and "intersection_vol" not in st.session_state:
                            st.session_state["intersection_vol"] = []

                    # If a prior search was committed, restore it
                    if qp.get("t2_ready") == "1":
                        # Build restored params
                        ds = qp.get("t2_date_start")
                        de = qp.get("t2_date_end")
                        try:
                            d_start = pd.to_datetime(ds).date() if ds else None
                            d_end = pd.to_datetime(de).date() if de else None
                        except Exception:
                            d_start = d_end = None
                        gran = qp.get("t2_granularity") or "Hourly"
                        direc = qp.get("t2_direction")
                        show_approaches_qp = qp.get("t2_show_all_approaches") == "1"
                        shade_periods_qp = qp.get("t2_shade_periods") == "1"
                        inter_val = qp.get_all("t2_intersection")
                        if not inter_val:
                            inter = st.session_state.get("intersection_vol", "All Intersections")
                        else:
                            inter = inter_val if any(i in intersections_ordered for i in inter_val) else "All Intersections"

                        t2_params_h = {
                            "corridor": st.session_state.get("corridor_vol", corridors[0]),
                            "intersection": inter or "SELECT",
                            "date_range_vol": (d_start, d_end) if d_start and d_end else None,
                            "granularity_vol": gran,
                            "direction_filter": direc if direc and direc != "All Directions" else None,
                            "show_all_approaches": show_approaches_qp,
                            "shade_periods": shade_periods_qp,
                        }
                        st.session_state["t2_params"] = t2_params_h
                        st.session_state["t2_ready"] = True
                        st.session_state["last_active_tab"] = "t2"
                finally:
                    st.session_state["t2_qp_hydrated"] = True

            # Corridor selector
            st.markdown("## 🛣️ Select Corridor")
            corridor = st.selectbox(
                "🛣️ Select Corridor",
                corridors,
                key="corridor_vol",
                label_visibility="collapsed",
            )

            # Build intersections list based on corridor
            if not volume_df.empty and "intersection_name" in volume_df.columns:
                corr_df = volume_df if corridor == "All Corridors" else volume_df[volume_df["corridor_id"] == corridor]
                avail = (corr_df["intersection_name"].dropna().unique().tolist() if not corr_df.empty else [])

                # Use appropriate ordered list based on selected corridor
                if corridor == "Highway 111":
                    current_ordered = hwy111_labels
                elif corridor == "Highway 111 - E Palm Canyon Drive":
                    current_ordered = epalm_canyon_labels
                elif corridor == "Washington Street":
                    current_ordered = washington_labels
                else:
                    current_ordered = ordered_labels

                intersections_ordered = [lbl for lbl in current_ordered if lbl in avail]
                intersections = ["SELECT"] + (
                    ["All Intersections"] + intersections_ordered if intersections_ordered else ["All Intersections"])
            else:
                intersections = ["SELECT", "All Intersections"]

            st.markdown("## 🚦 Select Intersection(s)")
            intersection = st.multiselect(
                "🚦 Select Intersection(s)",
                intersections_ordered,
                key="intersection_vol",
                label_visibility="collapsed",
                placeholder="All Intersections"
            )
            if not intersection:
                intersection = "All Intersections"

            # Info caption listing which corridor intersections are currently missing (no data in selection)
            try:
                if corridor == "Highway 111":
                    current_ordered = hwy111_labels
                elif corridor == "Highway 111 - E Palm Canyon Drive":
                    current_ordered = epalm_canyon_labels
                elif corridor == "Washington Street":
                    current_ordered = washington_labels
                else:
                    current_ordered = []  # Don't show missing for "All Corridors" or mixed

                if current_ordered:
                    missing_intersections = [lbl for lbl in current_ordered if lbl not in avail]
                    if missing_intersections:
                        st.caption("No data available for: " + ", ".join(missing_intersections))
            except Exception:
                pass

            # Detect corridor changes to update URL and reset readiness
            prev_corridor = st.session_state.get("corridor_vol_prev")
            if prev_corridor != corridor:
                st.session_state["corridor_vol_prev"] = corridor
                try:
                    st.query_params.update(t2_corridor=corridor)
                    # Reset readiness until user searches again
                    if "t2_ready" in st.session_state:
                        del st.session_state["t2_ready"]
                    if "t2_params" in st.session_state:
                        del st.session_state["t2_params"]
                    for k in ("t2_intersection", "t2_date_start", "t2_date_end", "t2_granularity", "t2_direction"):
                        if k in st.query_params:
                            del st.query_params[k]
                except Exception:
                    pass

            # Detect selection changes to drive loading animation and URL state
            prev_intersection = st.session_state.get("intersection_vol_prev")
            if prev_intersection != intersection:
                st.session_state["intersection_vol_prev"] = intersection
                # Update URL query params for persistence
                if intersection != "SELECT":
                    # Loading bar while we compute/refresh availability UI
                    pb = st.progress(0, text="Loading Data availability info...")
                    for i in range(0, 101, 10):
                        time.sleep(0.02)
                        pb.progress(i, text="Loading Data availability info...")
                    pb.empty()
                    try:
                        st.query_params.update(t2_corridor=corridor, t2_intersection=intersection)
                        # Do not mark ready unless user presses Search
                        if "t2_ready" in st.query_params:
                            del st.query_params["t2_ready"]
                        for k in ("t2_date_start", "t2_date_end", "t2_granularity", "t2_direction"):
                            if k in st.query_params:
                                del st.query_params[k]
                    except Exception:
                        pass
                else:
                    # User re-selected placeholder: hide controls and clear state + URL
                    for k in ("t2_ready", "t2_params"):
                        if k in st.session_state:
                            del st.session_state[k]
                    try:
                        for k in ("t2_corridor", "t2_intersection", "t2_ready", "t2_date_start", "t2_date_end",
                                  "t2_granularity", "t2_direction"):
                            if k in st.query_params:
                                del st.query_params[k]
                    except Exception:
                        pass

            # Availability preview for selected intersection (always visible)
            try:
                # When in initial state ("SELECT"), summarize overall dataset
                base_df = volume_df if volume_df is not None else pd.DataFrame()
                intersection_for_avail = None if intersection == "SELECT" else intersection
                avail = compute_data_availability(
                    base_df,
                    intersection_col="intersection_name",
                    intersection=intersection_for_avail,
                    max_gaps=3,
                    current_date=datetime.now(),
                )
                if avail.get("start") and avail.get("end"):
                    start_str = avail["start"].strftime("%b %d, %Y %I:%M %p")
                    end_str = avail["end"].strftime("%b %d, %Y %I:%M %p")
                    mb = avail.get("size_mb", 0.0)
                    size_str = f"({mb:.1f} MB)" if mb > 0 else ""
                    # Dynamic header based on selection
                    if intersection == "All Intersections":
                        header_label = "Available Data for this Corridor"
                    else:
                        header_label = f"Available Data for {', '.join(intersection) if isinstance(intersection, list) else intersection}"
                    st.caption(header_label)
                    st.caption(f"• Date Range: {start_str} → {end_str} {size_str}")
                    gaps = avail.get("gaps") or []
                    if len(gaps) == 0:
                        st.caption("• Missing Data: None")
                    else:
                        st.caption("• Missing Data: " + "; ".join(gaps))
            except Exception as _e:
                # Don't fail sidebar if preview errors
                pass

            # Progressive disclosure
            if True:
                if volume_df.empty or "local_datetime" not in volume_df.columns:
                    min_date = datetime.today().date() - timedelta(days=7)
                    max_date = datetime.today().date()
                else:
                    min_date = volume_df["local_datetime"].dt.date.min()
                    max_date = volume_df["local_datetime"].dt.date.max()

                st.markdown("## 📅 Date And Time")
                date_range_vol = date_range_preset_controls(min_date, max_date, key_prefix="vol")

                # Comparison toggle
                compare_mode = st.toggle(
                    "Compare with another period", 
                    value=st.session_state.get("t2_compare_mode", False), 
                    key="t2_compare_mode",
                    help="Enable to select a second date range for side-by-side comparison"
                )
                date_range_comp = None
                if compare_mode:
                    st.markdown("### 📅 Comparison Period")
                    date_range_comp = date_range_preset_controls(min_date, max_date, key_prefix="vol_comp")

                st.markdown("## Granularity")
                granularity_vol = st.selectbox(
                    "Data Aggregation",
                    ["Hourly", "Daily", "Weekly", "Monthly"],
                    index=0,
                    key="granularity_vol",
                )

                # Direction options scoped to selected corridor (and intersection if chosen)
                if not volume_df.empty and "direction" in volume_df.columns:
                    scope_df = volume_df
                    if corridor != "All Corridors":
                        scope_df = scope_df[scope_df["corridor_id"] == corridor]
                    if intersection not in ("All Intersections", "SELECT"):
                        if isinstance(intersection, list):
                            scope_df = scope_df[scope_df["intersection_name"].isin(intersection)]
                        else:
                            scope_df = scope_df[scope_df["intersection_name"] == intersection]
                    dirs = sorted(scope_df["direction"].dropna().unique().tolist()) if not scope_df.empty else []
                    direction_options = dirs if dirs else ["No Data"]
                else:
                    direction_options = ["No Data"]
                direction_filter = st.selectbox("🔄 Direction Filter", direction_options, key="direction_filter_vol")

                try:
                    show_all_approaches = st.toggle(
                        "Show All Approaches",
                        value=st.session_state.get("t2_show_all_approaches", False),
                        key="t2_show_all_approaches",
                        help="Compare the selected approach with all other available directions"
                    )
                except Exception:
                    show_all_approaches = st.checkbox(
                        "Show All Approaches",
                        value=st.session_state.get("t2_show_all_approaches", False),
                        key="t2_show_all_approaches",
                        help="Compare the selected approach with all other available directions"
                    )

                shade_periods = False
                if date_range_vol and len(date_range_vol) == 2 and date_range_vol[0] == date_range_vol[1]:
                    try:
                        shade_periods = st.toggle(
                            "Shade Time Periods",
                            value=st.session_state.get("t2_shade_periods", False),
                            key="t2_shade_periods",
                            help="Shade AM, Mid-day, and PM periods on the trend chart"
                        )
                    except Exception:
                        shade_periods = st.checkbox(
                            "Shade Time Periods",
                            value=st.session_state.get("t2_shade_periods", False),
                            key="t2_shade_periods",
                            help="Shade AM, Mid-day, and PM periods on the trend chart"
                        )

                # track uncommitted controls
                t2_current = {
                    "corridor": corridor,
                    "intersection": intersection,
                    "date_range_vol": tuple(date_range_vol) if date_range_vol else None,
                    "compare_mode": compare_mode,
                    "date_range_comp": tuple(date_range_comp) if date_range_comp else None,
                    "granularity_vol": granularity_vol,
                    "direction_filter": direction_filter,
                    "show_all_approaches": show_all_approaches,
                    "shade_periods": shade_periods,
                }
                st.session_state["t2_current"] = t2_current

                if st.button("🔍 **Generate**", key="search_tab2", type="primary", use_container_width=True):
                    st.session_state["t2_params"] = t2_current
                    st.session_state["t2_ready"] = True
                    set_active_search_tab("t2")
                    st.session_state["last_active_tab"] = "t2"
                    # Persist committed search to URL
                    try:
                        ds, de = None, None
                        if t2_current.get("date_range_vol"):
                            ds = t2_current["date_range_vol"][0].isoformat()
                            de = t2_current["date_range_vol"][1].isoformat()
                        
                        dcs, dce = None, None
                        if t2_current.get("date_range_comp"):
                            dcs = t2_current["date_range_comp"][0].isoformat()
                            dce = t2_current["date_range_comp"][1].isoformat()

                        st.query_params.update(
                            t2_ready="1",
                            t2_corridor=corridor,
                            t2_intersection=intersection,
                            t2_date_start=ds or "",
                            t2_date_end=de or "",
                            t2_compare_mode="1" if compare_mode else "0",
                            t2_comp_start=dcs or "",
                            t2_comp_end=dce or "",
                            t2_granularity=granularity_vol,
                            t2_direction=direction_filter,
                            t2_show_all_approaches="1" if show_all_approaches else "0",
                            t2_shade_periods="1" if shade_periods else "0",
                            last_tab="t2",
                        )
                    except Exception:
                        pass
            else:
                # Clean up any stale current params to avoid main area prompts depending on hidden inputs
                st.session_state["t2_current"] = {"intersection": intersection}
                # Also ensure main area resets and URL cleared when user re-selects placeholder
                if st.session_state.get("t2_ready"):
                    del st.session_state["t2_ready"]
                if st.session_state.get("t2_params"):
                    del st.session_state["t2_params"]
                try:
                    for k in ("t2_intersection", "t2_ready", "t2_date_start", "t2_date_end", "t2_granularity",
                              "t2_direction"):
                        if k in st.query_params:
                            del st.query_params[k]
                except Exception:
                    pass

    # -------- Main content area (render only when "Generate" committed) --------
    t2_ready = st.session_state.get("t2_ready", False)
    t2_params = st.session_state.get("t2_params", {})
    t2_pending = t2_ready and _freeze_params(t2_params) != _freeze_params(st.session_state.get("t2_current", {}))

    if not t2_ready:
        st.info("Choose your Intersection and Date Range in the settings to the left.")
    else:
        if t2_pending:
            st.warning("⚙️ Press **Generate** to refresh.")

        try:
            base_df = volume_df.copy() if not volume_df.empty else pd.DataFrame()
            if base_df.empty:
                st.error("❌ Failed to load volume Prediction. Please check your Prediction sources.")
            else:
                # Unpack committed params
                corridor = t2_params.get("corridor", "All Corridors")
                intersection = t2_params.get("intersection", "All Intersections")
                date_range_vol = t2_params.get("date_range_vol")
                granularity_vol = t2_params.get("granularity_vol", "Hourly")
                direction_filter = t2_params.get("direction_filter")
                show_all_approaches = t2_params.get("show_all_approaches", False)
                shade_periods = t2_params.get("shade_periods", False)

                if direction_filter is None and not show_all_approaches:
                    # Fallback if hydration didn't find a direction
                    direction_filter = "All Directions" 

                if corridor != "All Corridors" and "corridor_id" in base_df.columns:
                    base_df = base_df[base_df["corridor_id"] == corridor]
                if intersection != "All Intersections":
                    if isinstance(intersection, list):
                        base_df = base_df[base_df["intersection_name"].isin(intersection)]
                    else:
                        base_df = base_df[base_df["intersection_name"] == intersection]
                if not show_all_approaches and direction_filter not in ("All Directions", "No Data") and "direction" in base_df.columns:
                    base_df = base_df[base_df["direction"] == direction_filter]

                # Two-column layout with sticky right rail
                content_col, right_col = st.columns([7, 3.5], gap="large")

                # Right rail (sticky overview map)
                with right_col:
                    st.markdown('<div id="vol-map-anchor"></div>', unsafe_allow_html=True)
                    st.markdown("##### Corridor Map", help="Stays visible while you scroll the analysis on the left.")

                    satellite_t2 = st.toggle("🛰️ Satellite View", key="satellite_t2", value=False)
                    try:
                        t2_tooltip_map = st.session_state.get("t2_tooltip_map")
                        fig_over = build_intersections_overview(
                            selected_label=None if intersection == "All Intersections" else intersection,
                            corridor=corridor,
                            tooltip_map=t2_tooltip_map,
                            satellite=satellite_t2
                        )
                    except Exception:
                        fig_over = None

                    if fig_over:
                        try:
                            fig_over.update_layout(height=MAP_HEIGHT, margin=dict(l=0, r=0, t=32, b=0))
                        except Exception:
                            pass
                        st.markdown('<div class="cvag-map-card">', unsafe_allow_html=True)
                        st.plotly_chart(fig_over, use_container_width=True, config=PLOTLY_CONFIG)
                        if intersection != "All Intersections":
                            st.caption(f"Selected: **{', '.join(intersection) if isinstance(intersection, list) else intersection}**")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="cvag-map-card">', unsafe_allow_html=True)
                        st.caption("Map: unable to render overview (missing coordinates/GeoJSON).")
                        st.markdown('</div>', unsafe_allow_html=True)

                # Main analysis content
                with content_col:
                    if not date_range_vol or len(date_range_vol) != 2:
                        st.warning("⚠️ Please select both start and end dates to proceed with the volume analysis.")
                    else:
                        # ---- CAD-style loader starts here ----
                        with scoped_cad_loader("Fetching Data...", tab_id="t2") as step:
                            step("Applying filters & aggregations", 20)
                            filtered_volume_data = process_traffic_data(base_df, date_range_vol, granularity_vol)

                            if filtered_volume_data.empty:
                                step("No volume Prediction in selected range", 100)
                                st.warning("⚠️ No volume Prediction available for the selected range.")
                            else:
                                span = (date_range_vol[1] - date_range_vol[0]).days + 1
                                total_obs = len(filtered_volume_data)

                                # Spell out direction for the header
                                dir_map = {"NB": "Northbound", "SB": "Southbound", "EB": "Eastbound", "WB": "Westbound"}
                                if show_all_approaches:
                                    dir_display_vol = "All Approaches"
                                elif direction_filter:
                                    dir_display_vol = dir_map.get(direction_filter.upper(), direction_filter)
                                else:
                                    dir_display_vol = "N/A"

                                step("Preparing summary context", 35)
                                st.markdown(
                                    f"""
                                                                    <div style="
                                                                        background: linear-gradient(135deg, #2b77e5 0%, #19c3e6 100%);
                                                                        border-radius:16px; padding:18px 20px; color:#fff; margin:8px 0 14px;
                                                                        box-shadow:0 10px 26px rgba(25,115,210,.25); text-align:left;
                                                                        font-family: inherit;">
                                                                      <div style="display:flex; align-items:center; gap:10px;">
                                                                        <div>
                                                                            <div style="font-size:1.7rem; font-weight:800; letter-spacing:-0.01em; margin-bottom:2px;">
                                                                              <span style="color: #ffffff;">2025 BNP PARIBUS OPEN INDIAN WELLS DASHBOARD</span><span style="color: rgba(255,255,255,0.7);">: Q-FREE KINETIC MOBILITY VEHICLE VOLUME ANALYSIS</span>
                                                                            </div>
                                                                            <div style="font-size:1.1rem;font-weight:600;opacity:0.9; display:flex; flex-wrap:wrap; gap:15px; margin-top:4px; align-items:center;">
                                                                              <div style="background:rgba(255,255,255,0.15); padding:4px 12px; border-radius:8px; font-size:1.2rem; border:1px solid rgba(255,255,255,0.3); font-weight:700;">
                                                                                📅 {date_range_vol[0].strftime('%b %d, %Y')} to {date_range_vol[1].strftime('%b %d, %Y')} <span style="font-weight:400; opacity:0.8;">({span} days)</span>
                                                                              </div>
                                                                              <div style="background:rgba(255,255,255,0.15); padding:4px 12px; border-radius:8px; font-size:1.2rem; border:1px solid rgba(255,255,255,0.3);">
                                                                                <span style="opacity:0.9; font-weight:400;">Intersection(s):</span> <span style="font-weight:800;">{", ".join(intersection) if isinstance(intersection, list) else intersection}</span> <span style="background:rgba(255,255,255,0.2); padding:2px 8px; border-radius:4px; margin-left:8px; font-size:0.9rem; font-weight:700; color:#fff;">{dir_display_vol}</span>
                                                                              </div>
                                                                            </div>
                                                                            <div style="font-size:0.95rem; opacity:0.85; display:flex; flex-wrap:wrap; gap:15px; margin-top:10px; align-items:center;">
                                                                              <div style="display:flex; align-items:center; gap:15px;">
                                                                                <div><span style="opacity:0.8; font-weight:400;">Region:</span> Coachella Valley</div>
                                                                                <div style="border-left: 1px solid rgba(255,255,255,0.3); padding-left: 15px;"><span style="opacity:0.8; font-weight:400;">City:</span> Indian Wells / La Quinta</div>
                                                                                <div style="border-left: 1px solid rgba(255,255,255,0.3); padding-left: 15px;"><span style="opacity:0.8; font-weight:400;">Corridor:</span> {corridor}</div>
                                                                                <div style="border-left: 1px solid rgba(255,255,255,0.3); padding-left: 15px;">📊 {granularity_vol} Aggregation</div>
                                                                                <div style="border-left: 1px solid rgba(255,255,255,0.3); padding-left: 15px;">✅ {total_obs:,} observations</div>
                                                                              </div>
                                                                            </div>
                                                                        </div>
                                                                      </div>
                                                                    </div>
                                                                    """,
                                    unsafe_allow_html=True,
                                )

                                # ---- Windowed raw hourly Prediction for robust KPI math ----
                                step("Computing KPIs & risk indicators", 60)
                                raw = base_df[
                                    (base_df["local_datetime"].dt.date >= date_range_vol[0])
                                    & (base_df["local_datetime"].dt.date <= date_range_vol[1])
                                    ].copy()
                                raw["total_volume"] = pd.to_numeric(raw.get("total_volume", np.nan), errors="coerce")
                                raw["local_datetime"] = pd.to_datetime(raw["local_datetime"])

                                raw_comp = None
                                if t2_params.get("compare_mode") and t2_params.get("date_range_comp"):
                                    dr_comp = t2_params["date_range_comp"]
                                    raw_comp = base_df[
                                        (base_df["local_datetime"].dt.date >= dr_comp[0]) &
                                        (base_df["local_datetime"].dt.date <= dr_comp[1])
                                    ].copy()
                                    raw_comp["total_volume"] = pd.to_numeric(raw_comp.get("total_volume", np.nan), errors="coerce")
                                    raw_comp["local_datetime"] = pd.to_datetime(raw_comp["local_datetime"])

                                # 1. Generate Tooltip Map for all intersections in Tab 2
                                tooltip_map = {}
                                node_coords = get_node_coordinates()

                                # Calculate ADTs for ALL intersections in the full volume_df for the date range
                                # to ensure "blue dots" show data regardless of corridor/direction filters.
                                full_range_df = volume_df[
                                    (volume_df["local_datetime"].dt.date >= date_range_vol[0]) &
                                    (volume_df["local_datetime"].dt.date <= date_range_vol[1])
                                ].copy()

                                if not full_range_df.empty:
                                    # Calculate ADT as the average of daily totals for each intersection
                                    # This matches the KPI logic (Average Daily Traffic)
                                    full_range_df["date_only"] = full_range_df["local_datetime"].dt.date
                                    daily_totals = full_range_df.groupby(["intersection_name", "date_only"])["total_volume"].sum()
                                    adt_data = daily_totals.groupby("intersection_name").mean()
                                else:
                                    adt_data = pd.Series()

                                # Build tooltips for each known intersection
                                # Mapping for Cathedral City intersections
                                cath_city_data = {
                                    "Canyon Plaza Drive": {"no": "24", "phase": "1", "cycle": "140 sec"},
                                    "Perez Road": {"no": "25", "phase": "1", "cycle": "140 sec"},
                                    "Auto Park Drive": {"no": "26", "phase": "1", "cycle": "140 sec"},
                                    "Bankside Drive": {"no": "27", "phase": "1", "cycle": "140 sec"},
                                    "Cathedral Canyon Drive": {"no": "28", "phase": "1", "cycle": "140 sec"},
                                    "Buddy Rogers Avenue": {"no": "29", "phase": "1", "cycle": "140 sec"},
                                    "Van Fleet Street": {"no": "30", "phase": "1", "cycle": "140 sec"},
                                    "Date Palm Drive": {"no": "32", "phase": "1", "cycle": "140 sec"},
                                    "Sun Gate Way": {"no": "33", "phase": "1", "cycle": "140 sec"},
                                    "Officer Jermain Gibson": {"no": "34", "phase": "1", "cycle": "140 sec"},
                                }

                                # Mapping for Washington St corridor
                                washington_st_data = {
                                    "Washington St & Avenue 52": {"no": "102", "agency": "La Quinta", "cycle": "140 sec"},
                                    "Washington St & Calle Tampico": {"no": "103", "agency": "La Quinta", "cycle": "140 sec"},
                                    "Washington St & Village Shopping Center": {"no": "104", "agency": "La Quinta", "cycle": "140 sec"},
                                    "Washington St & Avenue 50": {"no": "105", "agency": "La Quinta", "cycle": "140 sec"},
                                    "Washington St & Sagebrush Avenue": {"no": "106", "agency": "La Quinta", "cycle": "140 sec"},
                                    "Washington St & Eisenhower Drive": {"no": "107", "agency": "La Quinta", "cycle": "140 sec"},
                                    "Washington St & Avenue 48": {"no": "108", "agency": "La Quinta", "cycle": "140 sec"},
                                    "Washington St & Avenue 47": {"no": "109", "agency": "La Quinta", "cycle": "140 sec"},
                                    "Washington St & Point Happy Simon": {"no": "110", "agency": "La Quinta", "cycle": "150 sec"},
                                    "Washington St & Hwy 111": {"no": "111", "agency": "La Quinta", "cycle": "150 sec"},
                                    "Washington St & Channel Drive": {"no": "112", "agency": "La Quinta", "cycle": "150 sec"},
                                    "Washington St & Miles Avenue": {"no": "113", "agency": "La Quinta", "cycle": "150 sec"},
                                    "Washington St & Via Sevilla": {"no": "114", "agency": "La Quinta", "cycle": "150 sec"},
                                    "Washington St & Fred Waring Drive": {"no": "115", "agency": "Palm Desert", "cycle": "150 sec"},
                                    "Washington St & Palm Royale Drive": {"no": "116", "agency": "Palm Desert", "cycle": "150 sec"},
                                    "Washington St & Avenue of the States": {"no": "117", "agency": "Palm Desert", "cycle": "150 sec"},
                                    "Washington St & Avenue 42": {"no": "118", "agency": "Palm Desert", "cycle": "150 sec"},
                                    "Washington St & Avenue 41": {"no": "119", "agency": "Palm Desert", "cycle": "150 sec"},
                                    "Washington St & Country Club Drive": {"no": "120", "agency": "Palm Desert", "cycle": "150 sec"},
                                    "Washington St & Varner Road": {"no": "122", "agency": "Palm Desert", "cycle": "150 sec"},
                                }

                                # Add aliases to ensure mapping works regardless of normalization
                                washington_st_data.update({
                                    "Washington St & Avenue50": washington_st_data["Washington St & Avenue 50"],
                                    "Washington St & Avenue 50": washington_st_data["Washington St & Avenue 50"],
                                    "Washington St & Ave48": washington_st_data["Washington St & Avenue 48"],
                                    "Washington St & Avenue 48": washington_st_data["Washington St & Avenue 48"],
                                    "Washington St & Ave47": washington_st_data["Washington St & Avenue 47"],
                                    "Washington St & Avenue 47": washington_st_data["Washington St & Avenue 47"],
                                    "Washington St & Eisenhower": washington_st_data["Washington St & Eisenhower Drive"],
                                    "Washington St & Eisenhower Drive": washington_st_data["Washington St & Eisenhower Drive"],
                                    "Washington St & Village Shop Ctr": washington_st_data["Washington St & Village Shopping Center"],
                                    "Washington St & Village Shopping Center": washington_st_data["Washington St & Village Shopping Center"],
                                    "Washington St & Sagebrush Ave": washington_st_data["Washington St & Sagebrush Avenue"],
                                    "Washington St & Sagebrush Avenue": washington_st_data["Washington St & Sagebrush Avenue"],
                                    "Washington St & Varner Road": washington_st_data["Washington St & Varner Road"],
                                    "Washington St & Country Club Drive": washington_st_data["Washington St & Country Club Drive"],
                                })

                                for disp_label, node_id in INTERSECTION_TO_NODE.items():
                                    coords = node_coords.get(node_id)
                                    gps_str = f"{coords[0]:.5f}, {coords[1]:.5f}" if coords else "Unknown"

                                    # Default values
                                    protocol = "Q-Free Kinetic Mobility"
                                    agency = "CVAG"
                                    cvag_phase = "N/A"
                                    intersection_no = "N/A"
                                    cycle_length = "N/A"

                                    # Apply Cathedral City updates
                                    if disp_label in cath_city_data:
                                        data = cath_city_data[disp_label]
                                        agency = "Cathedral City"
                                        cvag_phase = data["phase"]
                                        intersection_no = data["no"]
                                        cycle_length = data["cycle"]

                                    # Apply Washington St updates
                                    if disp_label in washington_st_data:
                                        data = washington_st_data[disp_label]
                                        agency = data.get("agency", agency)
                                        intersection_no = data["no"]
                                        cycle_length = data["cycle"]
                                        if disp_label == "Washington St & Avenue 52":
                                            cvag_phase = "1"

                                    # Kinetic Mobility labels usually match the map display labels or can be found in adt_data
                                    adt_val = adt_data.get(disp_label, 0)

                                    tt_html = f"""
                                    <b>{disp_label}</b><br>
                                    GPS Location: {gps_str}<br>
                                    Protocol: {protocol}<br>
                                    Agency: {agency}<br>
                                    CVAG Phase: {cvag_phase}<br>
                                    Intersection No.: {intersection_no}<br>
                                    Cycle Length: {cycle_length}<br>
                                    ADT: {adt_val:,.0f}
                                    """
                                    tooltip_map[disp_label] = tt_html

                                st.session_state["t2_tooltip_map"] = tooltip_map

                                st.subheader(" Traffic Demand Performance Indicators")
                                if raw.empty or raw["total_volume"].dropna().empty:
                                    st.info("No raw hourly volume in this window.")
                                else:
                                    bucket_all = \
                                    _prep_bucket(raw, granularity_vol).groupby("local_datetime", as_index=False)[
                                        "total_volume"].sum().sort_values("local_datetime")
                                    
                                    # Comparison KPI logic
                                    comp_metrics = None
                                    if raw_comp is not None and not raw_comp.empty:
                                        bucket_comp = _prep_bucket(raw_comp, granularity_vol).groupby("local_datetime", as_index=False)[
                                            "total_volume"].sum().sort_values("local_datetime")
                                        if granularity_vol == "Monthly":
                                            bucket_comp["bucket_hours"] = pd.to_datetime(
                                                bucket_comp["local_datetime"]).dt.days_in_month * 24
                                        else:
                                            bucket_comp["bucket_hours"] = AGG_META[granularity_vol]["fixed_hours"]
                                        
                                        peak_idx_c = int(bucket_comp["total_volume"].idxmax())
                                        peak_val_c = float(bucket_comp.loc[peak_idx_c, "total_volume"])
                                        avg_bucket_val_c = float(bucket_comp["total_volume"].mean())
                                        total_vehicles_c = float(np.nansum(raw_comp["total_volume"]))
                                        cv_bucket_c = (float(np.nanstd(bucket_comp["total_volume"])) / avg_bucket_val_c * 100) if avg_bucket_val_c > 0 else 0.0
                                        high_hours_c = int((raw_comp["total_volume"] > HIGH_VOLUME_THRESHOLD_VPH).sum())
                                        total_hours_c = int(raw_comp["total_volume"].count())
                                        risk_pct_c = (high_hours_c / total_hours_c * 100) if total_hours_c > 0 else 0.0
                                        
                                        comp_metrics = {
                                            "peak": peak_val_c,
                                            "avg": avg_bucket_val_c,
                                            "total": total_vehicles_c,
                                            "consistency": max(0, 100 - cv_bucket_c),
                                            "high_hours": high_hours_c,
                                            "risk_pct": risk_pct_c
                                        }

                                    if granularity_vol == "Monthly":
                                        bucket_all["bucket_hours"] = pd.to_datetime(
                                            bucket_all["local_datetime"]).dt.days_in_month * 24
                                    else:
                                        bucket_all["bucket_hours"] = AGG_META[granularity_vol]["fixed_hours"]

                                    bucket_all["cap"] = bucket_all["bucket_hours"] * THEORETICAL_LINK_CAPACITY_VPH
                                    util_series = np.where(bucket_all["cap"] > 0,
                                                           bucket_all["total_volume"] / bucket_all["cap"] * 100, np.nan)

                                    peak_idx = int(bucket_all["total_volume"].idxmax())
                                    peak_val = float(bucket_all.loc[peak_idx, "total_volume"])
                                    peak_cap = float(bucket_all.loc[peak_idx, "cap"])
                                    peak_util_pct = (peak_val / peak_cap * 100) if peak_cap > 0 else 0.0
                                    peak_date = pd.to_datetime(bucket_all.loc[peak_idx, "local_datetime"])

                                    avg_bucket_val = float(bucket_all["total_volume"].mean())
                                    avg_util_pct = float(np.nanmean(util_series)) if np.isfinite(
                                        util_series).any() else 0.0

                                    hourly_avg = float(np.nanmean(raw["total_volume"])) if raw[
                                        "total_volume"].notna().any() else 0.0
                                    cv_hourly = (float(
                                        np.nanstd(raw["total_volume"])) / hourly_avg * 100) if hourly_avg > 0 else 0.0
                                    cv_bucket = (float(np.nanstd(bucket_all[
                                                                     "total_volume"])) / avg_bucket_val * 100) if avg_bucket_val > 0 else 0.0

                                    high_hours = int((raw["total_volume"] > HIGH_VOLUME_THRESHOLD_VPH).sum())
                                    total_hours = int(raw["total_volume"].count())
                                    risk_pct = (high_hours / total_hours * 100) if total_hours > 0 else 0.0

                                    unit = AGG_META[granularity_vol]["unit"]
                                    if granularity_vol == "Hourly":
                                        avg_label = "Hourly Volume"
                                        peak_label = "🔥 Peak Hourly Volume"
                                        avg_suffix = "Vehicles"
                                        # Include the intersection that has the peak hourly volume at this time
                                        peak_intersection = None
                                        try:
                                            if "intersection_name" in raw.columns:
                                                hour_mask = pd.to_datetime(raw["local_datetime"]).dt.floor("H") == pd.to_datetime(peak_date).floor("H")
                                                hour_df = raw.loc[hour_mask]
                                                if not hour_df.empty and "total_volume" in hour_df.columns:
                                                    by_int = hour_df.groupby("intersection_name", as_index=False)["total_volume"].sum()
                                                    if not by_int.empty:
                                                        peak_intersection = str(by_int.loc[int(by_int["total_volume"].idxmax()), "intersection_name"])
                                        except Exception:
                                            peak_intersection = None
                                        base_period = f"{peak_date.strftime('%A')}, {peak_date.strftime('%m/%d/%Y %H:00')}"
                                        peak_period_str = base_period + (f" • {peak_intersection}" if peak_intersection else "")
                                    elif granularity_vol == "Daily":
                                        avg_label = "Daily Traffic (DT)"
                                        peak_label = "🔥 Peak Daily Volume"
                                        avg_suffix = "Vehicles"
                                        peak_period_str = f"{peak_date.strftime('%A')}, {peak_date.strftime('%m/%d/%Y')}"
                                    elif granularity_vol == "Weekly":
                                        avg_label = "Weekly Traffic (WT)"
                                        peak_label = "🔥 Peak Weekly Volume"
                                        avg_suffix = "Vehicles"
                                        _p = pd.Period(peak_date, freq='W')
                                        _ws, _we = _p.start_time, _p.end_time
                                        peak_period_str = f"{_ws.strftime('%m/%d/%Y')} – {_we.strftime('%m/%d/%Y')}"
                                    else:
                                        avg_label = "Monthly Traffic (MT)"
                                        peak_label = "🔥 Peak Monthly Volume"
                                        avg_suffix = "Vehicles"
                                        peak_period_str = peak_date.strftime('%B %Y')

                                    col1, col2, col3, col4, col5 = st.columns(5)

                                    with col1:
                                        badge = (
                                            "badge-critical" if peak_util_pct > 90 else
                                            "badge-poor" if peak_util_pct > 75 else
                                            "badge-fair" if peak_util_pct > 60 else
                                            "badge-good"
                                        )
                                        # Always display peak in vehicles and show the exact peak period in delta
                                        st.metric(peak_label, f"{peak_val:,.0f} Vehicles", delta=peak_period_str)
                                        if comp_metrics:
                                            diff = peak_val - comp_metrics["peak"]
                                            st.caption(f"vs Comp: {comp_metrics['peak']:,.0f} ({'+' if diff > 0 else ''}{diff:,.0f})")
                                        st.markdown(
                                            f'<span class="performance-badge {badge}">{peak_util_pct:.0f}% of Capacity</span>',
                                            unsafe_allow_html=True,
                                        )

                                    with col2:
                                        st.metric(
                                            f"📊 {avg_label}",
                                            f"{avg_bucket_val:,.0f} {avg_suffix}",
                                            help=("Traffic on the selected aggregation.\n"
                                                  "• DT = daily traffic\n• WT = weekly traffic\n• MT = monthly traffic"),
                                        )
                                        if comp_metrics:
                                            diff = avg_bucket_val - comp_metrics["avg"]
                                            st.caption(f"vs Comp: {comp_metrics['avg']:,.0f} ({'+' if diff > 0 else ''}{diff:,.0f})")
                                        if granularity_vol == "Hourly":
                                            avg_util_pct_hourly = (
                                                        hourly_avg / THEORETICAL_LINK_CAPACITY_VPH * 100) if THEORETICAL_LINK_CAPACITY_VPH else 0.0
                                            badge2 = "badge-good" if avg_util_pct_hourly <= 40 else (
                                                "badge-fair" if avg_util_pct_hourly <= 60 else "badge-poor")
                                            st.markdown(
                                                f'<span class="performance-badge {badge2}">{avg_util_pct_hourly:.0f}% Avg Util</span>',
                                                unsafe_allow_html=True,
                                            )
                                        else:
                                            badge2 = "badge-good" if avg_util_pct <= 40 else (
                                                "badge-fair" if avg_util_pct <= 60 else "badge-poor")
                                            st.markdown(
                                                f'<span class="performance-badge {badge2}">{avg_util_pct:.0f}% Avg Util</span>',
                                                unsafe_allow_html=True,
                                            )

                                    with col3:
                                        total_vehicles = float(np.nansum(raw["total_volume"]))
                                        st.metric(
                                            "🚗 Total Vehicles (period)",
                                            f"{total_vehicles:,.0f} Vehicles",
                                            help="Sum of vehicles across the selected time window (computed from hourly records).",
                                        )
                                        if comp_metrics:
                                            diff = total_vehicles - comp_metrics["total"]
                                            st.caption(f"vs Comp: {comp_metrics['total']:,.0f} ({'+' if diff > 0 else ''}{diff:,.0f})")
                                        state_badge = (
                                            "badge-good" if total_vehicles < 0.4 * THEORETICAL_LINK_CAPACITY_VPH * 24
                                            else "badge-fair" if total_vehicles < 0.7 * THEORETICAL_LINK_CAPACITY_VPH * 24
                                            else "badge-poor"
                                        )
                                        st.markdown(
                                            f'<span class="performance-badge {state_badge}">Period Total</span>',
                                            unsafe_allow_html=True,
                                        )

                                    with col4:
                                        st.metric(
                                            "🎯 Demand Consistency",
                                            f"{max(0, 100 - cv_bucket):.0f}%",
                                            delta=f"CV (bucket): {cv_bucket:.1f}%",
                                            help="Higher is steadier. CV calculated on bucket totals for the chosen aggregation."
                                        )
                                        if comp_metrics:
                                            diff = (max(0, 100 - cv_bucket)) - comp_metrics["consistency"]
                                            st.caption(f"vs Comp: {comp_metrics['consistency']:.0f}% ({'+' if diff > 0 else ''}{diff:.0f}%)")
                                        label_cons = "Consistent" if cv_bucket < 30 else (
                                            "Variable" if cv_bucket < 50 else "Highly Variable")
                                        badge_cons = "badge-good" if cv_bucket < 30 else (
                                            "badge-fair" if cv_bucket < 50 else "badge-poor")
                                        st.markdown(
                                            f'<span class="performance-badge {badge_cons}">{label_cons}</span>',
                                            unsafe_allow_html=True,
                                        )

                                    with col5:
                                        st.metric(
                                            "⚠️ High Volume Hours",
                                            f"{high_hours}",
                                            delta=f"{risk_pct:.1f}% of time",
                                            help=f"Hourly records with total_volume > {HIGH_VOLUME_THRESHOLD_VPH:,} vehicles (always computed on the hourly base).",
                                        )
                                        if comp_metrics:
                                            diff = high_hours - comp_metrics["high_hours"]
                                            st.caption(f"vs Comp: {comp_metrics['high_hours']} ({'+' if diff > 0 else ''}{diff})")
                                        level_badge = (
                                            "badge-critical" if risk_pct > 25 else
                                            "badge-poor" if risk_pct > 15 else
                                            "badge-fair" if risk_pct > 5 else
                                            "badge-good"
                                        )
                                        level = (
                                            "Very High" if risk_pct > 25 else
                                            "High" if risk_pct > 15 else
                                            "Moderate" if risk_pct > 5 else
                                            "Low"
                                        )
                                        st.markdown(
                                            f'<span class="performance-badge {level_badge}">{level} Risk</span>',
                                            unsafe_allow_html=True,
                                        )

                                # ---------------- Charts ----------------
                                step("Rendering charts", 80)
                                # Header with optional toggles
                                col_title, col_toggle = st.columns([0.85, 0.15])
                                with col_title:
                                    st.subheader("📈 Vehicle Volume Visualizations")
                                sum_all_monthly = False
                                with col_toggle:
                                    if granularity_vol == "Monthly":
                                        try:
                                            sum_all_monthly = st.toggle(
                                                "Sum All",
                                                value=st.session_state.get("t2_sum_all", False),
                                                key="t2_sum_all",
                                                help="Show summed total vehicles per month as a bar chart"
                                            )
                                        except Exception:
                                            sum_all_monthly = st.checkbox(
                                                "Sum All",
                                                value=st.session_state.get("t2_sum_all", False),
                                                key="t2_sum_all",
                                                help="Show summed total vehicles per month as a bar chart"
                                            )

                                if 'raw' in locals() and len(filtered_volume_data) > 1:
                                    try:
                                        # Build default charts
                                        dir_label_arg = dir_display_vol if direction_filter != "All Directions" else ""

                                        # All-directions slice for the heatmap (date-windowed, no direction filter)
                                        _base_no_dir = volume_df.copy()
                                        if corridor != "All Corridors" and "corridor_id" in _base_no_dir.columns:
                                            _base_no_dir = _base_no_dir[_base_no_dir["corridor_id"] == corridor]
                                        if intersection != "All Intersections":
                                            if isinstance(intersection, list):
                                                _base_no_dir = _base_no_dir[
                                                    _base_no_dir["intersection_name"].isin(intersection)]
                                            else:
                                                _base_no_dir = _base_no_dir[
                                                    _base_no_dir["intersection_name"] == intersection]
                                        _raw_all_dirs = _base_no_dir[
                                            (_base_no_dir["local_datetime"].dt.date >= date_range_vol[0]) &
                                            (_base_no_dir["local_datetime"].dt.date <= date_range_vol[1])
                                            ].copy()
                                        
                                        _raw_all_dirs_comp = None
                                        if t2_params.get("compare_mode") and t2_params.get("date_range_comp"):
                                            dr_comp = t2_params["date_range_comp"]
                                            _raw_all_dirs_comp = _base_no_dir[
                                                (_base_no_dir["local_datetime"].dt.date >= dr_comp[0]) &
                                                (_base_no_dir["local_datetime"].dt.date <= dr_comp[1])
                                            ].copy()

                                        # Determine primary street for the caption
                                        primary_street = ""
                                        if corridor == "Washington Street":
                                            primary_street = "Washington Street"
                                        elif "Highway 111" in corridor:
                                            primary_street = "Highway 111"

                                        if intersection == "All Intersections":
                                            int_label_arg = f"{corridor} Corridor" if corridor != "All Corridors" else "All Intersections"
                                        elif isinstance(intersection, list) and len(intersection) == 1:
                                            int_label_arg = f"{corridor} Corridor -- {intersection[0]}"
                                        elif isinstance(intersection, list):
                                            int_label_arg = f"{corridor} Corridor -- {len(intersection)} Intersections Selected"
                                        else:
                                            # Single string selection
                                            int_label_arg = f"{corridor} Corridor -- {intersection}" if corridor != "All Corridors" else intersection
                                        
                                        fig_trend, fig_box, fig_matrix = improved_volume_charts_for_tab2(
                                            raw_hourly_df=raw,
                                            granularity=granularity_vol,
                                            cap_vph=THEORETICAL_LINK_CAPACITY_VPH,
                                            high_vph=HIGH_VOLUME_THRESHOLD_VPH,
                                            direction_label=dir_label_arg,
                                            intersections=int_label_arg,
                                            date_range=date_range_vol,
                                            all_directions_df=_raw_all_dirs,
                                            show_all_approaches=show_all_approaches,
                                            shade_periods=shade_periods,
                                            comp_hourly_df=raw_comp,
                                            comp_date_range=t2_params.get("date_range_comp"),
                                            comp_all_directions_df=_raw_all_dirs_comp,
                                        )

                                        # If Monthly + Sum All, replace the trend with a monthly total bar chart
                                        if granularity_vol == "Monthly" and st.session_state.get("t2_sum_all", False):
                                            monthly = \
                                            _prep_bucket(raw, "Monthly").groupby("local_datetime", as_index=False)[
                                                "total_volume"].sum()
                                            if not monthly.empty:
                                                monthly["Month"] = pd.to_datetime(
                                                    monthly["local_datetime"]).dt.strftime("%B %Y")
                                                # Compute extra tooltip info
                                                dt_idx = pd.to_datetime(
                                                    monthly["local_datetime"])  # month period starts
                                                try:
                                                    days_in_month = dt_idx.dt.days_in_month
                                                except Exception:
                                                    # Fallback if dt accessor not available for index type
                                                    days_in_month = dt_idx.to_series().apply(
                                                        lambda d: pd.Period(d, freq='M').days_in_month).values
                                                monthly["Days"] = days_in_month
                                                monthly["Avg per day"] = monthly["total_volume"] / monthly[
                                                    "Days"].replace(0, np.nan)

                                                # Build hover text
                                                monthly["hover"] = monthly.apply(
                                                    lambda
                                                        r: f"{r['Month']}<br>Total: {int(r['total_volume']):,} vehicles" +
                                                           (f"<br>Avg/day: {r['Avg per day']:.0f}" if pd.notnull(
                                                               r['Avg per day']) else "") +
                                                           (f"<br>Days: {int(r['Days'])}" if pd.notnull(
                                                               r['Days']) else ""),
                                                    axis=1,
                                                )

                                                fig_bar = px.bar(
                                                    monthly, x="Month", y="total_volume",
                                                    title="Total Vehicles by Month",
                                                    labels={"total_volume": "Vehicles", "Month": "Month"},
                                                )
                                                fig_bar.update_traces(customdata=monthly[["hover"]].values,
                                                                      hovertemplate="%{customdata[0]}<extra></extra>")
                                                fig_bar.update_layout(yaxis_title="Vehicles", xaxis_title="Month",
                                                                      margin=dict(l=10, r=10, t=40, b=10))
                                                fig_trend = fig_bar

                                        if fig_trend:
                                            st.plotly_chart(fig_trend, use_container_width=True, config=PLOTLY_CONFIG)
                                        colA, colB = st.columns(2)
                                        with colA:
                                            if fig_box:
                                                st.plotly_chart(fig_box, use_container_width=True, config=PLOTLY_CONFIG)
                                        with colB:
                                            if fig_matrix:
                                                st.plotly_chart(fig_matrix, use_container_width=True,
                                                                config=PLOTLY_CONFIG)
                                    except Exception as e:
                                        st.error(f"❌ Error creating volume charts: {e}")

                                # ---------------- Insights ----------------
                                if 'raw' in locals() and not raw.empty:
                                    try:
                                        step("Generating insights & recommendations", 92)
                                        agg_all = \
                                        _prep_bucket(raw, granularity_vol).groupby("local_datetime", as_index=False)[
                                            "total_volume"].sum()
                                        if agg_all.empty:
                                            raise ValueError("No Prediction in selected window")

                                        if granularity_vol == "Monthly":
                                            agg_all["bucket_hours"] = pd.to_datetime(
                                                agg_all["local_datetime"]).dt.days_in_month * 24
                                        else:
                                            agg_all["bucket_hours"] = AGG_META[granularity_vol]["fixed_hours"]

                                        agg_all["cap"] = agg_all["bucket_hours"] * THEORETICAL_LINK_CAPACITY_VPH
                                        agg_all["thr"] = agg_all["bucket_hours"] * HIGH_VOLUME_THRESHOLD_VPH

                                        peak_idx = int(agg_all["total_volume"].idxmax())
                                        peak_val = float(agg_all.loc[peak_idx, "total_volume"])
                                        peak_ts = pd.to_datetime(agg_all.loc[peak_idx, "local_datetime"])
                                        avg_val = float(agg_all["total_volume"].mean())
                                        p95_val = float(np.nanpercentile(agg_all["total_volume"], 95)) if agg_all[
                                            "total_volume"].notna().any() else 0.0

                                        peak_cap = float(agg_all.loc[peak_idx, "cap"])
                                        peak_util_pct = (peak_val / peak_cap * 100) if peak_cap > 0 else 0.0

                                        util_series = np.where(agg_all["cap"] > 0,
                                                               agg_all["total_volume"] / agg_all["cap"], np.nan)
                                        p95_util_pct = float(np.nanpercentile(util_series * 100, 95)) if np.isfinite(
                                            util_series).any() else 0.0

                                        cv_bucket = (float(
                                            np.nanstd(agg_all["total_volume"])) / avg_val * 100) if avg_val > 0 else 0.0
                                        peak_to_avg = (peak_val / avg_val) if avg_val > 0 else 0.0

                                        hourly_over_thr = int((raw["total_volume"] > HIGH_VOLUME_THRESHOLD_VPH).sum())
                                        total_hours = int(raw["total_volume"].count())
                                        hourly_risk_pct = (
                                                    hourly_over_thr / total_hours * 100) if total_hours > 0 else 0.0

                                        bucket_over_80_cap = int(
                                            (agg_all["total_volume"] > 0.80 * agg_all["cap"]).sum())
                                        bucket_risk_pct = (bucket_over_80_cap / len(agg_all) * 100) if len(
                                            agg_all) else 0.0

                                        peak_bucket_all = _prep_bucket(raw, granularity_vol)
                                        top_in_peak = (
                                            peak_bucket_all.loc[peak_bucket_all["local_datetime"] == peak_ts]
                                            .groupby("intersection_name", as_index=False)["total_volume"].sum()
                                            .sort_values("total_volume", ascending=False)
                                        )
                                        top3 = top_in_peak.head(3)
                                        top3_list = " • ".join(
                                            [f"{r['intersection_name']}: {int(r['total_volume']):,}" for _, r in
                                             top3.iterrows()]) if not top3.empty else "N/A"

                                        unit = AGG_META[granularity_vol]["unit"]
                                        label = AGG_META[granularity_vol]["label"]
                                        peak_when = _fmt_period(peak_ts, granularity_vol)

                                        if peak_util_pct >= 95 or hourly_risk_pct >= 20:
                                            rec = (
                                                "Immediate capacity relief (short-term: retime signals, dynamic splits & queue management; "
                                                "mid-term: turn-lane/approach improvements; evaluate access control at peak contributors).")
                                            rec_badge = "badge-critical"
                                        elif peak_util_pct >= 85 or hourly_risk_pct >= 10 or bucket_risk_pct >= 25:
                                            rec = (
                                                "Prioritize signal optimization (AM/PM plans + progression), adjust cycle lengths, and "
                                                "pilot demand management (driveway control, TSP). Plan spot upgrades at top 2–3 intersections.")
                                            rec_badge = "badge-poor"
                                        elif peak_util_pct >= 70 or hourly_risk_pct >= 5:
                                            rec = (
                                                "Retiming & coordination refresh, monitor weekly trends, and stage TSP/ITS enhancements.")
                                            rec_badge = "badge-fair"
                                        else:
                                            rec = ("Monitor; current capacity is adequate with routine timing review.")
                                            rec_badge = "badge-good"

                                        st.markdown(
                                            f"""
                                            <div class="insight-box">
                                                <h4>💡 Volume Analysis Insights</h4>
                                                <p><strong> Capacity:</strong> Peak <b>{peak_val:,.0f} {unit}</b> on <b>{peak_when}</b>
                                                <p><strong> Typical {label.capitalize()} Volume:</strong> Average <b>{avg_val:,.0f} {unit}</b> •
                                                   Peak/Avg ratio <b>{peak_to_avg:.1f}×</b>
                                                <p><strong> Total Vehicles:</strong> <b>{float(np.nansum(raw['total_volume'])):,.0f}</b>.</p>
                                                <p><strong> Overcapacity Hours:</strong> Hourly > {HIGH_VOLUME_THRESHOLD_VPH:,} vehicles for <b>{hourly_over_thr}</b> hours
                                                   (<b>{hourly_risk_pct:.1f}%</b> of hours) 
                                                   {label.capitalize()}s above 80% of scaled capacity: <b>{bucket_over_80_cap}</b>
                                                <p><strong> Peak Contributors:</strong> {top3_list}</p>
                                                <p><strong>🎯 Recommendation for CVAG:</strong> {rec}</p>
                                                <div style="margin-top:.4rem;">
                                                    <span class="performance-badge {rec_badge}">Action Priority</span>
                                                </div>
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )
                                    except Exception as e:
                                        st.error(f"❌ Error computing insights: {e}")

                                # ---------------- Risk table ----------------
                                step("Calculating intersection risk table", 97)
                                st.subheader("🚨 Intersection Volume & Capacity Risk Analysis")

                                # Add legend for Risk Analysis methodology
                                with st.expander("📊 **Risk Analysis Methodology & Legend**", expanded=False):
                                    st.markdown("""
                                                                    ### Risk Score Calculation
                                                                    The **Risk Score** (0-120+) is a weighted composite metric that identifies capacity bottlenecks:
                                                                    - **50%** weight: Peak Capacity Utilization (maximum hourly volume ÷ theoretical capacity)
                                                                    - **30%** weight: Average Capacity Utilization (average hourly volume ÷ theoretical capacity)  
                                                                    - **20%** weight: Peak-to-Average Ratio (volatility indicator × 10 for scaling)

                                                                    Higher scores indicate higher risk of capacity issues and congestion.

                                                                    ### Capacity Calculations
                                                                    - **Theoretical Capacity**: 1,800 vehicles per hour per approach
                                                                    - **Peak Capacity %**: (Peak hourly volume ÷ 1,800 vehicles per hour) × 100
                                                                    - **Avg Capacity %**: (Average hourly volume ÷ 1,800 vehicles per hour) × 100

                                                                    ### Risk Level Categories
                                                                    """)

                                    # Create a visual legend with colored badges for risk levels
                                    risk_col1, risk_col2, risk_col3, risk_col4, risk_col5 = st.columns(5)
                                    with risk_col1:
                                        st.markdown('<span class="performance-badge badge-excellent">🟢 Low Risk</span>',
                                                    unsafe_allow_html=True)
                                        st.caption("Score: 0-40")
                                        st.caption("Monitor regularly")
                                    with risk_col2:
                                        st.markdown('<span class="performance-badge badge-good">🟡 Moderate Risk</span>',
                                                    unsafe_allow_html=True)
                                        st.caption("Score: 40-60")
                                        st.caption("Signal optimization")
                                    with risk_col3:
                                        st.markdown('<span class="performance-badge badge-fair">🟠 High Risk</span>',
                                                    unsafe_allow_html=True)
                                        st.caption("Score: 60-80")
                                        st.caption("Capacity improvements")
                                    with risk_col4:
                                        st.markdown('<span class="performance-badge badge-poor">🔴 Critical Risk</span>',
                                                    unsafe_allow_html=True)
                                        st.caption("Score: 80-90")
                                        st.caption("Urgent intervention")
                                    with risk_col5:
                                        st.markdown(
                                            '<span class="performance-badge badge-critical">🚨 Severe Risk</span>',
                                            unsafe_allow_html=True)
                                        st.caption("Score: 90+")
                                        st.caption("Immediate action")

                                    st.markdown("""
                                                                    ### Action Priority Guidelines
                                                                    - **🟢 Monitor**: Peak capacity < 60% - Routine monitoring sufficient
                                                                    - **🟡 Optimize**: Peak capacity 60-75% - Signal timing adjustments recommended  
                                                                    - **🟠 Upgrade**: Peak capacity 75-90% - Infrastructure improvements needed
                                                                    - **🔴 Urgent**: Peak capacity > 90% - Immediate capacity relief required
                                                                    """)
                                try:
                                    g = raw.groupby(["intersection_name", "direction"]).agg(
                                        total_volume_mean=("total_volume", "mean"),
                                        total_volume_max=("total_volume", "max"),
                                        total_volume_std=("total_volume", "std"),
                                        total_volume_count=("total_volume", "count"),
                                    ).reset_index()

                                    g["Peak_Capacity_Util"] = (
                                            g["total_volume_max"] / THEORETICAL_LINK_CAPACITY_VPH * 100
                                    ).round(1)
                                    g["Avg_Capacity_Util"] = (
                                            g["total_volume_mean"] / THEORETICAL_LINK_CAPACITY_VPH * 100
                                    ).round(1)
                                    g["Volume_Variability"] = (
                                            g["total_volume_std"] / g["total_volume_mean"] * 100
                                    ).replace([np.inf, -np.inf], np.nan).fillna(0).round(1)
                                    g["Peak_Avg_Ratio"] = (
                                            g["total_volume_max"] / g["total_volume_mean"]
                                    ).replace([np.inf, -np.inf], 0).fillna(0).round(1)

                                    g["🚨 Risk Score"] = (
                                            0.5 * g["Peak_Capacity_Util"]
                                            + 0.3 * g["Avg_Capacity_Util"]
                                            + 0.2 * (g["Peak_Avg_Ratio"] * 10)
                                    ).round(1)

                                    g["⚠️ Risk Level"] = pd.cut(
                                        g["🚨 Risk Score"],
                                        bins=[0, 40, 60, 80, 90, 999],
                                        labels=["🟢 Low Risk", "🟡 Moderate Risk", "🟠 High Risk", "🔴 Critical Risk",
                                                "🚨 Severe Risk"],
                                        include_lowest=True,
                                    )
                                    g["🎯 Action Priority"] = pd.cut(
                                        g["Peak_Capacity_Util"],
                                        bins=[0, 60, 75, 90, 999],
                                        labels=["🟢 Monitor", "🟡 Optimize", "🟠 Upgrade", "🔴 Urgent"],
                                        include_lowest=True,
                                    )

                                    final = g[
                                        [
                                            "intersection_name",
                                            "direction",
                                            "⚠️ Risk Level",
                                            "🎯 Action Priority",
                                            "🚨 Risk Score",
                                            "Peak_Capacity_Util",
                                            "Avg_Capacity_Util",
                                            "total_volume_mean",
                                            "total_volume_max",
                                            "Peak_Avg_Ratio",
                                            "total_volume_count",
                                        ]
                                    ].rename(
                                        columns={
                                            "intersection_name": "Intersection",
                                            "direction": "Dir",
                                            "Peak_Capacity_Util": "📊 Peak Capacity %",
                                            "Avg_Capacity_Util": "📊 Avg Capacity %",
                                            "total_volume_mean": "Volume (vehicles)",
                                            "total_volume_max": "Peak Volume (vehicles)",
                                            "total_volume_count": "Data Points",
                                        }
                                    ).sort_values("🚨 Risk Score", ascending=False)

                                    st.dataframe(
                                        final.head(15),
                                        use_container_width=True,
                                        column_config={
                                            "🚨 Risk Score": st.column_config.NumberColumn(
                                                "🚨 Capacity Risk Score",
                                                help="Composite of peak/avg util + peaking",
                                                format="%.1f",
                                                min_value=0,
                                                max_value=120,
                                            ),
                                            "📊 Peak Capacity %": st.column_config.NumberColumn("📊 Peak Capacity %",
                                                                                               format="%.1f%%"),
                                            "📊 Avg Capacity %": st.column_config.NumberColumn("📊 Avg Capacity %",
                                                                                              format="%.1f%%"),
                                        },
                                    )

                                    st.download_button(
                                        "⬇️ Download Capacity Risk Table (CSV)",
                                        data=final.to_csv(index=False).encode("utf-8"),
                                        file_name="capacity_risk.csv",
                                        mime="text/csv",
                                    )
                                    st.download_button(
                                        "⬇️ Download Filtered Volume (CSV)",
                                        data=filtered_volume_data.to_csv(index=False).encode("utf-8"),
                                        file_name="volume_filtered.csv",
                                        mime="text/csv",
                                    )
                                except Exception as e:
                                    st.error(f"❌ Error in volume analysis: {e}")
                                    simple = raw.groupby(["intersection_name", "direction"]).agg(
                                        Avg=("total_volume", "mean"), Peak=("total_volume", "max")
                                    ).reset_index().sort_values("Peak", ascending=False)
                                    st.dataframe(simple, use_container_width=True)

                                # Cycle Length Recommendations section
                                step("Rendering cycle length recommendations", 99)
                                render_cycle_length_section(_raw_all_dirs)

        except Exception as e:
            st.error(f"❌ Error processing traffic Prediction: {e}")
            st.info("Please check your Prediction sources and try again.")

# -------------------------
# TAB 3: Acyclica (Travel Time + AI Prediction Models)
# -------------------------
with tab3:
    # Use the enhanced Tab 3 renderer with proper sidebar controls
    render_tab3_analysis()
# -------------------------
# TAB 4: Iteris VantageLive (Bikes + Vehicles)
# -------------------------
with tab4:
    render_vantage_tab()

# -------------------------
# TAB 5: Bosch Multimodal Traffic Analysis
# -------------------------
with tab5:
    render_bosch_tab()

# -- Auth footer at the very bottom of the sidebar --
try:
    render_auth_sidebar_footer()
except Exception:
    pass

# =========================
# FOOTER
# =========================
FOOTER = """
<style>
  .footer-title { color:#2980b9; margin:0 0 .4rem; font-weight:700; }
  .social-btn {
    width: 40px; height: 40px; display:grid; place-items:center; border-radius:50%;
    background:#ffffff; border:1px solid rgba(41,128,185,.25);
    box-shadow:0 2px 8px rgba(0,0,0,.08); text-decoration:none;
    transition: transform .15s ease, box-shadow .15s ease;
  }
  .social-btn:hover { transform: translateY(-1px); box-shadow:0 4px 14px rgba(0,0,0,.12); }
  .website-pill {
    height:40px; display:inline-flex; align-items:center; gap:8px; padding:0 12px;
    border-radius:9999px; background:#ffffff; border:1px solid #2980b9; color:#2980b9;
    font-weight:700; text-decoration:none; box-shadow:0 2px 8px rgba(0,0,0,.08);
    transition: transform .15s ease, box-shadow .15s ease;
  }
  .website-pill:hover { transform: translateY(-1px); box-shadow:0 4px 14px rgba(0,0,0,.12); }
</style>

<div class="footer-card" style="text-align:center; padding: 1.25rem;
    background: linear-gradient(135deg, rgba(79,172,254,0.1), rgba(0,242,254,0.05));
    border-radius: 15px; margin-top: 1rem; border: 1px solid rgba(79,172,254,0.2);
    font-family: system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Liberation Sans', sans-serif;">

  <h4 class="footer-title">🛣️ Active Transportation & Operations Management Dashboard</h4>

  <p class="footer-sub" style="margin:.1rem 0 0; font-size:1.0rem; color:#0f2f52;">
    Powered by Advanced Machine Learning • Real-time Traffic Intelligence • Intelligent Transportation Solutions (ITS)
  </p>

  <div style="display:flex; justify-content:center; align-items:center; gap:14px; margin:12px 0 8px;">
    <a class="social-btn" href="https://www.instagram.com/advantec98/" target="_blank" rel="noopener noreferrer" aria-label="Instagram">
      <span style="font:700 13px/1 system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial; color:#444;">IG</span>
    </a>
    <a class="social-btn" href="https://www.linkedin.com/company/advantec-consulting-engineers-inc./posts/?feedView=all"
       target="_blank" rel="noopener noreferrer" aria-label="LinkedIn">
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 448 512" aria-hidden="true"><path fill="#0A66C2" d="M100.28 448H7.4V148.9h92.88zM53.79 108.1C24.09 108.1 0 83.5 0 53.8 0 24.1 24.1 0 53.79 0s53.8 24.1 53.8 53.8c0 29.7-24.1 54.3-53.8 54.3zM447.9 448h-92.68V302.4c0-34.7-.7-79.3-48.3-79.3-48.3 0-55.7 37.7-55.7 76.6V448h-92.7V148.9h89V185h1.3c12.4-23.6 42.7-48.3 87.8-48.3 93.9 0 111.2 61.8 111.2 142.3V448z"/></svg>
    </a>
    <a class="social-btn" href="https://www.facebook.com/advantecconsultingUSA" target="_blank" rel="noopener noreferrer" aria-label="Facebook">
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 320 512" aria-hidden="true"><path fill="#1877F2" d="M279.14 288l14.22-92.66h-88.91v-60.13c0-25.35 12.42-50.06 52.24-50.06h40.42V6.26S263.61 0 225.36 0c-73.22 0-121 44.38-121 124.72v70.62H22.89V288h81.47v224h100.2V288z"/></svg>
    </a>
    <a class="website-pill" href="https://advantec-usa.com/" target="_blank" rel="noopener noreferrer" aria-label="ADVANTEC Website">
      <span style="font-size:18px; line-height:1;">🌐</span>
      <span>Website</span>
    </a>
  </div>

  <p class="footer-copy" style="margin:.2rem 0 0; font-size:.9rem; color:#0f2f52;">
    © 2025 ADVANTEC Consulting Engineers, Inc. — "Because We Care"
  </p>
</div>

<script>
(function() {
  function updateFooterColors() {
    const body = document.body;
    const computed = getComputedStyle(body);
    const bgColor = computed.backgroundColor || getComputedStyle(document.documentElement).getPropertyValue('--background-color') || '#ffffff';

    let r=255,g=255,b=255;
    if (bgColor.startsWith('rgb')) {
      const m = bgColor.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
      if (m) { r = parseInt(m[1]); g = parseInt(m[2]); b = parseInt(m[3]); }
    }
    const luminance = (0.299*r + 0.587*g + 0.114*b) / 255;
    const isDark = luminance < 0.5;

    const subtitle = document.querySelector('.footer-sub');
    const copyright = document.querySelector('.footer-copy');
    const title = document.querySelector('.footer-title');

    if (subtitle && copyright) {
      if (isDark) {
        subtitle.style.color = '#ffffff';
        copyright.style.color = '#ffffff';
        if (title) title.style.color = '#7ec3ff';
      } else {
        subtitle.style.color = '#0f2f52';
        copyright.style.color = '#0f2f52';
        if (title) title.style.color = '#2980b9';
      }
    }
  }
  updateFooterColors();
  const observer = new MutationObserver(updateFooterColors);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['Prediction-theme', 'class'] });
  observer.observe(document.body, { attributes: true, attributeFilter: ['Prediction-theme', 'class', 'style'] });
  setInterval(updateFooterColors, 1000);
})();
</script>
"""
st.markdown(FOOTER, unsafe_allow_html=True)