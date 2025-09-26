import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import streamlit.components.v1 as components
import contextlib  # <-- added

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
    get_acyclica_df,  # NEW for Tab 3
    get_performance_rating,
    performance_chart,
    # volume_charts,  # <- no longer used in TAB 2
    date_range_preset_controls,
    compute_perf_kpis_interpretable,
    render_badge,
)

# Cycle length section (moved out)
from cycle_length_recommendations import render_cycle_length_section

# Map builders (return Plotly figures)
from Map import build_corridor_map, build_intersection_map, build_intersections_overview

# Acyclica analysis (Tab 3)
from acyclica_traveltime_analysis import render_acyclica_section

# AI Prediction Models (Tab 3 sub-sections)
from Prediction.peak_hour_prediction import render_peak_hour_section
from Prediction.incident_detection import render_incident_detection_section  
from Prediction.event_impact_analysis import render_event_impact_section


# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="Active Transportation & Operations Management Dashboard",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "toggleSpikelines"]
}
MAP_HEIGHT = 900  # default map height (px) for the right rail

# =========================
# Constants / Config
# =========================
THEORETICAL_LINK_CAPACITY_VPH = 1800
HIGH_VOLUME_THRESHOLD_VPH = 1200
CRITICAL_DELAY_SEC = 120
HIGH_DELAY_SEC = 60

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
    nb_mask = ser.str.contains(r"\b(nb|north|northbound)\b", regex=True)
    sb_mask = ser.str.contains(r"\b(sb|south|southbound)\b", regex=True)
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
    ss.setdefault("t1_ready", False)       # Tab 1: results committed?
    ss.setdefault("t2_ready", False)       # Tab 2: results committed?
    ss.setdefault("t1_params", {})         # Tab 1: committed parameters
    ss.setdefault("t2_params", {})         # Tab 2: committed parameters
    ss.setdefault("t1_current", {})        # Tab 1: current (uncommitted) values
    ss.setdefault("t2_current", {})        # Tab 2: current (uncommitted) values

def _freeze_params(d: dict) -> dict:
    """Normalize params for lightweight equality checks (esp. date_range)."""
    if not isinstance(d, dict):
        return {}
    out = dict(d)
    if "date_range" in out and isinstance(out["date_range"], (list, tuple)) and len(out["date_range"]) == 2:
        out["date_range"] = (str(out["date_range"][0]), str(out["date_range"][1]))
    if "date_range_vol" in out and isinstance(out["date_range_vol"], (list, tuple)) and len(out["date_range_vol"]) == 2:
        out["date_range_vol"] = (str(out["date_range_vol"][0]), str(out["date_range_vol"][1]))
    return out

_init_state()

# =========================
# Title / Intro
# =========================
st.markdown("""
<div class="main-container">
    <h1 style="text-align:center; margin:0; font-size:2.5rem; font-weight:800;">
        🛣️ Active Transportation & Operations Management Dashboard
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
    <div style="text-align:center; margin-bottom: 0.5rem;">
        <strong style="font-size: 1.2rem; color: #2980b9;">🌎 The ADVANTEC Web Service Platform</strong>
    </div>
    <p>Leverages <strong>millions of Prediction points</strong> trained on advanced Machine Learning algorithms to optimize traffic flow, reduce travel time, minimize fuel consumption, and decrease greenhouse gas emissions across the transportation network.</p>
    <p><strong>Key Capabilities:</strong> Real-time anomaly detection • Intelligent cycle length optimization • Predictive traffic modeling • Performance analytics</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, #3498db, #2980b9); color: white; padding: 1.1rem; border-radius: 15px;
    margin: 1rem 0; text-align: center; box-shadow: 0 6px 20px rgba(52, 152, 219, 0.25);">
    <h3 style="margin:0; font-weight:600;">🔍 Research Questions</h3>
    <p style="margin: 0.45rem 0 0; font-size: 1.0rem;">What are the main bottlenecks on Washington Street that most increase travel times?</p>
    <p style="margin: 0.45rem 0 0; font-size: 1.0rem;">Which direction on Washington Street causes the most congestion?</p>
</div>
""", unsafe_allow_html=True)

# =========================
# --------- NEW TAB 2 HELPERS (aggregation-aware) ----------
# =========================

AGG_META = {
    "Hourly":  {"unit": "vph", "bucket": "H", "label": "hour",  "fixed_hours": 1},
    "Daily":   {"unit": "vpd", "bucket": "D", "label": "day",   "fixed_hours": 24},
    "Weekly":  {"unit": "vpw", "bucket": "W", "label": "week",  "fixed_hours": 24*7},
    "Monthly": {"unit": "vpm", "bucket": "M", "label": "month", "fixed_hours": None},  # varies by month
}

def _prep_bucket(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """
    Aggregate hourly records to the selected bucket (sum of hourly volumes).
    Returns: df with columns [local_datetime, intersection_name, total_volume, bucket_hours].
    """
    if df.empty:
        return df.copy()

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
        d.groupby(["bucket", "intersection_name"], as_index=False)
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
    """Given unique x (local_datetime) and bucket_hours, produce y series for capacity/threshold."""
    xs = x_df[["local_datetime", "bucket_hours"]].drop_duplicates().sort_values("local_datetime")
    xs["capacity"] = xs["bucket_hours"] * float(cap_vph)
    xs["high"] = xs["bucket_hours"] * float(high_vph)
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
    top_k: int = 8
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
    agg = _prep_bucket(raw_hourly_df, granularity)
    if agg.empty:
        return None, None, None

    # Limit to top intersections by mean demand
    order = agg.groupby("intersection_name")["total_volume"].mean().sort_values(ascending=False)
    keep = order.index[:max(1, min(top_k, len(order)))]

    plot_df = agg[agg["intersection_name"].isin(keep)].copy().sort_values("local_datetime")
    unit = AGG_META[granularity]["unit"]
    label = AGG_META[granularity]["label"]

    # ---------- Trend ----------
    fig_trend = go.Figure()
    mode = "lines" if granularity == "Hourly" else "lines+markers"
    xfmt = "%Y-%m-%d %H:%M" if granularity == "Hourly" else "%Y-%m-%d"

    for name, g in plot_df.groupby("intersection_name"):
        fig_trend.add_trace(
            go.Scatter(
                x=g["local_datetime"],
                y=g["total_volume"],
                mode=mode,
                name=name,
                hovertemplate=(f"<b>%{{fullData.name}}</b><br>%{{x|{xfmt}}}<br>Volume: %{{y:,.0f}} {unit}<extra></extra>"),
            )
        )

    xs = _cap_series_for_x(plot_df, cap_vph, high_vph)
    fig_trend.add_trace(
        go.Scatter(
            x=xs["local_datetime"], y=xs["capacity"],
            name=f"Theoretical Capacity ({unit})", mode="lines",
            line=dict(dash="dash"),
            hovertemplate=(f"%{{x|{xfmt}}}<br>Capacity: %{{y:,.0f}} {unit}<extra></extra>"),
        )
    )
    fig_trend.add_trace(
        go.Scatter(
            x=xs["local_datetime"], y=xs["high"],
            name=f"High Volume Threshold ({unit})", mode="lines",
            line=dict(dash="dot"),
            hovertemplate=(f"%{{x|{xfmt}}}<br>Threshold: %{{y:,.0f}} {unit}<extra></extra>"),
        )
    )
    fig_trend.update_layout(
        xaxis_title="Date/Time",
        yaxis_title=f"Volume ({unit})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=40, b=10),
    )

    # ---------- Box ----------
    cat_order = order[order.index.isin(keep)].index.tolist()
    fig_box = px.box(
        plot_df, x="intersection_name", y="total_volume",
        category_orders={"intersection_name": cat_order},
        points=False, title=f"Volume Distribution by Intersection — {granularity}"
    )
    fig_box.update_layout(
        xaxis_title="Intersection",
        yaxis_title=f"Volume per {label} ({unit})",
        margin=dict(l=10, r=10, t=40, b=10)
    )

    # ---------- Matrix ----------
    mat = (
        plot_df.groupby("intersection_name", as_index=False)["total_volume"]
               .mean()
               .rename(columns={"total_volume": f"Avg {label} Volume"})
    )
    mat["Rank"] = mat[f"Avg {label} Volume"].rank(ascending=False, method="dense").astype(int)
    mat = mat.sort_values("Rank")
    fig_matrix = px.bar(
        mat, y="intersection_name", x=f"Avg {label} Volume",
        orientation="h", text=f"Avg {label} Volume",
        title=f"Average {label.capitalize()} Vehicle Volume by Intersection"
    )
    fig_matrix.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
    fig_matrix.update_layout(
        xaxis_title=f"Average {label} volume ({unit})",
        yaxis_title="",
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig_trend, fig_box, fig_matrix

# =========================
# Tabs
# =========================
st.markdown("## Select Page")
tab1, tab2, tab3 = st.tabs(["Pg.1 ITERIS CLEARGUIDE", "Pg.2 KINETIC MOBILITY", "Pg.3 ACYCLICA"])

# -------------------------
# TAB 1: Performance / Travel Time (Search-gated, NO forms)
# -------------------------
with tab1:

    # Load Prediction once to populate controls (safe to load; results stay blank until Search)
    corridor_df = get_corridor_df()

    # -------- Sidebar controls (commit on Search) --------
    with st.sidebar:
        st.image("Logos/ACE-logo-HiRes.jpg", width=210)
        st.image("Logos/CV Sync__.jpg", width=205)

        with st.expander("⚙️ Pg.1 SETTINGS", expanded=True):
            st.caption("Select Route and Date Range")
            st.caption("Data: Vehicle Speed, Delay, and Travel Time")
            st.markdown("## 🗺️ Select Route")

            od_mode = st.checkbox(
                "Analysis Pro",
                value=True,
                help="Allows you the power to select Origin and Destination. ",
                key="od_mode_perf",
            )

            origin, destination = None, None
            if od_mode and not corridor_df.empty:
                nodes_in_data = _canonical_order_in_data(corridor_df)
                node_list = nodes_in_data if len(nodes_in_data) >= 2 else _build_node_order(corridor_df)

                if len(node_list) >= 2:
                    cA, cB = st.columns(2)
                    with cA:
                        origin = st.selectbox("Origin", node_list, index=0, key="od_origin")
                    with cB:
                        destination = st.selectbox("Destination", node_list, index=len(node_list) - 1, key="od_destination")
                else:
                    st.info("Not enough nodes found to build O-D options.")

            # Analysis Period
            if corridor_df.empty or "local_datetime" not in corridor_df.columns:
                min_date = datetime.today().date() - timedelta(days=7)
                max_date = datetime.today().date()
            else:
                min_date = corridor_df["local_datetime"].dt.date.min()
                max_date = corridor_df["local_datetime"].dt.date.max()

            st.markdown("## 📅 Date And Time")
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

            if st.button("🔍 **Search**", key="search_tab1", type="primary", use_container_width=True):
                st.session_state["t1_params"] = t1_current
                st.session_state["t1_ready"] = True

    # -------- Main content area (render only when "Search" committed) --------
    t1_ready = st.session_state.get("t1_ready", False)
    t1_params = st.session_state.get("t1_params", {})
    t1_pending = t1_ready and _freeze_params(t1_params) != _freeze_params(st.session_state.get("t1_current", {}))

    if not t1_ready:
        st.info("Choose your Route and Date Range in the settings to the left.")
    else:
        if t1_pending:
            st.warning("⚙️ Press **Search** to refresh.")

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

                        if path_segments:
                            seg_df = base_df[base_df["segment_name"].isin(path_segments)].copy()
                            if "direction" in seg_df.columns and desired_dir is not None:
                                dnorm = normalize_dir(seg_df["direction"])
                                seg_df = seg_df.loc[dnorm == desired_dir].copy()

                            working_df = seg_df.copy()
                            route_label = f"{origin} → {destination}"
                        else:
                            st.info("No matching segments found for the selected O-D on the canonical path.")

                # ---------- Layout: wide content + sticky right rail ----------
                main_col_t1, right_col_t1 = st.columns([7, 3.5], gap="large")

                # Right rail (map code)
                with right_col_t1:
                    st.markdown('<div id="od-map-anchor"></div>', unsafe_allow_html=True)
                    st.markdown("##### Corridor Map", help="Stays visible while you scroll the analysis on the left.")
                    fig_od = None
                    if od_mode and origin and destination and origin != destination:
                        try:
                            fig_od = build_corridor_map(origin, destination)
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
                        with cad_loader("Fetching Data...") as step:
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

                                step("Preparing summary context", 35)
                                st.markdown(
                                    f"""
                                <div style="
                                    background: linear-gradient(135deg, #2b77e5 0%, #19c3e6 100%);
                                    border-radius:16px; padding:18px 20px; color:#fff; margin:8px 0 14px;
                                    box-shadow:0 10px 26px rgba(25,115,210,.25); text-align:left;
                                    font-family: inherit;">
                                  <div style="display:flex; align-items:center; gap:10px;">
                                    <div style="width:36px;height:36px;border-radius:10px;background:rgba(255,255,255,.18);
                                                display:flex;align-items:center;justify-content:center;
                                                box-shadow:inset 0 0 0 1px rgba(255,255,255,.15);">📊</div>
                                    <div style="font-size:1.9rem;font-weight:800;letter-spacing:.2px;">
                                      Travel Time Analysis: {route_label}
                                    </div>
                                  </div>
                                  <div style="margin-top:10px;display:flex;flex-direction:column;gap:6px;">
                                    <div>📅 {date_range[0].strftime('%b %d, %Y')} to {date_range[1].strftime('%b %d, %Y')} ({data_span} days) • {granularity} Aggregation{time_context}</div>
                                    <div>✅ Analyzing {total_records:,} Prediction points across the selected period</div>
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

                                    for c in ["average_traveltime", "average_delay"]:
                                        if c in od_hourly.columns:
                                            od_hourly[c] = pd.to_numeric(od_hourly[c], errors="coerce")

                                    if "segment_name" in od_hourly.columns and "local_datetime" in od_hourly.columns:
                                        od_hourly = (
                                            od_hourly.groupby(["local_datetime", "segment_name"], as_index=False)
                                            .agg({"average_traveltime": "mean", "average_delay": "mean"})
                                        )

                                    od_series = (
                                        od_hourly.groupby("local_datetime", as_index=False)
                                        .agg({"average_traveltime": "sum", "average_delay": "sum"})
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
                                    st.subheader("🚦 KPI's (Key Performance Indicators)")
                                    k = compute_perf_kpis_interpretable(raw_data, HIGH_DELAY_SEC)

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
                                        st.caption(k['congestion_freq'].get('extra', ''))
                                        st.markdown(render_badge(k['congestion_freq']['score']), unsafe_allow_html=True)
                                    with c3:
                                        st.metric(
                                            "⏱️ Average Travel Time",
                                            f"{k['avg_tt']['value']:.1f} {k['avg_tt']['unit']}",
                                            help=k['avg_tt']['help'],
                                        )
                                        st.markdown(render_badge(k['avg_tt']['score']), unsafe_allow_html=True)
                                    with c4:
                                        st.metric(
                                            "📈 Planning Time (95th Percentile)",
                                            f"{k['planning_time']['value']:.1f} {k['planning_time']['unit']}",
                                            help=k['planning_time']['help'],
                                        )
                                        st.markdown(render_badge(k['planning_time']['score']), unsafe_allow_html=True)
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

                                        if 'od_series' in locals() and not od_series.empty and granularity in ("Daily", "Weekly", "Monthly"):
                                            tmp = od_series.copy()
                                            tmp["local_datetime"] = pd.to_datetime(tmp["local_datetime"])
                                            if granularity == "Daily":
                                                tmp["date_group"] = tmp["local_datetime"].dt.date
                                                trends_df = (
                                                    tmp.groupby("date_group", as_index=False)
                                                    .agg({"average_traveltime": "mean", "average_delay": "mean"})
                                                    .rename(columns={"date_group": "local_datetime"})
                                                )
                                                trends_df["local_datetime"] = pd.to_datetime(trends_df["local_datetime"])
                                            elif granularity == "Weekly":
                                                tmp["week_group"] = tmp["local_datetime"].dt.to_period("W").dt.start_time
                                                trends_df = (
                                                    tmp.groupby("week_group", as_index=False)
                                                    .agg({"average_traveltime": "mean", "average_delay": "mean"})
                                                    .rename(columns={"week_group": "local_datetime"})
                                                )
                                            elif granularity == "Monthly":
                                                tmp["month_group"] = tmp["local_datetime"].dt.to_period("M").dt.start_time
                                                trends_df = (
                                                    tmp.groupby("month_group", as_index=False)
                                                    .agg({"average_traveltime": "mean", "average_delay": "mean"})
                                                    .rename(columns={"month_group": "local_datetime"})
                                                )

                                        with v1:
                                            dc = performance_chart(trends_df, "delay")
                                            if dc:
                                                st.plotly_chart(dc, use_container_width=True, config=PLOTLY_CONFIG)
                                        with v2:
                                            tc = performance_chart(trends_df, "travel")
                                            if tc:
                                                st.plotly_chart(tc, use_container_width=True, config=PLOTLY_CONFIG)

                                        if 'od_series' in locals() and not od_series.empty:
                                            st.subheader("🔍Which Dates/Times have the highest Travel Time and Delay?")
                                            st.dataframe(
                                                od_series.rename(
                                                    columns={
                                                        "local_datetime": "Timestamp",
                                                        "average_traveltime": "O-D Travel Time (min)",
                                                        "average_delay": "O-D Delay (min)",
                                                    }
                                                ),
                                                use_container_width=True,
                                            )

                                    # =========================
                                    # 🚨 Comprehensive Bottleneck Analysis
                                    # =========================
                                    step("Running bottleneck analysis", 95)
                                    st.subheader("🚨 Comprehensive Bottleneck Analysis")
                                    if 'raw_data' in locals() and not raw_data.empty and "segment_name" in working_df.columns:
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
                                                analysis_df = analysis_df.loc[analysis_df["dir_norm"] == desired_dir].copy()
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
                                                lambda r: f"{r['segment_name']} ({arrow_map.get(r['dir_norm'], '• UNK')})", axis=1
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
                                            g["🎯 Performance Rating"] = pd.cut(g["Bottleneck_Score"], bins=bins, labels=labels)

                                            final = g[
                                                [
                                                    "Segment (by Dir)",
                                                    "dir_norm",
                                                    "🎯 Performance Rating",
                                                    "Bottleneck_Score",
                                                    "average_delay_mean",
                                                    "average_delay_max",
                                                    "average_traveltime_mean",
                                                    "average_traveltime_max",
                                                    "average_speed_mean",
                                                    "average_speed_min",
                                                    "n",
                                                ]
                                            ].rename(
                                                columns={
                                                    "dir_norm": "Dir",
                                                    "average_delay_mean": "Avg Delay (min)",
                                                    "average_delay_max": "Peak Delay (min)",
                                                    "average_traveltime_mean": "Avg Time (min)",
                                                    "average_traveltime_max": "Peak Time (min)",
                                                    "average_speed_mean": "Avg Speed (mph)",
                                                    "average_speed_min": "Min Speed (mph)",
                                                    "n": "Obs",
                                                }
                                            ).sort_values("Bottleneck_Score", ascending=False)

                                            st.dataframe(
                                                final.head(15),
                                                use_container_width=True,
                                                column_config={
                                                    "Bottleneck_Score": st.column_config.NumberColumn(
                                                        "🚨 Impact Score",
                                                        help="Composite (0–100); higher ⇒ worse",
                                                        format="%.1f",
                                                    ),
                                                    "Dir": st.column_config.TextColumn("Dir"),
                                                },
                                            )

                                            st.download_button(
                                                "⬇️ Download Bottleneck Table (CSV)",
                                                data=final.to_csv(index=False).encode("utf-8"),
                                                file_name="bottlenecks.csv",
                                                mime="text/csv",
                                            )
                                            st.download_button(
                                                "⬇️ Download Filtered Performance (CSV)",
                                                data=filtered_data.to_csv(index=False).encode("utf-8"),
                                                file_name="performance_filtered.csv",
                                                mime="text/csv",
                                            )
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
        with st.expander("⚙️ Pg.2 SETTINGS", expanded=True):

            st.caption("Select Intersection(s) and Date Range")
            st.caption("Data: Vehicle Volume")
            intersections = ["All Intersections"] + sorted(
                volume_df["intersection_name"].dropna().unique().tolist()
            ) if not volume_df.empty and "intersection_name" in volume_df.columns else ["All Intersections"]

            st.markdown("## 🚦 Select Intersection")
            intersection = st.selectbox(
                "🚦 Select Intersection",
                intersections,
                key="intersection_vol",
                label_visibility="collapsed",
            )

            if volume_df.empty or "local_datetime" not in volume_df.columns:
                min_date = datetime.today().date() - timedelta(days=7)
                max_date = datetime.today().date()
            else:
                min_date = volume_df["local_datetime"].dt.date.min()
                max_date = volume_df["local_datetime"].dt.date.max()

            st.markdown("## 📅 Date And Time")
            date_range_vol = date_range_preset_controls(min_date, max_date, key_prefix="vol")

            st.markdown("## Granularity")
            granularity_vol = st.selectbox(
                "Data Aggregation",
                ["Hourly", "Daily", "Weekly", "Monthly"],
                index=0,
                key="granularity_vol",
            )

            if not volume_df.empty and "direction" in volume_df.columns:
                direction_options = ["All Directions"] + sorted(volume_df["direction"].dropna().unique().tolist())
            else:
                direction_options = ["All Directions"]
            direction_filter = st.selectbox("🔄 Direction Filter", direction_options, key="direction_filter_vol")

            # track uncommitted controls
            t2_current = {
                "intersection": intersection,
                "date_range_vol": tuple(date_range_vol) if date_range_vol else None,
                "granularity_vol": granularity_vol,
                "direction_filter": direction_filter,
            }
            st.session_state["t2_current"] = t2_current


            if st.button("🔍 **Search**", key="search_tab2", type="primary", use_container_width=True):
                st.session_state["t2_params"] = t2_current
                st.session_state["t2_ready"] = True


    # -------- Main content area (render only when "Search" committed) --------
    t2_ready = st.session_state.get("t2_ready", False)
    t2_params = st.session_state.get("t2_params", {})
    t2_pending = t2_ready and _freeze_params(t2_params) != _freeze_params(st.session_state.get("t2_current", {}))

    if not t2_ready:
        st.info("Choose your route and date range in the settings to the left.")
    else:
        if t2_pending:
            st.warning("⚙️ Press **Search** to refresh.")

        try:
            base_df = volume_df.copy() if not volume_df.empty else pd.DataFrame()
            if base_df.empty:
                st.error("❌ Failed to load volume Prediction. Please check your Prediction sources.")
            else:
                # Unpack committed params
                intersection = t2_params.get("intersection", "All Intersections")
                date_range_vol = t2_params.get("date_range_vol")
                granularity_vol = t2_params.get("granularity_vol", "Hourly")
                direction_filter = t2_params.get("direction_filter", "All Directions")

                if intersection != "All Intersections":
                    base_df = base_df[base_df["intersection_name"] == intersection]
                if direction_filter != "All Directions" and "direction" in base_df.columns:
                    base_df = base_df[base_df["direction"] == direction_filter]

                # Two-column layout with sticky right rail
                content_col, right_col = st.columns([7, 3.5], gap="large")

                # Right rail (sticky overview map)
                with right_col:
                    st.markdown('<div id="vol-map-anchor"></div>', unsafe_allow_html=True)
                    st.markdown("##### Corridor Map", help="Stays visible while you scroll the analysis on the left.")

                    try:
                        fig_over = build_intersections_overview(
                            selected_label=None if intersection == "All Intersections" else intersection
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
                            st.caption(f"Selected: **{intersection}**")
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
                        with cad_loader("Fetching Data...") as step:
                            step("Applying filters & aggregations", 20)
                            filtered_volume_data = process_traffic_data(base_df, date_range_vol, granularity_vol)

                            if filtered_volume_data.empty:
                                step("No volume Prediction in selected range", 100)
                                st.warning("⚠️ No volume Prediction available for the selected range.")
                            else:
                                span = (date_range_vol[1] - date_range_vol[0]).days + 1
                                total_obs = len(filtered_volume_data)

                                step("Preparing summary context", 35)
                                st.markdown(
                                    f"""
                                <div style="
                                    background: linear-gradient(135deg, #2b77e5 0%, #19c3e6 100%);
                                    border-radius:16px; padding:18px 20px; color:#fff; margin:8px 0 14px;
                                    box-shadow:0 10px 26px rgba(25,115,210,.25); text-align:left;
                                    font-family: system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
                                  <div style="display:flex; align-items:center; gap:10px;">
                                    <div style="width:36px;height:36px;border-radius:10px;background:rgba(255,255,255,.18);
                                                display:flex;align-items:center;justify-content:center;
                                                box-shadow:inset 0 0 0 1px rgba(255,255,255,.15);">📊</div>
                                    <div style="font-size:1.9rem;font-weight:800; letter-spacing:.2px;">
                                      Vehicle Volume Analysis: {intersection}
                                    </div>
                                  </div>
                                  <div style="margin-top:10px;display:flex;flex-direction:column;gap:6px;">
                                    <div>📅 {date_range_vol[0].strftime('%b %d, %Y')} to {date_range_vol[1].strftime('%b %d, %Y')} ({span} days) • {granularity_vol} Aggregation</div>
                                    <div>✅ {total_obs:,} observations • Direction: {direction_filter}</div>
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

                                st.subheader("🚦 Traffic Demand Performance Indicators")
                                if raw.empty or raw["total_volume"].dropna().empty:
                                    st.info("No raw hourly volume in this window.")
                                else:
                                    bucket_all = _prep_bucket(raw, granularity_vol).groupby("local_datetime", as_index=False)["total_volume"].sum().sort_values("local_datetime")
                                    if granularity_vol == "Monthly":
                                        bucket_all["bucket_hours"] = pd.to_datetime(bucket_all["local_datetime"]).dt.days_in_month * 24
                                    else:
                                        bucket_all["bucket_hours"] = AGG_META[granularity_vol]["fixed_hours"]

                                    bucket_all["cap"] = bucket_all["bucket_hours"] * THEORETICAL_LINK_CAPACITY_VPH
                                    util_series = np.where(bucket_all["cap"] > 0, bucket_all["total_volume"] / bucket_all["cap"] * 100, np.nan)

                                    peak_idx = int(bucket_all["total_volume"].idxmax())
                                    peak_val = float(bucket_all.loc[peak_idx, "total_volume"])
                                    peak_cap = float(bucket_all.loc[peak_idx, "cap"])
                                    peak_util_pct = (peak_val / peak_cap * 100) if peak_cap > 0 else 0.0

                                    p95_val = float(np.nanpercentile(bucket_all["total_volume"], 95)) if bucket_all["total_volume"].notna().any() else 0.0
                                    avg_bucket_val = float(bucket_all["total_volume"].mean())
                                    avg_util_pct = float(np.nanmean(util_series)) if np.isfinite(util_series).any() else 0.0

                                    hourly_avg = float(np.nanmean(raw["total_volume"])) if raw["total_volume"].notna().any() else 0.0
                                    cv_hourly = (float(np.nanstd(raw["total_volume"])) / hourly_avg * 100) if hourly_avg > 0 else 0.0
                                    cv_bucket = (float(np.nanstd(bucket_all["total_volume"])) / avg_bucket_val * 100) if avg_bucket_val > 0 else 0.0

                                    high_hours = int((raw["total_volume"] > HIGH_VOLUME_THRESHOLD_VPH).sum())
                                    total_hours = int(raw["total_volume"].count())
                                    risk_pct = (high_hours / total_hours * 100) if total_hours > 0 else 0.0

                                    unit = AGG_META[granularity_vol]["unit"]
                                    if granularity_vol == "Hourly":
                                        avg_label = "Average Hourly Volume"
                                        peak_label = "🔥 Peak Hourly Volume"
                                        avg_suffix = "vph"
                                    elif granularity_vol == "Daily":
                                        avg_label = "Average Daily Traffic (ADT)"
                                        peak_label = "🔥 Peak Daily Volume"
                                        avg_suffix = "vpd"
                                    elif granularity_vol == "Weekly":
                                        avg_label = "Average Weekly Traffic (AWT)"
                                        peak_label = "🔥 Peak Weekly Volume"
                                        avg_suffix = "vpw"
                                    else:
                                        avg_label = "Average Monthly Traffic (AMT)"
                                        peak_label = "🔥 Peak Monthly Volume"
                                        avg_suffix = "vpm"

                                    col1, col2, col3, col4, col5 = st.columns(5)

                                    with col1:
                                        badge = (
                                            "badge-critical" if peak_util_pct > 90 else
                                            "badge-poor" if peak_util_pct > 75 else
                                            "badge-fair" if peak_util_pct > 60 else
                                            "badge-good"
                                        )
                                        st.metric(peak_label, f"{peak_val:,.0f} {unit}", delta=f"95th: {p95_val:,.0f} {unit}")
                                        st.markdown(
                                            f'<span class="performance-badge {badge}">{peak_util_pct:.0f}% of Capacity</span>',
                                            unsafe_allow_html=True,
                                        )

                                    with col2:
                                        st.metric(
                                            f"📊 {avg_label}",
                                            f"{avg_bucket_val:,.0f} {avg_suffix}",
                                            help=("Average traffic on the selected aggregation.\n"
                                                  "• ADT = daily average\n• AWT = weekly average\n• AMT = monthly average"),
                                        )
                                        if granularity_vol == "Hourly":
                                            avg_util_pct_hourly = (hourly_avg / THEORETICAL_LINK_CAPACITY_VPH * 100) if THEORETICAL_LINK_CAPACITY_VPH else 0.0
                                            badge2 = "badge-good" if avg_util_pct_hourly <= 40 else ("badge-fair" if avg_util_pct_hourly <= 60 else "badge-poor")
                                            st.markdown(
                                                f'<span class="performance-badge {badge2}">{avg_util_pct_hourly:.0f}% Avg Util</span>',
                                                unsafe_allow_html=True,
                                            )
                                        else:
                                            badge2 = "badge-good" if avg_util_pct <= 40 else ("badge-fair" if avg_util_pct <= 60 else "badge-poor")
                                            st.markdown(
                                                f'<span class="performance-badge {badge2}">{avg_util_pct:.0f}% Avg Util</span>',
                                                unsafe_allow_html=True,
                                            )

                                    with col3:
                                        total_vehicles = float(np.nansum(raw["total_volume"]))
                                        st.metric(
                                            "🚗 Total Vehicles (period)",
                                            f"{total_vehicles:,.0f}",
                                            help="Sum of vehicles across the selected time window (computed from hourly records).",
                                        )
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
                                        label_cons = "Consistent" if cv_bucket < 30 else ("Variable" if cv_bucket < 50 else "Highly Variable")
                                        badge_cons = "badge-good" if cv_bucket < 30 else ("badge-fair" if cv_bucket < 50 else "badge-poor")
                                        st.markdown(
                                            f'<span class="performance-badge {badge_cons}">{label_cons}</span>',
                                            unsafe_allow_html=True,
                                        )

                                    with col5:
                                        st.metric(
                                            "⚠️ High Volume Hours",
                                            f"{high_hours}",
                                            delta=f"{risk_pct:.1f}% of time",
                                            help=f"Hourly records with total_volume > {HIGH_VOLUME_THRESHOLD_VPH:,} vph (always computed on the hourly base).",
                                        )
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
                                st.subheader("📈 Vehicle Volume Visualizations")
                                if 'raw' in locals() and len(filtered_volume_data) > 1:
                                    try:
                                        fig_trend, fig_box, fig_matrix = improved_volume_charts_for_tab2(
                                            raw_hourly_df=raw,
                                            granularity=granularity_vol,
                                            cap_vph=THEORETICAL_LINK_CAPACITY_VPH,
                                            high_vph=HIGH_VOLUME_THRESHOLD_VPH,
                                        )
                                        if fig_trend:
                                            st.plotly_chart(fig_trend, use_container_width=True, config=PLOTLY_CONFIG)
                                        colA, colB = st.columns(2)
                                        with colA:
                                            if fig_box:
                                                st.plotly_chart(fig_box, use_container_width=True, config=PLOTLY_CONFIG)
                                        with colB:
                                            if fig_matrix:
                                                st.plotly_chart(fig_matrix, use_container_width=True, config=PLOTLY_CONFIG)
                                    except Exception as e:
                                        st.error(f"❌ Error creating volume charts: {e}")

                                # ---------------- Insights ----------------
                                if 'raw' in locals() and not raw.empty:
                                    try:
                                        step("Generating insights & recommendations", 92)
                                        agg_all = _prep_bucket(raw, granularity_vol).groupby("local_datetime", as_index=False)["total_volume"].sum()
                                        if agg_all.empty:
                                            raise ValueError("No Prediction in selected window")

                                        if granularity_vol == "Monthly":
                                            agg_all["bucket_hours"] = pd.to_datetime(agg_all["local_datetime"]).dt.days_in_month * 24
                                        else:
                                            agg_all["bucket_hours"] = AGG_META[granularity_vol]["fixed_hours"]

                                        agg_all["cap"] = agg_all["bucket_hours"] * THEORETICAL_LINK_CAPACITY_VPH
                                        agg_all["thr"] = agg_all["bucket_hours"] * HIGH_VOLUME_THRESHOLD_VPH

                                        peak_idx = int(agg_all["total_volume"].idxmax())
                                        peak_val = float(agg_all.loc[peak_idx, "total_volume"])
                                        peak_ts = pd.to_datetime(agg_all.loc[peak_idx, "local_datetime"])
                                        avg_val = float(agg_all["total_volume"].mean())
                                        p95_val = float(np.nanpercentile(agg_all["total_volume"], 95)) if agg_all["total_volume"].notna().any() else 0.0

                                        peak_cap = float(agg_all.loc[peak_idx, "cap"])
                                        peak_util_pct = (peak_val / peak_cap * 100) if peak_cap > 0 else 0.0

                                        util_series = np.where(agg_all["cap"] > 0, agg_all["total_volume"] / agg_all["cap"], np.nan)
                                        p95_util_pct = float(np.nanpercentile(util_series * 100, 95)) if np.isfinite(util_series).any() else 0.0

                                        cv_bucket = (float(np.nanstd(agg_all["total_volume"])) / avg_val * 100) if avg_val > 0 else 0.0
                                        peak_to_avg = (peak_val / avg_val) if avg_val > 0 else 0.0

                                        hourly_over_thr = int((raw["total_volume"] > HIGH_VOLUME_THRESHOLD_VPH).sum())
                                        total_hours = int(raw["total_volume"].count())
                                        hourly_risk_pct = (hourly_over_thr / total_hours * 100) if total_hours > 0 else 0.0

                                        bucket_over_80_cap = int((agg_all["total_volume"] > 0.80 * agg_all["cap"]).sum())
                                        bucket_risk_pct = (bucket_over_80_cap / len(agg_all) * 100) if len(agg_all) else 0.0

                                        peak_bucket_all = _prep_bucket(raw, granularity_vol)
                                        top_in_peak = (
                                            peak_bucket_all.loc[peak_bucket_all["local_datetime"] == peak_ts]
                                                           .groupby("intersection_name", as_index=False)["total_volume"].sum()
                                                           .sort_values("total_volume", ascending=False)
                                        )
                                        top3 = top_in_peak.head(3)
                                        top3_list = " • ".join([f"{r['intersection_name']}: {int(r['total_volume']):,}" for _, r in top3.iterrows()]) if not top3.empty else "N/A"

                                        unit = AGG_META[granularity_vol]["unit"]
                                        label = AGG_META[granularity_vol]["label"]
                                        peak_when = _fmt_period(peak_ts, granularity_vol)

                                        if peak_util_pct >= 95 or hourly_risk_pct >= 20:
                                            rec = ("Immediate capacity relief (short-term: retime signals, dynamic splits & queue management; "
                                                   "mid-term: turn-lane/approach improvements; evaluate access control at peak contributors).")
                                            rec_badge = "badge-critical"
                                        elif peak_util_pct >= 85 or hourly_risk_pct >= 10 or bucket_risk_pct >= 25:
                                            rec = ("Prioritize signal optimization (AM/PM plans + progression), adjust cycle lengths, and "
                                                   "pilot demand management (driveway control, TSP). Plan spot upgrades at top 2–3 intersections.")
                                            rec_badge = "badge-poor"
                                        elif peak_util_pct >= 70 or hourly_risk_pct >= 5:
                                            rec = ("Retiming & coordination refresh, monitor weekly trends, and stage TSP/ITS enhancements.")
                                            rec_badge = "badge-fair"
                                        else:
                                            rec = ("Monitor; current capacity is adequate with routine timing review.")
                                            rec_badge = "badge-good"

                                        st.markdown(
                                            f"""
                                            <div class="insight-box">
                                                <h4>💡 Volume Analysis Insights</h4>
                                                <p><strong>📊 Capacity:</strong> Peak <b>{peak_val:,.0f} {unit}</b> on <b>{peak_when}</b>
                                                   ({peak_util_pct:.0f}% of scaled capacity) • 95th percentile <b>{p95_val:,.0f} {unit}</b> ({p95_util_pct:.0f}% of capacity).</p>
                                                <p><strong>🚗 Typical {label.capitalize()} Volume:</strong> Average <b>{avg_val:,.0f} {unit}</b> •
                                                   Peak/Avg ratio <b>{peak_to_avg:.1f}×</b> • Consistency <b>{max(0, 100 - cv_bucket):.0f}%</b>.</p>
                                                <p><strong>🧮 Total Vehicles (window):</strong> <b>{float(np.nansum(raw['total_volume'])):,.0f}</b>.</p>
                                                <p><strong>⚠️ Exposure:</strong> Hourly > {HIGH_VOLUME_THRESHOLD_VPH:,} vph for <b>{hourly_over_thr}</b> hours
                                                   (<b>{hourly_risk_pct:.1f}%</b> of hours) •
                                                   {label.capitalize()}s above 80% of scaled capacity: <b>{bucket_over_80_cap}</b>
                                                   (<b>{bucket_risk_pct:.1f}%</b> of {label}s).</p>
                                                <p><strong>📍 Peak Contributors:</strong> {top3_list}</p>
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
                                        labels=["🟢 Low Risk", "🟡 Moderate Risk", "🟠 High Risk", "🔴 Critical Risk", "🚨 Severe Risk"],
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
                                            "total_volume_mean": "Avg Volume (vph)",
                                            "total_volume_max": "Peak Volume (vph)",
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
                                            "📊 Peak Capacity %": st.column_config.NumberColumn("📊 Peak Capacity %", format="%.1f%%"),
                                            "📊 Avg Capacity %": st.column_config.NumberColumn("📊 Avg Capacity %", format="%.1f%%"),
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
                                render_cycle_length_section(raw)

        except Exception as e:
            st.error(f"❌ Error processing traffic Prediction: {e}")
            st.info("Please check your Prediction sources and try again.")

# -------------------------
# TAB 3: Acyclica (Travel Time + AI Prediction Models)
# -------------------------
with tab3:
    # Use the enhanced Tab 3 renderer with proper sidebar controls
    from acyclica_traveltime_analysis import render_tab3_analysis
    render_tab3_analysis()


    # Load Acyclica data for sidebar population
    acyclica_df = get_acyclica_df()

    with st.sidebar:
        with st.expander("⚙️ Pg.3 SETTINGS", expanded=True):
            st.caption("Acyclica Data: Speed + Travel Time Analysis")
            st.caption("AI Models: Peak Hour, Incident Detection, Event Impact")

            # Tab 3 Navigation
            st.markdown("## 🎯 Select Analysis Type")
            analysis_type = st.selectbox(
                "Choose Analysis",
                [
                    "🚗 Travel Time Analysis", 
                    "🔮 Peak Hour Prediction",
                    "⚠️ Incident Detection & Recovery", 
                    "🎪 Event Impact Analysis"
                ],
                key="tab3_analysis_type",
                help="Travel Time Analysis shows same metrics as Tab 1 but with Acyclica data + speed insights"
            )

    # Render the appropriate section based on selection
    if analysis_type == "🚗 Travel Time Analysis":
        render_acyclica_section()

    elif analysis_type == "🔮 Peak Hour Prediction":
        render_peak_hour_section()

    elif analysis_type == "⚠️ Incident Detection & Recovery":
        render_incident_detection_section()

    elif analysis_type == "🎪 Event Impact Analysis":
        render_event_impact_section()

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
