# VantageLivetab.py - Tab 4: Iteris VantageLive Analysis (Bikes + Vehicles + Pedestrians)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
import plotly.express as px

# Import shared utilities
from sidebar_functions import (
    date_range_preset_controls,
    render_badge,
    get_performance_rating,
    compute_data_availability,
)
from cycle_length_recommendations import render_cycle_length_section
from intersection_metadata import format_intersection_display, get_intersection_metadata
from Map import build_intersections_overview, INTERSECTION_TO_NODE

# Shared UI utils (scoped loader and tab highlight)
try:
    from ui_utils import cad_loader as scoped_cad_loader, set_active_search_tab, get_dynamic_xaxis_params, PerformanceMonitor, convert_df_to_csv
except ModuleNotFoundError:
    from core.ui_utils import cad_loader as scoped_cad_loader, set_active_search_tab, get_dynamic_xaxis_params, PerformanceMonitor, convert_df_to_csv

# =========================
# Constants
# =========================
# Vehicles
VEH_CAPACITY_VPH = 1800
VEH_HIGH_THRESHOLD_VPH = 1200

# Bikes — smaller so daily lines look sensible
BIKE_CAPACITY_VPH = 200
BIKE_HIGH_THRESHOLD_VPH = 60

# Pedestrians — align with bike defaults unless you provide field-calibrated caps
PED_CAPACITY_VPH = 200
PED_HIGH_THRESHOLD_VPH = 60

# Intersection list (exact order requested)
CANONICAL_INTERSECTIONS = [
    "Washington and Avenue 52",
    "Washington and Calle Tampico",
    "Washington and Village Shopping Center",
    "Washington and Avenue 50",
    "Washington and Sagebrush Avenue",
    "Washington and Eisenhower Drive",
    "Washington and Avenue 48",
    "Washington and Avenue 47",
    "Washington and Point Happy Simon",
]

# Map label aliases to match the map’s internal labels
MAP_LABEL_ALIASES = {
    "Washington and Avenue 52": "Washington St & Avenue 52",
    "Washington and Calle Tampico": "Washington St & Calle Tampico",
    "Washington and Village Shopping Center": "Washington St & Village Shopping Center",
    "Washington and Avenue 50": "Washington St & Avenue 50",
    "Washington and Sagebrush Avenue": "Washington St & Sagebrush Avenue",
    "Washington and Eisenhower Drive": "Washington St & Eisenhower Drive",
    "Washington and Avenue 48": "Washington St & Avenue 48",
    "Washington and Avenue 47": "Washington St & Avenue 47",
    "Washington and Point Happy Simon": "Washington St & Point Happy Simon",
}

# Aliases we normalize from raw files
INTERSECTION_ALIASES = {
    "Washington Street & Avenue 52": "Washington and Avenue 52",
    "Washington Street and Avenue 52": "Washington and Avenue 52",
    "Washington St & Avenue52": "Washington and Avenue 52",
    "Washington St & Ave 52": "Washington and Avenue 52",

    "Washington Street & Calle Tampico": "Washington and Calle Tampico",
    "Washington Street and Calle Tampico": "Washington and Calle Tampico",
    "Washington St & Calle Tampico": "Washington and Calle Tampico",

    "Washington Street & Village Shop Ctr": "Washington and Village Shopping Center",
    "Washington Street and Village Shop Ctr": "Washington and Village Shopping Center",
    "Washington St & Village Shop Ctr": "Washington and Village Shopping Center",
    "Washington and Village Shop Ctr": "Washington and Village Shopping Center",

    "Washington Street & Avenue 50": "Washington and Avenue 50",
    "Washington Street and Avenue 50": "Washington and Avenue 50",
    "Washington St & Avenue50": "Washington and Avenue 50",
    "Washington St & Ave 50": "Washington and Avenue 50",

    "Washington Street & Sagebrush Ave": "Washington and Sagebrush Avenue",
    "Washington Street and Sagebrush Ave": "Washington and Sagebrush Avenue",
    "Washington St & Sagebrush Ave": "Washington and Sagebrush Avenue",

    "Washington Street & Eisenhower Dr": "Washington and Eisenhower Drive",
    "Washington Street and Eisenhower Dr": "Washington and Eisenhower Drive",
    "Washington St & Eisenhower Dr": "Washington and Eisenhower Drive",

    "Washington Street & Avenue 48": "Washington and Avenue 48",
    "Washington Street and Avenue 48": "Washington and Avenue 48",
    "Washington St & Avenue48": "Washington and Avenue 48",
    "Washington St & Ave 48": "Washington and Avenue 48",

    "Washington Street & Avenue 47": "Washington and Avenue 47",
    "Washington Street and Avenue 47": "Washington and Avenue 47",
    "Washington St & Avenue47": "Washington and Avenue 47",
    "Washington St & Ave 47": "Washington and Avenue 47",
    "Washington Avenue and Avenue 47": "Washington and Avenue 47",
    "Washington Avenue and Ave 47": "Washington and Avenue 47",

    "Washington Street & Point Happy Way": "Washington and Point Happy Simon",
    "Washington Street and Point Happy Way": "Washington and Point Happy Simon",
    "Washington St & Point Happy Way": "Washington and Point Happy Simon",
    "Washington and Point Happy Way": "Washington and Point Happy Simon",
    "Washington and Point Happy Simon": "Washington and Point Happy Simon",
    "Washington Street & Point Happy Simon": "Washington and Point Happy Simon",
    "Washington St & Point Happy Simon": "Washington and Point Happy Simon",
}

# Aggregation metadata
AGG_META = {
    "Hourly": {"unit": "vehicles", "bucket": "H", "label": "hour", "fixed_hours": 1},
    "Daily": {"unit": "vehicles", "bucket": "D", "label": "day", "fixed_hours": 24},
    "Weekly": {"unit": "vehicles", "bucket": "W", "label": "week", "fixed_hours": 24 * 7},
    "Monthly": {"unit": "vehicles", "bucket": "M", "label": "month", "fixed_hours": None},
}

PLOTLY_CONFIG = {
    "displaylogo": False,
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "toggleSpikelines"]
}
MAP_HEIGHT = 1200

# ---------- Small helpers ----------
def kpi_title(label: str, help_text: str):
    safe_help = (help_text or "").replace('"', "&quot;")
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:2px;">
          <span style="font-weight:800;">{label}</span>
          <span title="{safe_help}"
                style="cursor:help;display:inline-flex;align-items:center;justify-content:center;
                       width:18px;height:18px;border-radius:50%;
                       background:rgba(0,0,0,.15);color:#fff;font-size:12px;line-height:18px;">?</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_kpi_card(title: str, value: str, delta: str, badge_text: str, badge_class: str, help_text: str):
    """Renders a single KPI card with title, value, delta, and status badge."""
    safe_help = (help_text or "").replace('"', "&quot;")
    badge_html = f'<div class="kpi-card-badge"><span class="performance-badge {badge_class}">{badge_text}</span></div>' if badge_text else ""
    return f'<div class="kpi-card"><div class="kpi-card-title"><span>{title}</span><span class="kpi-help-icon" title="{safe_help}">?</span></div><div class="kpi-card-value">{value}</div><div class="kpi-card-delta">{delta}</div>{badge_html}</div>'

def _canonicalize_intersection(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = " ".join(name.strip().split())
    if s in CANONICAL_INTERSECTIONS:
        return s
    if s in INTERSECTION_ALIASES:
        return INTERSECTION_ALIASES[s]
    s2 = s.replace("&", "and").replace("Street", "").replace("St", " ").replace("  ", " ").strip()
    s2 = (s2
          .replace("Avenue50", "Avenue 50").replace("Avenue52", "Avenue 52")
          .replace("Avenue48", "Avenue 48").replace("Avenue47", "Avenue 47")
          .replace("Sagebrush Ave", "Sagebrush Avenue")
          .replace("Eisenhower Dr", "Eisenhower Drive")
          .replace("Village Shop Ctr", "Village Shopping Center"))
    if s2 in CANONICAL_INTERSECTIONS:
        return s2
    if s2 in INTERSECTION_ALIASES:
        return INTERSECTION_ALIASES[s2]
    return ""

def _map_label_for(name: str) -> str:
    # Use the mapping if it exists, otherwise return the name as is.
    # This ensures "Washington and Avenue 48" maps to "Washington St & Avenue48"
    return MAP_LABEL_ALIASES.get(name, name)

def _mode_caps(mode_label: str):
    """Return (capacity_vph, high_threshold_vph) by mode."""
    if mode_label == "Bikes":
        return BIKE_CAPACITY_VPH, BIKE_HIGH_THRESHOLD_VPH
    if mode_label == "Pedestrians":
        return PED_CAPACITY_VPH, PED_HIGH_THRESHOLD_VPH
    return VEH_CAPACITY_VPH, VEH_HIGH_THRESHOLD_VPH

def _mode_noun(mode_label: str) -> str:
    if mode_label == "Bikes":
        return "bicycles"
    if mode_label == "Pedestrians":
        return "pedestrians"
    if "All Modes" in mode_label:
        return "movements"
    return "vehicles"

def _mode_emoji(mode_label: str) -> str:
    return "🚗" if mode_label == "Vehicles" else ("🚲" if mode_label == "Bikes" else ("🚶" if mode_label == "Pedestrians" else "📊"))

@st.cache_data(show_spinner=False)
def build_chart_context(mode_label, granularity, date_range, analysis_df, direction_filter, turn_filter, intersection_selection):
    """Centralized logic for building dynamic chart titles and subtitles."""
    # 1. Date Range String
    date_str = ""
    if date_range and len(date_range) == 2:
        d1, d2 = date_range
        if d1.month == d2.month and d1.year == d2.year:
            date_str = f"{d1.strftime('%b %d')}–{d2.strftime('%d, %Y')}"
        else:
            date_str = f"{d1.strftime('%b %d, %Y')} – {d2.strftime('%b %d, %Y')}"

    # 2. Intersection Count/Name
    if intersection_selection and intersection_selection != "All Intersections":
        intersection_str = format_intersection_display(intersection_selection)
    else:
        num_int = analysis_df["intersection_name"].nunique() if not analysis_df.empty else 0
        intersection_str = f"{num_int} intersections"

    # 3. Approach/Direction wording
    if not direction_filter or direction_filter == "All Directions":
        dir_str = "All approaches"
    else:
        # NORTHBOUND -> Northbound approach
        dir_str = f"{direction_filter.title()} approach"

    # 4. Turn wording
    if not turn_filter or turn_filter == "All Turns":
        turn_str = "All turns"
    else:
        if turn_filter == "Through":
            turn_str = "Through movements"
        elif turn_filter == "Left":
            turn_str = "Left turns"
        elif turn_filter == "Right":
            turn_str = "Right turns"
        else:
            turn_str = turn_filter

    # 5. Mode wording for Title
    mode_w = {
        "Combined (All Modes)": "Multimodal",
        "Vehicles": "Vehicle",
        "Bikes": "Bicycle",
        "Pedestrians": "Pedestrian"
    }.get(mode_label, mode_label)
    
    # 6. Specific Title/Subtitle Construction
    common_sub = f"{date_str} • {intersection_str} • {dir_str} • {turn_str}"
    
    if mode_label == "Combined (All Modes)":
        trend_title = f"Washington Street Corridor — {granularity} Multimodal Movement Volumes"
        box_title = f"Washington Street Corridor — {granularity} Multimodal Volume Distribution by Intersection"
        bar_title = f"Washington Street Corridor — {granularity} Multimodal Volume by Intersection"
        y_label = "Movement Volume"
        risk_mode = "Multimodal"
    else:
        trend_title = f"Washington Street Corridor — {granularity} {mode_w} Volumes"
        box_title = f"Washington Street Corridor — {granularity} {mode_w} Volume Distribution by Intersection"
        bar_title = f"Washington Street Corridor — {granularity} {mode_w} Volume by Intersection"
        y_label = f"{mode_w} Volume"
        risk_mode = mode_w

    risk_title = "Washington Street Corridor — Intersection Volume & Capacity Risk Analysis"
    risk_subtitle = f"{date_str} • {intersection_str} • {dir_str} • {risk_mode}"

    return {
        "trend": {"title": trend_title, "subtitle": common_sub},
        "box": {"title": box_title, "subtitle": common_sub},
        "bar": {"title": bar_title, "subtitle": common_sub},
        "risk": {"title": risk_title, "subtitle": risk_subtitle},
        "y_axis": y_label
    }

# =========================
# Data Loading
# =========================
@st.cache_data(show_spinner=False)
def load_vantage_bikes() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/Iteris_VantageLive/WashingtonStreet_ALL_Bikes.csv"
    try:
        df = pd.read_csv(url)
        df["local_datetime"] = pd.to_datetime(df["local_datetime"], format="%m/%d/%Y", errors="coerce")
        df = df.dropna(subset=["local_datetime"])
        df["intersection_name"] = df["segment_id"].astype(str).apply(_canonicalize_intersection)
        df = df[df["intersection_name"].isin(CANONICAL_INTERSECTIONS)]
        if "direction" in df.columns:
            df["direction"] = df["direction"].astype(str).str.strip().str.upper()
        if "turn_type" not in df.columns:
            df["turn_type"] = "Through"  # allow TMC + pies to work
        df["turn_type"] = df["turn_type"].astype(str).str.strip().str.title()
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        return df.sort_values("local_datetime").reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading bike data: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_vantage_vehicles() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/Iteris_VantageLive/WashingtonStreet_ALL_vehicles.csv"
    try:
        df = pd.read_csv(url)
        df["local_datetime"] = pd.to_datetime(df["local_datetime"], format="%m/%d/%Y", errors="coerce")
        df = df.dropna(subset=["local_datetime"])
        df["intersection_name"] = df["segment_id"].astype(str).apply(_canonicalize_intersection)
        df = df[df["intersection_name"].isin(CANONICAL_INTERSECTIONS)]
        if "direction" in df.columns:
            df["direction"] = df["direction"].astype(str).str.strip().str.upper()
        if "turn_type" in df.columns:
            df["turn_type"] = df["turn_type"].astype(str).str.strip().str.title()
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        return df.sort_values("local_datetime").reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading vehicle data: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_vantage_pedestrians() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/Iteris_VantageLive/WashingtonStreet_ALL_pedestrians.csv"
    try:
        df = pd.read_csv(url)
        df["local_datetime"] = pd.to_datetime(df["local_datetime"], errors="coerce")
        df = df.dropna(subset=["local_datetime"])
        df["intersection_name"] = df["segment_id"].astype(str).apply(_canonicalize_intersection)
        df = df[df["intersection_name"].isin(CANONICAL_INTERSECTIONS)]
        if "direction" in df.columns:
            df["direction"] = df["direction"].astype(str).str.strip().str.upper()
        # no turn_type in ped file -> treat as Through (crossing)
        df["turn_type"] = "Through"
        df["turn_type"] = df["turn_type"].astype(str).str.title()
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        return df.sort_values("local_datetime").reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading pedestrian data: {e}")
        return pd.DataFrame()

# =========================
# Processing Helpers
# =========================
@st.cache_data(show_spinner=False)
def _prep_bucket(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    meta = AGG_META[granularity]
    d = df.copy()
    d["local_datetime"] = pd.to_datetime(d["local_datetime"])
    if granularity == "Hourly":
        d["bucket"] = d["local_datetime"].dt.floor("H")
    elif granularity == "Daily":
        d["bucket"] = d["local_datetime"].dt.floor("D")
    elif granularity == "Weekly":
        d["bucket"] = d["local_datetime"].dt.to_period("W").dt.start_time
    else:
        d["bucket"] = d["local_datetime"].dt.to_period("M").dt.start_time

    group_cols = ["bucket", "intersection_name"]
    if "direction" in d.columns:
        group_cols.append("direction")
    if "turn_type" in d.columns:
        group_cols.append("turn_type")

    agg = (
        d.groupby(group_cols, as_index=False)
        .agg(volume=("volume", "sum"))
        .rename(columns={"bucket": "local_datetime"})
    )
    if granularity == "Monthly":
        agg["bucket_hours"] = pd.to_datetime(agg["local_datetime"]).dt.days_in_month * 24
    else:
        agg["bucket_hours"] = meta["fixed_hours"]
    return agg

@st.cache_data(show_spinner=False)
def _cap_series_for_x(x_df: pd.DataFrame, cap_vph: float, high_vph: float) -> pd.DataFrame:
    """Build capacity/threshold series aligned to unique time buckets only.
    Robust against duplicated rows from multiple directions/approaches.
    """
    if x_df.empty:
        return pd.DataFrame(columns=["local_datetime", "bucket_hours", "capacity", "high"]) 

    xs = (
        x_df[["local_datetime", "bucket_hours"]]
        .copy()
        .sort_values("local_datetime")
    )
    try:
        xs = xs.groupby("local_datetime", as_index=False)["bucket_hours"].first()
    except Exception:
        xs = xs.drop_duplicates(subset=["local_datetime", "bucket_hours"]) 

    xs["capacity"] = xs["bucket_hours"].astype(float) * float(cap_vph)
    xs["high"] = xs["bucket_hours"].astype(float) * float(high_vph)
    return xs

def _detect_data_resolution(df: pd.DataFrame) -> dict:
    """Checks if the dataframe contains hourly-level variation."""
    if df is None or df.empty:
        return {"hourly": False, "daily": True}
    
    d = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(d["local_datetime"]):
        d["local_datetime"] = pd.to_datetime(d["local_datetime"], errors='coerce')
    
    d = d.dropna(subset=["local_datetime"])
    if d.empty:
        return {"hourly": False, "daily": True}
        
    # Check if there are unique timestamps per intersection per day > 1
    d["date"] = d["local_datetime"].dt.date
    sample = d.head(5000) # Performance optimization
    counts = sample.groupby(["intersection_name", "date"])["local_datetime"].nunique()
    has_hourly = bool(counts.max() > 1)
    
    return {"hourly": has_hourly, "daily": True}

# =========================
# TMC Helpers
# =========================
def _filter_for_tmc(df: pd.DataFrame, intersection: str, date_range):
    if df is None or df.empty:
        return pd.DataFrame()
    out = df[
        (df["local_datetime"].dt.date >= date_range[0]) &
        (df["local_datetime"].dt.date <= date_range[1])
    ].copy()
    if intersection != "All Intersections":
        out = out[out["intersection_name"] == intersection]
    return out

@st.cache_data(show_spinner=False)
def _summarize_tmc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a dense table of approach x turn_type with volumes and shares.
    Expected columns: direction (NB/SB/EB/WB), turn_type (Left/Through/Right), volume.
    """
    if df.empty:
        return pd.DataFrame(columns=["approach", "turn", "volume", "pct", "total"])
    d = df.copy()
    d["direction"] = d["direction"].astype(str).str.upper()
    if "turn_type" not in d.columns:
        d["turn_type"] = "Through"
    d["turn_type"] = d["turn_type"].astype(str).str.title()

    dir_ok = ["NB", "SB", "EB", "WB"]
    turn_ok = ["Left", "Through", "Right"]
    d = d[d["direction"].isin(dir_ok)]
    d = d[d["turn_type"].isin(turn_ok)]

    g = d.groupby(["direction", "turn_type"], as_index=False)["volume"].sum()
    all_pairs = pd.MultiIndex.from_product([dir_ok, turn_ok], names=["direction", "turn_type"])
    g = g.set_index(["direction", "turn_type"]).reindex(all_pairs, fill_value=0).reset_index()

    tot = g.groupby("direction", as_index=False)["volume"].sum().rename(columns={"volume": "total"})
    x = g.merge(tot, on="direction", how="left")
    x["pct"] = np.where(x["total"] > 0, x["volume"] / x["total"], 0.0)
    x = x.rename(columns={"direction": "approach", "turn_type": "turn"})[
        ["approach", "turn", "volume", "pct", "total"]
    ]
    return x

def _pct_to_color(p):
    """0–10% green, 10–40% yellow, >40% orange; zero-total => gray."""
    if pd.isna(p):
        return "#cccccc"
    if p == -1:
        return "#e0e0e0"
    if p <= 0.10:
        return "#2ecc71"   # green
    if p <= 0.40:
        return "#f1c40f"   # yellow
    return "#e67e22"       # orange

def _add_badge(fig: go.Figure, x0, y0, x1, y1, text, fill, border="#ffffff"):
    fig.add_shape(
        type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
        line=dict(color=border, width=1),
        fillcolor=fill, layer="above"
    )
    fig.add_annotation(
        x=(x0 + x1) / 2.0, y=(y0 + y1) / 2.0,
        text=text, showarrow=False, font=dict(color="#000000", size=13)
    )

@st.cache_data(show_spinner=False)
def _tmc_figure(tmc_df: pd.DataFrame, intersection: str, date_range, mode_label: str):
    """
    Draw a schematic TMC diagram with four approaches and L/T/R shares.
    Coordinates: 0..10 in both axes, center at (5,5).
    Direction labels are positioned outside; EB/WB vertical.
    """
    turns = ["Left", "Through", "Right"]
    noun = _mode_noun(mode_label)

    approach_tot = tmc_df.groupby("approach", as_index=False)["total"].max()
    tot_map = dict(zip(approach_tot["approach"], approach_tot["total"]))

    def get_pair(app, turn):
        row = tmc_df[(tmc_df["approach"] == app) & (tmc_df["turn"] == turn)]
        if row.empty:
            return 0.0, 0
        return float(row["pct"].values[0]), int(row["volume"].values[0])

    fig = go.Figure()

    # Crosshair + center circle
    fig.add_shape(type="line", x0=5, y0=1.2, x1=5, y1=8.8, line=dict(color="#999999", width=2))
    fig.add_shape(type="line", x0=1.2, y0=5, x1=8.8, y1=5, line=dict(color="#999999", width=2))
    fig.add_shape(type="circle", x0=4.4, y0=4.4, x1=5.6, y1=5.6, line=dict(color="#888888", width=1))
    fig.add_annotation(x=5, y=5, text="N", showarrow=False, font=dict(size=12, color="#666"))

    fig.update_xaxes(visible=False, range=[-0.8, 10.8])
    fig.update_yaxes(visible=False, range=[-0.8, 10.8], scaleanchor="x")
    fig.update_layout(
        height=650,
        margin=dict(l=10, r=10, t=80, b=10),
        template="plotly_white",
        title=f"TMC — {mode_label} — {intersection}<br><sup>{date_range[0].strftime('%b %d, %Y')} to {date_range[1].strftime('%b %d, %Y')}</sup>"
    )

    # Badge positions
    top_boxes = {"Left": (3.4, 8.9, 4.9, 9.6), "Through": (4.95, 8.9, 5.95, 9.6), "Right": (6.1, 8.9, 7.6, 9.6)}
    bot_boxes = {"Left": (3.4, 0.4, 4.9, 1.1), "Through": (4.95, 0.4, 5.95, 1.1), "Right": (6.1, 0.4, 7.6, 1.1)}
    left_boxes = {"Left": (0.4, 6.6, 1.2, 7.2), "Through": (0.4, 5.5, 1.2, 6.1), "Right": (0.4, 4.4, 1.2, 5.0)}
    right_boxes = {"Left": (8.8, 6.6, 9.6, 7.2), "Through": (8.8, 5.5, 9.6, 6.1), "Right": (8.8, 4.4, 9.6, 5.0)}

    # Approach labels (outside) with totals; EB/WB rotated
    # Southbound (top): push higher
    fig.add_annotation(
        x=5.0, y=10.4, text="<b>SOUTHBOUND</b>", showarrow=False, xanchor="center",
        font=dict(size=15, color="#1a1a1a"),
        bgcolor="rgba(255,255,255,0.7)", borderpad=2
    )
    fig.add_annotation(
        x=5.0, y=10.05, text=f"({int(tot_map.get('SB',0)):,} total {noun})", showarrow=False, xanchor="center",
        font=dict(size=11, color="#777"),
        bgcolor="rgba(255,255,255,0.7)", borderpad=2
    )

    # Northbound (bottom): push lower
    fig.add_annotation(
        x=5.0, y=-0.4, text="<b>NORTHBOUND</b>", showarrow=False, xanchor="center",
        font=dict(size=15, color="#1a1a1a"),
        bgcolor="rgba(255,255,255,0.7)", borderpad=2
    )
    fig.add_annotation(
        x=5.0, y=-0.05, text=f"({int(tot_map.get('NB',0)):,} total {noun})", showarrow=False, xanchor="center",
        font=dict(size=11, color="#777"),
        bgcolor="rgba(255,255,255,0.7)", borderpad=2
    )

    # Eastbound (left side): move farther left & keep vertical text
    fig.add_annotation(
        x=-0.4, y=5.0, text="<b>EASTBOUND</b>", textangle=-90, showarrow=False, xanchor="center",
        font=dict(size=15, color="#1a1a1a"),
        bgcolor="rgba(255,255,255,0.7)", borderpad=2
    )
    fig.add_annotation(
        x=-0.05, y=5.0, text=f"({int(tot_map.get('EB',0)):,} total {noun})", textangle=-90,
        showarrow=False, xanchor="center", font=dict(size=11, color="#777"),
        bgcolor="rgba(255,255,255,0.7)", borderpad=2
    )

    # Westbound (right side): move farther right & keep vertical text
    fig.add_annotation(
        x=10.4, y=5.0, text="<b>WESTBOUND</b>", textangle=90, showarrow=False, xanchor="center",
        font=dict(size=15, color="#1a1a1a"),
        bgcolor="rgba(255,255,255,0.7)", borderpad=2
    )
    fig.add_annotation(
        x=10.05, y=5.0, text=f"({int(tot_map.get('WB',0)):,} total {noun})", textangle=90,
        showarrow=False, xanchor="center", font=dict(size=11, color="#777"),
        bgcolor="rgba(255,255,255,0.7)", borderpad=2
    )

    def draw_group(app_code, boxes):
        app_total = tot_map.get(app_code, 0)
        for turn in ["Left", "Through", "Right"]:
            p, v = get_pair(app_code, turn)
            if app_total <= 0:
                fill = _pct_to_color(-1)
                pct_txt = "0%"
            else:
                fill = _pct_to_color(p)
                pct_txt = f"{p*100:.0f}%"
            x0, y0, x1, y1 = boxes[turn]
            _add_badge(fig, x0, y0, x1, y1, pct_txt, fill)
            # counts near badges
            if boxes is top_boxes:
                ax, ay = (x0 + x1) / 2.0, y0 - 0.18
                anchor = "center"
            elif boxes is bot_boxes:
                ax, ay = (x0 + x1) / 2.0, y1 + 0.18
                anchor = "center"
            elif boxes is left_boxes:
                ax, ay = x1 + 0.20, (y0 + y1) / 2.0
                anchor = "left"
            else:
                ax, ay = x0 - 0.20, (y0 + y1) / 2.0
                anchor = "right"
            fig.add_annotation(
                x=ax, y=ay, text=f"<b>({v:,})</b>",
                showarrow=False, font=dict(size=11, color="#555"),
                xanchor=anchor, bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#e5e5e5", borderwidth=0.5, borderpad=2
            )

    draw_group("SB", top_boxes)
    draw_group("NB", bot_boxes)
    draw_group("EB", left_boxes)
    draw_group("WB", right_boxes)

    # Light cue arrows (tiny, unobtrusive)
    for (x,y,t) in [(4.25,8.7,"↙"), (5.45,8.7,"↓"), (6.7,8.7,"↘"),
                    (4.25,1.3,"↖"), (5.45,1.3,"↑"), (6.7,1.3,"↗"),
                    (1.35,6.85,"↗"), (1.35,5.75,"→"), (1.35,4.65,"↘"),
                    (8.65,6.85,"↖"), (8.65,5.75,"←"), (8.65,4.65,"↙")]:
        fig.add_annotation(x=x, y=y, text=t, showarrow=False, font=dict(size=12, color="#555"))
    return fig

# =========================
# Trend/Share Chart Helpers
# =========================
@st.cache_data(show_spinner=False)
def _calculate_kpis(df, granularity, mode_label, date_range, intersection, direction_filter, turn_filter):
    if df.empty:
        return None

    # Redundant numeric conversion removed (done during load)
    if not df["volume"].notna().any():
        return None

    # Optimization: Vectorized Link-Hour Counting
    d_temp = df.copy()
    d_temp["hour_bucket"] = d_temp["local_datetime"].dt.floor("H")
    
    # Define bucket for aggregation
    if granularity == "Hourly":
        d_temp["bucket"] = d_temp["hour_bucket"]
    elif granularity == "Daily":
        d_temp["bucket"] = d_temp["local_datetime"].dt.floor("D")
    elif granularity == "Weekly":
        d_temp["bucket"] = d_temp["local_datetime"].dt.to_period("W").dt.start_time
    else: # Monthly
        d_temp["bucket"] = d_temp["local_datetime"].dt.to_period("M").dt.start_time

    # Calculate volumes per bucket
    bucket_vol = d_temp.groupby("bucket")["volume"].sum()
    
    # Calculate unique link-hours per bucket efficiently
    # A link-hour is a unique combination of intersection, direction, turn_type, and HOUR.
    links = d_temp[["bucket", "intersection_name", "direction", "turn_type", "hour_bucket"]].drop_duplicates()
    bucket_links = links.groupby("bucket").size()
    
    bucket_all = pd.DataFrame({
        "volume": bucket_vol,
        "link_count": bucket_links
    }).reset_index().rename(columns={"bucket": "local_datetime"}).sort_values("local_datetime")

    if granularity == "Monthly":
        bucket_all["bucket_hours"] = pd.to_datetime(bucket_all["local_datetime"]).dt.days_in_month * 24
    else:
        bucket_all["bucket_hours"] = AGG_META[granularity]["fixed_hours"]

    # choose capacity by mode
    cap_key = "Vehicles" if mode_label == "All Modes" else mode_label
    CAP_VPH, HIGH_VPH = _mode_caps(cap_key)

    # Capacity scales by number of unique link-hours in the bucket
    bucket_all["cap"] = bucket_all["link_count"] * CAP_VPH
    bucket_all["threshold"] = bucket_all["link_count"] * HIGH_VPH
    util_series = np.where(bucket_all["cap"] > 0, bucket_all["volume"] / bucket_all["cap"] * 100, np.nan)

    peak_idx = int(bucket_all["volume"].idxmax())
    peak_val = float(bucket_all.loc[peak_idx, "volume"])
    peak_cap = float(bucket_all.loc[peak_idx, "cap"])
    peak_date = pd.to_datetime(bucket_all.loc[peak_idx, "local_datetime"])
    peak_util_pct = (peak_val / peak_cap * 100) if peak_cap > 0 else 0.0

    p95_val = float(np.nanpercentile(bucket_all["volume"], 95)) if bucket_all["volume"].notna().any() else 0.0
    avg_bucket_val = float(bucket_all["volume"].mean())
    avg_util_pct = float(np.nanmean(util_series)) if np.isfinite(util_series).any() else 0.0

    cv_bucket = (float(np.nanstd(bucket_all["volume"])) / avg_bucket_val * 100) if avg_bucket_val > 0 else 0.0
    
    bucket_all["threshold"] = bucket_all["bucket_hours"] * HIGH_VPH
    high_periods_df = bucket_all[bucket_all["volume"] > bucket_all["threshold"]]
    high_periods = int(len(high_periods_df))
    total_periods = int(len(bucket_all))
    risk_pct = (high_periods / total_periods * 100) if total_periods > 0 else 0.0

    return {
        "peak_val": peak_val,
        "peak_date": peak_date,
        "peak_util_pct": peak_util_pct,
        "avg_bucket_val": avg_bucket_val,
        "avg_util_pct": avg_util_pct,
        "cv_bucket": cv_bucket,
        "high_periods": high_periods,
        "risk_pct": risk_pct,
        "total_volume": float(bucket_all["volume"].sum()),
        "cap_total": float(bucket_all["cap"].sum()),
        "high_periods_df": high_periods_df[["local_datetime", "volume"]]
    }

def create_volume_charts(
    raw_df: pd.DataFrame,
    granularity: str,
    cap_vph: float,
    high_vph: float,
    mode_label: str = "Vehicles",
    direction_label: str = "",
    intersections: str = "",
    top_k: int = 10,
    date_range: tuple = None,
    turn_filter: str = "All Turns"
):
    if raw_df.empty:
        return None, None, None, None, None, None, None

    agg_all = _prep_bucket(raw_df, granularity)
    if agg_all.empty:
        return None, None, None, None, None, None, None

    # Context for Titles/Subtitles/Labels
    ctx = build_chart_context(mode_label, granularity, date_range, raw_df, direction_label, turn_filter, intersections)

    agg_for_plot = (
        agg_all.groupby(["local_datetime", "intersection_name"], as_index=False)["volume"]
        .sum()
    )

    order = agg_for_plot.groupby("intersection_name")["volume"].mean().sort_values(ascending=False)
    keep = order.index[:max(1, min(top_k, len(order)))]

    plot_df = agg_for_plot[agg_for_plot["intersection_name"].isin(keep)].copy().sort_values("local_datetime")
    
    # Optimization: Use mapping for formatting display names to avoid redundant work in loops
    unique_names = plot_df["intersection_name"].unique()
    name_map = {n: format_intersection_display(n) for n in unique_names}
    plot_df["display_name"] = plot_df["intersection_name"].map(name_map)
    
    unit = _mode_noun(mode_label)
    label = AGG_META[granularity]["label"]

    fig_trend = go.Figure()
    mode = "lines" if granularity == "Hourly" else "lines+markers"
    xfmt = "%Y-%m-%d %I:%M %p" if granularity == "Hourly" else "%Y-%m-%d"

    for name, g in plot_df.groupby("display_name"):
        fig_trend.add_trace(
            go.Scatter(
                x=g["local_datetime"],
                y=g["volume"],
                mode="lines+markers",
                name=name,
                marker=dict(size=6),
                hovertemplate=f"<b>%{{fullData.name}}</b><br>%{{x|{xfmt}}}<br>{mode_label}: %{{y:,.0f}} {unit}<extra></extra>",
            )
        )

    xs = _cap_series_for_x(agg_all[["local_datetime", "bucket_hours"]].drop_duplicates(), cap_vph, high_vph)
    fig_trend.add_trace(
        go.Scatter(
            x=xs["local_datetime"], y=xs["capacity"],
            name=f"Theoretical Capacity ({unit})", mode="lines",
            line=dict(dash="dash", color="red"),
            hovertemplate=f"%{{x|{xfmt}}}<br>Capacity: %{{y:,.0f}} {unit}<extra></extra>",
        )
    )
    fig_trend.add_trace(
        go.Scatter(
            x=xs["local_datetime"], y=xs["high"],
            name=f"High Volume Threshold ({unit})", mode="lines",
            line=dict(dash="dot", color="orange"),
            hovertemplate=f"%{{x|{xfmt}}}<br>Threshold: %{{y:,.0f}} {unit}<extra></extra>",
        )
    )
    # X-axis configuration
    xaxis_config = dict(
        title=dict(text="Date/Time", font=dict(size=16, weight="bold")),
        tickfont=dict(size=14),
    )
    if granularity == "Hourly" and not xs.empty:
        start_date = xs["local_datetime"].min()
        end_date = xs["local_datetime"].max()
        params = get_dynamic_xaxis_params(start_date, end_date)
        xaxis_config["dtick"] = params["dtick"]
        xaxis_config["tickformat"] = params["tickformat"]

    main_title = ctx["trend"]["title"]
    subtitle = ctx["trend"]["subtitle"]

    # Legend Row Estimation: ~3 items per row for better space utilization
    num_plotted = len(plot_df["intersection_name"].unique())
    num_legend_items = num_plotted + 2 # + Theoretical Capacity and High Volume Threshold
    estimated_legend_rows = max(1, (num_legend_items + 2) // 3)
    legend_height = estimated_legend_rows * 25
    top_margin = legend_height + 10

    fig_trend.update_layout(
        xaxis=xaxis_config,
        yaxis=dict(title=dict(text=f"{ctx['y_axis']} Counts", font=dict(size=16, weight="bold")), tickfont=dict(size=14)),
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1, 
            x=0
        ),
        margin=dict(l=10, r=10, t=top_margin, b=10),
        height=450 + (legend_height if estimated_legend_rows > 1 else 0),
        template="plotly_white",
    )

    cat_order = [format_intersection_display(n) for n in order[order.index.isin(keep)].index.tolist()]
    fig_box = px.box(
        plot_df, x="display_name", y="volume",
        category_orders={"display_name": cat_order},
        points=False,
    )
    box_title = ctx["box"]["title"]
    box_subtitle = ctx["box"]["subtitle"]
    fig_box.update_layout(
        xaxis_title="Intersection",
        yaxis_title=f"{ctx['y_axis']} per {label} ({unit})",
        margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_white",
    )

    mat = (
        plot_df.groupby("display_name", as_index=False)["volume"]
        .mean()
        .rename(columns={"volume": f"{label.capitalize()} Volume"})
    )
    mat["Rank"] = mat[f"{label.capitalize()} Volume"].rank(ascending=False, method="dense").astype(int)
    mat = mat.sort_values("Rank")
    fig_matrix = px.bar(
        mat, y="display_name", x=f"{label.capitalize()} Volume",
        orientation="h", text=f"{label.capitalize()} Volume",
    )
    matrix_title = ctx["bar"]["title"]
    matrix_subtitle = ctx["bar"]["subtitle"]
    fig_matrix.update_traces(
        texttemplate="<b>%{text:,.0f}</b>", 
        textposition="outside", 
        cliponaxis=False,
        textfont=dict(size=13, color="black")
    )
    fig_matrix.update_layout(
        xaxis_title=f"{label.capitalize()} {ctx['y_axis'].lower()} ({unit})",
        yaxis_title="",
        yaxis=dict(tickfont=dict(size=14)),
        margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_white",
        height=max(500, 150 + 60 * len(keep)),
        bargap=0.15,
        bargroupgap=0.05,
        uniformtext=dict(minsize=12, mode='show')
    )
    return fig_trend, fig_box, fig_matrix, main_title, subtitle, box_title, box_subtitle, matrix_title, matrix_subtitle

@st.cache_data(show_spinner=False)
def _get_map_adt_data(df_vehicles, date_range):
    if df_vehicles.empty or not date_range or len(date_range) < 2:
        return pd.Series(dtype=float)
    
    d1, d2 = date_range
    # Optimization: filter once
    mask = (df_vehicles["local_datetime"].dt.date >= d1) & (df_vehicles["local_datetime"].dt.date <= d2)
    subset = df_vehicles.loc[mask]
    
    if subset.empty:
        return pd.Series(dtype=float)
    
    # Avoid extra copy if we can
    subset = subset.copy()
    subset["date_only"] = subset["local_datetime"].dt.date
    daily_totals = subset.groupby(["intersection_name", "date_only"])["volume"].sum()
    return daily_totals.groupby("intersection_name").mean()

@st.fragment
def render_vantage_map_fragment(vehicles_df, bikes_df, peds_df, intersection, date_range):
    perf = PerformanceMonitor(enabled=st.session_state.get("show_perf", False))
    
    with perf.track("corridor map construction"):
        st.markdown("##### Map Display Settings")
        c1, c2 = st.columns(2)
        with c1:
            sat = st.toggle("🛰️ Satellite View", key="sat_toggle_frag", value=st.session_state.get("satellite_t4", False))
            st.session_state["satellite_t4"] = sat
            
            show_labels = st.toggle("🏷️ Show Map Labels", key="vantage_show_map_labels_frag", value=st.session_state.get("vantage_show_map_labels", False))
            st.session_state["vantage_show_map_labels"] = show_labels
        
        with c2:
            if show_labels:
                label_font_size = st.slider("Font Size", 8, 20, key="vantage_map_label_font_size_frag", value=st.session_state.get("vantage_map_label_font_size", 12))
                st.session_state["vantage_map_label_font_size"] = label_font_size
                
                bold_labels = st.toggle("Bold Labels", key="vantage_bold_map_labels_frag", value=st.session_state.get("vantage_bold_map_labels", False))
                st.session_state["vantage_bold_map_labels"] = bold_labels
            else:
                label_font_size = 12
                bold_labels = False

        try:
            # 1. Generate Tooltip Map for all intersections
            tooltip_map = {}
            from Map import INTERSECTION_TO_NODE, get_node_coordinates
            node_coords = get_node_coordinates()

            # Optimized ADT calculation
            adt_data = _get_map_adt_data(vehicles_df, date_range)

            # Build tooltips for each known intersection in the map
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
            }

            all_corridor_nodes = []
            for lbl in washington_st_data.keys():
                nid = INTERSECTION_TO_NODE.get(lbl)
                if nid and nid not in all_corridor_nodes:
                    all_corridor_nodes.append(nid)
            
            # Add nodes that have data
            vantage_intersection_names = set()
            if not vehicles_df.empty: vantage_intersection_names.update(vehicles_df["intersection_name"].unique())
            if not bikes_df.empty: vantage_intersection_names.update(bikes_df["intersection_name"].unique())
            if not peds_df.empty: vantage_intersection_names.update(peds_df["intersection_name"].unique())
            
            vantage_nodes = []
            vantage_map_labels_list = []
            for lbl in vantage_intersection_names:
                map_lbl = MAP_LABEL_ALIASES.get(lbl, lbl)
                node_id = INTERSECTION_TO_NODE.get(map_lbl)
                if node_id:
                    vantage_nodes.append(node_id)
                    vantage_map_labels_list.append(map_lbl)
                    if node_id not in all_corridor_nodes:
                        all_corridor_nodes.append(node_id)

            washington_st_data.update({
                "Washington St & Avenue50": washington_st_data["Washington St & Avenue 50"],
                "Washington St & Ave48": washington_st_data["Washington St & Avenue 48"],
                "Washington St & Ave47": washington_st_data["Washington St & Avenue 47"],
                "Washington St & Eisenhower": washington_st_data["Washington St & Eisenhower Drive"],
                "Washington St & Village Shop Ctr": washington_st_data["Washington St & Village Shopping Center"],
                "Washington St & Sagebrush Ave": washington_st_data["Washington St & Sagebrush Avenue"],
            })

            for disp_label, node_id in INTERSECTION_TO_NODE.items():
                canonical_name = _canonicalize_intersection(disp_label)
                adt_val = adt_data.get(canonical_name, adt_data.get(disp_label, 0))
                extra_info = f"ADT: {adt_val:,.0f}"
                if disp_label in cath_city_data:
                    extra_info += "<br>Agency: Cathedral City"
                    if "cycle" in cath_city_data[disp_label]:
                        extra_info += f"<br>Cycle Length: {cath_city_data[disp_label]['cycle']}"
                if disp_label in washington_st_data:
                    ws_d = washington_st_data[disp_label]
                    if "agency" in ws_d:
                        extra_info += f"<br>Agency: {ws_d['agency']}"
                    if "cycle" in ws_d:
                        extra_info += f"<br>Cycle Length: {ws_d['cycle']}"
                tooltip_map[disp_label] = extra_info

            if intersection == "All Intersections":
                selected_map_label = vantage_map_labels_list if vantage_map_labels_list else ["__NONE__"]
            else:
                selected_map_label = _map_label_for(intersection)

            fig_map = build_intersections_overview(
                selected_label=selected_map_label,
                corridor="Washington Street",
                tooltip_map=tooltip_map,
                satellite=sat,
                filter_to_nodes=all_corridor_nodes,
                show_labels=show_labels,
                label_font_size=label_font_size,
                bold_labels=bold_labels
            )
        except Exception:
            fig_map = None

        if fig_map:
            try:
                fig_map.update_layout(title="", height=MAP_HEIGHT, margin=dict(l=0, r=0, t=10, b=0))
            except Exception:
                pass
            
            if intersection == "All Intersections":
                key_suffix = "all"
            else:
                key_suffix = str(intersection).replace(' ', '_').replace('+', 'plus')
            
            map_key = f"t4_map_{key_suffix}_{int(show_labels)}_{label_font_size}_{int(bold_labels)}_{int(sat)}"
            st.plotly_chart(fig_map, use_container_width=True, config=PLOTLY_CONFIG, key=map_key)
            if intersection != "All Intersections":
                st.caption(f"Selected: **{intersection}**")
        else:
            st.caption("Map: Washington Street Corridor")
    
    perf.report()

# =========================
# Main Renderer
# =========================
def render_vantage_tab():
    perf = PerformanceMonitor(enabled=st.session_state.get("show_perf", False))

    st.markdown("""
    <style>
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 12px;
        width: 100%;
        margin: 10px 0 25px 0;
    }
    @media (max-width: 1200px) {
        .kpi-grid { grid-template-columns: repeat(3, 1fr); }
    }
    @media (max-width: 768px) {
        .kpi-grid { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 480px) {
        .kpi-grid { grid-template-columns: 1fr; }
    }
    .kpi-card {
        min-width: 0;
        box-sizing: border-box;
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #eaedf2;
        display: flex;
        flex-direction: column;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    .kpi-card-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        min-height: 2.5em;
        line-height: 1.2;
    }
    .kpi-card-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 0.25rem;
        line-height: 1.1;
    }
    .kpi-card-delta {
        font-size: 0.9rem;
        font-weight: 600;
        color: #27ae60;
        margin-bottom: 1.25rem;
    }
    .kpi-card-badge {
        margin-top: auto;
    }
    .kpi-help-icon {
        cursor: help;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #f1f5f9;
        color: #94a3b8;
        font-size: 11px;
        font-weight: bold;
    }
    /* Chart Card Styling via bordered containers */
    [data-testid="stVerticalBlockBordered"] {
        background: var(--secondary-background-color, white) !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        border-radius: 14px !important;
        padding: 24px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03) !important;
        margin-bottom: 24px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    with perf.track("raw VantageLive data load"):
        bikes_df = load_vantage_bikes()
        vehicles_df = load_vantage_vehicles()
        peds_df = load_vantage_pedestrians()

    # -------- Sidebar controls --------
    with st.sidebar:
        with st.expander("Pg.4 Iteris VantageLive Settings", expanded=False):
            st.caption("Select Mode, Intersection(s) and Date Range")
            st.caption("Data: Bike, Vehicle, and Pedestrian Volume from Iteris VantageLive")

            st.markdown("## Select Mode")
            mode = st.selectbox(
                "Analysis Mode",
                ["Vehicles", "Bikes", "Pedestrians", "Combined (All Modes)"],
                key="vantage_mode",
            )

            st.markdown("## Select Intersection")
            intersection = st.selectbox(
                "Intersection",
                ["SELECT", "All Intersections"] + CANONICAL_INTERSECTIONS,
                key="vantage_intersection",
                format_func=lambda x: format_intersection_display(x) if x not in ["SELECT", "All Intersections"] else x
            )

            # Loading animation when a real selection is made (leaving 'Select')
            prev_vantage_intersection = st.session_state.get("vantage_intersection_prev")
            if prev_vantage_intersection != intersection:
                st.session_state["vantage_intersection_prev"] = intersection

            # Availability preview for selected intersection (before date picker)
            try:
                # Choose the dataset based on mode selection
                if mode == "Vehicles":
                    base_df = vehicles_df
                elif mode == "Bikes":
                    base_df = bikes_df
                elif mode == "Pedestrians":
                    base_df = peds_df
                else:
                    # Combined (All Modes): concatenate the three (aligned on datetime/name only)
                    frames = []
                    for d in (vehicles_df, bikes_df, peds_df):
                        if d is not None and not d.empty:
                            frames.append(d[[c for c in d.columns if c in ("local_datetime", "intersection_name")]])
                    base_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

                # When 'Select' is chosen, summarize the whole (mode) dataset by passing intersection=None
                intersection_for_avail = None if intersection == "SELECT" else intersection
                avail = compute_data_availability(
                    base_df if base_df is not None else pd.DataFrame(),
                    intersection_col="intersection_name",
                    intersection=intersection_for_avail,
                    max_gaps=3,
                    current_date=pd.Timestamp.now().floor("H"),
                )
                if avail.get("start") and avail.get("end"):
                    start_str = avail["start"].strftime("%b %d, %Y %I:%M %p")
                    end_str = avail["end"].strftime("%b %d, %Y %I:%M %p")
                    mb = avail.get("size_mb", 0.0)
                    size_str = f"({mb:.1f} MB)" if mb > 0 else ""
                    # Dynamic header based on selection
                    if intersection == "SELECT":
                        header_label = "Available Data"
                    elif intersection == "All Intersections":
                        header_label = "Available Data for this Corridor"
                    else:
                        header_label = f"Available Data for {intersection}"
                    st.caption(header_label)
                    st.caption(f"• Date Range: {start_str} → {end_str} {size_str}")
                    tail_gap = avail.get("tail_gap")
                    if tail_gap:
                        st.caption("• Missing Data: " + tail_gap)
                    else:
                        gaps = avail.get("gaps") or []
                        if len(gaps) == 0:
                            st.caption("• Missing Data: None")
                        else:
                            st.caption("• Missing Data: " + "; ".join(gaps))
            except Exception:
                # Don’t break the sidebar if availability fails
                pass

            # Progressive disclosure: reveal the rest only after a valid selection
            if intersection != "SELECT": 
                # Date range bounds
                candidates = []
                if not bikes_df.empty: candidates += [bikes_df["local_datetime"]]
                if not vehicles_df.empty: candidates += [vehicles_df["local_datetime"]]
                if not peds_df.empty: candidates += [peds_df["local_datetime"]]
                if candidates:
                    min_date = min(c.min().date() for c in candidates)
                    max_date = max(c.max().date() for c in candidates)
                else:
                    min_date = datetime.today().date() - timedelta(days=7)
                    max_date = datetime.today().date()

                st.markdown("## Date And Time")
                date_range = date_range_preset_controls(min_date, max_date, key_prefix="vantage")

                st.markdown("## Granularity")
                granularity = st.selectbox(
                    "Data Aggregation",
                    ["Hourly", "Daily", "Weekly", "Monthly"],
                    index=1,  # Daily by default
                    key="granularity_vantage",
                )

                # Hourly support check
                hourly_supported = True
                check_df = pd.DataFrame()
                if mode == "Vehicles": check_df = vehicles_df
                elif mode == "Bikes": check_df = bikes_df
                elif mode == "Pedestrians": check_df = peds_df
                else: check_df = vehicles_df # Representative for Combined
                
                res = _detect_data_resolution(check_df)
                if not res["hourly"]:
                    hourly_supported = False
                    if granularity == "Hourly":
                        st.warning("⚠️ Hourly data is not available for this dataset.\nThe finest available resolution is Daily.")

                # (cleaner sidebar) — no extra headings for filters
                direction_filter = st.selectbox(
                    "Direction",
                    ["All Directions", "NB", "SB", "EB", "WB"],
                    key="direction_filter_vantage",
                )

                turn_filter = None
                if mode in ["Vehicles", "Combined (All Modes)"]:
                    turn_filter = st.selectbox(
                        "Turn Type",
                        ["All Turns", "Through", "Left", "Right"],
                        key="turn_filter_vantage",
                    )

                st.markdown("## Chart Type")
                chart_type = st.radio(
                    "Visualization",
                    ["Trend (Line)", "Share (Pie)", "TMC (Turning Movement Counts)"],
                    index=0 if mode != "Bikes" else 1,
                    horizontal=True,
                    key="vantage_chart_type",
                )

                vantage_current = {
                    "mode": mode,
                    "intersection": intersection,
                    "date_range": tuple(date_range) if date_range else None,
                    "granularity": granularity,
                    "direction_filter": direction_filter,
                    "turn_filter": turn_filter,
                    "chart_type": chart_type,
                }
                st.session_state["vantage_current"] = vantage_current

                # Block generate if Hourly is unsupported
                gen_disabled = (granularity == "Hourly" and not hourly_supported)
                if st.button("**Generate**", key="search_vantage", type="primary", use_container_width=True, disabled=gen_disabled):
                    st.session_state["vantage_params"] = vantage_current
                    st.session_state["vantage_ready"] = True
                    set_active_search_tab("t4")
                    st.session_state["last_active_tab"] = "t4"
                    # Persist committed search to URL
                    try:
                        ds, de = None, None
                        if vantage_current.get("date_range"):
                            ds = vantage_current["date_range"][0].isoformat()
                            de = vantage_current["date_range"][1].isoformat()
                        st.query_params.update(
                            t4_ready="1",
                            t4_mode=mode,
                            t4_intersection=intersection,
                            t4_date_start=ds or "",
                            t4_date_end=de or "",
                            t4_granularity=granularity,
                            t4_direction=direction_filter,
                            t4_turn=turn_filter,
                            t4_chart=chart_type,
                            last_tab="t4",
                        )
                    except Exception:
                        pass

    # -------- Main content area --------
    if not st.session_state.get("vantage_ready", False):
        st.info("Choose your Mode, Intersection and Date Range in the settings to the left.")
        return

    params = st.session_state.get("vantage_params", {})
    mode = params.get("mode", "Vehicles")
    intersection = params.get("intersection", "All Intersections")
    date_range = params.get("date_range")
    granularity = params.get("granularity", "Daily")
    direction_filter = params.get("direction_filter", "All Directions")
    turn_filter = params.get("turn_filter", "All Turns")
    chart_type = params.get("chart_type", "Trend (Line)")

    # Pending-change warning if sidebar controls differ from committed params
    t4_pending = st.session_state.get("vantage_ready", False) and (
        params != st.session_state.get("vantage_current", {})
    )
    if t4_pending:
        st.warning("Press **Generate** to refresh.")

    if not date_range or len(date_range) != 2:
        st.warning("Please select both start and end dates to proceed.")
        return

    try:
        # Filters for non-TMC charts (direction/turn filters apply here)
        def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            out = df[
                (df["local_datetime"].dt.date >= date_range[0]) &
                (df["local_datetime"].dt.date <= date_range[1])
            ].copy()
            if intersection != "All Intersections":
                out = out[out["intersection_name"] == intersection]
            if direction_filter != "All Directions" and "direction" in out.columns:
                out = out[out["direction"].str.upper() == direction_filter]
            return out

        with scoped_cad_loader("Fetching Data...", tab_id="t4") as step:
            with perf.track("date/mode/direction/turn filtering"):
                step("Applying filters", 20)
                working_bikes = apply_filters(bikes_df.copy()) if not bikes_df.empty else pd.DataFrame()
                working_vehicles = apply_filters(vehicles_df.copy()) if not vehicles_df.empty else pd.DataFrame()
                working_peds = apply_filters(peds_df.copy()) if not peds_df.empty else pd.DataFrame()

                if turn_filter and turn_filter != "All Turns" and not working_vehicles.empty and "turn_type" in working_vehicles.columns:
                    working_vehicles = working_vehicles[working_vehicles["turn_type"] == turn_filter]

                # Mode pick for non-TMC
                if mode == "Bikes":
                    analysis_df = working_bikes
                    mode_label = "Bikes"
                elif mode == "Vehicles":
                    analysis_df = working_vehicles
                    mode_label = "Vehicles"
                elif mode == "Pedestrians":
                    analysis_df = working_peds
                    mode_label = "Pedestrians"
                else:  # Combined (All Modes)
                    dfs = []
                    if not working_bikes.empty:
                        dfs.append(working_bikes.assign(mode="Bikes"))
                    if not working_vehicles.empty:
                        dfs.append(working_vehicles.assign(mode="Vehicles"))
                    if not working_peds.empty:
                        dfs.append(working_peds.assign(mode="Pedestrians"))
                    analysis_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
                    mode_label = "All Modes"

            step("Preparing layout", 55)

        # ---------- Right rail map ----------
        # Derive valid VantageLive intersections for the header and map
        vantage_intersection_names = set()
        if not vehicles_df.empty: vantage_intersection_names.update(vehicles_df["intersection_name"].unique())
        if not bikes_df.empty: vantage_intersection_names.update(bikes_df["intersection_name"].unique())
        if not peds_df.empty: vantage_intersection_names.update(peds_df["intersection_name"].unique())
        
        vantage_map_labels = []
        for lbl in vantage_intersection_names:
            map_lbl = MAP_LABEL_ALIASES.get(lbl, lbl)
            if map_lbl in INTERSECTION_TO_NODE:
                vantage_map_labels.append(map_lbl)

        content_col, right_col = st.columns([7, 3.5], gap="large")
        with right_col:
            st.markdown('<div id="vantage-map-anchor"></div>', unsafe_allow_html=True)
            st.markdown("##### Corridor Map", help="Stays visible while you scroll the analysis on the left.")

            # Call the optimized fragment for the map
            render_vantage_map_fragment(
                vehicles_df=vehicles_df, 
                bikes_df=bikes_df, 
                peds_df=peds_df, 
                intersection=intersection, 
                date_range=date_range
            )

        # ---------- Main content ----------
        with content_col:
            span = (date_range[1] - date_range[0]).days + 1
            total_obs = len(analysis_df)
            
            logo_url = "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/Logos/Iteris%20Vantage%20Live%20Logo%20TRANSPARENT.png"
            num_int = len(vantage_map_labels)
            intersection_display = (
                format_intersection_display(intersection) if intersection != "All Intersections" 
                else f"Washington Street Corridor — {num_int} {'Intersection' if num_int == 1 else 'Intersections'}"
            )

            st.markdown(
                f"""
                <div class="cvag-header-banner">
                  <div class="cvag-header-top">
                    <div>
                      <div class="cvag-header-title">Advantec Consulting Engineers <span style="font-weight:400; opacity:0.85; border-left:1px solid rgba(255,255,255,0.35); padding-left:12px; margin-left:10px; font-size:1.45rem;">Iteris VantageLive Volume Analysis</span></div>
                      <div class="cvag-header-subtitle">Multi-Modal Intersection Volume & Traffic Signal Performance</div>
                    </div>
                    <div class="cvag-tooltip-trigger" title="Iteris VantageLive high-resolution multi-modal vehicular, bicycle, and pedestrian volume monitoring for CVAG corridors.">?
                      <span class="cvag-tooltip-box">Iteris VantageLive high-resolution multi-modal vehicular, bicycle, and pedestrian volume monitoring for CVAG corridors.</span>
                    </div>
                  </div>
                  <div class="cvag-header-meta-grid">
                    <div class="cvag-meta-card cvag-cycle-card">
                      <div class="cvag-meta-card-label">Date Range</div>
                      <div class="cvag-meta-card-val" title="{date_range[0].strftime('%b %d, %Y')} — {date_range[1].strftime('%b %d, %Y')} ({span} days)">{date_range[0].strftime('%b %d, %Y')} — {date_range[1].strftime('%b %d, %Y')}</div>
                    </div>
                    <div class="cvag-meta-card">
                      <div class="cvag-meta-card-label">Intersection(s)</div>
                      <div class="cvag-meta-card-val" title="{intersection_display}">{intersection_display}</div>
                    </div>
                    <div class="cvag-meta-card">
                      <div class="cvag-meta-card-label">Mode</div>
                      <div class="cvag-meta-card-val" title="{mode_label} Volume">{mode_label} Volume</div>
                    </div>
                  </div>
                  <div class="cvag-header-chips-row">
                    <span class="cvag-sub-chip"><strong>Direction:</strong> {direction_filter}</span>
                    <span class="cvag-sub-chip"><strong>Granularity:</strong> {granularity} Aggregation</span>
                    <span class="cvag-sub-chip"><strong>Region:</strong> Coachella Valley</span>
                    <span class="cvag-sub-chip"><strong>Observations:</strong> {total_obs:,}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ============= TMC VIEW =============
            if chart_type == "TMC (Turning Movement Counts)":
                if intersection == "All Intersections":
                    st.info("TMC requires a single intersection. Please pick one in the sidebar.")
                    return
                if mode == "Combined (All Modes)":
                    st.info("TMC is designed for a single mode. Please select **Vehicles**, **Bikes**, or **Pedestrians**.")
                    return

                base_df = _filter_for_tmc(
                    vehicles_df if mode == "Vehicles" else (bikes_df if mode == "Bikes" else peds_df),
                    intersection, date_range
                )
                if base_df.empty:
                    st.warning("No data available for the selected dates and intersection.")
                    return

                tmc_table = _summarize_tmc(base_df)
                fig_tmc = _tmc_figure(tmc_table, intersection, date_range, mode_label)
                st.plotly_chart(
                    fig_tmc,
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                    key=f"t4_tmc_{mode}_{(intersection or 'all').replace(' ', '_')}_{date_range[0]}_{date_range[1]}"
                )
                with st.expander("📄 TMC Summary Table (volumes and shares)", expanded=False):
                    show = tmc_table.rename(columns={
                        "approach": "Approach", "turn": "Turn",
                        "volume": "Volume", "pct": "Approach volume (%)", "total": "Approach Total",
                    }).sort_values(["Approach", "Turn"])
                    # convert to percentage 0-100
                    show["Approach volume (%)"] = (show["Approach volume (%)"] * 100).round(1)

                    # append units for volume + totals
                    noun = _mode_noun(mode_label)
                    show["Volume"] = (
                        show["Volume"].astype(float).round(0).astype(int).map(lambda v: f"{v:,.0f} {noun.lower()}")
                    )
                    show["Approach Total"] = (
                        show["Approach Total"].astype(float).round(0).astype(int).map(lambda v: f"{v:,.0f} {noun.lower()}")
                    )

                    st.dataframe(
                        show,
                        use_container_width=True,
                        column_config={
                            "Approach volume (%)": st.column_config.NumberColumn("Approach volume (%)", format="%.1f%%")
                        }
                    )
                    st.download_button(
                        "Download TMC Summary (CSV)",
                        data=convert_df_to_csv(show),
                        file_name=f"tmc_{mode.lower()}_{intersection.replace(' ','_')}_{date_range[0]}_{date_range[1]}.csv",
                        mime="text/csv",
                    )
                return  # end TMC view

            # ---------- Analysis Context (Consolidated) ----------
            with perf.track("analysis context generation"):
                bucketed_df = _prep_bucket(analysis_df, granularity)
                chart_ctx = build_chart_context(mode_label, granularity, date_range, analysis_df, direction_filter, turn_filter, intersection)
                risk_ctx = chart_ctx # reuse for risk section

            # ---------- KPIs ----------
            # Note: analysis_df already has correct numeric/datetime types from load functions
            raw = analysis_df 

            st.subheader(f"{mode_label} Demand Performance Indicators")
            with perf.track("KPI calculation"):
                res = _calculate_kpis(analysis_df, granularity, mode_label, date_range, intersection, direction_filter, turn_filter)
            if res:
                peak_val = res["peak_val"]
                peak_date = res["peak_date"]
                peak_util_pct = res["peak_util_pct"]
                avg_bucket_val = res["avg_bucket_val"]
                avg_util_pct = res["avg_util_pct"]
                cv_bucket = res["cv_bucket"]
                high_periods = res["high_periods"]
                risk_pct = res["risk_pct"]
                total_volume = res["total_volume"]
                cap_total = res["cap_total"]
                high_dates_df = res["high_periods_df"]

                unit = _mode_noun(mode_label)
                label = AGG_META[granularity]["label"]
                avg_suffix = _mode_noun(mode_label)

                cap_key = "Vehicles" if mode_label == "All Modes" else mode_label
                _, HIGH_VPH = _mode_caps(cap_key)

                if granularity == "Hourly":
                    label = "hour"
                    avg_label = f"Hourly {mode_label}"
                    if mode_label == "All Modes":
                        avg_title = "AVERAGE HOURLY MOVEMENTS"
                        peak_title = "Peak Hourly Movements"
                    elif mode_label == "Vehicles":
                        avg_title = "Average Hourly Traffic"
                        peak_title = "Peak Hourly Traffic"
                    else:
                        avg_title = avg_label
                        peak_title = f"Peak Hourly {mode_label}"
                    display_threshold = HIGH_VPH
                    threshold_help = f"High-Volume Threshold: > {display_threshold:,} {avg_suffix}."
                elif granularity == "Daily":
                    label = "day"
                    avg_label = f"Daily {mode_label}"
                    if mode_label == "All Modes":
                        avg_title = "AVERAGE DAILY MOVEMENTS"
                        peak_title = "Maximum Daily Volume (Multi-Modal)"
                    elif mode_label == "Vehicles":
                        avg_title = "Average Daily Traffic (ADT)"
                        peak_title = "Peak Daily Traffic"
                    elif mode_label == "Bikes":
                        avg_title = "Average Daily Bicycle Volume"
                        peak_title = "Peak Daily Bicycles"
                    elif mode_label == "Pedestrians":
                        avg_title = "Average Daily Pedestrian Volume"
                        peak_title = "Peak Daily Pedestrians"
                    else:
                        avg_title = avg_label
                        peak_title = f"Peak Daily {mode_label}"
                    display_threshold = HIGH_VPH * 24
                    threshold_help = f"High-Volume Threshold (daily): > {display_threshold:,} {avg_suffix} (scaled from {HIGH_VPH:,} {avg_suffix} × 24h)."
                elif granularity == "Weekly":
                    label = "week"
                    avg_label = f"Weekly {mode_label}"
                    if mode_label == "All Modes":
                        avg_title = "AVERAGE WEEKLY MOVEMENTS"
                        peak_title = "Peak Weekly Movements"
                    else:
                        avg_title = avg_label
                        peak_title = f"Peak Weekly {mode_label}"
                    display_threshold = HIGH_VPH * 24 * 7
                    threshold_help = f"High-Volume Threshold (weekly): > {display_threshold:,} {avg_suffix} (scaled from {HIGH_VPH:,} {avg_suffix} × 168h)."
                else:
                    label = "month"
                    avg_label = f"Monthly {mode_label}"
                    if mode_label == "All Modes":
                        avg_title = "AVERAGE MONTHLY MOVEMENTS"
                        peak_title = "Peak Monthly Movements"
                    else:
                        avg_title = avg_label
                        peak_title = f"Peak Monthly {mode_label}"
                    display_threshold = HIGH_VPH * 24 * 30
                    threshold_help = f"High-Volume Threshold (monthly): > ~{display_threshold:,} {avg_suffix} (scaled from {HIGH_VPH:,} {avg_suffix} × hours in month)."

                # --- KPI Card 1: Peak ---
                badge_peak = (
                    "badge-critical" if peak_util_pct > 90 else
                    "badge-poor" if peak_util_pct > 75 else
                    "badge-fair" if peak_util_pct > 60 else
                    "badge-good"
                )
                if granularity == "Hourly":
                    peak_period_str = f"{peak_date.strftime('%A')}, {peak_date.strftime('%m/%d/%Y %H:00')}"
                elif granularity == "Daily":
                    peak_period_str = f"{peak_date.strftime('%A')}, {peak_date.strftime('%m/%d/%Y')}"
                elif granularity == "Weekly":
                    p = pd.Period(peak_date, freq='W')
                    ws, we = p.start_time, p.end_time
                    peak_period_str = f"{ws.strftime('%m/%d/%Y')} – {we.strftime('%m/%d/%Y')}"
                else:  # Monthly
                    peak_period_str = peak_date.strftime('%B %Y')

                card1 = render_kpi_card(
                    peak_title,
                    f"{peak_val:,.0f} {avg_suffix}",
                    peak_period_str,
                    f"{peak_util_pct:.0f}% of Capacity",
                    badge_peak,
                    f"Highest {mode_label} total within the selected period. Shows the exact peak period below."
                )

                # --- KPI Card 2: Average ---
                card2 = render_kpi_card(
                    avg_title,
                    f"{avg_bucket_val:,.0f} {avg_suffix}",
                    f"Average utilization: {avg_util_pct:.0f}%",
                    "",
                    "",
                    f"Mean of {mode_label} totals across the selected period (after all filters). Delta shows average utilization vs capacity."
                )

                # --- KPI Card 3: Total ---
                ratio = (total_volume / cap_total) if cap_total > 0 else 0.0
                state_badge = "badge-good" if ratio < 0.40 else ("badge-fair" if ratio < 0.70 else "badge-poor")
                
                if mode_label == "All Modes":
                    total_title = "TOTAL MOVEMENTS IN PERIOD"
                    badge_text = f"Vehicle Util: {ratio*100:.0f}%"
                else:
                    total_title = f"TOTAL {mode_label.upper()} IN PERIOD"
                    badge_text = f"Period utilization: {ratio*100:.0f}%"
                
                period_str = f"{date_range[0].strftime('%b %d')} – {date_range[1].strftime('%d, %Y')} • {span} days"
                if date_range[0].month != date_range[1].month or date_range[0].year != date_range[1].year:
                    period_str = f"{date_range[0].strftime('%b %d')} – {date_range[1].strftime('%b %d, %Y')} • {span} days"

                card3 = render_kpi_card(
                    total_title,
                    f"{total_volume:,.0f} {avg_suffix}",
                    f'<div style="font-size:0.8rem; opacity:0.8; margin-top:2px;">{period_str}</div>',
                    badge_text,
                    state_badge,
                    "Sum of all volumes across the selected time window and filters. The pill compares this against the period’s total theoretical capacity."
                )

                # --- KPI Card 4: Demand Consistency ---
                label_cons = "Consistent" if cv_bucket < 30 else ("Variable" if cv_bucket < 50 else "Highly Variable")
                badge_cons = "badge-good" if cv_bucket < 30 else ("badge-fair" if cv_bucket < 50 else "badge-poor")
                consistency_val = max(0, 100 - cv_bucket)
                card4 = render_kpi_card(
                    "DAILY DEMAND CONSISTENCY",
                    f"{consistency_val:.0f}%",
                    f"CV: {cv_bucket:.1f}%",
                    label_cons,
                    badge_cons,
                    "Measures day-to-day demand stability. Calculated as 100% minus the coefficient of variation (CV), with a minimum value of 0%. Higher values indicate more predictable daily demand."
                )

                # --- KPI Card 5: High Volume ---
                level_badge = (
                    "badge-critical" if risk_pct > 25 else
                    "badge-poor" if risk_pct > 15 else
                    "badge-fair" if risk_pct > 5 else
                    "badge-good"
                )
                
                label_plural = label + "s"
                card5_title = f"HIGH-VOLUME {label_plural.upper()}"
                
                details_html = ""
                if not high_dates_df.empty:
                    details_lines = []
                    for _, row in high_dates_df.head(30).iterrows():
                        dt = pd.to_datetime(row["local_datetime"])
                        if granularity == "Hourly":
                            d_str = dt.strftime('%b %d, %H:00')
                        elif granularity == "Daily":
                            d_str = dt.strftime('%b %d, %Y')
                        elif granularity == "Weekly":
                            d_str = f"Week of {dt.strftime('%b %d')}"
                        else:
                            d_str = dt.strftime('%b %Y')
                        v_str = f"{row['volume']:,.0f}"
                        details_lines.append(f"{d_str}: {v_str}")
                    
                    summary_text = f"View {high_periods} high-volume {label_plural}"
                    content_html = "<br>".join(details_lines)
                    if high_periods > 30:
                        content_html += "<br>..."
                    
                    details_html = f'<details style="font-size:0.75rem; margin-top:6px; color:inherit;"><summary style="cursor:pointer; text-decoration:underline; opacity:0.9;">{summary_text}</summary><div style="max-height:80px; overflow-y:auto; margin-top:4px; border-top:1px solid rgba(128,128,128,0.2); padding-top:4px; line-height:1.2;">{content_html}</div></details>'

                card5 = render_kpi_card(
                    card5_title,
                    f"{high_periods} {label_plural}",
                    f'Threshold: {display_threshold:,.0f} {avg_suffix}/{label}{details_html}',
                    f"{risk_pct:.1f}% of {label_plural}",
                    level_badge,
                    f"{threshold_help} Count and share of {label_plural} exceeding the threshold."
                )

                st.markdown(f'<div class="kpi-grid">{card1}{card2}{card3}{card4}{card5}</div>', unsafe_allow_html=True)
            # ---------- Visualizations ----------
            if chart_type == "Share (Pie)":
                if intersection == "All Intersections":
                    pie_df = raw.groupby("intersection_name", as_index=False)["volume"].sum()
                    pie_title = f"{mode_label} Volume Share by Intersection"
                    names = "intersection_name"
                else:
                    if "direction" in raw.columns:
                        pie_df = raw.groupby("direction", as_index=False)["volume"].sum()
                        pie_title = f"{mode_label} Volume Share by Direction — {intersection}"
                        names = "direction"
                    else:
                        pie_df = raw.groupby("intersection_name", as_index=False)["volume"].sum()
                        pie_title = f"{mode_label} Volume Share — {intersection}"
                        names = "intersection_name"

                show_turn_pie = (
                    ("turn_type" in raw.columns) and
                    (raw["turn_type"].nunique() > 1) and
                    (mode in ["Vehicles", "Combined (All Modes)"])
                )
                if show_turn_pie:
                    turn_df = raw.groupby("turn_type", as_index=False)["volume"].sum().sort_values("volume", ascending=False)

                if not pie_df.empty:
                    pie_df = pie_df.sort_values("volume", ascending=False)
                    pull = [0.06] + [0]*(len(pie_df)-1)
                    if show_turn_pie:
                        c1, c2 = st.columns(2)
                    else:
                        c1 = st.container()
                        c2 = None

                    with c1:
                        fig_pie = px.pie(
                            pie_df,
                            names=names,
                            values="volume",
                            title=pie_title,
                            hole=0.45,
                            template="simple_white",
                            color_discrete_sequence=px.colors.sequential.Blues,
                        )
                        fig_pie.update_traces(
                            sort=False,
                            pull=pull,
                            textposition="inside",
                            texttemplate="%{label}<br>%{percent} • %{value:,.0f}",
                            hovertemplate="<b>%{label}</b><br>Volume: %{value:,.0f}<br>Share: %{percent}<extra></extra>",
                            marker=dict(line=dict(color="white", width=1)),
                        )
                        fig_pie.update_layout(
                            margin=dict(l=10, r=10, t=50, b=10),
                            height=460,
                            legend=dict(orientation="v", yanchor="middle", y=0.5),
                        )
                        st.plotly_chart(
                            fig_pie,
                            use_container_width=True,
                            config=PLOTLY_CONFIG,
                            key=f"t4_pie_{mode}_{granularity}_{(intersection or 'all').replace(' ', '_').replace('+','plus')}"
                        )

                    if show_turn_pie and c2 is not None and not turn_df.empty:
                        with c2:
                            pull2 = [0.06] + [0]*(len(turn_df)-1)
                            fig_turn = px.pie(
                                turn_df,
                                names="turn_type",
                                values="volume",
                                title="Turn Volume Share",
                                hole=0.45,
                                template="simple_white",
                                color_discrete_sequence=px.colors.sequential.Oranges,
                            )
                            fig_turn.update_traces(
                                sort=False,
                                pull=pull2,
                                textposition="inside",
                                texttemplate="%{label}<br>%{percent} • %{value:,.0f}",
                                hovertemplate="<b>%{label}</b><br>Volume: %{value:,.0f}<br>Share: %{percent}<extra></extra>",
                                marker=dict(line=dict(color="white", width=1)),
                            )
                            fig_turn.update_layout(
                                margin=dict(l=10, r=10, t=50, b=10),
                                height=460,
                                legend=dict(orientation="v", yanchor="middle", y=0.5),
                            )
                            st.plotly_chart(
                                fig_turn,
                                use_container_width=True,
                                config=PLOTLY_CONFIG,
                                key=f"t4_turnpie_{granularity}_{(intersection or 'all').replace(' ', '_').replace('+','plus')}"
                            )
                else:
                    st.info("No data available to render the share (pie) view.")
            else:
                # Trend
                if len(analysis_df) > 1:
                    try:
                        cap_key = "Vehicles" if mode_label == "All Modes" else mode_label
                        CAP_VPH, HIGH_VPH = _mode_caps(cap_key)
                        dir_label_arg = "" if direction_filter == "All Directions" else direction_filter

                        # Trends and Distribution Charts
                        with perf.track("main charts"):
                            fig_trend, fig_box, fig_matrix, m_title, s_title, b_title, b_subtitle, x_title, x_subtitle = create_volume_charts(
                                raw_df=raw,
                                granularity=granularity,
                                cap_vph=CAP_VPH,
                                high_vph=HIGH_VPH,
                                mode_label=mode_label,
                                direction_label=direction_filter,
                                intersections=intersection,
                                date_range=date_range,
                                turn_filter=turn_filter,
                                top_k=20
                            )
                        if fig_trend:
                            with st.container(border=True):
                                st.markdown(f"<h3 style='margin-bottom:0px;'>{m_title}</h3>", unsafe_allow_html=True)
                                if s_title:
                                    st.markdown(f"<div style='font-size:16px; color:#666; margin-bottom:5px;'>{s_title}</div>", unsafe_allow_html=True)

                                st.plotly_chart(
                                    fig_trend,
                                    use_container_width=True,
                                    config=PLOTLY_CONFIG,
                                    key=f"t4_trend_{mode}_{granularity}_{(intersection or 'all').replace(' ', '_').replace('+','plus')}"
                                )
                        colA, colB = st.columns(2)
                        with colA:
                            if fig_box:
                                with st.container(border=True):
                                    st.markdown(f"<h5 style='margin-bottom:0px;'>{b_title}</h5>", unsafe_allow_html=True)
                                    if b_subtitle:
                                        st.markdown(f"<div style='font-size:14px; color:#666; margin-bottom:5px;'>{b_subtitle}</div>", unsafe_allow_html=True)
                                    st.plotly_chart(
                                        fig_box,
                                        use_container_width=True,
                                        config=PLOTLY_CONFIG,
                                        key=f"t4_box_{mode}_{granularity}_{(intersection or 'all').replace(' ', '_').replace('+','plus')}"
                                    )
                        with colB:
                            if fig_matrix:
                                with st.container(border=True):
                                    st.markdown(f"<h5 style='margin-bottom:0px;'>{x_title}</h5>", unsafe_allow_html=True)
                                    if x_subtitle:
                                        st.markdown(f"<div style='font-size:14px; color:#666; margin-bottom:5px;'>{x_subtitle}</div>", unsafe_allow_html=True)
                                    st.plotly_chart(
                                        fig_matrix,
                                        use_container_width=True,
                                        config=PLOTLY_CONFIG,
                                        key=f"t4_matrix_{mode}_{granularity}_{(intersection or 'all').replace(' ', '_').replace('+','plus')}"
                                    )
                    except Exception as e:
                        st.error(f"❌ Error creating volume charts: {e}")

            # ---------- Risk Analysis ----------
            with st.expander(f"⚠️ {risk_ctx['risk']['title']}", expanded=False):
                st.markdown(f"<div style='font-size:16px; color:#666; margin-bottom:15px;'>{risk_ctx['risk']['subtitle']}</div>", unsafe_allow_html=True)
                try:
                    cap_key = "Vehicles" if mode_label == "All Modes" else mode_label
                    CAP_VPH, _ = _mode_caps(cap_key)
                    
                    # Optimization: Use already computed bucketed_df
                    bucketed = bucketed_df.copy()
                    bucketed["per_hour_equiv"] = np.where(
                        bucketed["bucket_hours"] > 0,
                        bucketed["volume"] / bucketed["bucket_hours"],
                        np.nan
                    )

                    group_cols = ["intersection_name"]
                    if "direction" in bucketed.columns:
                        group_cols.append("direction")

                    g = bucketed.groupby(group_cols).agg(
                        volume_mean=("volume", "mean"),
                        volume_max=("volume", "max"),
                        volume_std=("volume", "std"),
                        volume_count=("volume", "count"),
                        hr_mean=("per_hour_equiv", "mean"),
                        hr_max=("per_hour_equiv", "max"),
                    ).reset_index()

                    g["Peak_Capacity_Util"] = (g["hr_max"] / CAP_VPH * 100).round(1)
                    g["Avg_Capacity_Util"] = (g["hr_mean"] / CAP_VPH * 100).round(1)
                    g["Volume_Variability"] = (g["volume_std"] / g["volume_mean"] * 100).replace([np.inf, -np.inf], np.nan).fillna(0).round(1)
                    g["Peak_Avg_Ratio"] = (g["volume_max"] / g["volume_mean"]).replace([np.inf, -np.inf], 0).fillna(0).round(2)

                    g["🚨 Risk Score"] = (0.5*g["Peak_Capacity_Util"] + 0.3*g["Avg_Capacity_Util"] + 0.2*(g["Peak_Avg_Ratio"]*10)).round(1)
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

                    unit = _mode_noun(mode_label)
                    label = AGG_META[granularity]["label"]

                    cols = ["intersection_name"]
                    if "direction" in g.columns:
                        cols.append("direction")
                    cols += [
                        "⚠️ Risk Level","🎯 Action Priority","🚨 Risk Score",
                        "Peak_Capacity_Util","Avg_Capacity_Util",
                        "volume_mean","volume_max","Peak_Avg_Ratio","volume_count",
                    ]

                    final = g[cols].rename(columns={
                        "intersection_name": "Intersection",
                        **({"direction": "Dir"} if "direction" in g.columns else {}),
                        "Peak_Capacity_Util": "📊 Peak Capacity %",
                        "Avg_Capacity_Util": "📊 Avg Capacity %",
                        "volume_mean": f"Avg {mode_label} ({unit})",
                        "volume_max": f"Peak {mode_label} ({unit})",
                        "volume_count": f"{label.capitalize()}s",
                    }).sort_values("🚨 Risk Score", ascending=False)
                    
                    # Optimization: Use mapping for formatting display names
                    unique_ints = final["Intersection"].unique()
                    disp_map = {n: format_intersection_display(n) for n in unique_ints}
                    final["Intersection"] = final["Intersection"].map(disp_map)

                    st.dataframe(
                        final.head(20),
                        use_container_width=True,
                        column_config={
                            "🚨 Risk Score": st.column_config.NumberColumn("🚨 Capacity Risk Score", help="Composite of peak/avg utilization and peaking severity.", format="%.1f", min_value=0, max_value=120),
                            "📊 Peak Capacity %": st.column_config.NumberColumn("📊 Peak Capacity %", format="%.1f%%"),
                            "📊 Avg Capacity %": st.column_config.NumberColumn("📊 Avg Capacity %", format="%.1f%%"),
                        },
                    )

                    st.download_button(
                        "⬇️ Download Full Risk Analysis Table (CSV)",
                        data=convert_df_to_csv(final),
                        file_name=f"vantage_{mode.lower().replace(' ','_')}_risk.csv",
                        mime="text/csv",
                        key="vantage_risk_download"
                    )
                except Exception as e:
                    st.error(f"❌ Error in risk analysis: {e}")

            # ---------- Cycle length ----------
            st.subheader("Cycle Length Recommendations for CVAG")
            with perf.track("cycle length section"):
                try:
                    # Optimization: Use already computed bucketed_df
                    cycle_bucketed = bucketed_df.copy()
                    cycle_bucketed["total_volume"] = np.where(
                        cycle_bucketed["bucket_hours"] > 0,
                        cycle_bucketed["volume"] / cycle_bucketed["bucket_hours"],
                        cycle_bucketed["volume"]
                    )
                    st.caption("Note: Using hourly-equivalent demand (bucket total ÷ hours in bucket) for cycle estimation.")
                    render_cycle_length_section(cycle_bucketed, key_prefix="vantage_cl")
                except Exception as e:
                    st.error(f"❌ Error rendering cycle length section: {e}")

    except Exception as e:
        st.error(f"❌ Error processing VantageLive data: {e}")
        import traceback
        st.text("Debug info:")
        st.text(traceback.format_exc())

    perf.report()
