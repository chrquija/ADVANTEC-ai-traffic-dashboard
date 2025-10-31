# VantageLivetab.py - Tab 4: Iteris VantageLive Analysis (Bikes + Vehicles + Pedestrians)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
from Map import build_intersections_overview

# Shared UI utils (scoped loader and tab highlight)
try:
    from ui_utils import cad_loader as scoped_cad_loader, set_active_search_tab, is_active_tab
except ModuleNotFoundError:
    from core.ui_utils import cad_loader as scoped_cad_loader, set_active_search_tab, is_active_tab

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
    "Washington and Point Happy Way",
]

# Map label aliases to match the map’s internal labels
MAP_LABEL_ALIASES = {
    "Washington and Avenue 52": "Washington St & Avenue52",
    "Washington and Calle Tampico": "Washington St & Calle Tampico",
    "Washington and Village Shopping Center": "Washington St & Village Shop Ctr",
    "Washington and Avenue 50": "Washington St & Avenue50",
    "Washington and Sagebrush Avenue": "Washington St & Sagebrush Ave",
    "Washington and Eisenhower Drive": "Washington St & Eisenhower Dr",
    "Washington and Avenue 48": "Washington St & Avenue48",
    "Washington and Avenue 47": "Washington St & Avenue47",
    "Washington and Point Happy Way": "Washington St & Point Happy Way",
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

    "Washington Street & Point Happy Way": "Washington and Point Happy Way",
    "Washington Street and Point Happy Way": "Washington and Point Happy Way",
    "Washington St & Point Happy Way": "Washington and Point Happy Way",
}

# Aggregation metadata
AGG_META = {
    "Hourly": {"unit": "vph", "bucket": "H", "label": "hour", "fixed_hours": 1},
    "Daily": {"unit": "vpd", "bucket": "D", "label": "day", "fixed_hours": 24},
    "Weekly": {"unit": "vpw", "bucket": "W", "label": "week", "fixed_hours": 24 * 7},
    "Monthly": {"unit": "vpm", "bucket": "M", "label": "month", "fixed_hours": None},
}

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "toggleSpikelines"]
}
MAP_HEIGHT = 900

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
    return MAP_LABEL_ALIASES.get(name, name)

def _mode_caps(mode_label: str):
    """Return (capacity_vph, high_threshold_vph) by mode."""
    if mode_label == "Bikes":
        return BIKE_CAPACITY_VPH, BIKE_HIGH_THRESHOLD_VPH
    if mode_label == "Pedestrians":
        return PED_CAPACITY_VPH, PED_HIGH_THRESHOLD_VPH
    return VEH_CAPACITY_VPH, VEH_HIGH_THRESHOLD_VPH

def _mode_noun(mode_label: str) -> str:
    return "vehicles" if mode_label == "Vehicles" else ("bikes" if mode_label == "Bikes" else "pedestrians")

def _mode_emoji(mode_label: str) -> str:
    return "🚗" if mode_label == "Vehicles" else ("🚲" if mode_label == "Bikes" else ("🚶" if mode_label == "Pedestrians" else "📊"))

# =========================
# Data Loading
# =========================
@st.cache_data(show_spinner=False)
def load_vantage_bikes() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/Iteris_VantageLive/WashingtonStreet_ALL_Bikes.csv"
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
    url = "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/Iteris_VantageLive/WashingtonStreet_ALL_vehicles.csv"
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
    url = "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/Iteris_VantageLive/WashingtonStreet_ALL_pedestrians.csv"
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

def _cap_series_for_x(x_df: pd.DataFrame, cap_vph: float, high_vph: float) -> pd.DataFrame:
    xs = x_df[["local_datetime", "bucket_hours"]].drop_duplicates().sort_values("local_datetime")
    xs["capacity"] = xs["bucket_hours"] * float(cap_vph)
    xs["high"] = xs["bucket_hours"] * float(high_vph)
    return xs

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

def _tmc_figure(tmc_df: pd.DataFrame, intersection: str, date_range, mode_label: str) -> go.Figure:
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

    fig.update_xaxes(visible=False, range=[0, 10])
    fig.update_yaxes(visible=False, range=[0, 10])
    fig.update_layout(
        height=650,
        margin=dict(l=20, r=20, t=80, b=30),
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
        x=5.0, y=9.98, text="SOUTHBOUND", showarrow=False, xanchor="center",
        font=dict(size=14, color="#2f2f2f"),
        bgcolor="rgba(255,255,255,0.95)", bordercolor="#cfcfcf", borderwidth=1, borderpad=4
    )
    fig.add_annotation(
        x=5.0, y=9.82, text=f"({int(tot_map.get('SB',0)):,} total {noun})", showarrow=False, xanchor="center",
        font=dict(size=11, color="#666"),
        bgcolor="rgba(255,255,255,0.90)", bordercolor="#e5e5e5", borderwidth=0.5, borderpad=2
    )

    # Northbound (bottom): push lower
    fig.add_annotation(
        x=5.0, y=0.02, text="NORTHBOUND", showarrow=False, xanchor="center",
        font=dict(size=14, color="#2f2f2f"),
        bgcolor="rgba(255,255,255,0.95)", bordercolor="#cfcfcf", borderwidth=1, borderpad=4
    )
    fig.add_annotation(
        x=5.0, y=0.22, text=f"({int(tot_map.get('NB',0)):,} total {noun})", showarrow=False, xanchor="center",
        font=dict(size=11, color="#666"),
        bgcolor="rgba(255,255,255,0.90)", bordercolor="#e5e5e5", borderwidth=0.5, borderpad=2
    )

    # Eastbound (left side): move farther left & keep vertical text
    fig.add_annotation(
        x=0.20, y=5.0, text="EASTBOUND", textangle=-90, showarrow=False, xanchor="center",
        font=dict(size=14, color="#2f2f2f"),
        bgcolor="rgba(255,255,255,0.95)", bordercolor="#cfcfcf", borderwidth=1, borderpad=4
    )
    fig.add_annotation(
        x=0.38, y=5.0, text=f"({int(tot_map.get('EB',0)):,} total {noun})", textangle=-90,
        showarrow=False, xanchor="center", font=dict(size=11, color="#666"),
        bgcolor="rgba(255,255,255,0.90)", bordercolor="#e5e5e5", borderwidth=0.5, borderpad=2
    )

    # Westbound (right side): move farther right & keep vertical text
    fig.add_annotation(
        x=9.80, y=5.0, text="WESTBOUND", textangle=90, showarrow=False, xanchor="center",
        font=dict(size=14, color="#2f2f2f"),
        bgcolor="rgba(255,255,255,0.95)", bordercolor="#cfcfcf", borderwidth=1, borderpad=4
    )
    fig.add_annotation(
        x=9.62, y=5.0, text=f"({int(tot_map.get('WB',0)):,} total {noun})", textangle=90,
        showarrow=False, xanchor="center", font=dict(size=11, color="#666"),
        bgcolor="rgba(255,255,255,0.90)", bordercolor="#e5e5e5", borderwidth=0.5, borderpad=2
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
                x=ax, y=ay, text=f"({v:,})",
                showarrow=False, font=dict(size=11, color="#777"),
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
def create_volume_charts(
    raw_df: pd.DataFrame,
    granularity: str,
    cap_vph: float,
    high_vph: float,
    mode_label: str = "Vehicles",
    top_k: int = 10
):
    if raw_df.empty:
        return None, None, None

    agg_all = _prep_bucket(raw_df, granularity)
    if agg_all.empty:
        return None, None, None

    agg_for_plot = (
        agg_all.groupby(["local_datetime", "intersection_name"], as_index=False)["volume"]
        .sum()
    )

    order = agg_for_plot.groupby("intersection_name")["volume"].mean().sort_values(ascending=False)
    keep = order.index[:max(1, min(top_k, len(order)))]

    plot_df = agg_for_plot[agg_for_plot["intersection_name"].isin(keep)].copy().sort_values("local_datetime")
    unit = AGG_META[granularity]["unit"]
    label = AGG_META[granularity]["label"]

    fig_trend = go.Figure()
    mode = "lines" if granularity == "Hourly" else "lines+markers"
    xfmt = "%Y-%m-%d %H:%M" if granularity == "Hourly" else "%Y-%m-%d"

    for name, g in plot_df.groupby("intersection_name"):
        fig_trend.add_trace(
            go.Scatter(
                x=g["local_datetime"],
                y=g["volume"],
                mode=mode,
                name=name,
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
    fig_trend.update_layout(
        title=f"{mode_label} Volume Trends - {granularity}",
        xaxis_title="Date/Time",
        yaxis_title=f"{mode_label} Volume ({unit})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_white",
    )

    cat_order = order[order.index.isin(keep)].index.tolist()
    fig_box = px.box(
        plot_df, x="intersection_name", y="volume",
        category_orders={"intersection_name": cat_order},
        points=False,
        title=f"{mode_label} Volume Distribution by Intersection — {granularity}"
    )
    fig_box.update_layout(
        xaxis_title="Intersection",
        yaxis_title=f"{mode_label} Volume per {label} ({unit})",
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_white",
    )

    mat = (
        plot_df.groupby("intersection_name", as_index=False)["volume"]
        .mean()
        .rename(columns={"volume": f"Avg {label} Volume"})
    )
    mat["Rank"] = mat[f"Avg {label} Volume"].rank(ascending=False, method="dense").astype(int)
    mat = mat.sort_values("Rank")
    fig_matrix = px.bar(
        mat, y="intersection_name", x=f"Avg {label} Volume",
        orientation="h", text=f"Avg {label} Volume",
        title=f"Average {label.capitalize()} {mode_label} Volume by Intersection"
    )
    fig_matrix.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
    fig_matrix.update_layout(
        xaxis_title=f"Average {label} volume ({unit})",
        yaxis_title="",
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_white",
    )
    return fig_trend, fig_box, fig_matrix

# =========================
# Main Renderer
# =========================
def render_vantage_tab():
    bikes_df = load_vantage_bikes()
    vehicles_df = load_vantage_vehicles()
    peds_df = load_vantage_pedestrians()

    # -------- Sidebar controls --------
    with st.sidebar:
        with st.expander("⚙️ Pg.4 ITERIS VANTAGE LIVE SETTINGS", expanded=False):
            active_t4 = is_active_tab("t4")
            if active_t4:
                st.markdown(
                    """
                    <div style="
                        background: linear-gradient(90deg, #ffe58f, #ffd666);
                        border: 1px solid #fadb14; color: #613400;
                        padding: 6px 10px; border-radius: 8px; font-weight: 700; margin-bottom: 6px;">
                        • You’re viewing: Pg.4 ITERIS VANTAGE LIVE
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.caption("Select Mode, Intersection(s) and Date Range")
            st.caption("Data: Bike, Vehicle, and Pedestrian Volume from Iteris VantageLive")

            st.markdown("## 🚲 Select Mode")
            mode = st.selectbox(
                "Analysis Mode",
                ["Vehicles", "Bikes", "Pedestrians", "Combined (All Modes)"],
                key="vantage_mode",
            )

            st.markdown("## 🚦 Select Intersection")
            intersection = st.selectbox(
                "Intersection",
                ["All Intersections"] + CANONICAL_INTERSECTIONS,
                key="vantage_intersection",
            )

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

                avail = compute_data_availability(
                    base_df if base_df is not None else pd.DataFrame(),
                    intersection_col="intersection_name",
                    intersection=intersection,
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
                        header_label = f"Available Data for {intersection}"
                    st.markdown(f"**{header_label}:**")
                    st.markdown(f"- Date Range: {start_str} → {end_str} {size_str}")
                    tail_gap = avail.get("tail_gap")
                    if tail_gap:
                        st.markdown("- Missing Data: " + tail_gap)
                    else:
                        gaps = avail.get("gaps") or []
                        if len(gaps) == 0:
                            st.markdown("- Missing Data: None")
                        else:
                            st.markdown("- Missing Data: " + "; ".join(gaps))
            except Exception:
                pass

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

            st.markdown("## 📅 Date And Time")
            date_range = date_range_preset_controls(min_date, max_date, key_prefix="vantage")

            st.markdown("## Granularity")
            granularity = st.selectbox(
                "Data Aggregation",
                ["Hourly", "Daily", "Weekly", "Monthly"],
                index=1,  # Daily by default
                key="granularity_vantage",
            )

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

            st.markdown("## 📊 Chart Type")
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

            if st.button("🔍 **Search**", key="search_vantage", type="primary", use_container_width=True):
                st.session_state["vantage_params"] = vantage_current
                st.session_state["vantage_ready"] = True
                set_active_search_tab("t4")
                st.session_state["last_active_tab"] = "t4"

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
        st.warning("⚙️ Press **Search** to refresh.")

    if not date_range or len(date_range) != 2:
        st.warning("⚠️ Please select both start and end dates to proceed.")
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
        content_col, right_col = st.columns([7, 3.5], gap="large")
        with right_col:
            st.markdown('<div id="vantage-map-anchor"></div>', unsafe_allow_html=True)
            st.markdown("##### Corridor Map", help="Stays visible while you scroll the analysis on the left.")
            try:
                selected_map_label = None if intersection == "All Intersections" else _map_label_for(intersection)
                fig_map = build_intersections_overview(selected_label=selected_map_label)
            except Exception:
                fig_map = None
            if fig_map:
                try:
                    fig_map.update_layout(height=MAP_HEIGHT, margin=dict(l=0, r=0, t=32, b=0))
                except Exception:
                    pass
                st.markdown('<div class="cvag-map-card">', unsafe_allow_html=True)
                map_key = f"t4_map_{(selected_map_label or 'all').replace(' ', '_').replace('+','plus')}"
                st.plotly_chart(fig_map, use_container_width=True, config=PLOTLY_CONFIG, key=map_key)
                if intersection != "All Intersections":
                    st.caption(f"Selected: **{intersection}**")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="cvag-map-card">', unsafe_allow_html=True)
                st.caption("Map: Washington Street Corridor")
                st.markdown('</div>', unsafe_allow_html=True)

        # ---------- Main content ----------
        with content_col:
            span = (date_range[1] - date_range[0]).days + 1
            total_obs = len(analysis_df)
            header_title = (
                f"Iteris Vantagelive {mode_label} Volume Analysis: {intersection}"
                if intersection != "All Intersections"
                else f"Iteris Vantagelive {mode_label} Volume Analysis: Washington Street Corridor"
            )
            emoji = _mode_emoji("Vehicles" if mode_label == "All Modes" else mode_label)
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #2b77e5 0%, #19c3e6 100%);
                    border-radius:16px; padding:18px 20px; color:#fff; margin:8px 0 14px;
                    box-shadow:0 10px 26px rgba(25,115,210,.25); text-align:left;">
                  <div style="display:flex; align-items:center; gap:10px;">
                    <div style="width:36px;height:36px;border-radius:10px;background:rgba(255,255,255,.18);
                                display:flex;align-items:center;justify-content:center;
                                box-shadow:inset 0 0 0 1px rgba(255,255,255,.15);">{emoji}</div>
                    <div style="font-size:1.9rem;font-weight:800; letter-spacing:.2px;">
                      {header_title}
                    </div>
                  </div>
                  <div style="margin-top:10px;display:flex;flex-direction:column;gap:6px;">
                    <div>📅 {date_range[0].strftime('%b %d, %Y')} to {date_range[1].strftime('%b %d, %Y')} ({span} days) • {granularity} Aggregation</div>
                    <div>✅ {total_obs:,} observations • Direction: {direction_filter}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ============= TMC VIEW =============
            if chart_type == "TMC (Turning Movement Counts)":
                if intersection == "All Intersections":
                    st.info("ℹ️ TMC requires a single intersection. Please pick one in the sidebar.")
                    return
                if mode == "Combined (All Modes)":
                    st.info("ℹ️ TMC is designed for a single mode. Please select **Vehicles**, **Bikes**, or **Pedestrians**.")
                    return

                base_df = _filter_for_tmc(
                    vehicles_df if mode == "Vehicles" else (bikes_df if mode == "Bikes" else peds_df),
                    intersection, date_range
                )
                if base_df.empty:
                    st.warning("No data available for the selected dates and intersection.")
                    return

                tmc_table = _summarize_tmc(base_df)
                fig_tmc = _tmc_figure(tmc_table, intersection, date_range, "Vehicles" if mode=="Vehicles" else ("Bikes" if mode=="Bikes" else "Pedestrians"))
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
                    noun = _mode_noun("Vehicles" if mode=="Vehicles" else ("Bikes" if mode=="Bikes" else "Pedestrians"))
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
                        "⬇️ Download TMC Summary (CSV)",
                        data=show.to_csv(index=False).encode("utf-8"),
                        file_name=f"tmc_{mode.lower()}_{intersection.replace(' ','_')}_{date_range[0]}_{date_range[1]}.csv",
                        mime="text/csv",
                    )
                return  # end TMC view

            # ---------- KPIs ----------
            raw = analysis_df.copy()
            raw["volume"] = pd.to_numeric(raw["volume"], errors="coerce")
            raw["local_datetime"] = pd.to_datetime(raw["local_datetime"])

            st.subheader(f" {mode_label} Demand Performance Indicators")
            if not raw.empty and raw["volume"].notna().any():
                bucket_all = _prep_bucket(raw, granularity).groupby("local_datetime", as_index=False)["volume"].sum().sort_values("local_datetime")
                if granularity == "Monthly":
                    bucket_all["bucket_hours"] = pd.to_datetime(bucket_all["local_datetime"]).dt.days_in_month * 24
                else:
                    bucket_all["bucket_hours"] = AGG_META[granularity]["fixed_hours"]

                # choose capacity by mode
                cap_key = "Vehicles" if mode_label == "All Modes" else mode_label
                CAP_VPH, HIGH_VPH = _mode_caps(cap_key)

                bucket_all["cap"] = bucket_all["bucket_hours"] * CAP_VPH
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

                unit = AGG_META[granularity]["unit"]
                label = AGG_META[granularity]["label"]
                bucket_all["threshold"] = bucket_all["bucket_hours"] * HIGH_VPH
                high_periods = int((bucket_all["volume"] > bucket_all["threshold"]).sum())
                total_periods = int(len(bucket_all))
                risk_pct = (high_periods / total_periods * 100) if total_periods > 0 else 0.0

                if granularity == "Hourly":
                    avg_label = f"Average Hourly {mode_label}"
                    peak_label = f"🔥 Peak Hourly {mode_label}"
                    avg_suffix = "vph"
                    display_threshold = HIGH_VPH
                    threshold_help = f"High-Volume Threshold: > {display_threshold:,} vph."
                elif granularity == "Daily":
                    avg_label = f"Average Daily {mode_label}"
                    peak_label = f"🔥 Peak Daily {mode_label}"
                    avg_suffix = "vpd"
                    display_threshold = HIGH_VPH * 24
                    threshold_help = f"High-Volume Threshold (daily): > {display_threshold:,} vpd (scaled from {HIGH_VPH:,} vph × 24h)."
                elif granularity == "Weekly":
                    avg_label = f"Average Weekly {mode_label}"
                    peak_label = f"🔥 Peak Weekly {mode_label}"
                    avg_suffix = "vpw"
                    display_threshold = HIGH_VPH * 24 * 7
                    threshold_help = f"High-Volume Threshold (weekly): > {display_threshold:,} vpw (scaled from {HIGH_VPH:,} vph × 168h)."
                else:
                    avg_label = f"Average Monthly {mode_label}"
                    peak_label = f"🔥 Peak Monthly {mode_label}"
                    avg_suffix = "vpm"
                    display_threshold = HIGH_VPH * 24 * 30
                    threshold_help = f"High-Volume Threshold (monthly): > ~{display_threshold:,} vpm (scaled from {HIGH_VPH:,} vph × hours in month)."

                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    kpi_title(
                        peak_label,
                        f"Highest {label} total within the selected period. Date shown below; 95th percentile is provided for context."
                    )
                    st.metric("", f"{peak_val:,.0f} {unit}", delta=f"on {peak_date.strftime('%b %d, %Y')}")
                    badge = (
                        "badge-critical" if peak_util_pct > 90 else
                        "badge-poor" if peak_util_pct > 75 else
                        "badge-fair" if peak_util_pct > 60 else
                        "badge-good"
                    )
                    st.markdown(
                        f'<span class="performance-badge {badge}">{peak_util_pct:.0f}% of Capacity</span>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"95th percentile: {p95_val:,.0f} {unit}")

                with col2:
                    kpi_title(
                        f"📊 {avg_label}",
                        f"Mean of {label} totals across the selected period (after all filters). Delta shows average utilization vs capacity."
                    )
                    st.metric("", f"{avg_bucket_val:,.0f} {avg_suffix}", delta=f"{avg_util_pct:.0f}% Avg Util")

                with col3:
                    total_volume = float(np.nansum(raw["volume"]))
                    cap_total = CAP_VPH * float(bucket_all["bucket_hours"].sum())
                    kpi_title(
                        f"🚗 Total {mode_label} (period)",
                        "Sum of all volumes across the selected time window and filters. The pill compares this against the period’s total theoretical capacity."
                    )
                    st.metric("", f"{total_volume:,.0f}")
                    if cap_total > 0:
                        ratio = total_volume / cap_total
                        state_badge = "badge-good" if ratio < 0.40 else ("badge-fair" if ratio < 0.70 else "badge-poor")
                    else:
                        state_badge = "badge-good"
                    st.markdown(
                        f'<span class="performance-badge {state_badge}">vs. period capacity</span>',
                        unsafe_allow_html=True,
                    )

                with col4:
                    kpi_title(
                        "🎯 Demand Consistency",
                        "Consistency is 100 − CV%, where CV is coefficient of variation of bucket totals (std ÷ mean). Higher → more consistent."
                    )
                    st.metric("", f"{max(0, 100 - cv_bucket):.0f}%", delta=f"CV: {cv_bucket:.1f}%")
                    label_cons = "Consistent" if cv_bucket < 30 else ("Variable" if cv_bucket < 50 else "Highly Variable")
                    badge_cons = "badge-good" if cv_bucket < 30 else ("badge-fair" if cv_bucket < 50 else "badge-poor")
                    st.markdown(
                        f'<span class="performance-badge {badge_cons}">{label_cons}</span>',
                        unsafe_allow_html=True,
                    )

                with col5:
                    kpi_title(
                        f"⚠️ High Volume {label.capitalize()}s",
                        f"{threshold_help} Count and share of {label}s exceeding the threshold."
                    )
                    st.metric("", f"{high_periods}", delta=f"{risk_pct:.1f}% of {label}s")
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

            # ---------- Visualizations ----------
            st.subheader(f"📈 {mode_label} Volume Visualizations")

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
                        fig_trend, fig_box, fig_matrix = create_volume_charts(
                            raw_df=raw,
                            granularity=granularity,
                            cap_vph=CAP_VPH,
                            high_vph=HIGH_VPH,
                            mode_label=mode_label,
                        )
                        if fig_trend:
                            st.plotly_chart(
                                fig_trend,
                                use_container_width=True,
                                config=PLOTLY_CONFIG,
                                key=f"t4_trend_{mode}_{granularity}_{(intersection or 'all').replace(' ', '_').replace('+','plus')}"
                            )
                        colA, colB = st.columns(2)
                        with colA:
                            if fig_box:
                                st.plotly_chart(
                                    fig_box,
                                    use_container_width=True,
                                    config=PLOTLY_CONFIG,
                                    key=f"t4_box_{mode}_{granularity}_{(intersection or 'all').replace(' ', '_').replace('+','plus')}"
                                )
                        with colB:
                            if fig_matrix:
                                st.plotly_chart(
                                    fig_matrix,
                                    use_container_width=True,
                                    config=PLOTLY_CONFIG,
                                    key=f"t4_matrix_{mode}_{granularity}_{(intersection or 'all').replace(' ', '_').replace('+','plus')}"
                                )
                    except Exception as e:
                        st.error(f"❌ Error creating volume charts: {e}")

            # ---------- Risk table ----------
            st.subheader(f"🚨 Intersection Volume & Capacity Risk Analysis ({mode_label})")
            try:
                cap_key = "Vehicles" if mode_label == "All Modes" else mode_label
                CAP_VPH, _ = _mode_caps(cap_key)
                bucketed = _prep_bucket(raw, granularity)
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

                unit = AGG_META[granularity]["unit"]
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

                st.dataframe(
                    final.head(15),
                    use_container_width=True,
                    column_config={
                        "🚨 Risk Score": st.column_config.NumberColumn("🚨 Capacity Risk Score", help="Composite of peak/avg utilization and peaking severity.", format="%.1f", min_value=0, max_value=120),
                        "📊 Peak Capacity %": st.column_config.NumberColumn("📊 Peak Capacity %", format="%.1f%%"),
                        "📊 Avg Capacity %": st.column_config.NumberColumn("📊 Avg Capacity %", format="%.1f%%"),
                    },
                )

                st.download_button(
                    "⬇️ Download Risk Analysis Table (CSV)",
                    data=final.to_csv(index=False).encode("utf-8"),
                    file_name=f"vantage_{mode.lower().replace(' ','_')}_risk.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"❌ Error in risk analysis: {e}")

            # ---------- Cycle length ----------
            st.subheader("🔄 Cycle Length Recommendations for CVAG")
            try:
                cycle_bucketed = _prep_bucket(raw, granularity)
                cycle_bucketed["total_volume"] = np.where(
                    cycle_bucketed["bucket_hours"] > 0,
                    cycle_bucketed["volume"] / cycle_bucketed["bucket_hours"],
                    cycle_bucketed["volume"]
                )
                st.caption("Note: Using hourly-equivalent demand (bucket total ÷ hours in bucket) for cycle estimation.")
                render_cycle_length_section(cycle_bucketed)
            except Exception as e:
                st.error(f"❌ Error rendering cycle length section: {e}")

    except Exception as e:
        st.error(f"❌ Error processing VantageLive data: {e}")
        import traceback
        st.text("Debug info:")
        st.text(traceback.format_exc())
