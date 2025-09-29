# Python
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Plotly for chart helpers
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add the Map import
from Map import build_all_segments_overview, build_intersections_overview


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _fix_raw_url(url: str) -> str:
    """
    GitHub RAW URLs must be:
      https://raw.githubusercontent.com/<user>/<repo>/<branch>/<path>
    Some links used '/refs/heads/main/'. This converts them.
    """
    return url.replace("/refs/heads/", "/")


def _safe_to_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _normalize_acyclica_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tolerant column normalizer for Acyclica CSVs.
    """
    if df is None or df.empty:
        return df
    lowmap = {c: "".join(str(c).strip().split()).lower() for c in df.columns}
    df = df.rename(columns=lowmap)
    canon = {
        "localdatetime": "local_datetime",
        "corridorid": "corridor_id",
        "direction": "direction",
        "metric": "metric",
        "strength": "Strength",
        "firsts": "Firsts",
        "lasts": "Lasts",
        "minimum": "Minimum",
        "maximum": "Maximum",
    }
    for src, tgt in canon.items():
        if src in df.columns:
            df = df.rename(columns={src: tgt})
    return df


# =========================
# Data loading
# =========================

# Iteris ClearGuide Data
@st.cache_data
def load_traffic_data():
    """
    Load and combine all corridor traffic data from GitHub (Iteris-style).
    Auto-fixes bad RAW URL pattern.
    """
    data_sources = {
        "Avenue 52 → Calle Tampico": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/1_2_LONG_NSB_Ave52_CalleTampico_WashSt_1hr_septojuly.csv",
        "Calle Tampico → Village Shopping Ctr": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/2_3_LONG_NSB_CalleTampico_VillageShoppingCtr_WashSt_1hr_septojuly.csv",
        "Village Shopping Ctr → Avenue 50": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/3_4_LONG_NSB_VillageShoppingCtr_Avenue50_WashSt_1hr_septojuly.csv",
        "Avenue 50 → Sagebrush Ave": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/4_5_LONG_NSB_Ave50_SagebrushAve_WashSt_1hr_septojuly.csv",
        "Sagebrush Ave → Eisenhower Dr": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/5_6_LONG_NSB_SagebrushAve_EisenhowerDr_WashSt_1hr_septojuly.csv",
        "Eisenhower Dr → Avenue 48": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/6_7_LONG_NSB_EisenhowerDr_Avenue48_WashSt_1hr_septojuly.csv",
        "Avenue 48 → Avenue 47": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/7_8_LONG_NSB_Ave48_Ave47_WashSt_1hr_septojuly.csv",
        "Avenue 47 → Point Happy Simon": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/8_9_LONG_NSB_Ave47_PointHappySimon_WashSt_1hr_septojuly.csv",
        "Point Happy Simon → Hwy 111": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/9_10_LONG_NSB_PointHappySimon_WashSt_1hr_septojuly.csv",
    }

    all_data = []
    for segment_name, url in data_sources.items():
        url = _fix_raw_url(url)
        try:
            df = pd.read_csv(url)
            df["segment_name"] = segment_name
            all_data.append(df)
        except Exception as e:
            st.error(f"Error loading {segment_name}: {e}")

    if not all_data:
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df["local_datetime"] = pd.to_datetime(combined_df["local_datetime"], errors="coerce")
    combined_df = combined_df.dropna(subset=["local_datetime"]).sort_values("local_datetime").reset_index(drop=True)
    return combined_df


# Kinetic mobility data
@st.cache_data
def load_volume_data():
    """
    Load consolidated volume data for all Washington Street intersections.
    Auto-fixes bad RAW URL pattern.
    """
    volume_url = _fix_raw_url(
        "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/VOLUME/KMOB_LONG/LONG_MASTER_Avenue52_to_Avenue47_1hr_NS_VOLUME_OctoberTOJune.csv"
    )

    try:
        volume_df = pd.read_csv(volume_url)
        volume_df["local_datetime"] = pd.to_datetime(volume_df["local_datetime"], errors="coerce")
        volume_df = volume_df.dropna(subset=["local_datetime"]).sort_values("local_datetime").reset_index(drop=True)

        # Create proper intersection names from intersection_id
        volume_df["intersection_name"] = (
            volume_df["intersection_id"]
            .str.replace("_", " ", regex=False)
            .str.replace("Washington St and ", "Washington St & ", regex=False)
            .str.replace(" and ", " & ", regex=False)
        )

        # Create a sorting order for intersections (from south to north along Washington St)
        intersection_order = {
            "Washington St & Avenue52": 1,
            "Washington St & Calle Tampico": 2,
            "Washington St & Village Shop Ctr": 3,
            "Washington St & Avenue50": 4,
            "Washington St & Sagebrush Ave": 5,
            "Washington St & Eisenhower": 6,
            "Washington St & Ave48": 7,
            "Washington St & Ave47": 8,
        }

        volume_df["sort_order"] = volume_df["intersection_name"].map(intersection_order).fillna(999)
        volume_df = volume_df.sort_values("sort_order").drop("sort_order", axis=1)
        return volume_df

    except Exception as e:
        st.error(f"Error loading volume data: {e}")
        return pd.DataFrame()


# -------------------------
# Acyclica (Long + Wide)
# -------------------------
ACYCLICA_URL = _fix_raw_url(
    "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/MASTER_Acyclica_Traveltime_speed.csv"
)

@st.cache_data(show_spinner=False)
def load_acyclica_data() -> pd.DataFrame:
    """
    Load Acyclica travel time & speed in LONG format.
    Returns a cleaned DataFrame or raises RuntimeError on failure.
    Columns (normalized):
      local_datetime, corridor_id, direction, metric, Strength, Firsts, Lasts, Minimum, Maximum
    """
    try:
        df = pd.read_csv(ACYCLICA_URL)
    except Exception as e:
        raise RuntimeError(f"Error loading Acyclica data: {e}")

    if df is None or df.empty:
        raise RuntimeError("Acyclica CSV is empty.")

    # Normalize headers
    df = _normalize_acyclica_headers(df)

    required = ["local_datetime","corridor_id","direction","metric","Strength","Firsts","Lasts","Minimum","Maximum"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Acyclica CSV missing required columns: {', '.join(missing)}")

    # Types & cleanup
    df["local_datetime"] = pd.to_datetime(df["local_datetime"], errors="coerce")
    df = df.dropna(subset=["local_datetime"])
    for c in ["Strength","Firsts","Lasts","Minimum","Maximum"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Metric and direction normalization:
    # - remove spaces/underscores and non-letters
    # - lowercase to stable keys: 'speed', 'traveltime'
    df["metric"] = (
        df["metric"]
        .astype(str)
        .str.strip()
        .str.replace(r"[^A-Za-z]+", "", regex=True)
        .str.lower()
    )
    df["direction"] = df["direction"].astype(str).str.strip().str.upper()

    df = df.sort_values(["local_datetime","direction","metric"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def acyclica_long_to_hourly(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long → wide for KPI/plots.
    Output columns:
      local_datetime, corridor_id, direction, average_traveltime, average_speed
    """
    if df_long is None or df_long.empty:
        return pd.DataFrame()

    piv = (
        df_long.pivot_table(
            index=["local_datetime","corridor_id","direction"],
            columns="metric",
            values="Strength",
            aggfunc="mean",
        )
        .reset_index()
    )

    # Normalize pivoted column labels then rename to canonical names
    piv.columns = [str(c).lower() for c in piv.columns]
    piv = piv.rename(columns={"traveltime": "average_traveltime", "speed": "average_speed"})

    # Ensure presence
    for col in ["average_traveltime","average_speed"]:
        if col not in piv.columns:
            piv[col] = np.nan

    piv["local_datetime"] = pd.to_datetime(piv["local_datetime"], errors="coerce")
    piv = piv.dropna(subset=["local_datetime"]).sort_values(["local_datetime","direction"]).reset_index(drop=True)
    return piv


# =========================
# Small data getters
# =========================
@st.cache_data(show_spinner=False)
def get_corridor_df() -> pd.DataFrame:
    df = load_traffic_data()
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = _safe_to_datetime(df.copy(), "local_datetime")
    needed = {"segment_name", "average_delay", "average_traveltime", "average_speed", "direction"}
    missing = needed - set(df.columns)
    if missing:
        st.warning(f"Traffic dataset is missing columns: {', '.join(missing)}")
    return df


@st.cache_data(show_spinner=False)
def get_volume_df() -> pd.DataFrame:
    df = load_volume_data()
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = _safe_to_datetime(df.copy(), "local_datetime")
    needed = {"intersection_name", "total_volume", "direction"}
    missing = needed - set(df.columns)
    if missing:
        st.warning(f"Volume dataset is missing columns: {', '.join(missing)}")
    return df


@st.cache_data(show_spinner=False)
def get_acyclica_long_df() -> pd.DataFrame:
    """Long-format Acyclica (for Incident/Peak/Event detection)."""
    return load_acyclica_data()


@st.cache_data(show_spinner=False)
def get_acyclica_df() -> pd.DataFrame:
    """
    Wide-format Acyclica for KPI/plots:
      local_datetime, corridor_id, direction, average_traveltime, average_speed
    """
    long_df = load_acyclica_data()
    if long_df is None or len(long_df) == 0:
        return pd.DataFrame()
    wide = acyclica_long_to_hourly(long_df)
    return _safe_to_datetime(wide.copy(), "local_datetime")


def get_performance_rating(score: float):
    if score > 80:
        return " Excellent", "badge-excellent"
    if score > 60:
        return " Good", "badge-good"
    if score > 40:
        return " Fair", "badge-fair"
    if score > 20:
        return " Poor", "badge-poor"
    return " Critical", "badge-critical"


# =========================
# Interpretable KPI helpers (for non-Acyclica Delay/TT tabs)
# =========================
def _coerce_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def compute_perf_kpis_interpretable(df: pd.DataFrame, high_delay_threshold: float) -> dict:
    """
    Compute five interpretable KPIs for Iteris-style (wide) data.
    """
    if df is None or df.empty:
        return {
            "avg_tt": {"value": 0.0, "unit": "min", "score": 50.0, "help": "Average Travel Time"},
            "planning_time": {"value": 0.0, "unit": "min", "score": 50.0, "help": "Planning Time (95th)"},
            "buffer_index": {"value": 0.0, "unit": "%", "score": 50.0, "help": "Buffer Index"},
            "reliability": {"value": 0.0, "unit": "%", "score": 50.0, "help": "Reliability Index"},
            "congestion_freq": {"value": 0.0, "unit": "%", "score": 50.0, "help": "Congestion Frequency"},
        }

    for c in ("average_delay", "average_traveltime", "average_speed"):
        if c in df:
            df[c] = _coerce_num(df[c])

    avg_tt = float(np.nanmean(df["average_traveltime"])) if "average_traveltime" in df else 0.0

    if "average_traveltime" in df and df["average_traveltime"].notna().any():
        p95_tt = float(np.nanpercentile(df["average_traveltime"].dropna(), 95))
    else:
        p95_tt = 0.0

    buffer_index = ((p95_tt - avg_tt) / avg_tt * 100.0) if avg_tt > 0 else 0.0

    if avg_tt > 0 and "average_traveltime" in df:
        cv_tt = float(np.nanstd(df["average_traveltime"])) / avg_tt * 100.0
    else:
        cv_tt = 0.0
    reliability = max(0.0, 100.0 - cv_tt)

    if "average_delay" in df and df["average_delay"].notna().any():
        total_hours = int(df["average_delay"].count())
        cong_hours = int((df["average_delay"] > high_delay_threshold).sum())
        cong_freq = (cong_hours / total_hours * 100.0) if total_hours > 0 else 0.0
    else:
        cong_freq, cong_hours, total_hours = 0.0, 0, 0

    def _minmax_score(series: pd.Series, val: float) -> float:
        series = pd.to_numeric(series, errors="coerce").dropna()
        if len(series) < 2:
            return 50.0
        mn, mx = float(series.min()), float(series.max())
        if mx <= mn:
            return 50.0
        frac = (val - mn) / (mx - mn)  # lower is better
        return float(max(0.0, min(100.0, 100.0 * (1.0 - frac))))

    if "average_traveltime" in df and df["average_traveltime"].notna().any():
        score_avg_tt = _minmax_score(df["average_traveltime"], avg_tt)
        score_plan = _minmax_score(df["average_traveltime"], p95_tt)
    else:
        score_avg_tt = score_plan = 50.0

    score_buffer = float(max(0.0, 100.0 - min(max(buffer_index, 0.0), 100.0)))
    score_reliability = float(max(0.0, min(100.0, reliability)))
    score_congestion = float(max(0.0, min(100.0, 100.0 - cong_freq)))

    return {
        "avg_tt": {
            "value": avg_tt,
            "unit": "min",
            "score": score_avg_tt,
            "help": "Average Travel Time (mean of hourly trip times).",
        },
        "planning_time": {
            "value": p95_tt,
            "unit": "min",
            "score": score_plan,
            "help": "Planning Time (95th percentile).",
        },
        "buffer_index": {
            "value": buffer_index,
            "unit": "%",
            "score": score_buffer,
            "help": "Buffer Index = (P95 − mean) / mean × 100.",
        },
        "reliability": {
            "value": reliability,
            "unit": "%",
            "score": score_reliability,
            "help": "Reliability Index = 100 − CV%.",
        },
        "congestion_freq": {
            "value": cong_freq,
            "unit": "%",
            "score": score_congestion,
            "extra": f"Hours > {high_delay_threshold:.0f}s: {cong_hours}/{total_hours}",
            "help": "Share of hours with delay above the threshold.",
        },
    }


def render_badge(score: float) -> str:
    label, css = get_performance_rating(score)
    return f'<span class="performance-badge {css}">{label}</span>'


# =========================
# Chart helpers
# =========================
def performance_chart(data: pd.DataFrame, metric_type: str = "delay"):
    if data.empty:
        return None
    metric_type = metric_type.lower().strip()
    if metric_type == "delay":
        y_col, title, color = "average_delay", "Traffic Delay Analysis", "#e74c3c"
        y_label = "Average Delay (seconds)"
        dist_x_label = "Average Delay (seconds)"
    else:
        y_col, title, color = "average_traveltime", "Travel Time Analysis", "#3498db"
        y_label = "Average Travel Time (minutes)"
        dist_x_label = "Average Travel Time (minutes)"

    dd = data.dropna(subset=["local_datetime", y_col]).sort_values("local_datetime")

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Time Series Analysis", "Distribution Analysis"),
        vertical_spacing=0.1,
    )

    fig.add_trace(
        go.Scatter(
            x=dd["local_datetime"],
            y=dd[y_col],
            mode="lines+markers",
            name=f"{metric_type.title()} Trend",
            line=dict(color=color, width=2),
            marker=dict(size=4),
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=dd[y_col],
            nbinsx=30,
            name=f"{metric_type.title()} Distribution",
            marker_color=color,
            opacity=0.75,
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=600,
        title=title,
        showlegend=True,
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(title_text="Date/Time", row=1, col=1)
    fig.update_yaxes(title_text=y_label, row=1, col=1)
    fig.update_xaxes(title_text=dist_x_label, row=2, col=1)
    fig.update_yaxes(title_text="Frequency (Number of Hours)", row=2, col=1)

    return fig


def volume_charts(
    data: pd.DataFrame,
    theoretical_link_capacity_vph: int,
    high_volume_threshold_vph: int,
):
    if data.empty:
        return None, None, None
    dd = data.dropna(subset=["local_datetime", "total_volume", "intersection_name"]).copy()
    dd.sort_values("local_datetime", inplace=True)

    # 1) Trend by intersection
    fig1 = px.line(
        dd,
        x="local_datetime",
        y="total_volume",
        color="intersection_name",
        title=" Traffic Volume Trends by Intersection",
        labels={"total_volume": "Volume (vehicles/hour)", "local_datetime": "Date/Time"},
        template="plotly_white",
    )
    fig1.update_layout(
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # 2) Distribution + Hourly heatmap
    fig2 = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Volume Distribution by Intersection", "Hourly Avg Volume Heatmap"),
        vertical_spacing=0.12,
    )

    # Box plots
    for name, g in dd.groupby("intersection_name", sort=False):
        fig2.add_trace(go.Box(y=g["total_volume"], name=name, boxpoints="outliers"), row=1, col=1)

    dd["hour"] = dd["local_datetime"].dt.hour
    hourly_avg = dd.groupby(["hour", "intersection_name"], as_index=False)["total_volume"].mean()
    hourly_pivot = hourly_avg.pivot(index="intersection_name", columns="hour", values="total_volume").sort_index()

    fig2.add_trace(
        go.Heatmap(
            z=hourly_pivot.values,
            x=hourly_pivot.columns,
            y=hourly_pivot.index,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Avg Volume (vph)"),
        ),
        row=2, col=1,
    )
    fig2.update_layout(
        height=800,
        title=" Volume Distribution & Capacity Analysis",
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    # 3) Peak hour by intersection
    hourly_volume = dd.groupby(["hour", "intersection_name"], as_index=False)["total_volume"].mean()
    fig3 = px.line(
        hourly_volume,
        x="hour",
        y="total_volume",
        color="intersection_name",
        title=" Average Hourly Volume Patterns",
        labels={"total_volume": "Average Volume (vph)", "hour": "Hour of Day"},
        template="plotly_white",
    )
    fig3.add_hline(
        y=theoretical_link_capacity_vph,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Theoretical Capacity ({theoretical_link_capacity_vph:,} vph)",
    )
    fig3.add_hline(
        y=high_volume_threshold_vph,
        line_dash="dot",
        line_color="orange",
        annotation_text=f"High Volume Threshold ({high_volume_threshold_vph:,} vph)",
    )
    fig3.update_layout(
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig1, fig2, fig3


# =========================
# Date range UI helper
# =========================
def date_range_preset_controls(min_date: datetime.date, max_date: datetime.date, key_prefix: str):
    """
    Presets that default to Last 30 Days on first load, persist in session_state,
    and won't clobber custom picks.
    """
    k_range = f"{key_prefix}_range"

    if k_range not in st.session_state:
        default_start = max(min_date, max_date - timedelta(days=30))
        st.session_state[k_range] = (default_start, max_date)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button(" Last 7 Days", key=f"{key_prefix}_7d"):
            st.session_state[k_range] = (max(min_date, max_date - timedelta(days=7)), max_date)
    with c2:
        if st.button(" Last 30 Days", key=f"{key_prefix}_30d"):
            st.session_state[k_range] = (max(min_date, max_date - timedelta(days=30)), max_date)
    with c3:
        if st.button(" Full Range", key=f"{key_prefix}_full"):
            st.session_state[k_range] = (min_date, max_date)

    custom = st.date_input(
        "Custom Date Range",
        value=st.session_state[k_range],
        min_value=min_date,
        max_value=max_date,
        key=f"{key_prefix}_custom",
    )
    if custom != st.session_state[k_range]:
        st.session_state[k_range] = custom

    return st.session_state[k_range]


# =========================
# Processing
# =========================
def process_traffic_data(df, date_range, granularity, time_filter=None, start_hour=None, end_hour=None):
    """
    Process traffic data based on date range and granularity selections.
    Works for:
      - Corridor/TT/Speed wide data (with or without 'segment_name')
      - Volume data (intersection_id)
    """
    df = df.copy()
    df["local_datetime"] = pd.to_datetime(df["local_datetime"])

    # Filter by date range
    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df[
            (df["local_datetime"].dt.date >= start_date)
            & (df["local_datetime"].dt.date <= end_date)
        ]

    # Optional time-of-day filters for Hourly
    if granularity == "Hourly" and time_filter:
        if time_filter == "Peak Hours (7-9 AM, 4-6 PM)":
            df = df[
                (df["local_datetime"].dt.hour.between(7, 9))
                | (df["local_datetime"].dt.hour.between(16, 18))
            ]
        elif time_filter == "AM Peak (7-9 AM)":
            df = df[df["local_datetime"].dt.hour.between(7, 9)]
        elif time_filter == "PM Peak (4-6 PM)":
            df = df[df["local_datetime"].dt.hour.between(16, 18)]
        elif time_filter == "Off-Peak":
            df = df[
                ~(df["local_datetime"].dt.hour.between(7, 9))
                & ~(df["local_datetime"].dt.hour.between(16, 18))
            ]
        elif time_filter == "Custom Range" and start_hour is not None and end_hour is not None:
            df = df[df["local_datetime"].dt.hour.between(start_hour, end_hour - 1)]

    # ---- Determine dataset type and aggregate accordingly ----
    has_corridor_measures = ("average_traveltime" in df.columns) or ("average_speed" in df.columns)

    if has_corridor_measures:
        # Corridor-wide data (Acyclica): aggregate by corridor_id/direction (segment_name optional)
        key_base = [k for k in ["corridor_id", "direction"] if k in df.columns]
        if "segment_name" in df.columns:
            key_base.append("segment_name")

        if granularity == "Daily":
            df["date_group"] = df["local_datetime"].dt.date
            grouped = df.groupby(["date_group"] + key_base).agg(
                {
                    "average_traveltime": "mean" if "average_traveltime" in df.columns else "first",
                    "average_speed": "mean" if "average_speed" in df.columns else "first",
                }
            ).reset_index()
            grouped["local_datetime"] = pd.to_datetime(grouped["date_group"])

        elif granularity == "Weekly":
            df["week_group"] = df["local_datetime"].dt.to_period("W").dt.start_time
            grouped = df.groupby(["week_group"] + key_base).agg(
                {
                    "average_traveltime": "mean" if "average_traveltime" in df.columns else "first",
                    "average_speed": "mean" if "average_speed" in df.columns else "first",
                }
            ).reset_index()
            grouped["local_datetime"] = grouped["week_group"]

        elif granularity == "Monthly":
            df["month_group"] = df["local_datetime"].dt.to_period("M").dt.start_time
            grouped = df.groupby(["month_group"] + key_base).agg(
                {
                    "average_traveltime": "mean" if "average_traveltime" in df.columns else "first",
                    "average_speed": "mean" if "average_speed" in df.columns else "first",
                }
            ).reset_index()
            grouped["local_datetime"] = grouped["month_group"]

        else:  # Hourly - no aggregation
            grouped = df

    elif "intersection_id" in df.columns:  # Volume data
        if granularity == "Daily":
            df["date_group"] = df["local_datetime"].dt.date
            grouped = df.groupby(["date_group", "intersection_id", "direction", "intersection_name"]).agg(
                {"total_volume": "sum"}
            ).reset_index()
            grouped["local_datetime"] = pd.to_datetime(grouped["date_group"])

        elif granularity == "Weekly":
            df["week_group"] = df["local_datetime"].dt.to_period("W").dt.start_time
            grouped = df.groupby(["week_group", "intersection_id", "direction", "intersection_name"]).agg(
                {"total_volume": "sum"}
            ).reset_index()
            grouped["local_datetime"] = grouped["week_group"]

        elif granularity == "Monthly":
            df["month_group"] = df["local_datetime"].dt.to_period("M").dt.start_time
            grouped = df.groupby(["month_group", "intersection_id", "direction", "intersection_name"]).agg(
                {"total_volume": "sum"}
            ).reset_index()
            grouped["local_datetime"] = grouped["month_group"]

        else:  # Hourly - no aggregation
            grouped = df

    else:
        # Fallback - just return filtered data
        grouped = df

    return grouped


# ... existing code ...

def render_tab3_analysis():
    """
    Main Tab 3 renderer for Acyclica travel time analysis.
    Matches the design and layout of Tabs 1 and 2.
    """
    try:
        # Load Acyclica data once to populate controls
        acyclica_df = get_acyclica_df()
    except RuntimeError as e:
        st.error(f"❌ Failed to load Acyclica data: {e}")
        return
    except Exception as e:
        st.error(f"❌ Unexpected error loading Acyclica data: {e}")
        return

    # -------- Sidebar controls (matching Tab 1 & 2 style) --------
    with st.sidebar:
        with st.expander("⚙️ Pg.3 ACYCLICA SETTINGS", expanded=True):
            st.caption("Select Corridor and Date Range")
            st.caption("Data: Travel Time & Speed from Acyclica sensors")

            # Get available corridors
            if not acyclica_df.empty and "corridor_id" in acyclica_df.columns:
                corridors = ["All Corridors"] + sorted(acyclica_df["corridor_id"].dropna().unique().tolist())
            else:
                corridors = ["All Corridors"]

            st.markdown("## 🛣️ Select Corridor")
            corridor = st.selectbox("Corridor", corridors, key="corridor_acyclica")

            # Date range
            if acyclica_df.empty or "local_datetime" not in acyclica_df.columns:
                min_date = datetime.today().date() - timedelta(days=7)
                max_date = datetime.today().date()
            else:
                min_date = acyclica_df["local_datetime"].dt.date.min()
                max_date = acyclica_df["local_datetime"].dt.date.max()

            st.markdown("## 📅 Date And Time")
            date_range = date_range_preset_controls(min_date, max_date, key_prefix="acyclica")

            # Granularity
            st.markdown("## Granularity")
            granularity = st.selectbox(
                "Data Aggregation",
                ["Hourly", "Daily", "Weekly", "Monthly"],
                index=0,
                key="granularity_acyclica",
            )

            # Direction filter
            if not acyclica_df.empty and "direction" in acyclica_df.columns:
                direction_options = ["All Directions"] + sorted(acyclica_df["direction"].dropna().unique().tolist())
            else:
                direction_options = ["All Directions"]
            direction_filter = st.selectbox("🔄 Direction Filter", direction_options, key="direction_filter_acyclica")

            # Time period focus for hourly
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
                    key="time_period_focus_acyclica",
                )
                if time_filter == "Custom Range":
                    c1, c2 = st.columns(2)
                    with c1:
                        start_hour = st.number_input("Start Hour (0–23)", 0, 23, 7, step=1, key="start_hour_acyclica")
                    with c2:
                        end_hour = st.number_input("End Hour (1–24)", 1, 24, 18, step=1, key="end_hour_acyclica")

            # Search button (matching other tabs)
            if st.button("🔍 **Search**", key="search_tab3", type="primary", use_container_width=True):
                st.session_state["t3_ready"] = True
                st.session_state["t3_params"] = {
                    "corridor": corridor,
                    "date_range": date_range,
                    "granularity": granularity,
                    "direction_filter": direction_filter,
                    "time_filter": time_filter,
                    "start_hour": start_hour,
                    "end_hour": end_hour,
                }

    # -------- Main content area (only render after Search) --------
    t3_ready = st.session_state.get("t3_ready", False)

    if not t3_ready:
        st.info("Choose your Corridor and Date Range in the settings to the left.")
        return

    t3_params = st.session_state.get("t3_params", {})
    corridor = t3_params.get("corridor", "All Corridors")
    date_range = t3_params.get("date_range")
    granularity = t3_params.get("granularity", "Hourly")
    direction_filter = t3_params.get("direction_filter", "All Directions")
    time_filter = t3_params.get("time_filter")
    start_hour = t3_params.get("start_hour")
    end_hour = t3_params.get("end_hour")

    if not date_range or len(date_range) != 2:
        st.warning("⚠️ Please select both start and end dates to proceed.")
        return

    try:
        # Apply filters
        working_df = acyclica_df.copy() if not acyclica_df.empty else pd.DataFrame()

        if working_df.empty:
            st.error("❌ No Acyclica data available.")
            return

        # Filter by corridor
        if corridor != "All Corridors":
            working_df = working_df[working_df["corridor_id"] == corridor]

        # Filter by direction
        if direction_filter != "All Directions" and "direction" in working_df.columns:
            working_df = working_df[working_df["direction"] == direction_filter]

        # ---------- Layout: wide content + sticky right rail (matching Tabs 1 & 2) ----------
        main_col_t3, right_col_t3 = st.columns([7, 3.5], gap="large")

        # Right rail (map code - now with actual map!)
        with right_col_t3:
            st.markdown('<div id="acyclica-map-anchor"></div>', unsafe_allow_html=True)
            st.markdown("##### Corridor Map", help="Stays visible while you scroll the analysis on the left.")

            # Use the existing map functions since Acyclica is on the same Washington Street corridor
            try:
                # Try to build the corridor overview map (shows all segments)
                fig_corridor = build_all_segments_overview()

                if fig_corridor:
                    try:
                        # Match the map height used in other tabs
                        fig_corridor.update_layout(height=900, margin=dict(l=0, r=0, t=32, b=0))
                    except Exception:
                        pass

                    st.markdown('<div class="cvag-map-card">', unsafe_allow_html=True)
                    st.plotly_chart(fig_corridor, use_container_width=True, config={"displaylogo": False})
                    st.caption(f"**Acyclica Corridor:** {corridor}")
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    # Fallback: try the intersections overview
                    fig_intersections = build_intersections_overview()
                    if fig_intersections:
                        try:
                            fig_intersections.update_layout(height=900, margin=dict(l=0, r=0, t=32, b=0))
                        except Exception:
                            pass
                        st.markdown('<div class="cvag-map-card">', unsafe_allow_html=True)
                        st.plotly_chart(fig_intersections, use_container_width=True, config={"displaylogo": False})
                        st.caption(f"**Acyclica Corridor:** {corridor}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        # Final fallback
                        st.markdown('<div class="cvag-map-card">', unsafe_allow_html=True)
                        st.info(f"**Acyclica Corridor:** {corridor}")
                        st.markdown("📍 Washington Street Corridor")
                        st.markdown("_Acyclica sensors monitor travel time and speed along the corridor_")
                        st.markdown("</div>", unsafe_allow_html=True)
            except Exception:
                # Error fallback
                st.markdown('<div class="cvag-map-card">', unsafe_allow_html=True)
                st.info(f"**Acyclica Corridor:** {corridor}")
                st.markdown("📍 Washington Street Corridor")
                st.markdown("_Acyclica sensors monitor travel time and speed along the corridor_")
                st.markdown("</div>", unsafe_allow_html=True)

        # Left/main content
        with main_col_t3:
            # Process the data
            filtered_data = process_traffic_data(
                working_df,
                date_range,
                granularity,
                time_filter if granularity == "Hourly" else None,
                start_hour,
                end_hour,
            )

            if filtered_data.empty:
                st.warning("⚠️ No data available for the selected filters.")
                return

            # Display header (matching Tab 1 & 2 style)
            total_records = len(filtered_data)
            data_span = (date_range[1] - date_range[0]).days + 1
            time_context = f" • {time_filter}" if (granularity == "Hourly" and time_filter) else ""

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
                                box-shadow:inset 0 0 0 1px rgba(255,255,255,.15);">🌐</div>
                    <div style="font-size:1.9rem;font-weight:800;letter-spacing:.2px;">
                      Acyclica Travel Time Analysis: {corridor}
                    </div>
                  </div>
                  <div style="margin-top:10px;display:flex;flex-direction:column;gap:6px;">
                    <div>📅 {date_range[0].strftime('%b %d, %Y')} to {date_range[1].strftime('%b %d, %Y')} ({data_span} days) • {granularity} Aggregation{time_context}</div>
                    <div>✅ Analyzing {total_records:,} data points from Acyclica sensors • Direction: {direction_filter}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Ensure numeric columns
            for col in ["average_traveltime", "average_speed"]:
                if col in filtered_data.columns:
                    filtered_data[col] = pd.to_numeric(filtered_data[col], errors="coerce")

            # ---- KPIs Section (matching Tab 1 order and style) ----
            st.subheader("🚦 KPI's (Key Performance Indicators)")
            if not filtered_data.empty:
                # Calculate KPI values
                avg_tt = float(np.nanmean(
                    filtered_data["average_traveltime"])) if "average_traveltime" in filtered_data.columns else 0.0
                avg_speed = float(
                    np.nanmean(filtered_data["average_speed"])) if "average_speed" in filtered_data.columns else 0.0

                if "average_traveltime" in filtered_data.columns and filtered_data["average_traveltime"].notna().any():
                    p95_tt = float(np.nanpercentile(filtered_data["average_traveltime"].dropna(), 95))
                else:
                    p95_tt = 0.0

                # Calculate buffer time and reliability
                buffer_minutes = max(0.0, p95_tt - avg_tt)
                if avg_tt > 0:
                    cv_tt = float(np.nanstd(filtered_data["average_traveltime"])) / avg_tt * 100.0
                    reliability = max(0.0, 100.0 - cv_tt)
                else:
                    reliability = 50.0

                # Calculate performance scores (matching Tab 1 logic)
                tt_score = max(0, 100 - (avg_tt * 5)) if avg_tt > 0 else 50
                planning_score = max(0, 100 - (p95_tt * 4)) if p95_tt > 0 else 50
                buffer_score = max(0, 100 - (buffer_minutes * 10)) if buffer_minutes >= 0 else 50
                speed_score = min(100, max(0, (avg_speed / 35) * 100)) if avg_speed > 0 else 50

                # Display metrics in the same order as Tab 1
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric(
                        "🎯 Reliability Index",
                        f"{reliability:.0f}%",
                        help="Travel time reliability (100% - coefficient of variation%)"
                    )
                    st.markdown(render_badge(reliability), unsafe_allow_html=True)

                with col2:
                    st.metric(
                        "⏱️ Average Travel Time",
                        f"{avg_tt:.1f} min",
                        help="Average travel time across the selected period and filters"
                    )
                    st.markdown(render_badge(tt_score), unsafe_allow_html=True)

                with col3:
                    st.metric(
                        "📈 Planning Time (95th Percentile)",
                        f"{p95_tt:.1f} min",
                        help="95th percentile travel time - plan for this to arrive on time 95% of the time"
                    )
                    st.markdown(render_badge(planning_score), unsafe_allow_html=True)

                with col4:
                    st.metric(
                        "🧭 Buffer Time (leave this much earlier)",
                        f"{buffer_minutes:.1f} min",
                        help="Extra time needed above average to arrive on time 95% of trips"
                    )
                    st.markdown(render_badge(buffer_score), unsafe_allow_html=True)

                with col5:
                    st.metric(
                        "🚗 Average Speed",
                        f"{avg_speed:.1f} mph",
                        help="Average speed across the corridor"
                    )
                    st.markdown(render_badge(speed_score), unsafe_allow_html=True)

            # ---- Charts Section ----
            if len(filtered_data) > 1:
                st.subheader("📈 Performance Trends")

                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    # Travel Time Chart
                    tt_chart = performance_chart(filtered_data, "travel")
                    if tt_chart:
                        st.plotly_chart(tt_chart, use_container_width=True, config={"displaylogo": False})

                with chart_col2:
                    # Speed Chart
                    if "average_speed" in filtered_data.columns:
                        speed_data = filtered_data.dropna(subset=["local_datetime", "average_speed"]).sort_values(
                            "local_datetime")

                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=("Speed Time Series Analysis", "Speed Distribution Analysis"),
                            vertical_spacing=0.1,
                        )

                        # Time series plot
                        fig.add_trace(
                            go.Scatter(
                                x=speed_data["local_datetime"],
                                y=speed_data["average_speed"],
                                mode="lines+markers",
                                name="Speed Trend",
                                line=dict(color="#2ecc71", width=2),
                                marker=dict(size=4),
                            ),
                            row=1, col=1,
                        )

                        # Distribution histogram
                        fig.add_trace(
                            go.Histogram(
                                x=speed_data["average_speed"],
                                nbinsx=30,
                                name="Speed Distribution",
                                marker_color="#2ecc71",
                                opacity=0.75,
                            ),
                            row=2, col=1,
                        )

                        fig.update_layout(
                            height=600,
                            title="Speed Analysis",
                            showlegend=True,
                            template="plotly_white",
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                        )
                        fig.update_xaxes(title_text="Date/Time", row=1, col=1)
                        fig.update_yaxes(title_text="Average Speed (mph)", row=1, col=1)
                        fig.update_xaxes(title_text="Average Speed (mph)", row=2, col=1)
                        fig.update_yaxes(title_text="Frequency (Number of Hours)", row=2, col=1)

                        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

            # ---- Data Table ----
            st.subheader("🔍 Which Dates/Times have the highest Travel Time?")
            display_columns = ["local_datetime", "corridor_id", "direction", "average_traveltime", "average_speed"]
            available_columns = [col for col in display_columns if col in filtered_data.columns]

            if available_columns:
                display_df = filtered_data[available_columns].copy()
                # Rename columns for better display
                column_renames = {
                    "local_datetime": "Timestamp",
                    "corridor_id": "Corridor",
                    "direction": "Direction",
                    "average_traveltime": "Travel Time (min)",
                    "average_speed": "Speed (mph)"
                }
                display_df = display_df.rename(
                    columns={k: v for k, v in column_renames.items() if k in display_df.columns})

                st.dataframe(display_df, use_container_width=True)

                # Download buttons (matching other tabs)
                st.download_button(
                    "⬇️ Download Acyclica Analysis (CSV)",
                    data=display_df.to_csv(index=False).encode("utf-8"),
                    file_name="acyclica_analysis.csv",
                    mime="text/csv",
                )
                st.download_button(
                    "⬇️ Download Filtered Data (CSV)",
                    data=filtered_data.to_csv(index=False).encode("utf-8"),
                    file_name="acyclica_filtered.csv",
                    mime="text/csv",
                )

            # ---- Performance Analysis by Corridor/Direction ----
            if "corridor_id" in filtered_data.columns and "direction" in filtered_data.columns:
                st.subheader("🚨 Corridor Performance Analysis")

                # Group by corridor and direction
                perf_analysis = filtered_data.groupby(["corridor_id", "direction"]).agg(
                    avg_travel_time=("average_traveltime", "mean"),
                    max_travel_time=("average_traveltime", "max"),
                    avg_speed=("average_speed", "mean"),
                    min_speed=("average_speed", "min"),
                    observations=("average_traveltime", "count")
                ).reset_index()

                # Calculate performance scores
                if not perf_analysis.empty:
                    def normalize_score(series, reverse=False):
                        series = pd.to_numeric(series, errors="coerce")
                        if len(series) < 2 or series.std() == 0:
                            return pd.Series([50.0] * len(series))
                        normalized = (series - series.min()) / (series.max() - series.min())
                        if reverse:
                            normalized = 1 - normalized
                        return normalized * 100

                    perf_analysis["Travel Time Score"] = normalize_score(perf_analysis["avg_travel_time"], reverse=True)
                    perf_analysis["Speed Score"] = normalize_score(perf_analysis["avg_speed"], reverse=False)
                    perf_analysis["Overall Score"] = (perf_analysis["Travel Time Score"] + perf_analysis[
                        "Speed Score"]) / 2

                    # Add performance ratings
                    def get_rating(score):
                        if score > 80:
                            return "🟢 Excellent"
                        elif score > 60:
                            return "🔵 Good"
                        elif score > 40:
                            return "🟡 Fair"
                        elif score > 20:
                            return "🟠 Poor"
                        else:
                            return "🔴 Critical"

                    perf_analysis["🎯 Performance Rating"] = perf_analysis["Overall Score"].apply(get_rating)

                    # Display the analysis
                    display_perf = perf_analysis.rename(columns={
                        "corridor_id": "Corridor",
                        "direction": "Direction",
                        "avg_travel_time": "Avg Travel Time (min)",
                        "max_travel_time": "Max Travel Time (min)",
                        "avg_speed": "Avg Speed (mph)",
                        "min_speed": "Min Speed (mph)",
                        "observations": "Data Points"
                    }).round(2)

                    st.dataframe(
                        display_perf.sort_values("Overall Score", ascending=False),
                        use_container_width=True,
                        column_config={
                            "Overall Score": st.column_config.NumberColumn(
                                "Overall Performance Score",
                                help="Composite score based on travel time and speed performance",
                                format="%.1f"
                            )
                        }
                    )

    except Exception as e:
        st.error(f"❌ Error processing Acyclica data: {e}")
        import traceback
        st.text("Debug info:")
        st.text(traceback.format_exc())
