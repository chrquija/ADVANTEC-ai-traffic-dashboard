# Python
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Plotly for chart helpers
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
