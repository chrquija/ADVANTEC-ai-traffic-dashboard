# Python
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Plotly for chart helpers
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================
# Data loading
# =========================
@st.cache_data
def load_traffic_data():
    """
    Load and combine all corridor traffic Prediction from GitHub
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
        try:
            df = pd.read_csv(url)
            df["segment_name"] = segment_name
            all_data.append(df)
        except Exception as e:
            st.error(f"Error loading {segment_name}: {e}")

    if not all_data:
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df["local_datetime"] = pd.to_datetime(combined_df["local_datetime"])
    combined_df = combined_df.sort_values("local_datetime").reset_index(drop=True)
    return combined_df


@st.cache_data
def load_volume_data():
    """
    Load consolidated volume Prediction for all Washington Street intersections
    """
    volume_url = "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/VOLUME/KMOB_LONG/LONG_MASTER_Avenue52_to_Avenue47_1hr_NS_VOLUME_OctoberTOJune.csv"

    try:
        volume_df = pd.read_csv(volume_url)

        volume_df["local_datetime"] = pd.to_datetime(volume_df["local_datetime"])
        volume_df = volume_df.sort_values("local_datetime").reset_index(drop=True)

        # Create proper intersection names from intersection_id
        volume_df["intersection_name"] = (
            volume_df["intersection_id"]
            .str.replace("_", " ")
            .str.replace("Washington St and ", "Washington St & ")
            .str.replace(" and ", " & ")
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
        st.error(f"Error loading volume Prediction: {e}")
        return pd.DataFrame()


@st.cache_data
def load_acyclica_data():
    """
    Load Acyclica travel time and speed data from GitHub.
    Data structure: local_datetime, corridor_id, direction, metric, Strength, Firsts, Lasts, Minimum, Maximum
    The 'metric' column contains either 'TravelTime' or 'Speed' values.
    """
    acyclica_url = "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/MASTER_Acyclica_Traveltime_speed.csv"

    try:
        df = pd.read_csv(acyclica_url)

        # Convert datetime
        df["local_datetime"] = pd.to_datetime(df["local_datetime"])

        # Pivot the data to get TravelTime and Speed as separate columns
        # Using 'Strength' as the main value
        pivoted = df.pivot_table(
            index=['local_datetime', 'corridor_id', 'direction'],
            columns='metric',
            values='Strength',
            aggfunc='mean'
        ).reset_index()

        # Flatten column names
        pivoted.columns.name = None

        # Rename columns to match existing schema
        column_mapping = {}
        if 'TravelTime' in pivoted.columns:
            column_mapping['TravelTime'] = 'average_traveltime'
        if 'Speed' in pivoted.columns:
            column_mapping['Speed'] = 'average_speed'

        pivoted = pivoted.rename(columns=column_mapping)

        # Add missing columns that existing code expects
        pivoted['average_delay'] = np.nan  # Acyclica doesn't have delay data
        pivoted['segment_name'] = pivoted['corridor_id'].astype(str)  # Create segment_name from corridor_id

        # Sort by datetime
        pivoted = pivoted.sort_values("local_datetime").reset_index(drop=True)

        return pivoted

    except Exception as e:
        st.error(f"Error loading Acyclica data: {e}")
        return pd.DataFrame()


# =========================
# Small Prediction utilities
# =========================
def _safe_to_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


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
def get_acyclica_df() -> pd.DataFrame:
    """
    Get Acyclica dataset for Tab 3 (speed-focused analysis).
    Returns data with: local_datetime, corridor_id, direction, average_traveltime, average_speed, average_delay (NaN)
    """
    df = load_acyclica_data()
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df = _safe_to_datetime(df.copy(), "local_datetime")

    # Check for expected columns
    needed = {"average_traveltime", "average_speed", "direction"}
    missing = needed - set(df.columns)
    if missing:
        st.warning(f"Acyclica dataset is missing columns: {', '.join(missing)}")

    return df


def get_performance_rating(score: float):
    """
    Map a 0..100 score to a label + CSS class used by the UI badges.
    """
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
# Interpretable KPI helpers (for Performance tab)
# =========================
def _coerce_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def compute_perf_kpis_interpretable(df: pd.DataFrame, high_delay_threshold: float) -> dict:
    """
    Compute five interpretable KPIs:
      - avg_tt: Average Travel Time (lower is better)
      - planning_time: 95th percentile travel time (lower is better)
      - buffer_index: (P95 - Mean)/Mean * 100 (lower is better)
      - reliability: 100 - CV(travel_time)% (higher is better)
      - congestion_freq: Share of hours with Delay > threshold (lower is better)

    Returns dict with values, units, normalized 'score' 0..100 (higher = better),
    and 'help' strings that explain formula + interpretation.
    """
    if df is None or df.empty:
        return {
            "avg_tt": {"value": 0.0, "unit": "min", "score": 50.0, "help": "Average Travel Time"},
            "planning_time": {"value": 0.0, "unit": "min", "score": 50.0, "help": "Planning Time (95th)"},
            "buffer_index": {"value": 0.0, "unit": "%", "score": 50.0, "help": "Buffer Index"},
            "reliability": {"value": 0.0, "unit": "%", "score": 50.0, "help": "Reliability Index"},
            "congestion_freq": {"value": 0.0, "unit": "%", "score": 50.0, "help": "Congestion Frequency"},
        }

    # Coerce numeric
    for c in ("average_delay", "average_traveltime", "average_speed"):
        if c in df:
            df[c] = _coerce_num(df[c])

    # Average TT
    avg_tt = float(np.nanmean(df["average_traveltime"])) if "average_traveltime" in df else 0.0

    # Planning time (P95)
    if "average_traveltime" in df and df["average_traveltime"].notna().any():
        p95_tt = float(np.nanpercentile(df["average_traveltime"].dropna(), 95))
    else:
        p95_tt = 0.0

    # Buffer Index
    buffer_index = ((p95_tt - avg_tt) / avg_tt * 100.0) if avg_tt > 0 else 0.0

    # Reliability Index = 100 - CV%
    if avg_tt > 0 and "average_traveltime" in df:
        cv_tt = float(np.nanstd(df["average_traveltime"])) / avg_tt * 100.0
    else:
        cv_tt = 0.0
    reliability = max(0.0, 100.0 - cv_tt)

    # Congestion Frequency (% of hours with delay > threshold)
    if "average_delay" in df and df["average_delay"].notna().any():
        total_hours = int(df["average_delay"].count())
        cong_hours = int((df["average_delay"] > high_delay_threshold).sum())
        cong_freq = (cong_hours / total_hours * 100.0) if total_hours > 0 else 0.0
    else:
        cong_freq, cong_hours, total_hours = 0.0, 0, 0

    # Normalized scores (0..100, higher = better)
    def _minmax_score(series: pd.Series, val: float) -> float:
        series = pd.to_numeric(series, errors="coerce").dropna()
        if len(series) < 2:
            return 50.0
        mn, mx = float(series.min()), float(series.max())
        if mx <= mn:
            return 50.0
        # lower is better -> invert
        frac = (val - mn) / (mx - mn)
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
            "help": "Average Travel Time\n\nWhat it means: The typical door-to-door trip time for this route with your current filters.\nWhy it exists: Gives a quick sense of what most trips take.\nHow it’s calculated: Average of the hourly O-D trip times.\nFormula: mean(travel_time)\nExample: 6, 6, 7, 7 minutes → (6 + 6 + 7 + 7) / 4 = 6.5 minutes.",
        },
        "planning_time": {
            "value": p95_tt,
            "unit": "min",
            "score": score_plan,
            "help": "Planning Time (95th)\n\nWhat it means: If you take all trip times in current filter, the 95th-percentile is the value such that 95% of the observations are at or below it.\nWhy it exists: Averages can hide variability. Planning Time being 95th percentile captures \"typical worst-case\".\nHow to read it: Realistically, your trip will in total, take this much time. Its the Travel Time you should plan for so you arrive on time about 95% of trips.",
        },
        "buffer_index": {
            "value": buffer_index,
            "unit": "%",
            "score": score_buffer,
            "help": "Buffer Index\n\nWhat it means: Extra time (as a percent) you should add on top of the average to be safe.\nHow it’s calculated: (Planning Time − Average Time) ÷ Average Time × 100%.\nFormula: (P95 − mean) / mean × 100%\nExample: (7.5 − 6.5) ÷ 6.5 × 100% ≈ 15.4%.",
        },
        "reliability": {
            "value": reliability,
            "unit": "%",
            "score": score_reliability,
            "help": "Reliability Index\n\nWhat it Means: Its your predictability score for travel time\n\nWhy it exists: An average travel time may not be reliable since the corridor has spiky and unpredictable periods. Higher RI = more dependable and easier arrival time planning.\n\nHow to read it:\n\nCalculate CV = Coefficient of Variation (a measure of variability in travel times in this case)\n\nReliability Index (RI) = 100 − CV%, where CV% = (Std Dev / Mean) × 100\n\nExample (travel time):\nIf mean = 6.5 min and stdev = 0.78 min, then CV = 0.78/6.5 ≈ 12%.\nReliability Index = 100 − CV%, \nso RI ≈ 88%.\n\nReliability Index Thresholds:\n≥ 85% → Excellent (very consistent; CV ≤ ~15%)\n\n70–84% → Good (moderately consistent)\n\n55–69% → Fair (noticeable variability)\n\n< 55% → Poor (highly variable; users can’t plan confidently)",
        },
        "congestion_freq": {
            "value": cong_freq,
            "unit": "%",
            "score": score_congestion,
            "extra": f"Hours > {high_delay_threshold:.0f}s: {cong_hours}/{total_hours}",
            "help": f"Congestion Frequency\n\nWhat it means: How often delay occurs above the chosen threshold of 60s during your selected period.\nWhy it exists: Highlights how frequently you encounter “too much” delay.\n\nExample: 0 of 4 hours above {high_delay_threshold:.0f}s → 0%.",
        },
    }


def render_badge(score: float) -> str:
    """
    Turn a 0..100 'goodness' score into your visual badge HTML, using get_performance_rating.
    """
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

    # Time series plot
    fig.add_trace(
        go.Scatter(
            x=dd["local_datetime"],
            y=dd[y_col],
            mode="lines+markers",
            name=f"{metric_type.title()} Trend",
            line=dict(color=color, width=2),
            marker=dict(size=4),
        ),
        row=1,
        col=1,
    )

    # Distribution histogram
    fig.add_trace(
        go.Histogram(
            x=dd[y_col],
            nbinsx=30,
            name=f"{metric_type.title()} Distribution",
            marker_color=color,
            opacity=0.75,
        ),
        row=2,
        col=1,
    )

    # Update layout with proper axis labels
    fig.update_layout(
        height=600,
        title=title,
        showlegend=True,
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    # Update x and y axis labels for both subplots
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
        row=2,
        col=1,
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

    # Default to LAST 30 DAYS (bounded by min_date)
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
    Process traffic Prediction based on date range and granularity selections
    """
    # Convert datetime if not already done
    df["local_datetime"] = pd.to_datetime(df["local_datetime"])

    # Filter by date range
    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df[
            (df["local_datetime"].dt.date >= start_date)
            & (df["local_datetime"].dt.date <= end_date)
        ]

    # Apply time filters for hourly Prediction
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

    # Determine Prediction type and aggregate accordingly
    if "segment_name" in df.columns:  # Corridor Prediction (delay/speed/travel time)
        if granularity == "Daily":
            df["date_group"] = df["local_datetime"].dt.date
            grouped = df.groupby(["date_group", "corridor_id", "direction", "segment_name"]).agg(
                {
                    "average_delay": "mean",
                    "average_traveltime": "mean",
                    "average_speed": "mean",
                }
            ).reset_index()
            grouped["local_datetime"] = pd.to_datetime(grouped["date_group"])

        elif granularity == "Weekly":
            df["week_group"] = df["local_datetime"].dt.to_period("W").dt.start_time
            grouped = df.groupby(["week_group", "corridor_id", "direction", "segment_name"]).agg(
                {
                    "average_delay": "mean",
                    "average_traveltime": "mean",
                    "average_speed": "mean",
                }
            ).reset_index()
            grouped["local_datetime"] = grouped["week_group"]

        elif granularity == "Monthly":
            df["month_group"] = df["local_datetime"].dt.to_period("M").dt.start_time
            grouped = df.groupby(["month_group", "corridor_id", "direction", "segment_name"]).agg(
                {
                    "average_delay": "mean",
                    "average_traveltime": "mean",
                    "average_speed": "mean",
                }
            ).reset_index()
            grouped["local_datetime"] = grouped["month_group"]

        else:  # Hourly - no aggregation needed
            grouped = df

    elif "intersection_id" in df.columns:  # Volume Prediction
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

        else:  # Hourly - no aggregation needed
            grouped = df

    else:
        # Fallback - just return filtered Prediction
        grouped = df

    return grouped