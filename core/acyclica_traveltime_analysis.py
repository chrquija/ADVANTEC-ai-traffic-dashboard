# core/acyclica_traveltime_analysis.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sidebar_functions import (
    get_acyclica_df,
    process_traffic_data,
    compute_perf_kpis_interpretable,
    render_badge,
    performance_chart,
    date_range_preset_controls,
    get_performance_rating,
)

# Import from timeline_scrubber for consistent headers
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Prediction.timeline_scrubber import render_gradient_header

# Import incident detection for Tab 3
from Prediction.incident_detection import render_incident_detection_section


def _safe_get_acyclica_df() -> pd.DataFrame:
    """
    Defensive loader: tries the wide-format getter first.
    If that fails, rebuilds wide-format from long-format and normalizes direction safely.
    """
    try:
        return get_acyclica_df()
    except Exception:
        # Late import to avoid any issues unless needed
        try:
            from sidebar_functions import get_acyclica_long_df, acyclica_long_to_hourly
        except Exception:
            return pd.DataFrame()

        long_df = get_acyclica_long_df()
        if long_df is None or long_df.empty:
            return pd.DataFrame()

        # Normalize direction safely on a Series
        if "direction" in long_df.columns:
            long_df["direction"] = long_df["direction"].astype(str).str.strip().str.upper()

        # Pivot long → wide
        try:
            wide = acyclica_long_to_hourly(long_df)
        except Exception:
            return pd.DataFrame()

        if wide is None or wide.empty:
            return pd.DataFrame()

        # Ensure datetime type and drop bad rows
        if "local_datetime" in wide.columns:
            wide["local_datetime"] = pd.to_datetime(wide["local_datetime"], errors="coerce")
            wide = wide.dropna(subset=["local_datetime"])
        return wide


def compute_acyclica_kpis(df: pd.DataFrame, low_speed_threshold: float) -> dict:
    """
    Compute KPIs for Acyclica data (speed-focused instead of delay-focused).
    Similar to compute_perf_kpis_interpretable but uses speed metrics instead of delay.
    """
    if df is None or df.empty:
        return {
            "avg_tt": {"value": 0.0, "unit": "min", "score": 50.0, "help": "Average Travel Time"},
            "planning_time": {"value": 0.0, "unit": "min", "score": 50.0, "help": "Planning Time (95th)"},
            "buffer_index": {"value": 0.0, "unit": "%", "score": 50.0, "help": "Buffer Index"},
            "reliability": {"value": 0.0, "unit": "%", "score": 50.0, "help": "Reliability Index"},
            "low_speed_freq": {"value": 0.0, "unit": "%", "score": 50.0, "help": "Low Speed Frequency"},
        }

    # Coerce numeric columns safely
    numeric_cols = ["average_speed", "average_traveltime", "average_delay"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Average TT
    if "average_traveltime" in df.columns and df["average_traveltime"].notna().any():
        avg_tt = float(df["average_traveltime"].mean())
    else:
        avg_tt = 0.0

    # Planning time (P95)
    if "average_traveltime" in df.columns and df["average_traveltime"].notna().any():
        p95_tt = float(df["average_traveltime"].quantile(0.95))
    else:
        p95_tt = 0.0

    # Buffer Index
    buffer_index = ((p95_tt - avg_tt) / avg_tt * 100.0) if avg_tt > 0 else 0.0

    # Reliability Index = 100 - CV%
    if avg_tt > 0 and "average_traveltime" in df.columns:
        std_tt = float(df["average_traveltime"].std())
        cv_tt = (std_tt / avg_tt * 100.0) if std_tt > 0 else 0.0
    else:
        cv_tt = 0.0
    reliability = max(0.0, 100.0 - cv_tt)

    # Low Speed Frequency (% of hours with speed < threshold)
    if "average_speed" in df.columns and df["average_speed"].notna().any():
        total_hours = int(df["average_speed"].count())
        low_speed_hours = int((df["average_speed"] < low_speed_threshold).sum())
        low_speed_freq = (low_speed_hours / total_hours * 100.0) if total_hours > 0 else 0.0
    else:
        low_speed_freq, low_speed_hours, total_hours = 0.0, 0, 0

    # Normalized scores (0..100, higher = better)
    def _minmax_score(series: pd.Series, val: float, invert: bool = True) -> float:
        series = pd.to_numeric(series, errors="coerce").dropna()
        if len(series) < 2:
            return 50.0
        mn, mx = float(series.min()), float(series.max())
        if mx <= mn:
            return 50.0
        frac = (val - mn) / (mx - mn)
        if invert:  # lower is better (travel time, buffer index)
            return float(max(0.0, min(100.0, 100.0 * (1.0 - frac))))
        else:  # higher is better (speed)
            return float(max(0.0, min(100.0, 100.0 * frac)))

    if "average_traveltime" in df.columns and df["average_traveltime"].notna().any():
        score_avg_tt = _minmax_score(df["average_traveltime"], avg_tt, invert=True)
        score_plan = _minmax_score(df["average_traveltime"], p95_tt, invert=True)
    else:
        score_avg_tt = score_plan = 50.0

    score_buffer = float(max(0.0, 100.0 - min(max(buffer_index, 0.0), 100.0)))
    score_reliability = float(max(0.0, min(100.0, reliability)))
    score_low_speed = float(max(0.0, min(100.0, 100.0 - low_speed_freq)))

    return {
        "avg_tt": {
            "value": avg_tt,
            "unit": "min",
            "score": score_avg_tt,
            "help": "Average Travel Time\n\nWhat it means: The typical door-to-door trip time for this route with your current filters.\nWhy it exists: Gives a quick sense of what most trips take.\nHow it's calculated: Average of the hourly O-D trip times.\nFormula: mean(travel_time)",
        },
        "planning_time": {
            "value": p95_tt,
            "unit": "min",
            "score": score_plan,
            "help": "Planning Time (95th)\n\nWhat it means: If you take all trip times in current filter, the 95th-percentile is the value such that 95% of the observations are at or below it.",
        },
        "buffer_index": {
            "value": buffer_index,
            "unit": "%",
            "score": score_buffer,
            "help": "Buffer Index\n\nWhat it means: Extra time (as a percent) you should add on top of the average to be safe.\nFormula: (P95 − mean) / mean × 100%",
        },
        "reliability": {
            "value": reliability,
            "unit": "%",
            "score": score_reliability,
            "help": "Reliability Index\n\nHigher = more predictable travel times.",
        },
        "low_speed_freq": {
            "value": low_speed_freq,
            "unit": "%",
            "score": score_low_speed,
            "extra": f"Hours < {low_speed_threshold:.0f}mph: {low_speed_hours}/{total_hours}",
            "help": f"Low Speed Frequency\n\nPercent of hours below {low_speed_threshold:.0f} mph.",
        },
    }


def speed_performance_chart(data: pd.DataFrame, metric_type: str = "speed"):
    """
    Create performance charts focused on speed or travel time.
    """
    if data.empty:
        return None

    # Ensure numeric columns
    numeric_cols = ["average_speed", "average_traveltime", "average_delay"]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    metric_type = metric_type.lower().strip()
    if metric_type == "speed":
        y_col, title, color = "average_speed", "Traffic Speed Analysis", "#2ecc71"
        y_label = "Average Speed (mph)"
        dist_x_label = "Average Speed (mph)"
    else:
        y_col, title, color = "average_traveltime", "Travel Time Analysis", "#3498db"
        y_label = "Average Travel Time (minutes)"
        dist_x_label = "Average Travel Time (minutes)"

    # Check if column exists and has data
    if y_col not in data.columns or data[y_col].isna().all():
        st.warning(f"No data available for {y_col}")
        return None

    dd = data.dropna(subset=["local_datetime", y_col]).sort_values("local_datetime")
    if dd.empty:
        st.warning(f"No valid data for {metric_type} analysis")
        return None

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
        row=1, col=1,
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


def render_acyclica_section():
    """
    Main function to render the Acyclica travel time analysis section.
    """
    # Load Acyclica data - use defensive loader
    # Load Acyclica data OUTSIDE the sidebar and stop if it fails/empty
    try:
        acyclica_df = _safe_get_acyclica_df()
    except Exception as e:
        st.error(f"Error loading Acyclica data: {e}")
        st.stop()

    if acyclica_df is None or acyclica_df.empty:
        st.error("❌ Failed to load Acyclica data. Please check your data sources.")
        st.stop()

    # Data preview
    with st.expander("🔍 Data Preview (First 5 rows)", expanded=False):
        st.write("**Available columns:**", list(acyclica_df.columns))
        st.dataframe(acyclica_df.head(), use_container_width=True)
        st.write("**Data shape:**", acyclica_df.shape)
        if "local_datetime" in acyclica_df.columns:
            st.write("**Date range:**", acyclica_df["local_datetime"].min(), "to", acyclica_df["local_datetime"].max())

    # Ensure required columns exist and are properly typed
    required_cols = ["local_datetime", "average_traveltime", "average_speed"]
    missing_cols = [col for col in required_cols if col not in acyclica_df.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns in Acyclica data: {', '.join(missing_cols)}")
        return

    # Clean and prepare the data
    try:
        acyclica_df["local_datetime"] = pd.to_datetime(acyclica_df["local_datetime"], errors="coerce")
        acyclica_df = acyclica_df.dropna(subset=["local_datetime"])

        for col in ["average_traveltime", "average_speed", "average_delay"]:
            if col in acyclica_df.columns:
                acyclica_df[col] = pd.to_numeric(acyclica_df[col], errors="coerce")

        if "segment_name" not in acyclica_df.columns:
            if "corridor_id" in acyclica_df.columns and "direction" in acyclica_df.columns:
                acyclica_df["segment_name"] = acyclica_df["corridor_id"].astype(str) + " (" + acyclica_df["direction"].astype(str) + ")"
            elif "corridor_id" in acyclica_df.columns:
                acyclica_df["segment_name"] = acyclica_df["corridor_id"].astype(str)
            else:
                acyclica_df["segment_name"] = "Washington Street"
    except Exception as e:
        st.error(f"❌ Error processing Acyclica data: {str(e)}")
        return

    # Header
    render_gradient_header(
        title="Travel Time Analysis: Acyclica Data",
        subtitle_left="🚗 Same comprehensive analytics as Iteris ClearGuide + Speed-focused insights",
        icon="⚡"
    )

    # Date range and basic filters
    if "local_datetime" in acyclica_df.columns:
        min_date_ts = acyclica_df["local_datetime"].min()
        max_date_ts = acyclica_df["local_datetime"].max()
    else:
        min_date_ts = datetime.today() - timedelta(days=7)
        max_date_ts = datetime.today()

    # Convert to date, ensure ordering and span
    try:
        min_date = pd.to_datetime(min_date_ts, errors="coerce").date()
        max_date = pd.to_datetime(max_date_ts, errors="coerce").date()
        if min_date > max_date:
            min_date, max_date = max_date, min_date
        if min_date == max_date:
            max_date = max_date + timedelta(days=1)
    except Exception:
        min_date = (datetime.today().date() - timedelta(days=30))
        max_date = datetime.today().date()

    st.markdown("#### 📅 Analysis Period")
    try:
        date_range = date_range_preset_controls(min_date, max_date, key_prefix="acyclica")
    except Exception:
        st.session_state.pop("acyclica_range", None)
        date_range = (max(min_date, max_date - timedelta(days=30)), max_date)
        st.info("Date range was reset to a safe default for this dataset.")

    if not date_range or len(date_range) != 2:
        st.warning("⚠️ Please select both start and end dates to proceed.")
        return

    # Process the data
    try:
        filtered_data = process_traffic_data(
            acyclica_df,
            date_range,
            "Hourly"
        )
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        filtered_data = pd.DataFrame()

    if filtered_data.empty:
        st.warning("⚠️ No Acyclica data available for the selected date range.")
        return

    total_records = len(filtered_data)
    data_span = (date_range[1] - date_range[0]).days + 1

    # Ensure numeric types again after processing
    for c in ["average_traveltime", "average_speed"]:
        if c in filtered_data.columns:
            filtered_data[c] = pd.to_numeric(filtered_data[c], errors="coerce")

    # KPIs Section
    st.subheader("🚦 KPI's (Key Performance Indicators)")
    st.info("✨ **Acyclica Advantage**: Speed-based congestion analysis instead of delay-based")

    LOW_SPEED_THRESHOLD = 25.0  # mph

    try:
        k = compute_acyclica_kpis(filtered_data, LOW_SPEED_THRESHOLD)
    except Exception as e:
        st.error(f"❌ Error computing KPIs: {str(e)}")
        return

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
            "🐌 Low Speed Frequency",
            f"{k['low_speed_freq']['value']:.1f}{k['low_speed_freq']['unit']}",
            help=k['low_speed_freq']['help'],
        )
        st.caption(k['low_speed_freq'].get('extra', ''))
        st.markdown(render_badge(k['low_speed_freq']['score']), unsafe_allow_html=True)
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

    # Performance Trends
    if len(filtered_data) > 1:
        st.subheader("📈 Performance Trends")
        st.info("✨ **Acyclica Advantage**: Speed analysis shows traffic flow efficiency")

        v1, v2 = st.columns(2)
        with v1:
            try:
                sc = speed_performance_chart(filtered_data, "speed")
                if sc:
                    st.plotly_chart(sc, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating speed chart: {str(e)}")
        with v2:
            try:
                tc = speed_performance_chart(filtered_data, "travel")
                if tc:
                    st.plotly_chart(tc, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating travel time chart: {str(e)}")

    # Speed-based Bottleneck Analysis
    st.subheader("🚨 Speed-Based Bottleneck Analysis")
    st.info("✨ **Acyclica Advantage**: Identifies bottlenecks using speed patterns instead of delay")

    if not filtered_data.empty:
        try:
            # Group by direction if available, otherwise use segment_name
            group_cols = ["segment_name"]
            if "direction" in filtered_data.columns:
                group_cols.append("direction")

            analysis_df = filtered_data.groupby(group_cols).agg(
                average_speed_mean=("average_speed", "mean"),
                average_speed_min=("average_speed", "min"),
                average_traveltime_mean=("average_traveltime", "mean"),
                average_traveltime_max=("average_traveltime", "max"),
                n=("average_speed", "count"),
            ).reset_index()

            if "direction" not in analysis_df.columns:
                analysis_df["direction"] = "All"

            # Speed-based scoring (lower speed = higher bottleneck score)
            def _norm_speed(s):
                s = pd.to_numeric(s, errors="coerce")
                s_clean = s.dropna()
                if len(s_clean) == 0:
                    return pd.Series(np.zeros(len(s)), index=s.index)
                mn, mx = s_clean.min(), s_clean.max()
                if mx > mn:
                    return (s - mn) / (mx - mn)
                return pd.Series(np.zeros(len(s)), index=s.index)

            # Invert speed scores (lower speed = worse performance)
            speed_score = (1.0 - _norm_speed(analysis_df["average_speed_mean"])) * 50
            min_speed_score = (1.0 - _norm_speed(analysis_df["average_speed_min"])) * 30
            travel_time_score = _norm_speed(analysis_df["average_traveltime_max"]) * 20

            analysis_df["Bottleneck_Score"] = (speed_score + min_speed_score + travel_time_score).round(1)

            bins = [-0.1, 20, 40, 60, 80, 200]
            labels = ["🟢 Excellent", "🔵 Good", "🟡 Fair", "🟠 Poor", "🔴 Critical"]
            analysis_df["🎯 Performance Rating"] = pd.cut(analysis_df["Bottleneck_Score"], bins=bins, labels=labels)

            final = analysis_df.rename(columns={
                "average_speed_mean": "Avg Speed (mph)",
                "average_speed_min": "Min Speed (mph)",
                "average_traveltime_mean": "Avg Time (min)",
                "average_traveltime_max": "Peak Time (min)",
                "direction": "Dir",
                "n": "Obs",
            }).sort_values("Bottleneck_Score", ascending=False)

            st.dataframe(
                final.head(15),
                use_container_width=True,
                column_config={
                    "Bottleneck_Score": st.column_config.NumberColumn(
                        "🚨 Speed Impact Score",
                        help="Speed-based composite (0–100); higher = worse performance",
                        format="%.1f",
                    ),
                },
            )

            st.download_button(
                "⬇️ Download Acyclica Analysis (CSV)",
                data=filtered_data.to_csv(index=False).encode("utf-8"),
                file_name="acyclica_analysis.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"❌ Error in speed-based analysis: {str(e)}")

    # Summary Stats
    with st.expander("📊 Data Summary"):
        st.write(f"**Total Records Analyzed:** {total_records:,}")
        st.write(f"**Date Range:** {date_range[0]} to {date_range[1]} ({data_span} days)")
        if "average_speed" in filtered_data.columns and filtered_data["average_speed"].notna().any():
            st.write(f"**Average Speed:** {filtered_data['average_speed'].mean():.1f} mph")
        if "average_traveltime" in filtered_data.columns and filtered_data["average_traveltime"].notna().any():
            st.write(f"**Average Travel Time:** {filtered_data['average_traveltime'].mean():.1f} minutes")

        if "direction" in filtered_data.columns:
            try:
                dir_summary = filtered_data.groupby("direction").agg({
                    "average_speed": "mean",
                    "average_traveltime": "mean"
                }).round(2)
                st.write("**By Direction:**")
                st.dataframe(dir_summary)
            except Exception as e:
                st.error(f"Error creating direction summary: {str(e)}")


def render_tab3_analysis():
    """
    Enhanced Tab 3 renderer with proper sidebar controls and sub-analysis routing.
    """
    # Load Acyclica data for sidebar population (defensive)
    try:
        acyclica_df = _safe_get_acyclica_df()
    except Exception as e:
        st.error(f"Error loading Acyclica data: {str(e)}")
        acyclica_df = pd.DataFrame()

    # Enhanced sidebar with all requested controls
    with st.sidebar:
        with st.expander("⚙️ Pg.3 SETTINGS", expanded=True):
            st.caption("Acyclica Data: Speed + Travel Time Analysis")
            st.caption("AI Models: Peak Hour Prediction, Incident Detection, Event Impact Analysis")

            # Corridor Selection
            st.markdown("## 🛣️ Select Corridor")
            corridor_options = ["Washington Street"]
            selected_corridor = st.selectbox(
                "Corridor",
                corridor_options,
                key="tab3_corridor",
                help="Currently focusing on Washington Street corridor"
            )

            # Direction Filter
            st.markdown("## 🔄 Direction Filter")
            if not acyclica_df.empty and "direction" in acyclica_df.columns:
                available_directions = sorted(pd.Series(acyclica_df["direction"]).dropna().astype(str).unique().tolist())
                direction_options = ["All Directions"] + available_directions
            else:
                direction_options = ["All Directions", "NB", "SB"]

            selected_direction = st.selectbox(
                "Direction",
                direction_options,
                key="tab3_direction",
                help="Filter analysis by travel direction"
            )

            # Date and Time Options — sanitize bounds
            if not acyclica_df.empty and "local_datetime" in acyclica_df.columns:
                min_date_ts = acyclica_df["local_datetime"].min()
                max_date_ts = acyclica_df["local_datetime"].max()
            else:
                min_date_ts = datetime.today() - timedelta(days=30)
                max_date_ts = datetime.today()

            try:
                min_date = pd.to_datetime(min_date_ts, errors="coerce").date()
                max_date = pd.to_datetime(max_date_ts, errors="coerce").date()
                if min_date > max_date:
                    min_date, max_date = max_date, min_date
                if min_date == max_date:
                    max_date = max_date + timedelta(days=1)
            except Exception:
                min_date = (datetime.today().date() - timedelta(days=30))
                max_date = datetime.today().date()

            st.markdown("## 📅 Date And Time")
            try:
                date_range = date_range_preset_controls(min_date, max_date, key_prefix="tab3")
            except Exception:
                # Reset stale session state if needed
                st.session_state.pop("tab3_range", None)
                date_range = (max(min_date, max_date - timedelta(days=30)), max_date)
                st.info("Date range was reset to a safe default for this dataset.")

            # Data Aggregation Selector
            st.markdown("## Data Aggregation")
            aggregation_options = ["Daily", "Weekly", "Monthly"]
            selected_aggregation = st.selectbox(
                "Aggregation Level",
                aggregation_options,
                key="tab3_aggregation",
                help="Choose temporal aggregation level. Daily shows day-by-day patterns, Weekly shows weekly trends, Monthly shows long-term patterns."
            )

            # Analysis Type Selection
            st.markdown("## Select Analysis Type")
            analysis_type = st.selectbox(
                "Choose Analysis",
                [
                    "Travel Time Analysis",
                    "Peak Hour Prediction",
                    "Incident Detection & Recovery",
                    "Event Impact Analysis"
                ],
                key="tab3_analysis_type",
                help="Travel Time Analysis shows same metrics as Tab 1 but with Acyclica data + speed insights"
            )

        # Apply filters to the data based on sidebar selections
        filtered_acyclica_df = acyclica_df.copy() if not acyclica_df.empty else pd.DataFrame()

        if not filtered_acyclica_df.empty:
            # Apply direction filter (guard .str use by casting to string)
            if selected_direction != "All Directions" and "direction" in filtered_acyclica_df.columns:
                dir_series = filtered_acyclica_df["direction"].astype(str)
                filtered_acyclica_df = filtered_acyclica_df[
                    dir_series.str.upper() == str(selected_direction).upper()
                ]
            # Apply date filter if provided
            if date_range and len(date_range) == 2 and "local_datetime" in filtered_acyclica_df.columns:
                filtered_acyclica_df = filtered_acyclica_df[
                    (filtered_acyclica_df["local_datetime"].dt.date >= date_range[0]) &
                    (filtered_acyclica_df["local_datetime"].dt.date <= date_range[1])
                ]

        # Route user to selected section
        if analysis_type == "Travel Time Analysis":
            render_acyclica_section_with_settings(
                filtered_acyclica_df,
                selected_corridor,
                selected_direction,
                date_range,
                selected_aggregation
            )
        elif analysis_type == "Peak Hour Prediction":
            from Prediction.peak_hour_prediction import render_peak_hour_section
            render_peak_hour_section()
        elif analysis_type == "Incident Detection & Recovery":
            render_incident_detection_section(
                df_source=filtered_acyclica_df,
                corridor=selected_corridor,
                direction=selected_direction if selected_direction != "All Directions" else "NB",
                day=date_range[1] if date_range and len(date_range) == 2 else datetime.now().date()
            )
        elif analysis_type == "Event Impact Analysis":
            from Prediction.event_impact_analysis import render_event_impact_section
            render_event_impact_section()


def render_acyclica_section_with_settings(data, corridor, direction, date_range, aggregation):
    """
    Render Acyclica section with the provided settings from sidebar.
    """
    if data.empty and date_range:
        st.warning("⚠️ No data available for the selected filters. Try adjusting your date range or direction filter.")
        return
    elif data.empty:
        st.error("❌ Failed to load Acyclica data. Please check your data sources.")
        return

    # Context banner
    st.info(f"📊 **Analysis Context**: {corridor} | {direction} | {aggregation} Aggregation")

    # Call the main analysis function
    render_acyclica_section()