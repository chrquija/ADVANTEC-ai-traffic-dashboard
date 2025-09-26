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
from Prediction.timeline_scrubber import render_gradient_header


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

    # Coerce numeric
    for c in ("average_speed", "average_traveltime"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

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

    # Low Speed Frequency (% of hours with speed < threshold) - replaces congestion frequency
    if "average_speed" in df and df["average_speed"].notna().any():
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

    if "average_traveltime" in df and df["average_traveltime"].notna().any():
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
            "help": "Planning Time (95th)\n\nWhat it means: If you take all trip times in current filter, the 95th-percentile is the value such that 95% of the observations are at or below it.\nWhy it exists: Averages can hide variability. Planning Time being 95th percentile captures \"typical worst-case\".\nHow to read it: Realistically, your trip will in total, take this much time.",
        },
        "buffer_index": {
            "value": buffer_index,
            "unit": "%",
            "score": score_buffer,
            "help": "Buffer Index\n\nWhat it means: Extra time (as a percent) you should add on top of the average to be safe.\nHow it's calculated: (Planning Time − Average Time) ÷ Average Time × 100%.\nFormula: (P95 − mean) / mean × 100%",
        },
        "reliability": {
            "value": reliability,
            "unit": "%",
            "score": score_reliability,
            "help": "Reliability Index\n\nWhat it Means: Its your predictability score for travel time\n\nWhy it exists: An average travel time may not be reliable since the corridor has spiky and unpredictable periods. Higher RI = more dependable and easier arrival time planning.",
        },
        "low_speed_freq": {
            "value": low_speed_freq,
            "unit": "%",
            "score": score_low_speed,
            "extra": f"Hours < {low_speed_threshold:.0f}mph: {low_speed_hours}/{total_hours}",
            "help": f"Low Speed Frequency\n\nWhat it means: How often speed drops below the chosen threshold of {low_speed_threshold}mph during your selected period.\nWhy it exists: Highlights how frequently you encounter slow speeds that indicate congestion.",
        },
    }


def speed_performance_chart(data: pd.DataFrame, metric_type: str = "speed"):
    """
    Create performance charts focused on speed instead of delay.
    """
    if data.empty:
        return None

    metric_type = metric_type.lower().strip()
    if metric_type == "speed":
        y_col, title, color = "average_speed", "Traffic Speed Analysis", "#2ecc71"
        y_label = "Average Speed (mph)"
        dist_x_label = "Average Speed (mph)"
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
    This mirrors Tab 1 functionality but uses Acyclica data and focuses on speed instead of delay.
    """
    # Load Acyclica data
    acyclica_df = get_acyclica_df()

    if acyclica_df.empty:
        st.error("❌ Failed to load Acyclica data. Please check your data sources.")
        return

    # Render gradient header
    render_gradient_header(
        title="Travel Time Analysis: Acyclica Data",
        subtitle_left="🚗 Same comprehensive analytics as Iteris ClearGuide + Speed-focused insights",
        icon="⚡"
    )

    # Date range and basic filters
    if "local_datetime" in acyclica_df.columns:
        min_date = acyclica_df["local_datetime"].dt.date.min()
        max_date = acyclica_df["local_datetime"].dt.date.max()
    else:
        min_date = datetime.today().date() - timedelta(days=7)
        max_date = datetime.today().date()

    st.markdown("#### 📅 Analysis Period")
    date_range = date_range_preset_controls(min_date, max_date, key_prefix="acyclica")

    if not date_range or len(date_range) != 2:
        st.warning("⚠️ Please select both start and end dates to proceed.")
        return

    # Process the data
    filtered_data = process_traffic_data(
        acyclica_df,
        date_range,
        "Hourly"  # Start with hourly for simplicity
    )

    if filtered_data.empty:
        st.warning("⚠️ No Acyclica data available for the selected date range.")
        return

    total_records = len(filtered_data)
    data_span = (date_range[1] - date_range[0]).days + 1

    # Ensure numeric types
    for c in ["average_traveltime", "average_speed"]:
        if c in filtered_data.columns:
            filtered_data[c] = pd.to_numeric(filtered_data[c], errors="coerce")

    # KPIs Section
    st.subheader("🚦 KPI's (Key Performance Indicators)")
    st.info("✨ **Acyclica Advantage**: Speed-based congestion analysis instead of delay-based")

    LOW_SPEED_THRESHOLD = 25.0  # mph threshold for low speed
    k = compute_acyclica_kpis(filtered_data, LOW_SPEED_THRESHOLD)

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
            sc = speed_performance_chart(filtered_data, "speed")
            if sc:
                st.plotly_chart(sc, use_container_width=True)
        with v2:
            tc = speed_performance_chart(filtered_data, "travel")
            if tc:
                st.plotly_chart(tc, use_container_width=True)

    # Speed-based Bottleneck Analysis
    st.subheader("🚨 Speed-Based Bottleneck Analysis")
    st.info("✨ **Acyclica Advantage**: Identifies bottlenecks using speed patterns instead of delay")

    if not filtered_data.empty:
        try:
            # Group by direction if available
            if "direction" in filtered_data.columns:
                analysis_df = filtered_data.groupby(["segment_name", "direction"]).agg(
                    average_speed_mean=("average_speed", "mean"),
                    average_speed_min=("average_speed", "min"),
                    average_traveltime_mean=("average_traveltime", "mean"),
                    average_traveltime_max=("average_traveltime", "max"),
                    n=("average_speed", "count"),
                ).reset_index()
            else:
                analysis_df = filtered_data.groupby(["segment_name"]).agg(
                    average_speed_mean=("average_speed", "mean"),
                    average_speed_min=("average_speed", "min"),
                    average_traveltime_mean=("average_traveltime", "mean"),
                    average_traveltime_max=("average_traveltime", "max"),
                    n=("average_speed", "count"),
                ).reset_index()
                analysis_df["direction"] = "All"

            # Speed-based scoring (lower speed = higher bottleneck score)
            def _norm_speed(s):
                s = s.astype(float)
                mn, mx = np.nanmin(s), np.nanmax(s)
                if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
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
            st.error(f"❌ Error in speed-based analysis: {e}")

    # Summary Stats
    with st.expander("📊 Data Summary"):
        st.write(f"**Total Records Analyzed:** {total_records:,}")
        st.write(f"**Date Range:** {date_range[0]} to {date_range[1]} ({data_span} days)")
        st.write(f"**Average Speed:** {filtered_data['average_speed'].mean():.1f} mph")
        st.write(f"**Average Travel Time:** {filtered_data['average_traveltime'].mean():.1f} minutes")

        if "direction" in filtered_data.columns:
            dir_summary = filtered_data.groupby("direction").agg({
                "average_speed": "mean",
                "average_traveltime": "mean"
            }).round(2)
            st.write("**By Direction:**")
            st.dataframe(dir_summary)