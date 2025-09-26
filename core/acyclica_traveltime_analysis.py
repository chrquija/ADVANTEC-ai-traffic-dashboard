# core/acyclica_traveltime_analysis.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- pull BOTH long + transformer from sidebar_functions ---
from sidebar_functions import (
    get_acyclica_long_df,          # LONG dataframe (Strength/Firsts/Lasts/Minimum/Maximum)
    acyclica_long_to_hourly,       # transformer → WIDE (average_traveltime, average_speed)
    process_traffic_data,
    render_badge,
    date_range_preset_controls,
)

# Import gradient header for visual consistency
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Prediction.timeline_scrubber import render_gradient_header

# Incident Detection tab renderer (expects long df)
from Prediction.incident_detection import render_incident_detection_section


# -------------------------------
# KPIs (speed-focused) you wrote
# -------------------------------
def compute_acyclica_kpis(df: pd.DataFrame, low_speed_threshold: float) -> dict:
    if df is None or df.empty:
        return {
            "avg_tt": {"value": 0.0, "unit": "min", "score": 50.0, "help": "Average Travel Time"},
            "planning_time": {"value": 0.0, "unit": "min", "score": 50.0, "help": "Planning Time (95th)"},
            "buffer_index": {"value": 0.0, "unit": "%", "score": 50.0, "help": "Buffer Index"},
            "reliability": {"value": 0.0, "unit": "%", "score": 50.0, "help": "Reliability Index"},
            "low_speed_freq": {"value": 0.0, "unit": "%", "score": 50.0, "help": "Low Speed Frequency"},
        }

    # Coerce numeric columns safely
    for col in ["average_speed", "average_traveltime", "average_delay"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    avg_tt = float(df["average_traveltime"].mean()) if "average_traveltime" in df.columns else 0.0
    p95_tt = float(df["average_traveltime"].quantile(0.95)) if "average_traveltime" in df.columns else 0.0
    buffer_index = ((p95_tt - avg_tt) / avg_tt * 100.0) if avg_tt > 0 else 0.0

    if avg_tt > 0 and "average_traveltime" in df.columns:
        sd = float(df["average_traveltime"].std()) if df["average_traveltime"].std() is not None else 0.0
        cv_tt = (sd / avg_tt * 100.0) if sd > 0 else 0.0
    else:
        cv_tt = 0.0
    reliability = max(0.0, 100.0 - cv_tt)

    if "average_speed" in df.columns and df["average_speed"].notna().any():
        total_hours = int(df["average_speed"].count())
        low_speed_hours = int((df["average_speed"] < low_speed_threshold).sum())
        low_speed_freq = (low_speed_hours / total_hours * 100.0) if total_hours > 0 else 0.0
    else:
        low_speed_freq, total_hours, low_speed_hours = 0.0, 0, 0

    def _minmax_score(series: pd.Series, val: float, invert: bool = True) -> float:
        series = pd.to_numeric(series, errors="coerce").dropna()
        if len(series) < 2:
            return 50.0
        mn, mx = float(series.min()), float(series.max())
        if mx <= mn:
            return 50.0
        frac = (val - mn) / (mx - mn)
        return float(max(0.0, min(100.0, 100.0 * ((1.0 - frac) if invert else frac))))

    if "average_traveltime" in df.columns and df["average_traveltime"].notna().any():
        score_avg_tt = _minmax_score(df["average_traveltime"], avg_tt, invert=True)
        score_plan   = _minmax_score(df["average_traveltime"], p95_tt, invert=True)
    else:
        score_avg_tt = score_plan = 50.0

    score_buffer     = float(max(0.0, 100.0 - min(max(buffer_index, 0.0), 100.0)))
    score_reliability= float(max(0.0, min(100.0, reliability)))
    score_low_speed  = float(max(0.0, min(100.0, 100.0 - low_speed_freq)))

    return {
        "avg_tt": {"value": avg_tt, "unit": "min", "score": score_avg_tt,
                   "help": "Average of hourly travel time (minutes)."},
        "planning_time": {"value": p95_tt, "unit": "min", "score": score_plan,
                          "help": "95th percentile travel time (minutes)."},
        "buffer_index": {"value": buffer_index, "unit": "%", "score": score_buffer,
                         "help": "Extra % to add to mean to hit P95."},
        "reliability": {"value": reliability, "unit": "%", "score": score_reliability,
                        "help": "100 − CV% (stdev/mean). Higher = steadier."},
        "low_speed_freq": {"value": low_speed_freq, "unit": "%", "score": score_low_speed,
                           "extra": f"Hours < {low_speed_threshold:.0f}mph: {low_speed_hours}/{total_hours}",
                           "help": "Share of hours below threshold speed."},
    }


def speed_performance_chart(data: pd.DataFrame, metric_type: str = "speed"):
    if data.empty:
        return None

    for c in ["average_speed", "average_traveltime", "average_delay"]:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")

    metric_type = metric_type.lower().strip()
    if metric_type == "speed":
        y_col, title, color = "average_speed", "Traffic Speed Analysis", "#2ecc71"
        y_label, dist_x_label = "Average Speed (mph)", "Average Speed (mph)"
    else:
        y_col, title, color = "average_traveltime", "Travel Time Analysis", "#3498db"
        y_label, dist_x_label = "Average Travel Time (minutes)", "Average Travel Time (minutes)"

    if y_col not in data.columns or data[y_col].isna().all():
        st.warning(f"No data available for {y_col}")
        return None

    dd = data.dropna(subset=["local_datetime", y_col]).sort_values("local_datetime")
    if dd.empty:
        st.warning(f"No valid data for {metric_type} analysis")
        return None

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Time Series Analysis", "Distribution Analysis"),
                        vertical_spacing=0.1)

    fig.add_trace(
        go.Scatter(x=dd["local_datetime"], y=dd[y_col], mode="lines+markers",
                   name=f"{metric_type.title()} Trend",
                   line=dict(color=color, width=2), marker=dict(size=4)),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=dd[y_col], nbinsx=30, name=f"{metric_type.title()} Distribution",
                     marker_color=color, opacity=0.75),
        row=2, col=1
    )
    fig.update_layout(height=600, title=title, showlegend=True,
                      template="plotly_white", plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title_text="Date/Time", row=1, col=1)
    fig.update_yaxes(title_text=y_label, row=1, col=1)
    fig.update_xaxes(title_text=dist_x_label, row=2, col=1)
    fig.update_yaxes(title_text="Frequency (Number of Hours)", row=2, col=1)
    return fig


# ---------------------------------------------------------
# MAIN TAB RENDERER  (now uses LONG → filter → WIDE)
# ---------------------------------------------------------
def render_tab3_analysis():
    """
    Tab 3 renderer with correct long/wide handling:
      - Build date bounds from LONG data
      - Filter LONG by corridor/direction/dates
      - Convert to WIDE for KPI/plots
      - Route LONG to incident/prediction modules
    """
    # 1) Load LONG Acyclica once
    try:
        acyclica_long = get_acyclica_long_df()
    except Exception as e:
        st.error(f"Error loading Acyclica data: {e}")
        acyclica_long = pd.DataFrame()

    # 2) Sidebar controls (bounds from LONG so you can select 2024 dates)
    with st.sidebar:
        st.image("Logos/ACE-logo-HiRes.jpg", width=210)
        st.image("Logos/CV Sync__.jpg", width=205)

        with st.expander("⚙️ Pg.3 SETTINGS", expanded=True):
            st.caption("Acyclica Data: Speed + Travel Time Analysis")
            st.caption("AI Models: Peak Hour, Incident Detection, Event Impact")

            st.markdown("## 🛣️ Select Corridor")
            corridor_options = ["Washington Street"]
            selected_corridor = st.selectbox("Corridor", corridor_options, key="tab3_corridor")

            st.markdown("## 🔄 Direction Filter")
            if not acyclica_long.empty and "direction" in acyclica_long.columns:
                dir_opts = ["All Directions"] + sorted(acyclica_long["direction"].dropna().unique().tolist())
            else:
                dir_opts = ["All Directions", "NB", "SB"]
            selected_direction = st.selectbox("Direction", dir_opts, key="tab3_direction")

            # Date bounds from LONG
            if not acyclica_long.empty and "local_datetime" in acyclica_long.columns:
                min_date = acyclica_long["local_datetime"].dt.date.min()
                max_date = acyclica_long["local_datetime"].dt.date.max()
            else:
                min_date = datetime.today().date() - timedelta(days=365)
                max_date = datetime.today().date()

            st.markdown("## 📅 Date And Time")
            date_range = date_range_preset_controls(min_date, max_date, key_prefix="tab3")

            st.markdown("## 📊 Data Aggregation")
            aggregation_options = ["Daily", "Weekly", "Monthly"]  # keep hourly out per your UX
            selected_aggregation = st.selectbox("Aggregation Level", aggregation_options, key="tab3_aggregation")

            st.markdown("## 🎯 Select Analysis Type")
            analysis_type = st.selectbox(
                "Choose Analysis",
                ["🚗 Travel Time Analysis", "🔮 Peak Hour Prediction",
                 "⚠️ Incident Detection & Recovery", "🎪 Event Impact Analysis"],
                key="tab3_analysis_type",
            )

    # 3) Filter LONG by direction + date
    filtered_long = acyclica_long.copy()
    if not filtered_long.empty:
        if selected_direction != "All Directions" and "direction" in filtered_long.columns:
            filtered_long = filtered_long[filtered_long["direction"].str.upper() == selected_direction.upper()]
        if date_range and len(date_range) == 2:
            d0, d1 = date_range
            mask = (filtered_long["local_datetime"].dt.date >= d0) & (filtered_long["local_datetime"].dt.date <= d1)
            filtered_long = filtered_long[mask]

    # 4) Convert LONG → WIDE for KPI/plots
    filtered_wide = acyclica_long_to_hourly(filtered_long) if not filtered_long.empty else pd.DataFrame()

    # 5) Route to sub-views
    if analysis_type == "🚗 Travel Time Analysis":
        render_acyclica_section_with_settings(
            filtered_wide, selected_corridor, selected_direction, date_range, selected_aggregation
        )

    elif analysis_type == "🔮 Peak Hour Prediction":
        from Prediction.peak_hour_prediction import render_peak_hour_section
        # Pass long data if your peak detector needs Firsts/Lasts
        render_peak_hour_section(df_source=filtered_long)

    elif analysis_type == "⚠️ Incident Detection & Recovery":
        render_incident_detection_section(
            df_source=filtered_long,
            corridor=selected_corridor,
            direction=selected_direction if selected_direction != "All Directions" else "NB",
            day=date_range[1] if date_range and len(date_range) == 2 else datetime.now().date(),
        )

    elif analysis_type == "🎪 Event Impact Analysis":
        from Prediction.event_impact_analysis import render_event_impact_section
        render_event_impact_section(df_source=filtered_long)


def render_acyclica_section_with_settings(data_wide, corridor, direction, date_range, aggregation):
    """
    Render the KPI/plots section using **WIDE** data (average_traveltime, average_speed).
    """
    if data_wide.empty and date_range:
        st.warning("⚠️ No data available for the selected filters. Try adjusting your date range or direction filter.")
        return
    elif data_wide.empty:
        st.error("❌ Failed to load Acyclica data. Please check your data sources.")
        return

    st.info(f"📊 **Analysis Context**: {corridor} | {direction} | {aggregation} Aggregation")

    # Defensive typing
    dfw = data_wide.copy()
    try:
        dfw["local_datetime"] = pd.to_datetime(dfw["local_datetime"], errors="coerce")
        dfw = dfw.dropna(subset=["local_datetime"])
        for col in ["average_traveltime", "average_speed", "average_delay"]:
            if col in dfw.columns:
                dfw[col] = pd.to_numeric(dfw[col], errors="coerce")
        if "segment_name" not in dfw.columns:
            if "direction" in dfw.columns:
                dfw["segment_name"] = "Washington Street (" + dfw["direction"].astype(str) + ")"
            else:
                dfw["segment_name"] = "Washington Street"
    except Exception as e:
        st.error(f"❌ Error processing Acyclica data: {str(e)}")
        return

    # Header
    render_gradient_header(
        title="Travel Time Analysis: Acyclica Data",
        subtitle_left="🚗 Same comprehensive analytics as Iteris ClearGuide + Speed-focused insights",
        icon="⚡"
    )

    # Process (aggregation is handled here)
    if not date_range or len(date_range) != 2:
        st.warning("⚠️ Date range not properly set.")
        return

    try:
        filtered_data = process_traffic_data(
            dfw, date_range, aggregation  # Daily/Weekly/Monthly
        )
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        return

    if filtered_data.empty:
        st.warning("⚠️ No Acyclica data available for the selected filters.")
        return

    total_records = len(filtered_data)
    data_span = (date_range[1] - date_range[0]).days + 1

    for c in ["average_traveltime", "average_speed"]:
        if c in filtered_data.columns:
            filtered_data[c] = pd.to_numeric(filtered_data[c], errors="coerce")

    # KPIs
    st.subheader("🚦 KPI's (Key Performance Indicators)")
    st.info("✨ **Acyclica Advantage**: Speed-based congestion analysis instead of delay-based")
    LOW_SPEED_THRESHOLD = 25.0  # mph

    k = compute_acyclica_kpis(filtered_data, LOW_SPEED_THRESHOLD)
    buffer_minutes = max(0.0, k["planning_time"]["value"] - k["avg_tt"]["value"])
    buffer_help = "Extra minutes to leave earlier to be on time 95% of the time."

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("🎯 Reliability Index", f"{k['reliability']['value']:.0f}{k['reliability']['unit']}")
        st.markdown(render_badge(k['reliability']['score']), unsafe_allow_html=True)
    with c2:
        st.metric("🐌 Low Speed Frequency", f"{k['low_speed_freq']['value']:.1f}{k['low_speed_freq']['unit']}")
        st.caption(k['low_speed_freq'].get('extra', ''))
        st.markdown(render_badge(k['low_speed_freq']['score']), unsafe_allow_html=True)
    with c3:
        st.metric("⏱️ Average Travel Time", f"{k['avg_tt']['value']:.1f} {k['avg_tt']['unit']}")
        st.markdown(render_badge(k['avg_tt']['score']), unsafe_allow_html=True)
    with c4:
        st.metric("📈 Planning Time (95th)", f"{k['planning_time']['value']:.1f} {k['planning_time']['unit']}")
        st.markdown(render_badge(k['planning_time']['score']), unsafe_allow_html=True)
    with c5:
        st.metric("🧭 Buffer Time", f"{buffer_minutes:.1f} min", help=buffer_help)
        st.markdown(render_badge(k['buffer_index']['score']), unsafe_allow_html=True)

    # Trends
    if len(filtered_data) > 1:
        st.subheader("📈 Performance Trends")
        st.info("✨ **Acyclica Advantage**: Speed analysis shows traffic flow efficiency")
        v1, v2 = st.columns(2)
        with v1:
            sc = speed_performance_chart(filtered_data, "speed")
            if sc: st.plotly_chart(sc, use_container_width=True)
        with v2:
            tc = speed_performance_chart(filtered_data, "travel")
            if tc: st.plotly_chart(tc, use_container_width=True)

    # Simple summary
    with st.expander("📊 Data Summary"):
        st.write(f"**Total Records Analyzed:** {total_records:,}")
        st.write(f"**Date Range:** {date_range[0]} to {date_range[1]} ({data_span} days)")
        if "average_speed" in filtered_data.columns and filtered_data["average_speed"].notna().any():
            st.write(f"**Average Speed:** {filtered_data['average_speed'].mean():.1f} mph")
        if "average_traveltime" in filtered_data.columns and filtered_data["average_traveltime"].notna().any():
            st.write(f"**Average Travel Time:** {filtered_data['average_traveltime'].mean():.1f} minutes")


# Kept for compatibility
def render_acyclica_section():
    st.warning("⚠️ This function has been replaced by the enhanced Tab 3 analysis with sidebar controls.")
