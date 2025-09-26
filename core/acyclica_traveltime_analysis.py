# core/acyclica_traveltime_analysis.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

from sidebar_functions import (
    get_acyclica_df,
    process_traffic_data,
    compute_perf_kpis_interpretable,  # ok to keep
    render_badge,
    performance_chart,                 # ok to keep
    date_range_preset_controls,
    get_performance_rating,            # ok to keep
)

# Ensure we can import from Prediction package
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Prediction.timeline_scrubber import render_gradient_header
from Prediction.incident_detection import render_incident_detection_section


# ---------------------------
# Safe loader for Acyclica DF
# ---------------------------
def _safe_get_acyclica_df() -> pd.DataFrame:
    try:
        return get_acyclica_df()
    except Exception:
        try:
            from sidebar_functions import get_acyclica_long_df, acyclica_long_to_hourly
        except Exception:
            return pd.DataFrame()

        long_df = get_acyclica_long_df()
        if long_df is None or long_df.empty:
            return pd.DataFrame()

        if "direction" in long_df.columns:
            long_df["direction"] = long_df["direction"].astype(str).str.strip().str.upper()

        try:
            wide = acyclica_long_to_hourly(long_df)
        except Exception:
            return pd.DataFrame()

        if wide is None or wide.empty:
            return pd.DataFrame()

        if "local_datetime" in wide.columns:
            wide["local_datetime"] = pd.to_datetime(wide["local_datetime"], errors="coerce")
            wide = wide.dropna(subset=["local_datetime"])
        return wide


# ---------------------------
# KPI computation (speed‑focused)
# ---------------------------
def compute_acyclica_kpis(df: pd.DataFrame, low_speed_threshold: float) -> dict:
    if df is None or df.empty:
        return {
            "avg_tt": {"value": 0.0, "unit": "min", "score": 50.0, "help": "Average Travel Time"},
            "planning_time": {"value": 0.0, "unit": "min", "score": 50.0, "help": "Planning Time (95th)"},
            "buffer_index": {"value": 0.0, "unit": "%", "score": 50.0, "help": "Buffer Index"},
            "reliability": {"value": 0.0, "unit": "%", "score": 50.0, "help": "Reliability Index"},
            "low_speed_freq": {"value": 0.0, "unit": "%", "score": 50.0, "help": "Low Speed Frequency"},
        }

    for col in ["average_speed", "average_traveltime", "average_delay"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    avg_tt = float(df["average_traveltime"].mean()) if "average_traveltime" in df.columns and df["average_traveltime"].notna().any() else 0.0
    p95_tt = float(df["average_traveltime"].quantile(0.95)) if "average_traveltime" in df.columns and df["average_traveltime"].notna().any() else 0.0
    buffer_index = ((p95_tt - avg_tt) / avg_tt * 100.0) if avg_tt > 0 else 0.0

    if avg_tt > 0 and "average_traveltime" in df.columns:
        std_tt = float(df["average_traveltime"].std())
        cv_tt = (std_tt / avg_tt * 100.0) if std_tt > 0 else 0.0
    else:
        cv_tt = 0.0
    reliability = max(0.0, 100.0 - cv_tt)

    if "average_speed" in df.columns and df["average_speed"].notna().any():
        total_hours = int(df["average_speed"].count())
        low_speed_hours = int((df["average_speed"] < low_speed_threshold).sum())
        low_speed_freq = (low_speed_hours / total_hours * 100.0) if total_hours > 0 else 0.0
    else:
        low_speed_freq, low_speed_hours, total_hours = 0.0, 0, 0

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
            "help": "Average Travel Time\n\nThe typical trip time for your current filters. mean(travel_time).",
        },
        "planning_time": {
            "value": p95_tt,
            "unit": "min",
            "score": score_plan,
            "help": "Planning Time (95th): 95th-percentile of hourly trip times.",
        },
        "buffer_index": {
            "value": buffer_index,
            "unit": "%",
            "score": score_buffer,
            "help": "Buffer Index = (P95 − mean) / mean × 100%.",
        },
        "reliability": {
            "value": reliability,
            "unit": "%",
            "score": score_reliability,
            "help": "Reliability Index = 100 − CV% of travel time (higher is better).",
        },
        "low_speed_freq": {
            "value": low_speed_freq,
            "unit": "%",
            "score": score_low_speed,
            "extra": f"Hours < {low_speed_threshold:.0f}mph: {low_speed_hours}/{total_hours}",
            "help": f"Percent of hours with average speed below {low_speed_threshold:.0f} mph.",
        },
    }


# ---------------------------
# Charts (speed / travel time)
# ---------------------------
def speed_performance_chart(data: pd.DataFrame, metric_type: str = "speed"):
    if data.empty:
        return None

    for col in ["average_speed", "average_traveltime", "average_delay"]:
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


# ---------------------------
# Main analysis section (center panel)
# ---------------------------
def render_acyclica_section(acyclica_df: pd.DataFrame, aggregation_level: str = "Daily"):
    """
    Render the Acyclica travel time analysis in the MAIN page.
    `aggregation_level` controls how `process_traffic_data` groups the data.
    """
    if acyclica_df is None or acyclica_df.empty:
        st.error("❌ Failed to load Acyclica data. Please check your data sources.")
        return

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

    try:
        acyclica_df["local_datetime"] = pd.to_datetime(acyclica_df["local_datetime"], errors="coerce")
        acyclica_df = acyclica_df.dropna(subset=["local_datetime"])

        for col in ["average_traveltime", "average_speed", "average_delay"]:
            if col in acyclica_df.columns:
                acyclica_df[col] = pd.to_numeric(acyclica_df[col], errors="coerce")

        if "segment_name" not in acyclica_df.columns:
            if "corridor_id" in acyclica_df.columns and "direction" in acyclica_df.columns:
                acyclica_df["segment_name"] = (
                    acyclica_df["corridor_id"].astype(str) + " (" + acyclica_df["direction"].astype(str) + ")"
                )
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

    # Determine bounds for local date picker in the main area
    if "local_datetime" in acyclica_df.columns:
        min_date_ts = acyclica_df["local_datetime"].min()
        max_date_ts = acyclica_df["local_datetime"].max()
    else:
        min_date_ts = datetime.today() - timedelta(days=7)
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

    # Apply grouping/aggregation as selected
    try:
        filtered_data = process_traffic_data(
            acyclica_df,
            date_range,
            aggregation_level  # <-- now uses the sidebar choice
        )
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        filtered_data = pd.DataFrame()

    if filtered_data.empty:
        st.warning("⚠️ No Acyclica data available for the selected date range.")
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
    try:
        k = compute_acyclica_kpis(filtered_data, LOW_SPEED_THRESHOLD)
    except Exception as e:
        st.error(f"❌ Error computing KPIs: {str(e)}")
        return

    buffer_minutes = max(0.0, k["planning_time"]["value"] - k["avg_tt"]["value"])
    buffer_help = "Extra minutes to leave earlier so you arrive on time 95% of the time.\nFormula: Planning Time (95th) − Average Travel Time."

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("🎯 Reliability Index", f"{k['reliability']['value']:.0f}{k['reliability']['unit']}", help=k['reliability']['help'])
        st.markdown(render_badge(k['reliability']['score']), unsafe_allow_html=True)
    with c2:
        st.metric("🐌 Low Speed Frequency", f"{k['low_speed_freq']['value']:.1f}{k['low_speed_freq']['unit']}", help=k['low_speed_freq']['help'])
        st.caption(k['low_speed_freq'].get('extra', ''))
        st.markdown(render_badge(k['low_speed_freq']['score']), unsafe_allow_html=True)
    with c3:
        st.metric("⏱️ Average Travel Time", f"{k['avg_tt']['value']:.1f} {k['avg_tt']['unit']}", help=k['avg_tt']['help'])
        st.markdown(render_badge(k['avg_tt']['score']), unsafe_allow_html=True)
    with c4:
        st.metric("📈 Planning Time (95th Percentile)", f"{k['planning_time']['value']:.1f} {k['planning_time']['unit']}", help=k['planning_time']['help'])
        st.markdown(render_badge(k['planning_time']['score']), unsafe_allow_html=True)
    with c5:
        st.metric("🧭 Buffer Time (leave this much earlier)", f"{buffer_minutes:.1f} min", help=buffer_help)
        st.markdown(render_badge(k['buffer_index']['score']), unsafe_allow_html=True)

    # Trends
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

    # Bottleneck analysis
    st.subheader("🚨 Speed-Based Bottleneck Analysis")
    st.info("✨ **Acyclica Advantage**: Identifies bottlenecks using speed patterns instead of delay")

    if not filtered_data.empty:
        try:
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

            def _norm_speed(s):
                s = pd.to_numeric(s, errors="coerce")
                s_clean = s.dropna()
                if len(s_clean) == 0:
                    return pd.Series(np.zeros(len(s)), index=s.index)
                mn, mx = s_clean.min(), s_clean.max()
                if mx > mn:
                    return (s - mn) / (mx - mn)
                return pd.Series(np.zeros(len(s)), index=s.index)

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

    # Summary
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


# ---------------------------
# Wrapper that adds context line
# ---------------------------
def render_acyclica_section_with_settings(data, corridor, direction, aggregation):
    if data.empty:
        st.warning("⚠️ No data available for the selected filters. Try adjusting your date range or direction filter.")
        return

    st.info(f"📊 **Analysis Context**: {corridor} | {direction} | {aggregation} Aggregation")
    render_acyclica_section(data, aggregation_level=aggregation)


# ---------------------------
# TAB 3 ENTRY POINT
# ---------------------------
def render_tab3_analysis():
    """
    Tab 3 renderer: sidebar only collects inputs; the main page renders content.
    """
    # Load data BEFORE sidebar so any errors display in the main area
    try:
        acyclica_df = _safe_get_acyclica_df()
    except Exception as e:
        st.error(f"Error loading Acyclica data: {e}")
        st.stop()

    if acyclica_df is None or acyclica_df.empty:
        st.error("❌ Failed to load Acyclica data. Please check your data sources.")
        st.stop()

    # ---------------- Sidebar (controls only) ----------------
    with st.sidebar:
        with st.expander("⚙️ Pg.3 SETTINGS", expanded=True):
            st.caption("Acyclica Data: Speed + Travel Time Analysis")
            st.caption("AI Models: Peak Hour Prediction, Incident Detection, Event Impact Analysis")

            # Corridor
            st.markdown("## 🛣️ Select Corridor")
            corridor_options = ["Washington Street"]
            selected_corridor = st.selectbox(
                "Corridor",
                corridor_options,
                key="tab3_corridor",
                help="Currently focusing on Washington Street corridor"
            )

            # Direction
            st.markdown("## 🔄 Direction Filter")
            if "direction" in acyclica_df.columns and not acyclica_df.empty:
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

            # Date bounds for sidebar picker
            if "local_datetime" in acyclica_df.columns:
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
                st.session_state.pop("tab3_range", None)
                date_range = (max(min_date, max_date - timedelta(days=30)), max_date)
                st.info("Date range was reset to a safe default for this dataset.")

            # Aggregation
            st.markdown("## Data Aggregation")
            aggregation_options = ["Hourly", "Daily", "Weekly", "Monthly"]
            selected_aggregation = st.selectbox(
                "Aggregation Level",
                aggregation_options,
                index=1,  # default to Daily
                key="tab3_aggregation",
                help="Daily shows day-by-day patterns, Weekly shows weekly trends, Monthly shows long-term patterns."
            )

            # Analysis type
            st.markdown("## Select Analysis Type")
            analysis_type = st.selectbox(
                "Choose Analysis",
                [
                    "Travel Time Analysis",
                    "Peak Hour Prediction",
                    "Incident Detection & Recovery",
                    "Event Impact Analysis",
                ],
                key="tab3_analysis_type",
                help="Travel Time Analysis shows the same metrics as Tab 1 but with Acyclica data + speed insights.",
            )

    # --------------- Main area (render) ----------------
    filtered_acyclica_df = acyclica_df.copy()

    # Apply direction filter
    if selected_direction != "All Directions" and "direction" in filtered_acyclica_df.columns:
        dir_series = filtered_acyclica_df["direction"].astype(str)
        filtered_acyclica_df = filtered_acyclica_df[dir_series.str.upper() == str(selected_direction).upper()]

    # Apply date filter from the sidebar
    if date_range and len(date_range) == 2 and "local_datetime" in filtered_acyclica_df.columns:
        filtered_acyclica_df = filtered_acyclica_df[
            (filtered_acyclica_df["local_datetime"].dt.date >= date_range[0]) &
            (filtered_acyclica_df["local_datetime"].dt.date <= date_range[1])
        ]

    # Route to the selected analysis (this is now OUTSIDE the sidebar)
    if analysis_type == "Travel Time Analysis":
        render_acyclica_section_with_settings(
            filtered_acyclica_df,
            selected_corridor,
            selected_direction,
            selected_aggregation,
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
