# Python
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Plotly for chart helpers
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add the Map import
from Map import build_all_segments_overview, build_intersections_overview

# Shared UI utils (scoped loader and tab highlight)
try:
    from ui_utils import cad_loader as scoped_cad_loader, set_active_search_tab, is_active_tab
except ModuleNotFoundError:
    from core.ui_utils import cad_loader as scoped_cad_loader, set_active_search_tab, is_active_tab

# Import ALL data functions from sidebar_functions (remove duplicates)
try:
    from sidebar_functions import (
        get_acyclica_df,
        get_acyclica_long_df,
        process_traffic_data,
        date_range_preset_controls,
        performance_chart,
        render_badge,
        get_performance_rating,
        compute_data_availability
    )
except ModuleNotFoundError:
    from core.sidebar_functions import (
        get_acyclica_df,
        get_acyclica_long_df,
        process_traffic_data,
        date_range_preset_controls,
        performance_chart,
        render_badge,
        get_performance_rating,
        compute_data_availability
    )

# Remove ALL duplicate functions - keep only the render function
def render_tab3_analysis():
    """
    Main Tab 3 renderer for Acyclica travel time analysis.
    Matches the design and layout of Tabs 1 and 2.
    """
    try:
        # Load Acyclica data once to populate controls
        acyclica_df = get_acyclica_df()  # Using imported function
    except RuntimeError as e:
        st.error(f"❌ Failed to load Acyclica data: {e}")
        return
    except Exception as e:
        st.error(f"❌ Unexpected error loading Acyclica data: {e}")
        return

    # -------- Sidebar controls (matching Tab 1 & 2 style) --------
    with st.sidebar:
        with st.expander("⚙️ Pg.3 ACYCLICA SETTINGS", expanded=False):

            # Data caption at top (before corridor selection)
            st.caption("Data: Travel Time, and Speed (Acyclica)")

            # Get available corridors
            if not acyclica_df.empty and "corridor_id" in acyclica_df.columns:
                corridors = ["SELECT", "All Corridors"] + sorted(acyclica_df["corridor_id"].dropna().unique().tolist())
            else:
                corridors = ["SELECT", "All Corridors"]

            st.markdown("## 🛣️ Select Corridor")
            corridor = st.selectbox("Corridor", corridors, key="corridor_acyclica")

            # Availability preview directly under the corridor selector (corridor-aware)
            try:
                base_df = acyclica_df if acyclica_df is not None else pd.DataFrame()
                if corridor == "SELECT":
                    corr_for_avail = None
                elif corridor == "All Corridors":
                    corr_for_avail = None
                else:
                    corr_for_avail = corridor
                avail = compute_data_availability(  # Using imported function
                    base_df if base_df is not None else pd.DataFrame(),
                    intersection_col="corridor_id",
                    intersection=corr_for_avail,
                    max_gaps=3,
                    current_date=datetime.now(),
                )
                if avail.get("start") and avail.get("end"):
                    start_str = avail["start"].strftime("%b %d, %Y %I:%M %p")
                    end_str = avail["end"].strftime("%b %d, %Y %I:%M %p")
                    mb = avail.get("size_mb", 0.0)
                    size_str = f"({mb:.1f} MB)" if mb > 0 else ""
                    if corridor == "SELECT":
                        header_label = "Available Data"
                    elif corridor == "All Corridors":
                        header_label = "Available Data for this Corridor"
                    else:
                        header_label = f"Available Data for {corridor}"
                    st.caption(header_label)
                    st.caption(f"• Date Range: {start_str} → {end_str} {size_str}")
                    gaps = avail.get("gaps") or []
                    if len(gaps) == 0:
                        st.caption("• Missing Data: None")
                    else:
                        st.caption("• Missing Data: " + "; ".join(gaps))
            except Exception:
                # Keep sidebar resilient if availability fails
                pass

            # Loading animation when corridor changes (only for real selections)
            prev_corridor = st.session_state.get("corridor_acyclica_prev")
            if prev_corridor != corridor:
                st.session_state["corridor_acyclica_prev"] = corridor
                if corridor != "SELECT":
                    pb = st.progress(0, text="Loading Data availability info...")
                    for i in range(0, 101, 10):
                        time.sleep(0.02)
                        pb.progress(i, text="Loading Data availability info...")
                    pb.empty()

            # Progressive disclosure: show controls only after a selection (including All Corridors)
            if corridor != "SELECT":
                # Date range
                if acyclica_df.empty or "local_datetime" not in acyclica_df.columns:
                    min_date = datetime.today().date() - timedelta(days=7)
                    max_date = datetime.today().date()
                else:
                    min_date = acyclica_df["local_datetime"].dt.date.min()
                    max_date = acyclica_df["local_datetime"].dt.date.max()

                st.markdown("## 📅 Date And Time")
                date_range = date_range_preset_controls(min_date, max_date, key_prefix="acyclica")  # Using imported function

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

                # Track uncommitted controls (current live sidebar values)
                t3_current = {
                    "corridor": corridor,
                    "date_range": tuple(date_range) if date_range else None,
                    "granularity": granularity,
                    "direction_filter": direction_filter,
                    "time_filter": time_filter,
                    "start_hour": start_hour,
                    "end_hour": end_hour,
                }
                st.session_state["t3_current"] = t3_current

                # Search button (matching other tabs)
                if st.button("🔍 **Search**", key="search_tab3", type="primary", use_container_width=True):
                    st.session_state["t3_ready"] = True
                    st.session_state["t3_params"] = t3_current
                    set_active_search_tab("t3")
                    st.session_state["last_active_tab"] = "t3"
            else:
                # Reset current minimal state when placeholder is selected
                st.session_state["t3_current"] = {"corridor": corridor}

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

    # Compare committed vs current sidebar values to detect pending changes
    t3_pending = t3_ready and (t3_params != st.session_state.get("t3_current", {}))
    if t3_pending:
        st.warning("⚙️ Press **Search** to refresh.")

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
            with scoped_cad_loader("Fetching Data...", tab_id="t3") as step:
                step("Applying filters & aggregations", 20)
                filtered_data = process_traffic_data(  # Using imported function
                    working_df,
                    date_range,
                    granularity,
                    time_filter if granularity == "Hourly" else None,
                    start_hour,
                    end_hour,
                )

            if filtered_data.empty:
                step("No data for selected filters", 100)
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
            st.subheader(" KPI's (Key Performance Indicators)")
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
                    st.markdown(render_badge(reliability), unsafe_allow_html=True)  # Using imported function

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
                    tt_chart = performance_chart(filtered_data, "travel")  # Using imported function
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
                            vertical_spacing=0.28,
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