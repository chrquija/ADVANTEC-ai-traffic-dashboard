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


def render_tab3_analysis():
    """
    Main Tab 3 renderer for Acyclica travel time analysis.
    Matches the design and layout of Tabs 1 and 2.
    """
    try:
        # Load Acyclica data once to populate controls
        acyclica_df = get_acyclica_df()

        # DEBUG: Check what we actually got
        if acyclica_df.empty:
            st.error("❌ Acyclica dataframe is empty")
            st.info("🔍 Debug: Check if `get_acyclica_df()` is returning data")
            return
        else:
            # Show debug info temporarily
            st.sidebar.caption(f"🔍 Debug: Loaded {len(acyclica_df)} Acyclica records")
            st.sidebar.caption(f"🔍 Columns: {', '.join(acyclica_df.columns[:5])}...")

    except RuntimeError as e:
        st.error(f"❌ Failed to load Acyclica data: {e}")
        return
    except Exception as e:
        st.error(f"❌ Unexpected error loading Acyclica data: {e}")
        st.exception(e)  # Show full traceback for debugging
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
                st.caption("⚠️ No corridor_id column found")

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
                avail = compute_data_availability(
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
            except Exception as e:
                # Keep sidebar resilient if availability fails
                st.caption(f"⚠️ Availability check failed: {e}")

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
                direction_filter = st.selectbox("🔄 Direction Filter", direction_options,
                                                key="direction_filter_acyclica")

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
                            start_hour = st.number_input("Start Hour (0–23)", 0, 23, 7, step=1,
                                                         key="start_hour_acyclica")
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

        # DEBUG: Show data info
        st.info(f"🔍 DEBUG: Working with {len(working_df)} records")
        if not working_df.empty:
            st.info(f"🔍 DEBUG: Columns available: {list(working_df.columns)}")
            # Show sample data
            st.dataframe(working_df.head(3), use_container_width=True)

        # Filter by corridor
        if corridor != "All Corridors" and "corridor_id" in working_df.columns:
            working_df = working_df[working_df["corridor_id"] == corridor]
            st.info(f"🔍 DEBUG: After corridor filter: {len(working_df)} records")

        # Filter by direction
        if direction_filter != "All Directions" and "direction" in working_df.columns:
            working_df = working_df[working_df["direction"] == direction_filter]
            st.info(f"🔍 DEBUG: After direction filter: {len(working_df)} records")

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
            except Exception as e:
                # Error fallback
                st.markdown('<div class="cvag-map-card">', unsafe_allow_html=True)
                st.info(f"**Acyclica Corridor:** {corridor}")
                st.markdown("📍 Washington Street Corridor")
                st.markdown("_Acyclica sensors monitor travel time and speed along the corridor_")
                st.caption(f"Map error: {e}")
                st.markdown("</div>", unsafe_allow_html=True)

        # Left/main content
        with main_col_t3:
            with scoped_cad_loader("Fetching Data...", tab_id="t3") as step:
                step("Applying filters & aggregations", 20)

                # DEBUG: Check the data before processing
                st.info(f"🔍 DEBUG: Before processing - {len(working_df)} records")
                if not working_df.empty:
                    # Check if required columns exist
                    required_cols = ["average_traveltime", "average_speed", "local_datetime"]
                    missing_cols = [col for col in required_cols if col not in working_df.columns]
                    if missing_cols:
                        st.error(f"❌ Missing required columns: {missing_cols}")
                        st.info(f"Available columns: {list(working_df.columns)}")
                        return

                try:
                    filtered_data = process_traffic_data(
                        working_df,
                        date_range,
                        granularity,
                        time_filter if granularity == "Hourly" else None,
                        start_hour,
                        end_hour,
                    )
                except Exception as process_error:
                    st.error(f"❌ Error in process_traffic_data: {process_error}")
                    st.info("🔍 DEBUG: This might be the source of the 'nan min' values")
                    # Show the working_df to debug
                    st.dataframe(working_df.head(), use_container_width=True)
                    return

                if filtered_data.empty:
                    step("No data for selected filters", 100)
                    st.warning("⚠️ No data available for the selected filters.")
                    return

                # DEBUG: Check filtered data
                st.info(f"🔍 DEBUG: After processing - {len(filtered_data)} records")

                # Continue with the rest of the analysis...
                # (Rest of your existing code remains the same)

    except Exception as e:
        st.error(f"❌ Error processing Acyclica data: {e}")
        import traceback
        st.text("Debug info:")
        st.text(traceback.format_exc())