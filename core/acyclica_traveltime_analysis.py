# Python
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import io

# Plotly for chart helpers
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add the Map import
from Map import build_all_segments_overview, build_intersections_overview, build_corridor_map

# Shared UI utils (scoped loader and tab highlight)
try:
    from ui_utils import cad_loader as scoped_cad_loader, set_active_search_tab, is_active_tab, get_dynamic_xaxis_params
except ModuleNotFoundError:
    from core.ui_utils import cad_loader as scoped_cad_loader, set_active_search_tab, is_active_tab, get_dynamic_xaxis_params

# Availability utility
try:
    from sidebar_functions import compute_data_availability
except ModuleNotFoundError:
    from core.sidebar_functions import compute_data_availability

# Import shared loaders/helpers from centralized sidebar_functions
try:
    from sidebar_functions import (
        get_acyclica_df,
        date_range_preset_controls,
        process_traffic_data,
        performance_chart,
        render_badge,
        compute_missing_strength_gaps,
    )
except ModuleNotFoundError:
    from core.sidebar_functions import (
        get_acyclica_df,
        date_range_preset_controls,
        process_traffic_data,
        performance_chart,
        render_badge,
        compute_missing_strength_gaps,
    )

# ------------------------------------------------------------------
# Highway 111 O→D helpers (Centralized)
# ------------------------------------------------------------------
def h111_valid_dests(o: str) -> list:
    if o == "Canyon Plaza West":
        return ["Jermaine Gibson"]
    if o == "Jermaine Gibson":
        return ["Canyon Plaza West"]
    if o == "Parkview Drive":
        return ["Cook Street"]
    if o == "Cook Street":
        return ["Parkview Drive", "Washington Street"]
    if o == "Washington Street":
        return ["Cook Street", "Monroe Street"]
    if o == "Monroe Street":
        return ["Washington Street", "Indio Blvd"]
    if o == "Indio Blvd":
        return ["Monroe Street"]
    # When origin is SELECT or unknown, allow all endpoints for discovery
    return [
        "Canyon Plaza West",
        "Jermaine Gibson",
        "Parkview Drive",
        "Cook Street",
        "Washington Street",
        "Monroe Street",
        "Indio Blvd",
    ]

def h111_resolve(o: str, d: str):
    # returns dict with: valid(bool), ui_dir(str|None), df_dir(str|None), pair(str)
    ui_dir = None
    df_dir = None
    valid = False
    pair = None
    if o == "Canyon Plaza West" and d == "Jermaine Gibson":
        ui_dir = "Eastbound"; df_dir = "EASTBOUND"; valid = True; pair = "CPW_JG"
    elif o == "Jermaine Gibson" and d == "Canyon Plaza West":
        ui_dir = "Westbound"; df_dir = "WESTBOUND"; valid = True; pair = "CPW_JG"
    elif o == "Parkview Drive" and d == "Cook Street":
        ui_dir = "Eastbound"; df_dir = "EASTBOUND"; valid = True; pair = "PV_CK"
    elif o == "Cook Street" and d == "Parkview Drive":
        ui_dir = "Westbound"; df_dir = "WESTBOUND"; valid = True; pair = "PV_CK"
    elif o == "Cook Street" and d == "Washington Street":
        ui_dir = "Eastbound"; df_dir = "EASTBOUND"; valid = True; pair = "CK_WS"
    elif o == "Washington Street" and d == "Cook Street":
        ui_dir = "Westbound"; df_dir = "WESTBOUND"; valid = True; pair = "CK_WS"
    elif o == "Washington Street" and d == "Monroe Street":
        ui_dir = "Eastbound"; df_dir = "EASTBOUND"; valid = True; pair = "WS_MS"
    elif o == "Monroe Street" and d == "Washington Street":
        ui_dir = "Westbound"; df_dir = "WESTBOUND"; valid = True; pair = "WS_MS"
    elif o == "Monroe Street" and d == "Indio Blvd":
        ui_dir = "Westbound"; df_dir = "WESTBOUND"; valid = True; pair = "MS_IB"
    elif o == "Indio Blvd" and d == "Monroe Street":
        ui_dir = "Eastbound"; df_dir = "EASTBOUND"; valid = True; pair = "MS_IB"
    return {"valid": valid, "ui_dir": ui_dir, "df_dir": df_dir, "pair": pair}

def h111_apply_filter(df: pd.DataFrame, o: str, d: str) -> pd.DataFrame:
    res = h111_resolve(o, d)
    if not res["valid"]:
        return df.iloc[0:0]
    if "segment_id" not in df.columns:
        return df
    seg_series = df["segment_id"].astype(str)
    seg_lower = seg_series.str.lower()
    if res["pair"] == "CPW_JG":
        ids = [
            "CanyonPlazaWest_to_JermaineGibsion",
            "CanyonPlazaWest_to_JermaineGibson",
            "JermaineGibson_to_CanyonPlazaWest",
        ]
        df = df[seg_series.isin(ids)]
    elif res["pair"] == "PV_CK":
        df = df[seg_lower.str.contains("parkview") & seg_lower.str.contains("cook")]
    elif res["pair"] == "CK_WS":
        exact_ids = ["CookStreet_to_WashingtonStreet", "WashingtonStreet_to_CookStreet"]
        present_exact = set(seg_series.unique()).intersection(exact_ids)
        if present_exact:
            df = df[seg_series.isin(list(present_exact))]
        else:
            df = df[seg_lower.str.contains("cook") & seg_lower.str.contains("washington")]
    elif res["pair"] == "WS_MS":
        exact_ids = ["WashingtonStreet_to_MonroeStreet", "MonroeStreet_to_WashingtonStreet"]
        present_exact = set(seg_series.unique()).intersection(exact_ids)
        if present_exact:
            df = df[seg_series.isin(list(present_exact))]
        else:
            df = df[seg_lower.str.contains("washington") & seg_lower.str.contains("monroe")]
    elif res["pair"] == "MS_IB":
        exact_ids = ["MonroeStreet_to_IndioBlvd", "IndioBlvd_to_MonroeStreet"]
        present_exact = set(seg_series.unique()).intersection(exact_ids)
        if present_exact:
            df = df[seg_series.isin(list(present_exact))]
        else:
            df = df[seg_lower.str.contains("monroe") & seg_lower.str.contains("indio")]
    
    # Direction refinement (uppercase in DF)
    if "direction" in df.columns and res["df_dir"]:
        df = df[df["direction"] == res["df_dir"]]
    return df

# ------------------------------------------------------------------
# Tab 3: Acyclica Travel Time Analysis (UI + logic only)
# ------------------------------------------------------------------




# Removed duplicate helpers and charts. Using centralized implementations from sidebar_functions.

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
                avail = compute_data_availability(
                    base_df if base_df is not None else pd.DataFrame(),
                    intersection_col="corridor_id",
                    intersection=corr_for_avail,
                    # Keep the Missing Data output concise like Pg.2 expander
                    max_gaps=1,
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
                    # Compact Missing Data display (match Pg.2 style)
                    tail_gap = avail.get("tail_gap")
                    gaps = avail.get("gaps") or []
                    if tail_gap:
                        st.caption(f"• Missing Data: {tail_gap}")
                    elif len(gaps) == 0:
                        st.caption("• Missing Data: None")
                    else:
                        first_gap = gaps[0]
                        more = len(gaps) - 1
                        suffix = f" (+{more} more)" if more > 0 else ""
                        st.caption(f"• Missing Data: {first_gap}{suffix}")
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
                # Optional O-D filter for specific corridors
                origin = destination = "SELECT"
                if corridor == "Washington Street":
                    st.markdown("## 🚦 Origin → Destination")
                    # Include Highway111 as a mid-corridor endpoint
                    od_options = ["SELECT", "Avenue 52", "Highway111", "Avenue 41", "Harris Lane", "Country Club Drive", "I-10 Interchange", "Varner Rd", "Market Pl", "Del Webb"]
                    oc, dc = st.columns(2)
                    with oc:
                        origin = st.selectbox("Origin", od_options, index=0, key="acyclica_od_origin")
                    with dc:
                        destination = st.selectbox("Destination", od_options, index=0, key="acyclica_od_destination")
                    # Sidebar route summary right under O-D
                    def _route_text_sb(corr: str, o: str, d: str) -> str:
                        if not o or not d or o == "SELECT" or d == "SELECT" or o == d:
                            return ""
                        # Node order for NB/SB inference
                        order = [
                            "Avenue 52","Calle Tampico","Village Shopping Ctr","Avenue 50","Sagebrush Ave","Eisenhower Dr",
                            "Avenue 48","Avenue 47","Point Happy Simon","Hwy 111","Channel Drive","Miles Avenue","Via Sevilla",
                            "Fred Waring Drive","Palm Royale Drive","Avenue of the States","Avenue 42","Avenue 41","Harris Lane","Country Club Drive",
                            "I-10 Interchange", "Varner Rd", "Market Pl", "Del Webb"
                        ]
                        o_fix = "Hwy 111" if o == "Highway111" else o
                        d_fix = "Hwy 111" if d == "Highway111" else d
                        friendly_dir = None
                        if o_fix in order and d_fix in order and order.index(o_fix) < order.index(d_fix):
                            friendly_dir = "Northbound"
                        elif o_fix in order and d_fix in order and order.index(o_fix) > order.index(d_fix):
                            friendly_dir = "Southbound"
                        if not friendly_dir:
                            return ""
                        return f"{o} → {d} ({friendly_dir})"

                    _pill = _route_text_sb(corridor, origin, destination)
                    if _pill:
                        st.markdown(
                            f"""
                            <div style="background:#e8f2ff;border:1px solid #c7dcff;color:#163f7a;padding:8px 12px;border-radius:10px;
                                        display:inline-block;font-weight:700;margin:6px 0 6px;">{_pill}</div>
                            """,
                            unsafe_allow_html=True,
                        )
                elif corridor == "Highway 111":
                    # -------------------------
                    # Highway 111 O→D resolver (single source of truth)
                    # -------------------------
                    od_endpoints = [
                        "Canyon Plaza West",
                        "Jermaine Gibson",
                        "Parkview Drive",
                        "Cook Street",
                        "Washington Street",
                        "Monroe Street",
                        "Indio Blvd",
                    ]

                    # Ensure session keys exist
                    if "acyclica_od_origin_h111" not in st.session_state:
                        st.session_state["acyclica_od_origin_h111"] = "SELECT"
                    if "acyclica_od_destination_h111" not in st.session_state:
                        st.session_state["acyclica_od_destination_h111"] = "SELECT"

                    oc, dc = st.columns(2)
                    with oc:
                        origin = st.selectbox(
                            "Origin",
                            ["SELECT"] + od_endpoints,
                            index=(0 if st.session_state.get("acyclica_od_origin_h111", "SELECT") == "SELECT" else (["SELECT"] + od_endpoints).index(st.session_state.get("acyclica_od_origin_h111"))),
                            key="acyclica_od_origin_h111",
                        )

                    # Constrain destination based on origin
                    valid_dests = h111_valid_dests(origin)
                    dest_options = ["SELECT"] + valid_dests
                    # Auto-reset destination if invalid
                    cur_dest = st.session_state.get("acyclica_od_destination_h111", "SELECT")
                    if cur_dest not in dest_options:
                        st.session_state["acyclica_od_destination_h111"] = "SELECT"
                        cur_dest = "SELECT"
                    with dc:
                        destination = st.selectbox(
                            "Destination",
                            dest_options,
                            index=dest_options.index(cur_dest) if cur_dest in dest_options else 0,
                            key="acyclica_od_destination_h111",
                        )

                    # Sidebar route summary right under O-D (uses resolver)
                    res = h111_resolve(origin, destination)
                    if res.get("valid") and res.get("ui_dir"):
                        _pill_h = f"{origin} → {destination} ({res['ui_dir']})"
                        st.markdown(
                            f"""
                            <div style="background:#e8f2ff;border:1px solid #c7dcff;color:#163f7a;padding:8px 12px;border-radius:10px;
                                        display:inline-block;font-weight:700;margin:6px 0 6px;">{_pill_h}</div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Corridor + O→D specific availability preview (uses resolver filtering)
                    try:
                        if res.get("valid"):
                            od_df = acyclica_df.copy()
                            od_df = od_df[(od_df.get("corridor_id") == "Highway 111")]
                            od_df = h111_apply_filter(od_df, origin, destination)
                            if not od_df.empty and "local_datetime" in od_df.columns:
                                _min = pd.to_datetime(od_df["local_datetime"], errors="coerce").min()
                                _max = pd.to_datetime(od_df["local_datetime"], errors="coerce").max()
                                if pd.notnull(_min) and pd.notnull(_max):
                                    st.caption(
                                        f"Available Data for this O→D: { _min.strftime('%b %d, %Y %I:%M %p') } → { _max.strftime('%b %d, %Y %I:%M %p') }"
                                    )
                                # For Cook ↔ Washington, also reveal detected exact segment_id(s) if present
                                if {origin, destination} == {"Cook Street", "Washington Street"} and "segment_id" in od_df.columns:
                                    ids_found = sorted(set(od_df["segment_id"].dropna().astype(str).unique().tolist()))
                                    if ids_found:
                                        st.caption("Detected segment_id(s) used: " + ", ".join(ids_found))
                    except Exception:
                        pass

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

                # Direction filter removed for Tab 3. Direction is implied by the selected Origin → Destination.
                direction_filter = "All Directions"

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
                    "origin": origin,
                    "destination": destination,
                    "date_range": tuple(date_range) if date_range else None,
                    "granularity": granularity,
                    "direction_filter": direction_filter,
                    "time_filter": time_filter,
                    "start_hour": start_hour,
                    "end_hour": end_hour,
                }
                st.session_state["t3_current"] = t3_current

                # Generate button (matching other tabs)
                if st.button("🔍 **Generate**", key="search_tab3", type="primary", use_container_width=True):
                    st.session_state["t3_ready"] = True
                    st.session_state["t3_params"] = t3_current
                    set_active_search_tab("t3")
                    st.session_state["last_active_tab"] = "t3"
                    # Persist committed search to URL
                    try:
                        ds, de = None, None
                        if t3_current.get("date_range"):
                            ds = t3_current["date_range"][0].isoformat()
                            de = t3_current["date_range"][1].isoformat()
                        st.query_params.update(
                            t3_ready="1",
                            t3_corridor=corridor,
                            t3_origin=origin or "",
                            t3_destination=destination or "",
                            t3_date_start=ds or "",
                            t3_date_end=de or "",
                            t3_granularity=granularity,
                            t3_direction=direction_filter,
                            t3_time_filter=t3_current.get("time_filter") or "",
                            t3_start_hour=str(t3_current.get("start_hour") or ""),
                            t3_end_hour=str(t3_current.get("end_hour") or ""),
                            last_tab="t3",
                        )
                    except Exception:
                        pass
            else:
                # Reset current minimal state when placeholder is selected
                st.session_state["t3_current"] = {"corridor": corridor}

    # -------- Main content area (only render after Generate) --------
    t3_ready = st.session_state.get("t3_ready", False)

    if not t3_ready:
        st.info("Choose your Corridor and Date Range in the settings to the left.")
        return

    t3_params = st.session_state.get("t3_params", {})
    corridor = t3_params.get("corridor", "All Corridors")
    origin = t3_params.get("origin", "SELECT")
    destination = t3_params.get("destination", "SELECT")
    date_range = t3_params.get("date_range")
    granularity = t3_params.get("granularity", "Hourly")
    direction_filter = t3_params.get("direction_filter", "All Directions")
    time_filter = t3_params.get("time_filter")
    start_hour = t3_params.get("start_hour")
    end_hour = t3_params.get("end_hour")

    # Compare committed vs current sidebar values to detect pending changes
    t3_pending = t3_ready and (t3_params != st.session_state.get("t3_current", {}))
    if t3_pending:
        st.warning("⚙️ Press **Generate** to refresh.")

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

        # Apply optional O-D mapping for Washington Street. Prefer segment_id filtering when available.
        if corridor == "Washington Street" and origin != "SELECT" and destination != "SELECT" and origin != destination:
            # Map O-D pairs to specific segment_ids when they represent mid-corridor legs
            od_to_segment = {
                ("Avenue 52", "Highway111"): "Avenue52_to_Highway111",
                ("Highway111", "Avenue 52"): "Highway111_to_Avenue52",
                ("Highway111", "Country Club Drive"): "Highway111_to_CountryClubDrive",
                ("Country Club Drive", "Highway111"): "CountryClubDrive_to_Highway111",
            }
            seg = od_to_segment.get((origin, destination))
            if seg and "segment_id" in working_df.columns:
                working_df = working_df[working_df["segment_id"].astype(str) == seg]
                # After segment selection, infer direction loosely for labeling if still all directions
                if direction_filter == "All Directions" and "direction" in working_df.columns and not working_df.empty:
                    # Keep unique direction if there's only one
                    dirs = working_df["direction"].dropna().unique().tolist()
                    if len(dirs) == 1:
                        direction_filter = dirs[0]
            else:
                # Fall back to direction-based mapping for end-to-end O-D
                if origin == "Avenue 52" and destination == "Country Club Drive":
                    direction_filter = "NB"
                elif origin == "Country Club Drive" and destination == "Avenue 52":
                    direction_filter = "SB"

        # Highway 111 specific O→D filtering
        if corridor == "Highway 111" and origin != "SELECT" and destination != "SELECT" and origin != destination:
            working_df = h111_apply_filter(working_df, origin, destination)

        # Filter by direction
        if direction_filter != "All Directions" and "direction" in working_df.columns:
            working_df = working_df[working_df["direction"] == direction_filter]

        # Guard: if no data after filters, exit gracefully to avoid rendering errors
        if working_df is None or len(working_df) == 0:
            # Attempt to hint available date span at the O→D level within this corridor
            hint = ""
            try:
                # Build an O→D-level availability hint using the base corridor filter
                base_df = acyclica_df.copy()
                base_df = base_df[(base_df.get("corridor_id") == corridor)] if corridor != "All Corridors" else base_df
                if corridor == "Highway 111" and origin != "SELECT" and destination != "SELECT" and origin != destination:
                    base_df = h111_apply_filter(base_df, origin, destination)

                if not base_df.empty and "local_datetime" in base_df.columns:
                    dt_min = pd.to_datetime(base_df["local_datetime"], errors="coerce").min()
                    dt_max = pd.to_datetime(base_df["local_datetime"], errors="coerce").max()
                    if pd.notnull(dt_min) and pd.notnull(dt_max):
                        hint = f"\n• Available data for this selection: {dt_min.strftime('%b %d, %Y')} → {dt_max.strftime('%b %d, %Y')}"
            except Exception:
                pass
            st.warning(
                "⚠️ No Acyclica data found for the selected corridor and O→D. Try adjusting the date range or O-D pair." + hint
            )
            return

        # ---------- Layout: wide content + sticky right rail (matching Tabs 1 & 2) ----------
        main_col_t3, right_col_t3 = st.columns([7, 3.5], gap="large")

        # Right rail (map code - now with actual map!)
        with right_col_t3:
            st.markdown('<div id="acyclica-map-anchor"></div>', unsafe_allow_html=True)
            st.markdown("##### Corridor Map", help="Stays visible while you scroll the analysis on the left.")

            satellite_t3 = st.toggle("🛰️ Satellite View", key="satellite_t3", value=False)
            # Prefer an O→D corridor map with green START and red END dots when O-D is selected; otherwise show overview
            try:
                fig_corridor = None
                # Attempt O-D route map when possible
                od_ok = (
                    corridor in ("Washington Street", "Highway 111")
                    and origin not in (None, "SELECT")
                    and destination not in (None, "SELECT")
                    and origin != destination
                )
                if od_ok:
                    o = origin
                    d = destination
                    # Map UI labels to Map.py node keys when needed
                    def _fix_node(n: str) -> str:
                        if n == "Highway111":
                            return "Hwy 111"
                        return n
                    o = _fix_node(o)
                    d = _fix_node(d)
                    try:
                        fig_corridor = build_corridor_map(o, d, satellite=satellite_t3)
                    except Exception:
                        fig_corridor = None

                # Fallback to the corridor overview (all segments)
                if not fig_corridor:
                    fig_corridor = build_all_segments_overview(satellite=satellite_t3)

                if fig_corridor:
                    try:
                        # Match the map height used in other tabs
                        fig_corridor.update_layout(height=1100, margin=dict(l=0, r=0, t=32, b=0))
                    except Exception:
                        pass

                    st.markdown('<div class="cvag-map-card">', unsafe_allow_html=True)
                    st.plotly_chart(fig_corridor, use_container_width=True, config={"displaylogo": False, "displayModeBar": True})
                    st.caption(f"**Acyclica Corridor:** {corridor}")
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    # Fallback: try the intersections overview
                    fig_intersections = build_intersections_overview(corridor=corridor, satellite=satellite_t3)
                    if fig_intersections:
                        try:
                            fig_intersections.update_layout(height=1100, margin=dict(l=0, r=0, t=32, b=0))
                        except Exception:
                            pass
                        st.markdown('<div class="cvag-map-card">', unsafe_allow_html=True)
                        st.plotly_chart(fig_intersections, use_container_width=True, config={"displaylogo": False, "displayModeBar": True})
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
                filtered_data = process_traffic_data(
                working_df,
                date_range,
                granularity,
                time_filter if granularity == "Hourly" else None,
                start_hour,
                end_hour,
            )

            if filtered_data.empty:
                step("No data for selected filters", 100)
                # Provide a more actionable message including O→D availability if we have it
                try:
                    dt_min = pd.to_datetime(working_df["local_datetime"], errors="coerce").min() if "local_datetime" in working_df else None
                    dt_max = pd.to_datetime(working_df["local_datetime"], errors="coerce").max() if "local_datetime" in working_df else None
                    if pd.notnull(dt_min) and pd.notnull(dt_max):
                        st.warning(
                            f"⚠️ No data for the selected date/time filters. Available range for this O→D is {dt_min.strftime('%b %d, %Y')} → {dt_max.strftime('%b %d, %Y')}."
                        )
                        return
                except Exception:
                    pass
                st.warning("⚠️ No data available for the selected filters.")
                return

            # Display header (matching Tab 1 & 2 style)
            total_records = len(filtered_data)
            data_span = (date_range[1] - date_range[0]).days + 1
            time_context = f" • {time_filter}" if (granularity == "Hourly" and time_filter) else ""
            
            # Map UI direction labels
            dir_display = "Northbound" if direction_filter == "NB" else "Southbound" if direction_filter == "SB" else direction_filter

            # Choose a display direction for the header subtext: prefer inferred direction from O→D
            def _infer_dir(corr: str, o: str, d: str, dir_filt: str) -> str:
                if not o or not d or o == "SELECT" or d == "SELECT" or o == d:
                    return dir_filt
                if corr == "Washington Street":
                    order = [
                        "Avenue 52","Calle Tampico","Village Shopping Ctr","Avenue 50","Sagebrush Ave","Eisenhower Dr",
                        "Avenue 48","Avenue 47","Point Happy Simon","Hwy 111","Channel Drive","Miles Avenue","Via Sevilla",
                        "Fred Waring Drive","Palm Royale Drive","Avenue of the States","Avenue 42","Avenue 41","Harris Lane","Country Club Drive",
                        "I-10 Interchange", "Varner Rd", "Market Pl", "Del Webb"
                    ]
                    o_fix = "Hwy 111" if o == "Highway111" else o
                    d_fix = "Hwy 111" if d == "Highway111" else d
                    if o_fix in order and d_fix in order and order.index(o_fix) < order.index(d_fix):
                        return "Northbound"
                    if o_fix in order and d_fix in order and order.index(o_fix) > order.index(d_fix):
                        return "Southbound"
                elif corr == "Highway 111":
                    if o == "Canyon Plaza West" and d == "Jermaine Gibson":
                        return "Eastbound"
                    if o == "Jermaine Gibson" and d == "Canyon Plaza West":
                        return "Westbound"
                    # New segment mappings: Parkview Drive ↔ Cook Street
                    if o == "Parkview Drive" and d == "Cook Street":
                        return "Eastbound"
                    if o == "Cook Street" and d == "Parkview Drive":
                        return "Westbound"
                    # Newly added segment: Cook Street ↔ Washington Street
                    if o == "Cook Street" and d == "Washington Street":
                        return "Eastbound"
                    if o == "Washington Street" and d == "Cook Street":
                        return "Westbound"
                    # Newly added segment: Washington Street ↔ Monroe Street
                    if o == "Washington Street" and d == "Monroe Street":
                        return "Eastbound"
                    if o == "Monroe Street" and d == "Washington Street":
                        return "Westbound"
                # Fallback to filter value mapping
                mapping = {"NB": "Northbound", "SB": "Southbound", "EB": "Eastbound", "WB": "Westbound"}
                return mapping.get(dir_filt, dir_filt)

            display_dir = _infer_dir(corridor, origin, destination, direction_filter)

            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #2b77e5 0%, #19c3e6 100%);
                    border-radius:16px; padding:18px 20px; color:#fff; margin:8px 0 14px;
                    box-shadow:0 10px 26px rgba(25,115,210,.25); text-align:left;
                    font-family: inherit;">
                  <div style="display:flex; align-items:center; gap:10px;">
                    <div>
                        <div style="font-size:1.7rem; font-weight:800; letter-spacing:-0.01em; margin-bottom:2px;">
                          <span style="color: #ffffff;">2025 BNP PARIBUS OPEN INDIAN WELLS DASHBOARD</span><span style="color: rgba(255,255,255,0.7);">: FLIR ACYCLICA TRAVEL TIME ANALYSIS</span>
                        </div>
                        <div style="font-size:1.1rem;font-weight:600;opacity:0.9; display:flex; flex-wrap:wrap; gap:15px; margin-top:4px; align-items:center;">
                          <div style="background:rgba(255,255,255,0.15); padding:4px 12px; border-radius:8px; font-size:1.2rem; border:1px solid rgba(255,255,255,0.3); font-weight:700;">
                            📅 {date_range[0].strftime('%b %d, %Y')} to {date_range[1].strftime('%b %d, %Y')} <span style="font-weight:400; opacity:0.8;">({data_span} days)</span>
                          </div>
                          <div style="background:rgba(255,255,255,0.15); padding:4px 12px; border-radius:8px; font-size:1.2rem; border:1px solid rgba(255,255,255,0.3);">
                            <span style="opacity:0.9; font-weight:400;">Corridor Segment:</span> <span style="font-weight:800;">{corridor}{f" ({origin} → {destination})" if (origin != "SELECT" and destination != "SELECT") else ""}</span> <span style="background:rgba(255,255,255,0.2); padding:2px 8px; border-radius:4px; margin-left:8px; font-size:0.9rem; font-weight:700; color:#fff;">{display_dir}</span>
                          </div>
                        </div>
                        <div style="font-size:0.95rem; opacity:0.85; display:flex; flex-wrap:wrap; gap:15px; margin-top:10px; align-items:center;">
                          <div style="display:flex; align-items:center; gap:15px;">
                            <div><span style="opacity:0.8; font-weight:400;">Region:</span> Coachella Valley</div>
                            <div style="border-left: 1px solid rgba(255,255,255,0.3); padding-left: 15px;"><span style="opacity:0.8; font-weight:400;">City:</span> Indian Wells / La Quinta</div>
                            <div style="border-left: 1px solid rgba(255,255,255,0.3); padding-left: 15px;"><span style="opacity:0.8; font-weight:400;">Corridor:</span> {corridor}</div>
                            <div style="border-left: 1px solid rgba(255,255,255,0.3); padding-left: 15px;">📊 {granularity} Aggregation{time_context}</div>
                            <div style="border-left: 1px solid rgba(255,255,255,0.3); padding-left: 15px;">✅ {total_records:,} data points</div>
                          </div>
                        </div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Removed separate route summary pill under the main title; it now appears inside the title.

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

                # Calculate LOS based on Average Speed
                if avg_speed > 35: los_letter = "A"
                elif avg_speed >= 28: los_letter = "B"
                elif avg_speed >= 22: los_letter = "C"
                elif avg_speed >= 17: los_letter = "D"
                elif avg_speed >= 13: los_letter = "E"
                elif avg_speed > 0: los_letter = "F"
                else: los_letter = "N/A"

                # Calculate Worst-case LOS based on Planning Time
                worst_speed_val = avg_speed * (avg_tt / p95_tt) if p95_tt > 0 else 0.0
                if worst_speed_val > 35: los_letter_worst = "A"
                elif worst_speed_val >= 28: los_letter_worst = "B"
                elif worst_speed_val >= 22: los_letter_worst = "C"
                elif worst_speed_val >= 17: los_letter_worst = "D"
                elif worst_speed_val >= 13: los_letter_worst = "E"
                elif worst_speed_val > 0: los_letter_worst = "F"
                else: los_letter_worst = "N/A"

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
                        help="Average travel time across the selected period. Used to estimate corridor Level of Service (LOS) per HCM 6th Edition (TRB, 2016) urban arterial standards based on average travel speed. LOS A = over 35 mph (free flow), LOS B = 28 to 35 mph (minor delay), LOS C = 22 to 28 mph (stable flow), LOS D = 17 to 22 mph (approaching capacity), LOS E = 13 to 17 mph (unstable flow), LOS F = under 13 mph (breakdown). Source: Highway Capacity Manual, 6th Edition, Transportation Research Board. https://www.trb.org/Main/Blurbs/175169.aspx"
                    )
                    st.markdown(render_badge(tt_score), unsafe_allow_html=True)
                    st.caption(f"Estimated Corridor LOS: {los_letter}")

                with col3:
                    st.metric(
                        "📈 Planning Time (95th Percentile)",
                        f"{p95_tt:.1f} min",
                        help="95th percentile travel time — only 5% of trips are slower than this. Used to estimate worst-case corridor Level of Service (LOS) per HCM 6th Edition (TRB, 2016). This represents the reliability ceiling: LOS A = over 35 mph, LOS B = 28 to 35 mph, LOS C = 22 to 28 mph, LOS D = 17 to 22 mph, LOS E = 13 to 17 mph, LOS F = under 13 mph. If this LOS is significantly worse than the Average Travel Time LOS, the corridor suffers from unreliable, variable conditions. Source: Highway Capacity Manual, 6th Edition, TRB. https://www.trb.org/Main/Blurbs/175169.aspx"
                    )
                    st.markdown(render_badge(planning_score), unsafe_allow_html=True)
                    st.caption(f"Worst-case Corridor LOS: {los_letter_worst}")

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
                st.caption("Shaded bands indicate periods with no data.")

                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    # Travel Time Chart
                    tt_chart = performance_chart(filtered_data, "travel", direction_label=dir_display)
                    if tt_chart:
                        st.plotly_chart(tt_chart, use_container_width=True, config={"displaylogo": False, "displayModeBar": True})

                with chart_col2:
                    # Speed Chart
                    if "average_speed" in filtered_data.columns:
                        speed_data = filtered_data.dropna(subset=["local_datetime", "average_speed"]).sort_values(
                            "local_datetime")

                        # Distribution stats for annotation and vertical lines
                        s_speed = speed_data["average_speed"]
                        mean_speed = float(s_speed.mean()) if not s_speed.empty else 0.0
                        p5 = float(s_speed.quantile(0.05)) if not s_speed.empty else 0.0
                        p25 = float(s_speed.quantile(0.25)) if not s_speed.empty else 0.0
                        p75 = float(s_speed.quantile(0.75)) if not s_speed.empty else 0.0
                        p95 = float(s_speed.quantile(0.95)) if not s_speed.empty else 0.0
                        dist_annotation = f"Most hours: {p25:.1f}–{p75:.1f} mph. Worst 5%: below {p5:.1f} mph."

                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=(f"{dir_display} Speed Time Series Analysis".strip(), f"Speed Distribution Analysis. {dist_annotation}"),
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
                                hovertemplate="%{x|%b %d, %Y %I:%M %p}\n🚗 Avg Speed: %{y:.1f} mph  ",
                            ),
                            row=1, col=1,
                        )

                        # Shade missing-data gaps on the speed time-series panel (row 1)
                        try:
                            times = pd.to_datetime(speed_data["local_datetime"]).sort_values().reset_index(drop=True)
                            if len(times) >= 3:
                                deltas = times.diff().dropna()
                                med = deltas.median()
                                if pd.notna(med) and med > pd.Timedelta(0):
                                    gap_threshold = med * 1.5
                                    gap_spans = []
                                    for i in range(1, len(times)):
                                        dt = times[i] - times[i - 1]
                                        if dt > gap_threshold:
                                            gap_spans.append((times[i - 1], times[i]))
                                    for start, end in gap_spans:
                                        fig.add_vrect(
                                            x0=start,
                                            x1=end,
                                            fillcolor="#95a5a6",
                                            opacity=0.18,
                                            line_width=0,
                                            layer="below",
                                            row=1,
                                            col=1,
                                        )
                        except Exception:
                            pass

                        # Distribution histogram
                        fig.add_trace(
                            go.Histogram(
                                x=speed_data["average_speed"],
                                nbinsx=30,
                                name="Speed Distribution",
                                marker_color="#2ecc71",
                                opacity=0.75,
                                hovertemplate="%{y} hours had a speed between %{x} mph  <extra></extra>",
                            ),
                            row=2, col=1,
                        )

                        # Add vertical reference lines for mean and 5th percentile
                        fig.add_vline(x=mean_speed, line_dash="dash", line_color="black", annotation_text="Avg", annotation_position="top right", row=2, col=1)
                        fig.add_vline(x=p5, line_dash="dot", line_color="#333", annotation_text="Worst 5%", annotation_position="top right", row=2, col=1)

                        fig.update_layout(
                            height=600,
                            title="Speed Analysis",
                            showlegend=True,
                            template="plotly_white",
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                        )
                        # Custom X-axis logic for better labels on long ranges
                        if not speed_data.empty:
                            start_date = speed_data["local_datetime"].min()
                            end_date = speed_data["local_datetime"].max()
                            params = get_dynamic_xaxis_params(start_date, end_date)
                            fig.update_xaxes(
                                title=dict(text="Date/Time", font=dict(size=16, weight="bold")),
                                tickfont=dict(size=14),
                                dtick=params["dtick"],
                                tickformat=params["tickformat"],
                                row=1, col=1
                            )
                        else:
                            fig.update_xaxes(
                                title=dict(text="Date/Time", font=dict(size=16, weight="bold")),
                                tickfont=dict(size=14),
                                dtick=21600000,
                                tickformat="%b %d\n%I:%M %p",
                                row=1, col=1
                            )
                        fig.update_yaxes(title=dict(text="Average Speed (mph)", font=dict(size=16, weight="bold")), tickfont=dict(size=14), row=1, col=1)
                        fig.update_xaxes(title=dict(text="Average Speed (mph)", font=dict(size=16, weight="bold")), tickfont=dict(size=14), row=2, col=1)
                        fig.update_yaxes(title=dict(text="Frequency (Number of Hours)", font=dict(size=16, weight="bold")), tickfont=dict(size=14), row=2, col=1)

                        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "displayModeBar": True})

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

                # --- NEW: Missing Data Button ---
                # Use current sidebar params for file naming and gap calculation
                t3_p = st.session_state.get("t3_params", {})
                corr_name = t3_p.get("corridor", "all").replace(" ", "_")
                gran = t3_p.get("granularity", "Hourly")
                d_start = date_range[0].strftime("%Y%m%d")
                d_end = date_range[1].strftime("%Y%m%d")

                missing_df = compute_missing_strength_gaps(
                    filtered_data,
                    pd.to_datetime(date_range[0]),
                    pd.to_datetime(date_range[1]) + pd.Timedelta(hours=23),
                    gran
                )

                # --- NEW: Styled Excel Download ---
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    missing_df.to_excel(writer, index=False, sheet_name='Missing Data Gaps')
                    workbook = writer.book
                    worksheet = writer.sheets['Missing Data Gaps']

                    # Define formats
                    header_format = workbook.add_format({
                        'bold': True,
                        'text_wrap': True,
                        'valign': 'vcenter',
                        'fg_color': '#E3F2FD',  # Light Blue
                        'border': 1
                    })

                    # Apply header format
                    for col_num, value in enumerate(missing_df.columns.values):
                        worksheet.write(0, col_num, value, header_format)

                    # Auto-adjust column widths
                    for i, col in enumerate(missing_df.columns):
                        column_len = max(missing_df[col].astype(str).str.len().max(), len(col)) + 2
                        worksheet.set_column(i, i, column_len)

                st.download_button(
                    "⬇️ Missing Data (XLSX)",
                    data=output.getvalue(),
                    file_name=f"missing_data_{corr_name}_{d_start}_{d_end}_{gran}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Reports contiguous gaps where Strength data is missing or null in a styled Excel format."
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
