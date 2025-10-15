# Boschtab.py - Tab 5: Bosch Multimodal Traffic Analysis

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import shared utilities
from sidebar_functions import (
    date_range_preset_controls,
    render_badge,
    get_performance_rating,
)
from cycle_length_recommendations import render_cycle_length_section
from Map import build_all_segments_overview

# Shared UI utils (scoped loader and tab highlight)
try:
    from ui_utils import cad_loader as scoped_cad_loader, set_active_search_tab, is_active_tab
except ModuleNotFoundError:
    from core.ui_utils import cad_loader as scoped_cad_loader, set_active_search_tab, is_active_tab

# =========================
# Constants
# =========================
THEORETICAL_LINK_CAPACITY_VPH = 1800
HIGH_VOLUME_THRESHOLD_VPH = 1200

# ---- Raw CSV URLs (update these if file names move) ----
BOSCH_RAW_URLS = {
    # Washington Street (multi-site file)
    "Washington Street": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/Bosch/WashingtonSt_1hr_Speed_Volume_Long.csv",
    # Jefferson St (Fred Waring)
    "Fred Waring": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/Bosch/JeffersonSt_1hr_Speed_Volume_Long.csv",
}

# ---- Canonical segment name mapping (Bosch device IDs → human-friendly)
SEGMENT_ID_ALIASES = {
    # Washington Street cameras
    "PTZ LQ-073 HWY111-Wash": "Washington Street and Hwy 111",
    "PTZ LQ-107 Wash-Ave48": "Washington Street and Avenue 48",
    "Washington and Sagebrush Ave": "Washington Street and Sagebrush Ave",
    "Washington St & Sagebrush Ave": "Washington Street and Sagebrush Ave",
    "Washington Street & Sagebrush Ave": "Washington Street and Sagebrush Ave",
    "Washington and Avenue 52": "Washington Street and Avenue 52",
    "Washington Street and Avenue 52": "Washington Street and Avenue 52",

    # Jefferson Street @ Fred Waring
    "TS266JeffersonStFredWaring": "Jefferson Street and Fred Waring",
    "Jefferson & Fred Waring": "Jefferson Street and Fred Waring",
    "Jefferson Street & Fred Waring": "Jefferson Street and Fred Waring",
}

def _canonicalize_segment(seg_id: str, corridor_label: str) -> str:
    s = str(seg_id).strip()
    if s in SEGMENT_ID_ALIASES:
        return SEGMENT_ID_ALIASES[s]
    # Fallback: keep original but prepend corridor for readability
    return f"{corridor_label} — {s}"

# Aggregation metadata
AGG_META = {
    "Hourly": {"unit": "vph", "bucket": "H", "label": "hour", "fixed_hours": 1},
    "Daily": {"unit": "vpd", "bucket": "D", "label": "day", "fixed_hours": 24},
    "Weekly": {"unit": "vpw", "bucket": "W", "label": "week", "fixed_hours": 24 * 7},
    "Monthly": {"unit": "vpm", "bucket": "M", "label": "month", "fixed_hours": None},
}

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "toggleSpikelines"]
}
MAP_HEIGHT = 900

# Vehicle class definitions (for pie charts and cycle length)
VEHICLE_CLASSES = ["trucks", "cars", "buses", "motorcycles"]
BICYCLE_ALIASES = {
    "bikes": "bicycles",
    "bikes_counts": "bicycles_counts",
    "bikes_speed_in_mph": "bicycles_speed_in_mph",
    "bikes_stopped": "bicycles_stopped",
}

# =========================
# Data Loading
# =========================
@st.cache_data(show_spinner=False)
def load_bosch_data(corridor: str) -> pd.DataFrame:
    """
    Load Bosch data for the specified corridor.
    corridor: keys of BOSCH_RAW_URLS (e.g., "Washington Street", "Fred Waring")
    """
    url = BOSCH_RAW_URLS.get(corridor)
    if not url:
        return pd.DataFrame()

    try:
        df = pd.read_csv(url)

        # Parse dates - Bosch format includes date+hour like "8/27/2025 2:00"
        df["local_datetime"] = pd.to_datetime(df["local_datetime"], errors="coerce")
        df = df.dropna(subset=["local_datetime"])

        # Map segment_id → human-friendly segment_name
        if "segment_id" in df.columns:
            df["segment_name"] = df["segment_id"].astype(str).apply(
                lambda x: _canonicalize_segment(x, corridor)
            )
        else:
            df["segment_name"] = corridor

        # Normalize measure names (lowercase + fix bicycle aliases)
        df["measure"] = df["measure"].astype(str).str.strip().str.lower().replace(BICYCLE_ALIASES)

        # Ensure value is numeric
        df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)

        # Keep expected columns if present (defensive)
        keep = [c for c in ["local_datetime", "segment_id", "segment_name", "weekday", "measure", "value"] if c in df.columns]
        df = df[keep].sort_values("local_datetime").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error loading Bosch data for {corridor}: {e}")
        return pd.DataFrame()

# =========================
# Processing Helpers
# =========================
def normalize_bicycle_measures(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d["measure"] = d["measure"].replace(BICYCLE_ALIASES)
    d = d.groupby(["local_datetime", "segment_name", "weekday", "measure"], as_index=False)["value"].sum()
    return d

def pivot_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    if df_long.empty:
        return pd.DataFrame()
    df = normalize_bicycle_measures(df_long)
    wide = df.pivot_table(
        index=["local_datetime", "segment_name"],
        columns="measure",
        values="value",
        aggfunc="sum",
        fill_value=0
    ).reset_index()
    wide.columns.name = None
    return wide

def compute_vehicle_totals(df_wide: pd.DataFrame) -> pd.DataFrame:
    if df_wide.empty:
        return df_wide
    d = df_wide.copy()
    vehicle_cols = [f"{cls}_counts" for cls in VEHICLE_CLASSES]
    existing = [c for c in vehicle_cols if c in d.columns]
    d["vehicular_volume"] = d[existing].sum(axis=1) if existing else 0
    return d

def _prep_bucket(df: pd.DataFrame, granularity: str, volume_col: str = "vehicular_volume") -> pd.DataFrame:
    if df.empty or volume_col not in df.columns:
        return df.copy()
    meta = AGG_META[granularity]
    d = df.copy()
    d["local_datetime"] = pd.to_datetime(d["local_datetime"])
    if granularity == "Hourly":
        d["bucket"] = d["local_datetime"].dt.floor("H")
    elif granularity == "Daily":
        d["bucket"] = d["local_datetime"].dt.floor("D")
    elif granularity == "Weekly":
        d["bucket"] = d["local_datetime"].dt.to_period("W").dt.start_time
    else:
        d["bucket"] = d["local_datetime"].dt.to_period("M").dt.start_time

    agg = d.groupby(["bucket", "segment_name"], as_index=False)[volume_col].sum().rename(
        columns={"bucket": "local_datetime", volume_col: "volume"}
    )
    agg["bucket_hours"] = (
        pd.to_datetime(agg["local_datetime"]).dt.days_in_month * 24
        if granularity == "Monthly" else
        meta["fixed_hours"]
    )
    return agg

# =========================
# Chart Helpers
# =========================
def create_volume_comparison_charts(df_wide: pd.DataFrame, granularity: str):
    if df_wide.empty:
        return None, None

    d = df_wide.copy()
    unit = AGG_META[granularity]["unit"]

    # Volume trends by class
    volume_cols = {f"{cls}_counts": cls.title() for cls in VEHICLE_CLASSES + ["bicycles"]}
    existing_vol = {k: v for k, v in volume_cols.items() if k in d.columns}
    if existing_vol:
        vol_melted = d.melt(
            id_vars=["local_datetime", "segment_name"],
            value_vars=list(existing_vol.keys()),
            var_name="class",
            value_name="volume"
        )
        vol_melted["class"] = vol_melted["class"].replace(existing_vol)
        fig_vol = px.line(
            vol_melted.groupby(["local_datetime", "class"], as_index=False)["volume"].sum(),
            x="local_datetime",
            y="volume",
            color="class",
            title=f"📊 Volume Trends by Mode ({granularity})",
            labels={"volume": f"Volume ({unit})", "local_datetime": "Date/Time", "class": "Mode"},
            template="plotly_white",
        )
        fig_vol.update_layout(
            height=450,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
    else:
        fig_vol = None

    # Speed by class
    speed_cols = {f"{cls}_speed_in_mph": cls.title() for cls in VEHICLE_CLASSES + ["bicycles"]}
    existing_speed = {k: v for k, v in speed_cols.items() if k in d.columns}
    if existing_speed:
        speed_melted = d.melt(
            id_vars=["local_datetime", "segment_name"],
            value_vars=list(existing_speed.keys()),
            var_name="class",
            value_name="speed"
        )
        speed_melted["class"] = speed_melted["class"].replace(existing_speed)
        speed_melted = speed_melted[speed_melted["speed"] > 0]
        fig_speed = px.line(
            speed_melted.groupby(["local_datetime", "class"], as_index=False)["speed"].mean(),
            x="local_datetime",
            y="speed",
            color="class",
            title=f"🚗 Average Speed by Mode ({granularity})",
            labels={"speed": "Speed (mph)", "local_datetime": "Date/Time", "class": "Mode"},
            template="plotly_white",
        )
        fig_speed.update_layout(
            height=450,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
    else:
        fig_speed = None

    return fig_vol, fig_speed

def create_mode_share_pies(df_wide: pd.DataFrame):
    if df_wide.empty:
        return None, None

    d = df_wide.copy()

    # Volume share
    volume_cols = [f"{cls}_counts" for cls in VEHICLE_CLASSES + ["bicycles"]]
    existing_vol = [c for c in volume_cols if c in d.columns]
    if existing_vol:
        vol_totals = d[existing_vol].sum()
        vol_totals.index = vol_totals.index.str.replace("_counts", "").str.title()
        fig_vol_pie = px.pie(
            values=vol_totals.values,
            names=vol_totals.index,
            title="📊 Volume Share by Mode",
            hole=0.4,
            template="simple_white",
        )
        fig_vol_pie.update_traces(
            textposition="inside",
            texttemplate="%{label}<br>%{percent}",
            hovertemplate="<b>%{label}</b><br>Count: %{value:,.0f}<br>Share: %{percent}<extra></extra>",
        )
        fig_vol_pie.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=True,
        )
    else:
        fig_vol_pie = None

    # Stopped share
    stopped_cols = [f"{cls}_stopped" for cls in VEHICLE_CLASSES + ["bicycles"]]
    existing_stopped = [c for c in stopped_cols if c in d.columns]
    if existing_stopped:
        stopped_totals = d[existing_stopped].sum()
        stopped_totals = stopped_totals[stopped_totals > 0]
        stopped_totals.index = stopped_totals.index.str.replace("_stopped", "").str.title()
        if len(stopped_totals) > 0:
            fig_stopped_pie = px.pie(
                values=stopped_totals.values,
                names=stopped_totals.index,
                title="🚦 Stopped Objects Share by Mode",
                hole=0.4,
                template="simple_white",
                color_discrete_sequence=px.colors.sequential.Reds,
            )
            fig_stopped_pie.update_traces(
                textposition="inside",
                texttemplate="%{label}<br>%{percent}",
                hovertemplate="<b>%{label}</b><br>Count: %{value:,.0f}<br>Share: %{percent}<extra></extra>",
            )
            fig_stopped_pie.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=50, b=10),
                showlegend=True,
            )
        else:
            fig_stopped_pie = None
    else:
        fig_stopped_pie = None

    return fig_vol_pie, fig_stopped_pie

# =========================
# Main Renderer
# =========================
def render_bosch_tab():
    """Main renderer for Tab 5: Bosch Multimodal Traffic Analysis."""

    # -------- Sidebar controls --------
    with st.sidebar:
        with st.expander("⚙️ Pg.5 BOSCH SETTINGS", expanded=False):
            active_t5 = is_active_tab("t5")
            if active_t5:
                st.markdown(
                    """
                    <div style="
                        background: linear-gradient(90deg, #ffe58f, #ffd666);
                        border: 1px solid #fadb14; color: #613400;
                        padding: 6px 10px; border-radius: 8px; font-weight: 700; margin-bottom: 6px;">
                        • You’re viewing: Pg.5 BOSCH CLOUD ANALYTICS
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.caption("Select Corridor and Date Range")
            st.caption("Data: Multimodal Traffic from Bosch Sensors")

            # Corridor selector (keys of BOSCH_RAW_URLS)
            st.markdown("## 🛣️ Select Corridor")
            corridor = st.selectbox(
                "Corridor",
                list(BOSCH_RAW_URLS.keys()),
                key="bosch_corridor",
            )

            # Load data for selected corridor
            bosch_df = load_bosch_data(corridor)

            # Date range
            if bosch_df.empty or "local_datetime" not in bosch_df.columns:
                min_date = datetime.today().date() - timedelta(days=7)
                max_date = datetime.today().date()
            else:
                min_date = bosch_df["local_datetime"].dt.date.min()
                max_date = bosch_df["local_datetime"].dt.date.max()

            st.markdown("## 📅 Date And Time")
            date_range = date_range_preset_controls(min_date, max_date, key_prefix="bosch")

            # Granularity
            st.markdown("## Granularity")
            granularity = st.selectbox(
                "Data Aggregation",
                ["Hourly", "Daily", "Weekly", "Monthly"],
                index=0,
                key="granularity_bosch",
            )

            # Mode filter
            st.markdown("## 🚗 Mode Filter")
            mode_filter = st.multiselect(
                "Select Modes to Include",
                ["Trucks", "Cars", "Buses", "Motorcycles", "Bicycles", "Persons"],
                default=["Trucks", "Cars", "Buses", "Motorcycles"],
                key="mode_filter_bosch",
            )

            bosch_current = {
                "corridor": corridor,
                "date_range": tuple(date_range) if date_range else None,
                "granularity": granularity,
                "mode_filter": tuple(sorted(mode_filter)) if mode_filter else None,
            }
            st.session_state["bosch_current"] = bosch_current

            if st.button("🔍 **Search**", key="search_bosch", type="primary", use_container_width=True):
                st.session_state["bosch_params"] = bosch_current
                st.session_state["bosch_ready"] = True
                set_active_search_tab("t5")
                st.session_state["last_active_tab"] = "t5"

    # -------- Main content area --------
    if not st.session_state.get("bosch_ready", False):
        st.info("Choose your Corridor and Date Range in the settings to the left.")
        return

    params = st.session_state.get("bosch_params", {})
    corridor = params.get("corridor", "Washington Street")
    date_range = params.get("date_range")
    granularity = params.get("granularity", "Hourly")
    mode_filter = params.get("mode_filter", ())

    # Pending-change warning if sidebar controls differ from committed params
    t5_pending = st.session_state.get("bosch_ready", False) and (
        params != st.session_state.get("bosch_current", {})
    )
    if t5_pending:
        st.warning("⚙️ Press **Search** to refresh.")

    if not date_range or len(date_range) != 2:
        st.warning("⚠️ Please select both start and end dates to proceed.")
        return

    try:
        with scoped_cad_loader("Fetching Data...", tab_id="t5") as step:
            step("Loading Bosch data", 15)
            bosch_df = load_bosch_data(corridor)
            if bosch_df.empty:
                step("No data for corridor", 100)
                st.warning(f"⚠️ No data available for {corridor}.")
                return

            step("Filtering by date range", 35)
            working_df = bosch_df[
                (bosch_df["local_datetime"].dt.date >= date_range[0]) &
                (bosch_df["local_datetime"].dt.date <= date_range[1])
            ].copy()

            if working_df.empty:
                step("No data for selected date range", 100)
                st.warning("⚠️ No data available for the selected date range.")
                return

            # Apply mode filter
            if mode_filter:
                step("Applying mode filter", 55)
                mode_measures = []
                for mode in mode_filter:
                    mode_measures.extend([
                        f"{mode.lower()}_counts",
                        f"{mode.lower()}_speed_in_mph",
                        f"{mode.lower()}_stopped",
                    ])
                mode_measures += ["total_counts", "average_speed_mph", "total_stopped_objects"]
                working_df = working_df[working_df["measure"].isin(mode_measures)]

            if working_df.empty:
                step("No data after mode filter", 100)
                st.warning("⚠️ No data available for the selected modes.")
                return

            # Convert to wide format & totals
            step("Aggregating & computing totals", 75)
            wide_df = pivot_to_wide(working_df)
            wide_df = compute_vehicle_totals(wide_df)

        # Layout: main content + sticky right rail
        content_col, right_col = st.columns([7, 3.5], gap="large")

        # ---------- Right rail (map) ----------
        with right_col:
            st.markdown('<div id="bosch-map-anchor"></div>', unsafe_allow_html=True)
            st.markdown("##### Corridor Map", help="Stays visible while you scroll the analysis on the left.")

            try:
                fig_map = build_all_segments_overview()
            except Exception:
                fig_map = None

            if fig_map:
                try:
                    fig_map.update_layout(height=MAP_HEIGHT, margin=dict(l=0, r=0, t=32, b=0))
                except Exception:
                    pass
                st.markdown('<div class="cvag-map-card">', unsafe_allow_html=True)
                st.plotly_chart(fig_map, use_container_width=True, config=PLOTLY_CONFIG, key="bosch_map_plot")
                st.caption(f"**Corridor:** {corridor}")
                # Data source link
                src = BOSCH_RAW_URLS.get(corridor)
                if src:
                    st.markdown(f"[Open data source ↗]({src})")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="cvag-map-card">', unsafe_allow_html=True)
                st.info(f"**Corridor:** {corridor}")
                st.markdown("📍 Multi-modal traffic monitoring with Bosch sensors")
                src = BOSCH_RAW_URLS.get(corridor)
                if src:
                    st.markdown(f"[Open data source ↗]({src})")
                st.markdown('</div>', unsafe_allow_html=True)

        # ---------- Main content ----------
        with content_col:
            span = (date_range[1] - date_range[0]).days + 1
            total_obs = len(working_df)

            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #2b77e5 0%, #19c3e6 100%);
                    border-radius:16px; padding:18px 20px; color:#fff; margin:8px 0 14px;
                    box-shadow:0 10px 26px rgba(25,115,210,.25); text-align:left;">
                  <div style="display:flex; align-items:center; gap:10px;">
                    <div style="width:36px;height:36px;border-radius:10px;background:rgba(255,255,255,.18);
                                display:flex;align-items:center;justify-content:center;
                                box-shadow:inset 0 0 0 1px rgba(255,255,255,.15);">🚦</div>
                    <div style="font-size:1.9rem;font-weight:800;letter-spacing:.2px;">
                      Bosch Multimodal Traffic Analysis: {corridor}
                    </div>
                  </div>
                  <div style="margin-top:10px;display:flex;flex-direction:column;gap:6px;">
                    <div>📅 {date_range[0].strftime('%b %d, %Y')} to {date_range[1].strftime('%b %d, %Y')} ({span} days) • {granularity} Aggregation</div>
                    <div>✅ {total_obs:,} measurements • Modes: {", ".join(mode_filter) if mode_filter else "All"}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ---------- KPIs ----------
            st.subheader("🚦 Multimodal Performance Indicators")

            if not wide_df.empty:
                total_volume = float(wide_df.get("total_counts", pd.Series([0])).sum())
                vehicular_volume = float(wide_df.get("vehicular_volume", pd.Series([0])).sum())
                bicycle_volume = float(wide_df.get("bicycles_counts", pd.Series([0])).sum())
                person_volume = float(wide_df.get("persons_counts", pd.Series([0])).sum())

                avg_speed = float(wide_df.get("average_speed_mph", pd.Series([0])).mean()) if "average_speed_mph" in wide_df.columns else 0.0
                total_stopped = float(wide_df.get("total_stopped_objects", pd.Series([0])).sum())
                stopped_pct = (total_stopped / total_volume * 100) if total_volume > 0 else 0.0

                if "vehicular_volume" in wide_df.columns:
                    peak_idx = wide_df["vehicular_volume"].idxmax()
                    peak_vol = float(wide_df.loc[peak_idx, "vehicular_volume"])
                    peak_time = wide_df.loc[peak_idx, "local_datetime"]
                    peak_util_pct = (peak_vol / THEORETICAL_LINK_CAPACITY_VPH * 100)
                else:
                    peak_vol, peak_util_pct = 0.0, 0.0
                    peak_time = None

                non_vehic_share = ((bicycle_volume + person_volume) / total_volume * 100) if total_volume > 0 else 0.0

                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric("🚗 Total Vehicle Volume", f"{vehicular_volume:,.0f}", help="Sum of trucks, cars, buses, and motorcycles")
                    badge = ("badge-critical" if peak_util_pct > 90 else "badge-poor" if peak_util_pct > 75 else "badge-fair" if peak_util_pct > 60 else "badge-good")
                    st.markdown(f'<span class="performance-badge {badge}">Peak: {peak_util_pct:.0f}% capacity</span>', unsafe_allow_html=True)

                with col2:
                    st.metric("🚴 Active Transportation", f"{bicycle_volume + person_volume:,.0f}", help="Bicycles + Pedestrians")
                    badge2 = "badge-excellent" if non_vehic_share > 15 else ("badge-good" if non_vehic_share > 8 else "badge-fair")
                    st.markdown(f'<span class="performance-badge {badge2}">{non_vehic_share:.1f}% of total</span>', unsafe_allow_html=True)

                with col3:
                    st.metric("📊 Total Traffic", f"{total_volume:,.0f}", help="Total Vehicle Volume + Active Transportation")
                    st.caption(f"Period: {span} days")

                with col4:
                    st.metric("🏎️ Average Speed", f"{avg_speed:.1f} mph", help="Average speed across all vehicles")
                    speed_badge = "badge-excellent" if 25 <= avg_speed <= 35 else ("badge-good" if 20 <= avg_speed <= 40 else "badge-fair")
                    st.markdown(f'<span class="performance-badge {speed_badge}">Corridor avg</span>', unsafe_allow_html=True)

                with col5:
                    st.metric("🚦 Stopped Objects", f"{total_stopped:,.0f}", help="Total stopped detections")
                    stopped_badge = "badge-good" if stopped_pct < 5 else ("badge-fair" if stopped_pct < 10 else "badge-poor")
                    st.markdown(f'<span class="performance-badge {stopped_badge}">{stopped_pct:.1f}% stopped</span>', unsafe_allow_html=True)

            # ---------- Comparison Charts ----------
            st.subheader("📈 Multimodal Traffic Visualizations")
            if len(wide_df) > 1:
                try:
                    fig_vol, fig_speed = create_volume_comparison_charts(wide_df, granularity)
                    colA, colB = st.columns(2)
                    with colA:
                        if fig_vol:
                            st.plotly_chart(fig_vol, use_container_width=True, config=PLOTLY_CONFIG, key="bosch_fig_vol")
                    with colB:
                        if fig_speed:
                            st.plotly_chart(fig_speed, use_container_width=True, config=PLOTLY_CONFIG, key="bosch_fig_speed")
                except Exception as e:
                    st.error(f"❌ Error creating comparison charts: {e}")

            # ---------- Mode Share Pies ----------
            st.subheader("🥧 Mode Share Analysis")
            try:
                fig_vol_pie, fig_stopped_pie = create_mode_share_pies(wide_df)
                colC, colD = st.columns(2)
                with colC:
                    if fig_vol_pie:
                        st.plotly_chart(fig_vol_pie, use_container_width=True, config=PLOTLY_CONFIG, key="bosch_pie_vol")
                with colD:
                    if fig_stopped_pie:
                        st.plotly_chart(fig_stopped_pie, use_container_width=True, config=PLOTLY_CONFIG, key="bosch_pie_stopped")
                    else:
                        st.info("No stopped object data available for the selected period.")
            except Exception as e:
                st.error(f"❌ Error creating pie charts: {e}")

            # ---------- Data Table ----------
            st.subheader("🔍 Detailed Multimodal Data")
            if not wide_df.empty:
                display_cols = ["local_datetime", "segment_name"]
                measure_cols = [c for c in wide_df.columns if "_counts" in c or "_speed" in c or "_stopped" in c]
                display_cols.extend(measure_cols[:10])
                display_df = wide_df[display_cols].head(20)
                st.dataframe(display_df, use_container_width=True)
                st.download_button(
                    "⬇️ Download Bosch Analysis (CSV)",
                    data=wide_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"bosch_{corridor.lower().replace(' ', '_')}_analysis.csv",
                    mime="text/csv",
                )

            # ---------- Cycle Length Recommendations ----------
            st.subheader("🔄 Cycle Length Recommendations for CVAG")
            try:
                if "vehicular_volume" in wide_df.columns:
                    cycle_df = wide_df[["local_datetime", "segment_name", "vehicular_volume"]].copy()
                    cycle_df = cycle_df.rename(columns={"vehicular_volume": "total_volume"})
                    cycle_df["intersection_name"] = cycle_df["segment_name"]
                    st.caption("Note: Using combined vehicular volume (trucks + cars + buses + motorcycles) for cycle length estimation.")
                    render_cycle_length_section(cycle_df)
                else:
                    st.info("No vehicular volume data available for cycle length recommendations.")
            except Exception as e:
                st.error(f"❌ Error rendering cycle length section: {e}")

    except Exception as e:
        st.error(f"❌ Error processing Bosch data: {e}")
        import traceback
        st.text("Debug info:")
        st.text(traceback.format_exc())
