# VantageLivetab.py - Tab 4: Iteris VantageLive Analysis (Bikes + Vehicles)

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
from Map import build_intersections_overview

# =========================
# Constants
# =========================
THEORETICAL_LINK_CAPACITY_VPH = 1800
HIGH_VOLUME_THRESHOLD_VPH = 1200

# Segment mapping (from south to north along Washington Street)
SEGMENT_ID_TO_NAME = {
    1: "Washington & Avenue 52",
    2: "Washington and Calle Tampico",
    3: "Washington and Village Shopping Center",
    4: "Washington and Avenue 50",
    5: "Washington and Sagebrush Avenue",
    6: "Washington and Eisenhower Drive",
    7: "Washington and Avenue 48",
    8: "Washington and Avenue 47",
    9: "Washington and Point Happy Way",
}

# Aggregation metadata (matching Tab 2 pattern)
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


# =========================
# Data Loading
# =========================
@st.cache_data(show_spinner=False)
def load_vantage_bikes() -> pd.DataFrame:
    """Load bike volume data from VantageLive."""
    url = "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/Iteris_VantageLive/WashingtonStreet_ALL_Bikes.csv"
    try:
        df = pd.read_csv(url)

        # Parse dates in American format (month/day/year)
        df["local_datetime"] = pd.to_datetime(df["local_datetime"], format="%m/%d/%Y", errors="coerce")
        df = df.dropna(subset=["local_datetime"])

        # Map segment_id to intersection names
        df["intersection_name"] = df["segment_id"].map(SEGMENT_ID_TO_NAME)
        df = df.dropna(subset=["intersection_name"])

        # Ensure volume is numeric
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

        return df.sort_values("local_datetime").reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading bike data: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_vantage_vehicles() -> pd.DataFrame:
    """Load vehicle volume data from VantageLive."""
    url = "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/Iteris_VantageLive/WashingtonStreet_ALL_vehicles.csv"
    try:
        df = pd.read_csv(url)

        # Parse dates in American format (month/day/year)
        df["local_datetime"] = pd.to_datetime(df["local_datetime"], format="%m/%d/%Y", errors="coerce")
        df = df.dropna(subset=["local_datetime"])

        # Map segment_id to intersection names
        df["intersection_name"] = df["segment_id"].map(SEGMENT_ID_TO_NAME)
        df = df.dropna(subset=["intersection_name"])

        # Ensure volume is numeric
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

        return df.sort_values("local_datetime").reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading vehicle data: {e}")
        return pd.DataFrame()


# =========================
# Processing Helpers
# =========================
def _prep_bucket(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """
    Aggregate hourly records to the selected bucket (sum of hourly volumes).
    Returns: df with columns [local_datetime, intersection_name, volume, bucket_hours].
    """
    if df.empty:
        return df.copy()

    g = granularity
    meta = AGG_META[g]
    d = df.copy()
    d["local_datetime"] = pd.to_datetime(d["local_datetime"])

    if g == "Hourly":
        d["bucket"] = d["local_datetime"].dt.floor("H")
    elif g == "Daily":
        d["bucket"] = d["local_datetime"].dt.floor("D")
    elif g == "Weekly":
        d["bucket"] = d["local_datetime"].dt.to_period("W").dt.start_time
    else:  # Monthly
        d["bucket"] = d["local_datetime"].dt.to_period("M").dt.start_time

    agg = (
        d.groupby(["bucket", "intersection_name"], as_index=False)
        .agg(volume=("volume", "sum"))
        .rename(columns={"bucket": "local_datetime"})
    )

    # Hours in the bucket (for capacity/threshold scaling)
    if g == "Monthly":
        agg["bucket_hours"] = pd.to_datetime(agg["local_datetime"]).dt.days_in_month * 24
    else:
        agg["bucket_hours"] = meta["fixed_hours"]
    return agg


def _cap_series_for_x(x_df: pd.DataFrame, cap_vph: float, high_vph: float) -> pd.DataFrame:
    """Given unique x (local_datetime) and bucket_hours, produce y series for capacity/threshold."""
    xs = x_df[["local_datetime", "bucket_hours"]].drop_duplicates().sort_values("local_datetime")
    xs["capacity"] = xs["bucket_hours"] * float(cap_vph)
    xs["high"] = xs["bucket_hours"] * float(high_vph)
    return xs


def _fmt_period(ts: pd.Timestamp, granularity: str) -> str:
    ts = pd.to_datetime(ts)
    if granularity == "Hourly":
        return ts.strftime("%b %d, %Y %H:%M")
    if granularity == "Daily":
        return ts.strftime("%b %d, %Y")
    if granularity == "Weekly":
        wk = ts.to_period("W")
        return f"Week of {wk.start_time.strftime('%b %d, %Y')}"
    return ts.strftime("%b %Y")


# =========================
# Chart Helpers
# =========================
def create_volume_charts(
        raw_hourly_df: pd.DataFrame,
        granularity: str,
        cap_vph: float,
        high_vph: float,
        mode_label: str = "Vehicles",
        top_k: int = 10
):
    """
    Returns (fig_trend, fig_box, fig_matrix)
    - fig_trend: Time series per intersection with scaled capacity/threshold overlays.
    - fig_box:   Distribution of bucket totals by intersection.
    - fig_matrix: Average bucket total by intersection (ranking).
    """
    if raw_hourly_df.empty:
        return None, None, None

    # Aggregate to the selected bucket
    agg = _prep_bucket(raw_hourly_df, granularity)
    if agg.empty:
        return None, None, None

    # Limit to top intersections by mean demand
    order = agg.groupby("intersection_name")["volume"].mean().sort_values(ascending=False)
    keep = order.index[:max(1, min(top_k, len(order)))]

    plot_df = agg[agg["intersection_name"].isin(keep)].copy().sort_values("local_datetime")
    unit = AGG_META[granularity]["unit"]
    label = AGG_META[granularity]["label"]

    # ---------- Trend ----------
    fig_trend = go.Figure()
    mode = "lines" if granularity == "Hourly" else "lines+markers"
    xfmt = "%Y-%m-%d %H:%M" if granularity == "Hourly" else "%Y-%m-%d"

    for name, g in plot_df.groupby("intersection_name"):
        fig_trend.add_trace(
            go.Scatter(
                x=g["local_datetime"],
                y=g["volume"],
                mode=mode,
                name=name,
                hovertemplate=(
                    f"<b>%{{fullData.name}}</b><br>%{{x|{xfmt}}}<br>{mode_label}: %{{y:,.0f}} {unit}<extra></extra>"),
            )
        )

    xs = _cap_series_for_x(plot_df, cap_vph, high_vph)
    fig_trend.add_trace(
        go.Scatter(
            x=xs["local_datetime"], y=xs["capacity"],
            name=f"Theoretical Capacity ({unit})", mode="lines",
            line=dict(dash="dash", color="red"),
            hovertemplate=(f"%{{x|{xfmt}}}<br>Capacity: %{{y:,.0f}} {unit}<extra></extra>"),
        )
    )
    fig_trend.add_trace(
        go.Scatter(
            x=xs["local_datetime"], y=xs["high"],
            name=f"High Volume Threshold ({unit})", mode="lines",
            line=dict(dash="dot", color="orange"),
            hovertemplate=(f"%{{x|{xfmt}}}<br>Threshold: %{{y:,.0f}} {unit}<extra></extra>"),
        )
    )
    fig_trend.update_layout(
        title=f"{mode_label} Volume Trends - {granularity}",
        xaxis_title="Date/Time",
        yaxis_title=f"{mode_label} Volume ({unit})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_white",
    )

    # ---------- Box ----------
    cat_order = order[order.index.isin(keep)].index.tolist()
    fig_box = px.box(
        plot_df, x="intersection_name", y="volume",
        category_orders={"intersection_name": cat_order},
        points=False,
        title=f"{mode_label} Volume Distribution by Intersection — {granularity}"
    )
    fig_box.update_layout(
        xaxis_title="Intersection",
        yaxis_title=f"{mode_label} Volume per {label} ({unit})",
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_white",
    )

    # ---------- Matrix ----------
    mat = (
        plot_df.groupby("intersection_name", as_index=False)["volume"]
        .mean()
        .rename(columns={"volume": f"Avg {label} Volume"})
    )
    mat["Rank"] = mat[f"Avg {label} Volume"].rank(ascending=False, method="dense").astype(int)
    mat = mat.sort_values("Rank")
    fig_matrix = px.bar(
        mat, y="intersection_name", x=f"Avg {label} Volume",
        orientation="h", text=f"Avg {label} Volume",
        title=f"Average {label.capitalize()} {mode_label} Volume by Intersection"
    )
    fig_matrix.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
    fig_matrix.update_layout(
        xaxis_title=f"Average {label} volume ({unit})",
        yaxis_title="",
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_white",
    )
    return fig_trend, fig_box, fig_matrix


# =========================
# Main Renderer
# =========================
def render_vantage_tab():
    """
    Main renderer for Tab 4: Iteris VantageLive (Bikes + Vehicles).
    Matches Tab 2 style with search-gated results.
    """
    # Load data once
    bikes_df = load_vantage_bikes()
    vehicles_df = load_vantage_vehicles()

    # -------- Sidebar controls --------
    with st.sidebar:
        with st.expander("⚙️ Pg.4 SETTINGS", expanded=True):
            st.caption("Select Mode, Intersection(s) and Date Range")
            st.caption("Data: Bike & Vehicle Volume from Iteris VantageLive")

            # Mode selector
            st.markdown("## 🚲 Select Mode")
            mode = st.selectbox(
                "Analysis Mode",
                ["Vehicles", "Bikes", "Both (Combined)"],
                key="vantage_mode",
            )

            # Intersection selector
            all_intersections = sorted(list(SEGMENT_ID_TO_NAME.values()))
            st.markdown("## 🚦 Select Intersection")
            intersection = st.selectbox(
                "Intersection",
                ["All Intersections"] + all_intersections,
                key="vantage_intersection",
            )

            # Date range
            if mode == "Bikes" and not bikes_df.empty:
                min_date = bikes_df["local_datetime"].dt.date.min()
                max_date = bikes_df["local_datetime"].dt.date.max()
            elif mode == "Vehicles" and not vehicles_df.empty:
                min_date = vehicles_df["local_datetime"].dt.date.min()
                max_date = vehicles_df["local_datetime"].dt.date.max()
            elif not bikes_df.empty and not vehicles_df.empty:
                min_date = min(bikes_df["local_datetime"].dt.date.min(),
                               vehicles_df["local_datetime"].dt.date.min())
                max_date = max(bikes_df["local_datetime"].dt.date.max(),
                               vehicles_df["local_datetime"].dt.date.max())
            else:
                min_date = datetime.today().date() - timedelta(days=7)
                max_date = datetime.today().date()

            st.markdown("## 📅 Date And Time")
            date_range = date_range_preset_controls(min_date, max_date, key_prefix="vantage")

            # Granularity
            st.markdown("## Granularity")
            granularity = st.selectbox(
                "Data Aggregation",
                ["Hourly", "Daily", "Weekly", "Monthly"],
                index=0,
                key="granularity_vantage",
            )

            # Direction filter
            st.markdown("## 🔄 Direction Filter")
            direction_filter = st.selectbox(
                "Direction",
                ["All Directions", "NB", "SB", "EB", "WB"],
                key="direction_filter_vantage",
            )

            # Turn type filter (vehicles only)
            turn_filter = None
            if mode in ["Vehicles", "Both (Combined)"]:
                st.markdown("## 🔄 Turn Type Filter")
                turn_filter = st.selectbox(
                    "Turn Type",
                    ["All Turns", "Through", "Left", "Right"],
                    key="turn_filter_vantage",
                )

            # Track uncommitted controls
            vantage_current = {
                "mode": mode,
                "intersection": intersection,
                "date_range": tuple(date_range) if date_range else None,
                "granularity": granularity,
                "direction_filter": direction_filter,
                "turn_filter": turn_filter,
            }
            st.session_state["vantage_current"] = vantage_current

            if st.button("🔍 **Search**", key="search_vantage", type="primary", use_container_width=True):
                st.session_state["vantage_params"] = vantage_current
                st.session_state["vantage_ready"] = True

    # -------- Main content area --------
    vantage_ready = st.session_state.get("vantage_ready", False)

    if not vantage_ready:
        st.info("Choose your Mode, Intersection and Date Range in the settings to the left.")
        return

    vantage_params = st.session_state.get("vantage_params", {})
    mode = vantage_params.get("mode", "Vehicles")
    intersection = vantage_params.get("intersection", "All Intersections")
    date_range = vantage_params.get("date_range")
    granularity = vantage_params.get("granularity", "Hourly")
    direction_filter = vantage_params.get("direction_filter", "All Directions")
    turn_filter = vantage_params.get("turn_filter", "All Turns")

    if not date_range or len(date_range) != 2:
        st.warning("⚠️ Please select both start and end dates to proceed.")
        return

    try:
        # Prepare working datasets
        working_bikes = bikes_df.copy() if not bikes_df.empty else pd.DataFrame()
        working_vehicles = vehicles_df.copy() if not vehicles_df.empty else pd.DataFrame()

        # Apply filters
        for df in [working_bikes, working_vehicles]:
            if df.empty:
                continue
            # Date filter
            df = df[(df["local_datetime"].dt.date >= date_range[0]) &
                    (df["local_datetime"].dt.date <= date_range[1])]
            # Intersection filter
            if intersection != "All Intersections":
                df = df[df["intersection_name"] == intersection]
            # Direction filter
            if direction_filter != "All Directions" and "direction" in df.columns:
                df = df[df["direction"].str.upper() == direction_filter]

            # Store back
            if df is working_bikes:
                working_bikes = df
            else:
                working_vehicles = df

        # Apply turn filter to vehicles
        if turn_filter and turn_filter != "All Turns" and not working_vehicles.empty:
            if "turn_type" in working_vehicles.columns:
                working_vehicles = working_vehicles[working_vehicles["turn_type"].str.title() == turn_filter]

        # Select data based on mode
        if mode == "Bikes":
            analysis_df = working_bikes
            mode_label = "Bikes"
        elif mode == "Vehicles":
            analysis_df = working_vehicles
            mode_label = "Vehicles"
        else:  # Both (Combined)
            # Combine bikes and vehicles
            if not working_bikes.empty and not working_vehicles.empty:
                # Aggregate vehicles by intersection/datetime (sum across turns)
                if "turn_type" in working_vehicles.columns:
                    working_vehicles = working_vehicles.groupby(
                        ["local_datetime", "intersection_name", "direction"],
                        as_index=False
                    )["volume"].sum()

                # Add mode column
                working_bikes["mode"] = "Bikes"
                working_vehicles["mode"] = "Vehicles"
                analysis_df = pd.concat([working_bikes, working_vehicles], ignore_index=True)
                mode_label = "Bikes + Vehicles"
            elif not working_bikes.empty:
                analysis_df = working_bikes
                mode_label = "Bikes"
            else:
                analysis_df = working_vehicles
                mode_label = "Vehicles"

        if analysis_df.empty:
            st.warning("⚠️ No data available for the selected filters.")
            return

        # Two-column layout with sticky right rail
        content_col, right_col = st.columns([7, 3.5], gap="large")

        # Right rail (sticky map)
        with right_col:
            st.markdown('<div id="vantage-map-anchor"></div>', unsafe_allow_html=True)
            st.markdown("##### Corridor Map", help="Stays visible while you scroll the analysis on the left.")

            try:
                fig_map = build_intersections_overview(
                    selected_label=None if intersection == "All Intersections" else intersection
                )
            except Exception:
                fig_map = None

            if fig_map:
                try:
                    fig_map.update_layout(height=MAP_HEIGHT, margin=dict(l=0, r=0, t=32, b=0))
                except Exception:
                    pass
                st.markdown('<div class="cvag-map-card">', unsafe_allow_html=True)
                st.plotly_chart(fig_map, use_container_width=True, config=PLOTLY_CONFIG)
                if intersection != "All Intersections":
                    st.caption(f"Selected: **{intersection}**")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="cvag-map-card">', unsafe_allow_html=True)
                st.caption("Map: Washington Street Corridor")
                st.markdown('</div>', unsafe_allow_html=True)

        # Main analysis content
        with content_col:
            span = (date_range[1] - date_range[0]).days + 1
            total_obs = len(analysis_df)

            # Gradient header (matching Tab 2)
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #2b77e5 0%, #19c3e6 100%);
                    border-radius:16px; padding:18px 20px; color:#fff; margin:8px 0 14px;
                    box-shadow:0 10px 26px rgba(25,115,210,.25); text-align:left;">
                  <div style="display:flex; align-items:center; gap:10px;">
                    <div style="width:36px;height:36px;border-radius:10px;background:rgba(255,255,255,.18);
                                display:flex;align-items:center;justify-content:center;
                                box-shadow:inset 0 0 0 1px rgba(255,255,255,.15);">📊</div>
                    <div style="font-size:1.9rem;font-weight:800; letter-spacing:.2px;">
                      Iteris VantageLive Analysis: {mode_label}
                    </div>
                  </div>
                  <div style="margin-top:10px;display:flex;flex-direction:column;gap:6px;">
                    <div>📅 {date_range[0].strftime('%b %d, %Y')} to {date_range[1].strftime('%b %d, %Y')} ({span} days) • {granularity} Aggregation</div>
                    <div>✅ {total_obs:,} observations • Intersection: {intersection} • Direction: {direction_filter}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Prepare aggregated data
            raw = analysis_df.copy()
            raw["volume"] = pd.to_numeric(raw["volume"], errors="coerce")
            raw["local_datetime"] = pd.to_datetime(raw["local_datetime"])

            # -------- KPIs Section --------
            st.subheader(f"🚦 {mode_label} Volume Performance Indicators")
            if not raw.empty and raw["volume"].notna().any():
                bucket_all = _prep_bucket(raw, granularity).groupby("local_datetime", as_index=False)[
                    "volume"].sum().sort_values("local_datetime")
                if granularity == "Monthly":
                    bucket_all["bucket_hours"] = pd.to_datetime(bucket_all["local_datetime"]).dt.days_in_month * 24
                else:
                    bucket_all["bucket_hours"] = AGG_META[granularity]["fixed_hours"]

                bucket_all["cap"] = bucket_all["bucket_hours"] * THEORETICAL_LINK_CAPACITY_VPH
                util_series = np.where(bucket_all["cap"] > 0, bucket_all["volume"] / bucket_all["cap"] * 100, np.nan)

                peak_idx = int(bucket_all["volume"].idxmax())
                peak_val = float(bucket_all.loc[peak_idx, "volume"])
                peak_cap = float(bucket_all.loc[peak_idx, "cap"])
                peak_util_pct = (peak_val / peak_cap * 100) if peak_cap > 0 else 0.0

                p95_val = float(np.nanpercentile(bucket_all["volume"], 95)) if bucket_all[
                    "volume"].notna().any() else 0.0
                avg_bucket_val = float(bucket_all["volume"].mean())
                avg_util_pct = float(np.nanmean(util_series)) if np.isfinite(util_series).any() else 0.0

                hourly_avg = float(np.nanmean(raw["volume"])) if raw["volume"].notna().any() else 0.0
                cv_hourly = (float(np.nanstd(raw["volume"])) / hourly_avg * 100) if hourly_avg > 0 else 0.0
                cv_bucket = (float(
                    np.nanstd(bucket_all["volume"])) / avg_bucket_val * 100) if avg_bucket_val > 0 else 0.0

                high_hours = int((raw["volume"] > HIGH_VOLUME_THRESHOLD_VPH).sum())
                total_hours = int(raw["volume"].count())
                risk_pct = (high_hours / total_hours * 100) if total_hours > 0 else 0.0

                unit = AGG_META[granularity]["unit"]
                if granularity == "Hourly":
                    avg_label = f"Average Hourly {mode_label}"
                    peak_label = f"🔥 Peak Hourly {mode_label}"
                    avg_suffix = "vph"
                elif granularity == "Daily":
                    avg_label = f"Average Daily {mode_label}"
                    peak_label = f"🔥 Peak Daily {mode_label}"
                    avg_suffix = "vpd"
                elif granularity == "Weekly":
                    avg_label = f"Average Weekly {mode_label}"
                    peak_label = f"🔥 Peak Weekly {mode_label}"
                    avg_suffix = "vpw"
                else:
                    avg_label = f"Average Monthly {mode_label}"
                    peak_label = f"🔥 Peak Monthly {mode_label}"
                    avg_suffix = "vpm"

                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    badge = (
                        "badge-critical" if peak_util_pct > 90 else
                        "badge-poor" if peak_util_pct > 75 else
                        "badge-fair" if peak_util_pct > 60 else
                        "badge-good"
                    )
                    st.metric(peak_label, f"{peak_val:,.0f} {unit}", delta=f"95th: {p95_val:,.0f} {unit}")
                    st.markdown(
                        f'<span class="performance-badge {badge}">{peak_util_pct:.0f}% of Capacity</span>',
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.metric(
                        f"📊 {avg_label}",
                        f"{avg_bucket_val:,.0f} {avg_suffix}",
                    )
                    if granularity == "Hourly":
                        avg_util_pct_hourly = (
                                    hourly_avg / THEORETICAL_LINK_CAPACITY_VPH * 100) if THEORETICAL_LINK_CAPACITY_VPH else 0.0
                        badge2 = "badge-good" if avg_util_pct_hourly <= 40 else (
                            "badge-fair" if avg_util_pct_hourly <= 60 else "badge-poor")
                        st.markdown(
                            f'<span class="performance-badge {badge2}">{avg_util_pct_hourly:.0f}% Avg Util</span>',
                            unsafe_allow_html=True,
                        )
                    else:
                        badge2 = "badge-good" if avg_util_pct <= 40 else (
                            "badge-fair" if avg_util_pct <= 60 else "badge-poor")
                        st.markdown(
                            f'<span class="performance-badge {badge2}">{avg_util_pct:.0f}% Avg Util</span>',
                            unsafe_allow_html=True,
                        )

                with col3:
                    total_volume = float(np.nansum(raw["volume"]))
                    st.metric(
                        f"🚗 Total {mode_label} (period)",
                        f"{total_volume:,.0f}",
                    )
                    state_badge = (
                        "badge-good" if total_volume < 0.4 * THEORETICAL_LINK_CAPACITY_VPH * 24
                        else "badge-fair" if total_volume < 0.7 * THEORETICAL_LINK_CAPACITY_VPH * 24
                        else "badge-poor"
                    )
                    st.markdown(
                        f'<span class="performance-badge {state_badge}">Period Total</span>',
                        unsafe_allow_html=True,
                    )

                with col4:
                    st.metric(
                        "🎯 Demand Consistency",
                        f"{max(0, 100 - cv_bucket):.0f}%",
                        delta=f"CV: {cv_bucket:.1f}%",
                    )
                    label_cons = "Consistent" if cv_bucket < 30 else (
                        "Variable" if cv_bucket < 50 else "Highly Variable")
                    badge_cons = "badge-good" if cv_bucket < 30 else ("badge-fair" if cv_bucket < 50 else "badge-poor")
                    st.markdown(
                        f'<span class="performance-badge {badge_cons}">{label_cons}</span>',
                        unsafe_allow_html=True,
                    )

                with col5:
                    st.metric(
                        "⚠️ High Volume Hours",
                        f"{high_hours}",
                        delta=f"{risk_pct:.1f}% of time",
                    )
                    level_badge = (
                        "badge-critical" if risk_pct > 25 else
                        "badge-poor" if risk_pct > 15 else
                        "badge-fair" if risk_pct > 5 else
                        "badge-good"
                    )
                    level = (
                        "Very High" if risk_pct > 25 else
                        "High" if risk_pct > 15 else
                        "Moderate" if risk_pct > 5 else
                        "Low"
                    )
                    st.markdown(
                        f'<span class="performance-badge {level_badge}">{level} Risk</span>',
                        unsafe_allow_html=True,
                    )

            # -------- Charts Section --------
            st.subheader(f"📈 {mode_label} Volume Visualizations")
            if len(analysis_df) > 1:
                try:
                    fig_trend, fig_box, fig_matrix = create_volume_charts(
                        raw_hourly_df=raw,
                        granularity=granularity,
                        cap_vph=THEORETICAL_LINK_CAPACITY_VPH,
                        high_vph=HIGH_VOLUME_THRESHOLD_VPH,
                        mode_label=mode_label,
                    )
                    if fig_trend:
                        st.plotly_chart(fig_trend, use_container_width=True, config=PLOTLY_CONFIG)
                    colA, colB = st.columns(2)
                    with colA:
                        if fig_box:
                            st.plotly_chart(fig_box, use_container_width=True, config=PLOTLY_CONFIG)
                    with colB:
                        if fig_matrix:
                            st.plotly_chart(fig_matrix, use_container_width=True, config=PLOTLY_CONFIG)
                except Exception as e:
                    st.error(f"❌ Error creating volume charts: {e}")

            # -------- Risk Analysis Table --------
            st.subheader(f"🚨 Intersection Volume & Capacity Risk Analysis ({mode_label})")
            try:
                g = raw.groupby(["intersection_name", "direction"]).agg(
                    volume_mean=("volume", "mean"),
                    volume_max=("volume", "max"),
                    volume_std=("volume", "std"),
                    volume_count=("volume", "count"),
                ).reset_index()

                g["Peak_Capacity_Util"] = (
                        g["volume_max"] / THEORETICAL_LINK_CAPACITY_VPH * 100
                ).round(1)
                g["Avg_Capacity_Util"] = (
                        g["volume_mean"] / THEORETICAL_LINK_CAPACITY_VPH * 100
                ).round(1)
                g["Volume_Variability"] = (
                        g["volume_std"] / g["volume_mean"] * 100
                ).replace([np.inf, -np.inf], np.nan).fillna(0).round(1)
                g["Peak_Avg_Ratio"] = (
                        g["volume_max"] / g["volume_mean"]
                ).replace([np.inf, -np.inf], 0).fillna(0).round(1)

                g["🚨 Risk Score"] = (
                        0.5 * g["Peak_Capacity_Util"]
                        + 0.3 * g["Avg_Capacity_Util"]
                        + 0.2 * (g["Peak_Avg_Ratio"] * 10)
                ).round(1)

                g["⚠️ Risk Level"] = pd.cut(
                    g["🚨 Risk Score"],
                    bins=[0, 40, 60, 80, 90, 999],
                    labels=["🟢 Low Risk", "🟡 Moderate Risk", "🟠 High Risk", "🔴 Critical Risk", "🚨 Severe Risk"],
                    include_lowest=True,
                )
                g["🎯 Action Priority"] = pd.cut(
                    g["Peak_Capacity_Util"],
                    bins=[0, 60, 75, 90, 999],
                    labels=["🟢 Monitor", "🟡 Optimize", "🟠 Upgrade", "🔴 Urgent"],
                    include_lowest=True,
                )

                final = g[
                    [
                        "intersection_name",
                        "direction",
                        "⚠️ Risk Level",
                        "🎯 Action Priority",
                        "🚨 Risk Score",
                        "Peak_Capacity_Util",
                        "Avg_Capacity_Util",
                        "volume_mean",
                        "volume_max",
                        "Peak_Avg_Ratio",
                        "volume_count",
                    ]
                ].rename(
                    columns={
                        "intersection_name": "Intersection",
                        "direction": "Dir",
                        "Peak_Capacity_Util": "📊 Peak Capacity %",
                        "Avg_Capacity_Util": "📊 Avg Capacity %",
                        "volume_mean": f"Avg {mode_label} (vph)",
                        "volume_max": f"Peak {mode_label} (vph)",
                        "volume_count": "Data Points",
                    }
                ).sort_values("🚨 Risk Score", ascending=False)

                st.dataframe(
                    final.head(15),
                    use_container_width=True,
                    column_config={
                        "🚨 Risk Score": st.column_config.NumberColumn(
                            "🚨 Capacity Risk Score",
                            help="Composite of peak/avg util + peaking",
                            format="%.1f",
                            min_value=0,
                            max_value=120,
                        ),
                        "📊 Peak Capacity %": st.column_config.NumberColumn("📊 Peak Capacity %", format="%.1f%%"),
                        "📊 Avg Capacity %": st.column_config.NumberColumn("📊 Avg Capacity %", format="%.1f%%"),
                    },
                )

                st.download_button(
                    "⬇️ Download Risk Analysis Table (CSV)",
                    data=final.to_csv(index=False).encode("utf-8"),
                    file_name=f"vantage_{mode.lower()}_risk.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"❌ Error in risk analysis: {e}")

            # -------- Cycle Length Recommendations --------
            st.subheader("🔄 Cycle Length Recommendations for CVAG")
            try:
                # Prepare data for cycle length function (expects 'total_volume' column)
                cycle_df = raw.copy()
                cycle_df["total_volume"] = cycle_df["volume"]
                render_cycle_length_section(cycle_df)
            except Exception as e:
                st.error(f"❌ Error rendering cycle length section: {e}")

    except Exception as e:
        st.error(f"❌ Error processing VantageLive data: {e}")
        import traceback
        st.text("Debug info:")
        st.text(traceback.format_exc())