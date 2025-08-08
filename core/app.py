import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# === External project functions ===
from sidebar_functions import process_traffic_data, load_traffic_data, load_volume_data

# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="Active Transportation & Operations Management Dashboard",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Constants / Config
# =========================
THEORETICAL_LINK_CAPACITY_VPH = 1800  # used for capacity references
HIGH_VOLUME_THRESHOLD_VPH = 1200
CRITICAL_DELAY_SEC = 120
HIGH_DELAY_SEC = 60

# =========================
# CSS (kept your aesthetic, trimmed a bit)
# =========================
st.markdown("""
<style>
    .main-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 15px; padding: 2rem; margin: 1rem 0; color: white;
        box-shadow: 0 8px 32px rgba(30, 60, 114, 0.3);
    }
    .context-header {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem; border-radius: 15px; margin: 1rem 0 2rem; color: white; text-align: center;
        box-shadow: 0 8px 32px rgba(79, 172, 254, 0.3); backdrop-filter: blur(10px);
    }
    .context-header h2 { margin: 0; font-size: 2rem; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.3); }
    .context-header p { margin: 1rem 0 0; font-size: 1.1rem; opacity: 0.9; font-weight: 300; }
    @media (prefers-color-scheme: dark) { .context-header { background: linear-gradient(135deg, #2980b9 0%, #3498db 100%); } }

    .metric-container { background: rgba(79, 172, 254, 0.1); border: 1px solid rgba(79, 172, 254, 0.3);
        border-radius: 15px; padding: 1.5rem; margin: 1rem 0; backdrop-filter: blur(10px); transition: all 0.3s ease; }
    .metric-container:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(79, 172, 254, 0.2); }

    .insight-box {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.15) 0%, rgba(0, 242, 254, 0.15) 100%);
        border-left: 5px solid #4facfe; border-radius: 12px; padding: 1.25rem 1.5rem; margin: 1.25rem 0;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.1);
    }
    .insight-box h4 { color: #1e3c72; margin-top: 0; font-weight: 600; }

    .performance-badge { display: inline-block; padding: 0.35rem 0.9rem; border-radius: 25px; font-size: 0.85rem;
        font-weight: 600; margin: 0.2rem; border: 2px solid transparent; transition: all 0.3s ease; }
    .performance-badge:hover { transform: scale(1.05); border-color: rgba(255,255,255,0.25); }
    .badge-excellent { background: linear-gradient(45deg, #2ecc71, #27ae60); color: white; }
    .badge-good { background: linear-gradient(45deg, #3498db, #2980b9); color: white; }
    .badge-fair { background: linear-gradient(45deg, #f39c12, #e67e22); color: white; }
    .badge-poor { background: linear-gradient(45deg, #e74c3c, #c0392b); color: white; }
    .badge-critical { background: linear-gradient(45deg, #e74c3c, #8e44ad); color: white; animation: pulse 2s infinite; }
    @keyframes pulse { 0% {opacity:1} 50% {opacity:.7} 100% {opacity:1} }

    .stTabs [data-baseweb="tab-list"] { gap: 16px; }
    .stTabs [data-baseweb="tab"] { height: 56px; padding: 0 18px; border-radius: 12px;
        background: rgba(79, 172, 254, 0.1); border: 1px solid rgba(79, 172, 254, 0.2); }

    .chart-container { background: rgba(79, 172, 254, 0.05); border-radius: 15px; padding: 1rem; margin: 1rem 0;
        border: 1px solid rgba(79, 172, 254, 0.1); }

    .volume-metric { background: linear-gradient(135deg, rgba(52, 152, 219, 0.1), rgba(41, 128, 185, 0.1));
        border: 1px solid rgba(52, 152, 219, 0.3); border-radius: 12px; padding: 1rem; margin: 0.5rem 0; }

    /* Reduce plotly modebar contrast */
    .modebar { filter: saturate(0.85) opacity(0.9); }
</style>
""", unsafe_allow_html=True)

# =========================
# Title / Intro
# =========================
st.markdown("""
<div class="main-container">
    <h1 style="text-align:center; margin:0; font-size:2.5rem; font-weight:800;">
        🛣️ Active Transportation & Operations Management Dashboard
    </h1>
    <p style="text-align:center; margin-top:1rem; font-size:1.1rem; opacity:0.9;">
        Advanced Traffic Engineering & Operations Management Platform
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    font-size: 1.05rem; font-weight: 400; color: var(--text-color);
    background: linear-gradient(135deg, rgba(79, 172, 254, 0.1), rgba(0, 242, 254, 0.05));
    padding: 1.5rem; border-radius: 18px; box-shadow: 0 8px 32px rgba(79,172,254,0.08);
    margin: 1.25rem 0; line-height: 1.7; border: 1px solid rgba(79,172,254,0.2); backdrop-filter: blur(8px);
">
    <div style="text-align:center; margin-bottom: 0.5rem;">
        <strong style="font-size: 1.2rem; color: #2980b9;">🚀 The ADVANTEC Platform</strong>
    </div>
    <p>Leverages <strong>millions of data points</strong> trained on advanced Machine Learning algorithms to optimize traffic flow, reduce travel time, minimize fuel consumption, and decrease greenhouse gas emissions across the Coachella Valley transportation network.</p>
    <p><strong>Key Capabilities:</strong> Real-time anomaly detection • Intelligent cycle length optimization • Predictive traffic modeling • Performance analytics</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, #3498db, #2980b9); color: white; padding: 1.1rem; border-radius: 15px;
    margin: 1rem 0; text-align: center; box-shadow: 0 6px 20px rgba(52, 152, 219, 0.25);">
    <h3 style="margin:0; font-weight:600;">🔍 Research Question</h3>
    <p style="margin: 0.45rem 0 0; font-size: 1.0rem;">What are the main bottlenecks (slowest intersections) on Washington St that are most prone to causing increased travel times?</p>
</div>
""", unsafe_allow_html=True)

# =========================
# Helpers / Utilities
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
    # Basic column sanity
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

def get_performance_rating(score: float):
    if score > 80: return "🟢 Excellent", "badge-excellent"
    if score > 60: return "🔵 Good", "badge-good"
    if score > 40: return "🟡 Fair", "badge-fair"
    if score > 20: return "🟠 Poor", "badge-poor"
    return "🔴 Critical", "badge-critical"

def performance_chart(data: pd.DataFrame, metric_type: str = "delay"):
    if data.empty: return None
    metric_type = metric_type.lower().strip()
    if metric_type == "delay":
        y_col, title, color = "average_delay", "Traffic Delay Analysis", "#e74c3c"
    else:
        y_col, title, color = "average_traveltime", "Travel Time Analysis", "#3498db"

    dd = data.dropna(subset=["local_datetime", y_col]).sort_values("local_datetime")

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Time Series Analysis", "Distribution Analysis"),
        vertical_spacing=0.1
    )
    fig.add_trace(
        go.Scatter(
            x=dd["local_datetime"], y=dd[y_col],
            mode="lines+markers", name=f"{metric_type.title()} Trend",
            line=dict(color=color, width=2), marker=dict(size=4)
        ), row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=dd[y_col], nbinsx=30, name=f"{metric_type.title()} Distribution", marker_color=color, opacity=0.75),
        row=2, col=1
    )
    fig.update_layout(height=600, title=title, showlegend=True, template="plotly_white",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig

def volume_charts(data: pd.DataFrame):
    if data.empty: return None, None, None
    dd = data.dropna(subset=["local_datetime", "total_volume", "intersection_name"]).copy()
    dd.sort_values("local_datetime", inplace=True)

    # 1) Trend by intersection
    fig1 = px.line(
        dd, x="local_datetime", y="total_volume", color="intersection_name",
        title="📈 Traffic Volume Trends by Intersection",
        labels={"total_volume": "Volume (vehicles/hour)", "local_datetime": "Date/Time"},
        template="plotly_white"
    )
    fig1.update_layout(height=500, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # 2) Distribution + Hourly heatmap
    fig2 = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Volume Distribution by Intersection", "Hourly Avg Volume Heatmap"),
        vertical_spacing=0.12
    )

    # Box plots
    for name, g in dd.groupby("intersection_name", sort=False):
        fig2.add_trace(
            go.Box(y=g["total_volume"], name=name, boxpoints="outliers"),
            row=1, col=1
        )

    dd["hour"] = dd["local_datetime"].dt.hour
    hourly_avg = dd.groupby(["hour", "intersection_name"], as_index=False)["total_volume"].mean()
    # Pivot with clean column names
    hourly_pivot = hourly_avg.pivot(index="intersection_name", columns="hour", values="total_volume").sort_index()

    fig2.add_trace(
        go.Heatmap(
            z=hourly_pivot.values,
            x=hourly_pivot.columns,
            y=hourly_pivot.index,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Avg Volume (vph)")
        ),
        row=2, col=1
    )
    fig2.update_layout(height=800, title="📊 Volume Distribution & Capacity Analysis", template="plotly_white",
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")

    # 3) Peak hour by intersection
    hourly_volume = dd.groupby(["hour", "intersection_name"], as_index=False)["total_volume"].mean()
    fig3 = px.line(
        hourly_volume, x="hour", y="total_volume", color="intersection_name",
        title="🕐 Average Hourly Volume Patterns",
        labels={"total_volume": "Average Volume (vph)", "hour": "Hour of Day"},
        template="plotly_white"
    )
    fig3.add_hline(y=THEORETICAL_LINK_CAPACITY_VPH, line_dash="dash", line_color="red",
                   annotation_text=f"Theoretical Capacity ({THEORETICAL_LINK_CAPACITY_VPH:,} vph)")
    fig3.add_hline(y=HIGH_VOLUME_THRESHOLD_VPH, line_dash="dot", line_color="orange",
                   annotation_text=f"High Volume Threshold ({HIGH_VOLUME_THRESHOLD_VPH:,} vph)")
    fig3.update_layout(height=500, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig1, fig2, fig3

def date_range_preset_controls(min_date: datetime.date, max_date: datetime.date, key_prefix: str):
    """Presets that don't clobber user custom ranges and persist in session_state."""
    k_range = f"{key_prefix}_range"
    if k_range not in st.session_state:
        st.session_state[k_range] = (min_date, max_date)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📅 Last 7 Days", key=f"{key_prefix}_7d"):
            st.session_state[k_range] = (max_date - timedelta(days=7), max_date)
    with c2:
        if st.button("📅 Last 30 Days", key=f"{key_prefix}_30d"):
            st.session_state[k_range] = (max_date - timedelta(days=30), max_date)
    with c3:
        if st.button("📅 Full Range", key=f"{key_prefix}_full"):
            st.session_state[k_range] = (min_date, max_date)

    custom = st.date_input(
        "Custom Date Range",
        value=st.session_state[k_range],
        min_value=min_date,
        max_value=max_date,
        key=f"{key_prefix}_custom"
    )
    # Keep session_state aligned if user changes the picker
    if custom != st.session_state[k_range]:
        st.session_state[k_range] = custom

    return st.session_state[k_range]

# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["🚧 Performance & Delay Analysis", "📊 Traffic Demand & Capacity Analysis"])

# -------------------------
# TAB 1: Performance / Travel Time
# -------------------------
with tab1:
    st.header("🚧 Comprehensive Performance & Travel Time Analysis")

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Loading corridor performance data...")
    progress_bar.progress(25)

    corridor_df = get_corridor_df()
    progress_bar.progress(100)

    if corridor_df.empty:
        st.error("❌ Failed to load corridor data. Please check your data sources.")
    else:
        status_text.text("✅ Data loaded successfully!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()

        with st.sidebar:
            st.markdown("### 🚧 Performance Analysis Controls")

            seg_options = ["All Segments"] + sorted(corridor_df["segment_name"].dropna().unique().tolist())
            corridor = st.selectbox("🛣️ Select Corridor Segment", seg_options,
                                    help="Choose a specific segment or analyze all segments")

            min_date = corridor_df["local_datetime"].dt.date.min()
            max_date = corridor_df["local_datetime"].dt.date.max()
            st.markdown("#### 📅 Analysis Period")
            date_range = date_range_preset_controls(min_date, max_date, key_prefix="perf")

            st.markdown("#### ⏰ Analysis Settings")
            granularity = st.selectbox("Data Aggregation", ["Hourly", "Daily", "Weekly", "Monthly"],
                                       index=0, help="Higher aggregation smooths trends but may hide peaks")

            time_filter, start_hour, end_hour = None, None, None
            if granularity == "Hourly":
                time_filter = st.selectbox(
                    "Time Period Focus",
                    ["All Hours", "Peak Hours (7–9 AM, 4–6 PM)", "AM Peak (7–9 AM)", "PM Peak (4–6 PM)", "Off-Peak", "Custom Range"]
                )
                if time_filter == "Custom Range":
                    c1, c2 = st.columns(2)
                    with c1:
                        start_hour = st.number_input("Start Hour (0–23)", 0, 23, 7, step=1)
                    with c2:
                        end_hour = st.number_input("End Hour (1–24)", 1, 24, 18, step=1)

        if len(date_range) == 2:
            try:
                # Segment filter
                base_df = corridor_df.copy()
                if corridor != "All Segments":
                    base_df = base_df[base_df["segment_name"] == corridor]

                # Guard for empty after filter
                if base_df.empty:
                    st.warning("⚠️ No data for the selected segment.")
                else:
                    # Processed (aggregated) data
                    filtered_data = process_traffic_data(
                        base_df,
                        date_range,
                        granularity,
                        time_filter if granularity == "Hourly" else None,
                        start_hour, end_hour
                    )

                    if filtered_data.empty:
                        st.warning("⚠️ No data available for the selected filters.")
                    else:
                        # Context header
                        total_records = len(filtered_data)
                        data_span = (date_range[1] - date_range[0]).days + 1
                        time_context = f" • {time_filter}" if (granularity == "Hourly" and time_filter) else ""

                        st.markdown(f"""
                        <div class="context-header">
                            <h2>📊 Performance Dashboard: {corridor}</h2>
                            <p>📅 {date_range[0].strftime('%b %d, %Y')} to {date_range[1].strftime('%b %d, %Y')}
                            ({data_span} days) • {granularity} Aggregation{time_context}</p>
                            <p>📈 Analyzing {total_records:,} data points across the selected period</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Raw window for KPIs (hourly base)
                        raw_data = base_df[
                            (base_df["local_datetime"].dt.date >= date_range[0]) &
                            (base_df["local_datetime"].dt.date <= date_range[1])
                        ].copy()

                        # KPI block
                        st.subheader("🎯 Key Performance Indicators")
                        if raw_data.empty:
                            st.info("No raw hourly data in this window.")
                        else:
                            # Clean NaNs
                            for col in ["average_delay", "average_traveltime", "average_speed"]:
                                if col in raw_data:
                                    raw_data[col] = pd.to_numeric(raw_data[col], errors="coerce")

                            col1, col2, col3, col4, col5 = st.columns(5)

                            with col1:
                                worst_delay = float(np.nanmax(raw_data["average_delay"])) if raw_data["average_delay"].notna().any() else 0.0
                                p95_delay = float(np.nanpercentile(raw_data["average_delay"].dropna(), 95)) if raw_data["average_delay"].notna().any() else 0.0
                                rating, badge = get_performance_rating(100 - min(worst_delay / 2, 100))
                                st.metric("🚨 Peak Delay", f"{worst_delay:.1f}s", delta=f"95th: {p95_delay:.1f}s")
                                st.markdown(f'<span class="performance-badge {badge}">{rating}</span>', unsafe_allow_html=True)

                            with col2:
                                worst_tt = float(np.nanmax(raw_data["average_traveltime"])) if raw_data["average_traveltime"].notna().any() else 0.0
                                avg_tt = float(np.nanmean(raw_data["average_traveltime"])) if raw_data["average_traveltime"].notna().any() else 0.0
                                tt_delta = ((worst_tt - avg_tt) / avg_tt * 100) if avg_tt > 0 else 0
                                impact_rating, badge = get_performance_rating(100 - min(max(tt_delta,0), 100))
                                st.metric("⏱️ Peak Travel Time", f"{worst_tt:.1f}min", delta=f"+{tt_delta:.0f}% vs avg")
                                st.markdown(f'<span class="performance-badge {badge}">{impact_rating}</span>', unsafe_allow_html=True)

                            with col3:
                                slowest = float(np.nanmin(raw_data["average_speed"])) if raw_data["average_speed"].notna().any() else 0.0
                                avg_speed = float(np.nanmean(raw_data["average_speed"])) if raw_data["average_speed"].notna().any() else 0.0
                                speed_drop = ((avg_speed - slowest) / avg_speed * 100) if avg_speed > 0 else 0
                                speed_rating, badge = get_performance_rating(min(slowest * 2, 100))
                                st.metric("🐌 Minimum Speed", f"{slowest:.1f}mph", delta=f"-{speed_drop:.0f}% vs avg")
                                st.markdown(f'<span class="performance-badge {badge}">{speed_rating}</span>', unsafe_allow_html=True)

                            with col4:
                                if avg_tt > 0:
                                    cv_tt = float(np.nanstd(raw_data["average_traveltime"]) / avg_tt) * 100
                                else:
                                    cv_tt = 0.0
                                reliability = max(0, 100 - cv_tt)
                                rel_rating, badge = get_performance_rating(reliability)
                                st.metric("🎯 Reliability Index", f"{reliability:.0f}%", delta=f"CV: {cv_tt:.1f}%")
                                st.markdown(f'<span class="performance-badge {badge}">{rel_rating}</span>', unsafe_allow_html=True)

                            with col5:
                                high_delay_pct = (raw_data["average_delay"] > HIGH_DELAY_SEC).mean() * 100 if raw_data["average_delay"].notna().any() else 0.0
                                hours = int((raw_data["average_delay"] > HIGH_DELAY_SEC).sum()) if raw_data["average_delay"].notna().any() else 0
                                freq_rating, badge = get_performance_rating(100 - high_delay_pct)
                                st.metric("⚠️ Congestion Frequency", f"{high_delay_pct:.1f}%", delta=f"{hours} hours")
                                st.markdown(f'<span class="performance-badge {badge}">{freq_rating}</span>', unsafe_allow_html=True)

                        # Charts
                        if len(filtered_data) > 1:
                            st.subheader("📈 Performance Trends")
                            v1, v2 = st.columns(2)
                            with v1:
                                dc = performance_chart(filtered_data, "delay")
                                if dc: st.plotly_chart(dc, use_container_width=True)
                            with v2:
                                tc = performance_chart(filtered_data, "travel")
                                if tc: st.plotly_chart(tc, use_container_width=True)

                        # Insights
                        if not raw_data.empty:
                            worst_delay = float(np.nanmax(raw_data["average_delay"])) if raw_data["average_delay"].notna().any() else 0.0
                            avg_tt = float(np.nanmean(raw_data["average_traveltime"])) if raw_data["average_traveltime"].notna().any() else 0.0
                            worst_tt = float(np.nanmax(raw_data["average_traveltime"])) if raw_data["average_traveltime"].notna().any() else 0.0
                            tt_delta = ((worst_tt - avg_tt) / avg_tt * 100) if avg_tt > 0 else 0
                            if avg_tt > 0:
                                cv_tt = float(np.nanstd(raw_data["average_traveltime"]) / avg_tt) * 100
                            else:
                                cv_tt = 0.0
                            reliability = max(0, 100 - cv_tt)
                            high_delay_pct = (raw_data["average_delay"] > HIGH_DELAY_SEC).mean() * 100 if raw_data["average_delay"].notna().any() else 0.0
                            st.markdown(f"""
                            <div class="insight-box">
                                <h4>💡 Advanced Performance Insights</h4>
                                <p><strong>📊 Data Overview:</strong> {len(filtered_data):,} {granularity.lower()} observations across {(date_range[1]-date_range[0]).days+1} days.</p>
                                <p><strong>🚨 Peaks:</strong> Delay up to {worst_delay:.0f}s ({worst_delay/60:.1f} min) • Travel time up to {worst_tt:.1f} min (+{tt_delta:.0f}% vs avg).</p>
                                <p><strong>🎯 Reliability:</strong> {reliability:.0f}% travel time reliability • Delays > {HIGH_DELAY_SEC}s occur {high_delay_pct:.1f}% of hours.</p>
                                <p><strong>📌 Action:</strong> {"Critical intervention needed" if worst_delay > CRITICAL_DELAY_SEC else "Optimization recommended" if worst_delay > HIGH_DELAY_SEC else "Monitor trends"}.</p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Bottleneck table
                        st.subheader("🚨 Comprehensive Bottleneck Analysis")
                        if not raw_data.empty:
                            try:
                                g = raw_data.groupby(["segment_name", "direction"]).agg(
                                    average_delay_mean=("average_delay", "mean"),
                                    average_delay_max=("average_delay", "max"),
                                    average_traveltime_mean=("average_traveltime", "mean"),
                                    average_traveltime_max=("average_traveltime", "max"),
                                    average_speed_mean=("average_speed", "mean"),
                                    average_speed_min=("average_speed", "min"),
                                    n=("average_delay", "count"),
                                ).reset_index()

                                # Robust scoring (normalize to avoid one metric dominating)
                                def _norm(s):
                                    s = s.astype(float)
                                    mn, mx = np.nanmin(s), np.nanmax(s)
                                    if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
                                        return (s - mn) / (mx - mn)
                                    return pd.Series(np.zeros(len(s)), index=s.index)

                                score = (
                                    0.45*_norm(g["average_delay_max"]) +
                                    0.35*_norm(g["average_delay_mean"]) +
                                    0.20*_norm(g["average_traveltime_max"])
                                ) * 100
                                g["Bottleneck_Score"] = score.round(1)

                                bins = [-0.1, 20, 40, 60, 80, 200]
                                labels = ['🟢 Excellent', '🔵 Good', '🟡 Fair', '🟠 Poor', '🔴 Critical']
                                g["🎯 Performance Rating"] = pd.cut(g["Bottleneck_Score"], bins=bins, labels=labels)

                                final = g[[
                                    "segment_name", "direction", "🎯 Performance Rating", "Bottleneck_Score",
                                    "average_delay_mean", "average_delay_max",
                                    "average_traveltime_mean", "average_traveltime_max",
                                    "average_speed_mean", "average_speed_min", "n"
                                ]].rename(columns={
                                    "segment_name": "Segment",
                                    "direction": "Dir",
                                    "average_delay_mean": "Avg Delay (s)",
                                    "average_delay_max": "Peak Delay (s)",
                                    "average_traveltime_mean": "Avg Time (min)",
                                    "average_traveltime_max": "Peak Time (min)",
                                    "average_speed_mean": "Avg Speed (mph)",
                                    "average_speed_min": "Min Speed (mph)",
                                    "n": "Obs"
                                }).sort_values("Bottleneck_Score", ascending=False)

                                st.dataframe(
                                    final.head(15),
                                    use_container_width=True,
                                    column_config={
                                        "Bottleneck_Score": st.column_config.NumberColumn(
                                            "🚨 Impact Score", help="Composite (0–100); higher ⇒ worse", format="%.1f"
                                        )
                                    }
                                )

                                # Downloads
                                st.download_button(
                                    "⬇️ Download Bottleneck Table (CSV)",
                                    data=final.to_csv(index=False).encode("utf-8"),
                                    file_name="bottlenecks.csv",
                                    mime="text/csv"
                                )
                                st.download_button(
                                    "⬇️ Download Filtered Performance (CSV)",
                                    data=filtered_data.to_csv(index=False).encode("utf-8"),
                                    file_name="performance_filtered.csv",
                                    mime="text/csv"
                                )
                            except Exception as e:
                                st.error(f"❌ Error in performance analysis: {e}")

            except Exception as e:
                st.error(f"❌ Error processing traffic data: {e}")
        else:
            st.warning("⚠️ Please select both start and end dates to proceed.")

# -------------------------
# TAB 2: Volume / Capacity
# -------------------------
with tab2:
    st.header("📊 Advanced Traffic Demand & Capacity Analysis")

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Loading traffic demand data...")
    progress_bar.progress(25)

    volume_df = get_volume_df()
    progress_bar.progress(100)

    if volume_df.empty:
        st.error("❌ Failed to load volume data. Please check your data sources.")
    else:
        status_text.text("✅ Volume data loaded successfully!")
        time.sleep(0.5); progress_bar.empty(); status_text.empty()

        with st.sidebar:
            st.markdown("### 📊 Volume Analysis Controls")
            intersections = ["All Intersections"] + sorted(volume_df["intersection_name"].dropna().unique().tolist())
            intersection = st.selectbox("🚦 Select Intersection", intersections)

            min_date = volume_df["local_datetime"].dt.date.min()
            max_date = volume_df["local_datetime"].dt.date.max()
            st.markdown("#### 📅 Analysis Period")
            date_range_vol = date_range_preset_controls(min_date, max_date, key_prefix="vol")

            st.markdown("#### ⏰ Analysis Settings")
            granularity_vol = st.selectbox("Data Aggregation", ["Hourly", "Daily", "Weekly", "Monthly"], index=0)

            direction_options = ["All Directions"] + sorted(volume_df["direction"].dropna().unique().tolist())
            direction_filter = st.selectbox("🔄 Direction Filter", direction_options)

        if len(date_range_vol) == 2:
            try:
                base_df = volume_df.copy()
                if intersection != "All Intersections":
                    base_df = base_df[base_df["intersection_name"] == intersection]
                if direction_filter != "All Directions":
                    base_df = base_df[base_df["direction"] == direction_filter]

                if base_df.empty:
                    st.warning("⚠️ No volume data for the selected filters.")
                else:
                    filtered_volume_data = process_traffic_data(base_df, date_range_vol, granularity_vol)

                    if filtered_volume_data.empty:
                        st.warning("⚠️ No volume data available for the selected range.")
                    else:
                        span = (date_range_vol[1] - date_range_vol[0]).days + 1
                        st.markdown(f"""
                        <div class="context-header">
                            <h2>📊 Volume Analysis: {intersection}</h2>
                            <p>📅 {date_range_vol[0].strftime('%b %d, %Y')} to {date_range_vol[1].strftime('%b %d, %Y')}
                            ({span} days) • {granularity_vol} Aggregation</p>
                            <p>📈 {len(filtered_volume_data):,} observations • Direction: {direction_filter}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Raw hourly for KPI
                        raw = base_df[
                            (base_df["local_datetime"].dt.date >= date_range_vol[0]) &
                            (base_df["local_datetime"].dt.date <= date_range_vol[1])
                        ].copy()

                        st.subheader("🚦 Traffic Demand Performance Indicators")
                        if raw.empty:
                            st.info("No raw hourly volume in this window.")
                        else:
                            raw["total_volume"] = pd.to_numeric(raw["total_volume"], errors="coerce")
                            col1, col2, col3, col4, col5 = st.columns(5)

                            with col1:
                                peak = float(np.nanmax(raw["total_volume"])) if raw["total_volume"].notna().any() else 0
                                p95 = float(np.nanpercentile(raw["total_volume"].dropna(), 95)) if raw["total_volume"].notna().any() else 0
                                util = (peak / THEORETICAL_LINK_CAPACITY_VPH) * 100 if THEORETICAL_LINK_CAPACITY_VPH else 0
                                if util > 90: badge = "badge-critical"
                                elif util > 75: badge = "badge-poor"
                                elif util > 60: badge = "badge-fair"
                                else: badge = "badge-good"
                                st.metric("🔥 Peak Demand", f"{peak:,.0f} vph", delta=f"95th: {p95:,.0f}")
                                st.markdown(f'<span class="performance-badge {badge}">{util:.0f}% Capacity</span>', unsafe_allow_html=True)

                            with col2:
                                avg = float(np.nanmean(raw["total_volume"])) if raw["total_volume"].notna().any() else 0
                                med = float(np.nanmedian(raw["total_volume"])) if raw["total_volume"].notna().any() else 0
                                st.metric("📊 Average Demand", f"{avg:,.0f} vph", delta=f"Median: {med:,.0f}")
                                avg_util = (avg / THEORETICAL_LINK_CAPACITY_VPH) * 100 if THEORETICAL_LINK_CAPACITY_VPH else 0
                                badge = "badge-good" if avg_util <= 40 else ("badge-fair" if avg_util <= 60 else "badge-poor")
                                st.markdown(f'<span class="performance-badge {badge}">{avg_util:.0f}% Avg Util</span>', unsafe_allow_html=True)

                            with col3:
                                ratio = (peak / avg) if avg > 0 else 0
                                st.metric("📈 Peak/Average Ratio", f"{ratio:.1f}x", help="Higher ⇒ more peaked demand")
                                badge = "badge-good" if ratio <= 2 else ("badge-fair" if ratio <= 3 else "badge-poor")
                                state = "Low" if ratio <= 2 else ("Moderate" if ratio <= 3 else "High")
                                st.markdown(f'<span class="performance-badge {badge}">{state} Peaking</span>', unsafe_allow_html=True)

                            with col4:
                                cv = (float(np.nanstd(raw["total_volume"])) / avg * 100) if avg > 0 else 0
                                st.metric("🎯 Demand Consistency", f"{max(0, 100 - cv):.0f}%", delta=f"CV: {cv:.1f}%")
                                badge = "badge-good" if cv < 30 else ("badge-fair" if cv < 50 else "badge-poor")
                                label = "Consistent" if cv < 30 else ("Variable" if cv < 50 else "Highly Variable")
                                st.markdown(f'<span class="performance-badge {badge}">{label}</span>', unsafe_allow_html=True)

                            with col5:
                                high_hours = int((raw["total_volume"] > HIGH_VOLUME_THRESHOLD_VPH).sum())
                                total_hours = int(raw["total_volume"].count())
                                risk_pct = (high_hours / total_hours * 100) if total_hours > 0 else 0
                                st.metric("⚠️ High Volume Hours", f"{high_hours}", delta=f"{risk_pct:.1f}% of time")
                                if risk_pct > 25: badge = "badge-critical"
                                elif risk_pct > 15: badge = "badge-poor"
                                elif risk_pct > 5: badge = "badge-fair"
                                else: badge = "badge-good"
                                level = "Very High" if risk_pct > 25 else ("High" if risk_pct > 15 else ("Moderate" if risk_pct > 5 else "Low"))
                                st.markdown(f'<span class="performance-badge {badge}">{level} Risk</span>', unsafe_allow_html=True)

                        # Charts
                        st.subheader("📈 Volume Analysis Visualizations")
                        if len(filtered_volume_data) > 1:
                            c1, c2 = None, None
                            chart1, chart2, chart3 = volume_charts(filtered_volume_data)
                            if chart1: st.plotly_chart(chart1, use_container_width=True)
                            colA, colB = st.columns(2)
                            with colA:
                                if chart3: st.plotly_chart(chart3, use_container_width=True)
                            with colB:
                                if chart2: st.plotly_chart(chart2, use_container_width=True)

                        # Insights
                        if not raw.empty:
                            peak = float(np.nanmax(raw["total_volume"])) if raw["total_volume"].notna().any() else 0
                            avg = float(np.nanmean(raw["total_volume"])) if raw["total_volume"].notna().any() else 0
                            ratio = (peak / avg) if avg > 0 else 0
                            cv = (float(np.nanstd(raw["total_volume"])) / avg * 100) if avg > 0 else 0
                            util = (peak / THEORETICAL_LINK_CAPACITY_VPH) * 100 if THEORETICAL_LINK_CAPACITY_VPH else 0
                            high_hours = int((raw["total_volume"] > HIGH_VOLUME_THRESHOLD_VPH).sum())
                            total_hours = int(raw["total_volume"].count())
                            risk_pct = (high_hours / total_hours * 100) if total_hours > 0 else 0

                            action = ("Immediate capacity expansion needed" if util > 90 else
                                      "Consider signal optimization" if util > 75 else
                                      "Monitor trends & optimize timing" if util > 60 else
                                      "Current capacity appears adequate")

                            st.markdown(f"""
                            <div class="insight-box">
                                <h4>💡 Advanced Volume Analysis Insights</h4>
                                <p><strong>📊 Capacity:</strong> Peak {peak:,.0f} vph ({util:.0f}% of {THEORETICAL_LINK_CAPACITY_VPH:,} vph) • Avg {avg:,.0f} vph.</p>
                                <p><strong>📈 Demand Shape:</strong> {ratio:.1f}× peak-to-average • Consistency {max(0,100-cv):.0f}%.</p>
                                <p><strong>⚠️ Risk:</strong> >{HIGH_VOLUME_THRESHOLD_VPH:,} vph occurs {high_hours} hours ({risk_pct:.1f}% of period).</p>
                                <p><strong>🎯 Recommendation:</strong> {action}.</p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Ranking
                        st.subheader("🚨 Intersection Volume & Capacity Risk Analysis")
                        try:
                            g = raw.groupby(["intersection_name", "direction"]).agg(
                                total_volume_mean=("total_volume", "mean"),
                                total_volume_max=("total_volume", "max"),
                                total_volume_std=("total_volume", "std"),
                                total_volume_count=("total_volume", "count")
                            ).reset_index()

                            g["Peak_Capacity_Util"] = (g["total_volume_max"] / THEORETICAL_LINK_CAPACITY_VPH * 100).round(1)
                            g["Avg_Capacity_Util"] = (g["total_volume_mean"] / THEORETICAL_LINK_CAPACITY_VPH * 100).round(1)
                            g["Volume_Variability"] = (g["total_volume_std"] / g["total_volume_mean"] * 100).replace([np.inf, -np.inf], np.nan).fillna(0).round(1)
                            g["Peak_Avg_Ratio"] = (g["total_volume_max"] / g["total_volume_mean"]).replace([np.inf, -np.inf], 0).fillna(0).round(1)

                            g["🚨 Risk Score"] = (
                                0.5 * g["Peak_Capacity_Util"] +
                                0.3 * g["Avg_Capacity_Util"] +
                                0.2 * (g["Peak_Avg_Ratio"] * 10)
                            ).round(1)

                            g["⚠️ Risk Level"] = pd.cut(
                                g["🚨 Risk Score"], bins=[0, 40, 60, 80, 90, 999],
                                labels=["🟢 Low Risk", "🟡 Moderate Risk", "🟠 High Risk", "🔴 Critical Risk", "🚨 Severe Risk"],
                                include_lowest=True
                            )
                            g["🎯 Action Priority"] = pd.cut(
                                g["Peak_Capacity_Util"], bins=[0, 60, 75, 90, 999],
                                labels=["🟢 Monitor", "🟡 Optimize", "🟠 Upgrade", "🔴 Urgent"], include_lowest=True
                            )

                            final = g[[
                                "intersection_name", "direction", "⚠️ Risk Level", "🎯 Action Priority", "🚨 Risk Score",
                                "Peak_Capacity_Util", "Avg_Capacity_Util",
                                "total_volume_mean", "total_volume_max", "Peak_Avg_Ratio", "total_volume_count"
                            ]].rename(columns={
                                "intersection_name": "Intersection",
                                "direction": "Dir",
                                "Peak_Capacity_Util": "📊 Peak Capacity %",
                                "Avg_Capacity_Util": "📊 Avg Capacity %",
                                "total_volume_mean": "Avg Volume (vph)",
                                "total_volume_max": "Peak Volume (vph)",
                                "total_volume_count": "Data Points"
                            }).sort_values("🚨 Risk Score", ascending=False)

                            st.dataframe(
                                final.head(15),
                                use_container_width=True,
                                column_config={
                                    "🚨 Risk Score": st.column_config.NumberColumn(
                                        "🚨 Capacity Risk Score", help="Composite of peak/avg util + peaking", format="%.1f", min_value=0, max_value=120
                                    ),
                                    "📊 Peak Capacity %": st.column_config.NumberColumn("📊 Peak Capacity %", format="%.1f%%"),
                                    "📊 Avg Capacity %": st.column_config.NumberColumn("📊 Avg Capacity %", format="%.1f%%"),
                                }
                            )

                            st.download_button(
                                "⬇️ Download Capacity Risk Table (CSV)",
                                data=final.to_csv(index=False).encode("utf-8"),
                                file_name="capacity_risk.csv",
                                mime="text/csv"
                            )
                            st.download_button(
                                "⬇️ Download Filtered Volume (CSV)",
                                data=filtered_volume_data.to_csv(index=False).encode("utf-8"),
                                file_name="volume_filtered.csv",
                                mime="text/csv"
                            )
                        except Exception as e:
                            st.error(f"❌ Error in volume analysis: {e}")
                            simple = raw.groupby(["intersection_name", "direction"]).agg(
                                Avg=("total_volume", "mean"), Peak=("total_volume", "max")
                            ).reset_index().sort_values("Peak", ascending=False)
                            st.dataframe(simple, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error processing volume data: {e}")
                st.info("Please check your data sources and try again.")
        else:
            st.warning("⚠️ Please select both start and end dates to proceed with the volume analysis.")

# =========================
# Footer
# =========================
st.markdown("""
---
<div style="text-align:center; padding: 1.25rem; background: linear-gradient(135deg, rgba(79,172,254,0.1), rgba(0,242,254,0.05));
    border-radius: 15px; margin-top: 1rem; border: 1px solid rgba(79,172,254,0.2);">
    <h4 style="color:#2980b9; margin-bottom: 0.5rem;">🛣️ Active Transportation & Operations Management Dashboard</h4>
    <p style="opacity:0.8; margin:0;">Powered by Advanced Machine Learning • Real-time Traffic Intelligence • Sustainable Transportation Solutions</p>
    <p style="opacity:0.6; margin-top:0.25rem; font-size:0.9rem;">© 2025 ADVANTEC Platform - Optimizing Transportation Networks</p>
</div>
""", unsafe_allow_html=True)
