# Python
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Plotly for chart helpers
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _fix_raw_url(url: str) -> str:
    """
    GitHub RAW URLs must be:
      https://raw.githubusercontent.com/<user>/<repo>/<branch>/<path>
    Some of your links used '/refs/heads/main/'. This converts them.
    """
    return url.replace("/refs/heads/", "/")


def _safe_to_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _normalize_acyclica_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tolerant column normalizer for Acyclica CSVs.
    """
    if df is None or df.empty:
        return df
    # Lower + strip spaces to catch variants, then map back to canon names
    lowmap = {c: "".join(str(c).strip().split()).lower() for c in df.columns}
    df = df.rename(columns=lowmap)
    canon = {
        "localdatetime": "local_datetime",
        "corridorid": "corridor_id",
        "direction": "direction",
        "metric": "metric",
        "strength": "Strength",
        "firsts": "Firsts",
        "lasts": "Lasts",
        "minimum": "Minimum",
        "maximum": "Maximum",
    }
    for src, tgt in canon.items():
        if src in df.columns:
            df = df.rename(columns={src: tgt})
    return df


# =========================
# Data loading
# =========================

# Iteris ClearGuide Data

@st.cache_data(show_spinner=False)
def load_traffic_data():
    """
    Load and combine all corridor traffic data from GitHub (Iteris-style).
    Optimized for memory and speed:
    - Fixes bad RAW URL pattern
    - Reads only necessary columns
    - Parses datetime on read
    - Downcasts numeric columns and categorizes strings
    """
    data_sources = {
        # Existing segments (Avenue 52 to Highway 111)
        "Avenue 52 → Calle Tampico": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/1_2_LONG_NSB_Ave52_CalleTampico_WashSt_1hr_septojuly.csv",
        "Calle Tampico → Village Shopping Ctr": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/2_3_LONG_NSB_CalleTampico_VillageShoppingCtr_WashSt_1hr_septojuly.csv",
        "Village Shopping Ctr → Avenue 50": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/3_4_LONG_NSB_VillageShoppingCtr_Avenue50_WashSt_1hr_septojuly.csv",
        "Avenue 50 → Sagebrush Ave": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/4_5_LONG_NSB_Ave50_SagebrushAve_WashSt_1hr_septojuly.csv",
        "Sagebrush Ave → Eisenhower Dr": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/5_6_LONG_NSB_SagebrushAve_EisenhowerDr_WashSt_1hr_septojuly.csv",
        "Eisenhower Dr → Avenue 48": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/6_7_LONG_NSB_EisenhowerDr_Avenue48_WashSt_1hr_septojuly.csv",
        "Avenue 48 → Avenue 47": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/7_8_LONG_NSB_Ave48_Ave47_WashSt_1hr_septojuly.csv",
        "Avenue 47 → Point Happy Simon": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/8_9_LONG_NSB_Ave47_PointHappySimon_WashSt_1hr_septojuly.csv",
        "Point Happy Simon → Hwy 111": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/9_10_LONG_NSB_PointHappySimon_WashSt_1hr_septojuly.csv",

        # New segments extending north from Highway 111
        "Hwy 111 → Channel Drive": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/10_11_LONG_NSB_Hwy111_to_ChannelDrive.csv",
        "Channel Drive → Miles Avenue": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/11_12_LONG_NSB_ChannelDrive_to_MilesAvenue.csv",
        "Miles Avenue → Via Sevilla": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/12_13_LONG_NSB_MilesAvenue_to_ViaSevilla.csv",
        "Via Sevilla → Fred Waring Drive": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/13_14_LONG_NSB_ViaSevilla_FredWaringDrive.csv",
        "Fred Waring Drive → Palm Royale Drive": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/14_15_LONG_NSB_FredWaringDrive_to_PalmRoyaleDrive.csv",
        "Palm Royale Drive → Avenue of the States": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/15_16_LONG_NSB_PalmRoyaleDrive_to_AvenueoftheStates.csv",
        "Avenue of the States → Avenue 42": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/16_17_LONG_NSB_AvenueoftheStates_to_Avenue42.csv",
        "Avenue 42 → Avenue 41": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/17_18_LONG_NSB_Avenue42_to_Avenue41.csv",
        "Avenue 41 → Country Club Drive": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/18_20_LONG_NB_Avenue41_to_Countryclubdrive.csv",

        # Southbound only segments
        "Harris Lane → Avenue 41 (SB)": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/19_18_LONG_SB_Harrislane_avenue41.csv",
        "Country Club Drive → Harris Lane (SB)": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/20_19_LONG_SB_CountryClubDrive_to_HarrisLane.csv",
    }

    usecols = [
        "local_datetime",
        "corridor_id",
        "direction",
        "average_delay",
        "average_traveltime",
        "average_speed",
    ]

    all_data = []
    for segment_name, url in data_sources.items():
        url = _fix_raw_url(url)
        try:
            df = pd.read_csv(
                url,
                usecols=lambda c: c in usecols,  # tolerate column mismatches
                parse_dates=["local_datetime"],
                dtype={
                    "corridor_id": "string",
                    "direction": "string",
                },
            )
            if df.empty:
                continue
            # Ensure needed numeric columns exist even if missing in source
            for c in ["average_delay", "average_traveltime", "average_speed"]:
                if c not in df.columns:
                    df[c] = np.nan
            # Assign segment name, reduce memory
            df["segment_name"] = segment_name
            for c in ("direction", "segment_name"):
                if c in df.columns:
                    df[c] = df[c].astype("category")
            for c in ("average_delay", "average_traveltime", "average_speed"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
            all_data.append(df)
        except Exception as e:
            st.error(f"Error loading {segment_name}: {e}")

    if not all_data:
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.dropna(subset=["local_datetime"]).sort_values("local_datetime").reset_index(drop=True)
    return combined_df

#Kinetic mobility data
@st.cache_data
def load_volume_data():
    """
    Load consolidated volume data for all Washington Street intersections.
    Auto-fixes bad RAW URL pattern.
    """
    volume_url = _fix_raw_url(
        "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/VOLUME/KMOB_LONG/LONG_MASTER_Avenue52_to_Avenue47_1hr_NS_VOLUME_OctoberTOJune.csv"
    )

    try:
        volume_df = pd.read_csv(volume_url)
        volume_df["local_datetime"] = pd.to_datetime(volume_df["local_datetime"], errors="coerce")
        volume_df = volume_df.dropna(subset=["local_datetime"]).sort_values("local_datetime").reset_index(drop=True)

        # Create proper intersection names from intersection_id
        volume_df["intersection_name"] = (
            volume_df["intersection_id"]
            .str.replace("_", " ", regex=False)
            .str.replace("Washington St and ", "Washington St & ", regex=False)
            .str.replace(" and ", " & ", regex=False)
        )

        # Create a sorting order for intersections (from south to north along Washington St)
        intersection_order = {
            "Washington St & Avenue52": 1,
            "Washington St & Calle Tampico": 2,
            "Washington St & Village Shop Ctr": 3,
            "Washington St & Avenue50": 4,
            "Washington St & Sagebrush Ave": 5,
            "Washington St & Eisenhower": 6,
            "Washington St & Ave48": 7,
            "Washington St & Ave47": 8,
        }

        volume_df["sort_order"] = volume_df["intersection_name"].map(intersection_order).fillna(999)
        volume_df = volume_df.sort_values("sort_order").drop("sort_order", axis=1)
        return volume_df

    except Exception as e:
        st.error(f"Error loading volume data: {e}")
        return pd.DataFrame()


# -------------------------
# Acyclica (Long + Wide)
# -------------------------
ACYCLICA_URL = _fix_raw_url(
    "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/MASTER_Acyclica_Traveltime_speed.csv"
)

@st.cache_data(show_spinner=False)
def load_acyclica_data() -> pd.DataFrame:
    """
    Load Acyclica travel time & speed in LONG format.
    Returns a cleaned DataFrame or raises RuntimeError on failure.
    Columns (normalized):
      local_datetime, corridor_id, direction, metric, Strength, Firsts, Lasts, Minimum, Maximum
    """
    try:
        df = pd.read_csv(ACYCLICA_URL)
    except Exception as e:
        # Raise so the caller decides where/how to display the error (main area, not sidebar)
        raise RuntimeError(f"Error loading Acyclica data: {e}")

    if df is None or df.empty:
        raise RuntimeError("Acyclica CSV is empty.")

    # Normalize headers
    df = _normalize_acyclica_headers(df)

    required = ["local_datetime","corridor_id","direction","metric","Strength","Firsts","Lasts","Minimum","Maximum"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Acyclica CSV missing required columns: {', '.join(missing)}")

    # Types & cleanup
    df["local_datetime"] = pd.to_datetime(df["local_datetime"], errors="coerce")
    df = df.dropna(subset=["local_datetime"])
    for c in ["Strength","Firsts","Lasts","Minimum","Maximum"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Metric and direction normalization (vectorized string ops)
    df["metric"] = (
        df["metric"].astype(str).str.strip().str.replace(" ", "", regex=False).str.title()
    )  # TravelTime / Speed
    df["direction"] = df["direction"].astype(str).str.strip().str.upper()

    df = df.sort_values(["local_datetime","direction","metric"]).reset_index(drop=True)
    return df



@st.cache_data(show_spinner=False)
def acyclica_long_to_hourly(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long → wide for KPI/plots.
    Output columns:
      local_datetime, corridor_id, direction, average_traveltime, average_speed, average_delay (NaN), segment_name
    """
    if df_long is None or df_long.empty:
        return pd.DataFrame()

    piv = (
        df_long.pivot_table(
            index=["local_datetime","corridor_id","direction"],
            columns="metric",
            values="Strength",
            aggfunc="mean",
        )
        .reset_index()
        .rename(columns={"TravelTime":"average_traveltime","Speed":"average_speed"})
    )

    # Ensure presence
    for col in ["average_traveltime","average_speed"]:
        if col not in piv.columns:
            piv[col] = np.nan

    piv["average_delay"] = np.nan  # Acyclica doesn't provide delay
    piv["segment_name"] = piv["corridor_id"].astype(str)

    piv["local_datetime"] = pd.to_datetime(piv["local_datetime"], errors="coerce")
    piv = piv.dropna(subset=["local_datetime"]).sort_values(["local_datetime","direction"]).reset_index(drop=True)
    return piv


# =========================
# Small data getters
# =========================
@st.cache_data(show_spinner=False)
def get_corridor_df() -> pd.DataFrame:
    df = load_traffic_data()
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = _safe_to_datetime(df.copy(), "local_datetime")
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


@st.cache_data(show_spinner=False)
def get_acyclica_long_df() -> pd.DataFrame:
    """
    Long-format Acyclica (for Incident/Peak/Event detection).
    """
    return load_acyclica_data()


@st.cache_data(show_spinner=False)
def get_acyclica_df() -> pd.DataFrame:
    """
    Wide-format Acyclica for KPI/plots (Iteris-like):
      local_datetime, corridor_id, direction, average_traveltime, average_speed, average_delay, segment_name
    """
    long_df = load_acyclica_data()
    if long_df is None or len(long_df) == 0:
        return pd.DataFrame()
    wide = acyclica_long_to_hourly(long_df)
    return _safe_to_datetime(wide.copy(), "local_datetime")


def get_performance_rating(score: float):
    """
    Map a 0..100 score to a label + CSS class used by the UI badges.
    """
    if score > 80:
        return " Excellent", "badge-excellent"
    if score > 60:
        return " Good", "badge-good"
    if score > 40:
        return " Fair", "badge-fair"
    if score > 20:
        return " Poor", "badge-poor"
    return " Critical", "badge-critical"


# =========================
# Interpretable KPI helpers (for Performance tab)
# =========================
def _coerce_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def compute_perf_kpis_interpretable(df: pd.DataFrame, high_delay_threshold: float) -> dict:
    """
    Compute five interpretable KPIs for Iteris-style (wide) data.
    """
    if df is None or df.empty:
        return {
            "avg_tt": {"value": 0.0, "unit": "min", "score": 50.0, "help": "Average Travel Time"},
            "planning_time": {"value": 0.0, "unit": "min", "score": 50.0, "help": "Planning Time (95th)"},
            "buffer_index": {"value": 0.0, "unit": "%", "score": 50.0, "help": "Buffer Index"},
            "reliability": {"value": 0.0, "unit": "%", "score": 50.0, "help": "Reliability Index"},
            "congestion_freq": {"value": 0.0, "unit": "%", "score": 50.0, "help": "Congestion Frequency"},
        }

    # Coerce numeric
    for c in ("average_delay", "average_traveltime", "average_speed"):
        if c in df:
            df[c] = _coerce_num(df[c])

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

    # Congestion Frequency (% of hours with delay > threshold)
    if "average_delay" in df and df["average_delay"].notna().any():
        total_hours = int(df["average_delay"].count())
        cong_hours = int((df["average_delay"] > high_delay_threshold).sum())
        cong_freq = (cong_hours / total_hours * 100.0) if total_hours > 0 else 0.0
    else:
        cong_freq, cong_hours, total_hours = 0.0, 0, 0

    # Normalized scores (0..100, higher = better)
    def _minmax_score(series: pd.Series, val: float) -> float:
        series = pd.to_numeric(series, errors="coerce").dropna()
        if len(series) < 2:
            return 50.0
        mn, mx = float(series.min()), float(series.max())
        if mx <= mn:
            return 50.0
        frac = (val - mn) / (mx - mn)  # lower is better
        return float(max(0.0, min(100.0, 100.0 * (1.0 - frac))))

    if "average_traveltime" in df and df["average_traveltime"].notna().any():
        score_avg_tt = _minmax_score(df["average_traveltime"], avg_tt)
        score_plan = _minmax_score(df["average_traveltime"], p95_tt)
    else:
        score_avg_tt = score_plan = 50.0

    score_buffer = float(max(0.0, 100.0 - min(max(buffer_index, 0.0), 100.0)))
    score_reliability = float(max(0.0, min(100.0, reliability)))
    score_congestion = float(max(0.0, min(100.0, 100.0 - cong_freq)))

    return {
        "avg_tt": {
            "value": avg_tt,
            "unit": "min",
            "score": score_avg_tt,
            "help": "Average Travel Time\n\nWhat it means: The typical door-to-door trip time for this route with your current filters.\nWhy it exists: Gives a quick sense of what most trips take.\nHow it’s calculated: Average of the hourly O-D trip times.\nFormula: mean(travel_time).",
        },
        "planning_time": {
            "value": p95_tt,
            "unit": "min",
            "score": score_plan,
            "help": "Planning Time (95th)\n\nWhat it means: 95th-percentile travel time in your filtered period.\nPurpose: captures a realistic worst-case for planning.",
        },
        "buffer_index": {
            "value": buffer_index,
            "unit": "%",
            "score": score_buffer,
            "help": "Buffer Index = (P95 − mean) / mean × 100.",
        },
        "reliability": {
            "value": reliability,
            "unit": "%",
            "score": score_reliability,
            "help": "Reliability Index = 100 − CV%, where CV% = stdev/mean × 100.",
        },
        "congestion_freq": {
            "value": cong_freq,
            "unit": "%",
            "score": score_congestion,
            "extra": f"Hours > {high_delay_threshold:.0f}s: {cong_hours}/{total_hours}",
            "help": "Share of hours with delay above your chosen threshold.",
        },
    }


def render_badge(score: float) -> str:
    """
    Turn a 0..100 'goodness' score into your visual badge HTML, using get_performance_rating.
    """
    label, css = get_performance_rating(score)
    return f'<span class="performance-badge {css}">{label}</span>'


# =========================
# Chart helpers
# =========================
def performance_chart(data: pd.DataFrame, metric_type: str = "delay"):
    if data.empty:
        return None
    metric_type = metric_type.lower().strip()
    if metric_type == "delay":
        y_col, title, color = "average_delay", "Traffic Delay Analysis", "#e74c3c"
        y_label = "Average Delay (seconds)"
        dist_x_label = "Average Delay (seconds)"
    else:
        y_col, title, color = "average_traveltime", "Travel Time Analysis", "#3498db"
        y_label = "Average Travel Time (minutes)"
        dist_x_label = "Average Travel Time (minutes)"

    dd = data.dropna(subset=["local_datetime", y_col]).sort_values("local_datetime")

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Time Series Analysis", "Distribution Analysis"),
        # Increase spacing between the top chart and the Distribution subtitle to prevent overlap
        vertical_spacing=0.22,
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


def volume_charts(
    data: pd.DataFrame,
    theoretical_link_capacity_vph: int,
    high_volume_threshold_vph: int,
):
    if data.empty:
        return None, None, None
    dd = data.dropna(subset=["local_datetime", "total_volume", "intersection_name"]).copy()
    dd.sort_values("local_datetime", inplace=True)

    # 1) Trend by intersection
    fig1 = px.line(
        dd,
        x="local_datetime",
        y="total_volume",
        color="intersection_name",
        title=" Traffic Volume Trends by Intersection",
        labels={"total_volume": "Volume (vehicles/hour)", "local_datetime": "Date/Time"},
        template="plotly_white",
    )
    fig1.update_layout(
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # 2) Distribution + Hourly heatmap
    fig2 = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Volume Distribution by Intersection", "Hourly Avg Volume Heatmap"),
        vertical_spacing=0.12,
    )

    # Box plots
    for name, g in dd.groupby("intersection_name", sort=False):
        fig2.add_trace(go.Box(y=g["total_volume"], name=name, boxpoints="outliers"), row=1, col=1)

    dd["hour"] = dd["local_datetime"].dt.hour
    hourly_avg = dd.groupby(["hour", "intersection_name"], as_index=False)["total_volume"].mean()
    hourly_pivot = hourly_avg.pivot(index="intersection_name", columns="hour", values="total_volume").sort_index()

    fig2.add_trace(
        go.Heatmap(
            z=hourly_pivot.values,
            x=hourly_pivot.columns,
            y=hourly_pivot.index,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Avg Volume (vph)"),
        ),
        row=2, col=1,
    )
    fig2.update_layout(
        height=800,
        title=" Volume Distribution & Capacity Analysis",
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    # 3) Peak hour by intersection
    hourly_volume = dd.groupby(["hour", "intersection_name"], as_index=False)["total_volume"].mean()
    fig3 = px.line(
        hourly_volume,
        x="hour",
        y="total_volume",
        color="intersection_name",
        title=" Average Hourly Volume Patterns",
        labels={"total_volume": "Average Volume (vph)", "hour": "Hour of Day"},
        template="plotly_white",
    )
    fig3.add_hline(
        y=theoretical_link_capacity_vph,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Theoretical Capacity ({theoretical_link_capacity_vph:,} vph)",
    )
    fig3.add_hline(
        y=high_volume_threshold_vph,
        line_dash="dot",
        line_color="orange",
        annotation_text=f"High Volume Threshold ({high_volume_threshold_vph:,} vph)",
    )
    fig3.update_layout(
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig1, fig2, fig3


# =========================
# Date range UI helper
# =========================
def date_range_preset_controls(min_date: datetime.date, max_date: datetime.date, key_prefix: str):
    """
    Presets that default to Last 30 Days on first load, persist in session_state,
    and won't clobber custom picks.
    """
    k_range = f"{key_prefix}_range"

    # Default to LAST 30 DAYS (bounded by min_date)
    if k_range not in st.session_state:
        default_start = max(min_date, max_date - timedelta(days=30))
        st.session_state[k_range] = (default_start, max_date)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button(" Last 7 Days", key=f"{key_prefix}_7d"):
            st.session_state[k_range] = (max(min_date, max_date - timedelta(days=7)), max_date)
    with c2:
        if st.button(" Last 30 Days", key=f"{key_prefix}_30d"):
            st.session_state[k_range] = (max(min_date, max_date - timedelta(days=30)), max_date)
    with c3:
        if st.button(" Full Range", key=f"{key_prefix}_full"):
            st.session_state[k_range] = (min_date, max_date)

    custom = st.date_input(
        "Custom Date Range",
        value=st.session_state[k_range],
        min_value=min_date,
        max_value=max_date,
        key=f"{key_prefix}_custom",
    )
    if custom != st.session_state[k_range]:
        st.session_state[k_range] = custom

    return st.session_state[k_range]


# =========================
# Processing
# =========================
def process_traffic_data(df, date_range, granularity, time_filter=None, start_hour=None, end_hour=None):
    """
    Process traffic data based on date range and granularity selections.
    Optimizations:
    - Early date/time filtering to minimize rows
    - Robust time_filter label handling (ASCII vs en dash)
    - If base data are sub-hourly (e.g., 5-min), aggregate to Hourly before applying hourly filters
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df = df.copy()

    # Ensure datetime
    df["local_datetime"] = pd.to_datetime(df.get("local_datetime", pd.NaT), errors="coerce")
    df = df.dropna(subset=["local_datetime"])  # safety

    # Early date range filter
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
        if start_date is not None and end_date is not None:
            mask = (df["local_datetime"].dt.date >= start_date) & (df["local_datetime"].dt.date <= end_date)
            df = df.loc[mask]
    if df.empty:
        return df

    # Normalize time filter label (handle en dash, extra spaces)
    tf = str(time_filter or "").replace("–", "-").strip()

    # If granular is Hourly but timestamps are sub-hourly, consolidate first
    def _hourly_group(gdf: pd.DataFrame, by_cols: list[str], metrics: list[str], how: str = "mean") -> pd.DataFrame:
        gdf = gdf.copy()
        gdf["hour"] = gdf["local_datetime"].dt.floor("H")
        agg = gdf.groupby(by_cols + ["hour"], observed=True)[metrics].agg(how).reset_index()
        agg = agg.rename(columns={"hour": "local_datetime"})
        return agg

    is_corridor = "segment_name" in df.columns
    is_volume = "intersection_id" in df.columns or "total_volume" in df.columns

    if is_corridor:
        # Ensure numeric dtypes
        for c in ("average_delay", "average_traveltime", "average_speed"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # Aggregate to requested granularity
        if granularity == "Hourly":
            # Consolidate sub-hourly to hourly if needed
            if (df["local_datetime"].dt.minute != 0).any():
                df = _hourly_group(df, ["corridor_id", "direction", "segment_name"], ["average_delay", "average_traveltime", "average_speed"], how="mean")
        elif granularity == "Daily":
            df["date_group"] = df["local_datetime"].dt.date
            df = df.groupby(["date_group", "corridor_id", "direction", "segment_name"], observed=True)[
                ["average_delay", "average_traveltime", "average_speed"]
            ].mean().reset_index().rename(columns={"date_group": "local_datetime"})
            df["local_datetime"] = pd.to_datetime(df["local_datetime"])
        elif granularity == "Weekly":
            df["week_group"] = df["local_datetime"].dt.to_period("W").dt.start_time
            df = df.groupby(["week_group", "corridor_id", "direction", "segment_name"], observed=True)[
                ["average_delay", "average_traveltime", "average_speed"]
            ].mean().reset_index().rename(columns={"week_group": "local_datetime"})
        elif granularity == "Monthly":
            df["month_group"] = df["local_datetime"].dt.to_period("M").dt.start_time
            df = df.groupby(["month_group", "corridor_id", "direction", "segment_name"], observed=True)[
                ["average_delay", "average_traveltime", "average_speed"]
            ].mean().reset_index().rename(columns={"month_group": "local_datetime"})

        # Apply time-of-day filters only when Hourly
        if granularity == "Hourly" and tf:
            hrs = df["local_datetime"].dt.hour
            if tf == "Peak Hours (7-9 AM, 4-6 PM)":
                df = df[hrs.between(7, 9) | hrs.between(16, 18)]
            elif tf == "AM Peak (7-9 AM)":
                df = df[hrs.between(7, 9)]
            elif tf == "PM Peak (4-6 PM)":
                df = df[hrs.between(16, 18)]
            elif tf == "Off-Peak":
                df = df[~hrs.between(7, 9) & ~hrs.between(16, 18)]
            elif tf == "Custom Range" and start_hour is not None and end_hour is not None:
                df = df[hrs.between(int(start_hour), int(end_hour) - 1)]

        return df.sort_values("local_datetime").reset_index(drop=True)

    if is_volume:
        # Volume aggregation
        if "total_volume" in df.columns:
            df["total_volume"] = pd.to_numeric(df["total_volume"], errors="coerce").fillna(0)
        if granularity == "Hourly":
            if (df["local_datetime"].dt.minute != 0).any():
                # sum volumes to the hour
                df = _hourly_group(df, ["intersection_id", "direction", "intersection_name" if "intersection_name" in df.columns else "intersection_id"], ["total_volume"], how="sum")
        elif granularity == "Daily":
            df["date_group"] = df["local_datetime"].dt.date
            df = df.groupby(["date_group", "intersection_id", "direction", "intersection_name"], observed=True)["total_volume"].sum().reset_index().rename(columns={"date_group": "local_datetime"})
            df["local_datetime"] = pd.to_datetime(df["local_datetime"])
        elif granularity == "Weekly":
            df["week_group"] = df["local_datetime"].dt.to_period("W").dt.start_time
            df = df.groupby(["week_group", "intersection_id", "direction", "intersection_name"], observed=True)["total_volume"].sum().reset_index().rename(columns={"week_group": "local_datetime"})
        elif granularity == "Monthly":
            df["month_group"] = df["local_datetime"].dt.to_period("M").dt.start_time
            df = df.groupby(["month_group", "intersection_id", "direction", "intersection_name"], observed=True)["total_volume"].sum().reset_index().rename(columns={"month_group": "local_datetime"})

        # Hourly time filter for volume
        if granularity == "Hourly" and tf:
            hrs = df["local_datetime"].dt.hour
            if tf == "Peak Hours (7-9 AM, 4-6 PM)":
                df = df[hrs.between(7, 9) | hrs.between(16, 18)]
            elif tf == "AM Peak (7-9 AM)":
                df = df[hrs.between(7, 9)]
            elif tf == "PM Peak (4-6 PM)":
                df = df[hrs.between(16, 18)]
            elif tf == "Off-Peak":
                df = df[~hrs.between(7, 9) & ~hrs.between(16, 18)]
            elif tf == "Custom Range" and start_hour is not None and end_hour is not None:
                df = df[hrs.between(int(start_hour), int(end_hour) - 1)]

        return df.sort_values("local_datetime").reset_index(drop=True)

    # Fallback - return filtered df
    return df.sort_values("local_datetime").reset_index(drop=True)

#Function to help download 5 minute CSV

def create_5min_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert filtered data to 5-minute intervals by interpolating between existing data points.
    This creates more granular data for detailed analysis without requiring actual 5-minute source data.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Make a copy and ensure datetime column exists
    result_df = df.copy()
    if "local_datetime" not in result_df.columns:
        return result_df

    # Ensure datetime is properly formatted
    result_df["local_datetime"] = pd.to_datetime(result_df["local_datetime"], errors="coerce")
    result_df = result_df.dropna(subset=["local_datetime"]).sort_values("local_datetime")

    if len(result_df) < 2:
        return result_df

    # Create 5-minute intervals between min and max dates
    start_time = result_df["local_datetime"].min().floor("5T")
    end_time = result_df["local_datetime"].max().ceil("5T")

    # Generate 5-minute time range
    time_range = pd.date_range(start=start_time, end=end_time, freq="5T")

    # For each unique combination of non-datetime columns, interpolate
    id_cols = [col for col in result_df.columns if col not in ["local_datetime"]
               and not pd.api.types.is_numeric_dtype(result_df[col])]
    numeric_cols = [col for col in result_df.columns
                    if col != "local_datetime" and pd.api.types.is_numeric_dtype(result_df[col])]

    if not id_cols:  # If no grouping columns, treat as single series
        # Simple case - just one series to interpolate
        temp_df = pd.DataFrame({"local_datetime": time_range})
        merged = pd.merge(temp_df, result_df, on="local_datetime", how="left")

        # Set datetime as index for time-based interpolation
        merged_indexed = merged.set_index("local_datetime")

        # Interpolate numeric columns using linear method (safer than time method)
        for col in numeric_cols:
            if col in merged_indexed.columns:
                merged_indexed[col] = merged_indexed[col].interpolate(method="linear")

        # Reset index to get datetime back as column
        merged = merged_indexed.reset_index()
        return merged.dropna()

    # Complex case - multiple groups to interpolate
    all_interpolated = []

    for group_vals, group_df in result_df.groupby(id_cols):
        if len(group_df) < 2:
            continue

        # Create time range for this group
        group_start = group_df["local_datetime"].min().floor("5T")
        group_end = group_df["local_datetime"].max().ceil("5T")
        group_time_range = pd.date_range(start=group_start, end=group_end, freq="5T")

        # Create base dataframe with 5-minute intervals
        temp_df = pd.DataFrame({"local_datetime": group_time_range})

        # Add the group identifier columns
        if isinstance(group_vals, tuple):
            for i, col in enumerate(id_cols):
                temp_df[col] = group_vals[i]
        else:
            temp_df[id_cols[0]] = group_vals

        # Merge with existing data
        merged = pd.merge(temp_df, group_df, on=["local_datetime"] + id_cols, how="left")

        # Set datetime as index for interpolation
        merged_indexed = merged.set_index("local_datetime")

        # Interpolate numeric columns using linear method
        for col in numeric_cols:
            if col in merged_indexed.columns:
                merged_indexed[col] = merged_indexed[col].interpolate(method="linear")

        # Reset index to get datetime back as column
        merged = merged_indexed.reset_index()

        # Only keep rows that have interpolated data
        merged = merged.dropna(subset=numeric_cols, how="all")
        all_interpolated.append(merged)

    if all_interpolated:
        return pd.concat(all_interpolated, ignore_index=True).sort_values("local_datetime")
    else:
        return pd.DataFrame()
