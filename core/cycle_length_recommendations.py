# cycle_length_recommendations.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# Time-period filter (AM/MD/PM/ALL)
# -------------------------
@st.cache_data
def filter_by_period(df: pd.DataFrame, time_col: str, period: str) -> pd.DataFrame:
    """Filter dataframe by time period (AM 05–10, MD 11–15, PM 16–20, ALL)."""
    if time_col not in df.columns or df.empty:
        return df
    df_copy = df.copy()
    df_copy[time_col] = pd.to_datetime(df_copy[time_col], errors="coerce")

    if period == "AM":
        return df_copy[(df_copy[time_col].dt.hour >= 5) & (df_copy[time_col].dt.hour <= 10)]
    if period == "MD":
        return df_copy[(df_copy[time_col].dt.hour >= 11) & (df_copy[time_col].dt.hour <= 15)]
    if period == "PM":
        return df_copy[(df_copy[time_col].dt.hour >= 16) & (df_copy[time_col].dt.hour <= 20)]
    return df_copy


# -------------------------
# Cycle length thresholds
# -------------------------
@st.cache_data
def get_hourly_cycle_length(volume):
    """
    Return recommended cycle length:
    ≥2400 → 140 sec, ≥1500 → 130 sec, ≥600 → 120 sec, ≥300 → 110 sec, else Free mode
    """
    if pd.isna(volume) or volume <= 0:
        return "Free mode"
    if volume >= 2400:
        return "140 sec"
    if volume >= 1500:
        return "130 sec"
    if volume >= 600:
        return "120 sec"
    if volume >= 300:
        return "110 sec"
    return "Free mode"


def _get_status(recommended: str, current: str) -> str:
    """Compare recommended cycle vs current and return status label."""
    if recommended == current:
        return "🟢 OPTIMAL"
    if recommended == "Free mode" and current != "Free mode":
        return "🔽 REDUCE"
    if recommended != "Free mode" and current == "Free mode":
        return "⬆️ INCREASE"
    rec_val = int(recommended.split()[0]) if recommended != "Free mode" else 0
    cur_val = int(current.split()[0]) if current != "Free mode" else 0
    if rec_val > cur_val:
        return "⬆️ INCREASE"
    if rec_val < cur_val:
        return "🔽 REDUCE"
    return "🟢 OPTIMAL"


# -------------------------
# Visual helpers (legend + colors) — theme-able & colorblind-safe
# -------------------------
CYCLE_ORDER = ["Free mode", "110 sec", "120 sec", "130 sec", "140 sec"]
THRESHOLD_TEXT = {
    "140 sec": "≥ 2400 vph",
    "130 sec": "≥ 1500 vph",
    "120 sec": "≥ 600 vph",
    "110 sec": "≥ 300 vph",
    "Free mode": "< 300 vph",
}

def _get_palettes(theme: str):
    """
    Returns (cycle_colors, status_colors, pattern_map) for the selected theme.
    Default is Okabe–Ito colorblind-safe palette.
    """
    if theme == "High Contrast":
        cycle_colors = {
            "Free mode": "#808080",
            "110 sec": "#1B9E77",
            "120 sec": "#386CB0",
            "130 sec": "#FDC827",
            "140 sec": "#D62728",
        }
        status_colors = {"🟢 OPTIMAL": "#1B9E77", "⬆️ INCREASE": "#D62728", "🔽 REDUCE": "#386CB0"}
    elif theme == "Greens → Red":
        cycle_colors = {
            "Free mode": "#9E9E9E",
            "110 sec": "#2ECC71",
            "120 sec": "#27AE60",
            "130 sec": "#E67E22",
            "140 sec": "#E74C3C",
        }
        status_colors = {"🟢 OPTIMAL": "#27AE60", "⬆️ INCREASE": "#E74C3C", "🔽 REDUCE": "#2E86C1"}
    elif theme == "Monochrome + Accents":
        cycle_colors = {
            "Free mode": "#95A5A6",
            "110 sec": "#34495E",
            "120 sec": "#2C3E50",
            "130 sec": "#8E44AD",
            "140 sec": "#E74C3C",
        }
        status_colors = {"🟢 OPTIMAL": "#2ECC71", "⬆️ INCREASE": "#E74C3C", "🔽 REDUCE": "#8E44AD"}
    else:  # "Colorblind Safe" (Okabe–Ito)
        cycle_colors = {
            "Free mode": "#8C8C8C",   # gray
            "110 sec": "#009E73",     # bluish green
            "120 sec": "#0072B2",     # blue
            "130 sec": "#E69F00",     # orange
            "140 sec": "#D55E00",     # vermillion
        }
        status_colors = {"🟢 OPTIMAL": "#009E73", "⬆️ INCREASE": "#D55E00", "🔽 REDUCE": "#0072B2"}

    pattern_map = {
        "Free mode": "",
        "110 sec": "/",
        "120 sec": "\\",
        "130 sec": "x",
        "140 sec": ".",
    }
    return cycle_colors, status_colors, pattern_map


def _inject_kpi_css():
    """Theme-aware CSS for legend and KPI cards (robust dark-mode support)."""
    st.markdown(
        """
<style>
/* ---------- Light defaults ---------- */
:root{
  /* Legend */
  --legend-bg: rgba(15,47,82,.06);
  --legend-border: rgba(79,172,254,.28);
  --legend-title: #0f2f52;

  /* KPI tiles */
  --kpi-bg: linear-gradient(135deg, rgba(79,172,254,.06), rgba(0,242,254,.04));
  --kpi-border: rgba(79,172,254,.28);
  --kpi-text: #0f2f52;
  --kpi-title: #0f2f52;   /* explicit */
  --kpi-muted: rgba(15,47,82,.78);
  --kpi-shadow: 0 8px 20px rgba(79,172,254,.10);
  --kpi-good: #2ecc71;
  --kpi-warn: #f39c12;
  --kpi-bad: #e74c3c;
  --kpi-pill: rgba(255,255,255,.65);
}

/* ---------- Dark mode overrides (common attributes) ---------- */
html.dark, [data-theme="dark"], [data-base-theme="dark"], body[data-theme="dark"]{
  --legend-bg: rgba(255,255,255,.08);
  --legend-border: rgba(255,255,255,.18);
  --legend-title: #ffffff;

  --kpi-bg: rgba(255,255,255,.07);
  --kpi-border: rgba(255,255,255,.22);
  --kpi-text: #ffffff;
  --kpi-title: #ffffff;
  --kpi-muted: rgba(255,255,255,.82);
  --kpi-shadow: 0 10px 26px rgba(0,0,0,.35);
  --kpi-pill: rgba(255,255,255,.10);
}

/* Fallback for environments that only expose prefers-color-scheme */
@media (prefers-color-scheme: dark){
  :root{
    --legend-bg: rgba(255,255,255,.08);
    --legend-border: rgba(255,255,255,.18);
    --legend-title: #ffffff;

    --kpi-bg: rgba(255,255,255,.07);
    --kpi-border: rgba(255,255,255,.22);
    --kpi-text: #ffffff;
    --kpi-title: #ffffff;
    --kpi-muted: rgba(255,255,255,.82);
    --kpi-pill: rgba(255,255,255,.10);
  }
}

/* ---------- Legend block ---------- */
.cvag-legend{
  border:1px solid var(--legend-border);
  background: var(--legend-bg);
  border-radius:12px;
  padding:.6rem 1rem;
  box-shadow:0 8px 24px rgba(0,0,0,.10);
  margin-top:.25rem;
}
.cvag-legend-title{
  font-weight:800;
  color:var(--legend-title) !important;
  margin-bottom:.35rem;
}

/* ---------- KPI grid/cards ---------- */
.cvag-kpi-grid{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; margin:4px 0 10px; }
@media (max-width:1500px){ .cvag-kpi-grid{ grid-template-columns:repeat(3,1fr);} }
@media (max-width:900px){ .cvag-kpi-grid{ grid-template-columns:repeat(2,1fr);} }
@media (max-width:600px){ .cvag-kpi-grid{ grid-template-columns:1fr;} }

.cvag-kpi-card{
  border-radius:16px; padding:14px 16px;
  background:var(--kpi-bg);
  border:1px solid var(--kpi-border);
  box-shadow:var(--kpi-shadow);
  color:var(--kpi-text);
}
.cvag-kpi-title{
  font-weight:800; font-size:.95rem; letter-spacing:.2px;
  color: var(--kpi-title) !important;
}
.cvag-kpi-value{
  font-size:2.0rem; line-height:1.05; font-weight:800; margin-top:.25rem; letter-spacing:.3px;
  color: var(--kpi-title) !important;
}
.cvag-kpi-delta{ margin-top:.15rem; font-size:.95rem; font-weight:700; color:var(--kpi-muted); }
.cvag-kpi-delta.good{ color:#2ECC71; } .cvag-kpi-delta.warn{ color:#F39C12; } .cvag-kpi-delta.bad{ color:#E74C3C; }
.cvag-kpi-foot{ margin-top:.35rem; font-size:.85rem; color:var(--kpi-muted); }
.cvag-pill{ display:inline-flex; align-items:center; gap:.4rem; padding:.25rem .55rem; border-radius:999px; background:var(--kpi-pill); font-weight:700; font-size:.82rem; }
</style>
        """,
        unsafe_allow_html=True,
    )


def _legend_html(cycle_colors: dict) -> str:
    """HTML legend for cycle length thresholds, generated from active palette."""
    pill_items = []
    for label in ["140 sec", "130 sec", "120 sec", "110 sec", "Free mode"]:
        color = cycle_colors.get(label, "#9A9A9A")
        text = THRESHOLD_TEXT[label]
        pill_items.append(
            f'<span style="display:inline-flex;align-items:center;margin:.25rem .5rem;'
            f'padding:.3rem .6rem;border-radius:999px;background:{color};color:#fff;'
            f'font-weight:800;font-size:.85rem;">{label}</span>'
            f'<span style="margin-right:1rem;opacity:.85;font-size:.9rem">{text}</span>'
        )
    return (
        '<div class="cvag-legend">'
        '<div class="cvag-legend-title">Cycle Length Thresholds</div>'
        + "".join(pill_items) +
        '</div>'
    )


def _sec_value(label: str) -> int:
    """Map label to numeric seconds for sorting/plotting."""
    return int(label.split()[0]) if label != "Free mode" else 0


# -------------------------
# KPI-card HTML
# -------------------------
def _kpi_card(title: str, value_html: str, delta_text: str, tone: str = "neutral",
              foot1: str | None = None, foot2: str | None = None) -> str:
    tone = tone if tone in {"good", "warn", "bad", "neutral"} else "neutral"
    foot1_html = f'<div class="cvag-kpi-foot">{foot1}</div>' if foot1 else ""
    foot2_html = f'<div class="cvag-kpi-foot">{foot2}</div>' if foot2 else ""
    tone_class = f" {tone}" if tone != "neutral" else " neutral"
    return f"""
    <div class="cvag-kpi-card">
      <div class="cvag-kpi-title">{title}</div>
      <div class="cvag-kpi-value">{value_html}</div>
      <div class="cvag-kpi-delta{tone_class}">{delta_text}</div>
      {foot1_html}{foot2_html}
    </div>
    """


# -------------------------
# Main renderer
# -------------------------
def render_cycle_length_section(raw: pd.DataFrame, key_prefix: str = "cycle") -> None:
    """Render the Cycle Length Recommendations section with theme-aware styles."""
    if raw is None or raw.empty:
        st.info("No hourly volume data available for cycle length recommendations.")
        return
    if "local_datetime" not in raw.columns or "total_volume" not in raw.columns:
        st.info("Required columns not found: 'local_datetime', 'total_volume'.")
        return

    # ✅ Make styles available BEFORE any HTML so colors work immediately
    _inject_kpi_css()

    # ---- Context values for header ----
    raw = raw.copy()
    raw["local_datetime"] = pd.to_datetime(raw["local_datetime"], errors="coerce")

    start_dt = raw["local_datetime"].min()
    end_dt = raw["local_datetime"].max()
    start_label = start_dt.strftime("%A, %b %d, %Y") if pd.notnull(start_dt) else "N/A"
    end_label = end_dt.strftime("%A, %b %d, %Y") if pd.notnull(end_dt) else "N/A"

    intersections = sorted(raw["intersection_name"].dropna().unique().tolist()) if "intersection_name" in raw else []
    if len(intersections) == 1:
        intersection_label = intersections[0]
    elif len(intersections) > 1:
        intersection_label = f"{len(intersections)} Intersections"
    else:
        intersection_label = "N/A"

    directions = sorted(raw["direction"].dropna().unique().tolist()) if "direction" in raw else []
    if len(directions) == 1:
        direction_label = directions[0]
    elif len(directions) > 1:
        direction_label = "All Directions"
    else:
        direction_label = "N/A"

    # ---- Header ----
    header_html = (
        '<div style="background: linear-gradient(135deg, #2b77e5 0%, #19c3e6 100%); border-radius: 16px; '
        'padding: 22px 24px 20px; color: #fff; box-shadow: 0 10px 26px rgba(25,115,210,.25); '
        'margin: 8px 0 16px; text-align: left;">'
        '<div style="display:flex; align-items:center; gap:12px;">'
        '<div style="width:40px; height:40px; border-radius:10px; background: rgba(255,255,255,.18); '
        'display:flex; align-items:center; justify-content:center; box-shadow: inset 0 0 0 1px rgba(255,255,255,.15);">'
        '<span style="font-size:20px;">🔁</span></div>'
        '<div style="font-size:2.1rem; font-weight:800; letter-spacing:.2px; line-height:1.1;">'
        'Cycle Length Recommendations for CVAG</div></div>'
        '<div style="display:flex; flex-wrap:wrap; gap:8px 10px; margin:12px 0 6px;">'
        '<span style="display:inline-flex; align-items:center; gap:6px; padding:6px 10px; border-radius:999px; '
        'background: rgba(255,255,255,.16); font-weight:700; font-size:.95rem; '
        'box-shadow: inset 0 0 0 1px rgba(255,255,255,.18);"><span style="opacity:.9;">Intersection:</span>'
        f'<span style="opacity:1;">{intersection_label}</span></span>'
        '<span style="display:inline-flex; align-items:center; gap:6px; padding:6px 10px; border-radius:999px; '
        'background: rgba(255,255,255,.16); font-weight:700; font-size:.95rem; '
        'box-shadow: inset 0 0 0 1px rgba(255,255,255,.18);"><span style="opacity:.9;">Direction:</span>'
        f'<span style="opacity:1;">{direction_label}</span></span>'
        '<span style="display:inline-flex; align-items:center; gap:6px; padding:6px 10px; border-radius:999px; '
        'background: rgba(255,255,255,.16); font-weight:700; font-size:.95rem; '
        'box-shadow: inset 0 0 0 1px rgba(255,255,255,.18);"><span style="opacity:.9;">Study Type:</span>'
        '<span style="opacity:1;">Hourly Analysis</span></span></div>'
        '<div style="display:flex; align-items:center; gap:8px; margin-top:2px;">'
        '<span style="width:24px; height:24px; border-radius:8px; background: rgba(255,255,255,.18); '
        'display:inline-flex; align-items:center; justify-content:center; font-size:13px; '
        'box-shadow: inset 0 0 0 1px rgba(255,255,255,.16);">📅</span>'
        f'<span style="font-size:1.05rem; font-weight:600; opacity:.95;">{start_label} — {end_label}</span>'
        '</div></div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

    # Controls
    c1, c2, c3 = st.columns([2, 1.6, 1.5])
    with c1:
        time_period = st.selectbox(
            "🕐 Time Period",
            ["AM (05:00-10:00)", "MD (11:00-15:00)", "PM (16:00-20:00)", "All Day"],
            index=0,
            help="Analyze AM, Midday, PM, or All Day periods",
            key=f"{key_prefix}_period",
        )
    with c2:
        current_cycle = st.selectbox(
            "⚙️ Current System Cycle",
            CYCLE_ORDER[::-1],  # 140, 130, 120, 110, Free
            index=0,
            help="Cycle used currently; compared against recommendations",
            key=f"{key_prefix}_current",
        )
    with c3:
        theme_choice = st.selectbox(
            "🎨 Color Theme",
            ["Colorblind Safe", "High Contrast", "Greens → Red", "Monochrome + Accents"],
            index=0,
            help="Pick a palette that's easy to read for presentations and printouts",
            key=f"{key_prefix}_theme",
        )

    # Resolve palettes & patterns from theme
    CYCLE_COLORS, STATUS_COLORS, PATTERN_MAP = _get_palettes(theme_choice)

    # Legend (uses theme variables, not hard-coded colors)
    st.markdown(_legend_html(CYCLE_COLORS), unsafe_allow_html=True)

    # Time period filtering
    period_map = {"AM (05:00-10:00)": "AM", "MD (11:00-15:00)": "MD", "PM (16:00-20:00)": "PM", "All Day": "ALL"}
    selected_period = period_map[time_period]
    period_data = raw if selected_period == "ALL" else filter_by_period(raw, "local_datetime", selected_period)
    if period_data.empty:
        st.warning("⚠️ No data available for the selected time period.")
        return

    # Hour window label for KPIs
    period_windows = {"AM": "05:00–10:00", "MD": "11:00–15:00", "PM": "16:00–20:00", "ALL": "00:00–23:00"}
    hours_window_str = period_windows.get(selected_period, "—")

    # Hourly aggregation
    period_data["hour"] = period_data["local_datetime"].dt.hour
    hourly = period_data.groupby("hour", as_index=False)["total_volume"].mean()
    hourly["Volume"] = hourly["total_volume"].fillna(0).round().astype(int)

    # Recommendations + Status
    hourly["CVAG Recommendation"] = hourly["Volume"].apply(get_hourly_cycle_length)
    hourly["Status"] = hourly["CVAG Recommendation"].apply(lambda rec: _get_status(rec, current_cycle))
    hourly["Hour"] = hourly["hour"].apply(lambda x: f"{x:02d}:00")
    hourly["Rec (sec)"] = hourly["CVAG Recommendation"].apply(_sec_value)

    # --- KPI calculations ---
    total_hours = len(hourly)
    optimal_hours = int((hourly["Status"] == "🟢 OPTIMAL").sum())
    changes_needed = total_hours - optimal_hours

    # Lists of hours needing changes (for display)
    inc_hours_list = hourly.loc[hourly["Status"] == "⬆️ INCREASE", "Hour"].tolist()
    red_hours_list = hourly.loc[hourly["Status"] == "🔽 REDUCE", "Hour"].tolist()

    # High-volume threshold KPI (based on raw rows in selected period)
    HIGH_VOLUME_THRESHOLD_VPH = 1200
    period_data["total_volume"] = pd.to_numeric(period_data["total_volume"], errors="coerce")
    total_rows = int(period_data["total_volume"].count())
    high_rows = period_data.loc[period_data["total_volume"] > HIGH_VOLUME_THRESHOLD_VPH]
    high_hours = int(len(high_rows)) if total_rows > 0 else 0
    high_share = (high_hours / total_rows * 100) if total_rows > 0 else 0.0

    # Unique hour-of-day labels that exceeded threshold
    exceed_hour_ids = sorted(high_rows["local_datetime"].dt.hour.unique().tolist()) if len(high_rows) else []
    exceed_hour_labels = [f"{h:02d}:00" for h in exceed_hour_ids]

    # Peak capacity utilization
    INTERSECTION_CAPACITY_VPH = 1800
    peak_volume_pd = float(period_data["total_volume"].max()) if total_rows > 0 else 0.0
    peak_capacity_util = (peak_volume_pd / INTERSECTION_CAPACITY_VPH * 100) if INTERSECTION_CAPACITY_VPH else 0.0

    # Helper to summarize lists
    def _hours_preview(lst, max_items=8):
        if not lst:
            return "None"
        tail = "" if len(lst) <= max_items else f" (+{len(lst)-max_items} more)"
        return ", ".join(lst[:max_items]) + tail

    # -------------------------
    # BOXED KPIs (card grid)
    # -------------------------
    system_eff = (optimal_hours / total_hours * 100) if total_hours else 0
    tone_eff = "good" if system_eff >= 80 else ("warn" if system_eff >= 60 else "bad")
    tone_changes = "good" if changes_needed == 0 else ("warn" if changes_needed <= (total_hours * 0.4) else "bad")
    tone_high = "bad" if high_share > 25 else ("warn" if high_share > 10 else "good")
    tone_util = "bad" if peak_capacity_util > 90 else ("warn" if peak_capacity_util > 75 else "good")

    cards_html = f"""
    <div class="cvag-kpi-grid">
      {_kpi_card("📅 Hours Analyzed", f"{total_hours}", hours_window_str, "neutral")}
      {_kpi_card("✅ Optimal Hours", f"{optimal_hours}", f"{system_eff:.0f}% efficiency", tone_eff)}
      {_kpi_card("🔧 Changes Needed", f"{changes_needed}", f"↑ {len(inc_hours_list)} • ↓ {len(red_hours_list)}", tone_changes)}
      {_kpi_card("⚠️ Hours Above High-Volume Threshold", f"{high_hours}", f"{high_share:.1f}% of time",
                 tone_high, foot1=f"Threshold: > {HIGH_VOLUME_THRESHOLD_VPH:,} vph",
                 foot2=f"Hours: {_hours_preview(exceed_hour_labels)}")}
      {_kpi_card("🚦 Peak Capacity Utilization", f"{peak_capacity_util:.0f}%", f"Peak {int(peak_volume_pd):,} vph",
                 tone_util, foot1=f"Capacity: {INTERSECTION_CAPACITY_VPH:,} vph")}
    </div>
    """
    st.markdown(cards_html, unsafe_allow_html=True)

    # -------------------------
    # Charts
    # -------------------------
    ch1, ch2 = st.columns([2.2, 1.8])
    with ch1:
        # Bar chart colored by recommended cycle (palette + patterns + outlines)
        fig = px.bar(
            hourly.sort_values("hour"),
            x="Hour",
            y="Volume",
            color="CVAG Recommendation",
            color_discrete_map=CYCLE_COLORS,
            category_orders={"CVAG Recommendation": CYCLE_ORDER, "Hour": [f"{h:02d}:00" for h in range(24)]},
            title="Hourly Volume with Recommended Cycle Length",
            labels={"Volume": "Avg Volume (vph)", "Hour": "Hour of Day"},
            template="simple_white",
        )
        for tr in fig.data:
            tr.update(marker_line_color="rgba(0,0,0,0.30)", marker_line_width=0.7)
            # Optional hatching for extra accessibility
            if tr.name in PATTERN_MAP:
                tr.update(marker_pattern=dict(shape=PATTERN_MAP[tr.name], size=4, solidity=0.25, fillmode="overlay"))

        status_symbols = {"🟢 OPTIMAL": "circle", "⬆️ INCREASE": "triangle-up", "🔽 REDUCE": "triangle-down"}
        fig.add_trace(
            go.Scatter(
                x=hourly["Hour"],
                y=hourly["Volume"],
                mode="markers",
                marker=dict(
                    size=11,
                    color=[_ for _ in (STATUS_COLORS[s] for s in hourly["Status"])],
                    symbol=[status_symbols[s] for s in hourly["Status"]],
                    line=dict(width=1, color="white"),
                ),
                name="Status",
                hovertemplate="Hour=%{x}<br>Volume=%{y:.0f}<extra></extra>",
            )
        )
        fig.update_layout(
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=50, b=10),
            bargap=0.15,
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="rgba(0,0,0,0.08)")
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        status_counts = hourly["Status"].value_counts().reindex(["🟢 OPTIMAL", "⬆️ INCREASE", "🔽 REDUCE"], fill_value=0)
        pie = px.pie(
            names=status_counts.index,
            values=status_counts.values,
            title="Hours by Status",
            color=status_counts.index,
            color_discrete_map={
                "🟢 OPTIMAL": STATUS_COLORS["🟢 OPTIMAL"],
                "⬆️ INCREASE": STATUS_COLORS["⬆️ INCREASE"],
                "🔽 REDUCE": STATUS_COLORS["🔽 REDUCE"],
            },
            hole=0.35,
            template="simple_white",
        )
        pie.update_traces(textposition="inside", textinfo="label+percent")
        pie.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(pie, use_container_width=True)

    # Stylized table
    hourly_display = hourly[["Hour", "Volume", "CVAG Recommendation", "Status"]].rename(
        columns={"Volume": "Avg Volume (vph)"}
    )
    st.dataframe(
        hourly_display,
        use_container_width=True,
        column_config={
            "Hour": st.column_config.TextColumn("Hour", width="small"),
            "Avg Volume (vph)": st.column_config.NumberColumn(
                "Total Vehicle Volume (Throughs, lefts, and rights)", format="%d"
            ),
            "CVAG Recommendation": st.column_config.TextColumn("Cycle Length Recommendation For CVAG", width="medium"),
            "Status": st.column_config.TextColumn("Cycle Length Status", width="medium"),
        },
    )

    # Insights + download
    if len(hourly):
        peak_volume = int(hourly["Volume"].max())
        peak_hour = hourly.loc[hourly["Volume"].idxmax(), "Hour"]
    else:
        peak_volume, peak_hour = 0, "—"

    st.markdown(
        f"""
        <div class="insight-box" style="margin-top:.5rem;">
            <h4>💡 Cycle Length Optimization Insights</h4>
            <p><strong>📊 System Efficiency:</strong> {optimal_hours}/{total_hours} hours optimal ({(optimal_hours/total_hours*100 if total_hours else 0):.0f}%)</p>
            <p><strong>📈 Volume Profile:</strong> Peak {peak_volume:,} vph at {peak_hour} • Threshold exceedance: {high_hours} hours ({high_share:.1f}% of time)</p>
            <p><strong>🔧 Actions:</strong> ↑ {int((hourly["Status"] == "⬆️ INCREASE").sum())} hours need longer cycles • ↓ {int((hourly["Status"] == "🔽 REDUCE").sum())} hours need shorter cycles</p>
            <p><strong>🚦 Capacity:</strong> Peak utilization {peak_capacity_util:.0f}% of intersection capacity ({INTERSECTION_CAPACITY_VPH:,} vph)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.download_button(
        "⬇️ Download Cycle Length Analysis (CSV)",
        data=hourly_display.to_csv(index=False).encode("utf-8"),
        file_name=f"cycle_length_recommendations_{selected_period.lower()}.csv",
        mime="text/csv",
        key=f"{key_prefix}_download",
    )
