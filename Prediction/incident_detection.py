# Prediction/incident_detection.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Optional UI helpers (use if available)
try:
    from Prediction.timeline_scrubber import (
        render_gradient_header,
        create_timeline_scrubber,
        create_hourly_forecast_table,
    )
except Exception:
    render_gradient_header = None
    create_timeline_scrubber = None
    create_hourly_forecast_table = None


# =========================
# ------ Data layer -------
# =========================

def _fmt_hour_label(dt: pd.Timestamp) -> str:
    # 24h -> “1PM”, “12AM”, etc.
    h = int(dt.hour)
    if h == 0:
        return "12AM"
    if h == 12:
        return "12PM"
    return f"{h-12}PM" if h > 12 else f"{h}AM"

def _color_for_error(pct):
    if pct is None or np.isnan(pct):
        return "#95a5a6"  # gray
    if pct <= 5:
        return "#27ae60"  # green
    if pct <= 15:
        return "#f39c12"  # amber
    return "#e74c3c"      # red

def _confidence_from_signals(tt_spike_x: float, ramp_x: float) -> int:
    """
    Transparent confidence: 0..100
    - tt_spike_x: Maximum / Strength
    - ramp_x: Lasts / Firsts  (TT); for Speed you’d invert—this module targets TT.
    """
    spike_term = max(0.0, tt_spike_x - 1.0)    # 0 when no spike
    ramp_term  = max(0.0, ramp_x - 1.0)        # 0 when no ramp
    score = 60 * spike_term + 40 * ramp_term   # weighted
    return int(max(0, min(100, round(score))))

def _synth_incident_df(start_date: datetime, hours=13) -> pd.DataFrame:
    """Synthetic day: columns = [ts, predicted, actual, status, label]."""
    base = pd.to_datetime(start_date).replace(hour=12, minute=0, second=0, microsecond=0)
    idx = pd.date_range(base, periods=hours, freq="h")
    # A shaped spike at 14:00–16:00 then recovery
    pred = np.array([4.1, 5.3, 8.8, 6.3, 4.4, 4.7, 4.2, 4.0, 3.8, 3.9, 4.1, 4.0, 3.9])
    act  = np.array([4.2, 5.3, 8.1, 6.4, 4.4, 4.7, 4.1, 4.0, 3.9, 3.8, 4.2, 4.1, 3.8])
    status = ["Normal","Normal","Incident","Recovery","Recovery"] + ["Normal"]*(hours-5)
    label  = ["🟢","🟢","🔴","🟡","🟡"] + ["🟢"]*(hours-5)
    df = pd.DataFrame({
        "local_datetime": idx,
        "travel_time_predicted": pred[:hours],
        "travel_time_actual": act[:hours],
        "incident_status": label[:hours],
        "conditions": status[:hours],
        "direction": "NB",
        "metric": "TravelTime",
    })
    return df

def build_incident_day(df_source: pd.DataFrame | None,
                       day: datetime,
                       corridor="Washington Street",
                       direction="NB",
                       metric="TravelTime") -> pd.DataFrame:
    """
    Assemble a single-day, hourly DF with columns:
    local_datetime, predicted, actual, incident_status, conditions, direction, metric
    - If df_source is provided, it should already include predictions (or you can adapt here).
    - If not, synthetic is used for demo.
    """
    if df_source is None or df_source.empty:
        return _synth_incident_df(day)

    d0 = pd.to_datetime(day).normalize()
    d1 = d0 + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    df = df_source.copy()
    df["local_datetime"] = pd.to_datetime(df["local_datetime"], errors="coerce")
    df = df.dropna(subset=["local_datetime"])
    df = df[(df["local_datetime"] >= d0) & (df["local_datetime"] <= d1)]
    if "direction" in df.columns:
        df = df[df["direction"].astype(str).str.upper() == direction.upper()]
    if "metric" in df.columns:
        df = df[df["metric"].astype(str).str.lower() == metric.lower()]

    # Expect columns travel_time_predicted/actual; if not present, fall back to Strength as “actual”
    if "travel_time_actual" not in df.columns and "Strength" in df.columns:
        df = df.rename(columns={"Strength": "travel_time_actual"})
    if "travel_time_predicted" not in df.columns:
        # naive baseline=actual shifted; purely to render; replace with your predictor output
        df = df.sort_values("local_datetime")
        df["travel_time_predicted"] = df["travel_time_actual"].shift(1).bfill()

    # Tag conditions from rules (optional). Here: simple heuristic on intra-hour % change.
    df = df.sort_values("local_datetime").copy()
    df["hour_label"] = df["local_datetime"].apply(_fmt_hour_label)
    df["error_pct"] = (df["travel_time_predicted"] - df["travel_time_actual"]).abs() / df["travel_time_actual"].replace(0, np.nan) * 100
    df["error_pct"] = df["error_pct"].clip(lower=0)

    # Volatility & intra-hour build-up if Firsts/Lasts are present
    if {"Firsts","Lasts","Maximum","Strength"}.issubset(df.columns):
        df["buildup_rate_pct"] = (df["Lasts"] - df["Firsts"]) / df["Firsts"].replace(0, np.nan) * 100
        df["volatility_pct"] = (df["Maximum"] - df["Minimum"]) / df["Strength"].replace(0, np.nan) * 100
        # confidence (TravelTime rules)
        tt_spike_x = df["Maximum"] / df["Strength"].replace(0, np.nan)
        ramp_x     = df["Lasts"] / df["Firsts"].replace(0, np.nan)
        df["incident_confidence"] = (60*(tt_spike_x-1).clip(lower=0) + 40*(ramp_x-1).clip(lower=0)).clip(lower=0)*100/100
        df["incident_confidence"] = df["incident_confidence"].fillna(0).clip(0,100).round().astype(int)
        df["conditions"] = np.where((tt_spike_x>2.0) & (ramp_x>1.5), "Incident",
                             np.where(df["buildup_rate_pct"]>10, "Recovery", "Normal"))
        df["incident_status"] = np.where(df["conditions"]=="Incident","🔴",
                                  np.where(df["conditions"]=="Recovery","🟡","🟢"))
    else:
        # Fallback if those cols absent
        df["buildup_rate_pct"] = np.nan
        df["volatility_pct"] = np.nan
        df["incident_confidence"] = 0
        df["incident_status"] = "🟢"
        df["conditions"] = "Normal"

    return df


# =========================
# --------- UI -----------
# =========================

def _card(title, big, sub=None, color="#3498db", border=None):
    border = border or color
    st.markdown(
        f"""
        <div style="background: rgba(0,0,0,0.02); border:1px solid {border}; border-radius:10px; padding:12px; margin:6px 0; font-family: ui-monospace,Consolas,Monaco;">
          <div style="font-size:0.8rem; color:#666;">{title}</div>
          <div style="font-size:1.35rem; font-weight:800; color:{color};">{big}</div>
          <div style="font-size:0.75rem; color:#999;">{sub or ''}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _kpi_row(df_day: pd.DataFrame):
    """Compute & render KPIs from df_day."""
    # time-to-clear: from last "Incident" to first "Normal" after it
    ttc_min = None
    if "conditions" in df_day.columns:
        inc_idx = df_day.index[df_day["conditions"]=="Incident"].tolist()
        if inc_idx:
            last_inc = inc_idx[-1]
            post = df_day.loc[last_inc+1:]
            normal_after = post.index[post["conditions"]=="Normal"].tolist()
            if normal_after:
                dt_start = df_day.loc[last_inc,"local_datetime"]
                dt_end   = df_day.loc[normal_after[0],"local_datetime"]
                ttc_min  = (dt_end - dt_start).total_seconds()/60.0

    mae = (df_day["travel_time_predicted"] - df_day["travel_time_actual"]).abs().mean()
    mape = ( (df_day["travel_time_predicted"] - df_day["travel_time_actual"]).abs()
            / df_day["travel_time_actual"].replace(0,np.nan) * 100 ).mean()

    vol = df_day.get("volatility_pct", pd.Series([np.nan])).median()
    conf = df_day.get("incident_confidence", pd.Series([0])).max()
    base = df_day["travel_time_actual"].median()

    c1, c2, c3 = st.columns(3)
    with c1:
        _card("⏱ Time to Clear", f"{ttc_min:.1f} min" if ttc_min else "—", "Last incident → back to normal", color="#e74c3c", border="#ff6b6b")
    with c2:
        _card("🎯 Prediction Error", f"{mae:.2f} min  |  {mape:.0f}%", "MAE | MAPE", color="#9b59b6")
    with c3:
        _card("📈 Volatility / Confidence", f"{vol:.0f}%  |  {conf}%", "Median volatility | Max incident confidence", color="#2ecc71")

def _plot_day_timeline(df_day: pd.DataFrame, title="Predicted vs Actual (hourly)"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_day["local_datetime"], y=df_day["travel_time_actual"], name="Actual", mode="lines+markers",
        line=dict(width=2), marker=dict(size=6)
    ))
    fig.add_trace(go.Scatter(
        x=df_day["local_datetime"], y=df_day["travel_time_predicted"], name="Predicted", mode="lines+markers",
        line=dict(width=2, dash="dash"), marker=dict(size=6)
    ))
    # Error bars (color via hover text)
    fig.update_layout(
        title=title,
        margin=dict(l=10,r=10,t=40,b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="Time"),
        yaxis=dict(title="Travel Time (min)"),
    )
    st.plotly_chart(fig, use_container_width=True, config=dict(displaylogo=False, responsive=True))

def _render_hourly_grid(df_day: pd.DataFrame):
    """Compact grid like your original, but auto-build from DF."""
    st.markdown("##### Hourly Snapshot — Predicted • Actual • Error")

    cols = st.columns(len(df_day))
    for i, (_, row) in enumerate(df_day.iterrows()):
        err = row["error_pct"]
        color = _color_for_error(err)
        with cols[i]:
            st.markdown(
                f"""
                <div style="text-align:center; font-family: ui-monospace,Consolas,Monaco;">
                  <div style="padding:6px; background:rgba(52,152,219,0.08); border:1px solid #3498db; border-radius:6px; font-weight:800;">[{_fmt_hour_label(row['local_datetime'])}]</div>
                  <div style="margin-top:4px; color:#3498db; font-weight:800;">{row['travel_time_predicted']:.1f} min</div>
                  <div style="font-size:0.75rem; color:#999;">PREDICT</div>
                  <div style="margin-top:4px; color:#27ae60; font-weight:800;">{row['travel_time_actual']:.1f} min</div>
                  <div style="font-size:0.75rem; color:#999;">ACTUAL</div>
                  <div style="margin-top:4px; color:{color}; font-weight:800;">±{(0 if pd.isna(err) else err):.1f}%</div>
                  <div style="font-size:0.70rem; color:#999;">ERROR</div>
                </div>
                """,
                unsafe_allow_html=True
            )

def render_incident_detection_section(
        df_source: pd.DataFrame | None = None,
        corridor: str = "Washington Street",
        direction: str = "NB",
        day: datetime | None = None
    ):
    """
    Complete, reusable section. If df_source is provided, it should include:
      - local_datetime (datetime)
      - direction (NB/SB)
      - metric == 'TravelTime'
      - Strength, Firsts, Lasts, Minimum, Maximum (optional but used for confidence)
      - travel_time_predicted (optional; if absent, a naive predictor is used)
      - travel_time_actual (optional; Strength is used if present)
    """
    day = day or datetime.now().date()
    df_day = build_incident_day(df_source, day, corridor=corridor, direction=direction, metric="TravelTime")

    # Header
    if render_gradient_header:
        render_gradient_header(
            title=f"Incident Detection & Recovery — {corridor} {direction}",
            subtitle_left=f"{pd.to_datetime(day).strftime('%b %d, %Y')} | Validated Pattern Analysis",
            icon="⚠️"
        )
    else:
        st.markdown(f"### ⚠️ Incident Detection & Recovery — {corridor} {direction}")

    # KPI row from data
    _kpi_row(df_day)

    # Timeline chart (Pred vs Actual)
    st.markdown("---")
    _plot_day_timeline(df_day, title="Travel Time — Predicted vs Actual")

    # Grid snapshot (like your original, but data-driven)
    st.markdown("---")
    _render_hourly_grid(df_day)

    # Characteristics (data-driven text)
    st.markdown("---")
    st.markdown("##### 🚨 Incident Characteristics")
    # pull peak impact & recovery text from DF
    peak_idx = df_day["travel_time_actual"].idxmax()
    peak_val = df_day.loc[peak_idx, "travel_time_actual"]
    base_val = df_day["travel_time_actual"].median()
    build_pct = df_day.get("buildup_rate_pct", pd.Series([np.nan])).median()
    vol_pct = df_day.get("volatility_pct", pd.Series([np.nan])).median()

    bullets = [
        f"• Peak Impact: {peak_val:.1f} min travel time ({(peak_val/base_val-1)*100:.0f}% vs {base_val:.1f} min median baseline)" if base_val and not np.isnan(base_val) else "• Peak Impact: —",
        f"• Within-Hour Volatility (median): {vol_pct:.0f}% (from Min–Max vs Strength)" if not np.isnan(vol_pct) else "• Within-Hour Volatility: —",
        f"• Median Intra-Hour Build: {build_pct:.0f}% (Lasts vs Firsts)" if not np.isnan(build_pct) else "• Median Intra-Hour Build: —",
    ]
    for b in bullets:
        st.markdown(f"<div style='font-family: ui-monospace,Consolas,Monaco; color:#2c3e50; font-size:0.95rem; margin:4px 0;'>{b}</div>", unsafe_allow_html=True)

    # Model performance summary (data-driven)
    st.info(
        f"""
**Model Performance Summary**
• Rows: {len(df_day)}  • MAE: {(df_day['travel_time_predicted']-df_day['travel_time_actual']).abs().mean():.2f} min  • MAPE: {((df_day['travel_time_predicted']-df_day['travel_time_actual']).abs()/df_day['travel_time_actual'].replace(0,np.nan)*100).mean():.0f}%
• Max Incident Confidence: {df_day.get('incident_confidence', pd.Series([0])).max()}%
• Median Volatility: {df_day.get('volatility_pct', pd.Series([np.nan])).median():.0f}%
"""
    )
