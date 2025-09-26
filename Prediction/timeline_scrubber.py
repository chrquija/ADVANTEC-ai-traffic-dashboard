# timeline_scrubber.py — Shared timeline + gradient header for prediction models
import streamlit as st
from typing import Optional, Tuple, List

# -------------------------
# Gradient Header (matches your existing style)
# -------------------------
def render_gradient_header(
    title: str,
    subtitle_left: Optional[str] = None,
    subtitle_right: Optional[str] = None,
    icon: str = "📊",
) -> None:
    """
    Renders a gradient header banner. Use this at the top of each analysis section
    to keep the look consistent across tabs and sub-tabs.
    """
    st.markdown(
        f"""
<div style="
    background: linear-gradient(135deg, #2b77e5 0%, #19c3e6 100%);
    border-radius: 16px; padding: 18px 20px; color: #fff; margin: 8px 0 14px;
    box-shadow: 0 10px 26px rgba(25,115,210,.25); text-align: left;">
  <div style="display:flex; align-items:center; gap:10px; justify-content:space-between;">
    <div style="display:flex; align-items:center; gap:10px;">
      <div style="width:36px;height:36px;border-radius:10px;background:rgba(255,255,255,.18);
                  display:flex;align-items:center;justify-content:center;
                  box-shadow: inset 0 0 0 1px rgba(255,255,255,.15);">{icon}</div>
      <div style="font-size:1.9rem; font-weight:800; letter-spacing:.2px;">{title}</div>
    </div>
    {"<div style='opacity:.95;font-weight:600;'>" + subtitle_right + "</div>" if subtitle_right else ""}
  </div>
  {"<div style='margin-top:10px;opacity:.95;font-weight:600;'>" + subtitle_left + "</div>" if subtitle_left else ""}
</div>
""",
        unsafe_allow_html=True,
    )


# -------------------------
# Timeline CSS
# -------------------------
def _inject_timeline_css():
    """Inject CSS for timeline components (theme-aware)."""
    st.markdown(
        """
<style>
:root {
  --timeline-bg: rgba(79,172,254,.06);
  --timeline-border: rgba(79,172,254,.28);
  --timeline-text: #0f2f52;
  --timeline-active: #2b77e5;
  --timeline-shadow: 0 4px 12px rgba(0,0,0,.08);
}

html.dark, [data-theme="dark"], [data-base-theme="dark"], body[data-theme="dark"] {
  --timeline-bg: rgba(255,255,255,.08);
  --timeline-border: rgba(255,255,255,.18);
  --timeline-text: #ffffff;
  --timeline-active: #19c3e6;
  --timeline-shadow: 0 6px 16px rgba(0,0,0,.35);
}

.timeline-container {
  background: var(--timeline-bg);
  border: 1px solid var(--timeline-border);
  border-radius: 12px;
  padding: 14px 14px 10px 14px;
  margin: 10px 0 6px 0;
  box-shadow: var(--timeline-shadow);
}

.timeline-row {
  display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom: 8px;
  color: var(--timeline-text);
}

.timeline-hour {
  text-align:center; padding: 8px 10px; font-weight: 700; font-size: .95rem;
  border-radius: 10px; border: 1px solid var(--timeline-border); cursor: pointer;
  background: rgba(255,255,255,.85); color: var(--timeline-text);
  transition: all .18s ease; user-select:none;
}
.timeline-hour:hover { background: var(--timeline-active); color: #fff; transform: translateY(-1px); }
.timeline-hour.active { background: var(--timeline-active); color: #fff; box-shadow: 0 2px 8px rgba(43,119,229,.3); }

.forecast-pill {
  text-align:center; padding: 8px; background: var(--timeline-bg); border-radius: 10px;
  border: 1px solid var(--timeline-border);
}
</style>
""",
        unsafe_allow_html=True,
    )


# -------------------------
# Timeline Scrubber
# -------------------------
def create_timeline_scrubber(
    center_hour: int = 17,
    date_label: str = "Sep 19, 2024",
    window_size: int = 5,
    key_prefix: str = "timeline",
) -> Tuple[List[int], int]:
    """
    Interactive hour selector for a small window (e.g., 5 hours) centered around center_hour.

    Returns:
      (hour_list, selected_hour)
    """
    _inject_timeline_css()

    # Compute hour window bounds
    window_size = max(3, min(9, int(window_size)))  # Clamp for layout sanity
    half = window_size // 2
    start_hour = max(0, center_hour - half)
    end_hour = min(23, center_hour + half)
    # Adjust if clipping at ends
    if end_hour - start_hour + 1 < window_size:
        if start_hour == 0:
            end_hour = min(23, start_hour + window_size - 1)
        elif end_hour == 23:
            start_hour = max(0, end_hour - window_size + 1)

    hour_list = list(range(start_hour, end_hour + 1))

    # Header row
    st.markdown(
        f"""
<div class="timeline-container">
  <div class="timeline-row">
    <div style="font-weight:700;">◄── Interactive Timeline ──►</div>
    <div style="opacity:.85;">📅 {date_label}</div>
  </div>
""",
        unsafe_allow_html=True,
    )

    # Hour buttons
    cols = st.columns(len(hour_list))
    selected_hour = center_hour
    for i, hr in enumerate(hour_list):
        label = f"{hr:02d}:00"
        is_active = (hr == center_hour)
        button_label = f"{'• ' if is_active else ''}{label}"
        if cols[i].button(
            button_label,
            key=f"{key_prefix}_hr_{hr}",
            use_container_width=True,
            help=f"Center view on {label}",
        ):
            selected_hour = hr
            # No rerun here; caller can decide recentering logic

    st.markdown("</div>", unsafe_allow_html=True)
    return hour_list, selected_hour


# -------------------------
# Forecast Table (Predicted vs Actual + optional emojis)
# -------------------------
def create_hourly_forecast_table(
    hour_list: List[int],
    predictions: List[float],
    actuals: Optional[List[float]] = None,
    conditions: Optional[List[str]] = None,
    unit_label: str = "min",
    key_prefix: str = "forecast",
) -> None:
    """
    Render a compact forecast grid for the hours in hour_list.

    - predictions: required; same length as hour_list
    - actuals: optional; if provided, shows actuals + error row
    - conditions: optional; emoji/indicator per hour (e.g., ☀️ 🌤️ ⛈️)
    """
    if len(predictions) != len(hour_list):
        st.error("Predictions length must match hour_list.")
        return
    if actuals is not None and len(actuals) != len(hour_list):
        st.error("Actuals length must match hour_list.")
        return

    # Header row: hours
    header_cols = st.columns(len(hour_list))
    for i, hr in enumerate(hour_list):
        with header_cols[i]:
            st.markdown(
                f"""
<div class="forecast-pill">
  <div style="font-weight:800;">[{hr:02d}:00]</div>
</div>
""",
                unsafe_allow_html=True,
            )

    # Predictions row
    pred_cols = st.columns(len(hour_list))
    for i, val in enumerate(predictions):
        with pred_cols[i]:
            emoji = conditions[i] if (conditions and i < len(conditions)) else "☀️"
            st.markdown(
                f"""
<div style="text-align:center;margin-top:6px;">
  <div style="font-size:1.1rem;">{emoji}</div>
  <div style="font-weight:700;color:#2b77e5;">{val:.1f} {unit_label}</div>
  <div style="font-size:.78rem;opacity:.75;">PREDICT</div>
</div>
""",
                unsafe_allow_html=True,
            )

    # Actuals row (optional)
    if actuals is not None:
        act_cols = st.columns(len(hour_list))
        for i, val in enumerate(actuals):
            with act_cols[i]:
                st.markdown(
                    f"""
<div style="text-align:center;margin-top:6px;">
  <div style="font-weight:700;color:#27ae60;">{val:.1f} {unit_label}</div>
  <div style="font-size:.78rem;opacity:.75;">ACTUAL</div>
</div>
""",
                    unsafe_allow_html=True,
                )

        # Error row
        err_cols = st.columns(len(hour_list))
        for i, act in enumerate(actuals):
            with err_cols[i]:
                pred = predictions[i]
                err = abs(pred - act) / act * 100 if act and act != 0 else 0.0
                color = "#27ae60" if err <= 5 else ("#f39c12" if err <= 15 else "#e74c3c")
                st.markdown(
                    f"""
<div style="text-align:center;margin-top:2px;">
  <div style="font-weight:800;color:{color};">±{err:.1f}%</div>
  <div style="font-size:.72rem;opacity:.65;">ERROR</div>
</div>
""",
                    unsafe_allow_html=True,
                )


# -------------------------
# Prev/Next navigation (shift window outside this module)
# -------------------------
def create_navigation_controls(key_prefix: str = "nav") -> Tuple[bool, bool]:
    """
    Simple Previous/Next window controls.
    Caller should handle updating center_hour and re-render.
    """
    c1, _, c3 = st.columns([2.2, 3.6, 2.2])
    with c1:
        prev_clicked = st.button("◄ Previous 5 Hours", key=f"{key_prefix}_prev", use_container_width=True)
    with c3:
        next_clicked = st.button("Next 5 Hours ►", key=f"{key_prefix}_next", use_container_width=True)
    return prev_clicked, next_clicked