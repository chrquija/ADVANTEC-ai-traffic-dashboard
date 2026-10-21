import time
import contextlib
import streamlit as st
import pandas as pd

class PerformanceMonitor:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.timings = {}
        self.start_times = {}

    def start(self, name):
        if self.enabled:
            self.start_times[name] = time.perf_counter()

    def stop(self, name):
        if self.enabled and name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[name]
            self.timings[name] = self.timings.get(name, 0) + elapsed
            del self.start_times[name]

    def report(self):
        if not self.enabled or not self.timings:
            return
        
        # Determine target: if in a fragment not in sidebar, st.sidebar will fail.
        # We try st.sidebar first, fallback to st if it fails due to fragment constraints.
        try:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### ⏱️ Performance Metrics")
            for name, elapsed in self.timings.items():
                st.sidebar.text(f"{name}: {elapsed:.3f}s")
        except Exception:
            # If st.sidebar fails (likely due to fragment), render in main area
            st.markdown("---")
            st.markdown("### ⏱️ Performance Metrics")
            for name, elapsed in self.timings.items():
                st.text(f"{name}: {elapsed:.3f}s")

    @contextlib.contextmanager
    def track(self, name):
        if not self.enabled:
            yield
            return
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

@contextlib.contextmanager
def cad_loader(title: str = "Processing…", tab_id: str | None = None):
    """Show the CAD-style progress UI only if this tab is the active search tab.
    If tab_id is None, always show the loader (legacy behavior).
    If tab_id is provided, animate only when it matches session_state['active_search_tab'].
    In inactive tabs, yield a no-op step so they render without animation.
    """
    active = True
    if tab_id is not None:
        active = (st.session_state.get("active_search_tab") == tab_id)

    if not active:
        def step(_msg: str, _pct: int | float):
            pass
        try:
            yield step
        finally:
            pass
        return

    title_placeholder = st.empty()
    log_placeholder = st.empty()
    bar_placeholder = st.empty()

    with title_placeholder:
        st.markdown(f"### {title}")

    log_container = log_placeholder.container()
    progress_bar = bar_placeholder.progress(0)

    def step(msg: str, pct: int | float):
        with log_container:
            st.write(f"• {msg}")
        try:
            pct_int = int(max(0, min(100, pct)))
        except Exception:
            pct_int = 0
        progress_bar.progress(pct_int)

    try:
        yield step
        progress_bar.progress(100)
        with log_container:
            st.success("✔️ Done")
        time.sleep(0.5)
    finally:
        title_placeholder.empty()
        log_placeholder.empty()
        bar_placeholder.empty()
        if tab_id is not None and st.session_state.get("active_search_tab") == tab_id:
            st.session_state["active_search_tab"] = None

def set_active_search_tab(tab_id: str):
    """Mark a tab as the active one initiating a Search."""
    st.session_state["active_search_tab"] = tab_id




@st.cache_data(show_spinner=False)
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Efficiently convert dataframe to CSV bytes for download buttons."""
    if df.empty:
        return b""
    return df.to_csv(index=False).encode("utf-8")

@st.cache_data(show_spinner=False)
def get_dynamic_xaxis_params(start_date, end_date):
    """
    Returns a dict with 'dtick' and 'tickformat' for Plotly xaxis
    based on the number of days between start_date and end_date.
    """
    import pandas as pd
    delta = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days

    if delta <= 1:
        # 1 day or less: 4 hours
        return {"dtick": 14400000, "tickformat": "%b %d\n%I:%M %p"}
    elif delta <= 3:
        # 2-3 days: 6 hours
        return {"dtick": 21600000, "tickformat": "%b %d\n%I:%M %p"}
    elif delta <= 7:
        # 4-7 days: 12 hours
        return {"dtick": 43200000, "tickformat": "%b %d\n%I:%M %p"}
    elif delta <= 14:
        # 8-14 days: 24 hours (1 day)
        return {"dtick": 86400000, "tickformat": "%b %d"}
    elif delta <= 31:
        # Up to a month: 2 days
        return {"dtick": 172800000, "tickformat": "%b %d"}
    else:
        # More than a month: 7 days
        return {"dtick": 604800000, "tickformat": "%b %d, %Y"}


# Boxed KPI cards (same look as the Acyclica and VantageLive tabs)
KPI_CARD_CSS = """
<style>
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 12px;
    width: 100%;
    margin: 10px 0 25px 0;
}
@media (max-width: 1200px) { .kpi-grid { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 768px) { .kpi-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 480px) { .kpi-grid { grid-template-columns: 1fr; } }
.kpi-card {
    min-width: 0;
    box-sizing: border-box;
    background: white;
    padding: 1.25rem;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    border: 1px solid #eaedf2;
    display: flex;
    flex-direction: column;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    height: 100%;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0,0,0,0.08); }
.kpi-card-title {
    font-size: 0.8rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    min-height: 2.5em;
    line-height: 1.2;
}
.kpi-card-value { font-size: 1.8rem; font-weight: 800; color: #1e293b; margin-bottom: 0.25rem; line-height: 1.1; }
.kpi-card-delta { font-size: 0.9rem; font-weight: 600; color: #27ae60; margin-bottom: 1.25rem; }
.kpi-card-badge { margin-top: auto; }
.kpi-help-icon {
    cursor: help;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #f1f5f9;
    color: #94a3b8;
    font-size: 11px;
    font-weight: bold;
}
</style>
"""


def render_kpi_card(title: str, value: str, delta: str, badge_text: str, badge_class: str, help_text: str) -> str:
    """HTML for a single boxed KPI card; place cards inside <div class="kpi-grid">."""
    # Encode newlines: a blank line inside st.markdown ends the HTML block and leaks the rest as text
    safe_help = (help_text or "").replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace("\n", "&#10;")
    badge_html = f'<div class="kpi-card-badge"><span class="performance-badge {badge_class}">{badge_text}</span></div>' if badge_text else ""
    return f'<div class="kpi-card"><div class="kpi-card-title"><span>{title}</span><span class="kpi-help-icon" title="{safe_help}">?</span></div><div class="kpi-card-value">{value}</div><div class="kpi-card-delta">{delta}</div>{badge_html}</div>'
