import time
import contextlib
import streamlit as st

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


def is_active_tab(tab_id: str) -> bool:
    """Decide if a tab should appear active in the sidebar.
    Priority: active_search_tab while loaders run; otherwise last_active_tab.
    """
    active = st.session_state.get("active_search_tab")
    if active:
        return active == tab_id
    return st.session_state.get("last_active_tab") == tab_id
