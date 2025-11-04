import streamlit as st
from typing import Optional

_ALLOWED_DOMAIN = "advantec-usa.com"


def _normalize_email(email: str | None) -> str:
    if not email:
        return ""
    return str(email).strip().lower()


def validate_company_email(email: str, domain: str = _ALLOWED_DOMAIN) -> bool:
    e = _normalize_email(email)
    return e.endswith("@" + domain)


def get_user_email() -> Optional[str]:
    email = st.session_state.get("auth_email")
    return _normalize_email(email) or None


def is_authenticated(domain: str = _ALLOWED_DOMAIN) -> bool:
    authed = bool(st.session_state.get("authenticated", False))
    email = get_user_email()
    if not authed or not email:
        return False
    return validate_company_email(email, domain)


def logout():
    for k in ("authenticated", "auth_email"):
        if k in st.session_state:
            del st.session_state[k]
    # Best-effort rerun to refresh UI
    try:
        st.rerun()
    except Exception:
        pass


def require_company_login(domain: str = _ALLOWED_DOMAIN) -> bool:
    """
    Render a minimal login gate that only accepts emails ending with @advantec-usa.com (by default).

    Returns True if the user is authenticated; False otherwise (and renders the login UI).
    """
    # If already authenticated and valid, nothing to do
    if is_authenticated(domain):
        # Optionally show a tiny account box in the sidebar
        with st.sidebar:
            st.markdown(
                f"<div style='padding:6px 10px;border:1px solid rgba(79,172,254,.25);"
                f"border-radius:8px;background:rgba(79,172,254,.06);margin-bottom:8px;'>"
                f"<strong>Signed in:</strong><br><code>{st.session_state.get('auth_email')}</code>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button("Sign out", key="auth_signout"):
                logout()
        return True

    # Not authenticated — show a simple login card in the main area
    st.markdown("""
        <div style='max-width:680px;margin:40px auto 10px; padding:24px; border-radius:16px;'
             'background:linear-gradient(135deg, rgba(79,172,254,.08), rgba(0,242,254,.08));'
             'border:1px solid rgba(79,172,254,.25); box-shadow:0 8px 26px rgba(0,0,0,.08);'>
          <h2 style='margin:0 0 8px 0;'>Welcome to the ADVANTEC Dashboard</h2>
          <p style='margin:0;opacity:.85;'>Please sign in with your company email</p>
        </div>
    """, unsafe_allow_html=True)

    with st.form("auth_form", clear_on_submit=False):
        email = st.text_input("Company Email", value=_normalize_email(st.session_state.get("auth_email", "")))
        submitted = st.form_submit_button("Sign In", type="primary")

    if submitted:
        if validate_company_email(email, domain):
            st.session_state["authenticated"] = True
            st.session_state["auth_email"] = _normalize_email(email)
            # Rerun so the rest of the app can render
            try:
                st.rerun()
            except Exception:
                pass
        else:
            st.error(f"Access restricted. Please use your @{domain} email address.")

    # Also offer a tiny hint
    st.caption(f"Only emails ending with @{domain} are allowed.")

    return False
