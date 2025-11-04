import streamlit as st
from typing import Optional, Dict, Any
import json
import time
import hashlib
import secrets
from pathlib import Path
from datetime import datetime, timedelta

_ALLOWED_DOMAIN = "advantec-usa.com"
_STORE_FILE = Path(__file__).resolve().parent / "users.json"
_RESET_TTL_MIN = 15

# ---------- Basic helpers ----------

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
    try:
        st.rerun()
    except Exception:
        pass


# ---------- User store (JSON) ----------

def _load_store() -> Dict[str, Any]:
    try:
        if _STORE_FILE.exists():
            with open(_STORE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"users": {}}


def _save_store(store: Dict[str, Any]):
    try:
        tmp = _STORE_FILE.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2)
        tmp.replace(_STORE_FILE)
    except Exception as e:
        st.error(f"Failed to save user store: {e}")


def _hash_password(password: str, salt: Optional[str] = None) -> Dict[str, str]:
    salt = salt or secrets.token_hex(16)
    h = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return {"salt": salt, "hash": h}


def _verify_password(password: str, salt: str, hash_hex: str) -> bool:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest() == hash_hex


def get_user(email: str) -> Optional[Dict[str, Any]]:
    email = _normalize_email(email)
    store = _load_store()
    return store.get("users", {}).get(email)


def create_or_set_user(email: str, password: str):
    email = _normalize_email(email)
    if not validate_company_email(email):
        raise ValueError("Email must be an @advantec-usa.com address")
    store = _load_store()
    creds = _hash_password(password)
    store.setdefault("users", {})[email] = {
        "email": email,
        "password": creds["hash"],
        "salt": creds["salt"],
        "reset_code": None,
        "reset_expiry": None,
        "created": int(time.time()),
        "updated": int(time.time()),
    }
    _save_store(store)


def set_password(email: str, new_password: str):
    email = _normalize_email(email)
    store = _load_store()
    user = store.setdefault("users", {}).get(email)
    if not user:
        raise ValueError("User not found")
    creds = _hash_password(new_password)
    user["password"], user["salt"] = creds["hash"], creds["salt"]
    user["updated"] = int(time.time())
    user["reset_code"], user["reset_expiry"] = None, None
    _save_store(store)


# ---------- Password reset via 5-digit code ----------

def _send_email(to_email: str, subject: str, body: str) -> bool:
    # Try SMTP via st.secrets; fallback to displaying the code inline for manual delivery
    smtp = st.secrets.get("smtp") if hasattr(st, "secrets") else None
    if smtp and all(k in smtp for k in ("host", "port", "user", "password", "sender")):
        try:
            import smtplib
            from email.mime.text import MIMEText
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = smtp["sender"]
            msg["To"] = to_email
            with smtplib.SMTP(smtp["host"], int(smtp["port"])) as server:
                server.starttls()
                server.login(smtp["user"], smtp["password"])
                server.send_message(msg)
            return True
        except Exception as e:
            st.warning(f"Email send failed, showing code here for now: {e}")
    # Fallback: show in UI for testing
    st.info(body)
    return False


def start_password_reset(email: str) -> Optional[str]:
    email = _normalize_email(email)
    if not validate_company_email(email):
        st.error("Please enter your @advantec-usa.com email")
        return None
    store = _load_store()
    users = store.setdefault("users", {})
    if email not in users:
        st.error("User not found. Ask admin to create your account.")
        return None
    code = f"{secrets.randbelow(90000)+10000:05d}"
    expiry = int((datetime.utcnow() + timedelta(minutes=_RESET_TTL_MIN)).timestamp())
    users[email]["reset_code"], users[email]["reset_expiry"] = code, expiry
    _save_store(store)
    _send_email(
        email,
        "Your ADVANTEC Dashboard reset code",
        f"Your 5-digit reset code is {code}. It expires in {_RESET_TTL_MIN} minutes.",
    )
    return code


def verify_password_reset(email: str, code: str, new_password: str) -> bool:
    email = _normalize_email(email)
    store = _load_store()
    user = store.get("users", {}).get(email)
    if not user:
        st.error("User not found")
        return False
    now = int(datetime.utcnow().timestamp())
    if not user.get("reset_code") or not user.get("reset_expiry"):
        st.error("No active reset code. Click 'Send Code' first.")
        return False
    if now > int(user["reset_expiry"]):
        st.error("Reset code expired. Please request a new code.")
        return False
    if str(code).strip() != str(user["reset_code"]).strip():
        st.error("Invalid code. Please check the email and try again.")
        return False
    # Set password and clear code
    set_password(email, new_password)
    return True


# ---------- UI: Login + Admin + Forgot Password ----------

def require_company_login(domain: str = _ALLOWED_DOMAIN) -> bool:
    """Render login UI and enforce company domain + password auth.
    Returns True when authenticated; otherwise renders the auth UI and returns False.
    """
    if is_authenticated(domain):
        return True

    # Center the entire sign-in experience within a narrower column so the page
    # doesn't feel "wide" on large displays while keeping the main app wide.
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.markdown(
            """
            <div style='max-width:720px;margin:40px auto 10px; padding:24px; border-radius:16px;'
                 'background:linear-gradient(135deg, rgba(79,172,254,.08), rgba(0,242,254,.08));'
                 'border:1px solid rgba(79,172,254,.25); box-shadow:0 8px 26px rgba(0,0,0,.08);'>
              <h2 style='margin:0 0 8px 0;'>ADVANTEC Dashboard — Sign in</h2>
              <p style='margin:0;opacity:.85;'>Use your company email and password</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

        tabs = st.tabs(["Sign In", "Forgot Password", "Admin: Add User"])

    # Sign In tab
    with tabs[0]:
        with st.form("auth_signin", clear_on_submit=False):
            email = st.text_input(
                "Company Email",
                value=_normalize_email(st.session_state.get("auth_email", "")),
            )
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In", type="primary")
        if submitted:
            if not validate_company_email(email, domain):
                st.error(f"Access restricted. Please use your @{domain} email address.")
            else:
                user = get_user(email)
                if not user:
                    st.error("Account not found. Ask admin to create it.")
                else:
                    if _verify_password(password, user.get("salt", ""), user.get("password", "")):
                        st.session_state["authenticated"] = True
                        st.session_state["auth_email"] = _normalize_email(email)
                        try:
                            st.rerun()
                        except Exception:
                            pass
                    else:
                        st.error("Incorrect password. Try again or use Forgot Password.")

    # Forgot Password tab
    with tabs[1]:
        st.write("Reset your password in two steps.")
        c1, c2 = st.columns([1, 1])
        with c1:
            email_fp = st.text_input("Your Company Email", key="fp_email")
            if st.button("Send Code", key="fp_send"):
                if validate_company_email(email_fp, domain):
                    start_password_reset(email_fp)
        with c2:
            code = st.text_input("5-digit Code", key="fp_code")
            new_pw = st.text_input("New Password", type="password", key="fp_new")
            if st.button("Change Password", key="fp_change"):
                if verify_password_reset(email_fp, code, new_pw):
                    st.success("Password updated. You can now sign in.")

    # Admin tab — gate with a shared setup key from secrets
    with tabs[2]:
        admin_key = (st.secrets.get("admin_setup_key") if hasattr(st, "secrets") else None)
        if not admin_key:
            st.info("To enable admin user creation, set st.secrets['admin_setup_key'].")
        provided = st.text_input("Setup Key", type="password")
        email_admin = st.text_input("User Email (@advantec-usa.com)")
        init_pw = st.text_input("Initial Password", type="password")
        if st.button("Create/Reset User"):
            if provided != (admin_key or ""):
                st.error("Invalid setup key.")
            elif not validate_company_email(email_admin, domain):
                st.error(f"Email must end with @{domain}")
            elif not init_pw:
                st.error("Please enter an initial password.")
            else:
                try:
                    create_or_set_user(email_admin, init_pw)
                    st.success(f"User {email_admin} created/updated.")
                except Exception as e:
                    st.error(str(e))

    st.caption(f"Only emails ending with @{domain} are allowed.")
    return False


def render_auth_sidebar_footer():
    """Render the signed-in chip and Sign out at the very bottom of the sidebar.
    Call this once at the end of the app script.
    """
    if not is_authenticated():
        return
    with st.sidebar:
        st.markdown(
            f"<div style='padding:10px;border:1px solid rgba(79,172,254,.25);"
            f"border-radius:8px;background:rgba(79,172,254,.06);margin-top:12px;'>"
            f"<strong>Signed in:</strong><br><code>{st.session_state.get('auth_email')}</code>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if st.button("Sign out", key="auth_signout_footer"):
            logout()
