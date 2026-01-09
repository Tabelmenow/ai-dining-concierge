from fastapi import FastAPI, HTTPException, Query, Form, Header, Depends, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
import httpx
import uuid
import os
import re
import jwt
import base64
import hashlib
import secrets
from supabase import create_client
from datetime import datetime, timezone, timedelta
from urllib.parse import quote


app = FastAPI()

# =====================================================
# Env + Supabase client
# =====================================================
SUPABASE_URL = os.environ.get("SUPABASE_URL")              # e.g. https://xxxx.supabase.co
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")              # service key (server)
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")    # anon key (browser auth flows)
SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET")

if not SUPABASE_URL or not SUPABASE_KEY:
    supabase = None
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Render base URL (single source of truth for redirects)
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "https://ai-dining-concierge.onrender.com").rstrip("/")

# Cookie names
COOKIE_PKCE_VERIFIER = "tabel_pkce_verifier"
COOKIE_ACCESS_TOKEN = "tabel_access_token"


# =====================================================
# PKCE helpers
# =====================================================
def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")

def pkce_generate_verifier() -> str:
    # 43-128 chars recommended; we use 64 bytes -> ~86 chars base64url
    return _b64url(secrets.token_bytes(64))

def pkce_challenge_from_verifier(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return _b64url(digest)


# =====================================================
# Phase 3 · Step 1 — Supabase Auth (Identity)
# Accepts Bearer header OR cookie set by /auth/callback
# =====================================================
def get_current_user_id(
    request: Request,
    authorization: str = Header(None),
) -> str:
    token = None

    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "").strip()
    else:
        token = request.cookies.get(COOKIE_ACCESS_TOKEN)

    if not token:
        raise HTTPException(status_code=401, detail="Missing auth token (login first)")

    if not SUPABASE_JWT_SECRET:
        raise HTTPException(status_code=500, detail="Missing SUPABASE_JWT_SECRET env var")

    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
        )
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID missing in token")

    return user_id


# =====================================================
# OAuth UI + Flow (Google)
# =====================================================
@app.get("/ui/login", response_class=HTMLResponse)
def ui_login_page():
    return f"""
    <html>
      <head><title>Tabel Login</title></head>
      <body style="font-family: Arial; max-width: 700px; margin: 40px auto;">
        <h1>Login</h1>
        <p>Step 1: Click the button.</p>
        <p><a href="/auth/google/start">Sign in with Google</a></p>
        <hr>
        <p><small><a href="/">Home</a> | <a href="/ui/me">Who am I?</a> | <a href="/docs">Swagger</a></small></p>
      </body>
    </html>
    """


@app.get("/auth/google/start")
def auth_google_start(response: Response):
    """
    1) Create a PKCE verifier (secret)
    2) Compute PKCE challenge (public)
    3) Store verifier in a cookie
    4) Redirect browser to Supabase /authorize with challenge
    """
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise HTTPException(status_code=500, detail="Missing SUPABASE_URL or SUPABASE_ANON_KEY")

    verifier = pkce_generate_verifier()
    challenge = pkce_challenge_from_verifier(verifier)

    redirect_to = f"{PUBLIC_BASE_URL}/auth/callback"
    redirect_to_enc = quote(redirect_to, safe="")

    # Supabase hosted auth endpoint (browser redirect)
    url = (
        f"{SUPABASE_URL}/auth/v1/authorize"
        f"?provider=google"
        f"&redirect_to={redirect_to_enc}"
        f"&code_challenge={challenge}"
        f"&code_challenge_method=s256"
    )

    # Store verifier so callback can exchange code -> session
    # secure=True is correct on HTTPS (Render). If testing locally, set SECURE_COOKIES=false and toggle.
    secure_cookies = os.environ.get("SECURE_COOKIES", "true").lower() == "true"
    resp = RedirectResponse(url=url, status_code=302)
    resp.set_cookie(
        key=COOKIE_PKCE_VERIFIER,
        value=verifier,
        httponly=True,
        secure=secure_cookies,
        samesite="lax",
        max_age=10 * 60,  # 10 minutes
        path="/",
    )
    return resp


@app.get("/auth/callback", response_class=HTMLResponse)
async def auth_callback(request: Request):
    """
    Supabase redirects here with ?code=...
    We read PKCE verifier from cookie and exchange code for a session.
    Then we store access_token in an HttpOnly cookie for the UI.
    """
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise HTTPException(status_code=500, detail="Missing SUPABASE_URL or SUPABASE_ANON_KEY")

    code = request.query_params.get("code")
    error = request.query_params.get("error")
    error_desc = request.query_params.get("error_description")

    if error:
        return f"""
        <html><body style="font-family: Arial; max-width: 700px; margin: 40px auto;">
          <h1>Login failed</h1>
          <p><strong>Error:</strong> {error}</p>
          <p><strong>Details:</strong> {error_desc}</p>
          <p><a href="/ui/login">Try again</a></p>
        </body></html>
        """

    if not code:
        return """
        <html><body style="font-family: Arial; max-width: 700px; margin: 40px auto;">
          <h1>Missing code</h1>
          <p>Supabase did not send a 'code'.</p>
          <p>Most common causes:</p>
          <ul>
            <li>Redirect URLs not saved in Supabase Authentication → URL Configuration</li>
            <li>Google Console redirect URI is wrong (must be https://PROJECT.supabase.co/auth/v1/callback)</li>
            <li>PKCE setup is incomplete</li>
          </ul>
          <p><a href="/ui/login">Try again</a></p>
        </body></html>
        """

    verifier = request.cookies.get(COOKIE_PKCE_VERIFIER)
    if not verifier:
        return """
        <html><body style="font-family: Arial; max-width: 700px; margin: 40px auto;">
          <h1>Missing PKCE verifier</h1>
          <p>Your browser did not have the PKCE cookie.</p>
          <p>Fix: start again from <a href="/ui/login">/ui/login</a> and do not open multiple tabs.</p>
        </body></html>
        """

    token_url = f"{SUPABASE_URL}/auth/v1/token?grant_type=pkce"

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            token_url,
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Content-Type": "application/json",
            },
            json={
                "auth_code": code,
                "code_verifier": verifier,
            },
        )

    if resp.status_code >= 400:
        return f"""
        <html><body style="font-family: Arial; max-width: 700px; margin: 40px auto;">
          <h1>Token exchange failed</h1>
          <p>Status: {resp.status_code}</p>
          <pre>{resp.text}</pre>
          <p><a href="/ui/login">Try again</a></p>
        </body></html>
        """

    data = resp.json()
    access_token = data.get("access_token")
    user = (data.get("user") or {})
    email = user.get("email", "(no email returned)")
    provider = user.get("app_metadata", {}).get("provider", "unknown")

    if not access_token:
        return f"""
        <html><body style="font-family: Arial; max-width: 700px; margin: 40px auto;">
          <h1>Login incomplete</h1>
          <p>No access_token returned.</p>
          <pre>{data}</pre>
          <p><a href="/ui/login">Try again</a></p>
        </body></html>
        """

    secure_cookies = os.environ.get("SECURE_COOKIES", "true").lower() == "true"
    html = f"""
    <html>
      <head><title>Login success</title></head>
      <body style="font-family: Arial; max-width: 700px; margin: 40px auto;">
        <h1>Login success</h1>
        <p><strong>Provider:</strong> {provider}</p>
        <p><strong>Email:</strong> {email}</p>
        <p>This proves Google sign-in worked.</p>
        <hr>
        <p><a href="/ui/me">Go to /ui/me</a> (shows your user_id)</p>
        <p><a href="/docs">Swagger</a> (now works with cookie auth in UI endpoints)</p>
        <p><a href="/">Home</a></p>
      </body>
    </html>
    """

    response = HTMLResponse(content=html)
    # Save access token for UI usage
    response.set_cookie(
        key=COOKIE_ACCESS_TOKEN,
        value=access_token,
        httponly=True,
        secure=secure_cookies,
        samesite="lax",
        max_age=60 * 60 * 24,  # 1 day
        path="/",
    )
    # Clean up verifier cookie
    response.delete_cookie(COOKIE_PKCE_VERIFIER, path="/")
    return response


@app.get("/ui/me", response_class=HTMLResponse)
def ui_me(request: Request, user_id: str = Depends(get_current_user_id)):
    return f"""
    <html>
      <head><title>Me</title></head>
      <body style="font-family: Arial; max-width: 700px; margin: 40px auto;">
        <h1>You are logged in</h1>
        <p><strong>User ID:</strong> {user_id}</p>
        <p><a href="/ui/login">Login page</a> | <a href="/docs">Swagger</a> | <a href="/">Home</a></p>
      </body>
    </html>
    """


# =====================================================
# Config + Booking Helpers (kept minimal)
# =====================================================
MAX_CALL_ATTEMPTS = 2
CALL_COOLDOWN_MINUTES = 10
BOOKING_TIMEOUT_MINUTES = 60

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def parse_iso(dt_str: str) -> datetime:
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))

def compute_expires_at(created_at_iso: str) -> str:
    return (parse_iso(created_at_iso) + timedelta(minutes=BOOKING_TIMEOUT_MINUTES)).isoformat()

def log_event(booking_data: dict, event_type: str, details: dict | None = None):
    booking_data.setdefault("events", []).append({
        "type": event_type,
        "time": now_iso(),
        "details": details or {}
    })
    booking_data["last_updated_at"] = now_iso()

def ai_suggest_strategy(_):
    return {
        "recommended_action": "try_digital_first",
        "reason": "Try digital first; call if unavailable",
        "confidence": "medium",
    }

def try_digital_booking(req: dict):
    return {"success": False, "reason": "No digital availability"}

def should_call_restaurant(ctx: dict) -> bool:
    return (
        not ctx["digital_attempt"]["success"]
        and ctx["strategy"]["recommended_action"] == "try_digital_first"
        and ctx["request"]["party_size"] <= 8
    )


# =====================================================
# Models
# =====================================================
class BookingRequest(BaseModel):
    name: str
    city: str
    date: str
    time: str
    party_size: int
    time_window_minutes: int = 30
    notes: str = ""
    restaurant_name: str = ""
    restaurant_phone: str = ""


# =====================================================
# Core API (protected)
# =====================================================
@app.post("/book")
def book(req: BookingRequest, user_id: str = Depends(get_current_user_id)):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    booking_id = str(uuid.uuid4())
    req_data = req.model_dump()

    strategy = ai_suggest_strategy(req_data)
    digital = try_digital_booking(req_data)
    call_allowed = should_call_restaurant({
        "request": req_data,
        "strategy": strategy,
        "digital_attempt": digital,
    })

    booking_data = {
        "user_id": user_id,
        "request": req_data,
        "status": "pending",
        "strategy": strategy,
        "digital_attempt": digital,
        "call_allowed": call_allowed,
        "created_at": now_iso(),
        "expires_at": None,
        "events": [],
    }

    booking_data["expires_at"] = compute_expires_at(booking_data["created_at"])
    log_event(booking_data, "booking_created")

    supabase.table("bookings").insert({"id": booking_id, "data": booking_data}).execute()

    return {"booking_id": booking_id, "status": "pending", "expires_at": booking_data["expires_at"]}


def load_booking(booking_id: str, user_id: str) -> dict:
    result = supabase.table("bookings").select("data").eq("id", booking_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Booking not found")

    booking_data = result.data[0]["data"]
    if booking_data.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not authorised")

    return booking_data


@app.get("/status/{booking_id}")
def status(booking_id: str, user_id: str = Depends(get_current_user_id)):
    return load_booking(booking_id, user_id)


@app.get("/timeline/{booking_id}")
def timeline(booking_id: str, user_id: str = Depends(get_current_user_id)):
    booking = load_booking(booking_id, user_id)
    return {
        "status": booking.get("status"),
        "expires_at": booking.get("expires_at"),
        "timeline": booking.get("events", []),
    }


# =====================================================
# UI
# =====================================================
@app.get("/", response_class=HTMLResponse)
def home():
    return f"""
    <html>
      <head><title>Tabel</title></head>
      <body style="font-family: Arial; max-width: 700px; margin: 40px auto;">
        <h1>Tabel</h1>
        <p>Phase 3: Auth + protected endpoints.</p>
        <ul>
          <li><a href="/ui/login">Login</a> (Google)</li>
          <li><a href="/ui/me">Who am I?</a> (requires login)</li>
          <li><a href="/docs">Swagger</a></li>
        </ul>
      </body>
    </html>
    """
