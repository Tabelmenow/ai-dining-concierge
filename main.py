from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    Form,
    Request,
    Header,
    Depends,
)
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
import uuid
import os
import re
import secrets
import base64
import hashlib
import httpx
import jwt
from supabase import create_client
from datetime import datetime, timezone, timedelta


app = FastAPI()

# =====================================================
# Env / Supabase clients
# =====================================================
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")  # server key (service role recommended)
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")  # for browser auth endpoints
SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET")  # for local JWT verification

if not SUPABASE_URL or not SUPABASE_KEY:
    supabase = None
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

if not SUPABASE_URL:
    raise RuntimeError("Missing SUPABASE_URL")

# Render is HTTPS, so secure cookies are OK
COOKIE_SECURE = True
COOKIE_SAMESITE = "lax"

# =====================================================
# Phase 3 · Identity (Google OAuth via Supabase) — PROPER PKCE
# =====================================================
def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")

def make_pkce_verifier() -> str:
    # 43-128 chars; secrets.token_urlsafe typically produces a safe-length string
    v = secrets.token_urlsafe(64)
    return v[:128]

def make_pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return _b64url(digest)

def get_token_from_request(request: Request) -> str | None:
    # 1) Authorization header (future mobile app)
    auth = request.headers.get("authorization") or ""
    if auth.startswith("Bearer "):
        return auth.replace("Bearer ", "").strip()

    # 2) Cookie (browser UI)
    return request.cookies.get("sb_access_token")

def get_current_user_id_from_request(request: Request) -> str:
    token = get_token_from_request(request)
    if not token:
        raise HTTPException(status_code=401, detail="Not logged in (no token found)")

    if not SUPABASE_JWT_SECRET:
        raise HTTPException(status_code=500, detail="SUPABASE_JWT_SECRET not configured on server")

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

def get_current_user_id(request: Request) -> str:
    return get_current_user_id_from_request(request)

# =====================================================
# UI A) Minimal flow
# /ui/login -> Google -> /auth/callback -> cookie set -> /ui/me -> /ui/book form
# =====================================================
@app.get("/ui/login", response_class=HTMLResponse)
def ui_login_page():
    return """
    <html>
      <head><title>Tabel · Login</title></head>
      <body style="font-family: Arial; max-width: 720px; margin: 40px auto;">
        <h1>Login</h1>
        <p>Click the button to sign in with Google.</p>
        <p>
          <a href="/auth/google/start"
             style="display:inline-block; padding:10px 14px; border:1px solid #333; text-decoration:none;">
            Sign in with Google
          </a>
        </p>
        <hr>
        <p><small><a href="/ui/me">Me</a> | <a href="/ui/book">Book</a> | <a href="/docs">Swagger</a></small></p>
      </body>
    </html>
    """

@app.get("/auth/google/start")
def auth_google_start():
    """
    Step 1: Create PKCE verifier+challenge.
    Step 2: Store verifier in a cookie (so callback can exchange code securely).
    Step 3: Redirect to Supabase authorize URL.
    """
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise HTTPException(status_code=500, detail="Missing SUPABASE_URL or SUPABASE_ANON_KEY")

    verifier = make_pkce_verifier()
    challenge = make_pkce_challenge(verifier)

    redirect_to = "https://ai-dining-concierge.onrender.com/auth/callback"

    authorize_url = (
        f"{SUPABASE_URL}/auth/v1/authorize"
        f"?provider=google"
        f"&redirect_to={redirect_to}"
        f"&code_challenge={challenge}"
        f"&code_challenge_method=s256"
    )

    resp = RedirectResponse(url=authorize_url, status_code=302)
    resp.set_cookie(
        key="sb_pkce_verifier",
        value=verifier,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        max_age=10 * 60,  # 10 minutes
    )
    return resp

@app.get("/auth/callback", response_class=HTMLResponse)
async def auth_callback(request: Request):
    """
    Supabase redirects back here with ?code=...
    We exchange that code + code_verifier for a session.
    Then set sb_access_token cookie for browser UI.
    """
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise HTTPException(status_code=500, detail="Missing SUPABASE_URL or SUPABASE_ANON_KEY")

    error = request.query_params.get("error")
    error_desc = request.query_params.get("error_description")
    if error:
        return f"""
        <html><body style="font-family: Arial; max-width: 720px; margin: 40px auto;">
          <h1>Login failed</h1>
          <p><strong>Error:</strong> {error}</p>
          <p><strong>Details:</strong> {error_desc}</p>
          <p><a href="/ui/login">Try again</a></p>
        </body></html>
        """

    code = request.query_params.get("code")
    if not code:
        return """
        <html><body style="font-family: Arial; max-width: 720px; margin: 40px auto;">
          <h1>Missing code</h1>
          <p>No <code>code</code> was returned. This usually means Redirect URLs in Supabase are wrong.</p>
          <p><a href="/ui/login">Try again</a></p>
        </body></html>
        """

    verifier = request.cookies.get("sb_pkce_verifier")
    if not verifier:
        return """
        <html><body style="font-family: Arial; max-width: 720px; margin: 40px auto;">
          <h1>Missing PKCE verifier</h1>
          <p>The server couldn't find the verifier cookie. Click Login again.</p>
          <p><a href="/ui/login">Try again</a></p>
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
        <html><body style="font-family: Arial; max-width: 720px; margin: 40px auto;">
          <h1>Token exchange failed</h1>
          <p>Status: {resp.status_code}</p>
          <pre style="white-space: pre-wrap;">{resp.text}</pre>
          <p><a href="/ui/login">Try again</a></p>
        </body></html>
        """

    data = resp.json()
    access_token = data.get("access_token")
    user = (data.get("user") or {})
    email = user.get("email", "(no email returned)")
    provider = user.get("app_metadata", {}).get("provider", "unknown")

    if not access_token:
        return """
        <html><body style="font-family: Arial; max-width: 720px; margin: 40px auto;">
          <h1>No access token returned</h1>
          <p>OAuth succeeded but token response was missing.</p>
          <p><a href="/ui/login">Try again</a></p>
        </body></html>
        """

    html = f"""
    <html>
      <head><title>Login success</title></head>
      <body style="font-family: Arial; max-width: 720px; margin: 40px auto;">
        <h1>Login success</h1>
        <p><strong>Provider:</strong> {provider}</p>
        <p><strong>Email:</strong> {email}</p>
        <p>Now go to <a href="/ui/me">/ui/me</a> or <a href="/ui/book">/ui/book</a>.</p>
      </body>
    </html>
    """

    response = HTMLResponse(html)

    # Store access token in cookie so UI pages work without manual headers
    response.set_cookie(
        key="sb_access_token",
        value=access_token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        max_age=60 * 60,  # 1 hour (token expiry may differ; this is just cookie lifetime)
    )

    # Clean up verifier cookie
    response.delete_cookie("sb_pkce_verifier")

    return response

@app.get("/ui/logout")
def ui_logout():
    resp = RedirectResponse(url="/ui/login", status_code=302)
    resp.delete_cookie("sb_access_token")
    resp.delete_cookie("sb_pkce_verifier")
    return resp

@app.get("/ui/me", response_class=HTMLResponse)
def ui_me(request: Request):
    try:
        user_id = get_current_user_id_from_request(request)
        return f"""
        <html>
          <head><title>Tabel · Me</title></head>
          <body style="font-family: Arial; max-width: 720px; margin: 40px auto;">
            <h1>You are logged in</h1>
            <p><strong>User ID:</strong> {user_id}</p>
            <p>
              <a href="/ui/book">Make a booking</a> |
              <a href="/ui/logout">Logout</a>
            </p>
          </body>
        </html>
        """
    except HTTPException as e:
        return f"""
        <html>
          <head><title>Tabel · Me</title></head>
          <body style="font-family: Arial; max-width: 720px; margin: 40px auto;">
            <h1>Not logged in</h1>
            <p>{e.detail}</p>
            <p><a href="/ui/login">Go to Login</a></p>
          </body>
        </html>
        """

# =====================================================
# Booking logic (kept minimal & identity-safe)
# =====================================================
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

@app.post("/book")
def book(req: BookingRequest, user_id: str = Depends(get_current_user_id)):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    booking_id = str(uuid.uuid4())

    try:
        req_data = req.model_dump()
    except AttributeError:
        req_data = req.dict()

    booking_data = {
        "user_id": user_id,
        "request": req_data,
        "status": "pending",
        "created_at": now_iso(),
        "expires_at": None,
        "events": [],
    }
    booking_data["expires_at"] = compute_expires_at(booking_data["created_at"])
    log_event(booking_data, "booking_created", {"city": req_data.get("city"), "party_size": req_data.get("party_size")})

    supabase.table("bookings").insert({
        "id": booking_id,
        "data": booking_data
    }).execute()

    return {"booking_id": booking_id, "status": booking_data["status"], "expires_at": booking_data["expires_at"]}

def load_booking(booking_id: str, user_id: str) -> dict:
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")

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
        "booking_id": booking_id,
        "status": booking.get("status"),
        "expires_at": booking.get("expires_at"),
        "timeline": booking.get("events", []),
    }

# =====================================================
# UI — booking form (Option A)
# =====================================================
@app.get("/ui/book", response_class=HTMLResponse)
def ui_book_form(request: Request):
    # Require login
    user_id = get_current_user_id_from_request(request)

    return f"""
    <html>
      <head><title>Tabel · Book</title></head>
      <body style="font-family: Arial; max-width: 720px; margin: 40px auto;">
        <h1>Make a booking</h1>
        <p><small>Logged in as: {user_id}</small></p>

        <form action="/ui/book" method="post">
          <label>Your name</label><br>
          <input name="name" required style="width: 100%; padding: 8px;"><br><br>

          <label>City</label><br>
          <input name="city" required style="width: 100%; padding: 8px;"><br><br>

          <label>Date</label><br>
          <input type="date" name="date" required style="width: 100%; padding: 8px;"><br><br>

          <label>Time</label><br>
          <input type="time" name="time" required style="width: 100%; padding: 8px;"><br><br>

          <label>Party size</label><br>
          <input type="number" name="party_size" min="1" required style="width: 100%; padding: 8px;"><br><br>

          <label>Flex window (minutes)</label><br>
          <input type="number" name="time_window_minutes" min="0" value="30" style="width: 100%; padding: 8px;"><br><br>

          <label>Notes (optional)</label><br>
          <input name="notes" style="width: 100%; padding: 8px;"><br><br>

          <hr style="margin: 20px 0;">

          <label>Restaurant name (optional)</label><br>
          <input name="restaurant_name" style="width: 100%; padding: 8px;"><br><br>

          <label>Restaurant phone (optional)</label><br>
          <input name="restaurant_phone" style="width: 100%; padding: 8px;"><br><br>

          <button type="submit" style="padding: 10px 16px;">Create booking</button>
          <a href="/ui/me" style="margin-left: 12px;">Me</a>
          <a href="/ui/logout" style="margin-left: 12px;">Logout</a>
        </form>

        <hr style="margin: 30px 0;">
        <p><small><a href="/docs">Swagger</a></small></p>
      </body>
    </html>
    """

@app.post("/ui/book")
def ui_book_submit(
    request: Request,
    name: str = Form(...),
    city: str = Form(...),
    date: str = Form(...),
    time: str = Form(...),
    party_size: int = Form(...),
    time_window_minutes: int = Form(30),
    notes: str = Form(""),
    restaurant_name: str = Form(""),
    restaurant_phone: str = Form(""),
):
    user_id = get_current_user_id_from_request(request)

    req = BookingRequest(
        name=name,
        city=city,
        date=date,
        time=time,
        party_size=party_size,
        time_window_minutes=time_window_minutes,
        notes=notes,
        restaurant_name=restaurant_name,
        restaurant_phone=restaurant_phone,
    )

    result = book(req, user_id)
    return RedirectResponse(url=f"/ui/status/{result['booking_id']}", status_code=303)

@app.get("/ui/status/{booking_id}", response_class=HTMLResponse)
def ui_status_page(request: Request, booking_id: str):
    user_id = get_current_user_id_from_request(request)
    booking = load_booking(booking_id, user_id)

    status_text = booking.get("status", "unknown")
    expires_at = booking.get("expires_at", "")
    events = booking.get("events", [])

    items = ""
    if not events:
        items = "<li>No events yet.</li>"
    else:
        for e in events:
            items += f"<li>{e.get('type')}<br><small>{e.get('time')}</small></li>"

    return f"""
    <html>
      <head><title>Tabel · Status</title></head>
      <body style="font-family: Arial; max-width: 720px; margin: 40px auto;">
        <h1>Booking Status</h1>
        <p><strong>Status:</strong> {status_text}</p>
        <p><strong>Expires at (UTC):</strong> {expires_at}</p>

        <h2>Progress</h2>
        <ul>{items}</ul>

        <p>
          <button onclick="location.reload()">Refresh</button>
        </p>

        <hr style="margin: 30px 0;">
        <p><small>Booking ID: {booking_id}</small></p>
        <p>
          <small>
            <a href="/ui/book">New booking</a> |
            <a href="/ui/me">Me</a> |
            <a href="/ui/logout">Logout</a>
          </small>
        </p>
      </body>
    </html>
    """

# =====================================================
# Root — send people to login
# =====================================================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
      <head><title>Tabel</title></head>
      <body style="font-family: Arial; max-width: 720px; margin: 40px auto;">
        <h1>Tabel</h1>
        <p>This is the Phase 3 minimal UI.</p>
        <p><a href="/ui/login">Go to Login</a></p>
        <p><small><a href="/docs">Swagger</a></small></p>
      </body>
    </html>
    """
