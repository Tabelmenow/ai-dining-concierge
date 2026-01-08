from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    Form,
    Header,
    Depends
)
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
import uuid
import os
import re
import jwt
from twilio.rest import Client
from supabase import create_client
from urllib.parse import urlparse
from datetime import datetime, timezone, timedelta


app = FastAPI()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET")

if not SUPABASE_URL or not SUPABASE_KEY:
    supabase = None
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# =====================================================
# Phase 3 · Step 1 — Supabase Auth (Identity)
# =====================================================
def get_current_user_id(
    authorization: str = Header(None),
) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    token = authorization.replace("Bearer ", "").strip()

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
# Config
# =====================================================
MAX_CALL_ATTEMPTS = 2
CALL_COOLDOWN_MINUTES = 10
BOOKING_TIMEOUT_MINUTES = 60
ALLOW_REAL_RESTAURANT_CALLS = os.environ.get(
    "ALLOW_REAL_RESTAURANT_CALLS", "false"
).lower() == "true"


# =====================================================
# Helpers
# =====================================================
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso(dt_str: str) -> datetime:
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))


def compute_expires_at(created_at_iso: str) -> str:
    return (parse_iso(created_at_iso) + timedelta(minutes=BOOKING_TIMEOUT_MINUTES)).isoformat()


def is_expired(booking_data: dict) -> bool:
    exp = booking_data.get("expires_at")
    return bool(exp and datetime.now(timezone.utc) >= parse_iso(exp))


def minutes_since(iso_time: str) -> float:
    return (datetime.now(timezone.utc) - parse_iso(iso_time)).total_seconds() / 60


def can_call_again(booking_data: dict):
    if booking_data.get("call_attempts", 0) >= MAX_CALL_ATTEMPTS:
        return False, "Max call attempts reached"

    last = booking_data.get("last_call_at")
    if last:
        mins = minutes_since(last)
        if mins < CALL_COOLDOWN_MINUTES:
            return False, f"Cooldown active ({int(CALL_COOLDOWN_MINUTES - mins)} mins)"

    return True, "OK"


def log_event(booking_data: dict, event_type: str, details: dict | None = None):
    booking_data.setdefault("events", []).append({
        "type": event_type,
        "time": now_iso(),
        "details": details or {}
    })
    booking_data["last_updated_at"] = now_iso()


# =====================================================
# Booking Logic
# =====================================================
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
# Core API
# =====================================================
@app.post("/book")
def book(
    req: BookingRequest,
    user_id: str = Depends(get_current_user_id),
):
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
        "call": None,
        "call_outcome": None,
        "confirmation": None,
        "call_attempts": 0,
        "last_call_at": None,
        "created_at": now_iso(),
        "expires_at": None,
        "final_reason": None,
        "events": [],
    }

    booking_data["expires_at"] = compute_expires_at(booking_data["created_at"])
    log_event(booking_data, "booking_created")

    supabase.table("bookings").insert({
        "id": booking_id,
        "data": booking_data
    }).execute()

    return {
        "booking_id": booking_id,
        "status": "pending",
        "expires_at": booking_data["expires_at"],
    }


def load_booking(booking_id: str, user_id: str) -> dict:
    result = supabase.table("bookings").select("data").eq("id", booking_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Booking not found")

    booking_data = result.data[0]["data"]
    if booking_data.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not authorised")

    return booking_data


@app.get("/status/{booking_id}")
def status(
    booking_id: str,
    user_id: str = Depends(get_current_user_id),
):
    return load_booking(booking_id, user_id)


@app.get("/timeline/{booking_id}")
def timeline(
    booking_id: str,
    user_id: str = Depends(get_current_user_id),
):
    booking = load_booking(booking_id, user_id)
    return {
        "status": booking["status"],
        "expires_at": booking["expires_at"],
        "timeline": booking.get("events", []),
    }


# =====================================================
# UI (unchanged, now identity-safe)
# =====================================================
@app.post("/ui/book")
def ui_book(
    name: str = Form(...),
    city: str = Form(...),
    date: str = Form(...),
    time: str = Form(...),
    party_size: int = Form(...),
    time_window_minutes: int = Form(30),
    notes: str = Form(""),
    restaurant_name: str = Form(""),
    restaurant_phone: str = Form(""),
    authorization: str = Header(None),
):
    user_id = get_current_user_id(authorization)

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
    return RedirectResponse(
        url=f"/ui/status/{result['booking_id']}",
        status_code=303
    )


@app.get("/", response_class=HTMLResponse)
def home():
    return "<h1>Tabel</h1><p>Use the app client to authenticate.</p>"
