from fastapi import FastAPI, HTTPException, Query, Form, Request, Header
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel
import uuid
import os
import re
import time
import json
import hmac
import hashlib
import base64
import urllib.parse
from twilio.rest import Client
from supabase import create_client
from urllib.parse import urlparse, quote
from datetime import datetime, timezone, timedelta


app = FastAPI()

# =====================================================
# Runtime toggles
# =====================================================
DRY_RUN_CALLS = os.environ.get("DRY_RUN_CALLS", "false").lower() == "true"

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

SESSION_SECRET = (os.environ.get("SESSION_SECRET") or "").encode("utf-8")
SESSION_COOKIE_NAME = "tabel_session"

if not SUPABASE_URL or not SUPABASE_KEY:
    supabase = None
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Separate client for auth flows (anon key). If missing, login will fail clearly.
supabase_auth = create_client(SUPABASE_URL, SUPABASE_ANON_KEY) if (SUPABASE_URL and SUPABASE_ANON_KEY) else None

# =====================================================
# Phase 2 Step 4: Retry + timeout config
# =====================================================
MAX_CALL_ATTEMPTS = 2
CALL_COOLDOWN_MINUTES = 10
BOOKING_TIMEOUT_MINUTES = 60

# =====================================================
# Phase 2 Step 5: Safe switch for real restaurant calling
# =====================================================
ALLOW_REAL_RESTAURANT_CALLS = os.environ.get("ALLOW_REAL_RESTAURANT_CALLS", "false").lower() == "true"

# =====================================================
# Phase 2 Step 7: Production safety + operator controls
# =====================================================
REQUIRE_OPERATOR_ARM_FOR_CALL = True

BOOK_RATE_LIMIT_MAX = int(os.environ.get("BOOK_RATE_LIMIT_MAX", "10"))
BOOK_RATE_LIMIT_WINDOW_SEC = int(os.environ.get("BOOK_RATE_LIMIT_WINDOW_SEC", "300"))

CALL_RATE_LIMIT_MAX = int(os.environ.get("CALL_RATE_LIMIT_MAX", "6"))
CALL_RATE_LIMIT_WINDOW_SEC = int(os.environ.get("CALL_RATE_LIMIT_WINDOW_SEC", "300"))

_rate_buckets: dict[str, list[float]] = {}

# =====================================================
# Phase 2 Step 8: Admin lookup + observability
# =====================================================
ADMIN_TOKEN = (os.environ.get("ADMIN_TOKEN") or "").strip()
ADMIN_TOKEN_HEADER = "X-Admin-Token"

# =====================================================
# Helpers
# =====================================================
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def parse_iso(dt_str: str) -> datetime:
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))

def compute_expires_at(created_at_iso: str) -> str:
    created = parse_iso(created_at_iso)
    return (created + timedelta(minutes=BOOKING_TIMEOUT_MINUTES)).isoformat()

def is_expired(booking_data: dict) -> bool:
    expires_at = booking_data.get("expires_at")
    if not expires_at:
        return False
    return datetime.now(timezone.utc) >= parse_iso(expires_at)

def minutes_since(iso_time: str) -> float:
    t = parse_iso(iso_time)
    diff = datetime.now(timezone.utc) - t
    return diff.total_seconds() / 60.0

def can_call_again(booking_data: dict) -> tuple[bool, str]:
    attempts = booking_data.get("call_attempts", 0)
    if attempts >= MAX_CALL_ATTEMPTS:
        return False, f"Max call attempts reached ({MAX_CALL_ATTEMPTS})"
    last_call_at = booking_data.get("last_call_at")
    if last_call_at:
        mins = minutes_since(last_call_at)
        if mins < CALL_COOLDOWN_MINUTES:
            wait = int(CALL_COOLDOWN_MINUTES - mins)
            return False, f"Cooldown active. Try again in ~{wait} minutes."
    return True, "OK"

def log_json(event: str, payload: dict) -> None:
    try:
        print(json.dumps({"event": event, "time": now_iso(), **payload}, ensure_ascii=False))
    except Exception:
        print(f"[{now_iso()}] {event} {payload}")

def log_event(booking_data: dict, event_type: str, details: dict | None = None) -> None:
    booking_data.setdefault("events", [])
    booking_data["events"].append({
        "type": event_type,
        "time": now_iso(),
        "details": details or {}
    })
    booking_data["last_updated_at"] = now_iso()

def ai_suggest_strategy(_booking: dict) -> dict:
    return {
        "recommended_action": "try_digital_first",
        "reason": "Try digital first; call only if digital fails.",
        "confidence": "medium"
    }

def try_digital_booking(req: dict) -> dict:
    party_size = req["party_size"]
    if party_size <= 4:
        return {"success": False, "reason": "No digital availability found"}
    return {"success": False, "reason": "Party size too large for digital"}

def should_call_restaurant(context: dict) -> bool:
    if context["digital_attempt"]["success"] is True:
        return False
    if context["strategy"]["recommended_action"] != "try_digital_first":
        return False
    if context["request"]["party_size"] > 8:
        return False
    return True

def is_valid_phone_e164(phone: str) -> bool:
    if not phone:
        return False
    return bool(re.fullmatch(r"\+[1-9]\d{7,14}", phone.strip()))

def get_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def _rate_key(scope: str, ip: str) -> str:
    return f"{scope}:{ip}"

def rate_limit_or_429(scope: str, ip: str, max_events: int, window_sec: int) -> None:
    now = time.time()
    key = _rate_key(scope, ip)
    bucket = _rate_buckets.setdefault(key, [])
    bucket[:] = [t for t in bucket if (now - t) <= window_sec]
    if len(bucket) >= max_events:
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Please wait and try again. (limit={max_events} per {window_sec}s)"
        )
    bucket.append(now)

def terminal_status(status: str) -> bool:
    return status in ("confirmed", "failed", "expired", "cancelled")

def ensure_not_terminal(booking_data: dict) -> None:
    if terminal_status(booking_data.get("status", "")):
        raise HTTPException(status_code=400, detail=f"Booking is already {booking_data.get('status')}.")

def ui_redirect_with_msg(url: str, msg: str) -> RedirectResponse:
    return RedirectResponse(url=f"{url}?msg={quote(msg)}", status_code=303)

def require_admin(token_q: str | None, x_admin_token: str | None) -> None:
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=500, detail="ADMIN_TOKEN is not configured on the server.")
    provided = (token_q or "").strip() or (x_admin_token or "").strip()
    if not provided or provided != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Admin access denied (missing or invalid token).")

# =====================================================
# Session cookies (Option 1)
# We store the Supabase access_token inside an HttpOnly cookie,
# signed so it can’t be tampered with.
# =====================================================
def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("utf-8").rstrip("=")

def _b64url_decode(s: str) -> bytes:
    pad = "=" * ((4 - (len(s) % 4)) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("utf-8"))

def sign_token(token: str) -> str:
    if not SESSION_SECRET:
        raise HTTPException(status_code=500, detail="SESSION_SECRET is not set.")
    mac = hmac.new(SESSION_SECRET, token.encode("utf-8"), hashlib.sha256).digest()
    return f"{_b64url(token.encode('utf-8'))}.{_b64url(mac)}"

def verify_signed_token(signed: str) -> str | None:
    try:
        raw_b64, mac_b64 = signed.split(".", 1)
        token = _b64url_decode(raw_b64).decode("utf-8")
        mac = _b64url_decode(mac_b64)
        expected = hmac.new(SESSION_SECRET, token.encode("utf-8"), hashlib.sha256).digest()
        if hmac.compare_digest(mac, expected):
            return token
        return None
    except Exception:
        return None

def get_access_token_from_cookie(request: Request) -> str | None:
    signed = request.cookies.get(SESSION_COOKIE_NAME)
    if not signed:
        return None
    return verify_signed_token(signed)

def require_login(request: Request) -> dict:
    """
    Returns supabase user object dict: {id, email, ...}
    """
    if not supabase_auth:
        raise HTTPException(status_code=500, detail="SUPABASE_ANON_KEY not configured (auth disabled).")

    token = get_access_token_from_cookie(request)
    if not token:
        raise HTTPException(status_code=401, detail="Not logged in. Please log in first.")

    # Validate token with Supabase
    try:
        res = supabase_auth.auth.get_user(token)
        user = getattr(res, "user", None) or (res.get("user") if isinstance(res, dict) else None)
        if not user:
            raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
        # python client user may be an object; normalize
        if hasattr(user, "id"):
            return {"id": user.id, "email": getattr(user, "email", None)}
        return {"id": user.get("id"), "email": user.get("email")}
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Session invalid. Please log in again.")

def assert_booking_owner(booking_data: dict, user_id: str) -> None:
    owner = booking_data.get("user_id")
    if not owner:
        # legacy bookings created pre-auth
        raise HTTPException(status_code=403, detail="This booking is not linked to a user. Create a new booking.")
    if owner != user_id:
        raise HTTPException(status_code=403, detail="You do not have access to this booking.")

# =====================================================
# Twilio calling
# =====================================================
def make_call(to_number: str) -> str:
    required_vars = ["TWILIO_SID", "TWILIO_AUTH", "TWILIO_NUMBER"]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing environment variables: {', '.join(missing)}")

    if not is_valid_phone_e164(to_number):
        raise HTTPException(status_code=400, detail="Invalid phone number format. Use E.164 like +6421...")

    client = Client(os.environ["TWILIO_SID"], os.environ["TWILIO_AUTH"])
    call = client.calls.create(
        to=to_number,
        from_=os.environ["TWILIO_NUMBER"],
        url="http://demo.twilio.com/docs/voice.xml"
    )
    return call.sid

def resolve_call_destination(booking_data: dict) -> tuple[str, str]:
    req = booking_data.get("request") or {}
    restaurant_phone = (req.get("restaurant_phone") or "").strip()

    if ALLOW_REAL_RESTAURANT_CALLS and is_valid_phone_e164(restaurant_phone):
        return restaurant_phone, "restaurant"

    founder = (os.environ.get("FOUNDER_PHONE") or "").strip()
    if not founder:
        raise HTTPException(status_code=500, detail="Missing environment variable: FOUNDER_PHONE")
    return founder, "founder_safe_default"

# =====================================================
# Call script
# =====================================================
def compute_time_window(time_str: str, window_minutes: int):
    t = datetime.strptime(time_str, "%H:%M")
    earliest = (t - timedelta(minutes=window_minutes)).strftime("%H:%M")
    latest = (t + timedelta(minutes=window_minutes)).strftime("%H:%M")
    return earliest, latest

def build_call_script(booking_data: dict, restaurant_name: str = "the restaurant") -> dict:
    req = booking_data["request"]
    name = req["name"]
    party = req["party_size"]
    date = req["date"]
    time_str = req["time"]
    window = req.get("time_window_minutes", 30)
    notes = req.get("notes", "")

    earliest, latest = compute_time_window(time_str, window)

    return {
        "restaurant_name": restaurant_name,
        "opening": "Hi there. Quick booking request, please.",
        "request": f"Could I please book a table for {party} on {date} around {time_str}, under the name {name}?",
        "fallback": f"If {time_str} isn’t available, anything between {earliest} and {latest} would work.",
        "notes": f"One note: {notes}" if notes.strip() else "",
        "proof_request": "If you can confirm it, what time is it booked for and what name should I put it under? Any reference number?",
        "close": "Thanks very much. Appreciate it.",
    }

def compute_next_action(booking_data: dict) -> str:
    status = booking_data.get("status", "unknown")
    call_allowed = booking_data.get("call_allowed") is True
    call_obj = booking_data.get("call") or {}
    call_sid = call_obj.get("call_sid")
    outcome_obj = booking_data.get("call_outcome") or {}
    outcome = outcome_obj.get("outcome")

    if status in ("confirmed", "failed", "expired", "cancelled"):
        return "No further action needed."
    if is_expired(booking_data):
        return "This booking is expired. (You can create a new one.)"
    if status == "needs_user_decision":
        return "Restaurant offered an alternative. Record details and decide what to do next."
    if status == "awaiting_confirmation":
        return "Call outcome is CONFIRMED. Next: confirm the booking (with proof)."
    if REQUIRE_OPERATOR_ARM_FOR_CALL and not booking_data.get("operator_call_armed"):
        if call_allowed:
            return "Next: operator must ARM the call (then you can place the call)."
        return "Call not allowed for this booking."
    if call_allowed and not call_sid:
        return "Next: place a call (and read the call script)."
    if call_sid and outcome != "CONFIRMED":
        return "Next: record call outcome (NO_ANSWER / DECLINED / OFFERED_ALTERNATIVE / CONFIRMED)."
    if call_sid and outcome == "CONFIRMED":
        return "Next: confirm the booking (with proof)."
    return "Next: review details and continue."

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
@app.get("/health")
def health():
    return {
        "ok": True,
        "time": now_iso(),
        "supabase_configured": bool(supabase),
        "supabase_auth_configured": bool(supabase_auth),
        "dry_run_calls": DRY_RUN_CALLS,
        "allow_real_restaurant_calls": ALLOW_REAL_RESTAURANT_CALLS,
        "require_operator_arm_for_call": REQUIRE_OPERATOR_ARM_FOR_CALL,
        "admin_token_configured": bool(ADMIN_TOKEN),
        "session_secret_configured": bool(SESSION_SECRET),
    }

@app.post("/book")
def book(req: BookingRequest, request: Request):
    # API booking creation also requires login now
    user = require_login(request)

    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    ip = get_client_ip(request)
    rate_limit_or_429("book", ip, BOOK_RATE_LIMIT_MAX, BOOK_RATE_LIMIT_WINDOW_SEC)

    booking_id = str(uuid.uuid4())
    req_data = req.model_dump()

    strategy = ai_suggest_strategy(req_data)
    digital_result = try_digital_booking(req_data)
    call_allowed = should_call_restaurant({"request": req_data, "strategy": strategy, "digital_attempt": digital_result})

    booking_data = {
        "user_id": user["id"],  # ownership
        "request": req_data,
        "status": "pending",
        "strategy": strategy,
        "events": [],
        "digital_attempt": digital_result,
        "call_allowed": call_allowed,
        "call": None,
        "call_outcome": None,
        "confirmation": None,
        "created_at": now_iso(),
        "last_updated_at": now_iso(),
        "call_attempts": 0,
        "last_call_at": None,
        "expires_at": None,
        "final_reason": None,
        "operator_call_armed": False,
        "operator_armed_at": None,
    }

    booking_data["expires_at"] = compute_expires_at(booking_data["created_at"])
    log_event(booking_data, "timeout_set", {"expires_at": booking_data["expires_at"]})
    log_event(booking_data, "booking_created", {"city": req_data["city"], "party_size": req_data["party_size"]})
    log_event(booking_data, "strategy_suggested", strategy)
    log_event(booking_data, "digital_attempted", digital_result)
    log_event(booking_data, "call_decision_made", {"call_allowed": call_allowed})
    log_event(booking_data, "operator_arm_required", {"required": REQUIRE_OPERATOR_ARM_FOR_CALL})

    supabase.table("bookings").insert({"id": booking_id, "data": booking_data}).execute()
    log_json("booking_created", {"booking_id": booking_id, "user_id": user["id"], "ip": ip, "call_allowed": call_allowed})

    return {"booking_id": booking_id, "status": booking_data["status"], "expires_at": booking_data["expires_at"]}

@app.get("/status/{booking_id}")
def status(booking_id: str, request: Request):
    user = require_login(request)
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    result = supabase.table("bookings").select("data").eq("id", booking_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Booking not found")
    booking_data = result.data[0]["data"]
    assert_booking_owner(booking_data, user["id"])
    return booking_data

@app.get("/timeline/{booking_id}")
def timeline(booking_id: str, request: Request):
    booking_data = status(booking_id, request)
    events = booking_data.get("events", [])
    timeline_steps = []
    for event in events:
        t = event.get("time")
        et = event.get("type")
        d = event.get("details") or {}

        if et == "booking_created":
            timeline_steps.append({"step": "Request received", "time": t})
        elif et == "digital_attempted":
            timeline_steps.append({"step": "Searching digitally", "time": t})
        elif et == "operator_call_armed":
            timeline_steps.append({"step": "Operator armed calling", "time": t})
        elif et == "call_attempt_incremented":
            timeline_steps.append({"step": f"Call attempt #{d.get('call_attempts', '?')} recorded", "time": t})
        elif et == "call_destination_resolved":
            timeline_steps.append({"step": f"Call destination set ({d.get('mode', 'unknown')})", "time": t})
        elif et == "call_script_generated":
            timeline_steps.append({"step": "Preparing call script", "time": t})
        elif et == "call_initiated":
            timeline_steps.append({"step": "Calling to check availability", "time": t})
        elif et == "dry_run_call_skipped":
            timeline_steps.append({"step": "Dry run enabled (no call placed)", "time": t})
        elif et == "call_recorded":
            timeline_steps.append({"step": "Call connected (SID recorded)", "time": t})
        elif et == "call_outcome_recorded":
            timeline_steps.append({"step": f"Call outcome recorded: {d.get('outcome', 'unknown')}", "time": t})
        elif et == "status_changed":
            timeline_steps.append({"step": f"Status changed: {d.get('status', 'unknown')}", "time": t})
        elif et == "confirmation_recorded":
            timeline_steps.append({"step": "Booking confirmed", "time": t})
        elif et == "booking_failed":
            timeline_steps.append({"step": f"Booking failed ({d.get('reason', 'unknown')})", "time": t})
        elif et == "booking_cancelled":
            timeline_steps.append({"step": f"Booking cancelled ({d.get('reason', 'unknown')})", "time": t})
        elif et == "booking_expired":
            timeline_steps.append({"step": "Booking expired (timeout)", "time": t})

    return {
        "booking_id": booking_id,
        "status": booking_data.get("status"),
        "expires_at": booking_data.get("expires_at"),
        "final_reason": booking_data.get("final_reason"),
        "timeline": timeline_steps
    }

@app.post("/arm-call/{booking_id}")
def arm_call(booking_id: str, request: Request):
    booking_data = status(booking_id, request)  # includes ownership check
    ensure_not_terminal(booking_data)

    if is_expired(booking_data):
        booking_data["status"] = "expired"
        booking_data["final_reason"] = "timeout"
        log_event(booking_data, "booking_expired", {"reason": booking_data["final_reason"]})
        supabase.table("bookings").update({"data": booking_data}).eq("id", booking_id).execute()
        raise HTTPException(status_code=400, detail="Booking has expired (timeout)")

    if booking_data.get("call_allowed") is not True:
        raise HTTPException(status_code=400, detail="Call not allowed for this booking (call_allowed is false)")

    booking_data["operator_call_armed"] = True
    booking_data["operator_armed_at"] = now_iso()
    log_event(booking_data, "operator_call_armed", {"armed_at": booking_data["operator_armed_at"]})
    supabase.table("bookings").update({"data": booking_data}).eq("id", booking_id).execute()
    return {"message": "Call armed", "booking_id": booking_id}

@app.post("/cancel/{booking_id}")
def cancel_booking(booking_id: str, request: Request, reason: str = Query("user_cancelled", max_length=80)):
    booking_data = status(booking_id, request)
    if terminal_status(booking_data.get("status", "")):
        return {"message": f"Already {booking_data.get('status')}", "status": booking_data.get("status")}
    booking_data["status"] = "cancelled"
    booking_data["final_reason"] = reason
    log_event(booking_data, "booking_cancelled", {"reason": reason})
    supabase.table("bookings").update({"data": booking_data}).eq("id", booking_id).execute()
    return {"message": "Cancelled", "booking_id": booking_id, "status": booking_data["status"]}

@app.post("/call-test/{booking_id}")
def call_test(booking_id: str, request: Request):
    booking_data = status(booking_id, request)
    ensure_not_terminal(booking_data)

    ip = get_client_ip(request)
    rate_limit_or_429("call", ip, CALL_RATE_LIMIT_MAX, CALL_RATE_LIMIT_WINDOW_SEC)

    if booking_data.get("status") != "pending":
        raise HTTPException(status_code=400, detail=f"Cannot call because status is '{booking_data.get('status')}'")
    if booking_data.get("call_allowed") is not True:
        raise HTTPException(status_code=400, detail="Call not allowed for this booking (call_allowed is false)")
    if REQUIRE_OPERATOR_ARM_FOR_CALL and not booking_data.get("operator_call_armed"):
        raise HTTPException(status_code=400, detail="Operator must ARM the call before calling.")
    if is_expired(booking_data):
        booking_data["status"] = "expired"
        booking_data["final_reason"] = "timeout"
        log_event(booking_data, "booking_expired", {"reason": booking_data["final_reason"]})
        supabase.table("bookings").update({"data": booking_data}).eq("id", booking_id).execute()
        raise HTTPException(status_code=400, detail="Booking has expired (timeout)")

    ok, msg = can_call_again(booking_data)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    req = booking_data.get("request") or {}
    restaurant_name = (req.get("restaurant_name") or "the restaurant").strip() or "the restaurant"
    script = build_call_script(booking_data, restaurant_name=restaurant_name)
    log_event(booking_data, "call_script_generated", {"script": script})

    booking_data["call_attempts"] = booking_data.get("call_attempts", 0) + 1
    booking_data["last_call_at"] = now_iso()
    log_event(booking_data, "call_attempt_incremented", {"call_attempts": booking_data["call_attempts"], "last_call_at": booking_data["last_call_at"]})

    to_number, mode = resolve_call_destination(booking_data)
    log_event(booking_data, "call_destination_resolved", {"mode": mode, "to": to_number})
    log_event(booking_data, "call_initiated", {"mode": "twilio"})

    if DRY_RUN_CALLS:
        log_event(booking_data, "dry_run_call_skipped", {"to": to_number, "to_mode": mode})
        supabase.table("bookings").update({"data": booking_data}).eq("id", booking_id).execute()
        return {"message": "Dry run: call skipped (no Twilio call placed)", "to_mode": mode}

    sid = make_call(to_number)
    booking_data["call"] = {"call_sid": sid, "called_at": now_iso(), "to": to_number, "to_mode": mode}
    log_event(booking_data, "call_recorded", {"call_sid": sid})
    supabase.table("bookings").update({"data": booking_data}).eq("id", booking_id).execute()
    return {"message": "Call placed and recorded", "call_sid": sid, "to_mode": mode}

@app.post("/call-outcome/{booking_id}")
def call_outcome(
    booking_id: str,
    request: Request,
    outcome: str = Query(..., pattern="^(NO_ANSWER|DECLINED|OFFERED_ALTERNATIVE|CONFIRMED)$"),
    notes: str = Query("", max_length=300),
    confirmed_time: str = Query(""),
    reference: str = Query("", max_length=100),
):
    booking_data = status(booking_id, request)
    ensure_not_terminal(booking_data)

    booking_data["call_outcome"] = {"outcome": outcome, "notes": notes, "confirmed_time": confirmed_time, "reference": reference, "recorded_at": now_iso()}
    log_event(booking_data, "call_outcome_recorded", booking_data["call_outcome"])

    if outcome == "CONFIRMED":
        booking_data["status"] = "awaiting_confirmation"
        log_event(booking_data, "status_changed", {"status": booking_data["status"]})
    elif outcome == "DECLINED":
        booking_data["status"] = "failed"
        booking_data["final_reason"] = "declined"
        log_event(booking_data, "booking_failed", {"reason": booking_data["final_reason"]})
    elif outcome == "NO_ANSWER":
        log_event(booking_data, "retry_possible", {"reason": "no_answer"})
    elif outcome == "OFFERED_ALTERNATIVE":
        booking_data["status"] = "needs_user_decision"
        log_event(booking_data, "status_changed", {"status": booking_data["status"]})

    supabase.table("bookings").update({"data": booking_data}).eq("id", booking_id).execute()
    return {"message": "Call outcome saved", "booking_id": booking_id, "outcome": outcome, "status": booking_data.get("status")}

@app.post("/confirm/{booking_id}")
def confirm_booking(
    booking_id: str,
    request: Request,
    proof: str = Query(..., min_length=10),
    confirmed_by: str = Query(..., min_length=2),
    method: str = Query(..., pattern="^(phone|digital|in_person)$"),
):
    booking_data = status(booking_id, request)

    if booking_data.get("status") == "confirmed":
        return {"message": "Already confirmed", "confirmation": booking_data.get("confirmation")}

    if terminal_status(booking_data.get("status", "")):
        raise HTTPException(status_code=400, detail=f"Cannot confirm because status is '{booking_data.get('status')}'")

    if is_expired(booking_data):
        booking_data["status"] = "expired"
        booking_data["final_reason"] = "timeout"
        log_event(booking_data, "booking_expired", {"reason": booking_data["final_reason"]})
        supabase.table("bookings").update({"data": booking_data}).eq("id", booking_id).execute()
        raise HTTPException(status_code=400, detail="Booking has expired (timeout)")

    if booking_data.get("status") not in ("pending", "awaiting_confirmation"):
        raise HTTPException(status_code=400, detail=f"Cannot confirm because status is '{booking_data.get('status')}'")

    if method == "phone":
        call_obj = booking_data.get("call") or {}
        if not call_obj.get("call_sid"):
            raise HTTPException(status_code=400, detail="Cannot confirm by phone without a recorded call SID")
        outcome_obj = booking_data.get("call_outcome") or {}
        if outcome_obj.get("outcome") != "CONFIRMED":
            raise HTTPException(status_code=400, detail="Cannot confirm by phone unless call outcome is CONFIRMED")

    booking_data["status"] = "confirmed"
    booking_data["confirmation"] = {"proof": proof, "confirmed_by": confirmed_by, "method": method, "confirmed_at": now_iso()}
    log_event(booking_data, "confirmation_recorded", {"method": method, "confirmed_by": confirmed_by})
    supabase.table("bookings").update({"data": booking_data}).eq("id", booking_id).execute()
    return {"message": "Confirmed", "booking_id": booking_id, "status": booking_data["status"]}

# =====================================================
# Admin API
# =====================================================
@app.get("/admin/booking/{booking_id}")
def admin_get_booking(
    booking_id: str,
    token: str | None = None,
    x_admin_token: str | None = Header(default=None, alias=ADMIN_TOKEN_HEADER),
):
    require_admin(token, x_admin_token)
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    result = supabase.table("bookings").select("data").eq("id", booking_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Booking not found")
    return {"booking_id": booking_id, "data": result.data[0]["data"]}

# =====================================================
# Auth UI (Phase 3.1)
# =====================================================
@app.get("/ui/login", response_class=HTMLResponse)
def ui_login(msg: str = ""):
    banner = f"<p style='background:#e7f3ff;padding:10px;border-radius:8px;border:1px solid #b3d7ff;'><strong>Info:</strong> {msg}</p>" if msg else ""
    return f"""
    <html>
      <head><title>Tabel Login</title></head>
      <body style="font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto;">
        <h1>Log in to Tabel</h1>
        {banner}
        <p>Enter your email. We will send you a magic link to log in.</p>
        <form action="/ui/login" method="post">
          <label>Email</label><br>
          <input name="email" type="email" required style="width: 100%; padding: 8px;"><br><br>
          <button type="submit" style="padding: 10px 16px;">Send magic link</button>
          <a href="/" style="margin-left: 12px;">Home</a>
        </form>
      </body>
    </html>
    """

@app.post("/ui/login")
def ui_login_submit(email: str = Form(...)):
    if not supabase_auth:
        raise HTTPException(status_code=500, detail="SUPABASE_ANON_KEY not configured (auth disabled).")

    # Send magic link that returns to /auth/callback
    redirect_to = "https://ai-dining-concierge.onrender.com/auth/callback"
    supabase_auth.auth.sign_in_with_otp({"email": email, "options": {"email_redirect_to": redirect_to}})

    return ui_redirect_with_msg("/ui/login", "Magic link sent. Check your email and click the link to finish login.")

@app.get("/auth/callback")
def auth_callback(request: Request):
    """
    Supabase magic link lands here with tokens in the URL fragment in many setups.
    Some clients send as query params. We'll support query params:
      ?access_token=...&refresh_token=...&type=magiclink
    If tokens are not present, we show instructions.
    """
    params = dict(request.query_params)
    access_token = params.get("access_token")

    if not access_token:
        # Many Supabase flows return tokens in URL fragment (#...) which the server cannot see.
        # So we provide a tiny page that converts fragment -> query params and reloads.
        return HTMLResponse("""
        <html>
          <head><title>Completing login...</title></head>
          <body style="font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto;">
            <h1>Completing login...</h1>
            <p>If you see this for more than a second, click the button below.</p>
            <button onclick="
              const hash = window.location.hash.startsWith('#') ? window.location.hash.slice(1) : window.location.hash;
              const q = new URLSearchParams(hash);
              if (q.get('access_token')) {
                window.location.href = '/auth/callback?' + q.toString();
              } else {
                document.getElementById('msg').innerText = 'No token found in link. Please request a new magic link.';
              }
            ">Finish login</button>
            <p id="msg" style="margin-top:12px;color:#555;"></p>
          </body>
        </html>
        """)

    signed = sign_token(access_token)

    resp = RedirectResponse(url="/ui/book", status_code=303)
    resp.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=signed,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 7,  # 7 days
    )
    return resp

@app.post("/ui/logout")
def ui_logout():
    resp = RedirectResponse(url="/ui/login?msg=" + quote("Logged out."), status_code=303)
    resp.delete_cookie(SESSION_COOKIE_NAME)
    return resp

# =====================================================
# Debug + UI
# =====================================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if request.url.path.startswith("/ui/") or request.url.path.startswith("/auth/"):
        # Redirect UI flows to login when not authenticated
        if exc.status_code == 401:
            return ui_redirect_with_msg("/ui/login", "Please log in first.")
        msg = exc.detail if isinstance(exc.detail, str) else "Something went wrong."
        return ui_redirect_with_msg("/ui/login", msg)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.get("/debug/env")
def debug_env():
    supa_url = os.environ.get("SUPABASE_URL", "")
    parsed = urlparse(supa_url) if supa_url else None
    return {
        "SUPABASE_URL_set": bool(supa_url),
        "SUPABASE_URL_scheme": parsed.scheme if parsed else None,
        "SUPABASE_URL_netloc": parsed.netloc if parsed else None,
        "SUPABASE_KEY_set": bool(os.environ.get("SUPABASE_KEY")),
        "SUPABASE_ANON_KEY_set": bool(os.environ.get("SUPABASE_ANON_KEY")),
        "SESSION_SECRET_set": bool(os.environ.get("SESSION_SECRET")),
        "ADMIN_TOKEN_set": bool(os.environ.get("ADMIN_TOKEN")),
        "DRY_RUN_CALLS": DRY_RUN_CALLS,
        "ALLOW_REAL_RESTAURANT_CALLS": ALLOW_REAL_RESTAURANT_CALLS,
    }

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
      <head><title>Tabel</title></head>
      <body style="font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto;">
        <h1>Tabel</h1>
        <p>Now includes login (magic link) + per-user booking access.</p>
        <ul>
          <li><a href="/ui/login">Login</a></li>
          <li><a href="/ui/book">New booking</a> (requires login)</li>
          <li><a href="/health">Health</a></li>
          <li><a href="/docs">Swagger</a></li>
        </ul>
        <form action="/ui/logout" method="post">
          <button type="submit" style="padding:8px 12px;">Logout</button>
        </form>
      </body>
    </html>
    """

@app.get("/ui/book", response_class=HTMLResponse)
def ui_book_form(request: Request, msg: str = ""):
    require_login(request)
    note = f"<p style='background:#fff3cd;padding:10px;border-radius:8px;border:1px solid #ffeeba;'><strong>Note:</strong> {msg}</p>" if msg else ""
    return f"""
    <html>
      <head><title>New Booking</title></head>
      <body style="font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto;">
        <h1>New booking</h1>
        {note}

        <form action="/ui/book" method="post">
          <h3>Restaurant (optional for now)</h3>
          <label>Restaurant name</label><br>
          <input name="restaurant_name" style="width: 100%; padding: 8px;"><br><br>

          <label>Restaurant phone (E.164, e.g. +64211234567)</label><br>
          <input name="restaurant_phone" style="width: 100%; padding: 8px;"><br><br>

          <hr style="margin: 20px 0;">

          <label>Your name</label><br>
          <input name="name" required style="width: 100%; padding: 8px;"><br><br>

          <label>City</label><br>
          <input name="city" required style="width: 100%; padding: 8px;"><br><br>

          <label>Date</label><br>
          <input type="date" name="date" required style="width: 100%; padding: 8px;"><br><br>

          <label>Time</label><br>
          <input type="time" name="time_str" required style="width: 100%; padding: 8px;"><br><br>

          <label>Party size</label><br>
          <input type="number" name="party_size" min="1" required style="width: 100%; padding: 8px;"><br><br>

          <label>Flex window (minutes)</label><br>
          <input type="number" name="time_window_minutes" min="0" value="30" style="width: 100%; padding: 8px;"><br><br>

          <label>Notes (optional)</label><br>
          <input name="notes" style="width: 100%; padding: 8px;"><br><br>

          <button type="submit" style="padding: 10px 16px;">Get me a table</button>
        </form>

        <hr style="margin: 20px 0;">
        <p><small><a href="/">Home</a></small></p>
        <form action="/ui/logout" method="post">
          <button type="submit" style="padding:8px 12px;">Logout</button>
        </form>
      </body>
    </html>
    """

@app.post("/ui/book")
def ui_book(
    request: Request,
    name: str = Form(...),
    city: str = Form(...),
    date: str = Form(...),
    time_str: str = Form(...),
    party_size: int = Form(...),
    time_window_minutes: int = Form(30),
    notes: str = Form(""),
    restaurant_name: str = Form(""),
    restaurant_phone: str = Form(""),
):
    # Create booking via core API logic
    req = BookingRequest(
        name=name,
        city=city,
        date=date,
        time=time_str,
        party_size=party_size,
        time_window_minutes=time_window_minutes,
        notes=notes,
        restaurant_name=restaurant_name,
        restaurant_phone=restaurant_phone,
    )
    result = book(req, request)
    return RedirectResponse(url=f"/ui/status/{result['booking_id']}", status_code=303)

@app.get("/ui/status/{booking_id}", response_class=HTMLResponse)
def ui_status(request: Request, booking_id: str, msg: str = ""):
    booking_data = status(booking_id, request)
    data = timeline(booking_id, request)

    status_text = data.get("status", "unknown")
    steps = data.get("timeline", [])
    expires_at = data.get("expires_at", "")
    final_reason = data.get("final_reason", "")
    next_action = compute_next_action(booking_data)

    call_allowed = booking_data.get("call_allowed") is True
    armed = booking_data.get("operator_call_armed") is True
    terminal = terminal_status(status_text)

    banner = f"<p style='background:#e7f3ff;padding:10px;border-radius:8px;border:1px solid #b3d7ff;'><strong>Info:</strong> {msg}</p>" if msg else ""
    reason_line = f"<p><strong>Final reason:</strong> {final_reason}</p>" if final_reason else ""

    controls = ""
    if not terminal:
        controls += f"""
        <form action="/ui/cancel/{booking_id}" method="post" style="display:inline;">
          <button type="submit" style="padding: 8px 12px; margin-right: 10px;">Cancel booking</button>
        </form>
        """
        if call_allowed and not armed and REQUIRE_OPERATOR_ARM_FOR_CALL:
            controls += f"""
            <form action="/ui/arm-call/{booking_id}" method="post" style="display:inline;">
              <button type="submit" style="padding: 8px 12px; margin-right: 10px;">ARM call (operator)</button>
            </form>
            """
        if call_allowed and (armed or not REQUIRE_OPERATOR_ARM_FOR_CALL):
            controls += f"""
            <form action="/ui/call/{booking_id}" method="post" style="display:inline;">
              <button type="submit" style="padding: 8px 12px; margin-right: 10px;">Place call</button>
            </form>
            """

    html = f"""
    <html>
      <head><title>Booking Status</title></head>
      <body style="font-family: Arial, sans-serif; max-width: 950px; margin: 40px auto;">
        <h1>Booking Status</h1>
        {banner}
        <p><strong>Status:</strong> {status_text}</p>
        <p><strong>Next action:</strong> {next_action}</p>
        <p><strong>Expires at (UTC):</strong> {expires_at}</p>
        {reason_line}

        <div style="padding:12px;background:#f6f6f6;border-radius:10px;">
          <p style="margin-top:0;"><strong>Controls</strong></p>
          {controls if controls else "<p>No actions available.</p>"}
          <p style="margin-bottom:0;">
            <a href="/ui/call-outcome/{booking_id}" style="margin-right: 12px;">Record call outcome</a>
            <a href="/ui/confirm/{booking_id}" style="margin-right: 12px;">Confirm booking</a>
          </p>
        </div>

        <h2>Progress</h2>
        <ul>
    """
    if not steps:
        html += "<li>No events yet. Refresh in a few seconds.</li>"
    else:
        for item in steps:
            html += f"<li>{item['step']}<br><small>{item['time']}</small></li>"

    html += f"""
        </ul>

        <p><button onclick="location.reload()">Refresh</button></p>

        <hr style="margin: 20px 0;">
        <p><small>Booking ID: {booking_id}</small></p>
        <p><small><a href="/ui/book">New booking</a> | <a href="/">Home</a></small></p>
        <form action="/ui/logout" method="post">
          <button type="submit" style="padding:8px 12px;">Logout</button>
        </form>
      </body>
    </html>
    """
    return html

@app.post("/ui/arm-call/{booking_id}")
def ui_arm_call(request: Request, booking_id: str):
    arm_call(booking_id, request)
    return ui_redirect_with_msg(f"/ui/status/{booking_id}", "Call armed. You can now place the call.")

@app.post("/ui/call/{booking_id}")
def ui_call(request: Request, booking_id: str):
    result = call_test(booking_id, request)
    msg = result.get("message", "Call action complete.")
    return ui_redirect_with_msg(f"/ui/status/{booking_id}", msg)

@app.post("/ui/cancel/{booking_id}")
def ui_cancel(request: Request, booking_id: str):
    cancel_booking(booking_id, request, reason="user_cancelled")
    return ui_redirect_with_msg(f"/ui/status/{booking_id}", "Booking cancelled.")

@app.get("/ui/call-outcome/{booking_id}", response_class=HTMLResponse)
def ui_call_outcome_form(request: Request, booking_id: str, msg: str = ""):
    booking_data = status(booking_id, request)
    req = booking_data.get("request") or {}
    rname = (req.get("restaurant_name") or "").strip() or "the restaurant"
    banner = f"<p style='background:#e7f3ff;padding:10px;border-radius:8px;border:1px solid #b3d7ff;'><strong>Info:</strong> {msg}</p>" if msg else ""

    return f"""
    <html>
      <head><title>Call Outcome</title></head>
      <body style="font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto;">
        <h1>Record Call Outcome</h1>
        {banner}
        <p><strong>Booking:</strong> {booking_id}</p>
        <p><strong>Restaurant:</strong> {rname}</p>

        <form action="/ui/call-outcome/{booking_id}" method="post">
          <label>Outcome</label><br>
          <select name="outcome" required style="width: 100%; padding: 8px;">
            <option value="NO_ANSWER">NO_ANSWER</option>
            <option value="DECLINED">DECLINED</option>
            <option value="OFFERED_ALTERNATIVE">OFFERED_ALTERNATIVE</option>
            <option value="CONFIRMED">CONFIRMED</option>
          </select><br><br>

          <label>Notes (optional)</label><br>
          <input name="notes" style="width: 100%; padding: 8px;"><br><br>

          <label>Confirmed time (only if CONFIRMED, e.g. 19:15)</label><br>
          <input name="confirmed_time" style="width: 100%; padding: 8px;"><br><br>

          <label>Reference (optional)</label><br>
          <input name="reference" style="width: 100%; padding: 8px;"><br><br>

          <button type="submit" style="padding: 10px 16px;">Save outcome</button>
          <a href="/ui/status/{booking_id}" style="margin-left: 12px;">Cancel</a>
        </form>
      </body>
    </html>
    """

@app.post("/ui/call-outcome/{booking_id}")
def ui_call_outcome_submit(
    request: Request,
    booking_id: str,
    outcome: str = Form(...),
    notes: str = Form(""),
    confirmed_time: str = Form(""),
    reference: str = Form(""),
):
    call_outcome(
        booking_id=booking_id,
        request=request,
        outcome=outcome,
        notes=notes,
        confirmed_time=confirmed_time,
        reference=reference,
    )
    return ui_redirect_with_msg(f"/ui/status/{booking_id}", "Call outcome saved.")

@app.get("/ui/confirm/{booking_id}", response_class=HTMLResponse)
def ui_confirm_form(request: Request, booking_id: str, msg: str = ""):
    booking_data = status(booking_id, request)
    req = booking_data.get("request") or {}
    rname = (req.get("restaurant_name") or "").strip() or "the restaurant"
    banner = f"<p style='background:#e7f3ff;padding:10px;border-radius:8px;border:1px solid #b3d7ff;'><strong>Info:</strong> {msg}</p>" if msg else ""

    return f"""
    <html>
      <head><title>Confirm Booking</title></head>
      <body style="font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto;">
        <h1>Confirm booking</h1>
        {banner}
        <p><strong>Booking:</strong> {booking_id}</p>
        <p><strong>Restaurant:</strong> {rname}</p>

        <p style="background:#f4f4f4;padding:12px;border-radius:8px;">
          Only confirm if you have real proof.
        </p>

        <form action="/ui/confirm/{booking_id}" method="post">
          <label>Method</label><br>
          <select name="method" required style="width: 100%; padding: 8px;">
            <option value="phone">phone</option>
            <option value="digital">digital</option>
            <option value="in_person">in_person</option>
          </select><br><br>

          <label>Confirmed by</label><br>
          <input name="confirmed_by" required style="width: 100%; padding: 8px;"><br><br>

          <label>Proof</label><br>
          <input name="proof" required style="width: 100%; padding: 8px;"><br><br>

          <button type="submit" style="padding: 10px 16px;">Confirm now</button>
          <a href="/ui/status/{booking_id}" style="margin-left: 12px;">Cancel</a>
        </form>
      </body>
    </html>
    """

@app.post("/ui/confirm/{booking_id}")
def ui_confirm_submit(
    request: Request,
    booking_id: str,
    method: str = Form(...),
    confirmed_by: str = Form(...),
    proof: str = Form(...),
):
    confirm_booking(
        booking_id=booking_id,
        request=request,
        proof=proof,
        confirmed_by=confirmed_by,
        method=method,
    )
    return ui_redirect_with_msg(f"/ui/status/{booking_id}", "Booking confirmed.")
