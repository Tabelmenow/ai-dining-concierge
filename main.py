from fastapi import FastAPI, HTTPException, Query, Form, Request, Header
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel
import uuid
import os
import re
import time
import json
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

if not SUPABASE_URL or not SUPABASE_KEY:
    supabase = None
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

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

# Basic rate limiting (in-memory)
BOOK_RATE_LIMIT_MAX = int(os.environ.get("BOOK_RATE_LIMIT_MAX", "10"))
BOOK_RATE_LIMIT_WINDOW_SEC = int(os.environ.get("BOOK_RATE_LIMIT_WINDOW_SEC", "300"))

CALL_RATE_LIMIT_MAX = int(os.environ.get("CALL_RATE_LIMIT_MAX", "6"))
CALL_RATE_LIMIT_WINDOW_SEC = int(os.environ.get("CALL_RATE_LIMIT_WINDOW_SEC", "300"))

_rate_buckets: dict[str, list[float]] = {}

# =====================================================
# Phase 2 Step 8: Admin lookup + observability
# =====================================================
ADMIN_TOKEN = (os.environ.get("ADMIN_TOKEN") or "").strip()
ADMIN_TOKEN_HEADER = "X-Admin-Token"  # send in header OR query param token


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
    # Keep Render logs grep-able
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
    # AI suggestion only. Not confirmation.
    return {
        "recommended_action": "try_digital_first",
        "reason": "Try digital first; call only if digital fails.",
        "confidence": "medium"
    }


def try_digital_booking(req: dict) -> dict:
    # Simulates digital attempt (for now).
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
# Polite deterministic call script
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

    opening = "Hi there. Quick booking request, please."
    request_line = f"Could I please book a table for {party} on {date} around {time_str}, under the name {name}?"
    fallback_line = f"If {time_str} isn’t available, anything between {earliest} and {latest} would work."
    notes_line = f"One note: {notes}" if notes.strip() else ""
    proof_line = "If you can confirm it, what time is it booked for and what name should I put it under? Any reference number?"
    close = "Thanks very much. Appreciate it."

    return {
        "restaurant_name": restaurant_name,
        "opening": opening,
        "request": request_line,
        "fallback": fallback_line,
        "notes": notes_line,
        "proof_request": proof_line,
        "close": close,
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
    restaurant_phone: str = ""  # E.164 preferred


# =====================================================
# Core API
# =====================================================
@app.get("/health")
def health():
    return {
        "ok": True,
        "time": now_iso(),
        "supabase_configured": bool(supabase),
        "dry_run_calls": DRY_RUN_CALLS,
        "allow_real_restaurant_calls": ALLOW_REAL_RESTAURANT_CALLS,
        "require_operator_arm_for_call": REQUIRE_OPERATOR_ARM_FOR_CALL,
        "admin_token_configured": bool(ADMIN_TOKEN),
    }


@app.post("/book")
def book(req: BookingRequest, request: Request):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    ip = get_client_ip(request)
    rate_limit_or_429("book", ip, BOOK_RATE_LIMIT_MAX, BOOK_RATE_LIMIT_WINDOW_SEC)

    booking_id = str(uuid.uuid4())
    req_data = req.model_dump()

    strategy = ai_suggest_strategy(req_data)
    digital_result = try_digital_booking(req_data)

    call_allowed = should_call_restaurant({
        "request": req_data,
        "strategy": strategy,
        "digital_attempt": digital_result
    })

    booking_data = {
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

    log_json("booking_created", {"booking_id": booking_id, "ip": ip, "call_allowed": call_allowed})
    return {
        "booking_id": booking_id,
        "status": booking_data["status"],
        "strategy": booking_data["strategy"],
        "digital_attempt": booking_data["digital_attempt"],
        "created_at": booking_data["created_at"],
        "call_allowed": booking_data["call_allowed"],
        "expires_at": booking_data["expires_at"]
    }


@app.get("/status/{booking_id}")
def status(booking_id: str):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    result = supabase.table("bookings").select("data").eq("id", booking_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Booking not found")
    return result.data[0]["data"]


@app.get("/timeline/{booking_id}")
def timeline(booking_id: str):
    booking_data = status(booking_id)
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
            outcome = d.get("outcome", "unknown")
            timeline_steps.append({"step": f"Call outcome record_
