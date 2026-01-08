from fastapi import FastAPI, HTTPException, Query, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
import uuid
import os
import re
from twilio.rest import Client
from supabase import create_client
from urllib.parse import urlparse
from datetime import datetime, timezone, timedelta


app = FastAPI()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    supabase = None
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# -----------------------------
# Step 4: Retry + timeout config
# -----------------------------
MAX_CALL_ATTEMPTS = 2          # never call more than this per booking
CALL_COOLDOWN_MINUTES = 10     # minimum minutes between calls
BOOKING_TIMEOUT_MINUTES = 60   # booking expires if not confirmed within this time


# -----------------------------
# Step 5: Safe switch for real restaurant calling
# Default: False (still calls founder)
# -----------------------------
ALLOW_REAL_RESTAURANT_CALLS = os.environ.get("ALLOW_REAL_RESTAURANT_CALLS", "false").lower() == "true"


def ai_suggest_strategy(booking: dict) -> dict:
    """
    AI suggestion only.
    This does NOT confirm availability.
    """
    return {
        "recommended_action": "try_digital_first",
        "reason": "Smaller party sizes are more likely to succeed digitally; call only if digital fails.",
        "confidence": "medium"
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso(dt_str: str) -> datetime:
    # Handles "+00:00" ISO strings and also "Z"
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


def log_event(booking_data: dict, event_type: str, details: dict | None = None) -> None:
    """
    Append-only event log. Never delete events.
    Keeps an audit trail of what happened.
    """
    if "events" not in booking_data or booking_data["events"] is None:
        booking_data["events"] = []

    booking_data["events"].append({
        "type": event_type,
        "time": now_iso(),
        "details": details or {}
    })

    booking_data["last_updated_at"] = now_iso()


def try_digital_booking(booking: dict) -> dict:
    """
    Simulates a digital booking attempt.
    Does NOT confirm availability.
    """
    party_size = booking["party_size"]

    if party_size <= 4:
        return {"success": False, "reason": "No digital availability found"}
    else:
        return {"success": False, "reason": "Party size too large for digital"}


def should_call_restaurant(context: dict) -> bool:
    """
    Deterministic decision: should we allow a call step?
    Uses rules, not AI guesses.
    """
    if context["digital_attempt"]["success"] is True:
        return False

    if context["strategy"]["recommended_action"] != "try_digital_first":
        return False

    if context["request"]["party_size"] > 8:
        return False

    return True


def is_valid_phone_e164(phone: str) -> bool:
    """
    Minimal E.164 check: + and 8-15 digits.
    Example: +64211234567
    """
    if not phone:
        return False
    return bool(re.fullmatch(r"\+[1-9]\d{7,14}", phone.strip()))


# -----------------------------
# Twilio: safe calling
# -----------------------------
def make_call(to_number: str) -> str:
    """
    Makes a Twilio call to a number (E.164).
    """
    required_vars = ["TWILIO_SID", "TWILIO_AUTH", "TWILIO_NUMBER"]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing environment variables: {', '.join(missing)}"
        )

    if not is_valid_phone_e164(to_number):
        raise HTTPException(status_code=400, detail="Invalid phone number format. Use E.164 like +6421...")

    client = Client(os.environ["TWILIO_SID"], os.environ["TWILIO_AUTH"])

    call = client.calls.create(
        to=to_number,
        from_=os.environ["TWILIO_NUMBER"],
        url="http://demo.twilio.com/docs/voice.xml"
    )

    return call.sid


def make_test_call() -> str:
    """
    Founder-only test call (unchanged behaviour).
    """
    founder = os.environ.get("FOUNDER_PHONE")
    if not founder:
        raise HTTPException(status_code=500, detail="Missing environment variable: FOUNDER_PHONE")
    return make_call(founder)


def resolve_call_destination(booking_data: dict) -> tuple[str, str]:
    """
    Returns (to_number, mode_string)
    - By default: founder (safe)
    - If ALLOW_REAL_RESTAURANT_CALLS=true AND restaurant_phone present/valid: restaurant
    """
    req = booking_data.get("request") or {}
    restaurant_phone = (req.get("restaurant_phone") or "").strip()

    if ALLOW_REAL_RESTAURANT_CALLS and is_valid_phone_e164(restaurant_phone):
        return restaurant_phone, "restaurant"

    founder = os.environ.get("FOUNDER_PHONE") or ""
    if not founder:
        raise HTTPException(status_code=500, detail="Missing environment variable: FOUNDER_PHONE")
    return founder, "founder_safe_default"


# -----------------------------
# Step 3: Polite deterministic call script
# -----------------------------
def compute_time_window(time_str: str, window_minutes: int):
    """
    Given '19:00' and 30 minutes, returns: earliest '18:30', latest '19:30'
    """
    t = datetime.strptime(time_str, "%H:%M")
    earliest = (t - timedelta(minutes=window_minutes)).strftime("%H:%M")
    latest = (t + timedelta(minutes=window_minutes)).strftime("%H:%M")
    return earliest, latest


def build_call_script(booking_data: dict, restaurant_name: str = "the restaurant") -> dict:
    """
    Deterministic script. No AI. No claims of availability.
    Returns a script object you can display and log.
    """
    req = booking_data["request"]
    name = req["name"]
    party = req["party_size"]
    date = req["date"]
    time = req["time"]
    window = req.get("time_window_minutes", 30)
    notes = req.get("notes", "")

    earliest, latest = compute_time_window(time, window)

    opening = "Hi there. Quick booking request, please."
    request_line = f"Could I please book a table for {party} on {date} around {time}, under the name {name}?"
    fallback_line = f"If {time} isn’t available, anything between {earliest} and {latest} would work."
    notes_line = f"One note: {notes}" if notes.strip() else ""
    proof_line = "If you can confirm it, what time is it booked for and what name should I put it under? Any reference number?"
    close = "Thanks very much. Appreciate it."

    busy_version = [
        "Sorry to bother you — quick one.",
        request_line,
        fallback_line,
        "Yes/no is perfect. Thank you."
    ]

    return {
        "restaurant_name": restaurant_name,
        "opening": opening,
        "request": request_line,
        "fallback": fallback_line,
        "notes": notes_line,
        "proof_request": proof_line,
        "close": close,
        "busy_version": busy_version
    }


def compute_next_action(booking_data: dict) -> str:
    """
    Human-friendly next action, used by /ui/status
    """
    status = booking_data.get("status", "unknown")
    call_allowed = booking_data.get("call_allowed") is True
    call_obj = booking_data.get("call") or {}
    call_sid = call_obj.get("call_sid")
    outcome_obj = booking_data.get("call_outcome") or {}
    outcome = outcome_obj.get("outcome")

    if status in ("confirmed", "failed", "expired"):
        return "No further action needed."

    if is_expired(booking_data):
        return "This booking is expired. (You can create a new one.)"

    if status == "needs_user_decision":
        return "Restaurant offered an alternative. Record details and decide what to do next."

    if status == "awaiting_confirmation":
        return "Call outcome is CONFIRMED. Next: confirm the booking (with proof)."

    # pending or unknown
    if call_allowed and not call_sid:
        return "Next: place a call (and read the call script)."

    if call_sid and outcome != "CONFIRMED":
        return "Next: record call outcome (NO_ANSWER / DECLINED / OFFERED_ALTERNATIVE / CONFIRMED)."

    if call_sid and outcome == "CONFIRMED":
        return "Next: confirm the booking (with proof)."

    return "Next: review details and continue."


class BookingRequest(BaseModel):
    name: str
    city: str
    date: str
    time: str
    party_size: int
    time_window_minutes: int = 30
    notes: str = ""

    # Step 3/5: restaurant details captured at request time (optional for now)
    restaurant_name: str = ""
    restaurant_phone: str = ""  # E.164 preferred: +64...


# -----------------------------
# Core API
# -----------------------------
@app.post("/book")
def book(req: BookingRequest):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    booking_id = str(uuid.uuid4())

    try:
        req_data = req.model_dump()
    except AttributeError:
        req_data = req.dict()

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

        # Step 4 fields
        "call_attempts": 0,
        "last_call_at": None,
        "expires_at": None,
        "final_reason": None
    }

    booking_data["expires_at"] = compute_expires_at(booking_data["created_at"])
    log_event(booking_data, "timeout_set", {"expires_at": booking_data["expires_at"]})

    log_event(booking_data, "booking_created", {"city": req_data["city"], "party_size": req_data["party_size"]})
    log_event(booking_data, "strategy_suggested", strategy)
    log_event(booking_data, "digital_attempted", digital_result)
    log_event(booking_data, "call_decision_made", {"call_allowed": call_allowed})

    supabase.table("bookings").insert({"id": booking_id, "data": booking_data}).execute()

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


@app.post("/call-test/{booking_id}")
def call_test(booking_id: str):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    result = supabase.table("bookings").select("data").eq("id", booking_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Booking not found")

    booking_data = result.data[0]["data"]

    # Basic guardrails
    if booking_data.get("status") != "pending":
        raise HTTPException(status_code=400, detail=f"Cannot call because status is '{booking_data.get('status')}'")

    if booking_data.get("call_allowed") is not True:
        raise HTTPException(status_code=400, detail="Call not allowed for this booking (call_allowed is false)")

    # Step 4: expiry
    if is_expired(booking_data):
        booking_data["status"] = "expired"
        booking_data["final_reason"] = "Timed out before confirmation"
        log_event(booking_data, "booking_expired", {"reason": booking_data["final_reason"]})
        supabase.table("bookings").update({"data": booking_data}).eq("id", booking_id).execute()
        raise HTTPException(status_code=400, detail="Booking has expired (timeout)")

    # Step 4: max attempts + cooldown
    ok, msg = can_call_again(booking_data)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    # Script: generate + log
    req = booking_data.get("request") or {}
    restaurant_name = (req.get("restaurant_name") or "the restaurant").strip() or "the restaurant"
    script = build_call_script(booking_data, restaurant_name=restaurant_name)
    log_event(booking_data, "call_script_generated", {"script": script})

    # Step 4: increment counters BEFORE calling Twilio
    booking_data["call_attempts"] = booking_data.get("call_attempts", 0) + 1
    booking_data["last_call_at"] = now_iso()
    log_event(booking_data, "call_attempt_incremented", {
        "call_attempts": booking_data["call_attempts"],
        "last_call_at": booking_data["last_call_at"]
    })

    # Choose destination (safe by default)
    to_number, mode = resolve_call_destination(booking_data)
    log_event(booking_data, "call_destination_resolved", {"mode": mode, "to": to_number})

    # Place Twilio call
    log_event(booking_data, "call_initiated", {"mode": "twilio"})
    sid = make_call(to_number)

    booking_data["call"] = {
        "call_sid": sid,
        "called_at": now_iso(),
        "to": to_number,
        "to_mode": mode
    }
    log_event(booking_data, "call_recorded", {"call_sid": sid})

    supabase.table("bookings").update({"data": booking_data}).eq("id", booking_id).execute()

    return {"message": "Call placed and recorded", "call_sid": sid, "to_mode": mode}


@app.get("/call-script/{booking_id}")
def call_script(booking_id: str, restaurant_name: str = "the restaurant"):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    result = supabase.table("bookings").select("data").eq("id", booking_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Booking not found")

    booking_data = result.data[0]["data"]
    script = build_call_script(booking_data, restaurant_name=restaurant_name)
    return {"booking_id": booking_id, "script": script}


@app.post("/call-outcome/{booking_id}")
def call_outcome(
    booking_id: str,
    outcome: str = Query(..., pattern="^(NO_ANSWER|DECLINED|OFFERED_ALTERNATIVE|CONFIRMED)$"),
    notes: str = Query("", max_length=300),
    confirmed_time: str = Query("", description="Only if outcome=CONFIRMED, e.g. 19:15"),
    reference: str = Query("", max_length=100)
):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    result = supabase.table("bookings").select("data").eq("id", booking_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Booking not found")

    booking_data = result.data[0]["data"]

    # Don’t allow outcomes on finished bookings
    if booking_data.get("status") in ("confirmed", "failed", "expired"):
        raise HTTPException(status_code=400, detail=f"Cannot record outcome because status is '{booking_data.get('status')}'")

    booking_data["call_outcome"] = {
        "outcome": outcome,
        "notes": notes,
        "confirmed_time": confirmed_time,
        "reference": reference,
        "recorded_at": now_iso()
    }
    log_event(booking_data, "call_outcome_recorded", booking_data["call_outcome"])

    # Step 4: behaviour rules after outcome
    if outcome == "CONFIRMED":
        booking_data["status"] = "awaiting_confirmation"  # user must still run /confirm with proof
        log_event(booking_data, "status_changed", {"status": booking_data["status"]})

    elif outcome == "DECLINED":
        booking_data["status"] = "failed"
        booking_data["final_reason"] = "Restaurant declined the booking request"
        log_event(booking_data, "booking_failed", {"reason": booking_data["final_reason"]})

    elif outcome == "NO_ANSWER":
        # stays pending, cooldown + max attempts will control future calls
        log_event(booking_data, "retry_possible", {"reason": "No answer"})

    elif outcome == "OFFERED_ALTERNATIVE":
        booking_data["status"] = "needs_user_decision"
        log_event(booking_data, "status_changed", {"status": booking_data["status"]})

    supabase.table("bookings").update({"data": booking_data}).eq("id", booking_id).execute()
    return {"message": "Call outcome saved", "booking_id": booking_id, "outcome": outcome, "status": booking_data.get("status")}


@app.post("/confirm/{booking_id}")
def confirm_booking(
    booking_id: str,
    proof: str = Query(..., min_length=10, description="Human-verifiable proof text"),
    confirmed_by: str = Query(..., min_length=2, description="Who confirmed it"),
    method: str = Query(..., pattern="^(phone|digital|in_person)$", description="How it was verified")
):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    result = supabase.table("bookings").select("data").eq("id", booking_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Booking not found")

    booking_data = result.data[0]["data"]

    if booking_data.get("status") == "confirmed":
        return {"message": "Already confirmed", "confirmation": booking_data.get("confirmation")}

    # Step 4: block if expired
    if is_expired(booking_data):
        booking_data["status"] = "expired"
        booking_data["final_reason"] = "Timed out before confirmation"
        log_event(booking_data, "booking_expired", {"reason": booking_data["final_reason"]})
        supabase.table("bookings").update({"data": booking_data}).eq("id", booking_id).execute()
        raise HTTPException(status_code=400, detail="Booking has expired (timeout)")

    # Allow confirming from pending OR awaiting_confirmation
    if booking_data.get("status") not in ("pending", "awaiting_confirmation"):
        raise HTTPException(status_code=400, detail=f"Cannot confirm because status is '{booking_data.get('status')}'")

    # Phone confirmations require call_sid + call_outcome CONFIRMED
    if method == "phone":
        call_obj = booking_data.get("call") or {}
        if not call_obj.get("call_sid"):
            raise HTTPException(status_code=400, detail="Cannot confirm by phone without a recorded call SID")

        outcome_obj = booking_data.get("call_outcome") or {}
        if outcome_obj.get("outcome") != "CONFIRMED":
            raise HTTPException(status_code=400, detail="Cannot confirm by phone unless call outcome is CONFIRMED")

    booking_data["status"] = "confirmed"
    booking_data["confirmation"] = {
        "proof": proof,
        "confirmed_by": confirmed_by,
        "method": method,
        "confirmed_at": now_iso()
    }
    log_event(booking_data, "confirmation_recorded", {"method": method, "confirmed_by": confirmed_by})

    supabase.table("bookings").update({"data": booking_data}).eq("id", booking_id).execute()
    return {"message": "Confirmed", "booking_id": booking_id, "status": booking_data["status"]}


@app.post("/advance/{booking_id}")
def advance(booking_id: str):
    """
    Optional helper: lets UI or you "tick" the booking forward (mainly: expire it if timed out).
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    result = supabase.table("bookings").select("data").eq("id", booking_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Booking not found")

    booking_data = result.data[0]["data"]

    if booking_data.get("status") in ("confirmed", "failed", "expired"):
        return {"message": "No change", "status": booking_data.get("status")}

    if is_expired(booking_data):
        booking_data["status"] = "expired"
        booking_data["final_reason"] = "Timed out before confirmation"
        log_event(booking_data, "booking_expired", {"reason": booking_data["final_reason"]})
        supabase.table("bookings").update({"data": booking_data}).eq("id", booking_id).execute()
        return {"message": "Expired", "status": booking_data["status"]}

    return {"message": "Still active", "status": booking_data.get("status"), "expires_at": booking_data.get("expires_at")}


@app.get("/timeline/{booking_id}")
def timeline(booking_id: str):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    result = supabase.table("bookings").select("data").eq("id", booking_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Booking not found")

    booking_data = result.data[0]["data"]
    events = booking_data.get("events", [])

    timeline_steps = []
    for event in events:
        t = event.get("time")
        et = event.get("type")

        if et == "booking_created":
            timeline_steps.append({"step": "Request received", "time": t})
        elif et == "digital_attempted":
            timeline_steps.append({"step": "Searching digitally", "time": t})
        elif et == "call_script_generated":
            timeline_steps.append({"step": "Preparing call script", "time": t})
        elif et == "call_initiated":
            timeline_steps.append({"step": "Calling to check availability", "time": t})
        elif et == "call_recorded":
            timeline_steps.append({"step": "Call connected (SID recorded)", "time": t})
        elif et == "call_outcome_recorded":
            outcome = (event.get("details") or {}).get("outcome", "unknown")
            timeline_steps.append({"step": f"Call outcome recorded: {outcome}", "time": t})
        elif et == "confirmation_recorded":
            timeline_steps.append({"step": "Booking confirmed", "time": t})
        elif et == "booking_failed":
            timeline_steps.append({"step": "Booking failed", "time": t})
        elif et == "booking_expired":
            timeline_steps.append({"step": "Booking expired (timeout)", "time": t})

    return {
        "booking_id": booking_id,
        "status": booking_data.get("status"),
        "expires_at": booking_data.get("expires_at"),
        "timeline": timeline_steps
    }


# -----------------------------
# Debug + Simple UI
# -----------------------------
@app.get("/debug/env")
def debug_env():
    supa_url = os.environ.get("SUPABASE_URL", "")
    parsed = urlparse(supa_url) if supa_url else None

    return {
        "SUPABASE_URL_set": bool(supa_url),
        "SUPABASE_URL_scheme": parsed.scheme if parsed else None,
        "SUPABASE_URL_netloc": parsed.netloc if parsed else None,
        "SUPABASE_KEY_set": bool(os.environ.get("SUPABASE_KEY")),
        "TWILIO_SID_set": bool(os.environ.get("TWILIO_SID")),
        "TWILIO_NUMBER_set": bool(os.environ.get("TWILIO_NUMBER")),
        "FOUNDER_PHONE_set": bool(os.environ.get("FOUNDER_PHONE")),
        "ALLOW_REAL_RESTAURANT_CALLS": ALLOW_REAL_RESTAURANT_CALLS,
    }


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
):
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

    result = book(req)
    booking_id = result["booking_id"]
    return RedirectResponse(url=f"/ui/status/{booking_id}", status_code=303)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Tabel</title>
        </head>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto;">
            <h1>Tabel</h1>
            <p>I will try get you a table. Fast. No lies.</p>

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
                <input type="time" name="time" required style="width: 100%; padding: 8px;"><br><br>

                <label>Party size</label><br>
                <input type="number" name="party_size" min="1" required style="width: 100%; padding: 8px;"><br><br>

                <label>Flex window (minutes)</label><br>
                <input type="number" name="time_window_minutes" min="0" value="30" style="width: 100%; padding: 8px;"><br><br>

                <label>Notes (optional)</label><br>
                <input name="notes" style="width: 100%; padding: 8px;"><br><br>

                <button type="submit" style="padding: 10px 16px;">Get me a table</button>
            </form>

            <hr style="margin: 30px 0;">
            <p><small>Developer tools: <a href="/docs">Swagger</a></small></p>
        </body>
    </html>
    """


@app.get("/ui/status/{booking_id}", response_class=HTMLResponse)
def ui_status(booking_id: str):
    # load booking (so we can compute next action)
    booking_data = status(booking_id)
    data = timeline(booking_id)

    status_text = data.get("status", "unknown")
    steps = data.get("timeline", [])
    expires_at = data.get("expires_at", "")
    next_action = compute_next_action(booking_data)

    html = f"""
    <html>
        <head>
            <title>Booking Status</title>
        </head>
        <body style="font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto;">
            <h1>Booking Status</h1>

            <p><strong>Status:</strong> {status_text}</p>
            <p><strong>Next action:</strong> {next_action}</p>
            <p><strong>Expires at (UTC):</strong> {expires_at}</p>

            <p>
                <a href="/call-script/{booking_id}" style="margin-right: 12px;">View call script</a>
                <a href="/ui/call-outcome/{booking_id}" style="margin-right: 12px;">Record call outcome</a>
                <a href="/docs" style="margin-right: 12px;">Confirm (via Swagger)</a>
            </p>

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

            <p>
                <button onclick="location.reload()">Refresh</button>
            </p>

            <hr style="margin: 30px 0;">
            <p><small>Booking ID: {booking_id}</small></p>
            <p><small><a href="/">New booking</a> | <a href="/docs">Swagger</a></small></p>
        </body>
    </html>
    """
    return html


# -----------------------------
# Step 6: Call outcome UI (no Swagger)
# -----------------------------
@app.get("/ui/call-outcome/{booking_id}", response_class=HTMLResponse)
def ui_call_outcome_form(booking_id: str):
    # will raise 404 if missing
    booking_data = status(booking_id)
    req = booking_data.get("request") or {}
    rname = (req.get("restaurant_name") or "").strip() or "the restaurant"

    return f"""
    <html>
        <head>
            <title>Call Outcome</title>
        </head>
        <body style="font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto;">
            <h1>Record Call Outcome</h1>
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

            <hr style="margin: 30px 0;">
            <p><small><a href="/ui/status/{booking_id}">Back to status</a></small></p>
        </body>
    </html>
    """


@app.post("/ui/call-outcome/{booking_id}")
def ui_call_outcome_submit(
    booking_id: str,
    outcome: str = Form(...),
    notes: str = Form(""),
    confirmed_time: str = Form(""),
    reference: str = Form(""),
):
    # Reuse the API logic (this will apply guardrails)
    call_outcome(
        booking_id=booking_id,
        outcome=outcome,
        notes=notes,
        confirmed_time=confirmed_time,
        reference=reference,
    )
    return RedirectResponse(url=f"/ui/status/{booking_id}", status_code=303)
