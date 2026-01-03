from fastapi import FastAPI, HTTPException
from fastapi import Query
from pydantic import BaseModel
import uuid
import os
from twilio.rest import Client
from supabase import create_client


app = FastAPI()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    supabase = None
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


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

from datetime import datetime, timezone

def now_iso():
    return datetime.now(timezone.utc).isoformat()

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

    # Simulated logic (always fails for now, but with different reasons)
    if party_size <= 4:
        return {"success": False, "reason": "No digital availability found"}
    else:
        return {"success": False, "reason": "Party size too large for digital"}

def should_call_restaurant(context: dict) -> bool:
    """
    Deterministic decision: should we allow a call step?
    Uses rules, not AI guesses.
    """
    # Never call if digital succeeded
    if context["digital_attempt"]["success"] is True:
        return False

    # Only consider calling if strategy supports it (adjust later when you add more strategies)
    if context["strategy"]["recommended_action"] != "try_digital_first":
        return False

    # Don't call for very large groups in this simple prototype
    if context["request"]["party_size"] > 8:
        return False

    return True

# -----------------------------
# Step 11.4: Add real Twilio call code (Founder phone only)
# -----------------------------
def make_test_call() -> str:
    """
    Makes a test call to the founder's verified phone number.
    This is for testing only (not calling restaurants yet).
    """
    # Basic env var validation (helps debugging)
    required_vars = ["TWILIO_SID", "TWILIO_AUTH", "TWILIO_NUMBER", "FOUNDER_PHONE"]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing environment variables: {', '.join(missing)}"
        )

    client = Client(
        os.environ["TWILIO_SID"],
        os.environ["TWILIO_AUTH"]
    )

    call = client.calls.create(
        to=os.environ["FOUNDER_PHONE"],
        from_=os.environ["TWILIO_NUMBER"],
        url="http://demo.twilio.com/docs/voice.xml"
    )

    return call.sid

class BookingRequest(BaseModel):
    name: str
    city: str
    date: str
    time: str
    party_size: int

@app.post("/book")
def book(req: BookingRequest):
    booking_id = str(uuid.uuid4())

    # Use Pydantic-safe dump (works in v1/v2)
    try:
        req_data = req.model_dump()
    except AttributeError:
        req_data = req.dict()
   
    # 1) AI suggests strategy (suggest only)
    strategy = ai_suggest_strategy(req_data)

    # 2) Try digital first (simulated for now)
    digital_result = try_digital_booking(req_data)

    # 3) Decide if a call is allowed (deterministic rules)
    call_allowed = should_call_restaurant({
        "request": req_data,
        "strategy": strategy,
        "digital_attempt": digital_result
    })

    # 4) Build booking record (named variable)
    booking_data = {
        "request": req_data,
        "status": "pending",
        "strategy": strategy,
        "events": [],
        "digital_attempt": digital_result,
        "call_allowed": call_allowed,
        "call": None,
        "confirmation": None,
        "created_at": now_iso(),
        "last_updated_at": now_iso()
    }
    log_event(booking_data, "booking_created", {"city": req_data["city"], "party_size": req_data["party_size"]})
    log_event(booking_data, "strategy_suggested", strategy)
    log_event(booking_data, "digital_attempted", digital_result)
    log_event(booking_data, "call_decision_made", {"call_allowed": call_allowed})


    # 5) Save to Supabase
    if supabase:
        supabase.table("bookings").insert({
            "id": booking_id,
            "data": booking_data
        }).execute()
    else:
        raise HTTPException(status_code=500, detail="Supabase not configured")


    # 6) Return what Swagger needs
    return {
        "booking_id": booking_id,
        "status": booking_data["status"],
        "strategy": booking_data["strategy"],
        "digital_attempt": booking_data["digital_attempt"],
        "created_at": booking_data["created_at"],
        "call_allowed": booking_data["call_allowed"]
    }


@app.get("/status/{booking_id}")
def status(booking_id: str):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    result = supabase.table("bookings").select("data").eq("id", booking_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Booking not found")

    return result.data[0]["data"]


from urllib.parse import urlparse

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
    }



# -----------------------------
# Step 11.5: Replace /call-test endpoint to place a real Twilio call
# -----------------------------
@app.post("/call-test/{booking_id}")
def call_test(booking_id: str):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    # 1) Load booking
    result = supabase.table("bookings").select("data").eq("id", booking_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Booking not found")

    booking_data = result.data[0]["data"]

    # 2) Guardrails
    if booking_data.get("status") != "pending":
        raise HTTPException(status_code=400, detail=f"Cannot call because status is '{booking_data.get('status')}'")

    if booking_data.get("call_allowed") is not True:
        raise HTTPException(status_code=400, detail="Call not allowed for this booking (call_allowed is false)")

    # Prevent repeated calls (simple throttle)
    if booking_data.get("call") and booking_data["call"].get("call_sid"):
        raise HTTPException(status_code=400, detail="Call already recorded for this booking")

    # 3) Place real Twilio call
    log_event(booking_data, "call_initiated", {"mode": "twilio"})
    sid = make_test_call()

    # 4) Store call record
    booking_data["call"] = {
        "call_sid": sid,
        "called_at": now_iso(),
        "to": "FOUNDER_PHONE"
    }
    log_event(booking_data, "call_recorded", {"call_sid": sid})

    # 5) Save back
    supabase.table("bookings").update({"data": booking_data}).eq("id", booking_id).execute()

    return {"message": "Call placed and recorded", "call_sid": sid}



@app.post("/confirm/{booking_id}")
def confirm_booking(
    booking_id: str,
    proof: str = Query(..., min_length=10),
    confirmed_by: str = Query(..., min_length=2),
    method: str = Query(..., pattern="^(phone|digital|in_person)$")
):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    # 1) Load booking
    result = supabase.table("bookings").select("data").eq("id", booking_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Booking not found")

    booking_data = result.data[0]["data"]

    # 2) Guardrails: cannot confirm twice
    if booking_data.get("status") == "confirmed":
        return {"message": "Already confirmed", "confirmation": booking_data.get("confirmation")}

    # Must be pending
    if booking_data.get("status") != "pending":
        raise HTTPException(status_code=400, detail=f"Cannot confirm because status is '{booking_data.get('status')}'")

    # If confirming by phone, require a recorded call SID
    if method == "phone":
        call_obj = booking_data.get("call") or {}
        if not call_obj.get("call_sid"):
            raise HTTPException(status_code=400, detail="Cannot confirm by phone without a recorded call SID")

    # 3) Set confirmation
    booking_data["status"] = "confirmed"
    booking_data["confirmation"] = {
        "proof": proof,
        "confirmed_by": confirmed_by,
        "method": method,
        "confirmed_at": now_iso()
    }

    log_event(booking_data, "confirmation_recorded", {
        "method": method,
        "confirmed_by": confirmed_by
    })

    # 4) Save back
    supabase.table("bookings").update({"data": booking_data}).eq("id", booking_id).execute()

    return {"message": "Confirmed", "booking_id": booking_id}


