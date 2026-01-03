from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import os
from twilio.rest import Client

app = FastAPI()
bookings = {}

from supabase import create_client

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"]
)


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
        "digital_attempt": digital_result,
        "call_allowed": call_allowed,
        "call": None,
        "confirmation": None,
        "created_at": now_iso(),
        "last_updated_at": now_iso()
    }


    # 5)  Save to Supabase
    supabase.table("bookings").insert({
        "id": booking_id,
        "data": booking_data
    }).execute()

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
    result = supabase.table("bookings") \
        .select("data") \
        .eq("id", booking_id) \
        .execute()

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
        "TWILIO_SID_set": bool(os.environ.get("TWILIO_SID")),
        "TWILIO_NUMBER_set": bool(os.environ.get("TWILIO_NUMBER")),
        "FOUNDER_PHONE_set": bool(os.environ.get("FOUNDER_PHONE")),
    }



# -----------------------------
# Step 11.5: Replace /call-test endpoint to place a real Twilio call
# -----------------------------
@app.post("/call-test/{booking_id}")
def call_test(booking_id: str):
    if booking_id not in bookings:
        return {"error": "Not found"}

    # If you have Twilio working, use make_test_call()
    # Otherwise simulate:
    # sid = "SIMULATED_CALL_SID"

    sid = "SIMULATED_CALL_SID"

    bookings[booking_id]["call"] = {
        "call_sid": sid,
        "called_at": now_iso(),
        "to": "FOUNDER_PHONE"
    }
    bookings[booking_id]["last_updated_at"] = now_iso()

    return {"message": "Call recorded", "call_sid": sid}


@app.post("/confirm/{booking_id}")
def confirm_booking(booking_id: str, proof: str):
    if booking_id not in bookings:
        return {"error": "Not found"}

    bookings[booking_id]["status"] = "confirmed"
    bookings[booking_id]["confirmation"] = {
        "proof": proof
    }

    return {"message": "Confirmed"}

