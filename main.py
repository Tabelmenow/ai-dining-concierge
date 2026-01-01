from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import os
from twilio.rest import Client

app = FastAPI()
bookings = {}
 
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

    # 1) AI suggests strategy (suggest only)
    strategy = ai_suggest_strategy(req.dict())

    # 2) Try digital first (simulated for now)
    digital_result = try_digital_booking(req.dict())

    # 3) Step 10.3: Save the call decision (deterministic rules)
    call_allowed = should_call_restaurant({
        "request": req.dict(),
        "strategy": strategy,
        "digital_attempt": digital_result
    })

    # 4) Store everything
    bookings[booking_id] = {
        "request": req.dict(),
        "status": "pending",
        "strategy": strategy,
        "digital_attempt": digital_result,
        "call_allowed": call_allowed,
        "confirmation": None
    }

    # 5) Return what you need for testing in Swagger
    return {
        "booking_id": booking_id,
        "status": "pending",
        "strategy": strategy,
        "digital_attempt": digital_result,
        "call_allowed": call_allowed
    }

@app.get("/status/{booking_id}")
def status(booking_id: str):
    return bookings.get(booking_id, "Not found")

# -----------------------------
# Step 11.5: Replace /call-test endpoint to place a real Twilio call
# -----------------------------
@app.post("/call-test")
def call_test():
    sid = make_test_call()
    return {"call_sid": sid}

@app.post("/confirm/{booking_id}")
def confirm_booking(booking_id: str, proof: str):
    if booking_id not in bookings:
        return {"error": "Not found"}

    bookings[booking_id]["status"] = "confirmed"
    bookings[booking_id]["confirmation"] = {
        "proof": proof
    }

    return {"message": "Confirmed"}

