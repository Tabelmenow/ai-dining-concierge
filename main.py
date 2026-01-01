from fastapi import FastAPI
from pydantic import BaseModel
import uuid

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
    # For now, strategy is always "try_digital_first", so this check mostly documents intent.
    if context["strategy"]["recommended_action"] != "try_digital_first":
        return False

    # Don't call for very large groups in this simple prototype
    if context["request"]["party_size"] > 8:
        return False
        "reason": "Large groups require special handling"
    return True

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

@app.post("/call-test")
def call_test():
    return {"message": "Call triggered (simulation)"}

