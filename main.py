from fastapi import FastAPI
from pydantic import BaseModel
import uuid
def ai_suggest_strategy(booking):
    """
    AI suggestion only.
    This does NOT confirm availability.
    """
    return {
        "recommended_action": "try_digital_first",
        "reason": "Mid-week booking with small party size has higher digital success rate",
        "confidence": "medium"
    }
def try_digital_booking(booking):
    """
    Simulates a digital booking attempt.
    Does NOT confirm availability.
    """
    city = booking["city"]
    party_size = booking["party_size"]

    if party_size <= 4:
        return {
            "success": False,
            "reason": "No digital availability found"
        }
    else:
        return {
            "success": False,
            "reason": "Party size too large for digital"
        }

app = FastAPI()
bookings = {}

class BookingRequest(BaseModel):
    name: str
    city: str
    date: str
    time: str
    party_size: int

@app.post("/book")
def book(req: BookingRequest):
    booking_id = str(uuid.uuid4())

    strategy = ai_suggest_strategy(req.dict())

    bookings[booking_id] = {
        "request": req.dict(),
        "status": "pending",
        "strategy": strategy,
        "confirmation": None
    }

    return {
        "booking_id": booking_id,
        "strategy": strategy
    }


@app.get("/status/{booking_id}")
def status(booking_id: str):
    return bookings.get(booking_id, "Not found")

@app.post("/call-test")
def call_test():
    return {"message": "Call triggered (simulation)"}
