from fastapi import FastAPI
from pydantic import BaseModel
import uuid

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
    bookings[booking_id] = {
        "request": req.dict(),
        "status": "pending",
        "confirmation": None
    }
    return {"booking_id": booking_id}

@app.get("/status/{booking_id}")
def status(booking_id: str):
    return bookings.get(booking_id, "Not found")

@app.post("/call-test")
def call_test():
    return {"message": "Call triggered (simulation)"}
