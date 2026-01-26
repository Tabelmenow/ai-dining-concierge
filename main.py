"""
Tabel AI Dining Concierge - Phase 3.3 Consolidated
Merges working auth/UI with evidence-backed confirmations and security hardening
"""

from fastapi import FastAPI, HTTPException, Query, Form, Request, Header, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import uuid
import os
import re
import time
import json
import hmac
import hashlib
import base64
import secrets
from urllib.parse import urlparse, quote
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from collections import defaultdict

try:
    from twilio.rest import Client as TwilioClient
    from twilio.base.exceptions import TwilioRestException
    from supabase import create_client, Client
    import stripe
    from gotrue.errors import AuthRetryableError
except ImportError as e:
    print(f"FATAL: Missing package: {e}")
    raise

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration"""
    
    # Supabase
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")  # Service role
    SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY", "")  # For auth
    
    # Twilio
    TWILIO_SID: str = os.getenv("TWILIO_SID", "")
    TWILIO_AUTH: str = os.getenv("TWILIO_AUTH", "")
    TWILIO_NUMBER: str = os.getenv("TWILIO_NUMBER", "")
    
    # Security
    ADMIN_TOKEN: str = os.getenv("ADMIN_TOKEN", "").strip()
    SESSION_SECRET: bytes = os.getenv("SESSION_SECRET", "").encode("utf-8")
    SESSION_COOKIE_NAME: str = "tabel_session"
    ADMIN_TOKEN_HEADER: str = "X-Admin-Token"
    
    # Safety toggles
    DRY_RUN_CALLS: bool = os.getenv("DRY_RUN_CALLS", "false").lower() == "true"
    ALLOW_REAL_RESTAURANT_CALLS: bool = os.getenv("ALLOW_REAL_RESTAURANT_CALLS", "false").lower() == "true"
    REQUIRE_OPERATOR_ARM_FOR_CALL: bool = True
    
    # Rate limiting
    BOOK_RATE_LIMIT_MAX: int = int(os.getenv("BOOK_RATE_LIMIT_MAX", "10"))
    BOOK_RATE_LIMIT_WINDOW_SEC: int = int(os.getenv("BOOK_RATE_LIMIT_WINDOW_SEC", "300"))
    CALL_RATE_LIMIT_MAX: int = int(os.getenv("CALL_RATE_LIMIT_MAX", "6"))
    CALL_RATE_LIMIT_WINDOW_SEC: int = int(os.getenv("CALL_RATE_LIMIT_WINDOW_SEC", "300"))
    
    # Retry & timeout
    MAX_CALL_ATTEMPTS: int = int(os.getenv("MAX_CALL_ATTEMPTS", "2"))
    CALL_COOLDOWN_MINUTES: int = int(os.getenv("CALL_COOLDOWN_MINUTES", "10"))
    BOOKING_TIMEOUT_MINUTES: int = int(os.getenv("BOOKING_TIMEOUT_MINUTES", "60"))
    
    # Allowlist
    PHONE_ALLOWLIST: List[str] = [
        p.strip() for p in os.getenv("PHONE_ALLOWLIST", "").split(",") if p.strip()
    ]
    
    # Founder safety
    FOUNDER_PHONE: str = os.getenv("FOUNDER_PHONE", "").strip()
    
    # App base URL for webhooks
    APP_BASE_URL: str = os.getenv("APP_BASE_URL", "https://ai-dining-concierge.onrender.com")
    
    # Stripe
    STRIPE_SECRET_KEY: str = os.getenv("STRIPE_SECRET_KEY", "")
    STRIPE_WEBHOOK_SECRET: str = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    STRIPE_PRICE_CENTS: int = int(os.getenv("STRIPE_PRICE_CENTS", "200"))
    STRIPE_CURRENCY: str = os.getenv("STRIPE_CURRENCY", "nzd")
    
    @classmethod
    def validate(cls):
        """Validate critical config"""
        errors = []
        if not cls.SUPABASE_URL:
            errors.append("SUPABASE_URL required")
        if not cls.SUPABASE_KEY:
            errors.append("SUPABASE_KEY required")
        if not cls.SESSION_SECRET:
            print("WARNING: SESSION_SECRET not set, using random (sessions won't survive restart)")
            cls.SESSION_SECRET = secrets.token_bytes(32)
        if errors:
            raise ValueError(f"Config errors: {', '.join(errors)}")

Config.validate()

# Initialize clients
supabase: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY) if Config.SUPABASE_URL and Config.SUPABASE_KEY else None
supabase_auth: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_ANON_KEY) if Config.SUPABASE_URL and Config.SUPABASE_ANON_KEY else None

# Initialize Stripe
if Config.STRIPE_SECRET_KEY:
    stripe.api_key = Config.STRIPE_SECRET_KEY

# ============================================================================
# HELPERS
# ============================================================================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def parse_iso(dt_str: str) -> datetime:
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))

def compute_expires_at(created_at_iso: str) -> str:
    created = parse_iso(created_at_iso)
    return (created + timedelta(minutes=Config.BOOKING_TIMEOUT_MINUTES)).isoformat()

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
    if attempts >= Config.MAX_CALL_ATTEMPTS:
        return False, f"Max call attempts reached ({Config.MAX_CALL_ATTEMPTS})"
    last_call_at = booking_data.get("last_call_at")
    if last_call_at:
        mins = minutes_since(last_call_at)
        if mins < Config.CALL_COOLDOWN_MINUTES:
            wait = int(Config.CALL_COOLDOWN_MINUTES - mins)
            return False, f"Cooldown active. Try again in ~{wait} minutes."
    return True, "OK"

def sanitize_for_log(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove PII from logs"""
    sensitive_keys = {'user_phone', 'restaurant_phone', 'phone', 'auth_token', 'password', 'email'}
    sanitized = {}
    for k, v in data.items():
        if k in sensitive_keys:
            sanitized[k] = "***REDACTED***"
        elif isinstance(v, dict):
            sanitized[k] = sanitize_for_log(v)
        else:
            sanitized[k] = v
    return sanitized

def log_json(event: str, payload: dict) -> None:
    """Structured logging with PII protection"""
    try:
        safe_payload = sanitize_for_log(payload)
        print(json.dumps({"event": event, "time": now_iso(), **safe_payload}, ensure_ascii=False))
    except Exception:
        print(f"[{now_iso()}] {event} {payload}")

def generate_integrity_hash(data: str) -> str:
    """SHA256 hash for evidence integrity"""
    return hashlib.sha256(data.encode()).hexdigest()

def log_event(booking_data: dict, event_type: str, details: dict | None = None) -> None:
    """Add event to booking's event log"""
    booking_data.setdefault("events", [])
    booking_data["events"].append({
        "type": event_type,
        "time": now_iso(),
        "details": sanitize_for_log(details or {})
    })
    booking_data["last_updated_at"] = now_iso()

def is_valid_phone_e164(phone: str) -> bool:
    if not phone:
        return False
    return bool(re.fullmatch(r"\+[1-9]\d{7,14}", phone.strip()))

def get_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def terminal_status(status: str) -> bool:
    return status in ("confirmed", "failed", "expired", "cancelled")

def ensure_not_terminal(booking_data: dict) -> None:
    if terminal_status(booking_data.get("status", "")):
        raise HTTPException(status_code=400, detail=f"Booking is already {booking_data.get('status')}.")

def ui_redirect_with_msg(url: str, msg: str) -> RedirectResponse:
    return RedirectResponse(url=f"{url}?msg={quote(msg)}", status_code=303)

# ============================================================================
# RATE LIMITING
# ============================================================================

_rate_buckets: dict[str, list[float]] = defaultdict(list)

def _rate_key(scope: str, ip: str) -> str:
    return f"{scope}:{ip}"

def rate_limit_or_429(scope: str, ip: str, max_events: int, window_sec: int) -> None:
    now = time.time()
    key = _rate_key(scope, ip)
    bucket = _rate_buckets[key]
    bucket[:] = [t for t in bucket if (now - t) <= window_sec]
    if len(bucket) >= max_events:
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Please wait and try again. (limit={max_events} per {window_sec}s)"
        )
    bucket.append(now)

# ============================================================================
# SESSION MANAGEMENT (HMAC-signed cookies)
# ============================================================================

def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("utf-8").rstrip("=")

def _b64url_decode(s: str) -> bytes:
    pad = "=" * ((4 - (len(s) % 4)) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("utf-8"))

def sign_token(token: str) -> str:
    if not Config.SESSION_SECRET:
        raise HTTPException(status_code=500, detail="SESSION_SECRET not set")
    mac = hmac.new(Config.SESSION_SECRET, token.encode("utf-8"), hashlib.sha256).digest()
    return f"{_b64url(token.encode('utf-8'))}.{_b64url(mac)}"

def verify_signed_token(signed: str) -> str | None:
    try:
        raw_b64, mac_b64 = signed.split(".", 1)
        token = _b64url_decode(raw_b64).decode("utf-8")
        mac = _b64url_decode(mac_b64)
        expected = hmac.new(Config.SESSION_SECRET, token.encode("utf-8"), hashlib.sha256).digest()
        if hmac.compare_digest(mac, expected):
            return token
        return None
    except Exception:
        return None

def get_access_token_from_cookie(request: Request) -> str | None:
    signed = request.cookies.get(Config.SESSION_COOKIE_NAME)
    if not signed:
        return None
    return verify_signed_token(signed)

def require_login(request: Request) -> dict:
    """Returns user dict: {id, email}"""
    if not supabase_auth:
        raise HTTPException(status_code=500, detail="SUPABASE_ANON_KEY not configured")
    
    token = get_access_token_from_cookie(request)
    if not token:
        raise HTTPException(status_code=401, detail="Not logged in")
    
    try:
        res = supabase_auth.auth.get_user(token)
        user = getattr(res, "user", None) or (res.get("user") if isinstance(res, dict) else None)
        if not user:
            raise HTTPException(status_code=401, detail="Session expired")
        
        if hasattr(user, "id"):
            return {"id": user.id, "email": getattr(user, "email", None)}
        return {"id": user.get("id"), "email": user.get("email")}
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Session invalid")

def assert_booking_owner(booking_data: dict, user_id: str) -> None:
    owner = booking_data.get("user_id")
    if not owner:
        raise HTTPException(status_code=403, detail="Booking not linked to user")
    if owner != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

def require_admin(token_q: str | None, x_admin_token: str | None) -> None:
    if not Config.ADMIN_TOKEN:
        raise HTTPException(status_code=500, detail="ADMIN_TOKEN not configured")
    provided = (token_q or "").strip() or (x_admin_token or "").strip()
    if not provided or not secrets.compare_digest(provided, Config.ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="Admin access denied")

# ============================================================================
# EVIDENCE MANAGEMENT (New - requirements compliance)
# ============================================================================

def create_evidence(booking_id: str, evidence_type: str, evidence_value: str, verifier: str = "system") -> dict:
    """Create evidence record in separate table"""
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
    evidence_data = {
        "booking_id": booking_id,
        "created_at": now_iso(),
        "evidence_type": evidence_type,
        "evidence_value": evidence_value,
        "verifier": verifier,
        "integrity_hash": generate_integrity_hash(evidence_value)
    }
    
    result = supabase.table("evidence").insert(evidence_data).execute()
    log_json("evidence_created", {
        "booking_id": booking_id,
        "evidence_type": evidence_type,
        "verifier": verifier
    })
    return result.data[0] if result.data else {}

def get_evidence_for_booking(booking_id: str) -> List[dict]:
    """Retrieve all evidence for a booking"""
    if not supabase:
        return []
    result = supabase.table("evidence").select("*").eq("booking_id", booking_id).execute()
    return result.data or []

def has_confirmation_evidence(booking_id: str) -> bool:
    """Check if booking has any confirmation evidence"""
    evidence = get_evidence_for_booking(booking_id)
    confirmation_types = {"CALL_SID", "HUMAN_NOTE", "TRANSCRIPT_SUMMARY"}
    return any(e.get("evidence_type") in confirmation_types for e in evidence)

# ============================================================================
# TWILIO INTEGRATION
# ============================================================================

def check_phone_allowlist(phone: str) -> bool:
    """Check allowlist (dev safety)"""
    if not Config.PHONE_ALLOWLIST:
        return True
    return phone in Config.PHONE_ALLOWLIST

def resolve_call_destination(booking_data: dict) -> tuple[str, str]:
    """Determine call destination with safety fallback"""
    req = booking_data.get("request") or {}
    restaurant_phone = (req.get("restaurant_phone") or "").strip()
    
    if Config.ALLOW_REAL_RESTAURANT_CALLS and is_valid_phone_e164(restaurant_phone):
        if check_phone_allowlist(restaurant_phone):
            return restaurant_phone, "restaurant"
    
    if not Config.FOUNDER_PHONE:
        raise HTTPException(status_code=500, detail="FOUNDER_PHONE not configured")
    return Config.FOUNDER_PHONE, "founder_safe_default"

def make_call(to_number: str, booking_id: str) -> str:
    """Place Twilio call with evidence recording"""
    if not Config.TWILIO_SID or not Config.TWILIO_AUTH or not Config.TWILIO_NUMBER:
        raise HTTPException(status_code=500, detail="Twilio not configured")
    
    if not is_valid_phone_e164(to_number):
        raise HTTPException(status_code=400, detail="Invalid phone number")
    
    client = TwilioClient(Config.TWILIO_SID, Config.TWILIO_AUTH)
    
    try:
        call = client.calls.create(
            to=to_number,
            from_=Config.TWILIO_NUMBER,
            url="http://demo.twilio.com/docs/voice.xml",  # TODO: Replace with your TwiML
            status_callback=f"{Config.APP_BASE_URL}/webhooks/call-status",
            status_callback_event=['completed'],
            record=True
        )
        
        # Create evidence for call SID
        create_evidence(booking_id, "CALL_SID", call.sid, verifier="system")
        
        return call.sid
        
    except TwilioRestException as e:
        log_json("twilio_error", {"error": str(e), "booking_id": booking_id})
        raise HTTPException(status_code=500, detail=f"Twilio error: {str(e)}")

# ============================================================================
# CALL SCRIPT GENERATION
# ============================================================================

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
        "fallback": f"If {time_str} isn't available, anything between {earliest} and {latest} would work.",
        "notes": f"One note: {notes}" if notes.strip() else "",
        "proof_request": "If you can confirm it, what time is it booked for and what name should I put it under? Any reference number?",
        "close": "Thanks very much. Appreciate it.",
    }

# ============================================================================
# AI STRATEGY (Placeholder)
# ============================================================================

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

# ============================================================================
# NEXT ACTION COMPUTATION
# ============================================================================

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
        return "Booking expired. Create a new one."
    if status == "needs_user_decision":
        return "Restaurant offered alternative. Record details and decide."
    if status == "awaiting_confirmation":
        return "Call outcome: CONFIRMED. Next: confirm booking with proof."
    if Config.REQUIRE_OPERATOR_ARM_FOR_CALL and not booking_data.get("operator_call_armed"):
        if call_allowed:
            return "Next: operator must ARM the call."
        return "Call not allowed for this booking."
    if call_allowed and not call_sid:
        return "Next: place a call and read the call script."
    if call_sid and outcome != "CONFIRMED":
        return "Next: record call outcome (NO_ANSWER/DECLINED/OFFERED_ALTERNATIVE/CONFIRMED)."
    if call_sid and outcome == "CONFIRMED":
        return "Next: confirm booking with proof."
    return "Next: review details and continue."

# ============================================================================
# MODELS
# ============================================================================

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

class ManualConfirmRequest(BaseModel):
    booking_id: str
    proof_text: str
    verifier_name: str
    
    @validator('proof_text')
    def validate_proof(cls, v):
        if len(v) < 10:
            raise ValueError("Proof must be at least 10 characters")
        if len(v) > 1000:
            raise ValueError("Proof too long (max 1000 chars)")
        return v

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Tabel AI Dining Concierge",
    description="Evidence-backed restaurant booking system",
    version="3.3.4"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CORE API ENDPOINTS
# ============================================================================

@app.get("/health")
def health():
    return {
        "ok": True,
        "time": now_iso(),
        "supabase_configured": bool(supabase),
        "supabase_auth_configured": bool(supabase_auth),
        "dry_run_calls": Config.DRY_RUN_CALLS,
        "allow_real_restaurant_calls": Config.ALLOW_REAL_RESTAURANT_CALLS,
        "require_operator_arm": Config.REQUIRE_OPERATOR_ARM_FOR_CALL,
        "admin_token_configured": bool(Config.ADMIN_TOKEN),
        "session_secret_configured": bool(Config.SESSION_SECRET),
        "stripe_configured": bool(Config.STRIPE_SECRET_KEY and Config.STRIPE_WEBHOOK_SECRET),
        "stripe_price_cents": Config.STRIPE_PRICE_CENTS,
        "stripe_currency": Config.STRIPE_CURRENCY,
    }

@app.post("/book")
def book(req: BookingRequest, request: Request):
    """Create new booking (requires auth)"""
    user = require_login(request)
    
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
    ip = get_client_ip(request)
    rate_limit_or_429("book", ip, Config.BOOK_RATE_LIMIT_MAX, Config.BOOK_RATE_LIMIT_WINDOW_SEC)
    
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
        "user_id": user["id"],
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
        "payment": {
            "payment_status": "unpaid",
            "amount_cents": Config.STRIPE_PRICE_CENTS,
            "currency": Config.STRIPE_CURRENCY,
            "stripe_checkout_session_id": None,
            "paid_at": None,
        },
    }
    
    booking_data["expires_at"] = compute_expires_at(booking_data["created_at"])
    
    log_event(booking_data, "booking_created", {"city": req_data["city"], "party_size": req_data["party_size"]})
    log_event(booking_data, "timeout_set", {"expires_at": booking_data["expires_at"]})
    log_event(booking_data, "strategy_suggested", strategy)
    log_event(booking_data, "digital_attempted", digital_result)
    log_event(booking_data, "call_decision_made", {"call_allowed": call_allowed})
    
    supabase.table("bookings").insert({"id": booking_id, "data": booking_data}).execute()
    log_json("booking_created", {"booking_id": booking_id, "user_id": user["id"], "ip": ip})
    
    return {
        "booking_id": booking_id,
        "status": booking_data["status"],
        "expires_at": booking_data["expires_at"]
    }

@app.get("/status/{booking_id}")
def status(booking_id: str, request: Request):
    """Get booking status (owner-only)"""
    user = require_login(request)
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
    result = supabase.table("bookings").select("data").eq("id", booking_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    booking_data = result.data[0]["data"]
    assert_booking_owner(booking_data, user["id"])
    
    # Attach evidence
    evidence = get_evidence_for_booking(booking_id)
    booking_data["_evidence"] = evidence
    
    return booking_data

@app.get("/timeline/{booking_id}")
def timeline(booking_id: str, request: Request):
    """Get booking timeline"""
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
            timeline_steps.append({"step": f"Call outcome: {d.get('outcome', 'unknown')}", "time": t})
        elif et == "status_changed":
            timeline_steps.append({"step": f"Status changed: {d.get('status', 'unknown')}", "time": t})
        elif et == "confirmation_recorded":
            timeline_steps.append({"step": "✅ Booking confirmed", "time": t})
        elif et == "manual_confirm":
            timeline_steps.append({"step": f"✅ Manual confirmation by {d.get('verifier', 'admin')}", "time": t})
        elif et == "booking_failed":
            timeline_steps.append({"step": f"❌ Booking failed: {d.get('reason', 'unknown')}", "time": t})
        elif et == "booking_cancelled":
            timeline_steps.append({"step": f"Cancelled: {d.get('reason', 'user_cancelled')}", "time": t})
        elif et == "booking_expired":
            timeline_steps.append({"step": "⏱️ Booking expired (timeout)", "time": t})
    
    return {
        "booking_id": booking_id,
        "status": booking_data.get("status"),
        "expires_at": booking_data.get("expires_at"),
        "final_reason": booking_data.get("final_reason"),
        "timeline": timeline_steps,
        "has_evidence": len(booking_data.get("_evidence", [])) > 0
    }

@app.post("/arm-call/{booking_id}")
def arm_call(booking_id: str, request: Request):
    """Operator arms the call"""
    booking_data = status(booking_id, request)
    ensure_not_terminal(booking_data)
    
    if is_expired(booking_data):
        booking_data["status"] = "expired"
        booking_data["final_reason"] = "timeout"
        log_event(booking_data, "booking_expired", {"reason": "timeout"})
        supabase.table("bookings").update({"data": booking_data}).eq("id", booking_id).execute()
        raise HTTPException(status_code=400, detail="Booking expired")
    
    if not booking_data.get("call_allowed"):
        raise HTTPException(status_code=400, detail="Call not allowed")
    
    booking_data["operator_call_armed"] = True
    booking_data["operator_armed_at"] = now_iso()
    log_event(booking_data, "operator_call_armed", {"armed_at": booking_data["operator_armed_at"]})
    
    supabase.table("bookings").update({"data": booking_data}).eq("id", booking_id).execute()
    return {"message": "Call armed", "booking_id": booking_id}

@app.post("/cancel/{booking_id}")
def cancel_booking(booking_id: str, request: Request, reason: str = Query("user_cancelled", max_length=80)):
    """Cancel booking"""
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
    """Initiate call to restaurant"""
    booking_data = status(booking_id, request)
    ensure_not_terminal(booking_data)
    
    ip = get_client_ip(request)
    rate_limit_or_429("call", ip, Config.CALL_RATE_LIMIT_MAX, Config.CALL_RATE_LIMIT_WINDOW_SEC)
    
    if booking_data.get("status") != "pending":
        raise HTTPException(status_code=400, detail=f"Cannot call from status: {booking_data.get('status')}")
    
    if not booking_data.get("call_allowed"):
        raise HTTPException(status_code=400, detail="Call not allowed")
    
    if Config.REQUIRE_OPERATOR_ARM_FOR_CALL and not booking_data.get("operator_call_armed"):
        raise HTTPException(status_code=400, detail="Operator must ARM call first")
    
    if is_expired(booking_data):
        booking_data["status"] = "expired"
        booking_data["final_reason"] = "timeout"
        log_event(booking_data, "booking_expired", {"reason": "timeout"})
        supabase.table("bookings").update({"data": booking_data}).eq("id", booking_id).execute()
        raise HTTPException(status_code=400, detail="Booking expired")
    
    ok, msg = can_call_again(booking_data)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    
    req = booking_data.get("request") or {}
    restaurant_name = (req.get("restaurant_name") or "the restaurant").strip() or "the restaurant"
    script = build_call_script(booking_data, restaurant_name=restaurant_name)
    log_event(booking_data, "call_script_generated", {"script": script})
    
    booking_data["call_attempts"] = booking_data.get("call_attempts", 0) + 1
    booking_data["last_call_at"] = now_iso()
    log_event(booking_data, "call_attempt_incremented", {
        "call_attempts": booking_data["call_attempts"],
        "last_call_at": booking_data["last_call_at"]
    })
    
    to_number, mode = resolve_call_destination(booking_data)
    log_event(booking_data, "call_destination_resolved", {"mode": mode, "to": to_number})
    log_event(booking_data, "call_initiated", {"mode": "twilio"})
    
    if Config.DRY_RUN_CALLS:
        log_event(booking_data, "dry_run_call_skipped", {"to": to_number, "to_mode": mode})
        # Create fake evidence for dry run
        fake_sid = f"DRY_RUN_{secrets.token_hex(8)}"
        create_evidence(booking_id, "CALL_SID", fake_sid, verifier="system_dry_run")
        supabase.table("bookings").update({"data": booking_data}).eq("id", booking_id).execute()
        return {"message": "Dry run: call skipped", "to_mode": mode, "dry_run": True}
    
    sid = make_call(to_number, booking_id)
    booking_data["call"] = {
        "call_sid": sid,
        "called_at": now_iso(),
        "to": to_number,
        "to_mode": mode
    }
    log_event(booking_data, "call_recorded", {"call_sid": sid})
    
    supabase.table("bookings").update({"data": booking_data}).eq("id", booking_id).execute()
    return {"message": "Call placed", "call_sid": sid, "to_mode": mode}

@app.post("/call-outcome/{booking_id}")
def call_outcome(
    booking_id: str,
    request: Request,
    outcome: str = Query(..., pattern="^(NO_ANSWER|DECLINED|OFFERED_ALTERNATIVE|CONFIRMED)$"),
    notes: str = Query("", max_length=300),
    confirmed_time: str = Query(""),
    reference: str = Query("", max_length=100),
):
    """Record call outcome"""
    booking_data = status(booking_id, request)
    ensure_not_terminal(booking_data)
    
    booking_data["call_outcome"] = {
        "outcome": outcome,
        "notes": notes,
        "confirmed_time": confirmed_time,
        "reference": reference,
        "recorded_at": now_iso()
    }
    log_event(booking_data, "call_outcome_recorded", booking_data["call_outcome"])
    
    # Create evidence for confirmed call outcome
    if outcome == "CONFIRMED":
        evidence_text = f"Call outcome: CONFIRMED"
        if confirmed_time:
            evidence_text += f" at {confirmed_time}"
        if reference:
            evidence_text += f" (ref: {reference})"
        if notes:
            evidence_text += f". Notes: {notes}"
        
        create_evidence(booking_id, "TRANSCRIPT_SUMMARY", evidence_text, verifier="operator")
        
        booking_data["status"] = "awaiting_confirmation"
        log_event(booking_data, "status_changed", {"status": "awaiting_confirmation"})
    elif outcome == "DECLINED":
        booking_data["status"] = "failed"
        booking_data["final_reason"] = "declined"
        log_event(booking_data, "booking_failed", {"reason": "declined"})
    elif outcome == "NO_ANSWER":
        log_event(booking_data, "retry_possible", {"reason": "no_answer"})
    elif outcome == "OFFERED_ALTERNATIVE":
        booking_data["status"] = "needs_user_decision"
        log_event(booking_data, "status_changed", {"status": "needs_user_decision"})
    
    supabase.table("bookings").update({"data": booking_data}).eq("id", booking_id).execute()
    return {
        "message": "Call outcome saved",
        "booking_id": booking_id,
        "outcome": outcome,
        "status": booking_data.get("status")
    }

@app.post("/confirm/{booking_id}")
def confirm_booking(
    booking_id: str,
    request: Request,
    proof: str = Query(..., min_length=10),
    confirmed_by: str = Query(..., min_length=2),
    method: str = Query(..., pattern="^(phone|digital|in_person)$"),
):
    """Confirm booking with proof (evidence-backed)"""
    booking_data = status(booking_id, request)
    
    if booking_data.get("status") == "confirmed":
        return {"message": "Already confirmed", "confirmation": booking_data.get("confirmation")}
    
    if terminal_status(booking_data.get("status", "")):
        raise HTTPException(status_code=400, detail=f"Cannot confirm: status is {booking_data.get('status')}")
    
    if is_expired(booking_data):
        booking_data["status"] = "expired"
        booking_data["final_reason"] = "timeout"
        log_event(booking_data, "booking_expired", {"reason": "timeout"})
        supabase.table("bookings").update({"data": booking_data}).eq("id", booking_id).execute()
        raise HTTPException(status_code=400, detail="Booking expired")
    
    if booking_data.get("status") not in ("pending", "awaiting_confirmation"):
        raise HTTPException(status_code=400, detail=f"Cannot confirm from status: {booking_data.get('status')}")
    
    # Validate evidence requirements for phone confirmations
    if method == "phone":
        call_obj = booking_data.get("call") or {}
        if not call_obj.get("call_sid"):
            raise HTTPException(status_code=400, detail="No call SID recorded")
        outcome_obj = booking_data.get("call_outcome") or {}
        if outcome_obj.get("outcome") != "CONFIRMED":
            raise HTTPException(status_code=400, detail="Call outcome must be CONFIRMED")
    
    # Create human confirmation evidence
    create_evidence(
        booking_id,
        "HUMAN_NOTE",
        f"Confirmation by {confirmed_by} via {method}: {proof}",
        verifier=confirmed_by
    )
    
    booking_data["status"] = "confirmed"
    booking_data["confirmation"] = {
        "proof": proof,
        "confirmed_by": confirmed_by,
        "method": method,
        "confirmed_at": now_iso()
    }
    log_event(booking_data, "confirmation_recorded", {"method": method, "confirmed_by": confirmed_by})
    
    supabase.table("bookings").update({"data": booking_data}).eq("id", booking_id).execute()
    
    log_json("booking_confirmed", {
        "booking_id": booking_id,
        "method": method,
        "confirmed_by": confirmed_by
    })
    
    return {
        "message": "✅ Confirmed with evidence",
        "booking_id": booking_id,
        "status": "confirmed",
        "evidence_count": len(get_evidence_for_booking(booking_id))
    }

# ============================================================================
# PAYMENT ENDPOINTS (Stripe)
# ============================================================================

@app.post("/pay/{booking_id}")
def create_payment_session(booking_id: str, request: Request):
    """Create Stripe checkout session for confirmed booking"""
    user = require_login(request)
    
    if not Config.STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Stripe not configured")
    
    booking_data = status(booking_id, request)
    
    # Only allow payment for confirmed bookings
    if booking_data.get("status") != "confirmed":
        raise HTTPException(status_code=400, detail="Payment only available for confirmed bookings")
    
    # Check if already paid
    payment = booking_data.get("payment") or {}
    if payment.get("payment_status") == "paid":
        return {"message": "Already paid", "paid_at": payment.get("paid_at")}
    
    # Create Stripe checkout session
    try:
        session = stripe.checkout.Session.create(
            mode="payment",
            line_items=[{
                "price_data": {
                    "currency": Config.STRIPE_CURRENCY,
                    "unit_amount": Config.STRIPE_PRICE_CENTS,
                    "product_data": {
                        "name": "Tabel booking success fee",
                    },
                },
                "quantity": 1,
            }],
            success_url=f"{Config.APP_BASE_URL}/ui/status/{booking_id}?msg=Payment+successful",
            cancel_url=f"{Config.APP_BASE_URL}/ui/status/{booking_id}?msg=Payment+cancelled",
            metadata={
                "booking_id": booking_id,
                "user_id": user["id"],
            },
        )
        
        # Update booking with session info
        if "payment" not in booking_data:
            booking_data["payment"] = {
                "payment_status": "unpaid",
                "amount_cents": Config.STRIPE_PRICE_CENTS,
                "currency": Config.STRIPE_CURRENCY,
                "stripe_checkout_session_id": None,
                "paid_at": None,
            }
        
        booking_data["payment"]["stripe_checkout_session_id"] = session.id
        booking_data["payment"]["payment_status"] = "pending"
        log_event(booking_data, "payment_session_created", {
            "session_id": session.id,
            "amount_cents": Config.STRIPE_PRICE_CENTS,
            "currency": Config.STRIPE_CURRENCY,
        })
        
        supabase.table("bookings").update({"data": booking_data}).eq("id", booking_id).execute()
        
        log_json("payment_session_created", {
            "booking_id": booking_id,
            "session_id": session.id,
            "amount_cents": Config.STRIPE_PRICE_CENTS,
        })
        
        return {"checkout_url": session.url}
        
    except stripe.error.StripeError as e:
        log_json("stripe_error", {"error": str(e), "booking_id": booking_id})
        raise HTTPException(status_code=500, detail=f"Payment error: {str(e)}")

@app.post("/webhooks/stripe")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events"""
    if not Config.STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Webhook secret not configured")
    
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    
    if not sig_header:
        raise HTTPException(status_code=400, detail="Missing signature")
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, Config.STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    # Handle checkout.session.completed
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        booking_id = session["metadata"].get("booking_id")
        user_id = session["metadata"].get("user_id")
        
        if not booking_id or not user_id:
            log_json("webhook_missing_metadata", {"session_id": session.get("id")})
            return {"status": "ignored"}
        
        # Fetch booking
        if not supabase:
            log_json("webhook_error", {"error": "Supabase not configured"})
            return {"status": "error"}
        
        result = supabase.table("bookings").select("data").eq("id", booking_id).execute()
        if not result.data:
            log_json("webhook_booking_not_found", {"booking_id": booking_id})
            return {"status": "ignored"}
        
        booking_data = result.data[0]["data"]
        
        # Verify user_id matches
        if booking_data.get("user_id") != user_id:
            log_json("webhook_user_mismatch", {
                "booking_id": booking_id,
                "expected_user": user_id,
                "actual_user": booking_data.get("user_id")
            })
            return {"status": "ignored"}
        
        # Verify session_id matches
        payment = booking_data.get("payment") or {}
        if payment.get("stripe_checkout_session_id") != session.get("id"):
            log_json("webhook_session_mismatch", {
                "booking_id": booking_id,
                "expected_session": session.get("id"),
                "actual_session": payment.get("stripe_checkout_session_id")
            })
            return {"status": "ignored"}
        
        # Update payment status
        if "payment" not in booking_data:
            booking_data["payment"] = {
                "amount_cents": Config.STRIPE_PRICE_CENTS,
                "currency": Config.STRIPE_CURRENCY,
                "stripe_checkout_session_id": session.get("id"),
            }
        
        booking_data["payment"]["payment_status"] = "paid"
        booking_data["payment"]["paid_at"] = now_iso()
        log_event(booking_data, "payment_confirmed", {"session_id": session.get("id")})
        
        supabase.table("bookings").update({"data": booking_data}).eq("id", booking_id).execute()
        
        log_json("payment_confirmed", {
            "booking_id": booking_id,
            "session_id": session.get("id"),
        })
    
    return {"status": "success"}

# ============================================================================
# ADMIN API (with evidence support)
# ============================================================================

@app.get("/admin/booking/{booking_id}")
def admin_get_booking(
    booking_id: str,
    token: str | None = None,
    x_admin_token: str | None = Header(default=None, alias=Config.ADMIN_TOKEN_HEADER),
):
    """Admin: Get any booking with evidence"""
    require_admin(token, x_admin_token)
    
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
    result = supabase.table("bookings").select("data").eq("id", booking_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    booking_data = result.data[0]["data"]
    evidence = get_evidence_for_booking(booking_id)
    
    return {
        "booking_id": booking_id,
        "data": booking_data,
        "evidence": evidence
    }

@app.post("/admin/manual-confirm")
def admin_manual_confirm(
    req: ManualConfirmRequest,
    token: str | None = Query(None),
    x_admin_token: str | None = Header(default=None, alias=Config.ADMIN_TOKEN_HEADER),
):
    """Admin: Manually confirm booking with proof"""
    require_admin(token, x_admin_token)
    
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
    result = supabase.table("bookings").select("data").eq("id", req.booking_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    booking_data = result.data[0]["data"]
    
    if booking_data.get("status") == "confirmed":
        return {"message": "Already confirmed", "booking_id": req.booking_id}
    
    if terminal_status(booking_data.get("status", "")):
        raise HTTPException(status_code=400, detail=f"Cannot confirm: status is {booking_data.get('status')}")
    
    # Create evidence
    create_evidence(
        req.booking_id,
        "HUMAN_NOTE",
        f"Admin manual confirmation by {req.verifier_name}: {req.proof_text}",
        verifier=req.verifier_name
    )
    
    booking_data["status"] = "confirmed"
    booking_data["confirmation"] = {
        "proof": req.proof_text,
        "confirmed_by": req.verifier_name,
        "method": "admin_manual",
        "confirmed_at": now_iso()
    }
    log_event(booking_data, "manual_confirm", {
        "verifier": req.verifier_name,
        "method": "admin_manual"
    })
    log_event(booking_data, "confirmation_recorded", {"method": "admin_manual"})
    
    supabase.table("bookings").update({"data": booking_data}).eq("id", req.booking_id).execute()
    
    log_json("admin_manual_confirm", {
        "booking_id": req.booking_id,
        "verifier": req.verifier_name
    })
    
    return {
        "message": "✅ Manually confirmed with evidence",
        "booking_id": req.booking_id,
        "status": "confirmed",
        "evidence_count": len(get_evidence_for_booking(req.booking_id))
    }

# ============================================================================
# WEBHOOKS
# ============================================================================

@app.post("/webhooks/call-status")
async def call_status_webhook(request: Request):
    """Twilio call status webhook"""
    form_data = await request.form()
    call_sid = form_data.get('CallSid')
    call_status = form_data.get('CallStatus')
    
    if not call_sid:
        return {"status": "ignored"}
    
    # Find booking by evidence table
    if supabase:
        evidence_result = supabase.table("evidence").select("booking_id").eq("evidence_value", call_sid).execute()
        
        if evidence_result.data:
            booking_id = evidence_result.data[0]["booking_id"]
            
            # Get booking
            booking_result = supabase.table("bookings").select("data").eq("id", booking_id).execute()
            if booking_result.data:
                booking_data = booking_result.data[0]["data"]
                
                log_event(booking_data, "twilio_callback", {
                    "call_sid": call_sid,
                    "status": call_status
                })
                
                # Update based on status
                if call_status == 'completed':
                    # Don't auto-confirm - require manual verification
                    log_event(booking_data, "call_completed", {"call_sid": call_sid})
                elif call_status in ['failed', 'busy', 'no-answer']:
                    if booking_data.get("status") == "pending":
                        log_event(booking_data, "call_failed", {"status": call_status})
                
                supabase.table("bookings").update({"data": booking_data}).eq("id", booking_id).execute()
    
    return {"status": "received"}

# ============================================================================
# AUTH UI
# ============================================================================

@app.get("/ui/login", response_class=HTMLResponse)
def ui_login(msg: str = ""):
    banner = f"<p style='background:#e7f3ff;padding:10px;border-radius:8px;border:1px solid #b3d7ff;'><strong>Info:</strong> {msg}</p>" if msg else ""
    return f"""
    <html>
      <head><title>Tabel Login</title></head>
      <body style="font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto;">
        <h1>🍽️ Tabel - Login</h1>
        {banner}
        <p>Enter your email for a magic link:</p>
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
        return ui_redirect_with_msg("/ui/login", "Authentication is not available right now. Please try again later.")
    
    # Normalize email
    normalized_email = email.strip().lower()
    
    redirect_to = f"{Config.APP_BASE_URL}/auth/callback"
    
    # Try with one retry on retryable errors
    for attempt in range(2):
        try:
            supabase_auth.auth.sign_in_with_otp({
                "email": normalized_email,
                "options": {"email_redirect_to": redirect_to}
            })
            
            log_json("magic_link_sent", {"email": normalized_email, "attempt": attempt + 1})
            return ui_redirect_with_msg("/ui/login", "Magic link sent. Check your email.")
            
        except AuthRetryableError as e:
            if attempt == 0:
                # First attempt failed with retryable error, sleep and retry
                log_json("login_retry", {"error": str(e), "email": normalized_email})
                time.sleep(1)
                continue
            else:
                # Second attempt also failed
                log_json("login_retryable_error", {"error": str(e), "email": normalized_email})
                return ui_redirect_with_msg("/ui/login", "Service temporarily unavailable. Please try again in a moment.")
        
        except Exception as e:
            # Unexpected error
            log_json("login_error", {"error": str(e), "email": normalized_email, "attempt": attempt + 1})
            return ui_redirect_with_msg("/ui/login", "Could not send magic link. Please check your email address and try again.")
    
    # Fallback (should never reach here)
    return ui_redirect_with_msg("/ui/login", "Please try again.")

@app.get("/auth/callback")
def auth_callback(request: Request):
    """Handle magic link callback"""
    params = dict(request.query_params)
    access_token = params.get("access_token")
    
    if not access_token:
        return HTMLResponse("""
        <html>
          <head><title>Completing login...</title></head>
          <body style="font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto;">
            <h1>Completing login...</h1>
            <button onclick="
              const hash = window.location.hash.startsWith('#') ? window.location.hash.slice(1) : window.location.hash;
              const q = new URLSearchParams(hash);
              if (q.get('access_token')) {
                window.location.href = '/auth/callback?' + q.toString();
              } else {
                document.getElementById('msg').innerText = 'No token found. Please request a new link.';
              }
            ">Finish login</button>
            <p id="msg" style="margin-top:12px;color:#555;"></p>
          </body>
        </html>
        """)
    
    signed = sign_token(access_token)
    
    resp = RedirectResponse(url="/ui/book", status_code=303)
    resp.set_cookie(
        key=Config.SESSION_COOKIE_NAME,
        value=signed,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 7,
    )
    return resp

@app.post("/ui/logout")
def ui_logout():
    resp = RedirectResponse(url="/ui/login?msg=" + quote("Logged out."), status_code=303)
    resp.delete_cookie(Config.SESSION_COOKIE_NAME)
    return resp

# ============================================================================
# UI ENDPOINTS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if request.url.path.startswith("/ui/") or request.url.path.startswith("/auth/"):
        msg = exc.detail if isinstance(exc.detail, str) else "Something went wrong."
        
        # Only redirect to login for auth errors
        if exc.status_code == 401:
            return ui_redirect_with_msg("/ui/login", "Please log in first.")
        
        # For other errors, redirect back to referer or to /ui/book
        referer = request.headers.get("referer")
        if referer and referer.startswith(Config.APP_BASE_URL):
            # Extract path from referer
            referer_path = referer.replace(Config.APP_BASE_URL, "")
            return ui_redirect_with_msg(referer_path, msg)
        
        return ui_redirect_with_msg("/ui/book", msg)
    
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
      <head><title>Tabel</title></head>
      <body style="font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto;">
        <h1>🍽️ Tabel</h1>
        <p>Evidence-backed restaurant bookings with no lies.</p>
        <ul>
          <li><a href="/ui/login">Login</a></li>
          <li><a href="/ui/book">New booking</a> (requires login)</li>
          <li><a href="/health">Health check</a></li>
          <li><a href="/docs">API Docs</a></li>
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
    note = f"<p style='background:#fff3cd;padding:10px;border-radius:8px;'><strong>Note:</strong> {msg}</p>" if msg else ""
    
    dry_run_banner = ""
    if Config.DRY_RUN_CALLS:
        dry_run_banner = "<p style='background:#e7f3ff;padding:10px;border-radius:8px;'><strong>DRY RUN MODE:</strong> No real calls will be placed.</p>"
    
    return f"""
    <html>
      <head><title>New Booking</title></head>
      <body style="font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto;">
        <h1>🍽️ New Booking</h1>
        {dry_run_banner}
        {note}
        
        <form action="/ui/book" method="post">
          <h3>Restaurant (optional)</h3>
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
        <p><a href="/">Home</a></p>
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
    evidence = booking_data.get("_evidence", [])
    
    call_allowed = booking_data.get("call_allowed") is True
    armed = booking_data.get("operator_call_armed") is True
    terminal = terminal_status(status_text)
    
    # Payment info
    payment = booking_data.get("payment") or {}
    payment_status = payment.get("payment_status", "unpaid")
    payment_amount_cents = payment.get("amount_cents", Config.STRIPE_PRICE_CENTS)
    payment_currency = payment.get("currency", Config.STRIPE_CURRENCY).upper()
    payment_display = f"${payment_amount_cents / 100:.2f} {payment_currency}"
    
    banner = f"<p style='background:#e7f3ff;padding:10px;border-radius:8px;'><strong>Info:</strong> {msg}</p>" if msg else ""
    reason_line = f"<p><strong>Final reason:</strong> {final_reason}</p>" if final_reason else ""
    
    # Payment status line
    payment_line = f"<p><strong>Payment:</strong> {payment_status}"
    if payment_status == "paid":
        paid_at = payment.get("paid_at", "")
        payment_line += f" ✅ <small>(paid at {paid_at})</small>"
    payment_line += "</p>"
    
    # Evidence display
    evidence_html = ""
    if evidence:
        evidence_html = """
        <div style="padding:12px;background:#f0f9ff;border-radius:8px;margin:20px 0;">
          <h3>📋 Evidence</h3>
          <ul style="margin:0;">
        """
        for ev in evidence:
            ev_type = ev.get("evidence_type", "unknown")
            ev_value = ev.get("evidence_value", "")[:150]
            verifier = ev.get("verifier", "system")
            created = ev.get("created_at", "")
            evidence_html += f"<li><strong>{ev_type}</strong> (by {verifier})<br><small>{ev_value}...</small><br><small>{created}</small></li>"
        evidence_html += "</ul></div>"
    
    controls = ""
    if not terminal:
        controls += f"""
        <form action="/ui/cancel/{booking_id}" method="post" style="display:inline;">
          <button type="submit" style="padding: 8px 12px; margin-right: 10px;">Cancel</button>
        </form>
        """
        if call_allowed and not armed and Config.REQUIRE_OPERATOR_ARM_FOR_CALL:
            controls += f"""
            <form action="/ui/arm-call/{booking_id}" method="post" style="display:inline;">
              <button type="submit" style="padding: 8px 12px; margin-right: 10px;">ARM call</button>
            </form>
            """
        if call_allowed and (armed or not Config.REQUIRE_OPERATOR_ARM_FOR_CALL):
            controls += f"""
            <form action="/ui/call/{booking_id}" method="post" style="display:inline;">
              <button type="submit" style="padding: 8px 12px; margin-right: 10px;">📞 Place call</button>
            </form>
            """
    
    # Payment button for confirmed bookings that aren't paid yet
    if status_text == "confirmed" and payment_status != "paid":
        controls += f"""
        <form action="/ui/pay/{booking_id}" method="post" style="display:inline;">
          <button type="submit" style="padding: 8px 12px; margin-right: 10px; background: #10b981; color: white; border: none; border-radius: 4px;">💳 Pay now ({payment_display})</button>
        </form>
        """
    
    html = f"""
    <html>
      <head><title>Booking Status</title></head>
      <body style="font-family: Arial, sans-serif; max-width: 950px; margin: 40px auto;">
        <h1>📊 Booking Status</h1>
        {banner}
        <p><strong>Status:</strong> {status_text}</p>
        {payment_line}
        <p><strong>Next action:</strong> {next_action}</p>
        <p><strong>Expires at:</strong> {expires_at}</p>
        {reason_line}
        
        <div style="padding:12px;background:#f6f6f6;border-radius:10px;margin:20px 0;">
          <p style="margin-top:0;"><strong>Controls</strong></p>
          {controls if controls else "<p>No actions available.</p>"}
          <p style="margin-bottom:0;">
            <a href="/ui/call-outcome/{booking_id}">Record call outcome</a> |
            <a href="/ui/confirm/{booking_id}">Confirm booking</a>
          </p>
        </div>
        
        {evidence_html}
        
        <h2>📅 Timeline</h2>
        <ul>
    """
    
    if not steps:
        html += "<li>No events yet. Refresh.</li>"
    else:
        for item in steps:
            html += f"<li>{item['step']}<br><small>{item['time']}</small></li>"
    
    html += f"""
        </ul>
        
        <p><button onclick="location.reload()">🔄 Refresh</button></p>
        
        <hr style="margin: 20px 0;">
        <p><small>Booking ID: {booking_id}</small></p>
        <p><a href="/ui/book">New booking</a> | <a href="/">Home</a></p>
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
    return ui_redirect_with_msg(f"/ui/status/{booking_id}", "✅ Call armed")

@app.post("/ui/call/{booking_id}")
def ui_call(request: Request, booking_id: str):
    result = call_test(booking_id, request)
    msg = result.get("message", "Call complete")
    if result.get("dry_run"):
        msg += " (DRY RUN)"
    return ui_redirect_with_msg(f"/ui/status/{booking_id}", msg)

@app.post("/ui/cancel/{booking_id}")
def ui_cancel(request: Request, booking_id: str):
    cancel_booking(booking_id, request, reason="user_cancelled")
    return ui_redirect_with_msg(f"/ui/status/{booking_id}", "Booking cancelled")

@app.post("/ui/pay/{booking_id}")
def ui_pay(request: Request, booking_id: str):
    """UI endpoint to initiate payment - redirects to Stripe checkout"""
    try:
        result = create_payment_session(booking_id, request)
        checkout_url = result.get("checkout_url")
        if checkout_url:
            return RedirectResponse(url=checkout_url, status_code=303)
        # Already paid case
        return ui_redirect_with_msg(f"/ui/status/{booking_id}", result.get("message", "Already paid"))
    except HTTPException as e:
        return ui_redirect_with_msg(f"/ui/status/{booking_id}", f"Payment error: {e.detail}")
    except Exception as e:
        log_json("ui_pay_error", {"error": str(e), "booking_id": booking_id})
        return ui_redirect_with_msg(f"/ui/status/{booking_id}", "Payment error. Please try again.")

@app.get("/ui/call-outcome/{booking_id}", response_class=HTMLResponse)
def ui_call_outcome_form(request: Request, booking_id: str, msg: str = ""):
    booking_data = status(booking_id, request)
    req = booking_data.get("request") or {}
    rname = (req.get("restaurant_name") or "").strip() or "the restaurant"
    banner = f"<p style='background:#e7f3ff;padding:10px;border-radius:8px;'><strong>Info:</strong> {msg}</p>" if msg else ""
    
    return f"""
    <html>
      <head><title>Call Outcome</title></head>
      <body style="font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto;">
        <h1>📞 Record Call Outcome</h1>
        {banner}
        <p><strong>Restaurant:</strong> {rname}</p>
        
        <form action="/ui/call-outcome/{booking_id}" method="post">
          <label>Outcome</label><br>
          <select name="outcome" required style="width: 100%; padding: 8px;">
            <option value="NO_ANSWER">NO_ANSWER</option>
            <option value="DECLINED">DECLINED</option>
            <option value="OFFERED_ALTERNATIVE">OFFERED_ALTERNATIVE</option>
            <option value="CONFIRMED">CONFIRMED</option>
          </select><br><br>
          
          <label>Notes</label><br>
          <input name="notes" style="width: 100%; padding: 8px;"><br><br>
          
          <label>Confirmed time (if CONFIRMED, e.g. 19:15)</label><br>
          <input name="confirmed_time" style="width: 100%; padding: 8px;"><br><br>
          
          <label>Reference</label><br>
          <input name="reference" style="width: 100%; padding: 8px;"><br><br>
          
          <button type="submit" style="padding: 10px 16px;">Save outcome</button>
          <a href="/ui/status/{booking_id}">Cancel</a>
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
    return ui_redirect_with_msg(f"/ui/status/{booking_id}", "✅ Call outcome saved")

@app.get("/ui/confirm/{booking_id}", response_class=HTMLResponse)
def ui_confirm_form(request: Request, booking_id: str, msg: str = ""):
    booking_data = status(booking_id, request)
    req = booking_data.get("request") or {}
    rname = (req.get("restaurant_name") or "").strip() or "the restaurant"
    banner = f"<p style='background:#e7f3ff;padding:10px;border-radius:8px;'><strong>Info:</strong> {msg}</p>" if msg else ""
    
    return f"""
    <html>
      <head><title>Confirm Booking</title></head>
      <body style="font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto;">
        <h1>✅ Confirm Booking</h1>
        {banner}
        <p><strong>Restaurant:</strong> {rname}</p>
        
        <p style="background:#fff3cd;padding:12px;border-radius:8px;">
          ⚠️ Only confirm if you have real proof. This creates evidence.
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
          
          <label>Proof (min 10 chars)</label><br>
          <textarea name="proof" required style="width: 100%; padding: 8px; height: 80px;"></textarea><br><br>
          
          <button type="submit" style="padding: 10px 16px;">✅ Confirm now</button>
          <a href="/ui/status/{booking_id}">Cancel</a>
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
    return ui_redirect_with_msg(f"/ui/status/{booking_id}", "✅ Booking confirmed with evidence")

# ============================================================================
# DEBUG
# ============================================================================

@app.get("/debug/env")
def debug_env():
    parsed = urlparse(Config.SUPABASE_URL) if Config.SUPABASE_URL else None
    return {
        "SUPABASE_URL_set": bool(Config.SUPABASE_URL),
        "SUPABASE_URL_scheme": parsed.scheme if parsed else None,
        "SUPABASE_URL_netloc": parsed.netloc if parsed else None,
        "SUPABASE_KEY_set": bool(Config.SUPABASE_KEY),
        "SUPABASE_ANON_KEY_set": bool(Config.SUPABASE_ANON_KEY),
        "SESSION_SECRET_set": bool(Config.SESSION_SECRET),
        "ADMIN_TOKEN_set": bool(Config.ADMIN_TOKEN),
        "DRY_RUN_CALLS": Config.DRY_RUN_CALLS,
        "ALLOW_REAL_RESTAURANT_CALLS": Config.ALLOW_REAL_RESTAURANT_CALLS,
    }

# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    print("=" * 60)
    print("Tabel AI Dining Concierge - Starting")
    print("=" * 60)
    print(f"DRY_RUN_CALLS: {Config.DRY_RUN_CALLS}")
    print(f"ALLOW_REAL_RESTAURANT_CALLS: {Config.ALLOW_REAL_RESTAURANT_CALLS}")
    print(f"REQUIRE_OPERATOR_ARM: {Config.REQUIRE_OPERATOR_ARM_FOR_CALL}")
    print(f"Max call attempts: {Config.MAX_CALL_ATTEMPTS}")
    print(f"Allowlist: {len(Config.PHONE_ALLOWLIST)} numbers" if Config.PHONE_ALLOWLIST else "Allowlist: DISABLED")
    print("=" * 60)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
