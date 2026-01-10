"""
FastAPI Backend - Privacy Proxy & Training Server
=================================================
This is NOT for routine inference (that's on-device).

This backend is used for:
1. Training ML models
2. Heavy AI fallback (when device can't handle)
3. Hash-based scam reporting aggregation
4. Signed blocklist distribution
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import hashlib
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector import ScamDetector, DetectionResult

# Initialize FastAPI
app = FastAPI(
    title="ScamShield Backend",
    description="Privacy-first backend for scam detection system",
    version="1.0.0"
)

# CORS for mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
model_dir = os.path.join(project_dir, 'models')
detector = ScamDetector(model_dir=model_dir)


# --- Request/Response Models ---

class AnalyzeRequest(BaseModel):
    """Request for message analysis."""
    text: str
    sender_id: Optional[str] = None
    modality: Optional[str] = "sms"  # sms, whatsapp, call


class AnalyzeResponse(BaseModel):
    """Response from message analysis."""
    risk_level: str
    confidence: float
    reason_en: str
    reason_hi: str
    signals: dict
    rules: List[str]
    action_en: str
    action_hi: str


class ReportRequest(BaseModel):
    """Hash-based scam report (privacy-preserving)."""
    message_hash: str  # SHA256 of message
    scam_type: str
    reporter_region: Optional[str] = None


class CallMetadataRequest(BaseModel):
    """Call risk analysis from metadata only."""
    caller_number_hash: str  # Hashed for privacy
    call_duration_seconds: int
    is_unknown_caller: bool
    user_reported_otp_request: bool
    user_reported_payment_request: bool


# --- Endpoints ---

@app.get("/")
async def root():
    """Health check."""
    return {
        "service": "ScamShield Backend",
        "status": "healthy",
        "note": "This backend is for fallback/training only. Primary detection is on-device."
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_message(request: AnalyzeRequest):
    """
    Analyze a message for scam indicators.
    
    NOTE: This endpoint is for FALLBACK only when on-device ML is insufficient.
    Normal flow should use on-device rule engine + TFLite model.
    """
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Empty message text")
    
    result = detector.detect(request.text, request.sender_id)
    
    return AnalyzeResponse(
        risk_level=result.risk_level,
        confidence=round(result.confidence, 2),
        reason_en=result.reason_en,
        reason_hi=result.reason_hi,
        signals=result.detected_signals,
        rules=result.triggered_rules,
        action_en=result.recommended_action,
        action_hi=result.recommended_action_hi
    )


@app.post("/api/analyze/call-metadata")
async def analyze_call_metadata(request: CallMetadataRequest):
    """
    Analyze call risk from METADATA only (no audio).
    
    This respects privacy - no call recording or transcription.
    """
    risk_score = 0
    reasons = []
    
    # Unknown caller
    if request.is_unknown_caller:
        risk_score += 20
        reasons.append("Unknown caller")
    
    # Long call (potential scam engagement)
    if request.call_duration_seconds > 180:  # 3 minutes
        risk_score += 15
        reasons.append("Long call duration")
    
    # User reported OTP request
    if request.user_reported_otp_request:
        risk_score += 40
        reasons.append("OTP was requested during call")
    
    # User reported payment request
    if request.user_reported_payment_request:
        risk_score += 40
        reasons.append("Payment was requested during call")
    
    # Determine risk level
    if risk_score >= 50:
        risk_level = "HIGH"
        action = "Consider blocking this number and informing family"
    elif risk_score >= 25:
        risk_level = "MEDIUM"
        action = "Be cautious about any future calls from this number"
    else:
        risk_level = "LOW"
        action = "Call appears normal"
    
    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "reasons": reasons,
        "action": action
    }


@app.post("/api/report")
async def report_scam(request: ReportRequest):
    """
    Submit a scam report using hash-based privacy.
    
    We store ONLY the hash, never raw message content.
    """
    # In production: store to PostgreSQL + Redis
    # For hackathon: just acknowledge
    return {
        "status": "received",
        "message_hash_prefix": request.message_hash[:8] + "...",
        "note": "Report stored. No personal data was collected."
    }


@app.get("/api/blocklist")
async def get_blocklist():
    """
    Get signed blocklist of known scam numbers/URLs.
    
    Updates are distributed periodically, not real-time.
    """
    # Sample blocklist for hackathon demo
    return {
        "version": "2026-01-11",
        "url_patterns": [
            "verify-now.co",
            "bank-kycverify.com",
            "secure-kyc-update.in",
            "pay-safe.link"
        ],
        "number_hashes": [
            # SHA256 hashes of known scam numbers
            hashlib.sha256("+91-86854-63467".encode()).hexdigest()[:16],
            hashlib.sha256("+91-98728-31517".encode()).hexdigest()[:16],
        ],
        "signature": "demo_signature_would_be_real_in_production"
    }


@app.get("/api/stats")
async def get_stats():
    """Get dataset statistics (for demo purposes)."""
    return {
        "total_messages_analyzed": 148,
        "scams_detected": 120,
        "legitimate": 28,
        "scam_types": [
            "Bank/KYC Fraud",
            "Digital Arrest",
            "Lottery/Prize Scam",
            "Tech Support Scam",
            "Job Offer Scam",
            "Family Impersonation"
        ]
    }


# --- Run server ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
