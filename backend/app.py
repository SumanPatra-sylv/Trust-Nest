"""
FastAPI Backend - Unified Scam Detection API
=============================================
Exposes the Rule Engine → DistilBERT → Guardian pipeline.

Endpoints:
- POST /api/analyze - Full analysis with explainability
- POST /api/analyze/quick - Rule engine only (fast)
- GET /api/health - Health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector import ScamDetector

# Initialize FastAPI
app = FastAPI(
    title="ScamShield API",
    description="Rule Engine → DistilBERT → Guardian Detection Pipeline",
    version="2.0.0"
)

# CORS
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
model_dir = os.path.join(project_dir, "models")
detector = ScamDetector(model_dir=model_dir)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class AnalyzeRequest(BaseModel):
    text: str
    sender_id: Optional[str] = None


class AnalyzeResponse(BaseModel):
    # Final verdict
    verdict: str
    confidence: float
    
    # Rule results
    rule_triggered: bool
    rule_triggers: List[str]
    rule_reasons_en: List[str]
    rule_reasons_hi: List[str]
    
    # ML results
    ml_label: Optional[str]
    ml_confidence: Optional[float]
    ml_used: bool
    
    # Explainability
    explanation_en: str
    explanation_hi: str
    action_en: str
    action_hi: str
    
    # Guardian
    should_escalate: bool
    escalation_reason: Optional[str]


class QuickAnalyzeResponse(BaseModel):
    verdict: str
    confidence: float
    rule_triggers: List[str]
    explanation: str


class HealthResponse(BaseModel):
    status: str
    rule_engine: bool
    distilbert: bool
    distilbert_path: Optional[str]


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check system health and model availability."""
    return HealthResponse(
        status="healthy",
        rule_engine=True,  # Always available
        distilbert=detector.distilbert is not None and detector.distilbert.loaded,
        distilbert_path=detector.distilbert.model_path if detector.distilbert else None
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_message(request: AnalyzeRequest):
    """
    Full analysis with Rule Engine → DistilBERT → Guardian pipeline.
    
    Returns complete explainability including:
    - Rule triggers
    - ML label + confidence
    - Bilingual explanations
    - Guardian escalation decision
    """
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Empty message text")
    
    result = detector.detect(request.text, request.sender_id)
    
    return AnalyzeResponse(
        verdict=result.verdict,
        confidence=result.confidence,
        rule_triggered=result.rule_triggered,
        rule_triggers=result.rule_triggers,
        rule_reasons_en=result.rule_reasons_en,
        rule_reasons_hi=result.rule_reasons_hi,
        ml_label=result.ml_label,
        ml_confidence=result.ml_confidence,
        ml_used=result.ml_used,
        explanation_en=result.explanation_en,
        explanation_hi=result.explanation_hi,
        action_en=result.action_en,
        action_hi=result.action_hi,
        should_escalate=result.should_escalate,
        escalation_reason=result.escalation_reason
    )


@app.post("/api/analyze/quick", response_model=QuickAnalyzeResponse)
async def quick_analyze(request: AnalyzeRequest):
    """
    Quick analysis using Rule Engine only (no ML).
    Use for high-throughput or when DistilBERT is unavailable.
    """
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Empty message text")
    
    # Use rule engine directly
    from rule_engine import RuleEngine
    engine = RuleEngine()
    result = engine.analyze(request.text, request.sender_id)
    
    return QuickAnalyzeResponse(
        verdict=result.risk_level.value,
        confidence=result.confidence,
        rule_triggers=result.triggered_rules,
        explanation="; ".join(result.reasons_en) if result.reasons_en else "No issues detected"
    )


@app.get("/")
async def root():
    """API info."""
    return {
        "service": "ScamShield Detection API",
        "version": "2.0.0",
        "pipeline": "Rule Engine → DistilBERT → Guardian",
        "endpoints": {
            "analyze": "/api/analyze",
            "quick": "/api/analyze/quick",
            "health": "/api/health"
        }
    }


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
