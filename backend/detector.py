"""
Unified Detection Pipeline - Rule Engine → DistilBERT → Guardian
================================================================
This is the main detection logic with strict ordering:

1. Rule Engine (ALWAYS runs first, can OVERRIDE ML)
2. DistilBERT (runs for uncertain cases)
3. Guardian Escalation (for high-risk)

Output includes:
- Rule triggers
- DistilBERT label + confidence
- Combined verdict with explainability
"""

import os
import sys
import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from enum import Enum

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rule_engine import RuleEngine, RuleResult, RiskLevel

# Try to import DistilBERT
DISTILBERT_AVAILABLE = False
try:
    import torch
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    DISTILBERT_AVAILABLE = True
except ImportError:
    print("[WARN] Transformers not available. Using rule engine only.")


class FinalVerdict(Enum):
    SAFE = "SAFE"
    SUSPICIOUS = "SUSPICIOUS"
    SCAM = "SCAM"


@dataclass
class DistilBERTResult:
    """Result from DistilBERT inference."""
    label: str  # SAFE or SCAM
    confidence: float
    raw_logits: Optional[List[float]] = None


@dataclass 
class DetectionResult:
    """Final combined detection result with full explainability."""
    # Final verdict
    verdict: str  # SAFE, SUSPICIOUS, SCAM
    confidence: float
    
    # Rule Engine results
    rule_triggered: bool
    rule_triggers: List[str]
    rule_reasons_en: List[str]
    rule_reasons_hi: List[str]
    rule_signals: Dict[str, bool]
    
    # DistilBERT results
    ml_label: Optional[str]
    ml_confidence: Optional[float]
    ml_used: bool
    
    # Combined explainability
    explanation_en: str
    explanation_hi: str
    
    # Recommended action
    action_en: str
    action_hi: str
    
    # Guardian escalation
    should_escalate: bool
    escalation_reason: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class DistilBERTClassifier:
    """DistilBERT classifier for scam detection."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded = False
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained DistilBERT model."""
        if not os.path.exists(self.model_path):
            print(f"[WARN] Model not found at {self.model_path}")
            return
        
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
            self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True
            print(f"[INFO] DistilBERT loaded from {self.model_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load DistilBERT: {e}")
    
    def predict(self, text: str) -> DistilBERTResult:
        """Run inference on a single text."""
        if not self.loaded:
            return DistilBERTResult(label="UNKNOWN", confidence=0.0)
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_class].item()
        
        # Map to label (0=SAFE, 1=SCAM)
        label = "SCAM" if pred_class == 1 else "SAFE"
        
        return DistilBERTResult(
            label=label,
            confidence=confidence,
            raw_logits=logits[0].tolist()
        )


class ScamDetector:
    """
    Unified Scam Detector with strict ordering:
    
    1. Rule Engine (ALWAYS first, can OVERRIDE)
    2. DistilBERT (for uncertain cases)
    3. Guardian Escalation (for high-risk)
    """
    
    # Thresholds for decision making
    RULE_SCAM_THRESHOLD = 60      # Rule score >= 60 = definite SCAM
    RULE_SUSPICIOUS_THRESHOLD = 30 # Rule score >= 30 = SUSPICIOUS
    ML_HIGH_CONFIDENCE = 0.8      # ML confidence >= 0.8 = trust it
    ML_LOW_CONFIDENCE = 0.6       # ML confidence < 0.6 = uncertain
    ESCALATION_THRESHOLD = 0.7    # Escalate to guardian if >= 0.7
    
    def __init__(self, model_dir: Optional[str] = None):
        # Always initialize Rule Engine
        self.rule_engine = RuleEngine()
        
        # Initialize DistilBERT if available
        self.distilbert = None
        if model_dir and DISTILBERT_AVAILABLE:
            distilbert_path = os.path.join(model_dir, "distilbert")
            if os.path.exists(distilbert_path):
                self.distilbert = DistilBERTClassifier(distilbert_path)
    
    def detect(self, text: str, sender_id: Optional[str] = None) -> DetectionResult:
        """
        Main detection method with strict ordering:
        
        STEP 1: Rule Engine (ALWAYS runs, can OVERRIDE)
        STEP 2: DistilBERT (if rules are uncertain)
        STEP 3: Combine results
        STEP 4: Determine guardian escalation
        """
        
        # =====================================================
        # STEP 1: RULE ENGINE (ALWAYS FIRST)
        # =====================================================
        rule_result = self.rule_engine.analyze(text, sender_id)
        
        rule_triggered = len(rule_result.triggered_rules) > 0
        rule_score = self._calculate_rule_score(rule_result)
        
        # Check for whitelist (legitimate messages that ML might misclassify)
        is_whitelisted = rule_result.detected_signals.get('is_whitelisted', False)
        is_lottery_scam = rule_result.detected_signals.get('is_lottery_scam', False)
        
        # Check if rules definitively determine outcome
        rule_override = False
        if is_whitelisted:
            # WHITELIST: Known safe pattern (e.g., OTP notification, balance check)
            rule_override = True
            final_verdict = FinalVerdict.SAFE
            final_confidence = 0.90
        elif is_lottery_scam:
            # LOTTERY SCAM: High confidence scam
            rule_override = True
            final_verdict = FinalVerdict.SCAM
            final_confidence = 0.85
        elif rule_score >= self.RULE_SCAM_THRESHOLD:
            # Rule Engine says SCAM - this OVERRIDES ML
            rule_override = True
            final_verdict = FinalVerdict.SCAM
            final_confidence = min(0.95, 0.6 + (rule_score - 60) * 0.01)
        else:
            # Let ML decide - rules are not strong enough to override
            rule_override = False
            final_verdict = None
            final_confidence = None
        
        # =====================================================
        # STEP 2: DISTILBERT (ALWAYS RUN)
        # =====================================================
        ml_result = None
        ml_used = False
        
        # ALWAYS run DistilBERT
        if self.distilbert and self.distilbert.loaded:
            ml_result = self.distilbert.predict(text)
            ml_used = not rule_override
        
        # If rules didn't override, use ML to determine verdict
        if not rule_override and ml_result:
            if ml_result.label == "SCAM":
                # ML says SCAM
                if ml_result.confidence >= 0.7:
                    # High confidence SCAM - definite scam
                    final_verdict = FinalVerdict.SCAM
                    final_confidence = ml_result.confidence
                elif ml_result.confidence >= 0.55:
                    # Medium confidence SCAM (55-70%) - suspicious
                    final_verdict = FinalVerdict.SUSPICIOUS
                    final_confidence = ml_result.confidence
                elif rule_triggered:
                    # Low confidence SCAM but rules also found something - suspicious
                    final_verdict = FinalVerdict.SUSPICIOUS
                    final_confidence = max(0.5, ml_result.confidence)
                else:
                    # Low confidence SCAM with NO rule backup - too uncertain, call SAFE
                    # 53% SCAM with 0 rules = ML is guessing, don't alarm user
                    final_verdict = FinalVerdict.SAFE
                    final_confidence = 1 - ml_result.confidence  # Flip to SAFE confidence
            else:
                # ML says SAFE
                if rule_score >= self.RULE_SUSPICIOUS_THRESHOLD:
                    # Rules found something - be cautious
                    final_verdict = FinalVerdict.SUSPICIOUS
                    final_confidence = 0.5 + (rule_score - 30) * 0.01
                else:
                    final_verdict = FinalVerdict.SAFE
                    final_confidence = ml_result.confidence
        
        # Fallback if ML not available
        if final_verdict is None:
            if rule_score >= self.RULE_SUSPICIOUS_THRESHOLD:
                final_verdict = FinalVerdict.SUSPICIOUS
                final_confidence = 0.5 + (rule_score - 30) * 0.01
            elif rule_triggered:
                final_verdict = FinalVerdict.SUSPICIOUS
                final_confidence = 0.5
            else:
                final_verdict = FinalVerdict.SAFE
                final_confidence = 0.7
        
        # =====================================================
        # STEP 3: GENERATE EXPLAINABILITY
        # =====================================================
        explanation_en, explanation_hi = self._generate_explanation(
            rule_result, ml_result, final_verdict, rule_override
        )
        
        action_en, action_hi = self._generate_action(final_verdict)
        
        # =====================================================
        # STEP 4: GUARDIAN ESCALATION
        # =====================================================
        should_escalate = (
            final_verdict == FinalVerdict.SCAM or
            (final_verdict == FinalVerdict.SUSPICIOUS and final_confidence >= self.ESCALATION_THRESHOLD) or
            "DIGITAL_ARREST" in rule_result.triggered_rules or
            "FAMILY_IMPERSONATION" in rule_result.triggered_rules
        )
        
        escalation_reason = None
        if should_escalate:
            if "DIGITAL_ARREST" in rule_result.triggered_rules:
                escalation_reason = "Digital Arrest scam detected"
            elif "FAMILY_IMPERSONATION" in rule_result.triggered_rules:
                escalation_reason = "Family impersonation detected"
            elif final_verdict == FinalVerdict.SCAM:
                escalation_reason = f"High-risk scam ({final_confidence:.0%} confidence)"
            else:
                escalation_reason = "Suspicious activity requires review"
        
        # =====================================================
        # BUILD FINAL RESULT
        # =====================================================
        return DetectionResult(
            verdict=final_verdict.value,
            confidence=final_confidence,
            rule_triggered=rule_triggered,
            rule_triggers=rule_result.triggered_rules,
            rule_reasons_en=rule_result.reasons_en,
            rule_reasons_hi=rule_result.reasons_hi,
            rule_signals=rule_result.detected_signals,
            ml_label=ml_result.label if ml_result else None,
            ml_confidence=ml_result.confidence if ml_result else None,
            ml_used=ml_used,
            explanation_en=explanation_en,
            explanation_hi=explanation_hi,
            action_en=action_en,
            action_hi=action_hi,
            should_escalate=should_escalate,
            escalation_reason=escalation_reason
        )
    
    def _calculate_rule_score(self, result: RuleResult) -> int:
        """Calculate a numeric score from rule triggers."""
        score = 0
        weights = {
            "DIGITAL_ARREST": 50,
            "FAMILY_IMPERSONATION": 45,
            "PHISHING_DOMAIN": 40,
            "OTP_REQUEST": 35,
            "THREAT": 30,
            "AUTHORITY_IMPERSONATION": 25,
            "URL_SHORTENER": 20,
            "UPI_PRESENT": 15,
            "URGENCY": 15,
            "PHONE_NUMBER": 10
        }
        for rule in result.triggered_rules:
            score += weights.get(rule, 10)
        return score
    
    def _generate_explanation(
        self, 
        rule_result: RuleResult, 
        ml_result: Optional[DistilBERTResult],
        verdict: FinalVerdict,
        rule_override: bool
    ) -> tuple:
        """Generate bilingual explanation."""
        parts_en = []
        parts_hi = []
        
        # Rule triggers
        if rule_result.reasons_en:
            parts_en.extend(rule_result.reasons_en)
            parts_hi.extend(rule_result.reasons_hi)
        
        # ML contribution
        if ml_result and not rule_override:
            ml_desc = f"AI analysis: {ml_result.label} ({ml_result.confidence:.0%})"
            parts_en.append(ml_desc)
            parts_hi.append(f"AI विश्लेषण: {ml_result.label} ({ml_result.confidence:.0%})")
        
        if not parts_en:
            parts_en.append("No issues detected")
            parts_hi.append("कोई समस्या नहीं मिली")
        
        return "; ".join(parts_en), "; ".join(parts_hi)
    
    def _generate_action(self, verdict: FinalVerdict) -> tuple:
        """Generate recommended action."""
        if verdict == FinalVerdict.SCAM:
            return (
                "⚠️ Block sender and report this scam",
                "⚠️ भेजने वाले को ब्लॉक करें और रिपोर्ट करें"
            )
        elif verdict == FinalVerdict.SUSPICIOUS:
            return (
                "❓ Ask a family member before responding",
                "❓ जवाब देने से पहले परिवार से पूछें"
            )
        else:
            return (
                "✅ Message appears safe",
                "✅ संदेश सुरक्षित लगता है"
            )


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    model_dir = os.path.join(project_dir, "models")
    
    detector = ScamDetector(model_dir=model_dir)
    
    test_cases = [
        # Safe
        "Hi, I'll reach by 7 PM. See you soon.",
        "Your Amazon order has been shipped. Track at amazon.in",
        
        # Suspicious
        "Pre-approved loan of Rs. 50000. Limited offer.",
        
        # Scam (should trigger rules)
        "Share OTP to verify your payment of Rs. 5000",
        "Digital arrest. Stay on video call. Transfer money now.",
        "Court notice: Settlement required today. Call +91-86854-63467",
        "Hi Mom, new number. Emergency. Send Rs. 9900 to xyz@oksbi",
    ]
    
    print("=" * 70)
    print("UNIFIED DETECTOR TEST (Rule Engine → DistilBERT → Guardian)")
    print("=" * 70)
    
    for text in test_cases:
        result = detector.detect(text)
        
        icon = {"SCAM": "🚨", "SUSPICIOUS": "⚠️", "SAFE": "✅"}[result.verdict]
        
        print(f"\n{icon} {result.verdict} ({result.confidence:.0%})")
        print(f"   Text: {text[:50]}...")
        print(f"   Rules: {', '.join(result.rule_triggers) if result.rule_triggers else 'None'}")
        print(f"   ML: {result.ml_label} ({result.ml_confidence:.0%})" if result.ml_used else "   ML: Not used (rule override)")
        print(f"   Explanation: {result.explanation_en}")
        print(f"   Escalate: {result.should_escalate} - {result.escalation_reason or 'N/A'}")
