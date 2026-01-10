"""
Unified Detection Pipeline - Combines Rule Engine + ML Classifier
==================================================================
This is the main detection logic that will be ported to Android.

Flow:
1. Rule Engine (deterministic, <10ms) → catches obvious scams
2. ML Classifier (if rule engine unsure) → probabilistic classification
3. Combined result with explainability
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rule_engine import RuleEngine, RuleResult, RiskLevel
from feature_extractor import FeatureExtractor
from classifier import ScamClassifier
from dataclasses import dataclass
from typing import List, Optional
import json


@dataclass
class DetectionResult:
    """Final detection result combining all signals."""
    risk_level: str  # SAFE, SUSPICIOUS, SCAM
    confidence: float
    reason_en: str
    reason_hi: str
    detected_signals: dict
    triggered_rules: List[str]
    ml_probability: Optional[float]
    recommended_action: str
    recommended_action_hi: str


class ScamDetector:
    """
    Unified scam detector combining Rule Engine + ML.
    This is the main API that will be exposed to Android.
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        self.rule_engine = RuleEngine()
        self.feature_extractor = FeatureExtractor()
        self.classifier = None
        
        if model_dir and os.path.exists(model_dir):
            self.classifier = ScamClassifier()
            try:
                self.classifier.load(model_dir)
                print("✓ ML classifier loaded")
            except Exception as e:
                print(f"⚠ Could not load ML classifier: {e}")
                self.classifier = None
    
    def detect(self, text: str, sender_id: Optional[str] = None) -> DetectionResult:
        """
        Main detection method.
        
        Step 1: Run Rule Engine (fast, deterministic)
        Step 2: If uncertain, run ML classifier  
        Step 3: Combine results and generate explanation
        """
        # Step 1: Rule Engine
        rule_result = self.rule_engine.analyze(text, sender_id)
        
        # Extract features for ML
        features = self.feature_extractor.extract(text)
        
        # Step 2: ML classifier (if rule engine is uncertain)
        ml_prob = None
        if self.classifier and rule_result.risk_level == RiskLevel.SUSPICIOUS:
            try:
                ml_result = self.classifier.predict(
                    text, 
                    structured_features=features.to_feature_vector()[:7]  # First 7 flags
                )
                ml_prob = ml_result['scam_probability']
                
                # Adjust confidence based on ML
                if ml_prob > 0.7:
                    rule_result.risk_level = RiskLevel.SCAM
                    rule_result.confidence = max(rule_result.confidence, ml_prob)
                elif ml_prob < 0.3:
                    rule_result.risk_level = RiskLevel.SAFE
                    rule_result.confidence = 1 - ml_prob
            except Exception as e:
                print(f"ML prediction error: {e}")
        
        # Step 3: Generate recommendation
        if rule_result.risk_level == RiskLevel.SCAM:
            action_en = "⚠️ Block sender and report this scam"
            action_hi = "⚠️ भेजने वाले को ब्लॉक करें और रिपोर्ट करें"
        elif rule_result.risk_level == RiskLevel.SUSPICIOUS:
            action_en = "❓ Ask a family member before responding"
            action_hi = "❓ जवाब देने से पहले परिवार से पूछें"
        else:
            action_en = "✅ Message appears safe"
            action_hi = "✅ संदेश सुरक्षित लगता है"
        
        # Combine reasons
        reason_en = "; ".join(rule_result.reasons_en) if rule_result.reasons_en else "No issues detected"
        reason_hi = "; ".join(rule_result.reasons_hi) if rule_result.reasons_hi else "कोई समस्या नहीं मिली"
        
        return DetectionResult(
            risk_level=rule_result.risk_level.value,
            confidence=rule_result.confidence,
            reason_en=reason_en,
            reason_hi=reason_hi,
            detected_signals=rule_result.detected_signals,
            triggered_rules=rule_result.triggered_rules,
            ml_probability=ml_prob,
            recommended_action=action_en,
            recommended_action_hi=action_hi
        )
    
    def to_json(self, result: DetectionResult) -> str:
        """Convert result to JSON for API response."""
        return json.dumps({
            'risk_level': result.risk_level,
            'confidence': round(result.confidence, 2),
            'reason': {
                'en': result.reason_en,
                'hi': result.reason_hi
            },
            'signals': result.detected_signals,
            'rules': result.triggered_rules,
            'ml_probability': round(result.ml_probability, 2) if result.ml_probability else None,
            'action': {
                'en': result.recommended_action,
                'hi': result.recommended_action_hi
            }
        }, indent=2, ensure_ascii=False)


# --- Test ---
if __name__ == "__main__":
    # Find model directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    model_dir = os.path.join(project_dir, 'models')
    
    # Initialize detector
    detector = ScamDetector(model_dir=model_dir)
    
    # Test messages
    test_cases = [
        # Safe messages
        "Hi, I'll reach by 7:00 PM. See you soon.",
        "BESCOM: Bill of ₹5000 due on 15-Jun. Pay via official app.",
        
        # Suspicious
        "Dear customer, your card will be blocked today. Verify at https://rebrand.ly/CqmeGMJ",
        "Pre-approved loan ₹9900. Activate by paying ₹199 at https://tinyurl.com/xyz",
        
        # Scam
        "You are under 'digital arrest'. Stay on line and transfer ₹1500.",
        "OLX: Buyer sent payment. To receive ₹50000, share OTP.",
        "Court notice: Settlement required today. Call +91-86854-63467 abhi.",
        "Hi Mom, new number. Emergency. Send ₹9900 to UPI xyz@oksbi. Don't call.",
    ]
    
    print("=" * 70)
    print("UNIFIED SCAM DETECTOR TEST")
    print("=" * 70)
    
    for msg in test_cases:
        result = detector.detect(msg)
        
        # Color code by risk
        if result.risk_level == "SCAM":
            icon = "🚨"
        elif result.risk_level == "SUSPICIOUS":
            icon = "⚠️"
        else:
            icon = "✅"
        
        print(f"\n{icon} {result.risk_level} ({result.confidence:.0%})")
        print(f"   Message: {msg[:55]}...")
        print(f"   Reason: {result.reason_en}")
        print(f"   Action: {result.recommended_action}")
