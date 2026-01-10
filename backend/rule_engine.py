"""
Rule Engine - Deterministic Scam Detection (On-Device First)
=============================================================
This runs FIRST, before any ML model. Fast, interpretable, no false negatives on obvious scams.
Target: <10ms execution on low-end phones.

Will be ported to Kotlin for Android on-device use.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum


class RiskLevel(Enum):
    SAFE = "SAFE"
    SUSPICIOUS = "SUSPICIOUS"
    SCAM = "SCAM"


@dataclass
class RuleResult:
    risk_level: RiskLevel
    confidence: float  # 0.0 to 1.0
    reasons_en: List[str]
    reasons_hi: List[str]
    detected_signals: dict
    triggered_rules: List[str]


class RuleEngine:
    """
    Deterministic rule-based scam detector.
    Order matters: rules are checked in priority order.
    """
    
    def __init__(self):
        # URL shorteners commonly used in scams
        self.url_shorteners = {
            'bit.ly', 'tinyurl.com', 'rebrand.ly', 'cutt.ly', 't.co',
            'goo.gl', 'ow.ly', 'is.gd', 'buff.ly', 'short.link'
        }
        
        # Suspicious domain patterns (phishing)
        self.suspicious_domains = [
            r'verify-now\.co', r'secure-kyc-update\.in', r'bank-kycverify\.com',
            r'track-parcel\.in', r'customs-clearance\.in', r'pay-safe\.link',
            r'gpay-help\.in', r'support-helpdesk\.in', r'cybercrime-case\.in'
        ]
        
        # UPI ID patterns (legitimate vs scam)
        self.upi_pattern = re.compile(r'[a-zA-Z0-9._-]+@[a-zA-Z]+', re.IGNORECASE)
        self.legitimate_upi_suffixes = {'okaxis', 'oksbi', 'okicici', 'okhdfc', 'paytm', 'ybl', 'ibl', 'upi'}
        
        # OTP-related phrases (English + Hindi)
        self.otp_phrases = [
            r'\botp\b', r'\bpin\b', r'one.?time.?password', r'verification.?code',
            r'share.?otp', r'tell.?otp', r'otp.?share', r'otp.?batao', r'otp.?bata',
            r'code.?share', r'code.?batao'
        ]
        
        # Urgency phrases
        self.urgency_phrases_en = [
            r'today only', r'expires today', r'within \d+ hours?', r'immediately',
            r'urgent', r'now or never', r'last chance', r'act now', r'hurry',
            r'limited time', r'before midnight', r'within \d+ minutes?'
        ]
        self.urgency_phrases_hi = [
            r'jaldi', r'abhi', r'turant', r'aaj hi', r'fauran', r'jald se jald'
        ]
        
        # Threat phrases
        self.threat_phrases_en = [
            r'account.?freeze', r'account.?block', r'will be.?blocked',
            r'legal.?action', r'fir.?filed?', r'police.?complaint', r'arrested?',
            r'digital.?arrest', r'under.?arrest', r'cyber.?crime', r'money.?laundering',
            r'power.?cut', r'disconnection?', r'deactivat', r'suspend'
        ]
        self.threat_phrases_hi = [
            r'giraftaar', r'arrest', r'kanoon', r'police', r'jail'
        ]
        
        # Authority impersonation
        self.authority_patterns = [
            r'\b(cbi|police|trai|income.?tax|cyber.?crime|rbi|sebi)\b',
            r'police.?control.?room', r'crime.?branch', r'enforcement.?directorate',
            r'court.?notice', r'settlement.?required', r'legal.?notice'
        ]
        
        # Family impersonation patterns
        self.family_impersonation_patterns = [
            r'hi.?mom', r'hi.?dad', r'hi.?papa', r'hi.?mummy',
            r'new.?number', r'phone.?lost', r'emergency',
            r"don'?t.?call", r'send.?money', r'transfer.?now',
            r"it'?s.?me", r'trouble', r'help.?me'
        ]
        
        # Known scam sender patterns
        self.scam_sender_patterns = [
            r'^\+91-\d{5}-\d{5}$',  # Random Indian mobile format
        ]
        
        # Compile all patterns for speed
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for faster matching."""
        self.otp_regex = re.compile('|'.join(self.otp_phrases), re.IGNORECASE)
        self.urgency_en_regex = re.compile('|'.join(self.urgency_phrases_en), re.IGNORECASE)
        self.urgency_hi_regex = re.compile('|'.join(self.urgency_phrases_hi), re.IGNORECASE)
        self.threat_en_regex = re.compile('|'.join(self.threat_phrases_en), re.IGNORECASE)
        self.threat_hi_regex = re.compile('|'.join(self.threat_phrases_hi), re.IGNORECASE)
        self.authority_regex = re.compile('|'.join(self.authority_patterns), re.IGNORECASE)
        self.suspicious_domain_regex = re.compile('|'.join(self.suspicious_domains), re.IGNORECASE)
    
    def analyze(self, text: str, sender_id: Optional[str] = None) -> RuleResult:
        """
        Analyze a message using deterministic rules.
        Returns risk level, confidence, and explanations.
        """
        text_lower = text.lower()
        signals = {
            'has_url': False,
            'has_shortener': False,
            'has_suspicious_domain': False,
            'has_upi': False,
            'has_otp_request': False,
            'has_urgency': False,
            'has_threat': False,
            'has_authority_claim': False,
            'has_phone_number': False,
        }
        reasons_en = []
        reasons_hi = []
        triggered_rules = []
        scam_score = 0
        
        # --- Rule 1: Check for URLs and shorteners ---
        urls = re.findall(r'https?://[^\s]+', text, re.IGNORECASE)
        if urls:
            signals['has_url'] = True
            for url in urls:
                for shortener in self.url_shorteners:
                    if shortener in url.lower():
                        signals['has_shortener'] = True
                        scam_score += 20
                        reasons_en.append("Contains shortened URL (often used to hide destination)")
                        reasons_hi.append("छोटा URL है (असली साइट छुपाने के लिए)")
                        triggered_rules.append("URL_SHORTENER")
                        break
                
                if self.suspicious_domain_regex.search(url):
                    signals['has_suspicious_domain'] = True
                    scam_score += 40
                    reasons_en.append("Contains known phishing domain")
                    reasons_hi.append("फिशिंग वेबसाइट है")
                    triggered_rules.append("PHISHING_DOMAIN")
        
        # --- Rule 2: Check for UPI IDs ---
        upi_matches = self.upi_pattern.findall(text)
        if upi_matches:
            signals['has_upi'] = True
            scam_score += 15
            reasons_en.append("Contains UPI payment ID")
            reasons_hi.append("UPI पेमेंट ID है")
            triggered_rules.append("UPI_PRESENT")
        
        # --- Rule 3: Check for OTP requests ---
        if self.otp_regex.search(text):
            signals['has_otp_request'] = True
            scam_score += 35
            reasons_en.append("Asks for OTP - Banks NEVER ask for OTP")
            reasons_hi.append("OTP माँग रहा है - बैंक कभी OTP नहीं माँगते")
            triggered_rules.append("OTP_REQUEST")
        
        # --- Rule 4: Check for urgency ---
        if self.urgency_en_regex.search(text) or self.urgency_hi_regex.search(text):
            signals['has_urgency'] = True
            scam_score += 15
            reasons_en.append("Creates false urgency")
            reasons_hi.append("जल्दी करने का दबाव बना रहा है")
            triggered_rules.append("URGENCY")
        
        # --- Rule 5: Check for threats ---
        if self.threat_en_regex.search(text) or self.threat_hi_regex.search(text):
            signals['has_threat'] = True
            scam_score += 30
            reasons_en.append("Contains threatening language")
            reasons_hi.append("धमकी दे रहा है")
            triggered_rules.append("THREAT")
        
        # --- Rule 6: Check for authority impersonation ---
        if self.authority_regex.search(text):
            signals['has_authority_claim'] = True
            scam_score += 25
            reasons_en.append("Claims to be from government/police - they don't call/message like this")
            reasons_hi.append("सरकार/पुलिस बताने का दावा - वे ऐसे संपर्क नहीं करते")
            triggered_rules.append("AUTHORITY_IMPERSONATION")
        
        # --- Rule 7: Check for phone numbers (potential callback scam) ---
        phone_pattern = re.compile(r'\+91[-\s]?\d{5}[-\s]?\d{5}')
        if phone_pattern.search(text):
            signals['has_phone_number'] = True
            scam_score += 10
            triggered_rules.append("PHONE_NUMBER")
        
        # --- Rule 8: Digital Arrest specific patterns ---
        digital_arrest_patterns = [
            r'digital.?arrest', r'stay.?on.?line', r'do.?not.?disconnect',
            r'video.?call', r'screen.?shar', r'install.?anydesk', r'install.?teamviewer'
        ]
        for pattern in digital_arrest_patterns:
            if re.search(pattern, text_lower):
                scam_score += 50
                reasons_en.append("CRITICAL: 'Digital Arrest' scam - Police NEVER video call")
                reasons_hi.append("खतरा: 'डिजिटल अरेस्ट' स्कैम - पुलिस कभी वीडियो कॉल नहीं करती")
                triggered_rules.append("DIGITAL_ARREST")
                break
        
        # --- Rule 9: Family Impersonation patterns ---
        family_score = 0
        family_indicators = ['new.?number', 'phone.?lost', "don'?t.?call", 'emergency', "it'?s.?me"]
        for pattern in family_indicators:
            if re.search(pattern, text_lower):
                family_score += 1
        
        if family_score >= 2 and signals['has_upi']:
            scam_score += 45
            reasons_en.append("SUSPICIOUS: Family impersonation + payment request. Verify by calling old number!")
            reasons_hi.append("संदेह: परिवार का नकली संदेश + पैसे माँग रहा है। पुराने नंबर पर कॉल करके पुष्टि करें!")
            triggered_rules.append("FAMILY_IMPERSONATION")
        
        # --- Determine final risk level ---
        if scam_score >= 60:
            risk_level = RiskLevel.SCAM
            confidence = min(0.95, 0.6 + (scam_score - 60) * 0.01)
        elif scam_score >= 30:
            risk_level = RiskLevel.SUSPICIOUS
            confidence = 0.5 + (scam_score - 30) * 0.01
        else:
            risk_level = RiskLevel.SAFE
            confidence = max(0.3, 1.0 - scam_score * 0.02)
            if not reasons_en:
                reasons_en.append("No scam indicators detected")
                reasons_hi.append("कोई स्कैम संकेत नहीं मिला")
        
        return RuleResult(
            risk_level=risk_level,
            confidence=confidence,
            reasons_en=reasons_en,
            reasons_hi=reasons_hi,
            detected_signals=signals,
            triggered_rules=triggered_rules
        )


# --- Quick test ---
if __name__ == "__main__":
    engine = RuleEngine()
    
    # Test cases from the dataset
    test_messages = [
        "TRAI verification pending. Confirm details by calling +91-98728-31517.",
        "Hi, I'll reach by 7:00 PM. See you soon.",
        "Dear customer, your card will be blocked today. Verify details at https://rebrand.ly/CqmeGMJ",
        "You are under 'digital arrest'. Stay on line and transfer ₹1500 to verify innocence.",
        "OLX: Buyer sent payment. To receive ₹50000, share OTP.",
        "BESCOM: Bill of ₹5000 due on 15-Jun. Pay via official app or https://icicibank.com/shzssaj386",
    ]
    
    print("=" * 60)
    print("RULE ENGINE TEST")
    print("=" * 60)
    
    for msg in test_messages:
        result = engine.analyze(msg)
        print(f"\n📩 Message: {msg[:60]}...")
        print(f"   Risk: {result.risk_level.value} ({result.confidence:.0%})")
        print(f"   Reason (EN): {result.reasons_en[0] if result.reasons_en else 'None'}")
        print(f"   Reason (HI): {result.reasons_hi[0] if result.reasons_hi else 'None'}")
        print(f"   Rules: {', '.join(result.triggered_rules)}")
