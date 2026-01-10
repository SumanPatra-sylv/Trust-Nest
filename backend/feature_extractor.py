"""
Feature Extractor - Extract structured signals from text for ML models
======================================================================
Extracts the boolean flags that the dataset provides (has_url, has_upi, etc.)
plus additional linguistic features for the TinyBERT classifier.

These features are used as auxiliary inputs alongside the text embedding.
"""

import re
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class ExtractedFeatures:
    """Structured features extracted from message text."""
    # Binary signal flags (matching dataset columns)
    has_url: bool = False
    has_upi: bool = False
    has_otp: bool = False
    has_qr: bool = False
    has_phone: bool = False
    has_threat: bool = False
    has_urgency: bool = False
    
    # Additional features for ML
    has_money_amount: bool = False
    has_bank_name: bool = False
    has_app_name: bool = False
    message_length: int = 0
    caps_ratio: float = 0.0
    special_char_ratio: float = 0.0
    
    # Language detection
    is_hinglish: bool = False
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_feature_vector(self) -> List[float]:
        """Convert to numerical vector for ML model input."""
        return [
            float(self.has_url),
            float(self.has_upi),
            float(self.has_otp),
            float(self.has_qr),
            float(self.has_phone),
            float(self.has_threat),
            float(self.has_urgency),
            float(self.has_money_amount),
            float(self.has_bank_name),
            float(self.has_app_name),
            min(self.message_length / 500.0, 1.0),  # Normalized length
            self.caps_ratio,
            self.special_char_ratio,
            float(self.is_hinglish),
        ]


class FeatureExtractor:
    """Extract structured features from text messages."""
    
    def __init__(self):
        # URL pattern
        self.url_pattern = re.compile(r'https?://[^\s]+', re.IGNORECASE)
        
        # UPI pattern
        self.upi_pattern = re.compile(r'[a-zA-Z0-9._-]+@[a-zA-Z]+', re.IGNORECASE)
        
        # OTP patterns
        self.otp_patterns = [
            r'\botp\b', r'\bpin\b', r'one.?time.?password', r'verification.?code',
            r'share.?otp', r'confirm.?otp', r'\d{4,6}\s*(code|otp)?'
        ]
        self.otp_regex = re.compile('|'.join(self.otp_patterns), re.IGNORECASE)
        
        # QR patterns
        self.qr_patterns = [r'\bqr\b', r'qr.?code', r'scan.?qr']
        self.qr_regex = re.compile('|'.join(self.qr_patterns), re.IGNORECASE)
        
        # Phone number pattern (Indian)
        self.phone_pattern = re.compile(r'\+91[-\s]?\d{5}[-\s]?\d{5}|\d{10}')
        
        # Threat patterns
        self.threat_patterns = [
            r'block', r'freeze', r'suspend', r'deactivat', r'disconnect',
            r'legal.?action', r'fir', r'police', r'arrest', r'court',
            r'power.?cut', r'penalty', r'fine'
        ]
        self.threat_regex = re.compile('|'.join(self.threat_patterns), re.IGNORECASE)
        
        # Urgency patterns (English + Hindi)
        self.urgency_patterns = [
            r'urgent', r'immediately', r'today', r'now', r'hurry', r'fast',
            r'within.?\d+.?(hour|minute|min|hr)', r'expires?', r'last.?chance',
            r'jaldi', r'abhi', r'turant', r'fauran'
        ]
        self.urgency_regex = re.compile('|'.join(self.urgency_patterns), re.IGNORECASE)
        
        # Money amount pattern (Indian Rupees)
        self.money_pattern = re.compile(r'₹\s*[\d,]+|rs\.?\s*[\d,]+|\d+\s*rupees?', re.IGNORECASE)
        
        # Bank names
        self.banks = [
            'sbi', 'hdfc', 'icici', 'axis', 'kotak', 'pnb', 'canara', 'bob',
            'idbi', 'yes bank', 'indusind', 'rbl', 'federal', 'bandhan'
        ]
        self.bank_regex = re.compile('|'.join(self.banks), re.IGNORECASE)
        
        # App names (often impersonated)
        self.apps = [
            'paytm', 'phonepe', 'gpay', 'google pay', 'amazon', 'flipkart',
            'whatsapp', 'anydesk', 'teamviewer', 'quicksupport'
        ]
        self.app_regex = re.compile('|'.join(self.apps), re.IGNORECASE)
        
        # Hindi/Hinglish indicators
        self.hinglish_words = [
            'karo', 'karo', 'abhi', 'jaldi', 'hai', 'haan', 'nahi', 'kya',
            'aap', 'aapka', 'batao', 'bata', 'bhejo', 'dijiye', 'karein'
        ]
        self.hinglish_regex = re.compile(r'\b(' + '|'.join(self.hinglish_words) + r')\b', re.IGNORECASE)
    
    def extract(self, text: str) -> ExtractedFeatures:
        """Extract all features from a text message."""
        features = ExtractedFeatures()
        
        # Basic signals
        features.has_url = bool(self.url_pattern.search(text))
        features.has_upi = bool(self.upi_pattern.search(text))
        features.has_otp = bool(self.otp_regex.search(text))
        features.has_qr = bool(self.qr_regex.search(text))
        features.has_phone = bool(self.phone_pattern.search(text))
        features.has_threat = bool(self.threat_regex.search(text))
        features.has_urgency = bool(self.urgency_regex.search(text))
        
        # Additional features
        features.has_money_amount = bool(self.money_pattern.search(text))
        features.has_bank_name = bool(self.bank_regex.search(text))
        features.has_app_name = bool(self.app_regex.search(text))
        
        # Length and character analysis
        features.message_length = len(text)
        
        alpha_chars = sum(1 for c in text if c.isalpha())
        if alpha_chars > 0:
            features.caps_ratio = sum(1 for c in text if c.isupper()) / alpha_chars
        
        total_chars = len(text)
        if total_chars > 0:
            features.special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / total_chars
        
        # Language detection
        features.is_hinglish = bool(self.hinglish_regex.search(text))
        
        return features


# --- Test ---
if __name__ == "__main__":
    extractor = FeatureExtractor()
    
    test_messages = [
        "TRAI verification pending. Confirm details by calling +91-98728-31517.",
        "Pre-approved loan ₹9900. Activate by paying processing ₹199 at https://tinyurl.com/eL2ZL5L",
        "OLX: Buyer sent payment. To receive ₹50000, share OTP.",
        "Jaldi karo, account block ho jayega. OTP batao abhi.",
        "Hi, I'll reach by 7:00 PM. See you soon.",
    ]
    
    print("=" * 60)
    print("FEATURE EXTRACTOR TEST")
    print("=" * 60)
    
    for msg in test_messages:
        features = extractor.extract(msg)
        print(f"\n📩 Message: {msg[:50]}...")
        print(f"   Features: {features.to_dict()}")
        print(f"   Vector: {features.to_feature_vector()}")
