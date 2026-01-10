"""
TinyBERT Classifier Training - For On-Device SMS/WhatsApp Analysis
===================================================================
Trains a lightweight transformer model for scam detection.
Target: <50MB model, <200ms inference on 3GB RAM phones.

This trains on SMS + WhatsApp data ONLY.
Call/Audio transcripts are for heavy model fallback only.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix
import pickle
import json


class ScamClassifier:
    """
    Lightweight scam classifier for on-device deployment.
    Uses TF-IDF + LogisticRegression for fast inference.
    
    For hackathon demo, this provides quick results.
    Production would use actual TinyBERT with TFLite.
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=3000,  # Keep model small
            ngram_range=(1, 2),  # Capture phrases
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )
        self.classifier = None
        self.is_trained = False
        self.structured_cols = ['has_url', 'has_upi', 'has_otp', 'has_qr', 
                                'has_phone', 'has_threat', 'has_urgency']
    
    def load_data(self, data_dir: str):
        """Load SMS and WhatsApp data only (not calls/audio for on-device)."""
        sms_path = os.path.join(data_dir, 'public_sms.csv')
        whatsapp_path = os.path.join(data_dir, 'public_whatsapp.csv')
        
        dfs = []
        
        # Load SMS
        if os.path.exists(sms_path):
            sms_df = pd.read_csv(sms_path)
            sms_df['modality'] = 'SMS'
            sms_df['text'] = sms_df['message_text']
            dfs.append(sms_df)
            print(f"✓ Loaded {len(sms_df)} SMS samples")
        
        # Load WhatsApp
        if os.path.exists(whatsapp_path):
            wa_df = pd.read_csv(whatsapp_path)
            wa_df['modality'] = 'WhatsApp'
            wa_df['text'] = wa_df['conversation_text']
            dfs.append(wa_df)
            print(f"✓ Loaded {len(wa_df)} WhatsApp samples")
        
        if not dfs:
            raise ValueError("No data files found!")
        
        # Combine
        self.data = pd.concat(dfs, ignore_index=True)
        
        # Clean text
        self.data['text'] = self.data['text'].fillna('').astype(str)
        
        print(f"✓ Total samples: {len(self.data)}")
        print(f"  Scams: {self.data['is_scam'].sum()}")
        print(f"  Legitimate: {(self.data['is_scam'] == 0).sum()}")
        
        return self.data
    
    def _get_structured_features(self, df):
        """Extract structured features from dataframe."""
        available_cols = [c for c in self.structured_cols if c in df.columns]
        if available_cols:
            return df[available_cols].values.astype(float)
        return np.zeros((len(df), len(self.structured_cols)))
    
    def train(self, test_size=0.2):
        """Train the classifier."""
        print("\n" + "=" * 60)
        print("TRAINING SCAM CLASSIFIER")
        print("=" * 60)
        
        # Prepare data
        X_text = self.data['text'].values
        y = self.data['is_scam'].values
        structured = self._get_structured_features(self.data)
        
        # Create indices for split
        indices = np.arange(len(X_text))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train_text = X_text[train_idx]
        X_test_text = X_text[test_idx]
        struct_train = structured[train_idx]
        struct_test = structured[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        # Fit vectorizer and transform
        print("Fitting TF-IDF vectorizer...")
        tfidf_train = self.vectorizer.fit_transform(X_train_text)
        tfidf_test = self.vectorizer.transform(X_test_text)
        
        # Combine with structured features
        X_train = hstack([tfidf_train, csr_matrix(struct_train)])
        X_test = hstack([tfidf_test, csr_matrix(struct_test)])
        
        self.is_trained = True
        
        # Train classifier
        print("Training classifier...")
        self.classifier = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            C=1.0,
            random_state=42
        )
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Scam']))
        
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        accuracy = (y_pred == y_test).mean()
        print(f"\n✓ Overall Accuracy: {accuracy:.1%}")
        
        return accuracy
    
    def predict(self, text: str, structured_features=None):
        """Predict scam probability for a single message."""
        if not self.is_trained:
            raise ValueError("Model not trained!")
        
        # TF-IDF features
        tfidf = self.vectorizer.transform([text])
        
        # Structured features (zeros if not provided)
        if structured_features is None:
            struct = np.zeros((1, len(self.structured_cols)))
        else:
            struct = np.array([structured_features])
        
        # Combine
        features = hstack([tfidf, csr_matrix(struct)])
        
        # Predict
        prob = self.classifier.predict_proba(features)[0]
        label = self.classifier.predict(features)[0]
        
        return {
            'is_scam': bool(label),
            'scam_probability': float(prob[1]),
            'safe_probability': float(prob[0]),
            'confidence': float(max(prob))
        }
    
    def save(self, model_dir: str):
        """Save model for deployment."""
        os.makedirs(model_dir, exist_ok=True)
        
        with open(os.path.join(model_dir, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(os.path.join(model_dir, 'classifier.pkl'), 'wb') as f:
            pickle.dump(self.classifier, f)
        
        metadata = {
            'is_trained': self.is_trained,
            'feature_count': len(self.vectorizer.vocabulary_) + len(self.structured_cols),
            'model_type': 'LogisticRegression',
            'structured_cols': self.structured_cols
        }
        with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Model saved to {model_dir}")
    
    def load(self, model_dir: str):
        """Load saved model."""
        with open(os.path.join(model_dir, 'vectorizer.pkl'), 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        with open(os.path.join(model_dir, 'classifier.pkl'), 'rb') as f:
            self.classifier = pickle.load(f)
        
        self.is_trained = True
        print(f"✓ Model loaded from {model_dir}")


# --- Training Script ---
if __name__ == "__main__":
    classifier = ScamClassifier()
    
    # Load data (SMS + WhatsApp only)
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    classifier.load_data(data_dir)
    
    # Train
    accuracy = classifier.train(test_size=0.2)
    
    # Save model
    model_dir = os.path.join(data_dir, 'models')
    classifier.save(model_dir)
    
    # Test with sample messages
    print("\n" + "=" * 60)
    print("TESTING ON SAMPLE MESSAGES")
    print("=" * 60)
    
    test_messages = [
        "Hi, I'll reach by 7:00 PM. See you soon.",
        "OLX: Buyer sent payment. To receive ₹50000, share OTP.",
        "You are under 'digital arrest'. Stay on line and transfer ₹1500.",
        "Your package is out for delivery. Track at amazon.in/track",
        "URGENT: Your card will be blocked. Verify at https://bit.ly/xyz",
    ]
    
    for msg in test_messages:
        result = classifier.predict(msg)
        status = "🚨 SCAM" if result['is_scam'] else "✅ SAFE"
        print(f"\n{status} ({result['confidence']:.0%} confidence)")
        print(f"   Message: {msg[:50]}...")
