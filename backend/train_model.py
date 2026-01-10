"""
REAL ML Training Pipeline - DistilBERT Scam Classifier
=======================================================
This trains an actual transformer model (DistilBERT) on the scam dataset.

For hackathon, we provide TWO options:
1. DistilBERT (preferred if torch is available) - ~250MB model
2. Baseline sklearn (fallback) - ~5MB model

Both are REAL ML models trained on the CSV data.
"""

import os
import sys
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack, csr_matrix
from tqdm import tqdm

# Try to import transformers (optional - for DistilBERT)
TRANSFORMERS_AVAILABLE = False
try:
    import torch
    from transformers import (
        DistilBertTokenizer, 
        DistilBertForSequenceClassification,
        TrainingArguments,
        Trainer
    )
    from torch.utils.data import Dataset
    TRANSFORMERS_AVAILABLE = torch.cuda.is_available() or True  # Allow CPU training
    print(f"[INFO] PyTorch available. CUDA: {torch.cuda.is_available()}")
except ImportError:
    print("[INFO] Transformers/PyTorch not installed. Using sklearn baseline.")


class ScamDataset:
    """Load and prepare the scam detection datasets."""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.data = None
        self.structured_cols = ['has_url', 'has_upi', 'has_otp', 'has_qr', 
                                'has_phone', 'has_threat', 'has_urgency']
    
    def load(self):
        """Load SMS and WhatsApp data only (on-device modalities)."""
        sms_path = os.path.join(self.data_dir, 'public_sms.csv')
        whatsapp_path = os.path.join(self.data_dir, 'public_whatsapp.csv')
        
        dfs = []
        
        if os.path.exists(sms_path):
            sms_df = pd.read_csv(sms_path)
            sms_df['modality'] = 'SMS'
            sms_df['text'] = sms_df['message_text']
            dfs.append(sms_df)
            print(f"[DATA] Loaded {len(sms_df)} SMS samples")
        
        if os.path.exists(whatsapp_path):
            wa_df = pd.read_csv(whatsapp_path)
            wa_df['modality'] = 'WhatsApp'
            wa_df['text'] = wa_df['conversation_text']
            dfs.append(wa_df)
            print(f"[DATA] Loaded {len(wa_df)} WhatsApp samples")
        
        if not dfs:
            raise ValueError("No data files found!")
        
        self.data = pd.concat(dfs, ignore_index=True)
        self.data['text'] = self.data['text'].fillna('').astype(str)
        
        print(f"[DATA] Total: {len(self.data)} samples")
        print(f"[DATA] Scams: {self.data['is_scam'].sum()} | Legit: {(self.data['is_scam'] == 0).sum()}")
        
        return self.data
    
    def get_structured_features(self):
        """Get structured signal flags."""
        available = [c for c in self.structured_cols if c in self.data.columns]
        if available:
            return self.data[available].values.astype(float)
        return np.zeros((len(self.data), len(self.structured_cols)))


# ============================================================================
# OPTION 1: DistilBERT Classifier (Preferred) - Only defined if torch available
# ============================================================================

if TRANSFORMERS_AVAILABLE:
    class ScamTorchDataset(Dataset):
        """PyTorch Dataset for scam classification."""
        
        def __init__(self, texts, labels, tokenizer, max_length=256):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = int(self.labels[idx])
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label)
            }

    class DistilBERTClassifier:
        """
        DistilBERT-based scam classifier.
        Smaller than BERT but still powerful.
        Can be exported to ONNX for mobile.
        """
        
        def __init__(self, model_name='distilbert-base-uncased'):
            self.model_name = model_name
            self.tokenizer = None
            self.model = None
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        def train(self, texts, labels, test_size=0.2, epochs=3):
            print(f"\n{'='*60}")
            print("TRAINING DistilBERT CLASSIFIER")
            print(f"{'='*60}")
            print(f"Device: {self.device}")
            
            # Load tokenizer and model
            print("[MODEL] Loading DistilBERT tokenizer...")
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
            
            print("[MODEL] Loading DistilBERT model...")
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=2
            ).to(self.device)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=test_size, random_state=42, stratify=labels
            )
            
            print(f"[SPLIT] Train: {len(X_train)} | Test: {len(X_test)}")
            
            # Create datasets
            train_dataset = ScamTorchDataset(X_train, y_train, self.tokenizer)
            test_dataset = ScamTorchDataset(X_test, y_test, self.tokenizer)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir='./results',
                num_train_epochs=epochs,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                warmup_steps=50,
                weight_decay=0.01,
                logging_dir='./logs',
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="accuracy"
            )
            
            # Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                compute_metrics=self._compute_metrics
            )
            
            # Train
            print("\n[TRAIN] Starting training...")
            trainer.train()
            
            # Evaluate
            print("\n[EVAL] Evaluating on test set...")
            results = trainer.evaluate()
            print(f"[EVAL] Accuracy: {results['eval_accuracy']:.2%}")
            
            return {'accuracy': results['eval_accuracy']}
        
        def _compute_metrics(self, pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            acc = accuracy_score(labels, preds)
            return {'accuracy': acc}
        
        def predict(self, text):
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred = torch.argmax(probs, dim=-1).item()
            
            return {
                'is_scam': bool(pred),
                'confidence': float(probs[0][pred]),
                'scam_probability': float(probs[0][1])
            }
        
        def save(self, model_dir):
            os.makedirs(model_dir, exist_ok=True)
            self.model.save_pretrained(os.path.join(model_dir, 'distilbert'))
            self.tokenizer.save_pretrained(os.path.join(model_dir, 'distilbert'))
            print(f"[SAVE] DistilBERT model saved to {model_dir}/distilbert")
        
        def load(self, model_dir):
            path = os.path.join(model_dir, 'distilbert')
            self.tokenizer = DistilBertTokenizer.from_pretrained(path)
            self.model = DistilBertForSequenceClassification.from_pretrained(path).to(self.device)
            print(f"[LOAD] DistilBERT model loaded from {path}")


# ============================================================================
# OPTION 2: Sklearn Baseline (Fallback)
# ============================================================================

class SklearnBaselineClassifier:
    """
    Sklearn baseline classifier.
    Uses TF-IDF + LogisticRegression.
    
    This is a REAL ML model, just simpler than transformers.
    Good for fast inference on low-end devices.
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )
        self.classifier = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            C=1.0,
            random_state=42
        )
        self.structured_cols = ['has_url', 'has_upi', 'has_otp', 'has_qr', 
                                'has_phone', 'has_threat', 'has_urgency']
    
    def train(self, texts, labels, structured_features=None, test_size=0.2):
        print(f"\n{'='*60}")
        print("TRAINING SKLEARN BASELINE CLASSIFIER")
        print(f"{'='*60}")
        
        # Split data
        indices = np.arange(len(texts))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=42, stratify=labels
        )
        
        X_train_text = [texts[i] for i in train_idx]
        X_test_text = [texts[i] for i in test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]
        
        print(f"[SPLIT] Train: {len(X_train_text)} | Test: {len(X_test_text)}")
        
        # TF-IDF
        print("[TRAIN] Fitting TF-IDF vectorizer...")
        tfidf_train = self.vectorizer.fit_transform(X_train_text)
        tfidf_test = self.vectorizer.transform(X_test_text)
        
        # Add structured features if available
        if structured_features is not None:
            struct_train = csr_matrix(structured_features[train_idx])
            struct_test = csr_matrix(structured_features[test_idx])
            X_train = hstack([tfidf_train, struct_train])
            X_test = hstack([tfidf_test, struct_test])
            print(f"[TRAIN] Using structured features: {self.structured_cols}")
        else:
            X_train = tfidf_train
            X_test = tfidf_test
        
        # Train
        print("[TRAIN] Training LogisticRegression classifier...")
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{'='*60}")
        print("CLASSIFICATION REPORT")
        print(f"{'='*60}")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Scam']))
        print(f"[EVAL] Accuracy: {accuracy:.2%}")
        
        return {'accuracy': accuracy}
    
    def predict(self, text, structured_features=None):
        tfidf = self.vectorizer.transform([text])
        
        if structured_features is not None:
            struct = csr_matrix([structured_features])
            features = hstack([tfidf, struct])
        else:
            # Use zeros if no structured features
            zeros = csr_matrix(np.zeros((1, len(self.structured_cols))))
            features = hstack([tfidf, zeros])
        
        prob = self.classifier.predict_proba(features)[0]
        pred = self.classifier.predict(features)[0]
        
        return {
            'is_scam': bool(pred),
            'confidence': float(max(prob)),
            'scam_probability': float(prob[1])
        }
    
    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        
        with open(os.path.join(model_dir, 'sklearn_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(os.path.join(model_dir, 'sklearn_classifier.pkl'), 'wb') as f:
            pickle.dump(self.classifier, f)
        
        metadata = {
            'model_type': 'sklearn_tfidf_logreg',
            'features': len(self.vectorizer.vocabulary_) + len(self.structured_cols),
            'structured_cols': self.structured_cols
        }
        with open(os.path.join(model_dir, 'sklearn_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[SAVE] Sklearn model saved to {model_dir}")
    
    def load(self, model_dir):
        with open(os.path.join(model_dir, 'sklearn_vectorizer.pkl'), 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        with open(os.path.join(model_dir, 'sklearn_classifier.pkl'), 'rb') as f:
            self.classifier = pickle.load(f)
        
        print(f"[LOAD] Sklearn model loaded from {model_dir}")


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    print("="*70)
    print("SCAMSHIELD ML TRAINING PIPELINE")
    print("="*70)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    model_dir = os.path.join(project_dir, 'models')
    
    # Load data
    print("\n[1/4] LOADING DATA")
    dataset = ScamDataset(project_dir)
    data = dataset.load()
    
    texts = data['text'].values
    labels = data['is_scam'].values
    structured = dataset.get_structured_features()
    
    # Choose model based on availability
    print("\n[2/4] SELECTING MODEL")
    
    if TRANSFORMERS_AVAILABLE and '--distilbert' in sys.argv:
        print("[MODEL] Using DistilBERT (transformer)")
        classifier = DistilBERTClassifier()
        results = classifier.train(texts, labels, epochs=3)
        classifier.save(model_dir)
        model_type = "DistilBERT"
    else:
        print("[MODEL] Using Sklearn baseline (TF-IDF + LogReg)")
        classifier = SklearnBaselineClassifier()
        results = classifier.train(texts, labels, structured_features=structured)
        classifier.save(model_dir)
        model_type = "Sklearn Baseline"
    
    # Test predictions
    print("\n[3/4] TESTING PREDICTIONS")
    test_messages = [
        ("Hi, I'll reach by 7 PM. See you soon.", False),
        ("OLX buyer. Share OTP to receive payment.", True),
        ("Digital arrest. Transfer money now.", True),
        ("Your Netflix subscription renewed.", False),
    ]
    
    correct = 0
    for msg, expected in test_messages:
        result = classifier.predict(msg)
        status = "PASS" if result['is_scam'] == expected else "FAIL"
        correct += 1 if result['is_scam'] == expected else 0
        print(f"  [{status}] {'SCAM' if result['is_scam'] else 'SAFE'} ({result['confidence']:.0%}) - {msg[:40]}...")
    
    print(f"\n[4/4] SUMMARY")
    print(f"  Model Type: {model_type}")
    print(f"  Test Accuracy: {results['accuracy']:.2%}")
    print(f"  Prediction Accuracy: {correct}/{len(test_messages)}")
    print(f"  Model saved to: {model_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
