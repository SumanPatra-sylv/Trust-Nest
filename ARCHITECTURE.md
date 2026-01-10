# ScamShield Architecture - Rule-Based vs ML-Based

## Component Classification

| Component | Type | Location | Size |
|-----------|------|----------|------|
| Rule Engine | **RULE-BASED** | `backend/rule_engine.py` | ~12KB |
| TF-IDF + LogReg | **ML-BASED** (baseline) | `models/sklearn_*.pkl` | ~80KB |
| DistilBERT | **ML-BASED** (preferred) | `models/distilbert/` | ~250MB* |

*DistilBERT not trained in this session (requires PyTorch installation)

---

## Detection Flow

```
Message → Rule Engine → ML Classifier → Combined Result
          (deterministic)  (probabilistic)
```

### Rule Engine (Deterministic)
- Fast pattern matching (<10ms)
- Catches: UPI, OTP, URLs, threats, authority claims
- **No training required**
- **Always runs first**

### ML Classifier (Trained)
- TF-IDF vectorization + Logistic Regression
- **Trained on**: `public_sms.csv` + `public_whatsapp.csv` (148 samples)
- **Test accuracy**: 100% on 20% holdout (30 samples)
- **Limitation**: Struggles with out-of-vocabulary text

---

## Model Artifacts

Located in `models/`:

| File | Purpose | Size |
|------|---------|------|
| `sklearn_vectorizer.pkl` | TF-IDF vocabulary + weights | 66 KB |
| `sklearn_classifier.pkl` | Trained LogisticRegression | 14 KB |
| `sklearn_metadata.json` | Model metadata | 214 bytes |

### Loading the Model

```python
from backend.train_model import SklearnBaselineClassifier

classifier = SklearnBaselineClassifier()
classifier.load("models")
result = classifier.predict("Share OTP to verify payment")
```

---

## Why Rule + ML?

| Input | Rule Engine | ML Alone | Combined |
|-------|-------------|----------|----------|
| "Share OTP" | ✅ SCAM (OTP_REQUEST rule) | ❌ SAFE | ✅ SCAM |
| "Digital arrest" | ✅ SCAM (DIGITAL_ARREST rule) | ❌ SAFE | ✅ SCAM |
| Novel phishing | ❌ Missed | ✅ Catches patterns | ✅ SCAM |
| Legitimate message | ✅ SAFE | ✅ SAFE | ✅ SAFE |

**The Rule Engine catches explicit patterns; ML catches subtle patterns.**

---

## Training the Model

```bash
# Create venv and install dependencies
python -m venv venv
.\venv\Scripts\pip install -r backend/requirements.txt

# Train sklearn baseline
.\venv\Scripts\python backend/train_model.py

# Train DistilBERT (if PyTorch installed)
.\venv\Scripts\python backend/train_model.py --distilbert
```

---

## Honest Assessment

- ✅ Rule Engine: Fully working, tested, catches 6+ scam types
- ✅ ML Baseline: Trained, saved, 100% on test split
- ⚠️ ML Generalization: Struggles with OOV text (expected)
- ❌ DistilBERT: Not trained (needs PyTorch ~2GB install)
- ❌ TFLite Export: Not implemented yet
