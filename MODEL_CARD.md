# Model Card: DistilBERT Scam Classifier

> **Audience**: Hackathon judges | **Status**: Prototype | **Last Updated**: 2026-01-11

---

## Model Overview

| Property | Value |
|----------|-------|
| **Model Name** | distilbert-scam-classifier |
| **Base Model** | distilbert-base-uncased |
| **Task** | Binary classification (SAFE / SCAM) |
| **Role in System** | Semantic layer after rule-based detection |
| **Parameters** | 66 million |

**System Context**: This model is the *secondary* detection layer. The rule engine always runs first and can **override** ML decisions for explicit scam patterns (OTP requests, digital arrest, etc.).

---

## Training Data

| Dataset | Samples | Scam % | Format |
|---------|---------|--------|--------|
| `public_sms.csv` | 74 | ~81% | SMS messages |
| `public_whatsapp.csv` | 74 | ~81% | WhatsApp conversations |
| **Total** | **148** | **81%** | Combined |

### ⚠️ Data Caveats

- **Synthetic data**: Generated for hackathon, not real-world scams
- **Class imbalance**: 81% scam, 19% legitimate (addressed via class weighting)
- **Language**: Primarily English with some Hindi transliteration
- **Small size**: 148 samples is insufficient for production

---

## Training Setup

| Parameter | Value |
|-----------|-------|
| Base model | `distilbert-base-uncased` |
| Epochs | 5 |
| Batch size | 8 |
| Learning rate | 2e-5 |
| Max sequence length | 256 |
| Class weights | Balanced (Legit: 2.11, Scam: 0.62) |
| Train/Test split | 80/20 (118 train, 30 test) |
| Optimizer | AdamW with warmup |
| Hardware | CPU only |

---

## Evaluation Metrics

**Test Set (n=30)**

| Metric | Value |
|--------|-------|
| Accuracy | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 Score | 1.0000 |

**Confusion Matrix**
```
              Predicted
              SAFE  SCAM
Actual SAFE     6     0
       SCAM     0    24
```

### ⚠️ Evaluation Caveats

- **Small test set**: Only 30 samples
- **Same distribution**: Test data from same synthetic source as training
- **Overfit risk**: Perfect accuracy on small set does not imply real-world performance
- **No cross-validation**: Single train/test split used

**Do not interpret 100% accuracy as production readiness.**

---

## Explainability

### How Confidence Is Used

```python
if ml_confidence >= 0.8:
    # Trust ML decision
elif rule_score >= 30:
    # Prefer rule engine
else:
    # Use ML as tiebreaker
```

### Rule Override Behavior

The rule engine **always** overrides ML when:
- OTP request detected
- Digital arrest pattern matched
- Authority impersonation found
- URL shortener with threat language

This ensures deterministic protection for known scam patterns.

---

## Limitations & Risks

### Technical Limitations
- **Small dataset**: 148 samples is far below production threshold
- **Synthetic data**: Patterns may not reflect real-world scam evolution
- **No adversarial testing**: Model not tested against evasion attempts
- **Single language**: Limited Hindi/regional language support

### Potential Failure Modes
| Risk | Impact | Mitigation |
|------|--------|------------|
| False positive | Blocks legitimate message | Guardian review, user override |
| False negative | Misses scam | Rule engine as safety net |
| OOV text | Low confidence | Falls back to rule engine |
| Code-switching | May misclassify | Future work: multilingual training |

### Bias Considerations
- Trained on Indian scam patterns only
- May not generalize to other regions
- Urgency detection biased toward English phrasing

---

## Privacy & Ethics

### ✅ Privacy Guarantees
- **No silent data collection**: User controls all data
- **No always-on audio**: Call analysis is metadata-only
- **On-device default**: Rule engine runs locally
- **No message upload**: Backend is fallback only

### ✅ Human-in-the-Loop
- Guardian mode requires explicit consent
- Users can override all ML decisions
- Block/allow lists are user-controlled

### ❌ Not Implemented
- Differential privacy for training
- Federated learning
- Model update consent flows

---

## Deployment Notes

| Deployment | Status | Notes |
|------------|--------|-------|
| Backend inference | ✅ Ready | FastAPI at `/api/analyze` |
| ONNX export | ✅ Complete | 255 MB, verified 3/3 test cases |
| TFLite | ❌ Not generated | Missing onnx_tf dependency |
| On-device (Android) | 📋 Planned | ONNX Runtime Mobile documented |

### Model Files

```
models/distilbert/
├── model.safetensors    # 256 MB (excluded from git)
├── config.json          # Model configuration
├── vocab.txt            # Tokenizer vocabulary
├── training_metadata.json
└── onnx/
    └── model.onnx       # 255 MB (excluded from git)
```

### Regenerating Model
```bash
.\venv\Scripts\python backend\train_distilbert.py
.\venv\Scripts\python backend\export_model.py
```

---

## Intended Use

✅ **Appropriate**
- Hackathon demonstration
- Research prototype
- Educational purposes
- Further development baseline

❌ **Not Appropriate**
- Production deployment
- Critical safety decisions without human review
- Sole protection mechanism

---

## Citation

```
@misc{scamshield2026,
  title={ScamShield: Privacy-First Scam Detection for Senior Citizens},
  author={KHISTIJ Hackathon Team},
  year={2026},
  note={Prototype - Not for production use}
}
```
