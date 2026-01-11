# ScamShield Architecture - Rule-Based vs ML-Based

## Component Classification

| Component | Type | Status | Size |
|-----------|------|--------|------|
| Rule Engine | **RULE-BASED** | ✅ Production | ~12KB |
| TF-IDF + LogReg | **ML-BASELINE** | ✅ Fallback | ~80KB |
| **DistilBERT** | **ML-PRIMARY** | ✅ Trained | ~256MB |

---

## DistilBERT Model (Primary ML)

**Location**: `models/distilbert/`

| File | Size | Purpose |
|------|------|---------|
| `model.safetensors` | 255 MB | Model weights (excluded from git) |
| `config.json` | 0.6 KB | Model configuration |
| `vocab.txt` | 256 KB | Tokenizer vocabulary |
| `tokenizer_config.json` | 1.3 KB | Tokenizer settings |
| `training_metadata.json` | 0.3 KB | Training metrics |

### Training Metrics
```
Accuracy:  1.0000
Precision: 1.0000
Recall:    1.0000
F1 Score:  1.0000

Confusion Matrix:
              Predicted
              SAFE  SCAM
Actual SAFE     6     0
       SCAM     0    24
```

### Training Config
- Model: `distilbert-base-uncased`
- Epochs: 5
- Batch Size: 8
- Learning Rate: 2e-5
- Class Weighting: Balanced (for 81% scam / 19% legit imbalance)
- Train: 118 samples | Test: 30 samples

---

## TF-IDF Baseline (Fallback ML)

**Location**: `models/sklearn_*.pkl`

- Lighter weight (~80KB total)
- Faster inference
- Used when DistilBERT is unavailable

---

## Detection Pipeline

```
Message → Rule Engine → DistilBERT → Combined Result
          (deterministic, <10ms)   (ML, ~100ms)
```

1. **Rule Engine** always runs first (catches explicit patterns)
2. **DistilBERT** runs for uncertain cases
3. Combined confidence determines final verdict

---

## Regenerating Models

```bash
# Activate venv
.\venv\Scripts\activate

# Train sklearn baseline
python backend/train_model.py

# Train DistilBERT (requires ~256MB download)
python backend/train_distilbert.py
```

---

## Note on 100% Accuracy

The dataset is small (148 samples) and synthetic. In production:
- Expect lower accuracy on real-world data
- Rule Engine provides robustness for known patterns
- DistilBERT generalizes to novel scam types
