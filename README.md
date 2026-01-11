# ScamShield - Trust Nest

**Privacy-first scam detection for senior citizens.** Android app with on-device rule engine + DistilBERT for SMS/WhatsApp protection.

> ⚠️ **Hackathon Project** - Not production-ready. Trained on synthetic data.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INCOMING MESSAGE                              │
│                   (SMS / WhatsApp / Call Metadata)                   │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     RULE ENGINE (Always First)                       │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐│
│  │   OTP   │ │   UPI   │ │  URL    │ │ Threat  │ │ Digital Arrest  ││
│  │ Request │ │  Check  │ │Shortener│ │Language │ │ Pattern Match   ││
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────────┬────────┘│
│       └───────────┴───────────┴───────────┴───────────────┘         │
│                           Rule Score ≥ 60?                          │
│                     YES → OVERRIDE (Skip ML)                        │
└─────────────────────────────────────────────────────────────────────┘
                                 │ NO (Uncertain)
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DistilBERT CLASSIFIER                          │
│         ┌─────────────────────────────────────────────┐              │
│         │  distilbert-base-uncased (66M params)       │              │
│         │  Trained on: SMS + WhatsApp (148 samples)   │              │
│         │  Output: SAFE/SCAM + confidence score       │              │
│         └─────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      GUARDIAN ESCALATION                            │
│      High-risk (SCAM) OR Digital Arrest OR Family Impersonation     │
│                    → Send FCM alert to guardian                      │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      USER-FACING OUTPUT                              │
│  ┌──────────────┐  ┌──────────────────┐  ┌─────────────────────────┐│
│  │  ✅ SAFE     │  │  ⚠️ SUSPICIOUS   │  │  🚨 SCAM               ││
│  │  No action   │  │  Ask family      │  │  Block + Report        ││
│  └──────────────┘  └──────────────────┘  └─────────────────────────┘│
│                  + Bilingual Explanation (EN/Hindi)                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Components

| Layer | Component | Location | Purpose |
|-------|-----------|----------|---------|
| **Detection** | Rule Engine | `backend/rule_engine.py` | Deterministic pattern matching |
| **Detection** | DistilBERT | `models/distilbert/` | Semantic classification |
| **Detection** | Unified Detector | `backend/detector.py` | Pipeline orchestration |
| **API** | FastAPI | `backend/app.py` | Backend inference |
| **Android** | RuleEngine.kt | `android/.../detection/` | On-device rules |
| **Android** | MessageShieldService | `android/.../services/` | Notification listener |
| **Android** | GuardianMode | `android/.../guardian/` | Family pairing + FCM |

---

## Quick Start

### Backend
```bash
# Create venv and install
python -m venv venv
.\venv\Scripts\pip install -r backend/requirements.txt

# Train model (optional - uses existing weights)
.\venv\Scripts\python backend/train_distilbert.py

# Run API
.\venv\Scripts\python backend/app.py
```

### Test Detection
```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Share OTP to verify your payment"}'
```

### Android
Open `android/` in Android Studio, sync Gradle, run on device.

---

## Detection Pipeline

```
Message → Rule Engine → DistilBERT → Guardian
          (<10ms)       (~100ms)     (if high-risk)
```

**Priority Order:**
1. **Rule Engine** catches explicit patterns (OTP, threats, URLs)
2. **DistilBERT** handles ambiguous cases
3. **Guardian** escalates high-risk to family

---

## Model Status

| Model | Type | Status | Size |
|-------|------|--------|------|
| Rule Engine | Deterministic | ✅ Production | - |
| TF-IDF | ML Baseline | ✅ Fallback | 80 KB |
| DistilBERT | ML Primary | ✅ Trained | 256 MB |
| ONNX | Export | ✅ Ready | 255 MB |

See [MODEL_CARD.md](MODEL_CARD.md) for training details.

---

## Privacy Promise

- ❌ No silent call recording
- ❌ No message uploading to server
- ❌ No contact scraping
- ✅ On-device rule engine
- ✅ User controls all data
- ✅ Guardian alerts require explicit pairing

---

## Project Structure

```
Trust-Nest/
├── backend/
│   ├── rule_engine.py      # Deterministic detection
│   ├── detector.py         # Unified pipeline
│   ├── train_distilbert.py # Model training
│   ├── export_model.py     # ONNX export
│   └── app.py              # FastAPI server
├── android/
│   └── app/src/main/java/com/scamshield/
│       ├── detection/      # On-device rules
│       ├── services/       # Background services
│       ├── guardian/       # Family pairing
│       └── ui/             # Compose screens
├── models/
│   └── distilbert/         # Trained model
└── *.csv                   # Training data
```

---

## Limitations

> ⚠️ This is a hackathon prototype, not production software.

- Trained on **synthetic data** (148 samples)
- Test accuracy may not generalize to real-world scams
- Model is large (256 MB) for mobile deployment
- Hindi support is partial (transliteration only)

---

## License

MIT License - Hackathon project for KHISTIJ.
