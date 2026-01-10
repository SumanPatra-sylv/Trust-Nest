# ScamShield - Android Scam Detection App

An Android-first, privacy-preserving scam detection system for senior citizens.

## Quick Start

### Backend (Optional - for fallback/training)
```bash
cd backend
pip install -r requirements.txt
python classifier.py  # Train model
python app.py         # Start server
```

### Android App
1. Open `android/` folder in Android Studio
2. Sync Gradle
3. Run on device/emulator

## Project Structure

```
KHISTIJ/
├── backend/                    # Python backend (fallback only)
│   ├── rule_engine.py         # Deterministic scam detection
│   ├── feature_extractor.py   # Extract signal flags
│   ├── classifier.py          # ML classifier (TF-IDF)
│   ├── detector.py            # Unified detection pipeline
│   └── app.py                 # FastAPI server
│
├── android/                    # Android app (Kotlin + Compose)
│   └── app/src/main/java/com/scamshield/
│       ├── detection/         # On-device rule engine
│       ├── services/          # Background services
│       ├── guardian/          # Family pairing
│       └── ui/                # Jetpack Compose screens
│
├── models/                     # Trained ML models
│   ├── vectorizer.pkl
│   └── classifier.pkl
│
└── public_*.csv               # Training datasets
```

## Features

| Feature | Status | Description |
|---------|--------|-------------|
| Message Shield | ✅ | SMS + WhatsApp monitoring |
| Call Shield | ✅ | Metadata-based warnings |
| Digital Arrest | ✅ | Video call interruption |
| Guardian Mode | ✅ | Family pairing + FCM |
| Teach-Me Mode | ✅ | Educational lessons |

## Detection Architecture

```
Message → Rule Engine → ML Classifier → Guardian Escalation
           (<10ms)       (fallback)      (if high risk)
```

## Privacy

- ❌ No silent call recording
- ❌ No contact scraping
- ❌ No message uploading
- ✅ All detection on-device
- ✅ Hash-based reporting only
