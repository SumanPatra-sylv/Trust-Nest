1. End Goal (What we are building)

Build a privacy-first Android app to protect senior citizens from digital scams across SMS, WhatsApp messages, voice calls, and video-call based “digital arrest” scams.

The system must:

Detect scams early

Warn before irreversible actions (OTP sharing, payments)

Involve trusted family members when risk is high

Be explainable in simple language

Work reliably on low-end Android phones

Respect OS, legal, and privacy constraints (no silent recording)

2. Core Design Philosophy

Rule-first, ML-second (deterministic checks catch obvious scams fast)

Tiny on-device ML for instant decisions

Heavy AI only on demand

Behavioral cues instead of spying (especially for calls)

Human-in-the-loop escalation (Family Guardian)

Explainability is mandatory

English + Hindi support

3. Full Feature Set (Build ALL of these)
A. Message & Link Shield (SMS + WhatsApp)

Monitor incoming SMS and WhatsApp notification text.

Classify each message into:

SAFE

SUSPICIOUS

SCAM

Always show 1-line reason for any warning.

One-tap actions:

Block / Delete

Ask family

Report scam

B. Call Shield (No silent audio recording)

Pre-pickup warning for unknown/high-risk callers.

In-call nudges based on call duration & risk:

Reminder banners at 60s / 120s.

Post-call summary:

Risk score

Reason

Actions if OTP/payment may have been shared.

Call intelligence is metadata + behavior based, not continuous ASR.

C. Digital Arrest Interrupter (Critical)

Detect incoming WhatsApp / Skype video calls from unknown numbers using notification metadata.

Immediately overlay a warning:

“Police/CBI never video call. Do not share screen or install apps.”

Warn if remote-control apps (AnyDesk, TeamViewer) are installed.

No audio/video capture.

D. Family / Care-Circle (Guardian Mode)

Guardian (child/caregiver) pairs with senior’s phone.

High-risk events trigger guardian alerts.

Guardian can:

Approve / Block numbers

Whitelist trusted contacts

Guardian actions override ML decisions.

E. Citizen Reporting & Intelligence

One-tap “Report Scam”.

Use hash-based reporting (no raw messages).

Confirm scams via cluster consensus.

Distribute signed blocklist updates.

F. Teach-Me Mode

Short micro-lessons (English + Hindi).

Examples:

Banks never ask OTP

Refunds don’t require QR receive

Police don’t video call

Text + Text-to-Speech.

G. User-Initiated Audio Analysis (Optional, Explicit)

User may manually record up to 30s of a call for analysis.

Requires explicit consent each time.

Uses local or proxy ASR.

Never always-on.

4. Tech Stack (Use exactly this)
Android Client

Language: Kotlin

UI: Jetpack Compose

Background:

NotificationListenerService

TelephonyManager / CallScreeningService

Foreground Service

ML:

TensorFlow Lite (TFLite) for TinyBERT

ExecuTorch + XNNPACK for local heavy model (optional)

ASR (manual only): Vosk or Sherpa-ONNX

VAD: Silero VAD

Storage: Room + Android Keystore AES-GCM

Networking: Retrofit + OkHttp

Push: Firebase Cloud Messaging

TTS: Android TextToSpeech

CI/CD: Gradle + GitHub Actions + Fastlane

Backend / Privacy Proxy

Framework: FastAPI (Python)

ML serving: ONNX Runtime / ExecuTorch

DB: PostgreSQL + Redis

Dashboard: React + Tailwind

Container: Docker

5. Detection Architecture (How things work)
Step 1: Rule Engine (Always First)

Use deterministic checks:

UPI handle validation

URL shorteners

OTP / urgency phrases

Homoglyph domains

Bloom filter for known scams

If high confidence scam → warn immediately.

Step 2: Tiny On-Device ML (SMS + WhatsApp only)

Model: TinyBERT / DistilBERT

Input:

Message text

Structured scam flags:

has_url

has_upi

has_otp

has_urgency

has_threat

Output:

Label: Safe / Suspicious / Scam

Score

Tokens for explanation

Target:

<50MB model

<200ms inference on 3GB phones

Step 3: Heavy AI (Only if Needed)

Triggered only when:

Rule engine is unsure

Tiny model returns SUSPICIOUS

Behavior:

If device capable → run local Sarvam-1

Else → send anonymized snippet to privacy proxy

Return label + reasons

Step 4: Human Escalation

Show “Ask family” prominently.

Guardian can approve or block.

6. Training Expectations (Important for Model Behavior)

Train separate logic for:

SMS / WhatsApp (single-text)

Calls / Audio transcripts (dialogue-aware, heavy model only)

Use structured CSV flags (has_otp, has_upi, etc.) as model inputs, not just labels.

Handle class imbalance (scam-heavy dataset).

Use multi-task heads:

is_scam

requested_action

scam_stage

Optimize for low false positives (warn > block).

7. Explainability (Mandatory)

Every warning must show:

Simple title

One-line reason (English + Hindi)

Recommended action highlighted

Example:

“Asks for OTP and urgent payment. Banks never ask OTP.”

8. Privacy & Permissions (Strict)

No silent call recording

No contact scraping

Audio only with explicit user action

Reporting uses hashes only

All server communication optional and consent-gated

9. Success Criteria (What “done” means)

App can:

Flag scam SMS & WhatsApp messages

Warn before risky calls

Interrupt digital arrest attempts

Escalate to family

Works on low-end phones

Fully explainable

Judges can demo it in 3–4 minutes

10. What NOT to Do

Do NOT run always-on ASR

Do NOT silently upload messages or audio

Do NOT auto-block without explanation

Do NOT rely only on heavy models