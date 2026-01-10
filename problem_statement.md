🧩 PROBLEM STATEMENT (AS PER PDF – IDE VERSION)
1. Core Problem (What we are solving)

Senior citizens in India are frequent targets of digital fraud, including:

Scam SMS and WhatsApp messages (phishing, fake KYC, courier scams)

Fraudulent voice calls (fake bank officials, tech support, “relative in distress”)

Malicious links and QR codes

New threats like “digital arrest” video calls

These scams cause:

Financial loss

Identity compromise

Severe emotional distress

The challenge is to build an AI-powered, privacy-first, deployable system that helps non-tech-savvy elders recognize, understand, and avoid scams before irreversible actions occur.

2. What the Organizers Explicitly Want Built

The PDF is very clear:
👉 This is not a research-only or slide-only problem.
👉 They want something that could realistically be piloted.

Mandatory Characteristics

Your solution must be:

Explainable (every warning needs a reason)

Privacy-preserving (no silent data collection)

Usable by seniors (large fonts, minimal taps)

Deployable on low to mid-range Android devices

Capable of working with real Indian scam patterns

3. Required Functional Capabilities (Problem Scopes)

You may choose one or combine multiple, but strong solutions usually combine them.

A. Message & Link Shield (SMS / WhatsApp)

The system should:

Classify incoming messages or links as:

Likely Scam

Suspicious

Safe

Provide plain-language explanation (English + at least one Indian language)

Offer one-tap actions, such as:

Call trusted contact

Report scam

Block sender

Verify merchant

📌 Important:
WhatsApp support is expected via notification analysis, not deep integration.

B. Call Shield (Voice / IVR)

The system should:

Detect suspicious caller behavior, such as:

Urgent money demands

OTP requests

Threats or pressure

Provide:

Real-time cues (warnings, reminders)

Post-call summary including:

Risk score

Reasons

Suggested next steps

📌 Important constraint:

No silent call recording

Privacy-first behavior-based detection is preferred

C. Family & Care-Circle Co-Pilot

The system should:

Allow seniors to opt in to a trusted contact (child/caregiver)

Notify that contact when high-risk events occur

Provide “Are you sure?” coaching moments that teach patterns like:

“Never share OTP”

“No refunds via QR receive”

📌 This is not optional fluff — it is a key differentiator.

D. Citizen Reporting & Intelligence

The system should:

Allow users to report scam incidents

Send reports to a centralized, privacy-preserving database

Support:

Deduplication

Regional/language-level analysis

Exportable evidence bundles (for authorities)

📌 Explicitly forbidden:

Uploading personal messages, contacts, or PII

4. Constraints You MUST Respect (Non-Negotiable)

These are hard constraints, not suggestions.

Privacy & Ethics

No collection of:

Contacts

Messages

Audio
without explicit, informed consent

No real elder PII in demos

Opt-out must be obvious

Default behavior must be non-destructive (warn before block)

Technical Constraints

Prefer on-device inference

If server is used:

Clearly describe what data is sent

Keep retention minimal

Low bandwidth aware

Runs on modest Android phones

Language Requirement

English plus at least one Indian language
(Hindi / Kannada / Bengali / Tamil etc.)

Explainability Requirement

Every detection must show:

A short, human-readable reason
Example:

“Urgent money demand + OTP ask”

This is mandatory and heavily judged.

5. Judging Criteria (How You’ll Be Scored)

This tells you what the system should optimize for:

Real-world viability (25%)

Runs on low-end phones

Minimal taps

Bilingual UI

Detection quality (25%)

Precision / recall on hidden test data

Not over-blocking legitimate messages

Explainability & safety (20%)

Clear reasons

Safe defaults

Privacy by design

Design for seniors (15%)

Large fonts

Voice guidance

Simple flows

Panic / quick-dial options

Path to piloting (15%)

How would this be deployed?

WhatsApp / IVR / app feasibility

Abuse handling & telemetry ethics