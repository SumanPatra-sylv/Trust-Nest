/**
 * ScamShield - Rule Engine (Kotlin)
 * ==================================
 * Deterministic scam detection that runs ON-DEVICE.
 * This is ported from Python rule_engine.py.
 * 
 * Target: <10ms execution on low-end phones
 */

package com.scamshield.detection

import java.util.regex.Pattern

enum class RiskLevel {
    SAFE, SUSPICIOUS, SCAM
}

data class DetectionResult(
    val riskLevel: RiskLevel,
    val confidence: Float,
    val reasonEn: String,
    val reasonHi: String,
    val signals: Map<String, Boolean>,
    val triggeredRules: List<String>,
    val recommendedAction: String,
    val recommendedActionHi: String
)

class RuleEngine {
    
    // URL shorteners commonly used in scams
    private val urlShorteners = setOf(
        "bit.ly", "tinyurl.com", "rebrand.ly", "cutt.ly", "t.co",
        "goo.gl", "ow.ly", "is.gd", "buff.ly", "short.link"
    )
    
    // Phishing domains
    private val suspiciousDomains = listOf(
        "verify-now.co", "secure-kyc-update.in", "bank-kycverify.com",
        "track-parcel.in", "customs-clearance.in", "pay-safe.link",
        "gpay-help.in", "support-helpdesk.in", "cybercrime-case.in"
    )
    
    // Pre-compiled patterns for speed
    private val urlPattern = Pattern.compile("https?://[^\\s]+", Pattern.CASE_INSENSITIVE)
    private val upiPattern = Pattern.compile("[a-zA-Z0-9._-]+@[a-zA-Z]+", Pattern.CASE_INSENSITIVE)
    private val phonePattern = Pattern.compile("\\+91[-\\s]?\\d{5}[-\\s]?\\d{5}")
    
    private val otpPattern = Pattern.compile(
        "\\botp\\b|\\bpin\\b|one.?time.?password|verification.?code|share.?otp|otp.?batao",
        Pattern.CASE_INSENSITIVE
    )
    
    private val urgencyPattern = Pattern.compile(
        "today only|expires today|within \\d+ hours?|immediately|urgent|jaldi|abhi|turant",
        Pattern.CASE_INSENSITIVE
    )
    
    private val threatPattern = Pattern.compile(
        "account.?freeze|account.?block|will be.?blocked|legal.?action|fir|arrested|" +
        "digital.?arrest|deactivat|suspend|power.?cut|disconnection",
        Pattern.CASE_INSENSITIVE
    )
    
    private val authorityPattern = Pattern.compile(
        "\\b(cbi|police|trai|income.?tax|cyber.?crime|rbi|sebi)\\b|" +
        "court.?notice|settlement.?required|legal.?notice",
        Pattern.CASE_INSENSITIVE
    )
    
    private val digitalArrestPattern = Pattern.compile(
        "digital.?arrest|stay.?on.?line|do.?not.?disconnect|video.?call|" +
        "screen.?shar|install.?anydesk|install.?teamviewer",
        Pattern.CASE_INSENSITIVE
    )
    
    private val familyScamPatterns = listOf(
        "new.?number", "phone.?lost", "don'?t.?call", "emergency", "it'?s.?me"
    )
    
    fun analyze(text: String): DetectionResult {
        val textLower = text.lowercase()
        val signals = mutableMapOf(
            "has_url" to false,
            "has_shortener" to false,
            "has_upi" to false,
            "has_otp_request" to false,
            "has_urgency" to false,
            "has_threat" to false,
            "has_authority_claim" to false,
            "has_phone_number" to false
        )
        val reasonsEn = mutableListOf<String>()
        val reasonsHi = mutableListOf<String>()
        val triggeredRules = mutableListOf<String>()
        var scamScore = 0
        
        // Rule 1: Check for URLs and shorteners
        val urlMatcher = urlPattern.matcher(text)
        while (urlMatcher.find()) {
            signals["has_url"] = true
            val url = urlMatcher.group().lowercase()
            
            for (shortener in urlShorteners) {
                if (url.contains(shortener)) {
                    signals["has_shortener"] = true
                    scamScore += 20
                    reasonsEn.add("Shortened URL detected")
                    reasonsHi.add("छोटा URL है")
                    triggeredRules.add("URL_SHORTENER")
                    break
                }
            }
            
            for (domain in suspiciousDomains) {
                if (url.contains(domain)) {
                    scamScore += 40
                    reasonsEn.add("Known phishing domain")
                    reasonsHi.add("फिशिंग वेबसाइट")
                    triggeredRules.add("PHISHING_DOMAIN")
                    break
                }
            }
        }
        
        // Rule 2: Check for UPI IDs
        if (upiPattern.matcher(text).find()) {
            signals["has_upi"] = true
            scamScore += 15
            reasonsEn.add("UPI payment ID present")
            reasonsHi.add("UPI पेमेंट ID है")
            triggeredRules.add("UPI_PRESENT")
        }
        
        // Rule 3: Check for OTP requests
        if (otpPattern.matcher(text).find()) {
            signals["has_otp_request"] = true
            scamScore += 35
            reasonsEn.add("OTP request - Banks NEVER ask!")
            reasonsHi.add("OTP माँग रहा है - बैंक कभी नहीं माँगते!")
            triggeredRules.add("OTP_REQUEST")
        }
        
        // Rule 4: Check for urgency
        if (urgencyPattern.matcher(text).find()) {
            signals["has_urgency"] = true
            scamScore += 15
            reasonsEn.add("Creates false urgency")
            reasonsHi.add("जल्दी का दबाव")
            triggeredRules.add("URGENCY")
        }
        
        // Rule 5: Check for threats
        if (threatPattern.matcher(text).find()) {
            signals["has_threat"] = true
            scamScore += 30
            reasonsEn.add("Threatening language")
            reasonsHi.add("धमकी दे रहा है")
            triggeredRules.add("THREAT")
        }
        
        // Rule 6: Authority impersonation
        if (authorityPattern.matcher(text).find()) {
            signals["has_authority_claim"] = true
            scamScore += 25
            reasonsEn.add("Fake authority claim")
            reasonsHi.add("नकली सरकारी संदेश")
            triggeredRules.add("AUTHORITY")
        }
        
        // Rule 7: Phone number
        if (phonePattern.matcher(text).find()) {
            signals["has_phone_number"] = true
            scamScore += 10
            triggeredRules.add("PHONE_NUMBER")
        }
        
        // Rule 8: Digital Arrest
        if (digitalArrestPattern.matcher(textLower).find()) {
            scamScore += 50
            reasonsEn.add("CRITICAL: Digital Arrest scam!")
            reasonsHi.add("खतरा: डिजिटल अरेस्ट स्कैम!")
            triggeredRules.add("DIGITAL_ARREST")
        }
        
        // Rule 9: Family Impersonation
        var familyScore = 0
        for (pattern in familyScamPatterns) {
            if (Pattern.compile(pattern, Pattern.CASE_INSENSITIVE).matcher(textLower).find()) {
                familyScore++
            }
        }
        if (familyScore >= 2 && signals["has_upi"] == true) {
            scamScore += 45
            reasonsEn.add("Family scam! Verify old number!")
            reasonsHi.add("परिवार का नकली संदेश! पुराने नंबर पर कॉल करें!")
            triggeredRules.add("FAMILY_IMPERSONATION")
        }
        
        // Determine risk level
        val (riskLevel, confidence) = when {
            scamScore >= 60 -> RiskLevel.SCAM to minOf(0.95f, 0.6f + (scamScore - 60) * 0.01f)
            scamScore >= 30 -> RiskLevel.SUSPICIOUS to (0.5f + (scamScore - 30) * 0.01f)
            else -> RiskLevel.SAFE to maxOf(0.3f, 1.0f - scamScore * 0.02f)
        }
        
        // Generate recommendation
        val (actionEn, actionHi) = when (riskLevel) {
            RiskLevel.SCAM -> "⚠️ Block and report!" to "⚠️ ब्लॉक करें और रिपोर्ट करें!"
            RiskLevel.SUSPICIOUS -> "❓ Ask family first" to "❓ पहले परिवार से पूछें"
            RiskLevel.SAFE -> "✅ Appears safe" to "✅ सुरक्षित लगता है"
        }
        
        val reasonEn = if (reasonsEn.isNotEmpty()) reasonsEn.joinToString("; ") else "No issues found"
        val reasonHi = if (reasonsHi.isNotEmpty()) reasonsHi.joinToString("; ") else "कोई समस्या नहीं"
        
        return DetectionResult(
            riskLevel = riskLevel,
            confidence = confidence,
            reasonEn = reasonEn,
            reasonHi = reasonHi,
            signals = signals,
            triggeredRules = triggeredRules,
            recommendedAction = actionEn,
            recommendedActionHi = actionHi
        )
    }
}
