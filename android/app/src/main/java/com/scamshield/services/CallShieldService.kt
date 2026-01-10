/**
 * CallShieldService - Phone Call Protection
 * ==========================================
 * Provides warnings for incoming calls using CallScreeningService.
 * 
 * IMPORTANT: This is METADATA-BASED only.
 * No audio recording or transcription during calls.
 */

package com.scamshield.services

import android.os.Build
import android.telecom.Call
import android.telecom.CallScreeningService
import android.util.Log
import androidx.annotation.RequiresApi

@RequiresApi(Build.VERSION_CODES.N)
class CallShieldService : CallScreeningService() {
    
    companion object {
        private const val TAG = "CallShield"
    }
    
    override fun onScreenCall(callDetails: Call.Details) {
        val handle = callDetails.handle
        val callerNumber = handle?.schemeSpecificPart ?: "Unknown"
        
        Log.d(TAG, "Screening call from: $callerNumber")
        
        // Analyze caller based on metadata only
        val riskLevel = analyzeCallerRisk(callerNumber, callDetails)
        
        val response = CallResponse.Builder().apply {
            when (riskLevel) {
                CallRisk.HIGH -> {
                    // Don't auto-reject, but show warning
                    setDisallowCall(false)
                    setRejectCall(false)
                    // Note: We NEVER silently reject. User always decides.
                    Log.w(TAG, "HIGH RISK caller: $callerNumber")
                }
                CallRisk.MEDIUM -> {
                    setDisallowCall(false)
                    setRejectCall(false)
                    Log.i(TAG, "MEDIUM RISK caller: $callerNumber")
                }
                CallRisk.LOW -> {
                    setDisallowCall(false)
                    setRejectCall(false)
                    Log.d(TAG, "LOW RISK caller: $callerNumber")
                }
            }
        }.build()
        
        respondToCall(callDetails, response)
        
        // Show overlay warning for high/medium risk
        if (riskLevel != CallRisk.LOW) {
            showPreCallWarning(callerNumber, riskLevel)
        }
    }
    
    private fun analyzeCallerRisk(number: String, details: Call.Details): CallRisk {
        // Check if number is in contacts (would need READ_CONTACTS permission)
        // For now, use simple heuristics
        
        // Unknown/hidden numbers are suspicious
        if (number == "Unknown" || number.isEmpty()) {
            return CallRisk.MEDIUM
        }
        
        // Check against local blocklist (stored in Room database)
        // if (BlocklistRepository.isBlocked(number)) return CallRisk.HIGH
        
        // International numbers (non-India) are suspicious
        if (!number.startsWith("+91") && number.startsWith("+")) {
            return CallRisk.MEDIUM
        }
        
        // Default to low risk
        return CallRisk.LOW
    }
    
    private fun showPreCallWarning(callerNumber: String, risk: CallRisk) {
        // Show overlay warning before user picks up
        // This is done via a system alert window or notification
        Log.i(TAG, "Showing pre-call warning for $callerNumber (Risk: $risk)")
        
        // In production: Use WindowManager to show overlay
        // Or send a high-priority notification
    }
    
    enum class CallRisk {
        LOW, MEDIUM, HIGH
    }
}


/**
 * InCallMonitor - Duration-based warning system
 * =============================================
 * Shows reminder banners during long calls.
 * 
 * This is NOT audio-based. It uses call state + duration only.
 */
class InCallMonitor {
    
    companion object {
        private const val WARNING_1_SECONDS = 60L   // 1 minute
        private const val WARNING_2_SECONDS = 120L  // 2 minutes
        private const val WARNING_3_SECONDS = 300L  // 5 minutes
    }
    
    private var callStartTime: Long = 0
    private var warningShown = mutableSetOf<Long>()
    
    fun onCallStarted() {
        callStartTime = System.currentTimeMillis()
        warningShown.clear()
    }
    
    fun checkDuration(): String? {
        val durationSeconds = (System.currentTimeMillis() - callStartTime) / 1000
        
        return when {
            durationSeconds >= WARNING_3_SECONDS && WARNING_3_SECONDS !in warningShown -> {
                warningShown.add(WARNING_3_SECONDS)
                "⚠️ Long call (5+ min). Remember: Never share OTP!"
            }
            durationSeconds >= WARNING_2_SECONDS && WARNING_2_SECONDS !in warningShown -> {
                warningShown.add(WARNING_2_SECONDS)
                "❓ 2 minutes on call. Is this a known contact?"
            }
            durationSeconds >= WARNING_1_SECONDS && WARNING_1_SECONDS !in warningShown -> {
                warningShown.add(WARNING_1_SECONDS)
                "💡 Reminder: Banks never ask for OTP on call."
            }
            else -> null
        }
    }
    
    fun onCallEnded(): PostCallSummary {
        val duration = (System.currentTimeMillis() - callStartTime) / 1000
        return PostCallSummary(
            durationSeconds = duration,
            // These would be filled by user post-call survey
            userReportedOtpRequest = false,
            userReportedPaymentRequest = false
        )
    }
}

data class PostCallSummary(
    val durationSeconds: Long,
    val userReportedOtpRequest: Boolean,
    val userReportedPaymentRequest: Boolean
)
