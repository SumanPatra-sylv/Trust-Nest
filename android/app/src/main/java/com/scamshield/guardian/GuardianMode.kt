/**
 * Guardian Mode - Family Care-Circle
 * ===================================
 * Allows seniors to pair with trusted family members (children/caregivers).
 * 
 * Features:
 * 1. Pairing via QR code or link
 * 2. FCM push to guardian on high-risk events
 * 3. Guardian can approve/block numbers remotely
 * 4. Guardian actions override ML decisions
 */

package com.scamshield.guardian

import android.content.Context
import android.util.Log
import com.google.firebase.messaging.FirebaseMessaging
import kotlinx.coroutines.tasks.await
import org.json.JSONObject
import java.util.UUID

/**
 * Guardian pairing and management
 */
class GuardianManager(private val context: Context) {
    
    companion object {
        private const val TAG = "GuardianMode"
        private const val PREFS_NAME = "guardian_prefs"
        private const val KEY_GUARDIAN_ID = "guardian_id"
        private const val KEY_GUARDIAN_TOKEN = "guardian_fcm_token"
        private const val KEY_PAIRING_CODE = "pairing_code"
        private const val KEY_IS_PAIRED = "is_paired"
    }
    
    private val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    
    /**
     * Check if a guardian is currently paired
     */
    fun isPaired(): Boolean = prefs.getBoolean(KEY_IS_PAIRED, false)
    
    /**
     * Generate a pairing code for QR/link sharing
     * Valid for 24 hours
     */
    fun generatePairingCode(): String {
        val code = UUID.randomUUID().toString().substring(0, 8).uppercase()
        prefs.edit()
            .putString(KEY_PAIRING_CODE, code)
            .putLong("pairing_code_expiry", System.currentTimeMillis() + 24 * 60 * 60 * 1000)
            .apply()
        
        Log.i(TAG, "Generated pairing code: $code")
        return code
    }
    
    /**
     * Complete pairing with guardian
     */
    fun completePairing(guardianId: String, guardianFcmToken: String) {
        prefs.edit()
            .putString(KEY_GUARDIAN_ID, guardianId)
            .putString(KEY_GUARDIAN_TOKEN, guardianFcmToken)
            .putBoolean(KEY_IS_PAIRED, true)
            .remove(KEY_PAIRING_CODE)
            .apply()
        
        Log.i(TAG, "Paired with guardian: $guardianId")
    }
    
    /**
     * Unpair from guardian
     */
    fun unpair() {
        prefs.edit()
            .remove(KEY_GUARDIAN_ID)
            .remove(KEY_GUARDIAN_TOKEN)
            .putBoolean(KEY_IS_PAIRED, false)
            .apply()
        
        Log.i(TAG, "Unpaired from guardian")
    }
    
    /**
     * Get guardian FCM token for sending alerts
     */
    fun getGuardianToken(): String? = prefs.getString(KEY_GUARDIAN_TOKEN, null)
}


/**
 * Alert sender - sends high-risk events to guardian via FCM
 */
class GuardianAlertSender(private val guardianManager: GuardianManager) {
    
    companion object {
        private const val TAG = "GuardianAlert"
    }
    
    /**
     * Send alert to guardian about a potential scam
     */
    suspend fun sendScamAlert(
        messagePreview: String,
        riskLevel: String,
        reason: String,
        senderNumber: String?
    ): Boolean {
        if (!guardianManager.isPaired()) {
            Log.d(TAG, "No guardian paired, skipping alert")
            return false
        }
        
        val guardianToken = guardianManager.getGuardianToken() ?: return false
        
        val alertData = JSONObject().apply {
            put("type", "SCAM_ALERT")
            put("risk_level", riskLevel)
            put("reason", reason)
            put("message_preview", messagePreview.take(100))
            put("sender", senderNumber ?: "Unknown")
            put("timestamp", System.currentTimeMillis())
            put("requires_action", true)
        }
        
        Log.i(TAG, "Sending scam alert to guardian: $riskLevel")
        
        // In production: Use FCM HTTP v1 API or your backend
        // For hackathon demo, we log the intent
        return sendFcmMessage(guardianToken, alertData)
    }
    
    /**
     * Send call warning to guardian
     */
    suspend fun sendCallAlert(
        callerNumber: String,
        callDuration: Long,
        userReportedOtpRequest: Boolean
    ): Boolean {
        if (!guardianManager.isPaired()) return false
        
        val guardianToken = guardianManager.getGuardianToken() ?: return false
        
        val alertData = JSONObject().apply {
            put("type", "CALL_ALERT")
            put("caller", callerNumber)
            put("duration_seconds", callDuration)
            put("otp_requested", userReportedOtpRequest)
            put("timestamp", System.currentTimeMillis())
        }
        
        Log.i(TAG, "Sending call alert to guardian")
        return sendFcmMessage(guardianToken, alertData)
    }
    
    private suspend fun sendFcmMessage(token: String, data: JSONObject): Boolean {
        // Placeholder for FCM sending
        // In production: Use Firebase Admin SDK or backend proxy
        Log.d(TAG, "Would send FCM to $token: $data")
        return true
    }
}


/**
 * Guardian response handler - processes approve/block decisions
 */
class GuardianResponseHandler(private val context: Context) {
    
    companion object {
        private const val TAG = "GuardianResponse"
    }
    
    /**
     * Handle guardian's decision to approve a number
     */
    fun handleApprove(senderNumber: String) {
        Log.i(TAG, "Guardian approved: $senderNumber")
        WhitelistManager(context).addToWhitelist(senderNumber)
    }
    
    /**
     * Handle guardian's decision to block a number
     */
    fun handleBlock(senderNumber: String) {
        Log.i(TAG, "Guardian blocked: $senderNumber")
        BlocklistManager(context).addToBlocklist(senderNumber)
    }
}


/**
 * Simple whitelist manager
 */
class WhitelistManager(context: Context) {
    private val prefs = context.getSharedPreferences("whitelist", Context.MODE_PRIVATE)
    
    fun addToWhitelist(number: String) {
        val current = getWhitelist().toMutableSet()
        current.add(normalizeNumber(number))
        prefs.edit().putStringSet("numbers", current).apply()
    }
    
    fun isWhitelisted(number: String): Boolean {
        return normalizeNumber(number) in getWhitelist()
    }
    
    fun getWhitelist(): Set<String> {
        return prefs.getStringSet("numbers", emptySet()) ?: emptySet()
    }
    
    private fun normalizeNumber(number: String): String {
        return number.replace(Regex("[^0-9+]"), "")
    }
}


/**
 * Simple blocklist manager
 */
class BlocklistManager(context: Context) {
    private val prefs = context.getSharedPreferences("blocklist", Context.MODE_PRIVATE)
    
    fun addToBlocklist(number: String) {
        val current = getBlocklist().toMutableSet()
        current.add(normalizeNumber(number))
        prefs.edit().putStringSet("numbers", current).apply()
    }
    
    fun isBlocked(number: String): Boolean {
        return normalizeNumber(number) in getBlocklist()
    }
    
    fun getBlocklist(): Set<String> {
        return prefs.getStringSet("numbers", emptySet()) ?: emptySet()
    }
    
    private fun normalizeNumber(number: String): String {
        return number.replace(Regex("[^0-9+]"), "")
    }
}
