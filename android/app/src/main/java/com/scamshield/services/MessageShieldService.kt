/**
 * MessageShieldService - NotificationListenerService
 * ===================================================
 * Monitors incoming SMS and WhatsApp notifications for scam detection.
 * 
 * This runs in background and triggers overlay warnings on scam detection.
 */

package com.scamshield.services

import android.app.Notification
import android.content.Intent
import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import android.util.Log
import com.scamshield.detection.RuleEngine
import com.scamshield.detection.RiskLevel

class MessageShieldService : NotificationListenerService() {
    
    companion object {
        private const val TAG = "MessageShield"
        
        // App packages to monitor
        private val MONITORED_PACKAGES = setOf(
            "com.whatsapp",
            "com.whatsapp.w4b", // WhatsApp Business
            "com.google.android.apps.messaging", // Google Messages
            "com.android.mms" // Default SMS
        )
    }
    
    private lateinit var ruleEngine: RuleEngine
    
    override fun onCreate() {
        super.onCreate()
        ruleEngine = RuleEngine()
        Log.i(TAG, "MessageShieldService started")
    }
    
    override fun onNotificationPosted(sbn: StatusBarNotification) {
        val packageName = sbn.packageName
        
        // Only process monitored apps
        if (packageName !in MONITORED_PACKAGES) return
        
        val notification = sbn.notification
        val extras = notification.extras
        
        // Extract message text
        val title = extras.getCharSequence(Notification.EXTRA_TITLE)?.toString() ?: ""
        val text = extras.getCharSequence(Notification.EXTRA_TEXT)?.toString() ?: ""
        val bigText = extras.getCharSequence(Notification.EXTRA_BIG_TEXT)?.toString() ?: ""
        
        val messageText = "$title $text $bigText".trim()
        
        if (messageText.isEmpty()) return
        
        Log.d(TAG, "Processing notification from $packageName")
        
        // Analyze message
        val result = ruleEngine.analyze(messageText)
        
        // Take action based on risk level
        when (result.riskLevel) {
            RiskLevel.SCAM -> {
                Log.w(TAG, "SCAM DETECTED: ${result.reasonEn}")
                showScamWarning(result, messageText)
            }
            RiskLevel.SUSPICIOUS -> {
                Log.i(TAG, "Suspicious message: ${result.reasonEn}")
                showWarningNotification(result, messageText)
            }
            RiskLevel.SAFE -> {
                Log.d(TAG, "Message appears safe")
            }
        }
    }
    
    private fun showScamWarning(result: com.scamshield.detection.DetectionResult, message: String) {
        // Show full-screen overlay warning
        val intent = Intent(this, WarningOverlayActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK
            putExtra("risk_level", result.riskLevel.name)
            putExtra("confidence", result.confidence)
            putExtra("reason_en", result.reasonEn)
            putExtra("reason_hi", result.reasonHi)
            putExtra("action_en", result.recommendedAction)
            putExtra("action_hi", result.recommendedActionHi)
            putExtra("message_preview", message.take(100))
        }
        startActivity(intent)
    }
    
    private fun showWarningNotification(result: com.scamshield.detection.DetectionResult, message: String) {
        // Show a warning notification (less intrusive than overlay)
        // Implementation would use NotificationCompat.Builder
        Log.i(TAG, "Would show notification for: ${result.reasonEn}")
    }
    
    override fun onNotificationRemoved(sbn: StatusBarNotification) {
        // Optional: cleanup if needed
    }
    
    override fun onListenerConnected() {
        Log.i(TAG, "NotificationListenerService connected")
    }
    
    override fun onListenerDisconnected() {
        Log.w(TAG, "NotificationListenerService disconnected")
    }
}


/**
 * WarningOverlayActivity - Full screen warning
 * Shown immediately when a scam is detected.
 */
class WarningOverlayActivity : android.app.Activity() {
    // This would be implemented with Jetpack Compose UI
    // See ui/WarningScreen.kt for the actual implementation
}
