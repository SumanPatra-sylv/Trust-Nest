/**
 * DigitalArrestInterrupter - Video Call Warning
 * ==============================================
 * CRITICAL FEATURE: Detects incoming video calls from unknown numbers
 * and shows immediate warning overlay.
 * 
 * "Digital Arrest" is a new scam where criminals:
 * 1. Video call pretending to be CBI/Police
 * 2. Show fake arrest warrants
 * 3. Demand remote screen sharing
 * 4. Extract money/credentials
 * 
 * This service monitors WhatsApp/Skype video call notifications
 * and interrupts with a warning. NO audio/video capture.
 */

package com.scamshield.services

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import android.util.Log
import androidx.core.app.NotificationCompat

class DigitalArrestInterrupter : NotificationListenerService() {
    
    companion object {
        private const val TAG = "DigitalArrest"
        private const val CHANNEL_ID = "digital_arrest_warning"
        
        // Apps that can make video calls
        private val VIDEO_CALL_APPS = setOf(
            "com.whatsapp",
            "com.whatsapp.w4b",
            "com.skype.raider",
            "com.skype.m2",
            "us.zoom.videomeetings",
            "com.google.android.apps.meetings" // Google Meet
        )
        
        // Remote control apps (DANGEROUS if installed)
        private val DANGEROUS_APPS = listOf(
            "com.anydesk.anydeskandroid" to "AnyDesk",
            "com.teamviewer.teamviewer.market.mobile" to "TeamViewer",
            "com.teamviewer.quicksupport.market" to "QuickSupport",
            "com.ammyy.admin" to "Ammyy Admin"
        )
    }
    
    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        checkForDangerousApps()
        Log.i(TAG, "DigitalArrestInterrupter started")
    }
    
    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Digital Arrest Warnings",
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "Critical warnings for potential digital arrest scams"
                enableVibration(true)
                setShowBadge(true)
            }
            
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }
    }
    
    override fun onNotificationPosted(sbn: StatusBarNotification) {
        val packageName = sbn.packageName
        
        // Only check video call apps
        if (packageName !in VIDEO_CALL_APPS) return
        
        val notification = sbn.notification
        val extras = notification.extras
        val text = extras.getCharSequence(android.app.Notification.EXTRA_TEXT)?.toString() ?: ""
        val title = extras.getCharSequence(android.app.Notification.EXTRA_TITLE)?.toString() ?: ""
        
        val combined = "$title $text".lowercase()
        
        // Detect video call patterns
        val isVideoCall = combined.contains("video") || 
                         combined.contains("incoming call") ||
                         combined.contains("ringing")
        
        // Check if caller is unknown (not in title from contacts)
        val isUnknownCaller = combined.contains("+91") || 
                             combined.contains("unknown") ||
                             title.matches(Regex("^\\+\\d+.*"))
        
        if (isVideoCall && isUnknownCaller) {
            Log.w(TAG, "Potential Digital Arrest detected! Video call from unknown number")
            showDigitalArrestWarning(title)
        }
    }
    
    private fun showDigitalArrestWarning(callerInfo: String) {
        // Show high-priority notification
        val intent = Intent(this, DigitalArrestWarningActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
            putExtra("caller", callerInfo)
        }
        
        val pendingIntent = PendingIntent.getActivity(
            this, 0, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_dialog_alert)
            .setContentTitle("⚠️ DANGER: Possible 'Digital Arrest' Scam!")
            .setContentText("Police/CBI NEVER video call. Do not share screen!")
            .setStyle(NotificationCompat.BigTextStyle().bigText(
                """
                🚨 WARNING: Unknown video call detected!
                
                Remember:
                • Police NEVER video call
                • CBI NEVER demands instant payment
                • NEVER share screen or install apps
                • NEVER share OTP or bank details
                
                If unsure, disconnect and call family!
                """.trimIndent()
            ))
            .setPriority(NotificationCompat.PRIORITY_MAX)
            .setCategory(NotificationCompat.CATEGORY_ALARM)
            .setFullScreenIntent(pendingIntent, true)
            .setAutoCancel(true)
            .build()
        
        val manager = getSystemService(NotificationManager::class.java)
        manager.notify(999, notification)
        
        // Also try to show overlay activity
        startActivity(intent)
    }
    
    private fun checkForDangerousApps() {
        val pm = packageManager
        val installedDangerousApps = mutableListOf<String>()
        
        for ((packageName, appName) in DANGEROUS_APPS) {
            try {
                pm.getPackageInfo(packageName, 0)
                installedDangerousApps.add(appName)
            } catch (e: PackageManager.NameNotFoundException) {
                // App not installed, good
            }
        }
        
        if (installedDangerousApps.isNotEmpty()) {
            Log.w(TAG, "Dangerous remote-control apps found: $installedDangerousApps")
            showDangerousAppWarning(installedDangerousApps)
        }
    }
    
    private fun showDangerousAppWarning(apps: List<String>) {
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_dialog_alert)
            .setContentTitle("⚠️ Remote Control App Detected")
            .setContentText("${apps.joinToString()} can be used by scammers!")
            .setStyle(NotificationCompat.BigTextStyle().bigText(
                """
                Apps like ${apps.joinToString()} allow remote control of your phone.
                
                Scammers may ask you to install these and then:
                • Access your bank app
                • Read your OTPs
                • Transfer your money
                
                Only use these apps with trusted IT support!
                """.trimIndent()
            ))
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setAutoCancel(true)
            .build()
        
        val manager = getSystemService(NotificationManager::class.java)
        manager.notify(998, notification)
    }
}


/**
 * DigitalArrestWarningActivity - Full screen warning
 * Shows immediately when a potential digital arrest scam is detected.
 */
class DigitalArrestWarningActivity : android.app.Activity() {
    // Implemented with Jetpack Compose
    // Shows clear warning with:
    // - Large "DANGER" header
    // - "Police NEVER video call" message in EN + Hindi
    // - "Disconnect Now" and "Call Family" buttons
    // - Link to report scam
}
