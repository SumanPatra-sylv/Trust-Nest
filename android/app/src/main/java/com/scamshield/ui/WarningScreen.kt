/**
 * Warning Screen - Jetpack Compose UI
 * ====================================
 * Full-screen warning shown when a scam is detected.
 * Senior-friendly design with large fonts and clear actions.
 */

package com.scamshield.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material.icons.filled.Block
import androidx.compose.material.icons.filled.Phone
import androidx.compose.material.icons.filled.Report
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

// Color scheme
private val DangerRed = Color(0xFFDC2626)
private val DarkRed = Color(0xFF7F1D1D)
private val WarningYellow = Color(0xFFFBBF24)
private val SafeGreen = Color(0xFF22C55E)
private val DarkBackground = Color(0xFF0F0F0F)

@Composable
fun ScamWarningScreen(
    riskLevel: String,
    confidence: Float,
    reasonEn: String,
    reasonHi: String,
    actionEn: String,
    actionHi: String,
    messagePreview: String,
    onBlock: () -> Unit,
    onAskFamily: () -> Unit,
    onReport: () -> Unit,
    onDismiss: () -> Unit
) {
    val backgroundColor = when (riskLevel) {
        "SCAM" -> Brush.verticalGradient(listOf(DarkRed, DarkBackground))
        "SUSPICIOUS" -> Brush.verticalGradient(listOf(Color(0xFF78350F), DarkBackground))
        else -> Brush.verticalGradient(listOf(Color(0xFF14532D), DarkBackground))
    }
    
    val accentColor = when (riskLevel) {
        "SCAM" -> DangerRed
        "SUSPICIOUS" -> WarningYellow
        else -> SafeGreen
    }
    
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(backgroundColor)
            .padding(24.dp)
    ) {
        Column(
            modifier = Modifier.fillMaxSize(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            // Header
            Column(
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Icon(
                    imageVector = Icons.Default.Warning,
                    contentDescription = "Warning",
                    tint = accentColor,
                    modifier = Modifier.size(80.dp)
                )
                
                Spacer(modifier = Modifier.height(16.dp))
                
                Text(
                    text = when (riskLevel) {
                        "SCAM" -> "⚠️ खतरा! SCAM DETECTED"
                        "SUSPICIOUS" -> "⚠️ संदेह! SUSPICIOUS"
                        else -> "✅ सुरक्षित SAFE"
                    },
                    fontSize = 32.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White,
                    textAlign = TextAlign.Center
                )
                
                Text(
                    text = "(${(confidence * 100).toInt()}% confidence)",
                    fontSize = 18.sp,
                    color = Color.White.copy(alpha = 0.7f)
                )
            }
            
            // Reason card
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = Color.White.copy(alpha = 0.1f)
                )
            ) {
                Column(
                    modifier = Modifier.padding(20.dp)
                ) {
                    Text(
                        text = "Why this is dangerous / यह खतरनाक क्यों है:",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Medium,
                        color = Color.White.copy(alpha = 0.7f)
                    )
                    
                    Spacer(modifier = Modifier.height(12.dp))
                    
                    Text(
                        text = reasonEn,
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color.White
                    )
                    
                    Spacer(modifier = Modifier.height(8.dp))
                    
                    Text(
                        text = reasonHi,
                        fontSize = 18.sp,
                        color = Color.White.copy(alpha = 0.9f)
                    )
                }
            }
            
            // Message preview
            if (messagePreview.isNotEmpty()) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = Color.Black.copy(alpha = 0.3f)
                    )
                ) {
                    Text(
                        text = "\"${messagePreview}...\"",
                        fontSize = 14.sp,
                        color = Color.White.copy(alpha = 0.6f),
                        modifier = Modifier.padding(16.dp),
                        maxLines = 3
                    )
                }
            }
            
            // Action buttons
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                // Primary action: Block
                Button(
                    onClick = onBlock,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(60.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = DangerRed
                    ),
                    shape = RoundedCornerShape(16.dp)
                ) {
                    Icon(Icons.Default.Block, contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text(
                        text = "Block & Delete / ब्लॉक करें",
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
                
                // Ask family
                OutlinedButton(
                    onClick = onAskFamily,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(56.dp),
                    shape = RoundedCornerShape(16.dp),
                    colors = ButtonDefaults.outlinedButtonColors(
                        contentColor = Color.White
                    )
                ) {
                    Icon(Icons.Default.Phone, contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text(
                        text = "Ask Family / परिवार से पूछें",
                        fontSize = 16.sp
                    )
                }
                
                // Report
                TextButton(
                    onClick = onReport,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(Icons.Default.Report, contentDescription = null, tint = Color.White.copy(alpha = 0.7f))
                    Spacer(Modifier.width(8.dp))
                    Text(
                        text = "Report Scam",
                        fontSize = 14.sp,
                        color = Color.White.copy(alpha = 0.7f)
                    )
                }
            }
        }
    }
}


/**
 * Teach-Me Mode - Educational screen
 */
@Composable
fun TeachMeScreen() {
    val lessons = listOf(
        Lesson(
            titleEn = "Banks NEVER ask OTP",
            titleHi = "बैंक कभी OTP नहीं माँगते",
            descEn = "If someone asks for OTP on call/message, it's 100% scam.",
            descHi = "अगर कोई कॉल/मैसेज पर OTP माँगे, तो यह 100% स्कैम है।"
        ),
        Lesson(
            titleEn = "Police NEVER video call",
            titleHi = "पुलिस कभी वीडियो कॉल नहीं करती",
            descEn = "'Digital Arrest' is fake. Real police sends summons by post.",
            descHi = "'डिजिटल अरेस्ट' नकली है। असली पुलिस डाक से समन भेजती है।"
        ),
        Lesson(
            titleEn = "No refund needs QR scan",
            titleHi = "रिफंड के लिए QR नहीं चाहिए",
            descEn = "To receive money, you don't scan QR. Only to pay!",
            descHi = "पैसे लेने के लिए QR स्कैन नहीं होता, सिर्फ देने के लिए!"
        ),
        Lesson(
            titleEn = "Government doesn't WhatsApp",
            titleHi = "सरकार WhatsApp नहीं करती",
            descEn = "IT, CBI, Police don't message on WhatsApp for official work.",
            descHi = "IT, CBI, पुलिस आधिकारिक काम के लिए WhatsApp नहीं करते।"
        )
    )
    
    // Implementation would show lessons in a scrollable list
    // with Text-to-Speech button for each lesson
}

data class Lesson(
    val titleEn: String,
    val titleHi: String,
    val descEn: String,
    val descHi: String
)
