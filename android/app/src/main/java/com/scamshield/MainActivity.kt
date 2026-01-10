/**
 * MainActivity - Main entry point
 * ================================
 * Home screen with dashboard showing protection status.
 */

package com.scamshield

import android.content.Intent
import android.os.Bundle
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

// Theme colors
private val DarkBackground = Color(0xFF0A0A0F)
private val CardBackground = Color(0xFF12121A)
private val AccentPurple = Color(0xFF8B5CF6)
private val AccentCyan = Color(0xFF06B6D4)
private val SafeGreen = Color(0xFF22C55E)
private val DangerRed = Color(0xFFEF4444)

class MainActivity : ComponentActivity() {
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        setContent {
            MaterialTheme(
                colorScheme = darkColorScheme(
                    primary = AccentPurple,
                    secondary = AccentCyan,
                    background = DarkBackground,
                    surface = CardBackground
                )
            ) {
                MainScreen(
                    onEnableServices = { openNotificationSettings() },
                    onOpenTeachMe = { /* Navigate to TeachMe */ },
                    onOpenGuardian = { /* Navigate to Guardian settings */ }
                )
            }
        }
    }
    
    private fun openNotificationSettings() {
        val intent = Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS)
        startActivity(intent)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(
    onEnableServices: () -> Unit,
    onOpenTeachMe: () -> Unit,
    onOpenGuardian: () -> Unit
) {
    var protectionEnabled by remember { mutableStateOf(false) }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { 
                    Text(
                        "ScamShield",
                        fontWeight = FontWeight.Bold
                    ) 
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = DarkBackground
                )
            )
        },
        containerColor = DarkBackground
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Protection Status Card
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = CardBackground)
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(
                            Brush.linearGradient(
                                colors = if (protectionEnabled) 
                                    listOf(SafeGreen.copy(alpha = 0.2f), CardBackground)
                                else 
                                    listOf(DangerRed.copy(alpha = 0.2f), CardBackground)
                            )
                        )
                        .padding(24.dp)
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Box(
                            modifier = Modifier
                                .size(100.dp)
                                .clip(CircleShape)
                                .background(
                                    if (protectionEnabled) SafeGreen.copy(alpha = 0.2f)
                                    else DangerRed.copy(alpha = 0.2f)
                                ),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                imageVector = if (protectionEnabled) Icons.Default.Shield else Icons.Default.Warning,
                                contentDescription = null,
                                tint = if (protectionEnabled) SafeGreen else DangerRed,
                                modifier = Modifier.size(48.dp)
                            )
                        }
                        
                        Spacer(modifier = Modifier.height(16.dp))
                        
                        Text(
                            text = if (protectionEnabled) "सुरक्षित / Protected" else "असुरक्षित / Not Protected",
                            fontSize = 24.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color.White
                        )
                        
                        Text(
                            text = if (protectionEnabled) 
                                "All shields are active"
                            else 
                                "Please enable notification access",
                            fontSize = 14.sp,
                            color = Color.White.copy(alpha = 0.7f)
                        )
                        
                        if (!protectionEnabled) {
                            Spacer(modifier = Modifier.height(16.dp))
                            Button(
                                onClick = onEnableServices,
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = AccentPurple
                                )
                            ) {
                                Text("Enable Protection")
                            }
                        }
                    }
                }
            }
            
            // Shield Cards
            Text(
                text = "Active Shields",
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                ShieldCard(
                    modifier = Modifier.weight(1f),
                    icon = Icons.Default.Message,
                    title = "Message Shield",
                    subtitle = "SMS & WhatsApp",
                    isActive = protectionEnabled
                )
                ShieldCard(
                    modifier = Modifier.weight(1f),
                    icon = Icons.Default.Phone,
                    title = "Call Shield",
                    subtitle = "Voice Calls",
                    isActive = protectionEnabled
                )
            }
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                ShieldCard(
                    modifier = Modifier.weight(1f),
                    icon = Icons.Default.Videocam,
                    title = "Digital Arrest",
                    subtitle = "Video Calls",
                    isActive = protectionEnabled
                )
                ShieldCard(
                    modifier = Modifier.weight(1f),
                    icon = Icons.Default.People,
                    title = "Guardian",
                    subtitle = "Family Alert",
                    isActive = false,
                    onClick = onOpenGuardian
                )
            }
            
            // Quick Actions
            Text(
                text = "Quick Actions",
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White,
                modifier = Modifier.padding(top = 8.dp)
            )
            
            ActionCard(
                icon = Icons.Default.School,
                title = "Teach Me / सीखें",
                subtitle = "Learn to spot scams",
                onClick = onOpenTeachMe
            )
            
            ActionCard(
                icon = Icons.Default.Report,
                title = "Report Scam",
                subtitle = "Help protect others",
                onClick = { }
            )
        }
    }
}

@Composable
fun ShieldCard(
    modifier: Modifier = Modifier,
    icon: ImageVector,
    title: String,
    subtitle: String,
    isActive: Boolean,
    onClick: (() -> Unit)? = null
) {
    Card(
        modifier = modifier,
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = CardBackground),
        onClick = onClick ?: {}
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Box(
                modifier = Modifier
                    .size(48.dp)
                    .clip(CircleShape)
                    .background(if (isActive) SafeGreen.copy(alpha = 0.2f) else Color.Gray.copy(alpha = 0.2f)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    tint = if (isActive) SafeGreen else Color.Gray
                )
            }
            Spacer(modifier = Modifier.height(8.dp))
            Text(title, fontSize = 14.sp, fontWeight = FontWeight.Medium, color = Color.White)
            Text(subtitle, fontSize = 12.sp, color = Color.White.copy(alpha = 0.5f))
        }
    }
}

@Composable
fun ActionCard(
    icon: ImageVector,
    title: String,
    subtitle: String,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = CardBackground),
        onClick = onClick
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(48.dp)
                    .clip(CircleShape)
                    .background(AccentPurple.copy(alpha = 0.2f)),
                contentAlignment = Alignment.Center
            ) {
                Icon(icon, contentDescription = null, tint = AccentPurple)
            }
            Spacer(modifier = Modifier.width(16.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(title, fontSize = 16.sp, fontWeight = FontWeight.Medium, color = Color.White)
                Text(subtitle, fontSize = 12.sp, color = Color.White.copy(alpha = 0.5f))
            }
            Icon(Icons.Default.ChevronRight, contentDescription = null, tint = Color.White.copy(alpha = 0.5f))
        }
    }
}
