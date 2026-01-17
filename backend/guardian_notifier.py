"""
Twilio Guardian Notification Service
Sends SMS alerts to family members when scam detected
"""
import os
from twilio.rest import Client
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GuardianNotifier:
    """Sends SMS alerts to family guardians via Twilio"""
    
    def __init__(self):
        self.account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        self.from_number = os.getenv('TWILIO_FROM_NUMBER')
        self.guardian_number = os.getenv('GUARDIAN_PHONE_NUMBER')
        
        self.client = None
        self.enabled = False
        
        if self.account_sid and self.auth_token and self.from_number:
            try:
                self.client = Client(self.account_sid, self.auth_token)
                self.enabled = True
                print("[INFO] Twilio Guardian Notifier initialized")
            except Exception as e:
                print(f"[WARN] Twilio init failed: {e}")
        else:
            print("[INFO] Twilio not configured - SMS alerts disabled")
    
    def notify_family(
        self,
        scam_type: str,
        message_preview: str,
        confidence: float,
        guardian_name: str = "Family Member",
        senior_name: str = "Your parent",
        custom_guardian_number: Optional[str] = None
    ) -> dict:
        """
        Send SMS alert to family guardian when scam detected
        
        Returns:
            dict with 'success', 'message_sid' (if sent), 'error' (if failed)
        """
        if not self.enabled:
            return {
                'success': False,
                'error': 'Twilio not configured',
                'simulated': True
            }
        
        to_number = custom_guardian_number or self.guardian_number
        if not to_number:
            return {
                'success': False,
                'error': 'No guardian phone number configured'
            }
        
        # Create personalized, caring message
        confidence_pct = int(confidence * 100)
        preview = message_preview[:100] + "..." if len(message_preview) > 100 else message_preview
        
        sms_body = f"""🛡️ Trust Nest Alert

Hi {guardian_name},

{senior_name} just received a suspicious message that looks like a {scam_type.lower().replace('_', ' ')} attempt.

📱 Message preview:
"{preview}"

⚠️ Risk Level: {confidence_pct}%

They may need your help right now. Give them a call to check in?

- Trust Nest (Protecting your family)"""
        
        try:
            # Use WhatsApp API (prefix with 'whatsapp:')
            from_whatsapp = f"whatsapp:{self.from_number}"
            to_whatsapp = f"whatsapp:{to_number}"
            
            message = self.client.messages.create(
                body=sms_body,
                from_=from_whatsapp,
                to=to_whatsapp
            )
            
            print(f"[WhatsApp] Alert sent to {to_number}: {message.sid}")
            
            return {
                'success': True,
                'message_sid': message.sid,
                'to': to_number,
                'channel': 'whatsapp'
            }
            
        except Exception as e:
            print(f"[ERROR] SMS send failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def send_safe_exit_confirmation(
        self,
        guardian_name: str = "Family Member",
        senior_name: str = "Your parent",
        custom_guardian_number: Optional[str] = None
    ) -> dict:
        """Send confirmation that senior safely exited the scam"""
        
        if not self.enabled:
            return {'success': False, 'error': 'Twilio not configured', 'simulated': True}
        
        to_number = custom_guardian_number or self.guardian_number
        if not to_number:
            return {'success': False, 'error': 'No guardian phone number'}
        
        sms_body = f"""✅ Trust Nest Update

Good news, {guardian_name}!

{senior_name} safely ended a suspicious interaction. They used the "Safely End" feature.

No action needed - they stayed safe! 🎉

- Trust Nest"""
        
        try:
            message = self.client.messages.create(
                body=sms_body,
                from_=self.from_number,
                to=to_number
            )
            return {'success': True, 'message_sid': message.sid}
        except Exception as e:
            return {'success': False, 'error': str(e)}


# Singleton instance
_notifier = None

def get_notifier() -> GuardianNotifier:
    global _notifier
    if _notifier is None:
        _notifier = GuardianNotifier()
    return _notifier
