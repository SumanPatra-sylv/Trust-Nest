"""
ScamShield Web App - Elder-Friendly UI
======================================
Full-featured web interface for judges to test all capabilities.
Large fonts, simple navigation, Hindi/English support.
"""

import streamlit as st
import sys
import os
import time
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="ScamShield - Trust Nest",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for elder-friendly design
st.markdown("""
<style>
    /* Large fonts for seniors */
    .stApp {
        font-size: 18px;
    }
    
    h1 {
        font-size: 2.5rem !important;
        color: #667eea !important;
    }
    
    h2 {
        font-size: 2rem !important;
        color: #764ba2 !important;
    }
    
    h3 {
        font-size: 1.5rem !important;
    }
    
    /* Large buttons */
    .stButton > button {
        font-size: 1.3rem !important;
        padding: 15px 30px !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
    }
    
    /* Large text area */
    .stTextArea textarea {
        font-size: 1.2rem !important;
        line-height: 1.6 !important;
    }
    
    /* Result cards */
    .safe-card {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 2px solid #28a745;
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
    }
    
    .suspicious-card {
        background: linear-gradient(135deg, #fff3cd, #ffeeba);
        border: 2px solid #ffc107;
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
    }
    
    .scam-card {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 2px solid #dc3545;
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
    }
    
    /* Large verdict text */
    .verdict-text {
        font-size: 2.5rem !important;
        font-weight: bold !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        font-size: 1.1rem;
    }
    
    /* Hindi text */
    .hindi-text {
        font-size: 1.3rem;
        color: #666;
        font-style: italic;
    }
    
    /* Feature cards */
    .feature-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    
    /* Pipeline steps */
    .pipeline-step {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        transition: all 0.3s;
    }
    
    .pipeline-step-active {
        border-color: #667eea;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Initialize detector
@st.cache_resource
def load_detector():
    """Load the scam detector (cached)."""
    try:
        from detector import ScamDetector
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, "models")
        detector = ScamDetector(model_dir=model_dir)
        return detector, True
    except Exception as e:
        st.error(f"Error loading detector: {e}")
        return None, False

# Load detector
detector, detector_loaded = load_detector()

# Sidebar navigation
st.sidebar.markdown("# 🛡️ ScamShield")
st.sidebar.markdown("### Trust Nest")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📌 Select Feature / सुविधा चुनें",
    [
        "🏠 Home / होम",
        "📱 Message Shield / संदेश सुरक्षा",
        "📞 Call Shield / कॉल सुरक्षा",
        "👨‍👩‍👧 Guardian Mode / अभिभावक मोड",
        "📚 Learn / सीखें",
        "⚙️ System Status"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Detection Status")
if detector_loaded:
    st.sidebar.success("✅ Rule Engine Ready")
    if detector.distilbert and detector.distilbert.loaded:
        st.sidebar.success("✅ DistilBERT Ready")
    else:
        st.sidebar.warning("⚠️ DistilBERT Loading...")
else:
    st.sidebar.error("❌ Detector Not Loaded")

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home / होम":
    st.markdown("# 🛡️ ScamShield - Trust Nest")
    st.markdown("### Protecting Seniors from Digital Scams / बुज़ुर्गों को डिजिटल स्कैम से बचाना")
    
    st.markdown("---")
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## What We Protect Against / हम किससे बचाते हैं
        
        <div class="feature-card">
        🔴 <b>OTP Scams</b> - "Share OTP to receive refund"<br>
        <span class="hindi-text">OTP स्कैम - "रिफंड पाने के लिए OTP शेयर करें"</span>
        </div>
        
        <div class="feature-card">
        🔴 <b>Digital Arrest</b> - "Police video call, transfer money"<br>
        <span class="hindi-text">डिजिटल अरेस्ट - "पुलिस वीडियो कॉल, पैसे भेजो"</span>
        </div>
        
        <div class="feature-card">
        🔴 <b>Family Impersonation</b> - "Mom, new number, emergency"<br>
        <span class="hindi-text">परिवार का बहाना - "मम्मी, नया नंबर, इमरजेंसी"</span>
        </div>
        
        <div class="feature-card">
        🔴 <b>Fake Authority</b> - "CBI/Court notice, pay fine"<br>
        <span class="hindi-text">नकली अधिकारी - "CBI/कोर्ट नोटिस, जुर्माना भरो"</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        ## Our Shields / हमारी सुरक्षा
        
        ### 📱 Message Shield
        SMS + WhatsApp protection
        
        ### 📞 Call Shield  
        Unknown call warnings
        
        ### 👨‍👩‍👧 Guardian Mode
        Family alerts for high-risk
        
        ### 🎓 Teach Me
        Learn scam patterns
        """)
    
    st.markdown("---")
    
    # Architecture
    st.markdown("## 🔧 How It Works / यह कैसे काम करता है")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="pipeline-step">
        <h3>📨</h3>
        <b>Message</b><br>
        संदेश
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="pipeline-step">
        <h3>⚙️</h3>
        <b>Rule Engine</b><br>
        नियम इंजन
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="pipeline-step">
        <h3>🤖</h3>
        <b>DistilBERT AI</b><br>
        AI मॉडल
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="pipeline-step">
        <h3>👨‍👩‍👧</h3>
        <b>Guardian Alert</b><br>
        परिवार को सूचना
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# MESSAGE SHIELD
# ============================================================================
elif page == "📱 Message Shield / संदेश सुरक्षा":
    st.markdown("# 📱 Message Shield / संदेश सुरक्षा")
    st.markdown("### Check any SMS or WhatsApp message / कोई भी SMS या WhatsApp संदेश जाँचें")
    
    st.markdown("---")
    
    # Input section
    message = st.text_area(
        "📝 Paste message here / संदेश यहाँ पेस्ट करें",
        height=150,
        placeholder="Enter or paste a suspicious message...\nसंदिग्ध संदेश यहाँ दर्ज करें..."
    )
    
    # Quick examples
    st.markdown("### 🔄 Quick Examples / त्वरित उदाहरण")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Safe Message", use_container_width=True):
            st.session_state.message = "Hi, I'll reach home by 7 PM. See you soon!"
            st.rerun()
    
    with col2:
        if st.button("⚠️ OTP Scam", use_container_width=True):
            st.session_state.message = "Your refund of Rs. 5000 is ready. Share OTP to receive."
            st.rerun()
    
    with col3:
        if st.button("🚨 Digital Arrest", use_container_width=True):
            st.session_state.message = "This is CBI. You are under digital arrest. Stay on video call and transfer Rs. 50000 to safe account."
            st.rerun()
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        if st.button("🚨 Family Scam", use_container_width=True):
            st.session_state.message = "Hi Mom, this is my new number. Emergency! Send Rs. 9900 to xyz@oksbi immediately."
            st.rerun()
    
    with col5:
        if st.button("⚠️ Court Notice", use_container_width=True):
            st.session_state.message = "Court notice: Settlement of Rs. 50000 required today. Call +91-86854-63467 immediately."
            st.rerun()
    
    with col6:
        if st.button("⚠️ Lottery Scam", use_container_width=True):
            st.session_state.message = "Congratulations! You won Rs. 10 Lakh lottery. Pay Rs. 5000 processing fee to claim."
            st.rerun()
    
    # Use session state message if set
    if 'message' in st.session_state:
        message = st.session_state.message
        del st.session_state.message
    
    st.markdown("---")
    
    # Analyze button
    if st.button("🔍 ANALYZE MESSAGE / संदेश जाँचें", type="primary", use_container_width=True):
        if message and message.strip():
            if detector_loaded:
                with st.spinner("Analyzing with Rule Engine + DistilBERT..."):
                    time.sleep(0.5)  # Small delay for effect
                    result = detector.detect(message)
                
                st.markdown("---")
                st.markdown("## 📊 Analysis Result / विश्लेषण परिणाम")
                
                # Verdict card
                if result.verdict == "SAFE":
                    st.markdown(f"""
                    <div class="safe-card">
                        <span class="verdict-text">✅ SAFE / सुरक्षित</span>
                        <p style="font-size: 1.5rem; margin-top: 10px;">Confidence: {result.confidence:.0%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif result.verdict == "SUSPICIOUS":
                    st.markdown(f"""
                    <div class="suspicious-card">
                        <span class="verdict-text">⚠️ SUSPICIOUS / संदेहास्पद</span>
                        <p style="font-size: 1.5rem; margin-top: 10px;">Confidence: {result.confidence:.0%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="scam-card">
                        <span class="verdict-text">🚨 SCAM / धोखाधड़ी</span>
                        <p style="font-size: 1.5rem; margin-top: 10px;">Confidence: {result.confidence:.0%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Explanation
                st.markdown("### 📖 Explanation / स्पष्टीकरण")
                st.info(f"**English:** {result.explanation_en}")
                st.info(f"**हिंदी:** {result.explanation_hi}")
                
                # Action
                st.markdown("### ✋ Recommended Action / सुझाव")
                st.warning(f"**{result.action_en}**")
                st.warning(f"**{result.action_hi}**")
                
                # Technical details
                with st.expander("🔧 Technical Details / तकनीकी विवरण"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Rule Engine:**")
                        if result.rule_triggers:
                            for rule in result.rule_triggers:
                                st.error(f"⚠️ {rule}")
                        else:
                            st.success("✅ No rules triggered")
                    
                    with col2:
                        st.markdown("**DistilBERT AI:**")
                        if result.ml_used:
                            st.info(f"🤖 {result.ml_label} ({result.ml_confidence:.0%})")
                        else:
                            st.info("⏭️ Skipped (rule override)")
                    
                    if result.should_escalate:
                        st.error(f"📱 **Guardian Alert:** {result.escalation_reason}")
            else:
                st.error("Detector not loaded. Please check system status.")
        else:
            st.warning("Please enter a message to analyze.")

# ============================================================================
# CALL SHIELD
# ============================================================================
elif page == "📞 Call Shield / कॉल सुरक्षा":
    st.markdown("# 📞 Call Shield / कॉल सुरक्षा")
    st.markdown("### Analyze suspicious calls / संदिग्ध कॉल की जाँच करें")
    
    st.markdown("---")
    
    st.markdown("""
    <div class="feature-card">
    <h3>How Call Shield Works / कॉल शील्ड कैसे काम करता है</h3>
    <ul style="font-size: 1.2rem;">
        <li>✅ Detects calls from unknown numbers</li>
        <li>✅ Warns before answering suspicious calls</li>
        <li>✅ Monitors call duration (long calls = higher risk)</li>
        <li>✅ NO audio recording - privacy first!</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Call metadata input
    st.markdown("### 📝 Enter Call Details / कॉल विवरण दर्ज करें")
    
    col1, col2 = st.columns(2)
    
    with col1:
        caller_number = st.text_input("📞 Caller Number", placeholder="+91-XXXXX-XXXXX")
        call_duration = st.number_input("⏱️ Call Duration (minutes)", min_value=0, max_value=120, value=5)
    
    with col2:
        caller_name = st.text_input("👤 Claimed Identity", placeholder="e.g., CBI Officer, Bank Manager")
        asked_for_money = st.checkbox("💰 Asked for money transfer?")
        asked_for_otp = st.checkbox("🔢 Asked for OTP/PIN?")
        video_call = st.checkbox("📹 Was it a video call?")
    
    if st.button("🔍 ANALYZE CALL / कॉल जाँचें", type="primary", use_container_width=True):
        risk_score = 0
        reasons = []
        
        # Analyze based on inputs
        if caller_number and ("+91" in caller_number or caller_number.startswith("0")):
            if len(caller_number.replace("-", "").replace(" ", "")) > 10:
                risk_score += 10
        
        if caller_name:
            authorities = ["cbi", "police", "court", "income tax", "customs", "trai"]
            if any(auth in caller_name.lower() for auth in authorities):
                risk_score += 35
                reasons.append("Claims to be from government authority")
        
        if call_duration > 15:
            risk_score += 15
            reasons.append("Long call duration (common in scams)")
        
        if asked_for_money:
            risk_score += 40
            reasons.append("Asked for money transfer")
        
        if asked_for_otp:
            risk_score += 35
            reasons.append("Asked for OTP/PIN - Never share!")
        
        if video_call and "cbi" in caller_name.lower() or "police" in caller_name.lower():
            risk_score += 50
            reasons.append("Digital Arrest pattern - Police NEVER video call!")
        
        # Display result
        st.markdown("---")
        st.markdown("## 📊 Call Analysis / कॉल विश्लेषण")
        
        if risk_score >= 50:
            st.markdown(f"""
            <div class="scam-card">
                <span class="verdict-text">🚨 HIGH RISK CALL</span>
                <p style="font-size: 1.5rem;">Risk Score: {risk_score}%</p>
            </div>
            """, unsafe_allow_html=True)
            st.error("⚠️ **DO NOT continue this call! End immediately and call family.**")
        elif risk_score >= 25:
            st.markdown(f"""
            <div class="suspicious-card">
                <span class="verdict-text">⚠️ SUSPICIOUS CALL</span>
                <p style="font-size: 1.5rem;">Risk Score: {risk_score}%</p>
            </div>
            """, unsafe_allow_html=True)
            st.warning("❓ **Ask a family member before taking any action.**")
        else:
            st.markdown(f"""
            <div class="safe-card">
                <span class="verdict-text">✅ LOW RISK</span>
                <p style="font-size: 1.5rem;">Risk Score: {risk_score}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        if reasons:
            st.markdown("### ⚠️ Warning Signs Found:")
            for reason in reasons:
                st.error(f"• {reason}")

# ============================================================================
# GUARDIAN MODE
# ============================================================================
elif page == "👨‍👩‍👧 Guardian Mode / अभिभावक मोड":
    st.markdown("# 👨‍👩‍👧 Guardian Mode / अभिभावक मोड")
    st.markdown("### Family protection circle / परिवार सुरक्षा वृत्त")
    
    st.markdown("---")
    
    st.markdown("""
    <div class="feature-card">
    <h3>What is Guardian Mode? / गार्डियन मोड क्या है?</h3>
    <p style="font-size: 1.2rem;">
    When a high-risk scam is detected, your trusted family member (son, daughter, caregiver) 
    receives an instant alert on their phone.
    </p>
    <p style="font-size: 1.2rem;" class="hindi-text">
    जब कोई खतरनाक स्कैम पता चलता है, तो आपके भरोसेमंद परिवार के सदस्य को तुरंत 
    अपने फोन पर अलर्ट मिलता है।
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📱 Senior's Device / बुज़ुर्ग का फोन")
        
        st.markdown("""
        **Pairing Code / जोड़ने का कोड:**
        """)
        
        pairing_code = "XKTY-8294"
        st.markdown(f"""
        <div style="background: #667eea; color: white; padding: 20px; border-radius: 10px; 
                    text-align: center; font-size: 2rem; font-weight: bold;">
        {pairing_code}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <p style="text-align: center; margin-top: 10px; color: #666;">
        Share this code with your family member<br>
        यह कोड अपने परिवार के सदस्य के साथ शेयर करें
        </p>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📲 Guardian's Device / अभिभावक का फोन")
        
        entered_code = st.text_input("Enter Pairing Code / कोड दर्ज करें", placeholder="XXXX-0000")
        
        if st.button("🔗 PAIR DEVICES / जोड़ें", use_container_width=True):
            if entered_code.upper() == pairing_code:
                st.success("✅ **Paired successfully! You will now receive alerts.**")
                st.balloons()
            else:
                st.error("❌ Invalid code. Please try again.")
    
    st.markdown("---")
    
    st.markdown("### 📢 Sample Alert / नमूना अलर्ट")
    
    if st.button("🔔 Simulate Guardian Alert / अलर्ट का नमूना देखें", use_container_width=True):
        st.markdown("""
        <div style="background: #dc3545; color: white; padding: 25px; border-radius: 15px; margin: 20px 0;">
            <h2 style="color: white;">🚨 SCAM ALERT from Mom's Phone</h2>
            <p style="font-size: 1.3rem;">
            <b>Detected:</b> Digital Arrest Scam<br>
            <b>Risk Level:</b> HIGH (85%)<br>
            <b>Time:</b> Just now<br><br>
            <b>Preview:</b> "CBI calling. You are under digital arrest..."
            </p>
            <div style="display: flex; gap: 10px; margin-top: 15px;">
                <button style="background: white; color: #dc3545; padding: 10px 20px; border: none; border-radius: 8px; font-weight: bold;">
                📞 CALL MOM NOW
                </button>
                <button style="background: rgba(255,255,255,0.2); color: white; padding: 10px 20px; border: none; border-radius: 8px;">
                ✅ Mark as False Alarm
                </button>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# LEARN / TEACH ME
# ============================================================================
elif page == "📚 Learn / सीखें":
    st.markdown("# 📚 Learn to Spot Scams / स्कैम पहचानना सीखें")
    st.markdown("### Simple lessons for everyone / सबके लिए आसान सबक")
    
    st.markdown("---")
    
    lessons = [
        {
            "title_en": "🏦 Banks NEVER Ask for OTP",
            "title_hi": "🏦 बैंक कभी OTP नहीं माँगते",
            "content_en": "If someone calls or messages asking for your OTP, it's 100% a scam. Banks will NEVER ask for OTP on call.",
            "content_hi": "अगर कोई फोन या मैसेज करके OTP माँगे, तो यह 100% स्कैम है। बैंक कभी भी OTP नहीं माँगते।",
            "icon": "🔐"
        },
        {
            "title_en": "👮 Police NEVER Video Call",
            "title_hi": "👮 पुलिस कभी वीडियो कॉल नहीं करती",
            "content_en": "'Digital Arrest' is FAKE. Real police sends summons by post, not WhatsApp/Video call.",
            "content_hi": "'डिजिटल अरेस्ट' नकली है। असली पुलिस डाक से समन भेजती है, WhatsApp से नहीं।",
            "icon": "🚔"
        },
        {
            "title_en": "📱 QR = Pay, Not Receive",
            "title_hi": "📱 QR = पैसे देना, लेना नहीं",
            "content_en": "To RECEIVE money, you don't scan QR. Scanning QR is for PAYING. If someone asks you to scan QR to receive refund, it's a scam!",
            "content_hi": "पैसे लेने के लिए QR स्कैन नहीं होता। QR स्कैन करना मतलब पैसे देना। अगर कोई रिफंड के लिए QR स्कैन करने को कहे, तो स्कैम है!",
            "icon": "📲"
        },
        {
            "title_en": "👨‍👩‍👧 Verify 'Family' on Old Number",
            "title_hi": "👨‍👩‍👧 परिवार को पुराने नंबर पर वेरिफाई करें",
            "content_en": "If someone says 'Mom, this is my new number, send money urgently' - ALWAYS call their OLD number first to verify.",
            "content_hi": "अगर कोई कहे 'मम्मी, यह मेरा नया नंबर है, जल्दी पैसे भेजो' - हमेशा पहले उनके पुराने नंबर पर कॉल करके पुष्टि करें।",
            "icon": "☎️"
        }
    ]
    
    for lesson in lessons:
        with st.expander(f"{lesson['icon']} {lesson['title_en']}", expanded=False):
            st.markdown(f"""
            <div style="font-size: 1.3rem; padding: 15px; background: #f8f9fa; border-radius: 10px;">
            <p><b>English:</b> {lesson['content_en']}</p>
            <p style="color: #666; font-style: italic;"><b>हिंदी:</b> {lesson['content_hi']}</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# SYSTEM STATUS
# ============================================================================
elif page == "⚙️ System Status":
    st.markdown("# ⚙️ System Status / सिस्टम स्थिति")
    st.markdown("### Technical details for judges / न्यायाधीशों के लिए तकनीकी विवरण")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔧 Components")
        
        if detector_loaded:
            st.success("✅ Rule Engine - Active")
            st.markdown("   - 8+ scam pattern rules")
            st.markdown("   - <10ms detection time")
            
            if detector.distilbert and detector.distilbert.loaded:
                st.success("✅ DistilBERT - Loaded")
                st.markdown("   - 66M parameters")
                st.markdown("   - Trained on SMS + WhatsApp")
            else:
                st.warning("⚠️ DistilBERT - Not Loaded")
        else:
            st.error("❌ Detector - Not Initialized")
    
    with col2:
        st.markdown("### 📊 Model Info")
        
        st.markdown("""
        | Property | Value |
        |----------|-------|
        | Base Model | distilbert-base-uncased |
        | Training Data | 148 samples |
        | Test Accuracy | 100% (on synthetic) |
        | ONNX Export | ✅ Available |
        """)
    
    st.markdown("---")
    
    st.markdown("### 🏗️ Architecture")
    st.code("""
    Message Input
         ↓
    ┌─────────────────┐
    │   Rule Engine   │  ← Pattern matching (<10ms)
    │   (Override)    │
    └────────┬────────┘
             ↓ (if uncertain)
    ┌─────────────────┐
    │   DistilBERT    │  ← Semantic classification
    │   (66M params)  │
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │ Guardian Alert  │  ← FCM to family
    │   (if high-risk)│
    └─────────────────┘
    """, language="text")
    
    st.markdown("---")
    
    st.markdown("### 🔒 Privacy Features")
    st.markdown("""
    - ❌ No silent call recording
    - ❌ No message upload to server
    - ❌ No contact scraping
    - ✅ On-device rule engine first
    - ✅ User controls all data
    - ✅ Guardian requires explicit consent
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 20px;">
<p>🛡️ <b>ScamShield - Trust Nest</b> | KHISTIJ Hackathon 2026</p>
<p>⚠️ Prototype - Trained on synthetic data</p>
</div>
""", unsafe_allow_html=True)
