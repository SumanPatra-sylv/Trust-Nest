"""
ScamShield Demo - Terminal Edition
===================================
Interactive demo showing Rule Engine → DistilBERT → Guardian pipeline.
Run this for a clean, impressive terminal demonstration.
"""

import os
import sys
import time

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Colors for terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

def print_banner():
    print(f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   🛡️  SCAMSHIELD - Trust Nest                                   ║
║   Privacy-First Scam Detection for Senior Citizens               ║
║                                                                  ║
║   Pipeline: Rule Engine → DistilBERT → Guardian                 ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
{Colors.ENDC}""")

def print_step(step_num, text, delay=0.3):
    print(f"{Colors.DIM}[{step_num}/4]{Colors.ENDC} {Colors.CYAN}{text}{Colors.ENDC}")
    time.sleep(delay)

def animate_loading(text, duration=0.5):
    chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    end_time = time.time() + duration
    i = 0
    while time.time() < end_time:
        print(f"\r{Colors.CYAN}{chars[i % len(chars)]} {text}{Colors.ENDC}", end="", flush=True)
        time.sleep(0.1)
        i += 1
    print(f"\r{Colors.GREEN}✓ {text}{Colors.ENDC}")

def demo_message(detector, text, label):
    print(f"\n{Colors.BOLD}{'─' * 70}{Colors.ENDC}")
    print(f"{Colors.BOLD}📨 INPUT MESSAGE:{Colors.ENDC}")
    print(f"   \"{text}\"")
    print()
    
    # Step 1: Rule Engine
    print_step(1, "Running Rule Engine...")
    time.sleep(0.3)
    
    # Get result
    result = detector.detect(text)
    
    # Show rules
    if result.rule_triggers:
        print(f"   {Colors.WARNING}⚠ Rules triggered: {', '.join(result.rule_triggers)}{Colors.ENDC}")
        for reason in result.rule_reasons_en[:2]:
            print(f"   {Colors.DIM}• {reason}{Colors.ENDC}")
    else:
        print(f"   {Colors.GREEN}✓ No rules triggered{Colors.ENDC}")
    
    # Step 2: DistilBERT
    print()
    print_step(2, "Running DistilBERT classifier...")
    time.sleep(0.5)
    
    if result.ml_used:
        print(f"   {Colors.BLUE}🤖 ML Prediction: {result.ml_label} ({result.ml_confidence:.0%}){Colors.ENDC}")
    else:
        print(f"   {Colors.DIM}⏭ Skipped (rule override){Colors.ENDC}")
    
    # Step 3: Combine results
    print()
    print_step(3, "Combining results...")
    time.sleep(0.3)
    
    # Step 4: Guardian check
    print()
    print_step(4, "Checking Guardian escalation...")
    time.sleep(0.2)
    
    if result.should_escalate:
        print(f"   {Colors.RED}📱 ESCALATION: {result.escalation_reason}{Colors.ENDC}")
    else:
        print(f"   {Colors.GREEN}✓ No escalation needed{Colors.ENDC}")
    
    # Final verdict
    print()
    if result.verdict == "SCAM":
        color = Colors.RED
        icon = "🚨"
    elif result.verdict == "SUSPICIOUS":
        color = Colors.WARNING
        icon = "⚠️"
    else:
        color = Colors.GREEN
        icon = "✅"
    
    print(f"{Colors.BOLD}{'─' * 70}{Colors.ENDC}")
    print(f"{color}{Colors.BOLD}")
    print(f"   {icon} VERDICT: {result.verdict} ({result.confidence:.0%} confidence)")
    print(f"{Colors.ENDC}")
    print(f"   {Colors.DIM}EN: {result.explanation_en}{Colors.ENDC}")
    print(f"   {Colors.DIM}HI: {result.explanation_hi}{Colors.ENDC}")
    print()
    print(f"   {Colors.BOLD}Action: {result.action_en}{Colors.ENDC}")
    print(f"{Colors.BOLD}{'─' * 70}{Colors.ENDC}")
    
    # Verify (only for labeled test cases)
    if label != "UNKNOWN":
        is_correct = (
            (label == "SAFE" and result.verdict == "SAFE") or
            (label == "SCAM" and result.verdict in ["SCAM", "SUSPICIOUS"])
        )
        if is_correct:
            print(f"   {Colors.GREEN}✓ Correctly detected{Colors.ENDC}")
        else:
            print(f"   {Colors.WARNING}△ Review needed{Colors.ENDC}")

def main():
    print_banner()
    
    # Load detector
    print(f"\n{Colors.CYAN}Initializing detection pipeline...{Colors.ENDC}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    model_dir = os.path.join(project_dir, "models")
    
    animate_loading("Loading Rule Engine", 0.3)
    
    from detector import ScamDetector
    detector = ScamDetector(model_dir=model_dir)
    
    if detector.distilbert and detector.distilbert.loaded:
        print(f"{Colors.GREEN}✓ DistilBERT loaded (66M parameters){Colors.ENDC}")
    else:
        print(f"{Colors.WARNING}⚠ DistilBERT not available, using rules only{Colors.ENDC}")
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}Pipeline ready!{Colors.ENDC}")
    
    # Demo cases
    test_cases = [
        ("Hi, I'll reach by 7 PM. See you soon!", "SAFE"),
        ("Share OTP to verify your payment of Rs. 5000", "SCAM"),
        ("Digital arrest. Stay on video call. Transfer money now.", "SCAM"),
        ("Hi Mom, new number. Emergency. Send Rs. 9900 to xyz@oksbi", "SCAM"),
        ("Court notice: Settlement required today. Call +91-86854-63467", "SCAM"),
    ]
    
    print(f"\n{Colors.BOLD}Running {len(test_cases)} test cases...{Colors.ENDC}")
    input(f"\n{Colors.DIM}Press Enter to start demo...{Colors.ENDC}")
    
    for text, label in test_cases:
        demo_message(detector, text, label)
        input(f"\n{Colors.DIM}Press Enter for next message...{Colors.ENDC}")
    
    # Summary
    print(f"""
{Colors.GREEN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════╗
║                     DEMO COMPLETE                                 ║
╠══════════════════════════════════════════════════════════════════╣
║  ✓ Rule Engine: Deterministic pattern matching (<10ms)          ║
║  ✓ DistilBERT: Semantic classification (66M params)             ║
║  ✓ Guardian Mode: Family escalation for high-risk               ║
║                                                                  ║
║  Architecture: On-device first, privacy-preserving               ║
╚══════════════════════════════════════════════════════════════════╝
{Colors.ENDC}""")
    
    # Interactive mode
    print(f"\n{Colors.CYAN}Interactive Mode - Enter your own messages:{Colors.ENDC}")
    print(f"{Colors.DIM}(Type 'quit' to exit){Colors.ENDC}\n")
    
    while True:
        try:
            text = input(f"{Colors.BOLD}Enter message: {Colors.ENDC}")
            if text.lower() in ['quit', 'exit', 'q']:
                break
            if text.strip():
                demo_message(detector, text, "UNKNOWN")
        except KeyboardInterrupt:
            break
    
    print(f"\n{Colors.CYAN}Thank you for using ScamShield!{Colors.ENDC}\n")

if __name__ == "__main__":
    main()
