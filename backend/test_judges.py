"""
Comprehensive Test Suite for Hackathon Judges
Tests edge cases that judges might use to evaluate the system
"""

import requests
import json

API = 'http://localhost:8000/api/analyze'

# Comprehensive test cases judges might try
test_cases = [
    # SAFE messages - should NOT be flagged
    ('SAFE', 'Good morning! How are you today?'),
    ('SAFE', 'Meeting rescheduled to 4pm tomorrow'),
    ('SAFE', 'Happy birthday! Wishing you all the best'),
    ('SAFE', 'Your Flipkart order has been delivered'),
    ('SAFE', 'Mom, I will be home for dinner'),
    ('SAFE', 'Please call me when you get free'),
    ('SAFE', 'The doctor appointment is confirmed for Monday'),
    ('SAFE', 'Your electricity bill of Rs 1500 is due'),
    ('SAFE', 'Train PNR status: Confirmed'),
    ('SAFE', 'Weather forecast: Sunny day ahead'),
    
    # SCAM messages - should be flagged (SCAM or SUSPICIOUS)
    ('SCAM', 'Share your OTP to complete KYC'),
    ('SCAM', 'You are under digital arrest'),
    ('SCAM', 'Send Rs 50000 immediately or face legal action'),
    ('SCAM', 'Your SBI account blocked. Click link to verify'),
    ('SCAM', 'Congratulations! You won 50 lakh lottery'),
    ('SCAM', 'This is CBI calling. You are involved in money laundering'),
    ('SCAM', 'Papa this is Rahul new number. Send money urgently'),
    ('SCAM', 'Court notice: Pay Rs 25000 settlement today'),
    ('SCAM', 'Your Aadhaar is used for terrorism. Call now'),
    ('SCAM', 'Transfer money to safe RBI account immediately'),
    
    # EDGE CASES - tricky ones
    ('SAFE', 'Your OTP is 123456. Do not share with anyone'),  # OTP notification, not request
    ('SAFE', 'Bank: Your account balance is Rs 5000'),
    ('SAFE', 'Amazon: Your refund of Rs 500 processed'),
    ('SCAM', 'Amazon refund failed. Share OTP to process'),
    ('SAFE', 'Reminder: EMI payment due tomorrow'),
    ('SCAM', 'EMI payment failed. Pay Rs 5000 now or account blocked'),
    ('SAFE', 'Your cab is arriving in 5 minutes'),
    ('SAFE', 'Food order delivered. Rate your experience'),
    ('SCAM', 'Your parcel stuck in customs. Pay Rs 2000 to release'),
    ('SAFE', 'School closed tomorrow due to holiday'),
]

def run_tests():
    print('='*70)
    print(f'COMPREHENSIVE JUDGE TEST - Testing {len(test_cases)} messages')
    print('='*70)
    print()

    passed = 0
    failed = 0
    errors = []

    for expected, text in test_cases:
        try:
            r = requests.post(API, json={'text': text}, timeout=10)
            data = r.json()
            verdict = data['verdict']
            conf = round(data['confidence'] * 100)
            ml_label = data.get('ml_label', 'N/A')
            ml_conf = round(data.get('ml_confidence', 0) * 100) if data.get('ml_confidence') else 0
            rules = data.get('rule_triggers', [])
            
            # Check if correct
            if expected == 'SAFE':
                correct = verdict == 'SAFE'
            else:  # SCAM
                correct = verdict in ['SCAM', 'SUSPICIOUS']
            
            status = '✓' if correct else '✗'
            if correct:
                passed += 1
            else:
                failed += 1
                errors.append((expected, verdict, text, ml_label, ml_conf, rules))
            
            rules_str = ','.join(rules) if rules else 'None'
            text_short = text[:45] + '...' if len(text) > 45 else text
            print(f'{status} [{expected:4}] -> {verdict:10} ({conf}%) ML:{ml_label}({ml_conf}%) Rules:[{rules_str}]')
            print(f'   "{text_short}"')
            
        except Exception as e:
            failed += 1
            errors.append((expected, 'ERROR', text, 'N/A', 0, []))
            print(f'✗ ERROR: {e} - {text[:40]}...')

    print()
    print('='*70)
    print(f'RESULTS: {passed}/{len(test_cases)} passed ({round(passed/len(test_cases)*100)}%)')
    print('='*70)

    if errors:
        print()
        print('FAILURES TO FIX:')
        for exp, got, text, ml_label, ml_conf, rules in errors:
            print(f'  Expected {exp}, Got {got}')
            print(f'    Text: "{text[:60]}..."' if len(text) > 60 else f'    Text: "{text}"')
            print(f'    ML: {ml_label} ({ml_conf}%), Rules: {rules}')
            print()
    
    return passed, failed, errors

if __name__ == '__main__':
    run_tests()
