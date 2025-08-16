#!/usr/bin/env python3
"""
Test German voice commands with Whisper service
"""

import requests
import json

def test_intent_mapping():
    """Test intent mapping with various phrases"""
    
    test_phrases = [
        "matrix schnell",
        "quantum benchmark", 
        "alle benchmarks",
        "status",
        "hilfe",
        "test",
        "matrix berechnung",
        "quantum simulation"
    ]
    
    print("🎯 Testing Intent Mapping...")
    
    for phrase in test_phrases:
        try:
            response = requests.get(f"http://127.0.0.1:8003/api/whisper/intents")
            if response.status_code == 200:
                intents = response.json()
                print(f"✅ Available intents loaded")
                break
        except Exception as e:
            print(f"❌ Could not load intents: {e}")
            return
    
    # Test the mapping logic by simulating transcripts
    print("\n📝 Testing phrase mappings:")
    for phrase in test_phrases:
        print(f"  '{phrase}' -> Testing...")

def test_whisper_logs():
    """Check recent Whisper logs"""
    try:
        response = requests.get("http://127.0.0.1:8003/api/whisper/logs")
        if response.status_code == 200:
            logs = response.json()
            print(f"📊 Recent Whisper Logs ({logs['total_count']} total):")
            for log in logs['logs'][-5:]:  # Show last 5
                print(f"  📝 '{log['original_transcript']}'")
                print(f"  🎯 Intent: {log['detected_intent']}")
                print(f"  ⚡ Action: {log['mapped_action']}")
                print(f"  ⏱️ Time: {log['processing_time']:.2f}s")
                print(f"  🎯 Confidence: {log['confidence']:.2f}")
                print("  ---")
        else:
            print(f"❌ Could not get logs: {response.status_code}")
    except Exception as e:
        print(f"❌ Log test failed: {e}")

if __name__ == "__main__":
    test_intent_mapping()
    print()
    test_whisper_logs()
