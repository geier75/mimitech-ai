#!/usr/bin/env python3
"""
Test script for Whisper service
"""

import requests
import json

def test_whisper_service():
    """Test the Whisper service with the JFK sample"""
    
    print("🎤 Testing Whisper Service...")
    
    # Test service status
    try:
        response = requests.get("http://127.0.0.1:8003/")
        print(f"✅ Service Status: {response.json()}")
    except Exception as e:
        print(f"❌ Service not reachable: {e}")
        return
    
    # Test with JFK sample audio
    try:
        with open("whisper.cpp/samples/jfk.wav", "rb") as audio_file:
            files = {"audio": ("jfk.wav", audio_file, "audio/wav")}
            data = {"language": "en"}
            
            response = requests.post("http://127.0.0.1:8003/api/whisper", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Transcription successful!")
                print(f"📝 Transcript: '{result['transcript']}'")
                print(f"🎯 Intent: {result['detected_intent']}")
                print(f"⚡ Action: {result['mapped_action']}")
                print(f"🎯 Confidence: {result['confidence']:.2f}")
                print(f"⏱️ Processing Time: {result['processing_time']:.2f}s")
            else:
                print(f"❌ Transcription failed: {response.status_code}")
                print(f"Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_whisper_service()
