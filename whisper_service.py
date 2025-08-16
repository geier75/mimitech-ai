#!/usr/bin/env python3
"""
🎤 MISO Ultimate - Whisper.cpp FastAPI Service
==============================================

FastAPI service for audio transcription using Whisper.cpp
Provides /api/whisper endpoint for voice-to-text conversion.

Author: MISO Ultimate Team
Date: 01.08.2025
"""

import os
import asyncio
import tempfile
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configuration
WHISPER_CPP_PATH = "./whisper.cpp"
WHISPER_MODEL_PATH = "./whisper.cpp/models/ggml-base.bin"
WHISPER_EXECUTABLE = "./whisper.cpp/build/bin/whisper-cli"

# Voice Control Intent Mapping
INTENT_MAPPING = {
    # Matrix Benchmark Intents
    "matrix": [
        "matrix", "matrizen", "matrix benchmark", "matrix test", "matrix berechnung",
        "metriks", "matrix rechnung", "matrix operation", "matrix multiplikation"
    ],
    "matrix_quick": [
        "matrix schnell", "matrix quick", "schneller matrix", "matrix fast", "matrix klein",
        "matrix kurz", "schnelle matrix", "matrix test schnell"
    ],
    "matrix_intensive": [
        "matrix intensiv", "matrix intensive", "matrix schwer", "matrix hard", "matrix groß",
        "matrix lang", "große matrix", "matrix vollständig"
    ],

    # Quantum Benchmark Intents
    "quantum": [
        "quantum", "quanten", "quantum benchmark", "quantum test", "quantum simulation",
        "kwantum", "quantom", "quantum rechnung", "quantum berechnung"
    ],
    "quantum_quick": [
        "quantum schnell", "quantum quick", "schneller quantum", "quantum fast", "quantum klein",
        "quantum kurz", "schnelle quantum", "quantum test schnell"
    ],
    "quantum_intensive": [
        "quantum intensiv", "quantum intensive", "quantum schwer", "quantum hard", "quantum groß",
        "quantum lang", "große quantum", "quantum vollständig"
    ],

    # Comprehensive Benchmark Intents
    "all": [
        "alle", "all", "alle benchmarks", "all benchmarks", "comprehensive", "komplett", "vollständig",
        "alles", "gesamt", "complete", "full", "alle tests", "alle benchmarks starten"
    ],
    "all_quick": [
        "alle schnell", "all quick", "alle benchmarks schnell", "comprehensive quick", "alles schnell",
        "alle tests schnell", "schnelle benchmarks", "quick all"
    ],

    # System Commands
    "status": [
        "status", "zustand", "system status", "server status", "wie geht es", "system",
        "state", "wie läuft es", "system zustand", "server zustand"
    ],
    "help": [
        "hilfe", "help", "befehle", "commands", "was kann ich sagen", "hilfe bitte",
        "hilf mir", "unterstützung", "anleitung", "kommandos"
    ],
    "stop": [
        "stop", "stopp", "beenden", "cancel", "abbrechen", "halt",
        "aufhören", "ende", "fertig", "schluss"
    ],

    # Test phrases for debugging
    "test": [
        "test", "testing", "probe", "versuch", "hallo", "hello",
        "funktioniert", "geht das", "hörst du mich", "kannst du mich hören",
        "hey", "hi", "servus", "moin", "guten tag", "guten morgen",
        "ask not what your country", "fellow americans", "jfk", "kennedy"
    ],

    # More German variations
    "benchmark": [
        "benchmark", "benchmarks", "bench mark", "test", "tests", "testing",
        "leistung", "performance", "messung", "prüfung", "check"
    ],

    # Simple commands
    "start": ["start", "starten", "los", "go", "anfangen", "beginnen"],
    "info": ["info", "information", "details", "zeige", "anzeigen"],

    # Presets
    "small": ["klein", "small", "wenig", "light"],
    "medium": ["mittel", "medium", "normal"],
    "large": ["groß", "large", "schwer", "heavy", "intensiv", "intensive"]
}

class TranscriptionRequest(BaseModel):
    """Request model for transcription"""
    language: Optional[str] = "de"
    model: Optional[str] = "base"

class TranscriptionResponse(BaseModel):
    """Response model for transcription"""
    transcript: str
    confidence: float
    processing_time: float
    detected_intent: Optional[str] = None
    mapped_action: Optional[str] = None
    timestamp: str

class VoiceControlLog(BaseModel):
    """Voice control log entry"""
    timestamp: str
    original_transcript: str
    detected_intent: str
    mapped_action: str
    execution_result: str
    processing_time: float
    confidence: float

# Initialize FastAPI app
app = FastAPI(
    title="MISO Ultimate Whisper Service",
    description="Voice transcription and intent mapping service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8082",
        "http://127.0.0.1:8081", 
        "http://127.0.0.1:8080",
        "http://localhost:8082",
        "http://localhost:8081",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Voice control logs
voice_logs = []

def map_transcript_to_intent(transcript: str) -> tuple[str, str]:
    """
    Map transcript to intent and action
    Returns: (intent, action)
    """
    transcript_lower = transcript.lower().strip()

    # Debug logging
    print(f"🔍 Mapping transcript: '{transcript}' -> '{transcript_lower}'")

    # Remove timestamps if present (Whisper sometimes includes them)
    import re
    transcript_clean = re.sub(r'\[[\d:.,\s-]+\]', '', transcript_lower).strip()
    print(f"🧹 Cleaned transcript: '{transcript_clean}'")

    # Check for exact matches first
    for intent, phrases in INTENT_MAPPING.items():
        for phrase in phrases:
            if phrase in transcript_clean:
                print(f"✅ Found match: '{phrase}' -> intent: {intent}")

                # Determine action based on intent
                if intent.startswith("matrix"):
                    if "schnell" in transcript_clean or "quick" in transcript_clean:
                        return intent, "benchmark_matrix_quick"
                    elif "intensiv" in transcript_clean or "intensive" in transcript_clean:
                        return intent, "benchmark_matrix_intensive"
                    else:
                        return intent, "benchmark_matrix"

                elif intent.startswith("quantum"):
                    if "schnell" in transcript_clean or "quick" in transcript_clean:
                        return intent, "benchmark_quantum_quick"
                    elif "intensiv" in transcript_clean or "intensive" in transcript_clean:
                        return intent, "benchmark_quantum_intensive"
                    else:
                        return intent, "benchmark_quantum"

                elif intent.startswith("all"):
                    if "schnell" in transcript_clean or "quick" in transcript_clean:
                        return intent, "benchmark_all_quick"
                    else:
                        return intent, "benchmark_all"

                elif intent == "status":
                    return intent, "get_status"
                elif intent == "help":
                    return intent, "show_help"
                elif intent == "stop":
                    return intent, "stop_execution"
                elif intent == "test":
                    return intent, "test_response"

                return intent, f"execute_{intent}"

    # Fuzzy matching for common variations
    fuzzy_matches = {
        "matrix": ["matrix", "matrizen", "matrix", "metriks"],
        "quantum": ["quantum", "quanten", "kwantum", "quantom"],
        "benchmark": ["benchmark", "benchmarks", "bench mark", "test"],
        "schnell": ["schnell", "quick", "fast", "snel"],
        "status": ["status", "state", "zustand"],
        "alle": ["alle", "all", "alles"]
    }

    for key, variations in fuzzy_matches.items():
        for variation in variations:
            if variation in transcript_clean:
                print(f"🔍 Fuzzy match found: '{variation}' -> {key}")

                if key == "matrix":
                    return "matrix", "benchmark_matrix"
                elif key == "quantum":
                    return "quantum", "benchmark_quantum"
                elif key == "status":
                    return "status", "get_status"
                elif key == "alle":
                    return "all", "benchmark_all"

    # Fallback: no clear intent detected
    print(f"❌ No intent found for: '{transcript_clean}'")
    print(f"📋 Available phrases: {list(INTENT_MAPPING.keys())}")
    return "unknown", "no_action"

async def transcribe_audio_whisper(audio_file_path: str, language: str = "de") -> Dict[str, Any]:
    """
    Transcribe audio using Whisper.cpp
    """
    start_time = time.time()
    
    try:
        # Check if Whisper executable exists
        if not os.path.exists(WHISPER_EXECUTABLE):
            raise HTTPException(status_code=500, detail="Whisper.cpp not found. Please build it first.")
        
        # Check if model exists
        if not os.path.exists(WHISPER_MODEL_PATH):
            raise HTTPException(status_code=500, detail="Whisper model not found. Please download it first.")
        
        # Check audio file size and content
        file_size = os.path.getsize(audio_file_path)
        print(f"🎤 Audio file size: {file_size} bytes")

        if file_size < 1000:
            print(f"❌ Audio file too small: {file_size} bytes")
            raise HTTPException(
                status_code=400,
                detail=f"Audio file too small ({file_size} bytes). Please record longer audio."
            )

        # Convert audio to WAV format using ffmpeg if available
        wav_file_path = audio_file_path.replace('.webm', '.wav').replace('.mp4', '.wav')

        try:
            # Try to use ffmpeg for conversion
            ffmpeg_cmd = [
                'ffmpeg', '-i', audio_file_path,
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',      # Mono
                '-y',            # Overwrite output
                wav_file_path
            ]

            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ Audio converted with ffmpeg: {wav_file_path}")
            else:
                print(f"⚠️ ffmpeg failed, using original file")
                wav_file_path = audio_file_path

        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(f"⚠️ ffmpeg not available, using original file")
            wav_file_path = audio_file_path

        # Run Whisper.cpp with better parameters
        cmd = [
            WHISPER_EXECUTABLE,
            "-m", WHISPER_MODEL_PATH,
            "-f", wav_file_path,
            "-l", language,
            "--output-txt",  # Use text output instead of JSON
            "--no-timestamps",  # Remove timestamps
            "--threads", "4",
            "--processors", "1"
        ]

        print(f"🎤 Running Whisper command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        print(f"🎤 Whisper return code: {result.returncode}")
        print(f"🎤 Whisper stdout: '{result.stdout}'")
        print(f"🎤 Whisper stderr: '{result.stderr}'")

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Whisper transcription failed: {result.stderr}"
            )

        # Parse output - Whisper.cpp outputs to stdout
        transcript = result.stdout.strip()

        # Remove any remaining timestamps or formatting
        import re
        transcript = re.sub(r'\[[\d:.,\s-]+\]', '', transcript).strip()

        # Simple confidence estimation
        confidence = 0.85 if len(transcript) > 5 else 0.5

        print(f"🎤 Final transcript: '{transcript}'")
        
        processing_time = time.time() - start_time
        
        return {
            "transcript": transcript,
            "confidence": confidence,
            "processing_time": processing_time
        }
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Transcription timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "MISO Ultimate Whisper Service",
        "version": "1.0.0",
        "status": "running",
        "whisper_available": os.path.exists(WHISPER_EXECUTABLE),
        "model_available": os.path.exists(WHISPER_MODEL_PATH),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/whisper", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = "de"
):
    """
    Transcribe uploaded audio file and map to intent
    """
    start_time = time.time()
    
    # Validate file type
    if not audio.content_type or not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid audio file type")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        content = await audio.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    try:
        # Transcribe audio
        transcription_result = await transcribe_audio_whisper(temp_file_path, language)
        
        # Map to intent
        transcript = transcription_result["transcript"]
        intent, action = map_transcript_to_intent(transcript)
        
        # Create response
        response = TranscriptionResponse(
            transcript=transcript,
            confidence=transcription_result["confidence"],
            processing_time=time.time() - start_time,
            detected_intent=intent,
            mapped_action=action,
            timestamp=datetime.now().isoformat()
        )
        
        # Log the voice control interaction
        log_entry = VoiceControlLog(
            timestamp=response.timestamp,
            original_transcript=transcript,
            detected_intent=intent,
            mapped_action=action,
            execution_result="pending",
            processing_time=response.processing_time,
            confidence=response.confidence
        )
        voice_logs.append(log_entry.dict())
        
        return response
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@app.get("/api/whisper/logs")
async def get_voice_logs():
    """Get voice control logs"""
    return {
        "logs": voice_logs[-50:],  # Return last 50 entries
        "total_count": len(voice_logs),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/whisper/test-jfk")
async def test_jfk_audio():
    """Test Whisper with JFK sample audio"""
    try:
        jfk_path = os.path.join(WHISPER_CPP_PATH, "samples", "jfk.wav")

        if not os.path.exists(jfk_path):
            raise HTTPException(
                status_code=404,
                detail=f"JFK sample not found at {jfk_path}"
            )

        print(f"🎵 Testing with JFK sample: {jfk_path}")

        # Run Whisper.cpp directly on JFK sample
        cmd = [
            WHISPER_EXECUTABLE,
            "-m", WHISPER_MODEL_PATH,
            "-f", jfk_path,
            "-l", "en",
            "--output-txt",
            "--no-timestamps",
            "--threads", "4"
        ]

        print(f"🎵 JFK Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        print(f"🎵 JFK return code: {result.returncode}")
        print(f"🎵 JFK stdout: '{result.stdout}'")

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"JFK test failed: {result.stderr}"
            )

        transcript = result.stdout.strip()

        # Remove any remaining timestamps
        import re
        transcript = re.sub(r'\[[\d:.,\s-]+\]', '', transcript).strip()

        # Map to intent
        detected_intent, mapped_action = map_transcript_to_intent(transcript)

        return {
            "transcript": transcript,
            "detected_intent": detected_intent,
            "mapped_action": mapped_action,
            "confidence": 0.95,
            "test_type": "jfk_sample",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"❌ JFK test error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/whisper/intents")
async def get_available_intents():
    """Get available voice intents"""
    return {
        "intents": INTENT_MAPPING,
        "examples": [
            "Matrix Benchmark schnell",
            "Quantum Test",
            "Alle Benchmarks",
            "Status",
            "Hilfe"
        ],
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("🎤 Starting MISO Ultimate Whisper Service...")
    print(f"🔧 Whisper.cpp path: {WHISPER_CPP_PATH}")
    print(f"📁 Model path: {WHISPER_MODEL_PATH}")
    print(f"🚀 Service will run on: http://127.0.0.1:8003")
    
    uvicorn.run(
        "whisper_service:app",
        host="127.0.0.1",
        port=8003,
        reload=True,
        log_level="info"
    )
