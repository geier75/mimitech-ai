#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - M-LINGUA Testskript

Dieses Skript testet die M-LINGUA-Implementierung und ihre Integration mit VXOR.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import json
from typing import Dict, List, Any

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [M-LINGUA-TEST] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MISO.M-LINGUA.Test")

# Füge das Stammverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Importiere die M-LINGUA-Komponenten
try:
    from miso.lang.mlingua.mlingua_interface import MLinguaInterface
    from miso.lang.mlingua.language_detector import LanguageDetector
    from miso.lang.mlingua.multilang_parser import MultilingualParser
    from miso.lang.mlingua.semantic_layer import SemanticLayer
    from miso.lang.mlingua.vxor_integration import VXORIntegration
    from miso.lang.mlingua.math_bridge import MathBridge
    logger.info("M-LINGUA-Komponenten erfolgreich importiert")
except ImportError as e:
    logger.error(f"Fehler beim Importieren der M-LINGUA-Komponenten: {e}")
    sys.exit(1)

def test_language_detector():
    """Testet den LanguageDetector"""
    logger.info("=== Test: LanguageDetector ===")
    
    detector = LanguageDetector()
    
    test_texts = {
        "de": "Hallo, wie geht es dir? Ich hoffe, es geht dir gut.",
        "en": "Hello, how are you? I hope you are doing well.",
        "es": "Hola, ¿cómo estás? Espero que estés bien.",
        "fr": "Bonjour, comment ça va? J'espère que vous allez bien.",
        "zh": "你好，你好吗？希望你一切都好。",
        "ja": "こんにちは、お元気ですか？元気であることを願っています。",
        "ru": "Привет, как дела? Надеюсь, у тебя все хорошо.",
        "ar": "مرحبا، كيف حالك؟ آمل أن تكون بخير."
    }
    
    success_count = 0
    for true_lang, text in test_texts.items():
        detected_lang, confidence = detector.detect_language(text)
        logger.info(f"Text: {text}")
        logger.info(f"Wahre Sprache: {true_lang}, Erkannte Sprache: {detected_lang}, Konfidenz: {confidence:.2f}")
        
        if detected_lang == true_lang:
            success_count += 1
            logger.info("✓ Erfolg")
        else:
            logger.warning("✗ Fehler")
        
        logger.info("-" * 50)
    
    success_rate = success_count / len(test_texts) * 100
    logger.info(f"Erfolgsrate: {success_rate:.2f}% ({success_count}/{len(test_texts)})")
    
    return success_rate >= 75.0

def test_multilingual_parser():
    """Testet den MultilingualParser"""
    logger.info("=== Test: MultilingualParser ===")
    
    parser = MultilingualParser()
    
    test_texts = {
        "de": "Öffne den Browser mit Suchparameter Wetter",
        "en": "Search for weather forecast in Berlin",
        "es": "Busca el pronóstico del tiempo en Madrid",
        "fr": "Ouvre le navigateur avec paramètre de recherche météo"
    }
    
    success_count = 0
    for lang, text in test_texts.items():
        parsed = parser.parse(text)
        logger.info(f"Text: {text}")
        logger.info(f"Sprache: {parsed.detected_language}, Intention: {parsed.intent}, Aktion: {parsed.action}")
        logger.info(f"Ziel: {parsed.target}, Parameter: {parsed.parameters}")
        logger.info(f"VXOR-Befehle: {parsed.vxor_commands}")
        
        if parsed.intent != "UNKNOWN" and parsed.action and parsed.target:
            success_count += 1
            logger.info("✓ Erfolg")
        else:
            logger.warning("✗ Fehler")
        
        logger.info("-" * 50)
    
    success_rate = success_count / len(test_texts) * 100
    logger.info(f"Erfolgsrate: {success_rate:.2f}% ({success_count}/{len(test_texts)})")
    
    return success_rate >= 75.0

def test_semantic_layer():
    """Testet die SemanticLayer"""
    logger.info("=== Test: SemanticLayer ===")
    
    semantic_layer = SemanticLayer()
    
    test_texts = {
        "de": "Öffne den Browser mit Suchparameter Wetter",
        "en": "Search for weather forecast in Berlin",
        "es": "Busca el pronóstico del tiempo en Madrid",
        "fr": "Ouvre le navigateur avec paramètre de recherche météo"
    }
    
    success_count = 0
    for lang, text in test_texts.items():
        result = semantic_layer.analyze(text)
        logger.info(f"Text: {text}")
        logger.info(f"Sprache: {result.parsed_command.detected_language}, Intention: {result.parsed_command.intent}")
        logger.info(f"Aktion: {result.parsed_command.action}, Ziel: {result.parsed_command.target}")
        logger.info(f"Parameter: {result.parsed_command.parameters}")
        logger.info(f"VXOR-Befehle: {[cmd['command_string'] for cmd in result.vxor_commands]}")
        logger.info(f"M-CODE: {result.m_code}")
        logger.info(f"Konfidenz: {result.confidence:.2f}")
        
        if result.parsed_command.intent != "UNKNOWN" and result.vxor_commands:
            success_count += 1
            logger.info("✓ Erfolg")
        else:
            logger.warning("✗ Fehler")
        
        logger.info("-" * 50)
    
    success_rate = success_count / len(test_texts) * 100
    logger.info(f"Erfolgsrate: {success_rate:.2f}% ({success_count}/{len(test_texts)})")
    
    return success_rate >= 75.0

def test_math_bridge():
    """Testet die MathBridge"""
    logger.info("=== Test: MathBridge ===")
    
    math_bridge = MathBridge()
    
    test_texts = {
        "de": [
            "Berechne 2 + 3 * 4",
            "Was ist 10 / 2 + 5",
            "Erstelle einen Vektor mit den Werten 1, 2, 3, 4",
            "Matrix [1, 2; 3, 4]",
            "Löse die Gleichung x^2 + 2*x - 3 = 0 für x"
        ],
        "en": [
            "Calculate 2 + 3 * 4",
            "What is 10 / 2 + 5",
            "Create a vector with values 1, 2, 3, 4",
            "Matrix [1, 2; 3, 4]",
            "Solve the equation x^2 + 2*x - 3 = 0 for x"
        ]
    }
    
    total_tests = sum(len(texts) for texts in test_texts.values())
    success_count = 0
    
    for lang, texts in test_texts.items():
        logger.info(f"\nSprache: {lang}")
        for text in texts:
            result = math_bridge.process_math_expression(text, lang)
            logger.info(f"Text: {text}")
            logger.info(f"Erfolg: {result.success}")
            
            if result.success:
                logger.info(f"Ausdruckstyp: {result.math_expression.expression_type}")
                logger.info(f"Ausdruck: {result.math_expression.expression_text}")
                logger.info(f"T-Mathematics-Befehl: {result.t_math_command}")
                logger.info(f"Ergebnis: {result.result}")
                
                success_count += 1
                logger.info("✓ Erfolg")
            else:
                logger.info(f"Fehlermeldung: {result.error_message}")
                logger.warning("✗ Fehler")
            
            logger.info("-" * 50)
    
    success_rate = success_count / total_tests * 100
    logger.info(f"Erfolgsrate: {success_rate:.2f}% ({success_count}/{total_tests})")
    
    return success_rate >= 75.0

def test_mlingua_interface():
    """Testet die MLinguaInterface"""
    logger.info("=== Test: MLinguaInterface ===")
    
    mlingua = MLinguaInterface()
    
    test_texts = {
        "de": "Öffne den Browser mit Suchparameter Wetter",
        "en": "Search for weather forecast in Berlin",
        "es": "Busca el pronóstico del tiempo en Madrid",
        "fr": "Ouvre le navigateur avec paramètre de recherche météo",
        "de_math": "Berechne 2 + 3 * 4",
        "en_math": "Calculate 10 / 2 + 5"
    }
    
    session_id = f"test_session_{int(time.time())}"
    success_count = 0
    
    for lang, text in test_texts.items():
        logger.info(f"Verarbeite Text: {text}")
        result = mlingua.process(text, session_id=session_id)
        
        logger.info(f"Erkannte Sprache: {result.detected_language}")
        logger.info(f"Verarbeitungszeit: {result.processing_time:.4f}s")
        logger.info(f"Erfolg: {result.success}")
        
        if result.success:
            semantic = result.semantic_result
            logger.info(f"Intention: {semantic.parsed_command.intent}")
            logger.info(f"Aktion: {semantic.parsed_command.action}")
            logger.info(f"Ziel: {semantic.parsed_command.target}")
            logger.info(f"Parameter: {semantic.parsed_command.parameters}")
            
            if semantic.requires_clarification:
                logger.info(f"Rückfrage erforderlich: {semantic.feedback}")
                logger.info(f"Optionen: {semantic.clarification_options}")
            else:
                logger.info(f"VXOR-Befehle: {[cmd['command_string'] for cmd in semantic.vxor_commands]}")
                logger.info(f"M-CODE: {semantic.m_code}")
            
            if semantic.parsed_command.intent != "UNKNOWN":
                success_count += 1
                logger.info("✓ Erfolg")
            else:
                logger.warning("✗ Fehler")
        else:
            logger.info(f"Fehlermeldung: {result.error_message}")
            logger.warning("✗ Fehler")
        
        logger.info("-" * 50)
    
    # Lösche die Testsitzung
    mlingua.clear_session(session_id)
    
    success_rate = success_count / len(test_texts) * 100
    logger.info(f"Erfolgsrate: {success_rate:.2f}% ({success_count}/{len(test_texts)})")
    
    return success_rate >= 75.0

def test_vxor_integration():
    """Testet die VXORIntegration"""
    logger.info("=== Test: VXORIntegration ===")
    
    vxor_integration = VXORIntegration()
    
    # Zeige verfügbare Module
    available_modules = vxor_integration.get_available_modules()
    logger.info(f"Verfügbare VXOR-Module: {', '.join(available_modules.keys())}")
    
    test_texts = {
        "de": "Öffne den Browser mit Suchparameter Wetter",
        "en": "Search for weather forecast in Berlin",
        "es": "Busca el pronóstico del tiempo en Madrid",
        "fr": "Ouvre le navigateur avec paramètre de recherche météo"
    }
    
    success_count = 0
    for lang, text in test_texts.items():
        logger.info(f"Verarbeite Text: {text}")
        semantic_result, command_results = vxor_integration.process_text(text)
        
        logger.info(f"Semantisches Ergebnis:")
        logger.info(f"  Sprache: {semantic_result.parsed_command.detected_language}")
        logger.info(f"  Intention: {semantic_result.parsed_command.intent}")
        logger.info(f"  Aktion: {semantic_result.parsed_command.action}")
        logger.info(f"  Ziel: {semantic_result.parsed_command.target}")
        logger.info(f"  Parameter: {semantic_result.parsed_command.parameters}")
        
        if semantic_result.requires_clarification:
            logger.info(f"  Rückfrage erforderlich: {semantic_result.feedback}")
            logger.info(f"  Optionen: {semantic_result.clarification_options}")
        else:
            logger.info(f"  VXOR-Befehle: {[cmd['command_string'] for cmd in semantic_result.vxor_commands]}")
            logger.info(f"  M-CODE: {semantic_result.m_code}")
        
        if semantic_result.parsed_command.intent != "UNKNOWN":
            success_count += 1
            logger.info("✓ Erfolg")
        else:
            logger.warning("✗ Fehler")
        
        logger.info("-" * 50)
    
    success_rate = success_count / len(test_texts) * 100
    logger.info(f"Erfolgsrate: {success_rate:.2f}% ({success_count}/{len(test_texts)})")
    
    return success_rate >= 75.0

def run_all_tests():
    """Führt alle Tests aus"""
    logger.info("=== M-LINGUA Testsuite ===")
    logger.info(f"Startzeit: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("LanguageDetector", test_language_detector),
        ("MultilingualParser", test_multilingual_parser),
        ("SemanticLayer", test_semantic_layer),
        ("MathBridge", test_math_bridge),
        ("VXORIntegration", test_vxor_integration),
        ("MLinguaInterface", test_mlingua_interface)
    ]
    
    results = {}
    all_passed = True
    
    for name, test_func in tests:
        logger.info(f"\n\n=== Starte Test: {name} ===")
        try:
            passed = test_func()
            results[name] = passed
            if not passed:
                all_passed = False
        except Exception as e:
            logger.error(f"Fehler bei Test {name}: {e}")
            results[name] = False
            all_passed = False
    
    logger.info("\n\n=== Testergebnisse ===")
    for name, passed in results.items():
        status = "✓ Bestanden" if passed else "✗ Fehlgeschlagen"
        logger.info(f"{name}: {status}")
    
    logger.info(f"\nGesamtergebnis: {'✓ Alle Tests bestanden' if all_passed else '✗ Einige Tests fehlgeschlagen'}")
    logger.info(f"Endzeit: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_passed

def create_vxor_mock_modules():
    """Erstellt Mock-Module für VXOR-Tests"""
    logger.info("Erstelle Mock-Module für VXOR-Tests")
    
    vxor_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'vxor'))
    os.makedirs(vxor_dir, exist_ok=True)
    
    # Erstelle vxor_manifest.json
    manifest = {
        "modules": {
            "VX-INTENT": {
                "module_path": "miso.vxor.vx_intent",
                "class_name": "VXIntent",
                "version": "1.0.0",
                "capabilities": ["intent_recognition", "action_execution"],
                "actions": {
                    "execute": {
                        "description": "Führt eine Aktion aus",
                        "parameters": ["target", "parameters"]
                    },
                    "terminate": {
                        "description": "Beendet eine Aktion",
                        "parameters": ["target", "parameters"]
                    },
                    "query": {
                        "description": "Führt eine Abfrage durch",
                        "parameters": ["target", "parameters"]
                    }
                }
            },
            "VX-MEMEX": {
                "module_path": "miso.vxor.vx_memex",
                "class_name": "VXMemex",
                "version": "1.0.0",
                "capabilities": ["memory_management", "information_retrieval"],
                "actions": {
                    "retrieve": {
                        "description": "Ruft Informationen ab",
                        "parameters": ["query", "context"]
                    },
                    "store": {
                        "description": "Speichert Informationen",
                        "parameters": ["data", "context"]
                    }
                }
            }
        }
    }
    
    with open(os.path.join(vxor_dir, "vxor_manifest.json"), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    # Erstelle Verzeichnisstruktur
    os.makedirs(os.path.join(vxor_dir, "vx_intent"), exist_ok=True)
    os.makedirs(os.path.join(vxor_dir, "vx_memex"), exist_ok=True)
    
    # Erstelle __init__.py-Dateien
    with open(os.path.join(vxor_dir, "__init__.py"), 'w') as f:
        f.write('"""VXOR-Modul-Paket"""\n')
    
    with open(os.path.join(vxor_dir, "vx_intent", "__init__.py"), 'w') as f:
        f.write('"""VX-INTENT-Modul"""\n\nclass VXIntent:\n    def execute(self, target, parameters):\n        return f"Executing {target} with {parameters}"\n\n    def terminate(self, target, parameters):\n        return f"Terminating {target} with {parameters}"\n\n    def query(self, target, parameters):\n        return f"Querying {target} with {parameters}"\n')
    
    with open(os.path.join(vxor_dir, "vx_memex", "__init__.py"), 'w') as f:
        f.write('"""VX-MEMEX-Modul"""\n\nclass VXMemex:\n    def retrieve(self, query, context):\n        return f"Retrieving information for {query} with context {context}"\n\n    def store(self, data, context):\n        return f"Storing data {data} with context {context}"\n')
    
    logger.info("Mock-Module für VXOR-Tests erstellt")

if __name__ == "__main__":
    # Erstelle Mock-Module für VXOR-Tests
    create_vxor_mock_modules()
    
    # Führe alle Tests aus
    success = run_all_tests()
    
    # Beende mit entsprechendem Exit-Code
    sys.exit(0 if success else 1)
