#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - M-LINGUA Interface

Hauptmodul für die M-LINGUA-Erweiterung, das alle Komponenten zusammenführt
und eine einheitliche Schnittstelle für die multilinguale Verarbeitung bietet.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field

# Importiere die Komponenten
from miso.lang.mlingua.language_detector import LanguageDetector
from miso.lang.mlingua.multilang_parser import MultilingualParser, ParsedCommand
from miso.lang.mlingua.semantic_layer import SemanticLayer, SemanticResult, SemanticContext
from miso.lang.mlingua.vxor_integration import VXORIntegration, VXORCommandResult

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [M-LINGUA] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MISO.M-LINGUA.Interface")

@dataclass
class MLinguaResult:
    """Ergebnis der M-LINGUA-Verarbeitung"""
    input_text: str
    detected_language: str
    semantic_result: SemanticResult
    vxor_results: List[VXORCommandResult] = field(default_factory=list)
    processing_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None

class MLinguaInterface:
    """
    Hauptschnittstelle für die M-LINGUA-Erweiterung
    
    Diese Klasse integriert alle Komponenten der M-LINGUA-Erweiterung und
    bietet eine einheitliche Schnittstelle für die multilinguale Verarbeitung
    natürlichsprachlicher Eingaben.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialisiert die M-LINGUA-Schnittstelle
        
        Args:
            config_path: Pfad zur Konfigurationsdatei (optional)
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "config", "mlingua_config.json"
        )
        self.language_detector = LanguageDetector()
        self.parser = MultilingualParser()
        self.semantic_layer = SemanticLayer()
        self.vxor_integration = VXORIntegration()
        self.config = {}
        self.sessions = {}
        self.load_config()
        logger.info("M-LINGUA-Schnittstelle initialisiert")
    
    def load_config(self):
        """Lädt die Konfiguration aus der Konfigurationsdatei"""
        try:
            # Stelle sicher, dass das Verzeichnis existiert
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Wenn die Datei nicht existiert, erstelle Standardkonfiguration
            if not os.path.exists(self.config_path):
                self._create_default_config()
            
            # Lade die Konfiguration
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            logger.info(f"M-LINGUA-Konfiguration geladen: {len(self.config)} Einträge")
        except Exception as e:
            logger.error(f"Fehler beim Laden der M-LINGUA-Konfiguration: {e}")
            # Erstelle Standardkonfiguration im Fehlerfall
            self._create_default_config()
    
    def _create_default_config(self):
        """Erstellt eine Standardkonfiguration"""
        default_config = {
            "general": {
                "default_language": "de",
                "fallback_language": "en",
                "min_confidence": 0.6,
                "max_session_age": 3600,  # 1 Stunde
                "log_all_interactions": True
            },
            "language_detection": {
                "enabled": True,
                "min_text_length": 3,
                "use_context": True
            },
            "semantic_processing": {
                "resolve_references": True,
                "use_conversation_history": True,
                "max_history_depth": 5
            },
            "vxor_integration": {
                "validate_commands": True,
                "log_commands": True,
                "retry_failed_commands": True,
                "max_retries": 3
            }
        }
        
        # Speichere die Standardkonfiguration
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        
        # Lade die Konfiguration in den Speicher
        self.config = default_config
        
        logger.info("Standard-M-LINGUA-Konfiguration erstellt")
    
    def process(self, text: str, session_id: Optional[str] = None, user_id: Optional[str] = None) -> MLinguaResult:
        """
        Verarbeitet einen natürlichsprachlichen Text
        
        Args:
            text: Zu verarbeitender Text
            session_id: ID der Sitzung (optional)
            user_id: ID des Benutzers (optional)
            
        Returns:
            MLinguaResult-Objekt mit dem Ergebnis der Verarbeitung
        """
        start_time = time.time()
        
        # Überprüfe den Text
        if not text or len(text.strip()) == 0:
            logger.warning("Leerer Text für Verarbeitung")
            return MLinguaResult(
                input_text=text,
                detected_language="unknown",
                semantic_result=SemanticResult(
                    parsed_command=ParsedCommand(
                        intent="UNKNOWN",
                        action="",
                        target="",
                        confidence=0.0,
                        original_text=text,
                        detected_language="unknown"
                    ),
                    context=SemanticContext(session_id=session_id or "unknown"),
                    confidence=0.0,
                    feedback="Leerer Text"
                ),
                success=False,
                error_message="Leerer Text"
            )
        
        # Erstelle oder hole Sitzung
        if not session_id:
            session_id = f"session_{hash(text + str(time.time())) % 10000:04d}"
        
        context = self._get_or_create_session(session_id, user_id)
        
        try:
            # Verarbeite den Text
            semantic_result, vxor_results = self.vxor_integration.process_text(text, context)
            
            # Erstelle Ergebnisobjekt
            result = MLinguaResult(
                input_text=text,
                detected_language=semantic_result.parsed_command.detected_language,
                semantic_result=semantic_result,
                vxor_results=vxor_results,
                processing_time=time.time() - start_time,
                success=True
            )
            
            # Aktualisiere die Sitzung
            self._update_session(session_id, context)
            
            # Logge die Interaktion, wenn aktiviert
            if self.config.get("general", {}).get("log_all_interactions", False):
                self._log_interaction(result)
            
            return result
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung: {e}")
            return MLinguaResult(
                input_text=text,
                detected_language="unknown",
                semantic_result=SemanticResult(
                    parsed_command=ParsedCommand(
                        intent="UNKNOWN",
                        action="",
                        target="",
                        confidence=0.0,
                        original_text=text,
                        detected_language="unknown"
                    ),
                    context=context,
                    confidence=0.0,
                    feedback=f"Fehler: {str(e)}"
                ),
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _get_or_create_session(self, session_id: str, user_id: Optional[str] = None) -> SemanticContext:
        """
        Holt oder erstellt eine Sitzung
        
        Args:
            session_id: ID der Sitzung
            user_id: ID des Benutzers (optional)
            
        Returns:
            SemanticContext-Objekt für die Sitzung
        """
        # Überprüfe, ob die Sitzung existiert
        if session_id in self.sessions:
            # Überprüfe das Alter der Sitzung
            session_age = time.time() - self.sessions[session_id].get("last_access", 0)
            max_age = self.config.get("general", {}).get("max_session_age", 3600)
            
            if session_age <= max_age:
                # Aktualisiere den Zeitstempel
                self.sessions[session_id]["last_access"] = time.time()
                return self.sessions[session_id]["context"]
        
        # Erstelle eine neue Sitzung
        context = SemanticContext(
            session_id=session_id,
            user_id=user_id
        )
        
        self.sessions[session_id] = {
            "context": context,
            "created": time.time(),
            "last_access": time.time()
        }
        
        return context
    
    def _update_session(self, session_id: str, context: SemanticContext):
        """
        Aktualisiert eine Sitzung
        
        Args:
            session_id: ID der Sitzung
            context: Kontext der Sitzung
        """
        if session_id in self.sessions:
            self.sessions[session_id]["context"] = context
            self.sessions[session_id]["last_access"] = time.time()
    
    def _log_interaction(self, result: MLinguaResult):
        """
        Loggt eine Interaktion
        
        Args:
            result: Ergebnis der Interaktion
        """
        log_entry = {
            "timestamp": time.time(),
            "input_text": result.input_text,
            "detected_language": result.detected_language,
            "intent": result.semantic_result.parsed_command.intent,
            "action": result.semantic_result.parsed_command.action,
            "target": result.semantic_result.parsed_command.target,
            "parameters": result.semantic_result.parsed_command.parameters,
            "success": result.success,
            "processing_time": result.processing_time,
            "vxor_commands": [
                {
                    "module": cmd.module,
                    "action": cmd.action,
                    "success": cmd.success,
                    "execution_time": cmd.execution_time
                }
                for cmd in result.vxor_results
            ]
        }
        
        # In einer realen Implementierung würde hier das Logging in eine Datei oder Datenbank erfolgen
        logger.info(f"Interaktion geloggt: {log_entry}")
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """
        Gibt eine Liste der unterstützten Sprachen zurück
        
        Returns:
            Liste mit Informationen zu unterstützten Sprachen
        """
        return self.language_detector.get_supported_languages()
    
    def get_available_vxor_modules(self) -> Dict[str, Any]:
        """
        Gibt eine Liste der verfügbaren VXOR-Module zurück
        
        Returns:
            Dictionary mit Informationen zu verfügbaren VXOR-Modulen
        """
        return self.vxor_integration.get_available_modules()
    
    def clear_session(self, session_id: str) -> bool:
        """
        Löscht eine Sitzung
        
        Args:
            session_id: ID der Sitzung
            
        Returns:
            True, wenn die Sitzung gelöscht wurde, sonst False
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Sitzung gelöscht: {session_id}")
            return True
        
        logger.warning(f"Sitzung nicht gefunden: {session_id}")
        return False
    
    def clear_all_sessions(self):
        """Löscht alle Sitzungen"""
        self.sessions.clear()
        logger.info("Alle Sitzungen gelöscht")

# Erstelle eine Instanz der M-LINGUA-Schnittstelle, wenn dieses Skript direkt ausgeführt wird
if __name__ == "__main__":
    mlingua = MLinguaInterface()
    
    # Zeige unterstützte Sprachen
    supported_languages = mlingua.get_supported_languages()
    print(f"Unterstützte Sprachen: {', '.join([lang['name'] for lang in supported_languages])}")
    
    # Zeige verfügbare VXOR-Module
    available_modules = mlingua.get_available_vxor_modules()
    print(f"Verfügbare VXOR-Module: {', '.join(available_modules.keys())}")
    
    # Beispieltexte
    test_texts = {
        "de": "Öffne den Browser mit Suchparameter Wetter",
        "en": "Search for weather forecast in Berlin",
        "es": "Busca el pronóstico del tiempo en Madrid",
        "fr": "Ouvre le navigateur avec paramètre de recherche météo"
    }
    
    # Teste die Verarbeitung
    session_id = f"test_session_{int(time.time())}"
    for lang, text in test_texts.items():
        print(f"\nVerarbeite Text: {text}")
        result = mlingua.process(text, session_id=session_id)
        
        print(f"Ergebnis:")
        print(f"  Erkannte Sprache: {result.detected_language}")
        print(f"  Verarbeitungszeit: {result.processing_time:.4f}s")
        print(f"  Erfolg: {result.success}")
        
        if result.success:
            semantic = result.semantic_result
            print(f"  Semantisches Ergebnis:")
            print(f"    Intention: {semantic.parsed_command.intent}")
            print(f"    Aktion: {semantic.parsed_command.action}")
            print(f"    Ziel: {semantic.parsed_command.target}")
            print(f"    Parameter: {semantic.parsed_command.parameters}")
            
            if semantic.requires_clarification:
                print(f"    Rückfrage erforderlich: {semantic.feedback}")
                print(f"    Optionen: {semantic.clarification_options}")
            else:
                print(f"    VXOR-Befehle: {[cmd['command_string'] for cmd in semantic.vxor_commands]}")
                print(f"    M-CODE: {semantic.m_code}")
            
            print(f"  VXOR-Ergebnisse:")
            for i, vxor_result in enumerate(result.vxor_results):
                print(f"    Befehl {i+1}:")
                print(f"      Modul: {vxor_result.module}")
                print(f"      Aktion: {vxor_result.action}")
                print(f"      Erfolg: {vxor_result.success}")
                if vxor_result.success:
                    print(f"      Ergebnis: {vxor_result.result}")
                    print(f"      Ausführungszeit: {vxor_result.execution_time:.4f}s")
                else:
                    print(f"      Fehlermeldung: {vxor_result.error_message}")
        else:
            print(f"  Fehlermeldung: {result.error_message}")
        
        print("-" * 50)
    
    # Lösche die Testsitzung
    mlingua.clear_session(session_id)
