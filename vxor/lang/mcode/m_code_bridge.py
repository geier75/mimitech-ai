#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - M-CODE zu M-LINGUA Bridge

Dieses Modul implementiert die Brücke zwischen M-CODE und M-LINGUA.
Es ermöglicht die Übersetzung von natürlichsprachlichen Befehlen in M-CODE
und die Integration von M-CODE in die M-LINGUA-Verarbeitung.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field

# Importiere M-LINGUA-Komponenten
from miso.lang.mlingua.mlingua_interface import MLinguaInterface, MLinguaResult
from miso.lang.mlingua.semantic_layer import SemanticResult, SemanticContext
from miso.lang.mlingua.multilang_parser import ParsedCommand

# Importiere M-CODE-Komponenten
from miso.code.m_code import MCodeCompiler, MCodeInterpreter, MCodeSyntaxTree
from miso.code.m_code import compile_m_code, execute_m_code, parse_m_code, optimize_m_code
from miso.code.m_code import initialize_m_code, get_runtime, reset_runtime

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [M-CODE-BRIDGE] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MISO.M-CODE-BRIDGE")

@dataclass
class MCodeBridgeResult:
    """Ergebnis der M-CODE-Bridge-Verarbeitung"""
    input_text: str
    generated_m_code: str
    execution_result: Any
    mlingua_result: Optional[MLinguaResult] = None
    processing_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None

class MCodeBridge:
    """
    Brücke zwischen M-CODE und M-LINGUA
    
    Diese Klasse implementiert die Brücke zwischen M-CODE und M-LINGUA.
    Sie ermöglicht die Übersetzung von natürlichsprachlichen Befehlen in M-CODE
    und die Integration von M-CODE in die M-LINGUA-Verarbeitung.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialisiert die M-CODE-Bridge
        
        Args:
            config_path: Pfad zur Konfigurationsdatei (optional)
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "config", "m_code_bridge_config.json"
        )
        self.mlingua_interface = MLinguaInterface()
        self.compiler = MCodeCompiler()
        self.interpreter = MCodeInterpreter()
        self.config = {}
        self.templates = {}
        self.load_config()
        
        # Initialisiere M-CODE-Runtime
        initialize_m_code(
            optimization_level=self.config.get("optimization_level", 2),
            use_jit=self.config.get("use_jit", True),
            memory_limit_mb=self.config.get("memory_limit_mb", 1024),
            enable_extensions=self.config.get("enable_extensions", True)
        )
        
        logger.info("M-CODE-Bridge initialisiert")
    
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
                config = json.load(f)
            
            self.config = config.get("config", {})
            self.templates = config.get("templates", {})
            
            logger.info(f"Konfiguration geladen: {len(self.templates)} Templates")
        except Exception as e:
            logger.error(f"Fehler beim Laden der Konfiguration: {e}")
            # Erstelle Standardkonfiguration im Fehlerfall
            self._create_default_config()
    
    def _create_default_config(self):
        """Erstellt eine Standardkonfiguration"""
        default_config = {
            "config": {
                "optimization_level": 2,
                "use_jit": True,
                "memory_limit_mb": 1024,
                "enable_extensions": True,
                "log_generated_code": True,
                "validate_code": True,
                "max_execution_time": 60.0,
                "default_language": "de"
            },
            "templates": {
                "EXECUTION": "func execute_command(target, params) {\n    // Führe Befehl aus\n    return system.execute(target, params);\n}",
                "QUERY": "func query_information(target, params) {\n    // Frage Informationen ab\n    return system.query(target, params);\n}",
                "CREATION": "func create_object(type, specs) {\n    // Erstelle Objekt\n    return system.create(type, specs);\n}",
                "MODIFICATION": "func modify_object(target, changes) {\n    // Modifiziere Objekt\n    return system.modify(target, changes);\n}",
                "DELETION": "func delete_object(target) {\n    // Lösche Objekt\n    return system.delete(target);\n}"
            }
        }
        
        # Speichere die Standardkonfiguration
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        
        # Lade die Konfiguration in den Speicher
        self.config = default_config["config"]
        self.templates = default_config["templates"]
        
        logger.info("Standard-M-CODE-Bridge-Konfiguration erstellt")
    
    def process_natural_language(self, text: str, session_id: Optional[str] = None) -> MCodeBridgeResult:
        """
        Verarbeitet natürlichsprachlichen Text und generiert M-CODE
        
        Args:
            text: Zu verarbeitender Text
            session_id: ID der Sitzung (optional)
            
        Returns:
            MCodeBridgeResult-Objekt mit dem Ergebnis der Verarbeitung
        """
        start_time = time.time()
        
        try:
            # Verarbeite den Text mit M-LINGUA
            mlingua_result = self.mlingua_interface.process(text, session_id)
            
            # Generiere M-CODE aus dem semantischen Ergebnis
            m_code = self._generate_m_code(mlingua_result.semantic_result)
            
            # Führe den M-CODE aus
            execution_result = self._execute_m_code(m_code)
            
            # Erstelle Ergebnisobjekt
            result = MCodeBridgeResult(
                input_text=text,
                generated_m_code=m_code,
                execution_result=execution_result,
                mlingua_result=mlingua_result,
                processing_time=time.time() - start_time,
                success=True
            )
            
            # Logge den generierten Code, wenn aktiviert
            if self.config.get("log_generated_code", True):
                logger.info(f"Generierter M-CODE:\n{m_code}")
            
            return result
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung: {e}")
            return MCodeBridgeResult(
                input_text=text,
                generated_m_code="",
                execution_result=None,
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def execute_m_code(self, m_code: str) -> MCodeBridgeResult:
        """
        Führt M-CODE direkt aus
        
        Args:
            m_code: Auszuführender M-CODE
            
        Returns:
            MCodeBridgeResult-Objekt mit dem Ergebnis der Ausführung
        """
        start_time = time.time()
        
        try:
            # Führe den M-CODE aus
            execution_result = self._execute_m_code(m_code)
            
            # Erstelle Ergebnisobjekt
            result = MCodeBridgeResult(
                input_text=m_code,
                generated_m_code=m_code,
                execution_result=execution_result,
                processing_time=time.time() - start_time,
                success=True
            )
            
            return result
        except Exception as e:
            logger.error(f"Fehler bei der Ausführung: {e}")
            return MCodeBridgeResult(
                input_text=m_code,
                generated_m_code=m_code,
                execution_result=None,
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _generate_m_code(self, semantic_result: SemanticResult) -> str:
        """
        Generiert M-CODE aus einem semantischen Ergebnis
        
        Args:
            semantic_result: Semantisches Ergebnis
            
        Returns:
            Generierter M-CODE
        """
        # Hole die relevanten Informationen aus dem semantischen Ergebnis
        intent = semantic_result.parsed_command.intent
        action = semantic_result.parsed_command.action
        target = semantic_result.parsed_command.target
        parameters = semantic_result.parsed_command.parameters
        
        # Konvertiere Parameter in einen String
        params_str = json.dumps(parameters, ensure_ascii=False)
        
        # Hole das passende Template
        template = self.templates.get(intent, "")
        
        if not template:
            # Fallback-Template
            template = "func execute(target, params) {\n    return system.execute(target, params);\n}"
        
        # Erstelle den M-CODE
        m_code = f"""
// Generierter M-CODE für Intent: {intent}, Action: {action}, Target: {target}
// Zeitstempel: {time.strftime('%Y-%m-%d %H:%M:%S')}

{template}

// Hauptausführung
func main() {{
    let target = "{target}";
    let params = {params_str};
    
    // Führe die entsprechende Funktion aus
    let result = null;
    if ("{intent}" == "EXECUTION") {{
        result = execute_command(target, params);
    }} else if ("{intent}" == "QUERY") {{
        result = query_information(target, params);
    }} else if ("{intent}" == "CREATION") {{
        result = create_object(target, params);
    }} else if ("{intent}" == "MODIFICATION") {{
        result = modify_object(target, params);
    }} else if ("{intent}" == "DELETION") {{
        result = delete_object(target);
    }} else {{
        // Fallback
        result = execute(target, params);
    }}
    
    return result;
}}

// Rufe die Hauptfunktion auf
main();
"""
        
        return m_code
    
    def _execute_m_code(self, m_code: str) -> Any:
        """
        Führt M-CODE aus
        
        Args:
            m_code: Auszuführender M-CODE
            
        Returns:
            Ergebnis der Ausführung
        """
        try:
            # Parse M-CODE
            syntax_tree = parse_m_code(m_code)
            
            # Optimiere M-CODE
            optimized_tree = optimize_m_code(syntax_tree, self.config.get("optimization_level", 2))
            
            # Kompiliere M-CODE
            bytecode = compile_m_code(m_code)
            
            # Führe M-CODE aus
            result = execute_m_code(bytecode)
            
            return result
        except Exception as e:
            logger.error(f"Fehler bei der Ausführung von M-CODE: {e}")
            raise

# Erstelle eine Instanz der M-CODE-Bridge, wenn dieses Skript direkt ausgeführt wird
if __name__ == "__main__":
    bridge = MCodeBridge()
    
    # Beispieltexte
    test_texts = [
        "Öffne die Datei 'beispiel.txt'",
        "Suche nach Informationen über Quantencomputer",
        "Erstelle eine neue Textdatei mit dem Namen 'notizen.txt'",
        "Beende das Programm"
    ]
    
    # Verarbeite die Beispieltexte
    for text in test_texts:
        print(f"\nVerarbeite: '{text}'")
        result = bridge.process_natural_language(text)
        
        if result.success:
            print(f"Generierter M-CODE:\n{result.generated_m_code}")
            print(f"Ausführungsergebnis: {result.execution_result}")
        else:
            print(f"Fehler: {result.error_message}")
