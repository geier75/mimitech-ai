#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - M-LINGUA T-Mathematics Bridge

Dieses Modul implementiert die Brücke zwischen dem M-LINGUA Interface und der T-Mathematics Engine,
um natürlichsprachliche Befehle in Tensor-Operationen umzuwandeln.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Union, Tuple

from miso_ultimate.engines.t_mathematics.engine import TMathematicsEngine
from miso_ultimate.lang.m_lingua.interface import MLinguaInterface
from miso_ultimate.lang.m_code.compiler import MCodeCompiler
from miso_ultimate.lang.m_code.executor import MCodeExecutor

logger = logging.getLogger("MISO.Ultimate.MLingua.TMathematicsBridge")

class TMathematicsBridge:
    """
    Brücke zwischen M-LINGUA und T-Mathematics Engine
    
    Diese Klasse ermöglicht die Steuerung der T-Mathematics Engine
    durch natürlichsprachliche Befehle über das M-LINGUA Interface.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialisiert die T-Mathematics Bridge
        
        Args:
            config: Konfigurationsobjekt für die Bridge
        """
        self.config = config
        self.language = config.get("language", "de")
        
        # Initialisiere M-LINGUA Interface
        m_lingua_config = config.get("m_lingua", {})
        self.m_lingua = MLinguaInterface(m_lingua_config)
        
        # Initialisiere T-Mathematics Engine
        t_math_config = config.get("t_mathematics", {})
        self.t_math = TMathematicsEngine(t_math_config)
        
        # Initialisiere M-CODE Compiler und Executor
        self.compiler = MCodeCompiler(config.get("m_code", {}))
        self.executor = MCodeExecutor(config.get("m_code", {}))
        
        # Initialisiere Kontext-Speicher
        self.context_memory = []
        
        # Initialisiere Tensor-Cache
        self.tensor_cache = {}
        
        logger.info("T-Mathematics Bridge initialisiert")
    
    def process(self, input_text: str) -> Dict[str, Any]:
        """
        Verarbeitet einen natürlichsprachlichen Befehl und führt entsprechende Tensor-Operationen aus
        
        Args:
            input_text: Eingabetext in natürlicher Sprache
            
        Returns:
            Dictionary mit Ergebnissen der Verarbeitung
        """
        # Initialisiere Ergebnis
        result = {
            "success": False,
            "message": "",
            "output": None,
            "error": None,
            "m_code": None,
            "execution_time": 0.0
        }
        
        try:
            # Verarbeite Text mit M-LINGUA
            m_lingua_result = self.m_lingua.process(input_text)
            
            # Extrahiere M-CODE
            m_code = m_lingua_result["m_code"]
            result["m_code"] = m_code
            
            # Erweitere M-CODE um T-Mathematics-spezifische Funktionen
            m_code = self._enhance_m_code(m_code)
            
            # Kompiliere M-CODE
            compiled_code = self.compiler.compile(m_code)
            
            # Führe kompilierten Code aus
            execution_result = self.executor.execute(compiled_code)
            
            # Aktualisiere Ergebnis
            result["success"] = True
            result["message"] = "Befehl erfolgreich ausgeführt"
            result["output"] = execution_result["output"]
            result["execution_time"] = execution_result["execution_time"]
            
            # Aktualisiere Kontext-Speicher
            self._update_context_memory(input_text, m_lingua_result, execution_result)
            
            # Aktualisiere Tensor-Cache
            self._update_tensor_cache(execution_result)
        
        except Exception as e:
            # Fehlerbehandlung
            result["success"] = False
            result["message"] = f"Fehler bei der Verarbeitung: {str(e)}"
            result["error"] = str(e)
            logger.error(f"Fehler bei der Verarbeitung von '{input_text}': {str(e)}")
        
        return result
    
    def _enhance_m_code(self, m_code: str) -> str:
        """
        Erweitert M-CODE um T-Mathematics-spezifische Funktionen
        
        Args:
            m_code: Generierter M-CODE
            
        Returns:
            Erweiterter M-CODE
        """
        # Füge Import-Anweisungen hinzu, falls nicht vorhanden
        if "from miso_ultimate.engines.t_mathematics.engine import TMathematicsEngine" not in m_code:
            m_code = "from miso_ultimate.engines.t_mathematics.engine import TMathematicsEngine\n" + m_code
        
        # Füge Initialisierung der T-Mathematics Engine hinzu, falls nicht vorhanden
        if "t_math = TMathematicsEngine" not in m_code:
            m_code = m_code.replace(
                "from miso_ultimate.engines.t_mathematics.engine import TMathematicsEngine\n",
                "from miso_ultimate.engines.t_mathematics.engine import TMathematicsEngine\n"
                "t_math = TMathematicsEngine(config)\n"
            )
        
        # Füge Tensor-Cache-Zugriff hinzu
        m_code = m_code.replace(
            "t_math = TMathematicsEngine(config)\n",
            "t_math = TMathematicsEngine(config)\n"
            "# Lade Tensoren aus dem Cache\n"
            "tensor_cache = bridge.tensor_cache\n"
        )
        
        return m_code
    
    def _update_context_memory(self, input_text: str, m_lingua_result: Dict[str, Any], 
                              execution_result: Dict[str, Any]) -> None:
        """
        Aktualisiert den Kontext-Speicher
        
        Args:
            input_text: Eingabetext in natürlicher Sprache
            m_lingua_result: Ergebnis der M-LINGUA-Verarbeitung
            execution_result: Ergebnis der M-CODE-Ausführung
        """
        # Erstelle Kontext-Eintrag
        context_entry = {
            "input": input_text,
            "intent": m_lingua_result.get("intent", {}),
            "symbols": m_lingua_result.get("symbols", []),
            "context": m_lingua_result.get("context", {}),
            "m_code": m_lingua_result.get("m_code", ""),
            "output": execution_result.get("output", None),
            "execution_time": execution_result.get("execution_time", 0.0),
            "timestamp": execution_result.get("timestamp", 0.0)
        }
        
        # Füge Eintrag zum Kontext-Speicher hinzu
        self.context_memory.append(context_entry)
        
        # Begrenze Größe des Kontext-Speichers
        max_context_size = self.config.get("max_context_size", 10)
        if len(self.context_memory) > max_context_size:
            self.context_memory = self.context_memory[-max_context_size:]
    
    def _update_tensor_cache(self, execution_result: Dict[str, Any]) -> None:
        """
        Aktualisiert den Tensor-Cache
        
        Args:
            execution_result: Ergebnis der M-CODE-Ausführung
        """
        # Extrahiere Tensoren aus dem Ausführungsergebnis
        variables = execution_result.get("variables", {})
        for name, value in variables.items():
            # Prüfe, ob es sich um einen Tensor handelt
            if hasattr(value, "__class__") and "Tensor" in value.__class__.__name__:
                # Speichere Tensor im Cache
                self.tensor_cache[name] = value
        
        # Begrenze Größe des Tensor-Caches
        max_cache_size = self.config.get("max_tensor_cache_size", 20)
        if len(self.tensor_cache) > max_cache_size:
            # Entferne älteste Einträge
            oldest_keys = list(self.tensor_cache.keys())[:len(self.tensor_cache) - max_cache_size]
            for key in oldest_keys:
                del self.tensor_cache[key]
    
    def get_available_backends(self) -> List[str]:
        """
        Gibt die verfügbaren Tensor-Backends zurück
        
        Returns:
            Liste der verfügbaren Backends
        """
        return self.t_math.get_available_backends()
    
    def set_default_backend(self, backend: str) -> bool:
        """
        Setzt das Standard-Backend für Tensor-Operationen
        
        Args:
            backend: Name des Backends
            
        Returns:
            True, wenn erfolgreich, sonst False
        """
        try:
            self.t_math.set_default_backend(backend)
            return True
        except Exception as e:
            logger.error(f"Fehler beim Setzen des Standard-Backends '{backend}': {str(e)}")
            return False
    
    def get_default_backend(self) -> str:
        """
        Gibt das aktuelle Standard-Backend zurück
        
        Returns:
            Name des Standard-Backends
        """
        return self.t_math.get_default_backend()
    
    def create_tensor(self, data, dtype=None, backend=None) -> Any:
        """
        Erstellt einen Tensor
        
        Args:
            data: Tensor-Daten
            dtype: Datentyp
            backend: Backend
            
        Returns:
            Erstellter Tensor
        """
        return self.t_math.create_tensor(data, dtype=dtype, backend=backend)
    
    def execute_operation(self, operation: str, *args, **kwargs) -> Any:
        """
        Führt eine Tensor-Operation aus
        
        Args:
            operation: Name der Operation
            *args: Positionsargumente
            **kwargs: Schlüsselwortargumente
            
        Returns:
            Ergebnis der Operation
        """
        # Prüfe, ob die Operation existiert
        if hasattr(self.t_math, operation):
            # Führe Operation aus
            operation_func = getattr(self.t_math, operation)
            return operation_func(*args, **kwargs)
        else:
            raise ValueError(f"Unbekannte Operation: {operation}")
