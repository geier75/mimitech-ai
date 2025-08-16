#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - NEXUS-OS LinguaMathBridge

Diese Datei implementiert die Brücke zwischen dem M-LINGUA Interface und der 
T-Mathematics Engine, um mathematische Ausdrücke und Tensor-Operationen direkt 
über natürliche Sprache zu steuern.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import threading
import json
import re
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Importiere erforderliche Komponenten
from miso.lang.mcode_engine import MCodeEngine, MCodeConfig, execute_m_code, get_engine
from miso.math.t_mathematics.engine import TMathEngine
from miso.math.t_mathematics.compat import TMathConfig

# Konfiguriere Logging
logger = logging.getLogger("MISO.nexus_os.lingua_math_bridge")

class LinguaMathBridge:
    """
    Brücke zwischen M-LINGUA Interface und T-Mathematics Engine
    
    Diese Klasse ermöglicht die Steuerung von Tensor-Operationen und mathematischen
    Berechnungen über natürliche Sprache, indem sie M-LINGUA mit der T-Mathematics
    Engine verbindet.
    """
    
    def __init__(self):
        """Initialisiert die LinguaMathBridge"""
        self.lock = threading.RLock()
        self.initialized = False
        self.m_code_engine = None
        self.t_math_engine = None
        self.tensor_cache = {}
        self.operation_history = []
        self.max_history_size = 100
        logger.info("LinguaMathBridge initialisiert")
    
    def initialize(self):
        """Initialisiert die Brücke mit den erforderlichen Engines"""
        if self.initialized:
            logger.warning("LinguaMathBridge bereits initialisiert")
            return
        
        with self.lock:
            # Initialisiere M-CODE Engine
            m_code_config = MCodeConfig(
                optimization_level=3,
                use_jit=True,
                use_gpu=True,
                use_neural_engine=True,
                security_level=2,  # Reduzierte Sicherheit für interne Verwendung
                memory_limit_mb=8192,
                allow_network_access=False,
                allow_file_access=True,  # Erlaube Dateizugriff für Tensor-Operationen
                debug_mode=False
            )
            self.m_code_engine = MCodeEngine(m_code_config)
            
            # Initialisiere T-Mathematics Engine
            t_math_config = TMathConfig(
                precision="mixed",
                device="auto",
                optimize_for_rdna=True,
                optimize_for_apple_silicon=True,
                use_flash_attention=True
            )
            self.t_math_engine = TMathEngine(t_math_config)
            
            # Registriere T-Mathematics Engine bei M-CODE
            self._register_math_engine()
            
            self.initialized = True
            logger.info("LinguaMathBridge vollständig initialisiert")
    
    def _register_math_engine(self):
        """Registriert die T-Mathematics Engine bei der M-CODE Engine"""
        # Erstelle globale Variablen für M-CODE
        global_vars = {
            "t_math_engine": self.t_math_engine,
            "tensor_ops": self.t_math_engine.tensor_ops if hasattr(self.t_math_engine, "tensor_ops") else {},
            "math_utils": self.t_math_engine.math_utils if hasattr(self.t_math_engine, "math_utils") else {}
        }
        
        # Registriere globale Variablen
        self.m_code_engine.runtime.register_globals(global_vars)
        
        # Registriere Callback für Tensor-Operationen
        self.m_code_engine.register_callback("tensor_operation", self._on_tensor_operation)
        
        logger.debug("T-Mathematics Engine bei M-CODE registriert")
    
    def _on_tensor_operation(self, **kwargs):
        """Callback für Tensor-Operationen"""
        operation = kwargs.get("operation", "")
        result = kwargs.get("result", None)
        
        # Füge Operation zur Historie hinzu
        self.operation_history.append({
            "operation": operation,
            "timestamp": kwargs.get("timestamp", None),
            "success": kwargs.get("success", False)
        })
        
        # Begrenze Historiengröße
        if len(self.operation_history) > self.max_history_size:
            self.operation_history = self.operation_history[-self.max_history_size:]
    
    def parse_natural_language(self, text: str) -> Dict[str, Any]:
        """
        Analysiert natürliche Sprache und extrahiert mathematische Operationen.
        
        Args:
            text: Natürlichsprachiger Text
            
        Returns:
            Dictionary mit extrahierten Operationen und Parametern
        """
        # Einfache Musteranalyse für mathematische Ausdrücke
        # In einer vollständigen Implementierung würde hier ein NLP-Modell verwendet werden
        
        operations = {
            "type": "unknown",
            "operation": None,
            "parameters": {},
            "raw_text": text
        }
        
        # Erkenne Tensoroperationen
        tensor_patterns = [
            (r"(?i)erstelle\s+(?:einen|eine|ein)?\s*tensor\s+mit\s+(?:den\s+)?werten\s+(.*)", "create_tensor"),
            (r"(?i)berechne\s+(?:die\s+)?matrix(?:en)?multiplikation\s+von\s+(.*)\s+und\s+(.*)", "matrix_multiply"),
            (r"(?i)berechne\s+(?:das\s+)?skalarprodukt\s+von\s+(.*)\s+und\s+(.*)", "dot_product"),
            (r"(?i)transponiere\s+(?:den\s+)?tensor\s+(.*)", "transpose"),
            (r"(?i)wende\s+(?:eine\s+)?(?:die\s+)?softmax(?:-funktion)?\s+auf\s+(.*)\s+an", "softmax"),
            (r"(?i)berechne\s+(?:die\s+)?summe\s+von\s+(.*)", "sum"),
            (r"(?i)berechne\s+(?:den\s+)?mittelwert\s+von\s+(.*)", "mean"),
            (r"(?i)normalisiere\s+(?:den\s+)?tensor\s+(.*)", "normalize")
        ]
        
        for pattern, op_type in tensor_patterns:
            match = re.search(pattern, text)
            if match:
                operations["type"] = "tensor"
                operations["operation"] = op_type
                
                # Extrahiere Parameter basierend auf Operation
                if op_type == "create_tensor":
                    values_str = match.group(1)
                    try:
                        # Versuche, die Werte zu parsen
                        values = self._parse_tensor_values(values_str)
                        operations["parameters"]["values"] = values
                    except Exception as e:
                        logger.error(f"Fehler beim Parsen der Tensorwerte: {e}")
                        operations["error"] = str(e)
                
                elif op_type in ["matrix_multiply", "dot_product"]:
                    operations["parameters"]["tensor1"] = match.group(1).strip()
                    operations["parameters"]["tensor2"] = match.group(2).strip()
                
                elif op_type in ["transpose", "softmax", "sum", "mean", "normalize"]:
                    operations["parameters"]["tensor"] = match.group(1).strip()
                
                break
        
        return operations
    
    def _parse_tensor_values(self, values_str: str) -> List:
        """
        Parst Tensorwerte aus einem String.
        
        Args:
            values_str: String mit Tensorwerten
            
        Returns:
            Liste oder verschachtelte Liste mit Werten
        """
        # Entferne unnötige Zeichen
        values_str = values_str.strip()
        
        # Prüfe auf Matrix-Notation
        if "[" in values_str and "]" in values_str:
            # Versuche als JSON zu parsen
            try:
                # Ersetze einzelne Anführungszeichen durch doppelte
                values_str = values_str.replace("'", "\"")
                return json.loads(values_str)
            except json.JSONDecodeError:
                # Fallback: Manuelles Parsen
                matrix = []
                rows = re.findall(r'\[(.*?)\]', values_str)
                for row in rows:
                    values = [float(x.strip()) for x in row.split(",") if x.strip()]
                    matrix.append(values)
                return matrix
        else:
            # Einfache Liste von Werten
            return [float(x.strip()) for x in values_str.split(",") if x.strip()]
    
    def execute_math_operation(self, operations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Führt eine mathematische Operation basierend auf natürlicher Sprache aus.
        
        Args:
            operations: Dictionary mit Operationen und Parametern
            
        Returns:
            Ergebnis der Operation
        """
        if not self.initialized:
            self.initialize()
        
        result = {
            "success": False,
            "result": None,
            "error": None,
            "operation": operations
        }
        
        try:
            op_type = operations.get("type", "unknown")
            operation = operations.get("operation", None)
            parameters = operations.get("parameters", {})
            
            if op_type == "tensor" and operation:
                # Generiere M-CODE für die Tensoroperation
                m_code = self._generate_tensor_m_code(operation, parameters)
                
                # Führe M-CODE aus
                execution_result = self.m_code_engine.execute(
                    m_code,
                    module_name="lingua_math_bridge",
                    verify=False  # Keine Verifizierung für interne Operationen
                )
                
                if execution_result.success:
                    result["success"] = True
                    result["result"] = execution_result.result
                else:
                    result["error"] = str(execution_result.error)
            else:
                result["error"] = f"Unbekannte Operation: {op_type}/{operation}"
        
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Fehler bei der Ausführung der mathematischen Operation: {e}")
        
        return result
    
    def _generate_tensor_m_code(self, operation: str, parameters: Dict[str, Any]) -> str:
        """
        Generiert M-CODE für eine Tensoroperation.
        
        Args:
            operation: Art der Operation
            parameters: Parameter für die Operation
            
        Returns:
            M-CODE als String
        """
        m_code = "// Automatisch generierter M-CODE für Tensoroperation\n"
        m_code += "using tensor_ops;\n\n"
        m_code += "function main() {\n"
        
        if operation == "create_tensor":
            values = parameters.get("values", [])
            m_code += f"    // Erstelle Tensor mit Werten\n"
            m_code += f"    let values = {json.dumps(values)};\n"
            m_code += f"    let tensor = tensor_ops.create_tensor(values);\n"
            m_code += f"    return tensor;\n"
        
        elif operation == "matrix_multiply":
            tensor1 = parameters.get("tensor1", "")
            tensor2 = parameters.get("tensor2", "")
            m_code += f"    // Matrix-Multiplikation\n"
            m_code += f"    let t1 = tensor_ops.get_tensor(\"{tensor1}\");\n"
            m_code += f"    let t2 = tensor_ops.get_tensor(\"{tensor2}\");\n"
            m_code += f"    let result = tensor_ops.matmul(t1, t2);\n"
            m_code += f"    return result;\n"
        
        elif operation == "dot_product":
            tensor1 = parameters.get("tensor1", "")
            tensor2 = parameters.get("tensor2", "")
            m_code += f"    // Skalarprodukt\n"
            m_code += f"    let t1 = tensor_ops.get_tensor(\"{tensor1}\");\n"
            m_code += f"    let t2 = tensor_ops.get_tensor(\"{tensor2}\");\n"
            m_code += f"    let result = tensor_ops.dot(t1, t2);\n"
            m_code += f"    return result;\n"
        
        elif operation == "transpose":
            tensor = parameters.get("tensor", "")
            m_code += f"    // Transponiere Tensor\n"
            m_code += f"    let t = tensor_ops.get_tensor(\"{tensor}\");\n"
            m_code += f"    let result = tensor_ops.transpose(t);\n"
            m_code += f"    return result;\n"
        
        elif operation == "softmax":
            tensor = parameters.get("tensor", "")
            m_code += f"    // Softmax-Funktion\n"
            m_code += f"    let t = tensor_ops.get_tensor(\"{tensor}\");\n"
            m_code += f"    let result = tensor_ops.softmax(t);\n"
            m_code += f"    return result;\n"
        
        elif operation == "sum":
            tensor = parameters.get("tensor", "")
            m_code += f"    // Summe\n"
            m_code += f"    let t = tensor_ops.get_tensor(\"{tensor}\");\n"
            m_code += f"    let result = tensor_ops.sum(t);\n"
            m_code += f"    return result;\n"
        
        elif operation == "mean":
            tensor = parameters.get("tensor", "")
            m_code += f"    // Mittelwert\n"
            m_code += f"    let t = tensor_ops.get_tensor(\"{tensor}\");\n"
            m_code += f"    let result = tensor_ops.mean(t);\n"
            m_code += f"    return result;\n"
        
        elif operation == "normalize":
            tensor = parameters.get("tensor", "")
            m_code += f"    // Normalisierung\n"
            m_code += f"    let t = tensor_ops.get_tensor(\"{tensor}\");\n"
            m_code += f"    let result = tensor_ops.normalize(t);\n"
            m_code += f"    return result;\n"
        
        else:
            m_code += f"    // Unbekannte Operation: {operation}\n"
            m_code += f"    return null;\n"
        
        m_code += "}\n"
        return m_code
    
    def process_language_to_math(self, text: str) -> Dict[str, Any]:
        """
        Verarbeitet natürliche Sprache zu mathematischen Operationen.
        
        Args:
            text: Natürlichsprachiger Text
            
        Returns:
            Ergebnis der Operation
        """
        # Parse natürliche Sprache
        operations = self.parse_natural_language(text)
        
        # Führe Operation aus
        result = self.execute_math_operation(operations)
        
        return result
    
    def get_operation_history(self) -> List[Dict[str, Any]]:
        """
        Gibt die Historie der ausgeführten Operationen zurück.
        
        Returns:
            Liste mit Operationshistorie
        """
        return self.operation_history
    
    def clear_tensor_cache(self):
        """Leert den Tensor-Cache"""
        with self.lock:
            self.tensor_cache.clear()
            logger.info("Tensor-Cache geleert")
    
    def shutdown(self):
        """Fährt die LinguaMathBridge herunter"""
        with self.lock:
            # Bereinige Ressourcen
            self.tensor_cache.clear()
            self.operation_history.clear()
            
            # Setze Engines zurück
            self.m_code_engine = None
            self.t_math_engine = None
            
            self.initialized = False
            logger.info("LinguaMathBridge heruntergefahren")
