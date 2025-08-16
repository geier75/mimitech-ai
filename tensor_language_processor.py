#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - NEXUS-OS TensorLanguageProcessor

Diese Datei implementiert den Sprachprozessor, der natürliche Sprache in 
Tensor-Operationen übersetzt und die optimalen Backends (MLX, PyTorch, NumPy) 
automatisch auswählt.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import re
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Konfiguriere Logging
logger = logging.getLogger("MISO.nexus_os.tensor_processor")

class TensorLanguageProcessor:
    """
    Prozessor für die Übersetzung natürlicher Sprache in Tensor-Operationen
    
    Diese Klasse analysiert natürlichsprachige Anfragen und übersetzt sie in
    Tensor-Operationen, wobei das optimale Backend (MLX, PyTorch, NumPy) 
    automatisch ausgewählt wird.
    """
    
    def __init__(self):
        """Initialisiert den TensorLanguageProcessor"""
        self.initialized = False
        self.backend_capabilities = {}
        self.available_backends = []
        self.default_backend = None
        self.tensor_registry = {}
        self.operation_patterns = {}
        logger.info("TensorLanguageProcessor initialisiert")
    
    def initialize(self):
        """Initialisiert den Prozessor mit den verfügbaren Backends"""
        if self.initialized:
            logger.warning("TensorLanguageProcessor bereits initialisiert")
            return
        
        # Erkenne verfügbare Backends
        self._detect_backends()
        
        # Lade Sprachmuster für Operationen
        self._load_operation_patterns()
        
        self.initialized = True
        logger.info(f"TensorLanguageProcessor vollständig initialisiert mit Backends: {', '.join(self.available_backends)}")
    
    def _detect_backends(self):
        """Erkennt verfügbare Tensor-Backends und ihre Fähigkeiten"""
        # Prüfe auf MLX (Apple Neural Engine)
        try:
            import mlx.core
            self.available_backends.append("mlx")
            self.backend_capabilities["mlx"] = {
                "device": "ane",  # Apple Neural Engine
                "precision": ["float16", "float32"],
                "operations": ["matmul", "conv", "attention", "softmax", "norm"],
                "performance_score": 9.5,  # Bewertung von 0-10
                "optimal_for": ["transformer", "attention", "small_batch"]
            }
            logger.info("MLX Backend erkannt (Apple Neural Engine)")
        except ImportError:
            logger.debug("MLX Backend nicht verfügbar")
        
        # Prüfe auf PyTorch mit MPS
        try:
            import torch
            if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                self.available_backends.append("torch_mps")
                self.backend_capabilities["torch_mps"] = {
                    "device": "mps",  # Metal Performance Shaders
                    "precision": ["float16", "float32"],
                    "operations": ["matmul", "conv", "rnn", "transformer", "norm"],
                    "performance_score": 8.5,
                    "optimal_for": ["large_batch", "training", "complex_models"]
                }
                logger.info("PyTorch MPS Backend erkannt (Metal GPU)")
            elif torch.cuda.is_available():
                self.available_backends.append("torch_cuda")
                self.backend_capabilities["torch_cuda"] = {
                    "device": "cuda",
                    "precision": ["float16", "float32", "bfloat16"],
                    "operations": ["matmul", "conv", "rnn", "transformer", "norm"],
                    "performance_score": 9.0,
                    "optimal_for": ["large_batch", "training", "inference"]
                }
                logger.info("PyTorch CUDA Backend erkannt (NVIDIA GPU)")
            else:
                self.available_backends.append("torch_cpu")
                self.backend_capabilities["torch_cpu"] = {
                    "device": "cpu",
                    "precision": ["float32", "float64"],
                    "operations": ["matmul", "conv", "rnn", "transformer", "norm"],
                    "performance_score": 6.0,
                    "optimal_for": ["cpu_only", "debugging"]
                }
                logger.info("PyTorch CPU Backend erkannt")
        except ImportError:
            logger.debug("PyTorch Backend nicht verfügbar")
        
        # Prüfe auf NumPy (immer verfügbar)
        try:
            import numpy
            self.available_backends.append("numpy")
            self.backend_capabilities["numpy"] = {
                "device": "cpu",
                "precision": ["float32", "float64"],
                "operations": ["matmul", "linalg", "fft", "random", "stats"],
                "performance_score": 5.0,
                "optimal_for": ["cpu_only", "scientific", "stats"]
            }
            logger.info("NumPy Backend erkannt")
        except ImportError:
            logger.warning("NumPy Backend nicht verfügbar (ungewöhnlich)")
        
        # Setze Default-Backend
        if "mlx" in self.available_backends:
            self.default_backend = "mlx"
        elif "torch_mps" in self.available_backends:
            self.default_backend = "torch_mps"
        elif "torch_cuda" in self.available_backends:
            self.default_backend = "torch_cuda"
        elif "torch_cpu" in self.available_backends:
            self.default_backend = "torch_cpu"
        else:
            self.default_backend = "numpy"
        
        logger.info(f"Standard-Backend: {self.default_backend}")
    
    def _load_operation_patterns(self):
        """Lädt Sprachmuster für Tensor-Operationen"""
        # Definiere Muster für verschiedene Sprachen (Deutsch, Englisch)
        self.operation_patterns = {
            # Matrixmultiplikation
            "matmul": [
                r"(?i)berechne\s+(?:die\s+)?matrix(?:en)?multiplikation\s+von\s+(.*)\s+und\s+(.*)",
                r"(?i)multipliziere\s+(?:die\s+)?matri(?:x|zen)\s+(.*)\s+und\s+(.*)",
                r"(?i)calculate\s+(?:the\s+)?matrix\s+multiplication\s+of\s+(.*)\s+and\s+(.*)",
                r"(?i)multiply\s+(?:the\s+)?matri(?:x|ces)\s+(.*)\s+and\s+(.*)"
            ],
            # Tensor erstellen
            "create_tensor": [
                r"(?i)erstelle\s+(?:einen|eine|ein)?\s*tensor\s+mit\s+(?:den\s+)?werten\s+(.*)",
                r"(?i)erzeuge\s+(?:einen|eine|ein)?\s*tensor\s+mit\s+(?:den\s+)?werten\s+(.*)",
                r"(?i)create\s+a\s+tensor\s+with\s+(?:the\s+)?values\s+(.*)",
                r"(?i)generate\s+a\s+tensor\s+with\s+(?:the\s+)?values\s+(.*)"
            ],
            # Transponieren
            "transpose": [
                r"(?i)transponiere\s+(?:den\s+)?tensor\s+(.*)",
                r"(?i)transpose\s+(?:the\s+)?tensor\s+(.*)"
            ],
            # Softmax
            "softmax": [
                r"(?i)wende\s+(?:eine\s+)?(?:die\s+)?softmax(?:-funktion)?\s+auf\s+(.*)\s+an",
                r"(?i)apply\s+softmax\s+(?:function\s+)?to\s+(.*)"
            ],
            # Summe
            "sum": [
                r"(?i)berechne\s+(?:die\s+)?summe\s+von\s+(.*)",
                r"(?i)summiere\s+(?:den\s+)?tensor\s+(.*)",
                r"(?i)calculate\s+(?:the\s+)?sum\s+of\s+(.*)",
                r"(?i)sum\s+(?:the\s+)?tensor\s+(.*)"
            ],
            # Mittelwert
            "mean": [
                r"(?i)berechne\s+(?:den\s+)?mittelwert\s+von\s+(.*)",
                r"(?i)calculate\s+(?:the\s+)?mean\s+of\s+(.*)",
                r"(?i)average\s+(?:the\s+)?tensor\s+(.*)"
            ],
            # Normalisierung
            "normalize": [
                r"(?i)normalisiere\s+(?:den\s+)?tensor\s+(.*)",
                r"(?i)normalize\s+(?:the\s+)?tensor\s+(.*)"
            ],
            # Konkatenation
            "concat": [
                r"(?i)konkateniere\s+(?:die\s+)?tensoren\s+(.*)\s+und\s+(.*)",
                r"(?i)concatenate\s+(?:the\s+)?tensors\s+(.*)\s+and\s+(.*)"
            ],
            # Elementweise Multiplikation
            "multiply": [
                r"(?i)multipliziere\s+(?:die\s+)?tensoren\s+(.*)\s+und\s+(.*)\s+elementweise",
                r"(?i)multiply\s+(?:the\s+)?tensors\s+(.*)\s+and\s+(.*)\s+elementwise"
            ]
        }
    
    def select_optimal_backend(self, operation: str, tensor_shapes: List[Any] = None) -> str:
        """
        Wählt das optimale Backend für eine Operation aus.
        
        Args:
            operation: Art der Operation
            tensor_shapes: Formen der beteiligten Tensoren
            
        Returns:
            Name des optimalen Backends
        """
        if not self.initialized:
            self.initialize()
        
        # Standardfall: Verwende das Default-Backend
        if not tensor_shapes or operation not in ["matmul", "conv", "attention"]:
            return self.default_backend
        
        # Spezifische Optimierungen basierend auf Operation und Tensorgröße
        if operation == "matmul":
            # Für große Matrizen: PyTorch mit MPS/CUDA
            if any(np.prod(shape) > 10000 for shape in tensor_shapes):
                if "torch_mps" in self.available_backends:
                    return "torch_mps"
                elif "torch_cuda" in self.available_backends:
                    return "torch_cuda"
            # Für kleinere Matrizen: MLX (wenn verfügbar)
            elif "mlx" in self.available_backends:
                return "mlx"
        
        elif operation == "attention":
            # Attention-Operationen sind optimal auf MLX
            if "mlx" in self.available_backends:
                return "mlx"
            # Fallback auf PyTorch
            elif "torch_mps" in self.available_backends:
                return "torch_mps"
            elif "torch_cuda" in self.available_backends:
                return "torch_cuda"
        
        # Fallback auf Default-Backend
        return self.default_backend
    
    def parse_natural_language(self, text: str) -> Dict[str, Any]:
        """
        Analysiert natürliche Sprache und extrahiert Tensor-Operationen.
        
        Args:
            text: Natürlichsprachiger Text
            
        Returns:
            Dictionary mit extrahierten Operationen und Parametern
        """
        if not self.initialized:
            self.initialize()
        
        result = {
            "operation": None,
            "parameters": {},
            "backend": None,
            "raw_text": text
        }
        
        # Prüfe auf explizite Backend-Angabe
        backend_match = re.search(r"(?i)verwende\s+(?:das\s+)?backend\s+([a-z_]+)", text)
        if backend_match and backend_match.group(1).lower() in self.available_backends:
            result["backend"] = backend_match.group(1).lower()
        
        # Durchsuche alle Operationsmuster
        for operation, patterns in self.operation_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    result["operation"] = operation
                    
                    # Extrahiere Parameter basierend auf Operation
                    if operation == "create_tensor":
                        values_str = match.group(1)
                        try:
                            values = self._parse_tensor_values(values_str)
                            result["parameters"]["values"] = values
                        except Exception as e:
                            logger.error(f"Fehler beim Parsen der Tensorwerte: {e}")
                            result["error"] = str(e)
                    
                    elif operation in ["matmul", "concat", "multiply"]:
                        result["parameters"]["tensor1"] = match.group(1).strip()
                        result["parameters"]["tensor2"] = match.group(2).strip()
                    
                    elif operation in ["transpose", "softmax", "sum", "mean", "normalize"]:
                        result["parameters"]["tensor"] = match.group(1).strip()
                    
                    # Wähle optimales Backend, falls nicht explizit angegeben
                    if not result["backend"]:
                        result["backend"] = self.select_optimal_backend(operation)
                    
                    return result
        
        # Keine bekannte Operation gefunden
        result["error"] = "Keine bekannte Tensor-Operation in der Anfrage erkannt"
        return result
    
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
    
    def create_tensor_operation(self, operation_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Erstellt eine Tensor-Operation basierend auf der analysierten Sprache.
        
        Args:
            operation_info: Informationen zur Operation
            
        Returns:
            Ergebnis der Operation
        """
        if not self.initialized:
            self.initialize()
        
        result = {
            "success": False,
            "result": None,
            "backend": operation_info.get("backend", self.default_backend),
            "operation": operation_info.get("operation"),
            "error": None
        }
        
        try:
            # Hole Operation und Parameter
            operation = operation_info.get("operation")
            parameters = operation_info.get("parameters", {})
            backend = result["backend"]
            
            # Erstelle Tensor-Operation
            if operation == "create_tensor":
                values = parameters.get("values", [])
                result["result"] = self._create_tensor(values, backend)
                result["success"] = True
            
            elif operation == "matmul":
                tensor1_name = parameters.get("tensor1", "")
                tensor2_name = parameters.get("tensor2", "")
                
                # Hole Tensoren aus Registry
                tensor1 = self.tensor_registry.get(tensor1_name)
                tensor2 = self.tensor_registry.get(tensor2_name)
                
                if tensor1 is None or tensor2 is None:
                    raise ValueError(f"Tensor nicht gefunden: {tensor1_name if tensor1 is None else tensor2_name}")
                
                result["result"] = self._matrix_multiply(tensor1, tensor2, backend)
                result["success"] = True
            
            # Weitere Operationen hier implementieren...
            
            else:
                result["error"] = f"Operation nicht implementiert: {operation}"
        
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Fehler bei der Ausführung der Tensor-Operation: {e}")
        
        return result
    
    def _create_tensor(self, values: List, backend: str) -> Any:
        """
        Erstellt einen Tensor mit dem angegebenen Backend.
        
        Args:
            values: Werte für den Tensor
            backend: Zu verwendendes Backend
            
        Returns:
            Tensor-Objekt
        """
        # Implementierung für verschiedene Backends
        if backend == "mlx":
            import mlx.core
            return mlx.core.array(values)
        
        elif backend.startswith("torch"):
            import torch
            device = "mps" if backend == "torch_mps" else "cuda" if backend == "torch_cuda" else "cpu"
            return torch.tensor(values, device=device)
        
        elif backend == "numpy":
            import numpy as np
            return np.array(values)
        
        else:
            raise ValueError(f"Unbekanntes Backend: {backend}")
    
    def _matrix_multiply(self, tensor1: Any, tensor2: Any, backend: str) -> Any:
        """
        Führt eine Matrixmultiplikation mit dem angegebenen Backend durch.
        
        Args:
            tensor1: Erster Tensor
            tensor2: Zweiter Tensor
            backend: Zu verwendendes Backend
            
        Returns:
            Ergebnis der Matrixmultiplikation
        """
        # Implementierung für verschiedene Backends
        if backend == "mlx":
            import mlx.core
            return mlx.core.matmul(tensor1, tensor2)
        
        elif backend.startswith("torch"):
            import torch
            return torch.matmul(tensor1, tensor2)
        
        elif backend == "numpy":
            import numpy as np
            return np.matmul(tensor1, tensor2)
        
        else:
            raise ValueError(f"Unbekanntes Backend: {backend}")
    
    def process_language_request(self, text: str) -> Dict[str, Any]:
        """
        Verarbeitet eine natürlichsprachige Anfrage zu Tensor-Operationen.
        
        Args:
            text: Natürlichsprachiger Text
            
        Returns:
            Ergebnis der Operation
        """
        # Parse natürliche Sprache
        operation_info = self.parse_natural_language(text)
        
        # Führe Operation aus, wenn erkannt
        if operation_info.get("operation"):
            result = self.create_tensor_operation(operation_info)
        else:
            result = {
                "success": False,
                "error": operation_info.get("error", "Keine Operation erkannt"),
                "raw_text": text
            }
        
        return result
    
    def register_tensor(self, name: str, tensor: Any):
        """
        Registriert einen Tensor unter einem Namen.
        
        Args:
            name: Name des Tensors
            tensor: Tensor-Objekt
        """
        self.tensor_registry[name] = tensor
        logger.debug(f"Tensor '{name}' registriert")
    
    def get_tensor(self, name: str) -> Any:
        """
        Gibt einen registrierten Tensor zurück.
        
        Args:
            name: Name des Tensors
            
        Returns:
            Tensor-Objekt oder None, falls nicht gefunden
        """
        return self.tensor_registry.get(name)
    
    def get_available_backends(self) -> List[str]:
        """
        Gibt die verfügbaren Backends zurück.
        
        Returns:
            Liste mit verfügbaren Backends
        """
        if not self.initialized:
            self.initialize()
        
        return self.available_backends
    
    def get_backend_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """
        Gibt die Fähigkeiten der verfügbaren Backends zurück.
        
        Returns:
            Dictionary mit Backend-Fähigkeiten
        """
        if not self.initialized:
            self.initialize()
        
        return self.backend_capabilities
