#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Q-LOGIK T-Mathematics Integration

Integrationsschicht zwischen Q-LOGIK und T-Mathematics Engine.
Ermöglicht die Verwendung von Tensor-Operationen in Q-LOGIK und
die Steuerung der T-Mathematics Engine durch Q-LOGIK-Entscheidungen.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import json
import logging
import math
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import os
import sys

# Importiere NumPy (erforderlich)
import numpy as np

# Importiere PyTorch (erforderlich)
import torch

# Importiere Q-LOGIK-Komponenten
from miso.logic.qlogik_engine import (
    BayesianDecisionCore,
    FuzzyLogicUnit,
    SymbolMap,
    ConflictResolver,
    simple_emotion_weight,
    simple_priority_mapping,
    advanced_qlogik_decision
)

# Importiere T-Mathematics-Komponenten (erforderlich)
from miso.math.t_mathematics.engine import TMathEngine, TMathConfig

# Logger einrichten
logger = logging.getLogger("MISO.Logic.Q-LOGIK.TMathIntegration")


class TensorOperationSelector:
    """
    Tensor-Operationsauswahl
    
    Wählt basierend auf Q-LOGIK-Entscheidungen die optimale
    Tensor-Operation und das Backend aus.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den Tensor-Operationsauswähler
        
        Args:
            config: Konfigurationsobjekt für den Auswähler
        """
        self.config = config or {}
        
        # Standardkonfiguration
        self.risk_threshold = self.config.get("risk_threshold", 0.6)
        self.auto_select_backend = self.config.get("auto_select_backend", True)
        self.preferred_backend = self.config.get("preferred_backend", "MLXTensor")
        
        # Verfügbare Backends
        self.available_backends = ["MLXTensor", "TorchTensor", "NumpyTensor"]
        
        logger.info("TensorOperationSelector initialisiert")
        
    def select_backend(self, operation_context: Dict[str, Any]) -> str:
        """
        Wählt das optimale Backend für eine Tensor-Operation
        
        Args:
            operation_context: Kontext der Operation
            
        Returns:
            Name des ausgewählten Backends
        """
        if not self.auto_select_backend:
            return self.preferred_backend
            
        # Extrahiere relevante Faktoren
        tensor_size = operation_context.get("tensor_size", 0)
        operation_type = operation_context.get("operation_type", "unknown")
        memory_available = operation_context.get("memory_available", 1.0)
        computation_complexity = operation_context.get("computation_complexity", 0.5)
        
        # Entscheidungsfaktoren
        backend_factors = {
            "MLXTensor": {
                "size_threshold": 1000000,  # Gut für große Tensoren
                "preferred_ops": ["matmul", "conv", "attention"],
                "memory_efficiency": 0.7,
                "computation_efficiency": 0.9,
                "apple_silicon_bonus": 0.3  # Bonus für Apple Silicon
            },
            "TorchTensor": {
                "size_threshold": 500000,  # Gut für mittlere Tensoren
                "preferred_ops": ["svd", "fft", "einsum"],
                "memory_efficiency": 0.8,
                "computation_efficiency": 0.8,
                "gpu_bonus": 0.2  # Bonus für GPU-Verfügbarkeit
            },
            "NumpyTensor": {
                "size_threshold": 100000,  # Gut für kleine Tensoren
                "preferred_ops": ["indexing", "reshape", "reduce"],
                "memory_efficiency": 0.9,
                "computation_efficiency": 0.6,
                "cpu_bonus": 0.1  # Bonus für CPU-Operationen
            }
        }
        
        # Berechne Scores für jedes Backend
        backend_scores = {}
        
        for backend, factors in backend_factors.items():
            # Basiswert
            score = 0.5
            
            # Größenanpassung
            if tensor_size > factors["size_threshold"]:
                score += 0.2
            else:
                score -= 0.1
                
            # Operationstyp
            if operation_type in factors["preferred_ops"]:
                score += 0.2
                
            # Speichereffizienz
            score += (memory_available * factors["memory_efficiency"])
            
            # Berechnungseffizienz (invertiert mit Komplexität)
            score += ((1 - computation_complexity) * factors["computation_efficiency"])
            
            # Hardware-Boni
            if "apple_silicon_bonus" in factors and self._is_apple_silicon():
                score += factors["apple_silicon_bonus"]
                
            if "gpu_bonus" in factors and self._has_gpu():
                score += factors["gpu_bonus"]
                
            if "cpu_bonus" in factors:
                score += factors["cpu_bonus"]
                
            backend_scores[backend] = score
            
        # Wähle das Backend mit dem höchsten Score
        selected_backend = max(backend_scores.items(), key=lambda x: x[1])[0]
        
        logger.info(f"Backend-Auswahl für {operation_type}: {selected_backend} (Score: {backend_scores[selected_backend]:.2f})")
        
        return selected_backend
        
    def _is_apple_silicon(self) -> bool:
        """Prüft, ob Apple Silicon verwendet wird"""
        import platform
        return platform.processor() == 'arm' and platform.system() == 'Darwin'
        
    def _has_gpu(self) -> bool:
        """Prüft, ob eine GPU verfügbar ist"""
        try:
            return torch.cuda.is_available() or hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except:
            return False


class QLOGIKTMathematicsIntegration:
    """
    Q-LOGIK T-Mathematics Integration
    
    Integrationsschicht zwischen Q-LOGIK und T-Mathematics Engine.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialisiert die Integrationsschicht
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
        """
        # Lade Konfiguration
        self.config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            except Exception as e:
                logger.error(f"Fehler beim Laden der Konfiguration: {str(e)}")
                
        # Initialisiere Q-LOGIK-Komponenten
        self.bayesian = BayesianDecisionCore()
        self.fuzzylogic = FuzzyLogicUnit()
        self.symbolmap = SymbolMap()
        self.conflict_resolver = ConflictResolver()
        
        # Verwende die vereinfachten Funktionen statt der Klassen
        self.emotion_weight_func = simple_emotion_weight
        self.priority_mapping_func = simple_priority_mapping
        
        # Initialisiere T-Mathematics-Engine
        t_math_config = TMathConfig(
            precision=self.config.get("precision", "mixed"),
            device=self.config.get("device", "auto"),
            optimize_for_rdna=self.config.get("optimize_for_amd", True),
            optimize_for_apple_silicon=self.config.get("optimize_for_apple", True)
        )
        self.t_math_engine = TMathEngine(config=t_math_config)
        
        # Initialisiere Tensor-Operationsauswähler
        tensor_ops_config = self.config.get("tensor_operations", {})
        self.tensor_selector = TensorOperationSelector(config=tensor_ops_config)
        
        logger.info("Q-LOGIK T-Mathematics Integration initialisiert")
        
    def decide_tensor_operation(self, operation_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Entscheidet über die Durchführung einer Tensor-Operation
        
        Args:
            operation_context: Kontext der Operation
            
        Returns:
            Entscheidungsergebnis
        """
        # Bereite Q-LOGIK-Kontext vor
        qlogik_context = {
            "data": {
                "hypothesis": "tensor_operation_feasible",
                "evidence": {
                    "memory_available": {
                        "value": operation_context.get("memory_available", 0.8),
                        "weight": 0.5
                    },
                    "computation_complexity": {
                        "value": operation_context.get("computation_complexity", 0.5),
                        "weight": 0.5
                    }
                }
            },
            "signal": {
                "system_load": operation_context.get("system_load", 0.5),
                "operation_priority": operation_context.get("priority", 0.5)
            },
            "risk": operation_context.get("risk", 0.3),
            "benefit": operation_context.get("benefit", 0.7),
            "urgency": operation_context.get("urgency", 0.5)
        }
        
        # Führe Q-LOGIK-Entscheidung durch
        decision_result = advanced_qlogik_decision(qlogik_context)
        
        # Wähle Backend basierend auf der Entscheidung
        if decision_result["decision"] == "JA":
            selected_backend = self.tensor_selector.select_backend(operation_context)
            
            # Füge Backend-Informationen hinzu
            decision_result["tensor_operation"] = {
                "execute": True,
                "backend": selected_backend,
                "priority": decision_result["priority"]["level"]
            }
        else:
            decision_result["tensor_operation"] = {
                "execute": False,
                "reason": "Operation nicht empfohlen",
                "alternative": "Verwende vereinfachte Operation oder verschiebe"
            }
            
        return decision_result
        
    def execute_tensor_operation(self, 
                               operation_type: str, 
                               tensors: List[Any], 
                               operation_context: Dict[str, Any] = None) -> Any:
        """
        Führt eine Tensor-Operation aus
        
        Args:
            operation_type: Typ der Operation (z.B. "matmul", "conv", "svd")
            tensors: Liste der Eingabetensoren
            operation_context: Kontext der Operation
            
        Returns:
            Ergebnis der Operation
        """
        # Standardkontext, falls nicht angegeben
        if operation_context is None:
            operation_context = {}
            
        # Füge Operationstyp hinzu
        operation_context["operation_type"] = operation_type
        
        # Schätze Tensorgröße
        tensor_size = 0
        for tensor in tensors:
            if hasattr(tensor, "numel"):
                tensor_size += tensor.numel()
            elif hasattr(tensor, "size"):
                tensor_size += np.prod(tensor.size)
            elif isinstance(tensor, np.ndarray):
                tensor_size += tensor.size
                
        operation_context["tensor_size"] = tensor_size
        
        # Entscheide über die Operation
        decision = self.decide_tensor_operation(operation_context)
        
        # Führe Operation aus, wenn empfohlen
        if decision["tensor_operation"]["execute"]:
            backend = decision["tensor_operation"]["backend"]
            
            try:
                # Führe Operation mit T-Mathematics Engine aus
                if operation_type == "matmul":
                    result = self.t_math_engine.matmul(tensors[0], tensors[1])
                elif operation_type == "conv":
                    result = self.t_math_engine.conv(tensors[0], tensors[1])
                elif operation_type == "svd":
                    result = self.t_math_engine.svd(tensors[0])
                elif operation_type == "attention":
                    result = self.t_math_engine.attention(tensors[0], tensors[1], tensors[2])
                else:
                    # Fallback für unbekannte Operationen
                    result = self._fallback_operation(operation_type, tensors, backend)
                    
                logger.info(f"Tensor-Operation {operation_type} erfolgreich mit {backend}")
                
                return {
                    "success": True,
                    "result": result,
                    "backend": backend,
                    "decision": decision
                }
                
            except Exception as e:
                logger.error(f"Fehler bei Tensor-Operation {operation_type}: {str(e)}")
                
                # Versuche Fallback
                try:
                    result = self._fallback_operation(operation_type, tensors, "NumpyTensor")
                    
                    return {
                        "success": True,
                        "result": result,
                        "backend": "NumpyTensor (Fallback)",
                        "decision": decision,
                        "warning": f"Ursprünglicher Fehler: {str(e)}"
                    }
                    
                except Exception as e2:
                    return {
                        "success": False,
                        "error": f"Operation fehlgeschlagen: {str(e2)}",
                        "decision": decision
                    }
        else:
            # Operation nicht empfohlen
            return {
                "success": False,
                "error": "Operation nicht empfohlen",
                "reason": decision["tensor_operation"]["reason"],
                "alternative": decision["tensor_operation"]["alternative"],
                "decision": decision
            }
            
    def _fallback_operation(self, operation_type: str, tensors: List[Any], backend: str) -> Any:
        """
        Führt eine Fallback-Operation aus
        
        Args:
            operation_type: Typ der Operation
            tensors: Liste der Eingabetensoren
            backend: Zu verwendendes Backend
            
        Returns:
            Ergebnis der Operation
        """
        # Konvertiere Tensoren zu NumPy, falls nötig
        numpy_tensors = []
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                numpy_tensors.append(tensor.detach().cpu().numpy())
            elif hasattr(tensor, "numpy"):
                numpy_tensors.append(tensor.numpy())
            else:
                numpy_tensors.append(tensor)
                
        # Führe Operation mit NumPy aus
        if operation_type == "matmul":
            return np.matmul(numpy_tensors[0], numpy_tensors[1])
        elif operation_type == "conv":
            # Einfache Faltung mit NumPy
            from scipy import signal
            return signal.convolve2d(numpy_tensors[0], numpy_tensors[1], mode='valid')
        elif operation_type == "svd":
            return np.linalg.svd(numpy_tensors[0])
        elif operation_type == "attention":
            # Einfache Attention-Implementierung
            q, k, v = numpy_tensors
            scores = np.matmul(q, k.transpose(-2, -1))
            scores = scores / np.sqrt(k.shape[-1])
            weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
            return np.matmul(weights, v)
        else:
            raise ValueError(f"Unbekannte Operation: {operation_type}")


class MLINGUATMathIntegration:
    """
    M-LINGUA T-Mathematics Integration
    
    Ermöglicht die Steuerung der T-Mathematics Engine durch
    natürliche Sprache über das M-LINGUA Interface.
    """
    
    def __init__(self, qlogik_tmath: QLOGIKTMathematicsIntegration = None):
        """
        Initialisiert die M-LINGUA T-Mathematics Integration
        
        Args:
            qlogik_tmath: Q-LOGIK T-Mathematics Integration
        """
        self.qlogik_tmath = qlogik_tmath or QLOGIKTMathematicsIntegration()
        
        # Sprachbefehle und ihre Zuordnungen
        self.command_mappings = {
            "multipliziere": "matmul",
            "falte": "conv",
            "zerlege": "svd",
            "berechne aufmerksamkeit": "attention",
            "berechne": "compute"
        }
        
        # Kontextparameter und ihre Standardwerte
        self.default_context = {
            "memory_available": 0.8,
            "computation_complexity": 0.5,
            "system_load": 0.3,
            "priority": 0.5,
            "risk": 0.2,
            "benefit": 0.7,
            "urgency": 0.5
        }
        
        logger.info("M-LINGUA T-Mathematics Integration initialisiert")
        
    def process_natural_language(self, text: str, tensors: List[Any] = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Verarbeitet natürlichsprachliche Anfragen für Tensor-Operationen
        
        Args:
            text: Natürlichsprachlicher Text
            tensors: Liste der Eingabetensoren (optional)
            context: Kontextinformationen (optional)
            
        Returns:
            Verarbeitungsergebnis
        """
        # Standardkontext, falls nicht angegeben
        operation_context = self.default_context.copy()
        if context:
            operation_context.update(context)
            
        # Bestimme Operationstyp aus Text
        operation_type = self._extract_operation_type(text)
        
        if not operation_type:
            return {
                "success": False,
                "error": "Konnte keine unterstützte Operation im Text erkennen",
                "text": text
            }
            
        # Extrahiere zusätzliche Kontextinformationen aus Text
        context_updates = self._extract_context_from_text(text)
        operation_context.update(context_updates)
        
        # Führe Operation aus, wenn Tensoren vorhanden sind
        if tensors:
            return self.qlogik_tmath.execute_tensor_operation(
                operation_type=operation_type,
                tensors=tensors,
                operation_context=operation_context
            )
        else:
            # Nur Entscheidung ohne Ausführung
            operation_context["operation_type"] = operation_type
            decision = self.qlogik_tmath.decide_tensor_operation(operation_context)
            
            return {
                "success": True,
                "operation_type": operation_type,
                "decision": decision,
                "context": operation_context,
                "note": "Keine Tensoren zur Ausführung bereitgestellt"
            }
            
    def _extract_operation_type(self, text: str) -> Optional[str]:
        """
        Extrahiert den Operationstyp aus natürlichsprachlichem Text
        
        Args:
            text: Natürlichsprachlicher Text
            
        Returns:
            Operationstyp oder None, wenn nicht erkannt
        """
        text = text.lower()
        
        for command, operation in self.command_mappings.items():
            if command in text:
                return operation
                
        return None
        
    def _extract_context_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extrahiert Kontextinformationen aus natürlichsprachlichem Text
        
        Args:
            text: Natürlichsprachlicher Text
            
        Returns:
            Extrahierte Kontextinformationen
        """
        context = {}
        text = text.lower()
        
        # Dringlichkeitsanalyse
        urgency_keywords = {
            "sofort": 0.9,
            "dringend": 0.8,
            "schnell": 0.7,
            "bald": 0.6,
            "irgendwann": 0.3,
            "wenn möglich": 0.4
        }
        
        for keyword, value in urgency_keywords.items():
            if keyword in text:
                context["urgency"] = value
                break
                
        # Prioritätsanalyse
        priority_keywords = {
            "höchste priorität": 0.9,
            "hohe priorität": 0.8,
            "wichtig": 0.7,
            "normal": 0.5,
            "niedrige priorität": 0.3,
            "unwichtig": 0.2
        }
        
        for keyword, value in priority_keywords.items():
            if keyword in text:
                context["priority"] = value
                break
                
        # Komplexitätsanalyse
        if "komplex" in text:
            context["computation_complexity"] = 0.8
        elif "einfach" in text:
            context["computation_complexity"] = 0.3
            
        # Risikoanalyse
        if "sicher" in text:
            context["risk"] = 0.2
        elif "riskant" in text or "unsicher" in text:
            context["risk"] = 0.7
            
        return context


# Beispiel für die Verwendung
if __name__ == "__main__":
    # Konfiguriere Logging
    logging.basicConfig(level=logging.INFO)
    
    # Erstelle Integration
    integration = QLOGIKTMathematicsIntegration()
    
    # Beispiel für Tensor-Operation
    import numpy as np
    
    # Erstelle Beispieltensoren
    tensor1 = np.random.rand(10, 20)
    tensor2 = np.random.rand(20, 15)
    
    # Operationskontext
    context = {
        "memory_available": 0.9,
        "computation_complexity": 0.4,
        "system_load": 0.3,
        "priority": 0.7,
        "risk": 0.2,
        "benefit": 0.8,
        "urgency": 0.6
    }
    
    # Führe Operation aus
    result = integration.execute_tensor_operation(
        operation_type="matmul",
        tensors=[tensor1, tensor2],
        operation_context=context
    )
    
    print("Ergebnis:", result)
    
    # M-LINGUA-Integration
    mlingua = MLINGUATMathIntegration(integration)
    
    # Natürlichsprachliche Anfrage
    nl_result = mlingua.process_natural_language(
        text="Multipliziere diese Matrizen mit hoher Priorität",
        tensors=[tensor1, tensor2]
    )
    
    print("M-LINGUA-Ergebnis:", nl_result)
