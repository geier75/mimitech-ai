#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate AGI - M-LINGUA-T-Mathematics Bridge

Die zentrale Komponente für die Integration zwischen dem M-LINGUA Interface 
(natürliche Sprache) und der T-Mathematics Engine (Tensor-Operationen).
Diese Bridge ermöglicht es, Tensor-Operationen in natürlicher Sprache zu steuern.
"""

import os
import sys
import logging
import json
import time
from typing import Dict, List, Any, Tuple, Optional, Union, Set
from enum import Enum
from pathlib import Path

# Füge Module-Verzeichnis zum Pfad hinzu
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Importiere die entwickelten Komponenten
try:
    from integration.multilingual_parser import MultilingualParser, ParserMode
    from integration.operation_mapper import OperationMapper, OperationType
    from integration.backend_optimizer import BackendOptimizer, Backend
    from integration.result_handler import ResultHandler, OperationStatus
except ImportError as e:
    raise ImportError(f"Fehler beim Importieren der erforderlichen Komponenten: {e}")

# Konfiguriere Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MISO.Bridge")


class MLinguaTMathBridge:
    """
    Die Bridge zwischen dem M-LINGUA Interface und der T-Mathematics Engine.
    
    Diese Klasse:
    1. Nimmt natürlichsprachliche Eingaben in verschiedenen Sprachen entgegen
    2. Parst diese mit dem MultilingualParser
    3. Extrahiert Tensor-Operationen mit dem OperationMapper
    4. Wählt das optimale Backend mit dem BackendOptimizer
    5. Führt die Operation aus und liefert formatierte Ergebnisse zurück
    """
    
    def __init__(self, 
                 language: str = "en", 
                 preferred_backend: Optional[str] = None,
                 workspace_dir: Optional[str] = None,
                 tensor_registry: Optional[Dict[str, Any]] = None):
        """
        Initialisiert die MLinguaTMathBridge.
        
        Args:
            language: Standardsprache (en, de, fr, es)
            preferred_backend: Bevorzugtes Backend (mlx, torch, tf, jax, numpy)
            workspace_dir: Verzeichnis für Arbeitsdateien und Logs
            tensor_registry: Dictionary zum Speichern von Tensor-Objekten
        """
        self.language = language.lower()
        
        # Speichere von Benutzern definierte Tensoren
        self.tensor_registry = tensor_registry or {}
        
        # Initialisiere die benötigten Komponenten
        logger.info(f"Initialisiere M-LINGUA-T-Mathematics Bridge...")
        
        # Sprachverarbeitung
        self.parser = MultilingualParser(language=self.language, context_memory_size=10)
        logger.info(f"Multilingual Parser initialisiert: Sprache={self.language}")
        
        # Backend-Optimierung
        self.backend_optimizer = BackendOptimizer(preferred_backend=preferred_backend, enable_benchmarking=True)
        available_backends = [b.value for b in self.backend_optimizer.get_available_backends()]
        logger.info(f"Backend Optimizer initialisiert: Verfügbare Backends={available_backends}")
        
        # Ergebnisformatierung und Fehlerbehandlung
        self.result_handler = ResultHandler(language=self.language, verbose=True)
        logger.info(f"Result Handler initialisiert: Sprache={self.language}")
        
        # Arbeitsverzeichnis
        self.workspace_dir = workspace_dir
        if self.workspace_dir:
            os.makedirs(self.workspace_dir, exist_ok=True)
            logger.info(f"Arbeitsverzeichnis: {self.workspace_dir}")
        
        logger.info("M-LINGUA-T-Mathematics Bridge erfolgreich initialisiert")
    
    def process_input(self, text: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Verarbeitet eine natürlichsprachliche Eingabe.
        
        Args:
            text: Natürlichsprachliche Eingabe
            language: Sprachcode (optional, sonst wird Standardsprache verwendet)
            
        Returns:
            Verarbeitungsergebnis als Dictionary
        """
        language = language or self.language
        
        # Starte Zeitmessung
        start_time = time.time()
        
        try:
            # Schritt 1: Parse die Eingabe
            logger.info(f"Verarbeite Eingabe: '{text}' (Sprache: {language})")
            parsed_result = self.parser.parse(text, language)
            
            # Schritt 2: Löse Kontextreferenzen auf
            if parsed_result["has_context_references"]:
                logger.info("Löse Kontextreferenzen auf...")
                parsed_result = self.parser.resolve_context_references(parsed_result)
            
            # Schritt 3: Ermittle die auszuführende Operation
            operation_type = OperationType(parsed_result["operation"]["type"])
            operation_params = parsed_result["operation"]["params"]
            
            # Wenn keine Operation erkannt wurde, gib eine Fehlermeldung zurück
            if operation_type == OperationType.UNKNOWN:
                return self.result_handler.format_error(
                    error=ValueError("Keine erkennbare Tensor-Operation in der Eingabe gefunden"),
                    operation_name="unknown",
                    status=OperationStatus.INVALID_INPUT,
                    input_data={"text": text, "language": language}
                )
            
            # Schritt 4: Wähle das optimale Backend
            backend = self.backend_optimizer.get_optimal_backend(operation_type.value)
            logger.info(f"Ausgewähltes Backend: {backend.value}")
            
            # Schritt 5: Führe die Operation aus
            result = self._execute_operation(operation_type, operation_params, backend)
            
            # Schritt 6: Formatiere das Ergebnis
            execution_time = time.time() - start_time
            formatted_result = self.result_handler.format_result(
                result=result,
                operation_name=operation_type.value,
                status=OperationStatus.SUCCESS,
                execution_time=execution_time,
                backend=backend.value,
                additional_info={"input_text": text, "language": language}
            )
            
            # Logge das Ergebnis
            self.result_handler.log_result(formatted_result)
            
            return formatted_result
        
        except Exception as e:
            # Bei Fehlern: Formatiere und logge den Fehler
            execution_time = time.time() - start_time
            error_result = self.result_handler.format_error(
                error=e,
                operation_name=getattr(operation_type, "value", "unknown_operation") if 'operation_type' in locals() else "unknown_operation",
                status=OperationStatus.FAILURE,
                input_data={"text": text, "language": language, "execution_time": execution_time}
            )
            
            # Logge den Fehler
            self.result_handler.log_result(error_result)
            
            return error_result
    
    def _execute_operation(self, 
                         operation_type: OperationType, 
                         params: Dict[str, Any],
                         backend: Backend) -> Any:
        """
        Führt eine Tensor-Operation aus.
        
        Args:
            operation_type: Auszuführende Operation
            params: Parameter für die Operation
            backend: Zu verwendendes Backend
            
        Returns:
            Ergebnis der Operation
        """
        # Hole Backend-Instanz
        backend_instance = self.backend_optimizer.get_backend_instance(backend)
        
        # Prüfe, ob die benötigten Tensor-Objekte im Registry vorhanden sind
        if operation_type in [OperationType.MATRIX_MULTIPLICATION, OperationType.TENSOR_ADDITION, OperationType.TENSOR_SUBTRACTION]:
            tensor1_name = params.get("tensor1_name", params.get("tensor1", None))
            tensor2_name = params.get("tensor2_name", params.get("tensor2", None))
            
            if not (tensor1_name and tensor2_name):
                raise ValueError(f"Für {operation_type.value} werden zwei Tensoren benötigt")
            
            # Prüfe, ob die Tensoren existieren, oder erstelle Dummy-Tensoren für Tests
            if tensor1_name in self.tensor_registry:
                tensor1 = self.tensor_registry[tensor1_name]
            else:
                logger.warning(f"Tensor '{tensor1_name}' nicht im Registry, erstelle Dummy")
                tensor1 = self._create_dummy_tensor(backend_instance, name=tensor1_name)
                self.tensor_registry[tensor1_name] = tensor1
            
            if tensor2_name in self.tensor_registry:
                tensor2 = self.tensor_registry[tensor2_name]
            else:
                logger.warning(f"Tensor '{tensor2_name}' nicht im Registry, erstelle Dummy")
                tensor2 = self._create_dummy_tensor(backend_instance, name=tensor2_name)
                self.tensor_registry[tensor2_name] = tensor2
            
            # Führe die entsprechende Operation aus
            if operation_type == OperationType.MATRIX_MULTIPLICATION:
                result = self._matrix_multiply(tensor1, tensor2, backend_instance)
            elif operation_type == OperationType.TENSOR_ADDITION:
                result = self._tensor_add(tensor1, tensor2, backend_instance)
            elif operation_type == OperationType.TENSOR_SUBTRACTION:
                result = self._tensor_subtract(tensor1, tensor2, backend_instance)
        
        elif operation_type == OperationType.MATRIX_TRANSPOSITION:
            tensor_name = params.get("tensor_name", params.get("tensor", None))
            
            if not tensor_name:
                raise ValueError(f"Für {operation_type.value} wird ein Tensor benötigt")
            
            # Prüfe, ob der Tensor existiert, oder erstelle Dummy-Tensor für Tests
            if tensor_name in self.tensor_registry:
                tensor = self.tensor_registry[tensor_name]
            else:
                logger.warning(f"Tensor '{tensor_name}' nicht im Registry, erstelle Dummy")
                tensor = self._create_dummy_tensor(backend_instance, name=tensor_name)
                self.tensor_registry[tensor_name] = tensor
            
            # Führe die Transposition aus
            result = self._matrix_transpose(tensor, backend_instance)
        
        else:
            raise ValueError(f"Operation nicht implementiert: {operation_type.value}")
        
        # Speichere das Ergebnis im Registry für spätere Referenzen
        self.tensor_registry["last_result"] = result
        
        return result
    
    def _create_dummy_tensor(self, backend, name: str, shape: Tuple[int, ...] = (3, 3)):
        """
        Erstellt einen Dummy-Tensor für Tests.
        
        Args:
            backend: Backend-Instanz
            name: Name des Tensors
            shape: Form des Tensors
            
        Returns:
            Dummy-Tensor
        """
        # Erzeuge einen Tensor mit Werten von 1 bis n
        if hasattr(backend, "ones"):
            # NumPy, PyTorch, TensorFlow, JAX
            return backend.ones(shape)
        elif hasattr(backend, "array") and hasattr(backend, "ones"):
            # MLX
            return backend.array(backend.ones(shape))
        else:
            # Fallback
            import numpy as np
            return np.ones(shape)
    
    def _matrix_multiply(self, tensor1, tensor2, backend):
        """Führt Matrix-Multiplikation aus."""
        if hasattr(backend, "matmul"):
            return backend.matmul(tensor1, tensor2)
        elif hasattr(backend, "dot"):
            return backend.dot(tensor1, tensor2)
        else:
            raise NotImplementedError(f"Matrix-Multiplikation nicht implementiert für {backend}")
    
    def _tensor_add(self, tensor1, tensor2, backend):
        """Führt Tensor-Addition aus."""
        return tensor1 + tensor2
    
    def _tensor_subtract(self, tensor1, tensor2, backend):
        """Führt Tensor-Subtraktion aus."""
        return tensor1 - tensor2
    
    def _matrix_transpose(self, tensor, backend):
        """Führt Matrix-Transposition aus."""
        if hasattr(backend, "transpose"):
            return backend.transpose(tensor)
        elif hasattr(tensor, "transpose"):
            return tensor.transpose()
        else:
            raise NotImplementedError(f"Matrix-Transposition nicht implementiert für {backend}")
    
    def register_tensor(self, name: str, tensor: Any) -> None:
        """
        Registriert einen Tensor für spätere Verwendung.
        
        Args:
            name: Name des Tensors
            tensor: Tensor-Objekt
        """
        self.tensor_registry[name] = tensor
        logger.info(f"Tensor '{name}' registriert mit Form {tensor.shape if hasattr(tensor, 'shape') else 'unbekannt'}")
    
    def get_tensor(self, name: str) -> Optional[Any]:
        """
        Gibt einen registrierten Tensor zurück.
        
        Args:
            name: Name des Tensors
            
        Returns:
            Tensor-Objekt oder None, falls nicht gefunden
        """
        if name in self.tensor_registry:
            return self.tensor_registry[name]
        else:
            logger.warning(f"Tensor '{name}' nicht im Registry gefunden")
            return None
    
    def set_language(self, language: str) -> None:
        """
        Ändert die Standardsprache der Bridge.
        
        Args:
            language: Neuer Sprachcode (en, de, fr, es)
        """
        self.language = language.lower()
        self.parser.language = self.language
        self.result_handler.language = self.language
        logger.info(f"Sprache geändert auf: {self.language}")
    
    def get_supported_operations(self) -> List[str]:
        """
        Gibt eine Liste der unterstützten Operationen zurück.
        
        Returns:
            Liste von Operationsnamen
        """
        return [op.value for op in OperationType if op != OperationType.UNKNOWN]
    
    def get_user_friendly_message(self, result: Dict[str, Any]) -> str:
        """
        Gibt eine benutzerfreundliche Nachricht für ein Ergebnis zurück.
        
        Args:
            result: Verarbeitungsergebnis
            
        Returns:
            Benutzerfreundliche Nachricht
        """
        return self.result_handler.get_user_friendly_message(result)
    
    def clear_context(self) -> None:
        """Löscht den Kontext des Parsers."""
        self.parser.clear_context()
        logger.info("Kontext gelöscht")


# Testfunktion
def test_bridge():
    """Testet die Funktionalität der MLinguaTMathBridge."""
    logger.info("=== Testing M-LINGUA-T-Mathematics Bridge ===")
    
    # Erstelle Bridge-Instanz
    bridge = MLinguaTMathBridge(language="en")
    
    # Test-Eingaben in verschiedenen Sprachen
    test_inputs = [
        # Englisch
        "Multiply matrix A with matrix B",
        "Transpose matrix C",
        "Add tensor X and tensor Y",
        
        # Deutsch
        "Multipliziere Matrix A mit Matrix B",
        "Transponiere Matrix C",
        "Addiere Tensor X und Tensor Y",
        
        # Französisch
        "Multiplie la matrice A avec la matrice B",
        "Transpose la matrice C",
        "Ajoute le tenseur X et le tenseur Y",
        
        # Spanisch
        "Multiplica la matriz A con la matriz B",
        "Transpone la matriz C",
        "Suma el tensor X y el tensor Y"
    ]
    
    # Verarbeite jede Eingabe
    for input_text in test_inputs:
        logger.info(f"\nVerarbeite: '{input_text}'")
        
        # Bestimme die Sprache (vereinfachte Erkennung für den Test)
        language = "en"  # Standardwert
        if any(word in input_text for word in ["Matrix", "Tensor", "multipliziere", "addiere", "transponiere"]):
            language = "de"
        elif any(word in input_text for word in ["matrice", "tenseur", "multiplie", "ajoute", "transpose"]):
            language = "fr"
        elif any(word in input_text for word in ["matriz", "tensor", "multiplica", "suma", "transpone"]):
            language = "es"
        
        # Verarbeite die Eingabe
        result = bridge.process_input(input_text, language)
        
        # Zeige benutzerfreundliche Nachricht
        user_message = bridge.get_user_friendly_message(result)
        print(f"Ergebnis ({language}): {user_message}")
        
        # Bei erfolgreicher Verarbeitung zeige Details zum Ergebnis
        if result["status"] == OperationStatus.SUCCESS.value:
            if "result" in result and "shape" in result["result"]:
                shape = result["result"]["shape"]
                print(f"Form des Ergebnisses: {shape}")
    
    # Teste Kontextreferenzen
    logger.info("\n=== Testing Context References ===")
    
    # Lösche vorherigen Kontext
    bridge.clear_context()
    
    # Sequenz von Eingaben mit Kontextreferenzen
    context_test = [
        "Calculate A + B",
        "Now multiply it with C",
        "What's the transpose of the result?"
    ]
    
    for input_text in context_test:
        logger.info(f"\nVerarbeite: '{input_text}'")
        result = bridge.process_input(input_text, "en")
        user_message = bridge.get_user_friendly_message(result)
        print(f"Ergebnis: {user_message}")


if __name__ == "__main__":
    # Führe Test aus
    test_bridge()
