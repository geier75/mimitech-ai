#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate AGI - Ergebnisbehandlung und Fehlerverarbeitung

Diese Komponente ist verantwortlich für die Fehlerbehandlung und Ergebnispräsentation
bei der Integration von M-LINGUA (natürliche Sprache) und T-Mathematics (Tensor-Operationen).
"""

import logging
import json
import traceback
from typing import Dict, Any, Union, Optional, List, Tuple
from enum import Enum

# Konfiguriere Logger
logger = logging.getLogger("MISO.ResultHandler")

class OperationStatus(Enum):
    """Status einer Operation."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    INVALID_INPUT = "invalid_input"
    NOT_IMPLEMENTED = "not_implemented"
    BACKEND_ERROR = "backend_error"


class LanguageCode(Enum):
    """Unterstützte Sprachcodes."""
    EN = "en"  # Englisch
    DE = "de"  # Deutsch
    FR = "fr"  # Französisch
    ES = "es"  # Spanisch


class ResultHandler:
    """
    Verarbeitet Ergebnisse und Fehler bei der Ausführung von Tensor-Operationen.
    
    Diese Klasse bietet:
    1. Mehrsprachige Fehlermeldungen
    2. Formatierte Ergebnisdarstellung
    3. Detaillierte Fehlerprotokolle
    4. Benutzerfreundliche Zusammenfassungen
    """
    
    def __init__(self, language: str = "en", verbose: bool = False):
        """
        Initialisiert den ResultHandler.
        
        Args:
            language: Sprachcode für Meldungen (en, de, fr, es)
            verbose: Ob detaillierte Informationen ausgegeben werden sollen
        """
        self.language = language if language in [lang.value for lang in LanguageCode] else "en"
        self.verbose = verbose
        
        # Lade Übersetzungsdaten
        self.translations = self._load_translations()
        
        logger.info(f"ResultHandler initialisiert mit Sprache: {self.language}, Verbose: {self.verbose}")
    
    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """
        Lädt Übersetzungen für Fehlermeldungen und Statusnachrichten.
        
        Returns:
            Dictionary mit Übersetzungen für verschiedene Sprachen
        """
        # Grundlegende Übersetzungen
        translations = {
            # Erfolgsmeldungen
            "operation_success": {
                "en": "Operation completed successfully.",
                "de": "Operation erfolgreich abgeschlossen.",
                "fr": "Opération terminée avec succès.",
                "es": "Operación completada con éxito."
            },
            "partial_success": {
                "en": "Operation partially successful with some limitations.",
                "de": "Operation teilweise erfolgreich mit Einschränkungen.",
                "fr": "Opération partiellement réussie avec certaines limitations.",
                "es": "Operación parcialmente exitosa con algunas limitaciones."
            },
            
            # Fehlermeldungen
            "operation_failed": {
                "en": "Operation failed.",
                "de": "Operation fehlgeschlagen.",
                "fr": "Échec de l'opération.",
                "es": "La operación ha fallado."
            },
            "invalid_input": {
                "en": "Invalid input provided.",
                "de": "Ungültige Eingabe.",
                "fr": "Entrée non valide fournie.",
                "es": "Entrada no válida proporcionada."
            },
            "not_implemented": {
                "en": "This operation is not yet implemented.",
                "de": "Diese Operation ist noch nicht implementiert.",
                "fr": "Cette opération n'est pas encore implémentée.",
                "es": "Esta operación aún no está implementada."
            },
            "backend_error": {
                "en": "Error in the backend processing.",
                "de": "Fehler bei der Backend-Verarbeitung.",
                "fr": "Erreur dans le traitement backend.",
                "es": "Error en el procesamiento del backend."
            },
            
            # Tensor-Operationen
            "matrix_multiplication": {
                "en": "Matrix multiplication",
                "de": "Matrixmultiplikation",
                "fr": "Multiplication de matrices",
                "es": "Multiplicación de matrices"
            },
            "tensor_addition": {
                "en": "Tensor addition",
                "de": "Tensor-Addition",
                "fr": "Addition tensorielle",
                "es": "Adición tensorial"
            },
            "tensor_subtraction": {
                "en": "Tensor subtraction",
                "de": "Tensor-Subtraktion",
                "fr": "Soustraction tensorielle",
                "es": "Sustracción tensorial"
            },
            "matrix_transposition": {
                "en": "Matrix transposition",
                "de": "Matrix-Transposition",
                "fr": "Transposition de matrice",
                "es": "Transposición de matriz"
            },
            
            # Resultatbeschreibungen
            "result_shape": {
                "en": "Result shape",
                "de": "Ergebnisform",
                "fr": "Forme du résultat",
                "es": "Forma del resultado"
            },
            "execution_time": {
                "en": "Execution time",
                "de": "Ausführungszeit",
                "fr": "Temps d'exécution",
                "es": "Tiempo de ejecución"
            },
            "backend_used": {
                "en": "Backend used",
                "de": "Verwendetes Backend",
                "fr": "Backend utilisé",
                "es": "Backend utilizado"
            }
        }
        
        return translations
    
    def get_message(self, key: str) -> str:
        """
        Gibt eine übersetzte Nachricht zurück.
        
        Args:
            key: Schlüssel der Nachricht
            
        Returns:
            Übersetzte Nachricht
        """
        if key in self.translations:
            return self.translations[key].get(self.language, self.translations[key]["en"])
        
        # Fallback auf Englisch
        logger.warning(f"Keine Übersetzung gefunden für Schlüssel: {key}")
        return key
    
    def format_result(self, 
                      result: Any, 
                      operation_name: str, 
                      status: OperationStatus = OperationStatus.SUCCESS,
                      execution_time: Optional[float] = None,
                      backend: Optional[str] = None,
                      additional_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Formatiert ein Operationsergebnis.
        
        Args:
            result: Tensor-Ergebnis oder Fehlerobjekt
            operation_name: Name der ausgeführten Operation
            status: Status der Operation
            execution_time: Ausführungszeit in Sekunden
            backend: Verwendetes Backend
            additional_info: Zusätzliche Informationen
            
        Returns:
            Formatiertes Ergebnis als Dictionary
        """
        # Grundstruktur des Ergebnisses
        formatted_result = {
            "status": status.value,
            "status_message": self.get_message(status.value) if hasattr(status, "value") else str(status),
            "operation": {
                "name": operation_name,
                "localized_name": self.get_message(operation_name) if operation_name in self.translations else operation_name
            }
        }
        
        # Füge Ergebnis hinzu, falls erfolgreich
        if status in [OperationStatus.SUCCESS, OperationStatus.PARTIAL_SUCCESS]:
            # Tensor-Ergebnis serialisieren
            if hasattr(result, "shape"):
                # Für numpy/torch/mlx Tensoren
                formatted_result["result"] = {
                    "data": result.tolist() if hasattr(result, "tolist") else str(result),
                    "shape": list(result.shape) if hasattr(result, "shape") else None,
                    "dtype": str(result.dtype) if hasattr(result, "dtype") else None
                }
            else:
                # Für andere Ergebnistypen
                formatted_result["result"] = {
                    "data": result if isinstance(result, (int, float, bool, str, list, dict)) else str(result)
                }
        
        # Metadaten hinzufügen
        metadata = {}
        
        if execution_time is not None:
            metadata["execution_time"] = {
                "value": execution_time,
                "unit": "seconds",
                "localized_name": self.get_message("execution_time")
            }
        
        if backend is not None:
            metadata["backend"] = {
                "name": backend,
                "localized_name": self.get_message("backend_used")
            }
        
        # Zusätzliche Informationen hinzufügen
        if additional_info:
            metadata.update(additional_info)
        
        if metadata:
            formatted_result["metadata"] = metadata
        
        return formatted_result
    
    def format_error(self,
                     error: Exception,
                     operation_name: str,
                     status: OperationStatus = OperationStatus.FAILURE,
                     input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Formatiert einen Fehler.
        
        Args:
            error: Ausnahme oder Fehlerobjekt
            operation_name: Name der fehlgeschlagenen Operation
            status: Status der Operation
            input_data: Eingabedaten, die zum Fehler geführt haben
            
        Returns:
            Formatierter Fehler als Dictionary
        """
        error_result = {
            "status": status.value,
            "status_message": self.get_message(status.value) if hasattr(status, "value") else str(status),
            "operation": {
                "name": operation_name,
                "localized_name": self.get_message(operation_name) if operation_name in self.translations else operation_name
            },
            "error": {
                "type": error.__class__.__name__,
                "message": str(error)
            }
        }
        
        # Detaillierte Fehlerinfos für Entwickler hinzufügen
        if self.verbose:
            error_result["error"]["traceback"] = traceback.format_exc()
        
        # Eingabedaten hinzufügen, falls vorhanden
        if input_data:
            # Sicherstellen, dass die Eingabedaten serialisierbar sind
            safe_input = {}
            for key, value in input_data.items():
                if hasattr(value, "shape"):
                    # Für Tensor-Objekte
                    safe_input[key] = {
                        "shape": list(value.shape) if hasattr(value, "shape") else None,
                        "dtype": str(value.dtype) if hasattr(value, "dtype") else None,
                        "summary": f"Tensor mit Form {value.shape}" if hasattr(value, "shape") else str(value)
                    }
                else:
                    # Für andere Datentypen
                    try:
                        # Versuche zu serialisieren
                        json.dumps({key: value})
                        safe_input[key] = value
                    except (TypeError, OverflowError):
                        # Falls nicht serialisierbar
                        safe_input[key] = str(value)
            
            error_result["input_data"] = safe_input
        
        return error_result
    
    def log_result(self, result: Dict[str, Any], log_level: int = logging.INFO) -> None:
        """
        Protokolliert ein Ergebnis oder einen Fehler.
        
        Args:
            result: Formatiertes Ergebnis oder Fehler
            log_level: Log-Level
        """
        status = result.get("status", "unknown")
        operation = result.get("operation", {}).get("name", "unknown")
        
        if status in [OperationStatus.SUCCESS.value, OperationStatus.PARTIAL_SUCCESS.value]:
            # Erfolgreiche Operation
            log_message = f"Operation '{operation}' erfolgreich: {result.get('status_message', '')}"
            
            # Details zum Ergebnis
            if "result" in result and "shape" in result["result"]:
                shape = result["result"]["shape"]
                log_message += f" (Form: {shape})"
            
            logger.log(log_level, log_message)
        else:
            # Fehlgeschlagene Operation
            error_type = result.get("error", {}).get("type", "Unknown")
            error_message = result.get("error", {}).get("message", "")
            
            log_message = f"Operation '{operation}' fehlgeschlagen ({error_type}): {error_message}"
            logger.error(log_message)
            
            # Traceback protokollieren, falls verfügbar
            if self.verbose and "traceback" in result.get("error", {}):
                logger.debug(f"Traceback: {result['error']['traceback']}")
    
    def get_user_friendly_message(self, result: Dict[str, Any]) -> str:
        """
        Erstellt eine benutzerfreundliche Nachricht aus einem Ergebnis.
        
        Args:
            result: Formatiertes Ergebnis oder Fehler
            
        Returns:
            Benutzerfreundliche Nachricht
        """
        status = result.get("status", "unknown")
        status_message = result.get("status_message", "")
        operation_name = result.get("operation", {}).get("localized_name", "Operation")
        
        if status in [OperationStatus.SUCCESS.value, OperationStatus.PARTIAL_SUCCESS.value]:
            # Erfolgreiche Operation
            message = f"{operation_name}: {status_message}"
            
            # Details zum Ergebnis
            if "metadata" in result and "execution_time" in result["metadata"]:
                exec_time = result["metadata"]["execution_time"]["value"]
                message += f" ({exec_time:.4f}s)"
            
            if "result" in result and "shape" in result["result"]:
                shape = result["result"]["shape"]
                message += f"\n{self.get_message('result_shape')}: {shape}"
                
            return message
        else:
            # Fehlgeschlagene Operation
            error_message = result.get("error", {}).get("message", "")
            
            if status == OperationStatus.INVALID_INPUT.value:
                return f"{operation_name}: {status_message} {error_message}"
            elif status == OperationStatus.NOT_IMPLEMENTED.value:
                return f"{operation_name}: {status_message}"
            else:
                return f"{operation_name}: {status_message} {error_message}"


# Testfunktion
def test_result_handler():
    """Testet die Funktionalität des ResultHandlers."""
    # Erstelle Handler-Instanzen für verschiedene Sprachen
    handlers = {
        "en": ResultHandler("en", verbose=True),
        "de": ResultHandler("de", verbose=True),
        "fr": ResultHandler("fr", verbose=True),
        "es": ResultHandler("es", verbose=True)
    }
    
    # Simuliere erfolgreiche Operation
    class MockTensor:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype
        
        def tolist(self):
            return [[1, 2], [3, 4]]
    
    mock_result = MockTensor((2, 2), "float32")
    
    # Test für erfolgreiche Operation in allen Sprachen
    for lang, handler in handlers.items():
        print(f"\n=== Testing success in {lang} ===")
        result = handler.format_result(
            result=mock_result,
            operation_name="matrix_multiplication",
            status=OperationStatus.SUCCESS,
            execution_time=0.054,
            backend="mlx"
        )
        
        print(json.dumps(result, indent=2))
        print(f"User message: {handler.get_user_friendly_message(result)}")
        handler.log_result(result)
    
    # Simuliere Fehler
    try:
        raise ValueError("Tensoren haben inkompatible Formen für Multiplikation")
    except Exception as e:
        error = e
    
    # Test für Fehler in allen Sprachen
    for lang, handler in handlers.items():
        print(f"\n=== Testing error in {lang} ===")
        error_result = handler.format_error(
            error=error,
            operation_name="matrix_multiplication",
            status=OperationStatus.INVALID_INPUT,
            input_data={"tensor1": MockTensor((3, 4), "float32"), "tensor2": MockTensor((5, 6), "float32")}
        )
        
        print(json.dumps(error_result, indent=2))
        print(f"User message: {handler.get_user_friendly_message(error_result)}")
        handler.log_result(error_result)


if __name__ == "__main__":
    # Konfiguriere Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Führe Test aus
    test_result_handler()
