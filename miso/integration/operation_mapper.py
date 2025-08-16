#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate AGI - Operation Mapper

Diese Komponente übersetzt natürlichsprachliche Befehle in konkrete Tensor-Operationen
und stellt die Verbindung zwischen M-LINGUA (Sprachverständnis) und T-Mathematics (Tensor-Engine) her.
"""

import re
import logging
import json
from typing import Dict, List, Any, Tuple, Optional, Union, Set
from enum import Enum

# Konfiguriere Logger
logger = logging.getLogger("MISO.OperationMapper")

class OperationType(Enum):
    """Unterstützte Tensor-Operationstypen."""
    MATRIX_MULTIPLICATION = "matrix_multiplication"
    TENSOR_ADDITION = "tensor_addition"
    TENSOR_SUBTRACTION = "tensor_subtraction"
    MATRIX_TRANSPOSITION = "matrix_transposition"
    ELEMENT_WISE_OPERATION = "element_wise_operation"
    TENSOR_NORM = "tensor_norm"
    CONVOLUTION = "convolution"
    UNKNOWN = "unknown"


class OperationMapper:
    """
    Übersetzt natürlichsprachliche Befehle in konkrete Tensor-Operationen.
    Unterstützt mehrere Sprachen (DE, EN, FR, ES) und verschiedene Ausdrucksweisen.
    """
    
    def __init__(self, language: str = "en"):
        """
        Initialisiert den OperationMapper.
        
        Args:
            language: Sprachcode (en, de, fr, es)
        """
        self.language = language.lower()
        self.supported_languages = {"en", "de", "fr", "es"}
        
        if self.language not in self.supported_languages:
            logger.warning(f"Sprache {language} nicht unterstützt, fallback auf Englisch")
            self.language = "en"
        
        # Lade Operationsbezeichnungen und Phrasen für alle unterstützten Sprachen
        self.operation_phrases = self._load_operation_phrases()
        
        logger.info(f"OperationMapper initialisiert mit Sprache: {self.language}")
    
    def _load_operation_phrases(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Lädt Phrasen für verschiedene Operationen in allen unterstützten Sprachen.
        
        Returns:
            Dictionary mit Operationstypen und zugehörigen Phrasen pro Sprache
        """
        # Phrasen für Matrixmultiplikation
        matrix_multiplication = {
            "en": ["multiply", "matrix multiplication", "matrix product", "dot product"],
            "de": ["multiplizieren", "matrixmultiplikation", "matrixprodukt", "skalarprodukt"],
            "fr": ["multiplier", "multiplie", "multiplication matricielle", "produit matriciel", "produit scalaire"],
            "es": ["multiplicar", "multiplica", "multiplicación de matrices", "producto matricial", "producto escalar"]
        }
        
        # Phrasen für Tensor-Addition
        tensor_addition = {
            "en": ["add", "sum", "addition", "plus"],
            "de": ["addieren", "summe", "addition", "plus"],
            "fr": ["ajouter", "somme", "addition", "plus"],
            "es": ["sumar", "suma", "adición", "más"]
        }
        
        # Phrasen für Tensor-Subtraktion
        tensor_subtraction = {
            "en": ["subtract", "difference", "minus"],
            "de": ["subtrahieren", "differenz", "unterschied", "minus"],
            "fr": ["soustraire", "différence", "moins"],
            "es": ["resta", "restar", "diferencia", "menos"]
        }
        
        # Phrasen für Matrix-Transposition
        matrix_transposition = {
            "en": ["transpose", "transposition", "flip"],
            "de": ["transponieren", "transposition", "spiegeln"],
            "fr": ["transpose", "transposer", "transposition", "retourner"],
            "es": ["transpone", "transponer", "transposición", "invertir"]
        }
        
        # Phrasen für elementweise Operationen
        element_wise_operation = {
            "en": ["element-wise", "elementwise", "each element", "per element"],
            "de": ["elementweise", "pro element", "jedes element"],
            "fr": ["élément par élément", "par élément", "chaque élément"],
            "es": ["elemento por elemento", "por elemento", "cada elemento"]
        }
        
        # Phrasen für Tensor-Norm
        tensor_norm = {
            "en": ["norm", "length", "magnitude"],
            "de": ["norm", "länge", "betrag"],
            "fr": ["norme", "longueur", "magnitude"],
            "es": ["norma", "longitud", "magnitud"]
        }
        
        # Phrasen für Faltung (Convolution)
        convolution = {
            "en": ["convolve", "convolution", "filter"],
            "de": ["falten", "faltung", "filter"],
            "fr": ["convoluer", "convolution", "filtre"],
            "es": ["convolucionar", "convolución", "filtro"]
        }
        
        # Kombiniere alle Phrasen
        operation_phrases = {
            OperationType.MATRIX_MULTIPLICATION.value: matrix_multiplication,
            OperationType.TENSOR_ADDITION.value: tensor_addition,
            OperationType.TENSOR_SUBTRACTION.value: tensor_subtraction,
            OperationType.MATRIX_TRANSPOSITION.value: matrix_transposition,
            OperationType.ELEMENT_WISE_OPERATION.value: element_wise_operation,
            OperationType.TENSOR_NORM.value: tensor_norm,
            OperationType.CONVOLUTION.value: convolution
        }
        
        return operation_phrases
    
    def detect_operation(self, text: str) -> Tuple[OperationType, Dict[str, Any]]:
        """
        Erkennt die in einem Text beschriebene Operation.
        
        Args:
            text: Natürlichsprachliche Eingabe
            
        Returns:
            Tuple aus Operationstyp und extrahierten Parametern
        """
        # Normalisiere Text (Kleinschreibung, Entfernung von Sonderzeichen)
        normalized_text = text.lower()
        normalized_text = re.sub(r'[^\w\s]', ' ', normalized_text)
        
        # Automatische Spracherkennung, falls möglich
        detected_language = self._detect_language(normalized_text)
        if detected_language:
            language = detected_language
        else:
            language = self.language
        
        # Log erkannte Sprache für Debugging
        logger.debug(f"Erkannte Sprache: {language} für Text: '{text}'")
        
        # Suche nach Operationsphrasen in der erkannten Sprache
        detected_ops = []
        matched_phrases = {}
        
        for op_type, phrases_by_lang in self.operation_phrases.items():
            # Versuche zuerst mit der erkannten Sprache
            if language in phrases_by_lang:
                phrases = phrases_by_lang[language]
                for phrase in phrases:
                    if phrase in normalized_text:
                        detected_ops.append(op_type)
                        matched_phrases[op_type] = phrase
                        break
            
            # Wenn keine Operation gefunden wurde, versuche mit allen unterstützten Sprachen
            if op_type not in detected_ops:
                for lang, phrases in phrases_by_lang.items():
                    if lang != language:  # Überspringe die bereits überprüfte Sprache
                        for phrase in phrases:
                            if phrase in normalized_text:
                                detected_ops.append(op_type)
                                matched_phrases[op_type] = phrase
                                # Aktualisiere die erkannte Sprache, da wir ein Muster in einer anderen Sprache gefunden haben
                                language = lang
                                break
                        if op_type in detected_ops:
                            break
        
        # Wenn mehrere Operationen erkannt wurden, wähle die spezifischste
        if not detected_ops:
            logger.warning(f"Keine Operation erkannt in: '{text}'")
            return OperationType.UNKNOWN, {}
        
        # Priorisiere spezifischere Operationen
        priority_order = [
            OperationType.MATRIX_MULTIPLICATION.value,
            OperationType.TENSOR_ADDITION.value,
            OperationType.TENSOR_SUBTRACTION.value,
            OperationType.MATRIX_TRANSPOSITION.value,
            OperationType.CONVOLUTION.value,
            OperationType.TENSOR_NORM.value,
            OperationType.ELEMENT_WISE_OPERATION.value
        ]
        
        for priority_op in priority_order:
            if priority_op in detected_ops:
                detected_op = priority_op
                break
        else:
            detected_op = detected_ops[0]
        
        # Extrahiere Parameter je nach Operation
        params = self._extract_parameters(normalized_text, detected_op, language)
        
        logger.info(f"Operation erkannt: {detected_op}, Parameter: {params}")
        return OperationType(detected_op), params
    
    def _detect_language(self, text: str) -> Optional[str]:
        """
        Versucht, die Sprache eines Textes zu erkennen.
        
        Args:
            text: Zu analysierender Text
            
        Returns:
            Sprachcode oder None, falls keine sichere Erkennung möglich
        """
        # Einfache sprachspezifische Indikatoren
        lang_indicators = {
            "en": ["the", "and", "matrix", "tensor", "calculate", "compute", "with", "from"],
            "de": ["die", "und", "matrix", "tensor", "berechne", "berechnen", "mit", "von"],
            "fr": ["le", "la", "et", "matrice", "tenseur", "calculer", "avec", "de", "du", "des"],
            "es": ["el", "la", "y", "matriz", "tensor", "calcular", "con", "de", "del"]
        }
        
        # Präfixe, die wahrscheinlich vor einer bestimmten Sprache erscheinen
        lang_prefixes = {
            "en": ["calculate", "compute", "find"],
            "de": ["berechne", "finde", "bestimme"],
            "fr": ["calcule", "trouve", "determine"],
            "es": ["calcula", "encuentra", "determina"]
        }
        
        # Überprüfe auf eindeutige Sprachpräfixe
        for lang, prefixes in lang_prefixes.items():
            for prefix in prefixes:
                if text.lower().startswith(prefix):
                    return lang
        
        # Zähle Indikatoren pro Sprache
        lang_scores = {lang: 0 for lang in self.supported_languages}
        
        for lang, indicators in lang_indicators.items():
            for indicator in indicators:
                # Überprüfe, ob das Indikatorwort im Text vorkommt
                if f" {indicator} " in f" {text} " or f"-{indicator} " in f" {text} " or f" {indicator}-" in f" {text} ":
                    lang_scores[lang] += 1
        
        # Bestimme die Sprache mit der höchsten Punktzahl
        max_score = max(lang_scores.values())
        if max_score > 0:
            # Bei Gleichstand bevorzuge die konfigurierte Sprache
            max_langs = [lang for lang, score in lang_scores.items() if score == max_score]
            if len(max_langs) == 1:
                return max_langs[0]
            elif self.language in max_langs:
                return self.language
            else:
                return max_langs[0]
        
        # Keine klare Erkennung möglich
        return None
    
    def _extract_parameters(self, text: str, operation_type: str, language: str) -> Dict[str, Any]:
        """
        Extrahiert Parameter für die erkannte Operation.
        
        Args:
            text: Normalisierter Text
            operation_type: Erkannter Operationstyp
            language: Erkannte Sprache
            
        Returns:
            Dictionary mit extrahierten Parametern
        """
        params = {}
        
        # Extrahiere Tensorbezeichnungen mit Regex
        tensor_regex = {
            "en": r"(?:matrix|tensor|vector)\s+([A-Za-z0-9_]+)",
            "de": r"(?:matrix|tensor|vektor)\s+([A-Za-z0-9_]+)",
            "fr": r"(?:matrice|tenseur|vecteur)\s+([A-Za-z0-9_]+)",
            "es": r"(?:matriz|tensor|vector)\s+([A-Za-z0-9_]+)"
        }
        
        if language in tensor_regex:
            tensor_matches = re.findall(tensor_regex[language], text)
            if tensor_matches:
                params["tensor_names"] = tensor_matches
        
        # Extrahiere numerische Parameter
        number_regex = r"(\d+(?:\.\d+)?)"
        number_matches = re.findall(number_regex, text)
        if number_matches:
            params["numeric_values"] = [float(n) for n in number_matches]
        
        # Operationsspezifische Parameter
        if operation_type == OperationType.MATRIX_MULTIPLICATION.value:
            # Für Matrixmultiplikation brauchen wir zwei Tensoren
            if "tensor_names" in params and len(params["tensor_names"]) >= 2:
                params["tensor1"] = params["tensor_names"][0]
                params["tensor2"] = params["tensor_names"][1]
        
        elif operation_type == OperationType.TENSOR_ADDITION.value or operation_type == OperationType.TENSOR_SUBTRACTION.value:
            # Für Addition/Subtraktion brauchen wir zwei Tensoren
            if "tensor_names" in params and len(params["tensor_names"]) >= 2:
                params["tensor1"] = params["tensor_names"][0]
                params["tensor2"] = params["tensor_names"][1]
        
        elif operation_type == OperationType.MATRIX_TRANSPOSITION.value:
            # Für Transposition brauchen wir einen Tensor
            if "tensor_names" in params and len(params["tensor_names"]) >= 1:
                params["tensor"] = params["tensor_names"][0]
        
        return params
    
    def get_operation_function(self, operation_type: OperationType) -> Optional[str]:
        """
        Gibt den Funktionsnamen für eine Operation zurück.
        
        Args:
            operation_type: Operationstyp
            
        Returns:
            Funktionsname oder None, falls nicht unterstützt
        """
        # Zuordnung von Operationstypen zu Funktionsnamen
        function_map = {
            OperationType.MATRIX_MULTIPLICATION: "matrix_multiply",
            OperationType.TENSOR_ADDITION: "tensor_add",
            OperationType.TENSOR_SUBTRACTION: "tensor_subtract",
            OperationType.MATRIX_TRANSPOSITION: "matrix_transpose",
            OperationType.ELEMENT_WISE_OPERATION: "element_wise_operation",
            OperationType.TENSOR_NORM: "tensor_norm",
            OperationType.CONVOLUTION: "tensor_convolve"
        }
        
        return function_map.get(operation_type)
    
    def generate_code_snippet(self, operation_type: OperationType, 
                            params: Dict[str, Any], 
                            backend: str = "numpy") -> str:
        """
        Generiert einen Code-Snippet für die erkannte Operation.
        
        Args:
            operation_type: Erkannter Operationstyp
            params: Extrahierte Parameter
            backend: Zu verwendendes Backend
            
        Returns:
            Code-Snippet als String
        """
        # Basisimporte für verschiedene Backends
        imports = {
            "numpy": "import numpy as np",
            "torch": "import torch",
            "mlx": "import mlx.core as mx",
            "tf": "import tensorflow as tf",
            "jax": "import jax\nimport jax.numpy as jnp"
        }
        
        # Operation nicht unterstützt
        if operation_type == OperationType.UNKNOWN:
            return "# Keine bekannte Operation erkannt"
        
        # Basis-Import
        code = [imports.get(backend, "# Unknown backend")]
        
        # Operationsspezifischer Code
        if operation_type == OperationType.MATRIX_MULTIPLICATION:
            if "tensor1" in params and "tensor2" in params:
                tensor1 = params["tensor1"]
                tensor2 = params["tensor2"]
                
                if backend == "numpy":
                    code.append(f"result = np.matmul({tensor1}, {tensor2})")
                elif backend == "torch":
                    code.append(f"result = torch.matmul({tensor1}, {tensor2})")
                elif backend == "mlx":
                    code.append(f"result = mx.matmul({tensor1}, {tensor2})")
                elif backend == "tf":
                    code.append(f"result = tf.matmul({tensor1}, {tensor2})")
                elif backend == "jax":
                    code.append(f"result = jnp.matmul({tensor1}, {tensor2})")
            else:
                code.append("# Fehlende Parameter für Matrixmultiplikation")
        
        elif operation_type == OperationType.TENSOR_ADDITION:
            if "tensor1" in params and "tensor2" in params:
                tensor1 = params["tensor1"]
                tensor2 = params["tensor2"]
                
                if backend == "numpy":
                    code.append(f"result = {tensor1} + {tensor2}")
                elif backend == "torch":
                    code.append(f"result = {tensor1} + {tensor2}")
                elif backend == "mlx":
                    code.append(f"result = {tensor1} + {tensor2}")
                elif backend == "tf":
                    code.append(f"result = {tensor1} + {tensor2}")
                elif backend == "jax":
                    code.append(f"result = {tensor1} + {tensor2}")
            else:
                code.append("# Fehlende Parameter für Tensor-Addition")
        
        elif operation_type == OperationType.TENSOR_SUBTRACTION:
            if "tensor1" in params and "tensor2" in params:
                tensor1 = params["tensor1"]
                tensor2 = params["tensor2"]
                
                if backend == "numpy":
                    code.append(f"result = {tensor1} - {tensor2}")
                elif backend == "torch":
                    code.append(f"result = {tensor1} - {tensor2}")
                elif backend == "mlx":
                    code.append(f"result = {tensor1} - {tensor2}")
                elif backend == "tf":
                    code.append(f"result = {tensor1} - {tensor2}")
                elif backend == "jax":
                    code.append(f"result = {tensor1} - {tensor2}")
            else:
                code.append("# Fehlende Parameter für Tensor-Subtraktion")
        
        elif operation_type == OperationType.MATRIX_TRANSPOSITION:
            if "tensor" in params:
                tensor = params["tensor"]
                
                if backend == "numpy":
                    code.append(f"result = np.transpose({tensor})")
                elif backend == "torch":
                    code.append(f"result = {tensor}.transpose(-2, -1)")
                elif backend == "mlx":
                    code.append(f"result = mx.transpose({tensor})")
                elif backend == "tf":
                    code.append(f"result = tf.transpose({tensor})")
                elif backend == "jax":
                    code.append(f"result = jnp.transpose({tensor})")
            else:
                code.append("# Fehlende Parameter für Matrix-Transposition")
        
        # Füge abschließenden Kommentar hinzu
        code.append(f"# Operation: {operation_type.value}")
        
        return "\n".join(code)
    
    def map_to_t_mathematics(self, operation_type: OperationType, 
                           params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mappt eine erkannte Operation auf die T-Mathematics Engine.
        
        Args:
            operation_type: Erkannter Operationstyp
            params: Extrahierte Parameter
            
        Returns:
            Dictionary mit T-Mathematics-kompatiblen Parametern
        """
        t_math_params = {
            "operation": operation_type.value,
            "params": {}
        }
        
        # Füge sprachübergreifende Parameter hinzu
        t_math_params["language"] = self.language
        
        # Operationsspezifische Parameter
        if operation_type == OperationType.MATRIX_MULTIPLICATION:
            if "tensor1" in params and "tensor2" in params:
                t_math_params["params"]["tensor1_name"] = params["tensor1"]
                t_math_params["params"]["tensor2_name"] = params["tensor2"]
        
        elif operation_type == OperationType.TENSOR_ADDITION:
            if "tensor1" in params and "tensor2" in params:
                t_math_params["params"]["tensor1_name"] = params["tensor1"]
                t_math_params["params"]["tensor2_name"] = params["tensor2"]
        
        elif operation_type == OperationType.TENSOR_SUBTRACTION:
            if "tensor1" in params and "tensor2" in params:
                t_math_params["params"]["tensor1_name"] = params["tensor1"]
                t_math_params["params"]["tensor2_name"] = params["tensor2"]
        
        elif operation_type == OperationType.MATRIX_TRANSPOSITION:
            if "tensor" in params:
                t_math_params["params"]["tensor_name"] = params["tensor"]
        
        # Füge alle extrahierten numerischen Werte hinzu
        if "numeric_values" in params:
            t_math_params["params"]["numeric_values"] = params["numeric_values"]
        
        # Füge alle Tensornamen hinzu
        if "tensor_names" in params:
            t_math_params["params"]["all_tensor_names"] = params["tensor_names"]
        
        return t_math_params


# Testfunktion
def test_operation_mapper():
    """Testet die Funktionalität des OperationMappers."""
    # Konfiguriere detailliertes Logging für den Test
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n=== Testing OperationMapper ===")
    
    # Testbeispiele in verschiedenen Sprachen
    test_cases = [
        # Englisch
        ("Multiply matrix A with matrix B", "en"),
        ("Calculate the dot product of vector x and vector y", "en"),
        ("Add tensor X and tensor Y", "en"),
        ("Subtract matrix C from matrix D", "en"),
        ("Transpose matrix M", "en"),
        
        # Deutsch
        ("Multipliziere die Matrix A mit der Matrix B", "de"),
        ("Berechne das Skalarprodukt von Vektor x und Vektor y", "de"),
        ("Addiere Tensor X und Tensor Y", "de"),
        ("Subtrahiere Matrix C von Matrix D", "de"),
        ("Transponiere die Matrix M", "de"),
        
        # Französisch
        ("Multiplie la matrice A avec la matrice B", "fr"),
        ("Calcule le produit scalaire du vecteur x et du vecteur y", "fr"),
        ("Ajoute le tenseur X et le tenseur Y", "fr"),
        ("Soustrais la matrice C de la matrice D", "fr"),
        ("Transpose la matrice M", "fr"),
        
        # Spanisch
        ("Multiplica la matriz A con la matriz B", "es"),
        ("Calcula el producto escalar del vector x y del vector y", "es"),
        ("Suma el tensor X y el tensor Y", "es"),
        ("Resta la matriz C de la matriz D", "es"),
        ("Transpone la matriz M", "es")
    ]
    
    # Teste jeden Fall mit entsprechender Sprache
    for text, language in test_cases:
        print(f"\nSprache: {language}, Text: '{text}'")
        
        # Erstelle OperationMapper mit der angegebenen Sprache
        mapper = OperationMapper(language)
        
        # Erkenne Operation
        operation_type, params = mapper.detect_operation(text)
        print(f"Erkannte Operation: {operation_type.value}")
        print(f"Extrahierte Parameter: {params}")
        
        # Generiere Code-Snippet
        for backend in ["numpy", "torch", "mlx"]:
            code = mapper.generate_code_snippet(operation_type, params, backend)
            print(f"\nCode für {backend}:\n{code}")
        
        # Mappe auf T-Mathematics
        t_math_params = mapper.map_to_t_mathematics(operation_type, params)
        print(f"\nT-Mathematics Mapping:\n{json.dumps(t_math_params, indent=2)}")


if __name__ == "__main__":
    # Führe Test aus
    test_operation_mapper()
