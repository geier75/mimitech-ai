#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate AGI - Mehrsprachlicher Parser

Diese Komponente verarbeitet natürlichsprachliche Eingaben in verschiedenen Sprachen
und extrahiert daraus strukturierte Daten für die Weiterverarbeitung durch die
T-Mathematics Engine.
"""

import re
import logging
import json
from typing import Dict, List, Any, Tuple, Optional, Union, Set
from pathlib import Path
import os
from enum import Enum

# Importiere Operation Mapper für die Erkennung von Tensor-Operationen
try:
    # Versuche zuerst den absoluten Import
    from miso.integration.operation_mapper import OperationMapper, OperationType
except ImportError:
    # Fallback auf relativen Import
    try:
        from .operation_mapper import OperationMapper, OperationType
    except ImportError:
        # Direkter Import als letzter Ausweg
        import sys
        import os
        
        # Füge das übergeordnete Verzeichnis zum Pfad hinzu
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        
        # Versuche erneut zu importieren
        try:
            from integration.operation_mapper import OperationMapper, OperationType
        except ImportError:
            # Absoluter Import mit vollständigem Pfad
            current_dir = os.path.dirname(os.path.abspath(__file__))
            operation_mapper_path = os.path.join(current_dir, 'operation_mapper.py')
            
            # Prüfe, ob die Datei existiert
            if os.path.exists(operation_mapper_path):
                sys.path.append(current_dir)
                from operation_mapper import OperationMapper, OperationType
            else:
                raise ImportError(f"Operation Mapper nicht gefunden unter: {operation_mapper_path}")

# Konfiguriere Logger
logger = logging.getLogger("MISO.MultilingualParser")

class LanguageCode(Enum):
    """Unterstützte Sprachcodes."""
    EN = "en"  # Englisch
    DE = "de"  # Deutsch
    FR = "fr"  # Französisch
    ES = "es"  # Spanisch


class ParserMode(Enum):
    """Verschiedene Modi für den Parser."""
    COMMAND = "command"        # Befehlsmodus (z.B. "Multipliziere Matrix A mit Matrix B")
    QUERY = "query"            # Abfragemodus (z.B. "Was ist die Transponierte von Matrix A?")
    EXPLANATION = "explanation"  # Erklärungsmodus (z.B. "Erkläre, wie man Matrizen multipliziert")


class MultilingualParser:
    """
    Parser für mehrsprachige natürlichsprachliche Eingaben.
    
    Unterstützt:
    - Verschiedene Sprachen (DE, EN, FR, ES)
    - Extraktion von Tensor-Operationen
    - Erkennung von Variablennamen und Werten
    - Umgang mit Kontextbezügen und Referenzen
    """
    
    def __init__(self, language: str = "en", context_memory_size: int = 5):
        """
        Initialisiert den MultilingualParser.
        
        Args:
            language: Standardsprache (en, de, fr, es)
            context_memory_size: Anzahl der zu speichernden Kontext-Elemente
        """
        self.language = language.lower()
        if self.language not in [lang.value for lang in LanguageCode]:
            logger.warning(f"Sprache {language} nicht unterstützt, fallback auf Englisch")
            self.language = "en"
        
        self.context_memory_size = context_memory_size
        self.context_memory = []
        
        # Operation Mapper initialisieren
        self.operation_mapper = OperationMapper(self.language)
        
        # Lade Sprach-spezifische Muster und Vokabular
        self.patterns = self._load_patterns()
        
        logger.info(f"MultilingualParser initialisiert mit Sprache: {self.language}")
    
    def _load_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Lädt sprachspezifische Muster für die Verarbeitung.
        
        Returns:
            Dictionary mit Mustern für jede unterstützte Sprache
        """
        patterns = {}
        
        # Englisch
        patterns["en"] = {
            # Befehlsmuster (imperativ)
            "command_patterns": [
                r"(calculate|compute|find|determine)\s+.+",
                r"(multiply|add|subtract|transpose)\s+.+",
                r"(perform|do|execute|run)\s+(a|the)?\s*(.+)\s+operation"
            ],
            
            # Abfragemuster (fragend)
            "query_patterns": [
                r"what\s+is\s+.+\?",
                r"how\s+(do|can|would)\s+I\s+.+\?",
                r"(can|could)\s+you\s+.+\?"
            ],
            
            # Erklärungsmuster
            "explanation_patterns": [
                r"explain\s+how\s+to\s+.+",
                r"describe\s+(the\s+)?(process|method|way)\s+to\s+.+",
                r"tell\s+me\s+about\s+.+"
            ],
            
            # Tensormusterm
            "tensor_patterns": [
                r"(matrix|tensor|vector)\s+([A-Za-z0-9_]+)",
                r"([A-Za-z0-9_]+)\s+(matrix|tensor|vector)"
            ],
            
            # Wertmuster
            "value_patterns": [
                r"(value|number|scalar)\s+([0-9]+\.?[0-9]*)",
                r"([0-9]+\.?[0-9]*)"
            ],
            
            # Muster für Kontextreferenzen
            "reference_patterns": [
                r"(it|this|that|the\s+result)",
                r"(previous|last|former)\s+(result|output|calculation)"
            ]
        }
        
        # Deutsch
        patterns["de"] = {
            # Befehlsmuster (imperativ)
            "command_patterns": [
                r"(berechne|bestimme|finde|ermittle)\s+.+",
                r"(multipliziere|addiere|subtrahiere|transponiere)\s+.+",
                r"(führe|mache|führe\s+durch)\s+(eine|die)?\s*(.+)\s+Operation"
            ],
            
            # Abfragemuster (fragend)
            "query_patterns": [
                r"was\s+ist\s+.+\?",
                r"wie\s+(kann|könnte|würde)\s+ich\s+.+\?",
                r"(kannst|könntest)\s+du\s+.+\?"
            ],
            
            # Erklärungsmuster
            "explanation_patterns": [
                r"erkläre\s+(mir)?\s+wie\s+man\s+.+",
                r"beschreibe\s+(den|die|das)?\s*(Prozess|Methode|Verfahren)\s+zur\s+.+",
                r"erzähl\s+(mir)?\s+(etwas)?\s+über\s+.+"
            ],
            
            # Tensormuster
            "tensor_patterns": [
                r"(Matrix|Tensor|Vektor)\s+([A-Za-z0-9_]+)",
                r"([A-Za-z0-9_]+)\s+(Matrix|Tensor|Vektor)"
            ],
            
            # Wertmuster
            "value_patterns": [
                r"(Wert|Zahl|Skalar)\s+([0-9]+\.?[0-9]*)",
                r"([0-9]+\.?[0-9]*)"
            ],
            
            # Muster für Kontextreferenzen
            "reference_patterns": [
                r"(es|diese[sr]?|das|dem|den|der|die|das\s+Ergebnis)",
                r"(vorherige[sn]?|letzte[sn]?|frühere[sn]?)\s+(Ergebnis|Ausgabe|Berechnung)"
            ]
        }
        
        # Französisch
        patterns["fr"] = {
            # Befehlsmuster (imperativ)
            "command_patterns": [
                r"(calcule|détermine|trouve)\s+.+",
                r"(multiplie|ajoute|soustrais|transpose)\s+.+",
                r"(effectue|fais|exécute)\s+(une|la)?\s*opération\s+de\s+(.+)"
            ],
            
            # Abfragemuster (fragend)
            "query_patterns": [
                r"qu'est-ce\s+que\s+.+\?",
                r"comment\s+(puis|pourrais|pourrai)-je\s+.+\?",
                r"(peux|pourrais)-tu\s+.+\?"
            ],
            
            # Erklärungsmuster
            "explanation_patterns": [
                r"explique\s+(moi)?\s+comment\s+.+",
                r"décris\s+(le|la)?\s*(processus|méthode|façon)\s+de\s+.+",
                r"parle\s+(moi)?\s+de\s+.+"
            ],
            
            # Tensormuster
            "tensor_patterns": [
                r"(matrice|tenseur|vecteur)\s+([A-Za-z0-9_]+)",
                r"([A-Za-z0-9_]+)\s+(matrice|tenseur|vecteur)"
            ],
            
            # Wertmuster
            "value_patterns": [
                r"(valeur|nombre|scalaire)\s+([0-9]+\.?[0-9]*)",
                r"([0-9]+\.?[0-9]*)"
            ],
            
            # Muster für Kontextreferenzen
            "reference_patterns": [
                r"(il|elle|ceci|cela|ce|le\s+résultat)",
                r"(précédent|dernier)\s+(résultat|sortie|calcul)"
            ]
        }
        
        # Spanisch
        patterns["es"] = {
            # Befehlsmuster (imperativ)
            "command_patterns": [
                r"(calcula|determina|encuentra)\s+.+",
                r"(multiplica|suma|resta|transpone)\s+.+",
                r"(realiza|haz|ejecuta)\s+(una|la)?\s*operación\s+de\s+(.+)"
            ],
            
            # Abfragemuster (fragend)
            "query_patterns": [
                r"(qué|cuál)\s+es\s+.+\?",
                r"cómo\s+(puedo|podría)\s+.+\?",
                r"(puedes|podrías)\s+.+\?"
            ],
            
            # Erklärungsmuster
            "explanation_patterns": [
                r"explica\s+(me)?\s+cómo\s+.+",
                r"describe\s+(el|la)?\s*(proceso|método|forma)\s+de\s+.+",
                r"háblame\s+de\s+.+"
            ],
            
            # Tensormuster
            "tensor_patterns": [
                r"(matriz|tensor|vector)\s+([A-Za-z0-9_]+)",
                r"([A-Za-z0-9_]+)\s+(matriz|tensor|vector)"
            ],
            
            # Wertmuster
            "value_patterns": [
                r"(valor|número|escalar)\s+([0-9]+\.?[0-9]*)",
                r"([0-9]+\.?[0-9]*)"
            ],
            
            # Muster für Kontextreferenzen
            "reference_patterns": [
                r"(esto|eso|él|ella|el\s+resultado)",
                r"(anterior|último|previo)\s+(resultado|salida|cálculo)"
            ]
        }
        
        return patterns
    
    def detect_mode(self, text: str, language: Optional[str] = None) -> ParserMode:
        """
        Erkennt den Modus der Eingabe (Befehl, Abfrage, Erklärung).
        
        Args:
            text: Zu analysierende Eingabe
            language: Sprachcode (optional, sonst wird Standardsprache verwendet)
            
        Returns:
            Erkannter Modus
        """
        language = language or self.language
        
        # Normalisiere Text
        normalized_text = text.lower()
        
        if language in self.patterns:
            patterns = self.patterns[language]
            
            # Prüfe auf Befehlsmuster
            for pattern in patterns["command_patterns"]:
                if re.match(pattern, normalized_text, re.IGNORECASE):
                    return ParserMode.COMMAND
            
            # Prüfe auf Abfragemuster
            for pattern in patterns["query_patterns"]:
                if re.match(pattern, normalized_text, re.IGNORECASE):
                    return ParserMode.QUERY
            
            # Prüfe auf Erklärungsmuster
            for pattern in patterns["explanation_patterns"]:
                if re.match(pattern, normalized_text, re.IGNORECASE):
                    return ParserMode.EXPLANATION
        
        # Standardmäßig als Befehl betrachten
        logger.info(f"Modus für Text konnte nicht erkannt werden, verwende Befehlsmodus: '{text}'")
        return ParserMode.COMMAND
    
    def extract_tensors(self, text: str, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extrahiert Tensor-Informationen aus dem Text.
        
        Args:
            text: Zu analysierender Text
            language: Sprachcode (optional, sonst wird Standardsprache verwendet)
            
        Returns:
            Liste von Tensor-Informationen
        """
        language = language or self.language
        
        tensors = []
        
        if language in self.patterns:
            patterns = self.patterns[language]
            
            # Suche nach Tensor-Mustern
            for pattern in patterns["tensor_patterns"]:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    # Extrahiere Tensorname und -typ
                    if len(match.groups()) >= 2:
                        # Standardformat: (matrix|tensor|vector) name
                        if match.group(1).lower() in ["matrix", "tensor", "vector", 
                                               "matrice", "tenseur", "vecteur",
                                               "matriz", "vector", 
                                               "matrix", "tensor", "vektor"]:
                            tensor_type = match.group(1).lower()
                            tensor_name = match.group(2).lower()
                        else:
                            # Alternatives Format: name (matrix|tensor|vector)
                            tensor_name = match.group(1).lower()
                            tensor_type = match.group(2).lower()
                        
                        # Normalisiere Tensor-Typ
                        normalized_type = self._normalize_tensor_type(tensor_type, language)
                        
                        # Speichere Tensor-Information
                        tensor_info = {
                            "name": tensor_name,
                            "type": normalized_type,
                            "span": (match.start(), match.end()),
                            "text": match.group(0)
                        }
                        
                        # Prüfe auf Duplikate und füge nur hinzu, wenn noch nicht vorhanden
                        if not any(t["name"] == tensor_name for t in tensors):
                            tensors.append(tensor_info)
        
        return tensors
    
    def _normalize_tensor_type(self, tensor_type: str, language: str) -> str:
        """
        Normalisiert einen Tensor-Typ auf einen einheitlichen Wert.
        
        Args:
            tensor_type: Zu normalisierender Tensor-Typ
            language: Sprachcode
            
        Returns:
            Normalisierter Tensor-Typ
        """
        # Normalisierungstabelle für verschiedene Sprachen
        normalization = {
            "en": {
                "matrix": "matrix",
                "tensor": "tensor",
                "vector": "vector"
            },
            "de": {
                "matrix": "matrix",
                "tensor": "tensor",
                "vektor": "vector"
            },
            "fr": {
                "matrice": "matrix",
                "tenseur": "tensor",
                "vecteur": "vector"
            },
            "es": {
                "matriz": "matrix",
                "tensor": "tensor",
                "vector": "vector"
            }
        }
        
        tensor_type = tensor_type.lower()
        
        if language in normalization:
            language_map = normalization[language]
            if tensor_type in language_map:
                return language_map[tensor_type]
        
        # Fallback
        if tensor_type in ["matrix", "matrice", "matriz"]:
            return "matrix"
        elif tensor_type in ["tensor", "tenseur"]:
            return "tensor"
        elif tensor_type in ["vector", "vektor", "vecteur"]:
            return "vector"
        else:
            return "tensor"  # Standardwert
    
    def extract_values(self, text: str, language: Optional[str] = None) -> List[float]:
        """
        Extrahiert numerische Werte aus dem Text.
        
        Args:
            text: Zu analysierender Text
            language: Sprachcode (optional, sonst wird Standardsprache verwendet)
            
        Returns:
            Liste von numerischen Werten
        """
        language = language or self.language
        values = []
        
        if language in self.patterns:
            patterns = self.patterns[language]
            
            # Suche nach Wert-Mustern
            for pattern in patterns["value_patterns"]:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    # Extrahiere den numerischen Wert
                    if len(match.groups()) >= 1:
                        # Das letzte Gruppen-Match sollte der numerische Wert sein
                        value_str = match.group(len(match.groups()))
                        
                        try:
                            value = float(value_str)
                            values.append(value)
                        except ValueError:
                            # Ignoriere nicht-numerische Werte
                            pass
        
        return values
    
    def detect_context_references(self, text: str, language: Optional[str] = None) -> bool:
        """
        Prüft, ob der Text Referenzen auf vorherige Ergebnisse enthält.
        
        Args:
            text: Zu analysierender Text
            language: Sprachcode (optional, sonst wird Standardsprache verwendet)
            
        Returns:
            True, wenn Kontextreferenzen gefunden wurden, sonst False
        """
        language = language or self.language
        
        if language in self.patterns:
            patterns = self.patterns[language]
            
            # Suche nach Referenz-Mustern
            for pattern in patterns["reference_patterns"]:
                if re.search(pattern, text, re.IGNORECASE):
                    return True
        
        return False
    
    def parse(self, text: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Parst einen Text und extrahiert strukturierte Informationen.
        
        Args:
            text: Zu analysierender Text
            language: Sprachcode (optional, sonst wird Standardsprache verwendet)
            
        Returns:
            Dictionary mit strukturierten Informationen
        """
        # Verwende die angegebene Sprache oder die Standardsprache
        language = language or self.language
        
        # Erkenne den Modus (Befehl, Abfrage, Erklärung)
        mode = self.detect_mode(text, language)
        
        # Erkenne Operation mit dem OperationMapper
        operation_type, operation_params = self.operation_mapper.detect_operation(text)
        
        # Extrahiere Tensor-Informationen
        tensors = self.extract_tensors(text, language)
        
        # Extrahiere numerische Werte
        values = self.extract_values(text, language)
        
        # Prüfe auf Kontextreferenzen
        has_context_references = self.detect_context_references(text, language)
        
        # Erstelle strukturiertes Ergebnis
        result = {
            "text": text,
            "language": language,
            "mode": mode.value,
            "operation": {
                "type": operation_type.value,
                "params": operation_params
            },
            "tensors": tensors,
            "values": values,
            "has_context_references": has_context_references
        }
        
        # Speichere im Kontextgedächtnis
        self._update_context_memory(result)
        
        logger.info(f"Parsing abgeschlossen: Modus={mode.value}, Operation={operation_type.value}, "
                   f"#Tensoren={len(tensors)}, #Werte={len(values)}")
        
        return result
    
    def _update_context_memory(self, parsed_result: Dict[str, Any]):
        """
        Aktualisiert das Kontextgedächtnis mit dem neuen Ergebnis.
        
        Args:
            parsed_result: Geparste Eingabe
        """
        # Füge neues Ergebnis zum Gedächtnis hinzu
        self.context_memory.append(parsed_result)
        
        # Begrenze die Größe des Gedächtnisses
        if len(self.context_memory) > self.context_memory_size:
            self.context_memory.pop(0)  # Entferne das älteste Element
    
    def get_context(self) -> List[Dict[str, Any]]:
        """
        Gibt das aktuelle Kontextgedächtnis zurück.
        
        Returns:
            Liste mit gespeicherten Kontext-Elementen
        """
        return self.context_memory
    
    def clear_context(self):
        """Löscht das Kontextgedächtnis."""
        self.context_memory = []
        logger.info("Kontextgedächtnis gelöscht")
    
    def resolve_context_references(self, parsed_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Löst Kontextreferenzen auf, indem fehlende Informationen aus dem Kontext ergänzt werden.
        
        Args:
            parsed_result: Geparste Eingabe mit möglichen Kontextreferenzen
            
        Returns:
            Vervollständigtes Ergebnis
        """
        # Wenn keine Kontextreferenzen erkannt wurden oder kein Kontext vorhanden ist
        if not parsed_result["has_context_references"] or not self.context_memory:
            return parsed_result
        
        # Erstelle eine Kopie des ursprünglichen Ergebnisses
        resolved_result = parsed_result.copy()
        
        # Das letzte Kontext-Element (ohne das aktuelle Ergebnis)
        previous_context = None
        for ctx in reversed(self.context_memory[:-1]):  # Alle außer dem letzten (aktuellen) Element
            if ctx["operation"]["type"] != OperationType.UNKNOWN.value:
                previous_context = ctx
                break
        
        if not previous_context:
            return resolved_result
        
        # Wenn keine Tensoren gefunden wurden, übernehme sie aus dem Kontext
        if not resolved_result["tensors"] and previous_context["tensors"]:
            resolved_result["tensors"] = previous_context["tensors"]
        
        # Wenn keine Operation erkannt wurde, übernehme sie aus dem Kontext
        if resolved_result["operation"]["type"] == OperationType.UNKNOWN.value:
            resolved_result["operation"] = previous_context["operation"]
        
        # Wenn keine Werte gefunden wurden, übernehme sie aus dem Kontext
        if not resolved_result["values"] and previous_context["values"]:
            resolved_result["values"] = previous_context["values"]
        
        logger.info(f"Kontextreferenzen aufgelöst: {len(resolved_result['tensors'])} Tensoren, "
                   f"Operation={resolved_result['operation']['type']}")
        
        return resolved_result


# Testfunktion
def test_multilingual_parser():
    """Testet die Funktionalität des MultilingualParsers."""
    # Konfiguriere detailliertes Logging für den Test
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n=== Testing MultilingualParser ===")
    
    # Testbeispiele in verschiedenen Sprachen
    test_cases = [
        # Englisch
        ("Multiply matrix A with matrix B", "en"),
        ("What is the transpose of matrix C?", "en"),
        ("Explain how to add two vectors", "en"),
        ("Calculate the dot product of vector x and vector y using scalar 5", "en"),
        
        # Deutsch
        ("Multipliziere Matrix A mit Matrix B", "de"),
        ("Was ist die Transponierte der Matrix C?", "de"),
        ("Erkläre, wie man zwei Vektoren addiert", "de"),
        ("Berechne das Skalarprodukt von Vektor x und Vektor y mit Skalar 5", "de"),
        
        # Französisch
        ("Multiplie la matrice A avec la matrice B", "fr"),
        ("Quelle est la transposée de la matrice C?", "fr"),
        ("Explique comment ajouter deux vecteurs", "fr"),
        ("Calcule le produit scalaire du vecteur x et du vecteur y avec scalaire 5", "fr"),
        
        # Spanisch
        ("Multiplica la matriz A con la matriz B", "es"),
        ("¿Cuál es la transpuesta de la matriz C?", "es"),
        ("Explica cómo sumar dos vectores", "es"),
        ("Calcula el producto escalar del vector x y del vector y con escalar 5", "es")
    ]
    
    # Erstelle Parser-Instanzen für alle Sprachen
    parsers = {
        "en": MultilingualParser("en"),
        "de": MultilingualParser("de"),
        "fr": MultilingualParser("fr"),
        "es": MultilingualParser("es")
    }
    
    # Teste jeden Fall
    for text, language in test_cases:
        parser = parsers[language]
        
        print(f"\nSprache: {language}, Text: '{text}'")
        
        # Parse den Text
        result = parser.parse(text)
        
        # Zeige Ergebnis
        print(f"Modus: {result['mode']}")
        print(f"Operation: {result['operation']['type']}")
        print(f"Parameter: {result['operation']['params']}")
        print(f"Tensoren: {len(result['tensors'])}")
        for tensor in result['tensors']:
            print(f"  - {tensor['type'].capitalize()} '{tensor['name']}'")
        print(f"Werte: {result['values']}")
        print(f"Kontextreferenzen: {result['has_context_references']}")
    
    # Teste Kontextreferenzen
    print("\n=== Testing Context References ===")
    
    context_test_cases = [
        # Englisch
        ["Calculate A + B", "Now multiply it with C", "What's the transpose of the result?"],
        
        # Deutsch
        ["Berechne A + B", "Jetzt multipliziere es mit C", "Was ist die Transponierte des Ergebnisses?"]
    ]
    
    for test_case in context_test_cases:
        language = "en" if test_case[0].startswith("C") else "de"
        parser = parsers[language]
        parser.clear_context()
        
        print(f"\nKontexttest in {language.upper()}:")
        
        for i, text in enumerate(test_case):
            print(f"\nEingabe {i+1}: '{text}'")
            result = parser.parse(text)
            
            # Für Eingaben mit Kontextreferenzen
            if i > 0:
                resolved = parser.resolve_context_references(result)
                print(f"Operation vor Auflösung: {result['operation']['type']}")
                print(f"Operation nach Auflösung: {resolved['operation']['type']}")
                print(f"Tensoren vor Auflösung: {len(result['tensors'])}")
                print(f"Tensoren nach Auflösung: {len(resolved['tensors'])}")
            else:
                print(f"Operation: {result['operation']['type']}")
                print(f"Tensoren: {len(result['tensors'])}")
                for tensor in result['tensors']:
                    print(f"  - {tensor['type'].capitalize()} '{tensor['name']}'")


if __name__ == "__main__":
    # Führe Test aus
    test_multilingual_parser()
