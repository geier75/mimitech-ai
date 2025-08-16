#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - M-LINGUA Symbol Mapper

Dieses Modul implementiert den Symbol Mapper für das M-LINGUA Interface,
der Worte und Phrasen in semantische Symbole umwandelt.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Union, Tuple

logger = logging.getLogger("MISO.Ultimate.MLingua.SymbolMapper")

class SymbolMapper:
    """
    Symbol Mapper für M-LINGUA
    
    Diese Klasse wandelt Worte und Phrasen in semantische Symbole um.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialisiert den Symbol Mapper
        
        Args:
            config: Konfigurationsobjekt für den Symbol Mapper
        """
        self.config = config
        self.language = config.get("language", "de")
        
        # Lade Symbol-Tabelle
        self._load_symbol_table()
        
        logger.info(f"Symbol Mapper initialisiert mit Sprache: {self.language}")
    
    def _load_symbol_table(self):
        """Lädt die Symbol-Tabelle aus der Konfiguration"""
        # In einer vollständigen Implementierung würden hier Symbole aus
        # einer Datei oder Datenbank geladen werden. Für dieses Beispiel verwenden
        # wir einige hartcodierte Symbole.
        
        # Definiere grundlegende Symbole
        self.symbol_table = {
            "math": {
                "de": {
                    "tensor": ["tensor", "tensorfeld", "tensorobjekt"],
                    "matrix": ["matrix", "matrize", "matrixfeld"],
                    "vector": ["vektor", "vektorfeld"],
                    "scalar": ["skalar", "zahl", "wert"],
                    "add": ["addiere", "plus", "summiere", "erhöhe um"],
                    "subtract": ["subtrahiere", "minus", "verringere um"],
                    "multiply": ["multipliziere", "mal", "vervielfache"],
                    "divide": ["dividiere", "geteilt durch", "teile"],
                    "transpose": ["transponiere", "transposition"],
                    "inverse": ["invertiere", "kehrwert", "umkehrung"],
                    "determinant": ["determinante", "det"],
                    "eigenvalues": ["eigenwerte", "charakteristische werte"],
                    "eigenvectors": ["eigenvektoren", "charakteristische vektoren"],
                    "norm": ["norm", "betrag", "länge"],
                    "dot_product": ["skalarprodukt", "inneres produkt", "punktprodukt"],
                    "cross_product": ["kreuzprodukt", "vektorprodukt"],
                    "gradient": ["gradient", "steigung"],
                    "optimize": ["optimiere", "maximiere", "minimiere"]
                },
                "en": {
                    "tensor": ["tensor", "tensor field", "tensor object"],
                    "matrix": ["matrix", "matrices"],
                    "vector": ["vector", "vector field"],
                    "scalar": ["scalar", "number", "value"],
                    "add": ["add", "plus", "sum", "increase by"],
                    "subtract": ["subtract", "minus", "decrease by"],
                    "multiply": ["multiply", "times", "multiply by"],
                    "divide": ["divide", "divided by", "divide by"],
                    "transpose": ["transpose", "transposition"],
                    "inverse": ["invert", "inverse", "reciprocal"],
                    "determinant": ["determinant", "det"],
                    "eigenvalues": ["eigenvalues", "characteristic values"],
                    "eigenvectors": ["eigenvectors", "characteristic vectors"],
                    "norm": ["norm", "magnitude", "length"],
                    "dot_product": ["dot product", "inner product", "scalar product"],
                    "cross_product": ["cross product", "vector product"],
                    "gradient": ["gradient", "slope"],
                    "optimize": ["optimize", "maximize", "minimize"]
                }
            },
            "backend": {
                "de": {
                    "mlx": ["mlx", "apple neural engine", "ane", "neural engine", "m4", "m4 max"],
                    "torch": ["pytorch", "torch", "gpu", "metal", "mps"],
                    "numpy": ["numpy", "np", "cpu"]
                },
                "en": {
                    "mlx": ["mlx", "apple neural engine", "ane", "neural engine", "m4", "m4 max"],
                    "torch": ["pytorch", "torch", "gpu", "metal", "mps"],
                    "numpy": ["numpy", "np", "cpu"]
                }
            },
            "dtype": {
                "de": {
                    "float32": ["float32", "float", "einfache genauigkeit", "single"],
                    "float64": ["float64", "double", "doppelte genauigkeit", "double"],
                    "int32": ["int32", "int", "integer", "ganzzahl"],
                    "int64": ["int64", "long", "lange ganzzahl"]
                },
                "en": {
                    "float32": ["float32", "float", "single precision", "single"],
                    "float64": ["float64", "double", "double precision", "double"],
                    "int32": ["int32", "int", "integer"],
                    "int64": ["int64", "long", "long integer"]
                }
            }
        }
    
    def map(self, text: str, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Wandelt einen Text in semantische Symbole um
        
        Args:
            text: Eingabetext in natürlicher Sprache
            intent: Erkannte Absicht
            
        Returns:
            Liste von semantischen Symbolen
        """
        # Initialisiere Ergebnis
        symbols = []
        
        # Normalisiere Text
        text_lower = text.lower()
        
        # Extrahiere Symbole basierend auf Intent-Typ
        if intent["type"] == "math_operation":
            # Extrahiere mathematische Symbole
            math_symbols = self._extract_symbols(text_lower, "math")
            symbols.extend(math_symbols)
            
            # Extrahiere Backend-Symbole
            backend_symbols = self._extract_symbols(text_lower, "backend")
            symbols.extend(backend_symbols)
            
            # Extrahiere Datentyp-Symbole
            dtype_symbols = self._extract_symbols(text_lower, "dtype")
            symbols.extend(dtype_symbols)
        
        return symbols
    
    def _extract_symbols(self, text: str, category: str) -> List[Dict[str, Any]]:
        """
        Extrahiert Symbole einer bestimmten Kategorie aus einem Text
        
        Args:
            text: Eingabetext in natürlicher Sprache
            category: Kategorie der Symbole
            
        Returns:
            Liste von Symbolen
        """
        # Initialisiere Ergebnis
        symbols = []
        
        # Hole Symbole für die aktuelle Sprache und Kategorie
        category_symbols = self.symbol_table.get(category, {}).get(self.language, {})
        
        # Prüfe jedes Symbol
        for symbol_name, aliases in category_symbols.items():
            for alias in aliases:
                # Suche nach dem Alias im Text
                pattern = r'\b' + re.escape(alias) + r'\b'
                for match in re.finditer(pattern, text):
                    symbols.append({
                        "category": category,
                        "name": symbol_name,
                        "alias": alias,
                        "start": match.start(),
                        "end": match.end()
                    })
        
        return symbols
