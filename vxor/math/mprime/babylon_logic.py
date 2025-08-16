#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - MPRIME Babylon Logic Core

Babylonisches Zahlensystem-Support (Basis 60 & Hybrid) für die MPRIME Engine.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import logging
import math
import re
from typing import Dict, Any, List, Optional, Union, Tuple

logger = logging.getLogger("MISO.Math.MPRIME.BabylonLogicCore")

class BabylonLogicCore:
    """
    Babylonisches Zahlensystem-Support (Basis 60 & Hybrid)
    
    Diese Klasse implementiert das babylonische Zahlensystem (Basis 60)
    und hybride Zahlensysteme für mathematische Berechnungen.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den BabylonLogicCore
        
        Args:
            config: Konfigurationsobjekt für den BabylonLogicCore
        """
        self.config = config or {}
        self.current_base = 10  # Standardbasis
        self.babylon_symbols = self._create_babylon_symbols()
        self.babylon_fractions = self._create_babylon_fractions()
        
        logger.info("BabylonLogicCore initialisiert")
    
    def _create_babylon_symbols(self) -> Dict[int, str]:
        """
        Erstellt babylonische Symbole für Zahlen
        
        Returns:
            Dictionary mit babylonischen Symbolen
        """
        # In einer vollständigen Implementierung würden hier echte
        # babylonische Symbole verwendet werden
        
        # Einfache Implementierung für dieses Beispiel
        symbols = {}
        
        # Symbole für 1-9
        for i in range(1, 10):
            symbols[i] = "𒐕" * i
        
        # Symbol für 10
        symbols[10] = "𒌋"
        
        # Symbole für 11-59
        for i in range(11, 60):
            tens = i // 10
            ones = i % 10
            symbols[i] = symbols[10] * tens
            if ones > 0:
                symbols[i] += symbols[ones]
        
        return symbols
    
    def _create_babylon_fractions(self) -> Dict[int, float]:
        """
        Erstellt babylonische Brüche
        
        Returns:
            Dictionary mit babylonischen Brüchen
        """
        # Babylonische Brüche sind Stammbrüche (1/n) und deren Kombinationen
        fractions = {}
        
        # Stammbrüche
        for i in range(2, 61):
            fractions[i] = 1.0 / i
        
        # Spezielle Brüche
        fractions[120] = 1.0 / 120  # 1/2 * 1/60
        fractions[180] = 1.0 / 180  # 1/3 * 1/60
        
        return fractions
    
    def use_base(self, base: int) -> bool:
        """
        Setzt die zu verwendende Zahlenbasis
        
        Args:
            base: Zahlenbasis (10, 60, etc.)
            
        Returns:
            True, wenn erfolgreich, sonst False
        """
        if base not in [10, 60]:
            logger.warning(f"Unbekannte Basis: {base}, verwende Standardbasis 10")
            self.current_base = 10
            return False
        
        self.current_base = base
        logger.info(f"Basis auf {base} gesetzt")
        return True
    
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transformiert Daten in das babylonische System
        
        Args:
            data: Eingabedaten (symbolischer Baum oder Zahlen)
            
        Returns:
            Transformierte Daten
        """
        # Initialisiere Ergebnis
        result = {
            "original_data": data,
            "babylon_data": None,
            "base": self.current_base,
            "system_type": "babylon"
        }
        
        try:
            # Transformiere Daten
            if isinstance(data, dict) and "symbol_tree" in data:
                # Symbolischer Baum
                babylon_data = self._transform_symbol_tree(data)
            elif isinstance(data, (int, float)):
                # Einzelne Zahl
                babylon_data = self._decimal_to_babylon(data)
            elif isinstance(data, str) and self._is_numeric(data):
                # Numerischer String
                babylon_data = self._decimal_to_babylon(float(data))
            else:
                # Anderer Datentyp
                babylon_data = self._transform_generic(data)
            
            result["babylon_data"] = babylon_data
            
            logger.info(f"Daten erfolgreich in das babylonische System (Basis {self.current_base}) transformiert")
        
        except Exception as e:
            logger.error(f"Fehler bei der Transformation in das babylonische System: {str(e)}")
            raise
        
        return result
    
    def _transform_symbol_tree(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transformiert einen symbolischen Baum in das babylonische System
        
        Args:
            data: Symbolischer Baum
            
        Returns:
            Transformierter symbolischer Baum
        """
        # Kopiere Daten
        result = data.copy()
        
        # Füge babylonische Informationen hinzu
        result["number_system"] = {
            "type": "babylon",
            "base": self.current_base
        }
        
        # Transformiere Konstanten
        if "constants" in result:
            babylon_constants = []
            for constant in result["constants"]:
                if self._is_numeric(constant):
                    # Konvertiere in babylonisches System
                    babylon_constant = self._decimal_to_babylon(float(constant))
                    babylon_constants.append(babylon_constant)
                else:
                    # Behalte nicht-numerische Konstanten bei
                    babylon_constants.append(constant)
            
            result["babylon_constants"] = babylon_constants
        
        # In einer vollständigen Implementierung würde hier eine komplexe
        # Transformation des gesamten symbolischen Baums stehen
        
        return result
    
    def _transform_generic(self, data: Any) -> Any:
        """
        Transformiert generische Daten in das babylonische System
        
        Args:
            data: Generische Daten
            
        Returns:
            Transformierte Daten
        """
        # In einer vollständigen Implementierung würde hier eine komplexe
        # Transformation für verschiedene Datentypen stehen
        
        # Einfache Implementierung für dieses Beispiel
        return {
            "original": data,
            "babylon_transform_applied": True,
            "base": self.current_base
        }
    
    def _decimal_to_babylon(self, value: float) -> Dict[str, Any]:
        """
        Konvertiert eine Dezimalzahl in das babylonische System
        
        Args:
            value: Dezimalzahl
            
        Returns:
            Babylonische Darstellung
        """
        # Trenne ganzzahligen Teil und Bruchteil
        integer_part = int(value)
        fractional_part = value - integer_part
        
        # Konvertiere ganzzahligen Teil
        if self.current_base == 60:
            # Basis 60 (Sexagesimal)
            babylon_integer = self._decimal_to_sexagesimal(integer_part)
        else:
            # Basis 10 (Dezimal)
            babylon_integer = integer_part
        
        # Konvertiere Bruchteil
        if fractional_part > 0:
            if self.current_base == 60:
                # Basis 60 (Sexagesimal)
                babylon_fraction = self._decimal_fraction_to_sexagesimal(fractional_part)
            else:
                # Basis 10 (Dezimal)
                babylon_fraction = fractional_part
        else:
            babylon_fraction = 0
        
        # Erstelle babylonische Darstellung
        babylon_representation = {
            "value": value,
            "base": self.current_base,
            "integer_part": babylon_integer,
            "fractional_part": babylon_fraction
        }
        
        # Füge symbolische Darstellung hinzu, falls Basis 60
        if self.current_base == 60:
            babylon_representation["symbolic"] = self._get_babylon_symbols(babylon_integer)
        
        return babylon_representation
    
    def _decimal_to_sexagesimal(self, value: int) -> List[int]:
        """
        Konvertiert eine Dezimalzahl in das Sexagesimalsystem (Basis 60)
        
        Args:
            value: Dezimalzahl
            
        Returns:
            Liste von Sexagesimalziffern
        """
        if value == 0:
            return [0]
        
        digits = []
        temp = abs(value)
        
        while temp > 0:
            digits.append(temp % 60)
            temp //= 60
        
        # Negatives Vorzeichen berücksichtigen
        if value < 0:
            digits[-1] = -digits[-1]
        
        # Umkehren, da die signifikantesten Ziffern zuerst kommen sollen
        return digits[::-1]
    
    def _decimal_fraction_to_sexagesimal(self, value: float, precision: int = 5) -> List[int]:
        """
        Konvertiert einen Dezimalbruch in das Sexagesimalsystem (Basis 60)
        
        Args:
            value: Dezimalbruch (0 <= value < 1)
            precision: Anzahl der Sexagesimalstellen
            
        Returns:
            Liste von Sexagesimalziffern
        """
        if value == 0:
            return [0]
        
        digits = []
        temp = value
        
        for _ in range(precision):
            temp *= 60
            digit = int(temp)
            digits.append(digit)
            temp -= digit
            
            if temp == 0:
                break
        
        return digits
    
    def _get_babylon_symbols(self, digits: List[int]) -> str:
        """
        Gibt die symbolische Darstellung einer babylonischen Zahl zurück
        
        Args:
            digits: Liste von Sexagesimalziffern
            
        Returns:
            Symbolische Darstellung
        """
        if not digits:
            return ""
        
        symbols = []
        
        for digit in digits:
            if digit == 0:
                symbols.append("𒁁")  # Symbol für 0 (eigentlich ein Platzhalter)
            elif 1 <= abs(digit) <= 59:
                symbols.append(self.babylon_symbols[abs(digit)])
            else:
                # Sollte nicht vorkommen, da alle Ziffern < 60 sein sollten
                symbols.append("?")
        
        # Negatives Vorzeichen berücksichtigen
        if digits[0] < 0:
            return "-" + " ".join(symbols)
        
        return " ".join(symbols)
    
    def _is_numeric(self, value: str) -> bool:
        """
        Prüft, ob ein String numerisch ist
        
        Args:
            value: Zu prüfender String
            
        Returns:
            True, wenn numerisch, sonst False
        """
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def babylon_to_decimal(self, babylon_value: Dict[str, Any]) -> float:
        """
        Konvertiert eine babylonische Zahl in eine Dezimalzahl
        
        Args:
            babylon_value: Babylonische Zahl
            
        Returns:
            Dezimalzahl
        """
        # Prüfe Eingabe
        if not isinstance(babylon_value, dict):
            raise ValueError("Babylonische Zahl muss als Dictionary vorliegen")
        
        # Extrahiere Komponenten
        base = babylon_value.get("base", 60)
        integer_part = babylon_value.get("integer_part", [0])
        fractional_part = babylon_value.get("fractional_part", [0])
        
        # Konvertiere ganzzahligen Teil
        decimal_integer = 0
        if isinstance(integer_part, list):
            # Sexagesimaldarstellung
            for i, digit in enumerate(integer_part):
                power = len(integer_part) - i - 1
                decimal_integer += digit * (base ** power)
        else:
            # Dezimaldarstellung
            decimal_integer = integer_part
        
        # Konvertiere Bruchteil
        decimal_fraction = 0
        if isinstance(fractional_part, list):
            # Sexagesimaldarstellung
            for i, digit in enumerate(fractional_part):
                power = i + 1
                decimal_fraction += digit * (base ** -power)
        else:
            # Dezimaldarstellung
            decimal_fraction = fractional_part
        
        # Kombiniere ganzzahligen Teil und Bruchteil
        return decimal_integer + decimal_fraction
    
    def add_babylon(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        """
        Addiert zwei babylonische Zahlen
        
        Args:
            a: Erste babylonische Zahl
            b: Zweite babylonische Zahl
            
        Returns:
            Summe als babylonische Zahl
        """
        # Konvertiere in Dezimal
        decimal_a = self.babylon_to_decimal(a)
        decimal_b = self.babylon_to_decimal(b)
        
        # Addiere
        decimal_sum = decimal_a + decimal_b
        
        # Konvertiere zurück in babylonisches System
        return self._decimal_to_babylon(decimal_sum)
    
    def multiply_babylon(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        """
        Multipliziert zwei babylonische Zahlen
        
        Args:
            a: Erste babylonische Zahl
            b: Zweite babylonische Zahl
            
        Returns:
            Produkt als babylonische Zahl
        """
        # Konvertiere in Dezimal
        decimal_a = self.babylon_to_decimal(a)
        decimal_b = self.babylon_to_decimal(b)
        
        # Multipliziere
        decimal_product = decimal_a * decimal_b
        
        # Konvertiere zurück in babylonisches System
        return self._decimal_to_babylon(decimal_product)
