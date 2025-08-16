#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - MPRIME Contextual Math Core

KI-gestützte, situationsabhängige Mathematik für die MPRIME Engine.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Union, Tuple, Set, Callable

logger = logging.getLogger("MISO.Math.MPRIME.ContextualMathCore")

class ContextualMathCore:
    """
    KI-gestützte, situationsabhängige Mathematik
    
    Diese Klasse implementiert kontextabhängige mathematische Operationen
    und Interpretationen basierend auf dem situativen Kontext.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den ContextualMathCore
        
        Args:
            config: Konfigurationsobjekt für den ContextualMathCore
        """
        self.config = config or {}
        
        # Kontexttypen
        self.context_types = {
            "scientific": self._scientific_context,
            "engineering": self._engineering_context,
            "financial": self._financial_context,
            "statistical": self._statistical_context,
            "geometric": self._geometric_context,
            "quantum": self._quantum_context,
            "logical": self._logical_context,
            "educational": self._educational_context
        }
        
        # Aktiver Kontext
        self.active_context = "scientific"
        
        # Kontextspezifische Variablen
        self.context_variables = {
            "scientific": {
                "c": 299792458,  # Lichtgeschwindigkeit (m/s)
                "h": 6.62607015e-34,  # Plancksches Wirkungsquantum (J·s)
                "G": 6.67430e-11,  # Gravitationskonstante (m³/(kg·s²))
                "e": 1.602176634e-19,  # Elementarladung (C)
                "N_A": 6.02214076e23  # Avogadro-Konstante (1/mol)
            },
            "engineering": {
                "g": 9.80665,  # Erdbeschleunigung (m/s²)
                "R": 8.31446261815324,  # Universelle Gaskonstante (J/(mol·K))
                "sigma": 5.670374419e-8,  # Stefan-Boltzmann-Konstante (W/(m²·K⁴))
                "mu_0": 1.25663706212e-6,  # Magnetische Feldkonstante (N/A²)
                "epsilon_0": 8.8541878128e-12  # Elektrische Feldkonstante (F/m)
            },
            "financial": {
                "r": 0.05,  # Zinssatz (5%)
                "inflation": 0.02,  # Inflationsrate (2%)
                "tax_rate": 0.3,  # Steuersatz (30%)
                "risk_free_rate": 0.01,  # Risikofreier Zinssatz (1%)
                "market_return": 0.08  # Marktrendite (8%)
            }
        }
        
        # Kontextspezifische Funktionen
        self.context_functions = {
            "scientific": {
                "energy_mass_equivalence": lambda m: m * (self.context_variables["scientific"]["c"] ** 2),
                "wavelength_frequency": lambda f: self.context_variables["scientific"]["c"] / f,
                "photon_energy": lambda f: self.context_variables["scientific"]["h"] * f
            },
            "financial": {
                "compound_interest": lambda P, r, t, n: P * (1 + r/n)**(n*t),
                "present_value": lambda FV, r, t: FV / (1 + r)**t,
                "future_value": lambda PV, r, t: PV * (1 + r)**t
            }
        }
        
        logger.info(f"ContextualMathCore initialisiert mit aktivem Kontext '{self.active_context}'")
    
    def set_context(self, context_type: str) -> bool:
        """
        Setzt den aktiven Kontext
        
        Args:
            context_type: Kontexttyp
            
        Returns:
            True, wenn erfolgreich, sonst False
        """
        if context_type not in self.context_types:
            logger.warning(f"Unbekannter Kontexttyp: {context_type}")
            return False
        
        self.active_context = context_type
        logger.info(f"Aktiver Kontext auf '{context_type}' gesetzt")
        return True
    
    def process(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Verarbeitet einen Eingabetext basierend auf dem mathematischen Kontext
        
        Diese Methode wird von der MPRIME Engine aufgerufen und bietet eine
        einheitliche Schnittstelle für die Verarbeitung von Ausdrücken.
        
        Args:
            input_text: Der zu verarbeitende Eingabetext/Ausdruck
            context: Zusätzlicher Kontext für die Verarbeitung (optional)
            
        Returns:
            Dictionary mit dem Verarbeitungsergebnis und relevanten Metadaten
        """
        # Initialisiere das Ergebnis
        result = {
            "success": False,
            "original_text": input_text,
            "processed_text": None,
            "context_type": self.active_context,
            "variables": {},
            "description": None,
            "error": None
        }
        
        try:
            # Prüfe, ob der Eingabetext gültig ist
            if not isinstance(input_text, str):
                if input_text is None:
                    input_text = ""
                else:
                    input_text = str(input_text)
            
            # Ignoriere leere Eingaben
            if not input_text.strip():
                result["description"] = "Leere Eingabe erhalten"
                return result
                
            # Bestimme den Kontexttyp (aus übergebenem Kontext oder aus aktivem Kontext)
            context_type = context.get("type", self.active_context) if context else self.active_context
            result["context_type"] = context_type
            
            # Bereite den Eingabetext für die Interpretation vor
            if context and "preprocess" in context and callable(context["preprocess"]):
                # Verwende benutzerdefinierte Vorverarbeitung, falls verfügbar
                processed_text = context["preprocess"](input_text)
            else:
                # Andernfalls führe Standardvorverarbeitung durch
                processed_text = self._preprocess_text(input_text)
                
            # Erstelle ein Dictionary für die Interpretation
            expr_dict = {"formula": processed_text}
            
            # Interpretiere den Ausdruck mit der vorhandenen interpret-Methode
            interpretation = self.interpret(expr_dict, {"type": context_type})
            
            # Übernehme die Ergebnisse in das Rückgabeformat
            result["success"] = True
            result["processed_text"] = interpretation["interpreted_expression"]
            result["variables"] = interpretation["variables"]
            result["description"] = interpretation["interpretation"]
            
            logger.info(f"Eingabetext '{input_text}' erfolgreich im Kontext '{context_type}' verarbeitet")
            
        except Exception as e:
            # Im Fehlerfall setze Fehlerinformationen und protokolliere sie
            result["error"] = str(e)
            logger.error(f"Fehler bei der Verarbeitung von '{input_text}': {str(e)}")
            
        return result
        
    def _preprocess_text(self, text: str) -> str:
        """
        Bereitet einen Text für die Verarbeitung vor
        
        Args:
            text: Der zu verarbeitende Text
            
        Returns:
            Der vorverarbeitete Text
        """
        # Entferne überflüssige Leerzeichen
        text = text.strip()
        
        # Ersetze mehrfache Leerzeichen durch ein einzelnes
        text = re.sub(r'\s+', ' ', text)
        
        # Standardisiere mathematische Operatoren
        replacements = {
            '×': '*',
            '÷': '/',
            '−': '-',
            '·': '*',
            '^': '**',
            '≈': '==',
            '≠': '!=',
            '≤': '<=',
            '≥': '>='
        }
        
        # Wende die Ersetzungen an
        for old, new in replacements.items():
            if old in text:
                text = text.replace(old, new)
                
        return text
    
    def interpret(self, expression: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Interpretiert einen mathematischen Ausdruck im aktuellen Kontext
        
        Args:
            expression: Mathematischer Ausdruck
            context: Zusätzlicher Kontext (optional)
            
        Returns:
            Dictionary mit der Interpretation und Metadaten
        """
        # Verwende aktiven Kontext, falls nicht angegeben
        context_type = context.get("type", self.active_context) if context else self.active_context
        
        # Initialisiere Ergebnis
        result = {
            "original_expression": expression,
            "context_type": context_type,
            "interpreted_expression": None,
            "variables": {},
            "functions": {},
            "interpretation": None
        }
        
        try:
            # Prüfe, ob Kontexttyp existiert
            if context_type not in self.context_types:
                raise ValueError(f"Unbekannter Kontexttyp: {context_type}")
            
            # Extrahiere Ausdrucksstring
            if isinstance(expression, dict) and "formula" in expression:
                expr_string = expression["formula"]
            elif isinstance(expression, dict) and "normalized_expression" in expression:
                expr_string = expression["normalized_expression"]
            elif isinstance(expression, str):
                expr_string = expression
            else:
                raise ValueError("Ungültiger Ausdruckstyp")
            
            # Wende kontextspezifische Interpretation an
            interpretation = self.context_types[context_type](expr_string, context)
            
            # Aktualisiere Ergebnis
            result["interpreted_expression"] = interpretation["expression"]
            result["variables"] = interpretation["variables"]
            result["functions"] = interpretation["functions"]
            result["interpretation"] = interpretation["description"]
            
            logger.info(f"Ausdruck erfolgreich im Kontext '{context_type}' interpretiert")
        
        except Exception as e:
            logger.error(f"Fehler bei der Interpretation des Ausdrucks im Kontext '{context_type}': {str(e)}")
            raise
        
        return result
    
    def _scientific_context(self, expression: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Interpretiert einen Ausdruck im wissenschaftlichen Kontext
        
        Args:
            expression: Ausdrucksstring
            context: Zusätzlicher Kontext
            
        Returns:
            Dictionary mit der Interpretation
        """
        # Initialisiere Ergebnis
        result = {
            "expression": expression,
            "variables": {},
            "functions": {},
            "description": "Wissenschaftliche Interpretation"
        }
        
        # Füge kontextspezifische Variablen hinzu
        for var, value in self.context_variables.get("scientific", {}).items():
            if var in expression:
                result["variables"][var] = value
        
        # Füge kontextspezifische Funktionen hinzu
        for func, func_obj in self.context_functions.get("scientific", {}).items():
            if func in expression:
                result["functions"][func] = func_obj
        
        # Interpretiere spezifische wissenschaftliche Ausdrücke
        if "E=mc^2" in expression or "E = mc^2" in expression:
            result["description"] = "Einsteins Energieäquivalenz: Energie (E) ist äquivalent zu Masse (m) multipliziert mit dem Quadrat der Lichtgeschwindigkeit (c²)."
        elif "F=ma" in expression or "F = ma" in expression:
            result["description"] = "Newtons zweites Gesetz: Kraft (F) ist gleich Masse (m) multipliziert mit Beschleunigung (a)."
        
        return result
    
    def _engineering_context(self, expression: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Interpretiert einen Ausdruck im technischen Kontext
        
        Args:
            expression: Ausdrucksstring
            context: Zusätzlicher Kontext
            
        Returns:
            Dictionary mit der Interpretation
        """
        # Initialisiere Ergebnis
        result = {
            "expression": expression,
            "variables": {},
            "functions": {},
            "description": "Technische Interpretation"
        }
        
        # Füge kontextspezifische Variablen hinzu
        for var, value in self.context_variables.get("engineering", {}).items():
            if var in expression:
                result["variables"][var] = value
        
        # Interpretiere spezifische technische Ausdrücke
        if "P=VI" in expression or "P = VI" in expression:
            result["description"] = "Elektrische Leistung: Leistung (P) ist gleich Spannung (V) multipliziert mit Strom (I)."
        elif "F=qvB" in expression or "F = qvB" in expression:
            result["description"] = "Lorentzkraft: Kraft (F) auf eine Ladung (q) mit Geschwindigkeit (v) in einem Magnetfeld (B)."
        
        return result
    
    def _financial_context(self, expression: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Interpretiert einen Ausdruck im finanziellen Kontext
        
        Args:
            expression: Ausdrucksstring
            context: Zusätzlicher Kontext
            
        Returns:
            Dictionary mit der Interpretation
        """
        # Initialisiere Ergebnis
        result = {
            "expression": expression,
            "variables": {},
            "functions": {},
            "description": "Finanzielle Interpretation"
        }
        
        # Füge kontextspezifische Variablen hinzu
        for var, value in self.context_variables.get("financial", {}).items():
            if var in expression:
                result["variables"][var] = value
        
        # Füge kontextspezifische Funktionen hinzu
        for func, func_obj in self.context_functions.get("financial", {}).items():
            if func in expression:
                result["functions"][func] = func_obj
        
        # Interpretiere spezifische finanzielle Ausdrücke
        if "FV=PV(1+r)^t" in expression or "FV = PV(1+r)^t" in expression:
            result["description"] = "Zinseszinsformel: Zukünftiger Wert (FV) ist gleich Barwert (PV) multipliziert mit (1 + Zinssatz (r)) hoch Zeit (t)."
        elif "NPV" in expression:
            result["description"] = "Nettogegenwartswert (NPV): Summe aller diskontierten zukünftigen Cashflows minus der anfänglichen Investition."
        
        return result
    
    def _statistical_context(self, expression: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Interpretiert einen Ausdruck im statistischen Kontext
        
        Args:
            expression: Ausdrucksstring
            context: Zusätzlicher Kontext
            
        Returns:
            Dictionary mit der Interpretation
        """
        # Initialisiere Ergebnis
        result = {
            "expression": expression,
            "variables": {},
            "functions": {},
            "description": "Statistische Interpretation"
        }
        
        # Interpretiere spezifische statistische Ausdrücke
        if "mu" in expression and "sigma" in expression:
            result["description"] = "Normalverteilung: Mittelwert (μ) und Standardabweichung (σ) definieren die Verteilung."
        elif "H_0" in expression or "H0" in expression:
            result["description"] = "Nullhypothese (H₀): Die zu testende Annahme in der Statistik."
        
        return result
    
    def _geometric_context(self, expression: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Interpretiert einen Ausdruck im geometrischen Kontext
        
        Args:
            expression: Ausdrucksstring
            context: Zusätzlicher Kontext
            
        Returns:
            Dictionary mit der Interpretation
        """
        # Initialisiere Ergebnis
        result = {
            "expression": expression,
            "variables": {},
            "functions": {},
            "description": "Geometrische Interpretation"
        }
        
        # Interpretiere spezifische geometrische Ausdrücke
        if "A=pi*r^2" in expression or "A = pi*r^2" in expression:
            result["description"] = "Kreisfläche: Fläche (A) ist gleich π multipliziert mit dem Quadrat des Radius (r)."
        elif "V=(4/3)*pi*r^3" in expression or "V = (4/3)*pi*r^3" in expression:
            result["description"] = "Kugelvolumen: Volumen (V) ist gleich (4/3) multipliziert mit π multipliziert mit der dritten Potenz des Radius (r)."
        
        return result
    
    def _quantum_context(self, expression: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Interpretiert einen Ausdruck im quantenmechanischen Kontext
        
        Args:
            expression: Ausdrucksstring
            context: Zusätzlicher Kontext
            
        Returns:
            Dictionary mit der Interpretation
        """
        # Initialisiere Ergebnis
        result = {
            "expression": expression,
            "variables": {},
            "functions": {},
            "description": "Quantenmechanische Interpretation"
        }
        
        # Interpretiere spezifische quantenmechanische Ausdrücke
        if "psi" in expression or "Ψ" in expression:
            result["description"] = "Wellenfunktion (Ψ): Beschreibt den Quantenzustand eines Systems."
        elif "H|psi" in expression or "H|Ψ" in expression:
            result["description"] = "Schrödingergleichung: Hamilton-Operator (H) angewendet auf die Wellenfunktion (Ψ)."
        
        return result
    
    def _logical_context(self, expression: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Interpretiert einen Ausdruck im logischen Kontext
        
        Args:
            expression: Ausdrucksstring
            context: Zusätzlicher Kontext
            
        Returns:
            Dictionary mit der Interpretation
        """
        # Initialisiere Ergebnis
        result = {
            "expression": expression,
            "variables": {},
            "functions": {},
            "description": "Logische Interpretation"
        }
        
        # Interpretiere spezifische logische Ausdrücke
        if "AND" in expression or "∧" in expression:
            result["description"] = "Logisches UND: Beide Aussagen müssen wahr sein, damit das Ergebnis wahr ist."
        elif "OR" in expression or "∨" in expression:
            result["description"] = "Logisches ODER: Mindestens eine Aussage muss wahr sein, damit das Ergebnis wahr ist."
        elif "NOT" in expression or "¬" in expression:
            result["description"] = "Logische Negation: Kehrt den Wahrheitswert einer Aussage um."
        
        return result
    
    def _educational_context(self, expression: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Interpretiert einen Ausdruck im pädagogischen Kontext
        
        Args:
            expression: Ausdrucksstring
            context: Zusätzlicher Kontext
            
        Returns:
            Dictionary mit der Interpretation
        """
        # Initialisiere Ergebnis
        result = {
            "expression": expression,
            "variables": {},
            "functions": {},
            "description": "Pädagogische Interpretation"
        }
        
        # Interpretiere spezifische pädagogische Ausdrücke
        if "+" in expression:
            result["description"] = "Addition: Zusammenzählen von Werten."
        elif "-" in expression:
            result["description"] = "Subtraktion: Abziehen eines Wertes von einem anderen."
        elif "*" in expression:
            result["description"] = "Multiplikation: Wiederholte Addition desselben Wertes."
        elif "/" in expression:
            result["description"] = "Division: Aufteilen eines Wertes in gleiche Teile."
        
        return result
    
    def apply_context_transformation(self, expression: Dict[str, Any], context_type: str = None) -> Dict[str, Any]:
        """
        Wendet eine kontextspezifische Transformation auf einen Ausdruck an
        
        Args:
            expression: Mathematischer Ausdruck
            context_type: Kontexttyp (optional)
            
        Returns:
            Dictionary mit dem transformierten Ausdruck und Metadaten
        """
        # Verwende aktiven Kontext, falls nicht angegeben
        context_type = context_type or self.active_context
        
        # Initialisiere Ergebnis
        result = {
            "original_expression": expression,
            "context_type": context_type,
            "transformed_expression": None,
            "transformation_type": None,
            "transformation_description": None
        }
        
        try:
            # Prüfe, ob Kontexttyp existiert
            if context_type not in self.context_types:
                raise ValueError(f"Unbekannter Kontexttyp: {context_type}")
            
            # Extrahiere Ausdrucksstring
            if isinstance(expression, dict) and "formula" in expression:
                expr_string = expression["formula"]
            elif isinstance(expression, dict) and "normalized_expression" in expression:
                expr_string = expression["normalized_expression"]
            elif isinstance(expression, str):
                expr_string = expression
            else:
                raise ValueError("Ungültiger Ausdruckstyp")
            
            # Wende kontextspezifische Transformation an
            if context_type == "scientific":
                transformed = self._transform_scientific(expr_string)
            elif context_type == "engineering":
                transformed = self._transform_engineering(expr_string)
            elif context_type == "financial":
                transformed = self._transform_financial(expr_string)
            else:
                # Standardtransformation für andere Kontexte
                transformed = {
                    "expression": expr_string,
                    "type": "identity",
                    "description": f"Identitätstransformation im Kontext '{context_type}'"
                }
            
            # Aktualisiere Ergebnis
            result["transformed_expression"] = transformed["expression"]
            result["transformation_type"] = transformed["type"]
            result["transformation_description"] = transformed["description"]
            
            logger.info(f"Ausdruck erfolgreich im Kontext '{context_type}' transformiert")
        
        except Exception as e:
            logger.error(f"Fehler bei der Transformation des Ausdrucks im Kontext '{context_type}': {str(e)}")
            raise
        
        return result
    
    def _transform_scientific(self, expression: str) -> Dict[str, Any]:
        """
        Transformiert einen Ausdruck im wissenschaftlichen Kontext
        
        Args:
            expression: Ausdrucksstring
            
        Returns:
            Dictionary mit dem transformierten Ausdruck
        """
        # In einer vollständigen Implementierung würde hier eine komplexe
        # Transformation im wissenschaftlichen Kontext stehen
        
        # Einfache Implementierung für dieses Beispiel
        transformed = expression
        
        # Ersetze Variablen durch ihre wissenschaftlichen Werte
        for var, value in self.context_variables.get("scientific", {}).items():
            pattern = r'\b' + re.escape(var) + r'\b'
            transformed = re.sub(pattern, str(value), transformed)
        
        return {
            "expression": transformed,
            "type": "scientific_substitution",
            "description": "Wissenschaftliche Variablen wurden durch ihre Werte ersetzt"
        }
    
    def _transform_engineering(self, expression: str) -> Dict[str, Any]:
        """
        Transformiert einen Ausdruck im technischen Kontext
        
        Args:
            expression: Ausdrucksstring
            
        Returns:
            Dictionary mit dem transformierten Ausdruck
        """
        # In einer vollständigen Implementierung würde hier eine komplexe
        # Transformation im technischen Kontext stehen
        
        # Einfache Implementierung für dieses Beispiel
        transformed = expression
        
        # Ersetze Variablen durch ihre technischen Werte
        for var, value in self.context_variables.get("engineering", {}).items():
            pattern = r'\b' + re.escape(var) + r'\b'
            transformed = re.sub(pattern, str(value), transformed)
        
        return {
            "expression": transformed,
            "type": "engineering_substitution",
            "description": "Technische Variablen wurden durch ihre Werte ersetzt"
        }
    
    def _transform_financial(self, expression: str) -> Dict[str, Any]:
        """
        Transformiert einen Ausdruck im finanziellen Kontext
        
        Args:
            expression: Ausdrucksstring
            
        Returns:
            Dictionary mit dem transformierten Ausdruck
        """
        # In einer vollständigen Implementierung würde hier eine komplexe
        # Transformation im finanziellen Kontext stehen
        
        # Einfache Implementierung für dieses Beispiel
        transformed = expression
        
        # Ersetze Variablen durch ihre finanziellen Werte
        for var, value in self.context_variables.get("financial", {}).items():
            pattern = r'\b' + re.escape(var) + r'\b'
            transformed = re.sub(pattern, str(value), transformed)
        
        # Ersetze spezifische finanzielle Ausdrücke
        transformed = transformed.replace("NPV", "Net Present Value")
        transformed = transformed.replace("IRR", "Internal Rate of Return")
        
        return {
            "expression": transformed,
            "type": "financial_substitution",
            "description": "Finanzielle Variablen und Begriffe wurden ersetzt"
        }
