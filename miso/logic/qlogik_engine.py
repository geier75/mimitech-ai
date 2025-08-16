#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Q-LOGIK Engine

Symbolisch-fuzzy-bayes'sches Entscheidungsmodul für MISO.
Kognitives Entscheidungszentrum für Unsicherheit, Wahrscheinlichkeiten & symbolisches Konfliktmanagement.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import json
import logging
import math
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import os
import sys

# Importiere Speicheroptimierung
from miso.logic.qlogik_memory_optimization import (
    get_from_cache, put_in_cache, clear_cache, register_lazy_loader
)

# Pfad zur Konfigurationsdatei
RULESET_PATH = os.path.join(os.path.dirname(__file__), "qlogik_ruleset.json")

# Logger einrichten
logger = logging.getLogger("MISO.Logic.Q-LOGIK")

class BayesianDecisionCore:
    """
    Hochoptimierter Bayesianischer Entscheidungskern
    
    Berechnet Wahrscheinlichkeiten aus unscharfen Daten mit
    fortschrittlichem Cache, vektorisierten Operationen und
    hierarchischer Datenverarbeitung für maximale Performance.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den BayesianDecisionCore mit erweiterter Konfiguration
        
        Args:
            config: Konfigurationsobjekt für den BayesianDecisionCore
        """
        self.config = config or {}
        self.priors = self.config.get("priors", {})
        self.cache_ttl = self.config.get("cache_ttl", 3600)  # Cache-Lebensdauer in Sekunden
        self.use_vector_calc = self.config.get("use_vector_calc", True)  # Vektorisierte Berechnung
        
        # Cache-Statistik
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Vorbereitung für Numpy-Vektorisierung
        if len(self.priors) > 0 and self.use_vector_calc:
            self._prepare_vector_calc()
            
        logger.info("BayesianDecisionCore initialisiert")
    
    def _prepare_vector_calc(self):
        """
        Bereitet Datenstrukturen für vektorisierte Berechnungen vor
        """
        # Konvertiere Priors in Numpy-Arrays für schnelle Berechnung
        self._hypotheses = np.array(list(self.priors.keys()))
        self._prior_values = np.array(list(self.priors.values()))
        # Numerische Stabilität durch Clipping
        self._prior_values = np.clip(self._prior_values, 0.01, 0.99)
    
    def _generate_cache_key(self, data: Dict[str, Any]) -> str:
        """
        Erzeugt einen robusten, deterministischen Cache-Schlüssel für Eingabedaten
        mit besserer Unterstützung für komplexe verschachtelte Strukturen
        
        Args:
            data: Eingabedaten für die Berechnung
            
        Returns:
            Cache-Schlüssel als String
        """
        if not data:
            return "bayes_default"
            
        # Extrahiere Schlüsselelemente von Daten
        hypothesis = data.get("hypothesis", "default")
        
        # Verwende eine stringbasierte Repräsentation für das Evidenzset
        evidence_hash = self._hash_complex_structure(data.get("evidence", {}))
        
        # Kombiniere zum finalen Schlüssel
        return f"bayes_{hypothesis}_{evidence_hash}"
    
    def _hash_complex_structure(self, obj: Any) -> str:
        """
        Erzeugt einen deterministischen Hash-String für beliebige Datenstrukturen,
        einschließlich verschachtelter Dictionaries und Listen
        
        Args:
            obj: Zu hashende Datenstruktur
            
        Returns:
            Deterministischer Hash als String
        """
        if obj is None:
            return "n"
        elif isinstance(obj, (int, float, bool, str)):
            # Für primitive Typen, direkte Stringkonvertierung mit Typpräfix
            type_prefix = "i" if isinstance(obj, int) else "f" if isinstance(obj, float) else \
                         "b" if isinstance(obj, bool) else "s"
            
            # Runde Floats auf 5 Nachkommastellen für Konsistenz
            if isinstance(obj, float):
                return f"{type_prefix}{obj:.5f}"
            return f"{type_prefix}{obj}"
        elif isinstance(obj, dict):
            # Sortiere Dictionary-Keys alphabetisch für Konsistenz
            sorted_items = sorted(obj.items())
            # Generiere Hash für jeden Key-Value-Paar und verbinde sie
            items_hash = "_".join([f"{k}:{self._hash_complex_structure(v)}" for k, v in sorted_items])
            return f"d{hash(items_hash)}"
        elif isinstance(obj, (list, tuple)):
            # Für Listen/Tuples, hashe jedes Element und verbinde sie
            items_hash = "_".join([self._hash_complex_structure(item) for item in obj])
            container_type = "l" if isinstance(obj, list) else "t"
            return f"{container_type}{hash(items_hash)}"
        else:
            # Für andere Objekte, verwende Stringrepräsentation (mit Warnung)
            try:
                return f"o{hash(str(obj))}"
            except Exception:
                # Fallback bei nicht hashbaren Objekten
                logger.warning(f"Nicht-hashbarer Typ in Cache-Key: {type(obj).__name__}")
                return f"o{id(obj)}"
    
    def evaluate(self, data: Dict[str, Any]) -> float:
        """
        Berechnet die Wahrscheinlichkeit einer Hypothese basierend auf den Daten
        Hochoptimierte Implementierung mit verbesserter Caching-Strategie und 
        vektorisierten Berechnungen für maximale Performance
        
        Args:
            data: Eingabedaten für die Berechnung
            
        Returns:
            Wahrscheinlichkeitswert P(H|D)
        """
        # Optimierter Cache-Key für schnellen Lookup
        cache_key = self._generate_cache_key(data)
        cached_result = get_from_cache(cache_key)
        
        if cached_result is not None:
            self._cache_hits += 1
            return cached_result
            
        self._cache_misses += 1
        
        # Extrahiere Daten
        hypothesis = data.get("hypothesis", "default")
        evidence = data.get("evidence", {})
        
        # Prior-Wahrscheinlichkeit P(H) mit Regularisierung
        prior = self.priors.get(hypothesis, 0.5)
        # Verhindere extreme Werte für numerische Stabilität
        prior = max(0.01, min(0.99, prior))
        
        # Likelihood P(D|H) - hochoptimierte Berechnung
        likelihood = self._calculate_likelihood(evidence, hypothesis)
        
        # Marginale Wahrscheinlichkeit P(D) - vektorisierte Berechnung wenn möglich
        if self.use_vector_calc and len(self.priors) > 1 and hypothesis in self.priors:
            marginal = self._calculate_marginal_vectorized(evidence)
        elif len(self.priors) > 1 and hypothesis in self.priors:
            # Optimierte iterative Berechnung
            marginal = self._calculate_marginal_iterative(evidence)
        else:
            # Effiziente Fallback-Berechnung
            marginal = max(0.01, (likelihood * prior + (1 - likelihood) * (1 - prior)))
        
        # Bayes-Theorem mit numerischer Stabilität
        # Logarithmische Berechnung für bessere Präzision bei sehr kleinen Werten
        if likelihood < 1e-7 or prior < 1e-7 or marginal < 1e-7:
            # Log-Domain-Berechnung für bessere numerische Stabilität
            log_likelihood = np.log(max(1e-10, likelihood))
            log_prior = np.log(max(1e-10, prior))
            log_marginal = np.log(max(1e-10, marginal))
            log_posterior = log_likelihood + log_prior - log_marginal
            posterior = np.exp(log_posterior)
        else:
            posterior = (likelihood * prior) / marginal
        
        # Sanfte Begrenzung für bessere Differenzierbarkeit und numerische Stabilität
        result = min(0.99, max(0.01, posterior))
        
        # Speichere Ergebnis im Cache mit TTL
        put_in_cache(cache_key, result, ttl=self.cache_ttl)
        
        return result
    
    def _calculate_marginal_vectorized(self, evidence: Dict[str, Any]) -> float:
        """
        Berechnet die marginale Wahrscheinlichkeit P(D) mit vektorisierten Operationen
        
        Args:
            evidence: Beobachtete Daten
            
        Returns:
            Marginale Wahrscheinlichkeit
        """
        # Vektorisierte Likelihood-Berechnung für alle Hypothesen
        likelihoods = np.zeros_like(self._prior_values)
        
        for i, hypothesis in enumerate(self._hypotheses):
            likelihoods[i] = self._calculate_likelihood(evidence, hypothesis)
        
        # Vektorisierte Berechnung von P(D) = ∑ P(D|H_i) * P(H_i)
        marginal = np.sum(likelihoods * self._prior_values)
        
        # Numerische Stabilität
        return max(0.01, marginal)
    
    def _calculate_marginal_iterative(self, evidence: Dict[str, Any]) -> float:
        """
        Berechnet die marginale Wahrscheinlichkeit P(D) iterativ mit Optimierungen
        
        Args:
            evidence: Beobachtete Daten
            
        Returns:
            Marginale Wahrscheinlichkeit
        """
        # Verwende einen lokalen Cache für wiederholte Likelihood-Berechnungen
        likelihood_cache = {}
        marginal = 0.0
        
        for alt_hyp, alt_prior in self.priors.items():
            # Prüfe, ob die Likelihood bereits berechnet wurde
            if alt_hyp not in likelihood_cache:
                likelihood_cache[alt_hyp] = self._calculate_likelihood(evidence, alt_hyp)
                
            marginal += likelihood_cache[alt_hyp] * alt_prior
            
        # Stelle sicher, dass marginal nicht zu klein ist
        return max(0.01, marginal)
    
    def _calculate_likelihood(self, evidence: Dict[str, Any], hypothesis: str) -> float:
        """
        Berechnet die Likelihood P(D|H) mit optimierten Vektoroperationen
        
        Args:
            evidence: Beobachtete Daten
            hypothesis: Hypothese
            
        Returns:
            Likelihood-Wert
        """
        # Cache-Key für die Likelihood-Berechnung mit robustem Hashing
        evidence_hash = self._hash_complex_structure(evidence)
        cache_key = f"likelihood_{hypothesis}_{evidence_hash}"
        cached_result = get_from_cache(cache_key)
        
        if cached_result is not None:
            return cached_result
        
        if not evidence:
            result = 0.5
            put_in_cache(cache_key, result, ttl=self.cache_ttl)
            return result
        
        # Optimierte Berechnung für gewichtete Summe
        if self.use_vector_calc and any(isinstance(v, dict) and "value" in v and "weight" in v for v in evidence.values()):
            # Vektorisierte Berechnung für eine Menge von Evidenz-Elementen
            values = []
            weights = []
            
            for key, value in evidence.items():
                if isinstance(value, dict) and "value" in value and "weight" in value:
                    values.append(value["value"])
                    weights.append(value["weight"])
            
            if not values:  # Keine gültigen Werte gefunden
                result = 0.5
            else:
                # Konvertiere zu Numpy-Arrays für schnelle Berechnung
                values_array = np.array(values, dtype=np.float64)
                weights_array = np.array(weights, dtype=np.float64)
                
                # Normalisiere Gewichte
                total_weight = np.sum(weights_array)
                if total_weight == 0:
                    result = 0.5
                else:
                    # Vektorisierte gewichtete Summe
                    result = np.sum(values_array * weights_array) / total_weight
        else:
            # Fallback für einfachere Fälle oder wenn keine Vektorisierung aktiviert ist
            total_weight = 0
            weighted_sum = 0
            
            for key, value in evidence.items():
                if isinstance(value, dict) and "value" in value and "weight" in value:
                    weight = value["weight"]
                    evidence_value = value["value"]
                    total_weight += weight
                    weighted_sum += weight * evidence_value
            
            if total_weight == 0:
                result = 0.5
            else:
                result = weighted_sum / total_weight
        
        # Begrenzen für numerische Stabilität
        result = min(0.99, max(0.01, result))
        
        # Cache das Ergebnis
        put_in_cache(cache_key, result, ttl=self.cache_ttl)
        
        return result
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Liefert Cache-Statistiken für Performance-Analyse
        
        Returns:
            Dictionary mit Cache-Hits und Misses
        """
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_ratio": self._cache_hits / max(1, (self._cache_hits + self._cache_misses))
        }
    
    def clear_cache_stats(self) -> None:
        """
        Setzt Cache-Statistiken zurück
        """
        self._cache_hits = 0
        self._cache_misses = 0

class FuzzyLogicUnit:
    """
    Fuzzy-Logik-Einheit
    
    Arbeitet mit Wahrheitsgraden zwischen 0 und 1 statt mit
    booleschen Werten und ermöglicht unscharfe Entscheidungen.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert die FuzzyLogicUnit
        
        Args:
            config: Konfigurationsobjekt für die FuzzyLogicUnit
        """
        self.config = config or {}
        self.membership_functions = self.config.get("membership_functions", {})
        logger.info("FuzzyLogicUnit initialisiert")
        
    def score(self, signal: Dict[str, Any]) -> float:
        """
        Berechnet den Wahrheitsgrad eines Signals mit optimierter Performance
        
        Args:
            signal: Eingabesignal für die Berechnung
            
        Returns:
            Wahrheitsgrad zwischen 0 und 1
        """
        if not signal:
            return 0.5
        
        # Cache-Key für häufig wiederkehrende Signalmuster
        cache_key = f"fuzzy_{hash(frozenset(sorted(str(signal.items()))))}" if signal else "fuzzy_default"
        cached_result = get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Optimierte Anwendung von Fuzzy-Regeln auf das Signal
        # Verwende List Comprehension für höhere Performance
        truth_values = [
            self._apply_membership_function(key, value)
            for key, value in signal.items()
            if key in self.membership_functions
        ]
        
        if not truth_values:
            return 0.5
        
        # Kombinationsmethode aus der Konfiguration wählen
        combination_method = self.config.get("combination_method", "mean")
        
        if combination_method == "mean" or combination_method == "average":
            # Schnelle Berechnung des Durchschnitts
            result = sum(truth_values) / len(truth_values)
        elif combination_method == "min":
            # Minimum (für UND-ähnliche Operationen)
            result = min(truth_values)
        elif combination_method == "max":
            # Maximum (für ODER-ähnliche Operationen)
            result = max(truth_values)
        elif combination_method == "weighted":
            # Gewichteter Durchschnitt basierend auf Konfiguration
            weights = [self.config.get("weights", {}).get(key, 1.0) 
                      for key in signal.keys() if key in self.membership_functions]
            if sum(weights) > 0:
                result = sum(t * w for t, w in zip(truth_values, weights)) / sum(weights)
            else:
                result = sum(truth_values) / len(truth_values)
        else:
            # Fallback auf Durchschnitt
            result = sum(truth_values) / len(truth_values)
        
        # Speichere Ergebnis im Cache
        put_in_cache(cache_key, result)
        
        return result
        
    def _apply_membership_function(self, key: str, value: float) -> float:
        """
        Wendet eine Zugehörigkeitsfunktion auf einen Wert an
        Optimierte Implementierung mit Cache und vektorisierten Operationen
        
        Args:
            key: Schlüssel der Zugehörigkeitsfunktion
            value: Eingabewert
            
        Returns:
            Zugehörigkeitsgrad
        """
        # Cache-Schlüssel für diese Anfrage
        cache_key = f"membership_{key}_{value}"
        cached_result = get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
            
        # Holen und validieren der Parameter
        function_info = self.membership_functions.get(key, {})
        function_type = function_info.get("type", "linear")
        params = function_info.get("params", {})
        
        # Map von Funktionstypen zu Implementierungsmethoden
        # Dies ist schneller als eine Reihe von if-elif-Statements
        function_map = {
            "linear": self._linear_membership,
            "triangular": self._triangular_membership, 
            "gaussian": self._gaussian_membership,
            "sigmoid": self._sigmoid_membership,  # Neue Funktion
            "trapezoid": self._trapezoid_membership,  # Neue Funktion
            "bell": self._bell_membership  # Neue Funktion
        }
        
        # Wähle die richtige Funktion oder Fallback auf Standard
        membership_func = function_map.get(function_type, lambda v, p: 0.5)
        
        # Berechne den Wert
        result = membership_func(value, params)
        
        # Speichere im Cache für zukünftige Anfragen
        put_in_cache(cache_key, result)
        
        return result
            
    def _linear_membership(self, value: float, params: Dict[str, float]) -> float:
        """
        Lineare Zugehörigkeitsfunktion (Optimiert)
        
        Args:
            value: Eingabewert
            params: Parameter der Funktion
            
        Returns:
            Zugehörigkeitsgrad
        """
        min_val = params.get("min", 0.0)
        max_val = params.get("max", 1.0)
        
        # Validiere Parameter
        if min_val >= max_val:
            # Fallback bei ungültigen Parametern
            return 0.5
        
        # Optimierte lineare Interpolation
        if value <= min_val:
            return 0.0
        elif value >= max_val:
            return 1.0
        else:
            # Effiziente Normalisierung
            return (value - min_val) / (max_val - min_val)
            
    def _gaussian_membership(self, value: float, params: Dict[str, float]) -> float:
        """
        Gaußsche Zugehörigkeitsfunktion (Optimiert)
        
        Args:
            value: Eingabewert
            params: Parameter der Funktion
            
        Returns:
            Zugehörigkeitsgrad
        """
        mean = params.get("mean", 0)
        std_dev = params.get("std_dev", 1)
        
        if std_dev == 0:  # Vermeide Division durch Null
            return 1.0 if value == mean else 0.0
        
        # Optimierter Berechnung mit numerischer Stabilität
        exponent = -((value - mean) ** 2) / (2 * (std_dev ** 2))
        # Begrenze Exponent, um NaN zu vermeiden
        exponent = max(-709, exponent)  # min exp value for float64
        return math.exp(exponent)

    def _sigmoid_membership(self, value: float, params: Dict[str, float]) -> float:
        """
        Sigmoid-Zugehörigkeitsfunktion für weiches Schwellwertverhalten
        
        Args:
            value: Eingabewert
            params: Parameter der Funktion (center, slope)
            
        Returns:
            Zugehörigkeitsgrad
        """
        center = params.get("center", 0.5)
        slope = params.get("slope", 10.0)  # Höherer Wert = steilere Kurve
        
        # Numerisch stabile Sigmoid-Funktion
        exponent = -slope * (value - center)
        if exponent > 709:  # Vermeide Overflow
            return 0.0
        if exponent < -709:  # Vermeide Underflow
            return 1.0
            
        return 1.0 / (1.0 + math.exp(exponent))

    def _trapezoid_membership(self, value: float, params: Dict[str, float]) -> float:
        """
        Trapezförmige Zugehörigkeitsfunktion mit 4 Definitionspunkten
        
        Args:
            value: Eingabewert
            params: Parameter der Funktion (a, b, c, d) wobei a ≤ b ≤ c ≤ d
            
        Returns:
            Zugehörigkeitsgrad
        """
        a = params.get("a", 0.0)
        b = params.get("b", 0.25)
        c = params.get("c", 0.75)
        d = params.get("d", 1.0)
        
        # Überprüfe Parameter (stellen sicher, dass a ≤ b ≤ c ≤ d)
        a, b, c, d = sorted([a, b, c, d])
        
        # Berechne Zugehörigkeitsgrad
        if value <= a or value >= d:
            return 0.0
        elif a < value <= b:
            # Ansteigende Linie von a bis b
            return (value - a) / (b - a) if b > a else 1.0
        elif b < value < c:
            # Plateau bei 1.0
            return 1.0
        else:  # c ≤ value < d
            # Absteigende Linie von c bis d
            return (d - value) / (d - c) if d > c else 1.0
            
    def _bell_membership(self, value: float, params: Dict[str, float]) -> float:
        """
        Glockenförmige Zugehörigkeitsfunktion
        f(x) = 1 / (1 + |x-c/a|^2b)
        
        Args:
            value: Eingabewert
            params: Parameter der Funktion (a=Breite, b=Steilheit, c=Zentrum)
            
        Returns:
            Zugehörigkeitsgrad
        """
        a = params.get("a", 1.0)  # Breite
        b = params.get("b", 2.0)  # Steilheit
        c = params.get("c", 0.5)  # Zentrum
        
        # Vermeide Division durch Null
        if a == 0:
            return 1.0 if value == c else 0.0
        
        # Berechne Zugehörigkeitsgrad
        distance = abs((value - c) / a)
        # Optimierter Potenzfunktion
        power = 2 * b
        # Begrenze den Wert, um numerische Stabilität zu gewährleisten
        powered_distance = min(1e15, distance ** power)
        
        return 1.0 / (1.0 + powered_distance)

    def _triangular_membership(self, value: float, params: Dict[str, float]) -> float:
        """
        Dreieckige Zugehörigkeitsfunktion
        
        Args:
            value: Eingabewert
            params: Parameter der Funktion
            
        Returns:
            Zugehörigkeitsgrad
        """
        a = params.get("a", 0)
        b = params.get("b", 0.5)
        c = params.get("c", 1)
        
        if value <= a or value >= c:
            return 0
        elif a < value <= b:
            return (value - a) / (b - a)
        else:  # b < value < c
            return (c - value) / (c - b)
            
    def _gaussian_membership(self, value: float, params: Dict[str, float]) -> float:
        """
        Gaußsche Zugehörigkeitsfunktion
        
        Args:
            value: Eingabewert
            params: Parameter der Funktion
            
        Returns:
            Zugehörigkeitsgrad
        """
        mean = params.get("mean", 0.5)
        std = params.get("std", 0.1)
        
        return math.exp(-((value - mean) ** 2) / (2 * std ** 2))


class SymbolMap:
    """
    Symbol-Map
    
    Übersetzt logische Konzepte in symbolische Darstellung und
    ermöglicht die Manipulation von symbolischen Ausdrücken.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert die SymbolMap
        
        Args:
            config: Konfigurationsobjekt für die SymbolMap
        """
        self.config = config or {}
        self.symbol_table = self.config.get("symbol_table", {})
        
        # Standard-Symboltabelle, falls keine konfiguriert wurde
        if not self.symbol_table:
            self.symbol_table = {
                "vertrauen": {"symbol": "T", "domain": [0, 1]},
                "gefahr": {"symbol": "R", "domain": [0, 1]},
                "widerspruch": {"symbol": "C", "domain": [0, 1]},
                "unsicherheit": {"symbol": "U", "domain": [0, 1]},
                "priorität": {"symbol": "P", "domain": [0, 10]},
                "nutzen": {"symbol": "B", "domain": [0, 1]},
                "kosten": {"symbol": "Co", "domain": [0, 1]},
                "zeit": {"symbol": "t", "domain": [0, float('inf')]}
            }
            
        logger.info("SymbolMap initialisiert")
        
    def translate(self, concept: str, value: float = None) -> Dict[str, Any]:
        """
        Übersetzt ein Konzept in eine symbolische Darstellung
        
        Args:
            concept: Das zu übersetzende Konzept
            value: Optionaler Wert des Konzepts
            
        Returns:
            Symbolische Darstellung des Konzepts
        """
        # Normalisiere das Konzept
        concept_lower = concept.lower()
        
        # Suche nach dem Konzept in der Symboltabelle
        if concept_lower in self.symbol_table:
            symbol_info = self.symbol_table[concept_lower]
            symbol = symbol_info["symbol"]
            domain = symbol_info.get("domain", [0, 1])
            
            # Normalisiere den Wert, falls vorhanden
            normalized_value = None
            if value is not None:
                # Begrenze den Wert auf die Domain
                bounded_value = max(domain[0], min(value, domain[1]))
                
                # Normalisiere auf [0,1]
                if domain[1] > domain[0]:
                    normalized_value = (bounded_value - domain[0]) / (domain[1] - domain[0])
                else:
                    normalized_value = 0
            
            return {
                "original": concept,
                "symbol": symbol,
                "value": value,
                "normalized_value": normalized_value,
                "domain": domain
            }
        else:
            # Fallback für unbekannte Konzepte
            return {
                "original": concept,
                "symbol": "?",
                "value": value,
                "normalized_value": None,
                "domain": [0, 1]
            }
            
    def combine(self, symbols: List[Dict[str, Any]], operation: str) -> Dict[str, Any]:
        """
        Kombiniert mehrere Symbole mit einer Operation
        
        Args:
            symbols: Liste von symbolischen Darstellungen
            operation: Zu verwendende Operation ("and", "or", "not", etc.)
            
        Returns:
            Kombinierte symbolische Darstellung
        """
        if not symbols:
            return {"symbol": "∅", "value": None}
            
        # Extrahiere Symbole und Werte
        symbol_strings = [s["symbol"] for s in symbols if "symbol" in s]
        values = [s["normalized_value"] for s in symbols if "normalized_value" in s and s["normalized_value"] is not None]
        
        # Kombiniere Symbole
        combined_symbol = ""
        combined_value = None
        
        if operation == "and":
            combined_symbol = "(" + " ∧ ".join(symbol_strings) + ")"
            combined_value = min(values) if values else None
        elif operation == "or":
            combined_symbol = "(" + " ∨ ".join(symbol_strings) + ")"
            combined_value = max(values) if values else None
        elif operation == "not" and symbols:
            combined_symbol = "¬" + symbols[0]["symbol"]
            combined_value = 1 - values[0] if values else None
        elif operation == "implies" and len(symbols) >= 2:
            combined_symbol = symbols[0]["symbol"] + " → " + symbols[1]["symbol"]
            # p → q is equivalent to ¬p ∨ q
            if len(values) >= 2:
                combined_value = max(1 - values[0], values[1])
        else:
            combined_symbol = "(" + ", ".join(symbol_strings) + ")"
            combined_value = sum(values) / len(values) if values else None
            
        return {
            "symbol": combined_symbol,
            "value": combined_value,
            "operation": operation,
            "components": symbols
        }


class ConflictResolver:
    """
    Konfliktlöser
    
    Erkennt, bewertet und löst Zielkonflikte oder Entscheidungsdilemmata
    basierend auf Prioritäten und Kontextinformationen.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den ConflictResolver
        
        Args:
            config: Konfigurationsobjekt für den ConflictResolver
        """
        self.config = config or {}
        self.resolution_strategies = self.config.get("resolution_strategies", {})
        
        # Standard-Strategien, falls keine konfiguriert wurden
        if not self.resolution_strategies:
            self.resolution_strategies = {
                "prioritize_safety": {
                    "weight": 0.8,
                    "conditions": {"risk": {"min": 0.6}}
                },
                "prioritize_utility": {
                    "weight": 0.7,
                    "conditions": {"benefit": {"min": 0.7}}
                },
                "prioritize_time": {
                    "weight": 0.6,
                    "conditions": {"urgency": {"min": 0.8}}
                },
                "compromise": {
                    "weight": 0.5,
                    "conditions": {}
                }
            }
            
        logger.info("ConflictResolver initialisiert")
        
    def resolve(self, conflict: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Löst einen Konflikt basierend auf dem Kontext
        
        Args:
            conflict: Konfliktbeschreibung
            context: Kontextinformationen für die Konfliktlösung
            
        Returns:
            Lösungsvorschlag
        """
        if not conflict or not context:
            return {"resolution": "no_action", "confidence": 0.0}
            
        # Extrahiere Konfliktparteien
        parties = conflict.get("parties", [])
        if not parties:
            return {"resolution": "no_action", "confidence": 0.0}
            
        # Bewerte die anwendbaren Strategien
        applicable_strategies = []
        
        for name, strategy in self.resolution_strategies.items():
            if self._check_conditions(strategy["conditions"], context):
                applicable_strategies.append((name, strategy["weight"]))
                
        if not applicable_strategies:
            # Fallback: Kompromiss-Strategie
            return {"resolution": "compromise", "confidence": 0.5}
            
        # Wähle die Strategie mit dem höchsten Gewicht
        applicable_strategies.sort(key=lambda x: x[1], reverse=True)
        best_strategy = applicable_strategies[0][0]
        confidence = applicable_strategies[0][1]
        
        # Wende die Strategie an
        resolution = self._apply_strategy(best_strategy, parties, context)
        resolution["confidence"] = confidence
        
        return resolution
        
    def _check_conditions(self, conditions: Dict[str, Dict[str, float]], context: Dict[str, Any]) -> bool:
        """
        Prüft, ob die Bedingungen einer Strategie erfüllt sind
        
        Args:
            conditions: Bedingungen der Strategie
            context: Kontextinformationen
            
        Returns:
            True, wenn alle Bedingungen erfüllt sind, sonst False
        """
        if not conditions:
            return True
            
        for key, condition in conditions.items():
            if key not in context:
                return False
                
            value = context[key]
            
            if "min" in condition and value < condition["min"]:
                return False
                
            if "max" in condition and value > condition["max"]:
                return False
                
        return True
        
    def _apply_strategy(self, strategy: str, parties: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wendet eine Strategie auf die Konfliktparteien an
        
        Args:
            strategy: Name der anzuwendenden Strategie
            parties: Konfliktparteien
            context: Kontextinformationen
            
        Returns:
            Lösungsvorschlag
        """
        if strategy == "prioritize_safety":
            # Priorisiere die sicherste Option
            parties.sort(key=lambda p: p.get("risk", 1.0))
            return {
                "resolution": "prioritize_safety",
                "selected": parties[0]["name"] if parties else None,
                "reasoning": "Sicherheit wurde priorisiert"
            }
        elif strategy == "prioritize_utility":
            # Priorisiere die nützlichste Option
            parties.sort(key=lambda p: p.get("benefit", 0.0), reverse=True)
            return {
                "resolution": "prioritize_utility",
                "selected": parties[0]["name"] if parties else None,
                "reasoning": "Nutzen wurde priorisiert"
            }
        elif strategy == "prioritize_time":
            # Priorisiere die zeitkritischste Option
            parties.sort(key=lambda p: p.get("urgency", 0.0), reverse=True)
            return {
                "resolution": "prioritize_time",
                "selected": parties[0]["name"] if parties else None,
                "reasoning": "Zeitkritikalität wurde priorisiert"
            }
        else:  # compromise
            # Kompromiss: Wähle die Option mit dem besten Gesamtwert
            for party in parties:
                party["score"] = (
                    party.get("benefit", 0.0) * 0.4 +
                    (1 - party.get("risk", 0.0)) * 0.4 +
                    party.get("urgency", 0.0) * 0.2
                )
                
            parties.sort(key=lambda p: p.get("score", 0.0), reverse=True)
            return {
                "resolution": "compromise",
                "selected": parties[0]["name"] if parties else None,
                "reasoning": "Kompromiss basierend auf Gesamtbewertung"
            }


# EmotionWeightedDecision-Klasse wurde entfernt, um das Q-LOGIK Framework zu vereinfachen
# Vereinfachte Version für emotionale Gewichtung
def simple_emotion_weight(decision_value: float, context: Dict[str, Any] = None) -> float:
    """Vereinfachte Funktion für emotionale Gewichtung ohne komplexe Klasse"""
    if decision_value is None:
        return 0.5
    if not context or "emotions" not in context:
        return decision_value
        
    # Einfache Implementierung: Nur positive/negative Emotion berücksichtigen
    emotion_factor = 0.2  # Reduzierter Einfluss
    emotion_value = context.get("emotion_value", 0.5)
    
    # Lineare Kombination
    weighted_value = (1 - emotion_factor) * decision_value + emotion_factor * emotion_value
    return min(1.0, max(0.0, weighted_value))


# Vereinfachte Version des MetaPriorityMapper
def simple_priority_mapping(risk: float, benefit: float, urgency: float = 0.5) -> Dict[str, Any]:
    """
    Vereinfachte Funktion für Prioritätsmapping ohne komplexe Klasse
    
    Args:
        risk: Risikowert zwischen 0 und 1
        benefit: Nutzenwert zwischen 0 und 1
        urgency: Dringlichkeitswert zwischen 0 und 1 (optional)
        
    Returns:
        Prioritätsinformation als Dictionary
    """
    # Begrenze die Werte auf [0,1]
    risk = min(1.0, max(0.0, risk))
    benefit = min(1.0, max(0.0, benefit))
    urgency = min(1.0, max(0.0, urgency))
    
    # Vereinfachte Gewichtung
    # Risiko wird invertiert, da ein höheres Risiko die Priorität senken sollte
    priority_value = ((1 - risk) * 0.4) + (benefit * 0.3) + (urgency * 0.3)
    
    # Vereinfachte Stufenzuordnung
    if priority_value >= 0.8:
        level = "critical"
        action = "immediate_action"
    elif priority_value >= 0.6:
        level = "high"
        action = "prioritize"
    elif priority_value >= 0.4:
        level = "medium"
        action = "schedule"
    elif priority_value >= 0.2:
        level = "low"
        action = "defer"
    else:
        level = "minimal"
        action = "ignore"
    
    return {
        "priority_value": priority_value,
        "level": level,
        "action": action,
        "components": {
            "risk": risk,
            "benefit": benefit,
            "urgency": urgency
        }
    }
        
def get_action_recommendation(priority: Dict[str, Any]) -> str:
    """
    Gibt eine Handlungsempfehlung basierend auf der Priorität
    
    Args:
        priority: Prioritätszuordnung
        
    Returns:
        Handlungsempfehlung
    """
    action = priority.get("action", "defer")
    level = priority.get("level", "low")
    
    if action == "immediate_action":
        return "Sofortige Handlung erforderlich"
    elif action == "prioritize":
        return "Hohe Priorität, baldige Bearbeitung empfohlen"
    elif action == "schedule":
        return "Mittlere Priorität, in Arbeitsplan aufnehmen"
    elif action == "defer":
        return "Niedrige Priorität, kann aufgeschoben werden"
    else:  # ignore
        return "Minimale Priorität, keine Handlung erforderlich"


# Globale Instanzen der Komponenten
bayesian = BayesianDecisionCore()
fuzzylogic = FuzzyLogicUnit()
symbolmap = SymbolMap()
conflict_resolver = ConflictResolver()
# Vereinfachte Komponenten: emotion_weighted und meta_priority wurden durch Funktionen ersetzt


def qlogik_decision(context: Dict[str, Any]) -> str:
    """
    Hauptfunktion für Q-LOGIK-Entscheidungen
    
    Args:
        context: Kontextinformationen für die Entscheidung
        
    Returns:
        Entscheidung als String ("JA", "NEIN", "WARNUNG")
    """
    # Extrahiere Daten aus dem Kontext
    data = context.get("data", {})
    signal = context.get("signal", {})
    
    # Berechne Wahrscheinlichkeit der Hypothese
    p_hypothesis = bayesian.evaluate(data)
    
    # Berechne Wahrheitsgrad des Signals
    truth_value = fuzzylogic.score(signal)
    
    # Verwende vereinfachte emotionale Gewichtung
    # Standardwert 0.5 für neutrale Emotion, falls nicht angegeben
    context["emotion_value"] = context.get("emotion_value", 0.5)
    weighted_decision = simple_emotion_weight(p_hypothesis, context)
    
    # Vereinfachte Entscheidungslogik
    if weighted_decision > 0.7:  # Schwellwert leicht reduziert
        return "JA"
    elif truth_value > 0.5:
        return "WARNUNG"
    else:
        return "NEIN"


def advanced_qlogik_decision(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Erweiterte Q-LOGIK-Entscheidungsfunktion mit detaillierten Informationen
    
    Args:
        context: Kontextinformationen für die Entscheidung
        
    Returns:
        Detaillierte Entscheidungsinformationen
    """
    # Extrahiere Daten aus dem Kontext
    data = context.get("data", {})
    signal = context.get("signal", {})
    risk = context.get("risk", 0.5)
    benefit = context.get("benefit", 0.5)
    urgency = context.get("urgency", 0.5)
    conflict = context.get("conflict", None)
    
    # Standardwert für emotionale Gewichtung
    context["emotion_value"] = context.get("emotion_value", 0.5)
    
    # Berechne Wahrscheinlichkeit der Hypothese
    p_hypothesis = bayesian.evaluate(data)
    
    # Berechne Wahrheitsgrad des Signals
    truth_value = fuzzylogic.score(signal)
    
    # Vereinfachte emotionale Gewichtung
    weighted_decision = simple_emotion_weight(p_hypothesis, context)
    
    # Vereinfachtes Prioritätsmapping
    priority = simple_priority_mapping(risk, benefit, urgency)
    
    # Löse Konflikte, falls vorhanden
    conflict_resolution = None
    if conflict:
        conflict_resolution = conflict_resolver.resolve(conflict, context)
    
    # Bestimme Entscheidung
    decision = "UNBEKANNT"
    confidence = weighted_decision
    
    if weighted_decision > 0.7:  # Schwellwert leicht reduziert
        decision = "JA"
    elif truth_value > 0.5:
        decision = "WARNUNG"
    else:
        decision = "NEIN"
    
    # Erstelle detaillierte Antwort
    result = {
        "decision": decision,
        "confidence": confidence,
        "components": {
            "bayesian_probability": p_hypothesis,
            "fuzzy_truth": truth_value,
            "emotion_value": context["emotion_value"],
            "weighted_decision": weighted_decision
        },
        "priority": priority,
        "action_recommendation": get_action_recommendation(priority)
    }
    
    if conflict_resolution:
        result["conflict_resolution"] = conflict_resolution
    
    return result
