#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - MPRIME Probabilistic Mapper

Wahrscheinlichkeits-Überlagerung von Gleichungspfaden für die MPRIME Engine.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import logging
import math
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

logger = logging.getLogger("MISO.Math.MPRIME.ProbabilisticMapper")

class ProbabilisticMapper:
    """
    Wahrscheinlichkeits-Überlagerung von Gleichungspfaden
    
    Diese Klasse implementiert probabilistische Methoden für die Analyse
    und Überlagerung von mathematischen Lösungspfaden.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den ProbabilisticMapper
        
        Args:
            config: Konfigurationsobjekt für den ProbabilisticMapper
        """
        self.config = config or {}
        self.max_paths = self.config.get("max_paths", 10)
        self.convergence_threshold = self.config.get("convergence_threshold", 1e-6)
        self.max_iterations = self.config.get("max_iterations", 100)
        
        # Initialisiere Wahrscheinlichkeitsverteilungen
        self.distributions = {
            "normal": self._normal_distribution,
            "uniform": self._uniform_distribution,
            "exponential": self._exponential_distribution,
            "cauchy": self._cauchy_distribution,
            "custom": None
        }
        
        logger.info(f"ProbabilisticMapper initialisiert mit max_paths={self.max_paths}")
    
    def set_custom_distribution(self, distribution_func: Callable) -> None:
        """
        Setzt eine benutzerdefinierte Wahrscheinlichkeitsverteilung
        
        Args:
            distribution_func: Funktion, die die Wahrscheinlichkeitsverteilung implementiert
        """
        self.distributions["custom"] = distribution_func
        logger.info("Benutzerdefinierte Wahrscheinlichkeitsverteilung gesetzt")
    
    def _normal_distribution(self, x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
        """
        Normalverteilung
        
        Args:
            x: Wert
            mu: Mittelwert
            sigma: Standardabweichung
            
        Returns:
            Wahrscheinlichkeitsdichte
        """
        return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    def _uniform_distribution(self, x: float, a: float = 0.0, b: float = 1.0) -> float:
        """
        Gleichverteilung
        
        Args:
            x: Wert
            a: Untere Grenze
            b: Obere Grenze
            
        Returns:
            Wahrscheinlichkeitsdichte
        """
        if a <= x <= b:
            return 1.0 / (b - a)
        return 0.0
    
    def _exponential_distribution(self, x: float, lambda_: float = 1.0) -> float:
        """
        Exponentialverteilung
        
        Args:
            x: Wert
            lambda_: Rate
            
        Returns:
            Wahrscheinlichkeitsdichte
        """
        if x >= 0:
            return lambda_ * math.exp(-lambda_ * x)
        return 0.0
    
    def _cauchy_distribution(self, x: float, x0: float = 0.0, gamma: float = 1.0) -> float:
        """
        Cauchy-Verteilung
        
        Args:
            x: Wert
            x0: Lageparameter
            gamma: Skalierungsparameter
            
        Returns:
            Wahrscheinlichkeitsdichte
        """
        return (1.0 / (math.pi * gamma)) * (gamma ** 2 / ((x - x0) ** 2 + gamma ** 2))
    
    def generate_solution_paths(self, equation: Dict[str, Any], num_paths: int = None) -> Dict[str, Any]:
        """
        Generiert probabilistische Lösungspfade für eine Gleichung
        
        Args:
            equation: Gleichung als symbolischer Baum
            num_paths: Anzahl der zu generierenden Pfade (optional)
            
        Returns:
            Dictionary mit Lösungspfaden und Metadaten
        """
        # Verwende Standardwert, falls nicht angegeben
        num_paths = num_paths or self.max_paths
        
        # Begrenze auf maximale Anzahl von Pfaden
        num_paths = min(num_paths, self.max_paths)
        
        # Initialisiere Ergebnis
        result = {
            "original_equation": equation,
            "num_paths": num_paths,
            "paths": [],
            "path_probabilities": [],
            "convergence": None,
            "most_probable_path": None,
            "most_probable_solution": None
        }
        
        try:
            # Generiere Lösungspfade
            paths, probabilities = self._generate_paths(equation, num_paths)
            result["paths"] = paths
            result["path_probabilities"] = probabilities
            
            # Bestimme Konvergenz
            convergence = self._calculate_convergence(paths)
            result["convergence"] = convergence
            
            # Bestimme wahrscheinlichsten Pfad
            if paths:
                most_probable_index = np.argmax(probabilities)
                result["most_probable_path"] = paths[most_probable_index]
                
                # Extrahiere Lösung aus dem wahrscheinlichsten Pfad
                if "solution" in paths[most_probable_index]:
                    result["most_probable_solution"] = paths[most_probable_index]["solution"]
            
            logger.info(f"{num_paths} Lösungspfade erfolgreich generiert")
        
        except Exception as e:
            logger.error(f"Fehler bei der Generierung von Lösungspfaden: {str(e)}")
            raise
        
        return result
    
    def _generate_paths(self, equation: Dict[str, Any], num_paths: int) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Generiert Lösungspfade für eine Gleichung
        
        Args:
            equation: Gleichung als symbolischer Baum
            num_paths: Anzahl der zu generierenden Pfade
            
        Returns:
            Tuple aus Liste von Pfaden und Liste von Wahrscheinlichkeiten
        """
        # In einer vollständigen Implementierung würde hier eine komplexe
        # Generierung von Lösungspfaden stehen
        
        # Einfache Implementierung für dieses Beispiel
        paths = []
        probabilities = []
        
        # Extrahiere Informationen aus der Gleichung
        equation_type = "algebraic"  # Standardwert
        if isinstance(equation, dict):
            if "symbol_types" in equation:
                # Bestimme Gleichungstyp aus Symboltypen
                if "calculus" in equation["symbol_types"]:
                    equation_type = "differential"
                elif "topology" in equation["symbol_types"]:
                    equation_type = "topological"
            
            if "complexity" in equation:
                complexity = equation["complexity"]
            else:
                complexity = 1
        else:
            complexity = 1
        
        # Generiere Pfade basierend auf Gleichungstyp
        for i in range(num_paths):
            # Berechne Pfadwahrscheinlichkeit
            # Höhere Indizes haben geringere Wahrscheinlichkeit
            probability = math.exp(-i / (num_paths / 2))
            
            # Normalisiere Wahrscheinlichkeit
            probabilities.append(probability)
            
            # Generiere Pfad
            path = self._generate_path(equation, equation_type, i, complexity)
            paths.append(path)
        
        # Normalisiere Wahrscheinlichkeiten
        total_probability = sum(probabilities)
        if total_probability > 0:
            probabilities = [p / total_probability for p in probabilities]
        
        return paths, probabilities
    
    def _generate_path(self, equation: Dict[str, Any], equation_type: str, path_index: int, complexity: int) -> Dict[str, Any]:
        """
        Generiert einen einzelnen Lösungspfad
        
        Args:
            equation: Gleichung als symbolischer Baum
            equation_type: Typ der Gleichung
            path_index: Index des Pfads
            complexity: Komplexität der Gleichung
            
        Returns:
            Lösungspfad als Dictionary
        """
        # In einer vollständigen Implementierung würde hier eine komplexe
        # Generierung eines Lösungspfads stehen
        
        # Einfache Implementierung für dieses Beispiel
        path = {
            "path_id": path_index,
            "equation_type": equation_type,
            "steps": [],
            "solution": None,
            "confidence": 1.0 - (path_index / self.max_paths)
        }
        
        # Generiere Schritte basierend auf Gleichungstyp und Komplexität
        num_steps = max(1, min(10, complexity))
        
        for step_index in range(num_steps):
            step = {
                "step_id": step_index,
                "operation": self._get_random_operation(equation_type, step_index, num_steps),
                "intermediate_result": f"Zwischenergebnis {step_index}",
                "confidence": 1.0 - (step_index / num_steps) * (path_index / self.max_paths)
            }
            
            path["steps"].append(step)
        
        # Generiere Lösung
        path["solution"] = {
            "value": f"Lösung für Pfad {path_index}",
            "type": equation_type,
            "exact": path_index == 0,  # Erster Pfad ist exakt
            "approximate": path_index > 0  # Andere Pfade sind approximativ
        }
        
        return path
    
    def _get_random_operation(self, equation_type: str, step_index: int, num_steps: int) -> str:
        """
        Gibt eine zufällige Operation für einen Lösungsschritt zurück
        
        Args:
            equation_type: Typ der Gleichung
            step_index: Index des Schritts
            num_steps: Gesamtanzahl der Schritte
            
        Returns:
            Operation als String
        """
        # Operationen basierend auf Gleichungstyp
        operations = {
            "algebraic": ["expand", "factor", "substitute", "simplify", "solve"],
            "differential": ["differentiate", "integrate", "substitute", "simplify", "solve"],
            "topological": ["transform", "dimension", "map", "project", "solve"]
        }
        
        # Wähle Operationen basierend auf Gleichungstyp
        if equation_type in operations:
            available_operations = operations[equation_type]
        else:
            available_operations = operations["algebraic"]
        
        # Wähle Operation basierend auf Schrittindex
        if step_index == 0:
            # Erster Schritt: Expansion oder Transformation
            return available_operations[0]
        elif step_index == num_steps - 1:
            # Letzter Schritt: Lösung
            return "solve"
        else:
            # Mittlere Schritte: Zufällige Operation
            index = step_index % (len(available_operations) - 2) + 1
            return available_operations[index]
    
    def _calculate_convergence(self, paths: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Berechnet die Konvergenz der Lösungspfade
        
        Args:
            paths: Liste von Lösungspfaden
            
        Returns:
            Konvergenzinformationen als Dictionary
        """
        # In einer vollständigen Implementierung würde hier eine komplexe
        # Berechnung der Konvergenz stehen
        
        # Einfache Implementierung für dieses Beispiel
        if not paths:
            return {
                "converged": False,
                "convergence_rate": 0.0,
                "convergence_measure": 0.0
            }
        
        # Extrahiere Lösungen
        solutions = []
        for path in paths:
            if "solution" in path and path["solution"] is not None:
                solutions.append(path["solution"])
        
        # Berechne Konvergenzrate
        convergence_rate = 1.0 - (len(set(str(s) for s in solutions)) / len(solutions)) if solutions else 0.0
        
        # Bestimme Konvergenz
        converged = convergence_rate >= 0.5
        
        return {
            "converged": converged,
            "convergence_rate": convergence_rate,
            "convergence_measure": convergence_rate * len(solutions) / len(paths) if paths else 0.0
        }
    
    def overlay_paths(self, paths: List[Dict[str, Any]], probabilities: List[float] = None) -> Dict[str, Any]:
        """
        Überlagert Lösungspfade
        
        Args:
            paths: Liste von Lösungspfaden
            probabilities: Liste von Pfadwahrscheinlichkeiten (optional)
            
        Returns:
            Überlagertes Ergebnis als Dictionary
        """
        # Prüfe Eingabe
        if not paths:
            return {
                "overlay_type": "empty",
                "paths": [],
                "probabilities": [],
                "result": None
            }
        
        # Verwende Gleichverteilung, falls keine Wahrscheinlichkeiten angegeben
        if probabilities is None:
            probabilities = [1.0 / len(paths)] * len(paths)
        
        # Normalisiere Wahrscheinlichkeiten
        total_probability = sum(probabilities)
        if total_probability > 0:
            normalized_probabilities = [p / total_probability for p in probabilities]
        else:
            normalized_probabilities = [1.0 / len(paths)] * len(paths)
        
        # Initialisiere Ergebnis
        result = {
            "overlay_type": "probabilistic",
            "paths": paths,
            "probabilities": normalized_probabilities,
            "result": None,
            "confidence": 0.0
        }
        
        try:
            # Überlagere Pfade
            overlay_result, confidence = self._compute_overlay(paths, normalized_probabilities)
            result["result"] = overlay_result
            result["confidence"] = confidence
            
            logger.info(f"{len(paths)} Lösungspfade erfolgreich überlagert")
        
        except Exception as e:
            logger.error(f"Fehler bei der Überlagerung von Lösungspfaden: {str(e)}")
            raise
        
        return result
    
    def _compute_overlay(self, paths: List[Dict[str, Any]], probabilities: List[float]) -> Tuple[Dict[str, Any], float]:
        """
        Berechnet die Überlagerung von Lösungspfaden
        
        Args:
            paths: Liste von Lösungspfaden
            probabilities: Liste von Pfadwahrscheinlichkeiten
            
        Returns:
            Tuple aus überlagertem Ergebnis und Konfidenz
        """
        # In einer vollständigen Implementierung würde hier eine komplexe
        # Berechnung der Überlagerung stehen
        
        # Einfache Implementierung für dieses Beispiel
        
        # Extrahiere Lösungen und Konfidenzen
        solutions = []
        confidences = []
        
        for i, path in enumerate(paths):
            if "solution" in path and path["solution"] is not None:
                solutions.append(path["solution"])
                
                # Berechne Konfidenz
                if "confidence" in path:
                    path_confidence = path["confidence"]
                else:
                    path_confidence = 1.0
                
                confidences.append(path_confidence * probabilities[i])
        
        # Berechne Gesamtkonfidenz
        total_confidence = sum(confidences) if confidences else 0.0
        
        # Wähle Lösung mit höchster Konfidenz
        if solutions and confidences:
            best_index = np.argmax(confidences)
            best_solution = solutions[best_index]
            
            # Erstelle überlagertes Ergebnis
            overlay_result = {
                "type": "probabilistic_overlay",
                "value": best_solution["value"] if isinstance(best_solution, dict) and "value" in best_solution else best_solution,
                "source_path": best_index,
                "contributing_paths": len(solutions),
                "total_paths": len(paths)
            }
        else:
            overlay_result = {
                "type": "empty_overlay",
                "value": None,
                "source_path": None,
                "contributing_paths": 0,
                "total_paths": len(paths)
            }
        
        return overlay_result, total_confidence
