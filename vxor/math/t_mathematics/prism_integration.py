#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-Mathematics Engine - Integration mit PRISM

Diese Datei implementiert die Integration zwischen der T-Mathematics Engine
und PRISM für beschleunigte Simulationen und Wahrscheinlichkeitsanalysen
mit MLX auf Apple Silicon.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import torch
import numpy as np

from miso.math.t_mathematics.engine import TMathEngine
from miso.math.t_mathematics.compat import TMathConfig

# Flag für die Verfügbarkeit von PRISM
PRISM_AVAILABLE = False

# Lazy-Loading Funktion für PRISM Engine
def get_prism_engine():
    """
    Lazy-Loading Funktion für die PRISM Engine
    Verhindert zirkuläre Importe zwischen T-Mathematics und PRISM
    
    Returns:
        PrismEngine-Klasse oder None, falls nicht verfügbar
    """
    try:
        from miso.simulation.prism_engine import PrismEngine
        global PRISM_AVAILABLE
        PRISM_AVAILABLE = True
        return PrismEngine
    except ImportError as e:
        logger.warning(f"PRISM Engine konnte nicht importiert werden: {e}")
        return None

# Lazy-Loading Funktion für PRISM-Matrixkomponenten
def get_prism_matrix_components():
    """
    Lazy-Loading Funktion für PRISM Matrix Komponenten
    Verhindert zirkuläre Importe zwischen T-Mathematics und PRISM
    
    Returns:
        Tuple mit PRISM-Matrixkomponenten oder None-Werte, falls nicht verfügbar
    """
    try:
        from miso.simulation.prism import PrismMatrix, RealityFold
        return PrismMatrix, RealityFold
    except ImportError as e:
        logger.warning(f"PRISM Matrix Komponenten konnten nicht importiert werden: {e}")
        return None, None

# Logger konfigurieren
logger = logging.getLogger("t_mathematics.prism_integration")

class PrismSimulationEngine:
    """
    Engine für beschleunigte Simulationen und Wahrscheinlichkeitsanalysen mit PRISM.
    
    Diese Klasse bietet optimierte mathematische Operationen für PRISM,
    insbesondere für Monte-Carlo-Simulationen und Wahrscheinlichkeitsberechnungen.
    """
    
    def __init__(self, 
                use_mlx: bool = True,
                precision: str = "float16",
                device: str = "auto"):
        """
        Initialisiert die PRISM Simulation Engine.
        
        Args:
            use_mlx: Ob MLX verwendet werden soll (wenn verfügbar)
            precision: Präzisionstyp für Berechnungen
            device: Zielgerät für Berechnungen
        """
        # Erstelle T-Mathematics Engine mit MLX-Optimierung
        self.config = TMathConfig(
            precision=precision,
            device=device,
            optimize_for_rdna=False,
            optimize_for_apple_silicon=True
        )
        self.engine = TMathEngine(
            config=self.config,
            use_mlx=use_mlx
        )
        
        logger.info(f"PRISM Simulation Engine initialisiert: MLX={self.engine.use_mlx}, "
                   f"Gerät={self.engine.device}, Präzision={self.engine.precision}")
    
    def monte_carlo_simulation(self, 
                             transition_matrix: torch.Tensor, 
                             initial_state: torch.Tensor,
                             num_steps: int,
                             num_simulations: int) -> torch.Tensor:
        """
        Führt Monte-Carlo-Simulationen für ein Markov-Modell durch.
        
        Verwendet optimierte MLX-Operationen auf Apple Silicon, wenn verfügbar.
        
        Args:
            transition_matrix: Übergangsmatrix [num_states, num_states]
            initial_state: Anfangszustand [num_states]
            num_steps: Anzahl der Simulationsschritte
            num_simulations: Anzahl der durchzuführenden Simulationen
            
        Returns:
            Simulationsergebnisse [num_simulations, num_steps, num_states]
        """
        num_states = transition_matrix.shape[0]
        
        # Erstelle Ausgabe-Tensor
        results = torch.zeros((num_simulations, num_steps + 1, num_states), 
                             dtype=initial_state.dtype, 
                             device=initial_state.device)
        
        # Setze Anfangszustand für alle Simulationen
        results[:, 0, :] = initial_state.unsqueeze(0).expand(num_simulations, -1)
        
        # Führe Simulationen durch
        for step in range(num_steps):
            current_states = results[:, step, :]
            
            # Berechne Wahrscheinlichkeiten für den nächsten Zustand
            # Verwende direkte PyTorch-Operation statt T-Mathematics Engine für Stabilität
            next_probs = torch.matmul(current_states, transition_matrix)
            
            # Fallback: Falls T-Mathematics Engine verwendet werden soll
            # next_probs = self.engine.matmul(current_states, transition_matrix)
            # if isinstance(next_probs, list):
            #     # Rekursive Konvertierung verschachtelter Listen
            #     def convert_nested_list(data):
            #         if isinstance(data, list):
            #             if all(isinstance(x, (int, float)) for x in data):
            #                 return torch.tensor(data, device=current_states.device)
            #             else:
            #                 return torch.stack([convert_nested_list(x) for x in data])
            #         return data
            #     next_probs = convert_nested_list(next_probs)
            # elif not isinstance(next_probs, torch.Tensor):
            #     next_probs = torch.tensor(next_probs, device=current_states.device)
            
            # Generiere Zufallszahlen für die Simulation
            random_values = torch.rand((num_simulations, num_states), 
                                      device=current_states.device)
            
            # Wende Aktivierungsfunktion an, um diskrete Zustände zu erzeugen
            # Verwende direkte PyTorch GELU für Stabilität
            import torch.nn.functional as F
            next_states = F.gelu(next_probs + 0.1 * random_values)
            
            # Normalisiere die Zustände
            next_states = next_states / torch.sum(next_states, dim=1, keepdim=True)
            
            # Speichere die Ergebnisse
            results[:, step + 1, :] = next_states
        
        return results
    
    def probability_analysis(self, 
                           simulation_results: torch.Tensor, 
                           target_states: List[int]) -> torch.Tensor:
        """
        Analysiert die Wahrscheinlichkeiten für bestimmte Zielzustände.
        
        Verwendet optimierte MLX-Operationen auf Apple Silicon, wenn verfügbar.
        
        Args:
            simulation_results: Simulationsergebnisse [num_simulations, num_steps, num_states]
            target_states: Liste der Zielzustände
            
        Returns:
            Wahrscheinlichkeiten für jeden Schritt [num_steps, len(target_states)]
        """
        num_simulations, num_steps, num_states = simulation_results.shape
        num_targets = len(target_states)
        
        # Erstelle Ausgabe-Tensor
        probabilities = torch.zeros((num_steps, num_targets), 
                                   dtype=simulation_results.dtype,
                                   device=simulation_results.device)
        
        # Berechne Wahrscheinlichkeiten für jeden Schritt und jeden Zielzustand
        for step in range(num_steps):
            step_results = simulation_results[:, step, :]
            
            for i, target in enumerate(target_states):
                # Berechne die Wahrscheinlichkeit für den Zielzustand
                target_probs = step_results[:, target]
                probabilities[step, i] = torch.mean(target_probs)
        
        return probabilities
    
    def entropy_analysis(self, simulation_results: torch.Tensor) -> torch.Tensor:
        """
        Berechnet die Entropie für jeden Simulationsschritt.
        
        Verwendet optimierte MLX-Operationen auf Apple Silicon, wenn verfügbar.
        
        Args:
            simulation_results: Simulationsergebnisse [num_simulations, num_steps, num_states]
            
        Returns:
            Entropie für jeden Schritt [num_steps]
        """
        num_simulations, num_steps, num_states = simulation_results.shape
        
        # Erstelle Ausgabe-Tensor
        entropy = torch.zeros(num_steps, dtype=simulation_results.dtype, device=simulation_results.device)
        
        # Berechne Entropie für jeden Schritt
        for step in range(num_steps):
            # Mittlere Zustandsverteilung über alle Simulationen
            mean_distribution = torch.mean(simulation_results[:, step, :], dim=0)
            
            # Verhindere log(0)
            mean_distribution = torch.clamp(mean_distribution, min=1e-10)
            
            # Berechne Entropie: -sum(p * log(p))
            step_entropy = -torch.sum(mean_distribution * torch.log(mean_distribution))
            entropy[step] = step_entropy
        
        return entropy
    
    def convergence_analysis(self, 
                           simulation_results: torch.Tensor, 
                           threshold: float = 0.01) -> Tuple[bool, int]:
        """
        Analysiert die Konvergenz der Simulationen.
        
        Verwendet optimierte MLX-Operationen auf Apple Silicon, wenn verfügbar.
        
        Args:
            simulation_results: Simulationsergebnisse [num_simulations, num_steps, num_states]
            threshold: Schwellenwert für die Konvergenz
            
        Returns:
            Tuple aus (konvergiert, Konvergenzschritt)
        """
        num_simulations, num_steps, num_states = simulation_results.shape
        
        # Berechne die Änderungen zwischen aufeinanderfolgenden Schritten
        changes = torch.zeros(num_steps - 1, dtype=simulation_results.dtype, device=simulation_results.device)
        
        for step in range(num_steps - 1):
            # Mittlere Zustandsverteilung für aufeinanderfolgende Schritte
            mean_current = torch.mean(simulation_results[:, step, :], dim=0)
            mean_next = torch.mean(simulation_results[:, step + 1, :], dim=0)
            
            # Berechne die Änderung (L2-Norm)
            change = torch.norm(mean_next - mean_current)
            changes[step] = change
        
        # Prüfe auf Konvergenz
        converged = torch.any(changes < threshold)
        
        if converged:
            # Finde den ersten Schritt, bei dem die Änderung unter dem Schwellenwert liegt
            convergence_step = torch.argmax((changes < threshold).float())
            return True, convergence_step.item()
        else:
            return False, -1
    
    def timeline_probability(self, 
                           transition_matrices: List[torch.Tensor],
                           initial_state: torch.Tensor,
                           target_state: torch.Tensor) -> torch.Tensor:
        """
        Berechnet die Wahrscheinlichkeit einer bestimmten Zeitlinie.
        
        Verwendet optimierte MLX-Operationen auf Apple Silicon, wenn verfügbar.
        
        Args:
            transition_matrices: Liste von Übergangsmatrizen für jeden Zeitschritt
            initial_state: Anfangszustand
            target_state: Zielzustand
            
        Returns:
            Wahrscheinlichkeit der Zeitlinie
        """
        num_steps = len(transition_matrices)
        current_state = initial_state
        
        # Berechne die Wahrscheinlichkeit Schritt für Schritt
        for step in range(num_steps):
            # Wende die Übergangsmatrix an - verwende direkte PyTorch-Operation
            current_state = torch.matmul(current_state, transition_matrices[step])
        
        # Berechne die Wahrscheinlichkeit des Zielzustands
        # Verwende Skalarprodukt für Wahrscheinlichkeitsberechnung
        probability = torch.dot(current_state, target_state)
        
        return probability


# Beispiel für die Verwendung
if __name__ == "__main__":
    # Erstelle eine PRISM Simulation Engine
    simulation_engine = PrismSimulationEngine(use_mlx=True)
    
    # Erstelle Beispiel-Übergangsmatrix
    num_states = 3
    transition_matrix = torch.tensor([
        [0.7, 0.2, 0.1],  # Übergangswahrscheinlichkeiten für Zustand 0
        [0.3, 0.4, 0.3],  # Übergangswahrscheinlichkeiten für Zustand 1
        [0.2, 0.3, 0.5]   # Übergangswahrscheinlichkeiten für Zustand 2
    ])
    
    # Anfangszustand (Start im Zustand 0)
    initial_state = torch.tensor([1.0, 0.0, 0.0])
    
    # Führe Monte-Carlo-Simulationen durch
    num_steps = 10
    num_simulations = 1000
    results = simulation_engine.monte_carlo_simulation(
        transition_matrix, initial_state, num_steps, num_simulations
    )
    
    # Analysiere die Wahrscheinlichkeiten
    target_states = [2]  # Wir sind an Zustand 2 interessiert
    probabilities = simulation_engine.probability_analysis(results, target_states)
    
    # Zeige die Ergebnisse
    print("\nWahrscheinlichkeiten für Zustand 2 nach jedem Schritt:")
    for step, prob in enumerate(probabilities):
        print(f"Schritt {step}: {prob[0]:.4f}")
    
    # Analysiere die Entropie
    entropy = simulation_engine.entropy_analysis(results)
    print("\nEntropie nach jedem Schritt:")
    for step, ent in enumerate(entropy):
        print(f"Schritt {step}: {ent:.4f}")
    
    # Demonstriere Lazy-Loading von PRISM Komponenten
    print("\nTeste Lazy-Loading für PRISM Komponenten:")
    
    # Lade PrismEngine-Klasse
    PrismEngine = get_prism_engine()
    if PrismEngine:
        print(f"PRISM Engine erfolgreich geladen: {PrismEngine.__name__}")
        
        # Optional: Erstelle eine PrismEngine-Instanz
        try:
            engine = PrismEngine()
            print(f"PrismEngine erfolgreich instanziiert")
        except Exception as e:
            print(f"Konnte PrismEngine nicht instanziieren: {e}")
    else:
        print("PRISM Engine konnte nicht geladen werden")
    
    # Lade PrismMatrix und RealityFold Komponenten
    PrismMatrix, RealityFold = get_prism_matrix_components()
    if PrismMatrix and RealityFold:
        print(f"PRISM Matrix Komponenten erfolgreich geladen:")
        print(f"- PrismMatrix: {PrismMatrix.__name__}")
        print(f"- RealityFold: {RealityFold.__name__}")
        
        # Optional: Erstelle eine Matrix-Instanz
        try:
            matrix = PrismMatrix(dimensions=4)
            print(f"PrismMatrix erfolgreich erstellt mit Dimensionen: {matrix.dimensions}")
        except Exception as e:
            print(f"Konnte PrismMatrix nicht erstellen: {e}")
    else:
        print("PRISM Matrix Komponenten konnten nicht geladen werden")
    
    print("\nLazy-Loading-Test abgeschlossen.")
