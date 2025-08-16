#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - PRISM-Engine

Echtzeit-Realitätsmodulation & Wahrscheinlichkeitssteuerung
Die PRISM-Engine ermöglicht MISO, Realität nicht nur zu analysieren, sondern zu simulieren,
vorwegzunehmen und gezielt zu modulieren.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import math
import logging
import uuid
import queue
import random
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable

# Importiere Basisklassen und -typen, um zirkuläre Importe zu vermeiden
from miso.simulation.prism_base import (
    SimulationConfig, SimulationStatus, TimelineType,
    TimeNode, Timeline, calculate_probability, sigmoid
)

# Konfiguriere Logging
logger = logging.getLogger("MISO.simulation.prism")

# Prüfen, ob Apple Silicon verfügbar ist und entsprechende Optimierungen aktivieren
is_apple_silicon = sys.platform == 'darwin' and 'arm' in os.uname().machine
if is_apple_silicon:
    # Apple Neural Engine Optimierungen
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    device = torch.device('mps')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import von internen Modulen
try:
    from miso.core.omega_core import OmegaCore
    from miso.math.mprime_engine import MPrimeEngine
    from miso.math.tensor_ops import MISOTensor, MLXTensor, TorchTensor
    from miso.logic.qlogik_integration import QLOGIKIntegrationManager
    # Verbesserte T-MATHEMATICS-Integration
    from miso.math.t_mathematics.integration_manager import get_t_math_integration_manager
    from miso.math.t_mathematics.compat import TMathConfig
    # Erweiterte Paradoxauflösung
    from miso.simulation.paradox_resolution import (
        get_paradox_resolution_manager,
        ParadoxSeverity,
        ParadoxCategory,
        ParadoxClassification,
        ResolutionStrategy
    )
    HAS_PARADOX_RESOLUTION = True
except ImportError as e:
    logger.warning(f"Einige Module konnten nicht importiert werden: {e}")
    HAS_PARADOX_RESOLUTION = False
    # Importiere TMathEngine direkt, ohne Unterklassen zu importieren
    import miso.math.t_mathematics.engine as t_math_engine_module
    TMathEngine = t_math_engine_module.TMathEngine
    HAS_DEPENDENCIES = True
except ImportError as e:
    logger.warning(f"Einige Abhängigkeiten konnten nicht importiert werden. PRISM-Engine läuft im eingeschränkten Modus: {e}")
    HAS_DEPENDENCIES = False

# Prüfen, ob MLX verfügbar ist
try:
    import mlx.core as mx
    HAS_MLX = True
    logger.info("MLX erfolgreich importiert - Apple Silicon Optimierung verfügbar")
except ImportError:
    HAS_MLX = False
    logger.warning("MLX konnte nicht importiert werden. Apple Silicon Optimierung nicht verfügbar.")

# Lazy imports für Untermodule werden durch die Factory behandelt
# Dies vermeidet zirkuläre Importe und verbessert die Modularität
try:
    from .prism_factory import (
        get_prism_registry, get_prism_matrix, get_time_scope_unit,
        get_predictive_stream_analyzer, get_pattern_dissonance_scanner
    )
    HAS_FACTORY = True
    logger.info("PRISM-Factory erfolgreich importiert - Optimierte Dependency Injection verfügbar")
except ImportError as e:
    logger.warning(f"PRISM-Factory konnte nicht importiert werden: {e}. Verwende direkte Importe.")
    HAS_FACTORY = False
    # Fallback auf direkte Importe nur wenn nötig
    from .prism_matrix import PrismMatrix
    from .predictive_stream import PredictiveStreamAnalyzer
    from .time_scope import TimeScopeUnit
    from .pattern_dissonance import PatternDissonanceScanner

# Importiere Timeline-Komponenten von ECHO-PRIME
from miso.timeline import TemporalEvent
# Hinweis: Timeline und TimeNode werden bereits aus prism_base importiert


class RealityForker:
    """
    Teilt Realitätslinien in Alternativstränge & simuliert Outcomes
    Erzeugt virtuelle Zeitachsen aus Ist-Zustand + potentiellen Varianten
    """
    
    def __init__(self):
        """Initialisiert den RealityForker"""
        self.current_reality = {}
        self.alternative_realities = []
        self.simulation_engine = None
        self.initialize_simulation_engine()
    
    def initialize_simulation_engine(self):
        """Initialisiert die Simulationsengine"""
        logger.info("Initialisiere Simulationsengine für RealityForker")
        
        if HAS_FACTORY:
            # Verwende die Factory, um zirkuläre Abhängigkeiten zu vermeiden
            # Wir holen hier nicht direkt die PrismEngine-Instanz, da diese den RealityForker verwendet
            # Stattdessen erstellen wir eine leichte Simulationsengine für interne Berechnungen
            registry = get_prism_registry()
            
            # Matrix für Berechnungen
            matrix = get_prism_matrix(dimensions=4, initial_size=10)
            self.simulation_engine = {
                "matrix": matrix,
                "compute": self._compute_simulation
            }
            logger.debug("Simulationsengine für RealityForker initialisiert mit Factory-Pattern")
        else:
            # Fallback: Minimale Eigenimplementierung wenn Factory nicht verfügbar
            from .prism_matrix import PrismMatrix
            matrix = PrismMatrix(dimensions=4, initial_size=10)
            self.simulation_engine = {
                "matrix": matrix,
                "compute": self._compute_simulation
            }
            logger.debug("Simulationsengine für RealityForker mit Fallback initialisiert")
    
    def _compute_simulation(self, data, steps):
        """Interne Berechnungsfunktion für Simulationen"""
        if self.simulation_engine is None or "matrix" not in self.simulation_engine:
            logger.error("Simulationsengine nicht korrekt initialisiert")
            return {"status": "error", "message": "Engine nicht initialisiert"}
            
        # Verwende die Matrix für Berechnungen
        matrix = self.simulation_engine["matrix"]
        
        # Basierend auf den Eingabedaten und Simulationsschritten
        # berechnen wir ein Ergebnis unter Verwendung unserer Matrix
        # (Vereinfachte Implementierung für Demonstration)
        
        # Erzeuge eine leichte Variation der Eingabedaten mit dem _apply_variation der Matrix
        if isinstance(data, dict):
            # Für Dict-Daten simulieren wir Wahrscheinlichkeiten
            result = {}
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    # Einfacher numerischer Wert: wende Variation an
                    variation_factor = 0.1 * (steps / 100)  # Skaliere mit Anzahl der Schritte
                    result[key] = matrix._apply_variation(
                        np.array([value]), variation_factor
                    )[0]
                else:
                    # Nicht-numerische Werte unverändert übernehmen
                    result[key] = value
        else:
            # Für andere Datentypen verwenden wir direkt _apply_variation
            variation_factor = 0.1 * (steps / 100)
            result = matrix._apply_variation(np.array(data), variation_factor)
        
        return {
            "status": "success", 
            "input": data, 
            "result": result, 
            "steps": steps
        }
    
    def fork_reality(self, current_state: Dict[str, Any], variations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Teilt eine Realitätslinie in mehrere Alternativen basierend auf Variationen
        
        Args:
            current_state: Aktueller Zustand der Realität
            variations: Liste von Variationen, die auf den aktuellen Zustand angewendet werden sollen
            
        Returns:
            Liste der alternativen Realitäten
        """
        # Prüfe, ob die Simulationsengine initialisiert ist
        if self.simulation_engine is None:
            logger.warning("Simulationsengine nicht initialisiert. Initialisiere jetzt.")
            self.initialize_simulation_engine()
        
        # Speichere den aktuellen Zustand
        self.current_reality = current_state
        self.alternative_realities = []
        
        # Verwende die optimierte Matrix für Variationen, wenn verfügbar
        matrix = self.simulation_engine.get("matrix") if self.simulation_engine else None
        
        # Wende jede Variation auf den aktuellen Zustand an
        for variation in variations:
            # Kombiniere aktuellen Zustand mit Variation
            alternative = current_state.copy()
            
            # Wende die Variation mit optimierter Methode an, falls verfügbar
            if matrix and hasattr(matrix, '_apply_variation'):
                # Verwende die Matrix-Variation für numerische Werte
                for key, value in variation.items():
                    if key in alternative and isinstance(alternative[key], (int, float)) and isinstance(value, (int, float)):
                        # Berechne Variationsfaktor basierend auf der Differenz
                        diff = abs(value - alternative[key])
                        base = max(abs(alternative[key]), 1.0)
                        var_factor = min(diff / base, 1.0)
                        
                        # Wende _apply_variation an
                        alternative[key] = float(matrix._apply_variation(
                            np.array([alternative[key]]), var_factor
                        )[0])
                    else:
                        # Nicht-numerische Werte oder neue Schlüssel direkt übernehmen
                        alternative[key] = value
            else:
                # Einfaches Update ohne optimierte Variation
                alternative.update(variation)
            
            # Füge Alternative mit Zeitstempel hinzu
            alternative["timestamp"] = time.time()
            self.alternative_realities.append(alternative)
            
        return self.alternative_realities
    
    def simulate_outcome(self, reality: Dict[str, Any], steps: int = 100) -> Dict[str, Any]:
        """
        Simuliert das Ergebnis einer alternativen Realität
        
        Args:
            reality: Zu simulierende Realität
            steps: Anzahl der Simulationsschritte
            
        Returns:
            Simulationsergebnis
        """
        # Prüfe, ob wir die optimierte Simulationsengine verwenden können
        if self.simulation_engine and "compute" in self.simulation_engine:
            # Verwende die optimierte Berechnungsfunktion
            try:
                simulation_result = self.simulation_engine["compute"](reality, steps)
                if simulation_result.get("status") == "success":
                    result = simulation_result.get("result", {})
                    # Stelle sicher, dass es ein Dictionary ist
                    if not isinstance(result, dict):
                        result = {"simulated_value": result}
                    
                    # Füge ursprüngliche Realitätswerte hinzu, die nicht simuliert wurden
                    for key, value in reality.items():
                        if key not in result:
                            result[key] = value
                    
                    # Füge Metadaten hinzu
                    result["steps"] = steps
                    return result
            except Exception as e:
                logger.error(f"Fehler bei optimierter Simulation: {e}")
                # Fallback auf einfache Simulation
        
        # Einfache Simulation: Füge Wahrscheinlichkeit hinzu
        result = reality.copy()
        
        # Berechne eine Wahrscheinlichkeit basierend auf Realitätsdaten
        if "impact" in reality and "complexity" in reality:
            probability = (reality["impact"] * 0.7 + reality["complexity"] * 0.3) / 2
        else:
            # Verwende Entropie-basierte Wahrscheinlichkeit für komplexere Fälle
            num_keys = len(reality)
            numeric_values = [v for v in reality.values() if isinstance(v, (int, float))]
            if numeric_values:
                # Normalisiere numerische Werte und berechne Varianz als Komplexitätsmaß
                normalized = np.array(numeric_values) / (max(numeric_values) if max(numeric_values) != 0 else 1)
                variance = np.var(normalized) if len(normalized) > 1 else 0
                probability = 0.5 + (variance - 0.5) * min(steps/200, 1)  # Skaliere mit Simulationsschritten
            else:
                probability = 0.5
            
        # Füge Wahrscheinlichkeit und Simulationsschritte hinzu
        result["probability"] = max(0.01, min(0.99, probability))  # Begrenze auf sinnvollen Bereich
        result["steps"] = steps
        
        return result
    
    def merge_realities(self, realities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Führt alternative Realitäten zu einer zusammen
        
        Args:
            realities: Liste von alternativen Realitäten
            
        Returns:
            Zusammengeführte Realität
        """
        if not realities:
            logger.warning("Keine Realitäten zum Zusammenführen")
            return {"probability": 0.0, "error": "Keine Realitäten"}
            
        # Prüfe, ob wir die Matrix für Berechnungen verwenden können
        matrix = self.simulation_engine.get("matrix") if self.simulation_engine else None
        
        # Erstelle eine leere zusammengeführte Realität
        merged = {}
        
        # Führe Realitäten zusammen, indem wir gewichtete Durchschnitte bilden
        total_probability = sum(r.get("probability", 0.5) for r in realities)
        if total_probability == 0:
            total_probability = 1.0
            
        # Sammle alle einzigartigen Schlüssel aus allen Realitäten
        all_keys = set().union(*(r.keys() for r in realities))
        
        for key in all_keys:
            if key in ("probability", "steps", "timestamp"):
                continue
                
            # Sammle Werte für diesen Schlüssel aus allen Realitäten
            values = []
            weights = []
            
            for reality in realities:
                if key in reality:
                    values.append(reality[key])
                    weights.append(reality.get("probability", 0.5) / total_probability)
                    
            if not values:
                continue
                
            # Berechne gewichteten Durchschnitt für numerische Werte
            if all(isinstance(v, (int, float)) for v in values):
                if matrix and hasattr(matrix, '_apply_variation') and len(values) > 1:
                    # Verwende Matrix für gewichtete Kombination mit Variation
                    # um realistischere Ergebnisse zu erzielen
                    base_value = sum(v * w for v, w in zip(values, weights))
                    weight_variance = np.var(weights) if len(weights) > 1 else 0
                    variation_factor = min(weight_variance * 3, 0.3)  # Skaliere mit Gewichtsvarianz
                    
                    merged[key] = float(matrix._apply_variation(
                        np.array([base_value]), variation_factor
                    )[0])
                else:
                    # Standard gewichteter Durchschnitt
                    merged[key] = sum(v * w for v, w in zip(values, weights))
            else:
                # Für nicht-numerische Werte: Wähle den Wert mit höchstem Gewicht
                merged[key] = values[weights.index(max(weights))]
                
        # Berechne Gesamtwahrscheinlichkeit
        # Wir verwenden hier einen modifizierten Durchschnitt, der höhere Wahrscheinlichkeiten bevorzugt
        probabilities = [r.get("probability", 0.5) for r in realities]
        if probabilities:
            # Logarithmisch gewichteter Durchschnitt für realistischere Wahrscheinlichkeiten
            weighted_prob = np.exp(sum(np.log(max(p, 0.01)) for p in probabilities) / len(probabilities))
            merged["probability"] = max(0.01, min(0.99, weighted_prob))
        else:
            merged["probability"] = 0.5
        
        # Füge Zeitstempel hinzu
        merged["timestamp"] = time.time()
        
        return merged


class FeedbackLoopModulator:
    """
    Schaltet auf Basis der Realitätssimulation Systempfade um
    Passt MISOs Verhalten an Simulationsergebnisse an
    """
    
    def __init__(self):
        """Initialisiert den FeedbackLoopModulator"""
        self.feedback_history = []
        self.adaptation_rate = 0.1
        self.threshold = 0.7
        logger.info("FeedbackLoopModulator initialisiert")
    
    def process_feedback(self, simulation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verarbeitet Feedback aus Simulationsergebnissen
        
        Args:
            simulation_result: Simulationsergebnis
            
        Returns:
            Verarbeitetes Feedback
        """
        logger.info("Verarbeite Feedback aus Simulationsergebnis")
        
        # Extrahiere Simulationsergebnisse
        results = simulation_result.get("results", [])
        
        if not results:
            return {"status": "error", "message": "Keine Simulationsergebnisse gefunden"}
        
        # Analysiere Ergebnisse
        feedback = {
            "status": "success",
            "timestamp": time.time(),
            "metrics": {},
            "actions": []
        }
        
        # Berechne Metriken
        metrics = {}
        for result in results:
            for key, value in result.items():
                if isinstance(value, (int, float)) and not key.startswith("_"):
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
        
        # Berechne Durchschnitt und Standardabweichung für jede Metrik
        for key, values in metrics.items():
            if values:
                avg = sum(values) / len(values)
                std = np.std(values) if len(values) > 1 else 0.0
                feedback["metrics"][key] = {"avg": avg, "std": std}
        
        # Generiere Aktionen basierend auf Metriken
        for key, metric in feedback["metrics"].items():
            if metric["std"] > self.threshold:
                feedback["actions"].append({
                    "type": "stabilize",
                    "target": key,
                    "confidence": 1.0 - (metric["std"] / (metric["avg"] if metric["avg"] != 0 else 1.0))
                })
        
        # Speichere Feedback in der Historie
        self.feedback_history.append(feedback)
        
        return feedback
    
    def adapt_system_behavior(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Passt das Systemverhalten basierend auf Feedback an
        
        Args:
            feedback: Feedback-Daten
            
        Returns:
            Anpassungsergebnis
        """
        logger.info("Passe Systemverhalten basierend auf Feedback an")
        
        if feedback.get("status") != "success":
            return {"status": "error", "message": "Ungültiges Feedback"}
        
        # Extrahiere Aktionen
        actions = feedback.get("actions", [])
        
        # Führe Anpassungen durch
        adaptations = []
        for action in actions:
            action_type = action.get("type")
            target = action.get("target")
            confidence = action.get("confidence", 0.0)
            
            if action_type == "stabilize" and confidence > self.threshold:
                # Implementiere Stabilisierungslogik
                adaptations.append({
                    "type": "stabilize",
                    "target": target,
                    "applied": True,
                    "adaptation_factor": self.adaptation_rate * confidence
                })
        
        return {
            "status": "success",
            "adaptations": adaptations,
            "adaptation_count": len(adaptations)
        }
    
    def detect_feedback_loop(self) -> bool:
        """
        Erkennt, ob ein Feedback-Loop vorhanden ist
        
        Returns:
            True, wenn ein Feedback-Loop erkannt wurde, sonst False
        """
        if len(self.feedback_history) < 3:
            return False
        
        # Analysiere die letzten drei Feedback-Einträge
        recent_feedback = self.feedback_history[-3:]
        
        # Prüfe, ob die gleichen Aktionen wiederholt auftreten
        action_counts = {}
        for feedback in recent_feedback:
            for action in feedback.get("actions", []):
                action_key = f"{action.get('type')}:{action.get('target')}"
                action_counts[action_key] = action_counts.get(action_key, 0) + 1
        
        # Wenn eine Aktion in allen drei Feedback-Einträgen vorkommt, haben wir einen Loop
        for count in action_counts.values():
            if count == 3:
                logger.warning("Feedback-Loop erkannt!")
                return True
        
        return False


class PrismEngine:
    """
    PRISM-Engine: Echtzeit-Realitätsmodulation & Wahrscheinlichkeitssteuerung
    Zentrales Modul für Realitätssimulation und Wahrscheinlichkeitsanalyse
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert die PRISM-Engine
        
        Args:
            config: Konfigurationsparameter
        """
        self.config = config or {}
        self.running = False
        self.initialized = False
        self.simulation_thread = None
        self.event_queue = queue.Queue()
        
        # MLX-Konfiguration
        self.use_mlx = self.config.get("use_mlx", True) and is_apple_silicon
        self.precision = self.config.get("precision", "float16")
        
        # Verbesserte T-MATHEMATICS-Integration
        if HAS_DEPENDENCIES:
            try:
                # Verwende den Integration Manager, um eine konsistente Engine-Instanz zu erhalten
                t_math_manager = get_t_math_integration_manager()
                self.t_math_engine = self.config.get("t_math_engine")
                
                # Wenn keine Engine übergeben wurde, erstelle eine neue
                if self.t_math_engine is None:
                    # Erstelle eine neue T-MATHEMATICS Engine-Instanz
                    t_math_config = TMathConfig(
                        optimize_for_apple_silicon=self.use_mlx,
                        precision=self.precision
                    )
                    self.t_math_engine = TMathEngine(config=t_math_config)
                    
                # Registriere die Engine beim Manager
                t_math_manager.register_engine("prism_engine", self.t_math_engine)
                
                # Hole die PRISM-spezifische Integration
                self.prism_simulation_engine = t_math_manager.get_prism_integration()
                
                logger.info(f"T-Mathematics Engine erfolgreich integriert mit MLX={self.use_mlx}")
            except Exception as e:
                logger.error(f"Fehler bei der Initialisierung der T-MATHEMATICS Engine: {e}")
                self.t_math_engine = None
                self.prism_simulation_engine = None
        else:
            self.t_math_engine = None
            self.prism_simulation_engine = None
        
        # Initialisiere Komponenten mit Factory-Pattern (wenn verfügbar)
        if HAS_FACTORY:
            # Verwende die Factory-Methoden, um zirkuläre Importe zu vermeiden
            self.matrix = get_prism_matrix(
                dimensions=self.config.get("matrix_dimensions", 4),
                initial_size=self.config.get("matrix_initial_size", 10)
            )
            
            self.stream_analyzer = get_predictive_stream_analyzer(
                sequence_length=self.config.get("sequence_length", 10),
                prediction_horizon=self.config.get("prediction_horizon", 5)
            )
            
            self.time_scope = get_time_scope_unit()
            
            self.dissonance_scanner = get_pattern_dissonance_scanner(
                dissonance_threshold=self.config.get("dissonance_threshold", 0.3)
            )
            
            # Registriere die Engine selbst bei der Registry
            # Dies ermöglicht anderen Komponenten den Zugriff auf die Engine
            get_prism_registry().register_component("prism_engine", self)
            logger.debug("PrismEngine bei Registry registriert")
        else:
            # Fallback: Direkter Konstruktoraufruf, wenn Factory nicht verfügbar ist
            self.matrix = PrismMatrix(
                dimensions=self.config.get("matrix_dimensions", 4),
                initial_size=self.config.get("matrix_initial_size", 10)
            )
            
            self.stream_analyzer = PredictiveStreamAnalyzer(
                sequence_length=self.config.get("sequence_length", 10),
                prediction_horizon=self.config.get("prediction_horizon", 5)
            )
            
            self.time_scope = TimeScopeUnit()
            
            self.dissonance_scanner = PatternDissonanceScanner(
                dissonance_threshold=self.config.get("dissonance_threshold", 0.3)
            )
        
        self.reality_forker = RealityForker()
        
        self.feedback_modulator = FeedbackLoopModulator()
        
        # Integration mit anderen MISO-Systemen
        self.omega_core = None
        self.mprime_engine = None
        self.qlogik_manager = None
        
        if HAS_DEPENDENCIES:
            self.initialize_dependencies()
        
        # Initialisiere Timeline-Registrierung
        self.registered_timelines = {}
        
        logger.info("PRISM-Engine initialisiert")
    
    def initialize_dependencies(self):
        """Initialisiert Abhängigkeiten zu anderen MISO-Systemen"""
        try:
            self.omega_core = OmegaCore()
            self.mprime_engine = MPrimeEngine()
            self.qlogik_manager = QLOGIKIntegrationManager()
            logger.info("MISO-Systemabhängigkeiten initialisiert")
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung von Abhängigkeiten: {e}")
    
    def start(self):
        """Startet die PRISM-Engine"""
        if self.running:
            logger.warning("PRISM-Engine läuft bereits")
            return False
        
        self.running = True
        self.simulation_thread = threading.Thread(target=self._simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        logger.info("PRISM-Engine gestartet")
        return True
    
    def stop(self):
        """Stoppt die PRISM-Engine"""
        if not self.running:
            logger.warning("PRISM-Engine läuft nicht")
            return False
        
        self.running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=5.0)
        
        logger.info("PRISM-Engine gestoppt")
        return True
    
    def _simulation_loop(self):
        """Hauptsimulationsschleife"""
        logger.info("Simulationsschleife gestartet")
        
        while self.running:
            try:
                # Verarbeite Ereignisse aus der Warteschlange
                try:
                    event = self.event_queue.get(block=False)
                    self._process_event(event)
                except queue.Empty:
                    pass
                
                # Führe Simulationsschritte durch
                self._perform_simulation_step()
                
                # Kurze Pause, um CPU-Last zu reduzieren
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Fehler in der Simulationsschleife: {e}")
        
        logger.info("Simulationsschleife beendet")
    
    def _process_event(self, event: Dict[str, Any]):
        """Verarbeitet ein Ereignis aus der Warteschlange"""
        event_type = event.get("type")
        event_data = event.get("data", {})
        
        logger.debug(f"Verarbeite Ereignis vom Typ {event_type}")
        
        if event_type == "register_stream":
            self.stream_analyzer.register_data_stream(
                event_data.get("stream_id"),
                event_data.get("initial_data")
            )
        
        elif event_type == "register_pattern":
            self.dissonance_scanner.register_pattern(
                event_data.get("pattern_id"),
                event_data.get("pattern_data"),
                event_data.get("pattern_type", "sequence"),
                event_data.get("metadata")
            )
        
        elif event_type == "fork_reality":
            self.reality_forker.fork_reality(
                event_data.get("current_state", {}),
                event_data.get("variations", [])
            )
        
        # Weitere Ereignistypen hier verarbeiten
    
    def _perform_simulation_step(self):
        """Führt einen Simulationsschritt durch"""
        # Hier würde die eigentliche Simulationslogik implementiert werden
        pass
    
    def add_data_point(self, stream_id: str, value: Any):
        """
        Fügt einen Datenpunkt zu einem Datenstrom hinzu
        
        Args:
            stream_id: ID des Datenstroms
            value: Datenpunkt
        """
        self.stream_analyzer.add_data_point(stream_id, value)
        
        # Aktualisiere auch die PrismMatrix
        # (Hier müsste eine Logik zur Koordinatenbestimmung implementiert werden)
        coordinates = self._determine_coordinates_for_stream(stream_id, value)
        if coordinates:
            self.matrix.set_data_point(coordinates, value)
    
    def _determine_coordinates_for_stream(self, stream_id: str, value: Any) -> Optional[Tuple]:
        """Bestimmt die Koordinaten für einen Datenpunkt in der Matrix"""
        # Diese Methode würde eine Logik implementieren, um Datenströme
        # in der n-dimensionalen Matrix zu positionieren
        # Für jetzt geben wir None zurück
        return None
        
    def evaluate_probability_recommendation(self, probability: float) -> Dict[str, Any]:
        """
        Bewertet eine Wahrscheinlichkeit und gibt eine Handlungsempfehlung zurück
        
        Args:
            probability: Wahrscheinlichkeitswert (0.0 bis 1.0)
            
        Returns:
            Handlungsempfehlung mit Risikobewertung
        """
        # Stelle sicher, dass die Wahrscheinlichkeit im gültigen Bereich liegt
        probability = max(0.0, min(1.0, probability))
        
        # Kategorisiere die Wahrscheinlichkeit nach den neuen Schwellenwerten (40-90%)
        if probability < 0.4:
            risk_level = "low"
            confidence = "Niedrig"
            recommendation = "Warnung: Geringe Wahrscheinlichkeit"
        elif probability <= 0.9:
            risk_level = "medium"
            confidence = "Mittel"
            recommendation = "Handlungsempfehlung mit Risiko"
        else:
            risk_level = "high"
            confidence = "Hoch"
            recommendation = "Starke Handlungsempfehlung"
        
        # Formatiere die Wahrscheinlichkeit als Prozentsatz
        probability_percentage = f"{probability * 100:.2f}%"
        
        return {
            "probability": probability,
            "probability_percentage": probability_percentage,
            "risk_level": risk_level,
            "confidence": confidence,
            "recommendation": recommendation,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def predict_trend(self, stream_id: str, steps: int = None) -> Dict[str, Any]:
        """
        Sagt einen Trend für einen Datenstrom voraus
        
        Args:
            stream_id: ID des Datenstroms
            steps: Anzahl der vorherzusagenden Schritte
            
        Returns:
            Vorhersageergebnis
        """
        # Trainiere das Modell, falls nötig
        training_result = self.stream_analyzer.train_model(stream_id)
        
        if training_result.get("status") != "success":
            return {"status": "error", "message": "Modelltraining fehlgeschlagen"}
        
        # Mache Vorhersage
        predictions = self.stream_analyzer.predict_next_values(stream_id, steps)
        
        return {
            "status": "success",
            "stream_id": stream_id,
            "predictions": predictions,
            "confidence": training_result.get("final_loss", 1.0)
        }
    
    def scan_pattern_dissonance(self, pattern_id: str, current_data: Any) -> Dict[str, Any]:
        """
        Scannt nach Abweichungen von einem erwarteten Muster
        
        Args:
            pattern_id: ID des Musters
            current_data: Aktuelle Daten
            
        Returns:
            Dissonanzergebnis
        """
        dissonance = self.dissonance_scanner.scan_for_dissonance(current_data, pattern_id)
        
        return {
            "pattern_id": pattern_id,
            "dissonance": dissonance,
            "is_significant": dissonance > self.dissonance_scanner.dissonance_threshold
        }
    
    def simulate_reality_fork(self, current_state: Dict[str, Any], 
                            variations: List[Dict[str, Any]], 
                            steps: int = 100) -> Dict[str, Any]:
        """
        Simuliert eine Realitätsverzweigung
        
        Args:
            current_state: Aktueller Zustand
            variations: Liste von Variationen
            steps: Anzahl der Simulationsschritte
            
        Returns:
            Simulationsergebnis
        """
        # Erzeuge Alternativrealitäten
        alternative_realities = self.reality_forker.fork_reality(current_state, variations)
        
        # Simuliere jede Realität
        simulation_results = []
        for reality in alternative_realities:
            result = self.reality_forker.simulate_outcome(reality, steps)
            simulation_results.append(result)
        
        # Führe die Realitäten zusammen
        merged_reality = self.reality_forker.merge_realities(simulation_results)
        
        # Verarbeite Feedback
        feedback = self.feedback_modulator.process_feedback({
            "results": simulation_results
        })
        
        # Passe Systemverhalten an
        adaptation = self.feedback_modulator.adapt_system_behavior(feedback)
        
        # Berechne die Wahrscheinlichkeit des Hauptergebnisses
        # (Dies würde in der tatsächlichen Implementierung komplexer sein)
        probability = merged_reality.get("probability", 0.0)
        recommendation = self.evaluate_probability_recommendation(probability)
        
        return {
            "original_state": current_state,
            "variations": variations,
            "alternative_realities": len(alternative_realities),
            "simulation_steps": steps,
            "merged_outcome": merged_reality,
            "feedback": feedback,
            "adaptation": adaptation,
            "probability_analysis": recommendation
        }
    
    def integrate_with_t_mathematics(self, tensor_operation: str, tensor_data: Any, backend: str = None, async_execution: bool = False):
        """
        Integriert mit der T-Mathematics Engine für Tensor-Operationen
        
        Args:
            tensor_operation: Gewünschte Tensor-Operation
            tensor_data: Eingabedaten für die Operation
            backend: Gewünschtes Backend (MLX, PyTorch, NumPy)
            async_execution: Ob die Operation asynchron ausgeführt werden soll
            
        Returns:
            Ergebnis der Tensor-Operation oder Task-ID bei asynchroner Ausführung
        """
        # Wenn die T-Mathematics Engine nicht initialisiert ist, versuche sie zu initialisieren
        if not self.t_math_engine and HAS_DEPENDENCIES:
            try:
                # Versuche, die Engine zu initialisieren
                logger.info("Versuche, die T-Mathematics Engine zu initialisieren...")
                t_math_manager = get_t_math_integration_manager()
                
                # Erstelle eine neue T-MATHEMATICS Engine-Instanz
                t_math_config = TMathConfig(
                    optimize_for_apple_silicon=self.use_mlx,
                    precision=self.precision
                )
                self.t_math_engine = TMathEngine(config=t_math_config)
                
                # Registriere die Engine beim Manager
                t_math_manager.register_engine("prism_engine", self.t_math_engine)
                
                # Hole die PRISM-spezifische Integration
                self.prism_simulation_engine = t_math_manager.get_prism_integration()
                
                logger.info(f"T-Mathematics Engine erfolgreich initialisiert mit MLX={self.use_mlx}")
            except Exception as e:
                logger.error(f"Fehler bei der Initialisierung der T-MATHEMATICS Engine: {e}")
                # Fallback: Verwende NumPy für die Berechnung
                return self._fallback_tensor_operation(tensor_operation, tensor_data)
        
        # Wenn die Engine immer noch nicht verfügbar ist, verwende Fallback
        if not self.t_math_engine:
            return self._fallback_tensor_operation(tensor_operation, tensor_data)
        
        try:
            # Verwende die PRISM-spezifische Integration, falls verfügbar
            if self.prism_simulation_engine and hasattr(self.prism_simulation_engine, tensor_operation):
                result = getattr(self.prism_simulation_engine, tensor_operation)(*tensor_data if isinstance(tensor_data, (list, tuple)) else [tensor_data])
                return {
                    "status": "success",
                    "operation": tensor_operation,
                    "backend": "prism_simulation_engine",
                    "result": result
                }
            
            # Führe die gewünschte Operation aus
            if tensor_operation == "matmul":
                result = self.t_math_engine.matmul(tensor_data[0], tensor_data[1])
            elif tensor_operation == "svd":
                result = self.t_math_engine.svd(tensor_data)
            elif tensor_operation == "attention":
                result = self.t_math_engine.attention(*tensor_data)
            elif hasattr(self.t_math_engine, tensor_operation):
                # Dynamischer Aufruf für andere Operationen
                result = getattr(self.t_math_engine, tensor_operation)(*tensor_data if isinstance(tensor_data, (list, tuple)) else [tensor_data])
            else:
                # Fallback für unbekannte Operationen
                return self._fallback_tensor_operation(tensor_operation, tensor_data)
            
            return {
                "status": "success",
                "operation": tensor_operation,
                "backend": backend or ("mlx" if self.use_mlx else "torch"),
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Fehler bei der Integration mit T-Mathematics: {e}")
            # Bei Fehler: Fallback auf NumPy
            return self._fallback_tensor_operation(tensor_operation, tensor_data)
    
    def _fallback_tensor_operation(self, tensor_operation: str, tensor_data: Any):
        """
        Fallback-Implementierung für Tensor-Operationen mit NumPy
        
        Args:
            tensor_operation: Gewünschte Tensor-Operation
            tensor_data: Eingabedaten für die Operation
            
        Returns:
            Ergebnis der Tensor-Operation mit NumPy
        """
        logger.warning(f"Verwende NumPy-Fallback für Tensor-Operation: {tensor_operation}")
        
        try:
            # Konvertiere Tensoren zu NumPy-Arrays, falls nötig
            numpy_data = []
            for tensor in tensor_data if isinstance(tensor_data, (list, tuple)) else [tensor_data]:
                if hasattr(tensor, 'detach') and hasattr(tensor, 'cpu') and hasattr(tensor, 'numpy'):
                    # PyTorch Tensor
                    numpy_data.append(tensor.detach().cpu().numpy())
                elif hasattr(tensor, 'to_numpy'):
                    # MLX Array
                    numpy_data.append(tensor.to_numpy())
                else:
                    # Vermutlich bereits NumPy oder kompatibel
                    numpy_data.append(np.array(tensor))
            
            # Führe die gewünschte Operation mit NumPy aus
            if tensor_operation == "matmul":
                result = np.matmul(numpy_data[0], numpy_data[1])
            elif tensor_operation == "svd":
                result = np.linalg.svd(numpy_data[0])
            elif tensor_operation == "attention":
                # Vereinfachte Attention-Implementierung
                q, k, v = numpy_data[:3]
                scores = np.matmul(q, k.T)
                scores = scores / np.sqrt(k.shape[-1])
                weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
                result = np.matmul(weights, v)
            else:
                # Generischer Fallback
                result = np.array(numpy_data[0])
                logger.warning(f"Unbekannte Tensor-Operation im Fallback: {tensor_operation}")
            
            return {
                "status": "success",
                "operation": tensor_operation,
                "backend": "numpy_fallback",
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Fehler im NumPy-Fallback: {e}")
            return {
                "status": "error",
                "operation": tensor_operation,
                "backend": "numpy_fallback",
                "error": str(e)
            }
    
    def integrate_with_m_lingua(self, natural_language_input: str) -> Dict[str, Any]:
        """
        Integriert mit M-LINGUA, um natürliche Sprache zu verarbeiten
        
        Args:
            natural_language_input: Natürlichsprachige Eingabe
            
        Returns:
            Verarbeitungsergebnis
        """
        if not HAS_DEPENDENCIES or not self.qlogik_manager:
            return {"status": "error", "message": "M-LINGUA-Integration nicht verfügbar"}
        
        logger.info(f"Verarbeite natürlichsprachige Eingabe: '{natural_language_input}'")
        
        try:
            # Analysiere die natürlichsprachige Eingabe mit M-LINGUA
            parsed_command = self.qlogik_manager.parse_natural_language(natural_language_input)
            
            # Extrahiere die mathematische Operation und die Parameter
            operation_type = parsed_command.get("operation_type")
            parameters = parsed_command.get("parameters", {})
            
            # Führe verschiedene Aktionen basierend auf dem Operationstyp aus
            if operation_type == "tensor_operation":
                # Für Tensor-Operationen verwenden wir die T-Mathematics-Integration
                return self.integrate_with_t_mathematics(
                    tensor_operation=parameters.get("operation"),
                    tensor_data=parameters.get("data"),
                    backend=parameters.get("backend")
                )
            
            elif operation_type == "reality_simulation":
                # Für Realitätssimulationen verwenden wir den RealityForker
                return self.simulate_reality_fork(
                    current_state=parameters.get("current_state", {}),
                    variations=parameters.get("variations", []),
                    steps=parameters.get("steps", 100)
                )
            
            elif operation_type == "pattern_analysis":
                # Für Musteranalysen verwenden wir den PatternDissonanceScanner
                return self.scan_pattern_dissonance(
                    pattern_id=parameters.get("pattern_id"),
                    current_data=parameters.get("data")
                )
            
            else:
                return {"status": "error", "message": f"Unbekannter Operationstyp: {operation_type}"}
            
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung der natürlichsprachigen Eingabe: {e}")
            return {"status": "error", "message": str(e)}
    
    def register_timeline(self, timeline: 'Timeline') -> None:
        """
        Registriert eine Zeitlinie bei der PRISM-Engine
        
        Args:
            timeline: Zu registrierende Zeitlinie
        """
        self.registered_timelines[timeline.id] = timeline
        logger.info(f"Zeitlinie {timeline.id} bei PRISM-Engine registriert")
    
    def get_registered_timeline_ids(self) -> List[str]:
        """
        Gibt die IDs aller registrierten Zeitlinien zurück
        
        Returns:
            Liste von Zeitlinien-IDs
        """
        return list(self.registered_timelines.keys())
        
    def get_registered_timeline(self, timeline_id: str) -> Optional['Timeline']:
        """
        Gibt eine registrierte Zeitlinie zurück
        
        Args:
            timeline_id: ID der Zeitlinie
            
        Returns:
            Zeitlinie oder None, falls nicht gefunden
        """
        return self.registered_timelines.get(timeline_id)
    
    def unregister_timeline(self, timeline_id: str) -> None:
        """
        Hebt die Registrierung einer Zeitlinie auf
        
        Args:
            timeline_id: ID der Zeitlinie
        """
        if timeline_id in self.registered_timelines:
            del self.registered_timelines[timeline_id]
            logger.info(f"Registrierung der Zeitlinie {timeline_id} aufgehoben")
            
    def calculate_timeline_probability(self, timeline_id: str) -> float:
        """
        Berechnet die Wahrscheinlichkeit einer Zeitlinie mit optimierter T-Mathematics Integration.
        
        Diese Methode verwendet das MLX-Framework auf Apple Silicon für hochperformante
        Berechnungen und hat Fallbacks für andere Hardware-Konfigurationen.
        
        Args:
            timeline_id: ID der Zeitlinie
            
        Returns:
            Wahrscheinlichkeit der Zeitlinie (0.0 bis 1.0)
        """
        # Hole die Zeitlinie aus den registrierten Zeitlinien
        timeline = self.get_registered_timeline(timeline_id)
        if not timeline:
            logger.warning(f"Zeitlinie {timeline_id} nicht gefunden")
            return 0.0
        
        # Wenn keine Knoten in der Zeitlinie vorhanden sind
        if not timeline.nodes or len(timeline.nodes) == 0:
            logger.debug(f"Zeitlinie {timeline_id} enthält keine Knoten, verwende Standardwert 0.5")
            return 0.5  # Standardwert für leere Zeitlinien
        
        # Performance-Messung starten
        start_time = time.time()
        
        # Optimierte Pfad für T-Mathematics mit MLX auf Apple Silicon
        if self.t_math_engine is not None and HAS_MLX and is_apple_silicon and self.use_mlx:
            try:
                # Nutze die T-Mathematics Engine für optimale Performance
                logger.debug(f"Verwende T-Mathematics Engine mit MLX für Zeitlinienwahrscheinlichkeit {timeline_id}")
                
                # Extrahiere alle Knotenwahrscheinlichkeiten als Vektor
                node_probs = [node.probability for node in timeline.nodes.values()]
                
                # Konvertiere zu MLX-Tensor für optimierte Berechnung
                node_probs_tensor = mx.array(node_probs)
                
                # Gewichte Knoten nach Wichtigkeit (neuere Knoten haben mehr Einfluss)
                weights = mx.array([1.0 + 0.1 * i for i in range(len(node_probs))])
                weights = weights / mx.sum(weights)  # Normalisierung
                
                # Berechne gewichteten Durchschnitt mit MLX (optimiert für Apple Silicon)
                combined_prob = mx.sum(node_probs_tensor * weights).item()
                
                # Berücksichtige Zeitlinientyp in der Berechnung
                type_factor = 1.0
                if hasattr(timeline, 'type') and timeline.type == TimelineType.THEORETICAL:
                    type_factor = 0.7  # Theoretische Zeitlinien sind weniger wahrscheinlich
                elif hasattr(timeline, 'type') and timeline.type == TimelineType.ALTERNATIVE:
                    type_factor = 0.85  # Alternative Zeitlinien sind etwas weniger wahrscheinlich
                
                # Wende die PRISM-Matrix an, um die Wahrscheinlichkeit zu verfeinern (mit MLX-Optimierung)
                if hasattr(self.matrix, 'apply_probability_transformation_mlx'):
                    refined_prob = self.matrix.apply_probability_transformation_mlx(combined_prob * type_factor)
                else:
                    refined_prob = self.matrix.apply_probability_transformation(combined_prob * type_factor)
                
                logger.debug(f"Zeitlinienwahrscheinlichkeit für {timeline_id} berechnet: {refined_prob:.6f} "
                           f"in {(time.time() - start_time)*1000:.2f}ms mit MLX-Optimierung")
                return max(0.0, min(1.0, refined_prob))
            except Exception as e:
                logger.warning(f"MLX-Optimierung für Zeitlinienwahrscheinlichkeit fehlgeschlagen: {e}, "
                              f"Fallback auf Standard-Implementierung")
        
        # Standard-Pfad als Fallback
        try:
            # Extrahiere alle Knotenwahrscheinlichkeiten
            node_probs = [node.probability for node in timeline.nodes.values()]
            
            # Gewichte Knoten nach Wichtigkeit (neuere Knoten haben mehr Einfluss)
            total_nodes = len(node_probs)
            weights = [(1.0 + 0.1 * i) for i in range(total_nodes)]
            weight_sum = sum(weights)
            normalized_weights = [w / weight_sum for w in weights]
            
            # Berechne gewichteten Durchschnitt
            combined_prob = sum(p * w for p, w in zip(node_probs, normalized_weights))
            
            # Wende die PRISM-Matrix an, um die Wahrscheinlichkeit zu verfeinern
            refined_prob = self.matrix.apply_probability_transformation(combined_prob)
            
            logger.debug(f"Zeitlinienwahrscheinlichkeit für {timeline_id} berechnet: {refined_prob:.6f} "
                       f"in {(time.time() - start_time)*1000:.2f}ms mit Standard-Methode")
            return max(0.0, min(1.0, refined_prob))
            
        except Exception as e:
            logger.error(f"Fehler bei der Berechnung der Zeitlinienwahrscheinlichkeit: {e}")
            return 0.5  # Standardwert im Fehlerfall
    
    def calculate_node_probability(self, timeline_id: str, node_id: str) -> float:
        """
        Berechnet die Wahrscheinlichkeit eines Knotens in einer Zeitlinie
        
        Args:
            timeline_id: ID der Zeitlinie
            node_id: ID des Knotens
            
        Returns:
            Wahrscheinlichkeit des Knotens (0.0 bis 1.0)
        """
        timeline = self.get_registered_timeline(timeline_id)
        if not timeline:
            logger.warning(f"Zeitlinie {timeline_id} nicht gefunden")
            return 0.0
            
        node = timeline.nodes.get(node_id)
        if not node:
            logger.warning(f"Knoten {node_id} in Zeitlinie {timeline_id} nicht gefunden")
            return 0.0
            
        # Basiswahrscheinlichkeit des Knotens
        base_prob = node.probability
        
        # Wende die PRISM-Matrix an, um die Wahrscheinlichkeit zu verfeinern
        refined_prob = self.matrix.apply_probability_transformation(base_prob)
        
        # Berücksichtige Verbindungen zu anderen Knoten
        connected_nodes = self._get_connected_nodes(timeline, node_id)
        if connected_nodes:
            connection_factor = sum(n.probability for n in connected_nodes) / len(connected_nodes)
            refined_prob = (refined_prob + connection_factor) / 2
        
        return max(0.0, min(1.0, refined_prob))
    
    def _get_connected_nodes(self, timeline: 'Timeline', node_id: str) -> List['TimeNode']:
        """
        Findet alle mit einem Knoten verbundenen Knoten
        
        Args:
            timeline: Zeitlinie
            node_id: ID des Knotens
            
        Returns:
            Liste von verbundenen Knoten
        """
        connected_nodes = []
        
        # Suche nach Verbindungen, bei denen der Knoten als Quelle oder Ziel fungiert
        for connection in timeline.connections:
            if connection.get('source_id') == node_id and connection.get('target_id') in timeline.nodes:
                connected_nodes.append(timeline.nodes[connection.get('target_id')])
            elif connection.get('target_id') == node_id and connection.get('source_id') in timeline.nodes:
                connected_nodes.append(timeline.nodes[connection.get('source_id')])
                
        return connected_nodes
    
    def analyze_timeline_stability(self, timeline_id: str) -> Dict[str, float]:
        """
        Führt eine Stabilitätsanalyse für eine Zeitlinie durch
        
        Args:
            timeline_id: ID der Zeitlinie
            
        Returns:
            Dictionary mit Stabilitätsmetriken
        """
        timeline = self.get_registered_timeline(timeline_id)
        if not timeline:
            logger.warning(f"Zeitlinie {timeline_id} nicht gefunden")
            return {
                "stability_index": 0.0,
                "paradox_risk": 1.0,
                "entropy": 1.0
            }
            
        # Berechne Stabilitätsindex basierend auf Knotenwahrscheinlichkeiten
        node_probs = [node.probability for node in timeline.nodes.values()]
        if not node_probs:
            return {
                "stability_index": 0.5,
                "paradox_risk": 0.5,
                "entropy": 0.5
            }
            
        # Stabilitätsindex: Höhere Werte bedeuten stabilere Zeitlinien
        avg_prob = sum(node_probs) / len(node_probs)
        stability_index = avg_prob
        
        # Paradoxrisiko: Niedrigere Werte bedeuten geringeres Risiko
        # Berechnet als inverse der Stabilität mit Rauschen
        noise = np.random.normal(0, 0.1)
        paradox_risk = max(0.0, min(1.0, 1.0 - stability_index + noise))
        
        # Entropie: Maß für die Unordnung in der Zeitlinie
        # Berechnet aus der Varianz der Knotenwahrscheinlichkeiten
        if len(node_probs) > 1:
            variance = np.var(node_probs)
            entropy = max(0.0, min(1.0, variance * 4))  # Skaliere Varianz auf [0,1]
        else:
            entropy = 0.5
            
        return {
            "stability_index": stability_index,
            "paradox_risk": paradox_risk,
            "entropy": entropy
        }
    
    def fork_timeline(self, source_timeline_id: str, fork_node_id: str, variation_factor: float = 0.5) -> Dict[str, Any]:
        """
        Erstellt eine Verzweigung einer Zeitlinie an einem bestimmten Knoten
        
        Args:
            source_timeline_id: ID der Quellzeitlinie
            fork_node_id: ID des Knotens, an dem verzweigt werden soll
            variation_factor: Faktor für die Variation der neuen Zeitlinie (0.0 bis 1.0)
            
        Returns:
            Dictionary mit Informationen zur verzweigten Zeitlinie
        """
        source_timeline = self.get_registered_timeline(source_timeline_id)
        if not source_timeline:
            logger.warning(f"Quellzeitlinie {source_timeline_id} nicht gefunden")
            return {"error": f"Zeitlinie {source_timeline_id} nicht gefunden"}
            
        if fork_node_id not in source_timeline.nodes:
            logger.warning(f"Verzweigungsknoten {fork_node_id} in Zeitlinie {source_timeline_id} nicht gefunden")
            return {"error": f"Knoten {fork_node_id} nicht gefunden"}
            
        # Erstelle eine Kopie der Zeitlinie
        from copy import deepcopy
        forked_timeline = deepcopy(source_timeline)
        forked_timeline.id = str(uuid.uuid4())
        forked_timeline.name = f"{source_timeline.name} (Verzweigung)"
        
        # Variiere die Wahrscheinlichkeiten der Knoten nach dem Verzweigungsknoten
        fork_timestamp = source_timeline.nodes[fork_node_id].timestamp
        for node_id, node in forked_timeline.nodes.items():
            if node.timestamp >= fork_timestamp:
                # Variiere die Wahrscheinlichkeit basierend auf dem Variationsfaktor
                variation = (np.random.random() - 0.5) * variation_factor
                node.probability = max(0.0, min(1.0, node.probability + variation))
                
        # Registriere die verzweigte Zeitlinie
        self.register_timeline(forked_timeline)
        
        # Berechne die Wahrscheinlichkeit der verzweigten Zeitlinie
        probability = self.calculate_timeline_probability(forked_timeline.id)
        
        return {
            "forked_timeline_id": forked_timeline.id,
            "probability": probability,
            "variation_factor": variation_factor
        }
        
    def merge_timelines(self, source_timeline_id: str, target_timeline_id: str, merge_point_source: str, merge_point_target: str) -> Dict[str, Any]:
        """
        Führt zwei Zeitlinien an bestimmten Knoten zusammen
        
        Args:
            source_timeline_id: ID der Quellzeitlinie
            target_timeline_id: ID der Zielzeitlinie
            merge_point_source: ID des Knotens in der Quellzeitlinie
            merge_point_target: ID des Knotens in der Zielzeitlinie
            
        Returns:
            Dictionary mit Informationen zur zusammengeführten Zeitlinie
        """
        source_timeline = self.get_registered_timeline(source_timeline_id)
        target_timeline = self.get_registered_timeline(target_timeline_id)
        
        if not source_timeline or not target_timeline:
            error_msg = f"Zeitlinien nicht gefunden: {source_timeline_id if not source_timeline else target_timeline_id}"
            logger.warning(error_msg)
            return {"error": error_msg}
            
        if merge_point_source not in source_timeline.nodes:
            error_msg = f"Zusammenführungspunkt {merge_point_source} in Quellzeitlinie nicht gefunden"
            logger.warning(error_msg)
            return {"error": error_msg}
            
        if merge_point_target not in target_timeline.nodes:
            error_msg = f"Zusammenführungspunkt {merge_point_target} in Zielzeitlinie nicht gefunden"
            logger.warning(error_msg)
            return {"error": error_msg}
            
        # Erstelle eine Kopie der Zielzeitlinie
        from copy import deepcopy
        merged_timeline = deepcopy(target_timeline)
        
        # Finde alle Knoten nach dem Zusammenführungspunkt in der Quellzeitlinie
        source_merge_timestamp = source_timeline.nodes[merge_point_source].timestamp
        nodes_to_merge = {}
        for node_id, node in source_timeline.nodes.items():
            if node.timestamp > source_merge_timestamp and node_id != merge_point_source:
                # Erstelle eine neue ID für den Knoten, um Konflikte zu vermeiden
                new_node_id = f"merged_{node_id}"
                node_copy = deepcopy(node)
                node_copy.id = new_node_id
                nodes_to_merge[new_node_id] = node_copy
                
        # Füge die Knoten zur zusammengeführten Zeitlinie hinzu
        for node_id, node in nodes_to_merge.items():
            merged_timeline.nodes[node_id] = node
            
        # Verbinde den Zusammenführungspunkt mit den neuen Knoten
        for node_id in nodes_to_merge.keys():
            merged_timeline.connect_nodes(
                source_id=merge_point_target,
                target_id=node_id,
                metadata={"merged": True, "source_timeline": source_timeline_id}
            )
            
        # Registriere die zusammengeführte Zeitlinie
        self.register_timeline(merged_timeline)
        
        # Berechne die Wahrscheinlichkeit der zusammengeführten Zeitlinie
        probability = self.calculate_timeline_probability(merged_timeline.id)
        
        # Berechne die Stabilität der zusammengeführten Zeitlinie
        stability_result = self.analyze_timeline_stability(merged_timeline.id)
        stability = stability_result.get("stability_score", 0.5)
        
        return {
            "merged_timeline_id": merged_timeline.id,
            "probability": probability,
            "stability": stability,
            "source_nodes_count": len(nodes_to_merge)
        }
    
    def _initialize_resolution_strategies(self, timeline: Timeline) -> Dict[str, Any]:
        """
        Initialisiert und konfiguriert die Auflösungsstrategien für Paradoxa
        
        Args:
            timeline: Die zu analysierende Zeitlinie
            
        Returns:
            Dictionary mit Strategien und Konfigurationen
        """
        strategies = {}
        
        # Verwende erweiterte Paradoxauflösung, wenn verfügbar
        if HAS_PARADOX_RESOLUTION:
            try:
                # Hole Paradox Resolution Manager
                manager = get_paradox_resolution_manager()
                
                # Integriere Timeline-spezifische Konfiguration
                config = {
                    "timeline_type": timeline.type.value,
                    "node_count": len(timeline.nodes),
                    "timeline_weight": timeline.weight,
                    "stability_threshold": 0.65 if timeline.type == TimelineType.PRIMARY else 0.45
                }
                
                # Erweiterte Konfiguration, wenn Q-LOGIK verfügbar
                if hasattr(self, 'qlogik_manager') and self.qlogik_manager:
                    try:
                        # Verwende Q-LOGIK für verbesserte Strategien
                        qlogik_factors = self.qlogik_manager.get_decision_factors({
                            "context": "paradox_resolution",
                            "timeline_id": timeline.id,
                            "stability": self._calculate_timeline_stability(timeline)
                        })
                        
                        if qlogik_factors:
                            config.update({
                                "qlogik_factors": qlogik_factors,
                                "use_quantum_strategies": qlogik_factors.get("quantum_applicability", 0.0) > 0.6
                            })
                    except Exception as e:
                        logger.warning(f"Q-LOGIK Integration für Strategien nicht verfügbar: {e}")
                
                # Gib Manager und Konfiguration zurück
                return {
                    "manager": manager,
                    "config": config
                }
            except Exception as e:
                logger.error(f"Fehler bei Initialisierung der Paradoxauflösungsstrategien: {e}")
        
        # Fallback: Verwende einfache Standardstrategien, wenn die erweiterte Auflösung nicht verfügbar ist
        strategies = {
            "direct_time_loop": {"score": 0.8, "method": "node_isolation"},
            "cyclic_time_loop": {"score": 0.7, "method": "probability_reduction"},
            "causal_paradox": {"score": 0.9, "method": "temporal_adjustment"},
            "quantum_paradox": {"score": 0.6, "method": "superposition"},
            "entropy_inversion": {"score": 0.5, "method": "entropy_injection"}
        }
        
        return {"strategies": strategies, "config": {"use_basic_resolution": True}}
    
    def _calculate_timeline_stability(self, timeline: Timeline) -> float:
        """
        Berechnet den Stabilitätsindex einer Zeitlinie basierend auf verschiedenen Faktoren
        
        Args:
            timeline: Die zu analysierende Zeitlinie
            
        Returns:
            Stabilitätsindex als Float zwischen 0.0 (instabil) und 1.0 (stabil)
        """
        # Basis-Stabilitätswert basierend auf Timeline-Typ
        base_stability = 0.8 if timeline.type == TimelineType.PRIMARY else 0.6
        
        # Wenn keine Knoten vorhanden sind, verwende Basis-Stabilität
        if not timeline.nodes:
            return base_stability
            
        # Faktoren zur Stabilitätsberechnung
        factors = [
            # 1. Knotenwahrscheinlichkeiten (höhere Wahrscheinlichkeiten = höhere Stabilität)
            self._calculate_node_probability_factor(timeline),
            
            # 2. Temporale Kohärenz (konsistente Zeitrichtung)
            self._calculate_temporal_coherence(timeline),
            
            # 3. Strukturelle Integrität (Verbindungen zwischen Knoten)
            self._calculate_structural_integrity(timeline),
            
            # 4. Entropiewachstum (naturgemäß steigend)
            self._calculate_entropy_growth(timeline)
        ]
        
        # Gewichteter Durchschnitt der Faktoren
        weights = [0.35, 0.25, 0.25, 0.15]  # Summe = 1.0
        weighted_stability = sum(f * w for f, w in zip(factors, weights))
        
        # Modifiziere Basis-Stabilität basierend auf Faktoren
        final_stability = base_stability * 0.4 + weighted_stability * 0.6
        
        # Begrenze auf Bereich [0.0, 1.0]
        return max(0.0, min(1.0, final_stability))
        
    def _calculate_node_probability_factor(self, timeline: Timeline) -> float:
        """
        Berechnet Stabilitätsfaktor basierend auf Knotenwahrscheinlichkeiten
        """
        if not timeline.nodes:
            return 0.5
            
        # Durchschnittliche Wahrscheinlichkeit aller Knoten
        probabilities = [node.probability for node in timeline.nodes.values() if hasattr(node, 'probability')]
        if not probabilities:
            return 0.5
            
        avg_probability = sum(probabilities) / len(probabilities)
        
        # Varianz der Wahrscheinlichkeiten (höhere Varianz = niedrigere Stabilität)
        if len(probabilities) > 1:
            variance = sum((p - avg_probability) ** 2 for p in probabilities) / len(probabilities)
            variance_penalty = min(0.3, variance * 2)  # Maximale Abzug von 0.3
        else:
            variance_penalty = 0.0
            
        # Stabilität basierend auf durchschnittlicher Wahrscheinlichkeit, reduziert durch Varianz
        return avg_probability - variance_penalty
    
    def _calculate_temporal_coherence(self, timeline: Timeline) -> float:
        """
        Berechnet temporale Kohärenz basierend auf konsistenter Zeitrichtung
        """
        if not timeline.nodes or len(timeline.nodes) < 2:
            return 0.7  # Standardwert für Zeitlinien mit zu wenigen Knoten
            
        # Prüfe auf zeitliche Inkonsistenzen in verbundenen Knoten
        inconsistencies = 0
        total_connections = 0
        
        # Erstelle temporale Adjazenzmatrix
        temporal_connections = {}
        for node_id, node in timeline.nodes.items():
            for connection in node.connections:
                if connection.target_id in timeline.nodes:
                    total_connections += 1
                    source_time = getattr(node, 'timestamp', 0)
                    target_time = getattr(timeline.nodes[connection.target_id], 'timestamp', 0)
                    
                    # Inkonsistenz: Zielknoten liegt zeitlich vor Quellknoten
                    if target_time < source_time and not getattr(connection, 'is_quantum', False):
                        inconsistencies += 1
        
        if total_connections == 0:
            return 0.7  # Standardwert
            
        # Berechne Kohärenzgrad
        coherence = 1.0 - (inconsistencies / total_connections)
        return coherence
    
    def _calculate_structural_integrity(self, timeline: Timeline) -> float:
        """
        Berechnet strukturelle Integrität basierend auf Knotenverbindungen
        """
        if not timeline.nodes:
            return 0.5
            
        # Anzahl Knoten ohne Verbindungen
        isolated_nodes = sum(1 for node in timeline.nodes.values() if not node.connections)
        isolated_ratio = isolated_nodes / len(timeline.nodes) if len(timeline.nodes) > 0 else 0
        
        # Strukturelle Integrität reduziert sich mit isolierten Knoten
        isolation_penalty = isolated_ratio * 0.5  # Maximal 0.5 Abzug
        
        # Verbindungsdichte (höhere Dichte = höhere Integrität, bis zu einem Punkt)
        total_connections = sum(len(node.connections) for node in timeline.nodes.values())
        avg_connections = total_connections / len(timeline.nodes) if len(timeline.nodes) > 0 else 0
        connection_factor = min(1.0, avg_connections / 3)  # Optimale Dichte bei 3 Verbindungen/Knoten
        
        return (0.8 * connection_factor) - isolation_penalty
    
    def _calculate_entropy_growth(self, timeline: Timeline) -> float:
        """
        Berechnet Stabilitätsfaktor basierend auf Entropiewachstum
        """
        if not timeline.nodes or len(timeline.nodes) < 2:
            return 0.6
            
        # Sammle Entropiewerte, falls vorhanden
        entropy_values = []
        for node in timeline.nodes.values():
            if hasattr(node, 'entropy') and node.entropy is not None:
                entropy_values.append((getattr(node, 'timestamp', 0), node.entropy))
        
        if len(entropy_values) < 2:
            return 0.6  # Standardwert
            
        # Sortiere nach Zeitstempel
        entropy_values.sort(key=lambda x: x[0])
        
        # Zähle Entropie-Invertierungen (Abnahmen)
        inversions = 0
        for i in range(1, len(entropy_values)):
            if entropy_values[i][1] < entropy_values[i-1][1]:
                inversions += 1
                
        inversion_ratio = inversions / (len(entropy_values) - 1)
        
        # Stabilität sinkt mit Entropie-Invertierungen
        return 1.0 - min(0.8, inversion_ratio * 2)  # Maximal 0.8 Abzug
        
    def detect_paradoxes(self, timeline_id: str, detect_only: bool = False) -> Dict[str, Any]:
        """
        Erweiterte Implementierung: Erkennt und löst komplexe Paradoxa in Zeitlinien
        
        Diese hochoptimierte Implementierung erkennt und klassifiziert Paradoxa hierarchisch
        in primäre, sekundäre und tertiäre Typen mit dynamischer Schweregrad-Bewertung:
        
        Primäre Paradoxtypen:
        - Temporale Paradoxa: Zeitschleifen, kausale Widersprüche, Zeitdisparitäten
        - Quantenmechanische Paradoxa: Superpositionen, Verschränkungskonflikte, Dekoherenzen
        - Informationstheoretische Paradoxa: Entropie-Inversionen, Informationsverluste
        
        Sekundäre Klassifikation:
        - Strukturell: In der Topologie der Zeitlinie verankert
        - Probabilistisch: In Wahrscheinlichkeitsverteilungen erkennbar
        - Semantisch: In der inhaltlichen Bedeutung und Beziehung von Knoten
        
        Tertiäre Klassifikation:
        - Kritisch: Erfordert sofortige Auflösung, destabilisiert die Zeitlinie
        - Signifikant: Beeinträchtigt die Funktionalität, aber nicht kritisch
        - Marginal: Geringe Auswirkung, kann oft ignoriert werden
        
        Args:
            timeline_id: ID der zu analysierenden Zeitlinie
            detect_only: Wenn True, werden Paradoxa nur erkannt, aber nicht automatisch aufgelöst
            
        Returns:
            Dictionary mit detaillierten Informationen zu erkannten Paradoxa, Klassifikationen,
            Auflösungsstrategien und ausgeführten Auflösungsmaßnahmen
        """
        # Initialisiere Telemetrie und Diagnosesystem mit expliziten Default-Werten
        paradox_detection_stats = {
            "detection_start_time": time.time(),
            "timeline_id": timeline_id,
            "detection_methods_used": [],
            "resolution_methods_applied": [],
            "performance_metrics": {},
            "warnings": [],
            "errors": []
        }
        
        # Performance-Messung starten
        start_time = time.time()
        method_start_time = start_time
        
        # Paradox-Liste initialisieren
        paradoxes = []
        
        # 1. Lade Zeitlinie mit spezifischer Fehlerbehandlung für verschiedene Exception-Typen
        try:
            timeline = self.get_registered_timeline(timeline_id)
            if not timeline:
                logger.warning(f"Zeitlinie {timeline_id} nicht gefunden")
                paradox_detection_stats["errors"].append(f"Zeitlinie {timeline_id} nicht gefunden")
                return {
                    "paradoxes": [], 
                    "paradox_count": 0,
                    "success": False,
                    "stats": paradox_detection_stats
                }
                
            # Leere Zeitlinie schnell behandeln
            if not timeline.nodes or len(timeline.nodes) == 0:
                logger.debug(f"Zeitlinie {timeline_id} enthält keine Knoten, keine Paradoxa möglich")
                paradox_detection_stats["result"] = "empty_timeline"
                paradox_detection_stats["performance_metrics"]["total_time_ms"] = (time.time() - start_time) * 1000
                return {
                    "paradoxes": [], 
                    "paradox_count": 0, 
                    "timeline_id": timeline_id,
                    "success": True,
                    "stats": paradox_detection_stats
                }
        except KeyError as e:
            # Spezifischere Fehlerbehandlung für Schlüsselfehler
            error_msg = f"Schlüsselfehler beim Laden der Zeitlinie {timeline_id}: {str(e)}"
            logger.error(error_msg)
            paradox_detection_stats["errors"].append(error_msg)
            return {
                "paradoxes": [], 
                "paradox_count": 0,
                "success": False,
                "stats": paradox_detection_stats
            }
        except (TypeError, ValueError) as e:
            # Spezifischere Fehlerbehandlung für Typfehler oder ungültige Werte
            error_msg = f"Fehler mit Datentypen oder Werten in Zeitlinie {timeline_id}: {str(e)}"
            logger.error(error_msg)
            paradox_detection_stats["errors"].append(error_msg)
            return {
                "paradoxes": [], 
                "paradox_count": 0,
                "success": False,
                "stats": paradox_detection_stats
            }
        except Exception as e:
            # Allgemeine Fehlerbehandlung mit mehr Kontextinformationen
            error_msg = f"Unerwarteter Fehler beim Laden der Zeitlinie {timeline_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)  # Log traceback for unexpected errors
            paradox_detection_stats["errors"].append(error_msg)
            return {
                "paradoxes": [], 
                "paradox_count": 0,
                "success": False,
                "stats": paradox_detection_stats
            }
        
        # 2. Bestimme optimalen Verarbeitungspfad (MLX oder Standard)
        use_optimized_path = False
        try:
            # Prüfe, ob MLX verfügbar und aktiviert ist
            if is_apple_silicon and self.use_mlx and 'mlx' in sys.modules:
                use_optimized_path = True
                paradox_detection_stats["used_mlx"] = True
                paradox_detection_stats["detection_methods_used"].append("mlx_optimized")
                logger.debug(f"Verwende MLX-optimierten Pfad für Paradoxerkennung in Zeitlinie {timeline_id}")
            else:
                logger.debug(f"Verwende Standard-Pfad für Paradoxerkennung in Zeitlinie {timeline_id}")
                paradox_detection_stats["detection_methods_used"].append("standard")
        except Exception as e:
            # Falls die MLX-Prüfung fehlschlägt, Standard-Pfad verwenden
            use_optimized_path = False
            error_msg = f"Fehler bei der Bestimmung des Optimierungspfads: {str(e)}"
            logger.warning(error_msg)
            paradox_detection_stats["warnings"].append(error_msg)
            paradox_detection_stats["detection_methods_used"].append("standard_fallback")
        
        # 3. Erkenne direkte Zeitschleifen (Zeitknoten, die sich selbst referenzieren)
        try:
            # Mit optimierter Zählmethode
            loops_found = 0
            for node_id, node in timeline.nodes.items():
                # Prüfe auf direkte Selbstreferenzen in Verbindungen
                for connection in timeline.connections:
                    if connection.get('source_id') == node_id and connection.get('target_id') == node_id:
                        loops_found += 1
                        paradoxes.append({
                            "type": "direct_time_loop",
                            "source_id": node_id,
                            "target_id": node_id,
                            "severity": 0.9,  # Direkte Schleifen sind meist kritisch
                            "resolution_difficulty": 0.7,
                            "description": "Direkte Zeitschleife: Knoten referenziert sich selbst",
                            "resolution_options": [
                                "Verbindung auflösen",
                                "Quantenverschränkung einführen"
                            ]
                        })
            
            # Erfasse Performance-Metrik
            paradox_detection_stats["direct_loops_found"] = loops_found
            paradox_detection_stats["performance_metrics"]["direct_loop_detection_time_ms"] = (time.time() - method_start_time) * 1000
        except Exception as e:
            error_msg = f"Fehler bei der Erkennung direkter Zeitschleifen: {str(e)}"
            logger.warning(error_msg)
            paradox_detection_stats["warnings"].append(error_msg)
        
        # Aktualisiere Startzeit für nächste Phase
        method_start_time = time.time()
        
        # 4. Erkenne zyklische Zeitschleifen mit Adjazenzmatrix-Analyse
        # Dies ist effizienter für die Erkennung komplexerer Zyklen
        try:
            if use_optimized_path:
                # Erstelle Adjazenzmatrix für Graph-Analyse mit MLX
                node_ids = list(timeline.nodes.keys())
                node_id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
                n = len(node_ids)
                
                # Initialisiere Adjazenzmatrix
                adjacency = mx.zeros((n, n))
                
                # Fülle Adjazenzmatrix
                for connection in timeline.connections:
                    source_id = connection.get('source_id')
                    target_id = connection.get('target_id')
                    if source_id in node_id_to_idx and target_id in node_id_to_idx:
                        i = node_id_to_idx[source_id]
                        j = node_id_to_idx[target_id]
                        adjacency = adjacency.at[i, j].set(1)
                
                # Berechne 3-Pfade und finde Zyklen (A^3 [i,i] > 0 bedeutet Zyklus der Länge 3 von i zu sich selbst)
                paths3 = mx.matmul(mx.matmul(adjacency, adjacency), adjacency)
                
                # Finde alle Knoten mit Zyklen
                cycles_found = 0
                for i in range(n):
                    if paths3[i, i] > 0 and node_ids[i] not in [p["source_id"] for p in paradoxes if p["type"] == "direct_time_loop"]:
                        cycles_found += 1
                        # Optimierte Erkennung von zyklischen Pfaden
                        paradoxes.append({
                            "type": "cyclic_time_loop",
                            "node_id": node_ids[i],
                            "cycle_length": 3,  # Wir haben explizit nach 3-Zyklen gesucht
                            "severity": 0.75,
                            "resolution_difficulty": 0.8,
                            "description": "Zyklische Zeitschleife: Knoten ist Teil eines zyklischen Pfades",
                            "resolution_options": [
                                "Parallele Zeitlinienextraktion",
                                "Quantensuperposition der Zyklusknoten"
                            ]
                        })
                
                # Erfasse Performance-Metrik
                paradox_detection_stats["cycles_found"] = cycles_found
                logger.debug(f"MLX-optimierte Zykluserkennung durchgeführt in {(time.time() - method_start_time)*1000:.2f}ms")
            else:
                # Standard-Implementierung für Zykluserkennung ohne MLX
                # Einfache Tiefensuche (DFS) zum Auffinden von Zyklen
                cycles_found = 0
                visited = {}
                rec_stack = {}
                
                def is_cyclic_util(node_id, visited, rec_stack):
                    visited[node_id] = True
                    rec_stack[node_id] = True
                    
                    # Für jeden verbundenen Knoten
                    for connection in timeline.connections:
                        if connection.get('source_id') == node_id:
                            target = connection.get('target_id')
                            # Wenn nicht besucht, rekursiv prüfen
                            if target not in visited:
                                if is_cyclic_util(target, visited, rec_stack):
                                    return target  # Rückgabe des Knotens, der einen Zyklus bildet
                            # Wenn im aktuellen Rekursionsstapel, Zyklus gefunden
                            elif rec_stack.get(target, False):
                                return target
                    
                    # Knoten aus Rekursionsstapel entfernen
                    rec_stack[node_id] = False
                    return None
                
                # Prüfe Zyklen für alle Knoten
                for node_id in timeline.nodes.keys():
                    if node_id not in visited:
                        cyclic_node = is_cyclic_util(node_id, visited, rec_stack)
                        if cyclic_node and cyclic_node not in [p["node_id"] for p in paradoxes if p["type"] == "cyclic_time_loop"]:
                            cycles_found += 1
                            paradoxes.append({
                                "type": "cyclic_time_loop",
                                "node_id": cyclic_node,
                                "severity": 0.7,
                                "resolution_difficulty": 0.7,
                                "description": "Zyklische Zeitschleife: Knoten ist Teil eines zyklischen Pfades",
                                "resolution_options": [
                                    "Parallele Zeitlinienextraktion",
                                    "Quantensuperposition der Zyklusknoten"
                                ]
                            })
                
                # Erfasse Performance-Metrik
                paradox_detection_stats["cycles_found"] = cycles_found
        except Exception as e:
            error_msg = f"Fehler bei der Zykluserkennung: {str(e)}"
            logger.warning(error_msg)
            paradox_detection_stats["warnings"].append(error_msg)
            
        # Aktualisiere Startzeit für nächste Phase und erfasse Performance-Metrik
        paradox_detection_stats["performance_metrics"]["cycle_detection_time_ms"] = (time.time() - method_start_time) * 1000
        method_start_time = time.time()
                "stats": paradox_detection_stats
            }
        except (TypeError, ValueError) as e:
            # Spezifischere Fehlerbehandlung für Typfehler oder ungültige Werte
            error_msg = f"Fehler mit Datentypen oder Werten: {str(e)}"
            logger.error(error_msg)
            paradox_detection_stats["error"] = error_msg
            return {
                "paradoxes": [], 
                "paradox_count": 0,
                "success": False,
                "stats": paradox_detection_stats
            }
        except Exception as e:
            # Allgemeine Fehlerbehandlung mit mehr Kontextinformationen
            error_msg = f"Allgemeiner Fehler beim Laden der Zeitlinie: {str(e)}"
            logger.error(error_msg, exc_info=True)  # Log traceback
            paradox_detection_stats["error"] = error_msg
            return {
                "paradoxes": [], 
                "paradox_count": 0,
                "success": False,
                "stats": paradox_detection_stats
            }
            
        # 2. Initialisiere Paradox-Tracking-Strukturen mit expliziten Default-Werten
        detected_paradoxes = []
        paradox_metrics = {
            "by_type": {},
            "by_severity": {
                "critical": 0,
                "significant": 0,
                "marginal": 0
            },
            "by_classification": {
                "temporal": 0,
                "quantum": 0,
                "informational": 0,
                "structural": 0,
                "probabilistic": 0,
                "semantic": 0
            }
        }
        
        # 3. Initialisiere Auflösungsstrategien und tracke Ausführungszeit
        try:
            method_start_time = time.time()
            resolution_strategies = self._initialize_resolution_strategies(timeline)
            paradox_detection_stats["performance_metrics"]["resolution_strategies_time_ms"] = (time.time() - method_start_time) * 1000
        except Exception as e:
            # Robuste Fehlerbehandlung für Strategieinitialisierung
            logger.error(f"Fehler bei Initialisierung der Auflösungsstrategien: {e}")
            paradox_detection_stats["error"] = f"Fehler bei Strategieinitialisierung: {str(e)}"
            resolution_strategies = {"strategies": {}, "config": {"use_basic_resolution": True}}
        
        # 4. Berechne Timeline-Stabilitätsindex für Paradox-Auswirkungsbewertung
        method_start_time = time.time()
        try:
            stability_index = self._calculate_timeline_stability(timeline)
            paradox_detection_stats["timeline_stability"] = stability_index
        except Exception as e:
            logger.warning(f"Fehler bei Stabilitätsberechnung: {e}")
            stability_index = 0.5  # Standardwert bei Fehler
            paradox_detection_stats["warning"] = f"Stabilitätsberechnung fehlgeschlagen: {str(e)}"
            paradox_detection_stats["timeline_stability"] = stability_index
            
        paradox_detection_stats["performance_metrics"]["stability_calculation_time_ms"] = (time.time() - method_start_time) * 1000
            "quantum": 0,
            "informational": 0,
            "structural": 0,
            "probabilistic": 0,
            "semantic": 0
        }
    }
        
    # Initialisiere Auflösungsstrategien
    resolution_strategies = self._initialize_resolution_strategies(timeline)
        if use_optimized_path:
            try:
                # Erstelle Adjazenzmatrix für Graph-Analyse mit MLX
                node_ids = list(timeline.nodes.keys())
                node_id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
                n = len(node_ids)
                
                # Initialisiere Adjazenzmatrix
                adjacency = mx.zeros((n, n))
                
                # Fülle Adjazenzmatrix
                for connection in timeline.connections:
                    source_id = connection.get('source_id')
                    target_id = connection.get('target_id')
                    if source_id in node_id_to_idx and target_id in node_id_to_idx:
                        i = node_id_to_idx[source_id]
                        j = node_id_to_idx[target_id]
                        adjacency = adjacency.at[i, j].set(1)
                
                # Berechne 3-Pfade und finde Zyklen (A^3 [i,i] > 0 bedeutet Zyklus der Länge 3 von i zu sich selbst)
                paths3 = mx.matmul(mx.matmul(adjacency, adjacency), adjacency)
                
                # Finde alle Knoten mit Zyklen
                for i in range(n):
                    if paths3[i, i] > 0 and node_ids[i] not in [p["source_id"] for p in paradoxes if p["type"] == "direct_time_loop"]:
                        # Optimierte Erkennung von zyklischen Pfaden
                        paradoxes.append({
                            "type": "cyclic_time_loop",
                            "node_id": node_ids[i],
                            "cycle_length": 3,  # Wir haben explizit nach 3-Zyklen gesucht
                            "severity": 0.75,
                            "resolution_difficulty": 0.8,
                            "description": "Zyklische Zeitschleife: Knoten ist Teil eines zyklischen Pfades",
                            "resolution_options": [
                                "Parallele Zeitlinienextraktion",
                                "Quantensuperposition der Zyklusknoten"
                            ]
                        })
                
                logger.debug(f"MLX-optimierte Zykluserkennung durchgeführt in {(time.time() - start_time)*1000:.2f}ms")
            except Exception as e:
                logger.warning(f"MLX-Zykluserkennung fehlgeschlagen: {e}, Fallback auf Standard-Implementierung")
                
        # --- 2. KAUSALE PARADOXA ---
        # Optimierte Erkennung kausaler Paradoxa
        if use_optimized_path:
            try:
                # Extrahiere Zeitstempel und IDs für optimierte Verarbeitung
                node_data = [(node_id, node.timestamp, node.probability) for node_id, node in timeline.nodes.items()]
                
                # Konvertiere zu MLX-Tensoren für optimierte Verarbeitung
                node_ids = [d[0] for d in node_data]
                timestamps = mx.array([d[1] for d in node_data])
                probabilities = mx.array([d[2] for d in node_data])
                
                # Für jede Verbindung prüfe kausale Paradoxa
                for connection in timeline.connections:
                    source_id = connection.get('source_id')
                    target_id = connection.get('target_id')
                    metadata = connection.get('metadata', {})
                    
                    # Prüfe, ob die Verbindung bereits als Paradox markiert ist
                    if metadata.get('is_paradox', False):
                        paradoxes.append({
                            "type": "marked_paradox",
                            "source_id": source_id,
                            "target_id": target_id,
                            "severity": 0.95,
                            "resolution_difficulty": 0.9,
                            "description": "Explizit markiertes Paradoxon",
                            "resolution_options": metadata.get('resolution_options', [])
                        })
                        continue
                    
                    # Prüfe kausale Paradoxa nur wenn beide Knoten existieren
                    if source_id in node_ids and target_id in node_ids:
                        source_idx = node_ids.index(source_id)
                        target_idx = node_ids.index(target_id)
                        
                        source_timestamp = timestamps[source_idx].item()
                        target_timestamp = timestamps[target_idx].item()
                        
                        if source_timestamp > target_timestamp:
                            time_diff = source_timestamp - target_timestamp
                            # Exponentiell wachsende Schwere mit der Zeitdifferenz
                            severity = 1.0 - mx.exp(-time_diff / 86400).item()  # Skaliert exponentiell
                            
                            # Berechne Auflösungsschwierigkeit basierend auf Zeitdifferenz und Wahrscheinlichkeiten
                            source_prob = probabilities[source_idx].item()
                            target_prob = probabilities[target_idx].item()
                            resolution_difficulty = 0.5 + (severity * 0.3) + (source_prob * target_prob * 0.2)
                            
                            paradoxes.append({
                                "type": "causal_paradox",
                                "source_id": source_id,
                                "target_id": target_id,
                                "time_difference": time_diff,
                                "severity": severity,
                                "resolution_difficulty": resolution_difficulty,
                                "description": "Kausales Paradoxon: Späterer Knoten beeinflusst früheren Knoten",
                                "resolution_options": [
                                    "Temporale Rekursion mit Quantenkorrektur",
                                    "Knotenaufspaltung mit Wahrscheinlichkeitsverteilung",
                                    "Kausale Entkopplung durch Zwischendimension"
                                ]
                            })
            except Exception as e:
                logger.warning(f"MLX-Optimierung für kausale Paradoxerkennung fehlgeschlagen: {e}, Fallback")
        
        # Standardpfad für kausale Paradoxa, wenn MLX nicht verwendet wird oder fehlgeschlagen ist
        if not use_optimized_path or len([p for p in paradoxes if p["type"] == "causal_paradox"]) == 0:
            for connection in timeline.connections:
                source_id = connection.get('source_id')
                target_id = connection.get('target_id')
                metadata = connection.get('metadata', {})
                
                # Prüfe, ob die Verbindung bereits als Paradox markiert ist
                if metadata.get('is_paradox', False) and not any(p["type"] == "marked_paradox" and 
                                                               p["source_id"] == source_id and 
                                                               p["target_id"] == target_id for p in paradoxes):
                    paradoxes.append({
                        "type": "marked_paradox",
                        "source_id": source_id,
                        "target_id": target_id,
                        "severity": 0.95,
                        "resolution_difficulty": 0.9,
                        "description": "Explizit markiertes Paradoxon",
                        "resolution_options": metadata.get('resolution_options', [])
                    })
                    continue
                
                if source_id in timeline.nodes and target_id in timeline.nodes:
                    source_node = timeline.nodes[source_id]
                    target_node = timeline.nodes[target_id]
                    
                    if source_node.timestamp > target_node.timestamp:
                        time_diff = source_node.timestamp - target_node.timestamp
                        severity = min(1.0, time_diff / 86400)  # Skaliere basierend auf der Zeitdifferenz (max 1 Tag)
                        
                        if not any(p["type"] == "causal_paradox" and 
                                  p["source_id"] == source_id and 
                                  p["target_id"] == target_id for p in paradoxes):
                            paradoxes.append({
                                "type": "causal_paradox",
                                "source_id": source_id,
                                "target_id": target_id,
                                "time_difference": time_diff,
                                "severity": severity,
                                "resolution_difficulty": 0.6 + (severity * 0.4),
                                "description": "Kausales Paradoxon: Späterer Knoten beeinflusst früheren Knoten",
                                "resolution_options": [
                                    "Temporale Rekursion mit Quantenkorrektur",
                                    "Knotenaufspaltung mit Wahrscheinlichkeitsverteilung"
                                ]
                            })
                    
        # --- 3. WAHRSCHEINLICHKEITSPARADOXA ---
        # Optimierte Erkennung von Wahrscheinlichkeitsparadoxa mit MLX
        if use_optimized_path:
            try:
                # Extrahiere alle Knotenwahrscheinlichkeiten für MLX-Verarbeitung
                node_ids = list(timeline.nodes.keys())
                probabilities = mx.array([node.probability for node in timeline.nodes.values()])
                
                # Finde alle Knoten mit Wahrscheinlichkeit < 0.1 (Optimierte Vektorisierung)
                low_prob_indices = mx.where(probabilities < 0.1)[0]
                
                # Füge Paradoxa hinzu für Knoten mit niedrigen Wahrscheinlichkeiten
                for idx in low_prob_indices:
                    node_id = node_ids[idx]
                    prob = probabilities[idx].item()
                    paradoxes.append({
                        "type": "probability_paradox",
                        "node_id": node_id,
                        "probability": prob,
                        "severity": 1.0 - prob,
                        "resolution_difficulty": 0.5,
                        "description": "Wahrscheinlichkeitsparadoxon: Knoten mit extrem niedriger Wahrscheinlichkeit",
                        "resolution_options": [
                            "Wahrscheinlichkeitsanreicherung mit Paralleldimension",
                            "Löschung mit Quantenrückschreibung",
                            "Verschmelzung mit höherwahrscheinlichem Knoten"
                        ]
                    })
            except Exception as e:
                logger.warning(f"MLX-Optimierung für Wahrscheinlichkeitsparadoxa fehlgeschlagen: {e}, Fallback")
        
        # Standardpfad für Wahrscheinlichkeitsparadoxa, wenn MLX nicht verwendet wird oder fehlgeschlagen ist
        if not use_optimized_path or len([p for p in paradoxes if p["type"] == "probability_paradox"]) == 0:
            for node_id, node in timeline.nodes.items():
                if node.probability < 0.1 and not any(p["type"] == "probability_paradox" and p["node_id"] == node_id for p in paradoxes):
                    paradoxes.append({
                        "type": "probability_paradox",
                        "node_id": node_id,
                        "probability": node.probability,
                        "severity": 1.0 - node.probability,
                        "resolution_difficulty": 0.5,
                        "description": "Wahrscheinlichkeitsparadoxon: Knoten mit extrem niedriger Wahrscheinlichkeit",
                        "resolution_options": [
                            "Wahrscheinlichkeitsanreicherung",
                            "Löschung mit Feedback-Korrektur"
                        ]
                    })
        
        # --- 4. ENTROPIE-INVERSIONSPARADOXA (neu) ---
        # Erkennung von Entropie-Inversionsparadoxa (Abnahme der Entropie ohne externe Energie)
        try:
            # Berechne die Entropieverteilung der Zeitlinie
            node_timestamps = [(node_id, node.timestamp) for node_id, node in timeline.nodes.items()]
            node_timestamps.sort(key=lambda x: x[1])  # Sortiere nach Zeitstempel
            
            # Vorbereitung der Entropiewerte
            entropy_values = []
            for node_id, _ in node_timestamps:
                node = timeline.nodes[node_id]
                # Einfache Entropieschätzung basierend auf Wahrscheinlichkeit und Verbindungen
                connected_nodes = self._get_connected_nodes(timeline, node_id)
                connection_count = len(connected_nodes)
                # Entropie = Wahrscheinlichkeit * log(Verbindungen + 1)
                if connection_count > 0:
                    entropy = -node.probability * np.log(node.probability) * np.log(connection_count + 1)
                else:
                    entropy = -node.probability * np.log(node.probability) if node.probability > 0 else 0
                entropy_values.append((node_id, entropy))
            
            # Suche nach signifikanten Entropieabfällen
            for i in range(1, len(entropy_values)):
                current_id, current_entropy = entropy_values[i]
                prev_id, prev_entropy = entropy_values[i-1]
                
                # Wenn die Entropie signifikant abnimmt (um mehr als 30%)
                if current_entropy < prev_entropy * 0.7:
                    delta_entropy = prev_entropy - current_entropy
                    severity = min(1.0, delta_entropy * 5)  # Skaliere mit der Entropiedifferenz
                    
                    paradoxes.append({
                        "type": "entropy_inversion",
                        "source_node_id": prev_id,
                        "target_node_id": current_id,
                        "entropy_delta": delta_entropy,
                        "severity": severity,
                        "resolution_difficulty": 0.8,
                        "description": "Entropie-Inversionsparadoxon: Unnatürliche Entropieabnahme ohne erkennbare Energiezufuhr",
                        "resolution_options": [
                            "Entropie-Ausgleich mit Quantenkälte",
                            "Dimensionsgrenzenverschiebung",
                            "Temporale Isolierung der Inversionsknoten"
                        ]
                    })
        except Exception as e:
            logger.warning(f"Entropie-Paradoxerkennung fehlgeschlagen: {e}")
        
        # --- 5. QUANTENPARADOXA (neu) ---
        # Prüfe auf Quantenparadoxa in Knoten mit Quantensuperpositionen
        try:
            for node_id, node in timeline.nodes.items():
                metadata = getattr(node, 'metadata', {})
                if metadata.get('quantum_state', False) or metadata.get('superposition', False):
                    # Prüfe auf widersprüchliche Quantenzustände
                    quantum_states = metadata.get('quantum_states', [])
                    if len(quantum_states) > 1:
                        # Berechne Konfliktrate zwischen Quantenzuständen
                        conflict_score = 0
                        for i, state1 in enumerate(quantum_states):
                            for state2 in quantum_states[i+1:]:
                                if state1.get('value') != state2.get('value') and state1.get('probability', 0) > 0.4 and state2.get('probability', 0) > 0.4:
                                    conflict_score += state1.get('probability', 0) * state2.get('probability', 0)
                        
                        if conflict_score > 0.2:  # Signifikanter Konflikt
                            paradoxes.append({
                                "type": "quantum_paradox",
                                "node_id": node_id,
                                "conflict_score": conflict_score,
                                "severity": min(1.0, conflict_score * 2),
                                "resolution_difficulty": 0.9,
                                "description": "Quantenparadoxon: Inkonsistente Superpositionen mit hoher Wahrscheinlichkeit",
                                "resolution_options": [
                                    "Wellenfunktionskollabierung mit Dimensionsselektion",
                                    "Quantenverschränkung auflösen",
                                    "Zustandsraum erweitern mit multidimensionaler Projektion"
                                ]
                            })
        except Exception as e:
            logger.warning(f"Quantum-Paradoxerkennung fehlgeschlagen: {e}")
        
        # Berechne eine Gesamtbewertung des Paradox-Schweregrades und Lösungsaufwandes
        total_severity = sum(p.get("severity", 0) for p in paradoxes)
        avg_severity = total_severity / len(paradoxes) if paradoxes else 0
        avg_resolution_difficulty = sum(p.get("resolution_difficulty", 0) for p in paradoxes) / len(paradoxes) if paradoxes else 0
        
        # Empfehle die beste Paradoxauflösungsstrategie basierend auf den Paradoxtypen
        resolution_strategy = None
        if paradoxes:
            paradox_types = [p["type"] for p in paradoxes]
            if "quantum_paradox" in paradox_types or "entropy_inversion" in paradox_types:
                resolution_strategy = "quantum_stabilization"
            elif "cyclic_time_loop" in paradox_types or "direct_time_loop" in paradox_types:
                resolution_strategy = "timeline_bifurcation"
            elif "causal_paradox" in paradox_types:
                resolution_strategy = "causal_reconstruction"
            else:
                resolution_strategy = "probability_reinforcement"
        
        # Leistungsmessung abschließen
        elapsed_time = time.time() - start_time
        logger.info(f"Paradoxerkennung für Zeitlinie {timeline_id} abgeschlossen: "
                   f"{len(paradoxes)} Paradoxa in {elapsed_time*1000:.2f}ms ")
                
        return {
            "paradoxes": paradoxes,
            "paradox_count": len(paradoxes),
            "timeline_id": timeline_id,
            "total_severity": total_severity,
            "avg_severity": avg_severity,
            "avg_resolution_difficulty": avg_resolution_difficulty,
            "recommended_resolution_strategy": resolution_strategy,
            "analysis_time_ms": elapsed_time * 1000
        }
    
    def generate_timeline_events(self, timeline_id: str, count: int = 5) -> List[Dict[str, Any]]:
        """
        Generiert Ereignisse für eine Zeitlinie
        
        Args:
            timeline_id: ID der Zeitlinie
            count: Anzahl der zu generierenden Ereignisse
            
        Returns:
            Liste von generierten Ereignissen
        """
        timeline = self.get_registered_timeline(timeline_id)
        if not timeline:
            logger.warning(f"Zeitlinie {timeline_id} nicht gefunden")
            return []
            
        # Prüfe, ob ein EventGenerator verfügbar ist
        if not hasattr(self, 'event_generator') or self.event_generator is None:
            from miso.simulation.event_generator import EventGenerator
            self.event_generator = EventGenerator(prism_engine=self)
            
        # Sammle alle Knoten-IDs
        node_ids = list(timeline.nodes.keys())
        
        # Generiere Ereignisse
        events = self.event_generator.generate_events_for_timeline(
            timeline_id=timeline_id,
            node_ids=node_ids,
            count=count
        )
        
        # Konvertiere Ereignisse in Dictionaries
        return [event.to_dict() for event in events]
    
    def simulate_timeline(self, timeline_id: str, steps: int = 10, variation_factor: float = 0.2) -> Dict[str, Any]:
        """
        Simuliert die Entwicklung einer Zeitlinie über mehrere Schritte
        
        Args:
            timeline_id: ID der Zeitlinie
            steps: Anzahl der Simulationsschritte
            variation_factor: Faktor für die Variation der Knotenwahrscheinlichkeiten
            
        Returns:
            Dictionary mit Simulationsergebnissen
        """
        timeline = self.get_registered_timeline(timeline_id)
        if not timeline:
            logger.warning(f"Zeitlinie {timeline_id} nicht gefunden")
            return {"error": f"Zeitlinie {timeline_id} nicht gefunden"}
            
        # Erstelle eine Kopie der Zeitlinie für die Simulation
        from copy import deepcopy
        simulation_timeline = deepcopy(timeline)
        simulation_timeline.id = f"{timeline_id}_sim"
        
        # Registriere die Simulationszeitlinie
        self.register_timeline(simulation_timeline)
        
        # Simulationsergebnisse
        results = {
            "initial_state": {
                "node_count": len(simulation_timeline.nodes),
                "connection_count": len(simulation_timeline.connections) if hasattr(simulation_timeline, 'connections') else 0,
                "probability": self.calculate_timeline_probability(simulation_timeline.id)
            },
            "steps": []
        }
        
        # Führe Simulationsschritte durch
        for step in range(steps):
            # Variiere die Wahrscheinlichkeiten der Knoten
            for node_id, node in simulation_timeline.nodes.items():
                variation = (np.random.random() - 0.5) * variation_factor
                node.probability = max(0.0, min(1.0, node.probability + variation))
                
            # Berechne die Wahrscheinlichkeit der Zeitlinie
            probability = self.calculate_timeline_probability(simulation_timeline.id)
            
            # Führe eine Stabilitätsanalyse durch
            stability = self.analyze_timeline_stability(simulation_timeline.id)
            
            # Generiere Ereignisse für diesen Schritt
            events = self.generate_timeline_events(simulation_timeline.id, count=2)
            
            # Speichere die Ergebnisse dieses Schritts
            results["steps"].append({
                "step": step + 1,
                "probability": probability,
                "stability": stability,
                "events": events
            })
            
        # Berechne die Endergebnisse
        results["final_state"] = {
            "node_count": len(simulation_timeline.nodes),
            "connection_count": len(simulation_timeline.connections) if hasattr(simulation_timeline, 'connections') else 0,
            "probability": self.calculate_timeline_probability(simulation_timeline.id),
            "stability": self.analyze_timeline_stability(simulation_timeline.id)
        }
        
        # Entferne die Simulationszeitlinie
        self.unregister_timeline(simulation_timeline.id)
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """
        Gibt den Status der PRISM-Engine zurück
        
        Returns:
            Statusbericht
        """
        return {
            "running": self.running,
            "matrix_dimensions": self.matrix.dimensions,
            "matrix_backend": self.matrix.tensor_backend,
            "registered_streams": list(self.stream_analyzer.data_buffer.keys()),
            "registered_patterns": list(self.dissonance_scanner.expected_patterns.keys()),
            "current_time_scale": self.time_scope.current_scale,
            "has_dependencies": HAS_DEPENDENCIES,
            "device": str(device),
            "t_mathematics_backend": "mlx" if is_apple_silicon else ("torch" if torch.cuda.is_available() else "numpy"),
            "probability_thresholds": {
                "low": "< 40%",
                "medium": "40-90%",
                "high": "> 90%"
            }
        }
