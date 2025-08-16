#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RecursiveEvaluator.py
=====================

Modul zur kontinuierlichen und systematischen Bewertung der VXOR AI-Performance,
Identifizierung von Verbesserungspotenzialen und Priorisierung von Optimierungsmaßnahmen.

Teil der MISO Ultimate AGI - Phase 7 (Recursive Self-Improvement)
"""

import os
import sys
import json
import time
import uuid
import logging
import datetime
import numpy as np
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Set

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MISO.RecursiveEvaluator')

class RecursiveEvaluator:
    """
    Evaluiert kontinuierlich die Performance des VXOR AI-Systems, identifiziert 
    Schwachstellen und Optimierungspotenziale und priorisiert Verbesserungsmaßnahmen.
    
    Features:
    - Systematische Analyse von Trainingsmetriken, Ausgabequalität und Reflexionsdaten
    - Intelligente Identifikation von Verbesserungspotenzialen
    - Priorisierung von Optimierungsmaßnahmen basierend auf Impact und Dringlichkeit
    - Integration mit VXOR-ETHICA und dem ValueAligner zur Berücksichtigung ethischer Prinzipien
    - Detaillierte JSON-Logs für Audit-Trails und Nachvollziehbarkeit
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 base_output_dir: Optional[str] = None,
                 metrics_dirs: Optional[List[str]] = None,
                 evaluation_interval: int = 3600,  # 1 Stunde
                 min_data_points: int = 100,
                 enable_continuous_evaluation: bool = False,
                 confidence_threshold: float = 0.75,
                 improvement_categories: Optional[List[str]] = None,
                 max_active_improvements: int = 5,
                 modules_to_monitor: Optional[List[str]] = None):
        """
        Initialisiert den RecursiveEvaluator.
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
            base_output_dir: Basisverzeichnis für Ausgaben
            metrics_dirs: Liste von Verzeichnissen mit Trainings- und Inferenzmetriken
            evaluation_interval: Intervall für kontinuierliche Evaluation in Sekunden
            min_data_points: Mindestanzahl von Datenpunkten für statistische Analysen
            enable_continuous_evaluation: Ob kontinuierliche Evaluation aktiviert werden soll
            confidence_threshold: Schwellenwert für Konfidenz bei Vorschlägen (0-1)
            improvement_categories: Kategorien von Verbesserungen (z.B. Modellarchitektur, Training)
            max_active_improvements: Maximale Anzahl gleichzeitig aktiver Verbesserungen
            modules_to_monitor: Liste von Modulen, die überwacht werden sollen
        """
        # Generiere eine eindeutige Session-ID
        self.session_id = str(uuid.uuid4())
        logger.info(f"Initialisiere RecursiveEvaluator mit Session-ID: {self.session_id}")
        
        # Lade Konfiguration
        self.config = self._load_config(config_path) if config_path else {}
        
        # Setze Basisparameter
        self.base_output_dir = base_output_dir or self.config.get('output_dir', './output')
        self.metrics_dirs = metrics_dirs or self.config.get('metrics_dirs', ['./logs/metrics', './logs/training'])
        self.evaluation_interval = evaluation_interval
        self.min_data_points = min_data_points
        self.enable_continuous_evaluation = enable_continuous_evaluation
        self.confidence_threshold = confidence_threshold
        
        # Verbesserungskategorien
        self.improvement_categories = improvement_categories or [
            "model_architecture",
            "training_data",
            "training_process",
            "inference_process",
            "optimization",
            "integration",
            "ethical_alignment"
        ]
        
        self.max_active_improvements = max_active_improvements
        
        # Module zu überwachen
        self.modules_to_monitor = modules_to_monitor or [
            "training_controller",
            "inference_engine",
            "data_processor",
            "ethical_framework",
            "federated_learning"
        ]
        
        # Erstelle die erforderlichen Verzeichnisse
        self._setup_directories()
        
        # Initialisiere Metriken-Speicher
        self.performance_metrics = {
            "training": {},
            "inference": {},
            "reflection": {},
            "ethics": {},
            "resource_usage": {}
        }
        
        # Speicher für identifizierte Verbesserungspotenziale
        self.improvement_opportunities = []
        
        # Prioritäts-Queue für Verbesserungen
        self.prioritized_improvements = []
        
        # Historie von durchgeführten Evaluationen
        self.evaluation_history = []
        
        # Status-Variablen
        self.is_evaluating = False
        self.evaluation_thread = None
        self.stop_evaluation = threading.Event()
        
        # Lade vorhandene Metriken, falls verfügbar
        self._load_existing_metrics()
        
        # Starte kontinuierliche Evaluation, wenn aktiviert
        if self.enable_continuous_evaluation:
            self.start_continuous_evaluation()
        
        logger.info("RecursiveEvaluator erfolgreich initialisiert")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Lädt die Konfigurationsdatei."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Konfiguration aus {config_path} geladen")
            return config
        except Exception as e:
            logger.warning(f"Konnte Konfiguration nicht laden: {e}. Verwende Standardeinstellungen.")
            return {}
    
    def _setup_directories(self) -> None:
        """Erstellt die erforderlichen Verzeichnisse."""
        try:
            # Erstelle Basisverzeichnis
            os.makedirs(self.base_output_dir, exist_ok=True)
            
            # Erstelle spezifische Unterverzeichnisse
            self.evaluation_dir = os.path.join(self.base_output_dir, 'evaluations')
            self.opportunities_dir = os.path.join(self.base_output_dir, 'improvement_opportunities')
            self.priorities_dir = os.path.join(self.base_output_dir, 'prioritized_improvements')
            
            os.makedirs(self.evaluation_dir, exist_ok=True)
            os.makedirs(self.opportunities_dir, exist_ok=True)
            os.makedirs(self.priorities_dir, exist_ok=True)
            
            logger.info(f"Verzeichnisstruktur unter {self.base_output_dir} eingerichtet")
        except Exception as e:
            logger.error(f"Fehler beim Einrichten der Verzeichnisse: {e}")
            raise
    
    def _load_existing_metrics(self) -> None:
        """Lädt existierende Metriken aus den Metrik-Verzeichnissen."""
        try:
            metrics_loaded = 0
            for metrics_dir in self.metrics_dirs:
                if not os.path.exists(metrics_dir):
                    logger.warning(f"Metriken-Verzeichnis {metrics_dir} existiert nicht.")
                    continue
                
                # Suche nach JSON-Dateien
                for root, _, files in os.walk(metrics_dir):
                    for file in files:
                        if file.endswith('.json'):
                            try:
                                file_path = os.path.join(root, file)
                                with open(file_path, 'r') as f:
                                    metrics_data = json.load(f)
                                
                                # Kategorisiere Metriken basierend auf Dateinamen oder Inhalt
                                category = self._determine_metric_category(file, metrics_data)
                                if category in self.performance_metrics:
                                    # Extrahiere Zeitstempel aus Datei oder verwende Modification Time
                                    timestamp = metrics_data.get('timestamp', datetime.datetime.fromtimestamp(
                                        os.path.getmtime(file_path)).isoformat())
                                    
                                    # Füge Metriken hinzu
                                    key = f"{timestamp}_{Path(file).stem}"
                                    self.performance_metrics[category][key] = metrics_data
                                    metrics_loaded += 1
                            except Exception as e:
                                logger.warning(f"Fehler beim Laden der Metrikdatei {file}: {e}")
            
            logger.info(f"Insgesamt {metrics_loaded} Metrikdateien aus {len(self.metrics_dirs)} Verzeichnissen geladen")
        except Exception as e:
            logger.error(f"Fehler beim Laden vorhandener Metriken: {e}")
    
    def _determine_metric_category(self, filename: str, data: Dict[str, Any]) -> str:
        """Bestimmt die Kategorie einer Metrikdatei basierend auf Dateinamen und Inhalt."""
        filename_lower = filename.lower()
        
        # Prüfe Dateinamen auf Hinweise
        if 'train' in filename_lower or 'training' in filename_lower:
            return "training"
        elif 'infer' in filename_lower or 'inference' in filename_lower:
            return "inference"
        elif 'reflect' in filename_lower or 'reflection' in filename_lower:
            return "reflection"
        elif 'ethic' in filename_lower or 'value' in filename_lower or 'align' in filename_lower:
            return "ethics"
        elif 'resource' in filename_lower or 'memory' in filename_lower or 'cpu' in filename_lower:
            return "resource_usage"
        
        # Prüfe Inhalt, falls Dateiname nicht eindeutig ist
        if data.get('type') in self.performance_metrics:
            return data['type']
        
        # Prüfe Schlüssel im Datensatz
        if any(k in data for k in ['loss', 'accuracy', 'epoch', 'learning_rate']):
            return "training"
        elif any(k in data for k in ['latency', 'throughput', 'precision', 'recall']):
            return "inference"
        elif any(k in data for k in ['self_evaluation', 'critique', 'confidence']):
            return "reflection"
        elif any(k in data for k in ['values', 'ethical_score', 'alignment']):
            return "ethics"
        elif any(k in data for k in ['memory_usage', 'cpu_usage', 'gpu_usage']):
            return "resource_usage"
        
        # Fallback
        return "training"
    
    def start_continuous_evaluation(self) -> bool:
        """Startet die kontinuierliche Evaluation in einem eigenen Thread."""
        if self.is_evaluating:
            logger.warning("Kontinuierliche Evaluation läuft bereits")
            return False
        
        try:
            # Setze Stop-Event zurück
            self.stop_evaluation.clear()
            
            # Starte Evaluation-Thread
            self.evaluation_thread = threading.Thread(target=self._continuous_evaluation_loop, daemon=True)
            self.evaluation_thread.start()
            
            self.is_evaluating = True
            logger.info("Kontinuierliche Evaluation gestartet")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Starten der kontinuierlichen Evaluation: {e}")
            return False
    
    def stop_continuous_evaluation(self) -> bool:
        """Stoppt die kontinuierliche Evaluation."""
        if not self.is_evaluating:
            logger.warning("Kontinuierliche Evaluation läuft nicht")
            return False
        
        try:
            # Signalisiere dem Thread, dass er stoppen soll
            self.stop_evaluation.set()
            
            # Warte auf das Ende des Threads mit Timeout
            if self.evaluation_thread:
                self.evaluation_thread.join(timeout=3.0)
            
            self.is_evaluating = False
            logger.info("Kontinuierliche Evaluation gestoppt")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Stoppen der kontinuierlichen Evaluation: {e}")
            return False
    
    def _continuous_evaluation_loop(self) -> None:
        """Hauptschleife für die kontinuierliche Evaluation."""
        try:
            while not self.stop_evaluation.is_set():
                # Führe vollständige Systemevaluation durch
                self.analyze_system_performance()
                
                # Identifiziere Verbesserungspotenziale
                self.identify_improvement_opportunities()
                
                # Priorisiere Verbesserungen
                self.prioritize_improvements()
                
                # Speichere Ergebnisse
                self._save_evaluation_results()
                
                # Warte für das nächste Evaluations-Intervall
                logger.info(f"Nächste Evaluation in {self.evaluation_interval} Sekunden")
                self.stop_evaluation.wait(self.evaluation_interval)
        
        except Exception as e:
            logger.error(f"Fehler in der Evaluationsschleife: {e}")
            self.is_evaluating = False
    
    def analyze_system_performance(self) -> Dict[str, Any]:
        """
        Analysiert die Systemleistung durch Auswertung von Trainingsmetriken, 
        Ausgabequalität und Reflexionsdaten.
        
        Returns:
            Dict mit Analyseergebnissen nach Kategorien
        """
        logger.info("Beginne Analyse der Systemperformance...")
        
        # Aktualisiere Metriken vor der Analyse
        self._load_latest_metrics()
        
        # Initialisiere Ergebnisstruktur
        analysis_results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "session_id": self.session_id,
            "categories": {},
            "trends": {},
            "anomalies": [],
            "overall_health": {
                "score": 0.0,
                "status": "unknown",
                "confidence": 0.0
            },
            "sufficient_data": False
        }
        
        # Prüfe, ob genügend Daten für statistische Analysen vorhanden sind
        data_counts = {category: len(metrics) for category, metrics in self.performance_metrics.items()}
        min_category_count = min(data_counts.values()) if data_counts else 0
        
        if min_category_count < self.min_data_points:
            logger.warning(f"Unzureichende Datenpunkte für vollständige Analyse ({min_category_count}/{self.min_data_points})")
            analysis_results["overall_health"]["status"] = "insufficient_data"
            analysis_results["sufficient_data"] = False
            return analysis_results
        
        # Genügend Daten vorhanden - führe vollständige Analyse durch
        analysis_results["sufficient_data"] = True
        
        # 1. Analysiere Trainingsmetriken
        training_analysis = self._analyze_training_metrics()
        analysis_results["categories"]["training"] = training_analysis
        
        # 2. Analysiere Inferenzmetriken
        inference_analysis = self._analyze_inference_metrics()
        analysis_results["categories"]["inference"] = inference_analysis
        
        # 3. Analysiere Reflexionsdaten
        reflection_analysis = self._analyze_reflection_data()
        analysis_results["categories"]["reflection"] = reflection_analysis
        
        # 4. Analysiere Ethikmetriken
        ethics_analysis = self._analyze_ethics_metrics()
        analysis_results["categories"]["ethics"] = ethics_analysis
        
        # 5. Analysiere Ressourcennutzung
        resource_analysis = self._analyze_resource_usage()
        analysis_results["categories"]["resource_usage"] = resource_analysis
        
        # 6. Identifiziere Trends über Zeit
        trends = self._identify_performance_trends()
        analysis_results["trends"] = trends
        
        # 7. Identifiziere Anomalien
        anomalies = self._detect_anomalies()
        analysis_results["anomalies"] = anomalies
        
        # 8. Berechne Gesundheitsscore des Gesamtsystems
        health_score, status, confidence = self._calculate_system_health(
            training_analysis, inference_analysis, reflection_analysis, ethics_analysis, resource_analysis
        )
        
        analysis_results["overall_health"] = {
            "score": health_score,
            "status": status,
            "confidence": confidence
        }
        
        # Speichere Analyseergebnisse
        self._save_analysis_results(analysis_results)
        
        # Aktualisiere Evaluationshistorie
        self.evaluation_history.append({
            "timestamp": analysis_results["timestamp"],
            "overall_health": analysis_results["overall_health"],
            "anomalies_count": len(anomalies)
        })
        
        logger.info(f"Systemanalyse abgeschlossen. Gesundheitsscore: {health_score:.2f}, Status: {status}")
        return analysis_results
    
    def _load_latest_metrics(self) -> None:
        """Lädt die neuesten Metriken aus den Metrik-Verzeichnissen."""
        # Implementierung ähnlich zu _load_existing_metrics, aber mit Fokus auf neuere Dateien
        latest_metrics_count = 0
        
        try:
            for metrics_dir in self.metrics_dirs:
                if not os.path.exists(metrics_dir):
                    continue
                
                # Suche nach neuen JSON-Dateien seit der letzten Aktualisierung
                for root, _, files in os.walk(metrics_dir):
                    for file in files:
                        if file.endswith('.json'):
                            file_path = os.path.join(root, file)
                            
                            # Prüfe, ob Datei bereits geladen wurde (basierend auf Pfad und Modifikationszeit)
                            file_mod_time = os.path.getmtime(file_path)
                            
                            # Erstelle Schlüssel basierend auf Pfad und Änderungszeit
                            file_key = f"{file_path}_{file_mod_time}"
                            
                            # Überspringe bereits geladene Dateien
                            skip_file = False
                            for category in self.performance_metrics.values():
                                for key in category.keys():
                                    if file_key in key:
                                        skip_file = True
                                        break
                                if skip_file:
                                    break
                            
                            if skip_file:
                                continue
                            
                            try:
                                with open(file_path, 'r') as f:
                                    metrics_data = json.load(f)
                                
                                # Kategorisiere Metriken
                                category = self._determine_metric_category(file, metrics_data)
                                if category in self.performance_metrics:
                                    # Extrahiere Zeitstempel oder verwende Modifikationszeit
                                    timestamp = metrics_data.get('timestamp', 
                                                    datetime.datetime.fromtimestamp(file_mod_time).isoformat())
                                    
                                    # Füge Metriken hinzu
                                    key = f"{timestamp}_{Path(file).stem}_{file_key}"
                                    self.performance_metrics[category][key] = metrics_data
                                    latest_metrics_count += 1
                            except Exception as e:
                                logger.warning(f"Fehler beim Laden der Metrikdatei {file}: {e}")
            
            if latest_metrics_count > 0:
                logger.info(f"{latest_metrics_count} neue Metrikdateien geladen")
            
        except Exception as e:
            logger.error(f"Fehler beim Laden der neuesten Metriken: {e}")
    
    def _analyze_training_metrics(self) -> Dict[str, Any]:
        """Analysiert die Trainingsmetriken auf Trends, Anomalien und Optimierungspotenziale."""
        logger.info("Analysiere Trainingsmetriken...")
        
        result = {
            "metrics_count": len(self.performance_metrics["training"]),
            "recent_trends": {},
            "issues": [],
            "strengths": [],
            "score": 0.0,
            "potential_improvements": []
        }
        
        if not self.performance_metrics["training"]:
            return result
        
        # Extrahiere Schlüsselmetriken aus den Trainingsdaten
        losses = []
        accuracies = []
        learning_rates = []
        timestamps = []
        convergence_rates = []
        
        # Sortiere nach Zeitstempel, falls vorhanden
        sorted_metrics = sorted(
            self.performance_metrics["training"].items(),
            key=lambda x: x[1].get("timestamp", "0")
        )
        
        for key, metrics in sorted_metrics:
            # Extrahiere Standardmetriken, falls vorhanden
            if "loss" in metrics:
                losses.append(float(metrics["loss"]))
            if "accuracy" in metrics:
                accuracies.append(float(metrics["accuracy"]))
            if "learning_rate" in metrics:
                learning_rates.append(float(metrics["learning_rate"]))
            if "timestamp" in metrics:
                timestamps.append(metrics["timestamp"])
            if "convergence_rate" in metrics:
                convergence_rates.append(float(metrics["convergence_rate"]))
        
        # Berechne Trends für jede Metrik, falls genügend Daten vorhanden sind
        if len(losses) >= 3:
            # Berechne Trend der letzten N Datenpunkte
            recent_losses = losses[-10:] if len(losses) >= 10 else losses
            loss_trend = self._calculate_trend(recent_losses)
            result["recent_trends"]["loss"] = loss_trend
            
            # Identifiziere problematische Trainingsmuster
            if loss_trend > 0.05:  # Loss steigt an
                result["issues"].append({
                    "type": "increasing_loss",
                    "severity": "high" if loss_trend > 0.2 else "medium",
                    "description": f"Verlustfunktion steigt an (Trend: {loss_trend:.3f})"
                })
                
                # Schlage mögliche Verbesserungen vor
                result["potential_improvements"].append({
                    "target": "learning_rate",
                    "action": "decrease",
                    "reason": "Steigende Verlustfunktion deutet auf zu hohe Lernrate hin",
                    "confidence": 0.8 if loss_trend > 0.2 else 0.6
                })
            elif loss_trend < -0.01 and loss_trend > -0.05:  # Gesunder Abstieg
                result["strengths"].append({
                    "type": "healthy_convergence",
                    "description": "Gesunde Konvergenz der Verlustfunktion"
                })
            elif loss_trend > -0.001:  # Stagnation
                result["issues"].append({
                    "type": "plateauing_loss",
                    "severity": "medium",
                    "description": "Verlustfunktion stagniert, Training könnte in einem lokalen Minimum stecken"
                })
                
                # Schlage mögliche Verbesserungen vor
                result["potential_improvements"].append({
                    "target": "optimizer",
                    "action": "change",
                    "reason": "Stagnierende Verlustfunktion deutet auf Bedarf eines adaptiveren Optimierers hin",
                    "confidence": 0.7
                })
                
                result["potential_improvements"].append({
                    "target": "learning_rate_schedule",
                    "action": "implement",
                    "reason": "Ein Lernratenplan könnte helfen, lokale Minima zu überwinden",
                    "confidence": 0.75
                })
        
        # Analyse der Genauigkeit (Accuracy)
        if len(accuracies) >= 3:
            recent_accuracies = accuracies[-10:] if len(accuracies) >= 10 else accuracies
            accuracy_trend = self._calculate_trend(recent_accuracies)
            result["recent_trends"]["accuracy"] = accuracy_trend
            
            # Identifiziere Probleme mit der Genauigkeit
            if accuracy_trend < -0.05:  # Accuracy sinkt
                result["issues"].append({
                    "type": "decreasing_accuracy",
                    "severity": "high" if accuracy_trend < -0.2 else "medium",
                    "description": f"Genauigkeit nimmt ab (Trend: {accuracy_trend:.3f})"
                })
                
                # Schlage mögliche Verbesserungen vor
                result["potential_improvements"].append({
                    "target": "regularization",
                    "action": "increase",
                    "reason": "Sinkende Genauigkeit könnte auf Überanpassung hindeuten",
                    "confidence": 0.7
                })
            elif accuracy_trend > 0.05:  # Gesunder Anstieg
                result["strengths"].append({
                    "type": "improving_accuracy",
                    "description": f"Stabile Verbesserung der Genauigkeit (Trend: {accuracy_trend:.3f})"
                })
        
        # Analyse der Konvergenzrate (falls vorhanden)
        if len(convergence_rates) >= 3:
            avg_convergence = sum(convergence_rates) / len(convergence_rates)
            result["recent_trends"]["convergence_rate"] = avg_convergence
            
            if avg_convergence < 0.01:  # Sehr langsame Konvergenz
                result["issues"].append({
                    "type": "slow_convergence",
                    "severity": "medium",
                    "description": "Training konvergiert sehr langsam"
                })
                
                # Schlage mögliche Verbesserungen vor
                result["potential_improvements"].append({
                    "target": "batch_size",
                    "action": "increase",
                    "reason": "Größere Batches könnten die Konvergenz beschleunigen",
                    "confidence": 0.6
                })
                
                result["potential_improvements"].append({
                    "target": "model_initialization",
                    "action": "improve",
                    "reason": "Bessere Gewichtsinitialisiering könnte schnellere Konvergenz ermöglichen",
                    "confidence": 0.65
                })
        
        # Berechne Gesamtscore für das Training zwischen 0.0-1.0
        if losses and accuracies:
            # Bewerte Verlust und Genauigkeit
            avg_recent_loss = sum(losses[-5:]) / len(losses[-5:]) if len(losses) >= 5 else sum(losses) / len(losses)
            avg_recent_accuracy = sum(accuracies[-5:]) / len(accuracies[-5:]) if len(accuracies) >= 5 else sum(accuracies) / len(accuracies)
            
            # Berechne Score basierend auf den Metriken und Trends
            # Niedrigerer Verlust und höhere Genauigkeit sind besser
            loss_score = max(0.0, 1.0 - avg_recent_loss) if avg_recent_loss < 1.0 else 0.1
            accuracy_score = avg_recent_accuracy
            
            # Berücksichtige auch Trends bei der Bewertung
            trend_modifier = 0.0
            if "loss" in result["recent_trends"] and result["recent_trends"]["loss"] < 0:  # Fallender Verlust ist gut
                trend_modifier += min(0.2, abs(result["recent_trends"]["loss"]) * 2)
            if "accuracy" in result["recent_trends"] and result["recent_trends"]["accuracy"] > 0:  # Steigende Genauigkeit ist gut
                trend_modifier += min(0.2, result["recent_trends"]["accuracy"] * 2)
            
            # Berechne Gesamtscore
            result["score"] = (0.4 * loss_score + 0.4 * accuracy_score + 0.2 * (1.0 + trend_modifier))
            # Begrenzen auf 0.0-1.0
            result["score"] = max(0.0, min(1.0, result["score"]))
        
        return result
    
    def _analyze_inference_metrics(self) -> Dict[str, Any]:
        """Analysiert die Inferenzmetriken auf Latenz, Durchsatz, Präzision und Rückruf."""
        logger.info("Analysiere Inferenzmetriken...")
        
        result = {
            "metrics_count": len(self.performance_metrics["inference"]),
            "recent_trends": {},
            "issues": [],
            "strengths": [],
            "score": 0.0,
            "potential_improvements": []
        }
        
        if not self.performance_metrics["inference"]:
            return result
        
        # Extrahiere Schlüsselmetriken aus den Inferenzdaten
        latencies = []
        throughputs = []
        precisions = []
        recalls = []
        f1_scores = []
        timestamps = []
        
        # Sortiere nach Zeitstempel, falls vorhanden
        sorted_metrics = sorted(
            self.performance_metrics["inference"].items(),
            key=lambda x: x[1].get("timestamp", "0")
        )
        
        for key, metrics in sorted_metrics:
            # Extrahiere Standardmetriken, falls vorhanden
            if "latency_ms" in metrics:
                latencies.append(float(metrics["latency_ms"]))
            if "throughput" in metrics:
                throughputs.append(float(metrics["throughput"]))
            if "precision" in metrics:
                precisions.append(float(metrics["precision"]))
            if "recall" in metrics:
                recalls.append(float(metrics["recall"]))
            if "f1_score" in metrics:
                f1_scores.append(float(metrics["f1_score"]))
            if "timestamp" in metrics:
                timestamps.append(metrics["timestamp"])
        
        # Analyse der Latenz
        if len(latencies) >= 3:
            recent_latencies = latencies[-10:] if len(latencies) >= 10 else latencies
            avg_latency = sum(recent_latencies) / len(recent_latencies)
            latency_trend = self._calculate_trend(recent_latencies)
            result["recent_trends"]["latency"] = latency_trend
            
            # Identifiziere Probleme mit der Latenz
            if latency_trend > 0.1:  # Latenz steigt an
                result["issues"].append({
                    "type": "increasing_latency",
                    "severity": "high" if latency_trend > 0.3 else "medium",
                    "description": f"Inferenzlatenz steigt an (Trend: {latency_trend:.3f})"
                })
                
                # Schlage mögliche Verbesserungen vor
                result["potential_improvements"].append({
                    "target": "model_optimization",
                    "action": "quantize",
                    "reason": "Quantisierung kann die Inferenzlatenz reduzieren",
                    "confidence": 0.85
                })
                
                result["potential_improvements"].append({
                    "target": "model_architecture",
                    "action": "simplify",
                    "reason": "Vereinfachung der Architektur könnte die Latenz verbessern",
                    "confidence": 0.7
                })
            elif latency_trend < -0.1:  # Latenz sinkt
                result["strengths"].append({
                    "type": "decreasing_latency",
                    "description": f"Inferenzlatenz verbessert sich (Trend: {latency_trend:.3f})"
                })
        
        # Analyse des Durchsatzes
        if len(throughputs) >= 3:
            recent_throughputs = throughputs[-10:] if len(throughputs) >= 10 else throughputs
            avg_throughput = sum(recent_throughputs) / len(recent_throughputs)
            throughput_trend = self._calculate_trend(recent_throughputs)
            result["recent_trends"]["throughput"] = throughput_trend
            
            if throughput_trend < -0.1:  # Durchsatz sinkt
                result["issues"].append({
                    "type": "decreasing_throughput",
                    "severity": "medium",
                    "description": f"Inferenzdurchsatz nimmt ab (Trend: {throughput_trend:.3f})"
                })
                
                # Schlage mögliche Verbesserungen vor
                result["potential_improvements"].append({
                    "target": "batch_inference",
                    "action": "implement",
                    "reason": "Batch-Inferenz könnte den Durchsatz verbessern",
                    "confidence": 0.8
                })
            elif throughput_trend > 0.1:  # Durchsatz steigt
                result["strengths"].append({
                    "type": "increasing_throughput",
                    "description": f"Inferenzdurchsatz verbessert sich (Trend: {throughput_trend:.3f})"
                })
        
        # Analyse der Präzision und des Rückrufs (wenn verfügbar)
        if len(precisions) >= 3 and len(recalls) >= 3:
            avg_precision = sum(precisions[-5:]) / len(precisions[-5:]) if len(precisions) >= 5 else sum(precisions) / len(precisions)
            avg_recall = sum(recalls[-5:]) / len(recalls[-5:]) if len(recalls) >= 5 else sum(recalls) / len(recalls)
            
            precision_trend = self._calculate_trend(precisions[-10:] if len(precisions) >= 10 else precisions)
            recall_trend = self._calculate_trend(recalls[-10:] if len(recalls) >= 10 else recalls)
            
            result["recent_trends"]["precision"] = precision_trend
            result["recent_trends"]["recall"] = recall_trend
            
            # Identifiziere Probleme mit Präzision/Rückruf
            if precision_trend < -0.05 and recall_trend < -0.05:  # Beide sinken
                result["issues"].append({
                    "type": "declining_accuracy_metrics",
                    "severity": "high",
                    "description": "Sowohl Präzision als auch Rückruf nehmen ab"
                })
                
                # Schlage mögliche Verbesserungen vor
                result["potential_improvements"].append({
                    "target": "training_data",
                    "action": "augment",
                    "reason": "Datenaugmentation könnte Präzision und Rückruf verbessern",
                    "confidence": 0.75
                })
            elif precision_trend < -0.05 and recall_trend > 0.05:  # Präzision sinkt, Rückruf steigt
                result["issues"].append({
                    "type": "precision_recall_tradeoff",
                    "severity": "medium",
                    "description": "Tradeoff zwischen Präzision und Rückruf verschiebt sich"
                })
        
        # Berechne Gesamtscore für die Inferenz
        if (latencies or throughputs) and (precisions or recalls or f1_scores):
            # Normalisiere Latenz (niedrigere ist besser)
            if latencies:
                avg_recent_latency = sum(latencies[-5:]) / len(latencies[-5:]) if len(latencies) >= 5 else sum(latencies) / len(latencies)
                # Annahme: 100ms ist ein guter Zielwert für Latenz (anpassen je nach Anwendungsfall)
                latency_score = max(0.0, 1.0 - (avg_recent_latency / 100.0))
            else:
                latency_score = 0.5  # Neutral, wenn keine Daten
            
            # Metriken für Genauigkeit
            if f1_scores:  # F1-Score ist ein kombiniertes Maß für Präzision und Rückruf
                avg_recent_f1 = sum(f1_scores[-5:]) / len(f1_scores[-5:]) if len(f1_scores) >= 5 else sum(f1_scores) / len(f1_scores)
                accuracy_score = avg_recent_f1
            elif precisions and recalls:
                avg_recent_precision = sum(precisions[-5:]) / len(precisions[-5:]) if len(precisions) >= 5 else sum(precisions) / len(precisions)
                avg_recent_recall = sum(recalls[-5:]) / len(recalls[-5:]) if len(recalls) >= 5 else sum(recalls) / len(recalls)
                # Harmonisches Mittel von Präzision und Rückruf
                if avg_recent_precision + avg_recent_recall > 0:
                    accuracy_score = 2 * (avg_recent_precision * avg_recent_recall) / (avg_recent_precision + avg_recent_recall)
                else:
                    accuracy_score = 0.0
            else:
                accuracy_score = 0.5  # Neutral, wenn keine Daten
            
            # Berücksichtige auch Trends bei der Bewertung
            trend_modifier = 0.0
            if "latency" in result["recent_trends"] and result["recent_trends"]["latency"] < 0:  # Fallende Latenz ist gut
                trend_modifier += min(0.1, abs(result["recent_trends"]["latency"]))
            if "throughput" in result["recent_trends"] and result["recent_trends"]["throughput"] > 0:  # Steigender Durchsatz ist gut
                trend_modifier += min(0.1, result["recent_trends"]["throughput"])
            
            # Berechne Gesamtscore
            result["score"] = (0.4 * latency_score + 0.4 * accuracy_score + 0.2 * (0.5 + trend_modifier))
            # Begrenzen auf 0.0-1.0
            result["score"] = max(0.0, min(1.0, result["score"]))
        
        return result
    
    def _calculate_trend(self, data_points: List[float]) -> float:
        """Berechnet den Trend einer Zeitreihe als normalisierte Steigung.
        
        Args:
            data_points: Liste von Datenpunkten (neueste zuletzt)
            
        Returns:
            Normalisierter Trend (positiv = steigend, negativ = fallend)
        """
        if len(data_points) < 2:
            return 0.0
        
        try:
            # Verwende einfache lineare Regression zur Trendermittlung
            n = len(data_points)
            x = np.array(range(n))
            y = np.array(data_points)
            
            # Berechne Steigung der Regressionslinie
            # Formel: Steigung = (n*Summe(xy) - Summe(x)*Summe(y)) / (n*Summe(x^2) - (Summe(x))^2)
            x_sum = np.sum(x)
            y_sum = np.sum(y)
            xy_sum = np.sum(x * y)
            x_squared_sum = np.sum(x**2)
            
            # Vermeidet Division durch Null
            denominator = n * x_squared_sum - x_sum**2
            if denominator == 0:
                return 0.0
            
            slope = (n * xy_sum - x_sum * y_sum) / denominator
            
            # Normalisiere die Steigung basierend auf dem Durchschnittswert
            avg_y = y_sum / n
            if avg_y != 0:  # Vermeidet Division durch Null
                normalized_slope = slope / abs(avg_y)
            else:
                normalized_slope = slope
            
            return normalized_slope
        except Exception as e:
            logger.warning(f"Fehler bei der Trendberechnung: {e}")
            return 0.0
    
    def _analyze_reflection_data(self) -> Dict[str, Any]:
        """Analysiert Selbstreflexions- und Selbstbewertungsdaten des Systems."""
        logger.info("Analysiere Reflexionsdaten...")
        
        result = {
            "metrics_count": len(self.performance_metrics["reflection"]),
            "recent_trends": {},
            "issues": [],
            "strengths": [],
            "score": 0.0,
            "potential_improvements": []
        }
        
        if not self.performance_metrics["reflection"]:
            return result
        
        # Extrahiere Schlüsselmetriken aus den Reflexionsdaten
        confidences = []
        self_critique_scores = []
        improvement_suggestions = []
        error_detections = []
        timestamps = []
        
        # Sortiere nach Zeitstempel, falls vorhanden
        sorted_metrics = sorted(
            self.performance_metrics["reflection"].items(),
            key=lambda x: x[1].get("timestamp", "0")
        )
        
        # Sammle relevante Reflexionsdaten
        for key, metrics in sorted_metrics:
            if "confidence" in metrics:
                confidences.append(float(metrics["confidence"]))
            if "self_critique_score" in metrics:
                self_critique_scores.append(float(metrics["self_critique_score"]))
            if "detected_errors" in metrics:
                error_detections.append(metrics["detected_errors"])
            if "improvement_suggestions" in metrics:
                improvement_suggestions.append(metrics["improvement_suggestions"])
            if "timestamp" in metrics:
                timestamps.append(metrics["timestamp"])
        
        # Analyse der Selbstvertrauenswerte
        if len(confidences) >= 3:
            recent_confidences = confidences[-10:] if len(confidences) >= 10 else confidences
            avg_confidence = sum(recent_confidences) / len(recent_confidences)
            confidence_trend = self._calculate_trend(recent_confidences)
            result["recent_trends"]["confidence"] = confidence_trend
            
            # Identifiziere Probleme mit dem Selbstvertrauen
            if avg_confidence < 0.4:  # Zu geringes Selbstvertrauen
                result["issues"].append({
                    "type": "low_confidence",
                    "severity": "medium",
                    "description": f"System zeigt niedriges Selbstvertrauen (Durchschnitt: {avg_confidence:.2f})"
                })
                
                # Schlage mögliche Verbesserungen vor
                result["potential_improvements"].append({
                    "target": "confidence_calibration",
                    "action": "improve",
                    "reason": "Bessere Kalibrierung des Selbstvertrauens erforderlich",
                    "confidence": 0.75
                })
            elif avg_confidence > 0.95:  # Übermäßiges Selbstvertrauen
                result["issues"].append({
                    "type": "overconfidence",
                    "severity": "high",
                    "description": f"System zeigt möglicherweise übermäßiges Selbstvertrauen (Durchschnitt: {avg_confidence:.2f})"
                })
                
                # Schlage mögliche Verbesserungen vor
                result["potential_improvements"].append({
                    "target": "uncertainty_estimation",
                    "action": "implement",
                    "reason": "Bessere Unsicherheitsschätzung könnte Übervertrauen reduzieren",
                    "confidence": 0.8
                })
            elif 0.6 <= avg_confidence <= 0.85:  # Gesundes Selbstvertrauen
                result["strengths"].append({
                    "type": "healthy_confidence",
                    "description": f"System zeigt gesundes und ausgewogenes Selbstvertrauen (Durchschnitt: {avg_confidence:.2f})"
                })
        
        # Analyse der Selbstkritikwerte
        if len(self_critique_scores) >= 3:
            recent_critiques = self_critique_scores[-10:] if len(self_critique_scores) >= 10 else self_critique_scores
            avg_critique = sum(recent_critiques) / len(recent_critiques)
            critique_trend = self._calculate_trend(recent_critiques)
            result["recent_trends"]["self_critique"] = critique_trend
            
            if avg_critique < 0.3:  # Zu wenig Selbstkritik
                result["issues"].append({
                    "type": "insufficient_self_critique",
                    "severity": "medium",
                    "description": f"System zeigt geringe Selbstkritikfähigkeit (Durchschnitt: {avg_critique:.2f})"
                })
                
                # Schlage mögliche Verbesserungen vor
                result["potential_improvements"].append({
                    "target": "reflection_module",
                    "action": "enhance",
                    "reason": "Verbesserung der Selbstreflexionsfähigkeit erforderlich",
                    "confidence": 0.7
                })
            elif avg_critique > 0.8:  # Übermäßige Selbstkritik
                result["issues"].append({
                    "type": "excessive_self_critique",
                    "severity": "low",
                    "description": f"System zeigt übermäßig hohe Selbstkritik (Durchschnitt: {avg_critique:.2f})"
                })
            elif 0.4 <= avg_critique <= 0.7:  # Gesunde Selbstkritik
                result["strengths"].append({
                    "type": "balanced_self_critique",
                    "description": f"System zeigt ausgeglichene Selbstkritikfähigkeit (Durchschnitt: {avg_critique:.2f})"
                })
        
        # Analyse erkannter Fehler (falls vorhanden)
        if error_detections:
            # Zähle die Anzahl erkannter Fehler
            total_errors = sum(len(errors) if isinstance(errors, list) else 1 for errors in error_detections)
            avg_errors_per_session = total_errors / len(error_detections)
            
            # Prüfe, ob die Fehlererkennungsrate stabil ist
            if len(error_detections) >= 5:
                recent_error_counts = [len(errors) if isinstance(errors, list) else 1 for errors in error_detections[-5:]]
                error_trend = self._calculate_trend(recent_error_counts)
                result["recent_trends"]["error_detection"] = error_trend
                
                if error_trend > 0.3:  # Steigende Fehlerrate
                    result["issues"].append({
                        "type": "increasing_errors",
                        "severity": "high",
                        "description": f"Steigende Anzahl erkannter Fehler (Trend: {error_trend:.2f})"
                    })
                elif error_trend < -0.3:  # Fallende Fehlerrate (positiv)
                    result["strengths"].append({
                        "type": "decreasing_errors",
                        "description": f"Rückgang der erkannten Fehler (Trend: {error_trend:.2f})"
                    })
        
        # Aggregiere Verbesserungsvorschläge aus der Selbstreflexion
        if improvement_suggestions:
            # Hier könnte eine komplexere Analyse der Verbesserungsvorschläge erfolgen,
            # z.B. Clustering nach Themen oder Bewertung der Qualität der Vorschläge
            recent_suggestions = []
            for suggestions in improvement_suggestions[-5:]:  # Betrachte nur die neuesten 5 Einträge
                if isinstance(suggestions, list):
                    recent_suggestions.extend(suggestions)
                else:
                    recent_suggestions.append(suggestions)
            
            # Extrahiere einzigartige Verbesserungsvorschläge
            unique_suggestions = set()
            for suggestion in recent_suggestions:
                if isinstance(suggestion, str):
                    unique_suggestions.add(suggestion)
                elif isinstance(suggestion, dict) and "description" in suggestion:
                    unique_suggestions.add(suggestion["description"])
            
            # Erfasse die Anzahl der einzigartigen Vorschläge
            result["unique_improvement_suggestions_count"] = len(unique_suggestions)
        
        # Berechne Gesamtscore für die Reflexionsfähigkeit
        if confidences and self_critique_scores:
            # Berechne Durchschnitte der neuesten Werte
            avg_recent_confidence = sum(confidences[-5:]) / len(confidences[-5:]) if len(confidences) >= 5 else sum(confidences) / len(confidences)
            avg_recent_critique = sum(self_critique_scores[-5:]) / len(self_critique_scores[-5:]) if len(self_critique_scores) >= 5 else sum(self_critique_scores) / len(self_critique_scores)
            
            # Bewerte Selbstvertrauen (zu niedrig oder zu hoch reduziert den Score)
            confidence_score = 1.0 - 2.0 * abs(avg_recent_confidence - 0.75)
            confidence_score = max(0.0, min(1.0, confidence_score))
            
            # Bewerte Selbstkritik (optimal zwischen 0.4 und 0.7)
            if 0.4 <= avg_recent_critique <= 0.7:
                critique_score = avg_recent_critique / 0.7
            elif avg_recent_critique > 0.7:
                critique_score = 1.0 - (avg_recent_critique - 0.7) / 0.3
            else:  # < 0.4
                critique_score = avg_recent_critique / 0.4
            critique_score = max(0.0, min(1.0, critique_score))
            
            # Berücksichtige auch Trends bei der Bewertung
            trend_modifier = 0.0
            if "error_detection" in result["recent_trends"] and result["recent_trends"]["error_detection"] < 0:  # Fallende Fehlerrate ist gut
                trend_modifier += min(0.1, abs(result["recent_trends"]["error_detection"]))
            
            # Berechne Gesamtscore
            result["score"] = (0.4 * confidence_score + 0.4 * critique_score + 0.2 * (0.5 + trend_modifier))
            # Begrenzen auf 0.0-1.0
            result["score"] = max(0.0, min(1.0, result["score"]))
        
        return result
    
    def _analyze_ethics_metrics(self) -> Dict[str, Any]:
        """Analysiert Metriken zur ethischen Ausrichtung und Wertebindung des Systems."""
        logger.info("Analysiere Ethikmetriken...")
        
        result = {
            "metrics_count": len(self.performance_metrics["ethics"]),
            "recent_trends": {},
            "issues": [],
            "strengths": [],
            "score": 0.0,
            "potential_improvements": []
        }
        
        if not self.performance_metrics["ethics"]:
            return result
        
        # Extrahiere Schlüsselmetriken aus den Ethikdaten
        value_alignment_scores = []
        fairness_scores = []
        bias_scores = []
        transparency_scores = []
        robustness_scores = []
        timestamps = []
        
        # Sortiere nach Zeitstempel, falls vorhanden
        sorted_metrics = sorted(
            self.performance_metrics["ethics"].items(),
            key=lambda x: x[1].get("timestamp", "0")
        )
        
        # Sammle relevante Ethikmetriken
        for key, metrics in sorted_metrics:
            if "value_alignment" in metrics:
                value_alignment_scores.append(float(metrics["value_alignment"]))
            if "fairness" in metrics:
                fairness_scores.append(float(metrics["fairness"]))
            if "bias_score" in metrics:
                # Niedrigerer Bias-Score ist besser
                bias_scores.append(float(metrics["bias_score"]))
            if "transparency" in metrics:
                transparency_scores.append(float(metrics["transparency"]))
            if "robustness" in metrics:
                robustness_scores.append(float(metrics["robustness"]))
            if "timestamp" in metrics:
                timestamps.append(metrics["timestamp"])
        
        # Analyse der Werteausrichtung
        if len(value_alignment_scores) >= 3:
            recent_values = value_alignment_scores[-10:] if len(value_alignment_scores) >= 10 else value_alignment_scores
            avg_alignment = sum(recent_values) / len(recent_values)
            alignment_trend = self._calculate_trend(recent_values)
            result["recent_trends"]["value_alignment"] = alignment_trend
            
            # Identifiziere Probleme mit der Werteausrichtung
            if avg_alignment < 0.7:  # Unzureichende Werteausrichtung
                result["issues"].append({
                    "type": "low_value_alignment",
                    "severity": "high" if avg_alignment < 0.5 else "medium",
                    "description": f"System zeigt unzureichende Werteausrichtung (Durchschnitt: {avg_alignment:.2f})"
                })
                
                # Schlage mögliche Verbesserungen vor
                result["potential_improvements"].append({
                    "target": "value_alignment",
                    "action": "enhance",
                    "reason": "Integration von VXOR-ETHICA muss verbessert werden",
                    "confidence": 0.85
                })
            elif avg_alignment > 0.9 and alignment_trend > 0:  # Hervorragende Werteausrichtung
                result["strengths"].append({
                    "type": "excellent_value_alignment",
                    "description": f"System zeigt hervorragende Werteausrichtung (Durchschnitt: {avg_alignment:.2f})"
                })
        
        # Analyse der Fairness (wenn verfügbar)
        if len(fairness_scores) >= 3:
            recent_fairness = fairness_scores[-10:] if len(fairness_scores) >= 10 else fairness_scores
            avg_fairness = sum(recent_fairness) / len(recent_fairness)
            fairness_trend = self._calculate_trend(recent_fairness)
            result["recent_trends"]["fairness"] = fairness_trend
            
            if avg_fairness < 0.6:  # Probleme mit Fairness
                result["issues"].append({
                    "type": "fairness_issues",
                    "severity": "high" if avg_fairness < 0.4 else "medium",
                    "description": f"System zeigt Fairnessprobleme (Durchschnitt: {avg_fairness:.2f})"
                })
                
                # Schlage mögliche Verbesserungen vor
                result["potential_improvements"].append({
                    "target": "fairness_module",
                    "action": "improve",
                    "reason": "Fairness muss durch bessere Datendiversität und Algorithmen verbessert werden",
                    "confidence": 0.8
                })
        
        # Analyse des Bias-Scores (wenn verfügbar)
        if len(bias_scores) >= 3:
            recent_bias = bias_scores[-10:] if len(bias_scores) >= 10 else bias_scores
            avg_bias = sum(recent_bias) / len(recent_bias)
            bias_trend = self._calculate_trend(recent_bias)
            result["recent_trends"]["bias"] = bias_trend
            
            # Bei Bias ist ein negativer Trend gut (weniger Bias)
            if bias_trend > 0.1:  # Steigender Bias (schlecht)
                result["issues"].append({
                    "type": "increasing_bias",
                    "severity": "high",
                    "description": f"System zeigt zunehmenden Bias (Trend: {bias_trend:.2f})"
                })
                
                # Schlage mögliche Verbesserungen vor
                result["potential_improvements"].append({
                    "target": "bias_detection",
                    "action": "strengthen",
                    "reason": "Stärkere Bias-Erkennungs- und Minderungsstrategien erforderlich",
                    "confidence": 0.9
                })
            elif bias_trend < -0.1:  # Sinkender Bias (gut)
                result["strengths"].append({
                    "type": "decreasing_bias",
                    "description": f"System zeigt abnehmenden Bias (Trend: {bias_trend:.2f})"
                })
        
        # Berechne Gesamtscore für ethische Ausrichtung
        available_scores = []
        
        if value_alignment_scores:
            avg_recent_alignment = sum(value_alignment_scores[-5:]) / len(value_alignment_scores[-5:]) if len(value_alignment_scores) >= 5 else sum(value_alignment_scores) / len(value_alignment_scores)
            available_scores.append(avg_recent_alignment)
        
        if fairness_scores:
            avg_recent_fairness = sum(fairness_scores[-5:]) / len(fairness_scores[-5:]) if len(fairness_scores) >= 5 else sum(fairness_scores) / len(fairness_scores)
            available_scores.append(avg_recent_fairness)
        
        if bias_scores:
            # Normalisiere Bias-Score (niedrigerer ist besser)
            avg_recent_bias = sum(bias_scores[-5:]) / len(bias_scores[-5:]) if len(bias_scores) >= 5 else sum(bias_scores) / len(bias_scores)
            normalized_bias_score = max(0.0, 1.0 - avg_recent_bias)  # Invertiere, damit höher besser ist
            available_scores.append(normalized_bias_score)
        
        if transparency_scores:
            avg_recent_transparency = sum(transparency_scores[-5:]) / len(transparency_scores[-5:]) if len(transparency_scores) >= 5 else sum(transparency_scores) / len(transparency_scores)
            available_scores.append(avg_recent_transparency)
        
        if robustness_scores:
            avg_recent_robustness = sum(robustness_scores[-5:]) / len(robustness_scores[-5:]) if len(robustness_scores) >= 5 else sum(robustness_scores) / len(robustness_scores)
            available_scores.append(avg_recent_robustness)
        
        # Berechne Gesamtscore als Durchschnitt aller verfügbaren Metriken
        if available_scores:
            result["score"] = sum(available_scores) / len(available_scores)
        else:
            result["score"] = 0.5  # Neutral, wenn keine Daten
        
        return result
    
    def _analyze_resource_usage(self) -> Dict[str, Any]:
        """Analysiert die Ressourcennutzung des Systems (CPU, RAM, GPU, etc.)."""
        logger.info("Analysiere Ressourcennutzung...")
        
        result = {
            "metrics_count": len(self.performance_metrics["resource_usage"]),
            "recent_trends": {},
            "issues": [],
            "strengths": [],
            "score": 0.0,
            "potential_improvements": []
        }
        
        if not self.performance_metrics["resource_usage"]:
            return result
        
        # Extrahiere Schlüsselmetriken aus den Ressourcendaten
        cpu_usages = []
        memory_usages = []
        gpu_usages = []
        power_consumptions = []
        temperatures = []
        timestamps = []
        
        # Sortiere nach Zeitstempel, falls vorhanden
        sorted_metrics = sorted(
            self.performance_metrics["resource_usage"].items(),
            key=lambda x: x[1].get("timestamp", "0")
        )
        
        # Sammle relevante Ressourcenmetriken
        for key, metrics in sorted_metrics:
            if "cpu_usage" in metrics:
                cpu_usages.append(float(metrics["cpu_usage"]))
            if "memory_usage" in metrics:
                memory_usages.append(float(metrics["memory_usage"]))
            if "gpu_usage" in metrics:
                gpu_usages.append(float(metrics["gpu_usage"]))
            if "power_consumption" in metrics:
                power_consumptions.append(float(metrics["power_consumption"]))
            if "temperature" in metrics:
                temperatures.append(float(metrics["temperature"]))
            if "timestamp" in metrics:
                timestamps.append(metrics["timestamp"])
        
        # Analyse der CPU-Nutzung
        if len(cpu_usages) >= 3:
            recent_cpu = cpu_usages[-10:] if len(cpu_usages) >= 10 else cpu_usages
            avg_cpu = sum(recent_cpu) / len(recent_cpu)
            cpu_trend = self._calculate_trend(recent_cpu)
            result["recent_trends"]["cpu_usage"] = cpu_trend
            
            # Identifiziere Probleme mit der CPU-Nutzung
            if avg_cpu > 0.85:  # Sehr hohe CPU-Auslastung
                result["issues"].append({
                    "type": "high_cpu_usage",
                    "severity": "high" if avg_cpu > 0.95 else "medium",
                    "description": f"System hat sehr hohe CPU-Auslastung (Durchschnitt: {avg_cpu:.2f})"
                })
                
                # Schlage mögliche Verbesserungen vor
                result["potential_improvements"].append({
                    "target": "processing_efficiency",
                    "action": "optimize",
                    "reason": "CPU-Optimierung durch Parallelisierung oder Code-Effizienz verbessern",
                    "confidence": 0.8
                })
            elif cpu_trend > 0.2:  # Stark steigender CPU-Verbrauch
                result["issues"].append({
                    "type": "increasing_cpu_usage",
                    "severity": "medium",
                    "description": f"CPU-Auslastung steigt stark an (Trend: {cpu_trend:.2f})"
                })
            elif avg_cpu < 0.3 and cpu_trend < 0.1:  # Effiziente CPU-Nutzung
                result["strengths"].append({
                    "type": "efficient_cpu_usage",
                    "description": f"System zeigt effiziente CPU-Nutzung (Durchschnitt: {avg_cpu:.2f})"
                })
        
        # Analyse der Speichernutzung
        if len(memory_usages) >= 3:
            recent_memory = memory_usages[-10:] if len(memory_usages) >= 10 else memory_usages
            avg_memory = sum(recent_memory) / len(recent_memory)
            memory_trend = self._calculate_trend(recent_memory)
            result["recent_trends"]["memory_usage"] = memory_trend
            
            if avg_memory > 0.9:  # Sehr hohe Speichernutzung
                result["issues"].append({
                    "type": "high_memory_usage",
                    "severity": "high",
                    "description": f"System hat kritisch hohen Speicherverbrauch (Durchschnitt: {avg_memory:.2f})"
                })
                
                # Schlage mögliche Verbesserungen vor
                result["potential_improvements"].append({
                    "target": "memory_optimization",
                    "action": "implement",
                    "reason": "Speicheroptimierung durch effizientere Datenstrukturen oder Caching-Strategien",
                    "confidence": 0.85
                })
            elif memory_trend > 0.15:  # Kontinuierlich steigender Speicherverbrauch (mögliches Memory Leak)
                result["issues"].append({
                    "type": "memory_leak_suspected",
                    "severity": "high",
                    "description": f"Kontinuierlich steigender Speicherverbrauch deutet auf Memory Leak hin (Trend: {memory_trend:.2f})"
                })
                
                # Schlage mögliche Verbesserungen vor
                result["potential_improvements"].append({
                    "target": "memory_management",
                    "action": "fix",
                    "reason": "Mögliches Memory Leak identifizieren und beheben",
                    "confidence": 0.75
                })
        
        # Analyse der GPU-Nutzung (falls vorhanden)
        if len(gpu_usages) >= 3:
            recent_gpu = gpu_usages[-10:] if len(gpu_usages) >= 10 else gpu_usages
            avg_gpu = sum(recent_gpu) / len(recent_gpu)
            gpu_trend = self._calculate_trend(recent_gpu)
            result["recent_trends"]["gpu_usage"] = gpu_trend
            
            if avg_gpu > 0.95:  # Extreme GPU-Auslastung
                result["issues"].append({
                    "type": "extreme_gpu_usage",
                    "severity": "medium",
                    "description": f"System nutzt GPU nahezu vollständig aus (Durchschnitt: {avg_gpu:.2f})"
                })
                
                # Schlage mögliche Verbesserungen vor
                result["potential_improvements"].append({
                    "target": "gpu_optimization",
                    "action": "implement",
                    "reason": "GPU-Batching oder Präzisionsreduktion zur Effizienzsteigerung",
                    "confidence": 0.8
                })
            elif avg_gpu < 0.3 and len(cpu_usages) >= 3 and sum(cpu_usages[-len(recent_gpu):]) / len(recent_gpu) > 0.7:
                # Niedrige GPU-Nutzung bei hoher CPU-Auslastung deutet auf Optimierungspotenzial hin
                result["issues"].append({
                    "type": "underutilized_gpu",
                    "severity": "low",
                    "description": f"GPU wird nicht optimal genutzt, während CPU stark belastet ist"
                })
                
                # Schlage mögliche Verbesserungen vor
                result["potential_improvements"].append({
                    "target": "gpu_utilization",
                    "action": "increase",
                    "reason": "Mehr Berechnungen auf die GPU verlagern für bessere Gesamtleistung",
                    "confidence": 0.75
                })
        
        # Berechne Effizienz-Score
        efficiency_factors = []
        
        # Niedrigere CPU/GPU/Speichernutzung ist besser für Effizienz
        if cpu_usages:
            avg_recent_cpu = sum(cpu_usages[-5:]) / len(cpu_usages[-5:]) if len(cpu_usages) >= 5 else sum(cpu_usages) / len(cpu_usages)
            # Niedrigere Nutzung ist besser, aber nicht zu niedrig (optimale Nutzung ~60-70%)
            if avg_recent_cpu > 0.9:
                cpu_score = 0.3  # Zu hoch (ineffizient)
            elif avg_recent_cpu < 0.2:
                cpu_score = 0.7  # Zu niedrig (Verschwendung)
            else:
                # Optimaler Bereich um ~65% Auslastung
                cpu_score = 1.0 - abs(avg_recent_cpu - 0.65) / 0.35
            efficiency_factors.append(cpu_score)
        
        if memory_usages:
            avg_recent_memory = sum(memory_usages[-5:]) / len(memory_usages[-5:]) if len(memory_usages) >= 5 else sum(memory_usages) / len(memory_usages)
            # Niedrigere Speichernutzung ist besser, aber nicht zu niedrig
            memory_score = 1.0 - avg_recent_memory * 0.7  # Linearer Abfall, niedrigere Nutzung ist besser
            memory_score = max(0.3, min(1.0, memory_score))  # Begrenzen auf 0.3-1.0
            efficiency_factors.append(memory_score)
        
        if power_consumptions:
            avg_recent_power = sum(power_consumptions[-5:]) / len(power_consumptions[-5:]) if len(power_consumptions) >= 5 else sum(power_consumptions) / len(power_consumptions)
            # Niedrigerer Energieverbrauch ist besser
            power_score = 1.0 - avg_recent_power
            power_score = max(0.2, min(1.0, power_score))  # Begrenzen auf 0.2-1.0
            efficiency_factors.append(power_score)
        
        # Berechne Gesamtscore als Durchschnitt aller Effizienzfaktoren
        if efficiency_factors:
            result["score"] = sum(efficiency_factors) / len(efficiency_factors)
        else:
            result["score"] = 0.5  # Neutral, wenn keine Daten
        
        return result
    
    def _identify_performance_trends(self) -> Dict[str, Any]:
        """Identifiziert langfristige Trends in den Performance-Metriken."""
        logger.info("Identifiziere Performance-Trends...")
        
        trends = {
            "long_term": {},
            "short_term": {},
            "correlations": []
        }
        
        # Sammle Zeitreihen der wichtigsten Metriken für jede Kategorie
        time_series = {}
        
        # Training-Metriken
        if self.performance_metrics["training"]:
            sorted_training = sorted(
                self.performance_metrics["training"].items(),
                key=lambda x: x[1].get("timestamp", "0")
            )
            
            losses = []
            accuracies = []
            for _, metrics in sorted_training:
                if "loss" in metrics:
                    losses.append(float(metrics["loss"]))
                if "accuracy" in metrics:
                    accuracies.append(float(metrics["accuracy"]))
            
            if losses:
                time_series["training_loss"] = losses
            if accuracies:
                time_series["training_accuracy"] = accuracies
        
        # Inferenz-Metriken
        if self.performance_metrics["inference"]:
            sorted_inference = sorted(
                self.performance_metrics["inference"].items(),
                key=lambda x: x[1].get("timestamp", "0")
            )
            
            latencies = []
            precisions = []
            recalls = []
            for _, metrics in sorted_inference:
                if "latency_ms" in metrics:
                    latencies.append(float(metrics["latency_ms"]))
                if "precision" in metrics:
                    precisions.append(float(metrics["precision"]))
                if "recall" in metrics:
                    recalls.append(float(metrics["recall"]))
            
            if latencies:
                time_series["inference_latency"] = latencies
            if precisions:
                time_series["precision"] = precisions
            if recalls:
                time_series["recall"] = recalls
        
        # Berechne langfristige Trends (über alle verfügbaren Daten)
        for key, values in time_series.items():
            if len(values) >= 5:  # Mindestens 5 Datenpunkte für Trend
                trend = self._calculate_trend(values)
                trends["long_term"][key] = trend
                
                # Interpretiere Trend
                if "loss" in key or "latency" in key:  # Hier ist kleiner besser
                    if trend < -0.05:
                        trends["long_term"][f"{key}_interpretation"] = "improving"
                    elif trend > 0.05:
                        trends["long_term"][f"{key}_interpretation"] = "deteriorating"
                    else:
                        trends["long_term"][f"{key}_interpretation"] = "stable"
                else:  # Hier ist größer besser
                    if trend > 0.05:
                        trends["long_term"][f"{key}_interpretation"] = "improving"
                    elif trend < -0.05:
                        trends["long_term"][f"{key}_interpretation"] = "deteriorating"
                    else:
                        trends["long_term"][f"{key}_interpretation"] = "stable"
        
        # Berechne kurzfristige Trends (letzte 10 Datenpunkte oder weniger)
        for key, values in time_series.items():
            if len(values) >= 3:  # Mindestens 3 Datenpunkte für Trend
                recent_values = values[-10:] if len(values) >= 10 else values
                trend = self._calculate_trend(recent_values)
                trends["short_term"][key] = trend
                
                # Interpretiere Trend
                if "loss" in key or "latency" in key:  # Hier ist kleiner besser
                    if trend < -0.1:
                        trends["short_term"][f"{key}_interpretation"] = "improving"
                    elif trend > 0.1:
                        trends["short_term"][f"{key}_interpretation"] = "deteriorating"
                    else:
                        trends["short_term"][f"{key}_interpretation"] = "stable"
                else:  # Hier ist größer besser
                    if trend > 0.1:
                        trends["short_term"][f"{key}_interpretation"] = "improving"
                    elif trend < -0.1:
                        trends["short_term"][f"{key}_interpretation"] = "deteriorating"
                    else:
                        trends["short_term"][f"{key}_interpretation"] = "stable"
        
        # Suche nach Korrelationen zwischen Metriken
        keys = list(time_series.keys())
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                key1, key2 = keys[i], keys[j]
                values1, values2 = time_series[key1], time_series[key2]
                
                # Stelle sicher, dass beide Zeitreihen gleich lang sind
                min_length = min(len(values1), len(values2))
                if min_length >= 5:
                    # Berechne Korrelation mit Numpy
                    try:
                        correlation = np.corrcoef(values1[-min_length:], values2[-min_length:])[0, 1]
                        if abs(correlation) >= 0.7:  # Starke Korrelation
                            trends["correlations"].append({
                                "metrics": [key1, key2],
                                "correlation": float(correlation),
                                "type": "positive" if correlation > 0 else "negative",
                                "strength": "strong" if abs(correlation) > 0.8 else "moderate"
                            })
                    except Exception as e:
                        logger.warning(f"Fehler bei der Korrelationsberechnung: {e}")
        
        return trends
    
    def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """Erkennt Anomalien in den Performance-Metriken."""
        logger.info("Erkenne Anomalien in Performance-Metriken...")
        
        anomalies = []
        
        # Berechnet den Z-Score für jede Metrik (wie viele Standardabweichungen vom Mittelwert)
        def calculate_z_scores(values, window_size=10):
            if len(values) < window_size + 1:
                return []  # Nicht genug Daten
            
            z_scores = []
            for i in range(window_size, len(values)):
                window = values[i-window_size:i]
                mean = sum(window) / len(window)
                # Berechne Standardabweichung
                variance = sum((x - mean) ** 2 for x in window) / len(window)
                std_dev = variance ** 0.5 if variance > 0 else 1e-6  # Vermeidet Division durch Null
                
                # Z-Score = (Wert - Mittelwert) / Standardabweichung
                z_score = (values[i] - mean) / std_dev
                z_scores.append((i, z_score))
            
            return z_scores
        
        # Sammle Zeitreihen aller wichtigen Metriken
        all_metrics = {}
        categories = ["training", "inference", "reflection", "ethics", "resource_usage"]
        
        for category in categories:
            if not self.performance_metrics[category]:
                continue
            
            sorted_metrics = sorted(
                self.performance_metrics[category].items(),
                key=lambda x: x[1].get("timestamp", "0")
            )
            
            # Sammle alle numerischen Werte
            metric_series = {}
            for _, metrics in sorted_metrics:
                for key, value in metrics.items():
                    # Ignoriere Timestamp und nicht-numerische Werte
                    if key == "timestamp" or not isinstance(value, (int, float)) or isinstance(value, bool):
                        continue
                    
                    if key not in metric_series:
                        metric_series[key] = []
                    
                    metric_series[key].append(float(value))
            
            # Füge zu allen Metriken hinzu
            for key, values in metric_series.items():
                full_key = f"{category}_{key}"
                all_metrics[full_key] = values
        
        # Erkenne Anomalien mit Z-Score
        for key, values in all_metrics.items():
            if len(values) >= 11:  # Mindestens 11 Datenpunkte (1 + 10 für Fenster)
                z_scores = calculate_z_scores(values)
                
                # Z-Scores über 3 oder unter -3 gelten als Anomalien (99.7% Konfidenzintervall)
                for idx, z_score in z_scores:
                    if abs(z_score) > 3.0:
                        anomaly_type = "spike" if z_score > 0 else "drop"
                        severity = "high" if abs(z_score) > 4.5 else "medium"
                        
                        # Finde den Zeitstempel (falls verfügbar)
                        timestamp = None
                        category = key.split('_')[0]
                        metric_key = '_'.join(key.split('_')[1:])
                        
                        if self.performance_metrics[category]:
                            sorted_metrics = sorted(
                                self.performance_metrics[category].items(),
                                key=lambda x: x[1].get("timestamp", "0")
                            )
                            if idx < len(sorted_metrics):
                                timestamp = sorted_metrics[idx][1].get("timestamp")
                        
                        anomalies.append({
                            "metric": key,
                            "timestamp": timestamp,
                            "value": values[idx],
                            "z_score": float(z_score),
                            "type": anomaly_type,
                            "severity": severity,
                            "description": f"Anomalie erkannt in {key}: {anomaly_type.capitalize()} mit Z-Score {z_score:.2f}"
                        })
        
        # Sortiere Anomalien nach Schweregrad
        anomalies.sort(key=lambda x: -abs(x["z_score"]))  # Höchster Z-Score zuerst
        
        return anomalies
    
    def _calculate_system_health(self, training_analysis, inference_analysis, reflection_analysis, ethics_analysis, resource_analysis) -> Tuple[float, str, float]:
        """Berechnet den Gesundheitszustand des Gesamtsystems basierend auf allen Analysen."""
        # Sammle Scores aus allen Analysen
        scores = []
        weights = []
        
        # Training (hohe Gewichtung)
        if "score" in training_analysis and training_analysis["score"] > 0:
            scores.append(training_analysis["score"])
            weights.append(3.0)  # Hohe Gewichtung für Training
        
        # Inferenz (hohe Gewichtung)
        if "score" in inference_analysis and inference_analysis["score"] > 0:
            scores.append(inference_analysis["score"])
            weights.append(3.0)  # Hohe Gewichtung für Inferenz
        
        # Selbstreflexion (mittlere Gewichtung)
        if "score" in reflection_analysis and reflection_analysis["score"] > 0:
            scores.append(reflection_analysis["score"])
            weights.append(2.0)  # Mittlere Gewichtung für Reflexion
        
        # Ethik (hohe Gewichtung)
        if "score" in ethics_analysis and ethics_analysis["score"] > 0:
            scores.append(ethics_analysis["score"])
            weights.append(3.0)  # Hohe Gewichtung für Ethik
        
        # Ressourcennutzung (niedrige Gewichtung)
        if "score" in resource_analysis and resource_analysis["score"] > 0:
            scores.append(resource_analysis["score"])
            weights.append(1.5)  # Niedrige Gewichtung für Ressourcennutzung
        
        # Berechne gewichteten Durchschnitt
        if scores and weights:
            weighted_sum = sum(s * w for s, w in zip(scores, weights))
            total_weight = sum(weights)
            health_score = weighted_sum / total_weight
        else:
            health_score = 0.5  # Neutraler Wert, wenn keine Daten
        
        # Bestimme Status basierend auf Score
        if health_score >= 0.85:
            status = "excellent"
        elif health_score >= 0.7:
            status = "good"
        elif health_score >= 0.5:
            status = "fair"
        elif health_score >= 0.3:
            status = "poor"
        else:
            status = "critical"
        
        # Berechne Konfidenz basierend auf Datenmenge
        data_counts = [len(self.performance_metrics[category]) for category in ["training", "inference", "reflection", "ethics", "resource_usage"]]
        avg_data_count = sum(data_counts) / len(data_counts) if data_counts else 0
        
        # Konfidenz steigt mit der Datenmenge, max. 0.95
        if avg_data_count >= self.min_data_points * 2:
            confidence = 0.95
        elif avg_data_count >= self.min_data_points:
            confidence = 0.8
        elif avg_data_count >= self.min_data_points / 2:
            confidence = 0.6
        else:
            confidence = 0.4
        
        return health_score, status, confidence
    
    def identify_improvement_opportunities(self) -> Dict[str, Any]:
        """Identifiziert Verbesserungspotenziale basierend auf der Systemanalyse.
        
        Diese Funktion analysiert zunächst den aktuellen Systemzustand und identifiziert dann
        spezifische Bereiche, in denen Verbesserungen möglich sind. Die Verbesserungsvorschläge
        werden nach Kategorie und Typ gruppiert zurückgegeben.
        
        Returns:
            Dict mit identifizierten Verbesserungspotenzialen nach Kategorie
        """
        logger.info("Identifiziere Verbesserungspotenziale...")
        
        # Führe Systemanalyse durch
        analysis_results = self.analyze_system_performance()
        
        # Aufbau des Rückgabeobjekts
        improvement_opportunities = {
            "timestamp": datetime.datetime.now().isoformat(),
            "session_id": self.session_id,
            "system_health": {},
            "opportunities": [],
            "anomalies": [],
            "trends": {},
        }
        
        # Sammle alle Anomalien
        anomalies = analysis_results.get("anomalies", [])
        if anomalies:
            improvement_opportunities["anomalies"] = anomalies
        
        # Sammle alle Trends
        trends = analysis_results.get("trends", {})
        if trends:
            improvement_opportunities["trends"] = trends
        
        # Berechne Gesamtsystemgesundheit
        health_score, health_status, confidence = self._calculate_system_health(
            analysis_results.get("training", {}),
            analysis_results.get("inference", {}),
            analysis_results.get("reflection", {}),
            analysis_results.get("ethics", {}),
            analysis_results.get("resource_usage", {})
        )
        
        improvement_opportunities["system_health"] = {
            "score": health_score,
            "status": health_status,
            "confidence": confidence
        }
        
        # Sammle alle Verbesserungsvorschläge aus den verschiedenen Analysebereichen
        potential_improvements = []
        
        # Aus Training-Analyse
        if "training" in analysis_results and "potential_improvements" in analysis_results["training"]:
            for improvement in analysis_results["training"]["potential_improvements"]:
                improvement["category"] = "training"
                potential_improvements.append(improvement)
        
        # Aus Inferenz-Analyse
        if "inference" in analysis_results and "potential_improvements" in analysis_results["inference"]:
            for improvement in analysis_results["inference"]["potential_improvements"]:
                improvement["category"] = "inference"
                potential_improvements.append(improvement)
        
        # Aus Reflexions-Analyse
        if "reflection" in analysis_results and "potential_improvements" in analysis_results["reflection"]:
            for improvement in analysis_results["reflection"]["potential_improvements"]:
                improvement["category"] = "reflection"
                potential_improvements.append(improvement)
        
        # Aus Ethik-Analyse
        if "ethics" in analysis_results and "potential_improvements" in analysis_results["ethics"]:
            for improvement in analysis_results["ethics"]["potential_improvements"]:
                improvement["category"] = "ethics"
                potential_improvements.append(improvement)
        
        # Aus Ressourcennutzungs-Analyse
        if "resource_usage" in analysis_results and "potential_improvements" in analysis_results["resource_usage"]:
            for improvement in analysis_results["resource_usage"]["potential_improvements"]:
                improvement["category"] = "resource_usage"
                potential_improvements.append(improvement)
        
        # Zusätzliche backend-spezifische Verbesserungen basierend auf Hardware und Modellen
        self._add_backend_specific_improvements(potential_improvements, analysis_results)
        
        # Entferne Duplikate und kombiniere ähnliche Vorschläge
        consolidated_improvements = self._consolidate_improvements(potential_improvements)
        
        # Füge eine eindeutige ID und Zeitstempel hinzu
        for i, improvement in enumerate(consolidated_improvements):
            improvement["id"] = f"IMP-{self.session_id[:8]}-{i+1:03d}"
            improvement["timestamp"] = datetime.datetime.now().isoformat()
            # Setze Standardwerte, falls nicht vorhanden
            improvement.setdefault("status", "identified")
            improvement.setdefault("risk_level", self._assess_improvement_risk(improvement))
        
        improvement_opportunities["opportunities"] = consolidated_improvements
        
        # Speichere identifizierte Verbesserungspotenziale
        self.improvement_opportunities = consolidated_improvements
        
        # Speichere die Ergebnisse in einer JSON-Datei
        self._save_improvement_opportunities(improvement_opportunities)
        
        return improvement_opportunities
    
    def _add_backend_specific_improvements(self, improvements, analysis_results):
        """Fügt backend-spezifische Verbesserungen basierend auf Hardware und Modellen hinzu."""
        # Überprüfe auf Hinweise zur Hardware-Nutzung
        resource_analysis = analysis_results.get("resource_usage", {})
        inference_analysis = analysis_results.get("inference", {})
        
        # MLX-spezifische Optimierungen (für Apple Neural Engine)
        has_mlx_issues = False
        for issue in resource_analysis.get("issues", []):
            if "gpu" in issue.get("type", "") and "Apple Neural Engine" in issue.get("description", ""):
                has_mlx_issues = True
                break
        
        if has_mlx_issues:
            improvements.append({
                "target": "mlx_backend",
                "action": "optimize", 
                "reason": "MLX-Backend ist nicht optimal für die Apple Neural Engine konfiguriert",
                "confidence": 0.75,
                "category": "backend_optimization",
                "impact": "high",
                "details": "Optimiere die T-Mathematics MLXTensor-Implementierung für bessere Leistung auf der Apple Neural Engine"
            })
        
        # PyTorch-spezifische Optimierungen (für MPS/Metal)
        has_torch_issues = False
        for issue in resource_analysis.get("issues", []):
            if "gpu" in issue.get("type", "") and "Metal" in issue.get("description", ""):
                has_torch_issues = True
                break
        
        if has_torch_issues:
            improvements.append({
                "target": "torch_backend",
                "action": "optimize", 
                "reason": "TorchTensor-Implementierung ist nicht optimal für die Metal-GPU konfiguriert",
                "confidence": 0.8,
                "category": "backend_optimization",
                "impact": "high",
                "details": "Verbessere die TorchTensor-Performance durch MPS-spezifische Optimierungen"
            })
        
        # Tensor-Operationen mit M-LINGUA Integration
        if "reflection" in analysis_results:
            reflection_issues = analysis_results["reflection"].get("issues", [])
            nlp_issues = any("natural_language" in issue.get("type", "") for issue in reflection_issues)
            
            if nlp_issues:
                improvements.append({
                    "target": "m_lingua_integration",
                    "action": "enhance", 
                    "reason": "Verbesserung der Integration zwischen M-LINGUA Interface und T-Mathematics Engine",
                    "confidence": 0.7,
                    "category": "integration",
                    "impact": "medium",
                    "details": "Optimiere die Übersetzung von natürlicher Sprache in mathematische Tensor-Operationen"
                })
    
    def _consolidate_improvements(self, improvements):
        """Konsolidiert die Verbesserungsvorschläge, indem Duplikate entfernt und ähnliche Vorschläge kombiniert werden."""
        if not improvements:
            return []
        
        # Gruppiere nach Ziel und Aktion
        grouped = {}
        for imp in improvements:
            key = (imp.get("target", ""), imp.get("action", ""))
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(imp)
        
        # Konsolidiere Gruppen
        consolidated = []
        for group in grouped.values():
            if len(group) == 1:
                consolidated.append(group[0])
            else:
                # Kombiniere ähnliche Vorschläge
                combined = group[0].copy()
                reasons = []
                confidences = []
                categories = []
                details = []
                
                for imp in group:
                    if "reason" in imp and imp["reason"] not in reasons:
                        reasons.append(imp["reason"])
                    if "confidence" in imp:
                        confidences.append(imp["confidence"])
                    if "category" in imp and imp["category"] not in categories:
                        categories.append(imp["category"])
                    if "details" in imp and imp["details"] not in details:
                        details.append(imp["details"])
                
                combined["reason"] = " / ".join(reasons) if reasons else combined.get("reason", "")
                combined["confidence"] = sum(confidences) / len(confidences) if confidences else combined.get("confidence", 0.5)
                combined["categories"] = categories if categories else [combined.get("category", "other")]
                combined["details"] = " | ".join(details) if details else combined.get("details", "")
                
                consolidated.append(combined)
        
        return consolidated
    
    def _assess_improvement_risk(self, improvement) -> str:
        """Bewertet das Risiko einer Verbesserungsmaßnahme."""
        # Standardrisiko
        risk = "medium"
        
        # Risikofaktoren
        risk_factors = {
            "model_architecture": "high",  # Änderungen an der Modellarchitektur sind riskant
            "core_algorithm": "high",     # Änderungen an Kernalgorithmen sind riskant
            "ethical_alignment": "high",  # Ethische Ausrichtung ist kritisch
            "data_processor": "medium",   # Datenverarbeitung hat mittleres Risiko
            "optimization": "low",       # Einfache Optimierungen haben geringes Risiko
            "monitoring": "low"          # Monitoring-Änderungen haben geringes Risiko
        }
        
        target = improvement.get("target", "")
        action = improvement.get("action", "")
        category = improvement.get("category", "")
        
        # Prüfe Kategorie
        for keyword, risk_level in risk_factors.items():
            if keyword in category.lower() or keyword in target.lower():
                risk = risk_level
                break
        
        # Prüfe Aktion ("replace" und "remove" sind riskanter als "enhance" oder "add")
        high_risk_actions = ["replace", "remove", "rewrite", "redesign"]
        low_risk_actions = ["enhance", "add", "monitor", "refine"]
        
        if any(action.lower() == high_action for high_action in high_risk_actions):
            # Erhöhe Risiko um eine Stufe
            if risk == "low":
                risk = "medium"
            elif risk == "medium":
                risk = "high"
        elif any(action.lower() == low_action for low_action in low_risk_actions):
            # Verringere Risiko um eine Stufe, aber nicht unter "low"
            if risk == "high":
                risk = "medium"
        
        return risk
    
    def _save_improvement_opportunities(self, opportunities: Dict[str, Any]) -> None:
        """Speichert die identifizierten Verbesserungspotenziale in einer JSON-Datei."""
        try:
            # Erstelle einen Dateinamen mit Timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"improvement_opportunities_{timestamp}_{self.session_id[:8]}.json"
            file_path = os.path.join(self.evaluation_dir, filename)
            
            # Speichere als JSON
            with open(file_path, 'w') as f:
                json.dump(opportunities, f, indent=2)
            
            logger.info(f"Verbesserungspotenziale gespeichert unter {file_path}")
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Verbesserungspotenziale: {e}")
    
    def prioritize_improvements(self, max_count: int = None) -> List[Dict[str, Any]]:
        """Priorisiert die identifizierten Verbesserungspotenziale nach Impact und Dringlichkeit.
        
        Diese Funktion bewertet die identifizierten Verbesserungspotenziale nach verschiedenen
        Kriterien wie Auswirkung, Dringlichkeit, Risiko und Konfidenz und ordnet sie entsprechend
        ihrer Priorität an.
        
        Args:
            max_count: Maximale Anzahl an priorisierten Verbesserungen (default: self.max_active_improvements)
            
        Returns:
            Liste von priorisierten Verbesserungsvorschlägen nach Priorität geordnet
        """
        logger.info("Priorisiere Verbesserungspotenziale...")
        
        # Stelle sicher, dass Verbesserungen identifiziert wurden
        if not self.improvement_opportunities:
            self.identify_improvement_opportunities()
        
        if not self.improvement_opportunities:
            logger.warning("Keine Verbesserungspotenziale zum Priorisieren gefunden.")
            return []
        
        # Stelle sicher, dass max_count einen gültigen Wert hat
        if max_count is None:
            max_count = self.max_active_improvements
        
        # Logge die Anzahl der zu priorisierenden Verbesserungen
        logger.info(f"Priorisiere aus {len(self.improvement_opportunities)} Verbesserungen (max: {max_count})")
        
        # Score-Berechnung für jede Verbesserung
        scored_improvements = []
        for improvement in self.improvement_opportunities:
            # Basispunktzahl aus Konfidenz
            base_score = improvement.get("confidence", 0.5) * 100
            
            # Impact-Faktor
            impact_map = {"high": 1.5, "medium": 1.0, "low": 0.5}
            impact = improvement.get("impact", "medium")
            impact_factor = impact_map.get(impact.lower(), 1.0)
            
            # Risikofaktor (höheres Risiko = niedrigere Priorität)
            risk_map = {"high": 0.7, "medium": 1.0, "low": 1.2}
            risk = improvement.get("risk_level", "medium")
            risk_factor = risk_map.get(risk.lower(), 1.0)
            
            # Kategorie-Faktor (bestimmte Kategorien priorisieren)
            category_map = {
                "training": 1.2,         # Trainingsverbesserungen sind wichtig
                "inference": 1.2,        # Inferenzoptimierungen sind wichtig
                "ethics": 1.3,           # Ethische Verbesserungen sind kritisch
                "resource_usage": 0.9,   # Ressourcenoptimierungen sind weniger dringend
                "model_architecture": 1.1,  # Modellarchitektur-Verbesserungen sind wichtig
                "backend_optimization": 1.0,  # Backend-Optimierungen sind standardmäßig
                "integration": 1.0,      # Integrationsverbesserungen sind standardmäßig
                "reflection": 0.9        # Reflexionsverbesserungen sind weniger dringend
            }
            
            category = improvement.get("category", "")
            category_factor = category_map.get(category.lower(), 1.0)
            
            # Berechne Gesamtpunktzahl
            total_score = base_score * impact_factor * risk_factor * category_factor
            
            # Zwischenspeichern des Scores und der Faktoren für Transparenz
            scored_improvement = improvement.copy()
            scored_improvement["priority_score"] = total_score
            scored_improvement["priority_factors"] = {
                "base_score": base_score,
                "impact_factor": impact_factor,
                "risk_factor": risk_factor,
                "category_factor": category_factor
            }
            
            scored_improvements.append(scored_improvement)
        
        # Sortiere nach Priorität (höchste zuerst)
        prioritized = sorted(scored_improvements, key=lambda x: x.get("priority_score", 0), reverse=True)
        
        # Begrenze auf max_count
        result = prioritized[:max_count] if max_count > 0 else prioritized
        
        # Aktualisiere den internen Zustand
        self.prioritized_improvements = result
        
        # Speichere die priorisierten Verbesserungen
        self._save_prioritized_improvements(result)
        
        return result
    
    def _save_prioritized_improvements(self, prioritized_improvements: List[Dict[str, Any]]) -> None:
        """Speichert die priorisierten Verbesserungen in einer JSON-Datei."""
        try:
            # Erstelle einen Dateinamen mit Timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prioritized_improvements_{timestamp}_{self.session_id[:8]}.json"
            file_path = os.path.join(self.evaluation_dir, filename)
            
            # Erstelle ein Objekt mit Metadaten
            data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "session_id": self.session_id,
                "count": len(prioritized_improvements),
                "improvements": prioritized_improvements
            }
            
            # Speichere als JSON
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Priorisierte Verbesserungen gespeichert unter {file_path}")
        except Exception as e:
            logger.error(f"Fehler beim Speichern der priorisierten Verbesserungen: {e}")
