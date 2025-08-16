#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate AGI - Ethics Integration

Dieses Modul integriert die ethischen Komponenten (BiasDetector, EthicsFramework,
ValueAligner) in das MISO Ultimate AGI System. Es implementiert Adapter und
Hooks zur Echtzeit-Analyse von Trainingsdaten und Modellausgaben.

Autor: MISO ULTIMATE AGI Team
Datum: 26.04.2025
"""

import os
import json
import logging
import datetime
import threading
import queue
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable

from miso.ethics.BiasDetector import BiasDetector
from miso.ethics.EthicsFramework import EthicsFramework
from miso.ethics.ValueAligner import ValueAligner

# Konfiguration für Logging
log_dir = Path("/Volumes/My Book/MISO_Ultimate 15.32.28/logs/ethics")
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("miso.ethics.integration")
logger.setLevel(logging.INFO)

# Datei-Handler für strukturierte Logs
file_handler = logging.FileHandler(log_dir / f"ethics_integration_log_{datetime.datetime.now().strftime('%Y%m%d')}.json")
logger.addHandler(file_handler)

# Konsolen-Handler für Entwicklungszwecke
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)


class EthicsSystem:
    """
    Hauptklasse zur Integration ethischer Komponenten in das MISO Ultimate AGI System.
    
    Diese Klasse dient als zentraler Koordinator für die ethischen Komponenten und
    bietet eine einheitliche Schnittstelle für die Integration in Trainings- und
    Antwortprozesse.
    """
    
    def __init__(self, 
                config_path: Optional[str] = None,
                async_processing: bool = True,
                block_unethical_outputs: bool = True):
        """
        Initialisiert das Ethics System.
        
        Args:
            config_path: Optional, Pfad zur Konfigurationsdatei.
            async_processing: Ob die ethische Verarbeitung asynchron erfolgen soll.
            block_unethical_outputs: Ob unethische Ausgaben blockiert werden sollen.
        """
        self.config = self._load_config(config_path)
        self.async_processing = async_processing
        self.block_unethical_outputs = block_unethical_outputs
        
        # Initialisiere ethische Komponenten
        self.bias_detector = BiasDetector()
        self.ethics_framework = EthicsFramework()
        self.value_aligner = ValueAligner()
        
        # Queue für asynchrone Verarbeitung
        self.ethics_queue = queue.Queue()
        self.worker_thread = None
        
        # Statistiken und Metriken
        self.integration_stats = {
            "processed_training_batches": 0,
            "processed_outputs": 0,
            "blocked_outputs": 0,
            "modified_outputs": 0,
            "detected_biases": 0,
            "ethical_violations": 0,
            "value_alignments": 0
        }
        
        # Starte Worker-Thread, wenn async aktiviert ist
        if self.async_processing:
            self._start_worker()
        
        logger.info(f"Ethics System initialisiert, async={self.async_processing}, "
                   f"block_unethical={self.block_unethical_outputs}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Lädt die Konfiguration aus einer JSON-Datei.
        
        Args:
            config_path: Pfad zur Konfigurationsdatei.
            
        Returns:
            Dictionary mit Konfigurationseinstellungen.
        """
        default_config = {
            "bias_detection": {
                "enabled": True,
                "threshold": 0.7,
                "log_all_results": True
            },
            "ethics_framework": {
                "enabled": True,
                "min_compliance_score": 60,
                "log_all_evaluations": True
            },
            "value_alignment": {
                "enabled": True,
                "always_document_rationale": True
            },
            "integration": {
                "block_threshold": 50,  # Compliance-Score unter dem Ausgaben blockiert werden
                "modification_threshold": 70,  # Compliance-Score unter dem Ausgaben modifiziert werden
                "bias_blocking": True,  # Ob Bias zu Blockierung führen soll
                "real_time_analysis": True  # Ob Echtzeit-Analyse aktiviert ist
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    custom_config = json.load(f)
                    
                    # Kombiniere Standard- und benutzerdefinierte Konfiguration
                    def update_recursive(d, u):
                        for k, v in u.items():
                            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                                update_recursive(d[k], v)
                            else:
                                d[k] = v
                    
                    update_recursive(default_config, custom_config)
            except Exception as e:
                logger.error(f"Fehler beim Laden der Konfiguration: {e}")
        
        return default_config
    
    def _start_worker(self) -> None:
        """Startet den Worker-Thread für asynchrone Verarbeitung."""
        if self.worker_thread is not None and self.worker_thread.is_alive():
            return
        
        self.worker_thread = threading.Thread(
            target=self._ethics_worker,
            daemon=True
        )
        self.worker_thread.start()
        logger.info("Asynchroner Ethics-Worker gestartet")
    
    def _ethics_worker(self) -> None:
        """Worker-Thread-Funktion für asynchrone ethische Verarbeitung."""
        while True:
            try:
                # Hole nächstes Item aus der Queue
                item = self.ethics_queue.get()
                if item is None:
                    break
                
                # Entpacke Item
                task_type, data, callback = item
                
                # Führe entsprechende Aufgabe aus
                if task_type == "training_data":
                    self._process_training_data_internal(data, callback)
                elif task_type == "output":
                    self._process_output_internal(data, callback)
                
                # Markiere Task als erledigt
                self.ethics_queue.task_done()
            except Exception as e:
                logger.error(f"Fehler im Ethics-Worker: {e}", exc_info=True)
    
    def stop_worker(self) -> None:
        """Stoppt den Worker-Thread."""
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.ethics_queue.put(None)
            self.worker_thread.join(timeout=2.0)
            logger.info("Ethics-Worker gestoppt")
    
    def process_training_data(self, 
                             data_batch: Any, 
                             callback: Optional[Callable] = None) -> Optional[Dict[str, Any]]:
        """
        Verarbeitet einen Trainingsdaten-Batch ethisch.
        
        Diese Methode analysiert Trainingsdaten auf Bias und ethische Probleme,
        bevor sie im Training verwendet werden.
        
        Args:
            data_batch: Der zu verarbeitende Trainingsdaten-Batch.
            callback: Optional, Callback-Funktion für asynchrone Verarbeitung.
            
        Returns:
            Bei synchroner Verarbeitung ein Dictionary mit Ergebnissen, sonst None.
        """
        if not self.config["bias_detection"]["enabled"]:
            # Wenn Bias-Erkennung deaktiviert ist, gib None zurück
            return None
        
        # Bei asynchroner Ausführung in die Queue einreihen
        if self.async_processing:
            self.ethics_queue.put(("training_data", data_batch, callback))
            return None
        
        # Bei synchroner Ausführung direkt ausführen
        return self._process_training_data_internal(data_batch, callback)
    
    def _process_training_data_internal(self, 
                                      data_batch: Any, 
                                      callback: Optional[Callable]) -> Dict[str, Any]:
        """
        Interne Methode zur ethischen Verarbeitung von Trainingsdaten.
        
        Args:
            data_batch: Der zu verarbeitende Trainingsdaten-Batch.
            callback: Optional, Callback-Funktion für asynchrone Verarbeitung.
            
        Returns:
            Ein Dictionary mit Ergebnissen.
        """
        start_time = datetime.datetime.now()
        self.integration_stats["processed_training_batches"] += 1
        
        # Erstelle Basis-Bericht
        results = {
            "process_id": f"training_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{self.integration_stats['processed_training_batches']}",
            "timestamp": start_time.isoformat(),
            "data_type": "training_data",
            "bias_results": None,
            "is_biased": False,
            "can_proceed": True,
            "recommendations": []
        }
        
        # Führe Bias-Erkennung durch
        bias_results = self.bias_detector.detect_bias_in_data(data_batch)
        results["bias_results"] = bias_results
        
        # Aktualisiere Statistiken und Ergebnisse
        if bias_results["bias_detected"]:
            results["is_biased"] = True
            self.integration_stats["detected_biases"] += 1
            
            # Füge Empfehlungen hinzu
            results["recommendations"].append(
                "Überprüfen Sie die Trainingsdaten auf potenzielle Verzerrungen und "
                "balancieren Sie den Datensatz entsprechend aus."
            )
            
            # Bestimme, ob Training fortgesetzt werden kann
            if self.config["integration"]["bias_blocking"] and self.block_unethical_outputs:
                # Entscheide basierend auf Schweregrad und Typ des Bias
                severe_bias = any(
                    result.get("severity", "medium") == "high" 
                    for bias_type, result in bias_results["detection_results"].items()
                    if result["bias_detected"]
                )
                
                if severe_bias:
                    results["can_proceed"] = False
                    results["recommendations"].append(
                        "Training mit diesen Daten wurde blockiert aufgrund schwerwiegender "
                        "Verzerrungen. Bitte überprüfen und korrigieren Sie die Daten."
                    )
        
        # Berechne Gesamtdauer
        end_time = datetime.datetime.now()
        results["processing_duration_ms"] = (end_time - start_time).total_seconds() * 1000
        
        # Rufe Callback auf, wenn vorhanden
        if callback is not None:
            try:
                callback(results)
            except Exception as e:
                logger.error(f"Fehler beim Aufrufen des Callbacks: {e}")
        
        return results
    
    def process_output(self, 
                      output_data: Any, 
                      input_data: Optional[Any] = None, 
                      context: Optional[Dict[str, Any]] = None,
                      callback: Optional[Callable] = None) -> Optional[Dict[str, Any]]:
        """
        Verarbeitet eine Modellausgabe ethisch.
        
        Diese Methode analysiert Modellausgaben auf Bias, ethische Probleme und
        Wertekonformität, bevor sie an den Benutzer weitergegeben werden.
        
        Args:
            output_data: Die zu verarbeitende Modellausgabe.
            input_data: Optional, die Eingabedaten, die zur Ausgabe geführt haben.
            context: Optional, zusätzlicher Kontext zur Ausgabe.
            callback: Optional, Callback-Funktion für asynchrone Verarbeitung.
            
        Returns:
            Bei synchroner Verarbeitung ein Dictionary mit Ergebnissen, sonst None.
        """
        if not (self.config["bias_detection"]["enabled"] or 
                self.config["ethics_framework"]["enabled"] or 
                self.config["value_alignment"]["enabled"]):
            # Wenn alle ethischen Komponenten deaktiviert sind, gib None zurück
            return None
        
        # Erstelle minimalen Kontext, wenn keiner angegeben wurde
        if context is None:
            context = {}
        
        # Bei asynchroner Ausführung in die Queue einreihen
        if self.async_processing:
            self.ethics_queue.put(("output", (output_data, input_data, context), callback))
            return None
        
        # Bei synchroner Ausführung direkt ausführen
        return self._process_output_internal((output_data, input_data, context), callback)
    
    def _process_output_internal(self, 
                              data: Tuple[Any, Optional[Any], Dict[str, Any]], 
                              callback: Optional[Callable]) -> Dict[str, Any]:
        """
        Interne Methode zur ethischen Verarbeitung von Modellausgaben.
        
        Args:
            data: Tuple aus (output_data, input_data, context).
            callback: Optional, Callback-Funktion für asynchrone Verarbeitung.
            
        Returns:
            Ein Dictionary mit Ergebnissen.
        """
        start_time = datetime.datetime.now()
        self.integration_stats["processed_outputs"] += 1
        
        # Entpacke Daten
        output_data, input_data, context = data
        
        # Erstelle Basis-Bericht
        results = {
            "process_id": f"output_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{self.integration_stats['processed_outputs']}",
            "timestamp": start_time.isoformat(),
            "data_type": "model_output",
            "original_output": output_data,
            "modified_output": output_data,  # Standard: keine Änderung
            "was_modified": False,
            "is_blocked": False,
            "bias_results": None,
            "ethics_results": None,
            "alignment_results": None,
            "can_proceed": True,
            "recommendations": []
        }
        
        # Führe Bias-Erkennung durch, wenn aktiviert
        if self.config["bias_detection"]["enabled"]:
            bias_results = self.bias_detector.detect_bias_in_outputs(output_data)
            results["bias_results"] = bias_results
            
            if bias_results["bias_detected"]:
                self.integration_stats["detected_biases"] += 1
                results["recommendations"].append(
                    "Die Ausgabe enthält potenzielle Verzerrungen. Überprüfen Sie die "
                    "Ausgabe auf problematische Inhalte."
                )
        
        # Führe ethische Bewertung durch, wenn aktiviert
        if self.config["ethics_framework"]["enabled"]:
            # Erstelle Aktionsrepräsentation aus der Ausgabe
            action = {
                "type": "model_output",
                "description": str(output_data) if len(str(output_data)) < 1000 else str(output_data)[:997] + "...",
                "context": context
            }
            
            ethics_results = self.ethics_framework.evaluate_action_against_ethics(action)
            results["ethics_results"] = ethics_results
            
            if not ethics_results["is_compliant"]:
                self.integration_stats["ethical_violations"] += 1
                
                # Füge Empfehlungen hinzu
                for recommendation in ethics_results["recommendations"]:
                    results["recommendations"].append(recommendation)
        
        # Führe Werteanpassung durch, wenn aktiviert
        if self.config["value_alignment"]["enabled"]:
            # Erstelle Entscheidungskontext aus der Ausgabe und den Empfehlungen
            decision_context = {
                "decision": output_data,
                "alternatives": self._generate_alternatives(output_data, context),
                "context": context
            }
            
            alignment_results = self.value_aligner.align_decision_with_values(decision_context)
            results["alignment_results"] = alignment_results
            
            if alignment_results["was_modified"]:
                self.integration_stats["value_alignments"] += 1
                results["modified_output"] = alignment_results["aligned_decision"]
                results["was_modified"] = True
                
                # Füge Begründung hinzu
                if alignment_results["rationale"]:
                    results["recommendations"].append(
                        f"Begründung für Anpassung: {alignment_results['rationale']}")
        
        # Bestimme, ob die Ausgabe blockiert werden soll
        should_block = False
        
        # Blockiere bei ethischen Verstößen, wenn konfiguriert
        if (self.config["ethics_framework"]["enabled"] and 
            results["ethics_results"] and 
            self.block_unethical_outputs):
            
            ethics_score = results["ethics_results"]["compliance_score"]
            if ethics_score < self.config["integration"]["block_threshold"]:
                should_block = True
                results["recommendations"].append(
                    f"Ausgabe blockiert: Ethik-Compliance-Score ({ethics_score}) "
                    f"unter dem Schwellenwert ({self.config['integration']['block_threshold']})."
                )
        
        # Blockiere bei schwerem Bias, wenn konfiguriert
        if (self.config["bias_detection"]["enabled"] and 
            results["bias_results"] and 
            results["bias_results"]["bias_detected"] and
            self.config["integration"]["bias_blocking"] and 
            self.block_unethical_outputs):
            
            # Prüfe auf schwerwiegende Bias-Erkennung
            severe_bias = False
            for bias_type, result in results["bias_results"]["detection_results"].items():
                if (result["bias_detected"] and 
                    result.get("details", {}).get("severity", "medium") == "high"):
                    severe_bias = True
                    break
            
            if severe_bias:
                should_block = True
                results["recommendations"].append(
                    "Ausgabe blockiert: Schwerwiegende Verzerrungen erkannt."
                )
        
        # Aktualisiere Blockierungsstatus
        if should_block:
            results["is_blocked"] = True
            results["can_proceed"] = False
            self.integration_stats["blocked_outputs"] += 1
        
        # Aktualisiere Modifikationsstatus
        if results["was_modified"]:
            self.integration_stats["modified_outputs"] += 1
        
        # Berechne Gesamtdauer
        end_time = datetime.datetime.now()
        results["processing_duration_ms"] = (end_time - start_time).total_seconds() * 1000
        
        # Rufe Callback auf, wenn vorhanden
        if callback is not None:
            try:
                callback(results)
            except Exception as e:
                logger.error(f"Fehler beim Aufrufen des Callbacks: {e}")
        
        return results
    
    def _generate_alternatives(self, output: Any, context: Dict[str, Any]) -> List[Any]:
        """
        Generiert alternative Ausgaben für die Werteanpassung.
        
        In einer vollständigen Implementierung würde hier das Modell mit
        verschiedenen Constraints aufgerufen, um alternative Ausgaben zu erzeugen.
        
        Args:
            output: Die ursprüngliche Ausgabe.
            context: Zusätzlicher Kontext.
            
        Returns:
            Eine Liste alternativer Ausgaben.
        """
        # Einfache Simulation für Demonstrationszwecke
        # In einer vollständigen Implementierung würden hier echte alternative
        # Ausgaben durch das Modell generiert werden
        
        # Konvertiere Ausgabe zu String für einfache Manipulation
        output_str = str(output)
        
        alternatives = []
        
        # Erstelle einfache Varianten durch Modifikation
        if len(output_str) > 10:
            # Alternative 1: Neutralere Version
            alternative1 = f"Neutraler: {output_str}"
            alternatives.append(alternative1)
            
            # Alternative 2: Version mit mehr Haftungsausschluss
            alternative2 = f"{output_str} (Dies ist eine KI-generierte Ausgabe, überprüfen Sie die Informationen.)"
            alternatives.append(alternative2)
            
            # Alternative 3: Kürzere Version
            max_len = min(len(output_str) - 1, int(len(output_str) * 0.7))
            alternative3 = output_str[:max_len] + "..."
            alternatives.append(alternative3)
        
        return alternatives
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Gibt Statistiken über die ethischen Verarbeitungsaktivitäten zurück.
        
        Returns:
            Ein Dictionary mit Statistiken.
        """
        # Kombiniere die Statistiken aller Komponenten
        combined_stats = {
            "integration": self.integration_stats,
            "bias_detector": self.bias_detector.get_statistics(),
            "ethics_framework": self.ethics_framework.get_statistics(),
            "value_aligner": self.value_aligner.get_statistics()
        }
        
        return combined_stats


# Integration mit dem TrainingController
def integrate_with_training_controller(training_controller, ethics_system=None):
    """
    Integriert das Ethics System mit dem TrainingController.
    
    Args:
        training_controller: Eine Instanz des TrainingController.
        ethics_system: Optional, eine bestehende Instanz des EthicsSystem.
            Wenn nicht angegeben, wird eine neue Instanz erstellt.
            
    Returns:
        Die Instanz des EthicsSystem.
    """
    if ethics_system is None:
        ethics_system = EthicsSystem()
    
    # Originale Trainingsfunktion sichern
    original_train_function = training_controller.train
    
    # Wrapper für die Trainingsfunktion erstellen
    def ethics_enhanced_train(*args, **kwargs):
        # Extrahiere Trainingsdaten aus den Argumenten (angepasst je nach API)
        data_batch = kwargs.get('data_batch', None)
        if data_batch is None and args:
            data_batch = args[0]
        
        # Ethische Verarbeitung der Trainingsdaten
        if data_batch is not None:
            ethics_results = ethics_system.process_training_data(data_batch)
            
            # Wenn Bias erkannt und Training blockiert werden sollte
            if ethics_results and not ethics_results["can_proceed"]:
                logger.warning("Training blockiert aufgrund ethischer Bedenken.")
                return {
                    "error": "Ethics violation",
                    "details": ethics_results,
                    "recommendations": ethics_results["recommendations"]
                }
        
        # Original-Trainingsfunktion aufrufen
        return original_train_function(*args, **kwargs)
    
    # Ersetze die Trainingsfunktion
    training_controller.train = ethics_enhanced_train
    
    logger.info("Ethics System wurde mit dem TrainingController integriert")
    return ethics_system


# Integration mit dem LiveReflectionSystem
def integrate_with_reflection_system(reflection_system, ethics_system=None):
    """
    Integriert das Ethics System mit dem LiveReflectionSystem.
    
    Args:
        reflection_system: Eine Instanz des LiveReflectionSystem.
        ethics_system: Optional, eine bestehende Instanz des EthicsSystem.
            Wenn nicht angegeben, wird eine neue Instanz erstellt.
            
    Returns:
        Die Instanz des EthicsSystem.
    """
    if ethics_system is None:
        ethics_system = EthicsSystem()
    
    # Originale Reflexionsfunktion sichern
    original_reflect_function = reflection_system.reflect_on_output
    
    # Wrapper für die Reflexionsfunktion erstellen
    def ethics_enhanced_reflect(input_data, output_data, metadata=None):
        # Erstelle Kontext aus den Metadaten
        context = metadata or {}
        
        # Ethische Verarbeitung der Ausgabe
        ethics_results = ethics_system.process_output(
            output_data, input_data, context)
        
        # Wenn ethische Verarbeitung die Ausgabe modifiziert hat
        if ethics_results and ethics_results["was_modified"]:
            modified_output = ethics_results["modified_output"]
            logger.info("Ausgabe wurde ethisch angepasst.")
            
            # Rufe Original-Reflexionsfunktion mit modifizierter Ausgabe auf
            reflection_results = original_reflect_function(
                input_data, modified_output, metadata)
            
            # Füge ethische Ergebnisse hinzu
            if reflection_results:
                reflection_results["ethics_results"] = ethics_results
            
            return reflection_results
        
        # Wenn ethische Verarbeitung die Ausgabe blockiert hat
        elif ethics_results and ethics_results["is_blocked"]:
            logger.warning("Ausgabe blockiert aufgrund ethischer Bedenken.")
            
            # Erstelle blockierte Reflexionsergebnisse
            blocked_results = {
                "reflection_id": f"blocked_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
                "timestamp": datetime.datetime.now().isoformat(),
                "blocked": True,
                "reason": "Ethics violation",
                "recommendations": ethics_results["recommendations"],
                "ethics_results": ethics_results
            }
            
            return blocked_results
        
        # Sonst rufe Original-Reflexionsfunktion unverändert auf
        return original_reflect_function(input_data, output_data, metadata)
    
    # Ersetze die Reflexionsfunktion
    reflection_system.reflect_on_output = ethics_enhanced_reflect
    
    logger.info("Ethics System wurde mit dem LiveReflectionSystem integriert")
    return ethics_system


# Einfache Testfunktion
def test_ethics_integration():
    """Einfacher Test für die Ethik-Integration."""
    # Erstelle Ethics System
    ethics_system = EthicsSystem(async_processing=False)
    
    # Teste Verarbeitung von Trainingsdaten
    test_data = ["Dies ist ein Testdatensatz für die ethische Verarbeitung."]
    
    print("Teste ethische Verarbeitung von Trainingsdaten...")
    data_result = ethics_system.process_training_data(test_data)
    print(f"Kann fortfahren: {data_result['can_proceed']}")
    print(f"Empfehlungen: {len(data_result['recommendations'])}")
    
    # Teste Verarbeitung von Ausgaben
    test_output = "Dies ist eine Testausgabe für die ethische Verarbeitung."
    test_input = "Wie funktioniert die ethische Verarbeitung?"
    test_context = {"type": "query_response", "sensitivity": "low"}
    
    print("\nTeste ethische Verarbeitung von Ausgaben...")
    output_result = ethics_system.process_output(test_output, test_input, test_context)
    print(f"Ausgabe blockiert: {output_result['is_blocked']}")
    print(f"Ausgabe modifiziert: {output_result['was_modified']}")
    if output_result['was_modified']:
        print(f"Modifizierte Ausgabe: {output_result['modified_output']}")
    print(f"Empfehlungen: {len(output_result['recommendations'])}")
    
    # Teste Statistiken
    print("\nStatistiken:")
    stats = ethics_system.get_statistics()
    print(f"Verarbeitete Trainingsbatches: {stats['integration']['processed_training_batches']}")
    print(f"Verarbeitete Ausgaben: {stats['integration']['processed_outputs']}")
    print(f"Blockierte Ausgaben: {stats['integration']['blocked_outputs']}")
    print(f"Modifizierte Ausgaben: {stats['integration']['modified_outputs']}")
    
    return data_result, output_result, stats


if __name__ == "__main__":
    test_ethics_integration()
