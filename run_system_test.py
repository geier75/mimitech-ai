#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MISO Systemtest

Dieser Test überprüft systematisch alle Komponenten des MISO-Systems
und gibt detaillierte Fehlermeldungen aus.
"""

import os
import sys
import logging
import importlib
import traceback
from datetime import datetime

# Konfiguration des Loggings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("miso_test.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("miso_test")

class CheckpointTester:
    """Testet die verschiedenen Checkpoints des MISO-Systems."""
    
    def __init__(self):
        self.results = {
            "checkpoint1": {"status": "nicht getestet", "details": []},
            "checkpoint2": {"status": "nicht getestet", "details": []},
            "checkpoint3": {"status": "nicht getestet", "details": []},
            "checkpoint4": {"status": "nicht getestet", "details": []},
            "checkpoint5": {"status": "nicht getestet", "details": []}
        }
        self.current_checkpoint = None
    
    def start_checkpoint(self, checkpoint_name):
        """Startet einen neuen Checkpoint-Test."""
        print(f"\n{'='*80}\n{checkpoint_name.upper()}\n{'='*80}")
        self.current_checkpoint = checkpoint_name
        self.results[checkpoint_name]["status"] = "in Bearbeitung"
        self.results[checkpoint_name]["details"] = []
    
    def add_result(self, test_name, success, message=""):
        """Fügt ein Testergebnis hinzu."""
        status = "✅ ERFOLG" if success else "❌ FEHLER"
        detail = {"test": test_name, "success": success, "message": message}
        self.results[self.current_checkpoint]["details"].append(detail)
        
        # Ausgabe formatieren
        print(f"{status} - {test_name}")
        if message:
            print(f"       {message}")
    
    def finish_checkpoint(self):
        """Schließt den aktuellen Checkpoint ab."""
        details = self.results[self.current_checkpoint]["details"]
        success_count = sum(1 for detail in details if detail["success"])
        total_count = len(details)
        
        if success_count == total_count and total_count > 0:
            self.results[self.current_checkpoint]["status"] = "erfolgreich"
            print(f"\n✅ Checkpoint {self.current_checkpoint} ERFOLGREICH: {success_count}/{total_count} Tests bestanden")
        else:
            self.results[self.current_checkpoint]["status"] = "fehlgeschlagen"
            print(f"\n❌ Checkpoint {self.current_checkpoint} FEHLGESCHLAGEN: {success_count}/{total_count} Tests bestanden")
    
    def print_summary(self):
        """Gibt eine Zusammenfassung aller Testergebnisse aus."""
        print("\n" + "="*80)
        print("ZUSAMMENFASSUNG DER TESTERGEBNISSE")
        print("="*80)
        
        all_success = True
        for checkpoint, result in self.results.items():
            status_symbol = "✅" if result["status"] == "erfolgreich" else "❌"
            print(f"{status_symbol} {checkpoint}: {result['status']}")
            
            if result["status"] != "erfolgreich":
                all_success = False
                # Zeige Details für fehlgeschlagene Checkpoints
                for detail in result["details"]:
                    if not detail["success"]:
                        print(f"   ❌ {detail['test']}: {detail['message']}")
        
        print("\n" + "="*80)
        if all_success:
            print("✅ ALLE TESTS ERFOLGREICH BESTANDEN")
        else:
            print("❌ EINIGE TESTS SIND FEHLGESCHLAGEN")
        print("="*80)


def test_checkpoint1(tester):
    """Test der grundlegenden Infrastruktur."""
    tester.start_checkpoint("checkpoint1")
    
    # Überprüfe Verzeichnisstruktur
    directories = [
        "miso",
        "miso/core",
        "miso/math",
        "miso/prism",
        "miso/timeline"
    ]
    
    for directory in directories:
        exists = os.path.isdir(directory)
        tester.add_result(
            f"Verzeichnis {directory}",
            exists,
            "" if exists else f"Verzeichnis {directory} nicht gefunden"
        )
    
    # Überprüfe, ob scipy installiert ist
    try:
        import scipy
        tester.add_result("scipy installiert", True, f"Version: {scipy.__version__}")
    except ImportError as e:
        tester.add_result("scipy installiert", False, str(e))
    
    # Überprüfe, ob numpy installiert ist
    try:
        import numpy
        tester.add_result("numpy installiert", True, f"Version: {numpy.__version__}")
    except ImportError as e:
        tester.add_result("numpy installiert", False, str(e))
    
    tester.finish_checkpoint()


def test_checkpoint2(tester):
    """Test der PRISM Engine."""
    tester.start_checkpoint("checkpoint2")
    
    # Überprüfe, ob die PRISM-Module existieren
    modules = [
        "miso.prism.prism_core",
        "miso.prism.event_generator",
        "miso.prism.visualization_engine"
    ]
    
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
            tester.add_result(f"Modul {module_name}", True)
        except ImportError as e:
            tester.add_result(f"Modul {module_name}", False, str(e))
    
    # Teste PRISMEngine Funktionalität
    try:
        from miso.prism.prism_core import PRISMEngine, Timeline, TimelineNode
        
        # Erstelle eine PRISMEngine-Instanz
        engine = PRISMEngine()
        tester.add_result("PRISMEngine Instanziierung", True)
        
        # Teste create_timeline
        timeline_id = engine.create_timeline("Test Timeline")
        timeline_created = timeline_id is not None
        tester.add_result(
            "create_timeline",
            timeline_created,
            f"Timeline ID: {timeline_id}" if timeline_created else "Timeline konnte nicht erstellt werden"
        )
        
        if timeline_created:
            # Teste create_node
            try:
                node_id = engine.create_node(timeline_id, {"test": "data"}, 1.0)
                node_created = node_id is not None
                tester.add_result(
                    "create_node",
                    node_created,
                    f"Node ID: {node_id}" if node_created else "Node konnte nicht erstellt werden"
                )
            except Exception as e:
                tester.add_result("create_node", False, str(e))
    
    except Exception as e:
        tester.add_result("PRISMEngine Tests", False, str(e))
    
    # Teste EventGenerator
    try:
        from miso.prism.event_generator import EventGenerator
        
        # Erstelle eine PRISMEngine-Instanz für den EventGenerator
        engine = PRISMEngine()
        generator = EventGenerator(prism_engine=engine)
        tester.add_result("EventGenerator Instanziierung", True)
        
        # Teste generate_event
        try:
            event = generator.generate_event("TEST_EVENT", {"test": "data"})
            tester.add_result(
                "generate_event",
                event is not None,
                f"Event: {event}" if event is not None else "Event konnte nicht generiert werden"
            )
        except Exception as e:
            tester.add_result("generate_event", False, str(e))
            
    except Exception as e:
        tester.add_result("EventGenerator Tests", False, str(e))
    
    # Teste VisualizationEngine
    try:
        from miso.prism.visualization_engine import VisualizationEngine
        
        # Erstelle eine PRISMEngine-Instanz für die VisualizationEngine
        engine = PRISMEngine()
        viz_engine = VisualizationEngine(prism_engine=engine)
        tester.add_result("VisualizationEngine Instanziierung", True)
        
    except Exception as e:
        tester.add_result("VisualizationEngine Tests", False, str(e))
    
    tester.finish_checkpoint()


def test_checkpoint3(tester):
    """Test der Math-Module."""
    tester.start_checkpoint("checkpoint3")
    
    # Überprüfe, ob die Math-Module existieren
    modules = [
        "miso.math.vector_operations",
        "miso.math.matrix_operations",
        "miso.math.tensor_operations",
        "miso.math.quantum_math",
        "miso.math.statistical_analysis",
        "miso.math.differential_equations",
        "miso.math.optimization_algorithms"
    ]
    
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
            tester.add_result(f"Modul {module_name}", True)
        except ImportError as e:
            tester.add_result(f"Modul {module_name}", False, str(e))
    
    # Teste VectorOperations
    try:
        from miso.math.vector_operations import VectorOperations
        
        vec_ops = VectorOperations()
        tester.add_result("VectorOperations Instanziierung", True)
        
        # Teste add
        result = vec_ops.add([1, 2, 3], [4, 5, 6])
        expected = [5, 7, 9]
        success = result == expected
        tester.add_result(
            "VectorOperations.add",
            success,
            f"Ergebnis: {result}, Erwartet: {expected}" if not success else ""
        )
        
    except Exception as e:
        tester.add_result("VectorOperations Tests", False, str(e))
    
    # Teste MatrixOperations
    try:
        from miso.math.matrix_operations import MatrixOperations
        
        matrix_ops = MatrixOperations()
        tester.add_result("MatrixOperations Instanziierung", True)
        
        # Teste add
        result = matrix_ops.add([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        expected = [[6, 8], [10, 12]]
        success = result == expected
        tester.add_result(
            "MatrixOperations.add",
            success,
            f"Ergebnis: {result}, Erwartet: {expected}" if not success else ""
        )
        
    except Exception as e:
        tester.add_result("MatrixOperations Tests", False, str(e))
    
    # Teste TensorOperations
    try:
        from miso.math.tensor_operations import TensorOperations
        import numpy as np
        
        tensor_ops = TensorOperations()
        tester.add_result("TensorOperations Instanziierung", True)
        
        # Teste add
        t1 = np.array([[1, 2], [3, 4]])
        t2 = np.array([[5, 6], [7, 8]])
        result = tensor_ops.add(t1, t2)
        expected = np.array([[6, 8], [10, 12]])
        success = np.array_equal(result, expected)
        tester.add_result(
            "TensorOperations.add",
            success,
            f"Ergebnis: {result}, Erwartet: {expected}" if not success else ""
        )
        
    except Exception as e:
        tester.add_result("TensorOperations Tests", False, str(e))
    
    # Teste QuantumMath
    try:
        from miso.math.quantum_math import QuantumMath
        
        quantum_math = QuantumMath()
        tester.add_result("QuantumMath Instanziierung", True)
        
        # Teste superposition
        result = quantum_math.superposition(1, 0)
        success = len(result) == 2
        tester.add_result(
            "QuantumMath.superposition",
            success,
            f"Ergebnis: {result}" if not success else ""
        )
        
    except Exception as e:
        tester.add_result("QuantumMath Tests", False, str(e))
    
    # Teste StatisticalAnalysis
    try:
        from miso.math.statistical_analysis import StatisticalAnalysis
        
        stats = StatisticalAnalysis()
        tester.add_result("StatisticalAnalysis Instanziierung", True)
        
        # Teste mean
        result = stats.mean([1, 2, 3, 4, 5])
        expected = 3
        success = result == expected
        tester.add_result(
            "StatisticalAnalysis.mean",
            success,
            f"Ergebnis: {result}, Erwartet: {expected}" if not success else ""
        )
        
    except Exception as e:
        tester.add_result("StatisticalAnalysis Tests", False, str(e))
    
    # Teste DifferentialEquations
    try:
        from miso.math.differential_equations import DifferentialEquations
        
        diff_eq = DifferentialEquations()
        tester.add_result("DifferentialEquations Instanziierung", True)
        
    except Exception as e:
        tester.add_result("DifferentialEquations Tests", False, str(e))
    
    # Teste OptimizationAlgorithms
    try:
        from miso.math.optimization_algorithms import OptimizationAlgorithms
        
        opt_alg = OptimizationAlgorithms()
        tester.add_result("OptimizationAlgorithms Instanziierung", True)
        
    except Exception as e:
        tester.add_result("OptimizationAlgorithms Tests", False, str(e))
    
    tester.finish_checkpoint()


def test_checkpoint4(tester):
    """Test des Timeline Frameworks."""
    tester.start_checkpoint("checkpoint4")
    
    # Überprüfe, ob die Timeline-Module existieren
    modules = [
        "miso.timeline"
    ]
    
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
            tester.add_result(f"Modul {module_name}", True)
        except ImportError as e:
            tester.add_result(f"Modul {module_name}", False, str(e))
    
    # Teste QTM_Modulator
    try:
        # Versuche, das QTM_Modulator-Modul zu importieren
        try:
            from miso.timeline.qtm_modulator import QTM_Modulator
            module_exists = True
        except ImportError:
            module_exists = False
        
        tester.add_result(
            "QTM_Modulator Modul existiert",
            module_exists,
            "" if module_exists else "miso.timeline.qtm_modulator konnte nicht importiert werden"
        )
        
        if module_exists:
            # Teste QTM_Modulator Instanziierung
            try:
                modulator = QTM_Modulator()
                tester.add_result("QTM_Modulator Instanziierung", True)
            except Exception as e:
                tester.add_result("QTM_Modulator Instanziierung", False, str(e))
    
    except Exception as e:
        tester.add_result("QTM_Modulator Tests", False, str(e))
    
    tester.finish_checkpoint()


def test_checkpoint5(tester):
    """Test des Gesamtsystems."""
    tester.start_checkpoint("checkpoint5")
    
    # Teste End-to-End Funktionalität
    try:
        # Importiere die Hauptkomponenten
        from miso.prism.prism_core import PRISMEngine
        from miso.prism.event_generator import EventGenerator
        
        # Erstelle eine PRISMEngine-Instanz
        engine = PRISMEngine()
        generator = EventGenerator(prism_engine=engine)
        
        # Erstelle eine Timeline
        timeline_id = engine.create_timeline("Test Timeline")
        
        # Erstelle Knoten
        node1_id = engine.create_node(timeline_id, {"data": "node1"}, 1.0)
        node2_id = engine.create_node(timeline_id, {"data": "node2"}, 2.0)
        
        # Verbinde Knoten
        success = True
        try:
            engine.connect_nodes(timeline_id, node1_id, node2_id)
        except Exception as e:
            success = False
        
        tester.add_result(
            "End-to-End Timeline Test",
            success,
            "" if success else "Fehler beim Verbinden von Knoten"
        )
        
        # Teste Paradoxerkennung (falls implementiert)
        try:
            # Diese Funktion könnte nicht existieren, daher in try-except
            paradox = engine.detect_paradox(timeline_id)
            tester.add_result("Paradoxerkennung", True, f"Paradox erkannt: {paradox}")
        except AttributeError:
            tester.add_result("Paradoxerkennung", False, "Methode detect_paradox nicht implementiert")
        except Exception as e:
            tester.add_result("Paradoxerkennung", False, str(e))
        
    except Exception as e:
        tester.add_result("End-to-End Test", False, str(e))
    
    tester.finish_checkpoint()


def main():
    """Hauptfunktion zum Ausführen aller Tests."""
    print(f"\n{'='*80}\nMISO SYSTEMTEST\n{'='*80}")
    print(f"Gestartet: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tester = CheckpointTester()
    
    try:
        # Führe alle Checkpoint-Tests aus
        test_checkpoint1(tester)
        test_checkpoint2(tester)
        test_checkpoint3(tester)
        test_checkpoint4(tester)
        test_checkpoint5(tester)
        
        # Gib Zusammenfassung aus
        tester.print_summary()
        
    except Exception as e:
        print(f"\n❌ KRITISCHER FEHLER: {e}")
        traceback.print_exc()
    
    print(f"\nTest abgeschlossen: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
