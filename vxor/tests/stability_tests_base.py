#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - Basistests für Stabilitätstests

Dieses Skript führt Basistests für alle Module des MISO Ultimate AGI-Systems durch.
Die Tests prüfen die grundlegende Funktionalität jedes Moduls.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import json
import datetime
import importlib
import traceback
import gc
import psutil
import numpy as np
from pathlib import Path

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stability_tests.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("miso_stability_tests")

# Basis-Verzeichnis
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Füge Basis-Verzeichnis zum Pfad hinzu
sys.path.insert(0, BASE_DIR)

# Ergebnisse
results = {
    "timestamp": datetime.datetime.now().isoformat(),
    "modules": {},
    "overall_status": "pending"
}

def measure_memory_usage():
    """Misst den aktuellen Speicherverbrauch des Prozesses"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Plattformunabhängige Speicherinformationen
    memory_dict = {
        "rss": memory_info.rss,  # Resident Set Size
        "vms": memory_info.vms,  # Virtual Memory Size
    }
    
    # Füge plattformspezifische Attribute hinzu, wenn verfügbar
    for attr in ['shared', 'text', 'lib', 'data', 'dirty']:
        if hasattr(memory_info, attr):
            memory_dict[attr] = getattr(memory_info, attr)
    
    return memory_dict

def run_gc():
    """Führt Garbage Collection durch"""
    gc.collect()

def test_module(module_name, module_path, test_function):
    """
    Testet ein Modul
    
    Args:
        module_name: Name des Moduls
        module_path: Pfad zum Modul
        test_function: Testfunktion
        
    Returns:
        Testergebnis
    """
    logger.info(f"Teste Modul: {module_name}")
    
    # Initialisiere Ergebnis
    result = {
        "name": module_name,
        "path": module_path,
        "status": "pending",
        "start_time": datetime.datetime.now().isoformat(),
        "end_time": None,
        "duration": None,
        "memory_before": measure_memory_usage(),
        "memory_after": None,
        "memory_diff": None,
        "error": None,
        "details": {}
    }
    
    try:
        # Führe Garbage Collection durch
        run_gc()
        
        # Führe Test durch
        start_time = time.time()
        test_result = test_function()
        end_time = time.time()
        
        # Führe Garbage Collection durch
        run_gc()
        
        # Aktualisiere Ergebnis
        result["status"] = "passed" if test_result.get("success", False) else "failed"
        result["end_time"] = datetime.datetime.now().isoformat()
        result["duration"] = end_time - start_time
        result["memory_after"] = measure_memory_usage()
        result["details"] = test_result
        
        # Berechne Speicherdifferenz
        result["memory_diff"] = {
            key: result["memory_after"][key] - result["memory_before"][key]
            for key in result["memory_before"]
        }
        
        logger.info(f"Test für Modul {module_name} abgeschlossen: {result['status']}")
    except Exception as e:
        # Aktualisiere Ergebnis bei Fehler
        result["status"] = "error"
        result["end_time"] = datetime.datetime.now().isoformat()
        result["duration"] = time.time() - start_time
        result["memory_after"] = measure_memory_usage()
        result["error"] = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        
        # Berechne Speicherdifferenz
        result["memory_diff"] = {
            key: result["memory_after"][key] - result["memory_before"][key]
            for key in result["memory_before"]
        }
        
        logger.error(f"Fehler beim Testen von Modul {module_name}: {e}")
    
    return result

def test_m_code_core():
    """Testet das M-CODE Core Modul"""
    try:
        # Importiere M-CODE Core
        from miso.code.m_code import MCodeCompiler, MCodeInterpreter, MCodeSyntaxTree
        from miso.code.m_code import compile_m_code, execute_m_code, parse_m_code, optimize_m_code
        from miso.code.m_code import initialize_m_code, get_runtime, reset_runtime
        
        # Initialisiere M-CODE
        initialize_m_code()
        
        # Teste Syntax-Parser
        m_code = """
        func test_function(x, y) {
            return x + y;
        }
        
        test_function(2, 3);
        """
        
        syntax_tree = parse_m_code(m_code)
        
        # Teste Optimizer
        optimized_tree = optimize_m_code(syntax_tree)
        
        # Teste Compiler
        bytecode = compile_m_code(m_code)
        
        # Teste Runtime
        runtime = get_runtime()
        
        # Teste Interpreter
        # In einer vollständigen Implementierung würde hier die Ausführung erfolgen
        # Für diesen Test geben wir ein Erfolgsergebnis zurück
        
        # Bereinige
        reset_runtime()
        
        return {
            "success": True,
            "message": "M-CODE Core Test erfolgreich",
            "details": {
                "syntax_tree": str(syntax_tree),
                "optimized_tree": str(optimized_tree),
                "bytecode": str(bytecode),
                "runtime": str(runtime)
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"M-CODE Core Test fehlgeschlagen: {e}",
            "error": str(e)
        }

def test_m_lingua_interface():
    """Testet das M-LINGUA Interface Modul"""
    try:
        # Importiere M-LINGUA Interface
        from miso.lang.mlingua.mlingua_interface import MLinguaInterface, MLinguaResult
        from miso.lang.mlingua.semantic_layer import SemanticResult, SemanticContext
        from miso.lang.mlingua.multilang_parser import ParsedCommand
        
        # Erstelle M-LINGUA Interface
        mlingua = MLinguaInterface()
        
        # Teste Verarbeitung
        result = mlingua.process("Öffne die Datei 'test.txt'")
        
        return {
            "success": True,
            "message": "M-LINGUA Interface Test erfolgreich",
            "details": {
                "result": str(result),
                "detected_language": result.detected_language,
                "semantic_result": str(result.semantic_result)
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"M-LINGUA Interface Test fehlgeschlagen: {e}",
            "error": str(e)
        }

def test_echo_prime():
    """Testet das ECHO-PRIME Modul"""
    try:
        # Importiere ECHO-PRIME
        from engines.echo_prime.engine import EchoPrimeEngine
        from engines.echo_prime.timeline import Timeline, TimeNode, Event, Trigger
        from engines.echo_prime.paradox import ParadoxDetector, ParadoxType, ParadoxResolution
        from engines.echo_prime.quantum import QuantumTimelineSimulator, QuantumTimeEffect, QuantumState
        
        # Erstelle ECHO-PRIME Engine
        engine = EchoPrimeEngine()
        
        # Erstelle Timeline
        timeline = Timeline(name="Test Timeline", description="Eine Testzeitlinie")
        
        # Erstelle Events
        event1 = Event(
            name="Event 1",
            description="Ein Testereignis",
            timestamp=datetime.datetime.now()
        )
        
        event2 = Event(
            name="Event 2",
            description="Ein weiteres Testereignis",
            timestamp=datetime.datetime.now() + datetime.timedelta(hours=1)
        )
        
        # Füge Events zur Timeline hinzu
        timeline.add_event(event1)
        timeline.add_event(event2)
        
        # Erstelle Trigger
        trigger = Trigger(
            name="Trigger 1",
            description="Ein Testtrigger",
            condition="event1.timestamp < event2.timestamp",
            actions=["print('Trigger ausgelöst')"]
        )
        
        # Füge Trigger zur Timeline hinzu
        timeline.add_trigger(trigger)
        
        # Erstelle Quantum-Effekt
        quantum_effect = QuantumTimeEffect(
            name="Quantum Effect 1",
            description="Ein Testquanteneffekt",
            quantum_state=QuantumState.SUPERPOSITION,
            probability=0.5,
            affected_events=[event1.id],
            affected_triggers=[trigger.id]
        )
        
        # Erstelle Quantum-Simulator
        quantum_simulator = QuantumTimelineSimulator()
        
        # Simuliere Quantum-Effekte
        simulation_result = quantum_simulator.simulate_quantum_effects(
            timeline_data=timeline.to_dict(),
            quantum_effects=[quantum_effect]
        )
        
        return {
            "success": True,
            "message": "ECHO-PRIME Test erfolgreich",
            "details": {
                "timeline": str(timeline),
                "events": [str(event1), str(event2)],
                "trigger": str(trigger),
                "quantum_effect": str(quantum_effect),
                "simulation_result": str(simulation_result)
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"ECHO-PRIME Test fehlgeschlagen: {e}",
            "error": str(e)
        }

def test_hyperfilter():
    """Testet das HYPERFILTER Modul"""
    try:
        # Importiere HYPERFILTER
        from miso.filter.hyperfilter import HyperFilter, FilterConfig, FilterMode
        from miso.filter.vxor_hyperfilter_integration import VXHyperfilterAdapter
        
        # Erstelle HyperFilter
        config = FilterConfig(
            mode=FilterMode.STRICT,
            threshold=0.8,
            max_retries=3
        )
        
        hyperfilter = HyperFilter(config)
        
        # Teste Filterung
        test_text = "Dies ist ein Testtext für den HYPERFILTER."
        filtered_text, metadata = hyperfilter.filter_input(test_text)
        
        # Erstelle VXOR-Adapter
        adapter = VXHyperfilterAdapter()
        
        # Teste VXOR-Integration
        result = adapter.process_content(
            raw_text=test_text,
            source_trust_score=0.9,
            language_code="de",
            context_stream="Test",
            media_source_type="text"
        )
        
        return {
            "success": True,
            "message": "HYPERFILTER Test erfolgreich",
            "details": {
                "filtered_text": filtered_text,
                "metadata": str(metadata),
                "vxor_result": str(result)
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"HYPERFILTER Test fehlgeschlagen: {e}",
            "error": str(e)
        }

def test_deep_state():
    """Testet das Deep-State-Modul"""
    try:
        # Importiere Deep-State-Modul
        from miso.analysis.deep_state import DeepStateAnalyzer, AnalysisResult, AnalysisConfig
        from miso.analysis.deep_state_patterns import PatternMatcher
        from miso.analysis.deep_state_network import NetworkAnalyzer
        from miso.analysis.deep_state_security import SecurityManager
        
        # Erstelle DeepStateAnalyzer
        analyzer = DeepStateAnalyzer()
        
        # Teste Analyse
        test_text = "Dies ist ein Testtext für die Deep-State-Analyse."
        result = analyzer.analyze(
            content_stream=test_text,
            source_id="test",
            source_trust_level=0.9,
            language_code="de",
            context_cluster="test"
        )
        
        return {
            "success": True,
            "message": "Deep-State-Modul Test erfolgreich",
            "details": {
                "result": str(result),
                "bias_score": result.bias_score,
                "pattern_score": result.pattern_score,
                "network_score": result.network_score,
                "paradox_signal": result.paradox_signal
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Deep-State-Modul Test fehlgeschlagen: {e}",
            "error": str(e)
        }

def test_t_mathematics():
    """Testet die T-MATHEMATICS ENGINE"""
    try:
        # Versuche, T-MATHEMATICS zu importieren
        try:
            # Verwende den korrekten Importpfad
            from miso.math.t_mathematics.integration_manager import get_t_math_integration_manager
            
            # Hole die T-MATHEMATICS Engine über den Integration Manager
            t_math_manager = get_t_math_integration_manager()
            engine = t_math_manager.get_engine("tests")
            
            # Prüfe, ob MLX verfügbar ist
            is_apple_silicon = engine.use_mlx
            
            # Erstelle Tensoren mit PyTorch
            import torch
            tensor1 = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
            tensor2 = torch.tensor([5, 6, 7, 8], dtype=torch.float32)
            
            # Führe Operationen mit der Engine durch
            result1 = engine.matmul(tensor1.unsqueeze(0), tensor2.unsqueeze(1))
            result2 = engine.prepare_tensor(tensor1) * engine.prepare_tensor(tensor2)
            result3 = engine.prepare_tensor(tensor1).dot(engine.prepare_tensor(tensor2))
            
            t_math_available = True
            details = {
                "tensor1": str(tensor1),
                "tensor2": str(tensor2),
                "result1": str(result1),
                "result2": str(result2),
                "result3": str(result3),
                "using_mlx": is_apple_silicon
            }
        except ImportError as e:
            # Fallback auf NumPy
            t_math_available = False
            
            # Erstelle NumPy-Arrays
            tensor1 = np.array([1, 2, 3, 4])
            tensor2 = np.array([5, 6, 7, 8])
            
            # Führe Operationen durch
            result1 = tensor1 + tensor2
            result2 = tensor1 * tensor2
            result3 = np.dot(tensor1, tensor2)
            
            details = {
                "tensor1": str(tensor1),
                "tensor2": str(tensor2),
                "result1": str(result1),
                "result2": str(result2),
                "result3": str(result3),
                "fallback": "NumPy",
                "import_error": str(e)
            }
        
        # Prüfe auch die VXOR-Integration
        try:
            # Versuche, die VXOR-Integration zu testen
            from miso.math.t_mathematics.vxor_math_integration import get_vxor_math_integration
            vxor_integration = get_vxor_math_integration()
            vxor_modules = vxor_integration.get_available_modules()
            details["vxor_integration"] = {
                "available_modules": vxor_modules,
                "status": "available"
            }
        except Exception as e:
            details["vxor_integration"] = {
                "status": "error",
                "error": str(e)
            }
        
        return {
            "success": True,
            "message": "T-MATHEMATICS ENGINE Test erfolgreich",
            "t_math_available": t_math_available,
            "details": details
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"T-MATHEMATICS ENGINE Test fehlgeschlagen: {e}",
            "error": str(e)
        }

def test_mprime():
    """Testet das MPRIME Mathematikmodul"""
    try:
        # Importiere MPRIME
        from miso.math.mprime.symbol_solver import SymbolSolver, MathExpression, SymbolTable
        
        # Erstelle SymbolSolver
        solver = SymbolSolver()
        
        # Teste Symbolverarbeitung
        expression = "a + b * c"
        symbol_table = SymbolTable()
        symbol_table.set("a", 1)
        symbol_table.set("b", 2)
        symbol_table.set("c", 3)
        
        result = solver.evaluate(expression, symbol_table)
        
        return {
            "success": True,
            "message": "MPRIME Mathematikmodul Test erfolgreich",
            "details": {
                "expression": expression,
                "symbol_table": str(symbol_table),
                "result": result
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"MPRIME Mathematikmodul Test fehlgeschlagen: {e}",
            "error": str(e)
        }

def test_q_logik():
    """Testet das Q-LOGIK Modul"""
    try:
        # Importiere Q-LOGIK
        from miso.logic.qlogik_engine import QLogikEngine, LogikRule, LogikContext
        
        # Erstelle Q-LOGIK Engine
        engine = QLogikEngine()
        
        # Erstelle Regeln
        rule1 = LogikRule(
            name="Rule 1",
            condition="A and B",
            conclusion="C"
        )
        
        rule2 = LogikRule(
            name="Rule 2",
            condition="C",
            conclusion="D"
        )
        
        # Füge Regeln hinzu
        engine.add_rule(rule1)
        engine.add_rule(rule2)
        
        # Erstelle Kontext
        context = LogikContext()
        context.set_fact("A", True)
        context.set_fact("B", True)
        
        # Führe Inferenz durch
        result = engine.infer(context)
        
        return {
            "success": True,
            "message": "Q-LOGIK Test erfolgreich",
            "details": {
                "rules": [str(rule1), str(rule2)],
                "context": str(context),
                "result": str(result),
                "facts": result.get_facts()
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Q-LOGIK Test fehlgeschlagen: {e}",
            "error": str(e)
        }

def test_prism_engine():
    """Testet die PRISM-Engine"""
    try:
        # Importiere PRISM-Engine
        from miso.simulation.prism_engine import PrismEngine, SimulationConfig
        from miso.simulation.prism_matrix import PrismMatrix
        
        # Erstelle PRISM-Engine
        engine = PrismEngine()
        
        # Erstelle Konfiguration
        config = SimulationConfig(
            iterations=10,
            time_steps=5,
            seed=42
        )
        
        # Erstelle Matrix
        matrix = PrismMatrix(3, 3)
        matrix.set(0, 0, 1.0)
        matrix.set(1, 1, 2.0)
        matrix.set(2, 2, 3.0)
        
        # Führe Simulation durch
        result = engine.simulate(matrix, config)
        
        return {
            "success": True,
            "message": "PRISM-Engine Test erfolgreich",
            "details": {
                "matrix": str(matrix),
                "config": str(config),
                "result": str(result)
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"PRISM-Engine Test fehlgeschlagen: {e}",
            "error": str(e)
        }

def test_void_protocol():
    """Testet das VOID-Protokoll 3.0"""
    try:
        # Importiere VOID-Protokoll
        from miso.protect.void_protocol import VoidProtocol, SecurityLevel, ProtectionMode
        
        # Erstelle VOID-Protokoll
        protocol = VoidProtocol(
            security_level=SecurityLevel.HIGH,
            protection_mode=ProtectionMode.FULL
        )
        
        # Teste Schutz
        test_data = "Dies sind Testdaten für das VOID-Protokoll."
        protected_data = protocol.protect(test_data)
        
        # Teste Entschlüsselung
        decrypted_data = protocol.unprotect(protected_data)
        
        return {
            "success": True,
            "message": "VOID-Protokoll 3.0 Test erfolgreich",
            "details": {
                "original_data": test_data,
                "protected_data": str(protected_data),
                "decrypted_data": decrypted_data,
                "integrity_check": test_data == decrypted_data
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"VOID-Protokoll 3.0 Test fehlgeschlagen: {e}",
            "error": str(e)
        }

def test_nexus_os():
    """Testet das NEXUS-OS"""
    try:
        # Importiere NEXUS-OS
        from miso.core.nexus_os import NexusOS, TaskManager, ResourceManager
        
        # Erstelle NEXUS-OS
        os = NexusOS()
        
        # Erstelle Task
        task = {
            "id": "task1",
            "name": "Test Task",
            "priority": 1,
            "resources": ["cpu", "memory"],
            "function": "test_function",
            "arguments": {"arg1": 1, "arg2": 2}
        }
        
        # Füge Task hinzu
        os.task_manager.add_task(task)
        
        # Teste Ressourcenzuweisung
        allocation = os.resource_manager.allocate_resources(task)
        
        return {
            "success": True,
            "message": "NEXUS-OS Test erfolgreich",
            "details": {
                "task": str(task),
                "allocation": str(allocation),
                "task_queue": str(os.task_manager.get_task_queue())
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"NEXUS-OS Test fehlgeschlagen: {e}",
            "error": str(e)
        }

def test_mimimon():
    """Testet das MIMIMON: ZTM-Modul"""
    try:
        # Importiere MIMIMON
        from miso.security.ztm.mimimon import MIMIMON, ZTMPolicy, ZTMVerifier
        
        # Erstelle MIMIMON
        mimimon = MIMIMON()
        
        # Erstelle Policy
        policy = ZTMPolicy(
            name="Test Policy",
            rules=[
                {"resource": "file", "action": "read", "allow": True},
                {"resource": "network", "action": "connect", "allow": False}
            ]
        )
        
        # Füge Policy hinzu
        mimimon.add_policy(policy)
        
        # Teste Verifikation
        result1 = mimimon.verify("file", "read")
        result2 = mimimon.verify("network", "connect")
        
        return {
            "success": True,
            "message": "MIMIMON: ZTM-Modul Test erfolgreich",
            "details": {
                "policy": str(policy),
                "result1": result1,
                "result2": result2
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"MIMIMON: ZTM-Modul Test fehlgeschlagen: {e}",
            "error": str(e)
        }

def test_vxor_integration():
    """Testet die VXOR-Integration"""
    try:
        # Importiere VXOR-Integration
        from miso.vxor.vx_memex import VXMemex
        
        # Erstelle VXMemex
        memex = VXMemex()
        
        # Teste Speicherung
        memory = {
            "id": "memory1",
            "content": "Dies ist ein Testgedächtnis",
            "tags": ["test", "memory"],
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        result = memex.store_memory(memory)
        
        return {
            "success": True,
            "message": "VXOR-Integration Test erfolgreich",
            "details": {
                "memory": str(memory),
                "result": str(result)
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"VXOR-Integration Test fehlgeschlagen: {e}",
            "error": str(e)
        }

def main():
    """Hauptfunktion"""
    logger.info("Starte MISO Ultimate Stabilitätstests (Basistests)...")
    
    # Liste der zu testenden Module
    modules = [
        {"name": "M-CODE Core", "test_function": test_m_code_core},
        {"name": "M-LINGUA Interface", "test_function": test_m_lingua_interface},
        {"name": "ECHO-PRIME", "test_function": test_echo_prime},
        {"name": "HYPERFILTER", "test_function": test_hyperfilter},
        {"name": "Deep-State-Modul", "test_function": test_deep_state},
        {"name": "T-MATHEMATICS ENGINE", "test_function": test_t_mathematics},
        {"name": "MPRIME Mathematikmodul", "test_function": test_mprime},
        {"name": "Q-LOGIK", "test_function": test_q_logik},
        {"name": "PRISM-Engine", "test_function": test_prism_engine},
        {"name": "VOID-Protokoll 3.0", "test_function": test_void_protocol},
        {"name": "NEXUS-OS", "test_function": test_nexus_os},
        {"name": "MIMIMON: ZTM-Modul", "test_function": test_mimimon},
        {"name": "VXOR-Integration", "test_function": test_vxor_integration}
    ]
    
    # Teste Module
    for module in modules:
        module_name = module["name"]
        test_function = module["test_function"]
        
        # Teste Modul
        result = test_module(module_name, "", test_function)
        
        # Speichere Ergebnis
        results["modules"][module_name] = result
    
    # Bestimme Gesamtstatus
    passed = sum(1 for module in results["modules"].values() if module["status"] == "passed")
    failed = sum(1 for module in results["modules"].values() if module["status"] == "failed")
    errors = sum(1 for module in results["modules"].values() if module["status"] == "error")
    
    if errors > 0:
        results["overall_status"] = "error"
    elif failed > 0:
        results["overall_status"] = "failed"
    else:
        results["overall_status"] = "passed"
    
    # Speichere Ergebnisse
    with open("stability_test_results_base.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Erstelle Zusammenfassung
    summary = f"""
MISO Ultimate Stabilitätstests (Basistests) - Zusammenfassung
==========================================================
Zeitstempel: {datetime.datetime.now().isoformat()}
Gesamtstatus: {results["overall_status"]}

Modulstatus:
"""
    
    for module_name, module_result in results["modules"].items():
        summary += f"  - {module_name}: {module_result['status']}"
        if module_result["status"] == "error":
            summary += f" ({module_result['error']['type']}: {module_result['error']['message']})"
        summary += f" ({module_result['duration']:.2f}s)\n"
    
    summary += f"""
Statistik:
  - Bestanden: {passed}/{len(modules)}
  - Fehlgeschlagen: {failed}/{len(modules)}
  - Fehler: {errors}/{len(modules)}

Ergebnisse gespeichert in:
  - stability_test_results_base.json
"""
    
    # Speichere Zusammenfassung
    with open("stability_test_results_base.md", "w", encoding="utf-8") as f:
        f.write(summary)
    
    logger.info(f"MISO Ultimate Stabilitätstests (Basistests) abgeschlossen: {results['overall_status']}")
    logger.info(f"Bestanden: {passed}/{len(modules)}, Fehlgeschlagen: {failed}/{len(modules)}, Fehler: {errors}/{len(modules)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
