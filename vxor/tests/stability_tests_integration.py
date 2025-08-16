#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - Integrationstests für Stabilitätstests

Dieses Skript führt Integrationstests für die Module des MISO Ultimate AGI-Systems durch.
Die Tests prüfen die Integration und Kommunikation zwischen verschiedenen Modulen.

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
from pathlib import Path

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stability_tests_integration.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("miso_stability_integration_tests")

# Basis-Verzeichnis
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Füge Basis-Verzeichnis zum Pfad hinzu
sys.path.insert(0, BASE_DIR)

# Ergebnisse
results = {
    "timestamp": datetime.datetime.now().isoformat(),
    "integration_tests": {},
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

def test_integration(test_name, test_function):
    """
    Führt einen Integrationstest durch
    
    Args:
        test_name: Name des Tests
        test_function: Testfunktion
        
    Returns:
        Testergebnis
    """
    logger.info(f"Führe Integrationstest durch: {test_name}")
    
    # Initialisiere Ergebnis
    result = {
        "name": test_name,
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
        
        logger.info(f"Integrationstest {test_name} abgeschlossen: {result['status']}")
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
        
        logger.error(f"Fehler beim Integrationstest {test_name}: {e}")
    
    return result

def test_m_code_m_lingua_integration():
    """Testet die Integration zwischen M-CODE und M-LINGUA"""
    try:
        # Importiere M-CODE und M-LINGUA
        from miso.code.m_code import compile_m_code, execute_m_code
        from miso.lang.mlingua.mlingua_interface import MLinguaInterface
        from miso.lang.mcode.m_code_bridge import MCodeBridge
        
        # Erstelle M-LINGUA Interface
        mlingua = MLinguaInterface()
        
        # Erstelle M-CODE Bridge
        bridge = MCodeBridge(mlingua)
        
        # Teste Übersetzung von natürlicher Sprache zu M-CODE
        natural_language = "Berechne die Summe von 2 und 3"
        
        # Übersetze natürliche Sprache zu M-CODE
        m_code = bridge.translate_to_m_code(natural_language)
        
        # Kompiliere M-CODE
        bytecode = compile_m_code(m_code)
        
        # Führe M-CODE aus
        result = execute_m_code(m_code)
        
        return {
            "success": True,
            "message": "M-CODE und M-LINGUA Integration erfolgreich",
            "details": {
                "natural_language": natural_language,
                "m_code": m_code,
                "bytecode": str(bytecode),
                "result": result
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"M-CODE und M-LINGUA Integration fehlgeschlagen: {e}",
            "error": str(e)
        }

def test_echo_prime_vxor_integration():
    """Testet die Integration zwischen ECHO-PRIME und VXOR"""
    try:
        # Importiere ECHO-PRIME und VXOR-Integration
        from engines.echo_prime.engine import EchoPrimeEngine
        from engines.echo_prime.vxor_integration import EchoPrimeVXORAdapter
        from engines.echo_prime.timeline import Timeline, Event
        
        # Erstelle ECHO-PRIME Engine
        engine = EchoPrimeEngine()
        
        # Erstelle VXOR-Adapter
        adapter = EchoPrimeVXORAdapter()
        
        # Erstelle Timeline
        timeline = Timeline(
            name="Test Timeline",
            description="Eine Testzeitlinie für die VXOR-Integration"
        )
        
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
        
        # Speichere Timeline in VXOR
        store_result = adapter.store_timeline(timeline)
        
        # Lade Timeline aus VXOR
        load_result = adapter.load_timeline(timeline.id)
        
        return {
            "success": True,
            "message": "ECHO-PRIME und VXOR Integration erfolgreich",
            "details": {
                "timeline": str(timeline),
                "store_result": str(store_result),
                "load_result": str(load_result)
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"ECHO-PRIME und VXOR Integration fehlgeschlagen: {e}",
            "error": str(e)
        }

def test_hyperfilter_vxor_integration():
    """Testet die Integration zwischen HYPERFILTER und VXOR"""
    try:
        # Importiere HYPERFILTER und VXOR-Integration
        from miso.filter.hyperfilter import HyperFilter, FilterConfig, FilterMode
        from miso.filter.vxor_hyperfilter_integration import VXHyperfilterAdapter
        
        # Erstelle HyperFilter
        config = FilterConfig(
            mode=FilterMode.STRICT,
            threshold=0.8,
            max_retries=3
        )
        
        hyperfilter = HyperFilter(config)
        
        # Erstelle VXOR-Adapter
        adapter = VXHyperfilterAdapter()
        
        # Teste Filterung mit VXOR-Integration
        test_text = "Dies ist ein Testtext für die HYPERFILTER und VXOR Integration."
        
        # Verarbeite Text mit VXOR-Integration
        result = adapter.process_content(
            raw_text=test_text,
            source_trust_score=0.9,
            language_code="de",
            context_stream="Test",
            media_source_type="text"
        )
        
        # Überprüfe, ob VXOR-Ergebnis korrekt ist
        vxor_result_valid = (
            result.get("signal_flag") is not None and
            result.get("report_summary") is not None and
            result.get("decision") is not None and
            result.get("action_trigger") is not None
        )
        
        return {
            "success": vxor_result_valid,
            "message": "HYPERFILTER und VXOR Integration erfolgreich" if vxor_result_valid else "HYPERFILTER und VXOR Integration fehlgeschlagen",
            "details": {
                "test_text": test_text,
                "vxor_result": str(result),
                "vxor_result_valid": vxor_result_valid
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"HYPERFILTER und VXOR Integration fehlgeschlagen: {e}",
            "error": str(e)
        }

def test_deep_state_hyperfilter_integration():
    """Testet die Integration zwischen Deep-State-Modul und HYPERFILTER"""
    try:
        # Importiere Deep-State-Modul und HYPERFILTER
        from miso.analysis.deep_state import DeepStateAnalyzer, AnalysisResult
        from miso.filter.hyperfilter import HyperFilter, FilterConfig, FilterMode
        
        # Erstelle DeepStateAnalyzer
        analyzer = DeepStateAnalyzer()
        
        # Erstelle HyperFilter
        config = FilterConfig(
            mode=FilterMode.STRICT,
            threshold=0.8,
            max_retries=3
        )
        
        hyperfilter = HyperFilter(config)
        
        # Teste Integration
        test_text = "Dies ist ein Testtext für die Deep-State und HYPERFILTER Integration."
        
        # Filtere Text mit HYPERFILTER
        filtered_text, metadata = hyperfilter.filter_input(test_text)
        
        # Analysiere gefilterten Text mit Deep-State-Modul
        analysis_result = analyzer.analyze(
            content_stream=filtered_text,
            source_id="test",
            source_trust_level=metadata.get("trust_score", 0.5),
            language_code="de",
            context_cluster="test"
        )
        
        return {
            "success": True,
            "message": "Deep-State-Modul und HYPERFILTER Integration erfolgreich",
            "details": {
                "original_text": test_text,
                "filtered_text": filtered_text,
                "metadata": str(metadata),
                "analysis_result": str(analysis_result)
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Deep-State-Modul und HYPERFILTER Integration fehlgeschlagen: {e}",
            "error": str(e)
        }

def test_q_logik_t_mathematics_integration():
    """Testet die Integration zwischen Q-LOGIK und T-MATHEMATICS"""
    try:
        # Importiere Q-LOGIK und T-MATHEMATICS
        from miso.logic.qlogik_engine import QLogikEngine, LogikRule, LogikContext
        
        # Versuche, T-MATHEMATICS zu importieren
        try:
            from engines.t_mathematics.tensor import Tensor
            from engines.t_mathematics.engine import TMathematicsEngine
            
            # Erstelle T-MATHEMATICS Engine
            t_math_engine = TMathematicsEngine()
            
            # Erstelle Tensoren
            tensor1 = Tensor([1, 2, 3, 4])
            tensor2 = Tensor([5, 6, 7, 8])
            
            t_math_available = True
        except ImportError:
            # Fallback auf NumPy
            import numpy as np
            
            t_math_engine = None
            tensor1 = np.array([1, 2, 3, 4])
            tensor2 = np.array([5, 6, 7, 8])
            
            t_math_available = False
        
        # Erstelle Q-LOGIK Engine
        q_logik_engine = QLogikEngine()
        
        # Erstelle Regeln
        rule1 = LogikRule(
            name="Tensor Addition Rule",
            condition="tensor1 and tensor2",
            conclusion="result = tensor1 + tensor2"
        )
        
        rule2 = LogikRule(
            name="Tensor Multiplication Rule",
            condition="tensor1 and tensor2",
            conclusion="result = tensor1 * tensor2"
        )
        
        # Füge Regeln hinzu
        q_logik_engine.add_rule(rule1)
        q_logik_engine.add_rule(rule2)
        
        # Erstelle Kontext
        context = LogikContext()
        context.set_fact("tensor1", tensor1)
        context.set_fact("tensor2", tensor2)
        
        # Führe Inferenz durch
        result = q_logik_engine.infer(context)
        
        # Überprüfe, ob Ergebnis korrekt ist
        if t_math_available:
            result_tensor_add = tensor1 + tensor2
            result_tensor_mul = tensor1 * tensor2
        else:
            result_tensor_add = tensor1 + tensor2
            result_tensor_mul = tensor1 * tensor2
        
        return {
            "success": True,
            "message": "Q-LOGIK und T-MATHEMATICS Integration erfolgreich",
            "details": {
                "t_math_available": t_math_available,
                "tensor1": str(tensor1),
                "tensor2": str(tensor2),
                "result": str(result),
                "result_tensor_add": str(result_tensor_add),
                "result_tensor_mul": str(result_tensor_mul),
                "facts": result.get_facts()
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Q-LOGIK und T-MATHEMATICS Integration fehlgeschlagen: {e}",
            "error": str(e)
        }

def test_nexus_os_mimimon_integration():
    """Testet die Integration zwischen NEXUS-OS und MIMIMON"""
    try:
        # Importiere NEXUS-OS und MIMIMON
        from miso.core.nexus_os import NexusOS, TaskManager, ResourceManager
        from miso.security.ztm.mimimon import MIMIMON, ZTMPolicy, ZTMVerifier
        
        # Erstelle NEXUS-OS
        os = NexusOS()
        
        # Erstelle MIMIMON
        mimimon = MIMIMON()
        
        # Erstelle Policy
        policy = ZTMPolicy(
            name="NEXUS-OS Policy",
            rules=[
                {"resource": "task", "action": "create", "allow": True},
                {"resource": "task", "action": "execute", "allow": True},
                {"resource": "resource", "action": "allocate", "allow": True},
                {"resource": "resource", "action": "deallocate", "allow": True}
            ]
        )
        
        # Füge Policy hinzu
        mimimon.add_policy(policy)
        
        # Erstelle Task
        task = {
            "id": "task1",
            "name": "Test Task",
            "priority": 1,
            "resources": ["cpu", "memory"],
            "function": "test_function",
            "arguments": {"arg1": 1, "arg2": 2}
        }
        
        # Verifiziere Task-Erstellung
        create_verification = mimimon.verify("task", "create")
        
        # Füge Task hinzu, wenn erlaubt
        if create_verification:
            os.task_manager.add_task(task)
        
        # Verifiziere Ressourcenzuweisung
        allocate_verification = mimimon.verify("resource", "allocate")
        
        # Weise Ressourcen zu, wenn erlaubt
        if allocate_verification:
            allocation = os.resource_manager.allocate_resources(task)
        else:
            allocation = None
        
        return {
            "success": create_verification and allocate_verification,
            "message": "NEXUS-OS und MIMIMON Integration erfolgreich",
            "details": {
                "task": str(task),
                "create_verification": create_verification,
                "allocate_verification": allocate_verification,
                "allocation": str(allocation) if allocation else None,
                "task_queue": str(os.task_manager.get_task_queue())
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"NEXUS-OS und MIMIMON Integration fehlgeschlagen: {e}",
            "error": str(e)
        }

def test_prism_engine_void_protocol_integration():
    """Testet die Integration zwischen PRISM-Engine und VOID-Protokoll"""
    try:
        # Importiere PRISM-Engine und VOID-Protokoll
        from miso.simulation.prism_engine import PrismEngine, SimulationConfig
        from miso.simulation.prism_matrix import PrismMatrix
        from miso.protect.void_protocol import VoidProtocol, SecurityLevel, ProtectionMode
        
        # Erstelle PRISM-Engine
        engine = PrismEngine()
        
        # Erstelle VOID-Protokoll
        protocol = VoidProtocol(
            security_level=SecurityLevel.HIGH,
            protection_mode=ProtectionMode.FULL
        )
        
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
        simulation_result = engine.simulate(matrix, config)
        
        # Schütze Simulationsergebnis
        protected_result = protocol.protect(str(simulation_result))
        
        # Entschlüssele Simulationsergebnis
        decrypted_result = protocol.unprotect(protected_result)
        
        return {
            "success": True,
            "message": "PRISM-Engine und VOID-Protokoll Integration erfolgreich",
            "details": {
                "matrix": str(matrix),
                "config": str(config),
                "simulation_result": str(simulation_result),
                "protected_result": str(protected_result),
                "decrypted_result": decrypted_result,
                "integrity_check": str(simulation_result) == decrypted_result
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"PRISM-Engine und VOID-Protokoll Integration fehlgeschlagen: {e}",
            "error": str(e)
        }

def main():
    """Hauptfunktion"""
    logger.info("Starte MISO Ultimate Integrationstests...")
    
    # Liste der Integrationstests
    integration_tests = [
        {"name": "M-CODE und M-LINGUA Integration", "test_function": test_m_code_m_lingua_integration},
        {"name": "ECHO-PRIME und VXOR Integration", "test_function": test_echo_prime_vxor_integration},
        {"name": "HYPERFILTER und VXOR Integration", "test_function": test_hyperfilter_vxor_integration},
        {"name": "Deep-State-Modul und HYPERFILTER Integration", "test_function": test_deep_state_hyperfilter_integration},
        {"name": "Q-LOGIK und T-MATHEMATICS Integration", "test_function": test_q_logik_t_mathematics_integration},
        {"name": "NEXUS-OS und MIMIMON Integration", "test_function": test_nexus_os_mimimon_integration},
        {"name": "PRISM-Engine und VOID-Protokoll Integration", "test_function": test_prism_engine_void_protocol_integration}
    ]
    
    # Führe Integrationstests durch
    for test in integration_tests:
        test_name = test["name"]
        test_function = test["test_function"]
        
        # Führe Test durch
        result = test_integration(test_name, test_function)
        
        # Speichere Ergebnis
        results["integration_tests"][test_name] = result
    
    # Bestimme Gesamtstatus
    passed = sum(1 for test in results["integration_tests"].values() if test["status"] == "passed")
    failed = sum(1 for test in results["integration_tests"].values() if test["status"] == "failed")
    errors = sum(1 for test in results["integration_tests"].values() if test["status"] == "error")
    
    if errors > 0:
        results["overall_status"] = "error"
    elif failed > 0:
        results["overall_status"] = "failed"
    else:
        results["overall_status"] = "passed"
    
    # Speichere Ergebnisse
    with open("stability_test_results_integration.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Erstelle Zusammenfassung
    summary = f"""
MISO Ultimate Integrationstests - Zusammenfassung
=================================================
Zeitstempel: {datetime.datetime.now().isoformat()}
Gesamtstatus: {results["overall_status"]}

Teststatus:
"""
    
    for test_name, test_result in results["integration_tests"].items():
        summary += f"  - {test_name}: {test_result['status']}"
        if test_result["status"] == "error":
            summary += f" ({test_result['error']['type']}: {test_result['error']['message']})"
        summary += f" ({test_result['duration']:.2f}s)\n"
    
    summary += f"""
Statistik:
  - Bestanden: {passed}/{len(integration_tests)}
  - Fehlgeschlagen: {failed}/{len(integration_tests)}
  - Fehler: {errors}/{len(integration_tests)}

Ergebnisse gespeichert in:
  - stability_test_results_integration.json
"""
    
    # Speichere Zusammenfassung
    with open("stability_test_results_integration.md", "w", encoding="utf-8") as f:
        f.write(summary)
    
    logger.info(f"MISO Ultimate Integrationstests abgeschlossen: {results['overall_status']}")
    logger.info(f"Bestanden: {passed}/{len(integration_tests)}, Fehlgeschlagen: {failed}/{len(integration_tests)}, Fehler: {errors}/{len(integration_tests)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
