#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate Verifikationsskript

Dieses Skript führt umfassende Tests für alle MISO Ultimate Module durch
und erstellt einen detaillierten Bericht über den aktuellen Implementierungsstand.

Copyright (c) 2025 MISO Team. All rights reserved.
"""

import os
import sys
import time
import logging
import json
from datetime import datetime
from pathlib import Path
import importlib.util
import numpy as np

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("miso_ultimate_verification.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("miso_ultimate_verification")

# Basis-Verzeichnis
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Liste der zu prüfenden Module
MODULES = [
    "Omega-Kern 4.0",
    "M-CODE Core",
    "M-LINGUA Interface",
    "MPRIME Mathematikmodul",
    "Q-LOGIK",
    "PRISM-Engine",
    "VOID-Protokoll 3.0",
    "NEXUS-OS",
    "ECHO-PRIME",
    "HYPERFILTER",
    "Deep-State-Modul",
    "T-MATHEMATICS ENGINE",
    "MIMIMON: ZTM-Modul"
]

# Modulpfade
MODULE_PATHS = {
    "Omega-Kern 4.0": ["miso/core/omega_core.py"],
    "M-CODE Core": ["miso/lang/mcode", "lang/mcode", "miso/code/m_code"],
    "M-LINGUA Interface": ["miso/lang/mlingua", "lang/mlingua", "miso/lang/mcode/m_code_bridge.py"],
    "MPRIME Mathematikmodul": [
        "miso/math/mprime/symbol_solver.py",
        "miso/math/mprime/topo_matrix.py",
        "miso/math/mprime/babylon_logic.py",
        "miso/math/mprime/prob_mapper.py",
        "miso/math/mprime/formula_builder.py",
        "miso/math/mprime/prime_resolver.py",
        "miso/math/mprime/contextual_math.py"
    ],
    "Q-LOGIK": ["miso/logic/qlogik_engine.py", "miso/logic/qlogik_integration.py"],
    "PRISM-Engine": ["miso/simulation/prism_engine.py", "miso/simulation/prism_matrix.py"],
    "VOID-Protokoll 3.0": ["miso/protect/void_protocol.py"],
    "NEXUS-OS": ["miso/core/nexus_os"],
    "ECHO-PRIME": [
        "miso/timeline/qtm_modulator.py",
        "miso/timeline/temporal_integrity_guard.py",
        "miso/timeline/trigger_matrix_analyzer.py",
        "engines/echo_prime"
    ],
    "HYPERFILTER": ["miso/filter/hyperfilter.py", "miso/filter/vxor_hyperfilter_integration.py"],
    "Deep-State-Modul": ["miso/analysis/deep_state"],
    "T-MATHEMATICS ENGINE": [
        "engines/t_mathematics/tensor.py",
        "engines/t_mathematics/tensor_torch.py",
        "tensor.py",
        "tensor_factory.py",
        "tensor_mlx.py",
        "tensor_numpy.py",
        "tensor_operations.py",
        "tensor_torch.py"
    ],
    "MIMIMON: ZTM-Modul": ["miso/security/ztm"]
}

class ModuleVerifier:
    """Klasse zur Überprüfung der MISO-Module"""
    
    def __init__(self):
        """Initialisiert den ModuleVerifier"""
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "modules": {},
            "overall_status": "pending"
        }
    
    def check_file_exists(self, file_path):
        """Prüft, ob eine Datei existiert"""
        full_path = os.path.join(BASE_DIR, file_path)
        return os.path.exists(full_path)
    
    def check_directory_exists(self, dir_path):
        """Prüft, ob ein Verzeichnis existiert"""
        full_path = os.path.join(BASE_DIR, dir_path)
        return os.path.isdir(full_path)
    
    def verify_module(self, module_name):
        """Überprüft ein einzelnes Modul"""
        logger.info(f"Überprüfe Modul: {module_name}")
        
        paths = MODULE_PATHS.get(module_name, [])
        if not paths:
            logger.warning(f"Keine Pfade für Modul {module_name} definiert")
            self.results["modules"][module_name] = {
                "status": "unknown",
                "files_found": 0,
                "total_files": 0,
                "details": "Keine Pfade definiert"
            }
            return
        
        files_found = 0
        total_files = len(paths)
        details = []
        
        for path in paths:
            if path.endswith(".py"):
                # Einzelne Python-Datei
                if self.check_file_exists(path):
                    files_found += 1
                    details.append(f"Datei gefunden: {path}")
                else:
                    details.append(f"Datei nicht gefunden: {path}")
            else:
                # Verzeichnis
                if self.check_directory_exists(path):
                    files_found += 1
                    details.append(f"Verzeichnis gefunden: {path}")
                else:
                    details.append(f"Verzeichnis nicht gefunden: {path}")
        
        # Bestimme Status
        if files_found == 0:
            status = "not_implemented"
        elif files_found < total_files:
            status = "partially_implemented"
        else:
            status = "implemented"
        
        self.results["modules"][module_name] = {
            "status": status,
            "files_found": files_found,
            "total_files": total_files,
            "details": details
        }
        
        logger.info(f"Modul {module_name}: {status} ({files_found}/{total_files})")
    
    def verify_all_modules(self):
        """Überprüft alle Module"""
        for module in MODULES:
            self.verify_module(module)
        
        # Bestimme Gesamtstatus
        implemented = 0
        partially_implemented = 0
        not_implemented = 0
        
        for module, data in self.results["modules"].items():
            if data["status"] == "implemented":
                implemented += 1
            elif data["status"] == "partially_implemented":
                partially_implemented += 1
            elif data["status"] == "not_implemented":
                not_implemented += 1
        
        if not_implemented == len(MODULES):
            self.results["overall_status"] = "not_implemented"
        elif implemented == len(MODULES):
            self.results["overall_status"] = "fully_implemented"
        else:
            self.results["overall_status"] = "partially_implemented"
        
        self.results["implementation_stats"] = {
            "implemented": implemented,
            "partially_implemented": partially_implemented,
            "not_implemented": not_implemented,
            "total": len(MODULES)
        }
        
        return self.results
    
    def save_results(self):
        """Speichert die Ergebnisse in einer JSON-Datei"""
        filename = f"miso_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Ergebnisse gespeichert in: {filename}")
        
        # Erstelle auch eine Zusammenfassung als Textdatei
        summary_filename = f"miso_verification_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_filename, 'w') as f:
            f.write("MISO Ultimate Verifikation - Zusammenfassung\n")
            f.write("==========================================\n\n")
            f.write(f"Zeitstempel: {self.results['timestamp']}\n")
            f.write(f"Gesamtstatus: {self.results['overall_status']}\n\n")
            
            f.write("Modulstatus:\n")
            for module, data in self.results["modules"].items():
                f.write(f"  - {module}: {data['status']} ({data['files_found']}/{data['total_files']})\n")
            
            f.write("\nImplementierungsstatistik:\n")
            stats = self.results["implementation_stats"]
            f.write(f"  - Vollständig implementiert: {stats['implemented']}/{stats['total']}\n")
            f.write(f"  - Teilweise implementiert: {stats['partially_implemented']}/{stats['total']}\n")
            f.write(f"  - Nicht implementiert: {stats['not_implemented']}/{stats['total']}\n")
        
        logger.info(f"Zusammenfassung gespeichert in: {summary_filename}")
        
        return filename, summary_filename

def verify_t_mathematics():
    """Führt spezifische Tests für die T-Mathematics Engine durch"""
    logger.info("Führe spezifische Tests für T-Mathematics Engine durch...")
    
    results = {
        "status": "pending",
        "tests": {}
    }
    
    # Prüfe, ob die Tensor-Dateien existieren
    tensor_files = [
        "engines/t_mathematics/tensor.py",
        "engines/t_mathematics/tensor_torch.py",
        "tensor.py",
        "tensor_factory.py",
        "tensor_mlx.py",
        "tensor_numpy.py",
        "tensor_operations.py",
        "tensor_torch.py"
    ]
    
    files_found = 0
    for file in tensor_files:
        if os.path.exists(os.path.join(BASE_DIR, file)):
            files_found += 1
    
    results["tests"]["files_exist"] = {
        "status": "passed" if files_found == len(tensor_files) else "failed",
        "details": f"{files_found}/{len(tensor_files)} Dateien gefunden"
    }
    
    # Versuche, die T-Mathematics Engine direkt zu importieren
    try:
        # Füge das Basis-Verzeichnis zum Pfad hinzu
        sys.path.insert(0, BASE_DIR)
        
        # Versuche, die Engine zu importieren
        from engines.t_mathematics.engine import TMathematicsEngine
        
        # Prüfe, ob die Engine initialisiert werden kann
        engine = TMathematicsEngine({"precision": "float32", "use_symbolic": False})
        
        results["tests"]["engine_init"] = {
            "status": "passed",
            "details": "TMathematicsEngine erfolgreich initialisiert"
        }
        
        # Prüfe, ob wir die Tensor-Klassen importieren können
        try:
            from engines.t_mathematics.tensor import MISOTensor
            results["tests"]["miso_tensor_class"] = {
                "status": "passed",
                "details": "MISOTensor-Klasse gefunden"
            }
        except ImportError:
            results["tests"]["miso_tensor_class"] = {
                "status": "failed",
                "details": "MISOTensor-Klasse nicht gefunden"
            }
        
        # Prüfe MLX-Unterstützung
        try:
            # Versuche, MLX zu importieren
            import mlx.core
            results["tests"]["mlx_support"] = {
                "status": "passed",
                "details": "MLX-Bibliothek gefunden"
            }
        except ImportError:
            results["tests"]["mlx_support"] = {
                "status": "failed",
                "details": "MLX-Bibliothek nicht gefunden"
            }
        
        # Prüfe PyTorch-Unterstützung
        try:
            # Versuche, PyTorch zu importieren
            import torch
            results["tests"]["torch_support"] = {
                "status": "passed",
                "details": "PyTorch-Bibliothek gefunden"
            }
        except ImportError:
            results["tests"]["torch_support"] = {
                "status": "failed",
                "details": "PyTorch-Bibliothek nicht gefunden"
            }
        
        # Prüfe, ob die Engine Tensoren erstellen kann
        try:
            tensor = engine.create_tensor([[1, 2], [3, 4]])
            results["tests"]["tensor_creation"] = {
                "status": "passed",
                "details": "Tensor erfolgreich erstellt"
            }
        except Exception as e:
            results["tests"]["tensor_creation"] = {
                "status": "failed",
                "details": f"Fehler beim Erstellen eines Tensors: {e}"
            }
        
    except Exception as e:
        logger.error(f"Fehler beim Importieren der Tensor-Module: {e}")
        results["tests"]["import_modules"] = {
            "status": "failed",
            "details": str(e)
        }
    
    # Bestimme Gesamtstatus
    failed_tests = sum(1 for test in results["tests"] if results["tests"][test]["status"] == "failed")
    if failed_tests == 0:
        results["status"] = "passed"
    elif failed_tests < len(results["tests"]):
        results["status"] = "partially_passed"
    else:
        results["status"] = "failed"
    
    return results

def verify_echo_prime():
    """Führt spezifische Tests für ECHO-PRIME durch"""
    logger.info("Führe spezifische Tests für ECHO-PRIME durch...")
    
    results = {
        "status": "pending",
        "tests": {}
    }
    
    # Prüfe, ob die ECHO-PRIME-Dateien existieren
    echo_prime_files = [
        "miso/timeline/qtm_modulator.py",
        "miso/timeline/temporal_integrity_guard.py",
        "miso/timeline/trigger_matrix_analyzer.py"
    ]
    
    files_found = 0
    for file in echo_prime_files:
        if os.path.exists(os.path.join(BASE_DIR, file)):
            files_found += 1
    
    results["tests"]["files_exist"] = {
        "status": "passed" if files_found == len(echo_prime_files) else "failed",
        "details": f"{files_found}/{len(echo_prime_files)} Dateien gefunden"
    }
    
    # Versuche, die ECHO-PRIME-Klassen zu importieren
    try:
        sys.path.insert(0, BASE_DIR)
        
        # Versuche, qtm_modulator.py zu importieren
        spec = importlib.util.spec_from_file_location("qtm_modulator", 
                                                     os.path.join(BASE_DIR, "miso/timeline/qtm_modulator.py"))
        qtm_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qtm_module)
        
        # Prüfe, ob QTM_Modulator existiert
        if hasattr(qtm_module, "QTM_Modulator"):
            results["tests"]["qtm_modulator_class"] = {
                "status": "passed",
                "details": "QTM_Modulator-Klasse gefunden"
            }
        else:
            results["tests"]["qtm_modulator_class"] = {
                "status": "failed",
                "details": "QTM_Modulator-Klasse nicht gefunden"
            }
        
        # Versuche, temporal_integrity_guard.py zu importieren
        spec = importlib.util.spec_from_file_location("temporal_integrity_guard", 
                                                     os.path.join(BASE_DIR, "miso/timeline/temporal_integrity_guard.py"))
        tig_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tig_module)
        
        # Prüfe, ob TemporalIntegrityGuard existiert
        if hasattr(tig_module, "TemporalIntegrityGuard"):
            results["tests"]["temporal_integrity_guard_class"] = {
                "status": "passed",
                "details": "TemporalIntegrityGuard-Klasse gefunden"
            }
        else:
            results["tests"]["temporal_integrity_guard_class"] = {
                "status": "failed",
                "details": "TemporalIntegrityGuard-Klasse nicht gefunden"
            }
        
        # Versuche, trigger_matrix_analyzer.py zu importieren
        spec = importlib.util.spec_from_file_location("trigger_matrix_analyzer", 
                                                     os.path.join(BASE_DIR, "miso/timeline/trigger_matrix_analyzer.py"))
        tma_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tma_module)
        
        # Prüfe, ob TriggerMatrixAnalyzer existiert
        if hasattr(tma_module, "TriggerMatrixAnalyzer"):
            results["tests"]["trigger_matrix_analyzer_class"] = {
                "status": "passed",
                "details": "TriggerMatrixAnalyzer-Klasse gefunden"
            }
        else:
            results["tests"]["trigger_matrix_analyzer_class"] = {
                "status": "failed",
                "details": "TriggerMatrixAnalyzer-Klasse nicht gefunden"
            }
        
    except Exception as e:
        logger.error(f"Fehler beim Importieren der ECHO-PRIME-Module: {e}")
        results["tests"]["import_modules"] = {
            "status": "failed",
            "details": str(e)
        }
    
    # Bestimme Gesamtstatus
    failed_tests = sum(1 for test in results["tests"] if results["tests"][test]["status"] == "failed")
    if failed_tests == 0:
        results["status"] = "passed"
    elif failed_tests < len(results["tests"]):
        results["status"] = "partially_passed"
    else:
        results["status"] = "failed"
    
    return results

def main():
    """Hauptfunktion"""
    logger.info("Starte MISO Ultimate Verifikation...")
    
    # Überprüfe alle Module
    verifier = ModuleVerifier()
    results = verifier.verify_all_modules()
    
    # Führe spezifische Tests für T-Mathematics durch
    t_math_results = verify_t_mathematics()
    results["t_mathematics_tests"] = t_math_results
    
    # Führe spezifische Tests für ECHO-PRIME durch
    echo_prime_results = verify_echo_prime()
    results["echo_prime_tests"] = echo_prime_results
    
    # Speichere Ergebnisse
    json_file, summary_file = verifier.save_results()
    
    # Gebe Zusammenfassung aus
    print("\nMISO Ultimate Verifikation - Zusammenfassung")
    print("==========================================")
    print(f"Gesamtstatus: {results['overall_status']}")
    print("\nModulstatus:")
    
    for module, data in results["modules"].items():
        print(f"  - {module}: {data['status']} ({data['files_found']}/{data['total_files']})")
    
    print("\nImplementierungsstatistik:")
    stats = results["implementation_stats"]
    print(f"  - Vollständig implementiert: {stats['implemented']}/{stats['total']}")
    print(f"  - Teilweise implementiert: {stats['partially_implemented']}/{stats['total']}")
    print(f"  - Nicht implementiert: {stats['not_implemented']}/{stats['total']}")
    
    print(f"\nT-Mathematics Tests: {t_math_results['status']}")
    print(f"ECHO-PRIME Tests: {echo_prime_results['status']}")
    
    print(f"\nErgebnisse gespeichert in:")
    print(f"  - {json_file}")
    print(f"  - {summary_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
