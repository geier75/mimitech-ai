#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - VXOR Integration Test

Dieses Skript testet die Integration zwischen MISO und den VXOR-Modulen,
insbesondere die Verbindung zur T-MATHEMATICS Engine und anderen Kernkomponenten.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import json
from pathlib import Path
from datetime import datetime

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MISO.VXORIntegrationTest")

# Prüfe, ob Apple Silicon verfügbar ist
is_apple_silicon = sys.platform == 'darwin' and 'arm' in os.uname().machine
logger.info(f"Apple Silicon: {is_apple_silicon}")

class VXORIntegrationTester:
    """Testet die Integration zwischen MISO und VXOR-Modulen"""
    
    def __init__(self):
        """Initialisiert den VXOR Integration Tester"""
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0
            }
        }
        
        # Lade VXOR-Manifest
        self.manifest = self._load_vxor_manifest()
        
        logger.info("VXOR Integration Tester initialisiert")
    
    def _load_vxor_manifest(self):
        """Lädt das VXOR-Manifest"""
        manifest_paths = [
            Path("vXor_Modules/vxor_manifest.json"),
            Path("miso/vxor/vxor_manifest.json")
        ]
        
        for path in manifest_paths:
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        manifest = json.load(f)
                    logger.info(f"VXOR-Manifest aus {path} geladen")
                    return manifest
                except Exception as e:
                    logger.error(f"Fehler beim Laden des VXOR-Manifests aus {path}: {e}")
        
        logger.warning("Kein VXOR-Manifest gefunden, verwende leeres Manifest")
        return {"modules": {"implemented": {}, "planned": {}}}
    
    def test_vxor_t_mathematics_integration(self):
        """Testet die Integration zwischen VXOR und T-MATHEMATICS"""
        test_name = "vxor_t_mathematics_integration"
        logger.info(f"Starte Test: {test_name}")
        
        try:
            # Importiere die erforderlichen Module
            from miso.math.t_mathematics.vxor_math_integration import VXORMathIntegration
            from miso.vXor_Modules.vxor_t_mathematics_bridge import get_vxor_t_math_bridge
            
            # Hole die VXOR-T-MATHEMATICS-Brücke
            vxor_bridge = get_vxor_t_math_bridge()
            
            # Prüfe, ob die Brücke korrekt initialisiert wurde
            if vxor_bridge is None:
                raise ValueError("VXOR-T-MATHEMATICS-Brücke konnte nicht initialisiert werden")
            
            # Prüfe, ob die Brücke die erforderlichen Methoden hat
            required_methods = ["get_engine", "register_module", "get_module_integration"]
            for method in required_methods:
                if not hasattr(vxor_bridge, method):
                    raise ValueError(f"VXOR-T-MATHEMATICS-Brücke hat keine Methode '{method}'")
            
            # Teste die Brücke mit einem einfachen Tensor
            test_tensor = [1.0, 2.0, 3.0, 4.0]
            result = vxor_bridge.test_tensor_operation(test_tensor)
            
            # Prüfe das Ergebnis
            if result is None or not isinstance(result, dict) or "success" not in result:
                raise ValueError(f"Ungültiges Ergebnis von test_tensor_operation: {result}")
            
            if not result.get("success", False):
                raise ValueError(f"Tensor-Operation fehlgeschlagen: {result.get('error', 'Unbekannter Fehler')}")
            
            logger.info(f"Test {test_name} erfolgreich")
            self.results["tests"][test_name] = {
                "status": "passed",
                "message": "VXOR-T-MATHEMATICS-Integration funktioniert korrekt"
            }
            self.results["summary"]["passed"] += 1
        except ImportError as e:
            logger.error(f"Fehler beim Importieren der Module: {e}")
            self.results["tests"][test_name] = {
                "status": "error",
                "message": f"Module konnten nicht importiert werden: {e}"
            }
            self.results["summary"]["errors"] += 1
        except Exception as e:
            logger.error(f"Fehler beim Test {test_name}: {e}")
            self.results["tests"][test_name] = {
                "status": "failed",
                "message": str(e)
            }
            self.results["summary"]["failed"] += 1
        
        self.results["summary"]["total"] += 1
    
    def test_hyperfilter_math_engine(self):
        """Testet die HyperfilterMathEngine"""
        test_name = "hyperfilter_math_engine"
        logger.info(f"Starte Test: {test_name}")
        
        try:
            # Importiere die erforderlichen Module
            from miso.vXor_Modules.hyperfilter_t_mathematics import get_hyperfilter_math_engine
            
            # Hole die HyperfilterMathEngine
            hyperfilter_engine = get_hyperfilter_math_engine()
            
            # Prüfe, ob die Engine korrekt initialisiert wurde
            if hyperfilter_engine is None:
                raise ValueError("HyperfilterMathEngine konnte nicht initialisiert werden")
            
            # Prüfe, ob die Engine die erforderlichen Methoden hat
            required_methods = ["analyze_text_embedding", "detect_sentiment", "normalize_context"]
            for method in required_methods:
                if not hasattr(hyperfilter_engine, method):
                    raise ValueError(f"HyperfilterMathEngine hat keine Methode '{method}'")
            
            # Teste die Engine mit einfachen Embeddings
            text_embedding = [0.1, 0.2, 0.3, 0.4]
            reference_embeddings = [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]
            trust_scores = [0.8, 0.6]
            
            result = hyperfilter_engine.analyze_text_embedding(
                text_embedding, reference_embeddings, trust_scores
            )
            
            # Prüfe das Ergebnis
            if result is None or not isinstance(result, dict) or "trust_score" not in result:
                raise ValueError(f"Ungültiges Ergebnis von analyze_text_embedding: {result}")
            
            logger.info(f"Test {test_name} erfolgreich")
            self.results["tests"][test_name] = {
                "status": "passed",
                "message": "HyperfilterMathEngine funktioniert korrekt"
            }
            self.results["summary"]["passed"] += 1
        except ImportError as e:
            logger.error(f"Fehler beim Importieren der Module: {e}")
            self.results["tests"][test_name] = {
                "status": "error",
                "message": f"Module konnten nicht importiert werden: {e}"
            }
            self.results["summary"]["errors"] += 1
        except Exception as e:
            logger.error(f"Fehler beim Test {test_name}: {e}")
            self.results["tests"][test_name] = {
                "status": "failed",
                "message": str(e)
            }
            self.results["summary"]["failed"] += 1
        
        self.results["summary"]["total"] += 1
    
    def test_vxor_manifest(self):
        """Testet das VXOR-Manifest"""
        test_name = "vxor_manifest"
        logger.info(f"Starte Test: {test_name}")
        
        try:
            # Prüfe, ob das Manifest die erforderlichen Schlüssel hat
            required_keys = ["manifest_version", "modules"]
            for key in required_keys:
                if key not in self.manifest:
                    raise ValueError(f"VXOR-Manifest hat keinen Schlüssel '{key}'")
            
            # Prüfe, ob das Manifest implementierte Module enthält
            if "implemented" not in self.manifest["modules"]:
                raise ValueError("VXOR-Manifest enthält keine implementierten Module")
            
            # Prüfe die implementierten Module
            implemented_modules = self.manifest["modules"]["implemented"]
            if not implemented_modules:
                raise ValueError("VXOR-Manifest enthält keine implementierten Module")
            
            # Liste der implementierten Module
            module_list = list(implemented_modules.keys())
            logger.info(f"Implementierte VXOR-Module: {', '.join(module_list)}")
            
            # Prüfe, ob VX-MEMEX und VX-SELFWRITER implementiert sind
            if "VX-MEMEX" not in implemented_modules:
                raise ValueError("VX-MEMEX ist nicht als implementiertes Modul im Manifest aufgeführt")
            
            if "VX-SELFWRITER" not in implemented_modules:
                raise ValueError("VX-SELFWRITER ist nicht als implementiertes Modul im Manifest aufgeführt")
            
            logger.info(f"Test {test_name} erfolgreich")
            self.results["tests"][test_name] = {
                "status": "passed",
                "message": "VXOR-Manifest ist korrekt",
                "implemented_modules": module_list
            }
            self.results["summary"]["passed"] += 1
        except Exception as e:
            logger.error(f"Fehler beim Test {test_name}: {e}")
            self.results["tests"][test_name] = {
                "status": "failed",
                "message": str(e)
            }
            self.results["summary"]["failed"] += 1
        
        self.results["summary"]["total"] += 1
    
    def test_vxor_echo_integration(self):
        """Testet die Integration zwischen VXOR und ECHO PRIME"""
        test_name = "vxor_echo_integration"
        logger.info(f"Starte Test: {test_name}")
        
        try:
            # Importiere die erforderlichen Module
            from miso.timeline.vxor_echo_integration import VXOREchoIntegration
            from miso.paradox.echo_prime import get_echo_prime_instance
            
            # Hole eine Instanz von ECHO PRIME
            echo_prime = get_echo_prime_instance()
            
            # Prüfe, ob ECHO PRIME korrekt initialisiert wurde
            if echo_prime is None:
                raise ValueError("ECHO PRIME konnte nicht initialisiert werden")
            
            # Erstelle eine VXOREchoIntegration-Instanz
            vxor_echo = VXOREchoIntegration(echo_prime)
            
            # Prüfe, ob die Integration korrekt initialisiert wurde
            if vxor_echo is None:
                raise ValueError("VXOREchoIntegration konnte nicht initialisiert werden")
            
            # Prüfe, ob die Integration die erforderlichen Methoden hat
            required_methods = ["register_timeline", "process_paradox", "get_timeline_status"]
            for method in required_methods:
                if not hasattr(vxor_echo, method):
                    raise ValueError(f"VXOREchoIntegration hat keine Methode '{method}'")
            
            logger.info(f"Test {test_name} erfolgreich")
            self.results["tests"][test_name] = {
                "status": "passed",
                "message": "VXOR-ECHO-Integration funktioniert korrekt"
            }
            self.results["summary"]["passed"] += 1
        except ImportError as e:
            logger.error(f"Fehler beim Importieren der Module: {e}")
            self.results["tests"][test_name] = {
                "status": "error",
                "message": f"Module konnten nicht importiert werden: {e}"
            }
            self.results["summary"]["errors"] += 1
        except Exception as e:
            logger.error(f"Fehler beim Test {test_name}: {e}")
            self.results["tests"][test_name] = {
                "status": "failed",
                "message": str(e)
            }
            self.results["summary"]["failed"] += 1
        
        self.results["summary"]["total"] += 1
    
    def test_vxor_prism_integration(self):
        """Testet die Integration zwischen VXOR und PRISM"""
        test_name = "vxor_prism_integration"
        logger.info(f"Starte Test: {test_name}")
        
        try:
            # Importiere die erforderlichen Module
            from miso.simulation.vxor_prism_integration import VXORPrismIntegration
            from miso.simulation.prism_engine import PrismEngine
            
            # Erstelle eine PrismEngine-Instanz
            prism_engine = PrismEngine()
            
            # Erstelle eine VXORPrismIntegration-Instanz
            vxor_prism = VXORPrismIntegration(prism_engine)
            
            # Prüfe, ob die Integration korrekt initialisiert wurde
            if vxor_prism is None:
                raise ValueError("VXORPrismIntegration konnte nicht initialisiert werden")
            
            # Prüfe, ob die Integration die erforderlichen Methoden hat
            required_methods = ["register_simulation", "process_matrix", "get_simulation_status"]
            for method in required_methods:
                if not hasattr(vxor_prism, method):
                    raise ValueError(f"VXORPrismIntegration hat keine Methode '{method}'")
            
            logger.info(f"Test {test_name} erfolgreich")
            self.results["tests"][test_name] = {
                "status": "passed",
                "message": "VXOR-PRISM-Integration funktioniert korrekt"
            }
            self.results["summary"]["passed"] += 1
        except ImportError as e:
            logger.error(f"Fehler beim Importieren der Module: {e}")
            self.results["tests"][test_name] = {
                "status": "error",
                "message": f"Module konnten nicht importiert werden: {e}"
            }
            self.results["summary"]["errors"] += 1
        except Exception as e:
            logger.error(f"Fehler beim Test {test_name}: {e}")
            self.results["tests"][test_name] = {
                "status": "failed",
                "message": str(e)
            }
            self.results["summary"]["failed"] += 1
        
        self.results["summary"]["total"] += 1
    
    def run_all_tests(self):
        """Führt alle Tests aus"""
        logger.info("Starte alle Tests")
        
        # Führe alle Tests aus
        self.test_vxor_manifest()
        self.test_vxor_t_mathematics_integration()
        self.test_hyperfilter_math_engine()
        self.test_vxor_echo_integration()
        self.test_vxor_prism_integration()
        
        # Aktualisiere die Zusammenfassung
        self.results["summary"]["total"] = len(self.results["tests"])
        self.results["summary"]["passed"] = sum(1 for test in self.results["tests"].values() if test["status"] == "passed")
        self.results["summary"]["failed"] = sum(1 for test in self.results["tests"].values() if test["status"] == "failed")
        self.results["summary"]["errors"] = sum(1 for test in self.results["tests"].values() if test["status"] == "error")
        
        # Speichere die Ergebnisse
        with open("vxor_integration_test_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Alle Tests abgeschlossen: {self.results['summary']['passed']} von {self.results['summary']['total']} Tests erfolgreich")
        
        return self.results

def main():
    """Hauptfunktion"""
    logger.info("=== MISO Ultimate VXOR Integration Test ===")
    
    try:
        # Initialisiere den Tester
        tester = VXORIntegrationTester()
        
        # Führe alle Tests aus
        results = tester.run_all_tests()
        
        # Gib die Ergebnisse aus
        print("\n=== Testergebnisse ===")
        print(f"Gesamt: {results['summary']['total']}")
        print(f"Erfolgreich: {results['summary']['passed']}")
        print(f"Fehlgeschlagen: {results['summary']['failed']}")
        print(f"Fehler: {results['summary']['errors']}")
        print("\nDetails:")
        
        for test_name, test_result in results["tests"].items():
            status = test_result["status"]
            message = test_result["message"]
            
            if status == "passed":
                print(f"✅ {test_name}: {message}")
            elif status == "failed":
                print(f"❌ {test_name}: {message}")
            else:
                print(f"⚠️ {test_name}: {message}")
        
        # Gib den Exit-Code basierend auf den Testergebnissen zurück
        if results["summary"]["failed"] > 0 or results["summary"]["errors"] > 0:
            return 1
        else:
            return 0
    except Exception as e:
        logger.error(f"Fehler bei der Ausführung der Tests: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
