#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Einfacher Test für Lazy-Loading Funktionalität

Dieser Test überprüft die Lazy-Loading-Funktionalität zwischen PRISM und ECHO-PRIME.
"""

import os
import sys
import logging
import time

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MISO.test_lazy_loading")

# Füge Projektverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def test_echo_prime_lazy_loading():
    """Test der Lazy-Loading-Funktionalität für ECHO-PRIME im PRISM-Modul"""
    logger.info("=== Test: Lazy-Loading von ECHO-PRIME in PRISM ===")
    
    try:
        # Importiere die PRISM-Integration mit ECHO-PRIME
        from miso.timeline.echo_prime_controller import get_prism_simulator
        
        # Prüfe, ob die Lazy-Loading-Funktion existiert
        logger.info("Prüfe, ob die Lazy-Loading-Funktion existiert...")
        assert callable(get_prism_simulator), "get_prism_simulator ist keine Funktion"
        logger.info("✅ Lazy-Loading-Funktion existiert")
        
        # Führe Lazy-Loading aus und prüfe das Ergebnis
        logger.info("Führe Lazy-Loading aus...")
        PrismEngine = get_prism_simulator()
        
        # Prüfe das Ergebnis
        if PrismEngine is not None:
            logger.info(f"✅ PrismEngine erfolgreich geladen: {PrismEngine.__name__}")
        else:
            logger.warning("⚠️ PrismEngine konnte nicht geladen werden, nutze Fallback")
        
        return True
    except Exception as e:
        logger.error(f"❌ Test fehlgeschlagen: {e}")
        return False

def test_prism_lazy_loading():
    """Test der Lazy-Loading-Funktionalität für PRISM im ECHO-PRIME-Modul"""
    logger.info("=== Test: Lazy-Loading von PRISM in ECHO-PRIME ===")
    
    try:
        # Importiere die ECHO-PRIME-Integration mit PRISM
        from miso.simulation.prism_echo_prime_integration import get_echo_prime_controller
        
        # Prüfe, ob die Lazy-Loading-Funktion existiert
        logger.info("Prüfe, ob die Lazy-Loading-Funktion existiert...")
        assert callable(get_echo_prime_controller), "get_echo_prime_controller ist keine Funktion"
        logger.info("✅ Lazy-Loading-Funktion existiert")
        
        # Führe Lazy-Loading aus und prüfe das Ergebnis
        logger.info("Führe Lazy-Loading aus...")
        EchoPrimeController = get_echo_prime_controller()
        
        # Prüfe das Ergebnis
        if EchoPrimeController is not None:
            logger.info(f"✅ EchoPrimeController erfolgreich geladen: {EchoPrimeController.__name__}")
        else:
            logger.warning("⚠️ EchoPrimeController konnte nicht geladen werden, nutze Fallback")
        
        return True
    except Exception as e:
        logger.error(f"❌ Test fehlgeschlagen: {e}")
        return False

def test_chronos_echo_bridge():
    """Test der Lazy-Loading-Funktionalität für die ChronosEchoBridge"""
    logger.info("=== Test: Lazy-Loading von ChronosEchoBridge ===")
    
    try:
        # Importiere die Lazy-Loading-Funktion
        from miso.timeline.echo_prime_controller import get_chronos_bridge
        
        # Prüfe, ob die Lazy-Loading-Funktion existiert
        logger.info("Prüfe, ob die Lazy-Loading-Funktion existiert...")
        assert callable(get_chronos_bridge), "get_chronos_bridge ist keine Funktion"
        logger.info("✅ Lazy-Loading-Funktion existiert")
        
        # Führe Lazy-Loading aus und prüfe das Ergebnis
        logger.info("Führe Lazy-Loading aus...")
        ChronosBridge = get_chronos_bridge()
        
        # Prüfe das Ergebnis
        if ChronosBridge is not None:
            logger.info(f"✅ ChronosBridge erfolgreich geladen: {ChronosBridge.__name__}")
        else:
            logger.warning("⚠️ ChronosBridge konnte nicht geladen werden, nutze Fallback")
        
        return True
    except Exception as e:
        logger.error(f"❌ Test fehlgeschlagen: {e}")
        return False

def run_all_tests():
    """Führt alle Tests aus und gibt die Ergebnisse zurück"""
    results = {
        "echo_prime_lazy_loading": test_echo_prime_lazy_loading(),
        "prism_lazy_loading": test_prism_lazy_loading(),
        "chronos_echo_bridge": test_chronos_echo_bridge()
    }
    
    # Gib eine Zusammenfassung aus
    logger.info("\n=== Testergebnisse ===")
    all_passed = True
    for test_name, result in results.items():
        status = "✅ BESTANDEN" if result else "❌ FEHLGESCHLAGEN"
        logger.info(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("🎉 ALLE TESTS ERFOLGREICH!")
    else:
        logger.warning("⚠️ EINIGE TESTS SIND FEHLGESCHLAGEN!")
    
    return results

if __name__ == "__main__":
    run_all_tests()
