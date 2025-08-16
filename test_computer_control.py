#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Computer Control Test

Dieses Skript demonstriert die Computersteuerungsfähigkeiten von MISO.
Es zeigt, wie MISO den Computer steuern kann, einschließlich Mausbewegungen und Klicks.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("MISO.ComputerControlTest")

# Stelle sicher, dass das MISO-Paket im Pfad ist
parent_dir = str(Path(__file__).parent)
sys.path.insert(0, parent_dir)

# Importiere die ComputerControl-Klasse
try:
    from miso.control.computer_control import ComputerControl
    logger.info("ComputerControl-Modul erfolgreich importiert")
except ImportError as e:
    logger.error(f"Fehler beim Import des ComputerControl-Moduls: {e}")
    # Versuche alternativen Import
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "computer_control", 
            os.path.join(parent_dir, "miso", "control", "computer_control.py")
        )
        computer_control_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(computer_control_module)
        ComputerControl = computer_control_module.ComputerControl
        logger.info("ComputerControl-Modul erfolgreich über alternativen Weg importiert")
    except Exception as e2:
        logger.error(f"Fehler beim alternativen Import: {e2}")
        sys.exit(1)

def test_mouse_movement():
    """Testet die Mausbewegung"""
    logger.info("Teste Mausbewegung...")
    
    # Erstelle ComputerControl mit niedrigem Sicherheitslevel für Tests
    config = {"security_level": "low", "command_delay": 0.5}
    computer_control = ComputerControl(config)
    
    # Bildschirmgröße ermitteln
    screen_width, screen_height = computer_control.get_screen_size()
    logger.info(f"Bildschirmgröße: {screen_width}x{screen_height}")
    
    # Bewege die Maus in einem Quadrat
    logger.info("Bewege die Maus in einem Quadrat...")
    
    # Startposition: Mitte des Bildschirms
    center_x, center_y = screen_width // 2, screen_height // 2
    size = 100  # Größe des Quadrats
    
    # Bewege die Maus zum Startpunkt
    computer_control.move_mouse(center_x - size, center_y - size, duration=1.0)
    time.sleep(1)
    
    # Bewege die Maus im Quadrat
    computer_control.move_mouse(center_x + size, center_y - size, duration=1.0)
    time.sleep(0.5)
    computer_control.move_mouse(center_x + size, center_y + size, duration=1.0)
    time.sleep(0.5)
    computer_control.move_mouse(center_x - size, center_y + size, duration=1.0)
    time.sleep(0.5)
    computer_control.move_mouse(center_x - size, center_y - size, duration=1.0)
    
    logger.info("Mausbewegungstest abgeschlossen")
    return True

def test_mouse_clicks():
    """Testet Mausklicks"""
    logger.info("Teste Mausklicks...")
    
    # Erstelle ComputerControl mit niedrigem Sicherheitslevel für Tests
    config = {"security_level": "low", "command_delay": 0.5}
    computer_control = ComputerControl(config)
    
    # Bildschirmgröße ermitteln
    screen_width, screen_height = computer_control.get_screen_size()
    
    # Bewege die Maus zur Mitte des Bildschirms und führe einen Linksklick aus
    center_x, center_y = screen_width // 2, screen_height // 2
    logger.info(f"Führe Linksklick in der Bildschirmmitte aus ({center_x}, {center_y})...")
    computer_control.move_mouse(center_x, center_y, duration=1.0)
    time.sleep(0.5)
    computer_control.click()
    
    # Führe einen Rechtsklick aus
    logger.info("Führe Rechtsklick aus...")
    time.sleep(1.0)
    computer_control.right_click()
    
    # Führe einen Doppelklick aus
    logger.info("Führe Doppelklick aus...")
    time.sleep(1.0)
    computer_control.double_click()
    
    logger.info("Mausklicktest abgeschlossen")
    return True

def test_keyboard_input():
    """Testet Tastatureingaben"""
    logger.info("Teste Tastatureingaben...")
    
    # Erstelle ComputerControl mit niedrigem Sicherheitslevel für Tests
    config = {"security_level": "low", "command_delay": 0.5}
    computer_control = ComputerControl(config)
    
    # Öffne eine Textdatei oder ein Textbearbeitungsprogramm
    # Hinweis: Dies ist nur ein Beispiel, Sie müssen möglicherweise den Pfad anpassen
    logger.info("Öffne ein Textbearbeitungsprogramm...")
    
    # Warte, bis das Programm geöffnet ist
    time.sleep(2.0)
    
    # Gib Text ein
    test_text = "Dies ist ein Test der MISO Computersteuerung. MISO kann den Computer selbständig steuern!"
    logger.info(f"Gebe Text ein: {test_text}")
    computer_control.type_text(test_text)
    
    # Drücke Enter
    time.sleep(1.0)
    computer_control.press_key("enter")
    
    # Gib weiteren Text ein
    time.sleep(0.5)
    computer_control.type_text("Tastatureingabetest abgeschlossen.")
    
    logger.info("Tastatureingabetest abgeschlossen")
    return True

def test_screenshot():
    """Testet Screenshots"""
    logger.info("Teste Screenshots...")
    
    # Erstelle ComputerControl mit niedrigem Sicherheitslevel für Tests
    config = {"security_level": "low", "command_delay": 0.5}
    computer_control = ComputerControl(config)
    
    # Mache einen Screenshot
    logger.info("Mache einen Screenshot...")
    screenshot = computer_control.take_screenshot()
    
    if screenshot is not None:
        # Speichere den Screenshot
        screenshot_path = os.path.join(os.path.dirname(__file__), "miso_screenshot.png")
        screenshot.save(screenshot_path)
        logger.info(f"Screenshot gespeichert unter: {screenshot_path}")
    else:
        logger.error("Fehler beim Erstellen des Screenshots")
    
    logger.info("Screenshottest abgeschlossen")
    return screenshot is not None

def main():
    """Hauptfunktion"""
    print("\n" + "=" * 80)
    print(" MISO Computer Control Test ".center(80, "="))
    print("=" * 80 + "\n")
    
    print("Dieser Test demonstriert die Fähigkeit von MISO, den Computer zu steuern.")
    print("Bitte stellen Sie sicher, dass keine wichtigen Anwendungen geöffnet sind,")
    print("da MISO die Maus bewegen und Klicks ausführen wird.\n")
    
    input("Drücken Sie Enter, um den Test zu starten...")
    
    # Führe Tests aus
    tests = [
        ("Mausbewegung", test_mouse_movement),
        ("Mausklicks", test_mouse_clicks),
        ("Tastatureingaben", test_keyboard_input),
        ("Screenshots", test_screenshot)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\nStarte Test: {test_name}")
        try:
            result = test_func()
            status = "✅ ERFOLG" if result else "❌ FEHLER"
            results[test_name] = result
        except Exception as e:
            logger.error(f"Fehler bei Test {test_name}: {e}")
            status = "❌ FEHLER"
            results[test_name] = False
        
        print(f"{test_name}: {status}")
    
    # Zeige Zusammenfassung
    print("\n" + "=" * 80)
    print(" Testergebnisse ".center(80, "="))
    print("=" * 80 + "\n")
    
    all_tests_passed = all(results.values())
    
    for test_name, result in results.items():
        status = "✅ ERFOLG" if result else "❌ FEHLER"
        print(f"{test_name}: {status}")
    
    print("\n" + "=" * 80)
    if all_tests_passed:
        print("\n✅ Alle Tests erfolgreich bestanden. MISO kann den Computer steuern!")
    else:
        print("\n❌ Einige Tests sind fehlgeschlagen. Bitte überprüfen Sie die Fehler.")
    
    print("\nHinweis: Die Computersteuerung kann in Ihren eigenen MISO-Anwendungen")
    print("verwendet werden, indem Sie die ComputerControl-Klasse importieren und")
    print("die entsprechenden Methoden aufrufen.")

if __name__ == "__main__":
    main()
