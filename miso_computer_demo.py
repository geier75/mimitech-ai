#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Computer Control Demo

Diese Demo zeigt, wie MISO den Computer steuern kann, indem es die
ComputerControl-Klasse in eine einfache Anwendung integriert.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import random
from pathlib import Path

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("MISO.ComputerDemo")

# Stelle sicher, dass das MISO-Paket im Pfad ist
parent_dir = str(Path(__file__).parent)
sys.path.insert(0, parent_dir)

# Importiere die benötigten Module
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

class MISOComputerDemo:
    """
    Demo-Klasse, die zeigt, wie MISO den Computer steuern kann.
    Diese Klasse integriert die ComputerControl-Funktionalität in eine einfache Anwendung.
    """
    
    def __init__(self):
        """Initialisiert die Demo"""
        logger.info("Initialisiere MISO Computer Demo")
        
        # Erstelle ComputerControl mit niedrigem Sicherheitslevel für die Demo
        config = {"security_level": "low", "command_delay": 0.5}
        self.computer_control = ComputerControl(config)
        
        # Hole Bildschirmgröße
        self.screen_width, self.screen_height = self.computer_control.get_screen_size()
        logger.info(f"Bildschirmgröße: {self.screen_width}x{self.screen_height}")
        
        # Initialisiere Zustand
        self.running = False
    
    def start(self):
        """Startet die Demo"""
        logger.info("Starte MISO Computer Demo")
        self.running = True
        
        print("\n" + "=" * 80)
        print(" MISO Computer Control Demo ".center(80, "="))
        print("=" * 80 + "\n")
        
        print("Diese Demo zeigt, wie MISO den Computer steuern kann.")
        print("MISO wird verschiedene Aktionen ausführen, um die Computersteuerung zu demonstrieren.")
        print("Sie können die Demo jederzeit mit Strg+C beenden.\n")
        
        input("Drücken Sie Enter, um zu starten...")
        
        try:
            self.run_demo()
        except KeyboardInterrupt:
            logger.info("Demo durch Benutzer unterbrochen")
        finally:
            self.running = False
            logger.info("Demo beendet")
    
    def run_demo(self):
        """Führt die Demo aus"""
        demos = [
            self.demo_mouse_movement,
            self.demo_mouse_clicks,
            self.demo_keyboard_input,
            self.demo_screenshot,
            self.demo_automated_task
        ]
        
        for demo_func in demos:
            if not self.running:
                break
            
            demo_name = demo_func.__name__.replace("demo_", "").replace("_", " ").title()
            print(f"\n{'-' * 40}")
            print(f" Demo: {demo_name} ".center(40, "-"))
            print(f"{'-' * 40}\n")
            
            try:
                demo_func()
                print(f"\n✅ {demo_name} Demo abgeschlossen")
            except Exception as e:
                logger.error(f"Fehler bei {demo_name} Demo: {e}")
                print(f"\n❌ {demo_name} Demo fehlgeschlagen")
            
            # Kurze Pause zwischen den Demos
            time.sleep(2.0)
    
    def demo_mouse_movement(self):
        """Demonstriert Mausbewegungen"""
        print("MISO bewegt die Maus in verschiedenen Mustern...")
        
        # Bewege die Maus in einem Kreis
        center_x, center_y = self.screen_width // 2, self.screen_height // 2
        radius = 100
        
        print("Bewege die Maus in einem Kreis...")
        self.computer_control.move_mouse(center_x, center_y, duration=1.0)
        time.sleep(1.0)
        
        steps = 12
        for i in range(steps + 1):
            if not self.running:
                break
            
            angle = 2 * 3.14159 * i / steps
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle))
            
            self.computer_control.move_mouse(x, y, duration=0.5)
            time.sleep(0.1)
        
        # Bewege die Maus in einem Zickzack-Muster
        print("Bewege die Maus in einem Zickzack-Muster...")
        start_x = center_x - 150
        start_y = center_y
        
        self.computer_control.move_mouse(start_x, start_y, duration=1.0)
        time.sleep(1.0)
        
        for i in range(5):
            if not self.running:
                break
            
            # Bewege nach oben/unten
            y_offset = 50 * (-1 if i % 2 == 0 else 1)
            self.computer_control.move_mouse(start_x + i * 75, start_y + y_offset, duration=0.5)
            time.sleep(0.3)
    
    def demo_mouse_clicks(self):
        """Demonstriert Mausklicks"""
        print("MISO führt verschiedene Mausklicks aus...")
        
        # Bewege die Maus zur Mitte des Bildschirms
        center_x, center_y = self.screen_width // 2, self.screen_height // 2
        
        # Führe einen Linksklick aus
        print("Führe Linksklick aus...")
        self.computer_control.move_mouse(center_x, center_y, duration=1.0)
        time.sleep(0.5)
        self.computer_control.click()
        time.sleep(1.0)
        
        # Führe einen Rechtsklick aus
        print("Führe Rechtsklick aus...")
        self.computer_control.right_click()
        time.sleep(1.0)
        
        # Drücke Escape, um das Kontextmenü zu schließen
        self.computer_control.press_key("escape")
        time.sleep(0.5)
        
        # Führe einen Doppelklick aus
        print("Führe Doppelklick aus...")
        self.computer_control.double_click()
        time.sleep(1.0)
    
    def demo_keyboard_input(self):
        """Demonstriert Tastatureingaben"""
        print("MISO gibt Text ein...")
        
        # Öffne eine Textdatei oder ein Textbearbeitungsprogramm
        # In einer realen Anwendung würde MISO hier ein Programm öffnen
        # Für diese Demo nehmen wir an, dass bereits ein Texteditor geöffnet ist
        
        # Gib Text ein
        text = "Dies ist eine Demonstration der MISO Computersteuerung."
        print(f"Gebe ein: '{text}'")
        self.computer_control.type_text(text)
        time.sleep(1.0)
        
        # Drücke Enter
        self.computer_control.press_key("enter")
        time.sleep(0.5)
        
        # Gib weiteren Text ein
        text2 = "MISO kann den Computer vollständig steuern, einschließlich Maus und Tastatur."
        print(f"Gebe ein: '{text2}'")
        self.computer_control.type_text(text2)
        time.sleep(1.0)
        
        # Drücke Tastenkombination (z.B. Strg+A für "Alles auswählen")
        print("Drücke Tastenkombination: Strg+A (Alles auswählen)")
        self.computer_control.hotkey("ctrl", "a")
        time.sleep(1.0)
    
    def demo_screenshot(self):
        """Demonstriert Screenshots"""
        print("MISO erstellt einen Screenshot...")
        
        # Mache einen Screenshot
        screenshot = self.computer_control.take_screenshot()
        
        if screenshot is not None:
            # Speichere den Screenshot
            screenshot_path = os.path.join(os.path.dirname(__file__), "miso_demo_screenshot.png")
            screenshot.save(screenshot_path)
            print(f"Screenshot gespeichert unter: {screenshot_path}")
            
            # In einer realen Anwendung könnte MISO den Screenshot analysieren
            print("MISO könnte diesen Screenshot analysieren, um den Bildschirminhalt zu verstehen")
        else:
            print("Fehler beim Erstellen des Screenshots")
    
    def demo_automated_task(self):
        """Demonstriert eine automatisierte Aufgabe"""
        print("MISO führt eine automatisierte Aufgabe aus...")
        
        # In einer realen Anwendung könnte MISO komplexe Aufgaben automatisieren
        # Hier ist ein einfaches Beispiel: Öffne den Browser und navigiere zu einer Webseite
        
        # Simuliere das Öffnen eines Browsers
        print("Simuliere das Öffnen eines Browsers...")
        
        # Bewege die Maus zu einer Position, wo ein Browser-Icon sein könnte
        browser_x = self.screen_width // 4
        browser_y = self.screen_height - 50
        
        self.computer_control.move_mouse(browser_x, browser_y, duration=1.0)
        time.sleep(0.5)
        self.computer_control.click()
        time.sleep(2.0)  # Warte, bis der Browser geöffnet ist
        
        # Simuliere die Eingabe einer URL
        print("Simuliere die Eingabe einer URL...")
        url = "https://www.example.com"
        self.computer_control.type_text(url)
        time.sleep(0.5)
        self.computer_control.press_key("enter")
        time.sleep(2.0)  # Warte, bis die Seite geladen ist
        
        print("Automatisierte Aufgabe abgeschlossen")

# Füge die fehlende math-Bibliothek hinzu
import math

if __name__ == "__main__":
    demo = MISOComputerDemo()
    demo.start()
