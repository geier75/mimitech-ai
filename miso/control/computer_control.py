#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Computer Control Module

Dieses Modul ermöglicht MISO die Steuerung des Computers, einschließlich Maus- und Tastatursteuerung.
Es nutzt plattformübergreifende Bibliotheken für die Interaktion mit dem Betriebssystem.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import subprocess
from typing import Dict, List, Any, Optional, Union, Tuple
import threading
import queue
import json

# Konfiguriere Logging
logger = logging.getLogger("MISO.control.computer")

# Prüfen, ob Apple Silicon verfügbar ist
is_apple_silicon = sys.platform == 'darwin' and 'arm' in os.uname().machine

# Importiere plattformspezifische Module
try:
    import pyautogui
    import keyboard
    import mouse
    HAS_CONTROL_DEPENDENCIES = True
except ImportError:
    logger.warning("Abhängigkeiten für Computersteuerung fehlen. Installiere sie mit: pip install pyautogui keyboard mouse")
    HAS_CONTROL_DEPENDENCIES = False


class ComputerControl:
    """
    Hauptklasse für die Computersteuerung, die Maus- und Tastatursteuerung sowie
    Bildschirminteraktionen ermöglicht.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert die ComputerControl-Klasse
        
        Args:
            config: Konfigurationsparameter
        """
        self.config = config or {}
        self.is_active = False
        self.command_queue = queue.Queue()
        self.processing_thread = None
        self.screen_size = None
        self.security_level = self.config.get("security_level", "high")
        
        # Initialisiere PyAutoGUI-Sicherheitseinstellungen
        if HAS_CONTROL_DEPENDENCIES:
            pyautogui.FAILSAFE = True  # Bewege Maus in obere linke Ecke, um Notfall-Stop auszulösen
            pyautogui.PAUSE = self.config.get("command_delay", 0.1)  # Verzögerung zwischen Befehlen
            self.screen_size = pyautogui.size()
        
        logger.info("ComputerControl initialisiert")
    
    def start(self):
        """Startet die Computersteuerung"""
        if not HAS_CONTROL_DEPENDENCIES:
            logger.error("Computersteuerung kann nicht gestartet werden: Abhängigkeiten fehlen")
            return False
        
        if self.is_active:
            logger.warning("Computersteuerung ist bereits aktiv")
            return True
        
        self.is_active = True
        
        # Starte Verarbeitungsthread für Befehle
        self.processing_thread = threading.Thread(target=self._process_commands)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Computersteuerung gestartet")
        return True
    
    def stop(self):
        """Stoppt die Computersteuerung"""
        if not self.is_active:
            logger.warning("Computersteuerung ist nicht aktiv")
            return
        
        self.is_active = False
        
        # Warte auf Beendigung des Verarbeitungsthreads
        if self.processing_thread and self.processing_thread.is_alive():
            self.command_queue.put(("STOP", None))
            self.processing_thread.join(timeout=2.0)
        
        logger.info("Computersteuerung gestoppt")
    
    def _process_commands(self):
        """Verarbeitet Befehle aus der Befehlswarteschlange"""
        while self.is_active:
            try:
                command, args = self.command_queue.get(timeout=0.5)
                
                if command == "STOP":
                    break
                
                # Verarbeite Befehl
                if command == "MOVE_MOUSE":
                    self._move_mouse(*args)
                elif command == "CLICK":
                    self._click(*args)
                elif command == "DOUBLE_CLICK":
                    self._double_click(*args)
                elif command == "RIGHT_CLICK":
                    self._right_click(*args)
                elif command == "DRAG":
                    self._drag(*args)
                elif command == "SCROLL":
                    self._scroll(*args)
                elif command == "TYPE":
                    self._type(*args)
                elif command == "PRESS_KEY":
                    self._press_key(*args)
                elif command == "HOTKEY":
                    self._hotkey(*args)
                elif command == "SCREENSHOT":
                    self._take_screenshot(*args)
                else:
                    logger.warning(f"Unbekannter Befehl: {command}")
                
                # Markiere Befehl als abgeschlossen
                self.command_queue.task_done()
                
            except queue.Empty:
                # Keine Befehle in der Warteschlange
                pass
            except Exception as e:
                logger.error(f"Fehler bei der Verarbeitung von Befehlen: {e}")
    
    def _check_security(self, action: str) -> bool:
        """
        Überprüft, ob eine Aktion aufgrund von Sicherheitseinstellungen erlaubt ist
        
        Args:
            action: Aktion, die überprüft werden soll
            
        Returns:
            True, wenn die Aktion erlaubt ist, sonst False
        """
        # Implementiere Sicherheitsrichtlinien basierend auf dem Sicherheitslevel
        if self.security_level == "high":
            # Nur grundlegende Mausbewegungen und Klicks erlauben
            allowed_actions = ["MOVE_MOUSE", "CLICK", "SCREENSHOT"]
            return action in allowed_actions
        elif self.security_level == "medium":
            # Die meisten Aktionen erlauben, aber keine Systemtasten
            disallowed_actions = ["PRESS_KEY_SYSTEM", "HOTKEY_SYSTEM"]
            return action not in disallowed_actions
        else:  # "low"
            # Alle Aktionen erlauben
            return True
    
    # Maussteuerungsfunktionen
    
    def move_mouse(self, x: int, y: int, duration: float = 0.5):
        """
        Bewegt die Maus zu einer Position
        
        Args:
            x: X-Koordinate
            y: Y-Koordinate
            duration: Dauer der Bewegung in Sekunden
        """
        if not self._check_security("MOVE_MOUSE"):
            logger.warning("Mausbewegung aufgrund von Sicherheitseinstellungen blockiert")
            return
        
        self.command_queue.put(("MOVE_MOUSE", (x, y, duration)))
    
    def _move_mouse(self, x: int, y: int, duration: float = 0.5):
        """Interne Funktion zur Mausbewegung"""
        try:
            pyautogui.moveTo(x, y, duration=duration)
            logger.debug(f"Maus bewegt zu ({x}, {y})")
        except Exception as e:
            logger.error(f"Fehler bei Mausbewegung: {e}")
    
    def click(self, x: Optional[int] = None, y: Optional[int] = None, button: str = "left"):
        """
        Führt einen Mausklick aus
        
        Args:
            x: X-Koordinate (None für aktuelle Position)
            y: Y-Koordinate (None für aktuelle Position)
            button: Maustaste ("left", "right", "middle")
        """
        if not self._check_security("CLICK"):
            logger.warning("Mausklick aufgrund von Sicherheitseinstellungen blockiert")
            return
        
        self.command_queue.put(("CLICK", (x, y, button)))
    
    def _click(self, x: Optional[int] = None, y: Optional[int] = None, button: str = "left"):
        """Interne Funktion für Mausklick"""
        try:
            pyautogui.click(x=x, y=y, button=button)
            pos = (x, y) if x is not None and y is not None else "aktueller Position"
            logger.debug(f"{button.capitalize()}-Klick bei {pos}")
        except Exception as e:
            logger.error(f"Fehler bei Mausklick: {e}")
    
    def double_click(self, x: Optional[int] = None, y: Optional[int] = None):
        """
        Führt einen Doppelklick aus
        
        Args:
            x: X-Koordinate (None für aktuelle Position)
            y: Y-Koordinate (None für aktuelle Position)
        """
        if not self._check_security("DOUBLE_CLICK"):
            logger.warning("Doppelklick aufgrund von Sicherheitseinstellungen blockiert")
            return
        
        self.command_queue.put(("DOUBLE_CLICK", (x, y)))
    
    def _double_click(self, x: Optional[int] = None, y: Optional[int] = None):
        """Interne Funktion für Doppelklick"""
        try:
            pyautogui.doubleClick(x=x, y=y)
            pos = (x, y) if x is not None and y is not None else "aktueller Position"
            logger.debug(f"Doppelklick bei {pos}")
        except Exception as e:
            logger.error(f"Fehler bei Doppelklick: {e}")
    
    def right_click(self, x: Optional[int] = None, y: Optional[int] = None):
        """
        Führt einen Rechtsklick aus
        
        Args:
            x: X-Koordinate (None für aktuelle Position)
            y: Y-Koordinate (None für aktuelle Position)
        """
        if not self._check_security("RIGHT_CLICK"):
            logger.warning("Rechtsklick aufgrund von Sicherheitseinstellungen blockiert")
            return
        
        self.command_queue.put(("RIGHT_CLICK", (x, y)))
    
    def _right_click(self, x: Optional[int] = None, y: Optional[int] = None):
        """Interne Funktion für Rechtsklick"""
        try:
            pyautogui.rightClick(x=x, y=y)
            pos = (x, y) if x is not None and y is not None else "aktueller Position"
            logger.debug(f"Rechtsklick bei {pos}")
        except Exception as e:
            logger.error(f"Fehler bei Rechtsklick: {e}")
    
    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5):
        """
        Führt eine Ziehbewegung aus
        
        Args:
            start_x: Start-X-Koordinate
            start_y: Start-Y-Koordinate
            end_x: End-X-Koordinate
            end_y: End-Y-Koordinate
            duration: Dauer der Bewegung in Sekunden
        """
        if not self._check_security("DRAG"):
            logger.warning("Drag-Operation aufgrund von Sicherheitseinstellungen blockiert")
            return
        
        self.command_queue.put(("DRAG", (start_x, start_y, end_x, end_y, duration)))
    
    def _drag(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5):
        """Interne Funktion für Drag-Operation"""
        try:
            pyautogui.moveTo(start_x, start_y)
            pyautogui.dragTo(end_x, end_y, duration=duration)
            logger.debug(f"Drag von ({start_x}, {start_y}) zu ({end_x}, {end_y})")
        except Exception as e:
            logger.error(f"Fehler bei Drag-Operation: {e}")
    
    def scroll(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None):
        """
        Scrollt die Maus
        
        Args:
            clicks: Anzahl der Scrollklicks (positiv = nach oben, negativ = nach unten)
            x: X-Koordinate (None für aktuelle Position)
            y: Y-Koordinate (None für aktuelle Position)
        """
        if not self._check_security("SCROLL"):
            logger.warning("Scroll-Operation aufgrund von Sicherheitseinstellungen blockiert")
            return
        
        self.command_queue.put(("SCROLL", (clicks, x, y)))
    
    def _scroll(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None):
        """Interne Funktion für Scroll-Operation"""
        try:
            if x is not None and y is not None:
                pyautogui.moveTo(x, y)
            pyautogui.scroll(clicks)
            direction = "oben" if clicks > 0 else "unten"
            logger.debug(f"Scroll nach {direction} ({abs(clicks)} Klicks)")
        except Exception as e:
            logger.error(f"Fehler bei Scroll-Operation: {e}")
    
    # Tastatursteuerungsfunktionen
    
    def type_text(self, text: str, interval: float = 0.01):
        """
        Gibt Text ein
        
        Args:
            text: Einzugebender Text
            interval: Verzögerung zwischen Tastenanschlägen in Sekunden
        """
        if not self._check_security("TYPE"):
            logger.warning("Texteingabe aufgrund von Sicherheitseinstellungen blockiert")
            return
        
        self.command_queue.put(("TYPE", (text, interval)))
    
    def _type(self, text: str, interval: float = 0.01):
        """Interne Funktion für Texteingabe"""
        try:
            pyautogui.write(text, interval=interval)
            logger.debug(f"Text eingegeben: {text[:20]}..." if len(text) > 20 else f"Text eingegeben: {text}")
        except Exception as e:
            logger.error(f"Fehler bei Texteingabe: {e}")
    
    def press_key(self, key: str):
        """
        Drückt eine Taste
        
        Args:
            key: Zu drückende Taste (z.B. 'enter', 'tab', 'a', 'F1')
        """
        # Prüfe, ob es sich um eine Systemtaste handelt
        system_keys = ['win', 'command', 'cmd', 'alt', 'option']
        action = "PRESS_KEY_SYSTEM" if key.lower() in system_keys else "PRESS_KEY"
        
        if not self._check_security(action):
            logger.warning(f"Tastendruck ({key}) aufgrund von Sicherheitseinstellungen blockiert")
            return
        
        self.command_queue.put(("PRESS_KEY", (key,)))
    
    def _press_key(self, key: str):
        """Interne Funktion für Tastendruck"""
        try:
            pyautogui.press(key)
            logger.debug(f"Taste gedrückt: {key}")
        except Exception as e:
            logger.error(f"Fehler bei Tastendruck: {e}")
    
    def hotkey(self, *keys):
        """
        Drückt mehrere Tasten gleichzeitig (Tastenkombination)
        
        Args:
            *keys: Zu drückende Tasten (z.B. 'ctrl', 'c' für Kopieren)
        """
        # Prüfe, ob es sich um eine System-Tastenkombination handelt
        system_keys = ['win', 'command', 'cmd']
        action = "HOTKEY_SYSTEM" if any(k.lower() in system_keys for k in keys) else "HOTKEY"
        
        if not self._check_security(action):
            logger.warning(f"Tastenkombination ({'+'.join(keys)}) aufgrund von Sicherheitseinstellungen blockiert")
            return
        
        self.command_queue.put(("HOTKEY", keys))
    
    def _hotkey(self, *keys):
        """Interne Funktion für Tastenkombination"""
        try:
            pyautogui.hotkey(*keys)
            logger.debug(f"Tastenkombination gedrückt: {'+'.join(keys)}")
        except Exception as e:
            logger.error(f"Fehler bei Tastenkombination: {e}")
    
    # Bildschirmfunktionen
    
    def take_screenshot(self, filepath: Optional[str] = None, region: Optional[Tuple[int, int, int, int]] = None):
        """
        Nimmt einen Screenshot auf
        
        Args:
            filepath: Pfad zum Speichern des Screenshots (None für Rückgabe als Image-Objekt)
            region: Region für den Screenshot (x, y, width, height)
        
        Returns:
            PIL.Image.Image oder None, wenn in Datei gespeichert
        """
        if not self._check_security("SCREENSHOT"):
            logger.warning("Screenshot aufgrund von Sicherheitseinstellungen blockiert")
            return None
        
        result_queue = queue.Queue()
        self.command_queue.put(("SCREENSHOT", (filepath, region, result_queue)))
        
        # Warte auf Ergebnis
        try:
            return result_queue.get(timeout=5.0)
        except queue.Empty:
            logger.error("Timeout beim Warten auf Screenshot")
            return None
    
    def _take_screenshot(self, filepath: Optional[str] = None, 
                        region: Optional[Tuple[int, int, int, int]] = None,
                        result_queue: Optional[queue.Queue] = None):
        """Interne Funktion für Screenshot"""
        try:
            screenshot = pyautogui.screenshot(region=region)
            
            if filepath:
                screenshot.save(filepath)
                logger.debug(f"Screenshot gespeichert unter: {filepath}")
                if result_queue:
                    result_queue.put(True)
            elif result_queue:
                result_queue.put(screenshot)
                logger.debug("Screenshot aufgenommen")
        except Exception as e:
            logger.error(f"Fehler bei Screenshot: {e}")
            if result_queue:
                result_queue.put(None)
    
    # Hilfsfunktionen
    
    def get_screen_size(self) -> Tuple[int, int]:
        """
        Gibt die Bildschirmgröße zurück
        
        Returns:
            Tuple mit (Breite, Höhe) des Bildschirms
        """
        if not HAS_CONTROL_DEPENDENCIES:
            logger.error("Bildschirmgröße kann nicht ermittelt werden: Abhängigkeiten fehlen")
            return (0, 0)
        
        return self.screen_size
    
    def get_mouse_position(self) -> Tuple[int, int]:
        """
        Gibt die aktuelle Mausposition zurück
        
        Returns:
            Tuple mit (x, y) der Mausposition
        """
        if not HAS_CONTROL_DEPENDENCIES:
            logger.error("Mausposition kann nicht ermittelt werden: Abhängigkeiten fehlen")
            return (0, 0)
        
        return pyautogui.position()
    
    def wait(self, seconds: float):
        """
        Wartet für eine bestimmte Zeit
        
        Args:
            seconds: Wartezeit in Sekunden
        """
        time.sleep(seconds)


# Beispielnutzung
if __name__ == "__main__":
    # Konfiguriere Logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Erstelle ComputerControl-Instanz
    control = ComputerControl(config={"security_level": "medium"})
    
    # Starte Computersteuerung
    if control.start():
        try:
            # Beispiel: Bewege Maus und klicke
            screen_size = control.get_screen_size()
            control.move_mouse(screen_size[0] // 2, screen_size[1] // 2)
            control.click()
            
            # Beispiel: Gib Text ein
            control.type_text("MISO Computer Control Test")
            
            # Warte kurz
            control.wait(2.0)
            
            # Beispiel: Nimm Screenshot auf
            screenshot = control.take_screenshot()
            if screenshot:
                screenshot.save("miso_control_test.png")
                
        finally:
            # Stoppe Computersteuerung
            control.stop()
