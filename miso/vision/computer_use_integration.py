#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VX-VISION Computer-Use Integration für VXOR.AI

Vollständige Computer-Use-Funktionalität mit visueller Erkennung,
Tastatur-/Maus-Steuerung und "Sehen und Drücken" Automatisierung.

Copyright (c) 2025 VXOR.AI Team. Alle Rechte vorbehalten.
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2
import pyautogui
import pynput
from pynput import mouse, keyboard
from pynput.mouse import Button, Listener as MouseListener
from pynput.keyboard import Key, Listener as KeyboardListener
import pytesseract
from PIL import Image, ImageGrab
import json
import os

# MISO-Module
from miso.math.t_mathematics.engine import TMathEngine
from miso.vision.vx_vision_core import VXVisionCore

# Logger konfigurieren
logger = logging.getLogger("MISO.Vision.ComputerUse")

# PyAutoGUI Sicherheitseinstellungen
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1

class InteractionType(Enum):
    """Typen von Computer-Interaktionen"""
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    DRAG = "drag"
    SCROLL = "scroll"
    TYPE_TEXT = "type_text"
    KEY_PRESS = "key_press"
    KEY_COMBINATION = "key_combination"
    SCREENSHOT = "screenshot"
    VISUAL_SEARCH = "visual_search"

class ScreenRegion(Enum):
    """Bildschirmbereiche für gezielte Interaktion"""
    FULL_SCREEN = "full_screen"
    ACTIVE_WINDOW = "active_window"
    MENU_BAR = "menu_bar"
    DOCK = "dock"
    DESKTOP = "desktop"
    CUSTOM = "custom"

@dataclass
class VisualElement:
    """Visuelles Element auf dem Bildschirm"""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    element_type: str
    text: Optional[str] = None
    attributes: Dict[str, Any] = None

@dataclass
class InteractionCommand:
    """Befehl für Computer-Interaktion"""
    action: InteractionType
    target: Optional[VisualElement] = None
    coordinates: Optional[Tuple[int, int]] = None
    text: Optional[str] = None
    keys: Optional[List[str]] = None
    duration: float = 0.1
    delay_before: float = 0.0
    delay_after: float = 0.1

class ComputerUseIntegration:
    """Hauptklasse für Computer-Use-Integration"""
    
    def __init__(self, tmath_engine: Optional[TMathEngine] = None, vx_vision: Optional[VXVisionCore] = None):
        self.tmath_engine = tmath_engine or TMathEngine()
        self.vx_vision = vx_vision or VXVisionCore()
        
        # Konfiguration
        self.screen_width, self.screen_height = pyautogui.size()
        self.confidence_threshold = 0.8
        self.interaction_delay = 0.1
        self.screenshot_cache = {}
        self.element_cache = {}
        
        # Event-Listener
        self.mouse_listener = None
        self.keyboard_listener = None
        self.monitoring_active = False
        
        # OCR-Konfiguration
        self.ocr_config = '--oem 3 --psm 6'
        
        # Template-Matching-Cache
        self.template_cache = {}
        
        logger.info("Computer-Use Integration initialisiert")
        logger.info(f"Bildschirmauflösung: {self.screen_width}x{self.screen_height}")
    
    def take_screenshot(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Erstelle Screenshot von Bildschirm oder Region"""
        try:
            if region:
                # Spezifische Region
                x, y, width, height = region
                screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
            else:
                # Vollbild
                screenshot = ImageGrab.grab()
            
            # Konvertiere zu OpenCV-Format
            screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            # Cache aktualisieren
            self.screenshot_cache['latest'] = screenshot_cv
            self.screenshot_cache['timestamp'] = time.time()
            
            return screenshot_cv
            
        except Exception as e:
            logger.error(f"Fehler beim Screenshot: {e}")
            return np.array([])
    
    def find_visual_elements(self, screenshot: Optional[np.ndarray] = None, 
                           element_types: List[str] = None) -> List[VisualElement]:
        """Finde visuelle Elemente auf dem Bildschirm"""
        if screenshot is None:
            screenshot = self.take_screenshot()
        
        if screenshot.size == 0:
            return []
        
        elements = []
        
        try:
            # Text-Erkennung mit OCR
            if not element_types or 'text' in element_types:
                text_elements = self._find_text_elements(screenshot)
                elements.extend(text_elements)
            
            # Button-Erkennung
            if not element_types or 'button' in element_types:
                button_elements = self._find_button_elements(screenshot)
                elements.extend(button_elements)
            
            # Icon-Erkennung
            if not element_types or 'icon' in element_types:
                icon_elements = self._find_icon_elements(screenshot)
                elements.extend(icon_elements)
            
            # Window-Erkennung
            if not element_types or 'window' in element_types:
                window_elements = self._find_window_elements(screenshot)
                elements.extend(window_elements)
            
            # Cache aktualisieren
            self.element_cache['latest'] = elements
            self.element_cache['timestamp'] = time.time()
            
            logger.info(f"Gefunden: {len(elements)} visuelle Elemente")
            return elements
            
        except Exception as e:
            logger.error(f"Fehler bei visueller Elementsuche: {e}")
            return []
    
    def _find_text_elements(self, screenshot: np.ndarray) -> List[VisualElement]:
        """Finde Text-Elemente mit OCR"""
        elements = []
        
        try:
            # OCR mit pytesseract
            data = pytesseract.image_to_data(screenshot, config=self.ocr_config, output_type=pytesseract.Output.DICT)
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if text and int(data['conf'][i]) > 30:  # Mindest-Konfidenz
                    element = VisualElement(
                        x=data['left'][i],
                        y=data['top'][i],
                        width=data['width'][i],
                        height=data['height'][i],
                        confidence=int(data['conf'][i]) / 100.0,
                        element_type='text',
                        text=text
                    )
                    elements.append(element)
            
        except Exception as e:
            logger.warning(f"OCR-Fehler: {e}")
        
        return elements
    
    def _find_button_elements(self, screenshot: np.ndarray) -> List[VisualElement]:
        """Finde Button-Elemente durch Kantenerkennung"""
        elements = []
        
        try:
            # Graustufen-Konvertierung
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            
            # Kantenerkennung
            edges = cv2.Canny(gray, 50, 150)
            
            # Konturen finden
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Rechteckige Bereiche (potentielle Buttons)
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter für Button-ähnliche Dimensionen
                if 20 < w < 300 and 15 < h < 80:
                    aspect_ratio = w / h
                    if 1.5 < aspect_ratio < 8:  # Typische Button-Proportionen
                        element = VisualElement(
                            x=x,
                            y=y,
                            width=w,
                            height=h,
                            confidence=0.7,
                            element_type='button'
                        )
                        elements.append(element)
            
        except Exception as e:
            logger.warning(f"Button-Erkennungsfehler: {e}")
        
        return elements
    
    def _find_icon_elements(self, screenshot: np.ndarray) -> List[VisualElement]:
        """Finde Icon-Elemente durch Template-Matching"""
        elements = []
        
        try:
            # Graustufen-Konvertierung
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            
            # Suche nach quadratischen Bereichen (typisch für Icons)
            # Verwende HoughCircles für runde Icons
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                param1=50, param2=30, minRadius=10, maxRadius=50
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    element = VisualElement(
                        x=x-r,
                        y=y-r,
                        width=2*r,
                        height=2*r,
                        confidence=0.6,
                        element_type='icon'
                    )
                    elements.append(element)
            
        except Exception as e:
            logger.warning(f"Icon-Erkennungsfehler: {e}")
        
        return elements
    
    def _find_window_elements(self, screenshot: np.ndarray) -> List[VisualElement]:
        """Finde Fenster-Elemente"""
        elements = []
        
        try:
            # Graustufen-Konvertierung
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            
            # Suche nach großen rechteckigen Bereichen
            edges = cv2.Canny(gray, 30, 100)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter für Fenster-ähnliche Dimensionen
                if w > 200 and h > 150:
                    element = VisualElement(
                        x=x,
                        y=y,
                        width=w,
                        height=h,
                        confidence=0.5,
                        element_type='window'
                    )
                    elements.append(element)
            
        except Exception as e:
            logger.warning(f"Fenster-Erkennungsfehler: {e}")
        
        return elements
    
    def find_element_by_text(self, text: str, exact_match: bool = False) -> Optional[VisualElement]:
        """Finde Element anhand von Text"""
        elements = self.find_visual_elements(element_types=['text'])
        
        for element in elements:
            if element.text:
                if exact_match:
                    if element.text == text:
                        return element
                else:
                    if text.lower() in element.text.lower():
                        return element
        
        return None
    
    def find_element_by_template(self, template_path: str, threshold: float = 0.8) -> Optional[VisualElement]:
        """Finde Element anhand eines Template-Bildes"""
        try:
            # Lade Template
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                logger.error(f"Template nicht gefunden: {template_path}")
                return None
            
            # Aktueller Screenshot
            screenshot = self.take_screenshot()
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            
            # Template-Matching
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val >= threshold:
                h, w = template.shape
                element = VisualElement(
                    x=max_loc[0],
                    y=max_loc[1],
                    width=w,
                    height=h,
                    confidence=max_val,
                    element_type='template_match'
                )
                return element
            
        except Exception as e:
            logger.error(f"Template-Matching-Fehler: {e}")
        
        return None
    
    def click_element(self, element: VisualElement, button: str = 'left') -> bool:
        """Klicke auf ein visuelles Element"""
        try:
            # Berechne Klick-Position (Zentrum des Elements)
            click_x = element.x + element.width // 2
            click_y = element.y + element.height // 2
            
            # Führe Klick aus
            if button == 'left':
                pyautogui.click(click_x, click_y)
            elif button == 'right':
                pyautogui.rightClick(click_x, click_y)
            elif button == 'double':
                pyautogui.doubleClick(click_x, click_y)
            
            logger.info(f"Geklickt auf Element bei ({click_x}, {click_y}) - {button}")
            return True
            
        except Exception as e:
            logger.error(f"Klick-Fehler: {e}")
            return False
    
    def type_text(self, text: str, interval: float = 0.05) -> bool:
        """Tippe Text"""
        try:
            pyautogui.typewrite(text, interval=interval)
            logger.info(f"Text getippt: {text[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Text-Eingabe-Fehler: {e}")
            return False
    
    def press_key(self, key: str) -> bool:
        """Drücke eine Taste"""
        try:
            pyautogui.press(key)
            logger.info(f"Taste gedrückt: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Tastendruck-Fehler: {e}")
            return False
    
    def press_key_combination(self, keys: List[str]) -> bool:
        """Drücke Tastenkombination"""
        try:
            pyautogui.hotkey(*keys)
            logger.info(f"Tastenkombination: {'+'.join(keys)}")
            return True
            
        except Exception as e:
            logger.error(f"Tastenkombination-Fehler: {e}")
            return False
    
    def drag_element(self, from_element: VisualElement, to_element: VisualElement, duration: float = 1.0) -> bool:
        """Ziehe Element von einer Position zur anderen"""
        try:
            from_x = from_element.x + from_element.width // 2
            from_y = from_element.y + from_element.height // 2
            to_x = to_element.x + to_element.width // 2
            to_y = to_element.y + to_element.height // 2
            
            pyautogui.drag(from_x, from_y, to_x - from_x, to_y - from_y, duration=duration)
            logger.info(f"Drag von ({from_x}, {from_y}) zu ({to_x}, {to_y})")
            return True
            
        except Exception as e:
            logger.error(f"Drag-Fehler: {e}")
            return False
    
    def scroll(self, direction: str = 'up', clicks: int = 3, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        """Scrolle in eine Richtung"""
        try:
            if x is None or y is None:
                x, y = self.screen_width // 2, self.screen_height // 2
            
            scroll_amount = clicks if direction == 'up' else -clicks
            pyautogui.scroll(scroll_amount, x=x, y=y)
            logger.info(f"Gescrollt {direction} ({clicks} Klicks) bei ({x}, {y})")
            return True
            
        except Exception as e:
            logger.error(f"Scroll-Fehler: {e}")
            return False
    
    def execute_command(self, command: InteractionCommand) -> bool:
        """Führe Interaktionsbefehl aus"""
        try:
            # Verzögerung vor Aktion
            if command.delay_before > 0:
                time.sleep(command.delay_before)
            
            success = False
            
            if command.action == InteractionType.CLICK:
                if command.target:
                    success = self.click_element(command.target)
                elif command.coordinates:
                    pyautogui.click(command.coordinates[0], command.coordinates[1])
                    success = True
            
            elif command.action == InteractionType.DOUBLE_CLICK:
                if command.target:
                    success = self.click_element(command.target, 'double')
                elif command.coordinates:
                    pyautogui.doubleClick(command.coordinates[0], command.coordinates[1])
                    success = True
            
            elif command.action == InteractionType.RIGHT_CLICK:
                if command.target:
                    success = self.click_element(command.target, 'right')
                elif command.coordinates:
                    pyautogui.rightClick(command.coordinates[0], command.coordinates[1])
                    success = True
            
            elif command.action == InteractionType.TYPE_TEXT:
                if command.text:
                    success = self.type_text(command.text)
            
            elif command.action == InteractionType.KEY_PRESS:
                if command.keys and len(command.keys) == 1:
                    success = self.press_key(command.keys[0])
            
            elif command.action == InteractionType.KEY_COMBINATION:
                if command.keys:
                    success = self.press_key_combination(command.keys)
            
            elif command.action == InteractionType.DRAG:
                if command.target and command.coordinates:
                    # Erstelle Ziel-Element für Drag
                    target_element = VisualElement(
                        x=command.coordinates[0], y=command.coordinates[1],
                        width=1, height=1, confidence=1.0, element_type='drag_target'
                    )
                    success = self.drag_element(command.target, target_element, command.duration)
            
            elif command.action == InteractionType.SCROLL:
                direction = 'up' if command.coordinates and command.coordinates[1] > 0 else 'down'
                success = self.scroll(direction, abs(command.coordinates[1]) if command.coordinates else 3)
            
            elif command.action == InteractionType.SCREENSHOT:
                screenshot = self.take_screenshot()
                success = screenshot.size > 0
            
            # Verzögerung nach Aktion
            if command.delay_after > 0:
                time.sleep(command.delay_after)
            
            return success
            
        except Exception as e:
            logger.error(f"Befehlsausführung-Fehler: {e}")
            return False
    
    def see_and_click(self, target_description: str, action: str = 'click') -> bool:
        """Hauptfunktion: Sehe und klicke - kombinierte Funktionalität"""
        try:
            logger.info(f"Sehe und {action}: {target_description}")
            
            # 1. Screenshot erstellen
            screenshot = self.take_screenshot()
            if screenshot.size == 0:
                return False
            
            # 2. Visuelle Elemente finden
            elements = self.find_visual_elements(screenshot)
            
            # 3. Ziel-Element identifizieren
            target_element = None
            
            # Suche nach Text
            target_element = self.find_element_by_text(target_description)
            
            # Falls nicht gefunden, suche in allen Elementen
            if not target_element:
                for element in elements:
                    if (element.text and target_description.lower() in element.text.lower()) or \
                       (element.element_type and target_description.lower() in element.element_type.lower()):
                        target_element = element
                        break
            
            # 4. Aktion ausführen
            if target_element:
                if action == 'click':
                    return self.click_element(target_element)
                elif action == 'double_click':
                    return self.click_element(target_element, 'double')
                elif action == 'right_click':
                    return self.click_element(target_element, 'right')
                else:
                    logger.warning(f"Unbekannte Aktion: {action}")
                    return False
            else:
                logger.warning(f"Ziel-Element nicht gefunden: {target_description}")
                return False
                
        except Exception as e:
            logger.error(f"See-and-click-Fehler: {e}")
            return False
    
    def see_and_type(self, target_description: str, text: str) -> bool:
        """Sehe Element und tippe Text hinein"""
        try:
            # Erst klicken, dann tippen
            if self.see_and_click(target_description):
                time.sleep(0.2)  # Kurze Pause
                return self.type_text(text)
            return False
            
        except Exception as e:
            logger.error(f"See-and-type-Fehler: {e}")
            return False
    
    def start_monitoring(self):
        """Starte Event-Monitoring für Maus und Tastatur"""
        try:
            self.monitoring_active = True
            
            # Maus-Listener
            self.mouse_listener = MouseListener(
                on_click=self._on_mouse_click,
                on_scroll=self._on_mouse_scroll
            )
            
            # Tastatur-Listener
            self.keyboard_listener = KeyboardListener(
                on_press=self._on_key_press,
                on_release=self._on_key_release
            )
            
            self.mouse_listener.start()
            self.keyboard_listener.start()
            
            logger.info("Event-Monitoring gestartet")
            
        except Exception as e:
            logger.error(f"Monitoring-Start-Fehler: {e}")
    
    def stop_monitoring(self):
        """Stoppe Event-Monitoring"""
        try:
            self.monitoring_active = False
            
            if self.mouse_listener:
                self.mouse_listener.stop()
            if self.keyboard_listener:
                self.keyboard_listener.stop()
            
            logger.info("Event-Monitoring gestoppt")
            
        except Exception as e:
            logger.error(f"Monitoring-Stop-Fehler: {e}")
    
    def _on_mouse_click(self, x, y, button, pressed):
        """Maus-Klick-Event-Handler"""
        if self.monitoring_active and pressed:
            logger.debug(f"Maus-Klick: ({x}, {y}) - {button}")
    
    def _on_mouse_scroll(self, x, y, dx, dy):
        """Maus-Scroll-Event-Handler"""
        if self.monitoring_active:
            logger.debug(f"Maus-Scroll: ({x}, {y}) - dx={dx}, dy={dy}")
    
    def _on_key_press(self, key):
        """Tastendruck-Event-Handler"""
        if self.monitoring_active:
            logger.debug(f"Taste gedrückt: {key}")
    
    def _on_key_release(self, key):
        """Taste-losgelassen-Event-Handler"""
        if self.monitoring_active:
            logger.debug(f"Taste losgelassen: {key}")
    
    def get_screen_info(self) -> Dict[str, Any]:
        """Hole Bildschirm-Informationen"""
        return {
            'width': self.screen_width,
            'height': self.screen_height,
            'elements_cached': len(self.element_cache.get('latest', [])),
            'screenshot_cached': 'latest' in self.screenshot_cache,
            'monitoring_active': self.monitoring_active
        }
    
    def save_screenshot(self, filename: str, region: Optional[Tuple[int, int, int, int]] = None) -> bool:
        """Speichere Screenshot"""
        try:
            screenshot = self.take_screenshot(region)
            if screenshot.size > 0:
                cv2.imwrite(filename, screenshot)
                logger.info(f"Screenshot gespeichert: {filename}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Screenshot-Speicher-Fehler: {e}")
            return False
