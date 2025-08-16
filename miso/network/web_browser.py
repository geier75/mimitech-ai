#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Web Browser Module

Dieses Modul ermöglicht MISO, einen Webbrowser zu steuern und im Internet zu navigieren.
Es bietet Funktionen zum Öffnen von Webseiten, Ausfüllen von Formularen, Klicken auf Links
und Extrahieren von Informationen aus Webseiten.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import json
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Versuche, optionale Abhängigkeiten zu importieren
try:
    import selenium
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

# Importiere InternetAccess
from miso.network.internet_access import InternetAccess, SecurityLevel

# Konfiguriere Logging
logger = logging.getLogger("MISO.network.web_browser")

class WebBrowser:
    """
    Hauptklasse für die Browsersteuerung.
    Ermöglicht MISO, einen Webbrowser zu steuern und im Internet zu navigieren.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den WebBrowser
        
        Args:
            config: Konfigurationsparameter
        """
        self.config = config or {}
        
        # Setze Standardkonfiguration
        self.config.setdefault("security_level", SecurityLevel.MEDIUM)
        self.config.setdefault("headless", False)  # Sichtbarer Browser für Debugging
        self.config.setdefault("browser_type", "chrome")
        self.config.setdefault("timeout", 30)
        self.config.setdefault("user_data_dir", str(Path.home() / ".miso" / "browser_data"))
        
        # Initialisiere InternetAccess
        self.internet_access = InternetAccess(self.config)
        
        # Initialisiere Webdriver
        self.driver = None
        self.is_initialized = False
        
        # Überprüfe, ob Selenium verfügbar ist
        if not SELENIUM_AVAILABLE:
            logger.warning("Selenium ist nicht installiert. Browser-Automatisierung ist nicht verfügbar.")
            return
        
        logger.info("WebBrowser initialisiert")
    
    def is_browser_open(self) -> bool:
        """
        Überprüft, ob der Browser geöffnet ist
        
        Returns:
            True, wenn der Browser geöffnet ist, sonst False
        """
        return self.is_initialized and self.driver is not None
    
    def initialize(self) -> bool:
        """
        Initialisiert den Webdriver
        
        Returns:
            True, wenn die Initialisierung erfolgreich war, sonst False
        """
        if not SELENIUM_AVAILABLE:
            logger.error("Selenium ist nicht installiert. Browser-Automatisierung ist nicht verfügbar.")
            return False
        
        if self.is_initialized:
            logger.info("Webdriver ist bereits initialisiert")
            return True
        
        try:
            browser_type = self.config["browser_type"].lower()
            
            if browser_type == "chrome":
                from selenium.webdriver.chrome.options import Options
                from selenium.webdriver.chrome.service import Service
                
                options = Options()
                if self.config["headless"]:
                    options.add_argument("--headless")
                
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument(f"--user-data-dir={self.config['user_data_dir']}")
                options.add_argument(f"user-agent={self.config.get('user_agent', 'MISO/1.0')}")
                
                self.driver = webdriver.Chrome(options=options)
                
            elif browser_type == "firefox":
                from selenium.webdriver.firefox.options import Options
                from selenium.webdriver.firefox.service import Service
                
                options = Options()
                if self.config["headless"]:
                    options.add_argument("--headless")
                
                options.set_preference("browser.download.folderList", 2)
                options.set_preference("browser.download.dir", self.config.get("download_dir", ""))
                options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/octet-stream")
                
                self.driver = webdriver.Firefox(options=options)
                
            elif browser_type == "safari":
                self.driver = webdriver.Safari()
                
            else:
                logger.error(f"Nicht unterstützter Browser-Typ: {browser_type}")
                return False
            
            # Setze Timeout
            self.driver.set_page_load_timeout(self.config["timeout"])
            
            self.is_initialized = True
            logger.info(f"Webdriver für {browser_type} erfolgreich initialisiert")
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung des Webdrivers: {e}")
            return False
    
    def close(self):
        """Schließt den Webdriver"""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("Webdriver geschlossen")
            except Exception as e:
                logger.error(f"Fehler beim Schließen des Webdrivers: {e}")
            finally:
                self.driver = None
                self.is_initialized = False
    
    def _check_url_security(self, url: str) -> bool:
        """
        Überprüft, ob eine URL den Sicherheitsrichtlinien entspricht
        
        Args:
            url: Die zu überprüfende URL
            
        Returns:
            True, wenn die URL sicher ist, sonst False
        """
        return self.internet_access._check_url_security(url)
    
    def open_url(self, url: str) -> bool:
        """
        Öffnet eine URL im Browser
        
        Args:
            url: Die zu öffnende URL
            
        Returns:
            True, wenn die URL erfolgreich geöffnet wurde, sonst False
        """
        # Überprüfe URL-Sicherheit
        if not self._check_url_security(url):
            logger.error(f"Sicherheitsüberprüfung fehlgeschlagen für URL: {url}")
            return False
        
        # Initialisiere Webdriver, falls noch nicht geschehen
        if not self.is_initialized and not self.initialize():
            return False
        
        logger.info(f"Öffne URL: {url}")
        
        try:
            self.driver.get(url)
            logger.info(f"URL erfolgreich geöffnet: {url}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Öffnen der URL {url}: {e}")
            return False
    
    def get_current_url(self) -> Optional[str]:
        """
        Gibt die aktuelle URL zurück
        
        Returns:
            Die aktuelle URL oder None, wenn kein Browser geöffnet ist
        """
        if not self.is_initialized:
            logger.error("Webdriver ist nicht initialisiert")
            return None
        
        try:
            return self.driver.current_url
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der aktuellen URL: {e}")
            return None
    
    def get_page_source(self) -> Optional[str]:
        """
        Gibt den Quellcode der aktuellen Seite zurück
        
        Returns:
            Der Quellcode der aktuellen Seite oder None bei Fehler
        """
        if not self.is_initialized:
            logger.error("Webdriver ist nicht initialisiert")
            return None
        
        try:
            return self.driver.page_source
        except Exception as e:
            logger.error(f"Fehler beim Abrufen des Seitenquellcodes: {e}")
            return None
    
    def find_element(self, selector: str, by: str = By.CSS_SELECTOR, wait_time: int = None) -> Optional[Any]:
        """
        Findet ein Element auf der aktuellen Seite
        
        Args:
            selector: Der Selektor für das Element
            by: Die Methode zum Finden des Elements (CSS_SELECTOR, XPATH, ID, etc.)
            wait_time: Optionale Wartezeit in Sekunden
            
        Returns:
            Das gefundene Element oder None, wenn kein Element gefunden wurde
        """
        if not self.is_initialized:
            logger.error("Webdriver ist nicht initialisiert")
            return None
        
        try:
            if wait_time is not None:
                element = WebDriverWait(self.driver, wait_time).until(
                    EC.presence_of_element_located((by, selector))
                )
            else:
                element = self.driver.find_element(by, selector)
            
            return element
        except (TimeoutException, NoSuchElementException) as e:
            logger.warning(f"Element nicht gefunden: {selector} (by={by})")
            return None
        except Exception as e:
            logger.error(f"Fehler beim Suchen des Elements {selector}: {e}")
            return None
    
    def find_elements(self, selector: str, by: str = By.CSS_SELECTOR) -> List[Any]:
        """
        Findet alle Elemente auf der aktuellen Seite, die dem Selektor entsprechen
        
        Args:
            selector: Der Selektor für die Elemente
            by: Die Methode zum Finden der Elemente (CSS_SELECTOR, XPATH, ID, etc.)
            
        Returns:
            Eine Liste der gefundenen Elemente oder eine leere Liste, wenn keine Elemente gefunden wurden
        """
        if not self.is_initialized:
            logger.error("Webdriver ist nicht initialisiert")
            return []
        
        try:
            elements = self.driver.find_elements(by, selector)
            return elements
        except Exception as e:
            logger.error(f"Fehler beim Suchen der Elemente {selector}: {e}")
            return []
    
    def click_element(self, selector: str, by: str = By.CSS_SELECTOR, wait_time: int = None) -> bool:
        """
        Klickt auf ein Element
        
        Args:
            selector: Der Selektor für das Element
            by: Die Methode zum Finden des Elements (CSS_SELECTOR, XPATH, ID, etc.)
            wait_time: Optionale Wartezeit in Sekunden
            
        Returns:
            True, wenn das Element erfolgreich angeklickt wurde, sonst False
        """
        element = self.find_element(selector, by, wait_time)
        if element is None:
            return False
        
        try:
            element.click()
            logger.info(f"Element angeklickt: {selector}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Klicken auf Element {selector}: {e}")
            return False
    
    def fill_input(self, selector: str, text: str, by: str = By.CSS_SELECTOR, clear_first: bool = True) -> bool:
        """
        Füllt ein Eingabefeld aus
        
        Args:
            selector: Der Selektor für das Eingabefeld
            text: Der einzugebende Text
            by: Die Methode zum Finden des Elements (CSS_SELECTOR, XPATH, ID, etc.)
            clear_first: True, wenn das Feld zuerst geleert werden soll
            
        Returns:
            True, wenn der Text erfolgreich eingegeben wurde, sonst False
        """
        element = self.find_element(selector, by)
        if element is None:
            return False
        
        try:
            if clear_first:
                element.clear()
            
            element.send_keys(text)
            logger.info(f"Text in Eingabefeld eingegeben: {selector}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Ausfüllen des Eingabefelds {selector}: {e}")
            return False
    
    def submit_form(self, form_selector: str = None, by: str = By.CSS_SELECTOR) -> bool:
        """
        Sendet ein Formular ab
        
        Args:
            form_selector: Der Selektor für das Formular, falls None, wird das erste gefundene Formular verwendet
            by: Die Methode zum Finden des Elements (CSS_SELECTOR, XPATH, ID, etc.)
            
        Returns:
            True, wenn das Formular erfolgreich abgesendet wurde, sonst False
        """
        if form_selector:
            form = self.find_element(form_selector, by)
            if form is None:
                return False
        else:
            forms = self.find_elements("form", By.TAG_NAME)
            if not forms:
                logger.error("Kein Formular gefunden")
                return False
            form = forms[0]
        
        try:
            form.submit()
            logger.info("Formular abgesendet")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Absenden des Formulars: {e}")
            return False
    
    def extract_text(self, selector: str, by: str = By.CSS_SELECTOR) -> Optional[str]:
        """
        Extrahiert den Text eines Elements
        
        Args:
            selector: Der Selektor für das Element
            by: Die Methode zum Finden des Elements (CSS_SELECTOR, XPATH, ID, etc.)
            
        Returns:
            Der Text des Elements oder None bei Fehler
        """
        element = self.find_element(selector, by)
        if element is None:
            return None
        
        try:
            return element.text
        except Exception as e:
            logger.error(f"Fehler beim Extrahieren des Textes aus Element {selector}: {e}")
            return None
    
    def extract_attribute(self, selector: str, attribute: str, by: str = By.CSS_SELECTOR) -> Optional[str]:
        """
        Extrahiert ein Attribut eines Elements
        
        Args:
            selector: Der Selektor für das Element
            attribute: Das zu extrahierende Attribut
            by: Die Methode zum Finden des Elements (CSS_SELECTOR, XPATH, ID, etc.)
            
        Returns:
            Der Wert des Attributs oder None bei Fehler
        """
        element = self.find_element(selector, by)
        if element is None:
            return None
        
        try:
            return element.get_attribute(attribute)
        except Exception as e:
            logger.error(f"Fehler beim Extrahieren des Attributs {attribute} aus Element {selector}: {e}")
            return None
    
    def take_screenshot(self, filename: Optional[str] = None) -> Optional[str]:
        """
        Erstellt einen Screenshot der aktuellen Seite
        
        Args:
            filename: Optionaler Dateiname, falls nicht angegeben, wird ein Zeitstempel verwendet
            
        Returns:
            Der Pfad zum Screenshot oder None bei Fehler
        """
        if not self.is_initialized:
            logger.error("Webdriver ist nicht initialisiert")
            return None
        
        if not filename:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"screenshot_{timestamp}.png"
        
        screenshot_dir = os.path.join(self.config.get("download_dir", ""), "screenshots")
        os.makedirs(screenshot_dir, exist_ok=True)
        
        screenshot_path = os.path.join(screenshot_dir, filename)
        
        try:
            self.driver.save_screenshot(screenshot_path)
            logger.info(f"Screenshot erstellt: {screenshot_path}")
            return screenshot_path
        except Exception as e:
            logger.error(f"Fehler beim Erstellen des Screenshots: {e}")
            return None
    
    def execute_javascript(self, script: str, *args) -> Any:
        """
        Führt JavaScript auf der aktuellen Seite aus
        
        Args:
            script: Das auszuführende JavaScript
            *args: Argumente für das JavaScript
            
        Returns:
            Das Ergebnis des JavaScript oder None bei Fehler
        """
        if not self.is_initialized:
            logger.error("Webdriver ist nicht initialisiert")
            return None
        
        # Überprüfe Sicherheitsstufe für JavaScript-Ausführung
        if self.config["security_level"] == SecurityLevel.HIGH:
            logger.error("JavaScript-Ausführung ist bei hoher Sicherheitsstufe nicht erlaubt")
            return None
        
        try:
            return self.driver.execute_script(script, *args)
        except Exception as e:
            logger.error(f"Fehler bei der Ausführung von JavaScript: {e}")
            return None
    
    def scroll_to(self, x: int = 0, y: int = 0):
        """
        Scrollt zu einer bestimmten Position
        
        Args:
            x: X-Koordinate
            y: Y-Koordinate
        """
        if not self.is_initialized:
            logger.error("Webdriver ist nicht initialisiert")
            return
        
        try:
            self.driver.execute_script(f"window.scrollTo({x}, {y});")
            logger.info(f"Zu Position ({x}, {y}) gescrollt")
        except Exception as e:
            logger.error(f"Fehler beim Scrollen zu Position ({x}, {y}): {e}")
    
    def scroll_to_element(self, selector: str, by: str = By.CSS_SELECTOR):
        """
        Scrollt zu einem Element
        
        Args:
            selector: Der Selektor für das Element
            by: Die Methode zum Finden des Elements (CSS_SELECTOR, XPATH, ID, etc.)
        """
        element = self.find_element(selector, by)
        if element is None:
            return
        
        try:
            self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
            logger.info(f"Zu Element {selector} gescrollt")
        except Exception as e:
            logger.error(f"Fehler beim Scrollen zu Element {selector}: {e}")
    
    def wait_for_element(self, selector: str, by: str = By.CSS_SELECTOR, timeout: int = None) -> bool:
        """
        Wartet auf das Erscheinen eines Elements
        
        Args:
            selector: Der Selektor für das Element
            by: Die Methode zum Finden des Elements (CSS_SELECTOR, XPATH, ID, etc.)
            timeout: Timeout in Sekunden, falls None, wird der Standard-Timeout verwendet
            
        Returns:
            True, wenn das Element erschienen ist, sonst False
        """
        if not self.is_initialized:
            logger.error("Webdriver ist nicht initialisiert")
            return False
        
        if timeout is None:
            timeout = self.config["timeout"]
        
        try:
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, selector))
            )
            logger.info(f"Element {selector} erschienen")
            return True
        except TimeoutException:
            logger.warning(f"Timeout beim Warten auf Element {selector}")
            return False
        except Exception as e:
            logger.error(f"Fehler beim Warten auf Element {selector}: {e}")
            return False
    
    def wait_for_url(self, url_pattern: str, timeout: int = None) -> bool:
        """
        Wartet, bis die URL einem Muster entspricht
        
        Args:
            url_pattern: Das Muster für die URL (regulärer Ausdruck)
            timeout: Timeout in Sekunden, falls None, wird der Standard-Timeout verwendet
            
        Returns:
            True, wenn die URL dem Muster entspricht, sonst False
        """
        if not self.is_initialized:
            logger.error("Webdriver ist nicht initialisiert")
            return False
        
        if timeout is None:
            timeout = self.config["timeout"]
        
        try:
            WebDriverWait(self.driver, timeout).until(
                lambda driver: re.search(url_pattern, driver.current_url) is not None
            )
            logger.info(f"URL entspricht Muster {url_pattern}")
            return True
        except TimeoutException:
            logger.warning(f"Timeout beim Warten auf URL-Muster {url_pattern}")
            return False
        except Exception as e:
            logger.error(f"Fehler beim Warten auf URL-Muster {url_pattern}: {e}")
            return False
    
    def back(self) -> bool:
        """
        Navigiert zurück
        
        Returns:
            True, wenn erfolgreich, sonst False
        """
        if not self.is_initialized:
            logger.error("Webdriver ist nicht initialisiert")
            return False
        
        try:
            self.driver.back()
            logger.info("Zurück navigiert")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Zurücknavigieren: {e}")
            return False
    
    def forward(self) -> bool:
        """
        Navigiert vorwärts
        
        Returns:
            True, wenn erfolgreich, sonst False
        """
        if not self.is_initialized:
            logger.error("Webdriver ist nicht initialisiert")
            return False
        
        try:
            self.driver.forward()
            logger.info("Vorwärts navigiert")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Vorwärtsnavigieren: {e}")
            return False
    
    def refresh(self) -> bool:
        """
        Aktualisiert die Seite
        
        Returns:
            True, wenn erfolgreich, sonst False
        """
        if not self.is_initialized:
            logger.error("Webdriver ist nicht initialisiert")
            return False
        
        try:
            self.driver.refresh()
            logger.info("Seite aktualisiert")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Aktualisieren der Seite: {e}")
            return False
    
    def __del__(self):
        """Destruktor"""
        self.close()
