#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Internet-Zugriff Demo (Eigenständige Version)

Dieses Skript demonstriert die Fähigkeit von MISO, auf das Internet zuzugreifen,
Webseiten zu besuchen und mit Webdiensten zu interagieren.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import json
import re
import urllib.parse
import urllib.request
import queue
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("miso_internet_demo.log")
    ]
)
logger = logging.getLogger("MISO.demo.internet")

# Prüfe, ob optionale Abhängigkeiten verfügbar sind
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests ist nicht installiert. Einige Funktionen sind eingeschränkt.")

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logger.warning("selenium ist nicht installiert. Browser-Automatisierung ist nicht verfügbar.")
    
    # Dummy-Klasse für By, falls Selenium nicht verfügbar ist
    class By:
        CSS_SELECTOR = "css selector"
        XPATH = "xpath"
        ID = "id"
        TAG_NAME = "tag name"

# Definiere Sicherheitsstufen
class SecurityLevel(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()

# Definiere Befehlsstatus
class CommandStatus(Enum):
    PENDING = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELED = auto()

# Definiere Befehlspriorität
class CommandPriority(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

class Command:
    """Repräsentiert einen Befehl, der von MISO ausgeführt werden soll"""
    
    def __init__(self, command_text: str, priority: CommandPriority = CommandPriority.MEDIUM):
        """
        Initialisiert einen neuen Befehl
        
        Args:
            command_text: Der Befehlstext
            priority: Die Priorität des Befehls
        """
        self.command_text = command_text
        self.priority = priority
        self.status = CommandStatus.PENDING
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None
        self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generiert eine eindeutige ID für den Befehl"""
        import uuid
        return str(uuid.uuid4())
    
    def execute(self, executor):
        """
        Führt den Befehl aus
        
        Args:
            executor: Der CommandExecutor, der den Befehl ausführt
        """
        self.status = CommandStatus.EXECUTING
        self.start_time = time.time()
        
        try:
            self.result = executor.process_command(self.command_text)
            self.status = CommandStatus.COMPLETED
        except Exception as e:
            self.error = str(e)
            self.status = CommandStatus.FAILED
            logger.error(f"Fehler bei der Ausführung des Befehls '{self.command_text}': {e}")
        
        self.end_time = time.time()
        return self.result
    
    def __str__(self) -> str:
        return f"Command({self.id}, '{self.command_text}', {self.status.name})"

class InternetAccess:
    """
    Hauptklasse für den Internetzugriff.
    Ermöglicht MISO, auf das Internet zuzugreifen und mit Webdiensten zu interagieren.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den InternetAccess
        
        Args:
            config: Konfigurationsparameter
        """
        self.config = config or {}
        
        # Setze Standardkonfiguration
        self.config.setdefault("security_level", SecurityLevel.MEDIUM)
        self.config.setdefault("user_agent", "MISO/1.0")
        self.config.setdefault("timeout", 30)
        self.config.setdefault("max_retries", 3)
        self.config.setdefault("download_dir", str(Path.home() / "Downloads" / "MISO"))
        
        # Erstelle Download-Verzeichnis, falls es nicht existiert
        os.makedirs(self.config["download_dir"], exist_ok=True)
        
        # Initialisiere Session, falls requests verfügbar ist
        self.session = None
        if REQUESTS_AVAILABLE:
            self.session = requests.Session()
            self.session.headers.update({"User-Agent": self.config["user_agent"]})
        
        logger.info("InternetAccess initialisiert")
    
    def is_url_safe(self, url: str) -> bool:
        """
        Überprüft, ob eine URL sicher ist
        
        Args:
            url: Die zu überprüfende URL
            
        Returns:
            True, wenn die URL sicher ist, sonst False
        """
        # Überprüfe, ob die URL ein gültiges Schema hat
        parsed_url = urllib.parse.urlparse(url)
        if parsed_url.scheme not in ["http", "https"]:
            logger.warning(f"Unsicheres URL-Schema: {parsed_url.scheme}")
            return False
        
        # Überprüfe auf bekannte bösartige Domains
        blacklisted_domains = ["malware.com", "phishing.net", "evil.org"]
        if any(domain in parsed_url.netloc for domain in blacklisted_domains):
            logger.warning(f"Blockierte Domain: {parsed_url.netloc}")
            return False
        
        # Überprüfe auf verdächtige Pfade
        suspicious_paths = ["/admin", "/login", "/password", "/hack"]
        if any(path in parsed_url.path for path in suspicious_paths) and self.config["security_level"] == SecurityLevel.HIGH:
            logger.warning(f"Verdächtiger Pfad: {parsed_url.path}")
            return False
        
        return True
    
    def get_webpage(self, url: str, params: Dict[str, str] = None) -> Optional[str]:
        """
        Ruft eine Webseite ab
        
        Args:
            url: Die URL der Webseite
            params: Optionale Parameter für die Anfrage
            
        Returns:
            Der Inhalt der Webseite als String oder None bei Fehler
        """
        if not self.is_url_safe(url):
            logger.error(f"URL {url} ist nicht sicher")
            return None
        
        logger.info(f"Rufe Webseite ab: {url}")
        
        try:
            if REQUESTS_AVAILABLE:
                response = self.session.get(url, params=params, timeout=self.config["timeout"])
                response.raise_for_status()
                return response.text
            else:
                # Fallback zu urllib
                full_url = url
                if params:
                    query_string = urllib.parse.urlencode(params)
                    full_url = f"{url}?{query_string}"
                
                headers = {"User-Agent": self.config["user_agent"]}
                req = urllib.request.Request(full_url, headers=headers)
                with urllib.request.urlopen(req, timeout=self.config["timeout"]) as response:
                    return response.read().decode('utf-8')
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Webseite {url}: {e}")
            return None
    
    def download_file(self, url: str, filename: Optional[str] = None) -> Optional[str]:
        """
        Lädt eine Datei herunter
        
        Args:
            url: Die URL der Datei
            filename: Optionaler Dateiname für die heruntergeladene Datei
            
        Returns:
            Der Pfad zur heruntergeladenen Datei oder None bei Fehler
        """
        if not self.is_url_safe(url):
            logger.error(f"URL {url} ist nicht sicher")
            return None
        
        # Bestimme Dateinamen, falls nicht angegeben
        if not filename:
            parsed_url = urllib.parse.urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename:
                filename = "download.dat"
        
        # Bestimme Pfad für die heruntergeladene Datei
        download_path = os.path.join(self.config["download_dir"], filename)
        
        logger.info(f"Lade Datei herunter: {url} -> {download_path}")
        
        try:
            if REQUESTS_AVAILABLE:
                response = self.session.get(url, stream=True, timeout=self.config["timeout"])
                response.raise_for_status()
                
                with open(download_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                # Fallback zu urllib
                headers = {"User-Agent": self.config["user_agent"]}
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=self.config["timeout"]) as response:
                    with open(download_path, 'wb') as f:
                        f.write(response.read())
            
            logger.info(f"Datei erfolgreich heruntergeladen: {download_path}")
            return download_path
        except Exception as e:
            logger.error(f"Fehler beim Herunterladen der Datei {url}: {e}")
            return None

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
            logger.info(f"Initialisiere Webdriver ({self.config['browser_type']})")
            
            # Konfiguriere Browser-Optionen
            if self.config["browser_type"].lower() == "chrome":
                from selenium.webdriver.chrome.options import Options
                options = Options()
                if self.config["headless"]:
                    options.add_argument("--headless")
                options.add_argument(f"--user-agent={self.config.get('user_agent', 'MISO/1.0')}")
                options.add_argument(f"--user-data-dir={self.config['user_data_dir']}")
                
                self.driver = webdriver.Chrome(options=options)
            
            elif self.config["browser_type"].lower() == "firefox":
                from selenium.webdriver.firefox.options import Options
                options = Options()
                if self.config["headless"]:
                    options.add_argument("--headless")
                
                self.driver = webdriver.Firefox(options=options)
            
            else:
                logger.error(f"Nicht unterstützter Browser-Typ: {self.config['browser_type']}")
                return False
            
            # Setze Timeout
            self.driver.implicitly_wait(self.config["timeout"])
            
            self.is_initialized = True
            logger.info("Webdriver erfolgreich initialisiert")
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung des Webdrivers: {e}")
            return False
    
    def close(self) -> bool:
        """
        Schließt den Browser
        
        Returns:
            True, wenn der Browser erfolgreich geschlossen wurde, sonst False
        """
        if not self.is_initialized or not self.driver:
            logger.warning("Browser ist nicht initialisiert")
            return False
        
        try:
            logger.info("Schließe Browser")
            self.driver.quit()
            self.driver = None
            self.is_initialized = False
            return True
        except Exception as e:
            logger.error(f"Fehler beim Schließen des Browsers: {e}")
            return False
    
    def open_url(self, url: str) -> bool:
        """
        Öffnet eine URL im Browser
        
        Args:
            url: Die zu öffnende URL
            
        Returns:
            True, wenn die URL erfolgreich geöffnet wurde, sonst False
        """
        # Überprüfe, ob die URL sicher ist
        if not self.internet_access.is_url_safe(url):
            logger.error(f"URL {url} ist nicht sicher")
            return False
        
        # Initialisiere Browser, falls noch nicht geschehen
        if not self.is_initialized:
            if not self.initialize():
                return False
        
        try:
            logger.info(f"Öffne URL: {url}")
            self.driver.get(url)
            return True
        except Exception as e:
            logger.error(f"Fehler beim Öffnen der URL {url}: {e}")
            return False
    
    def get_current_url(self) -> Optional[str]:
        """
        Gibt die aktuelle URL zurück
        
        Returns:
            Die aktuelle URL oder None, wenn der Browser nicht initialisiert ist
        """
        if not self.is_initialized or not self.driver:
            return None
        
        try:
            return self.driver.current_url
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der aktuellen URL: {e}")
            return None
    
    def take_screenshot(self, filename: Optional[str] = None) -> Optional[str]:
        """
        Erstellt einen Screenshot der aktuellen Seite
        
        Args:
            filename: Optionaler Dateiname für den Screenshot
            
        Returns:
            Der Pfad zum Screenshot oder None bei Fehler
        """
        if not self.is_initialized or not self.driver:
            logger.error("Browser ist nicht initialisiert")
            return None
        
        try:
            # Bestimme Dateinamen, falls nicht angegeben
            if not filename:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"screenshot_{timestamp}.png"
            
            # Stelle sicher, dass der Dateiname eine .png-Erweiterung hat
            if not filename.lower().endswith(".png"):
                filename += ".png"
            
            # Bestimme Pfad für den Screenshot
            screenshot_dir = os.path.join(self.config.get("download_dir", os.path.expanduser("~")), "screenshots")
            os.makedirs(screenshot_dir, exist_ok=True)
            screenshot_path = os.path.join(screenshot_dir, filename)
            
            logger.info(f"Erstelle Screenshot: {screenshot_path}")
            self.driver.save_screenshot(screenshot_path)
            
            return screenshot_path
        except Exception as e:
            logger.error(f"Fehler beim Erstellen des Screenshots: {e}")
            return None
    
    def fill_input(self, selector: str, text: str, by: str = By.CSS_SELECTOR) -> bool:
        """
        Füllt ein Eingabefeld aus
        
        Args:
            selector: Der Selektor für das Eingabefeld
            text: Der einzugebende Text
            by: Die Selektormethode (CSS_SELECTOR, XPATH, ID, ...)
            
        Returns:
            True, wenn das Eingabefeld erfolgreich ausgefüllt wurde, sonst False
        """
        if not self.is_initialized or not self.driver:
            logger.error("Browser ist nicht initialisiert")
            return False
        
        try:
            logger.info(f"Fülle Eingabefeld aus: {selector} -> {text}")
            element = self.driver.find_element(by, selector)
            element.clear()
            element.send_keys(text)
            return True
        except Exception as e:
            logger.error(f"Fehler beim Ausfüllen des Eingabefelds {selector}: {e}")
            return False
    
    def click_element(self, selector: str, by: str = By.CSS_SELECTOR) -> bool:
        """
        Klickt auf ein Element
        
        Args:
            selector: Der Selektor für das Element
            by: Die Selektormethode (CSS_SELECTOR, XPATH, ID, ...)
            
        Returns:
            True, wenn das Element erfolgreich angeklickt wurde, sonst False
        """
        if not self.is_initialized or not self.driver:
            logger.error("Browser ist nicht initialisiert")
            return False
        
        try:
            logger.info(f"Klicke auf Element: {selector}")
            element = self.driver.find_element(by, selector)
            element.click()
            return True
        except Exception as e:
            logger.error(f"Fehler beim Klicken auf Element {selector}: {e}")
            return False
    
    def execute_javascript(self, script: str, *args) -> Any:
        """
        Führt JavaScript auf der Seite aus
        
        Args:
            script: Das auszuführende JavaScript
            *args: Argumente für das JavaScript
            
        Returns:
            Das Ergebnis des JavaScript-Aufrufs oder None bei Fehler
        """
        if not self.is_initialized or not self.driver:
            logger.error("Browser ist nicht initialisiert")
            return None
        
        try:
            logger.info(f"Führe JavaScript aus: {script[:50]}...")
            return self.driver.execute_script(script, *args)
        except Exception as e:
            logger.error(f"Fehler beim Ausführen von JavaScript: {e}")
            return None

class CommandExecutor:
    """
    Hauptklasse für die Ausführung von Befehlen.
    Versteht und führt Befehle aus, die von MISO empfangen werden.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den CommandExecutor
        
        Args:
            config: Konfigurationsparameter
        """
        self.config = config or {}
        self.command_queue = None
        self.command_history = []
        self.running = False
        self.executor_thread = None
        
        # Initialisiere Module
        self.modules = {}
        
        # Befehlsmuster und zugehörige Handler
        self.command_patterns = {
            r'(?i)internet|web|browse|suche|download|url': self._handle_internet_command,
        }
        
        # Initialisiere Queue
        self.command_queue = queue.PriorityQueue()
        
        logger.info("CommandExecutor initialisiert")
    
    def start(self):
        """
        Startet den CommandExecutor
        """
        if self.running:
            logger.warning("CommandExecutor läuft bereits")
            return False
        
        self.running = True
        import threading
        self.executor_thread = threading.Thread(target=self._executor_loop)
        self.executor_thread.daemon = True
        self.executor_thread.start()
        
        logger.info("CommandExecutor gestartet")
        return True
    
    def stop(self):
        """
        Stoppt den CommandExecutor
        """
        if not self.running:
            logger.warning("CommandExecutor läuft nicht")
            return False
        
        self.running = False
        if self.executor_thread:
            self.executor_thread.join(timeout=5.0)
        
        logger.info("CommandExecutor gestoppt")
        return True
    
    def _executor_loop(self):
        """
        Hauptschleife für die Befehlsausführung
        """
        logger.info("Executor-Schleife gestartet")
        
        while self.running:
            try:
                # Hole den nächsten Befehl aus der Warteschlange
                priority, command = self.command_queue.get(block=True, timeout=1.0)
                
                # Führe den Befehl aus
                logger.info(f"Führe Befehl aus: {command}")
                command.execute(self)
                
                # Füge den Befehl zur Historie hinzu
                self.command_history.append(command)
                
                # Markiere den Befehl als erledigt
                self.command_queue.task_done()
                
            except queue.Empty:
                # Keine Befehle in der Warteschlange
                pass
            except Exception as e:
                logger.error(f"Fehler in der Executor-Schleife: {e}")
        
        logger.info("Executor-Schleife beendet")
    
    def add_command(self, command_text: str, priority: CommandPriority = CommandPriority.MEDIUM) -> Command:
        """
        Fügt einen Befehl zur Ausführungswarteschlange hinzu
        
        Args:
            command_text: Der Befehlstext
            priority: Die Priorität des Befehls
            
        Returns:
            Das erstellte Command-Objekt
        """
        command = Command(command_text, priority)
        self.command_queue.put((priority.value, command))
        logger.info(f"Befehl hinzugefügt: {command}")
        return command
    
    def process_command(self, command_text: str) -> Dict[str, Any]:
        """
        Verarbeitet einen Befehl
        
        Args:
            command_text: Der zu verarbeitende Befehlstext
            
        Returns:
            Das Ergebnis der Befehlsausführung
        """
        logger.info(f"Verarbeite Befehl: '{command_text}'")
        
        # Bestimme den Befehlstyp und rufe den entsprechenden Handler auf
        for pattern, handler in self.command_patterns.items():
            if re.search(pattern, command_text):
                return handler(command_text)
        
        # Fallback: Allgemeiner Befehlshandler
        return {"status": "error", "message": "Unbekannter Befehl"}
    
    def _handle_internet_command(self, command_text: str) -> Dict[str, Any]:
        """
        Verarbeitet einen Internet-Befehl
        
        Args:
            command_text: Der zu verarbeitende Befehlstext
            
        Returns:
            Das Ergebnis der Befehlsausführung
        """
        # Initialisiere Internet-Module, falls noch nicht geschehen
        if "internet_access" not in self.modules:
            self.modules["internet_access"] = {
                "internet_access": InternetAccess(self.config),
                "web_browser": WebBrowser(self.config)
            }
        
        # Hole die Internet-Module
        internet_access = self.modules["internet_access"]["internet_access"]
        web_browser = self.modules["internet_access"]["web_browser"]
        
        # Extrahiere relevante Informationen aus dem Befehlstext
        command_lower = command_text.lower()
        
        # Webseite öffnen
        if "öffne" in command_lower or "browse" in command_lower or "gehe zu" in command_lower:
            url_match = re.search(r'(?:öffne|browse|gehe zu)\s+(?:die\s+(?:seite|webseite|website)\s+)?(?:https?://)?([\w.-]+\.[a-z]{2,}(?:/[^\s]*)?)', command_text, re.IGNORECASE)
            if url_match:
                domain = url_match.group(1)
                url = f"https://{domain}" if not domain.startswith("http") else domain
                
                # Sicherheitscheck
                if not internet_access.is_url_safe(url):
                    return {"status": "error", "message": f"Die URL {url} wurde als unsicher eingestuft"}
                
                # Öffne die Webseite
                success = web_browser.open_url(url)
                if success:
                    return {"status": "success", "message": f"Webseite {url} geöffnet"}
                else:
                    return {"status": "error", "message": f"Konnte Webseite {url} nicht öffnen"}
        
        # Websuche durchführen
        elif "suche" in command_lower or "search" in command_lower:
            search_match = re.search(r'(?:suche|search)(?:\s+nach)?\s+"([^"]+)"', command_text)
            if not search_match:
                search_match = re.search(r'(?:suche|search)(?:\s+nach)?\s+([^\s].+)$', command_text)
            
            if search_match:
                query = search_match.group(1).strip()
                search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
                
                # Öffne die Suchseite
                success = web_browser.open_url(search_url)
                if success:
                    return {"status": "success", "message": f"Suche nach '{query}' durchgeführt"}
                else:
                    return {"status": "error", "message": f"Konnte Suche nach '{query}' nicht durchführen"}
        
        # Datei herunterladen
        elif "download" in command_lower or "herunterladen" in command_lower:
            url_match = re.search(r'(?:download|herunterladen)\s+(?:von\s+)?(?:https?://)?([\w.-]+\.[a-z]{2,}(?:/[^\s]*)?)', command_text, re.IGNORECASE)
            if url_match:
                domain = url_match.group(1)
                url = f"https://{domain}" if not domain.startswith("http") else domain
                
                # Sicherheitscheck
                if not internet_access.is_url_safe(url):
                    return {"status": "error", "message": f"Die URL {url} wurde als unsicher eingestuft"}
                
                # Extrahiere Dateinamen, falls angegeben
                filename_match = re.search(r'als\s+"([^"]+)"', command_text)
                filename = filename_match.group(1) if filename_match else None
                
                # Lade die Datei herunter
                downloaded_file = internet_access.download_file(url, filename)
                if downloaded_file:
                    return {"status": "success", "message": f"Datei von {url} heruntergeladen als {downloaded_file}"}
                else:
                    return {"status": "error", "message": f"Konnte Datei von {url} nicht herunterladen"}
        
        # Webseite abrufen
        elif "hole" in command_lower or "get" in command_lower:
            url_match = re.search(r'(?:hole|get)\s+(?:die\s+(?:seite|webseite|website)\s+)?(?:https?://)?([\w.-]+\.[a-z]{2,}(?:/[^\s]*)?)', command_text, re.IGNORECASE)
            if url_match:
                domain = url_match.group(1)
                url = f"https://{domain}" if not domain.startswith("http") else domain
                
                # Sicherheitscheck
                if not internet_access.is_url_safe(url):
                    return {"status": "error", "message": f"Die URL {url} wurde als unsicher eingestuft"}
                
                # Hole die Webseite
                content = internet_access.get_webpage(url)
                if content:
                    # Zeige nur die ersten 100 Zeichen des Inhalts
                    preview = content[:100] + "..." if len(content) > 100 else content
                    return {"status": "success", "message": f"Webseite {url} abgerufen", "content_preview": preview}
                else:
                    return {"status": "error", "message": f"Konnte Webseite {url} nicht abrufen"}
        
        # Screenshot erstellen
        elif "screenshot" in command_lower:
            # Prüfe, ob ein Browser geöffnet ist
            if not web_browser.is_browser_open():
                return {"status": "error", "message": "Es ist kein Browser geöffnet"}
            
            # Erstelle Screenshot
            screenshot_path = web_browser.take_screenshot()
            if screenshot_path:
                return {"status": "success", "message": f"Screenshot erstellt: {screenshot_path}"}
            else:
                return {"status": "error", "message": "Konnte keinen Screenshot erstellen"}
        
        # Fallback für unbekannte Internet-Befehle
        return {"status": "error", "message": "Unbekannter Internet-Befehl"}

# Hauptfunktion für die Demo
def run_demo():
    """
    Führt die Internet-Zugriff-Demo aus
    """
    print("\n=== MISO Internet-Zugriff Demo ===\n")
    print("Diese Demo zeigt, wie MISO auf das Internet zugreifen und Befehle ausführen kann.")
    print("\nBeispiel-Befehle:")
    print("1. Öffne die Webseite example.com")
    print("2. Suche nach MISO AI Assistent")
    print("3. Hole die Webseite python.org")
    print("4. Download von example.com/robots.txt")
    print("5. Screenshot erstellen (wenn Browser geöffnet ist)")
    print("\nGeben Sie 'exit' ein, um die Demo zu beenden.\n")
    
    # Initialisiere CommandExecutor
    executor = CommandExecutor()
    executor.start()
    
    # Interaktive Befehlsschleife
    while True:
        try:
            user_input = input("\nGeben Sie einen Befehl ein: ")
            
            if user_input.lower() in ['exit', 'quit', 'ende', 'beenden']:
                break
            
            # Führe den Befehl aus
            command = executor.add_command(user_input)
            
            # Warte kurz, damit der Befehl verarbeitet werden kann
            time.sleep(1)
            
            # Zeige das Ergebnis an
            if command.status.name == "COMPLETED":
                print(f"\nErgebnis: {command.result.get('message', 'Keine Nachricht')}")
                if 'content_preview' in command.result:
                    print(f"Inhalt: {command.result['content_preview']}")
            else:
                print(f"\nBefehl konnte nicht ausgeführt werden: {command.error}")
            
        except KeyboardInterrupt:
            print("\nDemo wird beendet...")
            break
        except Exception as e:
            print(f"\nFehler: {e}")
    
    # Beende den CommandExecutor
    executor.stop()
    
    # Schließe den Browser, falls geöffnet
    if "internet_access" in executor.modules and executor.modules["internet_access"]["web_browser"].is_browser_open():
        executor.modules["internet_access"]["web_browser"].close()
    
    print("\nDemo beendet.")

# Führe die Demo aus, wenn das Skript direkt ausgeführt wird
if __name__ == "__main__":
    run_demo()
