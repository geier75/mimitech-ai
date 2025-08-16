#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Internet-Zugriff Demo

Dieses Skript demonstriert die Fähigkeit von MISO, auf das Internet zuzugreifen,
Webseiten zu besuchen und mit Webdiensten zu interagieren.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import json
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union, Tuple

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

# Importiere MISO-Module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Definiere SecurityLevel-Enum, da wir das Modul nicht direkt importieren können
class SecurityLevel(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()

# Importiere die Module direkt
try:
    from miso.network.internet_access import InternetAccess
    from miso.network.web_browser import WebBrowser
    from miso.control.command_executor import CommandExecutor, CommandPriority
    # DirectCommandInterface wird in diesem Demo nicht verwendet
    # from miso.control.direct_command_interface import DirectCommandInterface
except ImportError as e:
    logger.error(f"Fehler beim Importieren der MISO-Module: {e}")
    print(f"\nFehler beim Importieren der MISO-Module: {e}")
    print("Stellen Sie sicher, dass die MISO-Module korrekt installiert sind.")
    sys.exit(1)

class InternetDemo:
    """
    Demo-Klasse für den Internet-Zugriff von MISO.
    Demonstriert verschiedene Internet-Funktionen wie Webseiten abrufen,
    Websuche durchführen und Browser-Automatisierung.
    """
    
    def __init__(self):
        """Initialisiert die Demo"""
        logger.info("Initialisiere Internet-Demo")
        
        # Konfiguration für Internet-Zugriff
        self.config = {
            "security_level": SecurityLevel.MEDIUM,
            "user_agent": "MISO/1.0 Demo",
            "timeout": 30,
            "download_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloads"),
            "headless": False,  # Sichtbarer Browser für Demo-Zwecke
            "browser_type": "chrome"
        }
        
        # Erstelle Download-Verzeichnis
        os.makedirs(self.config["download_dir"], exist_ok=True)
        
        # Initialisiere Internet-Zugriff
        self.internet_access = InternetAccess(self.config)
        
        # Initialisiere Web-Browser
        self.web_browser = WebBrowser(self.config)
        
        # Initialisiere Command Executor
        self.command_executor = CommandExecutor()
        
        # Erweitere Command Executor um Internet-Befehle
        self._extend_command_executor()
        
        logger.info("Internet-Demo initialisiert")
    
    def _extend_command_executor(self):
        """Erweitert den Command Executor um Internet-Befehle"""
        # Füge Internet-Module zum Command Executor hinzu
        self.command_executor.modules['internet_access'] = {
            'internet_access': self.internet_access,
            'web_browser': self.web_browser
        }
        
        logger.info("Command Executor um Internet-Module erweitert")
    
    def _handle_browse_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Behandelt den 'browse'-Befehl zum Öffnen einer Webseite
        
        Args:
            command_data: Befehlsdaten mit URL
            
        Returns:
            Ergebnis des Befehls
        """
        url = command_data.get("url")
        if not url:
            return {"success": False, "message": "Keine URL angegeben"}
        
        # Stelle sicher, dass die URL ein Schema hat
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        
        logger.info(f"Öffne Webseite: {url}")
        
        # Initialisiere Browser, falls noch nicht geschehen
        if not self.web_browser.is_initialized:
            if not self.web_browser.initialize():
                return {"success": False, "message": "Browser konnte nicht initialisiert werden"}
        
        # Öffne URL
        success = self.web_browser.open_url(url)
        
        if success:
            # Warte kurz, damit die Seite geladen werden kann
            time.sleep(2)
            
            # Mache Screenshot
            screenshot_path = self.web_browser.take_screenshot()
            
            # Hole Seitentitel
            title = self.web_browser.execute_javascript("return document.title;")
            
            return {
                "success": True,
                "message": f"Webseite {url} erfolgreich geöffnet",
                "title": title,
                "current_url": self.web_browser.get_current_url(),
                "screenshot": screenshot_path
            }
        else:
            return {"success": False, "message": f"Fehler beim Öffnen der Webseite {url}"}
    
    def _handle_search_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Behandelt den 'search'-Befehl zum Durchführen einer Websuche
        
        Args:
            command_data: Befehlsdaten mit Suchbegriff und optionaler Suchmaschine
            
        Returns:
            Ergebnis des Befehls
        """
        query = command_data.get("query")
        if not query:
            return {"success": False, "message": "Kein Suchbegriff angegeben"}
        
        search_engine = command_data.get("engine", "google").lower()
        
        # Bestimme URL basierend auf der Suchmaschine
        if search_engine == "google":
            url = f"https://www.google.com/search?q={query}"
        elif search_engine == "bing":
            url = f"https://www.bing.com/search?q={query}"
        elif search_engine == "duckduckgo":
            url = f"https://duckduckgo.com/?q={query}"
        else:
            return {"success": False, "message": f"Nicht unterstützte Suchmaschine: {search_engine}"}
        
        logger.info(f"Führe Websuche durch: {query} (mit {search_engine})")
        
        # Initialisiere Browser, falls noch nicht geschehen
        if not self.web_browser.is_initialized:
            if not self.web_browser.initialize():
                return {"success": False, "message": "Browser konnte nicht initialisiert werden"}
        
        # Öffne URL
        success = self.web_browser.open_url(url)
        
        if success:
            # Warte kurz, damit die Seite geladen werden kann
            time.sleep(2)
            
            # Mache Screenshot
            screenshot_path = self.web_browser.take_screenshot()
            
            return {
                "success": True,
                "message": f"Websuche nach '{query}' erfolgreich durchgeführt",
                "search_engine": search_engine,
                "query": query,
                "current_url": self.web_browser.get_current_url(),
                "screenshot": screenshot_path
            }
        else:
            return {"success": False, "message": f"Fehler bei der Websuche nach '{query}'"}
    
    def _handle_download_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Behandelt den 'download'-Befehl zum Herunterladen einer Datei
        
        Args:
            command_data: Befehlsdaten mit URL und optionalem Dateinamen
            
        Returns:
            Ergebnis des Befehls
        """
        url = command_data.get("url")
        if not url:
            return {"success": False, "message": "Keine URL angegeben"}
        
        filename = command_data.get("filename")
        
        logger.info(f"Lade Datei herunter: {url}")
        
        # Lade Datei herunter
        download_path = self.internet_access.download_file(url, filename)
        
        if download_path:
            return {
                "success": True,
                "message": f"Datei erfolgreich heruntergeladen",
                "url": url,
                "file_path": download_path,
                "file_size": os.path.getsize(download_path)
            }
        else:
            return {"success": False, "message": f"Fehler beim Herunterladen der Datei von {url}"}
    
    def _handle_fill_form_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Behandelt den 'fill_form'-Befehl zum Ausfüllen eines Formulars
        
        Args:
            command_data: Befehlsdaten mit Formularfeldern und optionaler URL
            
        Returns:
            Ergebnis des Befehls
        """
        url = command_data.get("url")
        form_data = command_data.get("form_data")
        
        if not form_data or not isinstance(form_data, dict):
            return {"success": False, "message": "Keine gültigen Formulardaten angegeben"}
        
        # Öffne URL, falls angegeben
        if url:
            # Stelle sicher, dass die URL ein Schema hat
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            
            logger.info(f"Öffne Webseite für Formular: {url}")
            
            # Initialisiere Browser, falls noch nicht geschehen
            if not self.web_browser.is_initialized:
                if not self.web_browser.initialize():
                    return {"success": False, "message": "Browser konnte nicht initialisiert werden"}
            
            # Öffne URL
            if not self.web_browser.open_url(url):
                return {"success": False, "message": f"Fehler beim Öffnen der Webseite {url}"}
            
            # Warte kurz, damit die Seite geladen werden kann
            time.sleep(2)
        
        if not self.web_browser.is_initialized:
            return {"success": False, "message": "Browser ist nicht initialisiert"}
        
        logger.info(f"Fülle Formular aus mit Daten: {form_data}")
        
        # Fülle Formularfelder aus
        for selector, value in form_data.items():
            # Versuche verschiedene Selektoren-Typen
            selectors_to_try = [
                (selector, By.CSS_SELECTOR),
                (selector, By.ID),
                (f"input[name='{selector}']", By.CSS_SELECTOR),
                (f"//input[@name='{selector}']", By.XPATH)
            ]
            
            filled = False
            for sel, by in selectors_to_try:
                if self.web_browser.fill_input(sel, value, by):
                    filled = True
                    break
            
            if not filled:
                logger.warning(f"Konnte Feld '{selector}' nicht finden oder ausfüllen")
        
        # Mache Screenshot
        screenshot_path = self.web_browser.take_screenshot()
        
        # Versuche, das Formular abzusenden
        submit_success = self.web_browser.submit_form()
        
        if submit_success:
            # Warte kurz, damit die Seite nach dem Absenden geladen werden kann
            time.sleep(2)
            
            # Mache einen weiteren Screenshot nach dem Absenden
            after_submit_screenshot = self.web_browser.take_screenshot("after_submit.png")
            
            return {
                "success": True,
                "message": "Formular erfolgreich ausgefüllt und abgesendet",
                "current_url": self.web_browser.get_current_url(),
                "before_submit_screenshot": screenshot_path,
                "after_submit_screenshot": after_submit_screenshot
            }
        else:
            return {
                "success": False,
                "message": "Formular ausgefüllt, konnte aber nicht abgesendet werden",
                "current_url": self.web_browser.get_current_url(),
                "screenshot": screenshot_path
            }
    
    def run_demo(self):
        """Führt die Internet-Demo aus"""
        logger.info("Starte Internet-Demo")
        
        print("\n" + "="*50)
        print("MISO Internet-Zugriff Demo")
        print("="*50 + "\n")
        
        try:
            # Demo 1: Webseite abrufen
            print("\n--- Demo 1: Webseite abrufen ---\n")
            
            url = "https://example.com"
            print(f"Rufe Webseite ab: {url}")
            
            content = self.internet_access.get_webpage(url)
            if content:
                print(f"Webseite erfolgreich abgerufen. Inhaltslänge: {len(content)} Zeichen")
                # Zeige einen Ausschnitt des Inhalts
                print(f"Ausschnitt: {content[:150]}...\n")
            else:
                print(f"Fehler beim Abrufen der Webseite {url}\n")
            
            # Demo 2: Browser-Automatisierung
            print("\n--- Demo 2: Browser-Automatisierung ---\n")
            
            print("Initialisiere Browser...")
            if not self.web_browser.is_initialized:
                if not self.web_browser.initialize():
                    print("Browser konnte nicht initialisiert werden")
                else:
                    print("Browser erfolgreich initialisiert")
            
            # Öffne eine Webseite
            url = "https://www.wikipedia.org"
            print(f"Öffne Webseite: {url}")
            
            if self.web_browser.open_url(url):
                print(f"Webseite erfolgreich geöffnet: {url}")
                
                # Warte kurz, damit die Seite geladen werden kann
                time.sleep(2)
                
                # Mache Screenshot
                screenshot_path = self.web_browser.take_screenshot()
                if screenshot_path:
                    print(f"Screenshot erstellt: {screenshot_path}")
                
                # Suche nach dem Suchfeld und gib einen Suchbegriff ein
                search_input = self.web_browser.find_element("input[name='search']")
                if search_input:
                    print("Suchfeld gefunden")
                    if self.web_browser.fill_input("input[name='search']", "Python programming"):
                        print("Suchbegriff 'Python programming' eingegeben")
                    
                    # Mache Screenshot nach Eingabe
                    self.web_browser.take_screenshot("after_input.png")
                else:
                    print("Suchfeld nicht gefunden")
            else:
                print(f"Fehler beim Öffnen der Webseite {url}")
            
            # Demo 3: Direkte Befehlsausführung
            print("\n--- Demo 3: Direkte Befehlsausführung ---\n")
            
            # Befehl zum Öffnen einer Webseite
            print("Führe Befehl aus: browse https://www.python.org")
            result = self.direct_command.execute_command("browse https://www.python.org")
            
            if result.get("success"):
                print(f"Befehl erfolgreich ausgeführt: {result.get('message')}")
                if "screenshot" in result:
                    print(f"Screenshot: {result['screenshot']}")
            else:
                print(f"Fehler bei der Befehlsausführung: {result.get('message')}")
            
            # Befehl zum Durchführen einer Websuche
            print("\nFühre Befehl aus: search 'MISO AI system'")
            result = self.direct_command.execute_command("search 'MISO AI system'")
            
            if result.get("success"):
                print(f"Befehl erfolgreich ausgeführt: {result.get('message')}")
                if "screenshot" in result:
                    print(f"Screenshot: {result['screenshot']}")
            else:
                print(f"Fehler bei der Befehlsausführung: {result.get('message')}")
            
            # Befehl zum Herunterladen einer Datei
            print("\nFühre Befehl aus: download https://www.python.org/static/img/python-logo.png python-logo.png")
            result = self.direct_command.execute_command("download https://www.python.org/static/img/python-logo.png python-logo.png")
            
            if result.get("success"):
                print(f"Befehl erfolgreich ausgeführt: {result.get('message')}")
                print(f"Heruntergeladene Datei: {result.get('file_path')}")
            else:
                print(f"Fehler bei der Befehlsausführung: {result.get('message')}")
            
            print("\n" + "="*50)
            print("Internet-Demo abgeschlossen")
            print("="*50 + "\n")
            
        except Exception as e:
            logger.error(f"Fehler in der Internet-Demo: {e}")
            print(f"\nFehler in der Internet-Demo: {e}\n")
        finally:
            # Schließe Browser
            if self.web_browser.is_initialized:
                print("Schließe Browser...")
                self.web_browser.close()
                print("Browser geschlossen")

if __name__ == "__main__":
    # Importiere Selenium-Komponenten, falls verfügbar
    try:
        from selenium.webdriver.common.by import By
        selenium_available = True
    except ImportError:
        selenium_available = False
        logger.warning("Selenium ist nicht verfügbar. Browser-Automatisierung wird eingeschränkt sein.")
        print("\nWARNUNG: Selenium ist nicht installiert. Einige Funktionen sind möglicherweise nicht verfügbar.")
        print("Installieren Sie Selenium mit: pip install selenium\n")
        # Definiere By-Klasse für den Fall, dass Selenium nicht verfügbar ist
        class By:
            CSS_SELECTOR = "css selector"
            XPATH = "xpath"
            ID = "id"
            TAG_NAME = "tag name"
    
    # Führe Demo aus
    demo = InternetDemo()
    
    # Starte den Command Executor
    demo.command_executor.start()
    
    print("\n=== MISO Internet-Zugriff Demo ===")
    print("Diese Demo zeigt, wie MISO auf das Internet zugreifen und Befehle ausführen kann.")
    print("\nBeispiel-Befehle:")
    print("1. Öffne die Webseite example.com")
    print("2. Suche nach MISO AI Assistent")
    print("3. Hole die Webseite python.org")
    print("4. Download von example.com/robots.txt")
    print("5. Screenshot erstellen (wenn Browser geöffnet ist)")
    print("\nGeben Sie 'exit' ein, um die Demo zu beenden.\n")
    
    # Interaktive Befehlsschleife
    while True:
        try:
            user_input = input("\nGeben Sie einen Befehl ein: ")
            
            if user_input.lower() in ['exit', 'quit', 'ende', 'beenden']:
                break
            
            # Führe den Befehl aus
            command = demo.command_executor.add_command(user_input)
            
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
    
    # Beende den Command Executor
    demo.command_executor.stop()
    
    # Schließe den Browser, falls geöffnet
    if demo.web_browser.is_initialized:
        demo.web_browser.close()
    
    print("\nDemo beendet.")
