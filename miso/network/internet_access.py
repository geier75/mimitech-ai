#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Internet Access Module

Dieses Modul ermöglicht MISO den Zugriff auf das Internet.
Es bietet Funktionen zum Abrufen von Webseiten, Durchführen von Websuchen,
Interaktion mit APIs und Herunterladen von Dateien.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import json
import re
import urllib.request
import urllib.parse
import urllib.error
import http.client
import ssl
import socket
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Versuche, optionale Abhängigkeiten zu importieren
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

# Konfiguriere Logging
logger = logging.getLogger("MISO.network.internet_access")

class SecurityLevel(object):
    """Sicherheitsstufen für den Internetzugriff"""
    LOW = "low"       # Wenige Einschränkungen
    MEDIUM = "medium" # Ausgewogene Sicherheit
    HIGH = "high"     # Strenge Einschränkungen

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
        Öffentliche Methode zur Überprüfung, ob eine URL sicher ist
        
        Args:
            url: Die zu überprüfende URL
            
        Returns:
            True, wenn die URL sicher ist, sonst False
        """
        return self._check_url_security(url)
        
    def _check_url_security(self, url: str) -> bool:
        """
        Überprüft, ob eine URL den Sicherheitsrichtlinien entspricht
        
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
        
        # Überprüfe auf bekannte bösartige Domains (in einer realen Implementierung würde hier eine umfangreiche Liste stehen)
        blacklisted_domains = ["malware.com", "phishing.net", "evil.org"]
        if any(domain in parsed_url.netloc for domain in blacklisted_domains):
            logger.warning(f"Blockierte Domain: {parsed_url.netloc}")
            return False
        
        # Überprüfe auf verdächtige Pfade oder Parameter
        suspicious_patterns = ["/admin", "/login", "/password", "script=", "eval=", "exec="]
        if any(pattern in url.lower() for pattern in suspicious_patterns):
            if self.config["security_level"] == SecurityLevel.HIGH:
                logger.warning(f"Verdächtiges Muster in URL: {url}")
                return False
            else:
                logger.warning(f"Verdächtiges Muster in URL, aber erlaubt aufgrund der Sicherheitsstufe: {url}")
        
        return True
    
    def get_webpage(self, url: str, params: Dict[str, str] = None) -> Optional[str]:
        """
        Ruft den Inhalt einer Webseite ab
        
        Args:
            url: Die URL der abzurufenden Webseite
            params: Optionale Query-Parameter
            
        Returns:
            Der Inhalt der Webseite als String oder None bei Fehler
        """
        # Überprüfe URL-Sicherheit
        if not self._check_url_security(url):
            logger.error(f"Sicherheitsüberprüfung fehlgeschlagen für URL: {url}")
            return None
        
        logger.info(f"Rufe Webseite ab: {url}")
        
        # Verwende requests, falls verfügbar
        if REQUESTS_AVAILABLE and self.session:
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.config["timeout"],
                    verify=True  # SSL-Zertifikate überprüfen
                )
                response.raise_for_status()  # Fehler bei HTTP-Statuscodes >= 400
                return response.text
            except requests.exceptions.RequestException as e:
                logger.error(f"Fehler beim Abrufen der Webseite {url}: {e}")
                return None
        
        # Fallback auf urllib
        else:
            full_url = url
            if params:
                query_string = urllib.parse.urlencode(params)
                full_url = f"{url}?{query_string}" if "?" not in url else f"{url}&{query_string}"
            
            try:
                headers = {"User-Agent": self.config["user_agent"]}
                req = urllib.request.Request(full_url, headers=headers)
                with urllib.request.urlopen(req, timeout=self.config["timeout"]) as response:
                    return response.read().decode("utf-8")
            except (urllib.error.URLError, urllib.error.HTTPError, socket.timeout) as e:
                logger.error(f"Fehler beim Abrufen der Webseite {full_url}: {e}")
                return None
    
    def post_data(self, url: str, data: Dict[str, Any], json_data: bool = False) -> Optional[str]:
        """
        Sendet Daten an eine URL mittels POST-Anfrage
        
        Args:
            url: Die Ziel-URL
            data: Die zu sendenden Daten
            json_data: True, wenn die Daten als JSON gesendet werden sollen
            
        Returns:
            Die Antwort als String oder None bei Fehler
        """
        # Überprüfe URL-Sicherheit
        if not self._check_url_security(url):
            logger.error(f"Sicherheitsüberprüfung fehlgeschlagen für URL: {url}")
            return None
        
        # Überprüfe Sicherheitsstufe für POST-Anfragen
        if self.config["security_level"] == SecurityLevel.HIGH:
            logger.error("POST-Anfragen sind bei hoher Sicherheitsstufe nicht erlaubt")
            return None
        
        logger.info(f"Sende POST-Anfrage an: {url}")
        
        # Verwende requests, falls verfügbar
        if REQUESTS_AVAILABLE and self.session:
            try:
                if json_data:
                    response = self.session.post(
                        url,
                        json=data,
                        timeout=self.config["timeout"],
                        verify=True
                    )
                else:
                    response = self.session.post(
                        url,
                        data=data,
                        timeout=self.config["timeout"],
                        verify=True
                    )
                response.raise_for_status()
                return response.text
            except requests.exceptions.RequestException as e:
                logger.error(f"Fehler bei POST-Anfrage an {url}: {e}")
                return None
        
        # Fallback auf urllib
        else:
            try:
                headers = {"User-Agent": self.config["user_agent"]}
                
                if json_data:
                    post_data = json.dumps(data).encode("utf-8")
                    headers["Content-Type"] = "application/json"
                else:
                    post_data = urllib.parse.urlencode(data).encode("utf-8")
                    headers["Content-Type"] = "application/x-www-form-urlencoded"
                
                req = urllib.request.Request(url, data=post_data, headers=headers)
                with urllib.request.urlopen(req, timeout=self.config["timeout"]) as response:
                    return response.read().decode("utf-8")
            except (urllib.error.URLError, urllib.error.HTTPError, socket.timeout) as e:
                logger.error(f"Fehler bei POST-Anfrage an {url}: {e}")
                return None
    
    def download_file(self, url: str, filename: Optional[str] = None) -> Optional[str]:
        """
        Lädt eine Datei von einer URL herunter
        
        Args:
            url: Die URL der herunterzuladenden Datei
            filename: Optionaler Dateiname, falls nicht angegeben, wird der Dateiname aus der URL extrahiert
            
        Returns:
            Der Pfad zur heruntergeladenen Datei oder None bei Fehler
        """
        # Überprüfe URL-Sicherheit
        if not self._check_url_security(url):
            logger.error(f"Sicherheitsüberprüfung fehlgeschlagen für URL: {url}")
            return None
        
        # Extrahiere Dateinamen aus URL, falls nicht angegeben
        if not filename:
            parsed_url = urllib.parse.urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename:
                filename = "downloaded_file"
        
        # Erstelle vollständigen Pfad
        download_path = os.path.join(self.config["download_dir"], filename)
        
        logger.info(f"Lade Datei herunter von {url} nach {download_path}")
        
        # Verwende requests, falls verfügbar
        if REQUESTS_AVAILABLE and self.session:
            try:
                response = self.session.get(
                    url,
                    stream=True,
                    timeout=self.config["timeout"],
                    verify=True
                )
                response.raise_for_status()
                
                with open(download_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Datei erfolgreich heruntergeladen: {download_path}")
                return download_path
            except requests.exceptions.RequestException as e:
                logger.error(f"Fehler beim Herunterladen der Datei von {url}: {e}")
                return None
            except IOError as e:
                logger.error(f"Fehler beim Schreiben der Datei {download_path}: {e}")
                return None
        
        # Fallback auf urllib
        else:
            try:
                headers = {"User-Agent": self.config["user_agent"]}
                req = urllib.request.Request(url, headers=headers)
                
                with urllib.request.urlopen(req, timeout=self.config["timeout"]) as response:
                    with open(download_path, "wb") as f:
                        f.write(response.read())
                
                logger.info(f"Datei erfolgreich heruntergeladen: {download_path}")
                return download_path
            except (urllib.error.URLError, urllib.error.HTTPError, socket.timeout) as e:
                logger.error(f"Fehler beim Herunterladen der Datei von {url}: {e}")
                return None
            except IOError as e:
                logger.error(f"Fehler beim Schreiben der Datei {download_path}: {e}")
                return None
