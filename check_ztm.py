#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate AGI - ZTM und VOID-Prüfung
"""

import os
import sys
import importlib
import logging
import glob
from datetime import datetime, timedelta

# Konfiguration
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("MISO.ZTM.Validator")

# ZTM aktivieren
os.environ["MISO_ZTM_MODE"] = "1"
os.environ["MISO_ZTM_LOG_LEVEL"] = "DEBUG"

# Basisverzeichnis
base_dir = os.path.dirname(os.path.abspath(__file__))

def check_ztm_logs():
    """Überprüft, ob aktuelle ZTM-Logs existieren"""
    log_dir = os.path.join(base_dir, "logs")
    log_files = glob.glob(os.path.join(log_dir, "ztm_session_*.log"))
    
    if not log_files:
        logger.warning("❌ Keine ZTM-Log-Dateien gefunden")
        return False
    
    # Prüfe auf aktuelle Logs (nicht älter als 24 Stunden)
    now = datetime.now()
    recent_logs = False
    
    for log_file in log_files:
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(log_file))
            if now - mtime < timedelta(hours=24):
                logger.info(f"✅ Aktuelle ZTM-Logs gefunden: {os.path.basename(log_file)}")
                recent_logs = True
        except Exception as e:
            logger.error(f"Fehler beim Prüfen der Log-Datei {log_file}: {e}")
    
    if not recent_logs:
        logger.warning("❌ Keine aktuellen ZTM-Logs (< 24h) gefunden")
        
    return recent_logs

def check_ztm_monitor():
    """Überprüft, ob ZTM-Monitor existiert und track()-Methode hat"""
    try:
        ztm_module = importlib.import_module("security.ztm_monitor")
        logger.info("✅ ZTM-Monitor-Modul gefunden")
        
        if hasattr(ztm_module, "track"):
            logger.info("✅ ZTM-Monitor hat track()-Methode")
            return True
        else:
            logger.warning("❌ ZTM-Monitor hat keine track()-Methode")
            return False
    except ImportError:
        logger.warning("❌ ZTM-Monitor-Modul nicht gefunden")
        
        # Suche nach alternativen ZTM-Modulen
        try:
            ztm_module = importlib.import_module("security.ztm_validator")
            logger.info("⚠️ Alternative: ZTM-Validator gefunden")
            
            if hasattr(ztm_module, "verify"):
                logger.info("✅ ZTM-Validator hat verify()-Methode")
                return True
            else:
                logger.warning("❌ ZTM-Validator hat keine verify()-Methode")
                return False
        except ImportError:
            logger.warning("❌ Kein alternatives ZTM-Modul gefunden")
            return False

def check_void_protocol():
    """Überprüft, ob VOID-Protokoll existiert und aktiv ist"""
    try:
        void_module = importlib.import_module("miso.security.void_protocol")
        logger.info("✅ VOID-Protokoll-Modul gefunden")
        
        has_verify = hasattr(void_module, "verify")
        has_secure = hasattr(void_module, "secure")
        
        if has_verify and has_secure:
            logger.info("✅ VOID-Protokoll hat verify() und secure()-Methoden")
            return True
        else:
            missing = []
            if not has_verify:
                missing.append("verify()")
            if not has_secure:
                missing.append("secure()")
            logger.warning(f"❌ VOID-Protokoll fehlen Methoden: {', '.join(missing)}")
            return False
    except ImportError:
        logger.warning("❌ VOID-Protokoll-Modul nicht gefunden")
        return False

# Führe Prüfungen aus
print("="*50)
print("PHASE 2: ZERO-TRUST & VOID-VERIFIKATION")
print("="*50)

ztm_logs_active = check_ztm_logs()
ztm_monitor_active = check_ztm_monitor()
void_active = check_void_protocol()

# Gesamtergebnis
print("\n" + "="*50)
if all([ztm_logs_active, ztm_monitor_active, void_active]):
    print("✅ ZTM/VOID vollständig aktiv")
else:
    print("⚠️ ZTM/VOID nicht vollständig aktiv:")
    if not ztm_logs_active:
        print("  - Keine aktuellen ZTM-Logs gefunden")
    if not ztm_monitor_active:
        print("  - ZTM-Monitor nicht aktiv oder track()-Methode nicht verfügbar")
    if not void_active:
        print("  - VOID-Protokoll nicht aktiv oder Methoden fehlen")
    print("\nEmpfohlene Schritte:")
    print("1. Erstellen/Aktivieren der fehlenden Komponenten")
    print("2. Überprüfen der ZTM-Konfiguration (MISO_ZTM_MODE=1)")
    print("3. Sicherstellen, dass ZTM-Hooks in kritischen Klassen eingebaut sind")
print("="*50)
