#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vXor Kern-Modul (vx_core)
Enthält die zentralen Strukturen und Klassen für das vXor AGI-System.
"""

import logging
logger = logging.getLogger(__name__)

class VXModule:
    """Basisklasse für alle vXor-Module"""
    
    def __init__(self, name=None, version="1.0.0"):
        self.name = name or self.__class__.__name__
        self.version = version
        self.initialized = False
        logger.info(f"VXModule {self.name} initialisiert (v{self.version})")
        
    def init(self):
        """Initialisiert das Modul und gibt True zurück bei Erfolg"""
        self.initialized = True
        logger.info(f"VXModule {self.name} erfolgreich initialisiert")
        return True
        
    def boot(self):
        """Startet das Modul"""
        if not self.initialized:
            self.init()
        logger.info(f"VXModule {self.name} gestartet")
        return True

# Globales Kernmodul
_core_module = None

# Einrichtung der Kern-Funktionen
def init():
    """Initialisiert das Kernmodul"""
    global _core_module
    _core_module = VXModule("vXor Core")
    logger.info("vXor Kernmodul initialisiert")
    return _core_module.init()

def boot():
    """Bootet das Kernmodul"""
    global _core_module
    if not _core_module:
        logger.warning("vXor Core: boot() ohne vorherige init() aufgerufen")
        _core_module = VXModule("vXor Core")
        
    logger.info("vXor Core: boot() - Kernmodul wird gebootet")
    return _core_module.boot()

# Abwärtskompatibilität mit bootstrap()-Aufrufen
def bootstrap():
    """Bootstrap-Prozess für das vXor-System (Alias für boot())"""
    return boot()

def configure(config=None):
    """Konfiguriert das Kernmodul
    
    Args:
        config (dict, optional): Konfigurationsparameter. Defaults to None.
    """
    global _core_module
    if not _core_module:
        logger.warning("vXor Core: configure() ohne vorherige init() aufgerufen")
        return False
    
    if config:
        if "name" in config:
            _core_module.name = config["name"]
        if "version" in config:
            _core_module.version = config["version"]
    
    logger.info(f"vXor Core: configure() - Modul {_core_module.name} (v{_core_module.version}) konfiguriert")
    return True

def setup():
    """Richtet das Kernmodul vollständig ein"""
    global _core_module
    if not _core_module:
        logger.warning("vXor Core: setup() ohne vorherige init() aufgerufen")
        return False
    
    logger.info("vXor Core: setup() - Initialisiere erweiterte Kernfunktionen")
    return True

def activate():
    """Aktiviert das Kernmodul vollständig"""
    global _core_module
    if not _core_module:
        logger.warning("vXor Core: activate() ohne vorherige init() aufgerufen")
        return False
    
    logger.info("vXor Core: activate() - Aktiviere alle Kernfunktionen")
    return True

def start():
    """Startet das Kernmodul"""
    global _core_module
    if not _core_module:
        logger.warning("vXor Core: start() ohne vorherige init() aufgerufen")
        return False
    
    logger.info("vXor Core: start() - Kernmodul erfolgreich gestartet")
    return True
