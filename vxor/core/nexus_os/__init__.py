#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - NEXUS-OS

NEXUS-OS ist die zentrale Integrationsschicht zwischen verschiedenen MISO-Komponenten,
insbesondere zwischen der T-Mathematics Engine und dem M-LINGUA Interface.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

from .lingua_math_bridge import LinguaMathBridge
from .tensor_language_processor import TensorLanguageProcessor
from .nexus_core import NexusCore, get_nexus_core
from .task_manager import TaskManager
from .resource_manager import ResourceManager
from .nexus_os import NexusOS

__all__ = [
    'LinguaMathBridge',
    'TensorLanguageProcessor',
    'NexusCore',
    'TaskManager',
    'ResourceManager',
    'NexusOS',
    'get_nexus_core',
    'init',
    'boot',
    'configure',
    'setup',
    'activate',
    'start'
]

# Globale NexusOS-Instanz
_nexus_os = None
import logging
logger = logging.getLogger("MISO.nexus_os")

def init():
    """Initialisiert das NEXUS-OS"""
    global _nexus_os
    _nexus_os = NexusOS()
    logger.info("NEXUS ResourceMonitor initialisiert.")
    return True

def boot():
    """Bootet das NEXUS-OS"""
    global _nexus_os
    if not _nexus_os:
        logger.warning("NEXUS-OS: boot() ohne vorherige init() aufgerufen")
        _nexus_os = NexusOS()
    
    logger.info("NEXUS-OS: boot() - Starte grundlegende Systemkomponenten")
    _nexus_os.initialize()
    return True

def configure(config=None):
    """Konfiguriert das NEXUS-OS
    
    Args:
        config (dict, optional): Konfigurationsparameter. Defaults to None.
    """
    global _nexus_os
    if not _nexus_os:
        logger.warning("NEXUS-OS: configure() ohne vorherige init() aufgerufen")
        return False
    
    logger.info("NEXUS-OS: configure() - Systemkonfiguration wird angewendet")
    _nexus_os.config.update(config or {})
    return True

def setup():
    """Richtet das NEXUS-OS vollständig ein"""
    global _nexus_os
    if not _nexus_os:
        logger.warning("NEXUS-OS: setup() ohne vorherige init() aufgerufen")
        return False
    
    logger.info("NEXUS-OS: setup() - Initialisiere erweiterte Systemkomponenten")
    _nexus_os.initialize()
    return True

def activate():
    """Aktiviert das NEXUS-OS vollständig"""
    global _nexus_os
    if not _nexus_os:
        logger.warning("NEXUS-OS: activate() ohne vorherige init() aufgerufen")
        return False
    
    logger.info("NEXUS-OS: activate() - Aktiviere alle Subsysteme")
    # Bei NEXUS-OS wird diese Funktion intern in start() abgebildet
    if not _nexus_os.running:
        _nexus_os.start()
    return True

def start():
    """Startet das NEXUS-OS"""
    global _nexus_os
    if not _nexus_os:
        logger.warning("NEXUS-OS: start() ohne vorherige init() aufgerufen")
        return False
    
    logger.info("NEXUS-OS: start() - System wird gestartet")
    _nexus_os.start()
    return True
