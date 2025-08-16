#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Konfigurationsmanagement für VXOR AI Blackbox
---------------------------------------------

Zentrale Komponente zur Verwaltung von Konfigurationen aller
VXOR AI Blackbox-Sicherheitsmodule mit Unterstützung für:
- Verschlüsselte Konfigurationswerte
- Hierarchische Konfigurationsstrukturen
- Umgebungsvariablen-Integration
- Sensible Standardwerte

© 2025 VXOR AI - Alle Rechte vorbehalten
"""

from .config_manager import ConfigManager, load_config, get_config
from .secure_config import SecureConfigValue, encrypt_config_value, decrypt_config_value

# Globaler ConfigManager-Singleton, der von allen Komponenten genutzt werden kann
_config_manager = None

def get_config_manager(reset=False):
    """
    Gibt den globalen ConfigManager zurück oder erstellt ihn, falls er noch nicht existiert.
    
    Args:
        reset: Wenn True, wird ein neuer ConfigManager erstellt, auch wenn bereits einer existiert
        
    Returns:
        ConfigManager-Instanz
    """
    global _config_manager
    
    if _config_manager is None or reset:
        _config_manager = ConfigManager()
    
    return _config_manager
