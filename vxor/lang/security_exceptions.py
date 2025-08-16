#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Sicherheitsausnahmen

Diese Datei definiert Ausnahmen für Sicherheitsverletzungen in MISO Ultimate.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

class SecurityViolationError(Exception):
    """
    Ausnahme für Sicherheitsverletzungen
    
    Wird ausgelöst, wenn eine Sicherheitsrichtlinie verletzt wird.
    """
    
    def __init__(self, message, module=None, severity="high"):
        """
        Initialisiert die Ausnahme
        
        Args:
            message: Fehlermeldung
            module: Name des Moduls, das die Verletzung verursacht hat
            severity: Schweregrad der Verletzung (low, medium, high, critical)
        """
        self.module = module
        self.severity = severity
        super().__init__(f"Sicherheitsverletzung ({severity}): {message}" + 
                        (f" in Modul {module}" if module else ""))

class AccessDeniedError(SecurityViolationError):
    """
    Ausnahme für verweigerten Zugriff
    
    Wird ausgelöst, wenn ein Zugriff auf eine Ressource verweigert wird.
    """
    
    def __init__(self, resource, module=None, severity="high"):
        """
        Initialisiert die Ausnahme
        
        Args:
            resource: Name der Ressource, auf die der Zugriff verweigert wurde
            module: Name des Moduls, das den Zugriff versucht hat
            severity: Schweregrad der Verletzung (low, medium, high, critical)
        """
        super().__init__(f"Zugriff auf {resource} verweigert", module, severity)

class InvalidTokenError(SecurityViolationError):
    """
    Ausnahme für ungültige Sicherheitstoken
    
    Wird ausgelöst, wenn ein ungültiges Sicherheitstoken verwendet wird.
    """
    
    def __init__(self, module=None, severity="high"):
        """
        Initialisiert die Ausnahme
        
        Args:
            module: Name des Moduls, das das ungültige Token verwendet hat
            severity: Schweregrad der Verletzung (low, medium, high, critical)
        """
        super().__init__("Ungültiges Sicherheitstoken", module, severity)

class ResourceLimitExceededError(SecurityViolationError):
    """
    Ausnahme für überschrittene Ressourcenlimits
    
    Wird ausgelöst, wenn ein Ressourcenlimit überschritten wird.
    """
    
    def __init__(self, resource, limit, actual, module=None, severity="medium"):
        """
        Initialisiert die Ausnahme
        
        Args:
            resource: Name der Ressource, deren Limit überschritten wurde
            limit: Limit der Ressource
            actual: Tatsächlicher Wert
            module: Name des Moduls, das das Limit überschritten hat
            severity: Schweregrad der Verletzung (low, medium, high, critical)
        """
        super().__init__(
            f"Ressourcenlimit für {resource} überschritten: {actual} > {limit}", 
            module, 
            severity
        )
