#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - VOID-Kontext Paket

Dieses Paket enthält die VOID-Kontext-Komponenten.
"""

import os
import sys

# Importiere die Kontext-Komponenten aus der Elterndatei
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from security.void.void_context import VoidContext
except ImportError:
    try:
        # Versuche einen direkten Import aus der parallelen Datei
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "void_context", 
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "void_context.py")
        )
        void_context = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(void_context)
        
        # Exportiere die relevante Klasse
        VoidContext = void_context.VoidContext
    except Exception as e:
        import logging
        logging.getLogger("MISO.Security.VOID.Context").error(f"Fehler beim Importieren des VOID-Kontexts: {e}")
        # Stelle eine Stub-Klasse bereit, damit die Imports nicht fehlschlagen
        class VoidContext: 
            def __init__(self, security_level="high"):
                self.security_level = security_level
                self.initialized = False
            
            def init(self):
                self.initialized = True
                return True

# Exportiere alle relevanten Klassen und Funktionen
__all__ = ['VoidContext']
