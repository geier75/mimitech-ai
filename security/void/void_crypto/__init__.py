#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - VOID-Kryptographie Paket

Dieses Paket enthält die VOID-Kryptographie-Komponenten.
"""

import os
import sys

# Importiere die Kryptographie-Komponenten aus der Elterndatei
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from security.void.void_crypto import VoidCrypto
except ImportError:
    try:
        # Versuche einen direkten Import aus der parallelen Datei
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "void_crypto", 
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "void_crypto.py")
        )
        void_crypto = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(void_crypto)
        
        # Exportiere die relevante Klasse
        VoidCrypto = void_crypto.VoidCrypto
    except Exception as e:
        import logging
        logging.getLogger("MISO.Security.VOID.Crypto").error(f"Fehler beim Importieren der VOID-Kryptographie: {e}")
        # Stelle eine Stub-Klasse bereit, damit die Imports nicht fehlschlagen
        class VoidCrypto: pass

# Exportiere alle relevanten Klassen und Funktionen
__all__ = ['VoidCrypto']
