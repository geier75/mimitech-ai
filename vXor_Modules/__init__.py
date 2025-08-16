#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VXOR Modules - Paketinitialisierung

Dieses Paket enthält die VXOR-Module, die von Manus AI implementiert wurden.
Es dient als Bridge zwischen MISO und den VXOR-Modulen.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

# Importiere die VXOR-Module
from . import vxor_integration
from .vxor_integration import VXORAdapter, check_module_availability, get_compatible_vxor_modules

# Versuche, die implementierten Module zu importieren
try:
    from . import vx_reflex
except ImportError:
    pass  # Modul nicht verfügbar
