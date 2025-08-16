#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VXOR AI Blackbox-Sicherheitsmodul
---------------------------------

Dieses Modul implementiert umfassende Sicherheitsmaßnahmen für die VXOR AI-Plattform,
einschließlich postquantenfester Kryptographie, Codeverschlüsselung, Obfuskation und
sicheren Bootloader-Mechanismen.

Enthält Komponenten für:
1. Quantum-Resistant Cryptography (QRC)
2. Secure Key Management System (SKMS)
3. Secure Build & Distribution Pipeline (SBDP)
4. Runtime Enclave & Anti-Debug (READ)
5. Secure Update System (SUS)

© 2025 VXOR AI - Alle Rechte vorbehalten
"""

__version__ = '0.1.0'
__author__ = 'VXOR AI Security Team'

# Version des Security-Pakets in einer kompatiblen Struktur
VERSION = (0, 1, 0)

# Exportiere zentrale Komponenten für einfachen Import
from . import crypto
from . import key_management
