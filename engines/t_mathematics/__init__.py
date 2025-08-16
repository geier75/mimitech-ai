#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - T-Mathematics Engine

Dieses Modul implementiert die T-Mathematics Engine für MISO Ultimate,
optimiert für Hochleistungs-Tensoralgebra und symbolische Mathematik.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

from .engine import TMathematicsEngine
from .tensor import MISOTensor

# Kommentiere nicht existierende Module aus
# from .symbolic import SymbolicMath
# from .operations import TensorOperations

__all__ = [
    'TMathematicsEngine',
    'MISOTensor'
    # 'SymbolicMath',
    # 'TensorOperations'
]
