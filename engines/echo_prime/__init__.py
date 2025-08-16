#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - ECHO-PRIME Engine

Dieses Paket implementiert die ECHO-PRIME Engine für MISO Ultimate, die für
Zeitlinienoperationen, Paradoxauflösung und temporale Analysen verwendet wird.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

from .engine import EchoPrimeEngine, get_echo_prime_engine
from .timeline import Timeline, TimeNode, TemporalEvent, Trigger
from .paradox import ParadoxDetector, ParadoxResolver, ParadoxType
from .quantum import QuantumTimeEffect, QuantumState, QuantumProbability

__all__ = [
    'EchoPrimeEngine',
    'get_echo_prime_engine',
    'Timeline',
    'TimeNode',
    'TemporalEvent',
    'Trigger',
    'ParadoxDetector',
    'ParadoxResolver',
    'ParadoxType',
    'QuantumTimeEffect',
    'QuantumState',
    'QuantumProbability'
]

__version__ = '1.0.0'
