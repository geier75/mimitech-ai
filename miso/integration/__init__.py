#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Integration Paket

Dieses Paket implementiert die Integration zwischen verschiedenen MISO-Komponenten,
insbesondere zwischen Q-LOGIK und ECHO-PRIME.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import logging

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.integration")

# Exportiere Hauptklassen für einfachen Import
try:
    from .ql_echo_bridge import QLEchoBridge, TemporalBeliefNetwork, get_ql_echo_bridge
    from .bayesian_time_analyzer import BayesianTimeNodeAnalyzer, BayesianTimeNodeResult, get_bayesian_time_analyzer
    from .temporal_decision_process import TemporalDecisionProcess, DecisionSequence, get_temporal_decision_process
    from .paradox_resolver import ParadoxResolver, ParadoxResolutionStrategy, get_paradox_resolver
    
    __all__ = [
        'QLEchoBridge', 'TemporalBeliefNetwork', 'get_ql_echo_bridge',
        'BayesianTimeNodeAnalyzer', 'BayesianTimeNodeResult', 'get_bayesian_time_analyzer',
        'TemporalDecisionProcess', 'DecisionSequence', 'get_temporal_decision_process',
        'ParadoxResolver', 'ParadoxResolutionStrategy', 'get_paradox_resolver'
    ]
    
    logger.info("MISO-Integration-Paket erfolgreich initialisiert")
except ImportError as e:
    logger.warning(f"Einige Integrationskomponenten konnten nicht importiert werden: {e}")
    # Importiere zumindest vorhandene Komponenten
    __all__ = []
