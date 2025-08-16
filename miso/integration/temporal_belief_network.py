#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Temporales Glaubensnetzwerk-Modul

Dieses Modul re-exportiert die TemporalBeliefNetwork-Klasse aus ql_echo_bridge.py,
um Kompatibilität mit Tests und anderen Modulen zu gewährleisten, die diese
Klasse an diesem Speicherort erwarten.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import logging
# Re-exportiere die TemporalBeliefNetwork und TimeNodeData aus ql_echo_bridge
from miso.integration.ql_echo_bridge import TemporalBeliefNetwork, TimeNodeData

# Setze den Logger für dieses Modul
logger = logging.getLogger(__name__)
logger.info("Temporales Glaubensnetzwerk-Modul erfolgreich geladen")

# Re-exportiere die Klassen, damit sie direkt aus diesem Modul importiert werden können
__all__ = ['TemporalBeliefNetwork', 'TimeNodeData']
