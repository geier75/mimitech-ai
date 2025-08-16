#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Network Module

Dieses Modul enthält Komponenten für den Netzwerkzugriff und die Internetkommunikation.
Es ermöglicht MISO, auf das Internet zuzugreifen, Webseiten zu besuchen und mit Webdiensten zu interagieren.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

from miso.network.internet_access import InternetAccess, SecurityLevel
from miso.network.web_browser import WebBrowser

__all__ = ['InternetAccess', 'SecurityLevel', 'WebBrowser']
