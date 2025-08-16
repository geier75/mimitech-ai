#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sichere Logging-Infrastruktur für VXOR AI Blackbox
--------------------------------------------------

Bietet eine umfassende Logging-Lösung mit folgenden Funktionen:
- Verschlüsselte Log-Dateien
- Log-Rotation mit Größenbegrenzung
- Audit-Trails für sicherheitsrelevante Ereignisse
- Datenschutzfilter für sensible Informationen
- Integration mit dem ConfigManager

© 2025 VXOR AI - Alle Rechte vorbehalten
"""

from .secure_logger import SecureLogger, setup_logger, get_logger
from .audit_logger import AuditLogger, audit_event, get_audit_logger
from .log_filter import SensitiveDataFilter, add_sensitive_pattern, remove_sensitive_pattern

# Exportierte Konstanten
LOG_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50
}

# Exportierte Functions für einfachen Zugriff auf das Logging-System
setup_global_logging = setup_logger
get_component_logger = get_logger
log_audit_event = audit_event
