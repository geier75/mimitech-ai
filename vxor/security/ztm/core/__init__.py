""
ZTM Core Package

This package contains the core functionality for the Zero-Trust Monitoring system.
"""

# Import core modules
from .ztm_core import ZeroTrustMonitor, SecurityEvent, Severity

__all__ = [
    'ZeroTrustMonitor',
    'SecurityEvent',
    'Severity',
]
