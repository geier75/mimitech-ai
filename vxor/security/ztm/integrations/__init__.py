""
ZTM Integrations Package

This package contains integrations for the Zero-Trust Monitoring system.
"""

# Import integration modules
from .void_integration import VOIDIntegration, VOIDVerificationResult

__all__ = [
    'VOIDIntegration',
    'VOIDVerificationResult',
]
