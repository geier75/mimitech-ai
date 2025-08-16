"""
VX-GESTALT - Emergenz- und Integrationsschicht für das vXor-System

Dieses Paket enthält die Kernkomponenten für die Zustandssynthese und
Konfliktlösung zwischen den VXOR-Agenten.
"""

from .gestalt_integrator import (
    GestaltIntegrator,
    get_gestalt_integrator,
    AgentState,
    EmergentState,
    Conflict
)

__all__ = [
    'GestaltIntegrator',
    'get_gestalt_integrator',
    'AgentState',
    'EmergentState',
    'Conflict'
]
