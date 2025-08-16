#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - PRISM-Engine VXOR Integration

Dieses Modul implementiert die Integration zwischen der PRISM-Engine und VX-REASON.
Es ermöglicht logisches Schließen, Kausalitätsanalyse und die Auflösung komplexer
paradoxer Szenarien in Zeitlinien und Simulationen.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple

# Importiere PRISM-Engine Komponenten
from miso.simulation.prism_engine import PrismEngine
from miso.simulation.prism_base import Timeline, TimeNode
from miso.timeline.advanced_paradox_resolution import EnhancedParadoxDetector, ParadoxResolver
from miso.simulation.paradox_resolution import get_paradox_resolution_manager, ParadoxResolutionManager
from miso.math.t_mathematics.integration_manager import TMathIntegrationManager

# Importiere VXOR-Adapter-Core
from miso.vxor.vx_adapter_core import get_module, get_module_status

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.simulation.vxor_integration")

class PRISMVXORIntegration:
    """
    Klasse zur Integration der PRISM-Engine mit VX-REASON
    
    Diese Klasse stellt die Verbindung zwischen der PRISM-Engine und
    dem VX-REASON Modul her, um logisches Schließen und Kausalitätsanalyse
    in Simulationen zu ermöglichen.
    """
    
    _instance = None  # Singleton-Pattern
    
    def __new__(cls, *args, **kwargs):
        """Implementiert das Singleton-Pattern"""
        if cls._instance is None:
            cls._instance = super(PRISMVXORIntegration, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialisiert die PRISM-VXOR-Integration
        
        Args:
            config_path: Pfad zur Konfigurationsdatei (optional)
        """
        # Initialisiere nur einmal (Singleton-Pattern)
        if hasattr(self, 'initialized') and self.initialized:
            return
            
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "config", "vxor_integration_config.json"
        )
        
        # Initialisiere PRISM-Engine-Komponenten
        self.prism_engine = PrismEngine()
        
        # Importiere ECHO-PRIME Controller für ParadoxDetector
        try:
            from miso.timeline.echo_prime_controller import EchoPrimeController
            echo_prime_controller = EchoPrimeController()
            self.paradox_detector = EnhancedParadoxDetector(echo_prime_controller)
        except Exception as e:
            logger.warning(f"Konnte ECHO-PRIME Controller nicht laden: {e}")
            # Fallback: Verwende None als echo_prime_controller
            self.paradox_detector = EnhancedParadoxDetector(None)
        
        # Verwende den gleichen echo_prime_controller für ParadoxResolver
        try:
            self.paradox_resolver = ParadoxResolver(echo_prime_controller)
        except Exception as e:
            logger.warning(f"Konnte ParadoxResolver nicht mit ECHO-PRIME Controller laden: {e}")
            # Fallback: Verwende None als echo_prime_controller
            self.paradox_resolver = ParadoxResolver(None)
        
        # Holen der TMathIntegrationManager Instanz
        self.t_math_manager = TMathIntegrationManager()
        
        # Lade oder erstelle Konfiguration
        self.vxor_config = {}
        self.load_config()
        
        # Dynamischer Import des VX-REASON-Moduls
        try:
            self.vx_reason = get_module("VX-REASON")
            self.reason_available = True
            logger.info("VX-REASON erfolgreich initialisiert")
        except Exception as e:
            self.vx_reason = None
            self.reason_available = False
            logger.warning(f"VX-REASON nicht verfügbar: {e}")
        
        self.initialized = True
        logger.info("PRISMVXORIntegration initialisiert")
    
    def load_config(self):
        """Lädt die Konfiguration aus der Konfigurationsdatei"""
        try:
            # Stelle sicher, dass das Verzeichnis existiert
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Wenn die Datei nicht existiert, erstelle Standardkonfiguration
            if not os.path.exists(self.config_path):
                self._create_default_config()
            
            # Lade die Konfiguration
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.vxor_config = json.load(f)
            
            logger.info(f"Konfiguration geladen: {len(self.vxor_config)} Einträge")
        except Exception as e:
            logger.error(f"Fehler beim Laden der Konfiguration: {e}")
            # Erstelle Standardkonfiguration im Fehlerfall
            self._create_default_config()
    
    def _create_default_config(self):
        """Erstellt eine Standardkonfiguration"""
        default_config = {
            "vx_reason": {
                "enabled": True,
                "reasoning_depth": 4,  # 1-5
                "use_quantum_logic": True,
                "paradox_resolution_mode": "adaptive",  # simple, complex, adaptive
                "causal_analysis_level": 3,  # 1-5
                "debug_mode": False
            }
        }
        
        try:
            # Stelle sicher, dass das Verzeichnis existiert
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Speichere die Standardkonfiguration
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2)
            
            self.vxor_config = default_config
            logger.info("Standardkonfiguration erstellt")
        except Exception as e:
            logger.error(f"Fehler beim Erstellen der Standardkonfiguration: {e}")
            self.vxor_config = default_config
    
    def analyze_timeline_causality(self, timeline: Timeline) -> Dict[str, Any]:
        """
        Analysiert die Kausalität einer Zeitlinie mit VX-REASON
        
        Args:
            timeline: Die zu analysierende Zeitlinie
            
        Returns:
            Ergebnisse der Kausalitätsanalyse
        """
        # Wenn VX-REASON nicht verfügbar ist, verwende die Standard-PRISM-Analyse
        if not self.reason_available:
            logger.warning("VX-REASON nicht verfügbar, verwende Standard-PRISM-Analyse")
            return self.prism_engine.analyze_timeline_integrity(timeline.id)
        
        try:
            # Bereite Daten für VX-REASON vor
            nodes_data = []
            for node_id, node in timeline.nodes.items():
                nodes_data.append({
                    "id": node.id,
                    "timestamp": node.timestamp,
                    "data": node.data,
                    "metadata": node.metadata,
                    "dependencies": [dep.id for dep in node.dependencies]
                })
            
            # Parameter für VX-REASON
            params = {
                "timeline_id": timeline.id,
                "nodes": nodes_data,
                "connections": timeline.connections,
                "analysis_type": "causal",
                "reasoning_depth": self.vxor_config.get("vx_reason", {}).get("reasoning_depth", 4),
                "use_quantum_logic": self.vxor_config.get("vx_reason", {}).get("use_quantum_logic", True),
                "context": {"engine": "prism"}
            }
            
            # Führe Analyse mit VX-REASON durch
            result = self.vx_reason.analyze_causality(params)
            
            if result and result.get("success", False):
                # Ergebnisse verarbeiten und zurückgeben
                return {
                    "timeline_id": timeline.id,
                    "causality_chains": result.get("causality_chains", []),
                    "integrity_score": result.get("integrity_score", 0.0),
                    "critical_points": result.get("critical_points", []),
                    "suggestions": result.get("suggestions", []),
                    "analysis_method": "vx_reason",
                    "reasoning_depth": params["reasoning_depth"]
                }
            
            # Fallback zur Standard-PRISM-Analyse bei Fehler
            logger.warning("VX-REASON-Analyse fehlgeschlagen, verwende Standard-PRISM-Analyse")
            return self.prism_engine.analyze_timeline_integrity(timeline.id)
            
        except Exception as e:
            logger.error(f"Fehler bei der VX-REASON-Analyse: {e}")
            # Fallback zur Standard-PRISM-Analyse
            return self.prism_engine.analyze_timeline_integrity(timeline.id)
    
    def resolve_paradox(self, timeline_id: str, paradox_id: str) -> Dict[str, Any]:
        """
        Löst ein Paradoxon mit VX-REASON
        
        Args:
            timeline_id: ID der Zeitlinie
            paradox_id: ID des Paradoxons
            
        Returns:
            Ergebnisse der Paradoxauflösung
        """
        # Wenn VX-REASON nicht verfügbar ist, verwende die Standard-Paradoxauflösung
        if not self.reason_available:
            logger.warning("VX-REASON nicht verfügbar, verwende Standard-Paradoxauflösung")
            return self.paradox_resolver.resolve_paradox(timeline_id, paradox_id)
        
        try:
            # Hole Paradoxon-Details
            timeline = self.prism_engine.get_registered_timeline(timeline_id)
            if not timeline:
                raise ValueError(f"Zeitlinie mit ID {timeline_id} nicht gefunden")
            
            paradox = self.paradox_detector.get_paradox(paradox_id)
            if not paradox:
                raise ValueError(f"Paradoxon mit ID {paradox_id} nicht gefunden")
            
            # Parameter für VX-REASON
            params = {
                "timeline_id": timeline_id,
                "paradox_id": paradox_id,
                "paradox_type": paradox.paradox_type,
                "paradox_data": paradox.data,
                "nodes_involved": paradox.nodes_involved,
                "resolution_mode": self.vxor_config.get("vx_reason", {}).get("paradox_resolution_mode", "adaptive"),
                "use_quantum_logic": self.vxor_config.get("vx_reason", {}).get("use_quantum_logic", True),
                "context": {"engine": "prism"}
            }
            
            # Führe Paradoxauflösung mit VX-REASON durch
            result = self.vx_reason.resolve_paradox(params)
            
            if result and result.get("success", False):
                # Ergebnisse verarbeiten und zurückgeben
                
                # Wenn Knotenänderungen vorgeschlagen wurden, wende sie an
                if "node_modifications" in result:
                    for node_mod in result["node_modifications"]:
                        node_id = node_mod.get("id")
                        if node_id in timeline.nodes:
                            node = timeline.nodes[node_id]
                            for key, value in node_mod.get("changes", {}).items():
                                setattr(node, key, value)
                
                # Wenn neue Knotenverbindungen vorgeschlagen wurden, wende sie an
                if "connection_modifications" in result:
                    for conn_mod in result["connection_modifications"]:
                        if conn_mod.get("action") == "add":
                            source_id = conn_mod.get("source_id")
                            target_id = conn_mod.get("target_id")
                            if source_id in timeline.nodes and target_id in timeline.nodes:
                                timeline.add_dependency(source_id, target_id)
                        elif conn_mod.get("action") == "remove":
                            source_id = conn_mod.get("source_id")
                            target_id = conn_mod.get("target_id")
                            if source_id in timeline.nodes and target_id in timeline.nodes:
                                timeline.remove_dependency(source_id, target_id)
                
                # Aktualisiere die Zeitlinie und Paradoxonstatus
                self.prism_engine.update_timeline(timeline)
                self.paradox_detector.update_paradox_status(paradox_id, "resolved")
                
                return {
                    "timeline_id": timeline_id,
                    "paradox_id": paradox_id,
                    "resolution_method": "vx_reason",
                    "resolution_mode": params["resolution_mode"],
                    "success": True,
                    "changes_applied": {
                        "node_modifications": result.get("node_modifications", []),
                        "connection_modifications": result.get("connection_modifications", [])
                    },
                    "resolution_details": result.get("resolution_details", {}),
                    "resolution_score": result.get("resolution_score", 0.0)
                }
            
            # Fallback zur Standard-Paradoxauflösung bei Fehler
            logger.warning("VX-REASON-Paradoxauflösung fehlgeschlagen, verwende Standard-Paradoxauflösung")
            return self.paradox_resolver.resolve_paradox(timeline_id, paradox_id)
            
        except Exception as e:
            logger.error(f"Fehler bei der VX-REASON-Paradoxauflösung: {e}")
            # Fallback zur Standard-Paradoxauflösung
            return self.paradox_resolver.resolve_paradox(timeline_id, paradox_id)


# Singleton-Instanz der Integration
_integration_instance = None

def get_prism_vxor_integration() -> PRISMVXORIntegration:
    """
    Gibt die Singleton-Instanz der PRISM-VXOR-Integration zurück
    
    Returns:
        PRISMVXORIntegration-Instanz
    """
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = PRISMVXORIntegration()
    return _integration_instance


# Initialisiere die Integration, wenn das Modul importiert wird
get_prism_vxor_integration()

# Hauptfunktion
if __name__ == "__main__":
    integration = get_prism_vxor_integration()
    print(f"PRISM ↔ VX-REASON Integration Status: {integration.reason_available}")
