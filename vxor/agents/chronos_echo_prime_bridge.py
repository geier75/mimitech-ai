#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECHO-PRIME ↔ VX-CHRONOS Brücke

Diese Brücke ermöglicht die nahtlose Integration zwischen dem ECHO-PRIME Zeitlinienmanagement
und dem VX-CHRONOS VXOR-Modul für temporale Entscheidungsoptimierung.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import json
import logging
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime
from pathlib import Path

# Konfiguriere Logging
logger = logging.getLogger("MISO.vxor.chronos_echo_prime_bridge")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ZTM-Protokoll-Initialisierung
ZTM_ACTIVE = os.environ.get('MISO_ZTM_MODE', '1') == '1'
ZTM_LOG_LEVEL = os.environ.get('MISO_ZTM_LOG_LEVEL', 'INFO')

def ztm_log(message: str, level: str = 'INFO', module: str = 'CHRONOS_BRIDGE'):
    """ZTM-konforme Logging-Funktion mit Audit-Trail"""
    if not ZTM_ACTIVE:
        return
    
    log_func = getattr(logger, level.lower())
    log_func(f"[ZTM:{module}] {message}")

# Definiere Pfade und füge sie zum Pythonpfad hinzu
MISO_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
PROJECT_ROOT = Path(os.path.abspath(os.path.join(MISO_ROOT, "..")))
VXOR_MODULES_PATH = PROJECT_ROOT / "vXor_Modules"

# Füge Pfade zum Pythonpfad hinzu
sys.path.insert(0, str(MISO_ROOT))
sys.path.insert(0, str(VXOR_MODULES_PATH))

# Verwende den VXOR-Adapter, um VX-CHRONOS zu importieren
try:
    from miso.vxor.vx_adapter_core import get_module, get_module_status
    VXOR_ADAPTER_AVAILABLE = True
except ImportError as e:
    VXOR_ADAPTER_AVAILABLE = False
    logger.error(f"VXOR-Adapter konnte nicht importiert werden: {e}")

# Importiere grundlegende ECHO-PRIME Komponenten
try:
    from miso.timeline.echo_prime import TimeNode, Timeline, TriggerLevel
    ECHO_PRIME_BASIC_AVAILABLE = True
except ImportError as e:
    ECHO_PRIME_BASIC_AVAILABLE = False
    logger.error(f"ECHO-PRIME Basiskomponenten konnten nicht importiert werden: {e}")

# Lazy-Loading für EchoPrimeController um zirkuläre Importe zu vermeiden
def get_echo_prime_controller():
    """
    Lazy-Loading Funktion für den EchoPrimeController
    Verhindert zirkuläre Importe zwischen ECHO-PRIME und VX-CHRONOS
    
    Returns:
        EchoPrimeController-Klasse oder None, falls nicht verfügbar
    """
    try:
        from miso.timeline.echo_prime_controller import EchoPrimeController
        return EchoPrimeController
    except ImportError as e:
        logger.error(f"EchoPrimeController konnte nicht importiert werden: {e}")
        return None

# Flag für die Verfügbarkeit von ECHO-PRIME
ECHO_PRIME_AVAILABLE = ECHO_PRIME_BASIC_AVAILABLE

class ChronosEchoBridge:
    """
    Brückenklasse für die Integration von ECHO-PRIME und VX-CHRONOS
    
    Diese Klasse stellt eine bidirektionale Kommunikation zwischen dem
    ECHO-PRIME Zeitlinienmanagement und dem VX-CHRONOS VXOR-Modul her.
    """
    
    def __init__(self, echo_prime_controller=None):
        """
        Initialisiert die Brücke zwischen ECHO-PRIME und VX-CHRONOS
        
        Args:
            echo_prime_controller: Optional vorhandener EchoPrimeController
        """
        self.chronos_module = None
        self.echo_prime = None
        self.bridge_id = str(uuid.uuid4())
        self.timeline_mappings = {}  # Zuordnung ECHO-PRIME Timeline IDs zu CHRONOS Timeline IDs
        self.node_mappings = {}      # Zuordnung ECHO-PRIME Node IDs zu CHRONOS Node IDs
        self.initialized = False
        
        # ZTM-Überwachung aktivieren
        if ZTM_ACTIVE:
            ztm_log("ChronosEchoBridge wird initialisiert", level="INFO")
        
        # Initialisiere EchoPrimeController, falls nicht übergeben
        if echo_prime_controller is not None:
            self.echo_prime = echo_prime_controller
        elif ECHO_PRIME_AVAILABLE:
            # Lazy-Loading für EchoPrimeController verwenden
            EchoPrimeController = get_echo_prime_controller()
            if EchoPrimeController:
                self.echo_prime = EchoPrimeController()
            else:
                logger.error("EchoPrimeController konnte nicht geladen werden")
                return
        else:
            logger.error("ECHO-PRIME ist nicht verfügbar, Bridge kann nicht initialisiert werden")
            return
        
        # Lade CHRONOS-Modul
        self._load_chronos_module()
        
        if self.chronos_module is not None and self.echo_prime is not None:
            self.initialized = True
            if ZTM_ACTIVE:
                ztm_log("ChronosEchoBridge erfolgreich initialisiert", level="INFO")
            logger.info("ChronosEchoBridge erfolgreich initialisiert")
    
    def _load_chronos_module(self):
        """Lädt das VX-CHRONOS-Modul über den VXOR-Adapter"""
        if not VXOR_ADAPTER_AVAILABLE:
            logger.error("VXOR-Adapter ist nicht verfügbar, VX-CHRONOS kann nicht geladen werden")
            return
        
        try:
            # Lade das Modul über den VXOR-Adapter
            self.chronos_module = get_module("VX-CHRONOS")
            
            if ZTM_ACTIVE:
                ztm_log("VX-CHRONOS-Modul erfolgreich geladen", level="INFO")
            logger.info("VX-CHRONOS-Modul erfolgreich geladen")
        except Exception as e:
            logger.error(f"Fehler beim Laden des VX-CHRONOS-Moduls: {e}")
            
            # Überprüfe den Status des Moduls
            chronos_status = get_module_status("VX-CHRONOS")
            if chronos_status and "error" in chronos_status:
                logger.error(f"Modulstatus für VX-CHRONOS: {chronos_status['error']}")
    
    def sync_timeline(self, echo_timeline_id: str) -> bool:
        """
        Synchronisiert eine ECHO-PRIME-Zeitlinie mit VX-CHRONOS
        
        Args:
            echo_timeline_id: ID der ECHO-PRIME-Zeitlinie
            
        Returns:
            True bei erfolgreicher Synchronisation, False sonst
        """
        if not self.initialized:
            logger.error("Bridge ist nicht initialisiert, Synchronisation nicht möglich")
            return False
        
        # Hole die Zeitlinie aus ECHO-PRIME
        echo_timeline = self.echo_prime.get_timeline(echo_timeline_id)
        if echo_timeline is None:
            logger.error(f"Zeitlinie mit ID {echo_timeline_id} nicht gefunden")
            return False
        
        try:
            # Konvertiere die ECHO-PRIME-Zeitlinie in ein Format, das VX-CHRONOS versteht
            chronos_timeline_data = self._convert_echo_to_chronos_timeline(echo_timeline)
            
            # Erstelle oder aktualisiere die Zeitlinie in VX-CHRONOS
            if echo_timeline_id in self.timeline_mappings:
                # Aktualisiere bestehende Zeitlinie
                chronos_timeline_id = self.timeline_mappings[echo_timeline_id]
                chronos_timeline = self.chronos_module.update_timeline(
                    chronos_timeline_id, chronos_timeline_data
                )
                if ZTM_ACTIVE:
                    ztm_log(f"Zeitlinie {echo_timeline_id} mit CHRONOS {chronos_timeline_id} aktualisiert", level="INFO")
            else:
                # Erstelle neue Zeitlinie
                chronos_timeline = self.chronos_module.create_timeline(chronos_timeline_data)
                chronos_timeline_id = chronos_timeline["timeline_id"]
                self.timeline_mappings[echo_timeline_id] = chronos_timeline_id
                if ZTM_ACTIVE:
                    ztm_log(f"Neue Zeitlinie in CHRONOS erstellt: {chronos_timeline_id}", level="INFO")
            
            # Synchronisiere alle Knoten
            self._sync_nodes(echo_timeline, chronos_timeline_id)
            
            return True
        except Exception as e:
            logger.error(f"Fehler bei der Synchronisation der Zeitlinie: {e}")
            if ZTM_ACTIVE:
                ztm_log(f"Synchronisationsfehler: {e}", level="ERROR")
            return False
    
    def _convert_echo_to_chronos_timeline(self, echo_timeline) -> Dict[str, Any]:
        """
        Konvertiert eine ECHO-PRIME-Zeitlinie in das VX-CHRONOS-Format
        
        Args:
            echo_timeline: ECHO-PRIME-Zeitlinienobjekt
            
        Returns:
            Dictionary mit Zeitliniendaten im VX-CHRONOS-Format
        """
        return {
            "name": echo_timeline.name,
            "description": echo_timeline.description,
            "metadata": {
                "source": "ECHO-PRIME",
                "echo_timeline_id": echo_timeline.id,
                "bridge_id": self.bridge_id,
                "creation_time": datetime.now().isoformat(),
                "ztm_compliant": ZTM_ACTIVE
            }
        }
    
    def _sync_nodes(self, echo_timeline, chronos_timeline_id: str) -> bool:
        """
        Synchronisiert alle Knoten einer Zeitlinie
        
        Args:
            echo_timeline: ECHO-PRIME-Zeitlinienobjekt
            chronos_timeline_id: ID der VX-CHRONOS-Zeitlinie
            
        Returns:
            True bei erfolgreicher Synchronisation, False sonst
        """
        try:
            # Hole alle Knoten aus der ECHO-PRIME-Zeitlinie
            nodes = echo_timeline.get_all_nodes()
            
            # Erstelle eine Abbildung von Parent-IDs zu Knoten für die Hierarchierekonstruktion
            node_hierarchy = {}
            for node in nodes:
                if node.parent_id not in node_hierarchy:
                    node_hierarchy[node.parent_id] = []
                node_hierarchy[node.parent_id].append(node)
            
            # Beginne mit den Wurzelknoten (keine Parent-ID)
            root_nodes = node_hierarchy.get(None, [])
            
            # Synchronisiere rekursiv, beginnend mit den Wurzelknoten
            for node in root_nodes:
                self._sync_node_recursive(node, chronos_timeline_id, node_hierarchy)
            
            return True
        except Exception as e:
            logger.error(f"Fehler bei der Synchronisation der Knoten: {e}")
            return False
    
    def _sync_node_recursive(self, node, chronos_timeline_id: str, node_hierarchy: Dict, parent_chronos_id: str = None) -> str:
        """
        Synchronisiert einen Knoten und seine Kinder rekursiv
        
        Args:
            node: ECHO-PRIME-Knotenobjekt
            chronos_timeline_id: ID der VX-CHRONOS-Zeitlinie
            node_hierarchy: Dictionary mit Parent-ID zu Knoten-Zuordnung
            parent_chronos_id: ID des Elternknotens in VX-CHRONOS
            
        Returns:
            ID des synchronisierten Knotens in VX-CHRONOS
        """
        # Konvertiere den Knoten in das VX-CHRONOS-Format
        chronos_node_data = {
            "description": node.description,
            "timestamp": node.timestamp,
            "probability": node.probability,
            "parent_id": parent_chronos_id,
            "metadata": {
                "source": "ECHO-PRIME",
                "echo_node_id": node.id,
                "bridge_id": self.bridge_id,
                "trigger_level": node.trigger_level.name if hasattr(node, 'trigger_level') else "MEDIUM",
                "original_metadata": json.dumps(node.metadata) if hasattr(node, 'metadata') else "{}"
            }
        }
        
        # Erstelle oder aktualisiere den Knoten in VX-CHRONOS
        if node.id in self.node_mappings:
            # Aktualisiere bestehenden Knoten
            chronos_node_id = self.node_mappings[node.id]
            chronos_node = self.chronos_module.update_node(
                chronos_timeline_id, chronos_node_id, chronos_node_data
            )
        else:
            # Erstelle neuen Knoten
            chronos_node = self.chronos_module.create_node(
                chronos_timeline_id, chronos_node_data
            )
            chronos_node_id = chronos_node["node_id"]
            self.node_mappings[node.id] = chronos_node_id
        
        # Synchronisiere Kindknoten
        child_nodes = node_hierarchy.get(node.id, [])
        for child_node in child_nodes:
            self._sync_node_recursive(child_node, chronos_timeline_id, node_hierarchy, chronos_node_id)
        
        return chronos_node_id
    
    def apply_chronos_optimizations(self, echo_timeline_id: str) -> Dict[str, Any]:
        """
        Wendet VX-CHRONOS-Optimierungen auf eine ECHO-PRIME-Zeitlinie an
        
        Args:
            echo_timeline_id: ID der ECHO-PRIME-Zeitlinie
            
        Returns:
            Dictionary mit Optimierungsergebnissen
        """
        if not self.initialized:
            logger.error("Bridge ist nicht initialisiert, Optimierungen nicht möglich")
            return {"success": False, "error": "Bridge nicht initialisiert"}
        
        # Synchronisiere die Zeitlinie, falls noch nicht geschehen
        if echo_timeline_id not in self.timeline_mappings:
            if not self.sync_timeline(echo_timeline_id):
                return {"success": False, "error": "Zeitlinie konnte nicht synchronisiert werden"}
        
        chronos_timeline_id = self.timeline_mappings[echo_timeline_id]
        
        try:
            # Führe die VX-CHRONOS-Optimierung durch
            optimization_results = self.chronos_module.optimize_timeline(
                chronos_timeline_id, 
                {
                    "depth": 3,
                    "methods": ["probability", "consistency", "efficiency"],
                    "metadata": {
                        "source": "ECHO-PRIME",
                        "echo_timeline_id": echo_timeline_id,
                        "bridge_id": self.bridge_id
                    }
                }
            )
            
            # Wende die Optimierungen auf die ECHO-PRIME-Zeitlinie an
            self._apply_optimizations_to_echo(echo_timeline_id, optimization_results)
            
            if ZTM_ACTIVE:
                ztm_log(f"VX-CHRONOS-Optimierungen auf Zeitlinie {echo_timeline_id} angewendet", level="INFO")
            
            return {
                "success": True, 
                "echo_timeline_id": echo_timeline_id,
                "chronos_timeline_id": chronos_timeline_id,
                "optimization_summary": optimization_results.get("summary", {}),
                "applied_changes": optimization_results.get("changes", [])
            }
        except Exception as e:
            logger.error(f"Fehler bei der Anwendung von VX-CHRONOS-Optimierungen: {e}")
            return {"success": False, "error": str(e)}
    
    def _apply_optimizations_to_echo(self, echo_timeline_id: str, optimization_results: Dict[str, Any]) -> bool:
        """
        Wendet VX-CHRONOS-Optimierungsergebnisse auf eine ECHO-PRIME-Zeitlinie an
        
        Args:
            echo_timeline_id: ID der ECHO-PRIME-Zeitlinie
            optimization_results: Optimierungsergebnisse von VX-CHRONOS
            
        Returns:
            True bei erfolgreicher Anwendung, False sonst
        """
        echo_timeline = self.echo_prime.get_timeline(echo_timeline_id)
        if echo_timeline is None:
            logger.error(f"Zeitlinie mit ID {echo_timeline_id} nicht gefunden")
            return False
        
        # Verarbeite die Änderungen aus den Optimierungsergebnissen
        changes = optimization_results.get("changes", [])
        for change in changes:
            change_type = change.get("type")
            echo_node_id = None
            
            # Suche die Echo-Node-ID aus den Metadaten oder der Zuordnung
            chronos_node_id = change.get("node_id")
            for echo_id, chron_id in self.node_mappings.items():
                if chron_id == chronos_node_id:
                    echo_node_id = echo_id
                    break
            
            if echo_node_id is None:
                logger.warning(f"Keine Zuordnung für CHRONOS-Knoten {chronos_node_id} gefunden")
                continue
            
            # Wende die Änderung an
            if change_type == "probability_update":
                self.echo_prime.update_time_node(
                    echo_timeline_id, echo_node_id,
                    {"probability": change.get("new_probability", 0.5)}
                )
            elif change_type == "node_add":
                # Erstelle einen neuen Knoten basierend auf den Optimierungsempfehlungen
                parent_chronos_id = change.get("parent_id")
                parent_echo_id = None
                
                # Suche die Echo-Parent-ID
                for echo_id, chron_id in self.node_mappings.items():
                    if chron_id == parent_chronos_id:
                        parent_echo_id = echo_id
                        break
                
                if parent_echo_id is not None:
                    new_node = self.echo_prime.add_time_node(
                        echo_timeline_id,
                        description=change.get("description", "Optimierter Knoten"),
                        parent_node_id=parent_echo_id,
                        probability=change.get("probability", 0.7),
                        trigger_level=TriggerLevel.MEDIUM,
                        metadata={"source": "VX-CHRONOS-OPTIMIZATION"}
                    )
                    
                    # Aktualisiere die Zuordnung
                    if new_node is not None:
                        self.node_mappings[new_node.id] = change.get("new_node_id")
            
            # Weitere Änderungstypen können hier hinzugefügt werden
        
        return True
    
    def detect_paradoxes(self, echo_timeline_id: str) -> Dict[str, Any]:
        """
        Erkennt Paradoxien in einer Zeitlinie mit Hilfe von VX-CHRONOS
        
        Args:
            echo_timeline_id: ID der ECHO-PRIME-Zeitlinie
            
        Returns:
            Dictionary mit Paradoxie-Erkennungsergebnissen
        """
        if not self.initialized:
            logger.error("Bridge ist nicht initialisiert, Paradoxieerkennung nicht möglich")
            return {"success": False, "error": "Bridge nicht initialisiert"}
        
        # Synchronisiere die Zeitlinie, falls noch nicht geschehen
        if echo_timeline_id not in self.timeline_mappings:
            if not self.sync_timeline(echo_timeline_id):
                return {"success": False, "error": "Zeitlinie konnte nicht synchronisiert werden"}
        
        chronos_timeline_id = self.timeline_mappings[echo_timeline_id]
        
        try:
            # Führe die VX-CHRONOS-Paradoxieerkennung durch
            paradox_results = self.chronos_module.detect_paradoxes(
                chronos_timeline_id,
                {
                    "depth": 5,
                    "types": ["causal", "temporal", "logical", "probabilistic"],
                    "metadata": {
                        "source": "ECHO-PRIME",
                        "echo_timeline_id": echo_timeline_id,
                        "bridge_id": self.bridge_id
                    }
                }
            )
            
            if ZTM_ACTIVE:
                ztm_log(f"VX-CHRONOS-Paradoxieerkennung für Zeitlinie {echo_timeline_id} durchgeführt", level="INFO")
            
            # Wandle die Ergebnisse in das ECHO-PRIME-Format um
            echo_paradoxes = self._convert_paradoxes_to_echo_format(echo_timeline_id, paradox_results)
            
            return {
                "success": True,
                "echo_timeline_id": echo_timeline_id,
                "paradoxes": echo_paradoxes,
                "total_count": len(echo_paradoxes),
                "severity_summary": {
                    "critical": len([p for p in echo_paradoxes if p.get("severity") == "CRITICAL"]),
                    "high": len([p for p in echo_paradoxes if p.get("severity") == "HIGH"]),
                    "medium": len([p for p in echo_paradoxes if p.get("severity") == "MEDIUM"]),
                    "low": len([p for p in echo_paradoxes if p.get("severity") == "LOW"])
                }
            }
        except Exception as e:
            logger.error(f"Fehler bei der VX-CHRONOS-Paradoxieerkennung: {e}")
            return {"success": False, "error": str(e)}
    
    def _convert_paradoxes_to_echo_format(self, echo_timeline_id: str, paradox_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Konvertiert VX-CHRONOS-Paradoxieergebnisse in das ECHO-PRIME-Format
        
        Args:
            echo_timeline_id: ID der ECHO-PRIME-Zeitlinie
            paradox_results: Paradoxieergebnisse von VX-CHRONOS
            
        Returns:
            Liste von Paradoxien im ECHO-PRIME-Format
        """
        echo_paradoxes = []
        
        paradoxes = paradox_results.get("paradoxes", [])
        for paradox in paradoxes:
            # Konvertiere die beteiligten Knoten-IDs
            involved_nodes = []
            for chronos_node_id in paradox.get("involved_node_ids", []):
                echo_node_id = None
                for echo_id, chron_id in self.node_mappings.items():
                    if chron_id == chronos_node_id:
                        echo_node_id = echo_id
                        break
                
                if echo_node_id is not None:
                    involved_nodes.append(echo_node_id)
            
            # Konvertiere die Paradoxklassifikation
            paradox_type = self._map_chronos_to_echo_paradox_type(paradox.get("type", "unknown"))
            
            # Erstelle die Paradoxie im ECHO-PRIME-Format
            echo_paradox = {
                "id": str(uuid.uuid4()),
                "timeline_id": echo_timeline_id,
                "type": paradox_type,
                "description": paradox.get("description", "Unbekannte Paradoxie"),
                "severity": paradox.get("severity", "MEDIUM"),
                "involved_node_ids": involved_nodes,
                "recommended_actions": paradox.get("recommendations", []),
                "metadata": {
                    "source": "VX-CHRONOS",
                    "chronos_paradox_id": paradox.get("id"),
                    "detection_time": datetime.now().isoformat()
                }
            }
            
            echo_paradoxes.append(echo_paradox)
        
        return echo_paradoxes
    
    def _map_chronos_to_echo_paradox_type(self, chronos_type: str) -> str:
        """
        Ordnet VX-CHRONOS-Paradoxtypen ECHO-PRIME-Paradoxtypen zu
        
        Args:
            chronos_type: Paradoxtyp von VX-CHRONOS
            
        Returns:
            Paradoxtyp im ECHO-PRIME-Format
        """
        mapping = {
            "causal": "CAUSAL",
            "temporal": "TEMPORAL",
            "logical": "LOGICAL",
            "probabilistic": "PROBABILISTIC",
            "consistency": "CONSISTENCY",
            "information": "INFORMATION",
            "quantum": "QUANTUM"
        }
        
        return mapping.get(chronos_type.lower(), "UNKNOWN")

# Singleton-Instanz für die Brücke
_bridge_instance = None

def get_bridge(echo_prime_controller=None) -> ChronosEchoBridge:
    """
    Gibt die Singleton-Instanz der ChronosEchoBridge zurück
    
    Args:
        echo_prime_controller: Optional vorhandener EchoPrimeController
        
    Returns:
        ChronosEchoBridge-Instanz
    """
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = ChronosEchoBridge(echo_prime_controller)
    return _bridge_instance

# Hilfsfunktionen für den direkten Zugriff
def sync_timeline(echo_timeline_id: str) -> bool:
    """
    Synchronisiert eine ECHO-PRIME-Zeitlinie mit VX-CHRONOS
    
    Args:
        echo_timeline_id: ID der ECHO-PRIME-Zeitlinie
        
    Returns:
        True bei erfolgreicher Synchronisation, False sonst
    """
    return get_bridge().sync_timeline(echo_timeline_id)

def apply_chronos_optimizations(echo_timeline_id: str) -> Dict[str, Any]:
    """
    Wendet VX-CHRONOS-Optimierungen auf eine ECHO-PRIME-Zeitlinie an
    
    Args:
        echo_timeline_id: ID der ECHO-PRIME-Zeitlinie
        
    Returns:
        Dictionary mit Optimierungsergebnissen
    """
    return get_bridge().apply_chronos_optimizations(echo_timeline_id)

def detect_paradoxes(echo_timeline_id: str) -> Dict[str, Any]:
    """
    Erkennt Paradoxien in einer Zeitlinie mit Hilfe von VX-CHRONOS
    
    Args:
        echo_timeline_id: ID der ECHO-PRIME-Zeitlinie
        
    Returns:
        Dictionary mit Paradoxie-Erkennungsergebnissen
    """
    return get_bridge().detect_paradoxes(echo_timeline_id)

# Initialisierung beim Import
if ZTM_ACTIVE:
    ztm_log("chronos_echo_prime_bridge.py geladen", level="INFO")
