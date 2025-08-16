#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Q-LOGIK VXOR Integration

Dieses Modul implementiert die Integration zwischen der Q-LOGIK Engine und dem VX-PSI Modul.
Es ermöglicht die Entscheidungsfindung mit Berücksichtigung von Bewusstseinssimulation.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple

# Importiere Q-LOGIK Komponenten
from miso.logic.qlogik_engine import BayesianDecisionCore, advanced_qlogik_decision, simple_emotion_weight
from miso.logic.qlogik_mprime import QLOGIKMPrimeIntegration
# Die apply_emotion_weighting Funktion ist in der Codebase direkt integriert
from miso.logic.qlogik_engine import simple_emotion_weight as apply_emotion_weighting
from miso.logic.qlogik_engine import simple_priority_mapping as map_priority_factors

# Erweiterte Paradoxauflösung
from miso.simulation.paradox_resolution import get_paradox_resolution_manager, ParadoxResolutionManager

# Importiere VXOR-Adapter-Core
from miso.vxor.vx_adapter_core import get_module, get_module_status

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.logic.vxor_integration")

class QLOGIKVXORIntegration:
    """
    Klasse zur Integration der Q-LOGIK Engine mit VX-PSI
    
    Diese Klasse stellt die Verbindung zwischen dem Q-LOGIK Framework und
    dem VX-PSI Modul her, um Entscheidungsfindung mit Bewusstseinssimulation
    zu ermöglichen.
    """
    
    _instance = None  # Singleton-Pattern
    
    def __new__(cls, *args, **kwargs):
        """Implementiert das Singleton-Pattern"""
        if cls._instance is None:
            cls._instance = super(QLOGIKVXORIntegration, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialisiert die Q-LOGIK-VXOR-Integration
        
        Args:
            config_path: Pfad zur Konfigurationsdatei (optional)
        """
        # Initialisiere nur einmal (Singleton-Pattern)
        if hasattr(self, 'initialized') and self.initialized:
            return
            
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "config", "vxor_integration_config.json"
        )
        
        # Initialisiere Q-LOGIK-Komponenten
        self.decision_core = BayesianDecisionCore()
        self.mprime_integration = QLOGIKMPrimeIntegration()
        
        # Lade oder erstelle Konfiguration
        self.vxor_config = {}
        self.load_config()
        
        # Dynamischer Import des VX-PSI-Moduls
        try:
            self.vx_psi = get_module("VX-PSI")
            self.psi_available = True
            logger.info("VX-PSI erfolgreich initialisiert")
        except Exception as e:
            self.vx_psi = None
            self.psi_available = False
            logger.warning(f"VX-PSI nicht verfügbar: {e}")
        
        self.initialized = True
        logger.info("QLOGIKVXORIntegration initialisiert")
    
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
            "vx_psi": {
                "enabled": True,
                "consciousness_simulation_level": 3,  # 1-5
                "self_reflection_enabled": True,
                "emotional_weight_factor": 0.7,
                "use_for_critical_decisions": True,
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
    
    def simulate_consciousness(self, context: Dict[str, Any], decision_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Führt eine Bewusstseinssimulation mit VX-PSI durch
        
        Args:
            context: Kontextinformationen für die Simulation
            decision_result: Ergebnis einer Q-LOGIK-Entscheidung
            
        Returns:
            Ergebnis der Bewusstseinssimulation mit Einsichten
        """
        if not self.psi_available:
            return {
                "success": False,
                "error": "VX-PSI nicht verfügbar",
                "insights": [],
                "confidence": decision_result.get("confidence", 0.5)
            }
        
        # Bereite Parameter für VX-PSI vor
        try:
            # Extrahiere relevante Informationen aus dem Entscheidungsergebnis
            hypothesis = decision_result.get("hypothesis", "Unbekannte Entscheidung")
            confidence = decision_result.get("confidence", 0.5)
            weighted_confidence = decision_result.get("weighted_confidence", confidence)
            decision = decision_result.get("decision", "UNBEKANNT")
            
            # Erstelle Parameter für die VX-PSI-Simulation
            psi_params = {
                "decision_context": context,
                "decision_result": {
                    "hypothesis": hypothesis,
                    "confidence": confidence,
                    "weighted_confidence": weighted_confidence,
                    "decision": decision,
                    "details": decision_result
                },
                "simulation_level": context.get("consciousness_level", 
                                       self.vxor_config.get("vx_psi", {}).get("consciousness_simulation_level", 3)),
                "self_reflection": self.vxor_config.get("vx_psi", {}).get("self_reflection_enabled", True)
            }
            
            # Führe Bewusstseinssimulation mit VX-PSI aus
            psi_result = self.vx_psi.simulate(psi_params)
            
            if psi_result and psi_result.get("success", False):
                # Extrahiere Einsichten aus der Simulation
                insights = psi_result.get("insights", [])
                consciousness_confidence = psi_result.get("confidence", weighted_confidence)
                
                # Tiefere Bewusstseinseinsichten basierend auf dem Q-LOGIK-Ergebnis
                return {
                    "success": True,
                    "insights": insights,
                    "confidence": consciousness_confidence,
                    "simulation_level": psi_params["simulation_level"],
                    "simulation_details": psi_result.get("details", {}),
                    "decision_enhanced": psi_result.get("decision_enhanced", False),
                    "self_reflection_result": psi_result.get("self_reflection_result", {})
                }
            else:
                # Simulation fehlgeschlagen oder keine Einsichten
                error = psi_result.get("error", "Unbekannter Fehler bei der Bewusstseinssimulation") if psi_result else "Keine Ergebnisse von VX-PSI"
                logger.warning(f"Bewusstseinssimulation fehlgeschlagen: {error}")
                
                # Verwende emotionale Gewichtung als Fallback
                return {
                    "success": False,
                    "error": error,
                    "insights": [],
                    "confidence": weighted_confidence,
                    "emotion_weighted": True
                }
                
        except Exception as e:
            logger.error(f"Fehler bei der Bewusstseinssimulation: {e}")
            return {
                "success": False,
                "error": str(e),
                "insights": [],
                "confidence": decision_result.get("confidence", 0.5),
                "emotion_weighted": True
            }
    
    def prioritize_with_consciousness(self, items: List[Dict[str, Any]], 
                                        context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Priorisiert Items mit Berücksichtigung von Bewusstseinssimulation
        
        Args:
            items: Liste von Items zur Priorisierung
            context: Kontextinformationen für die Priorisierung mit notwendigen Gewichtungsfaktoren
            
        Returns:
            Priorisierte Liste von Items
        """
        # Extrahiere Gewichtungsfaktoren aus dem Kontext oder verwende Standards
        risk_factor = context.get("risk_factor", 0.3)
        benefit_factor = context.get("benefit_factor", 0.4)
        urgency_factor = context.get("urgency_factor", 0.3)
        
        # Wende die einfache Prioritätszuordnung auf jedes Item an
        prioritized_items = []
        for item in items:
            risk = item.get("risk", 0.5)
            benefit = item.get("benefit", 0.5)
            urgency = item.get("urgency", 0.5)
            
            # Verwende die simple_priority_mapping Funktion aus der Q-LOGIK-Engine
            priority_data = map_priority_factors(risk, benefit, urgency)
            
            # Füge die Prioritätsinformationen zum Item hinzu
            item["priority"] = priority_data["priority"]
            item["priority_factors"] = priority_data["factors"]
            
            prioritized_items.append(item)
        
        # Wenn VX-PSI nicht verfügbar ist, verwende nur die Standardpriorisierung
        if not self.psi_available:
            return prioritized_items
        
        # Bereite Parameter für VX-PSI vor
        try:
            # Erstelle einen Faktoren-Dictionary für die VX-PSI-Simulation
            factors_dict = {
                "risk_factor": risk_factor,
                "benefit_factor": benefit_factor,
                "urgency_factor": urgency_factor
            }
            
            psi_params = {
                "items": items,
                "prioritized_items": prioritized_items,
                "factors": factors_dict,
                "context": context,
                "simulation_level": self.vxor_config.get("vx_psi", {}).get("consciousness_simulation_level", 3),
                "self_reflection": self.vxor_config.get("vx_psi", {}).get("self_reflection_enabled", True)
            }
            
            # Führe Bewusstseinssimulation mit VX-PSI aus
            psi_result = self.vx_psi.simulate(psi_params)
            
            if psi_result and psi_result.get("success", False):
                # Verwende die vom VX-PSI priorisierten Items
                psi_prioritized_items = psi_result.get("prioritized_items", prioritized_items)
                
                # Füge Metadaten hinzu
                for item in psi_prioritized_items:
                    if "psi_metadata" not in item:
                        item["psi_metadata"] = {
                            "consciousness_simulated": True,
                            "confidence": psi_result.get("confidence", 0.0),
                            "insights": psi_result.get("insights", [])
                        }
                
                return psi_prioritized_items
            
            # Fallback zur Standardpriorisierung
            logger.warning("VX-PSI-Simulation fehlgeschlagen, verwende Standardpriorisierung")
            return prioritized_items
            
        except Exception as e:
            logger.error(f"Fehler bei der VX-PSI-Simulation: {e}")
            # Fallback zur Standardpriorisierung
            return prioritized_items
    
    def handle_paradox(self, paradox_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Behandelt ein Paradox mit Hilfe des ParadoxResolutionManager
        
        Args:
            paradox_data: Paradoxinformationen (Kategorie, Schweregrad, betroffene Entities etc.)
            context: Zusätzlicher Kontext für die Paradoxauflösung
            
        Returns:
            Ergebnis der Paradoxauflösung mit Lösungsvorschlägen
        """
        # Initialisiere den ParadoxResolutionManager
        paradox_manager = get_paradox_resolution_manager()
        
        if paradox_manager is None:
            logger.error("Paradoxauflösungsmanager nicht verfügbar")
            return {
                "success": False,
                "error": "Paradoxauflösungsmanager nicht verfügbar",
                "resolution": None,
                "insights": []
            }
        
        # Bereite Paradoxdaten für die Auflösung vor
        prepared_data = paradox_data.copy()
        
        # Füge Bewusstseinssimulation hinzu, wenn VX-PSI verfügbar
        if self.psi_available and context:
            try:
                # Erstelle Parameter für Bewusstseinssimulation
                consciousness_context = {
                    "paradox": paradox_data,
                    "resolution_context": context,
                    "consciousness_level": context.get("consciousness_level", 3)
                }
                
                # Führe Bewusstseinssimulation durch
                consciousness_result = self._simulate_for_paradox(consciousness_context)
                
                if consciousness_result.get("success", False):
                    # Füge Bewusstseinseinsichten hinzu
                    prepared_data["consciousness_insights"] = consciousness_result.get("insights", [])
                    prepared_data["consciousness_confidence"] = consciousness_result.get("confidence", 0.5)
                    
                    # Füge Bewusstseinssimulationsdetails hinzu
                    if "details" not in prepared_data:
                        prepared_data["details"] = {}
                    prepared_data["details"]["consciousness_simulation"] = consciousness_result.get("simulation_details", {})
            except Exception as e:
                logger.warning(f"Fehler bei der Bewusstseinssimulation für Paradoxauflösung: {e}")
        
        # Verwende den ParadoxResolutionManager zur Auflösung
        try:
            resolution_result = paradox_manager.resolve_paradox(prepared_data)
            
            # Füge Metadaten hinzu
            return {
                "success": True,
                "resolution": resolution_result,
                "insights": resolution_result.get("insights", []),
                "resolution_strategy": resolution_result.get("strategy", "unknown"),
                "confidence": resolution_result.get("confidence", 0.0),
                "consciousness_enhanced": self.psi_available and "consciousness_insights" in prepared_data
            }
        except Exception as e:
            logger.error(f"Fehler bei der Paradoxauflösung: {e}")
            return {
                "success": False,
                "error": str(e),
                "resolution": None,
                "insights": []
            }
    
    def _simulate_for_paradox(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Führt eine spezielle Bewusstseinssimulation für Paradoxauflösung durch
        
        Args:
            context: Paradoxkontext mit allen notwendigen Informationen
            
        Returns:
            Simulationsergebnis mit Einsichten zur Paradoxauflösung
        """
        if not self.psi_available:
            return {
                "success": False,
                "error": "VX-PSI nicht verfügbar",
                "insights": []
            }
        
        try:
            # Bereite spezielle Parameter für die Paradoxsimulation vor
            psi_params = {
                "paradox_context": context,
                "simulation_type": "paradox_resolution",
                "simulation_level": context.get("consciousness_level", 3),
                "self_reflection": self.vxor_config.get("vx_psi", {}).get("self_reflection_enabled", True)
            }
            
            # Führe spezielle Bewusstseinssimulation für Paradoxa aus
            psi_result = self.vx_psi.simulate_special(psi_params)
            
            if psi_result and psi_result.get("success", False):
                return {
                    "success": True,
                    "insights": psi_result.get("insights", []),
                    "confidence": psi_result.get("confidence", 0.5),
                    "simulation_details": psi_result.get("details", {})
                }
            else:
                error = psi_result.get("error", "Unbekannter Fehler") if psi_result else "Keine Ergebnisse"
                logger.warning(f"Paradox-Bewusstseinssimulation fehlgeschlagen: {error}")
                return {
                    "success": False,
                    "error": error,
                    "insights": []
                }
        except Exception as e:
            logger.error(f"Fehler bei der Paradox-Bewusstseinssimulation: {e}")
            return {
                "success": False,
                "error": str(e),
                "insights": []
            }
        
        # Bereite Parameter für VX-PSI vor
        try:
            # Erstelle einen Faktoren-Dictionary für die VX-PSI-Simulation
            factors_dict = {
                "risk_factor": risk_factor,
                "benefit_factor": benefit_factor,
                "urgency_factor": urgency_factor
            }
            
            psi_params = {
                "items": items,
                "prioritized_items": prioritized_items,
                "factors": factors_dict,
                "context": context,
                "simulation_level": self.vxor_config.get("vx_psi", {}).get("consciousness_simulation_level", 3),
                "self_reflection": self.vxor_config.get("vx_psi", {}).get("self_reflection_enabled", True)
            }
            
            # Führe Bewusstseinssimulation mit VX-PSI aus
            psi_result = self.vx_psi.simulate(psi_params)
            
            if psi_result and psi_result.get("success", False):
                # Verwende die vom VX-PSI priorisierten Items
                psi_prioritized_items = psi_result.get("prioritized_items", prioritized_items)
                
                # Füge Metadaten hinzu
                for item in psi_prioritized_items:
                    if "psi_metadata" not in item:
                        item["psi_metadata"] = {
                            "consciousness_simulated": True,
                            "confidence": psi_result.get("confidence", 0.0),
                            "insights": psi_result.get("insights", [])
                        }
                
                return psi_prioritized_items
            
            # Fallback zur Standardpriorisierung
            logger.warning("VX-PSI-Simulation fehlgeschlagen, verwende Standardpriorisierung")
            return prioritized_items
            
        except Exception as e:
            logger.error(f"Fehler bei der VX-PSI-Simulation: {e}")
            # Fallback zur Standardpriorisierung
            return prioritized_items


# Singleton-Instanz der Integration
_integration_instance = None

def get_qlogik_vxor_integration() -> QLOGIKVXORIntegration:
    """
    Gibt die Singleton-Instanz der Q-LOGIK-VXOR-Integration zurück
    
    Returns:
        QLOGIKVXORIntegration-Instanz
    """
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = QLOGIKVXORIntegration()
    return _integration_instance




# Initialisiere die Integration, wenn das Modul importiert wird
get_qlogik_vxor_integration()

# Hauptfunktion
if __name__ == "__main__":
    integration = get_qlogik_vxor_integration()
    print(f"Q-LOGIK ↔ VX-PSI Integration Status: {integration.psi_available}")
    print(f"Paradoxauflösung vorbereitet mit ParadoxResolutionManager")
