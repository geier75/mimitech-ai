#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VXOR Integration - Bridge zwischen MISO und VXOR Modulen

Diese Datei stellt die Integrationsschnittstelle zwischen den MISO-Komponenten
und den VXOR-Modulen bereit, die von Manus AI implementiert werden.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import json
import logging
import importlib
from typing import Dict, List, Any, Optional, Union, Callable

# Konfiguriere Logging
logger = logging.getLogger("MISO.vxor_integration")

class VXORAdapter:
    """
    Adapter-Klasse für die Integration von VXOR-Modulen in MISO.
    
    Diese Klasse stellt sicher, dass die Kommunikation zwischen MISO und VXOR
    gemäß den definierten Regeln erfolgt und keine Redundanzen oder Kollisionen
    entstehen.
    """
    
    def __init__(self, manifest_path: Optional[str] = None):
        """
        Initialisiert den VXOR-Adapter.
        
        Args:
            manifest_path: Pfad zur vxor_manifest.json-Datei (optional)
        """
        self.manifest_path = manifest_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "vxor_manifest.json"
        )
        self.manifest = self._load_manifest()
        self.modules_cache = {}
        logger.info("VXOR-Adapter initialisiert")
    
    def _load_manifest(self) -> Dict[str, Any]:
        """
        Lädt das VXOR-Manifest.
        
        Returns:
            Dictionary mit dem Manifest-Inhalt
        """
        try:
            with open(self.manifest_path, 'r') as f:
                manifest = json.load(f)
            logger.info(f"VXOR-Manifest erfolgreich geladen: {len(manifest.get('modules', {}).get('implemented', {}))} implementierte Module, {len(manifest.get('modules', {}).get('planned', {}))} geplante Module")
            return manifest
        except Exception as e:
            logger.error(f"Fehler beim Laden des VXOR-Manifests: {e}")
            return {
                "modules": {
                    "implemented": {},
                    "planned": {}
                },
                "integration_points": {}
            }
    
    def is_module_available(self, module_name: str) -> bool:
        """
        Prüft, ob ein VXOR-Modul verfügbar ist.
        
        Args:
            module_name: Name des VXOR-Moduls (z.B. "VX-PSI")
            
        Returns:
            True, wenn das Modul implementiert ist, sonst False
        """
        return module_name in self.manifest.get("modules", {}).get("implemented", {})
    
    def is_module_planned(self, module_name: str) -> bool:
        """
        Prüft, ob ein VXOR-Modul geplant ist.
        
        Args:
            module_name: Name des VXOR-Moduls (z.B. "VX-REFLEX")
            
        Returns:
            True, wenn das Modul geplant ist, sonst False
        """
        return module_name in self.manifest.get("modules", {}).get("planned", {})
    
    def get_module_info(self, module_name: str) -> Dict[str, Any]:
        """
        Gibt Informationen zu einem VXOR-Modul zurück.
        
        Args:
            module_name: Name des VXOR-Moduls
            
        Returns:
            Dictionary mit Modulinformationen oder leeres Dictionary, wenn nicht gefunden
        """
        implemented = self.manifest.get("modules", {}).get("implemented", {})
        planned = self.manifest.get("modules", {}).get("planned", {})
        
        if module_name in implemented:
            return implemented[module_name]
        elif module_name in planned:
            return planned[module_name]
        else:
            return {}
    
    def get_compatible_modules(self, miso_component: str) -> List[str]:
        """
        Gibt eine Liste kompatibler VXOR-Module für eine MISO-Komponente zurück.
        
        Args:
            miso_component: Name der MISO-Komponente (z.B. "prism_engine")
            
        Returns:
            Liste kompatibler VXOR-Module
        """
        integration_points = self.manifest.get("integration_points", {}).get("miso", {})
        
        if miso_component in integration_points:
            return integration_points[miso_component].get("compatible_modules", [])
        else:
            return []
    
    def load_module(self, module_name: str) -> Any:
        """
        Lädt ein VXOR-Modul.
        
        Args:
            module_name: Name des VXOR-Moduls
            
        Returns:
            Das geladene Modul oder None, wenn nicht verfügbar
        """
        if not self.is_module_available(module_name):
            logger.warning(f"VXOR-Modul {module_name} ist nicht verfügbar")
            return None
        
        if module_name in self.modules_cache:
            return self.modules_cache[module_name]
        
        try:
            # Konvertiere Modulnamen in Python-Importpfad (z.B. VX-PSI -> vx_psi)
            module_path = module_name.lower().replace("-", "_")
            module = importlib.import_module(f"vXor_Modules.{module_path}")
            self.modules_cache[module_name] = module
            logger.info(f"VXOR-Modul {module_name} erfolgreich geladen")
            return module
        except Exception as e:
            logger.error(f"Fehler beim Laden des VXOR-Moduls {module_name}: {e}")
            return None
    
    def register_callback(self, miso_component: str, callback: Callable) -> bool:
        """
        Registriert einen Callback für eine MISO-Komponente.
        
        Args:
            miso_component: Name der MISO-Komponente
            callback: Callback-Funktion
            
        Returns:
            True, wenn erfolgreich registriert, sonst False
        """
        # Implementierung der Callback-Registrierung
        # (Wird in zukünftigen Versionen erweitert)
        logger.info(f"Callback für MISO-Komponente {miso_component} registriert")
        return True
    
    def update_manifest(self, new_module: Dict[str, Any], is_implemented: bool = False) -> bool:
        """
        Aktualisiert das VXOR-Manifest mit einem neuen Modul.
        
        Args:
            new_module: Dictionary mit Modulinformationen
            is_implemented: True, wenn das Modul implementiert ist, False wenn geplant
            
        Returns:
            True, wenn erfolgreich aktualisiert, sonst False
        """
        if not new_module.get("name"):
            logger.error("Modulname fehlt")
            return False
        
        module_name = new_module["name"]
        module_info = {
            "description": new_module.get("description", ""),
            "status": "vollständig implementiert" if is_implemented else "geplant",
            "owner": new_module.get("owner", "Manus AI"),
            "dependencies": new_module.get("dependencies", [])
        }
        
        category = "implemented" if is_implemented else "planned"
        self.manifest["modules"][category][module_name] = module_info
        
        try:
            with open(self.manifest_path, 'w') as f:
                json.dump(self.manifest, f, indent=2)
            logger.info(f"VXOR-Manifest erfolgreich aktualisiert: {module_name} ({category})")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Aktualisieren des VXOR-Manifests: {e}")
            return False


# Singleton-Instanz des VXOR-Adapters
vxor_adapter = VXORAdapter()

def check_module_availability(module_name: str) -> Dict[str, Any]:
    """
    Hilfsfunktion zur Prüfung der Verfügbarkeit eines VXOR-Moduls.
    
    Args:
        module_name: Name des VXOR-Moduls
        
    Returns:
        Dictionary mit Informationen zur Verfügbarkeit
    """
    is_available = vxor_adapter.is_module_available(module_name)
    is_planned = vxor_adapter.is_module_planned(module_name)
    module_info = vxor_adapter.get_module_info(module_name)
    
    result = {
        "name": module_name,
        "available": is_available,
        "planned": is_planned,
        "info": module_info
    }
    
    # Füge den Status hinzu, wenn verfügbar
    if is_available and "status" in module_info:
        result["status"] = module_info["status"]
    elif is_planned:
        result["status"] = "geplant"
    else:
        result["status"] = "nicht verfügbar"
    
    # Versuche, das Modul zu laden, um zu prüfen, ob es tatsächlich implementiert ist
    if is_available:
        try:
            module = vxor_adapter.load_module(module_name)
            if module and hasattr(module, "get_module"):
                module_instance = module.get_module()
                if hasattr(module_instance, "get_module_info"):
                    module_info = module_instance.get_module_info()
                    if "status" in module_info:
                        result["status"] = module_info["status"]
        except Exception as e:
            logger.warning(f"Fehler beim Laden des VXOR-Moduls {module_name}: {e}")
    
    return result

def get_compatible_vxor_modules(miso_component: str) -> List[Dict[str, Any]]:
    """
    Hilfsfunktion zur Ermittlung kompatibler VXOR-Module für eine MISO-Komponente.
    
    Args:
        miso_component: Name der MISO-Komponente
        
    Returns:
        Liste mit Informationen zu kompatiblen VXOR-Modulen
    """
    compatible_modules = vxor_adapter.get_compatible_modules(miso_component)
    result = []
    
    for module_name in compatible_modules:
        result.append(check_module_availability(module_name))
    
    return result
