#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VXOR-Adapter Core - Zentrale Integrationsschnittstelle für VXOR-Module

Diese Datei implementiert den zentralen Adapter für die nahtlose Integration
aller VXOR-Module mit den MISO-Kernkomponenten. Sie löst Importprobleme,
stellt eine einheitliche Schnittstelle bereit und gewährleistet ZTM-Konformität.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import json
import logging
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Type

# Konfiguriere Logging
logger = logging.getLogger("MISO.vxor.adapter_core")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ZTM-Protokoll-Initialisierung
ZTM_ACTIVE = os.environ.get('MISO_ZTM_MODE', '1') == '1'
ZTM_LOG_LEVEL = os.environ.get('MISO_ZTM_LOG_LEVEL', 'INFO')

def ztm_log(message: str, level: str = 'INFO', module: str = 'VXOR'):
    """ZTM-konforme Logging-Funktion mit Audit-Trail"""
    if not ZTM_ACTIVE:
        return
    
    log_func = getattr(logger, level.lower())
    log_func(f"[ZTM:{module}] {message}")

# Definiere Pfade zu VXOR-Modulen
MISO_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
PROJECT_ROOT = Path(os.path.abspath(os.path.join(MISO_ROOT, "..")))
# Externes VXOR.AI-Verzeichnis für die vollständige Integration
EXTERNAL_VXOR_ROOT = Path("/Volumes/My Book/VXOR.AI")
VXOR_MODULES_PATHS = [
    PROJECT_ROOT / "vXor_Modules",
    MISO_ROOT / "vxor",
    PROJECT_ROOT / "vxor.ai",
    EXTERNAL_VXOR_ROOT  # Externes VXOR.AI-Verzeichnis hinzugefügt
]

# Füge Pfade zum Pythonpfad hinzu
for path in VXOR_MODULES_PATHS:
    if path.exists() and str(path) not in sys.path:
        sys.path.append(str(path))

# Lade Manifest
def load_vxor_manifest() -> Dict[str, Any]:
    """
    Lädt das VXOR-Manifest aus der Datei vxor_manifest.json
    
    Returns:
        Dictionary mit dem Manifest-Inhalt
    """
    manifest_paths = [
        PROJECT_ROOT / "vXor_Modules" / "vxor_manifest.json",
        MISO_ROOT / "vxor" / "vxor_manifest.json"
    ]
    
    for path in manifest_paths:
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                    if ZTM_ACTIVE:
                        ztm_log(f"VXOR-Manifest geladen aus {path}", level="INFO")
                    return manifest
            except Exception as e:
                if ZTM_ACTIVE:
                    ztm_log(f"Fehler beim Laden des VXOR-Manifests aus {path}: {e}", level="ERROR")
                logger.error(f"Fehler beim Laden des VXOR-Manifests aus {path}: {e}")
    
    # Fallback: Leeres Manifest
    if ZTM_ACTIVE:
        ztm_log("Kein VXOR-Manifest gefunden, verwende leeres Manifest", level="WARNING")
    return {"modules": {"implemented": {}, "planned": {}}}

# Lade das Manifest
VXOR_MANIFEST = load_vxor_manifest()

class ModuleNotFoundError(ImportError):
    """Fehler, wenn ein VXOR-Modul nicht gefunden wurde"""
    pass

class VXORAdapterError(Exception):
    """Basisklasse für alle VXOR-Adapter-Fehler"""
    pass

class ModuleImportError(VXORAdapterError):
    """Fehler beim Importieren eines VXOR-Moduls"""
    pass

class ModuleIntegrationError(VXORAdapterError):
    """Fehler bei der Integration eines VXOR-Moduls"""
    pass

class VXORAdapter:
    """
    Zentraler Adapter für VXOR-Module
    
    Diese Klasse stellt eine einheitliche Schnittstelle für alle VXOR-Module bereit
    und kümmert sich um die korrekte Initialisierung, Kommunikation und Fehlerbehandlung.
    """
    
    def __init__(self):
        """Initialisiert den VXOR-Adapter"""
        self.modules = {}
        self.module_status = {}
        self.manifest = VXOR_MANIFEST
        
        # Initialisiere VXOR-Module
        self._initialize_modules()
        
        if ZTM_ACTIVE:
            ztm_log("VXOR-Adapter initialisiert", level="INFO")
    
    def _initialize_modules(self):
        """Initialisiert alle verfügbaren VXOR-Module"""
        # Durchsuche alle implementierten Module im Manifest
        implemented_modules = self.manifest.get("modules", {}).get("implemented", {})
        
        for module_name, module_info in implemented_modules.items():
            try:
                # Versuche, das Modul zu importieren
                self._import_module(module_name, module_info)
                
                if ZTM_ACTIVE:
                    ztm_log(f"VXOR-Modul {module_name} initialisiert", level="INFO")
            except (ModuleNotFoundError, ModuleImportError) as e:
                # Speichere den Fehler im Status
                self.module_status[module_name] = {
                    "status": "error",
                    "error": str(e)
                }
                
                if ZTM_ACTIVE:
                    ztm_log(f"Fehler bei der Initialisierung von {module_name}: {e}", level="ERROR")
                logger.error(f"Fehler bei der Initialisierung von {module_name}: {e}")
    
    def _import_module(self, module_name: str, module_info: Dict[str, Any]):
        """
        Importiert ein VXOR-Modul
        
        Args:
            module_name: Name des Moduls
            module_info: Informationen über das Modul aus dem Manifest
        """
        # Suche nach dem Modul in allen VXOR-Pfaden
        module_found = False
        module_path = None
        
        # Prüfe, ob ein expliziter Importpfad im Manifest angegeben ist
        if "import_path" in module_info:
            specific_path = Path(module_info["import_path"])
            if specific_path.exists():
                module_path = specific_path
                module_found = True
        
        # Wenn kein expliziter Pfad gefunden wurde, suche in allen Standardpfaden
        if not module_found:
            for base_path in VXOR_MODULES_PATHS:
                # Überprüfe verschiedene mögliche Pfadstrukturen
                potential_paths = [
                    base_path / module_name / "__init__.py",
                    base_path / module_name / f"{module_name.lower()}.py",
                    base_path / f"{module_name}.py",
                    base_path / module_name.replace("-", "_").lower() / "__init__.py",
                    base_path / f"{module_name.replace('-', '_').lower()}.py"
                ]
                
                for path in potential_paths:
                    if path.exists():
                        module_path = path.parent if path.name == "__init__.py" else path
                        module_found = True
                        break
                
                if module_found:
                    break
        
        if not module_found:
            self.module_status[module_name] = {
                "status": "not_found",
                "error": f"Modul {module_name} wurde in keinem der VXOR-Pfade gefunden"
            }
            raise ModuleNotFoundError(f"VXOR-Modul {module_name} nicht gefunden")
        
        try:
            # Importiere das Modul
            if module_path.is_dir():
                # Versuche, das Modul als Paket zu importieren
                module_spec = importlib.util.find_spec(f"{module_path.name}")
                if module_spec is None:
                    # Fallback: Füge den Pfad zum sys.path hinzu und importiere
                    if str(module_path.parent) not in sys.path:
                        sys.path.append(str(module_path.parent))
                    module = importlib.import_module(module_path.name)
                else:
                    module = importlib.util.module_from_spec(module_spec)
                    module_spec.loader.exec_module(module)
            else:
                # Importiere das Modul als Datei
                module_spec = importlib.util.spec_from_file_location(
                    module_name.replace("-", "_").lower(),
                    module_path
                )
                if module_spec is None:
                    raise ModuleImportError(f"Konnte kein Modul-Spec für {module_name} erstellen")
                
                module = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(module)
            
            # Speichere das Modul und seinen Status
            self.modules[module_name] = module
            self.module_status[module_name] = {
                "status": "loaded",
                "path": str(module_path)
            }
            
            logger.info(f"VXOR-Modul {module_name} importiert von {module_path}")
        except Exception as e:
            self.module_status[module_name] = {
                "status": "import_error",
                "error": str(e),
                "path": str(module_path) if module_path else "unknown"
            }
            raise ModuleImportError(f"Fehler beim Importieren von {module_name}: {e}")
    
    def get_module(self, module_name: str) -> Any:
        """
        Gibt ein VXOR-Modul zurück
        
        Args:
            module_name: Name des Moduls
            
        Returns:
            Modul-Instanz
            
        Raises:
            ModuleNotFoundError: Wenn das Modul nicht gefunden wurde
        """
        # Normalisiere den Modulnamen
        if not module_name.startswith("VX-") and "-" not in module_name:
            module_name = f"VX-{module_name.upper()}"
        
        if module_name not in self.modules:
            # Versuche, das Modul jetzt zu laden (falls es noch nicht initialisiert wurde)
            module_info = self.manifest.get("modules", {}).get("implemented", {}).get(module_name)
            if module_info:
                try:
                    self._import_module(module_name, module_info)
                except (ModuleNotFoundError, ModuleImportError) as e:
                    if ZTM_ACTIVE:
                        ztm_log(f"Fehler beim Laden von {module_name}: {e}", level="ERROR")
                    raise
            else:
                raise ModuleNotFoundError(f"VXOR-Modul {module_name} ist nicht im Manifest definiert")
        
        return self.modules[module_name]
    
    def get_module_status(self, module_name: str = None) -> Dict[str, Any]:
        """
        Gibt den Status eines oder aller VXOR-Module zurück
        
        Args:
            module_name: Name des Moduls oder None für alle Module
            
        Returns:
            Dictionary mit Statusinfomationen
        """
        if module_name:
            # Normalisiere den Modulnamen
            if not module_name.startswith("VX-") and "-" not in module_name:
                module_name = f"VX-{module_name.upper()}"
            
            return self.module_status.get(module_name, {"status": "unknown"})
        else:
            return self.module_status
    
    def integrate_with_miso(self, miso_module: str, vxor_module: str, method: str = None) -> bool:
        """
        Integriert ein VXOR-Modul mit einem MISO-Modul
        
        Args:
            miso_module: Name des MISO-Moduls
            vxor_module: Name des VXOR-Moduls
            method: Optionaler Name der Integrationsmethode
            
        Returns:
            True bei erfolgreicher Integration, False sonst
        """
        try:
            # Lade das VXOR-Modul
            module = self.get_module(vxor_module)
            
            # Überprüfe Integration im Manifest
            integration_points = self.manifest.get("integration_points", {}).get("miso", {})
            miso_module_info = integration_points.get(miso_module.lower(), {})
            
            # Überprüfe, ob das VXOR-Modul mit dem MISO-Modul kompatibel ist
            compatible_modules = miso_module_info.get("compatible_modules", [])
            if vxor_module not in compatible_modules and f"VX-{vxor_module.upper()}" not in compatible_modules:
                if ZTM_ACTIVE:
                    ztm_log(f"VXOR-Modul {vxor_module} ist nicht kompatibel mit MISO-Modul {miso_module}", level="WARNING")
                logger.warning(f"VXOR-Modul {vxor_module} ist nicht kompatibel mit MISO-Modul {miso_module}")
            
            # Bestimme die Integrationsmethode
            if method is None:
                method = miso_module_info.get("integration_method", "integrate")
            
            # Überprüfe, ob das Modul die Integrationsmethode hat
            if hasattr(module, method):
                integration_method = getattr(module, method)
                result = integration_method(miso_module=miso_module)
                
                if ZTM_ACTIVE:
                    ztm_log(f"VXOR-Modul {vxor_module} mit MISO-Modul {miso_module} integriert", level="INFO")
                
                return True
            else:
                # Fallback: Versuche eine generische Integration
                if hasattr(module, "integrate"):
                    result = module.integrate(miso_module=miso_module)
                    
                    if ZTM_ACTIVE:
                        ztm_log(f"VXOR-Modul {vxor_module} mit MISO-Modul {miso_module} generisch integriert", level="INFO")
                    
                    return True
                else:
                    if ZTM_ACTIVE:
                        ztm_log(f"VXOR-Modul {vxor_module} hat keine Integrationsmethode für {miso_module}", level="WARNING")
                    logger.warning(f"VXOR-Modul {vxor_module} hat keine Integrationsmethode für {miso_module}")
                    return False
        except (ModuleNotFoundError, ModuleImportError, ModuleIntegrationError) as e:
            if ZTM_ACTIVE:
                ztm_log(f"Fehler bei der Integration von {vxor_module} mit {miso_module}: {e}", level="ERROR")
            logger.error(f"Fehler bei der Integration von {vxor_module} mit {miso_module}: {e}")
            return False

# Singleton-Instanz für den Adapter
_adapter_instance = None

def get_adapter() -> VXORAdapter:
    """
    Gibt die Singleton-Instanz des VXOR-Adapters zurück
    
    Returns:
        VXORAdapter-Instanz
    """
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = VXORAdapter()
    return _adapter_instance

class VXORIntegration:
    """
    Integrationsklasse für VXOR-Module
    
    Diese Klasse bietet eine einfache Schnittstelle für andere MISO-Module,
    um mit VXOR-Modulen zu arbeiten.
    """
    
    def __init__(self):
        """Initialisiert die VXOR-Integration"""
        self.adapter = get_adapter()
    
    def get_module(self, module_name: str) -> Any:
        """
        Gibt ein VXOR-Modul zurück
        
        Args:
            module_name: Name des Moduls
            
        Returns:
            Modul-Instanz
        """
        return self.adapter.get_module(module_name)
    
    def get_module_status(self, module_name: str = None) -> Dict[str, Any]:
        """
        Gibt den Status eines oder aller VXOR-Module zurück
        
        Args:
            module_name: Name des Moduls oder None für alle Module
            
        Returns:
            Dictionary mit Statusinfomationen
        """
        return self.adapter.get_module_status(module_name)
    
    def integrate_with_miso(self, miso_module: str, vxor_module: str, method: str = None) -> bool:
        """
        Integriert ein VXOR-Modul mit einem MISO-Modul
        
        Args:
            miso_module: Name des MISO-Moduls
            vxor_module: Name des VXOR-Moduls
            method: Optionaler Name der Integrationsmethode
            
        Returns:
            True bei erfolgreicher Integration, False sonst
        """
        return self.adapter.integrate_with_miso(miso_module, vxor_module, method)
    
    def get_available_modules(self) -> List[str]:
        """
        Gibt eine Liste aller verfügbaren VXOR-Module zurück
        
        Returns:
            Liste der Modulnamen
        """
        return list(self.adapter.modules.keys())
    
    def get_implemented_modules(self) -> List[str]:
        """
        Gibt eine Liste aller im Manifest als implementiert markierten Module zurück
        
        Returns:
            Liste der Modulnamen
        """
        return list(self.adapter.manifest.get("modules", {}).get("implemented", {}).keys())
    
    def get_planned_modules(self) -> List[str]:
        """
        Gibt eine Liste aller im Manifest als geplant markierten Module zurück
        
        Returns:
            Liste der Modulnamen
        """
        return list(self.adapter.manifest.get("modules", {}).get("planned", {}).keys())

# Direkter Zugriff auf Module über Funktionen
def get_module(module_name: str) -> Any:
    """
    Gibt ein VXOR-Modul zurück
    
    Args:
        module_name: Name des Moduls
        
    Returns:
        Modul-Instanz
    """
    return get_adapter().get_module(module_name)

def integrate_with_miso(miso_module: str, vxor_module: str, method: str = None) -> bool:
    """
    Integriert ein VXOR-Modul mit einem MISO-Modul
    
    Args:
        miso_module: Name des MISO-Moduls
        vxor_module: Name des VXOR-Moduls
        method: Optionaler Name der Integrationsmethode
        
    Returns:
        True bei erfolgreicher Integration, False sonst
    """
    return get_adapter().integrate_with_miso(miso_module, vxor_module, method)

def get_module_status(module_name: str = None) -> Dict[str, Any]:
    """
    Gibt den Status eines oder aller VXOR-Module zurück
    
    Args:
        module_name: Name des Moduls oder None für alle Module
        
    Returns:
        Dictionary mit Statusinfomationen
    """
    return get_adapter().get_module_status(module_name)

# Initialisierung beim Import
if ZTM_ACTIVE:
    ztm_log("vx_adapter_core.py geladen", level="INFO")

# Main für Tests
if __name__ == "__main__":
    # Setze Log-Level auf DEBUG für Tests
    logger.setLevel(logging.DEBUG)
    
    # Teste den Adapter
    adapter = get_adapter()
    print(f"Verfügbare Module: {list(adapter.modules.keys())}")
    print(f"Modulstatus: {adapter.get_module_status()}")
    
    # Teste die Integration
    integration = VXORIntegration()
    print(f"Implementierte Module: {integration.get_implemented_modules()}")
    print(f"Geplante Module: {integration.get_planned_modules()}")
