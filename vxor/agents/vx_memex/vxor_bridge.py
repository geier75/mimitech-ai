#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-MEMEX: VXOR Bridge Module
----------------------------
Schnittstelle zwischen dem VX-MEMEX Gedächtnismodul und anderen VXOR-Komponenten.
Ermöglicht die Integration mit Q-LOGIK, M-CODE, M-LINGUA, T-MATHEMATICS und anderen
VXOR-Modulen.

Optimiert für Apple Silicon M4 Max.
"""

import json
import time
import logging
import importlib
import inspect
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

# Lokale Module importieren
from memory_core import MemoryCore

# Konstanten
DEFAULT_CONFIG_PATH = "/vXor_Modules/config/vx_memex_config.json"
DEFAULT_MANIFEST_PATH = "/vXor_Modules/manifest/vx_memex_manifest.json"

class VXORBridge:
    """
    Schnittstelle zwischen dem VX-MEMEX Gedächtnismodul und anderen VXOR-Komponenten.
    
    Verantwortlich für:
    - Bereitstellung einer einheitlichen API für andere VXOR-Komponenten
    - Verwaltung der Kommunikation zwischen VX-MEMEX und anderen Modulen
    - Registrierung von Callbacks und Event-Handlern
    - Konfiguration und Initialisierung des Gedächtnismoduls
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialisiert die VXOR-Bridge.
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
        """
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        
        # Logger initialisieren
        self._setup_logging()
        
        # Konfiguration laden
        self.config = self._load_config()
        
        # Memory Core initialisieren
        self.memory_core = MemoryCore(self.config.get('memory_core', {}))
        
        # Registrierte Module und Callbacks
        self.registered_modules = {}
        self.event_handlers = {}
        
        # Manifest-Informationen
        self.manifest = self._generate_manifest()
        
        self.logger.info("VX-MEMEX VXOR-Bridge initialisiert")
    
    def _setup_logging(self):
        """Konfiguriert das Logging für die VXOR-Bridge."""
        try:
            # Versuche, den VXOR-Logger zu importieren, falls verfügbar
            from vxor_logger import get_logger
            self.logger = get_logger('VX-MEMEX.bridge')
        except ImportError:
            # Fallback auf Standard-Logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger('VX-MEMEX.bridge')
            self.logger.info("VXOR-Logger nicht gefunden, Standard-Logger wird verwendet")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Lädt die Konfiguration aus der angegebenen Datei.
        
        Returns:
            Dictionary mit Konfigurationsparametern
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.logger.info(f"Konfiguration aus {self.config_path} geladen")
            return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.warning(f"Fehler beim Laden der Konfiguration: {e}")
            self.logger.info("Verwende Standardkonfiguration")
            return {}
    
    def _generate_manifest(self) -> Dict[str, Any]:
        """
        Generiert ein Manifest mit Informationen über das VX-MEMEX Modul.
        
        Returns:
            Dictionary mit Manifest-Informationen
        """
        manifest = {
            "module_name": "VX-MEMEX",
            "version": "1.0.0",
            "description": "Gedächtnismodul für das VXOR-System",
            "components": [
                {
                    "name": "memory_core",
                    "description": "Zentrale Verwaltung & Steuerung",
                    "api_methods": self._get_public_methods(self.memory_core)
                },
                {
                    "name": "semantic_store",
                    "description": "Semantisches Langzeitwissen (Begriffe, Konzepte)",
                    "api_methods": self._get_public_methods(self.memory_core.semantic_store)
                },
                {
                    "name": "episodic_store",
                    "description": "Zeitbasierte Erfahrungen (mit Timestamps)",
                    "api_methods": self._get_public_methods(self.memory_core.episodic_store)
                },
                {
                    "name": "working_memory",
                    "description": "Temporäre Speicher (kontextabhängig)",
                    "api_methods": self._get_public_methods(self.memory_core.working_memory)
                },
                {
                    "name": "vxor_bridge",
                    "description": "VXOR-Schnittstelle zur Anbindung an PSI, Q-LOGIK etc.",
                    "api_methods": self._get_public_methods(self)
                }
            ],
            "dependencies": [
                {
                    "name": "numpy",
                    "version": ">=1.20.0"
                },
                {
                    "name": "mlx",
                    "version": ">=1.0.0",
                    "optional": True
                }
            ],
            "integration_points": [
                {
                    "module": "Q-LOGIK",
                    "description": "Zugriff auf Gedächtnisinhalt für logische Operationen"
                },
                {
                    "module": "M-CODE",
                    "description": "Speicherung und Abruf von Code-Snippets und Algorithmen"
                },
                {
                    "module": "M-LINGUA",
                    "description": "Speicherung und Abruf von sprachlichen Konstrukten"
                },
                {
                    "module": "T-MATHEMATICS",
                    "description": "Speicherung und Abruf von mathematischen Konzepten und Formeln"
                }
            ],
            "created_at": time.time(),
            "platform": "Apple Silicon M4 Max"
        }
        
        return manifest
    
    def _get_public_methods(self, obj: Any) -> List[Dict[str, Any]]:
        """
        Extrahiert Informationen über öffentliche Methoden eines Objekts.
        
        Args:
            obj: Objekt, dessen Methoden extrahiert werden sollen
            
        Returns:
            Liste mit Informationen über öffentliche Methoden
        """
        methods = []
        
        for name, method in inspect.getmembers(obj, inspect.ismethod):
            # Private Methoden überspringen
            if name.startswith('_'):
                continue
            
            # Methodensignatur extrahieren
            signature = inspect.signature(method)
            
            # Parameter extrahieren
            params = []
            for param_name, param in signature.parameters.items():
                if param_name == 'self':
                    continue
                
                param_info = {
                    "name": param_name,
                    "required": param.default == inspect.Parameter.empty
                }
                
                # Typannotation extrahieren, falls vorhanden
                if param.annotation != inspect.Parameter.empty:
                    param_info["type"] = str(param.annotation)
                
                # Standardwert extrahieren, falls vorhanden
                if param.default != inspect.Parameter.empty:
                    param_info["default"] = str(param.default)
                
                params.append(param_info)
            
            # Rückgabetyp extrahieren, falls vorhanden
            return_type = None
            if signature.return_annotation != inspect.Signature.empty:
                return_type = str(signature.return_annotation)
            
            # Methodendokumentation extrahieren
            doc = inspect.getdoc(method)
            
            methods.append({
                "name": name,
                "parameters": params,
                "return_type": return_type,
                "doc": doc
            })
        
        return methods
    
    def save_manifest(self, path: Optional[str] = None) -> str:
        """
        Speichert das Manifest in einer Datei.
        
        Args:
            path: Pfad zur Ausgabedatei (None = Standardpfad verwenden)
            
        Returns:
            Pfad zur gespeicherten Manifest-Datei
        """
        manifest_path = path or DEFAULT_MANIFEST_PATH
        
        try:
            # Verzeichnis erstellen, falls nicht vorhanden
            import os
            os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
            
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(self.manifest, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Manifest in {manifest_path} gespeichert")
            return manifest_path
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern des Manifests: {e}")
            raise
    
    def register_module(self, module_name: str, module_instance: Any) -> bool:
        """
        Registriert ein VXOR-Modul für die Integration mit VX-MEMEX.
        
        Args:
            module_name: Name des Moduls
            module_instance: Instanz des Moduls
            
        Returns:
            True, wenn die Registrierung erfolgreich war
        """
        if module_name in self.registered_modules:
            self.logger.warning(f"Modul {module_name} bereits registriert, wird überschrieben")
        
        self.registered_modules[module_name] = module_instance
        self.logger.info(f"Modul {module_name} registriert")
        
        return True
    
    def unregister_module(self, module_name: str) -> bool:
        """
        Hebt die Registrierung eines VXOR-Moduls auf.
        
        Args:
            module_name: Name des Moduls
            
        Returns:
            True, wenn die Aufhebung erfolgreich war
        """
        if module_name not in self.registered_modules:
            self.logger.warning(f"Modul {module_name} nicht registriert")
            return False
        
        del self.registered_modules[module_name]
        self.logger.info(f"Registrierung von Modul {module_name} aufgehoben")
        
        return True
    
    def get_registered_modules(self) -> Dict[str, Any]:
        """
        Gibt eine Liste aller registrierten Module zurück.
        
        Returns:
            Dictionary mit registrierten Modulen
        """
        return self.registered_modules
    
    def register_event_handler(self, event_type: str, handler: Callable, module_name: Optional[str] = None) -> bool:
        """
        Registriert einen Event-Handler für ein bestimmtes Ereignis.
        
        Args:
            event_type: Typ des Ereignisses
            handler: Callback-Funktion für das Ereignis
            module_name: Optional, Name des Moduls, das den Handler registriert
            
        Returns:
            True, wenn die Registrierung erfolgreich war
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        handler_info = {
            "handler": handler,
            "module_name": module_name
        }
        
        self.event_handlers[event_type].append(handler_info)
        self.logger.debug(f"Event-Handler für {event_type} registriert" + (f" von Modul {module_name}" if module_name else ""))
        
        return True
    
    def unregister_event_handler(self, event_type: str, handler: Callable) -> bool:
        """
        Hebt die Registrierung eines Event-Handlers auf.
        
        Args:
            event_type: Typ des Ereignisses
            handler: Callback-Funktion für das Ereignis
            
        Returns:
            True, wenn die Aufhebung erfolgreich war
        """
        if event_type not in self.event_handlers:
            self.logger.warning(f"Keine Handler für Ereignistyp {event_type} registriert")
            return False
        
        # Handler suchen und entfernen
        for i, handler_info in enumerate(self.event_handlers[event_type]):
            if handler_info["handler"] == handler:
                self.event_handlers[event_type].pop(i)
                self.logger.debug(f"Event-Handler für {event_type} entfernt")
                return True
        
        self.logger.warning(f"Handler für Ereignistyp {event_type} nicht gefunden")
        return False
    
    def trigger_event(self, event_type: str, event_data: Any) -> int:
        """
        Löst ein Ereignis aus und ruft alle registrierten Handler auf.
        
        Args:
            event_type: Typ des Ereignisses
            event_data: Daten für das Ereignis
            
        Returns:
            Anzahl der aufgerufenen Handler
        """
        if event_type not in self.event_handlers:
            self.logger.debug(f"Keine Handler für Ereignistyp {event_type} registriert")
            return 0
        
        # Ereignis im episodischen Speicher protokollieren
        event_entry = {
            "type": event_type,
            "data": event_data,
            "timestamp": time.time()
        }
        
        self.memory_core.store(
            content=event_entry,
            memory_type='episodic',
            tags=['event', event_type],
            importance=0.7,
            context={"source": "vxor_bridge"}
        )
        
        # Handler aufrufen
        handler_count = 0
        for handler_info in self.event_handlers[event_type]:
            try:
                handler_info["handler"](event_data)
                handler_count += 1
            except Exception as e:
                module_name = handler_info.get("module_name", "unbekannt")
                self.logger.error(f"Fehler beim Aufruf des Event-Handlers für {event_type} von Modul {module_name}: {e}")
        
        self.logger.debug(f"Ereignis {event_type} ausgelöst, {handler_count} Handler aufgerufen")
        
        return handler_count
    
    def store(self, 
             content: Any, 
             memory_type: str = 'all', 
             tags: List[str] = None, 
             ttl: Optional[int] = None,
             importance: float = 0.5,
             context: Dict[str, Any] = None,
             source_module: Optional[str] = None) -> Dict[str, str]:
        """
        Speichert Inhalte im Gedächtnismodul.
        
        Args:
            content: Zu speichernder Inhalt
            memory_type: Zielgedächtnissystem ('semantic', 'episodic', 'working' oder 'all')
            tags: Liste von Schlagwörtern für die Indizierung
            ttl: Time-to-Live in Sekunden (None = Standardwert verwenden)
            importance: Wichtigkeit des Eintrags (0-1)
            context: Zusätzliche Kontextinformationen
            source_module: Name des Quellmoduls
            
        Returns:
            Dictionary mit den IDs der erstellten Einträge
        """
        # Kontext erweitern, falls Quellmodul angegeben
        if source_module:
            context = context or {}
            context["source_module"] = source_module
        
        # An Memory Core delegieren
        result = self.memory_core.store(
            content=content,
            memory_type=memory_type,
            tags=tags,
            ttl=ttl,
            importance=importance,
            context=context
        )
        
        # Ereignis auslösen
        self.trigger_event('memory_store', {
            'content': content,
            'memory_type': memory_type,
            'result': result,
            'source_module': source_module
        })
        
        return result
    
    def retrieve(self, 
                query: Union[str, Dict[str, Any]], 
                memory_type: str = 'all',
                limit: int = 10,
                min_similarity: float = 0.6,
                source_module: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Ruft Inhalte aus dem Gedächtnismodul ab.
        
        Args:
            query: Suchanfrage (Text oder strukturierte Abfrage)
            memory_type: Zielgedächtnissystem ('semantic', 'episodic', 'working' oder 'all')
            limit: Maximale Anzahl von Ergebnissen
            min_similarity: Minimale Ähnlichkeit für semantische Suche
            source_module: Name des Quellmoduls
            
        Returns:
            Dictionary mit Ergebnissen aus den abgefragten Speichersystemen
        """
        # An Memory Core delegieren
        results = self.memory_core.retrieve(
            query=query,
            memory_type=memory_type,
            limit=limit,
            min_similarity=min_similarity
        )
        
        # Ereignis auslösen
        self.trigger_event('memory_retrieve', {
            'query': query,
            'memory_type': memory_type,
            'result_count': {k: len(v) for k, v in results.items()},
            'source_module': source_module
        })
        
        return results
    
    def retrieve_by_id(self, 
                      entry_id: str, 
                      memory_type: str = 'all',
                      source_module: Optional[str] = None) -> Dict[str, Any]:
        """
        Ruft einen Eintrag anhand seiner ID aus dem Gedächtnismodul ab.
        
        Args:
            entry_id: ID des abzurufenden Eintrags
            memory_type: Zielgedächtnissystem ('semantic', 'episodic', 'working' oder 'all')
            source_module: Name des Quellmoduls
            
        Returns:
            Dictionary mit dem Eintrag aus den abgefragten Speichersystemen
        """
        # An Memory Core delegieren
        result = self.memory_core.retrieve_by_id(
            entry_id=entry_id,
            memory_type=memory_type
        )
        
        # Ereignis auslösen
        self.trigger_event('memory_retrieve_by_id', {
            'entry_id': entry_id,
            'memory_type': memory_type,
            'found': bool(result),
            'source_module': source_module
        })
        
        return result
    
    def retrieve_by_tags(self, 
                        tags: List[str], 
                        memory_type: str = 'all',
                        match_all: bool = False,
                        limit: int = 10,
                        source_module: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Ruft Einträge anhand von Schlagwörtern aus dem Gedächtnismodul ab.
        
        Args:
            tags: Liste von Schlagwörtern
            memory_type: Zielgedächtnissystem ('semantic', 'episodic', 'working' oder 'all')
            match_all: True, wenn alle Schlagwörter übereinstimmen müssen
            limit: Maximale Anzahl von Ergebnissen
            source_module: Name des Quellmoduls
            
        Returns:
            Dictionary mit Ergebnissen aus den abgefragten Speichersystemen
        """
        # An Memory Core delegieren
        results = self.memory_core.retrieve_by_tags(
            tags=tags,
            memory_type=memory_type,
            match_all=match_all,
            limit=limit
        )
        
        # Ereignis auslösen
        self.trigger_event('memory_retrieve_by_tags', {
            'tags': tags,
            'memory_type': memory_type,
            'match_all': match_all,
            'result_count': {k: len(v) for k, v in results.items()},
            'source_module': source_module
        })
        
        return results
    
    def retrieve_by_time_range(self, 
                              start_time: float, 
                              end_time: float,
                              limit: int = 10,
                              source_module: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Ruft Einträge aus dem episodischen Speicher anhand eines Zeitbereichs ab.
        
        Args:
            start_time: Startzeit als Unix-Timestamp
            end_time: Endzeit als Unix-Timestamp
            limit: Maximale Anzahl von Ergebnissen
            source_module: Name des Quellmoduls
            
        Returns:
            Liste mit Einträgen aus dem episodischen Speicher
        """
        # An Memory Core delegieren
        results = self.memory_core.retrieve_by_time_range(
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        # Ereignis auslösen
        self.trigger_event('memory_retrieve_by_time_range', {
            'start_time': start_time,
            'end_time': end_time,
            'result_count': len(results),
            'source_module': source_module
        })
        
        return results
    
    def update(self, 
              entry_id: str, 
              content: Any = None,
              tags: List[str] = None,
              ttl: Optional[int] = None,
              importance: Optional[float] = None,
              context: Dict[str, Any] = None,
              memory_type: str = 'all',
              source_module: Optional[str] = None) -> Dict[str, bool]:
        """
        Aktualisiert einen Eintrag im Gedächtnismodul.
        
        Args:
            entry_id: ID des zu aktualisierenden Eintrags
            content: Neuer Inhalt (None = unverändert)
            tags: Neue Schlagwörter (None = unverändert)
            ttl: Neue Time-to-Live (None = unverändert)
            importance: Neue Wichtigkeit (None = unverändert)
            context: Neuer Kontext (None = unverändert)
            memory_type: Zielgedächtnissystem ('semantic', 'episodic', 'working' oder 'all')
            source_module: Name des Quellmoduls
            
        Returns:
            Dictionary mit Erfolgs-Flags für jedes aktualisierte Speichersystem
        """
        # Kontext erweitern, falls Quellmodul angegeben
        if source_module and context is not None:
            context["source_module"] = source_module
        
        # An Memory Core delegieren
        results = self.memory_core.update(
            entry_id=entry_id,
            content=content,
            tags=tags,
            ttl=ttl,
            importance=importance,
            context=context,
            memory_type=memory_type
        )
        
        # Ereignis auslösen
        self.trigger_event('memory_update', {
            'entry_id': entry_id,
            'memory_type': memory_type,
            'results': results,
            'source_module': source_module
        })
        
        return results
    
    def delete(self, 
              entry_id: str, 
              memory_type: str = 'all',
              source_module: Optional[str] = None) -> Dict[str, bool]:
        """
        Löscht einen Eintrag aus dem Gedächtnismodul.
        
        Args:
            entry_id: ID des zu löschenden Eintrags
            memory_type: Zielgedächtnissystem ('semantic', 'episodic', 'working' oder 'all')
            source_module: Name des Quellmoduls
            
        Returns:
            Dictionary mit Erfolgs-Flags für jedes aktualisierte Speichersystem
        """
        # An Memory Core delegieren
        results = self.memory_core.delete(
            entry_id=entry_id,
            memory_type=memory_type
        )
        
        # Ereignis auslösen
        self.trigger_event('memory_delete', {
            'entry_id': entry_id,
            'memory_type': memory_type,
            'results': results,
            'source_module': source_module
        })
        
        return results
    
    def link_entries(self, 
                    source_id: str, 
                    target_id: str, 
                    link_type: str = 'related',
                    source_module: Optional[str] = None) -> bool:
        """
        Verknüpft zwei Einträge im Gedächtnismodul miteinander.
        
        Args:
            source_id: ID des Quelleintrags
            target_id: ID des Zieleintrags
            link_type: Art der Verknüpfung
            source_module: Name des Quellmoduls
            
        Returns:
            True, wenn die Verknüpfung erfolgreich war
        """
        # An Memory Core delegieren
        result = self.memory_core.link_entries(
            source_id=source_id,
            target_id=target_id,
            link_type=link_type
        )
        
        # Ereignis auslösen
        self.trigger_event('memory_link_entries', {
            'source_id': source_id,
            'target_id': target_id,
            'link_type': link_type,
            'result': result,
            'source_module': source_module
        })
        
        return result
    
    def get_linked_entries(self, 
                          entry_id: str, 
                          link_type: str = None,
                          source_module: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Ruft alle mit einem Eintrag verknüpften Einträge aus dem Gedächtnismodul ab.
        
        Args:
            entry_id: ID des Eintrags
            link_type: Optional, Art der Verknüpfung
            source_module: Name des Quellmoduls
            
        Returns:
            Dictionary mit verknüpften Einträgen aus allen Speichersystemen
        """
        # An Memory Core delegieren
        results = self.memory_core.get_linked_entries(
            entry_id=entry_id,
            link_type=link_type
        )
        
        # Ereignis auslösen
        self.trigger_event('memory_get_linked_entries', {
            'entry_id': entry_id,
            'link_type': link_type,
            'result_count': {k: len(v) for k, v in results.items()},
            'source_module': source_module
        })
        
        return results
    
    def cleanup(self, 
               memory_type: str = 'all',
               source_module: Optional[str] = None) -> Dict[str, int]:
        """
        Bereinigt abgelaufene Einträge im Gedächtnismodul.
        
        Args:
            memory_type: Zielgedächtnissystem ('semantic', 'episodic', 'working' oder 'all')
            source_module: Name des Quellmoduls
            
        Returns:
            Dictionary mit der Anzahl der bereinigten Einträge pro Speichersystem
        """
        # An Memory Core delegieren
        results = self.memory_core.cleanup(
            memory_type=memory_type
        )
        
        # Ereignis auslösen
        self.trigger_event('memory_cleanup', {
            'memory_type': memory_type,
            'results': results,
            'source_module': source_module
        })
        
        return results
    
    def get_stats(self, 
                 memory_type: str = 'all',
                 source_module: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Gibt Statistiken über das Gedächtnismodul zurück.
        
        Args:
            memory_type: Zielgedächtnissystem ('semantic', 'episodic', 'working' oder 'all')
            source_module: Name des Quellmoduls
            
        Returns:
            Dictionary mit Statistiken für jedes Speichersystem
        """
        # An Memory Core delegieren
        results = self.memory_core.get_stats(
            memory_type=memory_type
        )
        
        # Ereignis auslösen
        self.trigger_event('memory_get_stats', {
            'memory_type': memory_type,
            'results': results,
            'source_module': source_module
        })
        
        return results
    
    def export_to_json(self, 
                      memory_type: str = 'all', 
                      file_path: Optional[str] = None,
                      source_module: Optional[str] = None) -> Optional[str]:
        """
        Exportiert den Inhalt des Gedächtnismoduls als JSON.
        
        Args:
            memory_type: Zielgedächtnissystem ('semantic', 'episodic', 'working' oder 'all')
            file_path: Optional, Pfad für die Ausgabedatei
            source_module: Name des Quellmoduls
            
        Returns:
            JSON-String oder None, wenn in eine Datei exportiert wurde
        """
        # An Memory Core delegieren
        result = self.memory_core.export_to_json(
            memory_type=memory_type,
            file_path=file_path
        )
        
        # Ereignis auslösen
        self.trigger_event('memory_export_to_json', {
            'memory_type': memory_type,
            'file_path': file_path,
            'source_module': source_module
        })
        
        return result
    
    def import_from_json(self, 
                        json_data: Union[str, Dict], 
                        memory_type: str = 'all',
                        source_module: Optional[str] = None) -> Dict[str, int]:
        """
        Importiert Daten aus JSON in das Gedächtnismodul.
        
        Args:
            json_data: JSON-String oder Dictionary
            memory_type: Zielgedächtnissystem ('semantic', 'episodic', 'working' oder 'all')
            source_module: Name des Quellmoduls
            
        Returns:
            Dictionary mit der Anzahl der importierten Einträge pro Speichersystem
        """
        # An Memory Core delegieren
        results = self.memory_core.import_from_json(
            json_data=json_data,
            memory_type=memory_type
        )
        
        # Ereignis auslösen
        self.trigger_event('memory_import_from_json', {
            'memory_type': memory_type,
            'results': results,
            'source_module': source_module
        })
        
        return results
    
    def load_module(self, module_name: str) -> Any:
        """
        Lädt ein VXOR-Modul dynamisch.
        
        Args:
            module_name: Name des zu ladenden Moduls
            
        Returns:
            Instanz des geladenen Moduls
            
        Raises:
            ImportError: Wenn das Modul nicht geladen werden kann
        """
        try:
            # Versuche, das Modul zu importieren
            module_path = f"vxor_{module_name.lower()}"
            module = importlib.import_module(module_path)
            
            # Versuche, die Hauptklasse zu finden
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and name.lower() == module_name.lower():
                    # Instanz erstellen
                    instance = obj()
                    
                    # Modul registrieren
                    self.register_module(module_name, instance)
                    
                    self.logger.info(f"Modul {module_name} geladen und registriert")
                    return instance
            
            raise ImportError(f"Keine passende Klasse im Modul {module_path} gefunden")
        except ImportError as e:
            self.logger.error(f"Fehler beim Laden des Moduls {module_name}: {e}")
            raise
    
    def connect_to_q_logik(self) -> bool:
        """
        Stellt eine Verbindung zum Q-LOGIK-Modul her.
        
        Returns:
            True, wenn die Verbindung erfolgreich hergestellt wurde
        """
        try:
            # Q-LOGIK-Modul laden
            q_logik = self.load_module("Q-LOGIK")
            
            # Event-Handler für Gedächtnisabfragen registrieren
            q_logik.register_memory_provider(self)
            
            self.logger.info("Verbindung zu Q-LOGIK hergestellt")
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Verbinden mit Q-LOGIK: {e}")
            return False
    
    def connect_to_m_code(self) -> bool:
        """
        Stellt eine Verbindung zum M-CODE-Modul her.
        
        Returns:
            True, wenn die Verbindung erfolgreich hergestellt wurde
        """
        try:
            # M-CODE-Modul laden
            m_code = self.load_module("M-CODE")
            
            # Event-Handler für Code-Speicherung registrieren
            m_code.register_memory_provider(self)
            
            self.logger.info("Verbindung zu M-CODE hergestellt")
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Verbinden mit M-CODE: {e}")
            return False
    
    def connect_to_m_lingua(self) -> bool:
        """
        Stellt eine Verbindung zum M-LINGUA-Modul her.
        
        Returns:
            True, wenn die Verbindung erfolgreich hergestellt wurde
        """
        try:
            # M-LINGUA-Modul laden
            m_lingua = self.load_module("M-LINGUA")
            
            # Event-Handler für Sprachkonstrukte registrieren
            m_lingua.register_memory_provider(self)
            
            self.logger.info("Verbindung zu M-LINGUA hergestellt")
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Verbinden mit M-LINGUA: {e}")
            return False
    
    def connect_to_t_mathematics(self) -> bool:
        """
        Stellt eine Verbindung zum T-MATHEMATICS-Modul her.
        
        Returns:
            True, wenn die Verbindung erfolgreich hergestellt wurde
        """
        try:
            # T-MATHEMATICS-Modul laden
            t_mathematics = self.load_module("T-MATHEMATICS")
            
            # Event-Handler für mathematische Konzepte registrieren
            t_mathematics.register_memory_provider(self)
            
            self.logger.info("Verbindung zu T-MATHEMATICS hergestellt")
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Verbinden mit T-MATHEMATICS: {e}")
            return False
    
    def connect_to_all_modules(self) -> Dict[str, bool]:
        """
        Stellt Verbindungen zu allen bekannten VXOR-Modulen her.
        
        Returns:
            Dictionary mit Verbindungsstatus für jedes Modul
        """
        results = {}
        
        # Verbindung zu Q-LOGIK herstellen
        results["Q-LOGIK"] = self.connect_to_q_logik()
        
        # Verbindung zu M-CODE herstellen
        results["M-CODE"] = self.connect_to_m_code()
        
        # Verbindung zu M-LINGUA herstellen
        results["M-LINGUA"] = self.connect_to_m_lingua()
        
        # Verbindung zu T-MATHEMATICS herstellen
        results["T-MATHEMATICS"] = self.connect_to_t_mathematics()
        
        self.logger.info(f"Verbindungen zu VXOR-Modulen hergestellt: {results}")
        
        return results


# Wenn direkt ausgeführt, führe einen einfachen Test durch
if __name__ == "__main__":
    # VXOR-Bridge initialisieren
    bridge = VXORBridge()
    
    # Manifest generieren und speichern
    manifest_path = bridge.save_manifest("/vXor_Modules/VX-MEMEX/vx_memex_manifest.json")
    print(f"Manifest gespeichert in: {manifest_path}")
    
    # Testdaten speichern
    entry_id = bridge.store(
        content="Dies ist ein Testinhalt für die VXOR-Bridge.",
        memory_type='all',
        tags=['test', 'vxor-bridge'],
        importance=0.8,
        context={'test_run': True},
        source_module='test'
    )
    
    print(f"Eintrag gespeichert: {entry_id}")
    
    # Testdaten abrufen
    results = bridge.retrieve(
        query="Testinhalt für VXOR",
        memory_type='all',
        source_module='test'
    )
    
    print(f"Suchergebnisse: {results}")
    
    # Statistiken anzeigen
    stats = bridge.get_stats()
    
    print(f"Speicherstatistiken: {stats}")
    
    # Versuche, Verbindungen zu anderen Modulen herzustellen
    try:
        connection_results = bridge.connect_to_all_modules()
        print(f"Verbindungsergebnisse: {connection_results}")
    except ImportError:
        print("Module konnten nicht geladen werden (erwartet in Testumgebung)")
