#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-MEMEX: Integrationstest
--------------------------
Umfassender Integrationstest für das VX-MEMEX Gedächtnismodul.
Testet die Zusammenarbeit aller Komponenten und simuliert die Integration
mit anderen VXOR-Modulen.

Optimiert für Apple Silicon M4 Max.
"""

import os
import sys
import time
import json
import logging
import random
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple

# Pfad zum Projektverzeichnis hinzufügen
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Module importieren
from memory_core import MemoryCore
from semantic_store import SemanticStore
from episodic_store import EpisodicStore
from working_memory import WorkingMemory
from vxor_bridge import VXORBridge

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/home/ubuntu/vXor_Modules/VX-MEMEX/integration_test.log')
    ]
)

logger = logging.getLogger('vx_memex_integration_test')

class MockVXORModule:
    """
    Mock-Klasse für ein VXOR-Modul zur Simulation der Integration.
    """
    
    def __init__(self, name: str):
        """
        Initialisiert das Mock-Modul.
        
        Args:
            name: Name des Moduls
        """
        self.name = name
        self.memory_provider = None
        self.event_handlers = {}
        self.received_events = []
        
        logger.info(f"Mock-Modul {name} initialisiert")
    
    def register_memory_provider(self, provider: Any) -> bool:
        """
        Registriert einen Gedächtnisanbieter.
        
        Args:
            provider: Gedächtnisanbieter
            
        Returns:
            True, wenn die Registrierung erfolgreich war
        """
        self.memory_provider = provider
        logger.info(f"Gedächtnisanbieter bei {self.name} registriert")
        
        # Event-Handler registrieren
        if hasattr(provider, 'register_event_handler'):
            provider.register_event_handler(
                f'{self.name.lower()}_event',
                self.handle_event,
                self.name
            )
            logger.info(f"Event-Handler bei {self.name} registriert")
        
        return True
    
    def handle_event(self, event_data: Any) -> None:
        """
        Behandelt ein Ereignis.
        
        Args:
            event_data: Ereignisdaten
        """
        self.received_events.append(event_data)
        logger.info(f"{self.name} hat Ereignis empfangen: {event_data}")
    
    def store_data(self, content: Any, tags: List[str] = None) -> Dict[str, str]:
        """
        Speichert Daten im Gedächtnismodul.
        
        Args:
            content: Zu speichernder Inhalt
            tags: Liste von Schlagwörtern
            
        Returns:
            Dictionary mit den IDs der erstellten Einträge
        """
        if self.memory_provider is None:
            logger.warning(f"{self.name} hat keinen Gedächtnisanbieter")
            return {}
        
        result = self.memory_provider.store(
            content=content,
            tags=tags or [self.name.lower()],
            source_module=self.name
        )
        
        logger.info(f"{self.name} hat Daten gespeichert: {result}")
        
        return result
    
    def retrieve_data(self, query: Union[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Ruft Daten aus dem Gedächtnismodul ab.
        
        Args:
            query: Suchanfrage
            
        Returns:
            Dictionary mit Ergebnissen
        """
        if self.memory_provider is None:
            logger.warning(f"{self.name} hat keinen Gedächtnisanbieter")
            return {}
        
        result = self.memory_provider.retrieve(
            query=query,
            source_module=self.name
        )
        
        logger.info(f"{self.name} hat Daten abgerufen: {len(result)} Ergebnisse")
        
        return result
    
    def trigger_module_event(self, event_data: Any) -> None:
        """
        Löst ein modulspezifisches Ereignis aus.
        
        Args:
            event_data: Ereignisdaten
        """
        if self.memory_provider is None or not hasattr(self.memory_provider, 'trigger_event'):
            logger.warning(f"{self.name} kann kein Ereignis auslösen")
            return
        
        event_type = f'{self.name.lower()}_event'
        self.memory_provider.trigger_event(
            event_type=event_type,
            event_data=event_data
        )
        
        logger.info(f"{self.name} hat Ereignis ausgelöst: {event_type}")


def test_vxor_bridge_initialization():
    """Testet die Initialisierung der VXOR-Bridge."""
    logger.info("=== Test: VXOR-Bridge-Initialisierung ===")
    
    # VXOR-Bridge initialisieren
    bridge = VXORBridge()
    
    # Prüfen, ob die Bridge initialisiert wurde
    assert bridge is not None
    assert bridge.memory_core is not None
    
    # Manifest generieren
    manifest = bridge.manifest
    
    # Prüfen, ob das Manifest generiert wurde
    assert manifest is not None
    assert 'module_name' in manifest
    assert manifest['module_name'] == 'VX-MEMEX'
    
    logger.info("VXOR-Bridge erfolgreich initialisiert")
    logger.info(f"Manifest enthält {len(manifest['components'])} Komponenten")
    
    return bridge

def test_module_registration(bridge: VXORBridge):
    """
    Testet die Registrierung von Modulen.
    
    Args:
        bridge: VXOR-Bridge-Instanz
    """
    logger.info("=== Test: Modulregistrierung ===")
    
    # Mock-Module erstellen
    q_logik = MockVXORModule("Q-LOGIK")
    m_code = MockVXORModule("M-CODE")
    m_lingua = MockVXORModule("M-LINGUA")
    t_mathematics = MockVXORModule("T-MATHEMATICS")
    
    # Module registrieren
    bridge.register_module("Q-LOGIK", q_logik)
    bridge.register_module("M-CODE", m_code)
    bridge.register_module("M-LINGUA", m_lingua)
    bridge.register_module("T-MATHEMATICS", t_mathematics)
    
    # Prüfen, ob die Module registriert wurden
    registered_modules = bridge.get_registered_modules()
    assert len(registered_modules) == 4
    assert "Q-LOGIK" in registered_modules
    assert "M-CODE" in registered_modules
    assert "M-LINGUA" in registered_modules
    assert "T-MATHEMATICS" in registered_modules
    
    logger.info("Module erfolgreich registriert")
    
    return {
        "Q-LOGIK": q_logik,
        "M-CODE": m_code,
        "M-LINGUA": m_lingua,
        "T-MATHEMATICS": t_mathematics
    }

def test_memory_provider_registration(bridge: VXORBridge, modules: Dict[str, MockVXORModule]):
    """
    Testet die Registrierung des Gedächtnisanbieters bei den Modulen.
    
    Args:
        bridge: VXOR-Bridge-Instanz
        modules: Dictionary mit Mock-Modulen
    """
    logger.info("=== Test: Gedächtnisanbieter-Registrierung ===")
    
    # Gedächtnisanbieter bei den Modulen registrieren
    for name, module in modules.items():
        module.register_memory_provider(bridge)
    
    # Prüfen, ob der Gedächtnisanbieter registriert wurde
    for name, module in modules.items():
        assert module.memory_provider is not None
        assert module.memory_provider == bridge
    
    logger.info("Gedächtnisanbieter erfolgreich bei allen Modulen registriert")

def test_data_storage_and_retrieval(modules: Dict[str, MockVXORModule]):
    """
    Testet das Speichern und Abrufen von Daten durch die Module.
    
    Args:
        modules: Dictionary mit Mock-Modulen
    """
    logger.info("=== Test: Datenspeicherung und -abruf ===")
    
    # Testdaten für jedes Modul
    test_data = {
        "Q-LOGIK": {
            "content": {
                "type": "logical_rule",
                "rule": "IF temperature > 30 THEN status = 'hot'",
                "confidence": 0.95
            },
            "tags": ["rule", "temperature", "status"]
        },
        "M-CODE": {
            "content": {
                "type": "code_snippet",
                "language": "python",
                "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
            },
            "tags": ["algorithm", "fibonacci", "recursion"]
        },
        "M-LINGUA": {
            "content": {
                "type": "language_construct",
                "language": "de",
                "construct": "Gedächtnismodul",
                "translation": {
                    "en": "memory module",
                    "fr": "module de mémoire"
                }
            },
            "tags": ["translation", "german", "memory"]
        },
        "T-MATHEMATICS": {
            "content": {
                "type": "mathematical_formula",
                "name": "Euler's identity",
                "formula": "e^(i*pi) + 1 = 0",
                "components": ["e", "i", "pi"]
            },
            "tags": ["euler", "identity", "complex numbers"]
        }
    }
    
    # Daten speichern
    stored_ids = {}
    
    for name, module in modules.items():
        data = test_data[name]
        result = module.store_data(data["content"], data["tags"])
        stored_ids[name] = result
        
        # Prüfen, ob die Daten gespeichert wurden
        assert result is not None
        assert "semantic" in result
        assert "episodic" in result
        assert "working" in result
    
    logger.info(f"Daten erfolgreich von allen Modulen gespeichert: {len(stored_ids)} Einträge")
    
    # Daten abrufen
    for name, module in modules.items():
        # Nach Inhalt suchen
        if name == "Q-LOGIK":
            query = "logical rule temperature"
        elif name == "M-CODE":
            query = "fibonacci algorithm"
        elif name == "M-LINGUA":
            query = "Gedächtnismodul translation"
        elif name == "T-MATHEMATICS":
            query = "Euler's identity formula"
        
        results = module.retrieve_data(query)
        
        # Prüfen, ob Ergebnisse gefunden wurden
        assert results is not None
        assert "semantic" in results
        assert len(results["semantic"]) > 0
        
        logger.info(f"{name} hat erfolgreich Daten abgerufen: {len(results['semantic'])} semantische Ergebnisse")
    
    return stored_ids

def test_event_handling(bridge: VXORBridge, modules: Dict[str, MockVXORModule]):
    """
    Testet die Ereignisbehandlung zwischen der Bridge und den Modulen.
    
    Args:
        bridge: VXOR-Bridge-Instanz
        modules: Dictionary mit Mock-Modulen
    """
    logger.info("=== Test: Ereignisbehandlung ===")
    
    # Ereignisse von jedem Modul auslösen
    for name, module in modules.items():
        event_data = {
            "source": name,
            "timestamp": time.time(),
            "action": "test_event",
            "data": {
                "test_id": str(uuid.uuid4()),
                "message": f"Testevent von {name}"
            }
        }
        
        module.trigger_module_event(event_data)
    
    # Kurz warten, damit die Ereignisse verarbeitet werden können
    time.sleep(0.1)
    
    # Prüfen, ob die Ereignisse im episodischen Speicher protokolliert wurden
    for name, module in modules.items():
        # Nach Ereignissen suchen
        results = bridge.retrieve(
            query=f"Testevent von {name}",
            memory_type="episodic",
            source_module="integration_test"
        )
        
        # Prüfen, ob Ergebnisse gefunden wurden
        assert "episodic" in results
        assert len(results["episodic"]) > 0
        
        logger.info(f"Ereignis von {name} erfolgreich im episodischen Speicher protokolliert")
    
    logger.info("Ereignisbehandlung erfolgreich getestet")

def test_cross_module_integration(bridge: VXORBridge, modules: Dict[str, MockVXORModule], stored_ids: Dict[str, Dict[str, str]]):
    """
    Testet die modulübergreifende Integration durch Verknüpfung von Einträgen.
    
    Args:
        bridge: VXOR-Bridge-Instanz
        modules: Dictionary mit Mock-Modulen
        stored_ids: Dictionary mit gespeicherten Eintrags-IDs
    """
    logger.info("=== Test: Modulübergreifende Integration ===")
    
    # Einträge verknüpfen
    links = []
    
    # Q-LOGIK mit M-CODE verknüpfen
    link1 = bridge.link_entries(
        source_id=stored_ids["Q-LOGIK"]["semantic"],
        target_id=stored_ids["M-CODE"]["semantic"],
        link_type="implements",
        source_module="integration_test"
    )
    links.append(link1)
    
    # M-LINGUA mit T-MATHEMATICS verknüpfen
    link2 = bridge.link_entries(
        source_id=stored_ids["M-LINGUA"]["semantic"],
        target_id=stored_ids["T-MATHEMATICS"]["semantic"],
        link_type="describes",
        source_module="integration_test"
    )
    links.append(link2)
    
    # Prüfen, ob die Verknüpfungen erstellt wurden
    assert all(links)
    
    logger.info(f"Einträge erfolgreich verknüpft: {len(links)} Verknüpfungen")
    
    # Verknüpfte Einträge abrufen
    linked_entries1 = bridge.get_linked_entries(
        entry_id=stored_ids["Q-LOGIK"]["semantic"],
        source_module="integration_test"
    )
    
    linked_entries2 = bridge.get_linked_entries(
        entry_id=stored_ids["M-LINGUA"]["semantic"],
        source_module="integration_test"
    )
    
    # Prüfen, ob die verknüpften Einträge abgerufen wurden
    assert "semantic" in linked_entries1
    assert len(linked_entries1["semantic"]) > 0
    
    assert "semantic" in linked_entries2
    assert len(linked_entries2["semantic"]) > 0
    
    logger.info("Verknüpfte Einträge erfolgreich abgerufen")
    
    # Modulübergreifende Suche
    results = bridge.retrieve(
        query="algorithm formula",
        memory_type="all",
        source_module="integration_test"
    )
    
    # Prüfen, ob Ergebnisse gefunden wurden
    assert "semantic" in results
    assert len(results["semantic"]) > 0
    
    logger.info(f"Modulübergreifende Suche erfolgreich: {len(results['semantic'])} semantische Ergebnisse")

def test_memory_management(bridge: VXORBridge):
    """
    Testet die Speicherverwaltung des Gedächtnismoduls.
    
    Args:
        bridge: VXOR-Bridge-Instanz
    """
    logger.info("=== Test: Speicherverwaltung ===")
    
    # Einträge mit kurzer TTL speichern
    short_lived_entries = []
    
    for i in range(5):
        result = bridge.store(
            content=f"Kurzlebiger Eintrag {i}",
            memory_type="all",
            ttl=2,  # 2 Sekunden
            importance=0.5,
            source_module="integration_test"
        )
        short_lived_entries.append(result)
    
    logger.info(f"{len(short_lived_entries)} kurzlebige Einträge gespeichert")
    
    # Statistiken vor der Bereinigung abrufen
    stats_before = bridge.get_stats(source_module="integration_test")
    
    # Warten, bis die Einträge abgelaufen sind
    logger.info("Warte auf Ablauf der TTL...")
    time.sleep(3)
    
    # Bereinigung durchführen
    cleaned = bridge.cleanup(source_module="integration_test")
    
    # Prüfen, ob Einträge bereinigt wurden
    assert cleaned["semantic"] > 0
    assert cleaned["episodic"] > 0
    assert cleaned["working"] > 0
    
    logger.info(f"Bereinigung erfolgreich: {cleaned} Einträge entfernt")
    
    # Statistiken nach der Bereinigung abrufen
    stats_after = bridge.get_stats(source_module="integration_test")
    
    # Prüfen, ob die Anzahl der Einträge reduziert wurde
    assert stats_after["semantic"]["count"] < stats_before["semantic"]["count"]
    assert stats_after["episodic"]["count"] < stats_before["episodic"]["count"]
    assert stats_after["working"]["count"] < stats_before["working"]["count"]
    
    logger.info("Speicherverwaltung erfolgreich getestet")

def test_export_import(bridge: VXORBridge):
    """
    Testet den Export und Import von Daten.
    
    Args:
        bridge: VXOR-Bridge-Instanz
    """
    logger.info("=== Test: Export und Import ===")
    
    # Einige Testdaten speichern
    for i in range(10):
        bridge.store(
            content=f"Export-Test-Eintrag {i}",
            memory_type="all",
            tags=["export", "test", f"tag{i}"],
            importance=random.random(),
            source_module="integration_test"
        )
    
    logger.info("10 Testeinträge für Export gespeichert")
    
    # Statistiken vor dem Export abrufen
    stats_before = bridge.get_stats(source_module="integration_test")
    
    # Daten exportieren
    export_file = "/home/ubuntu/vXor_Modules/VX-MEMEX/export_test.json"
    bridge.export_to_json(file_path=export_file, source_module="integration_test")
    
    logger.info(f"Daten erfolgreich nach {export_file} exportiert")
    
    # Neuen Bridge erstellen
    new_bridge = VXORBridge()
    
    # Daten importieren
    with open(export_file, 'r', encoding='utf-8') as f:
        json_data = f.read()
    
    import_result = new_bridge.import_from_json(json_data, source_module="integration_test")
    
    # Prüfen, ob die Daten importiert wurden
    assert import_result["semantic"] > 0
    assert import_result["episodic"] > 0
    assert import_result["working"] > 0
    
    logger.info(f"Daten erfolgreich importiert: {import_result}")
    
    # Statistiken nach dem Import abrufen
    stats_after = new_bridge.get_stats(source_module="integration_test")
    
    # Prüfen, ob die Anzahl der Einträge übereinstimmt
    assert stats_after["semantic"]["count"] >= stats_before["semantic"]["count"]
    assert stats_after["episodic"]["count"] >= stats_before["episodic"]["count"]
    assert stats_after["working"]["count"] >= stats_before["working"]["count"]
    
    logger.info("Export und Import erfolgreich getestet")
    
    return export_file

def test_manifest_generation(bridge: VXORBridge):
    """
    Testet die Generierung des Manifests.
    
    Args:
        bridge: VXOR-Bridge-Instanz
    """
    logger.info("=== Test: Manifest-Generierung ===")
    
    # Manifest speichern
    manifest_file = "/home/ubuntu/vXor_Modules/VX-MEMEX/vx_memex_manifest.json"
    bridge.save_manifest(manifest_file)
    
    # Prüfen, ob das Manifest gespeichert wurde
    assert os.path.exists(manifest_file)
    
    # Manifest laden
    with open(manifest_file, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    # Prüfen, ob das Manifest die erforderlichen Informationen enthält
    assert manifest["module_name"] == "VX-MEMEX"
    assert "version" in manifest
    assert "components" in manifest
    assert len(manifest["components"]) == 5  # memory_core, semantic_store, episodic_store, working_memory, vxor_bridge
    
    logger.info(f"Manifest erfolgreich nach {manifest_file} gespeichert")
    
    return manifest_file

def run_integration_test():
    """Führt den Integrationstest durch."""
    logger.info("=== VX-MEMEX Integrationstest ===")
    logger.info(f"Startzeit: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # VXOR-Bridge initialisieren
        bridge = test_vxor_bridge_initialization()
        
        # Module registrieren
        modules = test_module_registration(bridge)
        
        # Gedächtnisanbieter registrieren
        test_memory_provider_registration(bridge, modules)
        
        # Daten speichern und abrufen
        stored_ids = test_data_storage_and_retrieval(modules)
        
        # Ereignisbehandlung testen
        test_event_handling(bridge, modules)
        
        # Modulübergreifende Integration testen
        test_cross_module_integration(bridge, modules, stored_ids)
        
        # Speicherverwaltung testen
        test_memory_management(bridge)
        
        # Export und Import testen
        export_file = test_export_import(bridge)
        
        # Manifest-Generierung testen
        manifest_file = test_manifest_generation(bridge)
        
        logger.info("=== Integrationstest erfolgreich abgeschlossen ===")
        logger.info(f"Endzeit: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return {
            "success": True,
            "export_file": export_file,
            "manifest_file": manifest_file
        }
    
    except Exception as e:
        logger.error(f"Fehler im Integrationstest: {e}", exc_info=True)
        
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    result = run_integration_test()
    
    if result["success"]:
        print("Integrationstest erfolgreich abgeschlossen.")
        print(f"Export-Datei: {result['export_file']}")
        print(f"Manifest-Datei: {result['manifest_file']}")
        sys.exit(0)
    else:
        print(f"Integrationstest fehlgeschlagen: {result['error']}")
        sys.exit(1)
