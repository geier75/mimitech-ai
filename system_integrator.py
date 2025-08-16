"""
System Integrator für Omega-Kern

Dieses Modul implementiert den System Integrator für das Omega-Kern-Modul.
Es ermöglicht die Integration und Kommunikation zwischen verschiedenen MISO-Komponenten.

Version: 1.0.0
"""

import logging
import threading
import queue
import time
import uuid
import json
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Tuple, Set, Union

# Logger konfigurieren
logger = logging.getLogger("miso.omega.system_integrator")

class ComponentType(Enum):
    """Typen von Komponenten im MISO-System"""
    CORE = "core"           # Kernkomponenten
    ENGINE = "engine"       # Engines
    FRAMEWORK = "framework" # Frameworks
    MODULE = "module"       # Module
    INTERFACE = "interface" # Schnittstellen
    UTILITY = "utility"     # Hilfsprogramme

class MessageType(Enum):
    """Typen von Nachrichten im MISO-System"""
    COMMAND = "command"     # Befehle
    EVENT = "event"         # Ereignisse
    DATA = "data"           # Daten
    QUERY = "query"         # Abfragen
    RESPONSE = "response"   # Antworten
    ERROR = "error"         # Fehler
    LOG = "log"             # Logs

class MessagePriority(Enum):
    """Prioritäten von Nachrichten im MISO-System"""
    LOW = 0       # Niedrige Priorität
    NORMAL = 1    # Normale Priorität
    HIGH = 2      # Hohe Priorität
    CRITICAL = 3  # Kritische Priorität

class SystemMessage:
    """Eine Nachricht im MISO-System"""
    
    def __init__(self, sender: str, receiver: str, message_type: MessageType, 
                 content: Any, priority: MessagePriority = MessagePriority.NORMAL,
                 message_id: Optional[str] = None, correlation_id: Optional[str] = None):
        """
        Initialisiert eine SystemMessage.
        
        Args:
            sender: Absender der Nachricht
            receiver: Empfänger der Nachricht
            message_type: Typ der Nachricht
            content: Inhalt der Nachricht
            priority: Priorität der Nachricht
            message_id: ID der Nachricht (wird generiert, wenn nicht angegeben)
            correlation_id: Korrelations-ID für zusammenhängende Nachrichten
        """
        self.sender = sender
        self.receiver = receiver
        self.message_type = message_type
        self.content = content
        self.priority = priority
        self.message_id = message_id or str(uuid.uuid4())
        self.correlation_id = correlation_id
        self.timestamp = time.time()
    
    def __str__(self) -> str:
        """String-Repräsentation der Nachricht"""
        return f"SystemMessage(id={self.message_id}, type={self.message_type.value}, sender={self.sender}, receiver={self.receiver})"
    
    def to_dict(self) -> Dict:
        """Konvertiert die Nachricht in ein Dictionary"""
        return {
            "message_id": self.message_id,
            "correlation_id": self.correlation_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "content": self.content,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SystemMessage':
        """
        Erstellt eine SystemMessage aus einem Dictionary.
        
        Args:
            data: Dictionary mit Nachrichtendaten
            
        Returns:
            SystemMessage: Die erstellte Nachricht
        """
        return cls(
            sender=data["sender"],
            receiver=data["receiver"],
            message_type=MessageType(data["message_type"]),
            content=data["content"],
            priority=MessagePriority(data["priority"]),
            message_id=data["message_id"],
            correlation_id=data.get("correlation_id")
        )

class ComponentInfo:
    """Informationen über eine Komponente im MISO-System"""
    
    def __init__(self, component_id: str, component_type: ComponentType, 
                 name: str, version: str, description: str = "",
                 dependencies: List[str] = None):
        """
        Initialisiert ComponentInfo.
        
        Args:
            component_id: ID der Komponente
            component_type: Typ der Komponente
            name: Name der Komponente
            version: Version der Komponente
            description: Beschreibung der Komponente
            dependencies: Abhängigkeiten der Komponente
        """
        self.component_id = component_id
        self.component_type = component_type
        self.name = name
        self.version = version
        self.description = description
        self.dependencies = dependencies or []
        self.status = "inactive"
        self.last_heartbeat = 0.0
        self.capabilities = []
    
    def __str__(self) -> str:
        """String-Repräsentation der Komponenteninformation"""
        return f"ComponentInfo(id={self.component_id}, name={self.name}, type={self.component_type.value}, status={self.status})"
    
    def to_dict(self) -> Dict:
        """Konvertiert die Komponenteninformation in ein Dictionary"""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "dependencies": self.dependencies,
            "status": self.status,
            "last_heartbeat": self.last_heartbeat,
            "capabilities": self.capabilities
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ComponentInfo':
        """
        Erstellt ComponentInfo aus einem Dictionary.
        
        Args:
            data: Dictionary mit Komponenteninformationen
            
        Returns:
            ComponentInfo: Die erstellte Komponenteninformation
        """
        info = cls(
            component_id=data["component_id"],
            component_type=ComponentType(data["component_type"]),
            name=data["name"],
            version=data["version"],
            description=data.get("description", ""),
            dependencies=data.get("dependencies", [])
        )
        info.status = data.get("status", "inactive")
        info.last_heartbeat = data.get("last_heartbeat", 0.0)
        info.capabilities = data.get("capabilities", [])
        return info

class SystemIntegrator:
    """
    System Integrator für das Omega-Kern-Modul.
    Ermöglicht die Integration und Kommunikation zwischen verschiedenen MISO-Komponenten.
    """
    
    def __init__(self):
        """Initialisiert den SystemIntegrator"""
        self.logger = logging.getLogger("miso.omega.system_integrator")
        self.components: Dict[str, ComponentInfo] = {}
        self.message_handlers: Dict[str, Dict[MessageType, List[Callable]]] = {}
        self.message_queue = queue.PriorityQueue()
        self.running = False
        self.message_processor_thread = None
        self.heartbeat_thread = None
        self.lock = threading.Lock()
        self.logger.info("SystemIntegrator initialisiert")
    
    def register_component(self, component_info: ComponentInfo) -> bool:
        """
        Registriert eine Komponente im System.
        
        Args:
            component_info: Informationen über die Komponente
            
        Returns:
            bool: True, wenn die Registrierung erfolgreich war, sonst False
        """
        with self.lock:
            if component_info.component_id in self.components:
                self.logger.warning(f"Komponente mit ID {component_info.component_id} ist bereits registriert")
                return False
            
            # Prüfe Abhängigkeiten
            for dependency in component_info.dependencies:
                if dependency not in self.components:
                    self.logger.warning(f"Abhängigkeit {dependency} für Komponente {component_info.component_id} nicht gefunden")
                    return False
            
            # Registriere die Komponente
            self.components[component_info.component_id] = component_info
            self.message_handlers[component_info.component_id] = {
                message_type: [] for message_type in MessageType
            }
            
            # Setze Status auf aktiv und aktualisiere Heartbeat
            component_info.status = "active"
            component_info.last_heartbeat = time.time()
            
            self.logger.info(f"Komponente {component_info} erfolgreich registriert")
            
            # Benachrichtige andere Komponenten über die Registrierung
            self.broadcast_message(
                sender="system_integrator",
                message_type=MessageType.EVENT,
                content={
                    "event_type": "component_registered",
                    "component_id": component_info.component_id,
                    "component_info": component_info.to_dict()
                }
            )
            
            return True
    
    def unregister_component(self, component_id: str) -> bool:
        """
        Meldet eine Komponente vom System ab.
        
        Args:
            component_id: ID der Komponente
            
        Returns:
            bool: True, wenn die Abmeldung erfolgreich war, sonst False
        """
        with self.lock:
            if component_id not in self.components:
                self.logger.warning(f"Komponente mit ID {component_id} ist nicht registriert")
                return False
            
            # Entferne die Komponente
            component_info = self.components.pop(component_id)
            self.message_handlers.pop(component_id)
            
            self.logger.info(f"Komponente {component_info} erfolgreich abgemeldet")
            
            # Benachrichtige andere Komponenten über die Abmeldung
            self.broadcast_message(
                sender="system_integrator",
                message_type=MessageType.EVENT,
                content={
                    "event_type": "component_unregistered",
                    "component_id": component_id
                }
            )
            
            return True
    
    def get_component_info(self, component_id: str) -> Optional[ComponentInfo]:
        """
        Gibt Informationen über eine Komponente zurück.
        
        Args:
            component_id: ID der Komponente
            
        Returns:
            Optional[ComponentInfo]: Informationen über die Komponente oder None, wenn nicht gefunden
        """
        with self.lock:
            return self.components.get(component_id)
    
    def get_all_components(self) -> Dict[str, ComponentInfo]:
        """
        Gibt Informationen über alle registrierten Komponenten zurück.
        
        Returns:
            Dict[str, ComponentInfo]: Ein Dictionary mit allen Komponenten
        """
        with self.lock:
            return self.components.copy()
    
    def register_message_handler(self, component_id: str, message_type: MessageType, 
                               handler: Callable[[SystemMessage], None]) -> bool:
        """
        Registriert einen Nachrichtenhandler für eine Komponente.
        
        Args:
            component_id: ID der Komponente
            message_type: Typ der Nachrichten, die behandelt werden sollen
            handler: Funktion, die Nachrichten behandelt
            
        Returns:
            bool: True, wenn die Registrierung erfolgreich war, sonst False
        """
        with self.lock:
            if component_id not in self.components:
                self.logger.warning(f"Komponente mit ID {component_id} ist nicht registriert")
                return False
            
            self.message_handlers[component_id][message_type].append(handler)
            self.logger.debug(f"Nachrichtenhandler für {component_id} und {message_type.value} registriert")
            return True
    
    def unregister_message_handler(self, component_id: str, message_type: MessageType, 
                                 handler: Callable[[SystemMessage], None]) -> bool:
        """
        Meldet einen Nachrichtenhandler für eine Komponente ab.
        
        Args:
            component_id: ID der Komponente
            message_type: Typ der Nachrichten
            handler: Der zu entfernende Handler
            
        Returns:
            bool: True, wenn die Abmeldung erfolgreich war, sonst False
        """
        with self.lock:
            if component_id not in self.components:
                self.logger.warning(f"Komponente mit ID {component_id} ist nicht registriert")
                return False
            
            if handler in self.message_handlers[component_id][message_type]:
                self.message_handlers[component_id][message_type].remove(handler)
                self.logger.debug(f"Nachrichtenhandler für {component_id} und {message_type.value} abgemeldet")
                return True
            else:
                self.logger.warning(f"Nachrichtenhandler für {component_id} und {message_type.value} nicht gefunden")
                return False
    
    def send_message(self, message: SystemMessage) -> bool:
        """
        Sendet eine Nachricht an eine Komponente.
        
        Args:
            message: Die zu sendende Nachricht
            
        Returns:
            bool: True, wenn die Nachricht gesendet wurde, sonst False
        """
        if not self.running:
            self.logger.warning("SystemIntegrator läuft nicht, Nachricht wird nicht gesendet")
            return False
        
        # Prüfe, ob Empfänger existiert
        if message.receiver != "broadcast" and message.receiver not in self.components:
            self.logger.warning(f"Empfänger {message.receiver} ist nicht registriert")
            return False
        
        # Füge die Nachricht zur Warteschlange hinzu
        self.message_queue.put((message.priority.value * -1, time.time(), message))
        self.logger.debug(f"Nachricht {message} zur Warteschlange hinzugefügt")
        return True
    
    def broadcast_message(self, sender: str, message_type: MessageType, 
                        content: Any, priority: MessagePriority = MessagePriority.NORMAL) -> bool:
        """
        Sendet eine Nachricht an alle Komponenten.
        
        Args:
            sender: Absender der Nachricht
            message_type: Typ der Nachricht
            content: Inhalt der Nachricht
            priority: Priorität der Nachricht
            
        Returns:
            bool: True, wenn die Nachricht gesendet wurde, sonst False
        """
        message = SystemMessage(
            sender=sender,
            receiver="broadcast",
            message_type=message_type,
            content=content,
            priority=priority
        )
        
        return self.send_message(message)
    
    def _process_messages(self):
        """Verarbeitet Nachrichten aus der Warteschlange"""
        self.logger.info("Nachrichtenverarbeitung gestartet")
        
        while self.running:
            try:
                # Hole die nächste Nachricht aus der Warteschlange
                _, _, message = self.message_queue.get(timeout=1.0)
                
                # Verarbeite die Nachricht
                if message.receiver == "broadcast":
                    # Broadcast-Nachricht an alle Komponenten senden
                    for component_id, handlers in self.message_handlers.items():
                        for handler in handlers[message.message_type]:
                            try:
                                handler(message)
                            except Exception as e:
                                self.logger.error(f"Fehler bei der Verarbeitung von Broadcast-Nachricht {message} durch {component_id}: {e}")
                else:
                    # Nachricht an einen bestimmten Empfänger senden
                    handlers = self.message_handlers.get(message.receiver, {}).get(message.message_type, [])
                    
                    for handler in handlers:
                        try:
                            handler(message)
                        except Exception as e:
                            self.logger.error(f"Fehler bei der Verarbeitung von Nachricht {message}: {e}")
                
                self.message_queue.task_done()
            
            except queue.Empty:
                # Timeout, weiter warten
                pass
            
            except Exception as e:
                self.logger.error(f"Fehler in der Nachrichtenverarbeitung: {e}")
        
        self.logger.info("Nachrichtenverarbeitung beendet")
    
    def _check_heartbeats(self):
        """Überprüft die Heartbeats der Komponenten"""
        self.logger.info("Heartbeat-Überprüfung gestartet")
        
        while self.running:
            try:
                current_time = time.time()
                
                with self.lock:
                    for component_id, component_info in list(self.components.items()):
                        # Prüfe, ob die Komponente noch aktiv ist
                        if component_info.status == "active" and current_time - component_info.last_heartbeat > 30.0:
                            # Komponente hat keinen Heartbeat mehr gesendet
                            component_info.status = "inactive"
                            self.logger.warning(f"Komponente {component_info} ist inaktiv (kein Heartbeat)")
                            
                            # Benachrichtige andere Komponenten
                            self.broadcast_message(
                                sender="system_integrator",
                                message_type=MessageType.EVENT,
                                content={
                                    "event_type": "component_inactive",
                                    "component_id": component_id
                                }
                            )
                
                # Warte 10 Sekunden bis zur nächsten Überprüfung
                time.sleep(10.0)
            
            except Exception as e:
                self.logger.error(f"Fehler bei der Heartbeat-Überprüfung: {e}")
        
        self.logger.info("Heartbeat-Überprüfung beendet")
    
    def update_heartbeat(self, component_id: str) -> bool:
        """
        Aktualisiert den Heartbeat einer Komponente.
        
        Args:
            component_id: ID der Komponente
            
        Returns:
            bool: True, wenn der Heartbeat aktualisiert wurde, sonst False
        """
        with self.lock:
            if component_id not in self.components:
                self.logger.warning(f"Komponente mit ID {component_id} ist nicht registriert")
                return False
            
            component_info = self.components[component_id]
            component_info.last_heartbeat = time.time()
            
            # Wenn die Komponente inaktiv war, reaktiviere sie
            if component_info.status == "inactive":
                component_info.status = "active"
                self.logger.info(f"Komponente {component_info} ist wieder aktiv")
                
                # Benachrichtige andere Komponenten
                self.broadcast_message(
                    sender="system_integrator",
                    message_type=MessageType.EVENT,
                    content={
                        "event_type": "component_active",
                        "component_id": component_id
                    }
                )
            
            return True
    
    def start(self) -> bool:
        """
        Startet den SystemIntegrator.
        
        Returns:
            bool: True, wenn der Start erfolgreich war, sonst False
        """
        if self.running:
            self.logger.warning("SystemIntegrator läuft bereits")
            return False
        
        self.running = True
        
        # Starte den Nachrichtenprozessor
        self.message_processor_thread = threading.Thread(
            target=self._process_messages,
            name="MessageProcessor"
        )
        self.message_processor_thread.daemon = True
        self.message_processor_thread.start()
        
        # Starte die Heartbeat-Überprüfung
        self.heartbeat_thread = threading.Thread(
            target=self._check_heartbeats,
            name="HeartbeatChecker"
        )
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
        
        self.logger.info("SystemIntegrator gestartet")
        return True
    
    def stop(self) -> bool:
        """
        Stoppt den SystemIntegrator.
        
        Returns:
            bool: True, wenn der Stopp erfolgreich war, sonst False
        """
        if not self.running:
            self.logger.warning("SystemIntegrator läuft nicht")
            return False
        
        self.running = False
        
        # Warte auf das Ende der Threads
        if self.message_processor_thread:
            self.message_processor_thread.join(timeout=5.0)
            self.message_processor_thread = None
        
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5.0)
            self.heartbeat_thread = None
        
        self.logger.info("SystemIntegrator gestoppt")
        return True
    
    def get_component_dependencies(self, component_id: str) -> List[ComponentInfo]:
        """
        Gibt die Abhängigkeiten einer Komponente zurück.
        
        Args:
            component_id: ID der Komponente
            
        Returns:
            List[ComponentInfo]: Liste der Abhängigkeiten
        """
        with self.lock:
            if component_id not in self.components:
                self.logger.warning(f"Komponente mit ID {component_id} ist nicht registriert")
                return []
            
            component_info = self.components[component_id]
            dependencies = []
            
            for dependency_id in component_info.dependencies:
                if dependency_id in self.components:
                    dependencies.append(self.components[dependency_id])
            
            return dependencies
    
    def get_component_dependents(self, component_id: str) -> List[ComponentInfo]:
        """
        Gibt die abhängigen Komponenten zurück.
        
        Args:
            component_id: ID der Komponente
            
        Returns:
            List[ComponentInfo]: Liste der abhängigen Komponenten
        """
        with self.lock:
            if component_id not in self.components:
                self.logger.warning(f"Komponente mit ID {component_id} ist nicht registriert")
                return []
            
            dependents = []
            
            for other_id, other_info in self.components.items():
                if component_id in other_info.dependencies:
                    dependents.append(other_info)
            
            return dependents
    
    def check_system_integrity(self) -> Dict:
        """
        Überprüft die Integrität des Systems.
        
        Returns:
            Dict: Ergebnis der Integritätsprüfung
        """
        with self.lock:
            result = {
                "total_components": len(self.components),
                "active_components": 0,
                "inactive_components": 0,
                "missing_dependencies": [],
                "circular_dependencies": [],
                "system_status": "ok"
            }
            
            # Zähle aktive und inaktive Komponenten
            for component_id, component_info in self.components.items():
                if component_info.status == "active":
                    result["active_components"] += 1
                else:
                    result["inactive_components"] += 1
            
            # Prüfe fehlende Abhängigkeiten
            for component_id, component_info in self.components.items():
                for dependency_id in component_info.dependencies:
                    if dependency_id not in self.components:
                        result["missing_dependencies"].append({
                            "component_id": component_id,
                            "missing_dependency": dependency_id
                        })
            
            # Prüfe zirkuläre Abhängigkeiten
            for component_id in self.components:
                visited = set()
                path = []
                self._check_circular_dependencies(component_id, visited, path, result["circular_dependencies"])
            
            # Setze Systemstatus
            if result["inactive_components"] > 0:
                result["system_status"] = "degraded"
            
            if result["missing_dependencies"] or result["circular_dependencies"]:
                result["system_status"] = "error"
            
            return result
    
    def _check_circular_dependencies(self, component_id: str, visited: Set[str], 
                                   path: List[str], circular_dependencies: List[List[str]]):
        """
        Prüft auf zirkuläre Abhängigkeiten.
        
        Args:
            component_id: ID der Komponente
            visited: Bereits besuchte Komponenten
            path: Aktueller Pfad
            circular_dependencies: Liste der gefundenen zirkulären Abhängigkeiten
        """
        if component_id in visited:
            # Zirkuläre Abhängigkeit gefunden
            if component_id in path:
                # Finde den Anfang des Zyklus
                start_index = path.index(component_id)
                cycle = path[start_index:] + [component_id]
                
                # Füge den Zyklus zur Liste hinzu, wenn er noch nicht enthalten ist
                if cycle not in circular_dependencies:
                    circular_dependencies.append(cycle)
            
            return
        
        visited.add(component_id)
        path.append(component_id)
        
        if component_id in self.components:
            for dependency_id in self.components[component_id].dependencies:
                self._check_circular_dependencies(dependency_id, visited, path, circular_dependencies)
        
        path.pop()
    
    def get_system_status(self) -> Dict:
        """
        Gibt den Status des Systems zurück.
        
        Returns:
            Dict: Systemstatus
        """
        with self.lock:
            status = {
                "running": self.running,
                "components": {
                    "total": len(self.components),
                    "active": sum(1 for info in self.components.values() if info.status == "active"),
                    "inactive": sum(1 for info in self.components.values() if info.status == "inactive")
                },
                "message_queue": {
                    "size": self.message_queue.qsize()
                },
                "timestamp": time.time()
            }
            
            return status
    
    def export_system_configuration(self) -> Dict:
        """
        Exportiert die Systemkonfiguration.
        
        Returns:
            Dict: Systemkonfiguration
        """
        with self.lock:
            config = {
                "components": {
                    component_id: component_info.to_dict()
                    for component_id, component_info in self.components.items()
                },
                "timestamp": time.time()
            }
            
            return config
    
    def import_system_configuration(self, config: Dict) -> bool:
        """
        Importiert eine Systemkonfiguration.
        
        Args:
            config: Systemkonfiguration
            
        Returns:
            bool: True, wenn der Import erfolgreich war, sonst False
        """
        try:
            with self.lock:
                # Lösche alle vorhandenen Komponenten
                self.components.clear()
                self.message_handlers.clear()
                
                # Importiere Komponenten
                for component_id, component_data in config["components"].items():
                    component_info = ComponentInfo.from_dict(component_data)
                    self.components[component_id] = component_info
                    self.message_handlers[component_id] = {
                        message_type: [] for message_type in MessageType
                    }
                
                self.logger.info(f"Systemkonfiguration mit {len(self.components)} Komponenten importiert")
                return True
        
        except Exception as e:
            self.logger.error(f"Fehler beim Import der Systemkonfiguration: {e}")
            return False
