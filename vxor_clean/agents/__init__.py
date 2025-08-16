#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VXOR Agents Module - Clean Implementation
Bereinigtes Multi-Agent System ohne Marketing-Hype

Copyright (c) 2025 VXOR AI. Alle Rechte vorbehalten.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum

# Setup logging
logger = logging.getLogger("VXOR.Agents")

class AgentStatus(Enum):
    """Agent Status"""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"

@dataclass
class AgentMetrics:
    """Agent Performance Metrics"""
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_response_time: float = 0.0
    uptime: float = 0.0
    last_activity: Optional[float] = None

class BaseAgent(ABC):
    """
    Base Agent Class - Clean Implementation
    
    Basis für alle VXOR-Agenten ohne übertriebene "AGI" Claims.
    Fokus auf echte Funktionalität und Performance.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialisiert den Agent"""
        self.name = name
        self.config = config or {}
        self.status = AgentStatus.INACTIVE
        self.metrics = AgentMetrics()
        self.lock = threading.RLock()
        self.start_time = time.time()
        
        logger.info(f"Agent {name} initialisiert")
        
    @abstractmethod
    def process(self, data: Any) -> Dict[str, Any]:
        """
        Verarbeitet Eingabedaten
        
        Args:
            data: Eingabedaten
            
        Returns:
            Verarbeitungsergebnis
        """
        pass
        
    def initialize(self) -> bool:
        """
        Initialisiert den Agent
        
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            self.status = AgentStatus.INITIALIZING
            
            # Agent-spezifische Initialisierung
            if self._initialize_agent():
                self.status = AgentStatus.ACTIVE
                logger.info(f"✅ Agent {self.name} erfolgreich initialisiert")
                return True
            else:
                self.status = AgentStatus.ERROR
                logger.error(f"❌ Agent {self.name} Initialisierung fehlgeschlagen")
                return False
                
        except Exception as e:
            self.status = AgentStatus.ERROR
            logger.error(f"❌ Agent {self.name} Initialisierung Fehler: {e}")
            return False
            
    def shutdown(self) -> bool:
        """
        Fährt den Agent herunter
        
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            self.status = AgentStatus.SHUTDOWN
            self._shutdown_agent()
            logger.info(f"🛑 Agent {self.name} heruntergefahren")
            return True
            
        except Exception as e:
            logger.error(f"❌ Agent {self.name} Shutdown Fehler: {e}")
            return False
            
    def get_status(self) -> Dict[str, Any]:
        """
        Gibt Agent-Status zurück
        
        Returns:
            Status-Dictionary
        """
        current_time = time.time()
        uptime = current_time - self.start_time
        
        return {
            "name": self.name,
            "status": self.status.value,
            "uptime": uptime,
            "metrics": {
                "tasks_completed": self.metrics.tasks_completed,
                "tasks_failed": self.metrics.tasks_failed,
                "average_response_time": self.metrics.average_response_time,
                "success_rate": self._calculate_success_rate()
            },
            "config": self.config
        }
        
    def health_check(self) -> bool:
        """
        Führt Health-Check durch
        
        Returns:
            True wenn gesund, False bei Problemen
        """
        return self.status in [AgentStatus.ACTIVE, AgentStatus.BUSY]
        
    def _initialize_agent(self) -> bool:
        """Agent-spezifische Initialisierung - Override in Subclasses"""
        return True
        
    def _shutdown_agent(self):
        """Agent-spezifische Shutdown-Logik - Override in Subclasses"""
        pass
        
    def _calculate_success_rate(self) -> float:
        """Berechnet Erfolgsrate"""
        total_tasks = self.metrics.tasks_completed + self.metrics.tasks_failed
        if total_tasks == 0:
            return 0.0
        return self.metrics.tasks_completed / total_tasks
        
    def _update_metrics(self, success: bool, response_time: float):
        """Aktualisiert Performance-Metriken"""
        with self.lock:
            if success:
                self.metrics.tasks_completed += 1
            else:
                self.metrics.tasks_failed += 1
                
            # Update average response time
            total_tasks = self.metrics.tasks_completed + self.metrics.tasks_failed
            if total_tasks == 1:
                self.metrics.average_response_time = response_time
            else:
                self.metrics.average_response_time = (
                    (self.metrics.average_response_time * (total_tasks - 1) + response_time) / total_tasks
                )
                
            self.metrics.last_activity = time.time()

class MemoryAgent(BaseAgent):
    """
    Memory Management Agent - Realistic Implementation
    
    Verwaltet Speicher und Wissensretrieval ohne "AGI" Übertreibung.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("VX-MEMEX", config)
        self.memory_store: Dict[str, Any] = {}
        self.index: Dict[str, List[str]] = {}
        
    def process(self, data: Any) -> Dict[str, Any]:
        """Verarbeitet Memory-Operationen"""
        start_time = time.time()
        
        try:
            self.status = AgentStatus.BUSY
            
            operation = data.get("operation", "retrieve")
            
            if operation == "store":
                result = self._store_data(data.get("key"), data.get("value"))
            elif operation == "retrieve":
                result = self._retrieve_data(data.get("key"))
            elif operation == "search":
                result = self._search_data(data.get("query"))
            else:
                result = {"error": f"Unknown operation: {operation}"}
                
            self.status = AgentStatus.ACTIVE
            response_time = time.time() - start_time
            self._update_metrics(True, response_time)
            
            return result
            
        except Exception as e:
            self.status = AgentStatus.ACTIVE
            response_time = time.time() - start_time
            self._update_metrics(False, response_time)
            
            return {"error": str(e)}
            
    def _store_data(self, key: str, value: Any) -> Dict[str, Any]:
        """Speichert Daten"""
        if not key:
            return {"error": "Key required"}
            
        self.memory_store[key] = value
        
        # Simple indexing
        if isinstance(value, str):
            words = value.lower().split()
            for word in words:
                if word not in self.index:
                    self.index[word] = []
                if key not in self.index[word]:
                    self.index[word].append(key)
                    
        return {"success": True, "key": key}
        
    def _retrieve_data(self, key: str) -> Dict[str, Any]:
        """Ruft Daten ab"""
        if key in self.memory_store:
            return {"success": True, "key": key, "value": self.memory_store[key]}
        else:
            return {"success": False, "error": "Key not found"}
            
    def _search_data(self, query: str) -> Dict[str, Any]:
        """Sucht Daten"""
        if not query:
            return {"error": "Query required"}
            
        query_words = query.lower().split()
        matching_keys = set()
        
        for word in query_words:
            if word in self.index:
                matching_keys.update(self.index[word])
                
        results = {}
        for key in matching_keys:
            results[key] = self.memory_store[key]
            
        return {"success": True, "results": results, "count": len(results)}

class ReasoningAgent(BaseAgent):
    """
    Reasoning Agent - Realistic Implementation
    
    Führt logische Operationen durch ohne "AGI" Übertreibung.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("VX-REASON", config)
        self.rules: List[Dict[str, Any]] = []
        
    def process(self, data: Any) -> Dict[str, Any]:
        """Verarbeitet Reasoning-Operationen"""
        start_time = time.time()
        
        try:
            self.status = AgentStatus.BUSY
            
            operation = data.get("operation", "infer")
            
            if operation == "infer":
                result = self._infer(data.get("facts", []))
            elif operation == "add_rule":
                result = self._add_rule(data.get("rule"))
            elif operation == "validate":
                result = self._validate_logic(data.get("statement"))
            else:
                result = {"error": f"Unknown operation: {operation}"}
                
            self.status = AgentStatus.ACTIVE
            response_time = time.time() - start_time
            self._update_metrics(True, response_time)
            
            return result
            
        except Exception as e:
            self.status = AgentStatus.ACTIVE
            response_time = time.time() - start_time
            self._update_metrics(False, response_time)
            
            return {"error": str(e)}
            
    def _infer(self, facts: List[str]) -> Dict[str, Any]:
        """Führt einfache Inferenz durch"""
        inferences = []
        
        # Simple rule-based inference
        for rule in self.rules:
            if self._check_rule_conditions(rule, facts):
                inferences.append(rule.get("conclusion"))
                
        return {"success": True, "inferences": inferences}
        
    def _add_rule(self, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Fügt Regel hinzu"""
        if not rule or "conditions" not in rule or "conclusion" not in rule:
            return {"error": "Invalid rule format"}
            
        self.rules.append(rule)
        return {"success": True, "rules_count": len(self.rules)}
        
    def _validate_logic(self, statement: str) -> Dict[str, Any]:
        """Validiert logische Aussage"""
        # Simple validation
        if not statement:
            return {"valid": False, "error": "Empty statement"}
            
        # Check for basic logical consistency
        contradictions = ["true and false", "not true and true"]
        
        for contradiction in contradictions:
            if contradiction in statement.lower():
                return {"valid": False, "error": "Logical contradiction detected"}
                
        return {"valid": True}
        
    def _check_rule_conditions(self, rule: Dict[str, Any], facts: List[str]) -> bool:
        """Überprüft Regelbedingungen"""
        conditions = rule.get("conditions", [])
        
        for condition in conditions:
            if condition not in facts:
                return False
                
        return True

class CoordinationAgent(BaseAgent):
    """
    Coordination Agent - Realistic Implementation
    
    Koordiniert Tasks zwischen Agenten ohne "AGI" Übertreibung.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("VX-NEXUS", config)
        self.task_queue: List[Dict[str, Any]] = []
        self.agents: Dict[str, BaseAgent] = {}
        
    def process(self, data: Any) -> Dict[str, Any]:
        """Verarbeitet Koordinations-Operationen"""
        start_time = time.time()
        
        try:
            self.status = AgentStatus.BUSY
            
            operation = data.get("operation", "coordinate")
            
            if operation == "coordinate":
                result = self._coordinate_task(data.get("task"))
            elif operation == "register_agent":
                result = self._register_agent(data.get("agent"))
            elif operation == "get_status":
                result = self._get_system_status()
            else:
                result = {"error": f"Unknown operation: {operation}"}
                
            self.status = AgentStatus.ACTIVE
            response_time = time.time() - start_time
            self._update_metrics(True, response_time)
            
            return result
            
        except Exception as e:
            self.status = AgentStatus.ACTIVE
            response_time = time.time() - start_time
            self._update_metrics(False, response_time)
            
            return {"error": str(e)}
            
    def _coordinate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Koordiniert Task-Ausführung"""
        if not task:
            return {"error": "Task required"}
            
        task_type = task.get("type")
        target_agent = self._select_agent_for_task(task_type)
        
        if not target_agent:
            return {"error": f"No agent available for task type: {task_type}"}
            
        # Delegate to agent
        result = target_agent.process(task.get("data", {}))
        
        return {
            "success": True,
            "agent": target_agent.name,
            "result": result
        }
        
    def _register_agent(self, agent: BaseAgent) -> Dict[str, Any]:
        """Registriert Agent"""
        if not agent or not hasattr(agent, 'name'):
            return {"error": "Invalid agent"}
            
        self.agents[agent.name] = agent
        return {"success": True, "agent": agent.name}
        
    def _get_system_status(self) -> Dict[str, Any]:
        """Gibt System-Status zurück"""
        agent_statuses = {}
        
        for name, agent in self.agents.items():
            agent_statuses[name] = agent.get_status()
            
        return {
            "success": True,
            "agents": agent_statuses,
            "task_queue_size": len(self.task_queue)
        }
        
    def _select_agent_for_task(self, task_type: str) -> Optional[BaseAgent]:
        """Wählt passenden Agent für Task"""
        task_agent_mapping = {
            "memory": "VX-MEMEX",
            "reasoning": "VX-REASON",
            "storage": "VX-MEMEX",
            "inference": "VX-REASON"
        }
        
        agent_name = task_agent_mapping.get(task_type)
        return self.agents.get(agent_name) if agent_name else None

# Agent Factory
def create_agent(agent_type: str, config: Optional[Dict[str, Any]] = None) -> Optional[BaseAgent]:
    """
    Erstellt Agent-Instanz
    
    Args:
        agent_type: Agent-Typ
        config: Konfiguration
        
    Returns:
        Agent-Instanz oder None
    """
    agent_classes = {
        "memory": MemoryAgent,
        "reasoning": ReasoningAgent,
        "coordination": CoordinationAgent
    }
    
    agent_class = agent_classes.get(agent_type.lower())
    if agent_class:
        return agent_class(config)
    else:
        logger.error(f"Unknown agent type: {agent_type}")
        return None

# Agent Manager
class AgentManager:
    """Verwaltet alle Agenten"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.lock = threading.RLock()
        
    def create_and_register_agent(self, agent_type: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Erstellt und registriert Agent"""
        agent = create_agent(agent_type, config)
        if agent and agent.initialize():
            with self.lock:
                self.agents[agent.name] = agent
            logger.info(f"Agent {agent.name} erstellt und registriert")
            return True
        return False
        
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Gibt Agent zurück"""
        with self.lock:
            return self.agents.get(name)
            
    def get_all_agents(self) -> Dict[str, BaseAgent]:
        """Gibt alle Agenten zurück"""
        with self.lock:
            return self.agents.copy()
            
    def shutdown_all(self):
        """Fährt alle Agenten herunter"""
        with self.lock:
            for agent in self.agents.values():
                agent.shutdown()
            self.agents.clear()

# Global Agent Manager
_agent_manager: Optional[AgentManager] = None

def get_agent_manager() -> AgentManager:
    """Gibt globalen Agent Manager zurück"""
    global _agent_manager
    if _agent_manager is None:
        _agent_manager = AgentManager()
    return _agent_manager
