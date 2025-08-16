"""
VX-GESTALT - Emergenz- und Integrationsschicht für das vXor-System

Diese Komponente ist verantwortlich für die Synthese von Agentenzuständen und die
Erzeugung eines einheitlichen Systemzustands (Emergent State).
"""

from typing import Dict, List, TypedDict, Optional, Any, Callable
from dataclasses import dataclass, field
import time
import uuid
import logging

# Import der benötigten Module
try:
    from miso.security.void_protocol import verify as void_verify, secure as void_secure
    VOID_AVAILABLE = True
except ImportError:
    def void_verify(data, signature=None): return True
    def void_secure(data): return data
    VOID_AVAILABLE = False

try:
    from miso.security.ztm_hooks import ztm_monitored, ztm_critical_operation
    ZTM_HOOKS_AVAILABLE = True
except ImportError:
    # Fallback-Decorators
    def ztm_monitored(component, operation, severity="INFO"):
        def decorator(func): return func
        return decorator
    def ztm_critical_operation(component, operation):
        def decorator(func): return func
        return decorator
    ZTM_HOOKS_AVAILABLE = False

# Lokale Imports
from .feedback_router import FeedbackType, FeedbackMessage, get_feedback_router

# Logger konfigurieren
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """Zustand eines einzelnen Agenten"""
    agent_id: str
    state: Dict[str, Any]
    confidence: float  # 0.0 bis 1.0
    timestamp: float
    signature: str  # VOID-Signatur

class EmergentState(TypedDict):
    """Vereinheitlichter Systemzustand"""
    state_id: str
    timestamp: float
    unified_state: Dict[str, Any]
    conflicts: List[Dict[str, Any]]
    decisions: List[Dict[str, Any]]
    signature: str  # VOID-Signatur

@dataclass
class Conflict:
    """Repräsentiert einen Konflikt zwischen Agenten"""
    conflict_id: str
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    involved_agents: List[str]
    resolution: Optional[str] = None

class GestaltIntegrator:
    """
    Kernklasse für die Integration von Agentenzuständen und Erzeugung
    eines einheitlichen Systemzustands.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialisiert den GestaltIntegrator.
        
        Args:
            config: Konfigurationsdictionary (optional)
        """
        self.config = config or {}
        # VOID-Protokoll-Integration
        if VOID_AVAILABLE:
            logger.info("VOID-Protokoll für VX-GESTALT verfügbar")
        else:
            logger.warning("VOID-Protokoll nicht verfügbar, verwende Fallback")
        self.agent_states: Dict[str, Dict] = {}
        self.last_emergent_state: Optional[EmergentState] = None
        self.conflicts: List[Conflict] = []
        
        # Feedback-Router initialisieren
        self.feedback_router = get_feedback_router()
        self._register_feedback_handlers()
        
        # Konfiguration der Agentenprioritäten (je höher der Wert, desto höher die Priorität)
        self.agent_priorities = {
            'VX-PSI': 100,
            'VX-INTENT': 90,
            'VX-CHRONOS': 80,
            'VX-MEMEX': 70,
            'VX-SOMA': 60
        }
        
        logger.info("GestaltIntegrator initialisiert")
    
    def register_agent(self, agent_id: str, agent_instance: Any):
        """
        Registriert einen Agenten beim Integrator.
        
        Args:
            agent_id: Eindeutige ID des Agenten
            agent_instance: Instanz des Agenten (muss get_state() implementieren)
        """
        if not hasattr(agent_instance, 'get_state'):
            raise ValueError("Agent muss eine get_state()-Methode implementieren")
            
        self.agent_states[agent_id] = {
            'instance': agent_instance,
            'last_state': None,
            'last_update': 0.0,
            'feedback_handlers': {}
        }
        
        # Agenten beim Feedback-Router registrieren
        if hasattr(agent_instance, 'receive_feedback'):
            self._register_agent_feedback_handler(agent_id, agent_instance.receive_feedback)
        logger.info(f"Agent registriert: {agent_id}")
    
    @ztm_monitored(component="GESTALT", operation="unify")
    def unify(self) -> EmergentState:
        """
        Führt die Vereinheitlichung der Agentenzustände durch und erzeugt
        einen einheitlichen Systemzustand.
        
        Returns:
            EmergentState: Der vereinheitlichte Systemzustand
        """
        logger.info("Starte Zustandsvereinheitlichung...")
        
        # 1. Aktuelle Zustände aller Agenten abrufen
        current_states = self._collect_agent_states()
        
        # 2. Konflikte zwischen Agentenzuständen erkennen
        self._detect_conflicts(current_states)
        
        # 3. Zustandssynthese durchführen
        unified_state = self._synthesize_state(current_states)
        
        # 4. Emergenten Zustand erstellen
        emergent_state = self._create_emergent_state(unified_state)
        
        # 5. Feedback an Agenten senden
        self._provide_feedback(emergent_state)
        
        # 6. Feedback-Nachrichten verarbeiten
        self.feedback_router.process_messages()
        
        self.last_emergent_state = emergent_state
        logger.info("Zustandsvereinheitlichung abgeschlossen")
        
        return emergent_state
    
    def _collect_agent_states(self) -> Dict[str, Dict]:
        """Sammelt die Zustände aller registrierten Agenten."""
        states = {}
        
        for agent_id, agent_data in self.agent_states.items():
            try:
                # Zustand vom Agenten abrufen
                state = agent_data['instance'].get_state()
                
                # VOID-Signatur überprüfen
                if VOID_AVAILABLE and not void_verify(state):
                    logger.warning(f"Ungültige Signatur für Agent {agent_id}")
                    continue
                
                # Zustand speichern
                states[agent_id] = {
                    'state': state,
                    'timestamp': time.time(),
                    'confidence': state.get('confidence', 0.5)
                }
                
                # Letzten bekannten Zustand aktualisieren
                agent_data['last_state'] = state
                agent_data['last_update'] = time.time()
                
            except Exception as e:
                logger.error(f"Fehler beim Abrufen des Zustands von {agent_id}: {str(e)}")
        
        return states
    
    def _detect_conflicts(self, states: Dict[str, Dict]):
        """Erkennt Konflikte zwischen Agentenzuständen."""
        self.conflicts = []
        
        # Beispielkonflikt: PSI vs. INTENT
        if 'VX-PSI' in states and 'VX-INTENT' in states:
            psi_state = states['VX-PSI']['state']
            intent_state = states['VX-INTENT']['state']
            
            # Beispiel: Wenn PSI eine Aktion als unsicher einstuft, die INTENT ausführen möchte
            if (intent_state.get('requested_action') and 
                psi_state.get('safety_assessment', {}).get(intent_state['requested_action'], {}).get('is_safe') is False):
                
                conflict = Conflict(
                    conflict_id=str(uuid.uuid4()),
                    description=f"Sicherheitsbedenken für Aktion {intent_state['requested_action']}",
                    severity='high',
                    involved_agents=['VX-PSI', 'VX-INTENT']
                )
                self.conflicts.append(conflict)
                logger.warning(f"Sicherheitskonflikt erkannt: {conflict.description}")
    
    def _synthesize_state(self, states: Dict[str, Dict]) -> Dict[str, Any]:
        """Führt die Zustandssynthese durch."""
        unified_state = {
            'timestamp': time.time(),
            'agents': {},
            'system_status': 'nominal',
            'active_conflicts': len(self.conflicts) > 0
        }
        
        # Zustände nach Priorität sortieren
        sorted_agents = sorted(
            states.items(),
            key=lambda x: self.agent_priorities.get(x[0], 0),
            reverse=True
        )
        
        # Zustände zusammenführen
        for agent_id, state_data in sorted_agents:
            unified_state['agents'][agent_id] = {
                'state': state_data['state'],
                'confidence': state_data['confidence'],
                'timestamp': state_data['timestamp']
            }
        
        # Konflikte hinzufügen
        unified_state['conflicts'] = [{
            'id': c.conflict_id,
            'description': c.description,
            'severity': c.severity,
            'involved_agents': c.involved_agents,
            'resolved': c.resolution is not None,
            'resolution': c.resolution
        } for c in self.conflicts]
        
        return unified_state
    
    def _create_emergent_state(self, unified_state: Dict[str, Any]) -> EmergentState:
        """Erstellt den finalen EmergentState."""
        emergent_state = {
            'state_id': str(uuid.uuid4()),
            'timestamp': time.time(),
            'unified_state': unified_state,
            'conflicts': unified_state.get('conflicts', []),
            'decisions': self._make_decisions(unified_state),
            'signature': ''  # Wird später signiert
        }
        
        # Mit VOID signieren
        if VOID_AVAILABLE:
            emergent_state['signature'] = void_secure(emergent_state).get('signature', '')
        else:
            emergent_state['signature'] = ''
        
        return EmergentState(emergent_state)
    
    def _make_decisions(self, unified_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Trifft Entscheidungen basierend auf dem aktuellen Zustand."""
        decisions = []
        
        # Beispiel: Bei Konflikten Entscheidung zur Konfliktlösung treffen
        for conflict in unified_state.get('conflicts', []):
            if conflict['severity'] in ['high', 'critical']:
                decisions.append({
                    'decision_id': str(uuid.uuid4()),
                    'type': 'safety_override',
                    'description': f"Sicherheitskonflikt erkannt: {conflict['description']}",
                    'action': 'block_action',
                    'target_agent': 'VX-INTENT',
                    'parameters': {
                        'blocked_action': conflict.get('action_id', 'unknown'),
                        'reason': 'safety_violation'
                    }
                })
        
        return decisions
    
    def _register_feedback_handlers(self):
        """Registriert die Feedback-Handler für den Integrator."""
        # Standard-Handler für Entscheidungen
        self.feedback_router.add_route(
            FeedbackType.DECISION,
            self._handle_decision_feedback
        )
        
        # Handler für Konfliktbenachrichtigungen
        self.feedback_router.add_route(
            FeedbackType.CONFLICT,
            self._handle_conflict_notification
        )
    
    def _register_agent_feedback_handler(self, agent_id: str, handler: Callable):
        """
        Registriert einen Feedback-Handler für einen Agenten.
        
        Args:
            agent_id: Die ID des Agenten
            handler: Die Handler-Funktion
        """
        if agent_id in self.agent_states:
            self.agent_states[agent_id]['feedback_handler'] = handler
            logger.debug(f"Feedback-Handler für Agent {agent_id} registriert")
    
    def _provide_feedback(self, emergent_state: EmergentState):
        """Sendet Feedback an die Agenten basierend auf dem EmergentState."""
        for decision in emergent_state.get('decisions', []):
            target_agent = decision.get('target_agent')
            if target_agent in self.agent_states:
                try:
                    # Feedback über den Router senden
                    self.feedback_router.send_feedback(
                        sender="GESTALT",
                        receiver=target_agent,
                        message_type=FeedbackType.DECISION,
                        payload={
                            'decision_id': decision['decision_id'],
                            'type': decision['type'],
                            'action': decision['action'],
                            'parameters': decision.get('parameters', {})
                        }
                    )
                except Exception as e:
                    logger.error(f"Fehler beim Senden von Feedback an {target_agent}: {str(e)}")
    
    def _handle_decision_feedback(self, message: FeedbackMessage):
        """Verarbeitet Entscheidungsfeedback."""
        target_agent = message.receiver
        if target_agent in self.agent_states and 'feedback_handler' in self.agent_states[target_agent]:
            try:
                handler = self.agent_states[target_agent]['feedback_handler']
                if callable(handler):
                    handler({
                        'message_id': message.message_id,
                        'sender': message.sender,
                        'type': 'decision',
                        'payload': message.payload,
                        'timestamp': message.timestamp
                    })
            except Exception as e:
                logger.error(f"Fehler im Entscheidungs-Handler für {target_agent}: {str(e)}")
    
    def _handle_conflict_notification(self, message: FeedbackMessage):
        """Verarbeitet Konfliktbenachrichtigungen."""
        # Hier können spezielle Aktionen bei Konflikten durchgeführt werden
        logger.warning(f"Konfliktbenachrichtigung empfangen: {message.payload}")
        
        # Beispiel: Konflikt an alle beteiligten Agenten weiterleiten
        if 'involved_agents' in message.payload:
            for agent_id in message.payload['involved_agents']:
                if agent_id in self.agent_states and 'feedback_handler' in self.agent_states[agent_id]:
                    try:
                        handler = self.agent_states[agent_id]['feedback_handler']
                        if callable(handler):
                            handler({
                                'message_id': message.message_id,
                                'sender': message.sender,
                                'type': 'conflict_notification',
                                'payload': message.payload,
                                'timestamp': message.timestamp
                            })
                    except Exception as e:
                        logger.error(f"Fehler im Konflikt-Handler für {agent_id}: {str(e)}")

# Singleton-Instanz
_gestalt_integrator = None

def get_gestalt_integrator(config: Optional[Dict] = None) -> 'GestaltIntegrator':
    """
    Gibt die Singleton-Instanz des GestaltIntegrators zurück.
    
    Args:
        config: Konfigurationsdictionary (optional)
        
    Returns:
        GestaltIntegrator: Die Singleton-Instanz
    """
    global _gestalt_integrator
    if _gestalt_integrator is None:
        _gestalt_integrator = GestaltIntegrator(config)
    return _gestalt_integrator
