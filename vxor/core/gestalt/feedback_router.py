"""
Feedback-Router für die bidirektionale Kommunikation zwischen VX-GESTALT und Agenten.

Dieses Modul implementiert ein leistungsfähiges Routing-System für Feedback und
Entscheidungen zwischen der GESTALT-Ebene und den einzelnen Agenten.
"""

from typing import Dict, List, Optional, Any, Callable
import logging
import uuid
from dataclasses import dataclass
from enum import Enum, auto

# Import der benötigten Module
try:
    from miso.security.void_protocol import verify as void_verify, secure as void_secure
    VOID_AVAILABLE = True
except ImportError:
    def void_verify(data, signature=None): return True
    def void_secure(data): return data
    VOID_AVAILABLE = False

try:
    from miso.security.ztm_hooks import ztm_monitored
    ZTM_AVAILABLE = True
except ImportError:
    def ztm_monitored(component, operation, severity="INFO"):
        def decorator(func): return func
        return decorator
    ZTM_AVAILABLE = False

# Logger konfigurieren
logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Typen von Feedback-Nachrichten"""
    DECISION = auto()       # Entscheidung der GESTALT-Ebene
    CONFLICT = auto()       # Konfliktmeldung
    PRIORITY_UPDATE = auto() # Prioritätsaktualisierung
    STATE_UPDATE = auto()   # Zustandsaktualisierung
    COMMAND = auto()        # Direkter Befehl

@dataclass
class FeedbackMessage:
    """Struktur einer Feedback-Nachricht"""
    message_id: str
    sender: str              # Absender (z.B. 'GESTALT', 'VX-INTENT')
    receiver: str            # Empfänger (Agenten-ID oder 'BROADCAST')
    message_type: FeedbackType
    payload: Dict[str, Any]
    timestamp: float
    ttl: float = 60.0        # Time-to-live in Sekunden
    signature: str = ""      # VOID-Signatur

class FeedbackRouter:
    """
    Verwaltet das Routing von Feedback-Nachrichten zwischen GESTALT und Agenten.
    """
    
    def __init__(self, void_verifier: Optional[Any] = None):
        """
        Initialisiert den Feedback-Router.
        
        Args:
            void_verifier: Optionaler VOID-Verifier für Signaturprüfungen (deprecated)
        """
        # VOID-Protokoll-Integration (modernisiert)
        if VOID_AVAILABLE:
            logger.info("VOID-Protokoll für Feedback-Router verfügbar")
        else:
            logger.warning("VOID-Protokoll nicht verfügbar, verwende Fallback")
        self.routes: Dict[str, List[Callable]] = {}
        self.message_queue: List[FeedbackMessage] = []
        self.message_history: Dict[str, FeedbackMessage] = {}
        self.subscriptions: Dict[str, List[str]] = {}  # agent_id -> List[message_types]
        
        # Standardrouten registrieren
        self._register_default_routes()
        
        logger.info("Feedback-Router initialisiert")
    
    def _register_default_routes(self):
        """Registriert Standard-Routen für grundlegende Nachrichtentypen."""
        # Standard-Handler für unbehandelte Nachrichten
        self.add_route("*", self._default_message_handler)
        
        # Handler für verschiedene Nachrichtentypen
        self.add_route(FeedbackType.DECISION, self._handle_decision_message)
        self.add_route(FeedbackType.CONFLICT, self._handle_conflict_message)
        self.add_route(FeedbackType.PRIORITY_UPDATE, self._handle_priority_update)
    
    def add_route(self, message_type: Any, handler: Callable):
        """
        Fügt eine neue Route für einen bestimmten Nachrichtentyp hinzu.
        
        Args:
            message_type: Der Nachrichtentyp (kann ein Enum-Wert oder "*" für alle sein)
            handler: Die aufzurufende Handler-Funktion
        """
        type_key = message_type.value if hasattr(message_type, 'value') else str(message_type)
        if type_key not in self.routes:
            self.routes[type_key] = []
        self.routes[type_key].append(handler)
        logger.debug(f"Neue Route für Nachrichtentyp {type_key} registriert")
    
    def subscribe(self, agent_id: str, message_types: List[FeedbackType]):
        """
        Registriert einen Agenten für bestimmte Nachrichtentypen.
        
        Args:
            agent_id: Die ID des Agenten
            message_types: Liste von Nachrichtentypen, für die der Agent Benachrichtigungen erhalten soll
        """
        self.subscriptions[agent_id] = [t.value if hasattr(t, 'value') else t for t in message_types]
        logger.info(f"Agent {agent_id} für Nachrichtentypen {message_types} registriert")
    
    def unsubscribe(self, agent_id: str):
        """
        Entfernt einen Agenten aus allen Abonnements.
        
        Args:
            agent_id: Die ID des Agenten
        """
        if agent_id in self.subscriptions:
            del self.subscriptions[agent_id]
            logger.info(f"Agent {agent_id} von allen Abonnements entfernt")
    
    def send_feedback(self, sender: str, receiver: str, 
                     message_type: FeedbackType, 
                     payload: Dict[str, Any],
                     ttl: float = 60.0) -> str:
        """
        Sendet eine Feedback-Nachricht an einen Empfänger.
        
        Args:
            sender: Absender der Nachricht
            receiver: Empfänger der Nachricht (Agenten-ID oder 'BROADCAST')
            message_type: Typ der Nachricht
            payload: Nutzdaten der Nachricht
            ttl: Time-to-live in Sekunden
            
        Returns:
            str: Eindeutige Nachrichten-ID
        """
        message_id = f"msg_{uuid.uuid4()}"
        timestamp = time.time()
        
        # Nachricht erstellen
        message = FeedbackMessage(
            message_id=message_id,
            sender=sender,
            receiver=receiver,
            message_type=message_type,
            payload=payload,
            timestamp=timestamp,
            ttl=ttl
        )
        
        # Nachricht signieren
        message.signature = self._sign_message(message)
        
        # Zur Warteschlange hinzufügen
        self.message_queue.append(message)
        self.message_history[message_id] = message
        
        logger.debug(f"Neue Nachricht von {sender} an {receiver}: {message_type}")
        return message_id
    
    def process_messages(self):
        """Verarbeitet alle ausstehenden Nachrichten in der Warteschlange."""
        current_time = time.time()
        processed_messages = []
        
        for message in self.message_queue[:]:
            # Auf abgelaufene Nachrichten prüfen
            if current_time > (message.timestamp + message.ttl):
                logger.debug(f"Nachricht {message.message_id} abgelaufen")
                self.message_queue.remove(message)
                continue
            
            # Signatur überprüfen
            if not self._verify_message(message):
                logger.warning(f"Ungültige Signatur für Nachricht {message.message_id}")
                self.message_queue.remove(message)
                continue
            
            # Nachricht weiterleiten
            self._route_message(message)
            processed_messages.append(message.message_id)
            
            # Aus der Warteschlange entfernen
            self.message_queue.remove(message)
        
        return processed_messages
    
    def _route_message(self, message: FeedbackMessage):
        """
        Leitet eine Nachricht an die entsprechenden Handler weiter.
        
        Args:
            message: Die zu routende Nachricht
        """
        type_key = message.message_type.value if hasattr(message.message_type, 'value') else str(message.message_type)
        
        # Spezifische Handler für diesen Nachrichtentyp aufrufen
        for handler in self.routes.get(type_key, []):
            try:
                handler(message)
            except Exception as e:
                logger.error(f"Fehler im Nachrichten-Handler {handler.__name__}: {str(e)}")
        
        # Allgemeine Handler aufrufen
        for handler in self.routes.get("*", []):
            try:
                handler(message)
            except Exception as e:
                logger.error(f"Fehler im allgemeinen Nachrichten-Handler {handler.__name__}: {str(e)}")
    
    def _sign_message(self, message: FeedbackMessage) -> str:
        """
        Signiert eine Nachricht mit VOID.
        
        Args:
            message: Die zu signierende Nachricht
            
        Returns:
            str: Die Signatur der Nachricht
        """
        # Erstelle ein Signatur-Dict mit den relevanten Nachrichtendaten
        sign_data = {
            'message_id': message.message_id,
            'sender': message.sender,
            'receiver': message.receiver,
            'message_type': message.message_type.value if hasattr(message.message_type, 'value') else str(message.message_type),
            'timestamp': message.timestamp,
            'ttl': message.ttl,
            'payload': message.payload
        }
        
        # Signatur erstellen
        if VOID_AVAILABLE:
            secured_data = void_secure(sign_data)
            return secured_data.get('signature', '')
        else:
            return ''
    
    def _verify_message(self, message: FeedbackMessage) -> bool:
        """
        Überprüft die Signatur einer Nachricht.
        
        Args:
            message: Die zu überprüfende Nachricht
            
        Returns:
            bool: True, wenn die Signatur gültig ist, sonst False
        """
        # Erstelle ein Signatur-Dict mit den relevanten Nachrichtendaten
        sign_data = {
            'message_id': message.message_id,
            'sender': message.sender,
            'receiver': message.receiver,
            'message_type': message.message_type.value if hasattr(message.message_type, 'value') else str(message.message_type),
            'timestamp': message.timestamp,
            'ttl': message.ttl,
            'payload': message.payload
        }
        
        # Signatur überprüfen
        if VOID_AVAILABLE:
            return void_verify(sign_data, message.signature)
        else:
            return True  # Fallback: immer gültig
    
    # --- Standard-Nachrichtenhandler ---
    
    def _default_message_handler(self, message: FeedbackMessage):
        """Standard-Handler für unbehandelte Nachrichten."""
        logger.debug(f"Unbehandelte Nachricht: {message.sender} -> {message.receiver}: {message.message_type}")
    
    def _handle_decision_message(self, message: FeedbackMessage):
        """Handler für Entscheidungsnachrichten."""
        logger.info(f"Entscheidung von {message.sender} an {message.receiver}: {message.payload}")
        
        # Wenn der Empfänger 'BROADCAST' ist, an alle Abonnenten senden
        if message.receiver == 'BROADCAST':
            for agent_id in self.subscriptions.keys():
                self._deliver_message(agent_id, message)
        else:
            # An bestimmten Empfänger senden
            self._deliver_message(message.receiver, message)
    
    def _handle_conflict_message(self, message: FeedbackMessage):
        """Handler für Konfliktnachrichten."""
        logger.warning(f"Konfliktmeldung von {message.sender}: {message.payload}")
        
        # Konfliktnachrichten werden standardmäßig an alle interessierten Agenten gesendet
        for agent_id, subscriptions in self.subscriptions.items():
            if message.message_type.value in subscriptions:
                self._deliver_message(agent_id, message)
    
    def _handle_priority_update(self, message: FeedbackMessage):
        """Handler für Prioritätsaktualisierungen."""
        logger.info(f"Prioritätsaktualisierung von {message.sender}: {message.payload}")
        
        # Prioritätsaktualisierungen werden an den Zielagenten gesendet
        target_agent = message.receiver
        if target_agent in self.subscriptions:
            self._deliver_message(target_agent, message)
    
    def _deliver_message(self, agent_id: str, message: FeedbackMessage):
        """
        Leitet eine Nachricht an einen bestimmten Agenten weiter.
        
        Args:
            agent_id: Die ID des Zielagenten
            message: Die zu übermittelnde Nachricht
        """
        # In einer echten Implementierung würde hier die Nachricht an den Agenten übergeben werden
        logger.debug(f"Nachricht {message.message_id} an {agent_id} übermittelt")
        
        # Hier würde normalerweise der Callback des Agenten aufgerufen werden
        # z.B.: agent.receive_feedback(message)

# Singleton-Instanz
_feedback_router = None

def get_feedback_router() -> FeedbackRouter:
    """
    Gibt die Singleton-Instanz des Feedback-Routers zurück.
    
    Returns:
        FeedbackRouter: Die Singleton-Instanz
    """
    global _feedback_router
    if _feedback_router is None:
        _feedback_router = FeedbackRouter()
    return _feedback_router
