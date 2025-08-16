"""
Integrationstests für die VX-GESTALT-Komponente.

Dieses Modul enthält Tests für die Funktionalität des GestaltIntegrators
und die Interaktion mit den VXOR-Agenten.
"""

import unittest
import time
import logging
from unittest.mock import MagicMock, patch
from typing import Dict, Any, Optional

# Konfiguration des Loggings für die Tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import der zu testenden Komponenten
from vxor.core.gestalt import (
    GestaltIntegrator,
    get_gestalt_integrator,
    AgentState,
    EmergentState
)

# Mock-Klassen für die VXOR-Agenten
class MockVXAgent:
    """Basisklasse für Mock-VXOR-Agenten."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.received_feedback = []
        self.state = {
            'agent_id': agent_id,
            'status': 'idle',
            'confidence': 0.9,
            'timestamp': time.time()
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Gibt den aktuellen Zustand des Agenten zurück."""
        self.state['timestamp'] = time.time()
        return self.state
    
    def receive_feedback(self, feedback: Dict[str, Any]):
        """Empfängt Feedback vom GestaltIntegrator."""
        self.received_feedback.append(feedback)
        logger.info(f"{self.agent_id} hat Feedback erhalten: {feedback}")

class MockVXIntent(MockVXAgent):
    """Mock für den VX-INTENT-Agenten."""
    
    def __init__(self):
        super().__init__("VX-INTENT")
        self.state.update({
            'requested_action': 'execute_operation',
            'action_parameters': {'operation': 'test_operation'},
            'confidence': 0.95
        })

class MockVXPSI(MockVXAgent):
    """Mock für den VX-PSI-Agenten."""
    
    def __init__(self):
        super().__init__("VX-PSI")
        self.state.update({
            'safety_assessment': {
                'execute_operation': {
                    'is_safe': False,
                    'confidence': 0.8,
                    'risks': ['potential_security_risk']
                }
            },
            'self_awareness': 'high',
            'confidence': 0.85
        })

class TestGestaltIntegration(unittest.TestCase):
    """Testfälle für die Gestalt-Integration."""
    
    def setUp(self):
        """Bereitet die Testumgebung vor."""
        # Neuen GestaltIntegrator erstellen
        self.gestalt = GestaltIntegrator()
        
        # Mock-Agenten erstellen
        self.intent_agent = MockVXIntent()
        self.psi_agent = MockVXPSI()
        
        # Agenten registrieren
        self.gestalt.register_agent("VX-INTENT", self.intent_agent)
        self.gestalt.register_agent("VX-PSI", self.psi_agent)
        
        # Logger für Tests konfigurieren
        self.logger = logging.getLogger(__name__)
    
    def test_agent_registration(self):
        """Testet die Registrierung von Agenten."""
        self.assertIn("VX-INTENT", self.gestalt.agent_states)
        self.assertIn("VX-PSI", self.gestalt.agent_states)
        self.assertEqual(len(self.gestalt.agent_states), 2)
    
    def test_state_collection(self):
        """Testet das Sammeln von Agentenzuständen."""
        states = self.gestalt._collect_agent_states()
        
        self.assertIn("VX-INTENT", states)
        self.assertIn("VX-PSI", states)
        self.assertEqual(states["VX-INTENT"]['state']['agent_id'], "VX-INTENT")
        self.assertEqual(states["VX-PSI"]['state']['agent_id'], "VX-PSI")
    
    def test_conflict_detection(self):
        """Testet die Konflikterkennung zwischen Agenten."""
        # Zustände sammeln
        states = {
            "VX-INTENT": {
                'state': {
                    'agent_id': 'VX-INTENT',
                    'requested_action': 'execute_operation',
                    'action_parameters': {'operation': 'test_operation'},
                    'confidence': 0.95,
                    'timestamp': time.time()
                },
                'timestamp': time.time(),
                'confidence': 0.95
            },
            "VX-PSI": {
                'state': {
                    'agent_id': 'VX-PSI',
                    'safety_assessment': {
                        'execute_operation': {
                            'is_safe': False,
                            'confidence': 0.8,
                            'risks': ['potential_security_risk']
                        }
                    },
                    'confidence': 0.85,
                    'timestamp': time.time()
                },
                'timestamp': time.time(),
                'confidence': 0.85
            }
        }
        
        # Konflikte erkennen
        self.gestalt._detect_conflicts(states)
        
        # Überprüfen, ob ein Konflikt erkannt wurde
        self.assertEqual(len(self.gestalt.conflicts), 1)
        self.assertEqual(self.gestalt.conflicts[0].severity, 'high')
        self.assertIn('VX-INTENT', self.gestalt.conflicts[0].involved_agents)
        self.assertIn('VX-PSI', self.gestalt.conflicts[0].involved_agents)
    
    def test_state_synthesis(self):
        """Testet die Synthese der Agentenzustände."""
        # Zustände sammeln
        states = {
            "VX-INTENT": {
                'state': {
                    'agent_id': 'VX-INTENT',
                    'status': 'active',
                    'confidence': 0.95,
                    'timestamp': time.time()
                },
                'timestamp': time.time(),
                'confidence': 0.95
            },
            "VX-PSI": {
                'state': {
                    'agent_id': 'VX-PSI',
                    'status': 'monitoring',
                    'confidence': 0.85,
                    'timestamp': time.time()
                },
                'timestamp': time.time(),
                'confidence': 0.85
            }
        }
        
        # Zustandssynthese durchführen
        unified_state = self.gestalt._synthesize_state(states)
        
        # Überprüfen des zusammengeführten Zustands
        self.assertIn('VX-INTENT', unified_state['agents'])
        self.assertIn('VX-PSI', unified_state['agents'])
        self.assertEqual(unified_state['agents']['VX-INTENT']['state']['status'], 'active')
        self.assertEqual(unified_state['agents']['VX-PSI']['state']['status'], 'monitoring')
        self.assertFalse(unified_state['active_conflicts'])
    
    def test_emergent_state_creation(self):
        """Testet die Erstellung des EmergentState."""
        # Zustandssynthese durchführen
        states = {
            "VX-INTENT": {
                'state': {'agent_id': 'VX-INTENT', 'status': 'active'},
                'timestamp': time.time(),
                'confidence': 0.95
            }
        }
        unified_state = self.gestalt._synthesize_state(states)
        
        # EmergentState erstellen
        emergent_state = self.gestalt._create_emergent_state(unified_state)
        
        # Überprüfen des erstellten EmergentState
        self.assertIn('state_id', emergent_state)
        self.assertIn('timestamp', emergent_state)
        self.assertIn('unified_state', emergent_state)
        self.assertIn('conflicts', emergent_state)
        self.assertIn('decisions', emergent_state)
        self.assertIn('signature', emergent_state)
    
    def test_full_unification_cycle(self):
        """Testet den vollständigen Vereinheitlichungszyklus."""
        # Unify-Methode aufrufen
        emergent_state = self.gestalt.unify()
        
        # Überprüfen des Rückgabewerts
        self.assertIsInstance(emergent_state, dict)
        self.assertIn('VX-INTENT', emergent_state['unified_state']['agents'])
        self.assertIn('VX-PSI', emergent_state['unified_state']['agents'])
        
        # Überprüfen, ob ein Konflikt erkannt wurde
        self.assertTrue(len(emergent_state['conflicts']) > 0)
        
        # Überprüfen, ob Entscheidungen getroffen wurden
        self.assertIsInstance(emergent_state['decisions'], list)
        
        # Überprüfen, ob Feedback gesendet wurde
        self.assertTrue(len(self.intent_agent.received_feedback) > 0)
    
    @patch('miso.security.void.VOIDVerifier.verify', return_value=True)
    @patch('miso.security.void.VOIDVerifier.sign', return_value="mocked_signature")
    def test_void_integration(self, mock_sign, mock_verify):
        """Testet die Integration mit dem VOID-Protokoll."""
        # Unify-Methode aufrufen
        emergent_state = self.gestalt.unify()
        
        # Überprüfen, ob die VOID-Methoden aufgerufen wurden
        self.assertTrue(mock_verify.called)
        self.assertTrue(mock_sign.called)
        
        # Überprüfen, ob die Signatur im EmergentState gesetzt wurde
        self.assertEqual(emergent_state['signature'], "mocked_signature")
    
    def test_feedback_routing(self):
        """Testet das Routing von Feedback-Nachrichten."""
        # Feedback-Nachricht senden
        feedback = {
            'decision_id': 'test_decision',
            'type': 'test_type',
            'action': 'test_action',
            'parameters': {'test_param': 'test_value'}
        }
        
        # Feedback direkt an den Agenten senden
        self.gestalt._provide_feedback({
            'decisions': [{
                'decision_id': 'test_decision',
                'type': 'test_type',
                'action': 'test_action',
                'parameters': {'test_param': 'test_value'},
                'target_agent': 'VX-INTENT'
            }]
        })
        
        # Verarbeiten der Nachrichten
        self.gestalt.feedback_router.process_messages()
        
        # Überprüfen, ob das Feedback angekommen ist
        self.assertEqual(len(self.intent_agent.received_feedback), 1)
        self.assertEqual(self.intent_agent.received_feedback[0]['type'], 'decision')
        self.assertEqual(self.intent_agent.received_feedback[0]['payload']['action'], 'test_action')

if __name__ == '__main__':
    unittest.main()
