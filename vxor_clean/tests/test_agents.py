#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VXOR Agents Tests - TDD Implementation
Comprehensive tests for VXOR Agent functionality

Copyright (c) 2025 VXOR AI. Alle Rechte vorbehalten.
"""

import unittest
import sys
import time
import threading
from pathlib import Path

# Add vxor_clean to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import (
    BaseAgent, MemoryAgent, ReasoningAgent, CoordinationAgent,
    AgentStatus, AgentMetrics, create_agent, AgentManager, get_agent_manager
)

class TestBaseAgent(unittest.TestCase):
    """Test Base Agent functionality"""
    
    def setUp(self):
        """Setup test environment"""
        self.test_config = {"test_mode": True}
        
    def test_agent_initialization(self):
        """Test agent initialization"""
        # Create concrete implementation for testing
        class TestAgent(BaseAgent):
            def process(self, data):
                return {"result": "test"}
                
        agent = TestAgent("test_agent", self.test_config)
        
        self.assertEqual(agent.name, "test_agent")
        self.assertEqual(agent.config, self.test_config)
        self.assertEqual(agent.status, AgentStatus.INACTIVE)
        self.assertEqual(agent.metrics.tasks_completed, 0)
        
    def test_agent_lifecycle(self):
        """Test agent lifecycle"""
        class TestAgent(BaseAgent):
            def process(self, data):
                return {"result": "test"}
                
        agent = TestAgent("test_agent")
        
        # Test initialization
        result = agent.initialize()
        self.assertTrue(result)
        self.assertEqual(agent.status, AgentStatus.ACTIVE)
        
        # Test shutdown
        result = agent.shutdown()
        self.assertTrue(result)
        self.assertEqual(agent.status, AgentStatus.SHUTDOWN)
        
    def test_agent_status_reporting(self):
        """Test agent status reporting"""
        class TestAgent(BaseAgent):
            def process(self, data):
                return {"result": "test"}
                
        agent = TestAgent("test_agent", self.test_config)
        agent.initialize()
        
        status = agent.get_status()
        
        self.assertEqual(status["name"], "test_agent")
        self.assertEqual(status["status"], AgentStatus.ACTIVE.value)
        self.assertIn("uptime", status)
        self.assertIn("metrics", status)
        self.assertEqual(status["config"], self.test_config)
        
    def test_health_check(self):
        """Test health check functionality"""
        class TestAgent(BaseAgent):
            def process(self, data):
                return {"result": "test"}
                
        agent = TestAgent("test_agent")
        
        # Inactive agent should not be healthy
        self.assertFalse(agent.health_check())
        
        # Active agent should be healthy
        agent.initialize()
        self.assertTrue(agent.health_check())
        
        # Shutdown agent should not be healthy
        agent.shutdown()
        self.assertFalse(agent.health_check())

class TestMemoryAgent(unittest.TestCase):
    """Test Memory Agent functionality"""
    
    def setUp(self):
        """Setup test environment"""
        self.agent = MemoryAgent()
        self.agent.initialize()
        
    def tearDown(self):
        """Cleanup after test"""
        self.agent.shutdown()
        
    def test_memory_storage(self):
        """Test memory storage functionality"""
        # Test store operation
        result = self.agent.process({
            "operation": "store",
            "key": "test_key",
            "value": "test_value"
        })
        
        self.assertTrue(result["success"])
        self.assertEqual(result["key"], "test_key")
        
    def test_memory_retrieval(self):
        """Test memory retrieval functionality"""
        # Store data first
        self.agent.process({
            "operation": "store",
            "key": "test_key",
            "value": "test_value"
        })
        
        # Test retrieve operation
        result = self.agent.process({
            "operation": "retrieve",
            "key": "test_key"
        })
        
        self.assertTrue(result["success"])
        self.assertEqual(result["key"], "test_key")
        self.assertEqual(result["value"], "test_value")
        
    def test_memory_search(self):
        """Test memory search functionality"""
        # Store test data
        self.agent.process({
            "operation": "store",
            "key": "doc1",
            "value": "This is a test document about AI"
        })
        
        self.agent.process({
            "operation": "store",
            "key": "doc2", 
            "value": "Another document about machine learning"
        })
        
        # Test search operation
        result = self.agent.process({
            "operation": "search",
            "query": "test AI"
        })
        
        self.assertTrue(result["success"])
        self.assertIn("doc1", result["results"])
        self.assertGreater(result["count"], 0)
        
    def test_memory_error_handling(self):
        """Test memory error handling"""
        # Test retrieve non-existent key
        result = self.agent.process({
            "operation": "retrieve",
            "key": "non_existent"
        })
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        
        # Test store without key
        result = self.agent.process({
            "operation": "store",
            "value": "test_value"
        })
        
        self.assertIn("error", result)

class TestReasoningAgent(unittest.TestCase):
    """Test Reasoning Agent functionality"""
    
    def setUp(self):
        """Setup test environment"""
        self.agent = ReasoningAgent()
        self.agent.initialize()
        
    def tearDown(self):
        """Cleanup after test"""
        self.agent.shutdown()
        
    def test_rule_addition(self):
        """Test rule addition functionality"""
        rule = {
            "conditions": ["A", "B"],
            "conclusion": "C"
        }
        
        result = self.agent.process({
            "operation": "add_rule",
            "rule": rule
        })
        
        self.assertTrue(result["success"])
        self.assertEqual(result["rules_count"], 1)
        
    def test_inference(self):
        """Test inference functionality"""
        # Add rule first
        rule = {
            "conditions": ["A", "B"],
            "conclusion": "C"
        }
        
        self.agent.process({
            "operation": "add_rule",
            "rule": rule
        })
        
        # Test inference with matching facts
        result = self.agent.process({
            "operation": "infer",
            "facts": ["A", "B", "D"]
        })
        
        self.assertTrue(result["success"])
        self.assertIn("C", result["inferences"])
        
    def test_logic_validation(self):
        """Test logic validation functionality"""
        # Test valid statement
        result = self.agent.process({
            "operation": "validate",
            "statement": "A and B"
        })
        
        self.assertTrue(result["valid"])
        
        # Test invalid statement
        result = self.agent.process({
            "operation": "validate",
            "statement": "true and false"
        })
        
        self.assertFalse(result["valid"])
        self.assertIn("error", result)

class TestCoordinationAgent(unittest.TestCase):
    """Test Coordination Agent functionality"""
    
    def setUp(self):
        """Setup test environment"""
        self.coordinator = CoordinationAgent()
        self.coordinator.initialize()
        
        # Create test agents
        self.memory_agent = MemoryAgent()
        self.memory_agent.initialize()
        
        self.reasoning_agent = ReasoningAgent()
        self.reasoning_agent.initialize()
        
    def tearDown(self):
        """Cleanup after test"""
        self.coordinator.shutdown()
        self.memory_agent.shutdown()
        self.reasoning_agent.shutdown()
        
    def test_agent_registration(self):
        """Test agent registration"""
        result = self.coordinator.process({
            "operation": "register_agent",
            "agent": self.memory_agent
        })
        
        self.assertTrue(result["success"])
        self.assertEqual(result["agent"], "VX-MEMEX")
        
    def test_task_coordination(self):
        """Test task coordination"""
        # Register agents first
        self.coordinator.process({
            "operation": "register_agent",
            "agent": self.memory_agent
        })
        
        # Coordinate memory task
        result = self.coordinator.process({
            "operation": "coordinate",
            "task": {
                "type": "memory",
                "data": {
                    "operation": "store",
                    "key": "test",
                    "value": "coordinated_value"
                }
            }
        })
        
        self.assertTrue(result["success"])
        self.assertEqual(result["agent"], "VX-MEMEX")
        self.assertIn("result", result)
        
    def test_system_status(self):
        """Test system status reporting"""
        # Register agents
        self.coordinator.process({
            "operation": "register_agent",
            "agent": self.memory_agent
        })
        
        self.coordinator.process({
            "operation": "register_agent", 
            "agent": self.reasoning_agent
        })
        
        # Get system status
        result = self.coordinator.process({
            "operation": "get_status"
        })
        
        self.assertTrue(result["success"])
        self.assertIn("agents", result)
        self.assertEqual(len(result["agents"]), 2)

class TestAgentFactory(unittest.TestCase):
    """Test Agent Factory functionality"""
    
    def test_create_memory_agent(self):
        """Test memory agent creation"""
        agent = create_agent("memory")
        
        self.assertIsInstance(agent, MemoryAgent)
        self.assertEqual(agent.name, "VX-MEMEX")
        
    def test_create_reasoning_agent(self):
        """Test reasoning agent creation"""
        agent = create_agent("reasoning")
        
        self.assertIsInstance(agent, ReasoningAgent)
        self.assertEqual(agent.name, "VX-REASON")
        
    def test_create_coordination_agent(self):
        """Test coordination agent creation"""
        agent = create_agent("coordination")
        
        self.assertIsInstance(agent, CoordinationAgent)
        self.assertEqual(agent.name, "VX-NEXUS")
        
    def test_create_invalid_agent(self):
        """Test invalid agent creation"""
        agent = create_agent("invalid_type")
        
        self.assertIsNone(agent)

class TestAgentManager(unittest.TestCase):
    """Test Agent Manager functionality"""
    
    def setUp(self):
        """Setup test environment"""
        self.manager = AgentManager()
        
    def tearDown(self):
        """Cleanup after test"""
        self.manager.shutdown_all()
        
    def test_create_and_register_agent(self):
        """Test agent creation and registration"""
        result = self.manager.create_and_register_agent("memory")
        
        self.assertTrue(result)
        
        agent = self.manager.get_agent("VX-MEMEX")
        self.assertIsNotNone(agent)
        self.assertIsInstance(agent, MemoryAgent)
        
    def test_get_all_agents(self):
        """Test getting all agents"""
        self.manager.create_and_register_agent("memory")
        self.manager.create_and_register_agent("reasoning")
        
        agents = self.manager.get_all_agents()
        
        self.assertEqual(len(agents), 2)
        self.assertIn("VX-MEMEX", agents)
        self.assertIn("VX-REASON", agents)
        
    def test_shutdown_all(self):
        """Test shutting down all agents"""
        self.manager.create_and_register_agent("memory")
        self.manager.create_and_register_agent("reasoning")
        
        # Verify agents exist
        self.assertEqual(len(self.manager.get_all_agents()), 2)
        
        # Shutdown all
        self.manager.shutdown_all()
        
        # Verify agents are gone
        self.assertEqual(len(self.manager.get_all_agents()), 0)

class TestAgentIntegration(unittest.TestCase):
    """Integration tests for VXOR Agents"""
    
    def test_multi_agent_workflow(self):
        """Test complete multi-agent workflow"""
        manager = AgentManager()
        
        try:
            # Create agents
            self.assertTrue(manager.create_and_register_agent("memory"))
            self.assertTrue(manager.create_and_register_agent("reasoning"))
            self.assertTrue(manager.create_and_register_agent("coordination"))
            
            # Get agents
            memory_agent = manager.get_agent("VX-MEMEX")
            reasoning_agent = manager.get_agent("VX-REASON")
            coordinator = manager.get_agent("VX-NEXUS")
            
            # Register agents with coordinator
            coordinator.process({
                "operation": "register_agent",
                "agent": memory_agent
            })
            
            coordinator.process({
                "operation": "register_agent",
                "agent": reasoning_agent
            })
            
            # Store knowledge via coordinator
            result = coordinator.process({
                "operation": "coordinate",
                "task": {
                    "type": "memory",
                    "data": {
                        "operation": "store",
                        "key": "fact1",
                        "value": "The sky is blue"
                    }
                }
            })
            
            self.assertTrue(result["success"])
            
            # Add reasoning rule via coordinator
            result = coordinator.process({
                "operation": "coordinate",
                "task": {
                    "type": "reasoning",
                    "data": {
                        "operation": "add_rule",
                        "rule": {
                            "conditions": ["sky_is_blue"],
                            "conclusion": "weather_is_clear"
                        }
                    }
                }
            })
            
            self.assertTrue(result["success"])
            
            # Get system status
            result = coordinator.process({
                "operation": "get_status"
            })
            
            self.assertTrue(result["success"])
            self.assertEqual(len(result["agents"]), 2)
            
        finally:
            manager.shutdown_all()

if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during tests
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    unittest.main(verbosity=2)
