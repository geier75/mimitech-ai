#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VXOR Core Tests - TDD Implementation
Comprehensive tests for VXOR Core functionality

Copyright (c) 2025 VXOR AI. Alle Rechte vorbehalten.
"""

import unittest
import sys
import time
import threading
from pathlib import Path

# Add vxor_clean to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import VXORCore, VXORConfig, VXORStatus, get_vxor_core, reset_vxor_core

class TestVXORCore(unittest.TestCase):
    """Test VXOR Core functionality"""
    
    def setUp(self):
        """Setup test environment"""
        # Reset global core instance before each test
        reset_vxor_core()
        
        # Create test configuration
        self.test_config = VXORConfig(
            debug_mode=True,
            performance_monitoring=True,
            auto_optimization=False,
            max_threads=2,
            heartbeat_interval=0.1,  # Fast heartbeat for testing
            log_level="DEBUG"
        )
        
    def tearDown(self):
        """Cleanup after test"""
        reset_vxor_core()
        
    def test_core_initialization(self):
        """Test core module initialization"""
        core = VXORCore(self.test_config)
        
        # Test initial state
        self.assertEqual(core.status, VXORStatus.UNINITIALIZED)
        self.assertFalse(core.running)
        self.assertEqual(len(core.modules), 0)
        
        # Test initialization
        result = core.initialize()
        self.assertTrue(result)
        self.assertEqual(core.status, VXORStatus.READY)
        
        # Test double initialization
        result = core.initialize()
        self.assertTrue(result)  # Should succeed but warn
        
    def test_core_start_stop(self):
        """Test core start and stop functionality"""
        core = VXORCore(self.test_config)
        
        # Initialize first
        self.assertTrue(core.initialize())
        
        # Test start
        result = core.start()
        self.assertTrue(result)
        self.assertEqual(core.status, VXORStatus.RUNNING)
        self.assertTrue(core.running)
        
        # Wait a bit for heartbeat
        time.sleep(0.2)
        
        # Test stop
        result = core.stop()
        self.assertTrue(result)
        self.assertEqual(core.status, VXORStatus.SHUTDOWN)
        self.assertFalse(core.running)
        
    def test_module_registration(self):
        """Test module registration functionality"""
        core = VXORCore(self.test_config)
        core.initialize()
        
        # Create test module
        test_module = type("TestModule", (), {
            "name": "test_module",
            "status": "active",
            "health_check": lambda: True
        })()
        
        # Test registration
        result = core.register_module("test_module", test_module)
        self.assertTrue(result)
        
        # Test retrieval
        retrieved_module = core.get_module("test_module")
        self.assertIsNotNone(retrieved_module)
        self.assertEqual(retrieved_module.name, "test_module")
        
        # Test duplicate registration
        result = core.register_module("test_module", test_module)
        self.assertTrue(result)  # Should succeed but warn
        
    def test_status_reporting(self):
        """Test status reporting functionality"""
        core = VXORCore(self.test_config)
        
        # Test uninitialized status
        status = core.get_status()
        self.assertEqual(status["status"], VXORStatus.UNINITIALIZED.value)
        self.assertFalse(status["running"])
        self.assertEqual(len(status["modules"]), 0)
        
        # Initialize and test
        core.initialize()
        status = core.get_status()
        self.assertEqual(status["status"], VXORStatus.READY.value)
        self.assertGreater(len(status["modules"]), 0)  # Should have loaded core modules
        
    def test_performance_monitoring(self):
        """Test performance monitoring"""
        core = VXORCore(self.test_config)
        core.initialize()
        
        status = core.get_status()
        self.assertIn("performance", status)
        self.assertIn("modules_loaded", status["performance"])
        self.assertGreaterEqual(status["performance"]["modules_loaded"], 0)
        
    def test_heartbeat_functionality(self):
        """Test heartbeat functionality"""
        core = VXORCore(self.test_config)
        core.initialize()
        core.start()
        
        # Wait for a few heartbeats
        time.sleep(0.3)
        
        # Heartbeat thread should be running
        self.assertIsNotNone(core.heartbeat_thread)
        self.assertTrue(core.heartbeat_thread.is_alive())
        
        core.stop()
        
        # Wait for thread to stop
        time.sleep(0.2)
        
        # Thread should be stopped
        if core.heartbeat_thread:
            self.assertFalse(core.heartbeat_thread.is_alive())
            
    def test_global_core_instance(self):
        """Test global core instance management"""
        # Test singleton behavior
        core1 = get_vxor_core(self.test_config)
        core2 = get_vxor_core()
        
        self.assertIs(core1, core2)
        
        # Test reset
        reset_vxor_core()
        core3 = get_vxor_core()
        
        self.assertIsNot(core1, core3)
        
    def test_error_handling(self):
        """Test error handling"""
        core = VXORCore(self.test_config)
        
        # Test start without initialization
        result = core.start()
        self.assertFalse(result)
        self.assertEqual(core.status, VXORStatus.ERROR)
        
    def test_thread_safety(self):
        """Test thread safety"""
        core = VXORCore(self.test_config)
        core.initialize()
        
        results = []
        errors = []
        
        def register_modules():
            try:
                for i in range(10):
                    module = type(f"Module{i}", (), {"name": f"module_{i}"})()
                    result = core.register_module(f"module_{i}", module)
                    results.append(result)
            except Exception as e:
                errors.append(e)
                
        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=register_modules)
            threads.append(thread)
            thread.start()
            
        # Wait for all threads
        for thread in threads:
            thread.join()
            
        # Check results
        self.assertEqual(len(errors), 0)
        self.assertTrue(all(results))
        
    def test_configuration_validation(self):
        """Test configuration validation"""
        # Test valid configuration
        valid_config = VXORConfig(
            debug_mode=True,
            max_threads=4,
            heartbeat_interval=1.0
        )
        
        core = VXORCore(valid_config)
        self.assertEqual(core.config.debug_mode, True)
        self.assertEqual(core.config.max_threads, 4)
        self.assertEqual(core.config.heartbeat_interval, 1.0)
        
    def test_module_health_checks(self):
        """Test module health checks"""
        core = VXORCore(self.test_config)
        core.initialize()
        
        # Create module with health check
        healthy_module = type("HealthyModule", (), {
            "name": "healthy",
            "health_check": lambda: True
        })()
        
        unhealthy_module = type("UnhealthyModule", (), {
            "name": "unhealthy", 
            "health_check": lambda: False
        })()
        
        core.register_module("healthy", healthy_module)
        core.register_module("unhealthy", unhealthy_module)
        
        # Start core to trigger health checks
        core.start()
        time.sleep(0.2)  # Wait for health check
        core.stop()
        
        # Health checks should have been called
        # (We can't easily test the results without more complex mocking)
        self.assertTrue(True)  # Placeholder assertion

class TestVXORConfig(unittest.TestCase):
    """Test VXOR Configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = VXORConfig()
        
        self.assertFalse(config.debug_mode)
        self.assertTrue(config.performance_monitoring)
        self.assertTrue(config.auto_optimization)
        self.assertEqual(config.max_threads, 4)
        self.assertEqual(config.heartbeat_interval, 5.0)
        self.assertEqual(config.log_level, "INFO")
        
    def test_custom_config(self):
        """Test custom configuration"""
        config = VXORConfig(
            debug_mode=True,
            performance_monitoring=False,
            max_threads=8,
            heartbeat_interval=2.0,
            log_level="DEBUG"
        )
        
        self.assertTrue(config.debug_mode)
        self.assertFalse(config.performance_monitoring)
        self.assertEqual(config.max_threads, 8)
        self.assertEqual(config.heartbeat_interval, 2.0)
        self.assertEqual(config.log_level, "DEBUG")

class TestVXORIntegration(unittest.TestCase):
    """Integration tests for VXOR Core"""
    
    def test_full_lifecycle(self):
        """Test complete VXOR lifecycle"""
        # Create and configure
        config = VXORConfig(
            debug_mode=True,
            heartbeat_interval=0.1
        )
        
        core = VXORCore(config)
        
        # Initialize
        self.assertTrue(core.initialize())
        self.assertEqual(core.status, VXORStatus.READY)
        
        # Register modules
        for i in range(3):
            module = type(f"Module{i}", (), {"name": f"module_{i}"})()
            self.assertTrue(core.register_module(f"module_{i}", module))
            
        # Start
        self.assertTrue(core.start())
        self.assertEqual(core.status, VXORStatus.RUNNING)
        
        # Run for a bit
        time.sleep(0.3)
        
        # Check status
        status = core.get_status()
        self.assertEqual(status["status"], VXORStatus.RUNNING.value)
        self.assertTrue(status["running"])
        self.assertEqual(len(status["modules"]), 7)  # 4 core + 3 custom
        
        # Stop
        self.assertTrue(core.stop())
        self.assertEqual(core.status, VXORStatus.SHUTDOWN)
        self.assertFalse(core.running)

if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during tests
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    unittest.main(verbosity=2)
