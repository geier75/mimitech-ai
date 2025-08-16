#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for ZTM (Zero-Trust Monitoring) integration.

This script tests the basic functionality of the ZTM system,
including the VOID protocol integration.
"""

import os
import sys
import time
import json
import logging
import unittest
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules to test
from miso.security.ztm.core import ZeroTrustMonitor, SecurityEvent, Severity
from miso.security.ztm.integrations import VOIDIntegration, VOIDVerificationResult

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_ztm.log')
    ]
)
logger = logging.getLogger('test_ztm')

class TestZTMIntegration(unittest.TestCase):
    """Test cases for ZTM integration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before any tests are run."""
        # Create a test configuration
        cls.test_config = {
            'core': {
                'enabled': True,
                'log_level': 'DEBUG',
                'audit_log': 'test_audit.log',
                'max_log_size_mb': 1,
                'max_log_backups': 3
            },
            'policies': {
                'access_control': {
                    'enforce_least_privilege': True,
                    'require_mfa': False,
                    'session_timeout_minutes': 30
                },
                'data_protection': {
                    'encrypt_sensitive_data': True,
                    'encryption_algorithm': 'AES-256-GCM',
                    'key_rotation_days': 30
                }
            },
            'custom_rules': [
                {
                    'name': 'test_rule',
                    'description': 'Test rule that always matches',
                    'condition': 'True',
                    'action': 'alert'
                }
            ],
            'integrations': {
                'void_protocol': {
                    'enabled': True,
                    'config_path': 'test_void_config.yaml',
                    'verify_interval_seconds': 60
                }
            }
        }
        
        # Create a test VOID config file
        cls.void_config = {
            'crypto': {
                'symmetric': {
                    'algorithm': 'AES-256-GCM',
                    'key_size': 32,
                    'iv_size': 12,
                    'auth_tag_size': 16
                },
                'asymmetric': {
                    'algorithm': 'RSA',
                    'key_size': 2048,
                    'padding': 'OAEP',
                    'hash_algorithm': 'SHA-256'
                },
                'hashing': {
                    'algorithm': 'SHA3-256',
                    'salt_size': 16,
                    'iterations': 100000
                },
                'key_management': {
                    'key_rotation_days': 30,
                    'max_key_age_days': 90,
                    'key_storage': 'local'
                }
            },
            'verification': {
                'enforce_timestamps': True,
                'max_clock_skew_seconds': 30,
                'require_nonce': True,
                'nonce_size': 16,
                'signature': {
                    'algorithm': 'ECDSA',
                    'curve': 'P-256',
                    'hash_algorithm': 'SHA-256'
                }
            },
            'key_storage': {
                'local': {
                    'enabled': True,
                    'private_key_path': 'test_private_key.pem',
                    'public_key_path': 'test_public_key.pem'
                }
            }
        }
        
        # Write the test VOID config to a file
        with open('test_void_config.yaml', 'w') as f:
            import yaml
            yaml.safe_dump(cls.void_config, f)
    
    def setUp(self):
        """Set up test fixtures before each test method is called."""
        # Create a new ZTM instance for each test
        self.ztm = ZeroTrustMonitor(self.test_config)
        
        # Create a mock VOID integration
        self.void_integration = VOIDIntegration(self.test_config)
        self.ztm.register_integration('void', self.void_integration)
        
        # Start the ZTM system
        self.ztm.start()
    
    def tearDown(self):
        """Tear down test fixtures after each test method is called."""
        # Stop the ZTM system
        self.ztm.shutdown()
        
        # Clean up any test files
        if os.path.exists('test_audit.log'):
            os.remove('test_audit.log')
        if os.path.exists('test_void_config.yaml'):
            os.remove('test_void_config.yaml')
    
    @classmethod
    def tearDownClass(cls):
        """Tear down test fixtures after all tests are run."""
        # Clean up any remaining test files
        if os.path.exists('test_void_config.yaml'):
            os.remove('test_void_config.yaml')
        if os.path.exists('test_private_key.pem'):
            os.remove('test_private_key.pem')
        if os.path.exists('test_public_key.pem'):
            os.remove('test_public_key.pem')
    
    def test_ztm_initialization(self):
        """Test that ZTM initializes correctly."""
        self.assertIsNotNone(self.ztm)
        self.assertTrue(hasattr(self.ztm, 'start'))
        self.assertTrue(hasattr(self.ztm, 'stop'))
        self.assertTrue(hasattr(self.ztm, 'log_event'))
    
    def test_void_integration_initialization(self):
        """Test that VOID integration initializes correctly."""
        self.assertIsNotNone(self.void_integration)
        self.assertTrue(hasattr(self.void_integration, 'process_event'))
        self.assertTrue(hasattr(self.void_integration, 'verify_operation'))
    
    def test_security_event_creation(self):
        """Test creation of security events."""
        event = SecurityEvent(
            event_id='test_event_123',
            timestamp=time.time(),
            source='test',
            event_type='test_event',
            severity=Severity.INFO,
            details={'message': 'Test event'}
        )
        
        self.assertEqual(event.event_id, 'test_event_123')
        self.assertEqual(event.event_type, 'test_event')
        self.assertEqual(event.severity, Severity.INFO)
        self.assertIn('message', event.details)
    
    def test_event_processing(self):
        """Test processing of security events."""
        # Create a test event
        event = SecurityEvent(
            event_id='test_event_456',
            timestamp=time.time(),
            source='test',
            event_type='test_event',
            severity=Severity.INFO,
            details={'message': 'Test event for processing'}
        )
        
        # Log the event
        self.ztm.log_event(event)
        
        # Give the event processor some time to process the event
        time.sleep(0.5)
        
        # Check metrics to verify the event was processed
        metrics = self.ztm.get_metrics()
        self.assertGreaterEqual(metrics['events_processed'], 1)
    
    def test_high_severity_event(self):
        """Test handling of high-severity events."""
        # Create a high-severity event
        event = SecurityEvent(
            event_id='test_event_789',
            timestamp=time.time(),
            source='test',
            event_type='sensitive_operation',
            severity=Severity.HIGH,
            details={
                'operation': 'user_login',
                'user_id': 'testuser',
                'ip_address': '192.168.1.1'
            }
        )
        
        # Log the event
        self.ztm.log_event(event)
        
        # Give the event processor some time to process the event
        time.sleep(0.5)
        
        # Check metrics to verify the event was processed
        metrics = self.ztm.get_metrics()
        self.assertGreaterEqual(metrics['events_processed'], 1)
    
    @patch('miso.security.ztm.integrations.void_integration.VOIDIntegration.verify_operation')
    def test_void_verification(self, mock_verify):
        """Test VOID verification of operations."""
        # Set up the mock
        mock_verify.return_value = VOIDVerificationResult(
            is_valid=True,
            timestamp=time.time(),
            operation='test_operation',
            verified_by='test',
            details={}
        )
        
        # Create a test event that should trigger verification
        event = SecurityEvent(
            event_id='test_event_verification',
            timestamp=time.time(),
            source='test',
            event_type='sensitive_operation',
            severity=Severity.HIGH,
            details={
                'operation': 'admin_action',
                'user_id': 'admin',
                'action': 'delete_user',
                'target_user': 'testuser'
            }
        )
        
        # Log the event
        self.ztm.log_event(event)
        
        # Give the event processor some time to process the event
        time.sleep(0.5)
        
        # Verify that verify_operation was called
        mock_verify.assert_called_once()
        
        # Check the call arguments
        args, kwargs = mock_verify.call_args
        self.assertEqual(kwargs['operation'], 'admin_action')
        self.assertTrue(kwargs.get('require_mfa', False))

if __name__ == '__main__':
    unittest.main()
