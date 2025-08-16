"""
Zero-Trust Monitoring (ZTM) Core Module

This module implements the core functionality of the Zero-Trust Monitoring system,
including policy enforcement, activity monitoring, and anomaly detection.
"""

import os
import sys
import time
import json
import logging
import threading
import importlib
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum, auto

# Configure logging
logger = logging.getLogger('ztm_core')

class Severity(Enum):
    """Severity levels for security events."""
    DEBUG = auto()
    INFO = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

@dataclass
class SecurityEvent:
    """Represents a security event captured by the ZTM system."""
    event_id: str
    timestamp: float
    source: str
    event_type: str
    severity: Severity
    details: Dict[str, Any]
    user: Optional[str] = None
    ip_address: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a dictionary."""
        result = asdict(self)
        result['severity'] = self.severity.name
        result['timestamp'] = datetime.fromtimestamp(self.timestamp).isoformat()
        return result

class ZeroTrustMonitor:
    """Core class for Zero-Trust Monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Zero-Trust Monitor.
        
        Args:
            config: Configuration dictionary loaded from ztm_config.yaml
        """
        self.config = config
        self.running = False
        self.modules: Dict[str, Any] = {}
        self.integrations: Dict[str, Any] = {}
        self.policies = self._load_policies()
        self.event_queue = []
        self.event_lock = threading.Lock()
        self.worker_thread = None
        
        # Initialize metrics
        self.metrics = {
            'events_processed': 0,
            'events_blocked': 0,
            'alerts_triggered': 0,
            'last_event_time': 0,
        }
        
        logger.info("Zero-Trust Monitor initialized")
    
    def _load_policies(self) -> Dict[str, Any]:
        """Load security policies from configuration."""
        policies = {}
        
        # Load access control policies
        if 'access_control' in self.config.get('policies', {}):
            policies['access_control'] = self.config['policies']['access_control']
        
        # Load data protection policies
        if 'data_protection' in self.config.get('policies', {}):
            policies['data_protection'] = self.config['policies']['data_protection']
        
        # Load custom rules
        if 'custom_rules' in self.config:
            policies['custom_rules'] = self.config['custom_rules']
        
        logger.info(f"Loaded {len(policies)} policy categories")
        return policies
    
    def register_integration(self, name: str, integration: Any) -> None:
        """Register an integration with the ZTM system.
        
        Args:
            name: Name of the integration (e.g., 'void', 'prometheus')
            integration: Integration instance
        """
        self.integrations[name] = integration
        logger.info(f"Registered integration: {name}")
    
    def monitor_module(self, module_path: str, priority: str = 'medium') -> bool:
        """Start monitoring a Python module.
        
        Args:
            module_path: Dotted path to the module (e.g., 'miso.math.t_mathematics')
            priority: Monitoring priority ('low', 'medium', 'high')
            
        Returns:
            bool: True if monitoring was successfully enabled
        """
        try:
            # Import the module
            module = importlib.import_module(module_path)
            
            # Store module reference
            self.modules[module_path] = {
                'module': module,
                'priority': priority,
                'hooked': False
            }
            
            # Apply monitoring hooks
            self._apply_hooks(module_path)
            
            logger.info(f"Started monitoring module: {module_path} (priority: {priority})")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to monitor module {module_path}: {e}")
            return False
    
    def _apply_hooks(self, module_path: str) -> None:
        """Apply monitoring hooks to a module.
        
        Args:
            module_path: Dotted path to the module
        """
        if module_path not in self.modules or self.modules[module_path]['hooked']:
            return
        
        module = self.modules[module_path]['module']
        
        # TODO: Implement method wrapping for monitoring
        # This is a placeholder for the actual hooking mechanism
        
        self.modules[module_path]['hooked'] = True
        logger.debug(f"Applied monitoring hooks to: {module_path}")
    
    def log_event(self, event: SecurityEvent) -> None:
        """Log a security event.
        
        Args:
            event: SecurityEvent instance
        """
        with self.event_lock:
            self.event_queue.append(event)
        
        # Update metrics
        self.metrics['events_processed'] += 1
        self.metrics['last_event_time'] = time.time()
        
        # Process high-severity events immediately
        if event.severity in [Severity.HIGH, Severity.CRITICAL]:
            self._process_event(event)
    
    def _process_events(self) -> None:
        """Process events from the queue."""
        while self.running:
            # Get events from the queue
            with self.event_lock:
                events = self.event_queue[:]
                self.event_queue = []
            
            # Process each event
            for event in events:
                self._process_event(event)
            
            # Sleep briefly to avoid high CPU usage
            time.sleep(0.1)
    
    def _process_event(self, event: SecurityEvent) -> None:
        """Process a single security event.
        
        Args:
            event: SecurityEvent instance
        """
        try:
            # Apply policies
            action = self._evaluate_policies(event)
            
            # Take action based on policy evaluation
            if action == 'block':
                self.metrics['events_blocked'] += 1
                logger.warning(f"Blocked event: {event.event_type} (ID: {event.event_id})")
            elif action == 'alert':
                self.metrics['alerts_triggered'] += 1
                logger.warning(f"Alert triggered: {event.event_type} (ID: {event.event_id})")
            
            # Forward to integrations
            self._forward_to_integrations(event)
            
        except Exception as e:
            logger.error(f"Error processing event {event.event_id}: {e}", exc_info=True)
    
    def _evaluate_policies(self, event: SecurityEvent) -> str:
        """Evaluate policies against an event.
        
        Args:
            event: SecurityEvent instance
            
        Returns:
            str: Action to take ('allow', 'block', 'alert')
        """
        # Default action is to allow
        action = 'allow'
        
        # Check custom rules
        for rule in self.policies.get('custom_rules', []):
            try:
                # Simple condition evaluation (in a real implementation, this would use a proper expression evaluator)
                condition_met = eval(rule['condition'], {
                    'event': event,
                    'severity': event.severity,
                    'Severity': Severity,
                    # Add more context as needed
                })
                
                if condition_met:
                    action = rule.get('action', 'alert')
                    logger.info(f"Rule '{rule['name']}' matched, action: {action}")
                    break
                    
            except Exception as e:
                logger.error(f"Error evaluating rule '{rule.get('name', 'unknown')}': {e}")
        
        return action
    
    def _forward_to_integrations(self, event: SecurityEvent) -> None:
        """Forward an event to all registered integrations.
        
        Args:
            event: SecurityEvent instance
        """
        for name, integration in self.integrations.items():
            try:
                if hasattr(integration, 'process_event'):
                    integration.process_event(event)
            except Exception as e:
                logger.error(f"Error forwarding event to integration {name}: {e}", exc_info=True)
    
    def start(self) -> None:
        """Start the ZTM system."""
        if self.running:
            logger.warning("ZTM system is already running")
            return
        
        self.running = True
        
        # Start event processing thread
        self.worker_thread = threading.Thread(
            target=self._process_events,
            name="ZTM-EventProcessor",
            daemon=True
        )
        self.worker_thread.start()
        
        logger.info("ZTM system started")
    
    def stop(self) -> None:
        """Stop the ZTM system."""
        if not self.running:
            return
        
        self.running = False
        
        # Wait for worker thread to finish
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        
        logger.info("ZTM system stopped")
    
    def shutdown(self) -> None:
        """Shut down the ZTM system and clean up resources."""
        self.stop()
        
        # Clean up integrations
        for name, integration in self.integrations.items():
            try:
                if hasattr(integration, 'shutdown'):
                    integration.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down integration {name}: {e}", exc_info=True)
        
        logger.info("ZTM system shutdown complete")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics.
        
        Returns:
            Dict containing system metrics
        """
        return {
            **self.metrics,
            'modules_monitored': len(self.modules),
            'integrations_active': len(self.integrations),
            'events_queued': len(self.event_queue),
            'uptime_seconds': time.time() - self.metrics.get('start_time', time.time())
        }
