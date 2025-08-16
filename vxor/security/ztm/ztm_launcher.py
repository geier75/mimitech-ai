#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZTM (Zero-Trust Monitoring) Launcher

This script initializes and starts the Zero-Trust Monitoring system.
"""

import os
import sys
import time
import signal
import logging
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ztm_launcher.log')
    ]
)
logger = logging.getLogger('ztm_launcher')

class ZTMLauncher:
    """ZTM Launcher for initializing and managing the Zero-Trust Monitoring system."""
    
    def __init__(self, config_path: str = None):
        """Initialize the ZTM Launcher.
        
        Args:
            config_path: Path to the ZTM configuration file.
        """
        self.config = self._load_config(config_path)
        self.running = False
        self.processes = {}
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        
        # Set up logging
        self._setup_logging()
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load the ZTM configuration.
        
        Args:
            config_path: Path to the configuration file.
            
        Returns:
            Dictionary containing the configuration.
        """
        default_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'ztm', 'config', 'ztm_config.yaml'
        )
        
        config_path = config_path or os.environ.get('ZTM_CONFIG_PATH', default_config_path)
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            sys.exit(1)
    
    def _setup_logging(self) -> None:
        """Configure logging based on the configuration."""
        log_level = self.config.get('core', {}).get('log_level', 'INFO')
        log_file = self.config.get('core', {}).get('audit_log', 'ztm_audit.log')
        
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # Add file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        logger.info(f"Logging configured. Log level: {log_level}, File: {log_file}")
    
    def _handle_signal(self, signum, frame) -> None:
        """Handle termination signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def start(self) -> None:
        """Start the ZTM system."""
        if self.running:
            logger.warning("ZTM system is already running")
            return
        
        logger.info("Starting ZTM system...")
        
        try:
            # Initialize components
            self._initialize_components()
            
            # Start monitoring
            self._start_monitoring()
            
            self.running = True
            logger.info("ZTM system started successfully")
            
            # Keep the main thread alive
            while self.running:
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Failed to start ZTM system: {e}", exc_info=True)
            self.stop()
            sys.exit(1)
    
    def _initialize_components(self) -> None:
        """Initialize ZTM components."""
        logger.info("Initializing ZTM components...")
        
        # Import components here to avoid circular imports
        from miso.security.ztm.core.ztm_core import ZeroTrustMonitor
        from miso.security.ztm.integrations.void_integration import VOIDIntegration
        
        # Initialize core components
        self.ztm = ZeroTrustMonitor(self.config)
        self.void_integration = VOIDIntegration(self.config)
        
        # Register components
        self.ztm.register_integration('void', self.void_integration)
        
        logger.info("ZTM components initialized")
    
    def _start_monitoring(self) -> None:
        """Start monitoring system activities."""
        logger.info("Starting system monitoring...")
        
        # Start monitoring critical modules
        for module in self.config.get('modules', {}).get('critical', []):
            self.ztm.monitor_module(module, priority='high')
        
        # Start monitoring standard modules
        for module in self.config.get('modules', {}).get('standard', []):
            self.ztm.monitor_module(module, priority='medium')
        
        # Start monitoring external modules
        for module in self.config.get('modules', {}).get('external', []):
            self.ztm.monitor_module(module, priority='low')
        
        logger.info("System monitoring started")
    
    def stop(self) -> None:
        """Stop the ZTM system."""
        if not self.running:
            return
        
        logger.info("Stopping ZTM system...")
        
        try:
            # Stop monitoring
            if hasattr(self, 'ztm'):
                self.ztm.shutdown()
            
            # Stop all child processes
            for name, process in self.processes.items():
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except Exception as e:
                    logger.warning(f"Error stopping process {name}: {e}")
            
            self.running = False
            logger.info("ZTM system stopped")
            
        except Exception as e:
            logger.error(f"Error during ZTM shutdown: {e}", exc_info=True)
            raise

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ZTM Launcher')
    parser.add_argument(
        '-c', '--config',
        dest='config_path',
        help='Path to the ZTM configuration file',
        default=None
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        launcher = ZTMLauncher(args.config_path)
        launcher.start()
    except KeyboardInterrupt:
        logger.info("ZTM launcher stopped by user")
    except Exception as e:
        logger.error(f"ZTM launcher failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
