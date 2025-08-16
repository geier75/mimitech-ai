#!/usr/bin/env python3
"""
MISO Port Configuration - Environment-based Port Management
==========================================================

Centralized port configuration to prevent conflicts between services.
All ports are configurable via environment variables with sensible defaults.
"""

import os
import socket
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PortConfig:
    """Centralized port configuration with conflict detection."""
    
    # Default port assignments
    DEFAULT_PORTS = {
        "BACKEND_API": 8000,
        "DASHBOARD": 8080,  # Changed from 5151 to avoid conflicts
        "WEBSOCKET": 8001,
        "MONITORING": 8002,
        "DEBUG_SERVER": 8003,
        "BENCHMARK_API": 8004,
        "METRICS": 9090,  # Prometheus standard
        "HEALTH_CHECK": 8005
    }
    
    def __init__(self):
        self.ports = {}
        self._load_configuration()
    
    def _load_configuration(self) -> None:
        """Load port configuration from environment variables."""
        for service, default_port in self.DEFAULT_PORTS.items():
            env_var = f"MISO_{service}_PORT"
            configured_port = os.environ.get(env_var, default_port)
            
            try:
                self.ports[service] = int(configured_port)
            except ValueError:
                logger.warning(f"Invalid port value for {env_var}: {configured_port}, using default {default_port}")
                self.ports[service] = default_port
    
    def get_port(self, service: str) -> int:
        """Get configured port for a service."""
        if service not in self.ports:
            raise ValueError(f"Unknown service: {service}")
        return self.ports[service]
    
    def is_port_available(self, port: int) -> bool:
        """Check if a port is available for binding."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return True
        except OSError:
            return False
    
    def find_available_port(self, start_port: int = 8000, max_attempts: int = 100) -> int:
        """Find next available port starting from start_port."""
        for port in range(start_port, start_port + max_attempts):
            if self.is_port_available(port):
                return port
        raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_attempts}")
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate port configuration and check for conflicts."""
        validation_result = {
            "valid": True,
            "conflicts": [],
            "unavailable": [],
            "services": {}
        }
        
        # Check for duplicate port assignments
        used_ports = {}
        for service, port in self.ports.items():
            if port in used_ports:
                validation_result["conflicts"].append({
                    "port": port,
                    "services": [used_ports[port], service]
                })
                validation_result["valid"] = False
            else:
                used_ports[port] = service
        
        # Check port availability
        for service, port in self.ports.items():
            available = self.is_port_available(port)
            validation_result["services"][service] = {
                "port": port,
                "available": available
            }
            
            if not available:
                validation_result["unavailable"].append({
                    "service": service,
                    "port": port
                })
                validation_result["valid"] = False
        
        return validation_result
    
    def auto_resolve_conflicts(self) -> Dict[str, int]:
        """Automatically resolve port conflicts by finding alternatives."""
        validation = self.validate_configuration()
        changes = {}
        
        if not validation["valid"]:
            # Resolve conflicts and unavailable ports
            for conflict in validation["conflicts"]:
                port = conflict["port"]
                services = conflict["services"]
                
                # Keep first service on original port, move others
                for service in services[1:]:
                    new_port = self.find_available_port(port + 1)
                    old_port = self.ports[service]
                    self.ports[service] = new_port
                    changes[service] = {"old": old_port, "new": new_port}
                    logger.info(f"Resolved conflict: {service} moved from port {old_port} to {new_port}")
            
            # Resolve unavailable ports
            for unavailable in validation["unavailable"]:
                service = unavailable["service"]
                old_port = unavailable["port"]
                new_port = self.find_available_port(old_port + 1)
                self.ports[service] = new_port
                changes[service] = {"old": old_port, "new": new_port}
                logger.info(f"Resolved unavailable port: {service} moved from port {old_port} to {new_port}")
        
        return changes
    
    def get_service_url(self, service: str, host: str = "127.0.0.1", scheme: str = "http") -> str:
        """Get full URL for a service."""
        port = self.get_port(service)
        return f"{scheme}://{host}:{port}"
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export current configuration for documentation or deployment."""
        return {
            "ports": self.ports.copy(),
            "environment_variables": {
                f"MISO_{service}_PORT": port 
                for service, port in self.ports.items()
            },
            "validation": self.validate_configuration()
        }

# Global instance
_port_config = None

def get_port_config() -> PortConfig:
    """Get global port configuration instance."""
    global _port_config
    if _port_config is None:
        _port_config = PortConfig()
    return _port_config

def get_port(service: str) -> int:
    """Convenience function to get port for a service."""
    return get_port_config().get_port(service)

def get_service_url(service: str, **kwargs) -> str:
    """Convenience function to get service URL."""
    return get_port_config().get_service_url(service, **kwargs)

# Environment variable documentation
ENV_VARS_DOCUMENTATION = """
MISO Port Configuration Environment Variables:

MISO_BACKEND_API_PORT=8000      # Main FastAPI backend
MISO_DASHBOARD_PORT=8080        # Web dashboard (Dash/Streamlit)
MISO_WEBSOCKET_PORT=8001        # WebSocket connections
MISO_MONITORING_PORT=8002       # System monitoring
MISO_DEBUG_SERVER_PORT=8003     # Debug/development server
MISO_BENCHMARK_API_PORT=8004    # Benchmark results API
MISO_METRICS_PORT=9090          # Prometheus metrics
MISO_HEALTH_CHECK_PORT=8005     # Health check endpoint

Example usage:
export MISO_DASHBOARD_PORT=3000
export MISO_BACKEND_API_PORT=5000
"""

if __name__ == "__main__":
    # CLI for port configuration management
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="MISO Port Configuration Manager")
    parser.add_argument("--validate", action="store_true", help="Validate current configuration")
    parser.add_argument("--resolve", action="store_true", help="Auto-resolve conflicts")
    parser.add_argument("--export", action="store_true", help="Export configuration")
    parser.add_argument("--service", help="Get port for specific service")
    
    args = parser.parse_args()
    
    config = get_port_config()
    
    if args.validate:
        validation = config.validate_configuration()
        print("Port Configuration Validation:")
        print(f"Valid: {validation['valid']}")
        
        if validation['conflicts']:
            print("Conflicts:")
            for conflict in validation['conflicts']:
                print(f"  Port {conflict['port']}: {', '.join(conflict['services'])}")
        
        if validation['unavailable']:
            print("Unavailable ports:")
            for unavailable in validation['unavailable']:
                print(f"  {unavailable['service']}: {unavailable['port']}")
        
        print("\nService assignments:")
        for service, info in validation['services'].items():
            status = "✅" if info['available'] else "❌"
            print(f"  {service}: {info['port']} {status}")
    
    elif args.resolve:
        changes = config.auto_resolve_conflicts()
        if changes:
            print("Port conflicts resolved:")
            for service, change in changes.items():
                print(f"  {service}: {change['old']} -> {change['new']}")
        else:
            print("No conflicts to resolve")
    
    elif args.export:
        import json
        export = config.export_configuration()
        print(json.dumps(export, indent=2))
    
    elif args.service:
        try:
            port = config.get_port(args.service)
            url = config.get_service_url(args.service)
            print(f"{args.service}: {port} ({url})")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    else:
        print(ENV_VARS_DOCUMENTATION)
