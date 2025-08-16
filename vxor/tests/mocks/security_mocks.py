#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SecurityLayer Mock-Objekte

Diese Datei enthält Mock-Implementierungen für die SecurityLayer-Komponenten.
Verwendet für Tests, um die Abhängigkeit von der echten SecurityLayer zu vermeiden.
"""

import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import time
import uuid

# Logger für Mocks
logger = logging.getLogger("vxor.test.mocks.security")

# Importiere Enum-Typen aus dem Original, um kompatibel zu sein
try:
    from security_layer import SecurityLevel, SecurityOperation, SecurityEvent
except ImportError:
    # Fallback, wenn das Originalmodul nicht importiert werden kann
    class SecurityLevel(Enum):
        """Sicherheitsstufen für MISO-Komponenten"""
        LOW = 1      # Niedrige Sicherheitsstufe
        MEDIUM = 2   # Mittlere Sicherheitsstufe
        HIGH = 3     # Hohe Sicherheitsstufe
        CRITICAL = 4 # Kritische Sicherheitsstufe
        
    class SecurityOperation(Enum):
        """Sicherheitsoperationen"""
        AUTHENTICATE = 1    # Authentifizierung
        AUTHORIZE = 2       # Autorisierung
        ENCRYPT = 3         # Verschlüsselung
        DECRYPT = 4         # Entschlüsselung
        SIGN = 5            # Signierung
        VERIFY = 6          # Verifizierung
    
    class SecurityEvent:
        """Repräsentiert ein Sicherheitsereignis"""
        def __init__(self, 
                    operation: SecurityOperation, 
                    component: str, 
                    level: SecurityLevel,
                    status: bool,
                    details: Optional[Dict[str, Any]] = None):
            self.id = str(uuid.uuid4())
            self.timestamp = time.time()
            self.operation = operation
            self.component = component
            self.level = level
            self.status = status
            self.details = details or {}
        
        def __str__(self):
            return f"SecurityEvent({self.operation}, {self.component}, {self.status})"
        
        def to_dict(self):
            return {
                "id": self.id,
                "timestamp": self.timestamp,
                "operation": self.operation.name,
                "component": self.component,
                "level": self.level.name,
                "status": self.status,
                "details": self.details
            }

class MockSecurityLayer:
    """Mock-Implementierung der SecurityLayer für Tests"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialisiert die Mock-Sicherheitsschicht.
        
        Args:
            config: Konfiguration für die Sicherheitsschicht (optional)
        """
        self.config = config or {}
        self.security_key = self.config.get("security_key", b"mock_security_key")
        self.security_level = SecurityLevel[self.config.get("security_level", "HIGH")]
        self.audit_trail = []
        self.component_security_levels = {}
        self.authorized_components = set()
        
        # Für Tests: Voreingestellte Authentifizierungs-/Autorisierungsergebnisse
        self.auth_results = self.config.get("auth_results", {"default": True})
        
        logger.info("[MOCK] SecurityLayer initialisiert")
    
    def authenticate(self, component_id: str, credentials: Dict[str, Any]) -> bool:
        """Mock-Authentifizierung, gibt voreingestelltes Ergebnis zurück"""
        result = self.auth_results.get(component_id, self.auth_results.get("default", True))
        event = SecurityEvent(
            SecurityOperation.AUTHENTICATE,
            component_id,
            self.security_level,
            result,
            {"mock": True, "credentials": str(credentials)}
        )
        self.audit_trail.append(event.to_dict())
        return result
    
    def authorize(self, component_id: str, operation: str, resource: str) -> bool:
        """Mock-Autorisierung, gibt voreingestelltes Ergebnis zurück"""
        result = self.auth_results.get(component_id, self.auth_results.get("default", True))
        event = SecurityEvent(
            SecurityOperation.AUTHORIZE,
            component_id,
            self.security_level,
            result,
            {"mock": True, "operation": operation, "resource": resource}
        )
        self.audit_trail.append(event.to_dict())
        return result
    
    def register_component(self, component_id: str, security_level: SecurityLevel) -> bool:
        """Registriert eine Komponente für Tests"""
        self.component_security_levels[component_id] = security_level
        self.authorized_components.add(component_id)
        return True
    
    def encrypt(self, data: bytes, component_id: str) -> Tuple[bytes, bytes]:
        """Mock-Verschlüsselung, gibt die Daten unverschlüsselt zurück"""
        # In einem echten System würde hier verschlüsselt werden
        iv = b"mock_iv_12345678"
        return data, iv
    
    def decrypt(self, encrypted_data: bytes, iv: bytes, component_id: str) -> bytes:
        """Mock-Entschlüsselung, gibt die Daten unverändert zurück"""
        # In einem echten System würde hier entschlüsselt werden
        return encrypted_data
    
    def sign(self, data: bytes, component_id: str) -> bytes:
        """Mock-Signierung, gibt eine konstante Signatur zurück"""
        return b"mock_signature_for_" + component_id.encode()
    
    def verify(self, data: bytes, signature: bytes, component_id: str) -> bool:
        """Mock-Verifizierung, gibt voreingestelltes Ergebnis zurück"""
        return self.auth_results.get(component_id, self.auth_results.get("default", True))
    
    def get_audit_trail(self, component_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Gibt den Mock-Audit-Trail zurück"""
        if component_id:
            filtered = [e for e in self.audit_trail if e["component"] == component_id]
            return filtered[:limit]
        return self.audit_trail[:limit]
    
    def export_audit_trail(self, file_path: str) -> bool:
        """Mock-Export des Audit-Trails"""
        logger.info(f"[MOCK] Audit-Trail würde nach {file_path} exportiert werden")
        return True
    
    def get_security_status(self) -> Dict[str, Any]:
        """Gibt einen Mock-Sicherheitsstatus zurück"""
        return {
            "security_level": self.security_level.name,
            "registered_components": len(self.component_security_levels),
            "authorized_components": len(self.authorized_components),
            "audit_events": len(self.audit_trail),
            "mock": True
        }

# Standardisierte Entry-Points für den Mock
_mock_security_layer = None

def init():
    """Standardisierter Entry-Point: Initialisiert die Mock-SecurityLayer"""
    global _mock_security_layer
    if _mock_security_layer is None:
        _mock_security_layer = MockSecurityLayer()
        logger.info("[MOCK] SecurityLayer erfolgreich initialisiert via standardisierten Entry-Point")
    return _mock_security_layer

def configure(config=None):
    """Standardisierter Entry-Point: Konfiguriert die Mock-SecurityLayer"""
    global _mock_security_layer
    if _mock_security_layer is None:
        _mock_security_layer = MockSecurityLayer(config=config)
    return _mock_security_layer

def boot():
    """Alias für init() für Kompatibilität mit Boot-Konvention"""
    return init()

def setup():
    """Alias für init() für Kompatibilität mit Setup-Konvention"""
    return init()

def activate():
    """Aktiviert die Mock-SecurityLayer mit Standardeinstellungen"""
    return init()
