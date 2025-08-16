"""
VOID-Protokoll 3.0 (Verified Origin ID and Defense)
-------------------------------------------------

Post-Quanten-Sicheres End-to-End-Verschlüsselungsprotokoll für MISO_Ultimate
mit Kontext-Enforcement und Anti-Debug-Schutz.

Hauptfunktionen:
- Post-Quanten-Kryptografie mit Kyber512 und Dilithium
- End-to-End-Verschlüsselung für alle IPC-Kanäle
- Automatische Schlüsselrotation
- Kontextbasierte Ausführungsrichtlinien
- Anti-Debug- und Anti-Tamper-Schutz
- Auto-Shutdown bei Sicherheitsverletzungen
"""

# Importiere die korrekten Klassen aus den Modulen - vereinfachte Version für Tests
try:
    from .void_protocol import VoidProtocol
    from .void_crypto import VoidCrypto
    from .void_context import VoidContext
    from .void_interface import VOIDInterface
except ImportError as e:
    # Fallback-Import für den Fall, dass wir in einem separaten Verzeichnis sind
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from security.void.void_protocol import VoidProtocol
        from security.void.void_crypto import VoidCrypto
        from security.void.void_context import VoidContext
        from security.void.void_interface import VOIDInterface
    except ImportError as e2:
        print(f"Fehler beim Importieren der VOID-Module: {e} / {e2}")
        # Stelle leere Stub-Klassen bereit, damit das __init__ nicht komplett fehlschlägt
        class VoidProtocol: pass
        class VoidCrypto: pass
        class VoidContext: pass
        class VOIDInterface: pass

__version__ = "3.0.0"
__all__ = [
    'VoidProtocol', 'VoidMessage', 'VoidEndpoint', 'HandshakeResult',
    'initialize', 'get_instance', 'encrypt', 'decrypt', 'handshake', 
    'verify_handshake', 'register_endpoint', 'close',
    'VoidContext', 'ContextVerifier'
]
