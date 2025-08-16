#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - M-LINGUA ↔ T-MATHEMATICS VXOR Integration

Dieses Modul implementiert die Integration zwischen M-LINGUA und der T-MATHEMATICS Engine
über die zentrale VXOR-Integrationsschicht. Es ermöglicht die Übersetzung natürlichsprachlicher
mathematischer Ausdrücke in optimierte T-MATHEMATICS Tensor-Operationen.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple

# Importiere M-LINGUA Komponenten
from miso.lang.mlingua.math_bridge import MathBridge, MathExpression, MathBridgeResult
from miso.lang.mlingua.semantic_layer import SemanticLayer, SemanticResult

# Importiere T-MATHEMATICS Komponenten
from miso.math.t_mathematics.config import TMathConfig
from miso.math.t_mathematics.integration_manager import TMathIntegrationManager

# Importiere VXOR-Adapter
from miso.vxor.vx_adapter_core import get_module, get_module_status

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.mlingua.vxor_t_math_integration")

class MLinguaTMathVXORIntegration:
    """
    Klasse zur Integration von M-LINGUA mit der T-MATHEMATICS Engine über VXOR
    
    Diese Klasse stellt die Verbindung zwischen der M-LINGUA Sprachverarbeitung
    und der optimierten T-MATHEMATICS Engine her, mit Unterstützung durch die
    VXOR-Module VX-MATRIX und VX-MEMEX.
    """
    
    _instance = None  # Singleton-Pattern
    
    def __new__(cls, *args, **kwargs):
        """Implementiert das Singleton-Pattern"""
        if cls._instance is None:
            cls._instance = super(MLinguaTMathVXORIntegration, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialisiert die M-LINGUA-T-MATHEMATICS-VXOR-Integration
        
        Args:
            config_path: Pfad zur Konfigurationsdatei (optional)
        """
        # Initialisiere nur einmal (Singleton-Pattern)
        if hasattr(self, 'initialized') and self.initialized:
            return
            
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "config", "vxor_t_math_integration_config.json"
        )
        
        # Initialisiere Kernkomponenten
        self.math_bridge = MathBridge()
        self.semantic_layer = SemanticLayer()
        
        # VXOR-Module
        self.vx_matrix = None
        self.vx_memex = None
        
        # Konfiguration
        self.config = {}
        self.load_config()
        
        # Initialisiere VXOR-Integration
        self._initialize_vxor_modules()
        
        # Initialisiere T-MATHEMATICS Integration Manager
        self.t_math_manager = TMathIntegrationManager()
        
        # Optimierte T-MATHEMATICS Engine für M-LINGUA
        self.t_math_engine = self._get_optimized_engine()
        
        self.initialized = True
        logger.info("M-LINGUA ↔ T-MATHEMATICS VXOR Integration initialisiert")
    
    def load_config(self):
        """Lädt die Konfiguration aus der Konfigurationsdatei"""
        try:
            # Stelle sicher, dass das Verzeichnis existiert
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Wenn die Datei nicht existiert, erstelle Standardkonfiguration
            if not os.path.exists(self.config_path):
                self._create_default_config()
            
            # Lade die Konfiguration
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            logger.info(f"Konfiguration geladen: {len(self.config)} Einträge")
        except Exception as e:
            logger.error(f"Fehler beim Laden der Konfiguration: {e}")
            # Erstelle Standardkonfiguration im Fehlerfall
            self._create_default_config()
    
    def _create_default_config(self):
        """Erstellt eine Standardkonfiguration"""
        default_config = {
            "vxor_integration": {
                "enabled": True,
                "vx_matrix_enabled": True,
                "vx_memex_enabled": True
            },
            "t_math_config": {
                "backend": "mlx",
                "optimization_level": 4,
                "use_hardware_acceleration": True,
                "use_caching": True,
                "debug_mode": False
            },
            "language_support": {
                "enabled_languages": ["de", "en"],
                "default_language": "de"
            },
            "performance": {
                "cache_results": True,
                "max_cache_size": 1000,
                "cache_ttl_seconds": 3600
            }
        }
        
        try:
            # Stelle sicher, dass das Verzeichnis existiert
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Speichere die Standardkonfiguration
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2)
            
            self.config = default_config
            logger.info("Standardkonfiguration erstellt")
        except Exception as e:
            logger.error(f"Fehler beim Erstellen der Standardkonfiguration: {e}")
            self.config = default_config
    
    def _initialize_vxor_modules(self):
        """Initialisiert die VXOR-Module"""
        # Prüfe, ob VXOR-Module aktiviert sind
        vxor_config = self.config.get("vxor_integration", {})
        vx_matrix_enabled = vxor_config.get("vx_matrix_enabled", True)
        vx_memex_enabled = vxor_config.get("vx_memex_enabled", True)
        
        # Initialisiere VX-MATRIX
        if vx_matrix_enabled:
            try:
                self.vx_matrix = get_module("VX-MATRIX")
                logger.info("VX-MATRIX erfolgreich initialisiert")
            except Exception as e:
                logger.warning(f"VX-MATRIX nicht verfügbar: {e}")
                self.vx_matrix = None
        
        # Initialisiere VX-MEMEX
        if vx_memex_enabled:
            try:
                self.vx_memex = get_module("VX-MEMEX")
                logger.info("VX-MEMEX erfolgreich initialisiert")
            except Exception as e:
                logger.warning(f"VX-MEMEX nicht verfügbar: {e}")
                self.vx_memex = None
    
    def _get_optimized_engine(self):
        """Erstellt oder holt eine optimierte T-MATHEMATICS Engine für M-LINGUA"""
        t_math_config = self.config.get("t_math_config", {})
        
        # Erstelle optimierte Konfiguration
        config = TMathConfig(
            backend=t_math_config.get("backend", "mlx"),
            optimization_level=t_math_config.get("optimization_level", 4),
            use_hardware_acceleration=t_math_config.get("use_hardware_acceleration", True),
            debug_mode=t_math_config.get("debug_mode", False)
        )
        
        # Hole oder registriere Engine im Integration Manager
        try:
            engine = self.t_math_manager.get_engine("m_lingua", config)
            logger.info("Optimierte T-MATHEMATICS Engine für M-LINGUA aus Integration Manager geladen")
        except Exception as e:
            logger.warning(f"Konnte keine Engine aus dem Integration Manager laden: {e}")
            
            # Erstelle direkt eine Engine
            from miso.math.t_mathematics.engine import TMathematicsEngine
            engine = TMathematicsEngine(config=config)
            
            # Registriere die Engine im Integration Manager für zukünftige Verwendung
            try:
                self.t_math_manager.register_engine("m_lingua", engine)
                logger.info("Neu erstellte T-MATHEMATICS Engine im Integration Manager registriert")
            except Exception as reg_err:
                logger.warning(f"Fehler bei der Registrierung der Engine im Integration Manager: {reg_err}")
        
        return engine
    
    def process_math_expression(self, text: str, language_code: str = "de") -> Dict[str, Any]:
        """
        Verarbeitet einen mathematischen Ausdruck mit optimierter T-MATHEMATICS Engine
        
        Args:
            text: Mathematischer Ausdruck in natürlicher Sprache
            language_code: Sprachcode (de, en)
            
        Returns:
            Dictionary mit Verarbeitungsergebnis
        """
        start_time = time.time()
        
        # Prüfe, ob wir Ergebnisse über VX-MEMEX zwischenspeichern können
        can_cache = self.vx_memex is not None and self.config.get("performance", {}).get("cache_results", True)
        
        if can_cache:
            # Versuche, Ergebnis aus dem Cache zu laden
            try:
                cache_key = f"math_expression:{language_code}:{text}"
                cached_result = self.vx_memex.retrieve({
                    "query": cache_key,
                    "context": {"source": "m_lingua_t_math"}
                })
                
                if cached_result and cached_result.get("success", False) and "result" in cached_result:
                    cached_data = cached_result.get("result")
                    cached_time = cached_data.get("timestamp", 0)
                    ttl = self.config.get("performance", {}).get("cache_ttl_seconds", 3600)
                    
                    if time.time() - cached_time < ttl:
                        logger.info(f"Ergebnis aus Cache geladen für: {text}")
                        return cached_data.get("data", {})
            except Exception as e:
                logger.warning(f"Fehler beim Laden aus dem Cache: {e}")
        
        # Prüfe, ob wir VX-MATRIX für die Berechnung verwenden können
        use_vx_matrix = self.vx_matrix is not None and self.config.get("vxor_integration", {}).get("vx_matrix_enabled", True)
        
        # Verarbeite den Ausdruck mit dem MathBridge
        bridge_result = self.math_bridge.process_math_expression(text, language_code)
        
        if not bridge_result.success:
            logger.warning(f"Fehler bei der Verarbeitung des mathematischen Ausdrucks: {bridge_result.error_message}")
            result = {
                "success": False,
                "error": bridge_result.error_message,
                "processing_time": time.time() - start_time
            }
        else:
            # Prüfe, ob wir optimierte Berechnungen mit VX-MATRIX durchführen können
            if use_vx_matrix and bridge_result.math_expression and bridge_result.math_expression.expression_type in ["matrix", "tensor"]:
                try:
                    math_expr = bridge_result.math_expression
                    operation_type = "tensor_operation" if math_expr.expression_type == "tensor" else "matrix_operation"
                    
                    # Bereite Parameter für VX-MATRIX vor
                    params = {
                        "operation": operation_type,
                        "inputs": [math_expr.parsed_expression],
                        "config": self.config.get("t_math_config", {})
                    }
                    
                    # Führe optimierte Berechnung mit VX-MATRIX durch
                    vx_result = self.vx_matrix.compute(params)
                    
                    if vx_result and vx_result.get("success", False):
                        logger.info(f"Optimierte Berechnung mit VX-MATRIX durchgeführt für: {text}")
                        
                        # Kombiniere Ergebnisse
                        result = {
                            "success": True,
                            "result": vx_result.get("result"),
                            "expression_type": math_expr.expression_type,
                            "expression_text": math_expr.expression_text,
                            "confidence": math_expr.confidence,
                            "optimized": True,
                            "optimizer": "VX-MATRIX",
                            "processing_time": time.time() - start_time
                        }
                    else:
                        # Fallback zum Standard-Ergebnis
                        result = self._create_standard_result(bridge_result, start_time)
                except Exception as e:
                    logger.warning(f"Fehler bei der optimierten Berechnung mit VX-MATRIX: {e}")
                    # Fallback zum Standard-Ergebnis
                    result = self._create_standard_result(bridge_result, start_time)
            else:
                # Standard-Ergebnis ohne Optimierung
                result = self._create_standard_result(bridge_result, start_time)
        
        # Speichere Ergebnis im Cache, wenn möglich
        if can_cache and result.get("success", False):
            try:
                cache_key = f"math_expression:{language_code}:{text}"
                cache_data = {
                    "data": result,
                    "timestamp": time.time()
                }
                
                self.vx_memex.store({
                    "data": cache_data,
                    "context": {
                        "key": cache_key,
                        "source": "m_lingua_t_math",
                        "ttl": self.config.get("performance", {}).get("cache_ttl_seconds", 3600)
                    }
                })
            except Exception as e:
                logger.warning(f"Fehler beim Speichern im Cache: {e}")
        
        return result
    
    def _create_standard_result(self, bridge_result: MathBridgeResult, start_time: float) -> Dict[str, Any]:
        """Erstellt ein Standard-Ergebnis ohne VXOR-Optimierung"""
        return {
            "success": True,
            "result": bridge_result.result,
            "expression_type": bridge_result.math_expression.expression_type if bridge_result.math_expression else "unknown",
            "expression_text": bridge_result.math_expression.expression_text if bridge_result.math_expression else "",
            "confidence": bridge_result.math_expression.confidence if bridge_result.math_expression else 0.0,
            "optimized": False,
            "processing_time": time.time() - start_time
        }
    
    def process_semantic_result(self, semantic_result: SemanticResult) -> Optional[Dict[str, Any]]:
        """
        Verarbeitet ein semantisches Ergebnis mit mathematischen Inhalten
        
        Args:
            semantic_result: Semantisches Ergebnis aus der M-LINGUA SemanticLayer
            
        Returns:
            Dictionary mit Verarbeitungsergebnis oder None, wenn kein mathematischer Inhalt
        """
        # Prüfe, ob das semantische Ergebnis mathematischen Inhalt hat
        if not semantic_result or not semantic_result.intent or semantic_result.intent not in ["math_expression", "calculation", "equation"]:
            return None
        
        # Extrahiere Text und Sprachcode
        text = semantic_result.text
        language_code = semantic_result.language_code or "de"
        
        # Verarbeite den mathematischen Ausdruck
        return self.process_math_expression(text, language_code)


# Singleton-Instanz der Integration
_integration_instance = None

def get_mlingua_tmath_integration() -> MLinguaTMathVXORIntegration:
    """
    Gibt die Singleton-Instanz der M-LINGUA-T-MATHEMATICS-Integration zurück
    
    Returns:
        MLinguaTMathVXORIntegration-Instanz
    """
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = MLinguaTMathVXORIntegration()
    return _integration_instance


# Initialisiere die Integration, wenn das Modul importiert wird
if __name__ != "__main__":  # Nicht initialisieren, wenn direkt ausgeführt
    _integration_instance = MLinguaTMathVXORIntegration()

# Hauptfunktion zum direkten Testen
if __name__ == "__main__":
    integration = get_mlingua_tmath_integration()
    
    # Teste einige mathematische Ausdrücke
    test_expressions = [
        ("berechne 3 + 4 * 2", "de"),
        ("what is the square root of 16", "en"),
        ("erstelle eine matrix mit den werten [1, 2; 3, 4]", "de"),
        ("vector [1, 2, 3, 4]", "en")
    ]
    
    for expr, lang in test_expressions:
        print(f"\nVerarbeite: {expr} (Sprache: {lang})")
        result = integration.process_math_expression(expr, lang)
        print(f"Ergebnis: {result}")
