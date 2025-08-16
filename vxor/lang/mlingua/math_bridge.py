#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - M-LINGUA Mathematics Bridge

Dieses Modul implementiert die Brücke zwischen M-LINGUA und der T-Mathematics-Engine.
Es ermöglicht die Übersetzung natürlichsprachlicher mathematischer Ausdrücke in
T-Mathematics-Operationen und -Befehle.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import re
import json
import logging
import importlib
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field

# Importiere die M-LINGUA-Komponenten
from miso.lang.mlingua.semantic_layer import SemanticResult, SemanticContext
from miso.lang.mlingua.multilang_parser import ParsedCommand

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [M-LINGUA] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MISO.M-LINGUA.MathBridge")

@dataclass
class MathExpression:
    """Repräsentation eines mathematischen Ausdrucks"""
    expression_type: str  # 'scalar', 'vector', 'matrix', 'tensor', 'equation', 'function'
    expression_text: str
    parsed_expression: Any
    variables: Dict[str, Any] = field(default_factory=dict)
    operations: List[Dict[str, Any]] = field(default_factory=list)
    result: Any = None
    confidence: float = 0.0
    original_text: str = ""
    detected_language: str = ""

@dataclass
class MathBridgeResult:
    """Ergebnis der Verarbeitung durch die Mathematik-Brücke"""
    success: bool
    math_expression: Optional[MathExpression] = None
    t_math_command: Optional[str] = None
    result: Any = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)

class MathBridge:
    """
    Brücke zwischen M-LINGUA und der T-Mathematics-Engine
    
    Diese Klasse übersetzt natürlichsprachliche mathematische Ausdrücke in
    T-Mathematics-Operationen und -Befehle und führt diese aus.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialisiert die Mathematik-Brücke
        
        Args:
            config_path: Pfad zur Konfigurationsdatei (optional)
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "config", "math_bridge_config.json"
        )
        self.config = {}
        self.math_patterns = {}
        self.t_math_engine = None
        self.load_config()
        self._initialize_t_math_engine()
        logger.info(f"MathBridge initialisiert mit {len(self.math_patterns)} mathematischen Mustern")
    
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
            
            self.math_patterns = self.config.get("math_patterns", {})
            
            logger.info(f"Konfiguration geladen: {len(self.math_patterns)} mathematische Muster")
        except Exception as e:
            logger.error(f"Fehler beim Laden der Konfiguration: {e}")
            # Erstelle Standardkonfiguration im Fehlerfall
            self._create_default_config()
    
    def _create_default_config(self):
        """Erstellt eine Standardkonfiguration"""
        default_config = {
            "t_math_engine": {
                "module_path": "miso.math.t_mathematics.engine",
                "class_name": "TMathematicsEngine",
                "use_mlx": True,
                "optimization_level": 3
            },
            "math_patterns": {
                "de": {
                    "scalar_patterns": [
                        {"pattern": r"berechne\s+([0-9+\-*/^().]+)", "type": "scalar_expression"},
                        {"pattern": r"was\s+ist\s+([0-9+\-*/^().]+)", "type": "scalar_expression"}
                    ],
                    "vector_patterns": [
                        {"pattern": r"vektor\s+\[([\d\s,.-]+)\]", "type": "vector_expression"},
                        {"pattern": r"erstelle\s+einen\s+vektor\s+mit\s+den\s+werten\s+([\d\s,.-]+)", "type": "vector_expression"}
                    ],
                    "matrix_patterns": [
                        {"pattern": r"matrix\s+\[([\d\s,;.-]+)\]", "type": "matrix_expression"},
                        {"pattern": r"erstelle\s+eine\s+matrix\s+mit\s+den\s+werten\s+\[([\d\s,;.-]+)\]", "type": "matrix_expression"}
                    ],
                    "equation_patterns": [
                        {"pattern": r"löse\s+die\s+gleichung\s+(.+?)(?:\s+für\s+(.+))?$", "type": "equation"},
                        {"pattern": r"berechne\s+(.+?)(?:\s+für\s+(.+))?$", "type": "equation"}
                    ],
                    "function_patterns": [
                        {"pattern": r"definiere\s+die\s+funktion\s+(\w+)\s*\((.+?)\)\s*=\s*(.+)$", "type": "function_definition"},
                        {"pattern": r"berechne\s+die\s+funktion\s+(\w+)\s*\((.+?)\)\s*$", "type": "function_evaluation"}
                    ]
                },
                "en": {
                    "scalar_patterns": [
                        {"pattern": r"calculate\s+([0-9+\-*/^().]+)", "type": "scalar_expression"},
                        {"pattern": r"what\s+is\s+([0-9+\-*/^().]+)", "type": "scalar_expression"}
                    ],
                    "vector_patterns": [
                        {"pattern": r"vector\s+\[([\d\s,.-]+)\]", "type": "vector_expression"},
                        {"pattern": r"create\s+a\s+vector\s+with\s+values\s+([\d\s,.-]+)", "type": "vector_expression"}
                    ],
                    "matrix_patterns": [
                        {"pattern": r"matrix\s+\[([\d\s,;.-]+)\]", "type": "matrix_expression"},
                        {"pattern": r"create\s+a\s+matrix\s+with\s+values\s+\[([\d\s,;.-]+)\]", "type": "matrix_expression"}
                    ],
                    "equation_patterns": [
                        {"pattern": r"solve\s+the\s+equation\s+(.+?)(?:\s+for\s+(.+))?$", "type": "equation"},
                        {"pattern": r"calculate\s+(.+?)(?:\s+for\s+(.+))?$", "type": "equation"}
                    ],
                    "function_patterns": [
                        {"pattern": r"define\s+the\s+function\s+(\w+)\s*\((.+?)\)\s*=\s*(.+)$", "type": "function_definition"},
                        {"pattern": r"calculate\s+the\s+function\s+(\w+)\s*\((.+?)\)\s*$", "type": "function_evaluation"}
                    ]
                }
            },
            "t_math_templates": {
                "scalar_expression": "TMath.evaluate('{{expression}}')",
                "vector_expression": "TMath.vector([{{values}}])",
                "matrix_expression": "TMath.matrix([{{values}}])",
                "tensor_expression": "TMath.tensor({{values}}, shape=[{{shape}}])",
                "equation": "TMath.solve_equation('{{equation}}', '{{variable}}')",
                "function_definition": "TMath.define_function('{{name}}', '{{parameters}}', '{{body}}')",
                "function_evaluation": "TMath.evaluate_function('{{name}}', {{parameters}})",
                "tensor_operation": "TMath.tensor_operation('{{operation}}', [{{inputs}}])",
                "mlx_optimized": "TMath.mlx_compute('{{operation}}', [{{inputs}}], {{config}})"
            }
        }
        
        # Speichere die Standardkonfiguration
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        
        # Lade die Konfiguration in den Speicher
        self.config = default_config
        self.math_patterns = default_config["math_patterns"]
        
        logger.info("Standardkonfiguration erstellt")
    
    def _initialize_t_math_engine(self):
        """Initialisiert die T-Mathematics-Engine"""
        try:
            t_math_config = self.config.get("t_math_engine", {})
            module_path = t_math_config.get("module_path", "")
            class_name = t_math_config.get("class_name", "")
            use_mlx = t_math_config.get("use_mlx", True)
            optimization_level = t_math_config.get("optimization_level", 3)
            
            if not module_path or not class_name:
                logger.warning("Keine T-Mathematics-Engine-Konfiguration gefunden")
                return
            
            # Versuche, das Modul zu laden
            try:
                # Importiere die T-Math Konfiguration
                from miso.math.t_mathematics.config import TMathConfig
                
                # Erstelle eine optimierte Konfiguration
                config = TMathConfig(
                    backend="mlx" if use_mlx else "auto",
                    optimization_level=optimization_level,
                    use_hardware_acceleration=True,
                    debug_mode=False
                )
                
                # Importiere die Integration Manager für VXOR-Unterstützung
                from miso.math.t_mathematics.integration_manager import TMathIntegrationManager
                manager = TMathIntegrationManager()
                
                # Hole Engine aus dem Manager oder erstelle direkt
                try:
                    self.t_math_engine = manager.get_engine("m_lingua", config)
                    logger.info("T-Mathematics-Engine aus Integration Manager geladen")
                except:
                    module = importlib.import_module(module_path)
                    engine_class = getattr(module, class_name)
                    self.t_math_engine = engine_class(config=config)
                    logger.info(f"T-Mathematics-Engine direkt geladen: {module_path}.{class_name}")
            except Exception as e:
                logger.warning(f"Fehler beim Laden der T-Mathematics-Engine: {e}")
                # Erstelle einen Mock für Tests
                self.t_math_engine = self._create_mock_t_math_engine()
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung der T-Mathematics-Engine: {e}")
            # Erstelle einen Mock für Tests
            self.t_math_engine = self._create_mock_t_math_engine()
    
    def _create_mock_t_math_engine(self):
        """Erstellt einen Mock der T-Mathematics-Engine für Tests"""
        class MockTMathEngine:
            def evaluate(self, expression):
                logger.info(f"Mock: Evaluiere Ausdruck: {expression}")
                try:
                    # Einfache Auswertung für Testzwecke
                    return eval(expression)
                except Exception as e:
                    logger.warning(f"Mock: Fehler bei der Auswertung: {e}")
                    return None
            
            def vector(self, values):
                logger.info(f"Mock: Erstelle Vektor: {values}")
                return values
            
            def matrix(self, values):
                logger.info(f"Mock: Erstelle Matrix: {values}")
                return values
                
            def tensor(self, values, shape=None):
                logger.info(f"Mock: Erstelle Tensor: {values} mit Shape {shape}")
                return values
                
            def tensor_operation(self, operation, inputs):
                logger.info(f"Mock: Führe Tensor-Operation aus: {operation} mit {inputs}")
                return inputs[0] if inputs else None
                
            def mlx_compute(self, operation, inputs, config):
                logger.info(f"Mock: MLX-Berechnung: {operation} mit {inputs} und Konfiguration {config}")
                return inputs[0] if inputs else None
            
            def solve_equation(self, equation, variable):
                logger.info(f"Mock: Löse Gleichung: {equation} für {variable}")
                return f"Lösung für {variable} in {equation}"
            
            def define_function(self, name, parameters, body):
                logger.info(f"Mock: Definiere Funktion: {name}({parameters}) = {body}")
                return True
            
            def evaluate_function(self, name, parameters):
                logger.info(f"Mock: Evaluiere Funktion: {name}({parameters})")
                return f"Ergebnis von {name}({parameters})"
        
        logger.info("Mock-T-Mathematics-Engine erstellt")
        return MockTMathEngine()
    
    def process_math_expression(self, text: str, language_code: str) -> MathBridgeResult:
        """
        Verarbeitet einen mathematischen Ausdruck
        
        Args:
            text: Zu verarbeitender Text
            language_code: Sprachcode
            
        Returns:
            MathBridgeResult-Objekt mit dem Ergebnis
        """
        if not text or len(text.strip()) == 0:
            logger.warning("Leerer Text für mathematische Verarbeitung")
            return MathBridgeResult(
                success=False,
                error_message="Leerer Text"
            )
        
        # Überprüfe, ob die Sprache unterstützt wird
        if language_code not in self.math_patterns:
            logger.warning(f"Sprache nicht unterstützt: {language_code}")
            return MathBridgeResult(
                success=False,
                error_message=f"Sprache nicht unterstützt: {language_code}"
            )
        
        # Extrahiere den mathematischen Ausdruck
        math_expression = self._extract_math_expression(text, language_code)
        
        if not math_expression:
            logger.warning(f"Kein mathematischer Ausdruck erkannt in: {text}")
            return MathBridgeResult(
                success=False,
                error_message="Kein mathematischer Ausdruck erkannt"
            )
        
        # Generiere T-Mathematics-Befehl
        t_math_command = self._generate_t_math_command(math_expression)
        
        if not t_math_command:
            logger.warning(f"Konnte keinen T-Mathematics-Befehl generieren für: {math_expression.expression_text}")
            return MathBridgeResult(
                success=False,
                math_expression=math_expression,
                error_message="Konnte keinen T-Mathematics-Befehl generieren"
            )
        
        # Führe den Befehl aus
        try:
            import time
            start_time = time.time()
            
            # Führe den Befehl aus
            result = self._execute_t_math_command(t_math_command, math_expression)
            
            execution_time = time.time() - start_time
            
            # Aktualisiere den mathematischen Ausdruck
            math_expression.result = result
            
            return MathBridgeResult(
                success=True,
                math_expression=math_expression,
                t_math_command=t_math_command,
                result=result,
                execution_time=execution_time
            )
        except Exception as e:
            logger.error(f"Fehler bei der Ausführung des T-Mathematics-Befehls: {e}")
            return MathBridgeResult(
                success=False,
                math_expression=math_expression,
                t_math_command=t_math_command,
                error_message=str(e)
            )
    
    def _extract_math_expression(self, text: str, language_code: str) -> Optional[MathExpression]:
        """
        Extrahiert einen mathematischen Ausdruck aus einem Text
        
        Args:
            text: Zu analysierender Text
            language_code: Sprachcode
            
        Returns:
            MathExpression-Objekt oder None
        """
        # Hole die Muster für die Sprache
        language_patterns = self.math_patterns.get(language_code, {})
        
        # Durchsuche alle Mustertypen
        for pattern_type, patterns in language_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info.get("pattern", "")
                expr_type = pattern_info.get("type", "")
                
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    # Extrahiere den Ausdruck
                    if expr_type == "scalar_expression":
                        expression_text = match.group(1).strip()
                        return MathExpression(
                            expression_type="scalar",
                            expression_text=expression_text,
                            parsed_expression=expression_text,
                            confidence=0.9,
                            original_text=text,
                            detected_language=language_code
                        )
                    elif expr_type == "vector_expression":
                        values_text = match.group(1).strip()
                        values = [float(v.strip()) for v in re.split(r'[,\s]+', values_text) if v.strip()]
                        return MathExpression(
                            expression_type="vector",
                            expression_text=values_text,
                            parsed_expression=values,
                            confidence=0.9,
                            original_text=text,
                            detected_language=language_code
                        )
                    elif expr_type == "matrix_expression":
                        matrix_text = match.group(1).strip()
                        rows = matrix_text.split(';')
                        matrix = []
                        for row in rows:
                            values = [float(v.strip()) for v in re.split(r'[,\s]+', row) if v.strip()]
                            if values:
                                matrix.append(values)
                        return MathExpression(
                            expression_type="matrix",
                            expression_text=matrix_text,
                            parsed_expression=matrix,
                            confidence=0.9,
                            original_text=text,
                            detected_language=language_code
                        )
                    elif expr_type == "equation":
                        equation = match.group(1).strip()
                        variable = match.group(2).strip() if match.lastindex >= 2 else None
                        return MathExpression(
                            expression_type="equation",
                            expression_text=equation,
                            parsed_expression={"equation": equation, "variable": variable},
                            variables={variable: None} if variable else {},
                            confidence=0.9,
                            original_text=text,
                            detected_language=language_code
                        )
                    elif expr_type == "function_definition":
                        function_name = match.group(1).strip()
                        parameters = match.group(2).strip()
                        body = match.group(3).strip()
                        return MathExpression(
                            expression_type="function",
                            expression_text=f"{function_name}({parameters}) = {body}",
                            parsed_expression={"name": function_name, "parameters": parameters, "body": body},
                            confidence=0.9,
                            original_text=text,
                            detected_language=language_code
                        )
                    elif expr_type == "function_evaluation":
                        function_name = match.group(1).strip()
                        parameters = match.group(2).strip()
                        return MathExpression(
                            expression_type="function_evaluation",
                            expression_text=f"{function_name}({parameters})",
                            parsed_expression={"name": function_name, "parameters": parameters},
                            confidence=0.9,
                            original_text=text,
                            detected_language=language_code
                        )
        
        return None
    
    def _generate_t_math_command(self, expression: MathExpression) -> Optional[str]:
        """
        Generiert einen T-Mathematics-Befehl aus einem mathematischen Ausdruck
        
        Args:
            expression: Mathematischer Ausdruck
            
        Returns:
            T-Mathematics-Befehl oder None
        """
        # Hole die Templates
        templates = self.config.get("t_math_templates", {})
        
        # Wähle das passende Template
        if expression.expression_type == "scalar":
            template = templates.get("scalar_expression", "")
            if not template:
                return None
            
            # Ersetze Platzhalter
            return template.replace("{{expression}}", expression.expression_text)
        elif expression.expression_type == "vector":
            template = templates.get("vector_expression", "")
            if not template:
                return None
            
            # Ersetze Platzhalter
            values = ", ".join(str(v) for v in expression.parsed_expression)
            return template.replace("{{values}}", values)
        elif expression.expression_type == "matrix":
            template = templates.get("matrix_expression", "")
            if not template:
                return None
            
            # Ersetze Platzhalter
            values = "], [".join(", ".join(str(v) for v in row) for row in expression.parsed_expression)
            return template.replace("{{values}}", values)
        elif expression.expression_type == "equation":
            template = templates.get("equation", "")
            if not template:
                return None
            
            # Ersetze Platzhalter
            equation = expression.parsed_expression["equation"]
            variable = expression.parsed_expression["variable"] or ""
            return template.replace("{{equation}}", equation).replace("{{variable}}", variable)
        elif expression.expression_type == "function":
            template = templates.get("function_definition", "")
            if not template:
                return None
            
            # Ersetze Platzhalter
            name = expression.parsed_expression["name"]
            parameters = expression.parsed_expression["parameters"]
            body = expression.parsed_expression["body"]
            return template.replace("{{name}}", name).replace("{{parameters}}", parameters).replace("{{body}}", body)
        elif expression.expression_type == "function_evaluation":
            template = templates.get("function_evaluation", "")
            if not template:
                return None
            
            # Ersetze Platzhalter
            name = expression.parsed_expression["name"]
            parameters = expression.parsed_expression["parameters"]
            return template.replace("{{name}}", name).replace("{{parameters}}", parameters)
        
        return None
    
    def _execute_t_math_command(self, command: str, expression: MathExpression) -> Any:
        """
        Führt einen T-Mathematics-Befehl aus
        
        Args:
            command: Auszuführender Befehl
            expression: Mathematischer Ausdruck
            
        Returns:
            Ergebnis der Ausführung
        """
        if not self.t_math_engine:
            logger.warning("T-Mathematics-Engine nicht initialisiert")
            return None
        
        # Führe den Befehl aus
        try:
            # Extrahiere die Methode und Parameter
            if command.startswith("TMath."):
                method_name = command.split("(")[0].split(".")[1]
                
                # Hole die Methode
                if not hasattr(self.t_math_engine, method_name):
                    logger.warning(f"Methode {method_name} nicht in T-Mathematics-Engine gefunden")
                    return None
                
                method = getattr(self.t_math_engine, method_name)
                
                # Extrahiere die Parameter
                params_str = command.split("(", 1)[1].rsplit(")", 1)[0]
                
                # Führe die Methode aus
                if expression.expression_type == "scalar":
                    return method(expression.expression_text)
                elif expression.expression_type == "vector":
                    return method(expression.parsed_expression)
                elif expression.expression_type == "matrix":
                    return method(expression.parsed_expression)
                elif expression.expression_type == "equation":
                    equation = expression.parsed_expression["equation"]
                    variable = expression.parsed_expression["variable"]
                    return method(equation, variable)
                elif expression.expression_type == "function":
                    name = expression.parsed_expression["name"]
                    parameters = expression.parsed_expression["parameters"]
                    body = expression.parsed_expression["body"]
                    return method(name, parameters, body)
                elif expression.expression_type == "function_evaluation":
                    name = expression.parsed_expression["name"]
                    parameters = expression.parsed_expression["parameters"]
                    return method(name, parameters)
            
            logger.warning(f"Ungültiger T-Mathematics-Befehl: {command}")
            return None
        except Exception as e:
            logger.error(f"Fehler bei der Ausführung des T-Mathematics-Befehls: {e}")
            raise
    
    def process_semantic_result(self, semantic_result: SemanticResult) -> Optional[MathBridgeResult]:
        """
        Verarbeitet ein semantisches Ergebnis
        
        Args:
            semantic_result: Semantisches Ergebnis
            
        Returns:
            MathBridgeResult-Objekt oder None
        """
        # Überprüfe, ob das semantische Ergebnis mathematische Ausdrücke enthält
        if semantic_result.parsed_command.intent != "QUERY" and "math" not in semantic_result.parsed_command.parameters:
            return None
        
        # Extrahiere den Text
        text = semantic_result.parsed_command.original_text
        language_code = semantic_result.parsed_command.detected_language
        
        # Verarbeite den mathematischen Ausdruck
        return self.process_math_expression(text, language_code)

# Erstelle eine Instanz der Mathematik-Brücke, wenn dieses Skript direkt ausgeführt wird
if __name__ == "__main__":
    math_bridge = MathBridge()
    
    # Beispieltexte
    test_texts = {
        "de": [
            "Berechne 2 + 3 * 4",
            "Was ist 10 / 2 + 5",
            "Erstelle einen Vektor mit den Werten 1, 2, 3, 4",
            "Matrix [1, 2; 3, 4]",
            "Löse die Gleichung x^2 + 2*x - 3 = 0 für x",
            "Definiere die Funktion f(x) = x^2 + 2*x - 3"
        ],
        "en": [
            "Calculate 2 + 3 * 4",
            "What is 10 / 2 + 5",
            "Create a vector with values 1, 2, 3, 4",
            "Matrix [1, 2; 3, 4]",
            "Solve the equation x^2 + 2*x - 3 = 0 for x",
            "Define the function f(x) = x^2 + 2*x - 3"
        ]
    }
    
    # Teste die Verarbeitung
    for lang, texts in test_texts.items():
        print(f"\nSprache: {lang}")
        for text in texts:
            print(f"\nVerarbeite Text: {text}")
            result = math_bridge.process_math_expression(text, lang)
            
            print(f"Ergebnis:")
            print(f"  Erfolg: {result.success}")
            
            if result.success:
                print(f"  Ausdruckstyp: {result.math_expression.expression_type}")
                print(f"  Ausdruck: {result.math_expression.expression_text}")
                print(f"  T-Mathematics-Befehl: {result.t_math_command}")
                print(f"  Ergebnis: {result.result}")
                print(f"  Ausführungszeit: {result.execution_time:.4f}s")
            else:
                print(f"  Fehlermeldung: {result.error_message}")
            
            print("-" * 50)
