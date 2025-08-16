#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - M-CODE Security Sandbox

Dieser Modul implementiert die Sicherheits-Sandbox für die M-CODE Programmiersprache.
Die Sandbox isoliert die Ausführung von M-CODE und verhindert schädliche Operationen.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import hashlib
import time
import uuid
import inspect
import re
import ast
from enum import Enum, auto
from typing import Dict, Any, List, Tuple, Optional, Union, Set, Callable

# Konfiguriere Logging
logger = logging.getLogger("MISO.m_code.security")


class SecurityLevel(Enum):
    """Sicherheitsstufen für die M-CODE Sandbox"""
    
    LOW = 0      # Minimale Sicherheit, für vertrauenswürdigen Code
    MEDIUM = 1   # Standard-Sicherheit, für normalen Code
    HIGH = 2     # Hohe Sicherheit, für unbekannten Code
    MAXIMUM = 3  # Maximale Sicherheit, für potenziell gefährlichen Code


class SecurityViolation(Exception):
    """Ausnahme bei Sicherheitsverletzungen"""
    
    def __init__(self, message: str, violation_type: str, code_snippet: Optional[str] = None):
        """
        Initialisiert eine neue Sicherheitsverletzung.
        
        Args:
            message: Fehlermeldung
            violation_type: Art der Verletzung
            code_snippet: Betroffener Code-Ausschnitt
        """
        super().__init__(message)
        self.violation_type = violation_type
        self.code_snippet = code_snippet
        self.timestamp = time.time()


class CodePattern:
    """Muster für die Erkennung von schädlichem Code"""
    
    def __init__(self, name: str, pattern: str, description: str, severity: int):
        """
        Initialisiert ein neues Code-Muster.
        
        Args:
            name: Name des Musters
            pattern: Regulärer Ausdruck für das Muster
            description: Beschreibung des Musters
            severity: Schweregrad (0-3)
        """
        self.name = name
        self.pattern = re.compile(pattern, re.DOTALL)
        self.description = description
        self.severity = severity
    
    def matches(self, code: str) -> bool:
        """
        Prüft, ob der Code dem Muster entspricht.
        
        Args:
            code: Zu prüfender Code
            
        Returns:
            True, wenn der Code dem Muster entspricht
        """
        return bool(self.pattern.search(code))


class SecuritySandbox:
    """Sicherheits-Sandbox für M-CODE"""
    
    def __init__(self, security_level=None):
        """
        Initialisiert eine neue Sicherheits-Sandbox.
        
        Args:
            security_level: Sicherheitsstufe (SecurityLevel-Enum oder Dict)
        """
        # Wenn security_level ein Dictionary ist, verwende MEDIUM als Standard
        if isinstance(security_level, dict) or security_level is None:
            self.security_level = SecurityLevel.MEDIUM
        else:
            self.security_level = security_level
        
        # Initialisiere Muster für schädlichen Code
        self.patterns = self._initialize_patterns()
        
        # Initialisiere erlaubte Module und Funktionen
        self.allowed_modules = self._initialize_allowed_modules()
        self.allowed_builtins = self._initialize_allowed_builtins()
        
        # Initialisiere Ressourcenbeschränkungen
        self.resource_limits = self._initialize_resource_limits()
        
        # Initialisiere Ausführungsstatistiken
        self.execution_stats = {
            "total_executions": 0,
            "total_violations": 0,
            "violations_by_type": {},
            "last_execution_time": 0.0
        }
        
        logger.info(f"Sicherheits-Sandbox initialisiert: Level={self.security_level.name}")
    
    def _initialize_patterns(self) -> List[CodePattern]:
        """
        Initialisiert Muster für schädlichen Code.
        
        Returns:
            Liste von Code-Mustern
        """
        patterns = [
            # Systemzugriff
            CodePattern(
                name="system_access",
                pattern=r"(os\.(system|popen|exec|spawn)|subprocess\.|eval\(|exec\()",
                description="Direkter Systemzugriff",
                severity=3
            ),
            
            # Dateisystemzugriff
            CodePattern(
                name="file_access",
                pattern=r"(open\(|file\(|os\.(remove|unlink|rmdir|mkdir)|shutil\.(copy|move|rmtree))",
                description="Dateisystemzugriff",
                severity=2
            ),
            
            # Netzwerkzugriff
            CodePattern(
                name="network_access",
                pattern=r"(socket\.|urllib\.|requests\.|http\.client|ftplib\.|smtplib\.|telnetlib\.|urllib\d?\.)",
                description="Netzwerkzugriff",
                severity=2
            ),
            
            # Speicherzugriff
            CodePattern(
                name="memory_access",
                pattern=r"(ctypes\.|mmap\.|multiprocessing\.sharedctypes)",
                description="Direkter Speicherzugriff",
                severity=3
            ),
            
            # Prozesszugriff
            CodePattern(
                name="process_access",
                pattern=r"(multiprocessing\.|threading\.|concurrent\.futures)",
                description="Prozesszugriff",
                severity=1
            ),
            
            # Importieren von gefährlichen Modulen
            CodePattern(
                name="dangerous_import",
                pattern=r"import\s+(os|sys|subprocess|shutil|socket|urllib|requests|http|ftplib|smtplib|telnetlib)",
                description="Import gefährlicher Module",
                severity=2
            ),
            
            # Endlosschleifen
            CodePattern(
                name="infinite_loop",
                pattern=r"while\s+True\s*:",
                description="Potenzielle Endlosschleife",
                severity=1
            ),
            
            # Gefährliche Attribute
            CodePattern(
                name="dangerous_attribute",
                pattern=r"\.__(\w+)__",
                description="Zugriff auf interne Attribute",
                severity=2
            )
        ]
        
        return patterns
    
    def _initialize_allowed_modules(self) -> Dict[str, Set[str]]:
        """
        Initialisiert erlaubte Module und Funktionen.
        
        Returns:
            Dictionary mit erlaubten Modulen und Funktionen
        """
        # Basismodule, die immer erlaubt sind
        base_modules = {
            "math": set(),  # Alle Funktionen erlaubt
            "random": {"random", "randint", "choice", "sample", "uniform", "normalvariate"},
            "datetime": {"datetime", "timedelta", "date", "time"},
            "collections": {"defaultdict", "Counter", "deque", "namedtuple"},
            "itertools": set(),  # Alle Funktionen erlaubt
            "functools": {"partial", "reduce", "lru_cache"},
            "operator": set(),  # Alle Funktionen erlaubt
            "re": {"match", "search", "findall", "sub", "compile"},
            "json": {"loads", "dumps"},
            "base64": {"b64encode", "b64decode"},
            "hashlib": {"md5", "sha1", "sha256", "sha512"},
            "uuid": {"uuid4"},
            "copy": {"copy", "deepcopy"}
        }
        
        # Wissenschaftliche Module
        science_modules = {
            "numpy": set(),  # Alle Funktionen erlaubt
            "pandas": set(),  # Alle Funktionen erlaubt
            "scipy": set(),  # Alle Funktionen erlaubt
            "sklearn": set(),  # Alle Funktionen erlaubt
            "torch": set(),  # Alle Funktionen erlaubt
            "tensorflow": set()  # Alle Funktionen erlaubt
        }
        
        # Kombiniere Module basierend auf Sicherheitsstufe
        allowed_modules = {}
        allowed_modules.update(base_modules)
        
        # Wenn die Sicherheitsstufe nicht HIGH oder höher ist, erlaube wissenschaftliche Module
        if (isinstance(self.security_level, SecurityLevel) and 
            self.security_level.value < SecurityLevel.HIGH.value):
            allowed_modules.update(science_modules)
        
        return allowed_modules
    
    def _initialize_allowed_builtins(self) -> Set[str]:
        """
        Initialisiert erlaubte Built-in-Funktionen.
        
        Returns:
            Set mit erlaubten Built-in-Funktionen
        """
        # Sichere Built-ins
        safe_builtins = {
            "abs", "all", "any", "ascii", "bin", "bool", "bytes", "callable",
            "chr", "complex", "dict", "dir", "divmod", "enumerate", "filter",
            "float", "format", "frozenset", "getattr", "hasattr", "hash",
            "hex", "id", "int", "isinstance", "issubclass", "iter", "len",
            "list", "map", "max", "min", "next", "object", "oct", "ord",
            "pow", "print", "range", "repr", "reversed", "round", "set",
            "slice", "sorted", "str", "sum", "tuple", "type", "zip"
        }
        
        # Potenziell gefährliche Built-ins, die bei niedrigeren Sicherheitsstufen erlaubt sind
        medium_builtins = {
            "classmethod", "compile", "delattr", "property", "setattr", "staticmethod",
            "super", "vars"
        }
        
        # Kombiniere Built-ins basierend auf Sicherheitsstufe
        allowed_builtins = safe_builtins.copy()
        
        if self.security_level.value < SecurityLevel.HIGH.value:
            allowed_builtins.update(medium_builtins)
        
        return allowed_builtins
    
    def _initialize_resource_limits(self) -> Dict[str, Any]:
        """
        Initialisiert Ressourcenbeschränkungen.
        
        Returns:
            Dictionary mit Ressourcenbeschränkungen
        """
        # Basisgrenzen
        base_limits = {
            "max_execution_time": 10.0,  # Sekunden
            "max_memory": 100 * 1024 * 1024,  # 100 MB
            "max_iterations": 1000000,  # Maximale Anzahl von Schleifeniterationen
            "max_recursion_depth": 100,  # Maximale Rekursionstiefe
            "max_tensor_size": 10000 * 10000  # Maximale Tensorgröße (Elemente)
        }
        
        # Passe Grenzen basierend auf Sicherheitsstufe an
        if self.security_level == SecurityLevel.LOW:
            base_limits["max_execution_time"] = 60.0
            base_limits["max_memory"] = 1024 * 1024 * 1024  # 1 GB
            base_limits["max_iterations"] = 10000000
            base_limits["max_recursion_depth"] = 1000
            base_limits["max_tensor_size"] = 100000 * 100000
        elif self.security_level == SecurityLevel.MEDIUM:
            base_limits["max_execution_time"] = 30.0
            base_limits["max_memory"] = 512 * 1024 * 1024  # 512 MB
            base_limits["max_iterations"] = 5000000
            base_limits["max_recursion_depth"] = 500
            base_limits["max_tensor_size"] = 50000 * 50000
        elif self.security_level == SecurityLevel.MAXIMUM:
            base_limits["max_execution_time"] = 5.0
            base_limits["max_memory"] = 50 * 1024 * 1024  # 50 MB
            base_limits["max_iterations"] = 100000
            base_limits["max_recursion_depth"] = 50
            base_limits["max_tensor_size"] = 1000 * 1000
        
        return base_limits
    
    def verify_code(self, code: str) -> Tuple[bool, Optional[SecurityViolation]]:
        """
        Überprüft Code auf Sicherheitsverletzungen.
        
        Args:
            code: Zu überprüfender Code
            
        Returns:
            Tupel aus (ist_sicher, Verletzung)
        """
        # Prüfe auf schädliche Muster
        for pattern in self.patterns:
            if pattern.matches(code):
                # Ignoriere Muster mit niedrigem Schweregrad bei niedriger Sicherheitsstufe
                if pattern.severity <= 1 and self.security_level == SecurityLevel.LOW:
                    continue
                
                violation = SecurityViolation(
                    message=f"Schädliches Codemuster erkannt: {pattern.description}",
                    violation_type=pattern.name,
                    code_snippet=code
                )
                
                # Aktualisiere Statistiken
                self.execution_stats["total_violations"] += 1
                if pattern.name not in self.execution_stats["violations_by_type"]:
                    self.execution_stats["violations_by_type"][pattern.name] = 0
                self.execution_stats["violations_by_type"][pattern.name] += 1
                
                return False, violation
        
        # Prüfe auf AST-Ebene
        try:
            tree = ast.parse(code)
            result, violation = self._verify_ast(tree)
            if not result:
                # Aktualisiere Statistiken
                self.execution_stats["total_violations"] += 1
                if violation.violation_type not in self.execution_stats["violations_by_type"]:
                    self.execution_stats["violations_by_type"][violation.violation_type] = 0
                self.execution_stats["violations_by_type"][violation.violation_type] += 1
                
                return False, violation
        except SyntaxError as e:
            violation = SecurityViolation(
                message=f"Syntaxfehler: {str(e)}",
                violation_type="syntax_error",
                code_snippet=code
            )
            
            # Aktualisiere Statistiken
            self.execution_stats["total_violations"] += 1
            if "syntax_error" not in self.execution_stats["violations_by_type"]:
                self.execution_stats["violations_by_type"]["syntax_error"] = 0
            self.execution_stats["violations_by_type"]["syntax_error"] += 1
            
            return False, violation
        
        return True, None
    
    def _verify_ast(self, tree: ast.AST) -> Tuple[bool, Optional[SecurityViolation]]:
        """
        Überprüft einen AST auf Sicherheitsverletzungen.
        
        Args:
            tree: Abstrakter Syntaxbaum
            
        Returns:
            Tupel aus (ist_sicher, Verletzung)
        """
        # Prüfe auf gefährliche Imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    if name.name.split('.')[0] not in self.allowed_modules:
                        return False, SecurityViolation(
                            message=f"Nicht erlaubter Import: {name.name}",
                            violation_type="forbidden_import",
                            code_snippet=ast.unparse(node) if hasattr(ast, 'unparse') else None
                        )
            
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                
                module = node.module.split('.')[0]
                if module not in self.allowed_modules:
                    return False, SecurityViolation(
                        message=f"Nicht erlaubter Import: {node.module}",
                        violation_type="forbidden_import",
                        code_snippet=ast.unparse(node) if hasattr(ast, 'unparse') else None
                    )
                
                # Prüfe, ob die importierten Namen erlaubt sind
                allowed_names = self.allowed_modules[module]
                if allowed_names:  # Leeres Set bedeutet, dass alle Namen erlaubt sind
                    for name in node.names:
                        if name.name not in allowed_names:
                            return False, SecurityViolation(
                                message=f"Nicht erlaubter Import: {node.module}.{name.name}",
                                violation_type="forbidden_import",
                                code_snippet=ast.unparse(node) if hasattr(ast, 'unparse') else None
                            )
            
            # Prüfe auf gefährliche Funktionsaufrufe
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name not in self.allowed_builtins and func_name not in {"print", "len", "range"}:
                        return False, SecurityViolation(
                            message=f"Nicht erlaubter Funktionsaufruf: {func_name}",
                            violation_type="forbidden_function",
                            code_snippet=ast.unparse(node) if hasattr(ast, 'unparse') else None
                        )
                
                elif isinstance(node.func, ast.Attribute):
                    # Prüfe auf gefährliche Attribute
                    if node.func.attr.startswith('__') and node.func.attr.endswith('__'):
                        return False, SecurityViolation(
                            message=f"Zugriff auf internes Attribut: {node.func.attr}",
                            violation_type="dangerous_attribute",
                            code_snippet=ast.unparse(node) if hasattr(ast, 'unparse') else None
                        )
        
        return True, None
    
    def create_secure_globals(self) -> Dict[str, Any]:
        """
        Erstellt ein sicheres globals-Dictionary für die Ausführung.
        
        Returns:
            Sicheres globals-Dictionary
        """
        secure_globals = {}
        
        # Füge erlaubte Built-ins hinzu
        for name in self.allowed_builtins:
            if name in __builtins__:
                secure_globals[name] = __builtins__[name]
        
        # Füge erlaubte Module hinzu
        for module_name in self.allowed_modules:
            try:
                module = __import__(module_name)
                
                # Wenn bestimmte Funktionen erlaubt sind, füge nur diese hinzu
                allowed_functions = self.allowed_modules[module_name]
                if allowed_functions:
                    module_dict = {}
                    for func_name in allowed_functions:
                        if hasattr(module, func_name):
                            module_dict[func_name] = getattr(module, func_name)
                    secure_globals[module_name] = module_dict
                else:
                    # Alle Funktionen sind erlaubt
                    secure_globals[module_name] = module
            
            except ImportError:
                # Modul nicht verfügbar, ignorieren
                pass
        
        return secure_globals
    
    def create_secure_locals(self) -> Dict[str, Any]:
        """
        Erstellt ein sicheres locals-Dictionary für die Ausführung.
        
        Returns:
            Sicheres locals-Dictionary
        """
        return {}
    
    def generate_code_hash(self, code: str, source: str = "unknown") -> str:
        """
        Generiert einen Hash für einen Code-Block.
        
        Args:
            code: Code-Block
            source: Quelle des Codes
            
        Returns:
            Hash-ID
        """
        # Erstelle Hash-Objekt
        hash_obj = hashlib.sha256()
        
        # Füge Code hinzu
        hash_obj.update(code.encode())
        
        # Füge Zeitstempel hinzu
        timestamp = str(time.time()).encode()
        hash_obj.update(timestamp)
        
        # Füge Quelle hinzu
        hash_obj.update(source.encode())
        
        # Erstelle Hash-ID
        hash_id = hash_obj.hexdigest()
        
        return hash_id
    
    def execute_code(self, code: str, globals_dict: Optional[Dict[str, Any]] = None,
                    locals_dict: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Führt Code in der Sandbox aus.
        
        Args:
            code: Auszuführender Code
            globals_dict: Globale Variablen
            locals_dict: Lokale Variablen
            
        Returns:
            Tupel aus (Rückgabewert, Lokale Variablen)
            
        Raises:
            SecurityViolation: Bei Sicherheitsverletzungen
        """
        # Aktualisiere Statistiken
        self.execution_stats["total_executions"] += 1
        
        # Überprüfe Code
        is_safe, violation = self.verify_code(code)
        if not is_safe:
            raise violation
        
        # Erstelle sichere Umgebung
        if globals_dict is None:
            globals_dict = self.create_secure_globals()
        
        if locals_dict is None:
            locals_dict = self.create_secure_locals()
        
        # Setze Ressourcenbeschränkungen
        old_recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(self.resource_limits["max_recursion_depth"])
        
        # Führe Code aus
        start_time = time.time()
        result = None
        
        try:
            # Kompiliere Code
            compiled_code = compile(code, "<string>", "exec")
            
            # Führe Code aus
            exec(compiled_code, globals_dict, locals_dict)
            
            # Prüfe auf Rückgabewert
            if "result" in locals_dict:
                result = locals_dict["result"]
        
        except Exception as e:
            # Fange alle Ausnahmen und gib sie weiter
            raise RuntimeError(f"Fehler bei der Ausführung: {str(e)}")
        
        finally:
            # Stelle Ressourcenbeschränkungen wieder her
            sys.setrecursionlimit(old_recursion_limit)
            
            # Aktualisiere Statistiken
            end_time = time.time()
            execution_time = end_time - start_time
            self.execution_stats["last_execution_time"] = execution_time
            
            # Prüfe Zeitlimit
            if execution_time > self.resource_limits["max_execution_time"]:
                raise SecurityViolation(
                    message=f"Zeitlimit überschritten: {execution_time:.2f}s (Limit: {self.resource_limits['max_execution_time']}s)",
                    violation_type="time_limit_exceeded"
                )
        
        return result, locals_dict
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Gibt Ausführungsstatistiken zurück.
        
        Returns:
            Ausführungsstatistiken
        """
        return self.execution_stats
    
    def set_security_level(self, level: SecurityLevel) -> None:
        """
        Setzt die Sicherheitsstufe.
        
        Args:
            level: Neue Sicherheitsstufe
        """
        self.security_level = level
        
        # Aktualisiere Ressourcenbeschränkungen
        self.resource_limits = self._initialize_resource_limits()
        
        # Aktualisiere erlaubte Module und Funktionen
        self.allowed_modules = self._initialize_allowed_modules()
        self.allowed_builtins = self._initialize_allowed_builtins()
        
        logger.info(f"Sicherheitsstufe geändert: {level.name}")


class CodeVerifier:
    """Verifizierer für M-CODE"""
    
    def __init__(self, sandbox: SecuritySandbox):
        """
        Initialisiert einen neuen Code-Verifizierer.
        
        Args:
            sandbox: Sicherheits-Sandbox
        """
        self.sandbox = sandbox
        self.verified_hashes = set()
    
    def verify_code(self, code: str, source: str = "unknown") -> Tuple[bool, str, Optional[SecurityViolation]]:
        """
        Verifiziert einen Code-Block.
        
        Args:
            code: Zu verifizierender Code
            source: Quelle des Codes
            
        Returns:
            Tupel aus (ist_verifiziert, Hash-ID, Verletzung)
        """
        # Generiere Hash-ID
        hash_id = self.sandbox.generate_code_hash(code, source)
        
        # Prüfe, ob der Code bereits verifiziert wurde
        if hash_id in self.verified_hashes:
            return True, hash_id, None
        
        # Überprüfe Code
        is_safe, violation = self.sandbox.verify_code(code)
        if not is_safe:
            return False, hash_id, violation
        
        # Markiere Code als verifiziert
        self.verified_hashes.add(hash_id)
        
        return True, hash_id, None
    
    def is_verified(self, hash_id: str) -> bool:
        """
        Prüft, ob ein Code-Block verifiziert ist.
        
        Args:
            hash_id: Hash-ID des Code-Blocks
            
        Returns:
            True, wenn der Code-Block verifiziert ist
        """
        return hash_id in self.verified_hashes
