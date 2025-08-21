"""
M-CODE Core - KI-native Programmiersprache für MISO Ultimate
=====================================================

M-CODE ist eine hochperformante, KI-optimierte Programmiersprache:
- Schneller als C, sicherer als Rust
- Optimiert für neuronale Netzwerke & KI-Logik
- Kompatibel mit GPU und Apple Neural Engine (M4 Max)
- Wird vom Omega-Kern kontrolliert

Hauptkomponenten:
1. MCodeParser - Zerlegt Code in Token, prüft Syntax
2. TypeChecker - Statische & dynamische Typanalyse, Sicherheitsprüfung
3. ASTCompiler - Erzeugt AST, wandelt in optimierten Bytecode/LLVM
4. GPU-JIT Execution Engine - Echtzeitkompilierung auf GPU/NE
5. Security Sandbox - Isolierte Ausführung, Erkennung schädlicher Muster
"""

import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Setup logging
logger = logging.getLogger("miso.lang.mcode")

# M-CODE Core Components
from .parser import MCodeParser
from .type_checker import MCodeTypeChecker  
from .ast_compiler import MCodeASTCompiler
from .gpu_jit_engine import MCodeGPUJITEngine
from .security_sandbox import MCodeSecuritySandbox
from .executor import MCodeExecutor

class MCodeCore:
    """
    M-CODE Core Engine - Zentrale Koordination aller M-CODE Komponenten
    
    Verwaltet den kompletten M-CODE Execution Pipeline:
    Parse → TypeCheck → Compile → Execute (with Security)
    """
    
    def __init__(self, omega_core_ref=None):
        """
        Initialisiert M-CODE Core Engine
        
        Args:
            omega_core_ref: Referenz zum Omega-Kern für Kontrolle und Verifikation
        """
        self.omega_core = omega_core_ref
        
        # Initialize core components
        self.parser = MCodeParser()
        self.type_checker = MCodeTypeChecker()
        self.ast_compiler = MCodeASTCompiler()
        self.gpu_jit_engine = MCodeGPUJITEngine()
        self.security_sandbox = MCodeSecuritySandbox(omega_core_ref)
        self.executor = MCodeExecutor(self)
        
        # M-CODE module registry (verified by Omega-Kern)
        self.verified_modules: Dict[str, Dict[str, Any]] = {}
        
        logger.info("M-CODE Core Engine initialized")
    
    def compile_and_verify(self, mcode_source: str, module_name: str = "inline") -> Dict[str, Any]:
        """
        Kompiliert M-CODE und verifiziert durch Omega-Kern
        
        Args:
            mcode_source: M-CODE Quellcode
            module_name: Name des Moduls
            
        Returns:
            Dict mit kompiliertem Modul und Metadata
        """
        try:
            # 1. Parse M-CODE source
            ast = self.parser.parse(mcode_source)
            
            # 2. Type check and security analysis
            type_info = self.type_checker.analyze(ast)
            
            # 3. Compile to optimized bytecode
            bytecode = self.ast_compiler.compile(ast, type_info)
            
            # 4. Security verification
            security_report = self.security_sandbox.verify_module(bytecode, module_name)
            
            # 5. Omega-Kern verification (if available)
            omega_verified = False
            if self.omega_core:
                omega_verified = self.omega_core.verify_mcode_module(bytecode, security_report)
            
            compiled_module = {
                "module_name": module_name,
                "source_hash": self._calculate_hash(mcode_source),
                "ast": ast,
                "type_info": type_info,
                "bytecode": bytecode,
                "security_report": security_report,
                "omega_verified": omega_verified,
                "timestamp": self._get_timestamp()
            }
            
            if omega_verified:
                self.verified_modules[module_name] = compiled_module
                logger.info(f"M-CODE module '{module_name}' verified by Omega-Kern")
            
            return compiled_module
            
        except Exception as e:
            logger.error(f"M-CODE compilation failed for '{module_name}': {e}")
            raise
    
    def execute_mcode(self, module_or_source: Union[str, Dict], context: Dict = None) -> Any:
        """
        Führt M-CODE aus (nur verifizierte Module)
        
        Args:
            module_or_source: Kompiliertes Modul oder M-CODE Quellcode
            context: Execution context
            
        Returns:
            Execution result
        """
        if isinstance(module_or_source, str):
            # Compile inline code
            module = self.compile_and_verify(module_or_source, "inline_exec")
        else:
            module = module_or_source
        
        # Security check - only execute verified modules
        if not module.get("omega_verified", False) and self.omega_core:
            raise SecurityError("Module not verified by Omega-Kern - execution denied")
        
        return self.executor.execute(module, context or {})
    
    def _calculate_hash(self, source: str) -> str:
        """Calculate deterministic hash of source code"""
        import hashlib
        return hashlib.sha256(source.encode()).hexdigest()
    
    def _get_timestamp(self) -> str:
        """Get ISO timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

class SecurityError(Exception):
    """Security violation in M-CODE execution"""
    pass

# Export main interface
__all__ = [
    'MCodeCore',
    'MCodeParser', 
    'MCodeTypeChecker',
    'MCodeASTCompiler',
    'MCodeGPUJITEngine', 
    'MCodeSecuritySandbox',
    'MCodeExecutor',
    'SecurityError'
]
