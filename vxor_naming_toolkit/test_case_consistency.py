#!/usr/bin/env python3
"""
VXOR Case-Consistency Test Suite - TDD FIRST APPROACH
====================================================

Diese Tests werden BEFORE Implementierung der Fixes geschrieben.
Sie werden initial FEHLSCHLAGEN und zeigen genau auf, was gefixt werden muss.

Test Strategy:
1. Scan alle Files/Directories für Case-Inkonsistenzen
2. Test alle Import-Statements auf Funktionsfähigkeit  
3. Detect Case-Kollisionen auf case-insensitive filesystems
4. Validate Naming-Policy Compliance
5. Test Alias-Shims funktionieren korrekt
"""

import pytest
import os
import sys
import json
import importlib
import ast
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple
import tempfile

# Repository root path
REPO_ROOT = Path("/Volumes/My Book/MISO_Ultimate 15.32.28")
CANONICAL_NAMING_PATH = Path("/Users/gecko365/vxor_naming_toolkit/canonical_naming.json")

class VxorCaseConsistencyTests:
    """Test Suite für VXOR Case-Consistency - TDD First"""
    
    def __init__(self):
        with open(CANONICAL_NAMING_PATH, 'r') as f:
            self.naming_policy = json.load(f)
        
        self.repo_root = REPO_ROOT
        self.case_violations = []
        self.import_failures = []
        self.case_collisions = []
        
    def scan_repo_for_vxor_patterns(self) -> Dict[str, List[str]]:
        """Scanne Repo nach allen vxor/VXOR/vx/VX Mustern"""
        patterns = {
            'files': [],
            'directories': [], 
            'symbols': [],
            'imports': [],
            'constants': []
        }
        
        vxor_patterns = re.compile(r'(vxor|VXOR|vXor|VxOr|vx|VX|Vx)', re.IGNORECASE)
        
        for root, dirs, files in os.walk(self.repo_root):
            # Check directory names
            for d in dirs:
                if vxor_patterns.search(d):
                    patterns['directories'].append(os.path.join(root, d))
            
            # Check file names
            for f in files:
                if f.endswith('.py') and vxor_patterns.search(f):
                    patterns['files'].append(os.path.join(root, f))
                
                # Parse Python files for symbols and imports
                if f.endswith('.py'):
                    try:
                        file_path = os.path.join(root, f)
                        with open(file_path, 'r', encoding='utf-8') as file:
                            content = file.read()
                            tree = ast.parse(content)
                            
                            for node in ast.walk(tree):
                                # Import statements
                                if isinstance(node, (ast.Import, ast.ImportFrom)):
                                    if hasattr(node, 'names'):
                                        for alias in node.names:
                                            if vxor_patterns.search(alias.name):
                                                patterns['imports'].append(f"{file_path}:{alias.name}")
                                    if hasattr(node, 'module') and node.module:
                                        if vxor_patterns.search(node.module):
                                            patterns['imports'].append(f"{file_path}:{node.module}")
                                
                                # Class/Function definitions
                                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                                    if vxor_patterns.search(node.name):
                                        patterns['symbols'].append(f"{file_path}:{node.name}")
                                
                                # Constants (assignments at module level)
                                if isinstance(node, ast.Assign):
                                    for target in node.targets:
                                        if isinstance(target, ast.Name):
                                            if vxor_patterns.search(target.id):
                                                patterns['constants'].append(f"{file_path}:{target.id}")
                    except Exception as e:
                        print(f"Warning: Could not parse {file_path}: {e}")
        
        return patterns

class TestVxorCaseConsistency:
    """Pytest Test Cases - Diese werden initial FEHLSCHLAGEN"""
    
    @pytest.fixture(scope="class")
    def consistency_checker(self):
        return VxorCaseConsistencyTests()
    
    @pytest.fixture(scope="class") 
    def repo_patterns(self, consistency_checker):
        return consistency_checker.scan_repo_for_vxor_patterns()
    
    def test_file_naming_consistency(self, consistency_checker, repo_patterns):
        """Test: Alle Dateinamen folgen lower_snake_case Pattern"""
        violations = []
        
        for file_path in repo_patterns['files']:
            filename = os.path.basename(file_path)
            # Check if filename follows lower_snake_case
            if not re.match(r'^[a-z][a-z0-9_]*\.py$', filename):
                if any(pattern in filename.lower() for pattern in ['vxor', 'vx']):
                    violations.append(f"File naming violation: {file_path}")
        
        assert len(violations) == 0, f"File naming violations found:\n" + "\n".join(violations)
    
    def test_directory_naming_consistency(self, consistency_checker, repo_patterns):
        """Test: Alle Directory-Namen folgen lower_snake_case Pattern"""
        violations = []
        
        for dir_path in repo_patterns['directories']:
            dirname = os.path.basename(dir_path)
            # Check if directory follows lower_snake_case
            if not re.match(r'^[a-z][a-z0-9_]*$', dirname):
                if any(pattern in dirname.lower() for pattern in ['vxor', 'vx']):
                    violations.append(f"Directory naming violation: {dir_path}")
        
        assert len(violations) == 0, f"Directory naming violations found:\n" + "\n".join(violations)
    
    def test_class_naming_consistency(self, consistency_checker, repo_patterns):
        """Test: Alle Klassen folgen PascalCase mit Vxor/Vx Präfix"""
        violations = []
        
        for symbol_entry in repo_patterns['symbols']:
            file_path, symbol_name = symbol_entry.split(':', 1)
            
            # Check if it's a class (starts with capital letter)
            if symbol_name[0].isupper():
                # Check if it contains vxor/vx patterns
                if any(pattern in symbol_name.lower() for pattern in ['vxor', 'vx']):
                    # Should start with Vxor or Vx and be PascalCase
                    if not (symbol_name.startswith('Vxor') or symbol_name.startswith('Vx')):
                        violations.append(f"Class naming violation: {file_path}:{symbol_name}")
                    
                    # Check PascalCase compliance
                    if not re.match(r'^[A-Z][a-zA-Z0-9]*$', symbol_name):
                        violations.append(f"Class PascalCase violation: {file_path}:{symbol_name}")
        
        assert len(violations) == 0, f"Class naming violations found:\n" + "\n".join(violations)
    
    def test_constant_naming_consistency(self, consistency_checker, repo_patterns):
        """Test: Alle Konstanten folgen UPPER_SNAKE_CASE mit VXOR_/VX_ Präfix"""
        violations = []
        
        for const_entry in repo_patterns['constants']:
            file_path, const_name = const_entry.split(':', 1)
            
            # Check if it's a constant (all uppercase)
            if const_name.isupper():
                # Check if it contains vxor/vx patterns
                if any(pattern in const_name.lower() for pattern in ['vxor', 'vx']):
                    # Should start with VXOR_ or VX_
                    if not (const_name.startswith('VXOR_') or const_name.startswith('VX_')):
                        violations.append(f"Constant naming violation: {file_path}:{const_name}")
        
        assert len(violations) == 0, f"Constant naming violations found:\n" + "\n".join(violations)
    
    def test_import_functionality(self, consistency_checker, repo_patterns):
        """Test: Alle Import-Statements funktionieren ohne Fehler"""
        sys.path.insert(0, str(REPO_ROOT))
        import_failures = []
        
        # Test critical VXOR modules
        critical_modules = [
            'vxor.core.vx_core',
            'agents.vx_memex',
            'agents.vx_planner', 
            'agents.vx_vision'
        ]
        
        for module_name in critical_modules:
            try:
                importlib.import_module(module_name)
            except ImportError as e:
                import_failures.append(f"Import failure: {module_name} - {str(e)}")
            except Exception as e:
                import_failures.append(f"Import error: {module_name} - {str(e)}")
        
        assert len(import_failures) == 0, f"Import failures found:\n" + "\n".join(import_failures)
    
    def test_case_collision_detection(self, consistency_checker):
        """Test: Keine Case-Kollisionen auf case-insensitive filesystems"""
        collisions = []
        seen_paths_lower = {}
        
        for root, dirs, files in os.walk(REPO_ROOT):
            all_items = dirs + files
            for item in all_items:
                if any(pattern in item.lower() for pattern in ['vxor', 'vx']):
                    item_lower = item.lower()
                    full_path = os.path.join(root, item)
                    
                    if item_lower in seen_paths_lower:
                        collisions.append(f"Case collision: {full_path} vs {seen_paths_lower[item_lower]}")
                    else:
                        seen_paths_lower[item_lower] = full_path
        
        assert len(collisions) == 0, f"Case collisions found:\n" + "\n".join(collisions)
    
    def test_deprecated_patterns_usage(self, consistency_checker, repo_patterns):
        """Test: Deprecated patterns (VXOR, vXor, etc.) werden nicht verwendet"""
        deprecated_usage = []
        deprecated_patterns = ['VXOR', 'vXor', 'VxOr', 'VX', 'Vx']
        
        # Check in imports
        for import_entry in repo_patterns['imports']:
            for pattern in deprecated_patterns:
                if pattern in import_entry:
                    deprecated_usage.append(f"Deprecated pattern in import: {import_entry}")
        
        # Check in symbols
        for symbol_entry in repo_patterns['symbols']:
            for pattern in deprecated_patterns:
                if pattern in symbol_entry:
                    deprecated_usage.append(f"Deprecated pattern in symbol: {symbol_entry}")
        
        assert len(deprecated_usage) == 0, f"Deprecated patterns found:\n" + "\n".join(deprecated_usage)

# Test runner for standalone execution
if __name__ == "__main__":
    print("🧪 VXOR Case-Consistency Test Suite - TDD FIRST")
    print("=" * 50)
    print("Diese Tests werden initial FEHLSCHLAGEN!")
    print("Sie zeigen genau auf, was gefixt werden muss.\n")
    
    # Run tests and capture results
    checker = VxorCaseConsistencyTests() 
    patterns = checker.scan_repo_for_vxor_patterns()
    
    print(f"📊 REPO SCAN RESULTS:")
    print(f"   Files with VXOR patterns: {len(patterns['files'])}")
    print(f"   Directories with VXOR patterns: {len(patterns['directories'])}")
    print(f"   Symbols with VXOR patterns: {len(patterns['symbols'])}")
    print(f"   Imports with VXOR patterns: {len(patterns['imports'])}")
    print(f"   Constants with VXOR patterns: {len(patterns['constants'])}")
    
    print(f"\n🔍 SAMPLE FINDINGS:")
    for category, items in patterns.items():
        if items:
            print(f"   {category.upper()}:")
            for item in items[:3]:  # Show first 3
                print(f"     - {item}")
            if len(items) > 3:
                print(f"     ... and {len(items) - 3} more")
    
    print(f"\n⚠️  Run 'pytest {__file__}' to see all test failures!")
