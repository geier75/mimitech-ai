#!/usr/bin/env python3
"""
VXOR Nano-Step Test Generators - TDD Failing Tests
================================================

Dynamically generates failing tests for each violation found by scanner.
Tests MUST fail before fix and pass after fix.

Test Categories:
1. Directory naming compliance
2. File naming compliance  
3. Import statement correctness
4. Symbol naming (classes/functions/constants)
5. Case collision detection
6. Shim functionality
"""

import pytest
import json
import sys
import os
import importlib
import ast
from pathlib import Path
from typing import List, Dict, Any

# Add repo to path
REPO_ROOT = Path("/Volumes/My Book/MISO_Ultimate 15.32.28")
sys.path.insert(0, str(REPO_ROOT))

class ViolationTestGenerator:
    """Generates failing tests based on current violations"""
    
    def __init__(self):
        self.toolkit_dir = Path("/Users/gecko365/vxor_naming_toolkit")
        self.violations_db = self.toolkit_dir / "reports" / "violations.json"
        self.violations = self._load_violations()
    
    def _load_violations(self) -> List[Dict]:
        """Load current violations from database"""
        if not self.violations_db.exists():
            return []
        
        with open(self.violations_db, 'r') as f:
            return json.load(f)
    
    def get_violations_by_category(self, category: str) -> List[Dict]:
        """Get violations filtered by category"""
        return [v for v in self.violations if v['category'] == category]
    
    def get_pending_violations(self) -> List[Dict]:
        """Get all pending violations"""
        return [v for v in self.violations if v['status'] == 'pending']

@pytest.fixture(scope="session")
def violation_generator():
    """Fixture to provide violation test generator"""
    return ViolationTestGenerator()

class TestDirectoryNaming:
    """Test directory naming compliance - MUST FAIL before fix"""
    
    def test_directory_naming_compliance(self, violation_generator):
        """Test that all directories follow lower_snake_case"""
        directory_violations = violation_generator.get_violations_by_category("directories")
        
        if not directory_violations:
            pytest.skip("No directory violations found")
        
        violations_found = []
        
        for violation in directory_violations:
            if violation['status'] == 'pending':
                path = Path(violation['file_path'])
                if path.exists():
                    dirname = path.name
                    # This test SHOULD FAIL if violations exist
                    if not self._is_valid_directory_name(dirname):
                        violations_found.append({
                            'path': str(path),
                            'name': dirname,
                            'suggestion': violation['suggested_name']
                        })
        
        # This assertion WILL FAIL initially (expected in TDD)
        assert len(violations_found) == 0, (
            f"Directory naming violations found:\n" + 
            "\n".join([f"  {v['path']}: {v['name']} → {v['suggestion']}" 
                      for v in violations_found])
        )
    
    def _is_valid_directory_name(self, name: str) -> bool:
        """Check if directory name follows naming policy"""
        import re
        # Must be lower_snake_case
        return re.match(r'^[a-z][a-z0-9_]*$', name) is not None

class TestFileNaming:
    """Test file naming compliance - MUST FAIL before fix"""
    
    def test_file_naming_compliance(self, violation_generator):
        """Test that all Python files follow lower_snake_case.py"""
        file_violations = violation_generator.get_violations_by_category("files")
        
        if not file_violations:
            pytest.skip("No file violations found")
        
        violations_found = []
        
        for violation in file_violations:
            if violation['status'] == 'pending':
                path = Path(violation['file_path'])
                if path.exists() and path.suffix == '.py':
                    filename = path.name
                    # This test SHOULD FAIL if violations exist
                    if not self._is_valid_file_name(filename):
                        violations_found.append({
                            'path': str(path),
                            'name': filename,
                            'suggestion': violation['suggested_name']
                        })
        
        # This assertion WILL FAIL initially (expected in TDD)
        assert len(violations_found) == 0, (
            f"File naming violations found:\n" + 
            "\n".join([f"  {v['path']}: {v['name']} → {v['suggestion']}" 
                      for v in violations_found])
        )
    
    def _is_valid_file_name(self, name: str) -> bool:
        """Check if file name follows naming policy"""
        import re
        # Must be lower_snake_case.py
        return re.match(r'^[a-z][a-z0-9_]*\.py$', name) is not None

class TestImportStability:
    """Test import functionality - MUST FAIL if imports broken"""
    
    def test_critical_imports_work(self, violation_generator):
        """Test that critical VXOR imports function correctly"""
        critical_modules = [
            'vxor.core.vx_core',
            'agents.vx_memex', 
            'agents.vx_planner',
            'agents.vx_vision'
        ]
        
        import_failures = []
        
        for module_name in critical_modules:
            try:
                # Clear module cache to ensure fresh import
                if module_name in sys.modules:
                    del sys.modules[module_name]
                
                importlib.import_module(module_name)
            except ImportError as e:
                import_failures.append(f"{module_name}: {str(e)}")
            except Exception as e:
                import_failures.append(f"{module_name}: {str(e)}")
        
        # This WILL FAIL if imports are broken
        assert len(import_failures) == 0, (
            f"Import failures detected:\n" + 
            "\n".join([f"  {failure}" for failure in import_failures])
        )
    
    def test_import_naming_compliance(self, violation_generator):
        """Test that import statements use canonical names"""
        import_violations = violation_generator.get_violations_by_category("imports")
        
        if not import_violations:
            pytest.skip("No import violations found")
        
        violations_found = []
        
        for violation in import_violations:
            if violation['status'] == 'pending':
                # Check if deprecated patterns are still used
                current_name = violation['current_name']
                if any(pattern in current_name for pattern in ['VXOR', 'vXor', 'VxOr']):
                    violations_found.append({
                        'file': violation['file_path'],
                        'import': current_name,
                        'suggestion': violation['suggested_name']
                    })
        
        # This WILL FAIL initially if deprecated imports exist
        assert len(violations_found) == 0, (
            f"Deprecated import patterns found:\n" + 
            "\n".join([f"  {v['file']}: {v['import']} → {v['suggestion']}" 
                      for v in violations_found])
        )

class TestSymbolNaming:
    """Test symbol naming compliance - MUST FAIL before fix"""
    
    def test_class_naming_compliance(self, violation_generator):
        """Test that classes follow PascalCase with Vxor/Vx prefix"""
        class_violations = [v for v in violation_generator.get_violations_by_category("symbols") 
                           if 'class' in v.get('violation_type', '').lower()]
        
        if not class_violations:
            pytest.skip("No class violations found")
        
        violations_found = []
        
        for violation in class_violations:
            if violation['status'] == 'pending':
                class_name = violation['current_name']
                # Check naming compliance
                if not self._is_valid_class_name(class_name):
                    violations_found.append({
                        'file': violation['file_path'],
                        'class': class_name,
                        'suggestion': violation['suggested_name'],
                        'rule': violation['rule']
                    })
        
        # This WILL FAIL initially
        assert len(violations_found) == 0, (
            f"Class naming violations found:\n" + 
            "\n".join([f"  {v['file']}: {v['class']} → {v['suggestion']} ({v['rule']})" 
                      for v in violations_found])
        )
    
    def test_constant_naming_compliance(self, violation_generator):
        """Test that constants follow UPPER_SNAKE_CASE with prefix"""
        constant_violations = [v for v in violation_generator.get_violations_by_category("symbols")
                              if 'constant' in v.get('violation_type', '').lower()]
        
        if not constant_violations:
            pytest.skip("No constant violations found")
        
        violations_found = []
        
        for violation in constant_violations:
            if violation['status'] == 'pending':
                const_name = violation['current_name']
                if not self._is_valid_constant_name(const_name):
                    violations_found.append({
                        'file': violation['file_path'],
                        'constant': const_name,
                        'suggestion': violation['suggested_name']
                    })
        
        # This WILL FAIL initially
        assert len(violations_found) == 0, (
            f"Constant naming violations found:\n" + 
            "\n".join([f"  {v['file']}: {v['constant']} → {v['suggestion']}" 
                      for v in violations_found])
        )
    
    def _is_valid_class_name(self, name: str) -> bool:
        """Check if class name is valid"""
        import re
        # Must be PascalCase with Vxor/Vx prefix
        if not re.match(r'^[A-Z][a-zA-Z0-9]*$', name):
            return False
        return name.startswith('Vxor') or name.startswith('Vx')
    
    def _is_valid_constant_name(self, name: str) -> bool:
        """Check if constant name is valid"""
        import re
        # Must be UPPER_SNAKE_CASE with VXOR_/VX_ prefix
        if not re.match(r'^[A-Z][A-Z0-9_]*$', name):
            return False
        return name.startswith('VXOR_') or name.startswith('VX_')

class TestCaseCollisions:
    """Test for case collisions on case-insensitive filesystems"""
    
    def test_no_case_collisions(self, violation_generator):
        """Test that no case collisions exist"""
        collision_violations = violation_generator.get_violations_by_category("case_collisions")
        
        if not collision_violations:
            pytest.skip("No case collision violations found")
        
        collisions_found = []
        
        for violation in collision_violations:
            if violation['status'] == 'pending':
                collisions_found.append({
                    'collision': violation['current_name'],
                    'paths': [violation['file_path'], violation.get('suggested_name', '')]
                })
        
        # This WILL FAIL if collisions exist
        assert len(collisions_found) == 0, (
            f"Case collisions found:\n" + 
            "\n".join([f"  {c['collision']}: {' vs '.join(c['paths'])}" 
                      for c in collisions_found])
        )

class TestSingleViolation:
    """Dynamic tests for individual violations - used by nano-step CLI"""
    
    @pytest.fixture(autouse=True)
    def setup_violation_tests(self, violation_generator):
        """Setup individual violation tests"""
        self.generator = violation_generator
        self.violations = violation_generator.get_pending_violations()
    
    def _generate_test_for_violation(self, violation_id: str):
        """Generate a specific test for a violation ID"""
        violation = next((v for v in self.violations if v['id'] == violation_id), None)
        
        if not violation:
            pytest.skip(f"Violation {violation_id} not found or already fixed")
        
        # Generate appropriate test based on category
        if violation['category'] == 'directories':
            return self._test_directory_violation(violation)
        elif violation['category'] == 'files':
            return self._test_file_violation(violation)
        elif violation['category'] == 'imports':
            return self._test_import_violation(violation)
        elif violation['category'] == 'symbols':
            return self._test_symbol_violation(violation)
        else:
            pytest.skip(f"No test generator for category: {violation['category']}")
    
    def _test_directory_violation(self, violation: Dict) -> bool:
        """Test specific directory violation"""
        path = Path(violation['file_path'])
        if not path.exists():
            return True  # Fixed - directory renamed or removed
        
        dirname = path.name
        # Should fail if violation still exists
        return self._is_valid_directory_name(dirname)
    
    def _test_file_violation(self, violation: Dict) -> bool:
        """Test specific file violation"""
        path = Path(violation['file_path'])
        if not path.exists():
            return True  # Fixed - file renamed or removed
        
        filename = path.name
        # Should fail if violation still exists
        return self._is_valid_file_name(filename)
    
    def _test_import_violation(self, violation: Dict) -> bool:
        """Test specific import violation"""
        # Check if deprecated pattern still exists in file
        try:
            with open(violation['file_path'], 'r') as f:
                content = f.read()
            
            deprecated_patterns = ['VXOR', 'vXor', 'VxOr']
            current_name = violation['current_name']
            
            # Should fail if deprecated pattern still in file
            for pattern in deprecated_patterns:
                if pattern in current_name and pattern in content:
                    return False
            
            return True  # Fixed
        except Exception:
            return False
    
    def _test_symbol_violation(self, violation: Dict) -> bool:
        """Test specific symbol violation"""
        try:
            with open(violation['file_path'], 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            current_name = violation['current_name']
            
            # Check if old symbol name still exists
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    if node.name == current_name:
                        # Check if it violates naming policy
                        if violation['violation_type'] == 'class':
                            return self._is_valid_class_name(node.name)
                        else:
                            return self._is_valid_function_name(node.name)
            
            return True  # Symbol not found or renamed
        except Exception:
            return False
    
    def _is_valid_directory_name(self, name: str) -> bool:
        import re
        return re.match(r'^[a-z][a-z0-9_]*$', name) is not None
    
    def _is_valid_file_name(self, name: str) -> bool:
        import re
        return re.match(r'^[a-z][a-z0-9_]*\.py$', name) is not None
    
    def _is_valid_class_name(self, name: str) -> bool:
        import re
        if not re.match(r'^[A-Z][a-zA-Z0-9]*$', name):
            return False
        return name.startswith('Vxor') or name.startswith('Vx')
    
    def _is_valid_function_name(self, name: str) -> bool:
        import re
        return re.match(r'^[a-z][a-z0-9_]*$', name) is not None

# Dynamic test generation for individual violations
def pytest_generate_tests(metafunc):
    """Generate tests dynamically based on current violations"""
    if 'violation_id' in metafunc.fixturenames:
        generator = ViolationTestGenerator()
        pending_violations = generator.get_pending_violations()
        
        # Generate test parameters for each pending violation
        violation_ids = [v['id'] for v in pending_violations[:10]]  # Limit for performance
        
        metafunc.parametrize('violation_id', violation_ids)

def test_violation_by_id(violation_id, violation_generator):
    """Test a specific violation by ID - used by nano-step CLI"""
    test_generator = TestSingleViolation()
    test_generator.generator = violation_generator
    test_generator.violations = violation_generator.get_pending_violations()
    
    # This test SHOULD FAIL before fix and PASS after fix
    result = test_generator._generate_test_for_violation(violation_id)
    
    assert result, f"Violation {violation_id} still exists and needs fixing"

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
