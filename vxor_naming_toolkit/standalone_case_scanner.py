#!/usr/bin/env python3
"""
VXOR Case-Consistency Scanner - Standalone TDD Implementation
===========================================================

Scant ohne pytest dependencies - zeigt alle Case-Inkonsistenzen auf.
"""

import os
import sys
import json
import ast
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple

# Repository root path
REPO_ROOT = Path("/Volumes/My Book/MISO_Ultimate 15.32.28")
CANONICAL_NAMING_PATH = Path("/Users/gecko365/vxor_naming_toolkit/canonical_naming.json")

class VxorCaseScanner:
    """Standalone Scanner für VXOR Case-Inkonsistenzen"""
    
    def __init__(self):
        with open(CANONICAL_NAMING_PATH, 'r') as f:
            self.naming_policy = json.load(f)
        
        self.repo_root = REPO_ROOT
        self.violations = {
            'files': [],
            'directories': [], 
            'classes': [],
            'functions': [],
            'constants': [],
            'imports': [],
            'case_collisions': []
        }
        
    def scan_all_patterns(self) -> Dict[str, List]:
        """Vollständiger Scan aller VXOR/VX Muster"""
        print("🔍 Scanning repository for VXOR/VX patterns...")
        
        vxor_patterns = re.compile(r'(vxor|VXOR|vXor|VxOr|vx|VX|Vx)', re.IGNORECASE)
        seen_lowercase = {}
        
        for root, dirs, files in os.walk(self.repo_root):
            rel_root = os.path.relpath(root, self.repo_root)
            
            # Check directory names
            for d in dirs:
                if vxor_patterns.search(d):
                    full_path = os.path.join(root, d)
                    self._check_directory_naming(d, full_path)
                    self._check_case_collision(d.lower(), full_path, seen_lowercase)
            
            # Check file names and contents
            for f in files:
                if vxor_patterns.search(f):
                    full_path = os.path.join(root, f)
                    self._check_file_naming(f, full_path)
                    self._check_case_collision(f.lower(), full_path, seen_lowercase)
                
                # Parse Python files
                if f.endswith('.py'):
                    self._scan_python_file(os.path.join(root, f))
        
        return self.violations
    
    def _check_file_naming(self, filename: str, full_path: str):
        """Check file naming conventions"""
        if filename.endswith('.py'):
            # Should be lower_snake_case.py
            if not re.match(r'^[a-z][a-z0-9_]*\.py$', filename):
                self.violations['files'].append({
                    'path': full_path,
                    'filename': filename,
                    'violation': 'Not lower_snake_case',
                    'suggestion': self._suggest_filename(filename)
                })
    
    def _check_directory_naming(self, dirname: str, full_path: str):
        """Check directory naming conventions"""
        # Should be lower_snake_case
        if not re.match(r'^[a-z][a-z0-9_]*$', dirname):
            self.violations['directories'].append({
                'path': full_path,
                'dirname': dirname,
                'violation': 'Not lower_snake_case', 
                'suggestion': self._suggest_dirname(dirname)
            })
    
    def _check_case_collision(self, lowercase_name: str, full_path: str, seen_lowercase: Dict):
        """Check for case collisions"""
        if lowercase_name in seen_lowercase:
            self.violations['case_collisions'].append({
                'collision': lowercase_name,
                'path1': seen_lowercase[lowercase_name],
                'path2': full_path
            })
        else:
            seen_lowercase[lowercase_name] = full_path
    
    def _scan_python_file(self, file_path: str):
        """Scan Python file for naming violations"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                # Import statements
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    self._check_imports(node, file_path)
                
                # Class definitions
                if isinstance(node, ast.ClassDef):
                    self._check_class_naming(node.name, file_path)
                
                # Function definitions  
                if isinstance(node, ast.FunctionDef):
                    self._check_function_naming(node.name, file_path)
                
                # Module-level assignments (constants)
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self._check_constant_naming(target.id, file_path)
                            
        except Exception as e:
            print(f"Warning: Could not parse {file_path}: {e}")
    
    def _check_imports(self, node, file_path: str):
        """Check import statement naming"""
        vxor_pattern = re.compile(r'(vxor|VXOR|vXor|VxOr|vx|VX|Vx)', re.IGNORECASE)
        
        if hasattr(node, 'names'):
            for alias in node.names:
                if vxor_pattern.search(alias.name):
                    # Check for deprecated patterns
                    if any(deprecated in alias.name for deprecated in ['VXOR', 'vXor', 'VxOr', 'VX', 'Vx']):
                        self.violations['imports'].append({
                            'file': file_path,
                            'import': alias.name,
                            'violation': 'Uses deprecated pattern',
                            'suggestion': self._suggest_import_name(alias.name)
                        })
        
        if hasattr(node, 'module') and node.module:
            if vxor_pattern.search(node.module):
                if any(deprecated in node.module for deprecated in ['VXOR', 'vXor', 'VxOr', 'VX', 'Vx']):
                    self.violations['imports'].append({
                        'file': file_path,
                        'module': node.module,
                        'violation': 'Uses deprecated pattern',
                        'suggestion': self._suggest_module_name(node.module)
                    })
    
    def _check_class_naming(self, class_name: str, file_path: str):
        """Check class naming conventions"""
        vxor_pattern = re.compile(r'(vxor|VXOR|vXor|VxOr|vx|VX|Vx)', re.IGNORECASE)
        
        if vxor_pattern.search(class_name):
            # Should be PascalCase with Vxor/Vx prefix
            if not (class_name.startswith('Vxor') or class_name.startswith('Vx')):
                self.violations['classes'].append({
                    'file': file_path,
                    'class': class_name,
                    'violation': 'Missing Vxor/Vx prefix',
                    'suggestion': self._suggest_class_name(class_name)
                })
            
            # Check PascalCase
            if not re.match(r'^[A-Z][a-zA-Z0-9]*$', class_name):
                self.violations['classes'].append({
                    'file': file_path,
                    'class': class_name,
                    'violation': 'Not PascalCase',
                    'suggestion': self._suggest_class_name(class_name)
                })
    
    def _check_function_naming(self, func_name: str, file_path: str):
        """Check function naming conventions"""
        vxor_pattern = re.compile(r'(vxor|VXOR|vXor|VxOr|vx|VX|Vx)', re.IGNORECASE)
        
        if vxor_pattern.search(func_name):
            # Should be lower_snake_case
            if not re.match(r'^[a-z][a-z0-9_]*$', func_name):
                self.violations['functions'].append({
                    'file': file_path,
                    'function': func_name,
                    'violation': 'Not lower_snake_case',
                    'suggestion': self._suggest_function_name(func_name)
                })
    
    def _check_constant_naming(self, const_name: str, file_path: str):
        """Check constant naming conventions"""
        vxor_pattern = re.compile(r'(vxor|VXOR|vXor|VxOr|vx|VX|Vx)', re.IGNORECASE)
        
        if vxor_pattern.search(const_name) and const_name.isupper():
            # Should have VXOR_ or VX_ prefix
            if not (const_name.startswith('VXOR_') or const_name.startswith('VX_')):
                self.violations['constants'].append({
                    'file': file_path,
                    'constant': const_name,
                    'violation': 'Missing VXOR_/VX_ prefix',
                    'suggestion': self._suggest_constant_name(const_name)
                })
    
    def _suggest_filename(self, filename: str) -> str:
        """Suggest canonical filename"""
        name = filename.replace('.py', '')
        suggested = re.sub(r'[^a-z0-9_]', '_', name.lower())
        return f"{suggested}.py"
    
    def _suggest_dirname(self, dirname: str) -> str:
        """Suggest canonical directory name"""
        return re.sub(r'[^a-z0-9_]', '_', dirname.lower())
    
    def _suggest_class_name(self, class_name: str) -> str:
        """Suggest canonical class name"""
        if 'vxor' in class_name.lower():
            return f"Vxor{class_name.split('vxor')[-1].title()}"
        elif 'vx' in class_name.lower():
            return f"Vx{class_name.split('vx')[-1].title()}"
        return class_name
    
    def _suggest_function_name(self, func_name: str) -> str:
        """Suggest canonical function name"""
        return re.sub(r'[^a-z0-9_]', '_', func_name.lower())
    
    def _suggest_constant_name(self, const_name: str) -> str:
        """Suggest canonical constant name"""
        if 'vxor' in const_name.lower():
            return f"VXOR_{const_name.upper()}"
        elif 'vx' in const_name.lower():
            return f"VX_{const_name.upper()}"
        return const_name
    
    def _suggest_import_name(self, import_name: str) -> str:
        """Suggest canonical import name"""
        return import_name.replace('VXOR', 'vxor').replace('vXor', 'vxor').replace('VxOr', 'vxor')
    
    def _suggest_module_name(self, module_name: str) -> str:
        """Suggest canonical module name"""
        return module_name.replace('VXOR', 'vxor').replace('vXor', 'vxor').replace('VxOr', 'vxor')
    
    def print_report(self):
        """Print comprehensive report"""
        print("\n" + "="*80)
        print("📊 VXOR CASE-CONSISTENCY SCAN REPORT")
        print("="*80)
        
        total_violations = sum(len(v) for v in self.violations.values())
        print(f"Total violations found: {total_violations}")
        
        for category, violations in self.violations.items():
            if violations:
                print(f"\n🚨 {category.upper()} VIOLATIONS ({len(violations)}):")
                print("-" * 40)
                
                for violation in violations[:5]:  # Show first 5
                    if category == 'files':
                        print(f"   File: {violation['filename']}")
                        print(f"   Path: {violation['path']}")
                        print(f"   Issue: {violation['violation']}")
                        print(f"   Fix: {violation['suggestion']}")
                    elif category == 'directories':
                        print(f"   Dir: {violation['dirname']}")
                        print(f"   Path: {violation['path']}")
                        print(f"   Issue: {violation['violation']}")
                        print(f"   Fix: {violation['suggestion']}")
                    elif category == 'case_collisions':
                        print(f"   Collision: {violation['collision']}")
                        print(f"   Path 1: {violation['path1']}")
                        print(f"   Path 2: {violation['path2']}")
                    else:
                        for key, value in violation.items():
                            print(f"   {key}: {value}")
                    print()
                
                if len(violations) > 5:
                    print(f"   ... and {len(violations) - 5} more violations")
        
        if total_violations == 0:
            print("✅ No case-consistency violations found!")
        else:
            print(f"\n⚠️  Found {total_violations} violations that need fixing")
            print("   Run the fix tool after reviewing these issues")

def main():
    print("🧪 VXOR Case-Consistency Scanner - TDD Approach")
    print("These violations show what needs to be fixed!\n")
    
    scanner = VxorCaseScanner()
    violations = scanner.scan_all_patterns()
    scanner.print_report()
    
    # Save results for fix tool
    report_path = "/Users/gecko365/vxor_naming_toolkit/case_violations_report.json"
    with open(report_path, 'w') as f:
        json.dump(violations, f, indent=2)
    
    print(f"\n📄 Full report saved to: {report_path}")
    return len(violations) > 0

if __name__ == "__main__":
    sys.exit(main())
