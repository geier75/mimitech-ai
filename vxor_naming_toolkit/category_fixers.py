#!/usr/bin/env python3
"""
VXOR Category-Specific Fixers
============================

Implements deterministic fixes for each violation category:
1. Directories - Rename directories with case inconsistencies
2. Files - Rename files following naming policy  
3. Imports - Update import statements in Python files
4. Symbols - Rename classes, functions, constants in code
5. Shims - Create backward compatibility aliases
6. Docs/CLI - Update documentation and CLI references
"""

import os
import re
import ast
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class FixResult:
    """Result of applying a fix"""
    success: bool
    violation_id: str
    old_value: str
    new_value: str
    files_affected: List[str]
    error: Optional[str] = None

class VxorCategoryFixers:
    """Category-specific fixers for VXOR naming violations"""
    
    def __init__(self, repo_root: Path, canonical_naming: Dict):
        self.repo_root = repo_root
        self.canonical = canonical_naming
        self.dry_run = True  # Default to dry-run mode
    
    def set_dry_run(self, dry_run: bool):
        """Enable/disable dry-run mode"""
        self.dry_run = dry_run
    
    def fix_directory_violation(self, violation: Dict) -> FixResult:
        """Fix directory naming violation"""
        violation_id = violation['id']
        old_path = Path(violation['file_path'])
        suggested_name = violation.get('suggested_fix', {}).get('new_name')
        
        if not suggested_name:
            return FixResult(
                success=False,
                violation_id=violation_id,
                old_value=str(old_path),
                new_value="",
                files_affected=[],
                error="No suggested fix provided"
            )
        
        # Calculate new path
        new_path = old_path.parent / suggested_name
        
        if self.dry_run:
            return FixResult(
                success=True,
                violation_id=violation_id,
                old_value=str(old_path),
                new_value=str(new_path),
                files_affected=[str(old_path)],
                error=None
            )
        
        try:
            # Rename directory
            if old_path.exists():
                shutil.move(str(old_path), str(new_path))
                
                # Update any imports that reference this directory
                affected_files = self._update_directory_imports(old_path.name, suggested_name)
                
                return FixResult(
                    success=True,
                    violation_id=violation_id,
                    old_value=str(old_path),
                    new_value=str(new_path),
                    files_affected=[str(new_path)] + affected_files
                )
            else:
                return FixResult(
                    success=False,
                    violation_id=violation_id,
                    old_value=str(old_path),
                    new_value=str(new_path),
                    files_affected=[],
                    error=f"Directory {old_path} does not exist"
                )
        except Exception as e:
            return FixResult(
                success=False,
                violation_id=violation_id,
                old_value=str(old_path),
                new_value=str(new_path),
                files_affected=[],
                error=str(e)
            )
    
    def fix_file_violation(self, violation: Dict) -> FixResult:
        """Fix file naming violation"""
        violation_id = violation['id']
        old_path = Path(violation['file_path'])
        suggested_name = violation.get('suggested_fix', {}).get('new_name')
        
        if not suggested_name:
            return FixResult(
                success=False,
                violation_id=violation_id,
                old_value=str(old_path),
                new_value="",
                files_affected=[],
                error="No suggested fix provided"
            )
        
        # Calculate new path  
        new_path = old_path.parent / suggested_name
        
        if self.dry_run:
            return FixResult(
                success=True,
                violation_id=violation_id,
                old_value=str(old_path),
                new_value=str(new_path),
                files_affected=[str(old_path)],
                error=None
            )
        
        try:
            if old_path.exists():
                # Rename file
                shutil.move(str(old_path), str(new_path))
                
                # Update imports that reference this file
                affected_files = self._update_file_imports(old_path, new_path)
                
                return FixResult(
                    success=True,
                    violation_id=violation_id,
                    old_value=str(old_path),
                    new_value=str(new_path),
                    files_affected=[str(new_path)] + affected_files
                )
            else:
                return FixResult(
                    success=False,
                    violation_id=violation_id,
                    old_value=str(old_path),
                    new_value=str(new_path),
                    files_affected=[],
                    error=f"File {old_path} does not exist"
                )
        except Exception as e:
            return FixResult(
                success=False,
                violation_id=violation_id,
                old_value=str(old_path),
                new_value=str(new_path),
                files_affected=[],
                error=str(e)
            )
    
    def fix_import_violation(self, violation: Dict) -> FixResult:
        """Fix import statement violation"""
        violation_id = violation['id']
        file_path = Path(violation['file_path'])
        line_number = violation.get('line_number', 1)
        old_import = violation.get('current_value', '')
        new_import = violation.get('suggested_fix', {}).get('new_value', '')
        
        if not new_import:
            return FixResult(
                success=False,
                violation_id=violation_id,
                old_value=old_import,
                new_value="",
                files_affected=[],
                error="No suggested fix provided"
            )
        
        if self.dry_run:
            return FixResult(
                success=True,
                violation_id=violation_id,
                old_value=old_import,
                new_value=new_import,
                files_affected=[str(file_path)],
                error=None
            )
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Update the specific line
            if 1 <= line_number <= len(lines):
                lines[line_number - 1] = lines[line_number - 1].replace(old_import, new_import)
                
                # Write back to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                return FixResult(
                    success=True,
                    violation_id=violation_id,
                    old_value=old_import,
                    new_value=new_import,
                    files_affected=[str(file_path)]
                )
            else:
                return FixResult(
                    success=False,
                    violation_id=violation_id,
                    old_value=old_import,
                    new_value=new_import,
                    files_affected=[],
                    error=f"Line number {line_number} out of range"
                )
                
        except Exception as e:
            return FixResult(
                success=False,
                violation_id=violation_id,
                old_value=old_import,
                new_value=new_import,
                files_affected=[],
                error=str(e)
            )
    
    def fix_symbol_violation(self, violation: Dict) -> FixResult:
        """Fix symbol naming violation (class, function, constant)"""
        violation_id = violation['id']
        file_path = Path(violation['file_path'])
        symbol_type = violation.get('symbol_type', 'unknown')
        old_name = violation.get('current_value', '')
        new_name = violation.get('suggested_fix', {}).get('new_value', '')
        
        if not new_name:
            return FixResult(
                success=False,
                violation_id=violation_id,
                old_value=old_name,
                new_value="",
                files_affected=[],
                error="No suggested fix provided"
            )
        
        if self.dry_run:
            return FixResult(
                success=True,
                violation_id=violation_id,
                old_value=old_name,
                new_value=new_name,
                files_affected=[str(file_path)],
                error=None
            )
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Apply symbol-specific renaming
            if symbol_type == 'class':
                updated_content = self._rename_class_symbol(content, old_name, new_name)
            elif symbol_type == 'function':
                updated_content = self._rename_function_symbol(content, old_name, new_name)
            elif symbol_type == 'constant':
                updated_content = self._rename_constant_symbol(content, old_name, new_name)
            else:
                # Generic regex-based rename
                updated_content = re.sub(
                    rf'\b{re.escape(old_name)}\b',
                    new_name,
                    content
                )
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            return FixResult(
                success=True,
                violation_id=violation_id,
                old_value=old_name,
                new_value=new_name,
                files_affected=[str(file_path)]
            )
            
        except Exception as e:
            return FixResult(
                success=False,
                violation_id=violation_id,
                old_value=old_name,
                new_value=new_name,
                files_affected=[],
                error=str(e)
            )
    
    def fix_shim_violation(self, violation: Dict) -> FixResult:
        """Create backward compatibility shim"""
        violation_id = violation['id']
        file_path = Path(violation['file_path'])
        old_name = violation.get('current_value', '')
        new_name = violation.get('suggested_fix', {}).get('new_value', '')
        
        if not new_name:
            return FixResult(
                success=False,
                violation_id=violation_id,
                old_value=old_name,
                new_value="",
                files_affected=[],
                error="No suggested fix provided"
            )
        
        if self.dry_run:
            return FixResult(
                success=True,
                violation_id=violation_id,
                old_value=old_name,
                new_value=f"{old_name} -> {new_name} (alias)",
                files_affected=[str(file_path)],
                error=None
            )
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add deprecation alias at end of file
            alias_code = f"""
# Backward compatibility alias (deprecated)
import warnings
{old_name} = {new_name}
warnings.warn(
    f"'{old_name}' is deprecated, use '{new_name}' instead",
    DeprecationWarning,
    stacklevel=2
)
"""
            
            updated_content = content + alias_code
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            return FixResult(
                success=True,
                violation_id=violation_id,
                old_value=old_name,
                new_value=f"{old_name} -> {new_name} (alias)",
                files_affected=[str(file_path)]
            )
            
        except Exception as e:
            return FixResult(
                success=False,
                violation_id=violation_id,
                old_value=old_name,
                new_value=new_name,
                files_affected=[],
                error=str(e)
            )
    
    def fix_docs_cli_violation(self, violation: Dict) -> FixResult:
        """Fix documentation/CLI reference violation"""
        violation_id = violation['id']
        file_path = Path(violation['file_path'])
        old_reference = violation.get('current_value', '')
        new_reference = violation.get('suggested_fix', {}).get('new_value', '')
        
        if not new_reference:
            return FixResult(
                success=False,
                violation_id=violation_id,
                old_value=old_reference,
                new_value="",
                files_affected=[],
                error="No suggested fix provided"
            )
        
        if self.dry_run:
            return FixResult(
                success=True,
                violation_id=violation_id,
                old_value=old_reference,
                new_value=new_reference,
                files_affected=[str(file_path)],
                error=None
            )
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace all occurrences
            updated_content = content.replace(old_reference, new_reference)
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            return FixResult(
                success=True,
                violation_id=violation_id,
                old_value=old_reference,
                new_value=new_reference,
                files_affected=[str(file_path)]
            )
            
        except Exception as e:
            return FixResult(
                success=False,
                violation_id=violation_id,
                old_value=old_reference,
                new_value=new_reference,
                files_affected=[],
                error=str(e)
            )
    
    def fix_violation(self, violation: Dict) -> FixResult:
        """Dispatch fix based on violation category"""
        category = violation.get('category', 'unknown')
        
        if category == 'directories':
            return self.fix_directory_violation(violation)
        elif category == 'files':
            return self.fix_file_violation(violation)
        elif category == 'imports':
            return self.fix_import_violation(violation)
        elif category == 'symbols':
            return self.fix_symbol_violation(violation)
        elif category == 'shims':
            return self.fix_shim_violation(violation)
        elif category == 'docs_cli':
            return self.fix_docs_cli_violation(violation)
        else:
            return FixResult(
                success=False,
                violation_id=violation.get('id', 'unknown'),
                old_value="",
                new_value="",
                files_affected=[],
                error=f"Unknown category: {category}"
            )
    
    # Helper methods
    
    def _update_directory_imports(self, old_dir_name: str, new_dir_name: str) -> List[str]:
        """Update imports that reference renamed directory"""
        affected_files = []
        
        # Find all Python files that might import from this directory
        for py_file in self.repo_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if file contains imports from the old directory
                old_import_pattern = rf'\bfrom\s+{re.escape(old_dir_name)}\b'
                if re.search(old_import_pattern, content):
                    if not self.dry_run:
                        # Update import statements
                        updated_content = re.sub(
                            old_import_pattern,
                            f'from {new_dir_name}',
                            content
                        )
                        with open(py_file, 'w', encoding='utf-8') as f:
                            f.write(updated_content)
                    
                    affected_files.append(str(py_file))
                    
            except Exception:
                continue  # Skip files we can't read
        
        return affected_files
    
    def _update_file_imports(self, old_path: Path, new_path: Path) -> List[str]:
        """Update imports that reference renamed file"""
        affected_files = []
        
        # Calculate module names
        old_module = old_path.stem
        new_module = new_path.stem
        
        if old_module == new_module:
            return []  # No module name change
        
        # Find all Python files that might import this module
        for py_file in self.repo_root.rglob("*.py"):
            if py_file == new_path:
                continue  # Skip the renamed file itself
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for imports of the old module
                import_patterns = [
                    rf'\bimport\s+{re.escape(old_module)}\b',
                    rf'\bfrom\s+{re.escape(old_module)}\s+import\b'
                ]
                
                updated = False
                for pattern in import_patterns:
                    if re.search(pattern, content):
                        if not self.dry_run:
                            # Update import statements
                            content = re.sub(
                                rf'\b{re.escape(old_module)}\b',
                                new_module,
                                content
                            )
                            updated = True
                
                if updated and not self.dry_run:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    affected_files.append(str(py_file))
                elif updated:  # dry_run mode
                    affected_files.append(str(py_file))
                    
            except Exception:
                continue  # Skip files we can't read
        
        return affected_files
    
    def _rename_class_symbol(self, content: str, old_name: str, new_name: str) -> str:
        """Rename class symbol with proper handling"""
        # Class definition
        content = re.sub(
            rf'^(\s*)class\s+{re.escape(old_name)}\b',
            rf'\1class {new_name}',
            content,
            flags=re.MULTILINE
        )
        
        # Class instantiation and references
        content = re.sub(
            rf'\b{re.escape(old_name)}\s*\(',
            f'{new_name}(',
            content
        )
        
        # Type annotations
        content = re.sub(
            rf'\b{re.escape(old_name)}\b(?=\s*[:\]])',
            new_name,
            content
        )
        
        return content
    
    def _rename_function_symbol(self, content: str, old_name: str, new_name: str) -> str:
        """Rename function symbol with proper handling"""
        # Function definition
        content = re.sub(
            rf'^(\s*)def\s+{re.escape(old_name)}\b',
            rf'\1def {new_name}',
            content,
            flags=re.MULTILINE
        )
        
        # Function calls
        content = re.sub(
            rf'\b{re.escape(old_name)}\s*\(',
            f'{new_name}(',
            content
        )
        
        return content
    
    def _rename_constant_symbol(self, content: str, old_name: str, new_name: str) -> str:
        """Rename constant symbol with proper handling"""
        # Constant assignment
        content = re.sub(
            rf'^(\s*){re.escape(old_name)}\s*=',
            rf'\1{new_name} =',
            content,
            flags=re.MULTILINE
        )
        
        # Constant references
        content = re.sub(
            rf'\b{re.escape(old_name)}\b',
            new_name,
            content
        )
        
        return content

def main():
    """CLI entry point for testing fixers"""
    import sys
    
    repo_root = Path("/Volumes/My Book/MISO_Ultimate 15.32.28")
    canonical_file = Path("/Users/gecko365/vxor_naming_toolkit/canonical_naming.json")
    
    with open(canonical_file, 'r') as f:
        canonical = json.load(f)
    
    fixers = VxorCategoryFixers(repo_root, canonical)
    
    # Test with a sample violation
    test_violation = {
        'id': 'test_violation_001',
        'category': 'files',
        'file_path': '/test/path/example.py',
        'current_value': 'vXor_test.py',
        'suggested_fix': {
            'new_name': 'vxor_test.py'
        }
    }
    
    # Dry run test
    fixers.set_dry_run(True)
    result = fixers.fix_violation(test_violation)
    print(f"Dry run result: {result}")

if __name__ == "__main__":
    main()
