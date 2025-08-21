#!/usr/bin/env python3
"""
VXOR-Naming CLI Toolkit - Case-Consistency & Import-Stability
============================================================

CLI Tool mit TDD-Approach:
- scan: Analysiert Case-Inkonsistenzen 
- fix: Behebt nur bestätigte Violations (nach Tests)
- verify: Testet Import-Stabilität und Compliance

Usage:
    vxor-naming scan [--repo-path PATH]
    vxor-naming fix [--dry-run] [--category CATEGORY] 
    vxor-naming verify [--imports] [--compliance]
"""

import argparse
import sys
import os
import json
import shutil
import re
from pathlib import Path
from typing import Dict, List, Set

from standalone_case_scanner import VxorCaseScanner

class VxorNamingCLI:
    """CLI Controller for VXOR Naming Toolkit"""
    
    def __init__(self):
        self.repo_root = Path("/Volumes/My Book/MISO_Ultimate 15.32.28")
        self.toolkit_dir = Path("/Users/gecko365/vxor_naming_toolkit")
        self.violations_file = self.toolkit_dir / "case_violations_report.json"
        self.canonical_naming_file = self.toolkit_dir / "canonical_naming.json"
        
        # Load naming policy
        with open(self.canonical_naming_file, 'r') as f:
            self.naming_policy = json.load(f)
    
    def cmd_scan(self, args) -> int:
        """Scan repository for case-consistency violations"""
        print("🔍 VXOR-NAMING SCAN - Analyzing Case-Consistency")
        print("=" * 60)
        
        scanner = VxorCaseScanner()
        violations = scanner.scan_all_patterns()
        scanner.print_report()
        
        # Generate summary
        total_violations = sum(len(v) for v in violations.values())
        print(f"\n📊 SCAN SUMMARY:")
        print(f"   Total Violations: {total_violations}")
        
        if total_violations > 0:
            print(f"   ⚠️  Run 'vxor-naming fix --dry-run' to see proposed fixes")
            return 1
        else:
            print(f"   ✅ Repository is case-consistency compliant!")
            return 0
    
    def cmd_fix(self, args) -> int:
        """Fix case-consistency violations"""
        print("🔧 VXOR-NAMING FIX - Applying Case-Consistency Fixes")
        print("=" * 60)
        
        if not self.violations_file.exists():
            print("❌ No violations report found. Run 'vxor-naming scan' first.")
            return 1
        
        with open(self.violations_file, 'r') as f:
            violations = json.load(f)
        
        total_violations = sum(len(v) for v in violations.values())
        
        if total_violations == 0:
            print("✅ No violations found to fix!")
            return 0
        
        if args.dry_run:
            return self._dry_run_fixes(violations, args.category)
        else:
            return self._apply_fixes(violations, args.category)
    
    def cmd_verify(self, args) -> int:
        """Verify import-stability and compliance"""
        print("✅ VXOR-NAMING VERIFY - Testing Import-Stability")
        print("=" * 60)
        
        success = True
        
        if args.imports:
            success &= self._verify_imports()
        
        if args.compliance:
            success &= self._verify_compliance()
        
        if not args.imports and not args.compliance:
            # Default: verify both
            success &= self._verify_imports()
            success &= self._verify_compliance()
        
        return 0 if success else 1
    
    def _dry_run_fixes(self, violations: Dict, category_filter: str = None) -> int:
        """Show proposed fixes without applying them"""
        print("🔍 DRY-RUN MODE - Showing proposed fixes:\n")
        
        fixes_count = 0
        
        for category, violation_list in violations.items():
            if category_filter and category != category_filter:
                continue
                
            if not violation_list:
                continue
                
            print(f"📂 {category.upper()} FIXES ({len(violation_list)}):")
            print("-" * 40)
            
            for i, violation in enumerate(violation_list[:10]):  # Show max 10
                fixes_count += 1
                
                if category == 'files':
                    current_path = violation['path']
                    suggested_name = violation['suggestion']
                    new_path = str(Path(current_path).parent / suggested_name)
                    print(f"   {i+1}. RENAME FILE:")
                    print(f"      FROM: {current_path}")
                    print(f"      TO:   {new_path}")
                
                elif category == 'directories':
                    current_path = violation['path']
                    suggested_name = violation['suggestion']
                    new_path = str(Path(current_path).parent / suggested_name)
                    print(f"   {i+1}. RENAME DIRECTORY:")
                    print(f"      FROM: {current_path}")
                    print(f"      TO:   {new_path}")
                
                elif category == 'classes':
                    print(f"   {i+1}. RENAME CLASS:")
                    print(f"      FILE: {violation['file']}")
                    print(f"      FROM: class {violation['class']}")
                    print(f"      TO:   class {violation['suggestion']}")
                
                elif category == 'imports':
                    print(f"   {i+1}. FIX IMPORT:")
                    print(f"      FILE: {violation['file']}")
                    if 'import' in violation:
                        print(f"      FROM: import {violation['import']}")
                        print(f"      TO:   import {violation['suggestion']}")
                    else:
                        print(f"      FROM: from {violation['module']}")
                        print(f"      TO:   from {violation['suggestion']}")
                
                print()
            
            if len(violation_list) > 10:
                print(f"   ... and {len(violation_list) - 10} more fixes")
            print()
        
        print(f"📊 SUMMARY: {fixes_count} fixes would be applied")
        print("   Run without --dry-run to apply these fixes")
        
        return 0
    
    def _apply_fixes(self, violations: Dict, category_filter: str = None) -> int:
        """Apply actual fixes to violations"""
        print("⚡ APPLYING FIXES - This will modify files!")
        
        if not self._confirm_fixes():
            print("❌ Fix operation cancelled by user")
            return 1
        
        applied_fixes = 0
        failed_fixes = 0
        
        # Apply fixes in safe order: imports -> symbols -> files -> directories
        fix_order = ['imports', 'classes', 'functions', 'constants', 'files', 'directories']
        
        for category in fix_order:
            if category_filter and category != category_filter:
                continue
                
            violation_list = violations.get(category, [])
            if not violation_list:
                continue
            
            print(f"\n🔧 Fixing {category.upper()} violations...")
            
            for violation in violation_list:
                try:
                    if category == 'files':
                        success = self._fix_file_rename(violation)
                    elif category == 'directories':
                        success = self._fix_directory_rename(violation)
                    elif category == 'imports':
                        success = self._fix_import_statement(violation)
                    elif category in ['classes', 'functions', 'constants']:
                        success = self._fix_symbol_rename(violation, category)
                    else:
                        success = False
                    
                    if success:
                        applied_fixes += 1
                        print(f"   ✅ Fixed: {self._get_violation_summary(violation)}")
                    else:
                        failed_fixes += 1
                        print(f"   ❌ Failed: {self._get_violation_summary(violation)}")
                        
                except Exception as e:
                    failed_fixes += 1
                    print(f"   ❌ Error: {self._get_violation_summary(violation)} - {e}")
        
        print(f"\n📊 FIX SUMMARY:")
        print(f"   Applied: {applied_fixes}")
        print(f"   Failed:  {failed_fixes}")
        
        if applied_fixes > 0:
            print(f"\n✅ Run 'vxor-naming verify' to test the fixes")
        
        return 0 if failed_fixes == 0 else 1
    
    def _verify_imports(self) -> bool:
        """Verify all VXOR imports work correctly"""
        print("🔍 Testing VXOR/VX import stability...")
        
        sys.path.insert(0, str(self.repo_root))
        
        critical_modules = [
            'vxor.core.vx_core',
            'agents.vx_memex',
            'agents.vx_planner',
            'agents.vx_vision'
        ]
        
        import_failures = []
        
        for module_name in critical_modules:
            try:
                __import__(module_name)
                print(f"   ✅ {module_name}")
            except ImportError as e:
                import_failures.append(f"{module_name}: {e}")
                print(f"   ❌ {module_name}: {e}")
            except Exception as e:
                import_failures.append(f"{module_name}: {e}")
                print(f"   ❌ {module_name}: {e}")
        
        if import_failures:
            print(f"\n❌ {len(import_failures)} import failures detected")
            return False
        else:
            print(f"\n✅ All critical imports working")
            return True
    
    def _verify_compliance(self) -> bool:
        """Verify naming policy compliance"""
        print("📋 Checking naming policy compliance...")
        
        # Re-run scanner to check current state
        scanner = VxorCaseScanner()
        violations = scanner.scan_all_patterns()
        
        total_violations = sum(len(v) for v in violations.values())
        
        if total_violations == 0:
            print("   ✅ Full compliance with naming policy")
            return True
        else:
            print(f"   ❌ {total_violations} violations remaining")
            return False
    
    def _confirm_fixes(self) -> bool:
        """Ask user confirmation for applying fixes"""
        response = input("This will modify files. Continue? [y/N]: ")
        return response.lower() in ['y', 'yes']
    
    def _fix_file_rename(self, violation: Dict) -> bool:
        """Rename a file according to naming policy"""
        current_path = Path(violation['path'])
        suggested_name = violation['suggestion']
        new_path = current_path.parent / suggested_name
        
        if new_path.exists():
            return False  # Avoid overwriting
            
        shutil.move(str(current_path), str(new_path))
        return True
    
    def _fix_directory_rename(self, violation: Dict) -> bool:
        """Rename a directory according to naming policy"""
        current_path = Path(violation['path'])
        suggested_name = violation['suggestion']
        new_path = current_path.parent / suggested_name
        
        if new_path.exists():
            return False  # Avoid overwriting
            
        shutil.move(str(current_path), str(new_path))
        return True
    
    def _fix_import_statement(self, violation: Dict) -> bool:
        """Fix import statement in a file"""
        # This would require LibCST for proper AST manipulation
        # For now, return False to indicate not implemented
        return False
    
    def _fix_symbol_rename(self, violation: Dict, category: str) -> bool:
        """Fix symbol (class/function/constant) naming"""
        # This would require LibCST for proper AST manipulation  
        # For now, return False to indicate not implemented
        return False
    
    def _get_violation_summary(self, violation: Dict) -> str:
        """Get a short summary of a violation for logging"""
        if 'path' in violation:
            return violation['path']
        elif 'file' in violation:
            return violation['file']
        else:
            return str(violation)

def create_cli_parser():
    """Create argument parser for CLI"""
    parser = argparse.ArgumentParser(
        prog='vxor-naming',
        description='VXOR Case-Consistency & Import-Stability Toolkit'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan for case-consistency violations')
    scan_parser.add_argument('--repo-path', help='Repository path to scan')
    
    # Fix command  
    fix_parser = subparsers.add_parser('fix', help='Fix case-consistency violations')
    fix_parser.add_argument('--dry-run', action='store_true', help='Show fixes without applying')
    fix_parser.add_argument('--category', help='Fix only specific category')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify import-stability and compliance')
    verify_parser.add_argument('--imports', action='store_true', help='Test import functionality')  
    verify_parser.add_argument('--compliance', action='store_true', help='Check naming compliance')
    
    return parser

def main():
    parser = create_cli_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    cli = VxorNamingCLI()
    
    if args.command == 'scan':
        return cli.cmd_scan(args)
    elif args.command == 'fix':
        return cli.cmd_fix(args)
    elif args.command == 'verify':
        return cli.cmd_verify(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
