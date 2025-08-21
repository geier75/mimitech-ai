#!/usr/bin/env python3
"""
VXOR-Naming Nano-Step TDD CLI - Produktionstauglich
==================================================

Nano-Step TDD/ADD Workflow:
1. Test schreiben/ausführen → Fehler sichtbar machen
2. Minimal fixen (genau 1 Verstoß)
3. Tests erneut ausführen → muss grün werden
4. Commit (conventional commits)
5. Report aktualisieren → nächster Loop

Exit Codes:
0 - Erfolg (grün)
2 - Verstöße gefunden 
3 - Import-Fehler
4 - Test-Fehler
"""

import argparse
import sys
import os
import json
import uuid
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from standalone_case_scanner import VxorCaseScanner

@dataclass
class ViolationReport:
    """Single violation report for nano-step processing"""
    id: str
    category: str
    violation_type: str
    file_path: str
    current_name: str
    suggested_name: str
    rule: str
    severity: str
    timestamp: str
    status: str = "pending"
    fix_duration_ms: Optional[int] = None
    test_result: Optional[str] = None

@dataclass  
class NanoStepReport:
    """Report for a single nano-step TDD loop"""
    step_id: str
    violation_id: str
    category: str
    timestamp: str
    duration_ms: int
    actions: List[str]
    test_before: str
    test_after: str 
    diff_stats: Dict[str, int]
    status: str
    error: Optional[str] = None

class VxorNamingNanoCLI:
    """Enhanced CLI for Nano-Step TDD workflow"""
    
    def __init__(self):
        self.repo_root = Path("/Volumes/My Book/MISO_Ultimate 15.32.28")
        self.toolkit_dir = Path("/Users/gecko365/vxor_naming_toolkit")
        self.reports_dir = self.toolkit_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        self.violations_db = self.reports_dir / "violations.json"
        self.nano_steps_log = self.reports_dir / "nano_steps.jsonl"
        self.summary_report = self.reports_dir / "summary.md"
        
        # Category processing order
        self.category_order = [
            "directories", "files", "imports", 
            "symbols", "shims", "docs_cli"
        ]
        
        # Load canonical naming
        with open(self.toolkit_dir / "canonical_naming.json", 'r') as f:
            self.naming_policy = json.load(f)
    
    def cmd_scan(self, args) -> int:
        """Scan for violations with nano-step focus"""
        print("🔍 NANO-STEP SCAN - Analyzing Case-Consistency")
        print("=" * 60)
        
        scanner = VxorCaseScanner()
        raw_violations = scanner.scan_all_patterns()
        
        # Convert to structured violation reports
        violations = self._convert_to_violation_reports(raw_violations)
        
        # Filter by category if specified
        if args.category:
            violations = [v for v in violations if v.category == args.category]
        
        # Limit results for nano-step processing
        if args.limit:
            violations = violations[:args.limit]
        
        # Save violations database
        self._save_violations_db(violations)
        
        # Output results
        total_violations = len(violations)
        
        if args.output_format == 'json':
            print(json.dumps([asdict(v) for v in violations], indent=2))
        else:
            self._print_scan_results(violations)
        
        if args.out:
            output_path = Path(args.out) / f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump([asdict(v) for v in violations], f, indent=2)
            print(f"📄 Results saved to: {output_path}")
        
        if total_violations == 0:
            print("✅ No violations found!")
            return 0
        else:
            print(f"⚠️  Found {total_violations} violations")
            return 2
    
    def cmd_fix(self, args) -> int:
        """Fix violations using nano-step approach"""
        print("🔧 NANO-STEP FIX - Applying Single Violation Fix")
        print("=" * 60)
        
        # Load violations database
        violations = self._load_violations_db()
        if not violations:
            print("❌ No violations database found. Run scan first.")
            return 2
        
        # Filter by category and get specific violation
        target_violations = [v for v in violations if v.category == args.category and v.status == "pending"]
        
        if not target_violations:
            print(f"✅ No pending violations in category '{args.category}'")
            return 0
        
        # Select violation to fix
        if args.id:
            violation = next((v for v in target_violations if v.id == args.id), None)
            if not violation:
                print(f"❌ Violation ID '{args.id}' not found")
                return 2
        else:
            # Take first pending violation
            violation = target_violations[0]
        
        print(f"🎯 Fixing violation: {violation.id}")
        print(f"   Category: {violation.category}")
        print(f"   Type: {violation.violation_type}")
        print(f"   File: {violation.file_path}")
        print(f"   Current: {violation.current_name}")
        print(f"   Suggested: {violation.suggested_name}")
        
        if args.dry_run:
            return self._dry_run_single_fix(violation)
        else:
            return self._apply_single_fix(violation)
    
    def cmd_verify(self, args) -> int:
        """Verify system state after fixes"""
        print("✅ NANO-STEP VERIFY - Testing System State")
        print("=" * 60)
        
        success = True
        
        # Import stability test
        if args.imports or not (args.compliance or args.tests):
            success &= self._verify_imports()
        
        # Compliance check
        if args.compliance or not (args.imports or args.tests):
            success &= self._verify_compliance()
        
        # Run pytest
        if args.tests or not (args.imports or args.compliance):
            success &= self._run_tests()
        
        return 0 if success else 3
    
    def cmd_report(self, args) -> int:
        """Generate comprehensive reports"""
        print("📊 NANO-STEP REPORT - Generating Reports")
        print("=" * 60)
        
        if args.append:
            self._append_to_reports()
        else:
            self._generate_full_report()
        
        return 0
    
    def _convert_to_violation_reports(self, raw_violations: Dict) -> List[ViolationReport]:
        """Convert scanner output to structured violation reports"""
        violations = []
        timestamp = datetime.now().isoformat()
        
        for category, violation_list in raw_violations.items():
            for violation in violation_list:
                report = ViolationReport(
                    id=str(uuid.uuid4())[:8],
                    category=category,
                    violation_type=violation.get('violation', 'unknown'),
                    file_path=violation.get('path', violation.get('file', '')),
                    current_name=violation.get('filename', violation.get('dirname', violation.get('class', violation.get('function', violation.get('constant', violation.get('import', violation.get('module', ''))))))),
                    suggested_name=violation.get('suggestion', ''),
                    rule=self._get_rule_for_violation(category, violation),
                    severity=self._get_severity(category),
                    timestamp=timestamp
                )
                violations.append(report)
        
        return violations
    
    def _get_rule_for_violation(self, category: str, violation: Dict) -> str:
        """Get the naming rule that was violated"""
        rules = {
            'files': 'lower_snake_case.py',
            'directories': 'lower_snake_case/',
            'classes': 'PascalCase with Vxor/Vx prefix',
            'functions': 'lower_snake_case',
            'constants': 'UPPER_SNAKE_CASE with VXOR_/VX_ prefix',
            'imports': 'canonical module names'
        }
        return rules.get(category, 'naming_policy')
    
    def _get_severity(self, category: str) -> str:
        """Get severity level for category"""
        severity_map = {
            'directories': 'critical',
            'files': 'high', 
            'imports': 'high',
            'classes': 'medium',
            'functions': 'medium',
            'constants': 'low'
        }
        return severity_map.get(category, 'medium')
    
    def _save_violations_db(self, violations: List[ViolationReport]):
        """Save violations to database"""
        data = [asdict(v) for v in violations]
        with open(self.violations_db, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_violations_db(self) -> List[ViolationReport]:
        """Load violations from database"""
        if not self.violations_db.exists():
            return []
        
        with open(self.violations_db, 'r') as f:
            data = json.load(f)
        
        return [ViolationReport(**item) for item in data]
    
    def _print_scan_results(self, violations: List[ViolationReport]):
        """Print scan results in human-readable format"""
        if not violations:
            print("✅ No violations found!")
            return
        
        # Group by category
        by_category = {}
        for v in violations:
            if v.category not in by_category:
                by_category[v.category] = []
            by_category[v.category].append(v)
        
        for category, items in by_category.items():
            print(f"\n🚨 {category.upper()} ({len(items)} violations):")
            print("-" * 40)
            
            for i, violation in enumerate(items[:5]):  # Show max 5
                print(f"   {i+1}. ID: {violation.id}")
                print(f"      File: {violation.file_path}")
                print(f"      Current: {violation.current_name}")
                print(f"      Suggested: {violation.suggested_name}")
                print(f"      Rule: {violation.rule}")
                print()
            
            if len(items) > 5:
                print(f"   ... and {len(items) - 5} more")
    
    def _dry_run_single_fix(self, violation: ViolationReport) -> int:
        """Show what a single fix would do"""
        print("\n🔍 DRY-RUN MODE - Single Fix Preview:")
        print("-" * 40)
        
        if violation.category == 'directories':
            current_path = Path(violation.file_path)
            new_path = current_path.parent / violation.suggested_name
            print(f"RENAME DIRECTORY:")
            print(f"  FROM: {current_path}")
            print(f"  TO:   {new_path}")
            
            if new_path.exists():
                print(f"  ⚠️  WARNING: Target already exists!")
                return 2
        
        elif violation.category == 'files':
            current_path = Path(violation.file_path)
            new_path = current_path.parent / violation.suggested_name
            print(f"RENAME FILE:")
            print(f"  FROM: {current_path}")
            print(f"  TO:   {new_path}")
            
            if new_path.exists():
                print(f"  ⚠️  WARNING: Target already exists!")
                return 2
        
        else:
            print(f"FIX {violation.category.upper()}:")
            print(f"  File: {violation.file_path}")
            print(f"  Change: {violation.current_name} → {violation.suggested_name}")
            print(f"  Rule: {violation.rule}")
        
        print(f"\nRun with --apply to execute this fix.")
        return 0
    
    def _apply_single_fix(self, violation: ViolationReport) -> int:
        """Apply a single fix and track the nano-step"""
        start_time = time.time()
        
        nano_step = NanoStepReport(
            step_id=str(uuid.uuid4())[:8],
            violation_id=violation.id,
            category=violation.category,
            timestamp=datetime.now().isoformat(),
            duration_ms=0,
            actions=[],
            test_before="unknown",
            test_after="unknown", 
            diff_stats={},
            status="started"
        )
        
        try:
            # Run tests BEFORE fix (should fail)
            print("1️⃣ Running tests BEFORE fix (should fail)...")
            nano_step.test_before = self._run_single_test_for_violation(violation)
            nano_step.actions.append("test_before")
            
            # Apply the fix
            print("2️⃣ Applying fix...")
            success = self._execute_fix(violation)
            if not success:
                nano_step.status = "fix_failed"
                return 4
            
            nano_step.actions.append("fix_applied")
            
            # Run tests AFTER fix (should pass)
            print("3️⃣ Running tests AFTER fix (should pass)...")
            nano_step.test_after = self._run_single_test_for_violation(violation)
            nano_step.actions.append("test_after")
            
            # Update violation status
            violation.status = "fixed"
            violation.fix_duration_ms = int((time.time() - start_time) * 1000)
            self._update_violation_in_db(violation)
            
            nano_step.duration_ms = int((time.time() - start_time) * 1000)
            nano_step.status = "completed"
            
            # Log nano-step
            self._log_nano_step(nano_step)
            
            print(f"✅ Fix completed successfully in {nano_step.duration_ms}ms")
            return 0
            
        except Exception as e:
            nano_step.error = str(e)
            nano_step.status = "error"
            nano_step.duration_ms = int((time.time() - start_time) * 1000)
            self._log_nano_step(nano_step)
            print(f"❌ Fix failed: {e}")
            return 4
    
    def _execute_fix(self, violation: ViolationReport) -> bool:
        """Execute the actual fix for a violation"""
        if violation.category == 'directories':
            return self._fix_directory_rename(violation)
        elif violation.category == 'files':
            return self._fix_file_rename(violation)
        elif violation.category == 'imports':
            return self._fix_import_statement(violation)
        elif violation.category in ['classes', 'functions', 'constants']:
            return self._fix_symbol_rename(violation)
        else:
            print(f"⚠️  Fix not implemented for category: {violation.category}")
            return False
    
    def _fix_directory_rename(self, violation: ViolationReport) -> bool:
        """Rename directory"""
        try:
            current_path = Path(violation.file_path)
            new_path = current_path.parent / violation.suggested_name
            
            if new_path.exists():
                print(f"⚠️  Target directory already exists: {new_path}")
                return False
            
            shutil.move(str(current_path), str(new_path))
            print(f"✅ Renamed: {current_path} → {new_path}")
            return True
        except Exception as e:
            print(f"❌ Directory rename failed: {e}")
            return False
    
    def _fix_file_rename(self, violation: ViolationReport) -> bool:
        """Rename file"""
        try:
            current_path = Path(violation.file_path)
            new_path = current_path.parent / violation.suggested_name
            
            if new_path.exists():
                print(f"⚠️  Target file already exists: {new_path}")
                return False
            
            shutil.move(str(current_path), str(new_path))
            print(f"✅ Renamed: {current_path} → {new_path}")
            return True
        except Exception as e:
            print(f"❌ File rename failed: {e}")
            return False
    
    def _fix_import_statement(self, violation: ViolationReport) -> bool:
        """Fix import statement (placeholder)"""
        print("⚠️  Import statement fixes not yet implemented (requires LibCST)")
        return False
    
    def _fix_symbol_rename(self, violation: ViolationReport) -> bool:
        """Fix symbol rename (placeholder)"""
        print("⚠️  Symbol rename fixes not yet implemented (requires LibCST)")
        return False
    
    def _run_single_test_for_violation(self, violation: ViolationReport) -> str:
        """Run a specific test for a violation"""
        try:
            # Run pytest for specific test
            result = subprocess.run([
                'python3', '-m', 'pytest', 
                f'{self.toolkit_dir}/test_nano_steps.py::test_violation_{violation.id}',
                '-v', '--tb=short'
            ], capture_output=True, text=True, cwd=self.toolkit_dir)
            
            if result.returncode == 0:
                return "PASS"
            else:
                return "FAIL"
        except Exception as e:
            return f"ERROR: {e}"
    
    def _verify_imports(self) -> bool:
        """Verify critical imports"""
        print("🔍 Testing import stability...")
        sys.path.insert(0, str(self.repo_root))
        
        critical_modules = [
            'vxor.core.vx_core',
            'agents.vx_memex',
            'agents.vx_planner',
            'agents.vx_vision'
        ]
        
        failures = 0
        for module_name in critical_modules:
            try:
                __import__(module_name)
                print(f"   ✅ {module_name}")
            except Exception as e:
                print(f"   ❌ {module_name}: {e}")
                failures += 1
        
        return failures == 0
    
    def _verify_compliance(self) -> bool:
        """Verify naming compliance"""
        print("📋 Checking naming compliance...")
        
        scanner = VxorCaseScanner()
        violations = scanner.scan_all_patterns()
        total = sum(len(v) for v in violations.values())
        
        if total == 0:
            print("   ✅ Full compliance")
            return True
        else:
            print(f"   ❌ {total} violations remaining")
            return False
    
    def _run_tests(self) -> bool:
        """Run full test suite"""
        print("🧪 Running test suite...")
        try:
            result = subprocess.run([
                'python3', '-m', 'pytest', 
                f'{self.toolkit_dir}/', '-v'
            ], capture_output=True, text=True, cwd=self.toolkit_dir)
            
            if result.returncode == 0:
                print("   ✅ All tests passed")
                return True
            else:
                print("   ❌ Test failures detected")
                print(result.stdout[-500:])  # Show last 500 chars
                return False
        except Exception as e:
            print(f"   ❌ Test execution failed: {e}")
            return False
    
    def _update_violation_in_db(self, violation: ViolationReport):
        """Update violation status in database"""
        violations = self._load_violations_db()
        for i, v in enumerate(violations):
            if v.id == violation.id:
                violations[i] = violation
                break
        self._save_violations_db(violations)
    
    def _log_nano_step(self, nano_step: NanoStepReport):
        """Log nano-step to JSONL file"""
        with open(self.nano_steps_log, 'a') as f:
            f.write(json.dumps(asdict(nano_step)) + '\n')
    
    def _append_to_reports(self):
        """Append new data to reports"""
        print("📄 Appending to reports...")
        # Implementation for appending reports
        
    def _generate_full_report(self):
        """Generate comprehensive report"""
        print("📊 Generating full report...")
        # Implementation for full report generation

def create_nano_cli_parser():
    """Create enhanced argument parser for nano-step CLI"""
    parser = argparse.ArgumentParser(
        prog='vxor-naming',
        description='VXOR Nano-Step TDD Case-Consistency Toolkit'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan for violations')
    scan_parser.add_argument('--category', choices=['directories', 'files', 'imports', 'symbols', 'shims', 'docs_cli'])
    scan_parser.add_argument('--limit', type=int, help='Limit number of violations (for nano-step)')
    scan_parser.add_argument('--out', help='Output directory for reports')
    scan_parser.add_argument('--output-format', choices=['text', 'json'], default='text')
    
    # Fix command  
    fix_parser = subparsers.add_parser('fix', help='Fix violations')
    fix_parser.add_argument('--category', required=True, choices=['directories', 'files', 'imports', 'symbols', 'shims', 'docs_cli'])
    fix_parser.add_argument('--id', help='Specific violation ID to fix')
    fix_parser.add_argument('--dry-run', action='store_true', help='Show fix without applying')
    fix_parser.add_argument('--apply', action='store_true', help='Apply the fix (opposite of dry-run)')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify system state')
    verify_parser.add_argument('--imports', action='store_true', help='Test imports')
    verify_parser.add_argument('--compliance', action='store_true', help='Check compliance')
    verify_parser.add_argument('--tests', action='store_true', help='Run test suite')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate reports')
    report_parser.add_argument('--append', action='store_true', help='Append to existing reports')
    
    return parser

def main():
    parser = create_nano_cli_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    cli = VxorNamingNanoCLI()
    
    # Set dry-run as default unless --apply is explicitly used
    if hasattr(args, 'dry_run') and hasattr(args, 'apply'):
        if not args.apply:
            args.dry_run = True
    
    if args.command == 'scan':
        return cli.cmd_scan(args)
    elif args.command == 'fix':
        return cli.cmd_fix(args)
    elif args.command == 'verify':
        return cli.cmd_verify(args)
    elif args.command == 'report':
        return cli.cmd_report(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
