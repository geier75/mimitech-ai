#!/usr/bin/env python3
"""
Benchmark Sanity Gates Validator
Detects duplicate dataset routing and suspicious execution times
"""

import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Set
import sys

class BenchmarkSanityValidator:
    """
    Validates benchmark runs for sanity issues:
    - Duplicate dataset_id + split combinations
    - Suspiciously fast execution times for complex benchmarks
    """
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.min_exec_times = {
            # Benchmarks that should not be "instant" (> 0.1s)
            "MBPP": 0.5,           # Code generation takes time
            "SWE-Bench": 1.0,      # Software engineering tasks
            "CodexGLUE": 0.3,      # Code understanding
            "HumanEval": 0.1,      # Basic code generation
            "MMLU": 0.2,           # Large knowledge base
            "HellaSwag": 0.1,      # Commonsense reasoning
            "WinoGrande": 0.2,     # Commonsense reasoning
            "PIQA": 0.15,          # Physical reasoning
            "GSM8K": 0.3,          # Math reasoning
            "ARC": 0.2,            # AI2 reasoning
        }
        
    def validate_all_sanity_checks(self) -> Tuple[bool, Dict[str, any]]:
        """
        Run all sanity validations on benchmark reports
        
        Returns:
            Tuple of (success, issues_found)
        """
        
        print("🔍 Running Benchmark Sanity Checks...")
        print("=" * 50)
        
        issues = {}
        
        # Find all benchmark reports
        report_pattern = str(self.project_root / "tests/reports/manifests/*.json")
        report_files = glob.glob(report_pattern)
        
        if not report_files:
            print("⚠️  No benchmark reports found")
            return True, {"warning": "No reports to validate"}
        
        # Load all reports
        reports = []
        for report_file in report_files:
            try:
                with open(report_file, 'r') as f:
                    report = json.load(f)
                    report['_file_path'] = report_file
                    reports.append(report)
            except Exception as e:
                issues[f"load_error_{Path(report_file).name}"] = str(e)
        
        print(f"📊 Analyzing {len(reports)} benchmark reports...")
        
        # Check for duplicate dataset routing
        duplicate_issues = self._check_duplicate_datasets(reports)
        if duplicate_issues:
            issues["duplicate_datasets"] = duplicate_issues
            
        # Check for suspicious execution times  
        timing_issues = self._check_execution_times(reports)
        if timing_issues:
            issues["suspicious_timing"] = timing_issues
            
        # Check for missing required fields
        field_issues = self._check_required_fields(reports)
        if field_issues:
            issues["missing_fields"] = field_issues
            
        # Determine overall success
        success = len(issues) == 0 or (len(issues) == 1 and "warning" in issues)
        
        return success, issues
    
    def _check_duplicate_datasets(self, reports: List[Dict]) -> List[Dict]:
        """Check for duplicate dataset_id + split combinations"""
        
        dataset_usage = {}  # (dataset_id, split) -> [benchmark_names]
        
        for report in reports:
            name = report.get('name', 'unknown')
            dataset_id = report.get('dataset_id')
            split = report.get('split')
            
            if dataset_id and split:
                key = (dataset_id, split)
                if key not in dataset_usage:
                    dataset_usage[key] = []
                dataset_usage[key].append(name)
        
        # Find conflicts (same dataset+split used by multiple benchmarks)
        conflicts = []
        for (dataset_id, split), benchmark_names in dataset_usage.items():
            if len(benchmark_names) > 1:
                conflicts.append({
                    "dataset_id": dataset_id,
                    "split": split,
                    "conflicting_benchmarks": benchmark_names,
                    "severity": "critical"
                })
        
        if conflicts:
            print(f"❌ Found {len(conflicts)} dataset routing conflicts:")
            for conflict in conflicts:
                benchmarks = ", ".join(conflict["conflicting_benchmarks"])
                print(f"  - {conflict['dataset_id']}/{conflict['split']} used by: {benchmarks}")
        
        return conflicts
    
    def _check_execution_times(self, reports: List[Dict]) -> List[Dict]:
        """Check for suspiciously fast execution times"""
        
        timing_issues = []
        
        for report in reports:
            name = report.get('name', 'unknown')
            exec_time = report.get('exec_time_s', 0.0)
            
            # Check if this benchmark has a minimum expected time
            if name in self.min_exec_times:
                min_expected = self.min_exec_times[name]
                
                if exec_time < min_expected:
                    timing_issues.append({
                        "benchmark": name,
                        "actual_time": exec_time,
                        "expected_minimum": min_expected,
                        "severity": "warning" if exec_time > 0.05 else "critical",
                        "file_path": report.get('_file_path', 'unknown')
                    })
        
        if timing_issues:
            print(f"⚠️  Found {len(timing_issues)} suspicious timing issues:")
            for issue in timing_issues:
                severity = issue["severity"].upper()
                print(f"  - {severity}: {issue['benchmark']} took {issue['actual_time']:.3f}s "
                      f"(expected >{issue['expected_minimum']:.1f}s)")
        
        return timing_issues
    
    def _check_required_fields(self, reports: List[Dict]) -> List[Dict]:
        """Check for missing required fields in reports"""
        
        required_fields = ['name', 'accuracy', 'samples_processed', 'duration_s', 'status']
        field_issues = []
        
        for report in reports:
            name = report.get('name', 'unknown')
            missing_fields = []
            
            for field in required_fields:
                if field not in report or report[field] is None:
                    missing_fields.append(field)
            
            if missing_fields:
                field_issues.append({
                    "benchmark": name,
                    "missing_fields": missing_fields,
                    "file_path": report.get('_file_path', 'unknown'),
                    "severity": "critical"
                })
        
        if field_issues:
            print(f"❌ Found {len(field_issues)} field validation issues:")
            for issue in field_issues:
                fields = ", ".join(issue["missing_fields"])
                print(f"  - {issue['benchmark']}: missing {fields}")
        
        return field_issues
    
    def generate_sanity_report(self, issues: Dict) -> str:
        """Generate a detailed sanity check report"""
        
        report_lines = [
            "# Benchmark Sanity Check Report",
            f"**Generated**: Sanity Gates Validator",
            ""
        ]
        
        if not issues or (len(issues) == 1 and "warning" in issues):
            report_lines.append("✅ **ALL SANITY CHECKS PASSED**")
            return "\n".join(report_lines)
        
        report_lines.append("❌ **SANITY CHECK FAILURES DETECTED**")
        report_lines.append("")
        
        # Duplicate dataset issues
        if "duplicate_datasets" in issues:
            conflicts = issues["duplicate_datasets"]
            report_lines.extend([
                f"## 🚨 Dataset Routing Conflicts ({len(conflicts)} found)",
                ""
            ])
            
            for conflict in conflicts:
                benchmarks = ", ".join(conflict["conflicting_benchmarks"])
                report_lines.extend([
                    f"**Conflict**: `{conflict['dataset_id']}/{conflict['split']}`",
                    f"- Used by: {benchmarks}",
                    f"- **Action Required**: Fix adapter routing to use unique datasets",
                    ""
                ])
        
        # Timing issues
        if "suspicious_timing" in issues:
            timing_issues = issues["suspicious_timing"]
            critical_timing = [t for t in timing_issues if t["severity"] == "critical"]
            warning_timing = [t for t in timing_issues if t["severity"] == "warning"]
            
            if critical_timing:
                report_lines.extend([
                    f"## ❌ Critical Timing Issues ({len(critical_timing)} found)",
                    ""
                ])
                
                for issue in critical_timing:
                    report_lines.extend([
                        f"**{issue['benchmark']}**: {issue['actual_time']:.3f}s (expected >{issue['expected_minimum']:.1f}s)",
                        f"- **Likely Cause**: Stub implementation or misrouted adapter",
                        f"- **File**: `{issue['file_path']}`",
                        ""
                    ])
            
            if warning_timing:
                report_lines.extend([
                    f"## ⚠️ Timing Warnings ({len(warning_timing)} found)",
                    ""
                ])
                
                for issue in warning_timing:
                    report_lines.extend([
                        f"**{issue['benchmark']}**: {issue['actual_time']:.3f}s (expected >{issue['expected_minimum']:.1f}s)",
                        f"- **Action**: Verify implementation is not a stub",
                        ""
                    ])
        
        # Field validation issues
        if "missing_fields" in issues:
            field_issues = issues["missing_fields"]
            report_lines.extend([
                f"## 📋 Missing Field Issues ({len(field_issues)} found)",
                ""
            ])
            
            for issue in field_issues:
                fields = ", ".join(issue["missing_fields"])
                report_lines.extend([
                    f"**{issue['benchmark']}**: Missing fields `{fields}`",
                    f"- **File**: `{issue['file_path']}`",
                    ""
                ])
        
        # Resolution steps
        report_lines.extend([
            "## 🔧 Resolution Steps",
            "",
            "1. **Fix Dataset Conflicts**: Ensure each benchmark uses unique dataset_id + split",
            "2. **Fix Stub implementations**: Replace placeholder functions with real benchmarks", 
            "3. **Add Missing Fields**: Ensure all reports contain required schema fields",
            "4. **Re-run Validation**: `python scripts/validate_benchmark_sanity.py`",
            ""
        ])
        
        return "\n".join(report_lines)

def main():
    validator = BenchmarkSanityValidator()
    
    print("🔍 MISO Benchmark Sanity Gates Validation")
    print("=" * 60)
    
    success, issues = validator.validate_all_sanity_checks()
    
    print("\n" + "=" * 60)
    print("📊 SANITY CHECK SUMMARY")
    print("=" * 60)
    
    report = validator.generate_sanity_report(issues)
    print(report)
    
    # Save report to file
    report_path = Path("benchmark_sanity_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n📄 Detailed report saved: {report_path}")
    
    if success:
        print("\n🎉 All sanity checks passed!")
        sys.exit(0)
    else:
        print("\n❌ Sanity check failures detected. Fix issues before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main()
