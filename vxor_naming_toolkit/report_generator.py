#!/usr/bin/env python3
"""
VXOR Nano-Step Reporting System
==============================

Generates comprehensive reports for nano-step TDD workflow:
- JSONL logs for each nano-step
- Markdown summary reports
- Progress tracking
- Performance metrics
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class ReportMetrics:
    """Metrics for reporting system"""
    total_violations_found: int
    violations_fixed: int
    violations_pending: int
    categories_completed: int
    total_nano_steps: int
    average_fix_time_ms: float
    success_rate: float
    test_pass_rate: float

class VxorReportGenerator:
    """Generate comprehensive reports for nano-step workflow"""
    
    def __init__(self, toolkit_dir: Path):
        self.toolkit_dir = toolkit_dir
        self.reports_dir = toolkit_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Report files
        self.nano_steps_log = self.reports_dir / "nano_steps.jsonl"
        self.violations_db = self.reports_dir / "violations.json"
        self.summary_report = self.reports_dir / "summary.md"
        self.progress_report = self.reports_dir / "progress.md"
        self.metrics_json = self.reports_dir / "metrics.json"
    
    def append_nano_step(self, nano_step_data: Dict):
        """Append nano-step to JSONL log"""
        with open(self.nano_steps_log, 'a') as f:
            f.write(json.dumps(nano_step_data) + '\n')
    
    def load_nano_steps(self) -> List[Dict]:
        """Load all nano-steps from JSONL log"""
        if not self.nano_steps_log.exists():
            return []
        
        steps = []
        with open(self.nano_steps_log, 'r') as f:
            for line in f:
                if line.strip():
                    steps.append(json.loads(line.strip()))
        return steps
    
    def load_violations(self) -> List[Dict]:
        """Load violations from database"""
        if not self.violations_db.exists():
            return []
        
        with open(self.violations_db, 'r') as f:
            return json.load(f)
    
    def calculate_metrics(self) -> ReportMetrics:
        """Calculate comprehensive metrics"""
        violations = self.load_violations()
        nano_steps = self.load_nano_steps()
        
        # Violation metrics
        total_violations = len(violations)
        fixed_violations = len([v for v in violations if v.get('status') == 'fixed'])
        pending_violations = len([v for v in violations if v.get('status') == 'pending'])
        
        # Category metrics
        categories = set(v.get('category') for v in violations)
        completed_categories = len([cat for cat in categories 
                                  if all(v.get('status') == 'fixed' 
                                        for v in violations 
                                        if v.get('category') == cat)])
        
        # Nano-step metrics
        total_nano_steps = len(nano_steps)
        successful_steps = len([s for s in nano_steps if s.get('status') == 'completed'])
        
        # Performance metrics
        fix_times = [s.get('duration_ms', 0) for s in nano_steps 
                    if s.get('status') == 'completed']
        avg_fix_time = sum(fix_times) / len(fix_times) if fix_times else 0
        
        # Test metrics
        test_results = [s for s in nano_steps if s.get('test_after')]
        passed_tests = len([s for s in test_results if s.get('test_after') == 'PASS'])
        test_pass_rate = passed_tests / len(test_results) if test_results else 0
        
        return ReportMetrics(
            total_violations_found=total_violations,
            violations_fixed=fixed_violations,
            violations_pending=pending_violations,
            categories_completed=completed_categories,
            total_nano_steps=total_nano_steps,
            average_fix_time_ms=avg_fix_time,
            success_rate=successful_steps / total_nano_steps if total_nano_steps else 0,
            test_pass_rate=test_pass_rate
        )
    
    def generate_summary_report(self) -> str:
        """Generate markdown summary report"""
        metrics = self.calculate_metrics()
        violations = self.load_violations()
        nano_steps = self.load_nano_steps()
        
        # Group violations by category
        by_category = {}
        for v in violations:
            category = v.get('category', 'unknown')
            if category not in by_category:
                by_category[category] = {'total': 0, 'fixed': 0, 'pending': 0}
            by_category[category]['total'] += 1
            if v.get('status') == 'fixed':
                by_category[category]['fixed'] += 1
            elif v.get('status') == 'pending':
                by_category[category]['pending'] += 1
        
        # Recent nano-steps
        recent_steps = sorted(nano_steps, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]
        
        report = f"""# VXOR Case-Consistency Toolkit - Summary Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Overview Metrics

| Metric | Value |
|--------|--------|
| Total Violations Found | **{metrics.total_violations_found}** |
| Violations Fixed | **{metrics.violations_fixed}** ✅ |
| Violations Pending | **{metrics.violations_pending}** ⏳ |
| Categories Completed | **{metrics.categories_completed}** |
| Total Nano-Steps | **{metrics.total_nano_steps}** |
| Success Rate | **{metrics.success_rate:.1%}** |
| Test Pass Rate | **{metrics.test_pass_rate:.1%}** |
| Avg Fix Time | **{metrics.average_fix_time_ms:.0f}ms** |

## 📂 Category Status

| Category | Total | Fixed | Pending | Status |
|----------|-------|--------|---------|--------|
"""
        
        for category, stats in by_category.items():
            status_icon = "✅" if stats['pending'] == 0 else "⏳"
            progress = f"{stats['fixed']}/{stats['total']}"
            report += f"| {category} | {stats['total']} | {stats['fixed']} | {stats['pending']} | {progress} {status_icon} |\n"
        
        if recent_steps:
            report += f"""
## 🔄 Recent Nano-Steps

| Timestamp | Category | Violation ID | Status | Duration |
|-----------|----------|--------------|--------|----------|
"""
            for step in recent_steps:
                timestamp = step.get('timestamp', '')[:19]  # Truncate to date+time
                category = step.get('category', 'unknown')
                violation_id = step.get('violation_id', 'unknown')
                status = step.get('status', 'unknown')
                duration = step.get('duration_ms', 0)
                status_icon = {"completed": "✅", "error": "❌", "started": "⏳"}.get(status, "❓")
                report += f"| {timestamp} | {category} | {violation_id} | {status} {status_icon} | {duration}ms |\n"
        
        # Progress indicators
        if metrics.violations_pending > 0:
            report += f"""
## 🎯 Next Steps

- **{metrics.violations_pending}** violations remaining
- Run: `make fix-one CURRENT_CATEGORY=<category>` to continue
- Run: `make all-categories` to process all remaining violations

"""
        else:
            report += """
## 🎉 Completion Status

✅ **All violations have been fixed!**

Your VXOR codebase is now fully compliant with the naming policy.

"""
        
        report += f"""
## 📈 Performance Stats

- **Average fix time:** {metrics.average_fix_time_ms:.0f}ms per violation
- **Success rate:** {metrics.success_rate:.1%} of nano-steps completed successfully  
- **Test reliability:** {metrics.test_pass_rate:.1%} of tests passed after fixes

## 🛠️ Nano-Step TDD Workflow

Each violation follows the strict TDD process:

1. **Test (Fail)** - Generate failing test for violation
2. **Fix** - Apply minimal fix to make test pass  
3. **Test (Pass)** - Verify fix resolves the violation
4. **Commit** - Commit single fix with conventional message
5. **Report** - Update progress tracking

---
*Report generated by VXOR Case-Consistency Toolkit*
"""
        
        return report
    
    def generate_progress_report(self) -> str:
        """Generate progress tracking report"""
        violations = self.load_violations()
        
        # Group by category and severity
        category_progress = {}
        for violation in violations:
            category = violation.get('category', 'unknown')
            severity = violation.get('severity', 'medium')
            status = violation.get('status', 'pending')
            
            if category not in category_progress:
                category_progress[category] = {
                    'critical': {'total': 0, 'fixed': 0},
                    'high': {'total': 0, 'fixed': 0}, 
                    'medium': {'total': 0, 'fixed': 0},
                    'low': {'total': 0, 'fixed': 0}
                }
            
            category_progress[category][severity]['total'] += 1
            if status == 'fixed':
                category_progress[category][severity]['fixed'] += 1
        
        report = f"""# VXOR Case-Consistency - Progress Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 Progress by Category and Severity

"""
        
        for category, severities in category_progress.items():
            report += f"""
### {category.title()}

| Severity | Progress | Status |
|----------|----------|--------|
"""
            for severity, stats in severities.items():
                if stats['total'] > 0:
                    progress_pct = (stats['fixed'] / stats['total']) * 100
                    progress_bar = "█" * int(progress_pct / 10) + "░" * (10 - int(progress_pct / 10))
                    status_text = f"{stats['fixed']}/{stats['total']} ({progress_pct:.0f}%)"
                    report += f"| {severity.title()} | {progress_bar} | {status_text} |\n"
        
        return report
    
    def save_metrics_json(self):
        """Save metrics as JSON for CI/tooling"""
        metrics = self.calculate_metrics()
        
        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': asdict(metrics),
            'summary': {
                'completion_rate': metrics.violations_fixed / metrics.total_violations_found if metrics.total_violations_found else 1.0,
                'is_complete': metrics.violations_pending == 0,
                'categories_total': len(set(v.get('category') for v in self.load_violations())),
                'categories_completed': metrics.categories_completed
            }
        }
        
        with open(self.metrics_json, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def generate_all_reports(self):
        """Generate all reports"""
        print("📊 Generating comprehensive reports...")
        
        # Generate summary report
        summary = self.generate_summary_report()
        with open(self.summary_report, 'w') as f:
            f.write(summary)
        print(f"   ✅ Summary: {self.summary_report}")
        
        # Generate progress report
        progress = self.generate_progress_report()
        with open(self.progress_report, 'w') as f:
            f.write(progress)
        print(f"   ✅ Progress: {self.progress_report}")
        
        # Save metrics JSON
        self.save_metrics_json()
        print(f"   ✅ Metrics: {self.metrics_json}")
        
        print("📊 All reports generated successfully!")

def main():
    """CLI entry point for report generation"""
    import sys
    
    toolkit_dir = Path("/Users/gecko365/vxor_naming_toolkit")
    generator = VxorReportGenerator(toolkit_dir)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--append":
        # Append mode - just update metrics
        generator.save_metrics_json()
        print("📊 Metrics updated")
    else:
        # Full report generation
        generator.generate_all_reports()

if __name__ == "__main__":
    main()
