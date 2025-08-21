#!/usr/bin/env python3
"""
Detect Drift - Phase 11
Script to detect performance drift against golden baseline
"""

import argparse
import json
import sys
from pathlib import Path
from miso.baseline.golden_baseline_manager import GoldenBaselineManager
from miso.baseline.drift_detector import DriftDetector, DriftSeverity

def main():
    parser = argparse.ArgumentParser(description="Detect performance drift against golden baseline")
    parser.add_argument("--report", "-r", required=True,
                       help="Current benchmark report JSON file")
    parser.add_argument("--baseline-dir",
                       help="Directory containing baselines (default: ./baseline)")
    parser.add_argument("--output", "-o",
                       help="Output file for drift report (default: auto-generated)")
    parser.add_argument("--fail-on-drift", action="store_true",
                       help="Exit with error code if drift detected")
    parser.add_argument("--fail-on-critical", action="store_true", default=True,
                       help="Exit with error code if critical drift detected (default)")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress output, only show summary")
    
    args = parser.parse_args()
    
    # Load current benchmark report
    report_path = Path(args.report)
    if not report_path.exists():
        print(f"❌ Report file not found: {report_path}")
        sys.exit(1)
    
    try:
        with open(report_path) as f:
            current_report = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load benchmark report: {e}")
        sys.exit(1)
    
    # Initialize drift detector
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else None
    manager = GoldenBaselineManager(baseline_dir)
    detector = DriftDetector(manager)
    
    try:
        # Detect drift
        if not args.quiet:
            print("🔍 Detecting drift against golden baseline...")
        
        drift_report = detector.detect_drift(current_report)
        
        # Print summary
        if not args.quiet:
            detector.print_drift_summary(drift_report)
        else:
            print(f"Drift Status: {drift_report.overall_status} ({drift_report.drifted_benchmarks}/{drift_report.total_benchmarks} drifted)")
        
        # Save drift report
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = None
            
        report_path = detector.save_drift_report(drift_report, output_path)
        
        if not args.quiet:
            print(f"\n📄 Drift report saved: {report_path}")
        
        # Exit codes based on drift detection
        if drift_report.has_critical_drift() and args.fail_on_critical:
            if not args.quiet:
                print("\n❌ Critical drift detected - failing build")
            sys.exit(1)
        elif (drift_report.has_warnings() or drift_report.has_critical_drift()) and args.fail_on_drift:
            if not args.quiet:
                print("\n⚠️  Drift detected - failing build")
            sys.exit(1)
        else:
            if not args.quiet:
                print("\n✅ Drift check completed")
            sys.exit(0)
            
    except FileNotFoundError as e:
        print(f"❌ No golden baseline found: {e}")
        print("💡 Create a golden baseline first using: python scripts/create_golden_baseline.py")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Drift detection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
