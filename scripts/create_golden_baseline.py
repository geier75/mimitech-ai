#!/usr/bin/env python3
"""
Create Golden Baseline - Phase 11
Script to establish a golden baseline from benchmark results
"""

import argparse
import json
import sys
from pathlib import Path
from miso.baseline.golden_baseline_manager import GoldenBaselineManager

def main():
    parser = argparse.ArgumentParser(description="Create golden baseline from benchmark results")
    parser.add_argument("--report", "-r", required=True,
                       help="Benchmark report JSON file")
    parser.add_argument("--name", "-n", default="default",
                       help="Baseline name (default: default)")
    parser.add_argument("--logs-dir", 
                       help="Directory containing JSONL log files")
    parser.add_argument("--datasets-dir",
                       help="Directory containing dataset manifests")
    parser.add_argument("--baseline-dir",
                       help="Directory to store baselines (default: ./baseline)")
    
    args = parser.parse_args()
    
    # Load benchmark report
    report_path = Path(args.report)
    if not report_path.exists():
        print(f"❌ Report file not found: {report_path}")
        sys.exit(1)
    
    try:
        with open(report_path) as f:
            benchmark_report = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load benchmark report: {e}")
        sys.exit(1)
    
    # Initialize baseline manager
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else None
    manager = GoldenBaselineManager(baseline_dir)
    
    # Collect log files
    log_files = []
    if args.logs_dir:
        logs_path = Path(args.logs_dir)
        if logs_path.exists():
            log_files = list(logs_path.glob("*.jsonl"))
            print(f"📋 Found {len(log_files)} log files")
    
    # Collect dataset manifests
    dataset_manifests = []
    if args.datasets_dir:
        datasets_path = Path(args.datasets_dir)
        if datasets_path.exists():
            dataset_manifests = list(datasets_path.glob("**/manifest.sha256"))
            print(f"📁 Found {len(dataset_manifests)} dataset manifests")
    
    try:
        # Create golden baseline
        baseline_path = manager.create_golden_baseline(
            benchmark_report=benchmark_report,
            log_files=log_files,
            dataset_manifests=dataset_manifests,
            baseline_name=args.name
        )
        
        print(f"\n🎯 Golden baseline created successfully!")
        print(f"📍 Location: {baseline_path}")
        
        # Show baseline summary
        metrics = manager.get_baseline_metrics()
        print(f"\n📊 Baseline Summary:")
        print(f"   Benchmarks: {len(metrics)}")
        
        total_samples = sum(m.samples_processed for m in metrics.values())
        avg_accuracy = sum(m.accuracy for m in metrics.values()) / len(metrics) if metrics else 0
        total_duration = sum(m.duration_s for m in metrics.values())
        
        print(f"   Total Samples: {total_samples:,}")
        print(f"   Average Accuracy: {avg_accuracy:.2%}")  
        print(f"   Total Duration: {total_duration:.1f}s")
        
        print(f"\n📋 Per-Benchmark Metrics:")
        for name, metric in sorted(metrics.items()):
            print(f"   {name}: {metric.accuracy:.2%} accuracy, {metric.samples_processed:,} samples, {metric.duration_s:.1f}s")
        
    except Exception as e:
        print(f"❌ Failed to create golden baseline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
