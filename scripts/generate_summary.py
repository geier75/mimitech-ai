#!/usr/bin/env python3
"""
Generate benchmark summary reports from JSON benchmark results
Phase 9: SUMMARY.md generation script
"""

import argparse
import json
import sys
from pathlib import Path
from miso.reporting.summary_generator import SummaryGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate benchmark summary reports")
    parser.add_argument("--input", "-i", required=True, 
                       help="Input JSON benchmark report file")
    parser.add_argument("--output", "-o", 
                       help="Output markdown file (default: auto-generated)")
    parser.add_argument("--output-dir", 
                       help="Output directory (default: ./reports)")
    parser.add_argument("--run-id", 
                       help="Custom run ID for the report")
    
    args = parser.parse_args()
    
    # Load benchmark report
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)
    
    try:
        with open(input_path) as f:
            benchmark_report = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load benchmark report: {e}")
        sys.exit(1)
    
    # Initialize summary generator
    output_dir = Path(args.output_dir) if args.output_dir else None
    generator = SummaryGenerator(output_dir)
    
    # Generate and save summary
    try:
        if args.output:
            output_path = generator.save_summary_report(benchmark_report, args.output)
        else:
            output_path = generator.save_summary_report(benchmark_report)
        
        print(f"✅ Summary report generated: {output_path}")
        
        # Print key metrics
        suite_summary = generator.generate_suite_summary(benchmark_report)
        print(f"\n📊 Key Metrics:")
        print(f"   Benchmarks: {suite_summary.passed_benchmarks}/{suite_summary.total_benchmarks} passed")
        print(f"   Average Accuracy: {suite_summary.average_accuracy:.2%}")
        print(f"   Total Samples: {suite_summary.total_samples:,}")
        print(f"   Duration: {suite_summary.total_duration_s:.1f}s")
        print(f"   Compute Mode: {suite_summary.compute_mode}")
        
    except Exception as e:
        print(f"❌ Failed to generate summary: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
