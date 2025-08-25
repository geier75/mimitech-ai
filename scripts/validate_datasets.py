#!/usr/bin/env python3
"""
Dataset Validation Script
Validates dataset integrity and minimum sample counts before benchmark execution
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from miso.datasets.dataset_integrity import DatasetIntegrityValidator, MINIMUM_SAMPLE_COUNTS

def setup_logging():
    """Setup logging for validation"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

def main():
    """Main validation function"""
    setup_logging()
    
    print("🔍 MISO Dataset Integrity Validation")
    print("=" * 50)
    
    # Initialize checker
    checker = DatasetIntegrityValidator()
    
    # Check if datasets root exists
    if not checker.datasets_root.exists():
        print(f"❌ CRITICAL: Datasets directory not found: {checker.datasets_root}")
        print("   Please create datasets directory or update VXOR_DATA_ROOT environment variable")
        return 1
    
    print(f"📁 Datasets root: {checker.datasets_root}")
    print(f"🎯 Required benchmarks: {len(MINIMUM_SAMPLE_COUNTS)}")
    print()
    
    # Generate checksums manifest if needed
    try:
        checksum_manager = checker._import_checksum_manager()
        if not checksum_manager.manifest_file.exists():
            print("📋 Generating checksums manifest...")
            manifest = checksum_manager.generate_checksums_manifest()
            print(f"✅ Generated manifest with {manifest['metadata']['total_files']} files")
        else:
            print("📋 Using existing checksums manifest")
    except Exception as e:
        print(f"⚠️  Warning: Could not generate checksums manifest: {e}")
    
    print()
    
    # Validate datasets
    print("🔍 Validating datasets...")
    passed, failed_datasets = checker.check_dataset_integrity_gate()
    
    # Generate detailed report
    report = checker.generate_dataset_report()
    
    print("\n📊 Validation Summary:")
    print(f"   Total datasets: {report['summary']['total_datasets']}")
    print(f"   ✅ Passed: {report['summary']['passed']}")
    print(f"   ❌ Failed: {report['summary']['failed']}")
    print(f"   ⚠️  Warnings: {report['summary']['warnings']}")
    print(f"   📈 Total samples: {report['summary']['total_samples']:,}")
    print(f"   🎯 Required samples: {report['summary']['total_required']:,}")
    
    # Show detailed results
    print("\n📋 Detailed Results:")
    for dataset_name, dataset_info in report['datasets'].items():
        status_icon = "✅" if dataset_info['status'] == "PASS" else "⚠️" if dataset_info['status'] == "WARNING" else "❌"
        checksum_icon = "✓" if dataset_info['checksum_valid'] else "✗"
        
        print(f"   {status_icon} {dataset_name}:")
        print(f"      Samples: {dataset_info['sample_count']:,} / {dataset_info['min_required']:,} required")
        print(f"      Checksum: {checksum_icon}")
        
        if dataset_info['error_message']:
            print(f"      Error: {dataset_info['error_message']}")
    
    # Exit with appropriate code
    if passed:
        print(f"\n✅ VALIDATION PASSED - All {len(MINIMUM_SAMPLE_COUNTS)} datasets meet requirements")
        return 0
    else:
        print(f"\n❌ VALIDATION FAILED - {len(failed_datasets)} dataset(s) failed:")
        for failure in failed_datasets:
            print(f"   - {failure}")
        print("\nPlease fix dataset issues before running benchmarks.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
