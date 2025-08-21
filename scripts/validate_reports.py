#!/usr/bin/env python3
"""
CI Script: Validate benchmark reports against JSON schemas
Usage: python scripts/validate_reports.py [reports_dir]
"""

import sys
import json
import logging
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from miso.validation.schema_validator import SchemaValidator

def setup_logging():
    """Setup logging for CI environment"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

def find_report_files(reports_dir: Path) -> List[Path]:
    """Find all JSON report files in directory"""
    if not reports_dir.exists():
        logging.info(f"Reports directory does not exist: {reports_dir}")
        return []
    
    json_files = list(reports_dir.glob("**/*.json"))
    logging.info(f"Found {len(json_files)} JSON files in {reports_dir}")
    return json_files

def validate_report_file(validator: SchemaValidator, report_path: Path) -> bool:
    """Validate single report file"""
    try:
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        
        # Check if this looks like a benchmark report
        if not all(key in report_data for key in ["report_id", "summary", "results"]):
            logging.info(f"⏭️  Skipping {report_path.name} - not a benchmark report")
            return True
        
        validator.validate_benchmark_report(report_data)
        logging.info(f"✅ {report_path.name} - validation passed")
        return True
        
    except Exception as e:
        logging.error(f"❌ {report_path.name} - validation failed: {e}")
        return False

def main():
    """Main CI validation function"""
    setup_logging()
    
    # Get reports directory from command line or default
    if len(sys.argv) > 1:
        reports_dir = Path(sys.argv[1])
    else:
        project_root = Path(__file__).parent.parent
        reports_dir = project_root / "tests" / "reports"
    
    logging.info(f"🔍 Validating benchmark reports in: {reports_dir}")
    
    validator = SchemaValidator()
    report_files = find_report_files(reports_dir)
    
    if not report_files:
        logging.info("✅ No report files to validate")
        return 0
    
    # Validate each report
    failed_count = 0
    for report_path in report_files:
        if not validate_report_file(validator, report_path):
            failed_count += 1
    
    # Summary
    total_files = len(report_files)
    passed_count = total_files - failed_count
    
    logging.info(f"\n📊 Validation Summary:")
    logging.info(f"   Total files: {total_files}")
    logging.info(f"   Passed: {passed_count}")
    logging.info(f"   Failed: {failed_count}")
    
    if failed_count > 0:
        logging.error(f"❌ {failed_count} report(s) failed schema validation")
        return 1
    else:
        logging.info("✅ All reports passed schema validation")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
