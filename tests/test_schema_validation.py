"""
Test Schema Validation for MISO Benchmark Reports
Tests both positive (valid) and negative (invalid) cases
"""

import unittest
import json
import tempfile
from pathlib import Path
from datetime import datetime

from miso.validation.schema_validator import SchemaValidator

class TestSchemaValidation(unittest.TestCase):
    """Test suite for JSON schema validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = SchemaValidator()
        self.test_timestamp = datetime.now().isoformat() + "Z"
        
    def test_valid_bench_result(self):
        """Test validation of valid benchmark result"""
        valid_result = {
            "schema_version": "v1.0.0",
            "test_name": "matrix_10x10",
            "status": "passed",
            "execution_time_ms": 1250.5,
            "timestamp": self.test_timestamp,
            "accuracy": 85.7,
            "samples_processed": 1000,
            "metadata": {
                "dataset_paths": ["/data/matrix_10x10.allow"],
                "finished_at": 1692345678.9
            }
        }
        
        # Should pass validation without raising exception
        result = self.validator.validate_bench_result(valid_result)
        self.assertTrue(result)
        
    def test_invalid_bench_result_missing_required(self):
        """Test validation fails for missing required fields"""
        invalid_result = {
            "schema_version": "v1.0.0",
            "test_name": "incomplete_test",
            # Missing required fields: status, execution_time_ms, timestamp
            "accuracy": 75.0,
            "samples_processed": 500
        }
        
        with self.assertRaises(ValueError) as cm:
            self.validator.validate_bench_result(invalid_result)
        self.assertIn("schema validation failed", str(cm.exception))
        
    def test_invalid_bench_result_accuracy_range(self):
        """Test validation fails for accuracy outside 0-100 range"""
        invalid_result = {
            "schema_version": "v1.0.0",
            "test_name": "accuracy_overflow",
            "status": "passed",
            "execution_time_ms": 1000.0,
            "timestamp": self.test_timestamp,
            "accuracy": 150.0,  # Invalid: > 100
            "samples_processed": 1000
        }
        
        with self.assertRaises(ValueError):
            self.validator.validate_bench_result(invalid_result)
            
    def test_invalid_bench_result_negative_samples(self):
        """Test validation fails for negative samples_processed"""
        invalid_result = {
            "schema_version": "v1.0.0",
            "test_name": "negative_samples",
            "status": "passed",
            "execution_time_ms": 1000.0,
            "timestamp": self.test_timestamp,
            "accuracy": 80.0,
            "samples_processed": -10  # Invalid: negative
        }
        
        with self.assertRaises(ValueError):
            self.validator.validate_bench_result(invalid_result)
            
    def test_invalid_bench_result_wrong_status(self):
        """Test validation fails for invalid status values"""
        invalid_result = {
            "schema_version": "v1.0.0",
            "test_name": "wrong_status",
            "status": "SUCCESS",  # Invalid: not in enum
            "execution_time_ms": 1000.0,
            "timestamp": self.test_timestamp,
            "accuracy": 80.0,
            "samples_processed": 500
        }
        
        with self.assertRaises(ValueError):
            self.validator.validate_bench_result(invalid_result)
    
    def test_valid_benchmark_report(self):
        """Test validation of valid benchmark report"""
        valid_report = {
            "schema_version": "v1.0.0",
            "report_id": "test_report_001",
            "timestamp": self.test_timestamp,
            "summary": {
                "total_tests": 2,
                "passed": 1,
                "failed": 1,
                "skipped": 0,
                "errors": 0,
                "total_samples_processed": 1500,
                "average_accuracy": 75.0,
                "total_execution_time_ms": 2500.0
            },
            "results": [
                {
                    "schema_version": "v1.0.0",
                    "test_name": "test1",
                    "status": "passed",
                    "execution_time_ms": 1000.0,
                    "timestamp": self.test_timestamp,
                    "accuracy": 90.0,
                    "samples_processed": 1000
                },
                {
                    "schema_version": "v1.0.0",
                    "test_name": "test2",
                    "status": "failed",
                    "execution_time_ms": 1500.0,
                    "timestamp": self.test_timestamp,
                    "accuracy": 60.0,
                    "samples_processed": 500
                }
            ],
            "system_info": {
                "platform": "macOS Apple Silicon",
                "python_version": "3.11+",
                "hardware": "MacBook Pro M4 Max"
            }
        }
        
        result = self.validator.validate_benchmark_report(valid_report)
        self.assertTrue(result)
        
    def test_invalid_report_inconsistent_totals(self):
        """Test validation fails for inconsistent summary totals"""
        invalid_report = {
            "schema_version": "v1.0.0",
            "report_id": "inconsistent_report",
            "timestamp": self.test_timestamp,
            "summary": {
                "total_tests": 3,  # Says 3 tests
                "passed": 1,
                "failed": 1,
                "skipped": 0,
                "errors": 0,  # But 1+1+0+0 = 2, not 3
                "total_samples_processed": 1000
            },
            "results": [  # And only 1 result provided
                {
                    "schema_version": "v1.0.0",
                    "test_name": "test1",
                    "status": "passed",
                    "execution_time_ms": 1000.0,
                    "timestamp": self.test_timestamp,
                    "accuracy": 90.0,
                    "samples_processed": 1000
                }
            ]
        }
        
        with self.assertRaises(ValueError) as cm:
            self.validator.validate_benchmark_report(invalid_report)
        self.assertIn("total_tests", str(cm.exception))
        
    def test_invalid_report_samples_mismatch(self):
        """Test validation fails for sample count mismatch"""
        invalid_report = {
            "schema_version": "v1.0.0",
            "report_id": "samples_mismatch",
            "timestamp": self.test_timestamp,
            "summary": {
                "total_tests": 1,
                "passed": 1,
                "failed": 0,
                "skipped": 0,
                "errors": 0,
                "total_samples_processed": 2000  # Claims 2000
            },
            "results": [
                {
                    "schema_version": "v1.0.0",
                    "test_name": "test1",
                    "status": "passed",
                    "execution_time_ms": 1000.0,
                    "timestamp": self.test_timestamp,
                    "accuracy": 90.0,
                    "samples_processed": 1000  # But actual is 1000
                }
            ]
        }
        
        with self.assertRaises(ValueError) as cm:
            self.validator.validate_benchmark_report(invalid_report)
        self.assertIn("samples_processed mismatch", str(cm.exception))
    
    def test_bench_result_conversion(self):
        """Test conversion from BenchResult object to schema dict"""
        # Mock BenchResult object
        class MockBenchResult:
            def __init__(self):
                self.name = "test_conversion"
                self.status = "PASS"
                self.duration_s = 1.5
                self.started_at = 1692345678.0
                self.accuracy = 0.857  # Internal format [0.0, 1.0]
                self.samples_processed = 1000
                self.dataset_paths = ["/data/test.allow"]
                self.finished_at = 1692345679.5
        
        mock_result = MockBenchResult()
        schema_dict = self.validator.convert_bench_result_to_schema(mock_result)
        
        # Verify conversion
        self.assertEqual(schema_dict["test_name"], "test_conversion")
        self.assertEqual(schema_dict["status"], "passed")
        self.assertEqual(schema_dict["execution_time_ms"], 1500.0)  # Converted to ms
        self.assertEqual(schema_dict["accuracy"], 85.7)  # Converted to percentage
        self.assertEqual(schema_dict["samples_processed"], 1000)
        self.assertEqual(schema_dict["schema_version"], "v1.0.0")
        
        # Should pass schema validation
        self.validator.validate_bench_result(schema_dict)

if __name__ == '__main__':
    unittest.main()
