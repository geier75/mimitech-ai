#!/usr/bin/env python3
"""
Phase 10: Mutation tests for schema validation robustness
Tests edge cases and malformed data to ensure validation catches errors
"""

import json
import pytest
import tempfile
from pathlib import Path
from miso.validation.schema_validator import SchemaValidator

class TestMutationValidation:
    """Test suite for schema validation robustness against mutations"""
    
    @pytest.fixture
    def validator(self):
        """Create schema validator instance"""
        return SchemaValidator()
    
    @pytest.fixture
    def valid_benchmark_report(self):
        """Valid benchmark report for mutation testing"""
        return {
            "schema_version": "1.0.0",
            "suite_name": "mutation_test_suite",
            "timestamp": "2024-03-25T10:00:00Z",
            "summary": {
                "total_benchmarks": 1,
                "passed_benchmarks": 1,
                "failed_benchmarks": 0,
                "total_samples": 100,
                "total_duration_s": 10.5,
                "average_accuracy": 0.85
            },
            "reproducibility": {
                "git_commit": "abc123def456",
                "git_tag": "v1.0.0",
                "python_version": "3.11.5",
                "platform": "Darwin-23.1.0-arm64",
                "env_flags": {
                    "PYTHONHASHSEED": "0",
                    "OMP_NUM_THREADS": "1"
                },
                "seed": 42
            },
            "metrics": {
                "compute_mode": "full"
            },
            "results": [
                {
                    "benchmark_name": "test_benchmark",
                    "status": "PASS",
                    "samples_processed": 100,
                    "predictions_count": 100,
                    "accuracy": 0.85,
                    "duration_s": 10.5,
                    "throughput_samples_per_sec": 9.52,
                    "metadata": {
                        "dataset_version": "1.0",
                        "model_config": "default"
                    }
                }
            ]
        }
    
    def test_mutation_missing_required_field(self, validator, valid_benchmark_report):
        """Test 1: Missing required field - should fail validation"""
        # Remove required field 'schema_version'
        mutated_report = valid_benchmark_report.copy()
        del mutated_report['schema_version']
        
        with pytest.raises(Exception) as exc_info:
            validator.validate_benchmark_report(mutated_report)
        
        assert "schema_version" in str(exc_info.value).lower() or "required" in str(exc_info.value).lower()
    
    def test_mutation_missing_reproducibility_block(self, validator, valid_benchmark_report):
        """Test: Missing reproducibility block - should fail validation"""
        mutated_report = valid_benchmark_report.copy()
        del mutated_report['reproducibility']
        
        with pytest.raises(Exception) as exc_info:
            validator.validate_benchmark_report(mutated_report)
        
        assert "reproducibility" in str(exc_info.value).lower()
    
    def test_mutation_missing_compute_mode(self, validator, valid_benchmark_report):
        """Test: Missing compute_mode - should fail validation"""
        mutated_report = valid_benchmark_report.copy()
        del mutated_report['metrics']['compute_mode']
        
        with pytest.raises(Exception) as exc_info:
            validator.validate_benchmark_report(mutated_report)
        
        assert "compute_mode" in str(exc_info.value).lower()
    
    def test_mutation_invalid_accuracy_range(self, validator, valid_benchmark_report):
        """Test 2: Invalid accuracy value - should fail validation"""
        # Set accuracy > 1.0 (invalid range)
        mutated_report = valid_benchmark_report.copy()
        mutated_report['results'][0]['accuracy'] = 1.5
        mutated_report['summary']['average_accuracy'] = 1.5
        
        with pytest.raises(Exception) as exc_info:
            validator.validate_benchmark_report(mutated_report)
        
        assert "accuracy" in str(exc_info.value).lower() or "range" in str(exc_info.value).lower()
    
    def test_mutation_negative_accuracy(self, validator, valid_benchmark_report):
        """Test: Negative accuracy value - should fail validation"""
        mutated_report = valid_benchmark_report.copy()
        mutated_report['results'][0]['accuracy'] = -0.1
        mutated_report['summary']['average_accuracy'] = -0.1
        
        with pytest.raises(Exception) as exc_info:
            validator.validate_benchmark_report(mutated_report)
        
        assert "accuracy" in str(exc_info.value).lower()
    
    def test_mutation_invalid_status(self, validator, valid_benchmark_report):
        """Test 3: Invalid status value - should fail validation"""
        # Set invalid status (not PASS/FAIL)
        mutated_report = valid_benchmark_report.copy()
        mutated_report['results'][0]['status'] = "INVALID_STATUS"
        
        with pytest.raises(Exception) as exc_info:
            validator.validate_benchmark_report(mutated_report)
        
        assert "status" in str(exc_info.value).lower() or "enum" in str(exc_info.value).lower()
    
    def test_mutation_invalid_compute_mode(self, validator, valid_benchmark_report):
        """Test: Invalid compute_mode - should fail validation"""
        mutated_report = valid_benchmark_report.copy()
        mutated_report['metrics']['compute_mode'] = "invalid_mode"
        
        with pytest.raises(Exception) as exc_info:
            validator.validate_benchmark_report(mutated_report)
        
        assert "compute_mode" in str(exc_info.value).lower()
    
    def test_mutation_negative_duration(self, validator, valid_benchmark_report):
        """Test: Negative duration - should fail validation"""
        mutated_report = valid_benchmark_report.copy()
        mutated_report['results'][0]['duration_s'] = -1.0
        mutated_report['summary']['total_duration_s'] = -1.0
        
        with pytest.raises(Exception) as exc_info:
            validator.validate_benchmark_report(mutated_report)
        
        assert "duration" in str(exc_info.value).lower() or "negative" in str(exc_info.value).lower()
    
    def test_mutation_inconsistent_sample_counts(self, validator, valid_benchmark_report):
        """Test: Inconsistent sample counts - should fail cross-check"""
        mutated_report = valid_benchmark_report.copy()
        # Mismatch between samples_processed and predictions_count
        mutated_report['results'][0]['samples_processed'] = 100
        mutated_report['results'][0]['predictions_count'] = 90  # Mismatch
        
        with pytest.raises(Exception) as exc_info:
            validator.validate_benchmark_report(mutated_report)
        
        assert "cross" in str(exc_info.value).lower() or "mismatch" in str(exc_info.value).lower()
    
    def test_mutation_invalid_schema_version(self, validator, valid_benchmark_report):
        """Test: Invalid schema version format - should fail validation"""
        mutated_report = valid_benchmark_report.copy()
        mutated_report['schema_version'] = "invalid.version.format"
        
        with pytest.raises(Exception) as exc_info:
            validator.validate_benchmark_report(mutated_report)
        
        assert "schema_version" in str(exc_info.value).lower() or "version" in str(exc_info.value).lower()

class TestMutationFileValidation:
    """Test file-based mutation validation scenarios"""
    
    @pytest.fixture
    def validator(self):
        return SchemaValidator()
    
    def test_mutation_malformed_json(self, validator):
        """Test: Malformed JSON - should fail gracefully"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": "json"')  # Missing closing brace
            temp_path = f.name
        
        try:
            with pytest.raises(Exception) as exc_info:
                validator.validate_benchmark_report_file(temp_path)
            
            assert "json" in str(exc_info.value).lower() or "parse" in str(exc_info.value).lower()
        finally:
            Path(temp_path).unlink()
    
    def test_mutation_empty_file(self, validator):
        """Test: Empty file - should fail validation"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name  # Empty file
        
        try:
            with pytest.raises(Exception) as exc_info:
                validator.validate_benchmark_report_file(temp_path)
            
            assert "empty" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()
        finally:
            Path(temp_path).unlink()

class TestMutationPlausibilityChecks:
    """Test mutation scenarios for plausibility monitoring"""
    
    @pytest.fixture
    def validator(self):
        return SchemaValidator()
    
    @pytest.fixture
    def outlier_report(self):
        """Report with outlier values that should trigger warnings"""
        return {
            "schema_version": "1.0.0",
            "suite_name": "outlier_test",
            "timestamp": "2024-03-25T10:00:00Z",
            "summary": {
                "total_benchmarks": 1,
                "passed_benchmarks": 1,
                "failed_benchmarks": 0,
                "total_samples": 100,
                "total_duration_s": 0.001,  # Suspiciously fast
                "average_accuracy": 0.99999  # Suspiciously high
            },
            "reproducibility": {
                "git_commit": "abc123def456",
                "python_version": "3.11.5",
                "platform": "Darwin-23.1.0-arm64",
                "env_flags": {"PYTHONHASHSEED": "0"},
                "seed": 42
            },
            "metrics": {
                "compute_mode": "full"
            },
            "results": [
                {
                    "benchmark_name": "mmlu",
                    "status": "PASS",
                    "samples_processed": 100,
                    "predictions_count": 100,
                    "accuracy": 0.99999,  # Suspiciously high for MMLU
                    "duration_s": 0.001,  # Suspiciously fast
                    "throughput_samples_per_sec": 100000,  # Impossibly fast
                    "metadata": {"dataset_version": "1.0"}
                }
            ]
        }
    
    def test_mutation_outlier_detection(self, validator, outlier_report):
        """Test: Outlier values should trigger warnings"""
        # This should not fail validation but generate warnings
        validation_result = validator.validate_benchmark_report(outlier_report, allow_warnings=True)
        
        # Check that warnings were generated for outliers
        assert validation_result.warnings, "Expected warnings for outlier values"
        
        warning_text = " ".join(validation_result.warnings).lower()
        assert any(keyword in warning_text for keyword in 
                  ["outlier", "suspicious", "plausibility", "unusual"])

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
