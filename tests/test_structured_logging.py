"""
Test structured logging and cross-check functionality
"""

import json
import tempfile
import time
from pathlib import Path
import pytest

from miso.logging import BenchmarkLogger, StructuredLogger, JSONLHandler


class TestStructuredLogger:
    """Test basic structured logging functionality"""
    
    def test_structured_logger_basic(self):
        """Test basic structured logging operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = StructuredLogger("test_benchmark", Path(temp_dir))
            
            # Test different log levels
            logger.info("test_event", "Test info message", key1="value1")
            logger.warning("warning_event", "Test warning", key2="value2") 
            logger.error("error_event", "Test error", key3="value3")
            logger.debug("debug_event", "Test debug", key4="value4")
            
            # Verify JSONL file was created
            jsonl_file = Path(temp_dir) / "test_benchmark.jsonl"
            assert jsonl_file.exists()
            
            # Read and validate JSONL content
            lines = jsonl_file.read_text().strip().split('\n')
            assert len(lines) == 4
            
            for line in lines:
                entry = json.loads(line)
                assert "timestamp" in entry
                assert "level" in entry
                assert "benchmark_name" in entry
                assert "event_type" in entry
                assert "message" in entry
                assert "execution_id" in entry
                assert entry["benchmark_name"] == "test_benchmark"


class TestBenchmarkLogger:
    """Test benchmark-specific logging with cross-checks"""
    
    def test_benchmark_logger_lifecycle(self):
        """Test complete benchmark logging lifecycle"""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = BenchmarkLogger("mmlu", Path(temp_dir))
            
            # Start benchmark
            dataset_paths = ["/path/to/mmlu_train.jsonl", "/path/to/mmlu_test.jsonl"]
            logger.start_benchmark(dataset_paths, expected_samples=1000)
            
            # Simulate processing samples with predictions
            for i in range(10):
                sample_id = f"sample_{i:03d}"
                prediction = f"answer_{i % 4}"  # A, B, C, D
                confidence = 0.7 + (i * 0.02)
                
                logger.log_prediction(sample_id, prediction, confidence)
                logger.log_sample_processed(sample_id, processing_time_ms=50.0 + i)
            
            # Update accuracy metrics
            logger.log_accuracy_update(0.80, correct_predictions=8, total_predictions=10)
            
            # Finish benchmark
            summary = logger.finish_benchmark(final_accuracy=0.80, duration_s=30.5)
            
            # Verify summary
            assert summary["benchmark_name"] == "mmlu"
            assert summary["final_accuracy"] == 0.80
            assert summary["predictions_count"] == 10
            assert summary["samples_processed"] == 10
            assert summary["cross_check_passed"] is True
            
            # Verify cross-check status
            status = logger.get_cross_check_status()
            assert status["predictions_count"] == 10
            assert status["samples_processed"] == 10
            assert status["match"] is True
            assert status["difference"] == 0
    
    def test_cross_check_mismatch_detection(self):
        """Test cross-check failure detection"""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = BenchmarkLogger("arc", Path(temp_dir))
            
            logger.start_benchmark(["/path/to/arc.jsonl"], expected_samples=100)
            
            # Create mismatch: more predictions than samples processed
            for i in range(5):
                logger.log_prediction(f"sample_{i}", f"answer_{i}")
            
            # Process fewer samples than predictions
            for i in range(3):  # Only 3 samples processed vs 5 predictions
                logger.log_sample_processed(f"sample_{i}")
            
            # Update accuracy with inconsistent total
            logger.log_accuracy_update(0.60, correct_predictions=3, total_predictions=7)  # Wrong total
            
            # Finish should detect cross-check failure
            summary = logger.finish_benchmark(0.60)
            
            assert summary["predictions_count"] == 5
            assert summary["samples_processed"] == 3
            assert summary["cross_check_passed"] is False
            
            # Verify JSONL logs contain warnings/errors
            jsonl_file = Path(temp_dir) / "benchmark.arc.jsonl"
            assert jsonl_file.exists()
            
            content = jsonl_file.read_text()
            assert "accuracy_inconsistency" in content
            assert "cross_check_failed" in content
    
    def test_jsonl_log_parsing(self):
        """Test that JSONL logs can be parsed correctly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = BenchmarkLogger("hellaswag", Path(temp_dir))
            
            logger.start_benchmark(["/test/path.jsonl"])
            logger.log_prediction("test_sample", "option_a", confidence=0.85)
            logger.log_sample_processed("test_sample", processing_time_ms=125.5)
            logger.log_accuracy_update(1.0, correct_predictions=1, total_predictions=1)
            logger.finish_benchmark(1.0)
            
            # Read and parse all JSONL entries
            jsonl_file = Path(temp_dir) / "benchmark.hellaswag.jsonl"
            entries = []
            
            for line in jsonl_file.read_text().strip().split('\n'):
                if line.strip():
                    entry = json.loads(line)
                    entries.append(entry)
            
            # Verify we have all expected event types
            event_types = [entry["event_type"] for entry in entries]
            expected_events = [
                "benchmark_start", 
                "prediction_made", 
                "sample_processed", 
                "accuracy_update", 
                "cross_check_passed",
                "benchmark_complete"
            ]
            
            for expected in expected_events:
                assert expected in event_types
            
            # Verify structured data
            start_entry = next(e for e in entries if e["event_type"] == "benchmark_start")
            assert "dataset_paths" in start_entry["data"]
            
            prediction_entry = next(e for e in entries if e["event_type"] == "prediction_made")
            assert prediction_entry["data"]["confidence"] == 0.85
            assert prediction_entry["data"]["predictions_count"] == 1
            
            complete_entry = next(e for e in entries if e["event_type"] == "benchmark_complete")
            assert complete_entry["data"]["final_accuracy"] == 1.0
            assert complete_entry["data"]["cross_check_passed"] is True


class TestJSONLHandler:
    """Test JSONL handler functionality"""
    
    def test_jsonl_handler_basic(self):
        """Test basic JSONL handler functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            import logging
            
            # Create logger with JSONL handler
            logger = logging.getLogger("test_jsonl")
            handler = JSONLHandler(Path(temp_dir) / "test.jsonl")
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            
            # Log some messages
            logger.info("Simple message")  # Standard log record
            logger.info('{"event": "structured", "data": {"key": "value"}}')  # JSON string
            
            handler.close()
            
            # Verify file content
            jsonl_file = Path(temp_dir) / "test.jsonl"
            assert jsonl_file.exists()
            
            lines = jsonl_file.read_text().strip().split('\n')
            assert len(lines) == 2
            
            # First line should be standard log
            entry1 = json.loads(lines[0])
            assert entry1["message"] == "Simple message"
            assert entry1["level"] == "INFO"
            
            # Second line should be structured
            entry2 = json.loads(lines[1])
            assert entry2["event"] == "structured"
            assert entry2["data"]["key"] == "value"


if __name__ == "__main__":
    # Run basic functionality test
    test_logger = TestBenchmarkLogger()
    test_logger.test_benchmark_logger_lifecycle()
    print("✓ Benchmark logging lifecycle test passed")
    
    test_logger.test_cross_check_mismatch_detection()
    print("✓ Cross-check mismatch detection test passed")
    
    test_logger.test_jsonl_log_parsing()
    print("✓ JSONL log parsing test passed")
    
    print("\n✓ All structured logging tests passed!")
