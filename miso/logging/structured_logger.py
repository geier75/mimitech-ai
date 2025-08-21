"""
Structured logging with per-benchmark namespaces and JSONL output
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict

@dataclass
class LogEntry:
    """Structured log entry for JSONL output"""
    timestamp: str
    level: str
    benchmark_name: str
    event_type: str
    message: str
    data: Dict[str, Any]
    execution_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

class StructuredLogger:
    """Base structured logger with JSONL output"""
    
    def __init__(self, name: str, log_dir: Path = None, execution_id: str = None):
        self.name = name
        self.execution_id = execution_id or f"exec_{int(time.time())}"
        
        if log_dir is None:
            project_root = Path(__file__).parent.parent.parent
            log_dir = project_root / "logs"
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup standard logger
        self.logger = logging.getLogger(f"miso.{name}")
        self.logger.setLevel(logging.DEBUG)
        
        # Setup JSONL handler
        from .jsonl_handler import JSONLHandler
        self.jsonl_handler = JSONLHandler(self.log_dir / f"{name}.jsonl")
        self.logger.addHandler(self.jsonl_handler)
        
        # Prevent duplicate logs in parent loggers
        self.logger.propagate = False
        
    def _create_entry(self, level: str, event_type: str, message: str, **kwargs) -> LogEntry:
        """Create structured log entry"""
        return LogEntry(
            timestamp=datetime.now().isoformat() + "Z",
            level=level.upper(),
            benchmark_name=self.name,
            event_type=event_type,
            message=message,
            data=kwargs,
            execution_id=self.execution_id
        )
    
    def info(self, event_type: str, message: str, **kwargs):
        """Log info-level structured message"""
        entry = self._create_entry("INFO", event_type, message, **kwargs)
        self.logger.info(json.dumps(entry.to_dict()))
        
    def warning(self, event_type: str, message: str, **kwargs):
        """Log warning-level structured message"""
        entry = self._create_entry("WARNING", event_type, message, **kwargs)
        self.logger.warning(json.dumps(entry.to_dict()))
        
    def error(self, event_type: str, message: str, **kwargs):
        """Log error-level structured message"""
        entry = self._create_entry("ERROR", event_type, message, **kwargs)
        self.logger.error(json.dumps(entry.to_dict()))
        
    def debug(self, event_type: str, message: str, **kwargs):
        """Log debug-level structured message"""
        entry = self._create_entry("DEBUG", event_type, message, **kwargs)
        self.logger.debug(json.dumps(entry.to_dict()))

class BenchmarkLogger(StructuredLogger):
    """Specialized logger for benchmark operations with cross-checks"""
    
    def __init__(self, benchmark_name: str, log_dir: Path = None, execution_id: str = None):
        super().__init__(f"benchmark.{benchmark_name}", log_dir, execution_id)
        self.benchmark_name = benchmark_name
        
        # Cross-check tracking
        self._predictions_count = 0
        self._samples_processed = 0
        self._start_time = None
        self._metrics = {}
        
    def start_benchmark(self, dataset_paths: list, expected_samples: int = None):
        """Log benchmark start with initialization"""
        self._start_time = time.time()
        self._predictions_count = 0
        self._samples_processed = 0
        
        self.info("benchmark_start", 
                 f"Starting benchmark: {self.benchmark_name}",
                 dataset_paths=dataset_paths,
                 expected_samples=expected_samples,
                 start_time=self._start_time)
    
    def log_prediction(self, sample_id: str, prediction: Any, confidence: float = None):
        """Log individual prediction with tracking"""
        self._predictions_count += 1
        
        self.debug("prediction_made",
                  f"Prediction for sample {sample_id}",
                  sample_id=sample_id,
                  prediction=str(prediction),
                  confidence=confidence,
                  predictions_count=self._predictions_count)
    
    def log_sample_processed(self, sample_id: str, processing_time_ms: float = None):
        """Log sample processing with tracking"""
        self._samples_processed += 1
        
        self.debug("sample_processed",
                  f"Processed sample {sample_id}",
                  sample_id=sample_id,
                  processing_time_ms=processing_time_ms,
                  samples_processed=self._samples_processed)
    
    def log_accuracy_update(self, current_accuracy: float, correct_predictions: int, total_predictions: int):
        """Log accuracy updates with validation"""
        self._metrics["accuracy"] = current_accuracy
        self._metrics["correct_predictions"] = correct_predictions
        self._metrics["total_predictions"] = total_predictions
        
        # Cross-check: total_predictions should match predictions_count
        if total_predictions != self._predictions_count:
            self.warning("accuracy_inconsistency",
                        f"Total predictions mismatch: metric={total_predictions}, tracked={self._predictions_count}",
                        metric_total=total_predictions,
                        tracked_count=self._predictions_count,
                        difference=abs(total_predictions - self._predictions_count))
        
        self.info("accuracy_update",
                 f"Accuracy: {current_accuracy:.2%}",
                 accuracy=current_accuracy,
                 correct_predictions=correct_predictions,
                 total_predictions=total_predictions)
    
    def finish_benchmark(self, final_accuracy: float, duration_s: float = None) -> Dict[str, Any]:
        """Log benchmark completion with cross-checks"""
        if duration_s is None and self._start_time:
            duration_s = time.time() - self._start_time
        
        # Cross-check: predictions_count == samples_processed
        cross_check_passed = self._predictions_count == self._samples_processed
        
        if not cross_check_passed:
            self.error("cross_check_failed",
                      f"Cross-check FAILED: predictions_count != samples_processed",
                      predictions_count=self._predictions_count,
                      samples_processed=self._samples_processed,
                      difference=abs(self._predictions_count - self._samples_processed))
        else:
            self.info("cross_check_passed",
                     f"Cross-check PASSED: predictions_count == samples_processed",
                     predictions_count=self._predictions_count,
                     samples_processed=self._samples_processed)
        
        summary = {
            "benchmark_name": self.benchmark_name,
            "final_accuracy": final_accuracy,
            "duration_s": duration_s,
            "predictions_count": self._predictions_count,
            "samples_processed": self._samples_processed,
            "cross_check_passed": cross_check_passed,
            "execution_id": self.execution_id
        }
        
        self.info("benchmark_complete",
                 f"Benchmark completed: {self.benchmark_name}",
                 **summary)
        
        return summary
    
    def get_cross_check_status(self) -> Dict[str, Any]:
        """Get current cross-check status"""
        return {
            "predictions_count": self._predictions_count,
            "samples_processed": self._samples_processed,
            "match": self._predictions_count == self._samples_processed,
            "difference": abs(self._predictions_count - self._samples_processed)
        }
