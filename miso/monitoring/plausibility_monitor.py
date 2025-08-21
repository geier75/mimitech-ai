"""
Plausibility monitoring for benchmark results
Detects outliers in duration, throughput, and accuracy metrics
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics

logger = logging.getLogger(__name__)

class ComputeMode(Enum):
    """Compute modes for benchmark execution"""
    FULL = "full"      # Complete benchmark with all samples
    LIGHT = "light"    # Reduced sample set for faster execution  
    STUB = "stub"      # Minimal execution for testing/CI

@dataclass
class PlausibilityWindow:
    """Plausibility thresholds for benchmark metrics"""
    min_duration_s: float
    max_duration_s: float
    min_throughput_samples_per_sec: float
    max_throughput_samples_per_sec: float
    min_accuracy: float
    max_accuracy: float
    
@dataclass
class PlausibilityResult:
    """Result of plausibility check"""
    is_plausible: bool
    warnings: List[str]
    outliers: List[str]
    details: Dict[str, Any]

class PlausibilityMonitor:
    """Monitor benchmark results for plausible ranges and outliers"""
    
    # Default plausibility windows per benchmark and compute mode
    PLAUSIBILITY_WINDOWS = {
        "mmlu": {
            ComputeMode.FULL: PlausibilityWindow(
                min_duration_s=300.0, max_duration_s=3600.0,
                min_throughput_samples_per_sec=5.0, max_throughput_samples_per_sec=50.0,
                min_accuracy=0.20, max_accuracy=0.95
            ),
            ComputeMode.LIGHT: PlausibilityWindow(
                min_duration_s=30.0, max_duration_s=600.0,
                min_throughput_samples_per_sec=10.0, max_throughput_samples_per_sec=100.0,
                min_accuracy=0.15, max_accuracy=0.95
            ),
            ComputeMode.STUB: PlausibilityWindow(
                min_duration_s=1.0, max_duration_s=60.0,
                min_throughput_samples_per_sec=5.0, max_throughput_samples_per_sec=200.0,
                min_accuracy=0.10, max_accuracy=1.00
            )
        },
        "hellaswag": {
            ComputeMode.FULL: PlausibilityWindow(
                min_duration_s=60.0, max_duration_s=900.0,
                min_throughput_samples_per_sec=1.0, max_throughput_samples_per_sec=20.0,
                min_accuracy=0.25, max_accuracy=0.90
            ),
            ComputeMode.LIGHT: PlausibilityWindow(
                min_duration_s=10.0, max_duration_s=180.0,
                min_throughput_samples_per_sec=2.0, max_throughput_samples_per_sec=40.0,
                min_accuracy=0.20, max_accuracy=0.90
            ),
            ComputeMode.STUB: PlausibilityWindow(
                min_duration_s=1.0, max_duration_s=30.0,
                min_throughput_samples_per_sec=1.0, max_throughput_samples_per_sec=100.0,
                min_accuracy=0.10, max_accuracy=1.00
            )
        },
        # Default windows for other benchmarks
        "_default": {
            ComputeMode.FULL: PlausibilityWindow(
                min_duration_s=30.0, max_duration_s=1800.0,
                min_throughput_samples_per_sec=1.0, max_throughput_samples_per_sec=100.0,
                min_accuracy=0.15, max_accuracy=0.95
            ),
            ComputeMode.LIGHT: PlausibilityWindow(
                min_duration_s=5.0, max_duration_s=300.0,
                min_throughput_samples_per_sec=2.0, max_throughput_samples_per_sec=200.0,
                min_accuracy=0.10, max_accuracy=0.95
            ),
            ComputeMode.STUB: PlausibilityWindow(
                min_duration_s=0.5, max_duration_s=60.0,
                min_throughput_samples_per_sec=1.0, max_throughput_samples_per_sec=500.0,
                min_accuracy=0.05, max_accuracy=1.00
            )
        }
    }
    
    def __init__(self):
        self.historical_results = {}  # Store for outlier detection
        
    def validate_compute_mode(self, compute_mode: str) -> bool:
        """Validate that compute_mode is one of the allowed values"""
        try:
            ComputeMode(compute_mode)
            return True
        except ValueError:
            return False
    
    def enforce_compute_mode(self, metrics: Dict[str, Any]) -> None:
        """Enforce that compute_mode is present and valid"""
        if "compute_mode" not in metrics:
            raise ValueError("compute_mode is required in metrics but was not provided")
        
        if not self.validate_compute_mode(metrics["compute_mode"]):
            valid_modes = [mode.value for mode in ComputeMode]
            raise ValueError(f"compute_mode '{metrics['compute_mode']}' is invalid. Must be one of: {valid_modes}")
    
    def get_plausibility_window(self, benchmark_name: str, compute_mode: ComputeMode) -> PlausibilityWindow:
        """Get plausibility window for benchmark and compute mode"""
        benchmark_windows = self.PLAUSIBILITY_WINDOWS.get(benchmark_name)
        if not benchmark_windows:
            benchmark_windows = self.PLAUSIBILITY_WINDOWS["_default"]
        
        return benchmark_windows.get(compute_mode, benchmark_windows[ComputeMode.STUB])
    
    def check_plausibility(self, 
                          benchmark_name: str,
                          duration_s: float,
                          samples_processed: int,
                          accuracy: float,
                          compute_mode: str) -> PlausibilityResult:
        """Check if benchmark results are within plausible ranges"""
        
        # Enforce compute_mode requirement
        if not compute_mode:
            return PlausibilityResult(
                is_plausible=False,
                warnings=[],
                outliers=["compute_mode is required but not provided"],
                details={"error": "missing_compute_mode"}
            )
        
        try:
            mode_enum = ComputeMode(compute_mode)
        except ValueError:
            valid_modes = [mode.value for mode in ComputeMode]
            return PlausibilityResult(
                is_plausible=False,
                warnings=[],
                outliers=[f"compute_mode '{compute_mode}' invalid. Must be one of: {valid_modes}"],
                details={"error": "invalid_compute_mode"}
            )
        
        window = self.get_plausibility_window(benchmark_name, mode_enum)
        warnings = []
        outliers = []
        details = {
            "window": {
                "duration_range": [window.min_duration_s, window.max_duration_s],
                "throughput_range": [window.min_throughput_samples_per_sec, window.max_throughput_samples_per_sec],
                "accuracy_range": [window.min_accuracy, window.max_accuracy]
            },
            "actual": {
                "duration_s": duration_s,
                "samples_processed": samples_processed,
                "accuracy": accuracy,
                "compute_mode": compute_mode
            }
        }
        
        # Calculate throughput
        throughput = samples_processed / duration_s if duration_s > 0 else 0
        details["actual"]["throughput_samples_per_sec"] = throughput
        
        # Check duration
        if duration_s < window.min_duration_s:
            outliers.append(f"Duration {duration_s:.2f}s below minimum {window.min_duration_s}s")
        elif duration_s > window.max_duration_s:
            outliers.append(f"Duration {duration_s:.2f}s exceeds maximum {window.max_duration_s}s")
        elif duration_s < window.min_duration_s * 1.5:
            warnings.append(f"Duration {duration_s:.2f}s is unusually low")
        elif duration_s > window.max_duration_s * 0.8:
            warnings.append(f"Duration {duration_s:.2f}s is approaching upper limit")
        
        # Check throughput
        if throughput < window.min_throughput_samples_per_sec:
            outliers.append(f"Throughput {throughput:.2f} samples/s below minimum {window.min_throughput_samples_per_sec}")
        elif throughput > window.max_throughput_samples_per_sec:
            outliers.append(f"Throughput {throughput:.2f} samples/s exceeds maximum {window.max_throughput_samples_per_sec}")
        elif throughput < window.min_throughput_samples_per_sec * 1.2:
            warnings.append(f"Throughput {throughput:.2f} samples/s is unusually low")
        elif throughput > window.max_throughput_samples_per_sec * 0.9:
            warnings.append(f"Throughput {throughput:.2f} samples/s is unusually high")
        
        # Check accuracy
        if accuracy < window.min_accuracy:
            outliers.append(f"Accuracy {accuracy:.2%} below minimum {window.min_accuracy:.2%}")
        elif accuracy > window.max_accuracy:
            outliers.append(f"Accuracy {accuracy:.2%} exceeds maximum {window.max_accuracy:.2%}")
        elif accuracy < window.min_accuracy * 1.1:
            warnings.append(f"Accuracy {accuracy:.2%} is unusually low")
        elif accuracy > window.max_accuracy * 0.95:
            warnings.append(f"Accuracy {accuracy:.2%} is unusually high")
        
        is_plausible = len(outliers) == 0
        
        return PlausibilityResult(
            is_plausible=is_plausible,
            warnings=warnings,
            outliers=outliers,
            details=details
        )
    
    def detect_statistical_outliers(self, 
                                   benchmark_name: str,
                                   current_result: Dict[str, Any],
                                   historical_threshold: int = 5) -> List[str]:
        """Detect outliers based on historical results using statistical methods"""
        if benchmark_name not in self.historical_results:
            self.historical_results[benchmark_name] = []
        
        history = self.historical_results[benchmark_name]
        
        # Need at least historical_threshold results for meaningful statistics
        if len(history) < historical_threshold:
            history.append(current_result)
            return []
        
        outliers = []
        
        # Check duration outliers (using z-score)
        durations = [r["duration_s"] for r in history]
        mean_duration = statistics.mean(durations)
        stdev_duration = statistics.stdev(durations) if len(durations) > 1 else 0
        
        if stdev_duration > 0:
            z_score = abs(current_result["duration_s"] - mean_duration) / stdev_duration
            if z_score > 2.5:  # More than 2.5 standard deviations
                outliers.append(f"Duration {current_result['duration_s']:.2f}s is statistical outlier (z-score: {z_score:.2f})")
        
        # Check accuracy outliers
        accuracies = [r["accuracy"] for r in history]
        mean_accuracy = statistics.mean(accuracies)
        stdev_accuracy = statistics.stdev(accuracies) if len(accuracies) > 1 else 0
        
        if stdev_accuracy > 0:
            z_score = abs(current_result["accuracy"] - mean_accuracy) / stdev_accuracy
            if z_score > 2.0:  # More than 2 standard deviations for accuracy
                outliers.append(f"Accuracy {current_result['accuracy']:.2%} is statistical outlier (z-score: {z_score:.2f})")
        
        # Add current result to history
        history.append(current_result)
        
        # Keep only recent results
        if len(history) > 20:
            history.pop(0)
        
        return outliers
    
    def generate_monitoring_report(self, results: List[PlausibilityResult]) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        total_results = len(results)
        plausible_count = sum(1 for r in results if r.is_plausible)
        warning_count = sum(len(r.warnings) for r in results)
        outlier_count = sum(len(r.outliers) for r in results)
        
        return {
            "summary": {
                "total_benchmarks": total_results,
                "plausible_results": plausible_count,
                "plausibility_rate": plausible_count / total_results if total_results > 0 else 0,
                "total_warnings": warning_count,
                "total_outliers": outlier_count
            },
            "details": [
                {
                    "is_plausible": r.is_plausible,
                    "warnings": r.warnings,
                    "outliers": r.outliers,
                    "metrics": r.details
                }
                for r in results
            ],
            "quality_gate": plausible_count == total_results and outlier_count == 0
        }
