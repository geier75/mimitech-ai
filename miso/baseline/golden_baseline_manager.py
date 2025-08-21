#!/usr/bin/env python3
"""
Golden Baseline Manager - Phase 11
Manages golden baseline storage, retrieval, and validation
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from miso.validation.schema_validator import SchemaValidator

@dataclass
class BaselineMetrics:
    """Baseline metrics for a specific benchmark"""
    benchmark_name: str
    accuracy: float
    samples_processed: int
    duration_s: float
    throughput_samples_per_sec: float
    
    # Tolerance thresholds
    accuracy_tolerance: float = 0.02  # ±2%
    duration_tolerance_factor: float = 1.5  # 50% slower allowed
    sample_count_tolerance: int = 0  # Exact match required
    
    def __post_init__(self):
        """Validate metrics on initialization"""
        if not 0 <= self.accuracy <= 1.0:
            raise ValueError(f"Invalid accuracy: {self.accuracy}")
        if self.samples_processed <= 0:
            raise ValueError(f"Invalid sample count: {self.samples_processed}")
        if self.duration_s <= 0:
            raise ValueError(f"Invalid duration: {self.duration_s}")

class GoldenBaselineManager:
    """
    Manages golden baseline creation, storage, and validation
    Phase 11: Golden Baseline & Drift-Detection
    """
    
    def __init__(self, baseline_dir: Path = None):
        self.baseline_dir = baseline_dir or Path("baseline")
        self.baseline_dir.mkdir(exist_ok=True)
        
        # Standard baseline structure
        self.reports_dir = self.baseline_dir / "reports"
        self.logs_dir = self.baseline_dir / "logs"  
        self.datasets_dir = self.baseline_dir / "datasets"
        self.metadata_dir = self.baseline_dir / "metadata"
        
        # Create subdirectories
        for dir_path in [self.reports_dir, self.logs_dir, self.datasets_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
            
        self.validator = SchemaValidator()
    
    def create_golden_baseline(self, 
                             benchmark_report: Dict[str, Any], 
                             log_files: List[Path] = None,
                             dataset_manifests: List[Path] = None,
                             baseline_name: str = "default") -> Path:
        """
        Create a new golden baseline from benchmark results
        
        Args:
            benchmark_report: Validated benchmark report
            log_files: Optional JSONL log files to include  
            dataset_manifests: Optional dataset manifests to include
            baseline_name: Name for this baseline version
            
        Returns:
            Path to created baseline directory
        """
        
        # Validate the report first
        validation_result = self.validator.validate_benchmark_report(benchmark_report)
        if not validation_result.is_valid:
            raise ValueError(f"Invalid benchmark report: {validation_result.errors}")
        
        # Create baseline version directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        baseline_version = f"{baseline_name}_{timestamp}"
        version_dir = self.baseline_dir / baseline_version
        version_dir.mkdir(exist_ok=True)
        
        # Save benchmark report
        report_path = version_dir / "benchmark_report.json"
        with open(report_path, 'w') as f:
            json.dump(benchmark_report, f, indent=2)
        
        # Extract and save baseline metrics
        metrics = self._extract_baseline_metrics(benchmark_report)
        metrics_path = version_dir / "baseline_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({name: asdict(metric) for name, metric in metrics.items()}, f, indent=2)
        
        # Copy log files if provided
        if log_files:
            logs_version_dir = version_dir / "logs"
            logs_version_dir.mkdir(exist_ok=True)
            for log_file in log_files:
                if log_file.exists():
                    shutil.copy2(log_file, logs_version_dir / log_file.name)
        
        # Copy dataset manifests if provided
        if dataset_manifests:
            datasets_version_dir = version_dir / "datasets"
            datasets_version_dir.mkdir(exist_ok=True)
            for manifest in dataset_manifests:
                if manifest.exists():
                    shutil.copy2(manifest, datasets_version_dir / manifest.name)
        
        # Create baseline metadata
        metadata = {
            "baseline_name": baseline_name,
            "created_at": datetime.now().isoformat(),
            "git_commit": benchmark_report.get("reproducibility", {}).get("git_commit"),
            "schema_version": benchmark_report.get("schema_version"),
            "total_benchmarks": benchmark_report.get("summary", {}).get("total_benchmarks"),
            "compute_mode": benchmark_report.get("metrics", {}).get("compute_mode"),
            "baseline_version": baseline_version,
            "files": {
                "report": "benchmark_report.json",
                "metrics": "baseline_metrics.json",
                "logs": "logs/" if log_files else None,
                "datasets": "datasets/" if dataset_manifests else None
            }
        }
        
        metadata_path = version_dir / "baseline_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update current baseline symlink
        current_link = self.baseline_dir / "current"
        if current_link.exists() or current_link.is_symlink():
            current_link.unlink()
        current_link.symlink_to(baseline_version)
        
        print(f"✅ Golden baseline created: {version_dir}")
        print(f"📊 Baseline metrics extracted for {len(metrics)} benchmarks")
        
        return version_dir
    
    def get_current_baseline(self) -> Optional[Dict[str, Any]]:
        """Get the current golden baseline"""
        current_link = self.baseline_dir / "current"
        if not current_link.exists():
            return None
        
        baseline_dir = current_link.resolve()
        report_path = baseline_dir / "benchmark_report.json"
        
        if not report_path.exists():
            return None
            
        with open(report_path) as f:
            return json.load(f)
    
    def get_baseline_metrics(self, baseline_name: str = "current") -> Dict[str, BaselineMetrics]:
        """Get baseline metrics for drift comparison"""
        if baseline_name == "current":
            current_link = self.baseline_dir / "current"
            if not current_link.exists():
                raise FileNotFoundError("No current baseline found")
            baseline_dir = current_link.resolve()
        else:
            baseline_dir = self.baseline_dir / baseline_name
            
        metrics_path = baseline_dir / "baseline_metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Baseline metrics not found: {metrics_path}")
        
        with open(metrics_path) as f:
            metrics_data = json.load(f)
        
        return {
            name: BaselineMetrics(**data) 
            for name, data in metrics_data.items()
        }
    
    def list_baselines(self) -> List[Dict[str, Any]]:
        """List all available baselines with metadata"""
        baselines = []
        
        for baseline_dir in self.baseline_dir.iterdir():
            if baseline_dir.is_dir() and baseline_dir.name != "current":
                metadata_path = baseline_dir / "baseline_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    metadata["path"] = str(baseline_dir)
                    baselines.append(metadata)
        
        # Sort by creation date (newest first)
        baselines.sort(key=lambda x: x["created_at"], reverse=True)
        return baselines
    
    def _extract_baseline_metrics(self, benchmark_report: Dict[str, Any]) -> Dict[str, BaselineMetrics]:
        """Extract baseline metrics from benchmark report"""
        metrics = {}
        
        results = benchmark_report.get("results", [])
        for result in results:
            if result.get("status") == "PASS":
                benchmark_name = result["benchmark_name"]
                
                # Set tolerance based on benchmark type
                accuracy_tolerance = self._get_accuracy_tolerance(benchmark_name)
                duration_tolerance = self._get_duration_tolerance(benchmark_name)
                
                metrics[benchmark_name] = BaselineMetrics(
                    benchmark_name=benchmark_name,
                    accuracy=result["accuracy"],
                    samples_processed=result["samples_processed"],
                    duration_s=result["duration_s"],
                    throughput_samples_per_sec=result.get("throughput_samples_per_sec", 0),
                    accuracy_tolerance=accuracy_tolerance,
                    duration_tolerance_factor=duration_tolerance
                )
        
        return metrics
    
    def _get_accuracy_tolerance(self, benchmark_name: str) -> float:
        """Get accuracy tolerance by benchmark type"""
        # More strict tolerances for easier benchmarks
        strict_benchmarks = ["piqa", "hellaswag", "winogrande"]
        if any(name in benchmark_name.lower() for name in strict_benchmarks):
            return 0.01  # ±1%
        
        # Standard tolerance for MMLU, ARC, etc.
        return 0.02  # ±2%
    
    def _get_duration_tolerance(self, benchmark_name: str) -> float:
        """Get duration tolerance factor by benchmark type"""
        # Larger datasets can have more variable timing
        large_benchmarks = ["mmlu"]
        if any(name in benchmark_name.lower() for name in large_benchmarks):
            return 2.0  # 100% slower allowed
        
        # Standard tolerance
        return 1.5  # 50% slower allowed
