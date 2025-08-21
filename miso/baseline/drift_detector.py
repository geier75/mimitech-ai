#!/usr/bin/env python3
"""
Drift Detector - Phase 11
Detects and reports performance drift against golden baseline
"""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from .golden_baseline_manager import GoldenBaselineManager, BaselineMetrics

class DriftSeverity(Enum):
    """Severity levels for drift detection"""
    INFO = "info"
    WARNING = "warning" 
    CRITICAL = "critical"

@dataclass
class DriftAlert:
    """Individual drift alert for a specific metric"""
    benchmark_name: str
    metric_name: str
    baseline_value: float
    current_value: float
    drift_percent: float
    tolerance: float
    severity: DriftSeverity
    message: str

@dataclass
class DriftReport:
    """Comprehensive drift detection report"""
    timestamp: str
    baseline_version: str
    current_git_commit: str
    total_benchmarks: int
    drifted_benchmarks: int
    alerts: List[DriftAlert]
    overall_status: str  # "PASS", "WARNING", "CRITICAL"
    
    def has_critical_drift(self) -> bool:
        """Check if any critical drift was detected"""
        return any(alert.severity == DriftSeverity.CRITICAL for alert in self.alerts)
    
    def has_warnings(self) -> bool:
        """Check if any warnings were generated"""
        return any(alert.severity == DriftSeverity.WARNING for alert in self.alerts)

class DriftDetector:
    """
    Detects performance drift against golden baseline
    Phase 11: Golden Baseline & Drift-Detection
    """
    
    def __init__(self, baseline_manager: GoldenBaselineManager = None):
        self.baseline_manager = baseline_manager or GoldenBaselineManager()
    
    def detect_drift(self, current_report: Dict[str, Any]) -> DriftReport:
        """
        Detect drift between current results and golden baseline
        
        Args:
            current_report: Current benchmark report to compare
            
        Returns:
            DriftReport with detected drift and alerts
        """
        
        # Get baseline metrics
        try:
            baseline_metrics = self.baseline_manager.get_baseline_metrics()
        except FileNotFoundError:
            raise RuntimeError("No golden baseline found. Create baseline first.")
        
        # Extract current metrics
        current_metrics = self._extract_current_metrics(current_report)
        
        # Perform drift detection
        alerts = []
        drifted_benchmarks = set()
        
        for benchmark_name, baseline in baseline_metrics.items():
            if benchmark_name in current_metrics:
                current = current_metrics[benchmark_name]
                benchmark_alerts = self._detect_benchmark_drift(baseline, current)
                alerts.extend(benchmark_alerts)
                
                if benchmark_alerts:
                    drifted_benchmarks.add(benchmark_name)
            else:
                # Missing benchmark is critical
                alerts.append(DriftAlert(
                    benchmark_name=benchmark_name,
                    metric_name="presence",
                    baseline_value=1.0,
                    current_value=0.0,
                    drift_percent=100.0,
                    tolerance=0.0,
                    severity=DriftSeverity.CRITICAL,
                    message=f"Benchmark {benchmark_name} missing from current results"
                ))
                drifted_benchmarks.add(benchmark_name)
        
        # Check for new benchmarks not in baseline
        for benchmark_name in current_metrics:
            if benchmark_name not in baseline_metrics:
                alerts.append(DriftAlert(
                    benchmark_name=benchmark_name,
                    metric_name="presence", 
                    baseline_value=0.0,
                    current_value=1.0,
                    drift_percent=100.0,
                    tolerance=0.0,
                    severity=DriftSeverity.WARNING,
                    message=f"New benchmark {benchmark_name} not in baseline"
                ))
                drifted_benchmarks.add(benchmark_name)
        
        # Determine overall status
        overall_status = self._determine_overall_status(alerts)
        
        # Create drift report
        return DriftReport(
            timestamp=datetime.now().isoformat(),
            baseline_version=self._get_baseline_version(),
            current_git_commit=current_report.get("reproducibility", {}).get("git_commit", "unknown"),
            total_benchmarks=len(baseline_metrics),
            drifted_benchmarks=len(drifted_benchmarks),
            alerts=sorted(alerts, key=lambda x: x.severity.value, reverse=True),
            overall_status=overall_status
        )
    
    def _extract_current_metrics(self, report: Dict[str, Any]) -> Dict[str, BaselineMetrics]:
        """Extract metrics from current report for comparison"""
        metrics = {}
        
        results = report.get("results", [])
        for result in results:
            if result.get("status") == "PASS":
                benchmark_name = result["benchmark_name"]
                metrics[benchmark_name] = BaselineMetrics(
                    benchmark_name=benchmark_name,
                    accuracy=result["accuracy"],
                    samples_processed=result["samples_processed"],
                    duration_s=result["duration_s"],
                    throughput_samples_per_sec=result.get("throughput_samples_per_sec", 0)
                )
        
        return metrics
    
    def _detect_benchmark_drift(self, baseline: BaselineMetrics, current: BaselineMetrics) -> List[DriftAlert]:
        """Detect drift for a specific benchmark"""
        alerts = []
        
        # Check accuracy drift
        accuracy_drift = abs(current.accuracy - baseline.accuracy)
        if accuracy_drift > baseline.accuracy_tolerance:
            severity = DriftSeverity.CRITICAL if accuracy_drift > baseline.accuracy_tolerance * 2 else DriftSeverity.WARNING
            drift_percent = (accuracy_drift / baseline.accuracy) * 100 if baseline.accuracy > 0 else 100
            
            alerts.append(DriftAlert(
                benchmark_name=baseline.benchmark_name,
                metric_name="accuracy",
                baseline_value=baseline.accuracy,
                current_value=current.accuracy,
                drift_percent=drift_percent,
                tolerance=baseline.accuracy_tolerance,
                severity=severity,
                message=f"Accuracy drift: {current.accuracy:.3f} vs baseline {baseline.accuracy:.3f} (±{baseline.accuracy_tolerance:.3f})"
            ))
        
        # Check sample count drift (must be exact)
        if current.samples_processed != baseline.samples_processed:
            alerts.append(DriftAlert(
                benchmark_name=baseline.benchmark_name,
                metric_name="samples_processed",
                baseline_value=baseline.samples_processed,
                current_value=current.samples_processed,
                drift_percent=abs(current.samples_processed - baseline.samples_processed) / baseline.samples_processed * 100,
                tolerance=0.0,
                severity=DriftSeverity.CRITICAL,
                message=f"Sample count mismatch: {current.samples_processed} vs baseline {baseline.samples_processed}"
            ))
        
        # Check duration drift (performance regression)
        duration_ratio = current.duration_s / baseline.duration_s if baseline.duration_s > 0 else float('inf')
        if duration_ratio > baseline.duration_tolerance_factor:
            severity = DriftSeverity.CRITICAL if duration_ratio > baseline.duration_tolerance_factor * 1.5 else DriftSeverity.WARNING
            drift_percent = (duration_ratio - 1) * 100
            
            alerts.append(DriftAlert(
                benchmark_name=baseline.benchmark_name,
                metric_name="duration_s",
                baseline_value=baseline.duration_s,
                current_value=current.duration_s,
                drift_percent=drift_percent,
                tolerance=baseline.duration_tolerance_factor,
                severity=severity,
                message=f"Performance regression: {current.duration_s:.2f}s vs baseline {baseline.duration_s:.2f}s ({duration_ratio:.1f}x slower)"
            ))
        
        # Check throughput for significant drops
        if baseline.throughput_samples_per_sec > 0 and current.throughput_samples_per_sec > 0:
            throughput_ratio = current.throughput_samples_per_sec / baseline.throughput_samples_per_sec
            if throughput_ratio < 0.7:  # 30% throughput drop
                severity = DriftSeverity.CRITICAL if throughput_ratio < 0.5 else DriftSeverity.WARNING
                drift_percent = (1 - throughput_ratio) * 100
                
                alerts.append(DriftAlert(
                    benchmark_name=baseline.benchmark_name,
                    metric_name="throughput_samples_per_sec",
                    baseline_value=baseline.throughput_samples_per_sec,
                    current_value=current.throughput_samples_per_sec,
                    drift_percent=drift_percent,
                    tolerance=0.3,  # 30% tolerance
                    severity=severity,
                    message=f"Throughput drop: {current.throughput_samples_per_sec:.2f} vs baseline {baseline.throughput_samples_per_sec:.2f} samples/sec"
                ))
        
        return alerts
    
    def _determine_overall_status(self, alerts: List[DriftAlert]) -> str:
        """Determine overall drift status from alerts"""
        if any(alert.severity == DriftSeverity.CRITICAL for alert in alerts):
            return "CRITICAL"
        elif any(alert.severity == DriftSeverity.WARNING for alert in alerts):
            return "WARNING"
        else:
            return "PASS"
    
    def _get_baseline_version(self) -> str:
        """Get current baseline version identifier"""
        try:
            current_baseline = self.baseline_manager.get_current_baseline()
            if current_baseline:
                return current_baseline.get("reproducibility", {}).get("git_commit", "unknown")
        except:
            pass
        return "unknown"
    
    def save_drift_report(self, report: DriftReport, output_path: Path = None) -> Path:
        """Save drift report to file"""
        if output_path is None:
            output_path = Path(f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Convert report to JSON-serializable format
        report_data = asdict(report)
        
        # Convert enums to strings
        for alert in report_data["alerts"]:
            alert["severity"] = alert["severity"].value if hasattr(alert["severity"], "value") else alert["severity"]
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return output_path
    
    def print_drift_summary(self, report: DriftReport):
        """Print human-readable drift report summary"""
        print(f"\n📊 Drift Detection Report - {report.timestamp}")
        print("=" * 60)
        print(f"Baseline: {report.baseline_version}")
        print(f"Current:  {report.current_git_commit}")
        print(f"Status:   {report.overall_status}")
        print(f"Drifted:  {report.drifted_benchmarks}/{report.total_benchmarks} benchmarks")
        
        if report.alerts:
            print(f"\n🚨 Alerts ({len(report.alerts)}):")
            
            # Group by severity
            critical_alerts = [a for a in report.alerts if a.severity == DriftSeverity.CRITICAL]
            warning_alerts = [a for a in report.alerts if a.severity == DriftSeverity.WARNING]
            
            if critical_alerts:
                print(f"\n❌ CRITICAL ({len(critical_alerts)}):")
                for alert in critical_alerts:
                    print(f"   {alert.benchmark_name}.{alert.metric_name}: {alert.message}")
            
            if warning_alerts:
                print(f"\n⚠️  WARNING ({len(warning_alerts)}):")
                for alert in warning_alerts:
                    print(f"   {alert.benchmark_name}.{alert.metric_name}: {alert.message}")
        else:
            print("\n✅ No drift detected - all metrics within tolerance")
        
        print("=" * 60)
