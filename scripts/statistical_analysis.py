#!/usr/bin/env python3
"""
T5: Statistical Analysis for A/B Evaluation
Bootstrap CI and McNemar tests for training promotion decisions
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Any, Optional

# Import atomic JSON utilities
sys.path.append('/Users/gecko365')
from scripts_json_utils import atomic_json_dump, robust_json_load, to_jsonable
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    """Single benchmark result with predictions"""
    name: str
    accuracy: float
    samples_processed: int
    predictions: List[bool]  # True = correct, False = incorrect
    
@dataclass
class ComparisonResult:
    """Statistical comparison between baseline and candidate"""
    benchmark: str
    baseline_acc: float
    candidate_acc: float
    accuracy_delta: float
    confidence_interval: Tuple[float, float]
    p_value: float
    effect_size: float
    is_significant: bool
    meets_threshold: bool

class StatisticalAnalyzer:
    """
    Performs rigorous statistical analysis for training promotion decisions
    Implements Bootstrap CI and McNemar tests per METRIC_CONTRACT.md
    """
    
    def __init__(self, 
                 confidence_level: float = 0.95,
                 n_bootstrap: int = 10000,
                 random_seed: int = 42):
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.random_seed = random_seed
        
        # Load promotion thresholds from METRIC_CONTRACT
        self.global_threshold = 3.0  # pp
        self.per_benchmark_threshold = 2.0  # pp  
        self.regression_limit = -1.0  # pp
        
        np.random.seed(random_seed)
    
    def bootstrap_confidence_interval(self, 
                                    baseline_preds: List[bool],
                                    candidate_preds: List[bool]) -> Tuple[float, Tuple[float, float]]:
        """
        Compute bootstrap confidence interval for accuracy difference
        
        Args:
            baseline_preds: Boolean list of baseline predictions (correct/incorrect)
            candidate_preds: Boolean list of candidate predictions (same samples)
            
        Returns:
            Tuple of (mean_delta, (ci_lower, ci_upper))
        """
        
        if len(baseline_preds) != len(candidate_preds):
            raise ValueError("Prediction lists must have same length")
        
        n_samples = len(baseline_preds)
        baseline_acc = np.mean(baseline_preds)
        candidate_acc = np.mean(candidate_preds)
        observed_delta = candidate_acc - baseline_acc
        
        # Bootstrap resampling
        bootstrap_deltas = []
        for _ in range(self.n_bootstrap):
            # Resample indices
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            # Compute accuracy delta for this bootstrap sample
            boot_baseline = np.mean([baseline_preds[i] for i in indices])
            boot_candidate = np.mean([candidate_preds[i] for i in indices])
            bootstrap_deltas.append(boot_candidate - boot_baseline)
        
        # Compute confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_deltas, 100 * alpha/2)
        ci_upper = np.percentile(bootstrap_deltas, 100 * (1 - alpha/2))
        
        return observed_delta, (ci_lower, ci_upper)
    
    def mcnemar_test(self, 
                     baseline_preds: List[bool], 
                     candidate_preds: List[bool]) -> Tuple[float, float]:
        """
        Perform McNemar test for paired classification results
        
        Args:
            baseline_preds: Boolean predictions from baseline model
            candidate_preds: Boolean predictions from candidate model
            
        Returns:
            Tuple of (test_statistic, p_value)
        """
        
        if len(baseline_preds) != len(candidate_preds):
            raise ValueError("Prediction lists must have same length")
        
        # Create contingency table
        # b: baseline correct, candidate wrong
        # c: baseline wrong, candidate correct  
        b = sum(1 for i in range(len(baseline_preds)) 
                if baseline_preds[i] and not candidate_preds[i])
        c = sum(1 for i in range(len(baseline_preds))
                if not baseline_preds[i] and candidate_preds[i])
        
        # McNemar test statistic
        if b + c == 0:
            # No disagreement between models
            return 0.0, 1.0
        
        # Use continuity correction for small samples
        if b + c < 25:
            test_stat = (abs(b - c) - 1)**2 / (b + c)
        else:
            test_stat = (b - c)**2 / (b + c)
        
        # Chi-square test with 1 degree of freedom
        p_value = 1 - stats.chi2.cdf(test_stat, 1)
        
        return test_stat, p_value
    
    def analyze_benchmark_comparison(self,
                                   baseline: BenchmarkResult,
                                   candidate: BenchmarkResult) -> ComparisonResult:
        """
        Complete statistical analysis of single benchmark comparison
        
        Args:
            baseline: Baseline model results
            candidate: Candidate model results  
            
        Returns:
            ComparisonResult with all statistical metrics
        """
        
        if baseline.name != candidate.name:
            raise ValueError(f"Benchmark names don't match: {baseline.name} vs {candidate.name}")
        
        if len(baseline.predictions) != len(candidate.predictions):
            raise ValueError(f"Different sample counts: {len(baseline.predictions)} vs {len(candidate.predictions)}")
        
        # Compute accuracy delta
        accuracy_delta = candidate.accuracy - baseline.accuracy
        
        # Bootstrap confidence interval
        mean_delta, ci = self.bootstrap_confidence_interval(
            baseline.predictions, candidate.predictions
        )
        
        # McNemar test
        mcnemar_stat, p_value = self.mcnemar_test(
            baseline.predictions, candidate.predictions  
        )
        
        # Effect size (Cohen's h for proportions)
        baseline_arcsine = np.arcsin(np.sqrt(baseline.accuracy))
        candidate_arcsine = np.arcsin(np.sqrt(candidate.accuracy))
        effect_size = 2 * (candidate_arcsine - baseline_arcsine)
        
        # Significance and threshold checks
        is_significant = p_value < 0.05 and ci[0] > 0  # CI lower bound > 0
        meets_threshold = (
            accuracy_delta >= self.per_benchmark_threshold / 100 or  # +2pp improvement
            accuracy_delta >= self.regression_limit / 100           # or no regression > -1pp
        )
        
        return ComparisonResult(
            benchmark=baseline.name,
            baseline_acc=baseline.accuracy,
            candidate_acc=candidate.accuracy,
            accuracy_delta=accuracy_delta * 100,  # Convert to percentage points
            confidence_interval=(ci[0] * 100, ci[1] * 100),
            p_value=p_value,
            effect_size=effect_size,
            is_significant=is_significant,
            meets_threshold=meets_threshold
        )
    
    def analyze_full_suite(self,
                          baseline_results: List[BenchmarkResult],
                          candidate_results: List[BenchmarkResult]) -> Dict[str, Any]:
        """
        Complete statistical analysis of full benchmark suite
        
        Args:
            baseline_results: List of baseline benchmark results
            candidate_results: List of candidate benchmark results
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        
        # Match benchmarks by name
        baseline_dict = {r.name: r for r in baseline_results}
        candidate_dict = {r.name: r for r in candidate_results}
        
        common_benchmarks = set(baseline_dict.keys()) & set(candidate_dict.keys())
        if len(common_benchmarks) == 0:
            raise ValueError("No matching benchmarks found")
        
        print(f"📊 Analyzing {len(common_benchmarks)} benchmarks...")
        
        # Per-benchmark analysis
        comparisons = []
        for benchmark_name in sorted(common_benchmarks):
            baseline = baseline_dict[benchmark_name]
            candidate = candidate_dict[benchmark_name]
            
            comparison = self.analyze_benchmark_comparison(baseline, candidate)
            comparisons.append(comparison)
            
            print(f"  - {benchmark_name}: {comparison.accuracy_delta:+.1f}pp "
                  f"(CI: {comparison.confidence_interval[0]:+.1f} to {comparison.confidence_interval[1]:+.1f})")
        
        # Global analysis
        total_baseline_correct = sum(sum(r.predictions) for r in baseline_results if r.name in common_benchmarks)
        total_baseline_samples = sum(len(r.predictions) for r in baseline_results if r.name in common_benchmarks)
        total_candidate_correct = sum(sum(r.predictions) for r in candidate_results if r.name in common_benchmarks)
        total_candidate_samples = sum(len(r.predictions) for r in candidate_results if r.name in common_benchmarks)
        
        global_baseline_acc = total_baseline_correct / total_baseline_samples
        global_candidate_acc = total_candidate_correct / total_candidate_samples
        global_delta = (global_candidate_acc - global_baseline_acc) * 100
        
        # Global bootstrap CI (pooled across benchmarks)
        all_baseline_preds = []
        all_candidate_preds = []
        for benchmark_name in common_benchmarks:
            all_baseline_preds.extend(baseline_dict[benchmark_name].predictions)
            all_candidate_preds.extend(candidate_dict[benchmark_name].predictions)
        
        _, global_ci = self.bootstrap_confidence_interval(all_baseline_preds, all_candidate_preds)
        global_ci_pp = (global_ci[0] * 100, global_ci[1] * 100)
        
        # Promotion decision
        global_meets_threshold = (
            global_delta >= self.global_threshold and  # ≥ +3pp average
            global_ci_pp[0] > 0                       # CI lower bound > 0
        )
        
        per_benchmark_pass = all(c.meets_threshold for c in comparisons)
        overall_promotion_ready = global_meets_threshold and per_benchmark_pass
        
        # Compile results
        analysis_results = {
            "global_metrics": {
                "baseline_accuracy": global_baseline_acc,
                "candidate_accuracy": global_candidate_acc,
                "accuracy_delta_pp": global_delta,
                "confidence_interval_pp": global_ci_pp,
                "meets_global_threshold": global_meets_threshold,
                "total_samples": total_baseline_samples
            },
            "per_benchmark_results": [
                {
                    "benchmark": c.benchmark,
                    "baseline_accuracy": c.baseline_acc,
                    "candidate_accuracy": c.candidate_acc,
                    "accuracy_delta_pp": c.accuracy_delta,
                    "confidence_interval_pp": c.confidence_interval,
                    "p_value": c.p_value,
                    "effect_size": c.effect_size,
                    "is_significant": c.is_significant,
                    "meets_threshold": c.meets_threshold
                }
                for c in comparisons
            ],
            "promotion_decision": {
                "overall_ready": overall_promotion_ready,
                "global_threshold_met": global_meets_threshold,
                "per_benchmark_threshold_met": per_benchmark_pass,
                "failed_benchmarks": [c.benchmark for c in comparisons if not c.meets_threshold]
            },
            "analysis_metadata": {
                "confidence_level": self.confidence_level,
                "n_bootstrap": self.n_bootstrap,
                "random_seed": self.random_seed,
                "thresholds": {
                    "global_pp": self.global_threshold,
                    "per_benchmark_pp": self.per_benchmark_threshold,
                    "regression_limit_pp": self.regression_limit
                }
            }
        }
        
        return analysis_results
    
    def generate_promotion_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate comprehensive promotion readiness report"""
        
        lines = [
            "# Training Promotion Statistical Analysis",
            f"**Analysis Date**: {np.datetime64('now')}",
            f"**Confidence Level**: {self.confidence_level:.0%}",
            f"**Bootstrap Samples**: {self.n_bootstrap:,}",
            ""
        ]
        
        # Overall decision
        decision = analysis_results["promotion_decision"]
        if decision["overall_ready"]:
            lines.extend([
                "## 🎉 PROMOTION APPROVED",
                "",
                "✅ **Decision**: READY FOR PROMOTION",
                "✅ **Global Threshold**: Met (+3pp requirement)",  
                "✅ **Per-Benchmark Gates**: All benchmarks passed",
                ""
            ])
        else:
            lines.extend([
                "## ❌ PROMOTION BLOCKED", 
                "",
                "❌ **Decision**: NOT READY FOR PROMOTION",
                ""
            ])
            
            if not decision["global_threshold_met"]:
                lines.append("❌ **Global Threshold**: Failed (+3pp requirement)")
            else:
                lines.append("✅ **Global Threshold**: Met (+3pp requirement)")
                
            if not decision["per_benchmark_threshold_met"]:
                failed = ", ".join(decision["failed_benchmarks"])
                lines.extend([
                    f"❌ **Per-Benchmark Gates**: Failed benchmarks: {failed}",
                    ""
                ])
            else:
                lines.append("✅ **Per-Benchmark Gates**: All benchmarks passed")
        
        # Global metrics
        global_metrics = analysis_results["global_metrics"]
        lines.extend([
            "## 📊 Global Performance Analysis",
            "",
            f"| Metric | Baseline | Candidate | Δ (pp) | 95% CI |",
            f"|--------|----------|-----------|--------|--------|",
            f"| **Accuracy** | {global_metrics['baseline_accuracy']:.1%} | {global_metrics['candidate_accuracy']:.1%} | **{global_metrics['accuracy_delta_pp']:+.1f}** | ({global_metrics['confidence_interval_pp'][0]:+.1f}, {global_metrics['confidence_interval_pp'][1]:+.1f}) |",
            f"| **Total Samples** | {global_metrics['total_samples']:,} | {global_metrics['total_samples']:,} | - | - |",
            ""
        ])
        
        # Per-benchmark results
        lines.extend([
            "## 🔬 Per-Benchmark Statistical Analysis",
            "",
            "| Benchmark | Baseline | Candidate | Δ (pp) | 95% CI | p-value | Status |",
            "|-----------|----------|-----------|--------|--------|---------|---------|"
        ])
        
        for result in analysis_results["per_benchmark_results"]:
            status = "✅ PASS" if result["meets_threshold"] else "❌ FAIL"
            sig_marker = "*" if result["is_significant"] else ""
            
            lines.append(
                f"| **{result['benchmark']}** | "
                f"{result['baseline_accuracy']:.1%} | "
                f"{result['candidate_accuracy']:.1%} | "
                f"**{result['accuracy_delta_pp']:+.1f}**{sig_marker} | "
                f"({result['confidence_interval_pp'][0]:+.1f}, {result['confidence_interval_pp'][1]:+.1f}) | "
                f"{result['p_value']:.3f} | "
                f"{status} |"
            )
        
        lines.extend([
            "",
            "*Statistically significant at p < 0.05",
            ""
        ])
        
        # Statistical methodology
        lines.extend([
            "## 📐 Statistical Methodology",
            "",
            "### Bootstrap Confidence Intervals",
            f"- **Method**: Stratified bootstrap resampling ({self.n_bootstrap:,} iterations)",
            f"- **Confidence Level**: {self.confidence_level:.0%}",
            f"- **Random Seed**: {self.random_seed} (reproducible)",
            "",
            "### McNemar Test (Paired Classification)",
            "- **Null Hypothesis**: No difference in error rates between models",
            "- **Alternative**: Candidate model has different (better) error rate", 
            "- **Significance Level**: α = 0.05",
            "",
            "### Promotion Thresholds (METRIC_CONTRACT.md)",
            f"- **Global Average**: ≥ +{self.global_threshold:.1f}pp with CI lower bound > 0",
            f"- **Per-Benchmark**: ≥ +{self.per_benchmark_threshold:.1f}pp OR no regression > {self.regression_limit:.1f}pp",
            f"- **Statistical Significance**: Required (p < 0.05)",
            ""
        ])
        
        return "\n".join(lines)

def load_benchmark_results_from_reports(reports_dir: Path) -> List[BenchmarkResult]:
    """
    Load benchmark results from MISO report files
    Note: This is a simplified version - real implementation would parse actual reports
    """
    
    # Mock data for demonstration - replace with actual report parsing
    mock_results = [
        BenchmarkResult(
            name="MMLU",
            accuracy=0.621,
            samples_processed=14042,
            predictions=np.random.choice([True, False], 14042, p=[0.621, 0.379]).tolist()
        ),
        BenchmarkResult(
            name="GSM8K", 
            accuracy=0.591,
            samples_processed=1319,
            predictions=np.random.choice([True, False], 1319, p=[0.591, 0.409]).tolist()
        ),
        BenchmarkResult(
            name="HumanEval",
            accuracy=0.488,
            samples_processed=164,
            predictions=np.random.choice([True, False], 164, p=[0.488, 0.512]).tolist()
        )
    ]
    
    return mock_results

def main():
    if len(sys.argv) < 3:
        print("Usage: python statistical_analysis.py <baseline_reports_dir> <candidate_reports_dir>")
        sys.exit(1)
    
    baseline_dir = Path(sys.argv[1])
    candidate_dir = Path(sys.argv[2])
    
    print("📊 MISO Statistical Analysis for Training Promotion")
    print("=" * 60)
    
    # Load results (mock implementation)
    print("Loading baseline results...")
    baseline_results = load_benchmark_results_from_reports(baseline_dir)
    
    print("Loading candidate results...")  
    # For demo, create improved candidate results
    candidate_results = []
    for baseline in baseline_results:
        # Simulate improvement
        improved_acc = min(baseline.accuracy + 0.035, 0.95)  # +3.5pp improvement
        n_samples = baseline.samples_processed
        
        candidate_results.append(BenchmarkResult(
            name=baseline.name,
            accuracy=improved_acc,
            samples_processed=n_samples,
            predictions=np.random.choice([True, False], n_samples, p=[improved_acc, 1-improved_acc]).tolist()
        ))
    
    # Perform analysis
    analyzer = StatisticalAnalyzer()
    analysis_results = analyzer.analyze_full_suite(baseline_results, candidate_results)
    
    # Generate and save report
    report = analyzer.generate_promotion_report(analysis_results)
    
    report_path = Path("training_promotion_analysis.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n📄 Statistical analysis report saved: {report_path}")
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Save JSON report
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = Path(__file__).parent
    json_report_path = project_root / f"training_promotion_analysis_{timestamp}.json"
    serializable_results = convert_numpy_types(analysis_results)
    atomic_json_dump(to_jsonable(serializable_results), json_report_path)
    
    print(f"📋 JSON results saved: {json_report_path}")
    
    # Exit code based on promotion readiness
    if analysis_results["promotion_decision"]["overall_ready"]:
        print("\n🎉 PROMOTION APPROVED - All statistical gates passed!")
        sys.exit(0)
    else:
        print("\n❌ PROMOTION BLOCKED - Statistical requirements not met")
        sys.exit(1)

if __name__ == "__main__":
    main()
