#!/usr/bin/env python3
"""
📈 ALGORITHMIC OPTIMIZATION (Optional)
Dual-pass evaluation system for algorithm_* problems to improve Pass@1 rate
"""

import json
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# Add benchmarks to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

@dataclass
class DualPassResult:
    """Results from dual-pass evaluation"""
    task_id: str
    entry_point: str
    first_pass_success: bool
    second_pass_success: bool
    first_pass_code: str
    second_pass_code: str
    recovery_achieved: bool
    execution_time_first: float
    execution_time_second: float
    error_first_pass: str = ""
    error_second_pass: str = ""

class AlgorithmicOptimizer:
    """Dual-pass optimization system for algorithmic problems"""
    
    def __init__(self):
        self.version = "v1.0.0"
        self.results: List[DualPassResult] = []
        
    def isolate_algorithmic_problems(self) -> List[Dict]:
        """Isolate all algorithm_* cases in own subset"""
        print("🔹 ISOLATE: algorithm_* cases → algorithmic subset")
        print("=" * 60)
        
        # Create test data
        from create_real_111_problems import create_real_111_problems
        temp_dir = create_real_111_problems()
        
        try:
            # Load problems
            problems = []
            dataset_path = temp_dir / "HumanEval.jsonl"
            
            with open(dataset_path, 'r') as f:
                for line in f:
                    if line.strip():
                        problem = json.loads(line.strip())
                        problems.append(problem)
            
            # Filter algorithmic problems
            algorithmic_problems = []
            from benchmarks.humaneval_benchmark import HumanEvalSolver
            solver = HumanEvalSolver()
            
            for problem in problems:
                problem_type = solver._classify_problem_type(problem['prompt'])
                if problem_type == "algorithmic" or problem['entry_point'].startswith('algorithm_'):
                    algorithmic_problems.append(problem)
            
            print(f"   ✅ Total problems: {len(problems)}")
            print(f"   ✅ Algorithmic problems: {len(algorithmic_problems)}")
            print(f"   📊 Algorithmic ratio: {len(algorithmic_problems)/len(problems)*100:.1f}%")
            
            # Show sample algorithmic problems
            print(f"\n   📋 Sample algorithmic problems:")
            for i, problem in enumerate(algorithmic_problems[:5]):
                print(f"      {i+1}. {problem['task_id']} - {problem['entry_point']}")
            
            return algorithmic_problems
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def build_dual_pass_test_suite(self, algorithmic_problems: List[Dict]):
        """Build test suite for dual-pass evaluation"""
        print(f"\n🔹 BUILD: Dual-Pass Test Suite")
        print("=" * 60)
        
        print(f"   🧪 Test Suite Configuration:")
        print(f"      First-Pass: Baseline Model (current implementation)")
        print(f"      Second-Pass: Retry with Enhanced Prompt")
        print(f"      Logging: Error capture + Recovery tracking")
        print(f"      Analysis: Delta calculation + Net gain measurement")
        
        # Test each algorithmic problem with dual-pass
        for i, problem in enumerate(algorithmic_problems):
            print(f"\n   🧪 Dual-Pass Test {i+1}/{len(algorithmic_problems)}: {problem['entry_point']}")
            
            result = self._execute_dual_pass_test(problem)
            self.results.append(result)
            
            # Show immediate result
            status_first = "✅" if result.first_pass_success else "❌"
            status_second = "✅" if result.second_pass_success else "❌"
            recovery = "🔄" if result.recovery_achieved else "  "
            
            print(f"      First-Pass: {status_first} ({result.execution_time_first:.3f}s)")
            print(f"      Second-Pass: {status_second} ({result.execution_time_second:.3f}s)")
            print(f"      Recovery: {recovery} {'YES' if result.recovery_achieved else 'NO'}")
    
    def _execute_dual_pass_test(self, problem: Dict) -> DualPassResult:
        """Execute dual-pass test for single problem"""
        from benchmarks.humaneval_benchmark import HumanEvalSolver, HumanEvalProblem, HumanEvalEvaluator
        
        # Create problem object
        humaneval_problem = HumanEvalProblem(
            task_id=problem['task_id'],
            prompt=problem['prompt'],
            canonical_solution=problem['canonical_solution'],
            test=problem['test'],
            entry_point=problem['entry_point']
        )
        
        solver = HumanEvalSolver()
        
        # First Pass - Baseline
        start_time = time.time()
        try:
            first_code, _ = solver.solve_problem(humaneval_problem)
            first_time = time.time() - start_time
            
            # Test first pass
            evaluator = HumanEvalEvaluator(None, None)
            first_success = evaluator._test_generated_code(humaneval_problem, first_code)
            first_error = ""
            
        except Exception as e:
            first_time = time.time() - start_time
            first_code = ""
            first_success = False
            first_error = str(e)
        
        # Second Pass - Enhanced (only if first failed)
        start_time = time.time()
        if not first_success:
            try:
                # Enhanced prompt for retry
                enhanced_problem = self._create_enhanced_prompt(humaneval_problem)
                second_code, _ = solver.solve_problem(enhanced_problem)
                second_time = time.time() - start_time
                
                # Test second pass
                second_success = evaluator._test_generated_code(humaneval_problem, second_code)
                second_error = ""
                
            except Exception as e:
                second_time = time.time() - start_time
                second_code = ""
                second_success = False
                second_error = str(e)
        else:
            # Skip second pass if first succeeded
            second_code = first_code
            second_success = first_success
            second_time = 0.0
            second_error = ""
        
        # Determine recovery
        recovery_achieved = not first_success and second_success
        
        return DualPassResult(
            task_id=problem['task_id'],
            entry_point=problem['entry_point'],
            first_pass_success=first_success,
            second_pass_success=second_success,
            first_pass_code=first_code[:100] + "..." if len(first_code) > 100 else first_code,
            second_pass_code=second_code[:100] + "..." if len(second_code) > 100 else second_code,
            recovery_achieved=recovery_achieved,
            execution_time_first=first_time,
            execution_time_second=second_time,
            error_first_pass=first_error,
            error_second_pass=second_error
        )
    
    def _create_enhanced_prompt(self, problem) -> object:
        """Create enhanced prompt for second pass"""
        # Enhanced prompt with more specific guidance
        enhanced_prompt = problem.prompt.replace(
            '"""',
            '"""\n    ENHANCED: Focus on robust algorithmic implementation.\n    Consider edge cases and ensure proper return types.\n    """'
        )
        
        # Create new problem object with enhanced prompt
        from benchmarks.humaneval_benchmark import HumanEvalProblem
        return HumanEvalProblem(
            task_id=problem.task_id,
            prompt=enhanced_prompt,
            canonical_solution=problem.canonical_solution,
            test=problem.test,
            entry_point=problem.entry_point
        )
    
    def analyze_dual_pass_results(self) -> Dict[str, Any]:
        """Analyze dual-pass results"""
        print(f"\n🔹 ANALYZE: Dual-Pass Results")
        print("=" * 60)
        
        if not self.results:
            print("   ⚠️ No results to analyze")
            return {}
        
        # Calculate metrics
        total_problems = len(self.results)
        first_pass_successes = sum(1 for r in self.results if r.first_pass_success)
        second_pass_successes = sum(1 for r in self.results if r.second_pass_success)
        recoveries = sum(1 for r in self.results if r.recovery_achieved)
        
        # Calculate rates
        first_pass_rate = (first_pass_successes / total_problems) * 100
        second_pass_rate = (second_pass_successes / total_problems) * 100
        recovery_rate = (recoveries / (total_problems - first_pass_successes)) * 100 if total_problems > first_pass_successes else 0
        net_gain = second_pass_rate - first_pass_rate
        
        # Calculate timing
        avg_first_time = sum(r.execution_time_first for r in self.results) / total_problems
        avg_second_time = sum(r.execution_time_second for r in self.results if r.execution_time_second > 0)
        avg_second_time = avg_second_time / sum(1 for r in self.results if r.execution_time_second > 0) if avg_second_time else 0
        
        analysis = {
            "total_problems": total_problems,
            "first_pass": {
                "successes": first_pass_successes,
                "rate": first_pass_rate,
                "avg_time": avg_first_time
            },
            "second_pass": {
                "successes": second_pass_successes,
                "rate": second_pass_rate,
                "avg_time": avg_second_time
            },
            "recovery": {
                "count": recoveries,
                "rate": recovery_rate,
                "net_gain": net_gain
            },
            "performance_impact": {
                "single_pass_rate": first_pass_rate,
                "dual_pass_rate": second_pass_rate,
                "improvement": net_gain,
                "cost_benefit": net_gain / (avg_first_time + avg_second_time) if (avg_first_time + avg_second_time) > 0 else 0
            }
        }
        
        # Display analysis
        print(f"   📊 DUAL-PASS ANALYSIS RESULTS:")
        print(f"      Total Problems: {total_problems}")
        print(f"      First-Pass Rate: {first_pass_rate:.1f}% ({first_pass_successes}/{total_problems})")
        print(f"      Second-Pass Rate: {second_pass_rate:.1f}% ({second_pass_successes}/{total_problems})")
        print(f"      Recovery Rate: {recovery_rate:.1f}% ({recoveries} recoveries)")
        print(f"      Net Gain: {net_gain:+.1f}%")
        print(f"      Avg First-Pass Time: {avg_first_time:.3f}s")
        print(f"      Avg Second-Pass Time: {avg_second_time:.3f}s")
        
        # Recommendations
        print(f"\n   🎯 RECOMMENDATIONS:")
        if net_gain > 5:
            print(f"      ✅ IMPLEMENT: Dual-pass shows significant improvement (+{net_gain:.1f}%)")
        elif net_gain > 0:
            print(f"      ⚖️ CONSIDER: Dual-pass shows modest improvement (+{net_gain:.1f}%)")
        else:
            print(f"      ❌ SKIP: Dual-pass shows no improvement ({net_gain:+.1f}%)")
        
        return analysis
    
    def export_optimization_report(self, analysis: Dict[str, Any]):
        """Export optimization report"""
        print(f"\n🔹 EXPORT: Optimization Report")
        print("=" * 60)
        
        report = {
            "optimization_info": {
                "version": self.version,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "optimization_type": "DUAL_PASS_ALGORITHMIC"
            },
            "analysis": analysis,
            "detailed_results": [
                {
                    "task_id": r.task_id,
                    "entry_point": r.entry_point,
                    "first_pass_success": r.first_pass_success,
                    "second_pass_success": r.second_pass_success,
                    "recovery_achieved": r.recovery_achieved,
                    "execution_time_first": r.execution_time_first,
                    "execution_time_second": r.execution_time_second,
                    "error_first_pass": r.error_first_pass,
                    "error_second_pass": r.error_second_pass
                }
                for r in self.results
            ]
        }
        
        report_path = f"algorithmic_optimization_{self.version}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"   ✅ Report exported: {report_path}")
        print(f"   📊 Size: {Path(report_path).stat().st_size} bytes")
        
        return report_path

def main():
    """Main algorithmic optimization execution"""
    print("📈 ALGORITHMIC OPTIMIZATION (Optional)")
    print("=" * 80)
    print("🎯 Dual-pass evaluation system for algorithm_* problems")
    print()
    
    optimizer = AlgorithmicOptimizer()
    
    # Execute optimization steps
    algorithmic_problems = optimizer.isolate_algorithmic_problems()
    
    if algorithmic_problems:
        optimizer.build_dual_pass_test_suite(algorithmic_problems)
        analysis = optimizer.analyze_dual_pass_results()
        report_path = optimizer.export_optimization_report(analysis)
        
        # Summary
        print(f"\n" + "=" * 80)
        print("🎉 ALGORITHMIC OPTIMIZATION COMPLETED")
        print("=" * 80)
        print(f"📊 Problems Analyzed: {len(algorithmic_problems)}")
        print(f"📈 Net Gain: {analysis.get('recovery', {}).get('net_gain', 0):+.1f}%")
        print(f"📄 Report: {report_path}")
        
        if analysis.get('recovery', {}).get('net_gain', 0) > 5:
            print("✅ RECOMMENDATION: Implement dual-pass for significant improvement")
        else:
            print("ℹ️ RECOMMENDATION: Current single-pass performance is sufficient")
    else:
        print("⚠️ No algorithmic problems found for optimization")

if __name__ == "__main__":
    main()
