"""
Comprehensive Test Suite for HumanEval Benchmark
Enterprise-Grade Testing with Security Validation and Determinism Checks

This test suite validates the HumanEval benchmark implementation for:
- Security measures and sandboxed execution
- Deterministic evaluation consistency
- Core functionality with mock data
- Integration testing with sample problems
"""

import pytest
import tempfile
import json
import time
import hashlib
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the parent directory to the path to import the benchmark
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.humaneval_benchmark import (
    HumanEvalProblem, HumanEvalResult, HumanEvalDataLoader, 
    HumanEvalSolver, HumanEvalEvaluator, create_humaneval_evaluator
)

class TestHumanEvalSolver:
    """Test suite for HumanEvalSolver class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.solver = HumanEvalSolver()
    
    def test_classify_problem_type_string_manipulation(self):
        """Test problem type classification for string manipulation"""
        prompt = "def reverse_string(s: str) -> str: return the reversed string"
        result = self.solver._classify_problem_type(prompt)
        assert result == "string_manipulation"
    
    def test_classify_problem_type_mathematical(self):
        """Test problem type classification for mathematical problems"""
        prompt = "def factorial(n: int) -> int: return the factorial of n"
        result = self.solver._classify_problem_type(prompt)
        assert result == "mathematical"
    
    def test_classify_problem_type_list_operations(self):
        """Test problem type classification for list operations"""
        prompt = "def sort_list(lst: List[int]) -> List[int]: sort the list"
        result = self.solver._classify_problem_type(prompt)
        assert result == "list_operations"
    
    def test_classify_problem_type_algorithmic(self):
        """Test problem type classification for algorithmic problems"""
        prompt = "def binary_search(arr: List[int], target: int) -> int: find target"
        result = self.solver._classify_problem_type(prompt)
        assert result == "algorithmic"
    
    def test_identify_required_operations_iteration(self):
        """Test identification of iteration operations"""
        prompt = "iterate through the list and process each element"
        result = self.solver._identify_required_operations(prompt)
        assert "iteration" in result
    
    def test_identify_required_operations_conditional(self):
        """Test identification of conditional operations"""
        prompt = "check if the condition is met and validate the input"
        result = self.solver._identify_required_operations(prompt)
        assert "conditional" in result
    
    def test_identify_required_operations_multiple(self):
        """Test identification of multiple operations"""
        prompt = "loop through items, check conditions, and return result"
        result = self.solver._identify_required_operations(prompt)
        assert "iteration" in result
        assert "conditional" in result
        assert "return_value" in result
    
    def test_solve_problem_deterministic(self):
        """Test that solve_problem produces deterministic results"""
        problem = HumanEvalProblem(
            task_id="test/0",
            prompt="def add_numbers(a: int, b: int) -> int:\n    \"\"\"Add two numbers\"\"\"",
            canonical_solution="    return a + b",
            test="assert add_numbers(2, 3) == 5",
            entry_point="add_numbers"
        )
        
        # Run multiple times to check determinism
        results = []
        for _ in range(5):
            code, trace = self.solver.solve_problem(problem)
            results.append((code, trace["problem_type"], trace["required_operations"]))
        
        # All results should be identical (deterministic)
        first_result = results[0]
        for result in results[1:]:
            assert result[0] == first_result[0]  # Same generated code
            assert result[1] == first_result[1]  # Same problem type
            assert result[2] == first_result[2]  # Same operations

class TestHumanEvalDataLoader:
    """Test suite for HumanEvalDataLoader class"""
    
    def setup_method(self):
        """Set up test fixtures with temporary data"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.data_loader = HumanEvalDataLoader(self.temp_dir)
    
    def teardown_method(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_problems_with_valid_data(self):
        """Test loading problems with valid JSONL data"""
        # Create mock data file
        test_data = [
            {
                "task_id": "HumanEval/0",
                "prompt": "def test_function():\n    pass",
                "canonical_solution": "    return True",
                "test": "assert test_function() == True",
                "entry_point": "test_function"
            },
            {
                "task_id": "HumanEval/1", 
                "prompt": "def another_function():\n    pass",
                "canonical_solution": "    return False",
                "test": "assert another_function() == False",
                "entry_point": "another_function"
            }
        ]
        
        # Write test data to file
        data_file = self.temp_dir / "HumanEval.jsonl"
        with open(data_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        # Test loading
        problems = self.data_loader.load_problems(sample_size=2)
        
        assert len(problems) == 2
        assert problems[0].task_id == "HumanEval/0"
        assert problems[1].task_id == "HumanEval/1"
        assert problems[0].entry_point == "test_function"
    
    def test_load_problems_file_not_found(self):
        """Test error handling when data file is not found"""
        with pytest.raises(FileNotFoundError):
            self.data_loader.load_problems()
    
    def test_load_problems_sample_size_limit(self):
        """Test that sample_size parameter limits the number of problems loaded"""
        # Create mock data with 5 problems
        test_data = []
        for i in range(5):
            test_data.append({
                "task_id": f"HumanEval/{i}",
                "prompt": f"def func_{i}(): pass",
                "canonical_solution": "    return True",
                "test": "assert True",
                "entry_point": f"func_{i}"
            })
        
        data_file = self.temp_dir / "HumanEval.jsonl"
        with open(data_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        # Load only 3 problems
        problems = self.data_loader.load_problems(sample_size=3)
        assert len(problems) == 3

class TestHumanEvalEvaluator:
    """Test suite for HumanEvalEvaluator class with security validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.evaluator = HumanEvalEvaluator(self.temp_dir)
        
        # Create minimal test data
        self.create_test_data()
    
    def teardown_method(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_data(self):
        """Create minimal test data for evaluation"""
        test_problems = [
            {
                "task_id": "HumanEval/0",
                "prompt": "def add_two(a: int, b: int) -> int:\n    \"\"\"Add two numbers\"\"\"",
                "canonical_solution": "    return a + b",
                "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(0, 0) == 0\n\ncheck(add_two)",
                "entry_point": "add_two"
            },
            {
                "task_id": "HumanEval/1",
                "prompt": "def is_even(n: int) -> bool:\n    \"\"\"Check if number is even\"\"\"",
                "canonical_solution": "    return n % 2 == 0",
                "test": "def check(candidate):\n    assert candidate(2) == True\n    assert candidate(3) == False\n\ncheck(is_even)",
                "entry_point": "is_even"
            }
        ]
        
        data_file = self.temp_dir / "HumanEval.jsonl"
        with open(data_file, 'w') as f:
            for problem in test_problems:
                f.write(json.dumps(problem) + '\n')
    
    def test_secure_code_execution_timeout(self):
        """Test that code execution has timeout protection"""
        # Create a problem with infinite loop
        malicious_problem = HumanEvalProblem(
            task_id="test/malicious",
            prompt="def infinite_loop():",
            canonical_solution="    while True: pass",
            test="infinite_loop()",
            entry_point="infinite_loop"
        )
        
        start_time = time.time()
        result = self.evaluator._test_generated_code(malicious_problem, "def infinite_loop():\n    while True: pass")
        execution_time = time.time() - start_time
        
        # Should timeout and return False within reasonable time (< 15 seconds)
        assert result == False
        assert execution_time < 15
    
    def test_secure_code_execution_restricted_imports(self):
        """Test that dangerous imports are restricted"""
        dangerous_problem = HumanEvalProblem(
            task_id="test/dangerous",
            prompt="def dangerous_function():",
            canonical_solution="    import os; os.system('rm -rf /')",
            test="dangerous_function()",
            entry_point="dangerous_function"
        )
        
        # Should fail due to restricted imports
        result = self.evaluator._test_generated_code(
            dangerous_problem, 
            "def dangerous_function():\n    import os\n    os.system('echo test')\n    return True"
        )
        assert result == False
    
    def test_evaluate_with_mock_data(self):
        """Test full evaluation with mock data"""
        result = self.evaluator.evaluate(sample_size=2)
        
        # Validate result structure
        assert isinstance(result, HumanEvalResult)
        assert result.total_problems == 2
        assert result.passed_problems >= 0
        assert 0 <= result.pass_at_1 <= 100
        assert result.execution_time > 0
        assert len(result.solver_trace) == 2
        assert isinstance(result.problem_categories, dict)
        assert isinstance(result.category_pass_rates, dict)
        assert isinstance(result.execution_stats, dict)
        assert isinstance(result.data_provenance, dict)
    
    def test_deterministic_evaluation(self):
        """Test that evaluation produces deterministic results"""
        # Run evaluation multiple times
        results = []
        for _ in range(3):
            result = self.evaluator.evaluate(sample_size=2)
            results.append({
                'total_problems': result.total_problems,
                'passed_problems': result.passed_problems,
                'pass_at_1': result.pass_at_1,
                'problem_categories': result.problem_categories,
                'data_hash': result.data_provenance['data_hash']
            })
        
        # All results should be identical (deterministic)
        first_result = results[0]
        for result in results[1:]:
            assert result['total_problems'] == first_result['total_problems']
            assert result['passed_problems'] == first_result['passed_problems']
            assert result['pass_at_1'] == first_result['pass_at_1']
            assert result['problem_categories'] == first_result['problem_categories']
            assert result['data_hash'] == first_result['data_hash']
    
    def test_data_provenance_calculation(self):
        """Test data provenance hash calculation"""
        problems = self.evaluator.data_loader.load_problems(sample_size=2)
        provenance = self.evaluator._calculate_data_provenance(problems)
        
        assert 'data_hash' in provenance
        assert 'problem_count' in provenance
        assert 'hash_algorithm' in provenance
        assert 'data_source' in provenance
        assert provenance['problem_count'] == 2
        assert provenance['hash_algorithm'] == 'SHA256'
        assert len(provenance['data_hash']) == 64  # SHA256 hex length

class TestHumanEvalResult:
    """Test suite for HumanEvalResult export functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create sample result
        self.result = HumanEvalResult(
            total_problems=10,
            passed_problems=7,
            pass_at_1=70.0,
            execution_time=5.5,
            solver_trace=[{"test": "trace"}],
            problem_categories={"mathematical": {"correct": 3, "total": 4}},
            category_pass_rates={"mathematical": 75.0},
            execution_stats={"mathematical": {"avg_time": 0.5, "success_rate": 75.0}},
            data_provenance={"data_hash": "abc123", "problem_count": 10},
            evaluation_timestamp=time.time()
        )
    
    def teardown_method(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_export_results_creates_file(self):
        """Test that export_results creates a properly formatted JSON file"""
        result_file = self.result.export_results(self.temp_dir)
        
        # Check file was created
        assert result_file.exists()
        assert result_file.suffix == '.json'
        assert 'humaneval_' in result_file.name
        
        # Check file content
        with open(result_file, 'r') as f:
            exported_data = json.load(f)
        
        assert exported_data['benchmark'] == 'HumanEval'
        assert exported_data['version'] == '2.0.0'
        assert 'results' in exported_data
        assert 'audit_trail' in exported_data
        assert 'security_measures' in exported_data['audit_trail']
        assert 'sandboxed_execution' in exported_data['audit_trail']['security_measures']

class TestIntegration:
    """Integration tests for the complete HumanEval benchmark system"""
    
    def test_factory_function(self):
        """Test the factory function creates evaluator correctly"""
        evaluator = create_humaneval_evaluator("test/path")
        assert isinstance(evaluator, HumanEvalEvaluator)
        assert str(evaluator.data_loader.data_path) == "test/path"
    
    def test_end_to_end_with_minimal_data(self):
        """Test complete end-to-end evaluation with minimal data"""
        # Create temporary directory with test data
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Create minimal test problem
            test_problem = {
                "task_id": "HumanEval/0",
                "prompt": "def return_true() -> bool:\n    \"\"\"Return True\"\"\"",
                "canonical_solution": "    return True",
                "test": "def check(candidate):\n    assert candidate() == True\n\ncheck(return_true)",
                "entry_point": "return_true"
            }
            
            data_file = temp_dir / "HumanEval.jsonl"
            with open(data_file, 'w') as f:
                f.write(json.dumps(test_problem) + '\n')
            
            # Run complete evaluation
            evaluator = create_humaneval_evaluator(str(temp_dir))
            result = evaluator.evaluate(sample_size=1)
            
            # Export results
            result_file = result.export_results(temp_dir / "results")
            
            # Validate complete pipeline worked
            assert result.total_problems == 1
            assert result_file.exists()
            
            # Validate exported data
            with open(result_file, 'r') as f:
                exported = json.load(f)
            assert exported['benchmark'] == 'HumanEval'
            assert 'sandboxed_execution' in exported['audit_trail']['security_measures']
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    """Run tests directly"""
    pytest.main([__file__, "-v", "--tb=short"])
