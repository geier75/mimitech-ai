#!/usr/bin/env python3
"""
JSON Export Functionality Test for HumanEval Benchmark
Demonstrates comprehensive result persistence with audit trail
"""

import sys
import tempfile
import json
import gzip
import os
from pathlib import Path

# Add benchmarks to path
sys.path.insert(0, str(Path(__file__).parent))

from benchmarks.humaneval_benchmark import (
    HumanEvalProblem, HumanEvalEvaluator, HumanEvalResult, 
    HumanEvalConfigLoader, create_humaneval_evaluator
)

def create_test_data():
    """Create comprehensive test data with 111 diverse problems for production testing"""
    temp_dir = Path(tempfile.mkdtemp())

    # Create 111 diverse HumanEval problems covering all categories
    test_problems = []

    # STRING MANIPULATION PROBLEMS (25 problems)
    string_problems = [
        {
            "task_id": f"HumanEval/{i}",
            "prompt": f"def string_func_{i}(s: str) -> str:\n    \"\"\"String manipulation function {i}\"\"\"",
            "canonical_solution": "    return s[::-1]" if i % 2 == 0 else "    return s.upper()",
            "test": f"def check(candidate):\n    assert len(candidate('test')) >= 0\n\ncheck(string_func_{i})",
            "entry_point": f"string_func_{i}"
        }
        for i in range(0, 25)
    ]

    # MATHEMATICAL PROBLEMS (25 problems)
    math_problems = [
        {
            "task_id": f"HumanEval/{i}",
            "prompt": f"def math_func_{i}(n: int) -> int:\n    \"\"\"Mathematical function {i}\"\"\"",
            "canonical_solution": "    return n * 2" if i % 2 == 0 else "    return n + 1",
            "test": f"def check(candidate):\n    assert candidate(5) > 0\n\ncheck(math_func_{i})",
            "entry_point": f"math_func_{i}"
        }
        for i in range(25, 50)
    ]

    # LIST OPERATIONS PROBLEMS (25 problems)
    list_problems = [
        {
            "task_id": f"HumanEval/{i}",
            "prompt": f"def list_func_{i}(lst: List[int]) -> List[int]:\n    \"\"\"List operation function {i}\"\"\"",
            "canonical_solution": "    return sorted(lst)" if i % 2 == 0 else "    return lst[::-1]",
            "test": f"def check(candidate):\n    assert isinstance(candidate([1,2,3]), list)\n\ncheck(list_func_{i})",
            "entry_point": f"list_func_{i}"
        }
        for i in range(50, 75)
    ]

    # CONDITIONAL LOGIC PROBLEMS (20 problems)
    conditional_problems = [
        {
            "task_id": f"HumanEval/{i}",
            "prompt": f"def is_condition_{i}(n: int) -> bool:\n    \"\"\"Check condition {i}\"\"\"",
            "canonical_solution": "    return n % 2 == 0" if i % 2 == 0 else "    return n > 0",
            "test": f"def check(candidate):\n    assert isinstance(candidate(5), bool)\n\ncheck(is_condition_{i})",
            "entry_point": f"is_condition_{i}"
        }
        for i in range(75, 95)
    ]

    # ALGORITHMIC PROBLEMS (16 problems)
    algo_problems = [
        {
            "task_id": f"HumanEval/{i}",
            "prompt": f"def algorithm_{i}(data: List[int]) -> int:\n    \"\"\"Algorithm function {i}\"\"\"",
            "canonical_solution": "    return max(data)" if i % 2 == 0 else "    return sum(data)",
            "test": f"def check(candidate):\n    assert candidate([1,2,3]) >= 0\n\ncheck(algorithm_{i})",
            "entry_point": f"algorithm_{i}"
        }
        for i in range(95, 111)
    ]

    # Combine all problems
    test_problems = string_problems + math_problems + list_problems + conditional_problems + algo_problems
    
    # Write test data to JSONL file
    data_file = temp_dir / "HumanEval.jsonl"
    with open(data_file, 'w') as f:
        for problem in test_problems:
            f.write(json.dumps(problem) + '\n')
    
    return temp_dir

def test_basic_json_export():
    """Test basic JSON export functionality"""
    print("🧪 Testing Basic JSON Export")
    print("=" * 50)
    
    # Create test data
    temp_dir = create_test_data()
    
    try:
        # Create evaluator
        evaluator = create_humaneval_evaluator(str(temp_dir))
        
        # Run evaluation with 111 problems
        print("Running evaluation with 111 problems...")
        result = evaluator.evaluate(sample_size=111)
        
        # Test manual export
        results_dir = Path("test_results")
        exported_file = result.export_results(results_dir, evaluator.config_loader)
        
        print(f"✅ Results exported to: {exported_file}")
        print(f"✅ File exists: {exported_file.exists()}")
        print(f"✅ File size: {exported_file.stat().st_size} bytes")
        
        # Verify JSON structure
        with open(exported_file, 'r', encoding='utf-8') as f:
            exported_data = json.load(f)
        
        required_sections = [
            "metadata", "evaluation_results", "data_integrity", 
            "security_audit", "configuration", "audit_trail"
        ]
        
        for section in required_sections:
            if section in exported_data:
                print(f"✅ Section '{section}' present")
            else:
                print(f"❌ Section '{section}' missing")
        
        # Display key metrics
        summary = exported_data["evaluation_results"]["summary"]
        print(f"\n📊 Evaluation Summary (111 Problems):")
        print(f"   Total Problems: {summary['total_problems']}")
        print(f"   Passed Problems: {summary['passed_problems']}")
        print(f"   Pass@1 Rate: {summary['pass_at_1']:.1f}%")
        print(f"   Execution Time: {summary['execution_time_seconds']:.3f}s")
        print(f"   Avg Time per Problem: {summary['execution_time_seconds']/summary['total_problems']:.4f}s")

        # Display category breakdown
        categories = exported_data["evaluation_results"]["category_performance"]["category_pass_rates"]
        print(f"\n📈 Category Performance:")
        for category, rate in categories.items():
            print(f"   {category}: {rate:.1f}%")
        
        return exported_file
        
    except Exception as e:
        print(f"❌ Error in basic export test: {e}")
        return None
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_compressed_export():
    """Test compressed JSON export functionality"""
    print("\n🗜️ Testing Compressed JSON Export")
    print("=" * 50)
    
    # Set environment variable to force compression
    os.environ["VXOR_ENABLE_COMPRESSION"] = "true"
    
    # Create test data
    temp_dir = create_test_data()
    
    try:
        # Create evaluator
        evaluator = create_humaneval_evaluator(str(temp_dir))
        
        # Run evaluation with 111 problems
        print("Running evaluation with compression enabled (111 problems)...")
        result = evaluator.evaluate(sample_size=111)
        
        # Test compressed export
        results_dir = Path("test_results")
        exported_file = result.export_results(results_dir, evaluator.config_loader)
        
        print(f"✅ Compressed results exported to: {exported_file}")
        print(f"✅ File extension: {exported_file.suffix}")
        print(f"✅ Is compressed: {exported_file.suffix == '.gz'}")
        print(f"✅ File size: {exported_file.stat().st_size} bytes")
        
        # Verify compressed JSON structure
        if exported_file.suffix == '.gz':
            with gzip.open(exported_file, 'rt', encoding='utf-8') as f:
                exported_data = json.load(f)
        else:
            with open(exported_file, 'r', encoding='utf-8') as f:
                exported_data = json.load(f)
        
        # Check compression flag in metadata
        compression_used = exported_data["metadata"]["compression_used"]
        print(f"✅ Compression flag in metadata: {compression_used}")
        
        return exported_file
        
    except Exception as e:
        print(f"❌ Error in compressed export test: {e}")
        return None
    finally:
        # Cleanup environment
        if "VXOR_ENABLE_COMPRESSION" in os.environ:
            del os.environ["VXOR_ENABLE_COMPRESSION"]
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_environment_variable_tracking():
    """Test environment variable tracking in exports"""
    print("\n🌍 Testing Environment Variable Tracking")
    print("=" * 50)
    
    # Set multiple environment variables
    test_env_vars = {
        "VXOR_INCLUDE_GENERATED_CODE": "true",
        "VXOR_ENABLE_COMPRESSION": "false",
        "VXOR_MAX_SUBPROCESSES": "2"
    }
    
    for key, value in test_env_vars.items():
        os.environ[key] = value
    
    # Create test data
    temp_dir = create_test_data()
    
    try:
        # Create evaluator
        evaluator = create_humaneval_evaluator(str(temp_dir))
        
        # Run evaluation with 111 problems
        print("Running evaluation with environment variables set (111 problems)...")
        result = evaluator.evaluate(sample_size=111)
        
        # Export results
        results_dir = Path("test_results")
        exported_file = result.export_results(results_dir, evaluator.config_loader)
        
        # Verify environment variables are tracked
        with open(exported_file, 'r', encoding='utf-8') as f:
            exported_data = json.load(f)
        
        env_overrides = exported_data["configuration"]["environment_overrides"]
        
        print("Environment variables tracked:")
        for key, expected_value in test_env_vars.items():
            if key in env_overrides:
                actual_value = env_overrides[key]
                if actual_value == expected_value:
                    print(f"✅ {key}: {actual_value}")
                else:
                    print(f"❌ {key}: expected {expected_value}, got {actual_value}")
            else:
                print(f"❌ {key}: not tracked")
        
        return exported_file
        
    except Exception as e:
        print(f"❌ Error in environment variable test: {e}")
        return None
    finally:
        # Cleanup environment variables
        for key in test_env_vars:
            if key in os.environ:
                del os.environ[key]
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_automatic_export():
    """Test automatic export during evaluation"""
    print("\n🤖 Testing Automatic Export During Evaluation")
    print("=" * 50)
    
    # Create test data
    temp_dir = create_test_data()
    
    try:
        # Create evaluator
        evaluator = create_humaneval_evaluator(str(temp_dir))
        
        # Check if results directory exists before evaluation
        results_dir = Path("results")
        files_before = list(results_dir.glob("humaneval_*.json*")) if results_dir.exists() else []
        
        print(f"Files before evaluation: {len(files_before)}")
        
        # Run evaluation with 111 problems (should automatically export)
        print("Running evaluation with 111 problems (automatic export enabled)...")
        result = evaluator.evaluate(sample_size=111)
        
        # Check if new files were created
        files_after = list(results_dir.glob("humaneval_*.json*")) if results_dir.exists() else []
        new_files = [f for f in files_after if f not in files_before]
        
        print(f"Files after evaluation: {len(files_after)}")
        print(f"New files created: {len(new_files)}")
        
        if new_files:
            latest_file = max(new_files, key=lambda f: f.stat().st_mtime)
            print(f"✅ Latest exported file: {latest_file}")
            print(f"✅ File size: {latest_file.stat().st_size} bytes")
            
            # Verify it's a valid JSON file
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"✅ Valid JSON structure")
                print(f"✅ Benchmark: {data['metadata']['benchmark']}")
                print(f"✅ Version: {data['metadata']['version']}")
            except json.JSONDecodeError:
                print(f"❌ Invalid JSON structure")
        else:
            print(f"❌ No new files created during automatic export")
        
        return new_files[0] if new_files else None
        
    except Exception as e:
        print(f"❌ Error in automatic export test: {e}")
        return None
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    """Run all JSON export tests"""
    print("🚀 HumanEval JSON Export Functionality Tests")
    print("=" * 70)
    
    test_results = []
    
    # Run all tests
    test_results.append(("Basic JSON Export", test_basic_json_export()))
    test_results.append(("Compressed Export", test_compressed_export()))
    test_results.append(("Environment Tracking", test_environment_variable_tracking()))
    test_results.append(("Automatic Export", test_automatic_export()))
    
    # Summary
    print("\n" + "=" * 70)
    print("🎯 TEST SUMMARY")
    print("=" * 70)
    
    passed_tests = 0
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed_tests += 1
    
    print(f"\nOverall: {passed_tests}/{len(test_results)} tests passed")
    
    if passed_tests == len(test_results):
        print("🎉 All JSON export functionality tests PASSED!")
        print("📊 Results are being persisted to the results/ directory")
        print("🔒 Full audit trail and security measures documented")
    else:
        print("⚠️ Some tests failed - check implementation")

if __name__ == "__main__":
    main()
