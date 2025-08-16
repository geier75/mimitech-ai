#!/usr/bin/env python3
"""
DIAGNOSE LIST OPERATIONS - ROOT CAUSE ANALYSIS
Systematic analysis of why List Operations show 0.0% pass rate
"""

import sys
import tempfile
import json
import re
from pathlib import Path

# Add benchmarks to path
sys.path.insert(0, str(Path(__file__).parent))

from benchmarks.humaneval_benchmark import HumanEvalProblem, HumanEvalSolver

def diagnose_classification():
    """Test if classification works correctly"""
    print("🔍 STEP 1: CLASSIFICATION DIAGNOSIS")
    print("=" * 60)
    
    solver = HumanEvalSolver()
    
    test_prompts = [
        "def sort_list(lst: List[int]) -> List[int]:\n    \"\"\"Sort list in ascending order\"\"\"",
        "def reverse_list(lst: List[int]) -> List[int]:\n    \"\"\"Reverse the list\"\"\"",
        "def list_operation_55(lst: List[int]) -> List[int]:\n    \"\"\"List operation 55\"\"\"",
        "def max_element(lst: List[int]) -> int:\n    \"\"\"Find maximum element\"\"\"",
        "def sum_list(lst: List[int]) -> int:\n    \"\"\"Calculate sum of list elements\"\"\""
    ]
    
    for prompt in test_prompts:
        classification = solver._classify_problem_type(prompt)
        print(f"Prompt: {prompt.split('(')[0]}...")
        print(f"   Classification: {classification}")
        print(f"   Expected: list_operations")
        print(f"   Correct: {'✅' if classification == 'list_operations' else '❌'}")
        print()

def diagnose_parameter_extraction():
    """Test parameter extraction"""
    print("🔍 STEP 2: PARAMETER EXTRACTION DIAGNOSIS")
    print("=" * 60)
    
    test_prompts = [
        "def sort_list(lst: List[int]) -> List[int]:",
        "def reverse_list(data: List[int]) -> List[int]:",
        "def list_operation_55(lst: List[int]) -> List[int]:",
        "def max_element(arr: List[int]) -> int:",
        "def process_list(items: List[int]) -> List[int]:"
    ]
    
    for prompt in test_prompts:
        # Test parameter extraction
        param_match = re.search(r'def\s+\w+\s*\(\s*(\w+):', prompt)
        list_param = param_match.group(1) if param_match else "lst"
        
        print(f"Prompt: {prompt}")
        print(f"   Extracted Parameter: '{list_param}'")
        print(f"   Match Found: {'✅' if param_match else '❌'}")
        print()

def diagnose_code_generation():
    """Test code generation for list operations"""
    print("🔍 STEP 3: CODE GENERATION DIAGNOSIS")
    print("=" * 60)
    
    solver = HumanEvalSolver()
    
    test_problems = [
        {
            "prompt": "def sort_list(lst: List[int]) -> List[int]:\n    \"\"\"Sort list in ascending order\"\"\"",
            "entry_point": "sort_list",
            "expected_behavior": "should sort the list"
        },
        {
            "prompt": "def reverse_list(lst: List[int]) -> List[int]:\n    \"\"\"Reverse the list\"\"\"",
            "entry_point": "reverse_list", 
            "expected_behavior": "should reverse the list"
        },
        {
            "prompt": "def list_operation_55(lst: List[int]) -> List[int]:\n    \"\"\"List operation 55\"\"\"",
            "entry_point": "list_operation_55",
            "expected_behavior": "should perform some list operation"
        }
    ]
    
    for problem_data in test_problems:
        problem = HumanEvalProblem(
            task_id="test",
            prompt=problem_data["prompt"],
            canonical_solution="",
            test="",
            entry_point=problem_data["entry_point"]
        )
        
        print(f"Problem: {problem.entry_point}")
        print(f"   Expected: {problem_data['expected_behavior']}")
        
        try:
            generated_code, reasoning_trace = solver.solve_problem(problem)
            print(f"   Generated Code: {generated_code}")
            print(f"   Problem Type: {reasoning_trace['problem_type']}")

            # Test if code is syntactically valid
            complete_code = problem.prompt + "\n" + generated_code
            try:
                compile(complete_code, '<string>', 'exec')
                print(f"   Syntax Valid: ✅")
            except SyntaxError as e:
                print(f"   Syntax Valid: ❌ - {e}")

        except Exception as e:
            print(f"   Code Generation Error: ❌ - {e}")
        print()

def diagnose_execution():
    """Test actual code execution"""
    print("🔍 STEP 4: CODE EXECUTION DIAGNOSIS")
    print("=" * 60)
    
    solver = HumanEvalSolver()
    
    # Test with a simple sort_list problem
    problem = HumanEvalProblem(
        task_id="test",
        prompt="def sort_list(lst: List[int]) -> List[int]:\n    \"\"\"Sort list in ascending order\"\"\"",
        canonical_solution="    return sorted(lst)",
        test="def check(candidate):\n    assert candidate([3, 1, 2]) == [1, 2, 3]\n    assert candidate([]) == []\n\ncheck(sort_list)",
        entry_point="sort_list"
    )
    
    print("Testing sort_list problem:")
    
    try:
        # Generate code
        generated_code, reasoning_trace = solver.solve_problem(problem)
        print(f"   Generated: {generated_code}")
        print(f"   Problem Type: {reasoning_trace['problem_type']}")
        
        # Create complete code
        complete_code = problem.prompt + "\n" + generated_code
        print(f"   Complete Code:\n{complete_code}")
        
        # Execute code
        exec_globals = {}
        exec(complete_code, exec_globals)
        func = exec_globals[problem.entry_point]
        
        # Test with sample data
        test_input = [3, 1, 2]
        result = func(test_input)
        expected = [1, 2, 3]
        
        print(f"   Input: {test_input}")
        print(f"   Output: {result}")
        print(f"   Expected: {expected}")
        print(f"   Correct: {'✅' if result == expected else '❌'}")
        
        # Test with empty list
        empty_result = func([])
        print(f"   Empty Input Result: {empty_result}")
        print(f"   Empty Correct: {'✅' if empty_result == [] else '❌'}")
        
        # Run actual test
        try:
            exec(problem.test, {**exec_globals, problem.entry_point: func})
            print(f"   Test Execution: ✅ PASSED")
        except Exception as e:
            print(f"   Test Execution: ❌ FAILED - {e}")
            
    except Exception as e:
        print(f"   Execution Error: ❌ - {e}")

def diagnose_generic_list_operations():
    """Test generic list operations that are failing"""
    print("\n🔍 STEP 5: GENERIC LIST OPERATIONS DIAGNOSIS")
    print("=" * 60)
    
    solver = HumanEvalSolver()
    
    # Test generic list operation like those in our benchmark
    problem = HumanEvalProblem(
        task_id="test",
        prompt="def list_operation_55(lst: List[int]) -> List[int]:\n    \"\"\"List operation 55\"\"\"",
        canonical_solution="    return sorted(lst)",
        test="def check(candidate):\n    result = candidate([3,1,2])\n    assert isinstance(result, list)\n    assert len(result) == 3\n\ncheck(list_operation_55)",
        entry_point="list_operation_55"
    )
    
    print("Testing generic list_operation_55:")
    
    try:
        # Generate code
        generated_code, reasoning_trace = solver.solve_problem(problem)
        print(f"   Generated: {generated_code}")
        print(f"   Problem Type: {reasoning_trace['problem_type']}")
        
        # Create complete code
        complete_code = problem.prompt + "\n" + generated_code
        
        # Execute code
        exec_globals = {}
        exec(complete_code, exec_globals)
        func = exec_globals[problem.entry_point]
        
        # Test with sample data
        test_input = [3, 1, 2]
        result = func(test_input)
        
        print(f"   Input: {test_input}")
        print(f"   Output: {result}")
        print(f"   Is List: {'✅' if isinstance(result, list) else '❌'}")
        print(f"   Length 3: {'✅' if len(result) == 3 else '❌'}")
        
        # Run actual test
        try:
            exec(problem.test, {**exec_globals, problem.entry_point: func})
            print(f"   Test Execution: ✅ PASSED")
        except Exception as e:
            print(f"   Test Execution: ❌ FAILED - {e}")
            
    except Exception as e:
        print(f"   Execution Error: ❌ - {e}")

def main():
    """Run complete diagnosis"""
    print("🚀 LIST OPERATIONS ROOT CAUSE ANALYSIS")
    print("=" * 80)
    print("Diagnosing why 44 List Operations problems are detected but 0.0% pass")
    print()
    
    diagnose_classification()
    diagnose_parameter_extraction()
    diagnose_code_generation()
    diagnose_execution()
    diagnose_generic_list_operations()
    
    print("\n" + "=" * 80)
    print("🎯 DIAGNOSIS COMPLETE")
    print("Review the output to identify the specific failure points")

if __name__ == "__main__":
    main()
