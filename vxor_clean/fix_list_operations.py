#!/usr/bin/env python3
"""
DEFINITIV LIST OPERATIONS AUF 100% BRINGEN
Analysiere und behebe das spezifische Problem
"""

import sys
import tempfile
import json
from pathlib import Path

# Add benchmarks to path
sys.path.insert(0, str(Path(__file__).parent))

from benchmarks.humaneval_benchmark import HumanEvalProblem, HumanEvalSolver, HumanEvalEvaluator

def test_specific_list_problem():
    """Teste ein spezifisches List Problem um das Problem zu identifizieren"""
    print("🔍 SPEZIFISCHER LIST OPERATIONS TEST")
    print("=" * 60)
    
    # Erstelle ein einfaches List Problem
    problem_data = {
        "task_id": "HumanEval/50",
        "prompt": "def sort_list(lst: List[int]) -> List[int]:\n    \"\"\"Sort list in ascending order\"\"\"",
        "canonical_solution": "    return sorted(lst)",
        "test": "def check(candidate):\n    assert candidate([3, 1, 4, 1, 5]) == [1, 1, 3, 4, 5]\n    assert candidate([]) == []\n    assert candidate([1]) == [1]\n\ncheck(sort_list)",
        "entry_point": "sort_list"
    }
    
    problem = HumanEvalProblem(
        task_id=problem_data["task_id"],
        prompt=problem_data["prompt"],
        canonical_solution=problem_data["canonical_solution"],
        test=problem_data["test"],
        entry_point=problem_data["entry_point"]
    )
    
    solver = HumanEvalSolver()
    
    # Teste Klassifikation
    problem_type = solver._classify_problem_type(problem.prompt)
    print(f"Klassifikation: {problem_type}")
    
    # Teste Code-Generierung
    generated_code = solver._generate_code(problem)
    print(f"Generierter Code: {generated_code}")
    
    # Teste vollständigen Code
    complete_code = problem.prompt + "\n" + generated_code
    print(f"Vollständiger Code:\n{complete_code}")
    
    # Teste Ausführung
    try:
        exec_globals = {}
        exec(complete_code, exec_globals)
        func = exec_globals[problem.entry_point]
        
        # Teste mit den echten Test-Daten
        result1 = func([3, 1, 4, 1, 5])
        result2 = func([])
        result3 = func([1])
        
        print(f"Test 1: {result1} (erwartet: [1, 1, 3, 4, 5])")
        print(f"Test 2: {result2} (erwartet: [])")
        print(f"Test 3: {result3} (erwartet: [1])")
        
        # Führe den echten Test aus
        try:
            exec(problem.test, {**exec_globals, problem.entry_point: func})
            print("✅ ECHTER TEST: BESTANDEN")
        except Exception as e:
            print(f"❌ ECHTER TEST: FEHLGESCHLAGEN - {e}")
            
    except Exception as e:
        print(f"❌ CODE AUSFÜHRUNG FEHLGESCHLAGEN: {e}")

def test_generic_list_problem():
    """Teste ein generisches List Problem"""
    print("\n🔍 GENERISCHES LIST OPERATIONS TEST")
    print("=" * 60)
    
    # Erstelle ein generisches List Problem wie in unserem Benchmark
    problem_data = {
        "task_id": "HumanEval/55",
        "prompt": "def list_operation_55(lst: List[int]) -> List[int]:\n    \"\"\"List operation 55\"\"\"",
        "canonical_solution": "    return sorted(lst)",
        "test": "def check(candidate):\n    result = candidate([3,1,2])\n    assert isinstance(result, list)\n    assert len(result) == 3\n\ncheck(list_operation_55)",
        "entry_point": "list_operation_55"
    }
    
    problem = HumanEvalProblem(
        task_id=problem_data["task_id"],
        prompt=problem_data["prompt"],
        canonical_solution=problem_data["canonical_solution"],
        test=problem_data["test"],
        entry_point=problem_data["entry_point"]
    )
    
    solver = HumanEvalSolver()
    
    # Teste Klassifikation
    problem_type = solver._classify_problem_type(problem.prompt)
    print(f"Klassifikation: {problem_type}")
    
    # Teste Code-Generierung
    generated_code = solver._generate_code(problem)
    print(f"Generierter Code: {generated_code}")
    
    # Teste vollständigen Code
    complete_code = problem.prompt + "\n" + generated_code
    print(f"Vollständiger Code:\n{complete_code}")
    
    # Teste Ausführung
    try:
        exec_globals = {}
        exec(complete_code, exec_globals)
        func = exec_globals[problem.entry_point]
        
        # Teste mit den Test-Daten
        result = func([3, 1, 2])
        print(f"Test Ergebnis: {result}")
        print(f"Ist Liste: {isinstance(result, list)}")
        print(f"Länge: {len(result) if isinstance(result, list) else 'N/A'}")
        
        # Führe den echten Test aus
        try:
            exec(problem.test, {**exec_globals, problem.entry_point: func})
            print("✅ ECHTER TEST: BESTANDEN")
        except Exception as e:
            print(f"❌ ECHTER TEST: FEHLGESCHLAGEN - {e}")
            
    except Exception as e:
        print(f"❌ CODE AUSFÜHRUNG FEHLGESCHLAGEN: {e}")

def main():
    """Hauptfunktion für List Operations Fix"""
    test_specific_list_problem()
    test_generic_list_problem()
    
    print("\n" + "=" * 60)
    print("🎯 ANALYSE ABGESCHLOSSEN")
    print("Identifiziere das spezifische Problem mit List Operations")

if __name__ == "__main__":
    main()
