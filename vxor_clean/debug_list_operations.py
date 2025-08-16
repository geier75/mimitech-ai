#!/usr/bin/env python3
"""
Debug List Operations - Analysiere warum List Operations 0.0% zeigen
"""

import sys
import tempfile
import json
from pathlib import Path

# Add benchmarks to path
sys.path.insert(0, str(Path(__file__).parent))

from benchmarks.humaneval_benchmark import HumanEvalProblem, HumanEvalSolver

def test_list_operations_debug():
    """Debug List Operations Probleme"""
    print("🔍 DEBUG: LIST OPERATIONS ANALYSE")
    print("=" * 60)
    
    # Erstelle spezifische List Operations Probleme
    list_problems = [
        {
            "task_id": "HumanEval/50",
            "prompt": "def sort_list(lst: List[int]) -> List[int]:\n    \"\"\"Sort list in ascending order\"\"\"",
            "canonical_solution": "    return sorted(lst)",
            "test": "def check(candidate):\n    assert candidate([3, 1, 4, 1, 5]) == [1, 1, 3, 4, 5]\n    assert candidate([]) == []\n    assert candidate([1]) == [1]\n\ncheck(sort_list)",
            "entry_point": "sort_list"
        },
        {
            "task_id": "HumanEval/51",
            "prompt": "def reverse_list(lst: List[int]) -> List[int]:\n    \"\"\"Reverse the list\"\"\"",
            "canonical_solution": "    return lst[::-1]",
            "test": "def check(candidate):\n    assert candidate([1, 2, 3]) == [3, 2, 1]\n    assert candidate([]) == []\n    assert candidate([5]) == [5]\n\ncheck(reverse_list)",
            "entry_point": "reverse_list"
        },
        {
            "task_id": "HumanEval/52",
            "prompt": "def list_operation_52(lst: List[int]) -> List[int]:\n    \"\"\"List operation 52\"\"\"",
            "canonical_solution": "    return sorted(lst)",
            "test": "def check(candidate):\n    assert isinstance(candidate([1,2,3]), list)\n\ncheck(list_operation_52)",
            "entry_point": "list_operation_52"
        }
    ]
    
    solver = HumanEvalSolver()
    
    for i, problem_data in enumerate(list_problems):
        print(f"\n🧪 TEST {i+1}: {problem_data['entry_point']}")
        print("-" * 40)
        
        # Erstelle HumanEvalProblem
        problem = HumanEvalProblem(
            task_id=problem_data["task_id"],
            prompt=problem_data["prompt"],
            canonical_solution=problem_data["canonical_solution"],
            test=problem_data["test"],
            entry_point=problem_data["entry_point"]
        )
        
        # Teste Klassifikation
        problem_type = solver._classify_problem_type(problem.prompt)
        print(f"   Klassifikation: {problem_type}")
        
        # Teste Code-Generierung
        try:
            generated_code = solver.solve(problem)
            print(f"   Generierter Code: {generated_code}")
            
            # Teste Ausführung
            complete_code = problem.prompt + "\n" + generated_code
            print(f"   Vollständiger Code:")
            print(f"   {complete_code}")
            
            # Versuche Code auszuführen
            try:
                exec_globals = {}
                exec(complete_code, exec_globals)
                func = exec_globals[problem.entry_point]
                
                # Teste mit einfachen Daten
                if "sort" in problem.entry_point:
                    result = func([3, 1, 2])
                    print(f"   Test Ergebnis: {result}")
                    print(f"   Test Status: {'✅ PASS' if result == [1, 2, 3] else '❌ FAIL'}")
                elif "reverse" in problem.entry_point:
                    result = func([1, 2, 3])
                    print(f"   Test Ergebnis: {result}")
                    print(f"   Test Status: {'✅ PASS' if result == [3, 2, 1] else '❌ FAIL'}")
                else:
                    result = func([1, 2, 3])
                    print(f"   Test Ergebnis: {result}")
                    print(f"   Test Status: {'✅ PASS' if isinstance(result, list) else '❌ FAIL'}")
                    
            except Exception as e:
                print(f"   Ausführung FEHLER: {e}")
                
        except Exception as e:
            print(f"   Code-Generierung FEHLER: {e}")

def test_parameter_extraction():
    """Teste Parameter Extraktion"""
    print("\n🔍 DEBUG: PARAMETER EXTRAKTION")
    print("=" * 60)
    
    test_prompts = [
        "def sort_list(lst: List[int]) -> List[int]:",
        "def reverse_list(data: List[int]) -> List[int]:",
        "def list_operation_52(lst: List[int]) -> List[int]:",
        "def process_array(arr: List[int]) -> List[int]:"
    ]
    
    import re
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        
        # Teste Parameter Extraktion
        param_match = re.search(r'def\s+\w+\s*\(\s*(\w+):', prompt)
        list_param = param_match.group(1) if param_match else "lst"
        print(f"   Extrahierter Parameter: {list_param}")
        
        # Teste Klassifikation
        solver = HumanEvalSolver()
        problem_type = solver._classify_problem_type(prompt)
        print(f"   Klassifikation: {problem_type}")

def main():
    """Hauptfunktion für List Operations Debug"""
    test_list_operations_debug()
    test_parameter_extraction()
    
    print("\n" + "=" * 60)
    print("🎯 DEBUG ANALYSE ABGESCHLOSSEN")
    print("Überprüfe die Ausgabe um List Operations Probleme zu identifizieren")

if __name__ == "__main__":
    main()
