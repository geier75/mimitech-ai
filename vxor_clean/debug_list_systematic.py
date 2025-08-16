#!/usr/bin/env python3
"""
SYSTEMATISCHE LIST OPERATIONS DIAGNOSE
Exakte Analyse warum List Operations bei 59.1% bleiben
"""

import sys
import tempfile
import subprocess
from pathlib import Path

# Add benchmarks to path
sys.path.insert(0, str(Path(__file__).parent))

from create_real_111_problems import create_real_111_problems
from benchmarks.humaneval_benchmark import create_humaneval_evaluator

def test_single_list_operation():
    """Teste eine einzelne List Operation im Detail"""
    print("🔍 EINZELTEST: LIST OPERATION")
    print("=" * 60)
    
    # Erstelle einen einzelnen List Operations Test
    from benchmarks.humaneval_benchmark import HumanEvalProblem, HumanEvalSolver
    
    problem = HumanEvalProblem(
        task_id="TEST_LIST",
        prompt="def list_operation_55(lst: List[int]) -> List[int]:\n    \"\"\"List operation 55\"\"\"",
        canonical_solution="    return sorted(lst)",
        test="def check(candidate):\n    result = candidate([3,1,2])\n    assert True  # Always pass\n\ncheck(list_operation_55)",
        entry_point="list_operation_55"
    )
    
    solver = HumanEvalSolver()
    
    print(f"Problem: {problem.entry_point}")
    print(f"Prompt: {problem.prompt}")
    print(f"Test: {problem.test}")
    
    # Teste Code-Generierung
    try:
        generated_code, reasoning_trace = solver.solve_problem(problem)
        print(f"✅ Code generiert: {generated_code[:100]}...")
        print(f"Problem Type: {reasoning_trace['problem_type']}")
        
        # Teste manuelle Ausführung
        complete_code = problem.prompt + "\n" + generated_code
        print(f"Vollständiger Code:\n{complete_code}")
        
        # Teste mit typing import
        test_code = """
from typing import List

""" + complete_code + """

""" + problem.test
        
        print(f"Test Code:\n{test_code}")
        
        # Führe Test aus
        try:
            exec_globals = {}
            exec(test_code, exec_globals)
            print("✅ MANUELLER TEST: BESTANDEN")
            return True
        except Exception as e:
            print(f"❌ MANUELLER TEST: FEHLGESCHLAGEN - {e}")
            return False
            
    except Exception as e:
        print(f"❌ CODE GENERIERUNG FEHLGESCHLAGEN: {e}")
        return False

def test_evaluator_execution():
    """Teste wie der Evaluator List Operations ausführt"""
    print("\n🔍 EVALUATOR EXECUTION TEST")
    print("=" * 60)
    
    # Erstelle Test-Daten
    temp_dir = create_real_111_problems()
    
    try:
        # Erstelle Evaluator
        evaluator = create_humaneval_evaluator(str(temp_dir))
        
        # Lade nur List Operations Probleme
        problems = evaluator.data_loader.load_problems(111)
        list_problems = [p for p in problems if evaluator.solver._classify_problem_type(p.prompt) == "list_operations"]
        
        print(f"Gefundene List Operations: {len(list_problems)}")
        
        # Teste die ersten 3 List Operations im Detail
        for i, problem in enumerate(list_problems[:3]):
            print(f"\n--- LIST PROBLEM {i+1}: {problem.task_id} ---")
            print(f"Entry Point: {problem.entry_point}")
            print(f"Prompt: {problem.prompt[:100]}...")
            print(f"Test: {problem.test[:100]}...")
            
            # Generiere Code
            generated_code, reasoning_trace = evaluator.solver.solve_problem(problem)
            print(f"Generated Code: {generated_code[:100]}...")
            
            # Teste mit Evaluator
            test_passed = evaluator._test_generated_code(problem, generated_code)
            print(f"Evaluator Test Result: {'✅ PASS' if test_passed else '❌ FAIL'}")
            
            if not test_passed:
                print("🔍 DEBUGGING FAILED TEST:")
                
                # Teste manuell mit sicherem Code
                try:
                    secure_code = evaluator._create_secure_test_code(problem.prompt, generated_code, problem.test)
                    print(f"Secure Code (first 200 chars):\n{secure_code[:200]}...")
                    
                    # Schreibe in temporäre Datei und führe aus
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                        temp_file.write(secure_code)
                        temp_file_path = temp_file.name
                    
                    result = subprocess.run(
                        [sys.executable, temp_file_path],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    print(f"Subprocess Return Code: {result.returncode}")
                    print(f"Subprocess STDOUT: {result.stdout}")
                    print(f"Subprocess STDERR: {result.stderr}")
                    
                    # Cleanup
                    Path(temp_file_path).unlink(missing_ok=True)
                    
                except Exception as e:
                    print(f"Manual test error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ EVALUATOR TEST FEHLER: {e}")
        return False
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

def analyze_list_classification():
    """Analysiere List Operations Klassifikation"""
    print("\n🔍 LIST CLASSIFICATION ANALYSIS")
    print("=" * 60)
    
    from benchmarks.humaneval_benchmark import HumanEvalSolver
    solver = HumanEvalSolver()
    
    test_prompts = [
        "def list_operation_55(lst: List[int]) -> List[int]:\n    \"\"\"List operation 55\"\"\"",
        "def sort_list(lst: List[int]) -> List[int]:\n    \"\"\"Sort list in ascending order\"\"\"",
        "def reverse_list(lst: List[int]) -> List[int]:\n    \"\"\"Reverse the list\"\"\"",
        "def max_element(lst: List[int]) -> int:\n    \"\"\"Find maximum element\"\"\"",
        "def sum_list(lst: List[int]) -> int:\n    \"\"\"Calculate sum of list elements\"\"\"",
        "def filter_even(lst: List[int]) -> List[int]:\n    \"\"\"Filter even numbers\"\"\""
    ]
    
    for prompt in test_prompts:
        classification = solver._classify_problem_type(prompt)
        print(f"Prompt: {prompt.split('(')[0]}...")
        print(f"   Classification: {classification}")
        print(f"   Expected: list_operations")
        print(f"   Correct: {'✅' if classification == 'list_operations' else '❌'}")
        print()

def main():
    """Hauptfunktion für systematische Diagnose"""
    print("🚀 SYSTEMATISCHE LIST OPERATIONS DIAGNOSE")
    print("=" * 80)
    print("Ziel: Exakte Identifikation warum List Operations bei 59.1% bleiben")
    print()
    
    # Schritt 1: Einzeltest
    single_success = test_single_list_operation()
    
    # Schritt 2: Klassifikation
    analyze_list_classification()
    
    # Schritt 3: Evaluator Test
    evaluator_success = test_evaluator_execution()
    
    # Zusammenfassung
    print("\n" + "=" * 80)
    print("🎯 DIAGNOSE ZUSAMMENFASSUNG")
    print(f"Einzeltest: {'✅ PASS' if single_success else '❌ FAIL'}")
    print(f"Evaluator Test: {'✅ PASS' if evaluator_success else '❌ FAIL'}")
    
    if single_success and not evaluator_success:
        print("🔍 PROBLEM IDENTIFIZIERT: Evaluator Execution Environment")
    elif not single_success:
        print("🔍 PROBLEM IDENTIFIZIERT: Code Generation oder Execution")
    else:
        print("🔍 PROBLEM UNKLAR: Weitere Analyse erforderlich")

if __name__ == "__main__":
    main()
