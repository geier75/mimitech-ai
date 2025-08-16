#!/usr/bin/env python3
"""
IDENTIFIZIERE FEHLGESCHLAGENE LIST OPERATIONS
Finde genau welche List Operations bei 59.1% fehlschlagen
"""

import sys
from pathlib import Path

# Add benchmarks to path
sys.path.insert(0, str(Path(__file__).parent))

from create_real_111_problems import create_real_111_problems
from benchmarks.humaneval_benchmark import create_humaneval_evaluator

def find_all_failing_list_operations():
    """Finde alle fehlgeschlagenen List Operations"""
    print("🔍 IDENTIFIZIERE ALLE FEHLGESCHLAGENEN LIST OPERATIONS")
    print("=" * 80)
    
    # Erstelle Test-Daten
    temp_dir = create_real_111_problems()
    
    try:
        # Erstelle Evaluator
        evaluator = create_humaneval_evaluator(str(temp_dir))
        
        # Lade alle Probleme
        problems = evaluator.data_loader.load_problems(111)
        
        # Filtere List Operations
        list_problems = []
        for problem in problems:
            problem_type = evaluator.solver._classify_problem_type(problem.prompt)
            if problem_type == "list_operations":
                list_problems.append(problem)
        
        print(f"📊 Gefundene List Operations: {len(list_problems)}")
        print()
        
        # Teste jede List Operation einzeln
        passed = 0
        failed = 0
        failed_problems = []
        
        for i, problem in enumerate(list_problems):
            print(f"🧪 TEST {i+1}/{len(list_problems)}: {problem.task_id} ({problem.entry_point})")
            
            try:
                # Generiere Code
                generated_code, reasoning_trace = evaluator.solver.solve_problem(problem)
                
                # Teste Code
                test_passed = evaluator._test_generated_code(problem, generated_code)
                
                if test_passed:
                    print(f"   ✅ PASS")
                    passed += 1
                else:
                    print(f"   ❌ FAIL")
                    failed += 1
                    failed_problems.append({
                        'task_id': problem.task_id,
                        'entry_point': problem.entry_point,
                        'prompt': problem.prompt[:100] + "...",
                        'generated_code': generated_code[:100] + "...",
                        'test': problem.test[:100] + "..."
                    })
                    
            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                failed += 1
                failed_problems.append({
                    'task_id': problem.task_id,
                    'entry_point': problem.entry_point,
                    'error': str(e)
                })
        
        # Zusammenfassung
        print("\n" + "=" * 80)
        print("📊 ERGEBNISSE:")
        print(f"   Gesamt: {len(list_problems)}")
        print(f"   Bestanden: {passed}")
        print(f"   Fehlgeschlagen: {failed}")
        print(f"   Pass Rate: {(passed/len(list_problems)*100):.1f}%")
        
        # Detailanalyse der Failures
        if failed_problems:
            print(f"\n❌ FEHLGESCHLAGENE PROBLEME ({len(failed_problems)}):")
            print("=" * 60)
            
            for i, failure in enumerate(failed_problems):
                print(f"\n{i+1}. {failure['task_id']} - {failure['entry_point']}")
                if 'prompt' in failure:
                    print(f"   Prompt: {failure['prompt']}")
                    print(f"   Generated: {failure['generated_code']}")
                    print(f"   Test: {failure['test']}")
                if 'error' in failure:
                    print(f"   Error: {failure['error']}")
        
        return failed_problems
        
    except Exception as e:
        print(f"❌ ANALYSE FEHLER: {e}")
        return []
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

def analyze_failure_patterns(failed_problems):
    """Analysiere Muster in den fehlgeschlagenen Problemen"""
    print(f"\n🔍 FAILURE PATTERN ANALYSE")
    print("=" * 60)
    
    if not failed_problems:
        print("Keine fehlgeschlagenen Probleme zu analysieren.")
        return
    
    # Analysiere Entry Points
    entry_points = [f.get('entry_point', 'unknown') for f in failed_problems]
    print(f"Fehlgeschlagene Entry Points: {entry_points}")
    
    # Analysiere Task IDs
    task_ids = [f.get('task_id', 'unknown') for f in failed_problems]
    print(f"Fehlgeschlagene Task IDs: {task_ids}")
    
    # Suche nach Mustern
    patterns = {}
    for failure in failed_problems:
        entry_point = failure.get('entry_point', 'unknown')
        if 'list_operation_' in entry_point:
            patterns['generic_list_operations'] = patterns.get('generic_list_operations', 0) + 1
        elif entry_point in ['sort_list', 'reverse_list', 'max_element', 'sum_list', 'filter_even']:
            patterns['specific_list_operations'] = patterns.get('specific_list_operations', 0) + 1
        else:
            patterns['other'] = patterns.get('other', 0) + 1
    
    print(f"Failure Patterns: {patterns}")

def main():
    """Hauptfunktion"""
    print("🚀 FINDE FEHLGESCHLAGENE LIST OPERATIONS")
    print("=" * 90)
    print("Ziel: Identifiziere genau welche List Operations fehlschlagen")
    print()
    
    # Finde alle fehlgeschlagenen List Operations
    failed_problems = find_all_failing_list_operations()
    
    # Analysiere Muster
    analyze_failure_patterns(failed_problems)
    
    print("\n" + "=" * 90)
    print("🎯 NÄCHSTE SCHRITTE:")
    if failed_problems:
        print("1. Behebe die identifizierten fehlgeschlagenen Probleme")
        print("2. Fokussiere auf die häufigsten Failure Patterns")
        print("3. Teste die Fixes systematisch")
    else:
        print("Keine fehlgeschlagenen Probleme gefunden - Problem liegt woanders")

if __name__ == "__main__":
    main()
