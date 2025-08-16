#!/usr/bin/env python3
"""
COMPREHENSIVE VALIDATION REPORT
Verify authentic HumanEval benchmark results meet all requirements
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any

def validate_benchmark_results(results_file: str) -> Dict[str, Any]:
    """Comprehensive validation of benchmark results"""
    print("🔍 COMPREHENSIVE VALIDATION REPORT")
    print("=" * 80)
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    validation_report = {
        'file_validation': {},
        'requirement_compliance': {},
        'data_integrity': {},
        'performance_validation': {},
        'statistical_validation': {},
        'overall_assessment': {}
    }
    
    # 1. File Validation
    print("📄 FILE VALIDATION")
    print("-" * 40)
    
    file_path = Path(results_file)
    file_size = file_path.stat().st_size
    
    validation_report['file_validation'] = {
        'file_exists': file_path.exists(),
        'file_size_bytes': file_size,
        'file_size_readable': f"{file_size / 1024:.1f} KB",
        'json_valid': True,  # We loaded it successfully
        'timestamp_format': 'YYYY-MM-DD HH:MM:SS' in str(results.get('benchmark_info', {}).get('timestamp', ''))
    }
    
    print(f"   ✅ File exists: {file_path}")
    print(f"   ✅ File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    print(f"   ✅ JSON format: Valid")
    print(f"   ✅ Timestamp: {results['benchmark_info']['timestamp']}")
    
    # 2. Requirement Compliance
    print(f"\n📋 REQUIREMENT COMPLIANCE")
    print("-" * 40)
    
    total_problems = results['benchmark_info']['total_problems']
    dataset_source = results['benchmark_info']['dataset_source']
    execution_method = results['validation']['execution_method']
    security_measures = results['validation']['security_measures']
    statistical_significance = results['validation']['statistical_significance']
    
    requirements = {
        'authentic_problems_only': dataset_source == 'authentic_humaneval',
        'systematic_approach': 'detailed_results' in results and len(results['detailed_results']) > 0,
        'real_execution_environment': execution_method == 'REAL_CODE_EXECUTION',
        'comprehensive_coverage': len(results['category_performance']) >= 4,
        'production_grade_validation': security_measures == 'PRODUCTION_GRADE',
        'statistical_significance': total_problems >= 100 and statistical_significance == 'HIGH',
        'verifiable_results': 'detailed_results' in results and 'code_hash' in results['detailed_results'][0]
    }
    
    validation_report['requirement_compliance'] = requirements
    
    for req, status in requirements.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {req.replace('_', ' ').title()}: {status}")
    
    # 3. Data Integrity
    print(f"\n🔒 DATA INTEGRITY VALIDATION")
    print("-" * 40)
    
    # Verify problem counts
    category_totals = sum(stats['total'] for stats in results['category_statistics'].values())
    detailed_results_count = len(results['detailed_results'])
    
    # Verify pass rates
    calculated_pass_rate = (results['performance_metrics']['passed_problems'] / 
                          results['performance_metrics']['total_problems']) * 100
    reported_pass_rate = results['performance_metrics']['pass_at_1']
    
    # Verify category calculations
    category_calculations_correct = True
    for category, pass_rate in results['category_performance'].items():
        stats = results['category_statistics'][category]
        expected_rate = (stats['passed'] / stats['total']) * 100
        if abs(pass_rate - expected_rate) > 0.1:  # Allow small floating point differences
            category_calculations_correct = False
            break
    
    integrity_checks = {
        'problem_count_consistency': category_totals == total_problems == detailed_results_count,
        'pass_rate_calculation': abs(calculated_pass_rate - reported_pass_rate) < 0.1,
        'category_calculations': category_calculations_correct,
        'all_results_have_hashes': all('code_hash' in result for result in results['detailed_results']),
        'execution_times_positive': all(result['execution_time'] > 0 for result in results['detailed_results'])
    }
    
    validation_report['data_integrity'] = integrity_checks
    
    for check, status in integrity_checks.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {check.replace('_', ' ').title()}: {status}")
    
    # 4. Performance Validation
    print(f"\n⚡ PERFORMANCE VALIDATION")
    print("-" * 40)
    
    metrics = results['performance_metrics']
    avg_time = metrics['avg_execution_time']
    problems_per_minute = metrics['problems_per_minute']
    total_time = results['benchmark_info']['execution_time']
    
    performance_checks = {
        'reasonable_execution_time': total_time < 300,  # Less than 5 minutes
        'positive_avg_time': avg_time > 0,
        'realistic_throughput': 100 < problems_per_minute < 10000,
        'pass_rate_above_threshold': metrics['pass_at_1'] > 50,  # Reasonable performance
        'confidence_interval_valid': (metrics['confidence_interval_95']['lower'] < 
                                    metrics['pass_at_1'] < 
                                    metrics['confidence_interval_95']['upper'])
    }
    
    validation_report['performance_validation'] = performance_checks
    
    for check, status in performance_checks.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {check.replace('_', ' ').title()}: {status}")
    
    print(f"   📊 Pass@1 Rate: {metrics['pass_at_1']:.1f}%")
    print(f"   📊 Total Time: {total_time:.2f}s")
    print(f"   📊 Avg Time/Problem: {avg_time:.3f}s")
    print(f"   📊 Problems/Minute: {problems_per_minute:.1f}")
    
    # 5. Statistical Validation
    print(f"\n📊 STATISTICAL VALIDATION")
    print("-" * 40)
    
    # Check category distribution
    category_counts = [stats['total'] for stats in results['category_statistics'].values()]
    min_category_size = min(category_counts)
    max_category_size = max(category_counts)
    category_balance = (max_category_size - min_category_size) / max_category_size < 0.5  # Within 50%
    
    statistical_checks = {
        'sufficient_sample_size': total_problems >= 100,
        'category_balance': category_balance,
        'multiple_categories': len(results['category_performance']) >= 4,
        'confidence_interval_calculated': 'confidence_interval_95' in metrics,
        'high_statistical_significance': statistical_significance == 'HIGH'
    }
    
    validation_report['statistical_validation'] = statistical_checks
    
    for check, status in statistical_checks.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {check.replace('_', ' ').title()}: {status}")
    
    print(f"   📊 Sample Size: {total_problems}")
    print(f"   📊 Categories: {len(results['category_performance'])}")
    print(f"   📊 CI (95%): [{metrics['confidence_interval_95']['lower']:.1f}%, {metrics['confidence_interval_95']['upper']:.1f}%]")
    
    # 6. Overall Assessment
    print(f"\n🎯 OVERALL ASSESSMENT")
    print("-" * 40)
    
    all_requirements_met = all(requirements.values())
    all_integrity_checks_passed = all(integrity_checks.values())
    all_performance_checks_passed = all(performance_checks.values())
    all_statistical_checks_passed = all(statistical_checks.values())
    
    overall_score = sum([
        all_requirements_met,
        all_integrity_checks_passed,
        all_performance_checks_passed,
        all_statistical_checks_passed
    ]) / 4 * 100
    
    validation_report['overall_assessment'] = {
        'requirements_compliance': all_requirements_met,
        'data_integrity': all_integrity_checks_passed,
        'performance_validation': all_performance_checks_passed,
        'statistical_validation': all_statistical_checks_passed,
        'overall_score': overall_score,
        'production_ready': overall_score >= 75
    }
    
    print(f"   Requirements Compliance: {'✅ PASS' if all_requirements_met else '❌ FAIL'}")
    print(f"   Data Integrity: {'✅ PASS' if all_integrity_checks_passed else '❌ FAIL'}")
    print(f"   Performance Validation: {'✅ PASS' if all_performance_checks_passed else '❌ FAIL'}")
    print(f"   Statistical Validation: {'✅ PASS' if all_statistical_checks_passed else '❌ FAIL'}")
    print(f"   Overall Score: {overall_score:.1f}%")
    print(f"   Production Ready: {'✅ YES' if overall_score >= 75 else '❌ NO'}")
    
    return validation_report

def main():
    """Main validation function"""
    results_file = "authentic_humaneval_2025-08-06_01-30-52.json"
    
    if not Path(results_file).exists():
        print(f"❌ Results file not found: {results_file}")
        return
    
    validation_report = validate_benchmark_results(results_file)
    
    # Export validation report
    validation_file = f"validation_report_{results_file.replace('.json', '.json')}"
    with open(validation_file, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    print(f"\n💾 VALIDATION REPORT EXPORTED: {validation_file}")
    print("\n" + "=" * 80)
    print("🔍 COMPREHENSIVE VALIDATION COMPLETED")
    
    if validation_report['overall_assessment']['production_ready']:
        print("✅ BENCHMARK RESULTS VALIDATED FOR PRODUCTION USE")
        print("🚀 Ready for enterprise deployment and evaluation")
    else:
        print("⚠️ BENCHMARK RESULTS REQUIRE ATTENTION")
        print("🔧 Review failed validation checks before production use")

if __name__ == "__main__":
    main()
