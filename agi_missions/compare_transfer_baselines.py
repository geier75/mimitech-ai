#!/usr/bin/env python3
"""
A/B-Test Vergleich: Alte vs. Neue Transfer-Baseline
Statistische Signifikanz-Tests und Effektgrößen-Analyse
"""

import json
import numpy as np
import argparse
import yaml
import logging
from typing import Dict, List, Tuple
from datetime import datetime
from scipy import stats
from dataclasses import dataclass, asdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StatisticalTest:
    """Ergebnis eines statistischen Tests"""
    metric: str
    old_mean: float
    new_mean: float
    old_std: float
    new_std: float
    difference: float
    percent_improvement: float
    t_statistic: float
    p_value: float
    is_significant: bool
    cohens_d: float
    confidence_interval_95: Tuple[float, float]
    interpretation: str

class TransferBaselineComparator:
    """Vergleicht alte und neue Transfer-Baselines statistisch"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.results = {}
        logger.info("🔬 Transfer Baseline Comparator initialisiert")
    
    def load_baseline_data(self, filename: str) -> List[Dict]:
        """Lädt Baseline-Daten aus JSON"""
        with open(filename, 'r') as f:
            data = json.load(f)
        return data['runs']
    
    def calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Berechnet Cohen's d Effektgröße"""
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                             (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
        return (np.mean(group2) - np.mean(group1)) / pooled_std
    
    def interpret_cohens_d(self, d: float) -> str:
        """Interpretiert Cohen's d Effektgröße"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "Kleiner Effekt"
        elif abs_d < 0.5:
            return "Mittlerer Effekt"
        elif abs_d < 0.8:
            return "Großer Effekt"
        else:
            return "Sehr großer Effekt"
    
    def perform_t_test(self, metric: str, old_values: np.ndarray, new_values: np.ndarray) -> StatisticalTest:
        """Führt t-Test für eine Metrik durch"""
        logger.info(f"📊 T-Test für {metric}")
        
        # Grundstatistiken
        old_mean = np.mean(old_values)
        new_mean = np.mean(new_values)
        old_std = np.std(old_values, ddof=1)
        new_std = np.std(new_values, ddof=1)
        
        # T-Test (unabhängige Stichproben)
        t_stat, p_value = stats.ttest_ind(new_values, old_values)
        
        # Effektgröße
        cohens_d = self.calculate_cohens_d(old_values, new_values)
        
        # Konfidenzintervall für Differenz
        diff_mean = new_mean - old_mean
        pooled_se = np.sqrt((old_std**2 / len(old_values)) + (new_std**2 / len(new_values)))
        df = len(old_values) + len(new_values) - 2
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        ci_lower = diff_mean - t_critical * pooled_se
        ci_upper = diff_mean + t_critical * pooled_se
        
        # Prozentuale Verbesserung
        percent_improvement = (diff_mean / old_mean) * 100 if old_mean != 0 else 0
        
        return StatisticalTest(
            metric=metric,
            old_mean=old_mean,
            new_mean=new_mean,
            old_std=old_std,
            new_std=new_std,
            difference=diff_mean,
            percent_improvement=percent_improvement,
            t_statistic=t_stat,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            cohens_d=cohens_d,
            confidence_interval_95=(ci_lower, ci_upper),
            interpretation=self.interpret_cohens_d(cohens_d)
        )
    
    def compare_baselines(self, old_file: str, new_file: str) -> Dict:
        """Führt vollständigen Baseline-Vergleich durch"""
        logger.info("🔬 Starte A/B-Test Vergleich")
        
        # Lade Daten
        old_runs = self.load_baseline_data(old_file)
        new_runs = self.load_baseline_data(new_file)
        
        logger.info(f"📈 Alte Baseline: {len(old_runs)} Runs")
        logger.info(f"🚀 Neue Baseline: {len(new_runs)} Runs")
        
        # Extrahiere Metriken
        metrics = ['accuracy', 'sharpe_ratio', 'quantum_speedup', 'latency_ms', 
                  'drawdown', 'confidence', 'transfer_effectiveness']
        
        test_results = {}
        
        for metric in metrics:
            old_values = np.array([run[metric] for run in old_runs if metric in run])
            new_values = np.array([run[metric] for run in new_runs if metric in run])
            
            if len(old_values) > 0 and len(new_values) > 0:
                test_result = self.perform_t_test(metric, old_values, new_values)
                test_results[metric] = test_result
                
                # Log Ergebnis
                significance = "✅ SIGNIFIKANT" if test_result.is_significant else "❌ NICHT SIGNIFIKANT"
                logger.info(f"  {metric}: {test_result.percent_improvement:+.1f}% {significance} (p={test_result.p_value:.4f})")
        
        # Gesamtbewertung
        overall_assessment = self._assess_overall_performance(test_results)
        
        return {
            "metadata": {
                "comparison_date": datetime.now().isoformat(),
                "old_baseline_file": old_file,
                "new_baseline_file": new_file,
                "alpha_level": self.alpha,
                "sample_sizes": {
                    "old_baseline": len(old_runs),
                    "new_baseline": len(new_runs)
                }
            },
            "statistical_tests": {metric: asdict(result) for metric, result in test_results.items()},
            "overall_assessment": overall_assessment
        }
    
    def _assess_overall_performance(self, test_results: Dict[str, StatisticalTest]) -> Dict:
        """Bewertet Gesamtperformance der neuen Baseline"""
        
        # Kritische Metriken
        critical_metrics = ['accuracy', 'sharpe_ratio', 'transfer_effectiveness']
        critical_improvements = []
        
        for metric in critical_metrics:
            if metric in test_results:
                result = test_results[metric]
                critical_improvements.append({
                    'metric': metric,
                    'improvement': result.percent_improvement,
                    'is_significant': result.is_significant,
                    'effect_size': result.interpretation
                })
        
        # Entscheidungslogik
        significant_improvements = sum(1 for imp in critical_improvements if imp['is_significant'] and imp['improvement'] > 0)
        significant_degradations = sum(1 for imp in critical_improvements if imp['is_significant'] and imp['improvement'] < 0)
        
        if significant_improvements >= 2 and significant_degradations == 0:
            recommendation = "DEPLOY_NEW_BASELINE"
            confidence = "HIGH"
        elif significant_improvements >= 1 and significant_degradations == 0:
            recommendation = "DEPLOY_NEW_BASELINE"
            confidence = "MEDIUM"
        elif significant_degradations > 0:
            recommendation = "KEEP_OLD_BASELINE"
            confidence = "HIGH"
        else:
            recommendation = "FURTHER_TESTING_NEEDED"
            confidence = "LOW"
        
        return {
            "critical_metrics_analysis": critical_improvements,
            "significant_improvements": significant_improvements,
            "significant_degradations": significant_degradations,
            "recommendation": recommendation,
            "confidence": confidence,
            "summary": self._generate_summary(test_results, recommendation)
        }
    
    def _generate_summary(self, test_results: Dict[str, StatisticalTest], recommendation: str) -> str:
        """Generiert Zusammenfassung der Ergebnisse"""
        accuracy_result = test_results.get('accuracy')
        sharpe_result = test_results.get('sharpe_ratio')
        
        summary_parts = []
        
        if accuracy_result:
            summary_parts.append(f"Accuracy: {accuracy_result.percent_improvement:+.1f}% ({'signifikant' if accuracy_result.is_significant else 'nicht signifikant'})")
        
        if sharpe_result:
            summary_parts.append(f"Sharpe Ratio: {sharpe_result.percent_improvement:+.1f}% ({'signifikant' if sharpe_result.is_significant else 'nicht signifikant'})")
        
        summary = "; ".join(summary_parts)
        
        if recommendation == "DEPLOY_NEW_BASELINE":
            summary += " → Empfehlung: Neue Baseline deployen"
        elif recommendation == "KEEP_OLD_BASELINE":
            summary += " → Empfehlung: Alte Baseline beibehalten"
        else:
            summary += " → Empfehlung: Weitere Tests erforderlich"
        
        return summary
    
    def save_report(self, results: Dict, output_file: str):
        """Speichert A/B-Test Report als YAML"""
        with open(output_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"📁 A/B-Test Report gespeichert: {output_file}")

def main():
    """Hauptfunktion"""
    parser = argparse.ArgumentParser(description='A/B-Test Vergleich Transfer-Baselines')
    parser.add_argument('old_baseline', help='JSON-Datei mit alter Baseline')
    parser.add_argument('new_baseline', help='JSON-Datei mit neuer Baseline')
    parser.add_argument('-o', '--output', default='ab_test_report.yaml', help='Output YAML-Datei')
    parser.add_argument('--alpha', type=float, default=0.05, help='Signifikanz-Level (default: 0.05)')
    
    args = parser.parse_args()
    
    # Führe Vergleich durch
    comparator = TransferBaselineComparator(alpha=args.alpha)
    results = comparator.compare_baselines(args.old_baseline, args.new_baseline)
    comparator.save_report(results, args.output)
    
    # Zeige Zusammenfassung
    assessment = results['overall_assessment']
    logger.info("\n" + "="*60)
    logger.info("🏆 A/B-TEST ERGEBNIS")
    logger.info("="*60)
    logger.info(f"📊 {assessment['summary']}")
    logger.info(f"🎯 Empfehlung: {assessment['recommendation']}")
    logger.info(f"🔒 Confidence: {assessment['confidence']}")
    logger.info("="*60)

if __name__ == "__main__":
    main()
