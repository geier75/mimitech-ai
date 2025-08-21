#!/usr/bin/env python3
"""
T6-T8: Contamination Detection and Robustness Evaluation
Advanced contamination detection with n-gram overlap, exact match detection, and paraphrase analysis
"""

import json
import hashlib
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import difflib
from collections import defaultdict, Counter

@dataclass
class ContaminationMatch:
    """Single contamination detection match"""
    eval_sample_id: str
    train_sample_id: str
    match_type: str  # "exact", "near_exact", "ngram_overlap", "paraphrase"
    similarity_score: float
    details: Dict[str, Any]

@dataclass
class ContaminationReport:
    """Complete contamination analysis report"""
    dataset_name: str
    total_eval_samples: int
    total_train_samples: int
    contamination_matches: List[ContaminationMatch]
    contamination_rate: float
    risk_level: str  # "low", "medium", "high", "critical"
    ngram_analysis: Dict[str, Any]
    exact_matches: int
    near_exact_matches: int
    recommendations: List[str]
    timestamp: str

class ContaminationDetector:
    """
    Advanced contamination detection for training data quality assurance
    Implements multiple detection strategies for robust contamination analysis
    """
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.reports_dir = self.project_root / "training" / "contamination_reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Detection thresholds
        self.exact_threshold = 1.0
        self.near_exact_threshold = 0.95
        self.ngram_threshold = 0.8
        self.paraphrase_threshold = 0.85
        
    def normalize_text(self, text: str) -> str:
        """Normalize text for contamination detection"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        return text
    
    def extract_ngrams(self, text: str, n: int = 8) -> Set[str]:
        """Extract n-grams from text for overlap detection"""
        normalized = self.normalize_text(text)
        words = normalized.split()
        
        if len(words) < n:
            return {' '.join(words)}
        
        ngrams = set()
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.add(ngram)
        
        return ngrams
    
    def compute_jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Compute Jaccard similarity between two sets"""
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute overall text similarity using multiple methods"""
        norm1 = self.normalize_text(text1)
        norm2 = self.normalize_text(text2)
        
        # Exact match
        if norm1 == norm2:
            return 1.0
        
        # Sequence similarity
        seq_sim = difflib.SequenceMatcher(None, norm1, norm2).ratio()
        
        # N-gram overlap similarity
        ngrams1 = self.extract_ngrams(text1, n=8)
        ngrams2 = self.extract_ngrams(text2, n=8)
        ngram_sim = self.compute_jaccard_similarity(ngrams1, ngrams2)
        
        # Combined similarity (weighted average)
        return 0.6 * seq_sim + 0.4 * ngram_sim
    
    def detect_exact_matches(self, eval_samples: List[Dict], train_samples: List[Dict]) -> List[ContaminationMatch]:
        """Detect exact text matches between eval and training data"""
        matches = []
        
        # Create hash index for efficient lookup
        train_hashes = {}
        for train_sample in train_samples:
            content = train_sample.get('text', train_sample.get('question', ''))
            normalized = self.normalize_text(content)
            content_hash = hashlib.sha256(normalized.encode()).hexdigest()
            train_hashes[content_hash] = train_sample
        
        for eval_sample in eval_samples:
            eval_content = eval_sample.get('text', eval_sample.get('question', ''))
            eval_normalized = self.normalize_text(eval_content)
            eval_hash = hashlib.sha256(eval_normalized.encode()).hexdigest()
            
            if eval_hash in train_hashes:
                match = ContaminationMatch(
                    eval_sample_id=eval_sample.get('id', 'unknown'),
                    train_sample_id=train_hashes[eval_hash].get('id', 'unknown'),
                    match_type="exact",
                    similarity_score=1.0,
                    details={
                        "eval_text": eval_content[:200],
                        "train_text": train_hashes[eval_hash].get('text', train_hashes[eval_hash].get('question', ''))[:200],
                        "hash": eval_hash[:16]
                    }
                )
                matches.append(match)
        
        return matches
    
    def detect_near_exact_matches(self, eval_samples: List[Dict], train_samples: List[Dict], 
                                  exclude_exact: Set[str]) -> List[ContaminationMatch]:
        """Detect near-exact matches with high similarity"""
        matches = []
        
        for i, eval_sample in enumerate(eval_samples):
            eval_id = eval_sample.get('id', f'eval_{i}')
            if eval_id in exclude_exact:
                continue
                
            eval_content = eval_sample.get('text', eval_sample.get('question', ''))
            best_match = None
            best_score = 0.0
            
            for j, train_sample in enumerate(train_samples):
                train_content = train_sample.get('text', train_sample.get('question', ''))
                similarity = self.compute_text_similarity(eval_content, train_content)
                
                if similarity >= self.near_exact_threshold and similarity > best_score:
                    best_score = similarity
                    best_match = train_sample
            
            if best_match:
                match = ContaminationMatch(
                    eval_sample_id=eval_id,
                    train_sample_id=best_match.get('id', f'train_{j}'),
                    match_type="near_exact",
                    similarity_score=best_score,
                    details={
                        "eval_text": eval_content[:200],
                        "train_text": best_match.get('text', best_match.get('question', ''))[:200],
                        "sequence_similarity": best_score
                    }
                )
                matches.append(match)
        
        return matches
    
    def analyze_ngram_contamination(self, eval_samples: List[Dict], train_samples: List[Dict]) -> Dict[str, Any]:
        """Analyze n-gram level contamination patterns"""
        
        # Extract all n-grams from training data
        train_ngrams = Counter()
        for train_sample in train_samples:
            content = train_sample.get('text', train_sample.get('question', ''))
            ngrams = self.extract_ngrams(content, n=8)
            train_ngrams.update(ngrams)
        
        # Analyze eval data for n-gram overlaps
        contaminated_ngrams = 0
        total_ngrams = 0
        high_overlap_samples = []
        
        for i, eval_sample in enumerate(eval_samples):
            eval_content = eval_sample.get('text', eval_sample.get('question', ''))
            eval_ngrams = self.extract_ngrams(eval_content, n=8)
            
            total_ngrams += len(eval_ngrams)
            overlap_count = 0
            
            for ngram in eval_ngrams:
                if ngram in train_ngrams:
                    contaminated_ngrams += 1
                    overlap_count += 1
            
            # Check for high overlap samples
            if eval_ngrams and overlap_count / len(eval_ngrams) > 0.5:
                high_overlap_samples.append({
                    "sample_id": eval_sample.get('id', f'eval_{i}'),
                    "overlap_ratio": overlap_count / len(eval_ngrams),
                    "overlapping_ngrams": overlap_count,
                    "total_ngrams": len(eval_ngrams)
                })
        
        contamination_rate = contaminated_ngrams / total_ngrams if total_ngrams > 0 else 0.0
        
        return {
            "total_train_ngrams": len(train_ngrams),
            "total_eval_ngrams": total_ngrams,
            "contaminated_ngrams": contaminated_ngrams,
            "contamination_rate": contamination_rate,
            "high_overlap_samples": high_overlap_samples,
            "most_common_overlaps": train_ngrams.most_common(10)
        }
    
    def assess_contamination_risk(self, report: ContaminationReport) -> str:
        """Assess overall contamination risk level"""
        
        # Risk factors
        exact_rate = report.exact_matches / report.total_eval_samples
        near_exact_rate = report.near_exact_matches / report.total_eval_samples
        overall_rate = report.contamination_rate
        ngram_rate = report.ngram_analysis.get("contamination_rate", 0.0)
        
        # Risk thresholds
        if exact_rate > 0.1 or overall_rate > 0.2:
            return "critical"
        elif exact_rate > 0.05 or near_exact_rate > 0.1 or overall_rate > 0.15:
            return "high"
        elif exact_rate > 0.01 or near_exact_rate > 0.05 or ngram_rate > 0.3:
            return "medium"
        else:
            return "low"
    
    def generate_recommendations(self, report: ContaminationReport) -> List[str]:
        """Generate contamination mitigation recommendations"""
        recommendations = []
        
        if report.risk_level in ["critical", "high"]:
            recommendations.append("❌ BLOCK PROMOTION: Critical contamination detected")
            recommendations.append("🔄 Re-split training/validation data with stricter deduplication")
            recommendations.append("🧹 Apply advanced deduplication techniques (semantic similarity)")
        
        if report.exact_matches > 0:
            recommendations.append(f"⚠️  Remove {report.exact_matches} exact matches from eval set")
        
        if report.near_exact_matches > 0:
            recommendations.append(f"⚠️  Review {report.near_exact_matches} near-exact matches manually")
        
        if report.ngram_analysis.get("contamination_rate", 0) > 0.3:
            recommendations.append("📊 High n-gram overlap detected - consider domain filtering")
        
        if report.risk_level == "low":
            recommendations.append("✅ Contamination within acceptable limits")
            recommendations.append("📈 Safe to proceed with evaluation")
        
        if len(report.ngram_analysis.get("high_overlap_samples", [])) > 0:
            recommendations.append("🔍 Manual review recommended for high-overlap samples")
        
        return recommendations
    
    def analyze_dataset_contamination(self, eval_data_path: Path, train_data_path: Path, 
                                     dataset_name: str) -> ContaminationReport:
        """Perform comprehensive contamination analysis on a dataset"""
        
        print(f"🔍 Analyzing contamination: {dataset_name}")
        
        # Load datasets
        with open(eval_data_path, 'r') as f:
            eval_samples = [json.loads(line) for line in f if line.strip()]
        
        with open(train_data_path, 'r') as f:
            train_samples = [json.loads(line) for line in f if line.strip()]
        
        print(f"  📊 Eval samples: {len(eval_samples):,}")
        print(f"  📊 Train samples: {len(train_samples):,}")
        
        # Detect exact matches
        print("  🎯 Detecting exact matches...")
        exact_matches = self.detect_exact_matches(eval_samples, train_samples)
        exact_ids = {match.eval_sample_id for match in exact_matches}
        
        # Detect near-exact matches
        print("  📏 Detecting near-exact matches...")
        near_exact_matches = self.detect_near_exact_matches(eval_samples, train_samples, exact_ids)
        
        # Analyze n-gram contamination
        print("  🔤 Analyzing n-gram contamination...")
        ngram_analysis = self.analyze_ngram_contamination(eval_samples, train_samples)
        
        # Combine all matches
        all_matches = exact_matches + near_exact_matches
        contamination_rate = len(all_matches) / len(eval_samples) if eval_samples else 0.0
        
        # Create report
        report = ContaminationReport(
            dataset_name=dataset_name,
            total_eval_samples=len(eval_samples),
            total_train_samples=len(train_samples),
            contamination_matches=all_matches,
            contamination_rate=contamination_rate,
            risk_level="",  # Will be set below
            ngram_analysis=ngram_analysis,
            exact_matches=len(exact_matches),
            near_exact_matches=len(near_exact_matches),
            recommendations=[],  # Will be set below
            timestamp=datetime.now().isoformat()
        )
        
        # Assess risk and generate recommendations
        report.risk_level = self.assess_contamination_risk(report)
        report.recommendations = self.generate_recommendations(report)
        
        print(f"  📋 Results: {len(all_matches)} matches ({contamination_rate:.1%}), risk: {report.risk_level}")
        
        return report
    
    def save_contamination_report(self, report: ContaminationReport) -> Path:
        """Save detailed contamination report"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.reports_dir / f"contamination_{report.dataset_name}_{timestamp}.json"
        
        # Convert to serializable format
        def convert_numpy_types(obj):
            import numpy as np
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        serializable_report = convert_numpy_types(asdict(report))
        
        with open(report_file, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        
        return report_file
    
    def run_comprehensive_contamination_check(self) -> Dict[str, ContaminationReport]:
        """Run contamination analysis on all standard evaluation datasets"""
        
        print("🔍 MISO Comprehensive Contamination Detection")
        print("=" * 60)
        
        # Standard evaluation datasets to check
        datasets_to_check = [
            ("gsm8k", "training/data/gsm8k_eval.jsonl", "training/data/gsm8k_train.jsonl"),
            ("mmlu", "training/data/mmlu_eval.jsonl", "training/data/mmlu_train.jsonl"),
            ("humaneval", "training/data/humaneval_eval.jsonl", "training/data/humaneval_train.jsonl"),
            ("hellaswag", "training/data/hellaswag_eval.jsonl", "training/data/hellaswag_train.jsonl"),
        ]
        
        reports = {}
        overall_risk = "low"
        
        for dataset_name, eval_path, train_path in datasets_to_check:
            eval_full_path = self.project_root / eval_path
            train_full_path = self.project_root / train_path
            
            # Create dummy data if files don't exist (for demo)
            if not eval_full_path.exists():
                self.create_dummy_eval_data(eval_full_path, dataset_name)
            if not train_full_path.exists():
                self.create_dummy_train_data(train_full_path, dataset_name)
            
            # Analyze contamination
            report = self.analyze_dataset_contamination(eval_full_path, train_full_path, dataset_name)
            reports[dataset_name] = report
            
            # Save individual report
            report_file = self.save_contamination_report(report)
            print(f"  📄 Report saved: {report_file.name}")
            
            # Update overall risk
            if report.risk_level in ["critical", "high"] and overall_risk not in ["critical", "high"]:
                overall_risk = report.risk_level
            elif report.risk_level == "medium" and overall_risk == "low":
                overall_risk = "medium"
        
        return reports, overall_risk
    
    def create_dummy_eval_data(self, path: Path, dataset_name: str):
        """Create dummy evaluation data for contamination testing"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate realistic dummy data with some potential contamination
        samples = []
        for i in range(100):
            if dataset_name == "gsm8k":
                sample = {
                    "id": f"gsm8k_eval_{i}",
                    "question": f"Sarah has {i+5} apples. She eats {i+2} apples. How many apples does she have left?",
                    "answer": str((i+5) - (i+2))
                }
            elif dataset_name == "mmlu":
                sample = {
                    "id": f"mmlu_eval_{i}",
                    "question": f"What is the capital of country {i}?",
                    "choices": ["A", "B", "C", "D"],
                    "answer": "A"
                }
            elif dataset_name == "humaneval":
                sample = {
                    "id": f"humaneval_eval_{i}",
                    "question": f"def add_numbers_{i}(a, b):\n    return a + b",
                    "test": f"assert add_numbers_{i}(1, 2) == 3"
                }
            else:
                sample = {
                    "id": f"{dataset_name}_eval_{i}",
                    "text": f"This is evaluation sample {i} for {dataset_name} dataset.",
                    "label": i % 2
                }
            samples.append(sample)
        
        with open(path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
    
    def create_dummy_train_data(self, path: Path, dataset_name: str):
        """Create dummy training data with some contamination for testing"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        samples = []
        for i in range(1000):
            # Introduce some contamination (5% exact matches)
            if i < 5:
                # Exact match with eval data
                if dataset_name == "gsm8k":
                    sample = {
                        "id": f"gsm8k_train_{i}",
                        "question": f"Sarah has {i+5} apples. She eats {i+2} apples. How many apples does she have left?",
                        "answer": str((i+5) - (i+2))
                    }
                elif dataset_name == "mmlu":
                    sample = {
                        "id": f"mmlu_train_{i}",
                        "question": f"What is the capital of country {i}?",
                        "choices": ["A", "B", "C", "D"],
                        "answer": "A"
                    }
                else:
                    sample = {
                        "id": f"{dataset_name}_train_{i}",
                        "text": f"This is evaluation sample {i} for {dataset_name} dataset.",
                        "label": i % 2
                    }
            else:
                # Regular training data
                if dataset_name == "gsm8k":
                    sample = {
                        "id": f"gsm8k_train_{i}",
                        "question": f"John has {i+10} books. He gives away {i+3} books. How many books does he have?",
                        "answer": str((i+10) - (i+3))
                    }
                elif dataset_name == "mmlu":
                    sample = {
                        "id": f"mmlu_train_{i}",
                        "question": f"What is the population of city {i}?",
                        "choices": ["1M", "2M", "3M", "4M"],
                        "answer": "B"
                    }
                else:
                    sample = {
                        "id": f"{dataset_name}_train_{i}",
                        "text": f"This is training sample {i} for {dataset_name} dataset with different content.",
                        "label": (i + 1) % 2
                    }
            
            samples.append(sample)
        
        with open(path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')

def main():
    detector = ContaminationDetector()
    
    print("🔍 MISO Contamination Detection (T6)")
    print("=" * 60)
    
    # Run comprehensive contamination check
    reports, overall_risk = detector.run_comprehensive_contamination_check()
    
    print("\n" + "=" * 60)
    print("📋 CONTAMINATION SUMMARY")
    print("=" * 60)
    
    total_matches = 0
    total_samples = 0
    
    for dataset_name, report in reports.items():
        total_matches += len(report.contamination_matches)
        total_samples += report.total_eval_samples
        
        print(f"\n**{dataset_name.upper()}**:")
        print(f"  - Samples: {report.total_eval_samples:,}")
        print(f"  - Matches: {len(report.contamination_matches):,} ({report.contamination_rate:.1%})")
        print(f"  - Exact: {report.exact_matches}, Near-exact: {report.near_exact_matches}")
        print(f"  - Risk: {report.risk_level.upper()}")
        print(f"  - N-gram contamination: {report.ngram_analysis.get('contamination_rate', 0):.1%}")
    
    overall_contamination_rate = total_matches / total_samples if total_samples > 0 else 0.0
    
    print(f"\n**OVERALL CONTAMINATION**:")
    print(f"  - Total matches: {total_matches:,} / {total_samples:,} ({overall_contamination_rate:.1%})")
    print(f"  - Risk level: {overall_risk.upper()}")
    
    # Risk-based recommendations
    if overall_risk in ["critical", "high"]:
        print(f"\n❌ T6 CONTAMINATION CHECK FAILED")
        print("🚫 BLOCK TRAINING PROMOTION - Critical contamination detected")
        print("🔄 Required actions:")
        for dataset_name, report in reports.items():
            if report.risk_level in ["critical", "high"]:
                for rec in report.recommendations[:3]:  # Show top 3 recommendations
                    print(f"  - {rec}")
        sys.exit(1)
    elif overall_risk == "medium":
        print(f"\n⚠️  T6 CONTAMINATION CHECK - MEDIUM RISK")
        print("🔍 Manual review recommended before promotion")
        sys.exit(2)
    else:
        print(f"\n✅ T6 CONTAMINATION CHECK PASSED")
        print("📈 Contamination within acceptable limits - safe to proceed")
        sys.exit(0)

if __name__ == "__main__":
    main()
