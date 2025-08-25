"""
Dataset integrity validation and minimum sample count enforcement
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Minimum sample counts per benchmark (Quality Gates)
MINIMUM_SAMPLE_COUNTS = {
    "mmlu": 14000,
    "hellaswag": 800, 
    "winogrande": 800,
    "piqa": 800,
    "arc": 1000,
    # Additional benchmarks
    "gsm8k": 500,
    "humaneval": 164,
    "truthfulqa": 400,
    "boolq": 600,
    "race": 500,
    "squad": 1000,
    "cnn_dailymail": 1000
}

@dataclass
class DatasetStatus:
    """Dataset validation status"""
    name: str
    path: Path
    exists: bool
    sample_count: int
    min_required: int
    checksum_valid: bool
    status: str  # "PASS", "FAIL", "WARNING"
    error_message: str = ""

class DatasetIntegrityValidator:
    """Validates dataset integrity and sample counts"""
    
    def __init__(self, datasets_root: Path = None):
        if datasets_root is None:
            # Default to project datasets directory
            project_root = Path(__file__).parent.parent.parent
            datasets_root = project_root / "datasets"
        
        self.datasets_root = datasets_root
        self.checksum_manager = None
        
    def _import_checksum_manager(self):
        """Lazy import to avoid circular dependencies"""
        if self.checksum_manager is None:
            from .checksum_manager import ChecksumManager
            self.checksum_manager = ChecksumManager(self.datasets_root)
        return self.checksum_manager
    
    def count_samples_in_dataset(self, dataset_path: Path) -> int:
        """Count samples in dataset file"""
        if not dataset_path.exists():
            return 0
            
        try:
            # Handle different file formats
            if dataset_path.suffix == '.json':
                with open(dataset_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return len(data)
                    elif isinstance(data, dict) and 'examples' in data:
                        return len(data['examples'])
                    elif isinstance(data, dict) and 'data' in data:
                        return len(data['data'])
                    else:
                        return len(data)
                        
            elif dataset_path.suffix == '.jsonl':
                with open(dataset_path, 'r') as f:
                    return sum(1 for line in f if line.strip())
                    
            elif dataset_path.suffix in ['.csv', '.tsv']:
                with open(dataset_path, 'r') as f:
                    lines = f.readlines()
                    return max(0, len(lines) - 1)  # Subtract header
                    
            elif dataset_path.suffix == '.txt':
                with open(dataset_path, 'r') as f:
                    return len([line for line in f if line.strip()])
                    
            else:
                logger.warning(f"Unknown dataset format: {dataset_path.suffix}")
                return 0
                
        except Exception as e:
            logger.error(f"Error counting samples in {dataset_path}: {e}")
            return 0
    
    def validate_dataset(self, dataset_name: str) -> DatasetStatus:
        """Validate single dataset"""
        # Find dataset file(s)
        dataset_patterns = [
            f"{dataset_name}.json",
            f"{dataset_name}.jsonl", 
            f"{dataset_name}_test.json",
            f"{dataset_name}_valid.json",
            f"{dataset_name}.csv"
        ]
        
        dataset_path = None
        for pattern in dataset_patterns:
            candidate = self.datasets_root / pattern
            if candidate.exists():
                dataset_path = candidate
                break
        
        if dataset_path is None:
            return DatasetStatus(
                name=dataset_name,
                path=self.datasets_root / f"{dataset_name}.json",
                exists=False,
                sample_count=0,
                min_required=MINIMUM_SAMPLE_COUNTS.get(dataset_name.lower(), 0),
                checksum_valid=False,
                status="FAIL",
                error_message=f"Dataset file not found for {dataset_name}"
            )
        
        # Count samples
        sample_count = self.count_samples_in_dataset(dataset_path)
        min_required = MINIMUM_SAMPLE_COUNTS.get(dataset_name.lower(), 0)
        
        # Validate checksum
        checksum_manager = self._import_checksum_manager()
        checksum_valid = checksum_manager.verify_file_integrity(dataset_path)
        
        # Determine status
        if sample_count < min_required:
            status = "FAIL"
            error_message = f"Insufficient samples: {sample_count} < {min_required}"
        elif not checksum_valid:
            status = "WARNING"
            error_message = "Checksum validation failed - data may be corrupted"
        else:
            status = "PASS"
            error_message = ""
        
        return DatasetStatus(
            name=dataset_name,
            path=dataset_path,
            exists=True,
            sample_count=sample_count,
            min_required=min_required,
            checksum_valid=checksum_valid,
            status=status,
            error_message=error_message
        )
    
    def validate_all_required_datasets(self, required_datasets: List[str] = None) -> Dict[str, DatasetStatus]:
        """Validate all required datasets"""
        if required_datasets is None:
            required_datasets = list(MINIMUM_SAMPLE_COUNTS.keys())
        
        results = {}
        for dataset_name in required_datasets:
            results[dataset_name] = self.validate_dataset(dataset_name)
        
        return results
    
    def check_dataset_integrity_gate(self, required_datasets: List[str] = None) -> Tuple[bool, List[str]]:
        """Quality Gate: Check if all datasets pass integrity requirements"""
        validation_results = self.validate_all_required_datasets(required_datasets)
        
        failed_datasets = []
        for dataset_name, status in validation_results.items():
            if status.status == "FAIL":
                failed_datasets.append(f"{dataset_name}: {status.error_message}")
                
        all_passed = len(failed_datasets) == 0
        
        if all_passed:
            logger.info("✅ All dataset integrity checks passed")
        else:
            logger.error(f"❌ Dataset integrity check failed: {len(failed_datasets)} datasets failed")
            for failure in failed_datasets:
                logger.error(f"   - {failure}")
        
        return all_passed, failed_datasets
    
    def generate_dataset_report(self, required_datasets: List[str] = None) -> Dict[str, Any]:
        """Generate comprehensive dataset validation report"""
        validation_results = self.validate_all_required_datasets(required_datasets)
        
        total_datasets = len(validation_results)
        passed = len([r for r in validation_results.values() if r.status == "PASS"])
        failed = len([r for r in validation_results.values() if r.status == "FAIL"])
        warnings = len([r for r in validation_results.values() if r.status == "WARNING"])
        
        total_samples = sum(r.sample_count for r in validation_results.values())
        total_required = sum(r.min_required for r in validation_results.values())
        
        return {
            "summary": {
                "total_datasets": total_datasets,
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
                "total_samples": total_samples,
                "total_required": total_required
            },
            "datasets": {
                name: {
                    "path": str(status.path),
                    "exists": status.exists,
                    "sample_count": status.sample_count,
                    "min_required": status.min_required,
                    "checksum_valid": status.checksum_valid,
                    "status": status.status,
                    "error_message": status.error_message
                }
                for name, status in validation_results.items()
            }
        }
