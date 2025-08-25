#!/usr/bin/env python3
"""
MISO Training Data Integration Script
====================================

Integrates VXOR Phase-1 training data with MISO component training pipelines.
Supports T-Mathematics Engine, M-PRIME Framework, ECHO-PRIME, and other modules.

Author: MISO Development Team
Version: 1.0.0
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Iterator, Optional, Union
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for MISO training integration."""
    data_dir: Path
    batch_size: int = 32
    max_seq_length: int = 512
    train_split_ratio: float = 0.96
    val_split_ratio: float = 0.02
    test_split_ratio: float = 0.02
    seed: int = 42
    device: str = "auto"
    backend: str = "auto"  # auto, pytorch, mlx
    
    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        if self.device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

class MISOTrainingDataset(Dataset):
    """PyTorch Dataset for MISO training data."""
    
    def __init__(self, jsonl_files: List[Path], max_seq_length: int = 512):
        self.samples = []
        self.max_seq_length = max_seq_length
        
        for jsonl_file in jsonl_files:
            self._load_jsonl_file(jsonl_file)
        
        logger.info(f"Loaded {len(self.samples)} samples from {len(jsonl_files)} files")
    
    def _load_jsonl_file(self, jsonl_file: Path):
        """Load samples from a JSONL file."""
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        if 'prompt' in data and 'completion' in data:
                            self.samples.append({
                                'prompt': data['prompt'].strip(),
                                'completion': data['completion'].strip(),
                                'source_file': jsonl_file.name,
                                'line_num': line_num
                            })
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON in {jsonl_file}:{line_num}: {e}")
        except Exception as e:
            logger.error(f"Error loading {jsonl_file}: {e}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.samples[idx]

class MISODataIntegrator:
    """Main class for integrating VXOR training data with MISO components."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.datasets = {}
        self.data_loaders = {}
        
        # Set random seeds for reproducibility
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        logger.info(f"Initialized MISO Data Integrator")
        logger.info(f"Device: {config.device}, Backend: {config.backend}")
    
    def discover_datasets(self) -> Dict[str, List[Path]]:
        """Discover available JSONL datasets."""
        datasets = {}
        
        for split in ['train', 'val', 'test']:
            split_dir = self.config.data_dir / 'jsonl' / split
            if split_dir.exists():
                jsonl_files = []
                for jsonl_file in split_dir.glob('*.jsonl'):
                    if not jsonl_file.name.startswith('._'):
                        jsonl_files.append(jsonl_file)
                datasets[split] = sorted(jsonl_files)
                logger.info(f"Found {len(jsonl_files)} datasets in {split} split")
        
        return datasets
    
    def load_datasets(self) -> None:
        """Load all available datasets."""
        dataset_files = self.discover_datasets()
        
        for split, files in dataset_files.items():
            if files:
                self.datasets[split] = MISOTrainingDataset(
                    files, 
                    max_seq_length=self.config.max_seq_length
                )
                logger.info(f"Loaded {split} dataset: {len(self.datasets[split])} samples")
    
    def create_data_loaders(self) -> None:
        """Create PyTorch DataLoaders for training."""
        for split, dataset in self.datasets.items():
            shuffle = (split == 'train')
            self.data_loaders[split] = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                num_workers=0,  # Avoid multiprocessing on external drive
                pin_memory=(self.config.device != 'cpu')
            )
            logger.info(f"Created {split} DataLoader: {len(self.data_loaders[split])} batches")
    
    def get_domain_specific_data(self, domain_pattern: str) -> Dict[str, List[Dict]]:
        """Extract samples for specific AGI domains."""
        domain_data = {'train': [], 'val': [], 'test': []}
        
        for split, dataset in self.datasets.items():
            for sample in dataset.samples:
                if domain_pattern.lower() in sample['source_file'].lower():
                    domain_data[split].append(sample)
        
        total_samples = sum(len(samples) for samples in domain_data.values())
        logger.info(f"Extracted {total_samples} samples for domain '{domain_pattern}'")
        
        return domain_data
    
    def prepare_t_mathematics_data(self) -> Dict[str, List[Dict]]:
        """Prepare data specifically for T-Mathematics Engine training."""
        math_domains = [
            'mathematics_logic',
            'probability_statistics', 
            'abstract_reasoning',
            'pattern_recognition'
        ]
        
        math_data = {'train': [], 'val': [], 'test': []}
        
        for domain in math_domains:
            domain_samples = self.get_domain_specific_data(domain)
            for split in math_data:
                math_data[split].extend(domain_samples[split])
        
        # Add mathematical reasoning tags
        for split in math_data:
            for sample in math_data[split]:
                sample['component'] = 'T-Mathematics'
                sample['training_type'] = 'mathematical_reasoning'
        
        total_samples = sum(len(samples) for samples in math_data.values())
        logger.info(f"Prepared {total_samples} samples for T-Mathematics Engine")
        
        return math_data
    
    def prepare_mprime_data(self) -> Dict[str, List[Dict]]:
        """Prepare data for M-PRIME Framework training."""
        mprime_domains = [
            'mathematics_logic',
            'probability_statistics',
            'abstract_reasoning',
            'temporal_sequential_logic'
        ]
        
        mprime_data = {'train': [], 'val': [], 'test': []}
        
        for domain in mprime_domains:
            domain_samples = self.get_domain_specific_data(domain)
            for split in mprime_data:
                mprime_data[split].extend(domain_samples[split])
        
        # Add M-PRIME specific tags
        for split in mprime_data:
            for sample in mprime_data[split]:
                sample['component'] = 'M-PRIME'
                sample['training_type'] = 'symbolic_mathematics'
        
        total_samples = sum(len(samples) for samples in mprime_data.values())
        logger.info(f"Prepared {total_samples} samples for M-PRIME Framework")
        
        return mprime_data
    
    def prepare_echo_prime_data(self) -> Dict[str, List[Dict]]:
        """Prepare data for ECHO-PRIME temporal logic training."""
        temporal_domains = [
            'temporal_sequential_logic',
            'hypothesis_testing',
            'metacognitive_reasoning',
            'creative_problem_solving'
        ]
        
        echo_data = {'train': [], 'val': [], 'test': []}
        
        for domain in temporal_domains:
            domain_samples = self.get_domain_specific_data(domain)
            for split in echo_data:
                echo_data[split].extend(domain_samples[split])
        
        # Add ECHO-PRIME specific tags
        for split in echo_data:
            for sample in echo_data[split]:
                sample['component'] = 'ECHO-PRIME'
                sample['training_type'] = 'temporal_reasoning'
        
        total_samples = sum(len(samples) for samples in echo_data.values())
        logger.info(f"Prepared {total_samples} samples for ECHO-PRIME")
        
        return echo_data
    
    def prepare_mlingua_data(self) -> Dict[str, List[Dict]]:
        """Prepare data for M-LINGUA Interface training."""
        language_domains = [
            'language_communication',
            'explainable_ai',
            'knowledge_transfer',
            'bias_detection'
        ]
        
        lingua_data = {'train': [], 'val': [], 'test': []}
        
        for domain in language_domains:
            domain_samples = self.get_domain_specific_data(domain)
            for split in lingua_data:
                lingua_data[split].extend(domain_samples[split])
        
        # Add M-LINGUA specific tags
        for split in lingua_data:
            for sample in lingua_data[split]:
                sample['component'] = 'M-LINGUA'
                sample['training_type'] = 'natural_language'
        
        total_samples = sum(len(samples) for samples in lingua_data.values())
        logger.info(f"Prepared {total_samples} samples for M-LINGUA Interface")
        
        return lingua_data
    
    def export_component_data(self, component_name: str, data: Dict[str, List[Dict]], 
                            output_dir: Optional[Path] = None) -> None:
        """Export component-specific training data."""
        if output_dir is None:
            output_dir = self.config.data_dir.parent / 'component_training' / component_name.lower()
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split, samples in data.items():
            if samples:
                output_file = output_dir / f'{component_name.lower()}_{split}.jsonl'
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    for sample in samples:
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                
                logger.info(f"Exported {len(samples)} {split} samples to {output_file}")
    
    def generate_training_report(self) -> Dict:
        """Generate comprehensive training data report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'data_dir': str(self.config.data_dir),
                'batch_size': self.config.batch_size,
                'max_seq_length': self.config.max_seq_length,
                'device': self.config.device,
                'backend': self.config.backend
            },
            'datasets': {},
            'components': {}
        }
        
        # Dataset statistics
        for split, dataset in self.datasets.items():
            report['datasets'][split] = {
                'total_samples': len(dataset),
                'source_files': len(set(sample['source_file'] for sample in dataset.samples))
            }
        
        # Component-specific data preparation
        components = {
            'T-Mathematics': self.prepare_t_mathematics_data,
            'M-PRIME': self.prepare_mprime_data,
            'ECHO-PRIME': self.prepare_echo_prime_data,
            'M-LINGUA': self.prepare_mlingua_data
        }
        
        for comp_name, prep_func in components.items():
            comp_data = prep_func()
            report['components'][comp_name] = {
                split: len(samples) for split, samples in comp_data.items()
            }
        
        return report

def main():
    """Main execution function for training data integration."""
    # Configuration
    config = TrainingConfig(
        data_dir=Path("/Volumes/My Book/MISO_Ultimate 15.32.28/data/type_training/phase1/processed"),
        batch_size=32,
        max_seq_length=512,
        device="auto",
        backend="auto"
    )
    
    # Initialize integrator
    integrator = MISODataIntegrator(config)
    
    # Load datasets
    logger.info("🔄 Loading VXOR Phase-1 training datasets...")
    integrator.load_datasets()
    integrator.create_data_loaders()
    
    # Prepare component-specific data
    logger.info("🔄 Preparing component-specific training data...")
    
    components_data = {
        'T-Mathematics': integrator.prepare_t_mathematics_data(),
        'M-PRIME': integrator.prepare_mprime_data(),
        'ECHO-PRIME': integrator.prepare_echo_prime_data(),
        'M-LINGUA': integrator.prepare_mlingua_data()
    }
    
    # Export component data
    for comp_name, comp_data in components_data.items():
        integrator.export_component_data(comp_name, comp_data)
    
    # Generate and save training report
    report = integrator.generate_training_report()
    report_file = config.data_dir / 'training_integration_report.json'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📊 Training integration report saved to: {report_file}")
    
    # Summary
    total_samples = sum(len(integrator.datasets[split]) for split in integrator.datasets)
    logger.info(f"✅ MISO Training Integration Complete!")
    logger.info(f"📊 Total Samples: {total_samples:,}")
    logger.info(f"🔧 Components Prepared: {len(components_data)}")
    logger.info(f"💾 Device: {config.device} | Backend: {config.backend}")

if __name__ == "__main__":
    main()
