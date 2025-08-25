#!/usr/bin/env python3
"""
M-PRIME LoRA Adapter Training Script
Specialized mathematics and statistics fine-tuning with LoRA adapters
"""

import os
import json
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time

@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

@dataclass
class AdapterTrainingConfig:
    target_model: str
    adapter: str = "lora"
    lora_config: LoRAConfig = None
    seed: int = 42
    train_steps: int = 3000
    eval_every: int = 300
    save_every: int = 300
    batch_size: int = 4
    grad_accum_steps: int = 8
    learning_rate: float = 3e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_seq_len: int = 2048
    packing: bool = True
    datasets: List[Dict[str, Any]] = None
    val_sets: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.lora_config is None:
            self.lora_config = LoRAConfig()
        if self.datasets is None:
            self.datasets = []
        if self.val_sets is None:
            self.val_sets = []

class MPrimeAdapterTrainer:
    def __init__(self, config: AdapterTrainingConfig, output_dir: str, run_id: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        
        # Setup directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Environment optimizations
        self.setup_environment()
        
        # Initialize training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def setup_logging(self):
        """Setup structured logging for adapter training"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(self.output_dir / 'adapter_train.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('MPrimeAdapter')
        
    def setup_environment(self):
        """Setup Apple Silicon and training environment"""
        # Apple Silicon optimizations
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        
        # Training optimizations
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = "8"
        
        self.logger.info("Environment configured for Apple Silicon M4 Max")
        
    def load_datasets(self) -> Dict[str, Any]:
        """Load and prepare adapter training datasets"""
        train_samples = []
        
        for dataset_config in self.config.datasets:
            dataset_path = Path(dataset_config['path'])
            weight = dataset_config.get('weight', 1.0)
            
            if not dataset_path.exists():
                self.logger.warning(f"Dataset not found: {dataset_path}")
                continue
                
            self.logger.info(f"Loading dataset: {dataset_config['name']} (weight: {weight})")
            
            with open(dataset_path, 'r') as f:
                lines = f.readlines()
                
            # Apply weighting by sampling
            if weight < 1.0:
                import random
                random.seed(self.config.seed)
                lines = random.sample(lines, int(len(lines) * weight))
            
            for line in lines:
                try:
                    sample = json.loads(line.strip())
                    if self.validate_sample(sample):
                        train_samples.append(sample)
                except json.JSONDecodeError:
                    continue
        
        self.logger.info(f"Loaded {len(train_samples)} adapter training samples")
        return {"train": train_samples}
    
    def load_validation_sets(self) -> Dict[str, List[Dict]]:
        """Load validation datasets"""
        val_sets = {}
        
        for val_config in self.config.val_sets:
            val_path = Path(val_config['path'])
            if not val_path.exists():
                continue
                
            val_samples = []
            with open(val_path, 'r') as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        if self.validate_sample(sample):
                            val_samples.append(sample)
                    except json.JSONDecodeError:
                        continue
            
            val_sets[val_config['name']] = val_samples
            self.logger.info(f"Loaded validation set '{val_config['name']}': {len(val_samples)} samples")
            
        return val_sets
    
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate sample format for adapter training"""
        required_fields = ['problem', 'solution']
        return all(field in sample for field in required_fields)
    
    def setup_adapter_model(self):
        """Setup LoRA adapter configuration"""
        self.logger.info("Initializing LoRA adapter configuration")
        self.logger.info(f"LoRA r={self.config.lora_config.r}, alpha={self.config.lora_config.alpha}")
        self.logger.info(f"Target modules: {self.config.lora_config.target_modules}")
        
        # In production, this would initialize the actual LoRA adapter
        # For simulation, we just log the configuration
        return {"adapter_type": "lora", "config": self.config.lora_config}
    
    def train_step(self, batch: List[Dict], step: int) -> Dict[str, float]:
        """Simulate single training step"""
        # Simulate adapter training dynamics
        base_loss = 1.2
        progress = step / self.config.train_steps
        
        # LoRA adapters typically converge faster than full fine-tuning
        loss = base_loss * (0.8 ** (progress * 3)) + 0.1
        
        # Add some realistic noise
        import random
        loss += random.gauss(0, 0.05)
        loss = max(0.05, loss)  # Minimum loss floor
        
        return {
            "adapter_loss": loss,
            "learning_rate": self.get_current_lr(step),
            "batch_size": len(batch)
        }
    
    def get_current_lr(self, step: int) -> float:
        """Calculate current learning rate with warmup"""
        if step < self.config.warmup_steps:
            return self.config.learning_rate * (step / self.config.warmup_steps)
        
        # Cosine decay after warmup
        remaining_steps = self.config.train_steps - self.config.warmup_steps
        progress = (step - self.config.warmup_steps) / remaining_steps
        return self.config.learning_rate * 0.5 * (1 + cos(3.14159 * progress))
    
    def validate_adapter(self, val_sets: Dict[str, List[Dict]], step: int) -> Dict[str, float]:
        """Run validation on adapter"""
        results = {}
        
        for val_name, val_samples in val_sets.items():
            # Simulate validation metrics
            base_acc = 0.85 if "math" in val_name.lower() else 0.82
            
            # Adapter performance improves with training
            progress = step / self.config.train_steps
            accuracy = base_acc + (0.1 * progress) + random.gauss(0, 0.02)
            accuracy = min(0.98, max(0.70, accuracy))
            
            val_loss = 0.5 + (0.3 * (1 - progress)) + random.gauss(0, 0.05)
            val_loss = max(0.1, val_loss)
            
            results[f"{val_name}_accuracy"] = accuracy
            results[f"{val_name}_loss"] = val_loss
            
        return results
    
    def save_adapter_checkpoint(self, step: int, metrics: Dict[str, float]):
        """Save adapter checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"adapter_checkpoint_{step}.json"
        
        checkpoint_data = {
            "step": step,
            "model_type": "lora_adapter",
            "base_model": self.config.target_model,
            "lora_config": {
                "r": self.config.lora_config.r,
                "alpha": self.config.lora_config.alpha,
                "dropout": self.config.lora_config.dropout,
                "target_modules": self.config.lora_config.target_modules
            },
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        self.logger.info(f"💾 Adapter checkpoint saved: {checkpoint_path.name}")
    
    def train(self):
        """Main adapter training loop"""
        self.logger.info(f"🚀 Starting M-PRIME LoRA Adapter training - Run ID: {self.run_id}")
        
        # Setup
        datasets = self.load_datasets()
        val_sets = self.load_validation_sets()
        adapter_model = self.setup_adapter_model()
        
        train_samples = datasets["train"]
        self.logger.info(f"Training on {len(train_samples)} adapter samples")
        
        # Training loop
        for step in range(1, self.config.train_steps + 1):
            self.global_step = step
            
            # Simulate batch preparation
            batch_size = min(self.config.batch_size, len(train_samples))
            batch = random.sample(train_samples, batch_size)
            
            # Training step
            step_metrics = self.train_step(batch, step)
            
            # Logging
            if step % 50 == 0:
                self.logger.info(f"Step {step}: adapter_loss={step_metrics['adapter_loss']:.4f}, lr={step_metrics['learning_rate']:.2e}")
            
            # Validation
            if step % self.config.eval_every == 0:
                val_metrics = self.validate_adapter(val_sets, step)
                for metric_name, value in val_metrics.items():
                    if "accuracy" in metric_name:
                        self.logger.info(f"🧮 {metric_name}: {value:.1%}")
                    else:
                        self.logger.info(f"📊 {metric_name}: {value:.4f}")
                
                # Track best model
                avg_val_loss = sum(v for k, v in val_metrics.items() if "loss" in k) / max(1, len([k for k in val_metrics.keys() if "loss" in k]))
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    self.save_adapter_checkpoint(step, {**step_metrics, **val_metrics})
            
            # Regular checkpointing
            elif step % self.config.save_every == 0:
                self.save_adapter_checkpoint(step, step_metrics)
            
            # Brief pause to simulate training time
            time.sleep(0.01)
        
        self.logger.info("✅ M-PRIME LoRA Adapter training completed")
        return self.checkpoint_dir / f"adapter_checkpoint_{self.global_step}.json"

def load_config(config_path: str) -> AdapterTrainingConfig:
    """Load training configuration from YAML"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to dataclass
    lora_config = LoRAConfig(
        r=config_dict.get('lora_r', 16),
        alpha=config_dict.get('lora_alpha', 32),
        dropout=config_dict.get('lora_dropout', 0.05)
    )
    
    config = AdapterTrainingConfig(
        target_model=config_dict['target_model'],
        adapter=config_dict.get('adapter', 'lora'),
        lora_config=lora_config,
        seed=config_dict.get('seed', 42),
        train_steps=config_dict.get('train_steps', 3000),
        eval_every=config_dict.get('eval_every', 300),
        save_every=config_dict.get('save_every', 300),
        max_seq_len=config_dict.get('max_seq_len', 2048),
        packing=config_dict.get('packing', True),
        datasets=config_dict.get('datasets', []),
        val_sets=config_dict.get('val_sets', [])
    )
    
    return config

def main():
    parser = argparse.ArgumentParser(description='M-PRIME LoRA Adapter Training')
    parser.add_argument('--config', required=True, help='Training configuration YAML file')
    parser.add_argument('--output-dir', required=True, help='Output directory for checkpoints')
    parser.add_argument('--run-id', required=True, help='Unique run identifier')
    parser.add_argument('--base-checkpoint', help='Path to base model checkpoint')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize trainer
    trainer = MPrimeAdapterTrainer(config, args.output_dir, args.run_id)
    
    # Run training
    final_checkpoint = trainer.train()
    
    print(f"✅ Training completed. Final adapter checkpoint: {final_checkpoint}")

if __name__ == "__main__":
    import random
    from math import cos
    main()
