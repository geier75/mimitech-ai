#!/usr/bin/env python3
"""
M-LINGUA SFT Training Pipeline for VXOR Phase-1
===============================================

Production-ready supervised fine-tuning for M-LINGUA component.
Optimized for Apple Silicon M4 Max with comprehensive monitoring.

Author: MISO Development Team
Version: 1.0.0
"""

import json
import logging
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MLinguaTrainingConfig:
    """M-LINGUA SFT training configuration."""
    # Model settings
    model_name: str = "microsoft/DialoGPT-small"
    max_seq_length: int = 2048
    
    # Training settings
    batch_size: int = 4
    learning_rate: float = 3e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    
    # Data settings
    data_config_path: str = ""
    packing: bool = True
    
    # Device settings
    device: str = "auto"
    use_mixed_precision: bool = True
    
    # Checkpoint settings
    save_steps: int = 300
    eval_steps: int = 300
    output_dir: str = ""
    
    # Logging
    log_steps: int = 50
    seed: int = 42
    
    def __post_init__(self):
        if self.device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("🔋 Using Apple Metal Performance Shaders (MPS)")
            elif torch.cuda.is_available():
                self.device = "cuda"
                logger.info("🎮 Using NVIDIA CUDA")
            else:
                self.device = "cpu"
                logger.info("💻 Using CPU")

class MLinguaDataset(Dataset):
    """Multi-dataset loader for M-LINGUA training."""
    
    def __init__(self, config_path: Path, split: str = "train"):
        self.split = split
        self.samples = []
        self.dataset_weights = {}
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load datasets with weights
        for dataset_config in config['datasets']:
            dataset_path = Path(dataset_config['path'])
            if split in str(dataset_path):
                weight = dataset_config.get('weight', 1.0)
                self.dataset_weights[dataset_config['name']] = weight
                self._load_dataset(dataset_path, dataset_config['name'], weight)
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
        logger.info(f"Dataset weights: {self.dataset_weights}")
    
    def _load_dataset(self, path: Path, name: str, weight: float):
        """Load individual dataset with weighted sampling."""
        dataset_samples = []
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'prompt' in data and 'completion' in data:
                        dataset_samples.append({
                            'prompt': data['prompt'].strip(),
                            'completion': data['completion'].strip(),
                            'dataset': name,
                            'weight': weight
                        })
                except json.JSONDecodeError:
                    continue
        
        # Apply dataset weighting through replication
        if weight != 1.0:
            replications = max(1, int(weight * len(dataset_samples)))
            dataset_samples = (dataset_samples * replications)[:replications]
        
        self.samples.extend(dataset_samples)
        logger.info(f"  {name}: {len(dataset_samples)} samples (weight: {weight})")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.samples[idx]

class MLinguaTrainer:
    """Main trainer for M-LINGUA SFT."""
    
    def __init__(self, config: MLinguaTrainingConfig):
        self.config = config
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Initialize model and tokenizer
        self.tokenizer = self._setup_tokenizer()
        self.model = self._setup_model()
        
        # Setup training components
        self.optimizer = self._setup_optimizer()
        self.scheduler = None  # Will be set after dataset loading
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        
        logger.info(f"✅ M-LINGUA Trainer initialized")
        logger.info(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _setup_tokenizer(self):
        """Initialize tokenizer for M-LINGUA."""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Add M-LINGUA specific tokens
        special_tokens = {
            "additional_special_tokens": [
                "[MLINGUA]", "[HELPFUL]", "[BRIEF]", "[FRIENDLY]",
                "[LANG]", "[COMM]", "[EXPLAIN]", "[TRANSFER]"
            ]
        }
        tokenizer.add_special_tokens(special_tokens)
        
        logger.info(f"🔤 Tokenizer ready with vocab size: {len(tokenizer)}")
        return tokenizer
    
    def _setup_model(self):
        """Initialize M-LINGUA model."""
        from transformers import AutoModelForCausalLM
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.use_mixed_precision else torch.float32
        )
        
        # Resize for new tokens
        model.resize_token_embeddings(len(self.tokenizer))
        
        # Move to device
        model = model.to(self.config.device)
        
        return model
    
    def _setup_optimizer(self):
        """Setup AdamW optimizer."""
        return AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
    
    def load_datasets(self) -> Tuple[Dataset, Dataset]:
        """Load training and validation datasets."""
        config_path = Path(self.config.data_config_path)
        
        # Create datasets
        train_dataset = MLinguaDataset(config_path, "train")
        val_dataset = MLinguaDataset(config_path, "val")
        
        # Setup scheduler now that we know dataset size
        total_steps = (len(train_dataset) // self.config.batch_size) * self.config.num_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config.learning_rate * 0.1
        )
        
        logger.info(f"📈 Scheduler configured for {total_steps} total steps")
        
        return train_dataset, val_dataset
    
    def create_dataloaders(self, train_dataset: Dataset, val_dataset: Dataset) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation dataloaders."""
        
        def collate_fn(batch):
            """Custom collation for variable length sequences."""
            prompts = [item['prompt'] for item in batch]
            completions = [item['completion'] for item in batch]
            
            # Create input-output pairs
            full_texts = [f"{p} [SEP] {c}" for p, c in zip(prompts, completions)]
            
            # Tokenize
            encodings = self.tokenizer(
                full_texts,
                truncation=True,
                padding=True,
                max_length=self.config.max_seq_length,
                return_tensors="pt"
            )
            
            # Create labels (mask prompt tokens)
            labels = encodings['input_ids'].clone()
            for i, (prompt, full_text) in enumerate(zip(prompts, full_texts)):
                sep_tokens = self.tokenizer.encode('[SEP]', add_special_tokens=False)
                if sep_tokens:
                    # Find SEP position and mask everything before it
                    input_ids = encodings['input_ids'][i]
                    sep_positions = []
                    for j in range(len(input_ids) - len(sep_tokens) + 1):
                        if input_ids[j:j+len(sep_tokens)].tolist() == sep_tokens:
                            sep_positions.append(j)
                    
                    if sep_positions:
                        sep_pos = sep_positions[0] + len(sep_tokens)
                        labels[i, :sep_pos] = -100
            
            return {
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask'],
                'labels': labels
            }
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config.log_steps == 0:
                avg_loss = total_loss / (batch_idx + 1)
                lr = self.scheduler.get_last_lr()[0]
                logger.info(f"Step {self.global_step}: loss={loss.item():.4f}, avg_loss={avg_loss:.4f}, lr={lr:.2e}")
            
            # Validation
            if self.global_step % self.config.eval_steps == 0:
                val_loss = self.validate()
                logger.info(f"🔍 Validation loss: {val_loss:.4f}")
            
            # Save checkpoint
            if self.global_step % self.config.save_steps == 0:
                self.save_checkpoint()
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        """Run validation."""
        self.model.eval()
        
        # Quick validation on a subset
        val_samples = 100  # Quick validation
        total_loss = 0.0
        count = 0
        
        # Load a few validation samples
        config_path = Path(self.config.data_config_path)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        val_files = [d['path'] for d in config.get('val_sets', [])]
        
        with torch.no_grad():
            for val_file in val_files[:1]:  # Use first val file
                if Path(val_file).exists():
                    with open(val_file, 'r') as f:
                        for i, line in enumerate(f):
                            if i >= val_samples:
                                break
                            
                            try:
                                data = json.loads(line.strip())
                                text = f"{data['prompt']} [SEP] {data['completion']}"
                                
                                # Tokenize
                                inputs = self.tokenizer(
                                    text,
                                    return_tensors="pt",
                                    max_length=512,
                                    truncation=True,
                                    padding=True
                                ).to(self.config.device)
                                
                                # Forward pass
                                inputs['labels'] = inputs['input_ids'].clone()
                                outputs = self.model(**inputs)
                                total_loss += outputs.loss.item()
                                count += 1
                                
                            except:
                                continue
        
        avg_loss = total_loss / max(count, 1)
        
        # Save best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_best_model()
        
        self.model.train()
        return avg_loss
    
    def save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        torch.save({
            'global_step': self.global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss
        }, checkpoint_dir / "training_state.pt")
        
        logger.info(f"💾 Checkpoint saved to {checkpoint_dir}")
    
    def save_best_model(self):
        """Save the best model."""
        best_model_dir = Path(self.config.output_dir) / "best_model"
        best_model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(best_model_dir)
        self.tokenizer.save_pretrained(best_model_dir)
        
        logger.info(f"🏆 Best model saved (val_loss: {self.best_val_loss:.4f})")
    
    def train(self):
        """Main training loop."""
        logger.info("🚀 Starting M-LINGUA SFT training...")
        
        # Load datasets
        train_dataset, val_dataset = self.load_datasets()
        train_loader, val_loader = self.create_dataloaders(train_dataset, val_dataset)
        
        logger.info(f"📊 Training samples: {len(train_dataset)}")
        logger.info(f"📊 Validation samples: {len(val_dataset)}")
        logger.info(f"📊 Training batches per epoch: {len(train_loader)}")
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\n🔄 Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train epoch
            epoch_loss = self.train_epoch(train_loader)
            self.train_losses.append(epoch_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            logger.info(f"📈 Epoch {epoch + 1} complete: train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Final checkpoint
        self.save_checkpoint()
        
        # Training summary
        total_time = time.time() - start_time
        logger.info(f"\n✅ M-LINGUA training complete!")
        logger.info(f"⏱️  Total training time: {total_time:.1f}s")
        logger.info(f"🏆 Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"📊 Total steps: {self.global_step}")
        
        return {
            'best_val_loss': self.best_val_loss,
            'final_train_loss': epoch_loss,
            'total_steps': self.global_step,
            'training_time': total_time
        }

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='M-LINGUA SFT Training')
    parser.add_argument('--config', required=True, help='Path to training config YAML')
    parser.add_argument('--output-dir', required=True, help='Output directory for checkpoints')
    parser.add_argument('--run-id', default='', help='Run ID for this training')
    
    args = parser.parse_args()
    
    # Configuration
    config = MLinguaTrainingConfig(
        data_config_path=args.config,
        output_dir=args.output_dir,
        batch_size=4,
        learning_rate=3e-5,
        num_epochs=2,
        max_seq_length=1024,  # Reduced for stability
        device="auto"
    )
    
    # Initialize trainer
    trainer = MLinguaTrainer(config)
    
    # Start training
    results = trainer.train()
    
    # Save training report
    report = {
        'timestamp': datetime.now().isoformat(),
        'run_id': args.run_id,
        'config': config.__dict__,
        'results': results
    }
    
    report_file = Path(config.output_dir) / "mlingua_training_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"📊 Training report saved to: {report_file}")

if __name__ == "__main__":
    main()
