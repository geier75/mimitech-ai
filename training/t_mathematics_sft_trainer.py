#!/usr/bin/env python3
"""
T-Mathematics Engine SFT Training Pipeline
==========================================

Supervised Fine-Tuning pipeline for MISO T-Mathematics Engine.
Optimized for Apple Silicon (M4 Max) with MLX and PyTorch MPS support.

Author: MISO Development Team
Version: 1.0.0
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import math
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TMathTrainingConfig:
    """Configuration for T-Mathematics Engine training."""
    # Model settings
    model_name: str = "microsoft/DialoGPT-small"  # Lightweight base model
    max_seq_length: int = 512
    vocab_size: int = 50257
    
    # Training settings
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    
    # Data settings
    train_data_path: str = "/Volumes/My Book/MISO_Ultimate 15.32.28/data/type_training/phase1/component_training/t-mathematics"
    validation_split: float = 0.1
    
    # Device settings
    device: str = "auto"
    use_mixed_precision: bool = True
    compile_model: bool = False  # torch.compile for Apple Silicon
    
    # Checkpoint settings
    save_steps: int = 500
    eval_steps: int = 250
    output_dir: str = "/Volumes/My Book/MISO_Ultimate 15.32.28/training/checkpoints/t_mathematics"
    
    # Logging
    log_steps: int = 50
    wandb_project: Optional[str] = None
    
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
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

class TMathematicsDataset(Dataset):
    """Dataset class for T-Mathematics training data."""
    
    def __init__(self, jsonl_file: Path, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # Load and preprocess data
        self._load_data(jsonl_file)
        logger.info(f"Loaded {len(self.samples)} T-Mathematics training samples")
    
    def _load_data(self, jsonl_file: Path):
        """Load and tokenize training data."""
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    
                    # Create input-output format for mathematical reasoning
                    prompt = data.get('prompt', '').strip()
                    completion = data.get('completion', '').strip()
                    
                    if prompt and completion:
                        # Format: <prompt> [SEP] <completion>
                        full_text = f"{prompt} [SEP] {completion}"
                        
                        # Tokenize
                        tokens = self.tokenizer.encode(
                            full_text,
                            add_special_tokens=True,
                            max_length=self.max_length,
                            truncation=True,
                            padding='max_length'
                        )
                        
                        # Create labels (same as input for causal LM)
                        labels = tokens.copy()
                        
                        # Mask prompt tokens in loss calculation (only train on completion)
                        sep_pos = self._find_sep_position(tokens)
                        if sep_pos is not None:
                            for i in range(sep_pos + 1):
                                labels[i] = -100  # Ignore index
                        
                        self.samples.append({
                            'input_ids': tokens,
                            'labels': labels,
                            'attention_mask': [1 if token != self.tokenizer.pad_token_id else 0 for token in tokens]
                        })
                
                except json.JSONDecodeError:
                    continue
    
    def _find_sep_position(self, tokens: List[int]) -> Optional[int]:
        """Find position of [SEP] token."""
        sep_token = self.tokenizer.encode('[SEP]', add_special_tokens=False)
        if not sep_token:
            return None
        
        sep_id = sep_token[0]
        try:
            return tokens.index(sep_id)
        except ValueError:
            return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            'input_ids': torch.tensor(sample['input_ids'], dtype=torch.long),
            'labels': torch.tensor(sample['labels'], dtype=torch.long),
            'attention_mask': torch.tensor(sample['attention_mask'], dtype=torch.long)
        }

class TMathematicsTrainer:
    """Main trainer class for T-Mathematics Engine."""
    
    def __init__(self, config: TMathTrainingConfig):
        self.config = config
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Initialize tokenizer and model
        self.tokenizer = self._setup_tokenizer()
        self.model = self._setup_model()
        
        # Setup training components
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        
        logger.info(f"✅ T-Mathematics Trainer initialized")
        logger.info(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _setup_tokenizer(self):
        """Initialize tokenizer with mathematical tokens."""
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Add pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Add special mathematical tokens
        special_tokens = {
            "additional_special_tokens": [
                "[SEP]", "[MATH]", "[TENSOR]", "[MLX]", "[PYTORCH]",
                "[SOLVE]", "[PROOF]", "[VERIFY]", "[OPTIMIZE]"
            ]
        }
        tokenizer.add_special_tokens(special_tokens)
        
        logger.info(f"🔤 Tokenizer ready with vocab size: {len(tokenizer)}")
        return tokenizer
    
    def _setup_model(self):
        """Initialize and configure the T-Mathematics model."""
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.use_mixed_precision else torch.float32
        )
        
        # Resize token embeddings for new special tokens
        model.resize_token_embeddings(len(self.tokenizer))
        
        # Move to device
        model = model.to(self.config.device)
        
        # Enable torch.compile for Apple Silicon optimization (if available)
        if self.config.compile_model and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
                logger.info("⚡ Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Could not compile model: {e}")
        
        return model
    
    def _setup_optimizer(self):
        """Setup AdamW optimizer."""
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        # Estimate total steps
        train_data_path = Path(self.config.train_data_path) / "t-mathematics_train.jsonl"
        if train_data_path.exists():
            with open(train_data_path, 'r') as f:
                num_samples = sum(1 for _ in f)
        else:
            num_samples = 1000  # Default estimate
        
        total_steps = (num_samples // self.config.batch_size) * self.config.num_epochs
        
        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config.learning_rate * 0.1
        )
        
        logger.info(f"📈 Scheduler configured for {total_steps} total steps")
        return scheduler
    
    def load_datasets(self) -> Tuple[Dataset, Dataset]:
        """Load training and validation datasets."""
        train_file = Path(self.config.train_data_path) / "t-mathematics_train.jsonl"
        val_file = Path(self.config.train_data_path) / "t-mathematics_val.jsonl"
        
        train_dataset = TMathematicsDataset(train_file, self.tokenizer, self.config.max_seq_length)
        val_dataset = TMathematicsDataset(val_file, self.tokenizer, self.config.max_seq_length)
        
        return train_dataset, val_dataset
    
    def create_dataloaders(self, train_dataset: Dataset, val_dataset: Dataset) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation dataloaders."""
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid issues with external storage
            pin_memory=(self.config.device != 'cpu')
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.config.device != 'cpu')
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
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
        total_loss = 0.0
        num_batches = 0
        
        val_file = Path(self.config.train_data_path) / "t-mathematics_val.jsonl"
        val_dataset = TMathematicsDataset(val_file, self.tokenizer, self.config.max_seq_length)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        self.val_losses.append(avg_loss)
        
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
        
        # Save model and tokenizer
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        torch.save({
            'global_step': self.global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
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
        logger.info("🚀 Starting T-Mathematics Engine training...")
        
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
            
            logger.info(f"📈 Epoch {epoch + 1} complete: train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Final checkpoint
        self.save_checkpoint()
        
        # Training summary
        total_time = time.time() - start_time
        logger.info(f"\n✅ T-Mathematics training complete!")
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
    # Configuration
    config = TMathTrainingConfig(
        batch_size=4,  # Conservative for external storage
        learning_rate=3e-5,
        num_epochs=2,
        max_seq_length=384,
        use_mixed_precision=True,
        device="auto"
    )
    
    # Initialize trainer
    trainer = TMathematicsTrainer(config)
    
    # Start training
    results = trainer.train()
    
    # Save training report
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': config.__dict__,
        'results': results
    }
    
    report_file = Path(config.output_dir) / "training_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"📊 Training report saved to: {report_file}")

if __name__ == "__main__":
    main()
