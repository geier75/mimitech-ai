#!/usr/bin/env python3
"""Enhanced Phase 2 Training Script with Prometheus Telemetry and Optimizations"""

import os, json, random, math
from pathlib import Path
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, 
    DataCollatorForLanguageModeling, EarlyStoppingCallback
)
from transformers.trainer_callback import TrainerCallback
from peft import LoraConfig, get_peft_model
from prometheus_callback import create_prometheus_callback

ROOT = Path(os.environ.get("ROOT", "."))
BASE_MODEL = (ROOT/"models/tinyllama").as_posix()
MIXING_JSON = ROOT/"training/configs/mixing_phase2.json"
OUTDIR = (ROOT/"runs/phase2_mps_enhanced").as_posix()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🔥 Using device: {device}")
torch.manual_seed(42)

class StabilityWatcher(TrainerCallback):
    """Stability monitoring callback with automatic interventions"""
    
    def __init__(self, patience: int = 2, lr_decay_factor: float = 0.5, min_lr: float = 1e-6):
        self.patience = patience
        self.lr_decay_factor = lr_decay_factor
        self.min_lr = min_lr
        self.eval_losses = []
        self.stagnant_count = 0
        self.best_loss = float('inf')
        
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        if not logs or "eval_loss" in logs:
            return
            
        current_loss = logs["eval_loss"]
        self.eval_losses.append(current_loss)
        
        # Check for improvement
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.stagnant_count = 0
        else:
            self.stagnant_count += 1
            
        # Check for stagnation (loss within ±0.01 for patience evaluations)
        if len(self.eval_losses) >= self.patience:
            recent_losses = self.eval_losses[-self.patience:]
            loss_range = max(recent_losses) - min(recent_losses)
            
            if loss_range <= 0.01:  # Stagnant
                current_lr = state.learning_rate if hasattr(state, 'learning_rate') else args.learning_rate
                new_lr = max(current_lr * self.lr_decay_factor, self.min_lr)
                
                if new_lr != current_lr:
                    print(f"⚠️ Loss stagnant, reducing LR: {current_lr:.2e} → {new_lr:.2e}")
                    # Update optimizer LR
                    if hasattr(kwargs.get('optimizer'), 'param_groups'):
                        for group in kwargs['optimizer'].param_groups:
                            group['lr'] = new_lr
                    
                self.stagnant_count = 0
        
        # Alert on anomalies
        if not math.isfinite(current_loss):
            print(f"🚨 ALERT: Non-finite loss detected: {current_loss}")
            control.should_training_stop = True
            
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
            
        # Check for gradient norm anomalies
        if "grad_norm" in logs:
            grad_norm = logs["grad_norm"]
            if grad_norm > 10.0:  # High gradient norm
                print(f"⚠️ High gradient norm detected: {grad_norm:.4f}")

def format_example(ex, tokenizer):
    """Enhanced formatting with better prompt structure"""
    prompt = ex.get("problem", "")
    response = ex.get("solution", "") 
    
    # Improved prompt format
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    
    # Tokenize with consistent length
    tokenized = tokenizer(
        text, 
        truncation=True, 
        padding="max_length", 
        max_length=512,
        return_tensors=None
    )
    
    # Create labels (same as input_ids for causal LM)
    labels = tokenized["input_ids"].copy()
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }

def load_mixed_dataset(tokenizer, validation_split: float = 0.1):
    """Load dataset with train/validation split"""
    with open(MIXING_JSON) as f:
        config = json.load(f)
    
    datasets = []
    for item in config["train"]:
        agi_name = item["name"]
        jsonl_path = Path(item["path"])
        weight = item["weight"]
        
        # Try absolute path first, then relative to ROOT
        if not jsonl_path.exists():
            jsonl_path = ROOT / jsonl_path
            
        if jsonl_path.exists():
            dataset = load_dataset("json", data_files=str(jsonl_path), split="train")
            # Sample according to weight
            sample_size = int(len(dataset) * weight)
            dataset = dataset.shuffle().select(range(min(sample_size, len(dataset))))
            datasets.append(dataset)
            print(f"✅ Loaded {agi_name}: {len(dataset)} samples (weight: {weight})")
        else:
            print(f"⚠️  Skipping {agi_name}: file not found at {jsonl_path}")
    
    combined = concatenate_datasets(datasets)
    formatted = combined.map(
        lambda ex: format_example(ex, tokenizer), 
        remove_columns=combined.column_names,
        num_proc=4  # Parallel processing
    ).shuffle()
    
    # Split train/validation
    split_idx = int(len(formatted) * (1 - validation_split))
    train_dataset = formatted.select(range(split_idx))
    eval_dataset = formatted.select(range(split_idx, len(formatted)))
    
    return train_dataset, eval_dataset

def create_optimized_dataloader_kwargs():
    """Optimized DataLoader parameters for better throughput"""
    return {
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 6,
        "drop_last": True  # For consistent batch sizes
    }

def main():
    print("🚀 Starting Enhanced Phase 2 MPS Training")
    
    # Create output directory
    os.makedirs(OUTDIR, exist_ok=True)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add chat tokens if not present
    special_tokens = ["<|im_start|>", "<|im_end|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)
    model.resize_token_embeddings(len(tokenizer))
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # More targets
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    
    # Load datasets
    train_dataset, eval_dataset = load_mixed_dataset(tokenizer)
    print(f"📊 Dataset split: {len(train_dataset)} train, {len(eval_dataset)} eval")
    
    # Enhanced training arguments with all optimizations
    training_args = TrainingArguments(
        output_dir=OUTDIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,  # Larger eval batch
        gradient_accumulation_steps=4,
        learning_rate=5e-4,
        weight_decay=0.01,
        warmup_steps=100,
        
        # Optimized evaluation and saving strategy
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps", 
        save_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Logging
        logging_steps=50,
        logging_strategy="steps",
        report_to=None,  # Disable wandb/tensorboard
        
        # Performance optimizations
        fp16=False,  # MPS doesn't support fp16
        dataloader_pin_memory=False,  # Handled by DataLoader kwargs
        dataloader_num_workers=0,  # Will be overridden
        remove_unused_columns=False,
        
        # Stability features
        max_grad_norm=1.0,  # Gradient clipping
        lr_scheduler_type="cosine",
        
        # Early stopping
        load_best_model_at_end=True,
        
        # Memory optimization
        gradient_checkpointing=True,
        optim="adamw_torch",
        
        # Reproducibility
        seed=42,
        data_seed=42
    )
    
    # Data collator with optimized settings
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=8  # Optimize for tensor cores
    )
    
    # Create callbacks
    callbacks = [
        create_prometheus_callback(port=9108, prefix="mimikcompute"),
        StabilityWatcher(patience=2, lr_decay_factor=0.5),
        EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)
    ]
    
    # Get optimized DataLoader kwargs
    dataloader_kwargs = create_optimized_dataloader_kwargs()
    
    # Create trainer with all optimizations
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataloader_kwargs  # Apply DataLoader optimizations
    )
    
    # Performance summary before training
    print("🔧 Training Configuration:")
    print(f"   Model: {BASE_MODEL}")
    print(f"   Device: {device}")
    print(f"   Train samples: {len(train_dataset):,}")
    print(f"   Eval samples: {len(eval_dataset):,}")
    print(f"   Batch size: {training_args.per_device_train_batch_size}")
    print(f"   Grad accumulation: {training_args.gradient_accumulation_steps}")
    print(f"   Effective batch: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"   DataLoader workers: {dataloader_kwargs['num_workers']}")
    print(f"   Prometheus port: 9108")
    print(f"   Checkpoints: Every {training_args.save_steps} steps (max {training_args.save_total_limit})")
    
    # Train with enhanced monitoring
    print("🔥 Starting enhanced MPS-accelerated training...")
    print("📊 Telemetry available at: http://localhost:9108/metrics")
    
    try:
        result = trainer.train()
        print("✅ Training completed successfully!")
        
        # Save final model and metrics
        trainer.save_model(OUTDIR)
        tokenizer.save_pretrained(OUTDIR)
        
        # Save training summary
        summary = {
            "training_loss": result.training_loss,
            "train_runtime": result.metrics.get("train_runtime", 0),
            "train_samples_per_second": result.metrics.get("train_samples_per_second", 0),
            "total_steps": result.global_step,
            "best_eval_loss": trainer.state.best_metric,
            "configuration": {
                "batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "learning_rate": training_args.learning_rate,
                "epochs": training_args.num_train_epochs
            }
        }
        
        with open(f"{OUTDIR}/training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        print(f"📋 Training summary saved to {OUTDIR}/training_summary.json")
        print(f"🎯 Best eval loss: {trainer.state.best_metric:.4f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
    
    finally:
        # Cleanup
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None

if __name__ == "__main__":
    main()
