#!/usr/bin/env python3
"""Simplified Training Optimizer - Drop-in without external dependencies"""

import json
import time
import logging
from pathlib import Path
from transformers.trainer_callback import TrainerCallback

class SimpleTrainingMonitor(TrainerCallback):
    """Lightweight training monitor without external dependencies"""
    
    def __init__(self, log_file: str = "training_monitor.jsonl"):
        self.log_file = Path(log_file)
        self.start_time = None
        self.step_times = []
        self.losses = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.logger.info("🚀 Training started with enhanced monitoring")
        self._log_metrics({
            "event": "train_start",
            "timestamp": time.time(),
            "total_steps_planned": state.max_steps
        })
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
            
        current_time = time.time()
        metrics = {
            "event": "training_log",
            "timestamp": current_time,
            "step": state.global_step,
            "epoch": logs.get("epoch", 0)
        }
        
        # Core metrics
        if "loss" in logs:
            loss = float(logs["loss"])
            metrics["loss"] = loss
            self.losses.append(loss)
            
            # Check for anomalies
            if not (0 <= loss < float('inf')):
                self.logger.warning(f"🚨 Abnormal loss detected: {loss}")
                metrics["anomaly"] = "abnormal_loss"
                
        if "learning_rate" in logs:
            metrics["learning_rate"] = float(logs["learning_rate"])
            
        if "grad_norm" in logs:
            grad_norm = float(logs["grad_norm"])
            metrics["grad_norm"] = grad_norm
            
            if grad_norm > 10.0:
                self.logger.warning(f"⚠️ High gradient norm: {grad_norm:.4f}")
                
        # Performance metrics
        if self.start_time:
            elapsed = current_time - self.start_time
            metrics["elapsed_time"] = elapsed
            
            if state.global_step > 0:
                avg_step_time = elapsed / state.global_step
                metrics["avg_step_time"] = avg_step_time
                
                # Estimate completion
                if state.max_steps:
                    remaining_steps = state.max_steps - state.global_step
                    eta_seconds = remaining_steps * avg_step_time
                    metrics["eta_hours"] = eta_seconds / 3600
        
        self._log_metrics(metrics)
        
        # Periodic summary
        if state.global_step % 1000 == 0:
            self._print_summary(state, metrics)
            
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        if logs and "eval_loss" in logs:
            eval_loss = float(logs["eval_loss"])
            self.logger.info(f"📊 Evaluation at step {state.global_step}: loss={eval_loss:.4f}")
            
            self._log_metrics({
                "event": "evaluation",
                "timestamp": time.time(),
                "step": state.global_step,
                "eval_loss": eval_loss
            })
            
    def on_save(self, args, state, control, **kwargs):
        self.logger.info(f"💾 Checkpoint saved at step {state.global_step}")
        self._log_metrics({
            "event": "checkpoint_saved",
            "timestamp": time.time(),
            "step": state.global_step
        })
        
    def _log_metrics(self, metrics):
        """Log metrics to JSON Lines file"""
        with open(self.log_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
            
    def _print_summary(self, state, metrics):
        """Print periodic training summary"""
        print("\n" + "="*60)
        print(f"📊 Training Summary - Step {state.global_step}")
        print("="*60)
        
        if "loss" in metrics:
            print(f"Current Loss: {metrics['loss']:.4f}")
            
        if "learning_rate" in metrics:
            print(f"Learning Rate: {metrics['learning_rate']:.2e}")
            
        if "avg_step_time" in metrics:
            print(f"Avg Step Time: {metrics['avg_step_time']:.2f}s")
            
        if "eta_hours" in metrics:
            print(f"ETA: {metrics['eta_hours']:.1f} hours")
            
        if len(self.losses) >= 10:
            recent_loss = sum(self.losses[-10:]) / 10
            print(f"Recent Loss (10 steps): {recent_loss:.4f}")
            
        print("="*60 + "\n")


def create_optimized_training_args(output_dir: str, **kwargs):
    """Create optimized TrainingArguments with all improvements"""
    from transformers import TrainingArguments
    
    defaults = {
        "output_dir": output_dir,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5e-4,
        "weight_decay": 0.01,
        "warmup_steps": 100,
        
        # Optimized evaluation and saving
        "evaluation_strategy": "steps",
        "eval_steps": 1000,
        "save_strategy": "steps", 
        "save_steps": 1000,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        
        # Performance optimizations
        "logging_steps": 50,
        "max_grad_norm": 1.0,
        "fp16": False,  # MPS doesn't support fp16
        "dataloader_pin_memory": False,
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        
        # Stability
        "lr_scheduler_type": "cosine",
        "optim": "adamw_torch",
        
        # Reproducibility
        "seed": 42,
        "data_seed": 42
    }
    
    # Override with user kwargs
    defaults.update(kwargs)
    
    return TrainingArguments(**defaults)


def create_optimized_dataloader_kwargs():
    """Create optimized DataLoader kwargs"""
    return {
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 6,
        "drop_last": True
    }


def integrate_with_existing_trainer(trainer, monitor_log="training_monitor.jsonl"):
    """
    Integrate monitoring with existing trainer
    
    Usage:
        trainer = Trainer(...)
        integrate_with_existing_trainer(trainer)
        trainer.train()
    """
    
    # Add our monitor to existing callbacks
    monitor = SimpleTrainingMonitor(log_file=monitor_log)
    
    if hasattr(trainer, 'callback_handler'):
        trainer.callback_handler.add_callback(monitor)
    else:
        if not hasattr(trainer, 'callbacks'):
            trainer.callbacks = []
        trainer.callbacks.append(monitor)
    
    print(f"✅ Enhanced monitoring added - logs will be saved to {monitor_log}")
    return monitor


if __name__ == "__main__":
    print("🔧 Simple Training Optimizer")
    print("="*50)
    print("Usage examples:")
    print()
    print("1. Add to existing trainer:")
    print("   integrate_with_existing_trainer(trainer)")
    print()
    print("2. Create optimized training args:")
    print("   args = create_optimized_training_args('/path/to/output')")
    print()
    print("3. Get DataLoader optimizations:")
    print("   dl_kwargs = create_optimized_dataloader_kwargs()")
    print("   trainer = Trainer(..., **dl_kwargs)")
    print()
    print("✅ No external dependencies required!")
