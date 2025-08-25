#!/usr/bin/env python3
"""Phase 3 Training Configuration - Advanced Fine-tuning"""

import json
from pathlib import Path
from transformers import TrainingArguments

def create_phase3_config():
    """Create Phase 3 training configuration for advanced fine-tuning"""
    
    # Phase 3 focuses on specialized capabilities and performance optimization
    config = {
        "phase": 3,
        "description": "Advanced fine-tuning with specialized datasets and optimizations",
        "base_model": "output from Phase 2",
        "training_strategy": "continual_learning_with_specialization",
        
        "agi_types_priority": [
            # High-priority specialized types
            {"name": "AGI_Type_01_Physics_Causal_Reasoning", "weight": 1.2, "focus": "scientific_reasoning"},
            {"name": "AGI_Type_03_Spatial_Reasoning", "weight": 1.1, "focus": "3d_spatial_tasks"},
            {"name": "AGI_Type_09_Ethical_Reasoning", "weight": 1.0, "focus": "moral_decisions"},
            {"name": "AGI_Type_12_Meta_Learning", "weight": 1.3, "focus": "learning_to_learn"},
            
            # Advanced cognitive types
            {"name": "AGI_Type_15_Analogical_Reasoning", "weight": 0.9, "focus": "pattern_transfer"},
            {"name": "AGI_Type_20_Counterfactual_Reasoning", "weight": 0.8, "focus": "what_if_scenarios"},
            {"name": "AGI_Type_25_Strategic_Planning", "weight": 0.9, "focus": "long_term_planning"},
            
            # Specialized domains
            {"name": "AGI_Type_30_Code_Generation", "weight": 1.1, "focus": "programming_tasks"},
            {"name": "AGI_Type_35_Mathematical_Proofs", "weight": 1.0, "focus": "formal_reasoning"}
        ],
        
        "training_parameters": {
            "epochs": 2,
            "batch_size": 1,  # Smaller batch for complex reasoning tasks
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-5,  # Lower LR for fine-tuning
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "max_sequence_length": 1024,  # Longer sequences for complex reasoning
            
            # Advanced training techniques
            "use_gradient_checkpointing": True,
            "fp16": False,  # MPS compatibility
            "dataloader_num_workers": 2,
            "eval_strategy": "steps",
            "eval_steps": 250,
            "save_steps": 250,
            "logging_steps": 50
        },
        
        "optimization_strategies": [
            "curriculum_learning",
            "difficulty_annealing", 
            "adaptive_batch_sizing",
            "loss_reweighting"
        ],
        
        "evaluation_criteria": {
            "reasoning_accuracy": 0.85,
            "coherence_score": 0.80,
            "safety_compliance": 0.95,
            "factual_accuracy": 0.88
        }
    }
    
    return config

def get_phase3_training_args(output_dir):
    """Get TrainingArguments for Phase 3"""
    return TrainingArguments(
        output_dir=output_dir,
        
        # Core training params
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        
        # Evaluation and saving
        evaluation_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Logging and performance
        logging_steps=50,
        logging_dir=f"{output_dir}/logs",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        
        # Advanced features
        gradient_checkpointing=True,
        fp16=False,  # MPS compatibility
        
        # Other
        seed=42,
        report_to=None,
        push_to_hub=False
    )

def create_phase3_mixing_config():
    """Create mixing configuration for Phase 3 datasets"""
    mixing_config = {
        "train": [
            {"name": "AGI_Type_01_Physics_Causal_Reasoning_100K", "weight": 1.2, "sample_ratio": 0.15},
            {"name": "AGI_Type_12_Meta_Learning_50K", "weight": 1.3, "sample_ratio": 0.20},
            {"name": "AGI_Type_30_Code_Generation_75K", "weight": 1.1, "sample_ratio": 0.18},
            {"name": "AGI_Type_35_Mathematical_Proofs_60K", "weight": 1.0, "sample_ratio": 0.12},
            {"name": "AGI_Type_03_Spatial_Reasoning_80K", "weight": 1.1, "sample_ratio": 0.15},
            {"name": "AGI_Type_09_Ethical_Reasoning_40K", "weight": 1.0, "sample_ratio": 0.08},
            {"name": "AGI_Type_25_Strategic_Planning_55K", "weight": 0.9, "sample_ratio": 0.12}
        ],
        "validation_split": 0.1,
        "total_target_samples": 50000,
        "curriculum_phases": 3
    }
    
    return mixing_config

if __name__ == "__main__":
    # Save configurations
    config = create_phase3_config()
    mixing_config = create_phase3_mixing_config()
    
    print("Phase 3 Configuration:")
    print(json.dumps(config, indent=2))
    
    # Save to files
    with open("phase3_full_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    with open("phase3_mixing.json", "w") as f:
        json.dump(mixing_config, f, indent=2)
    
    print("✅ Phase 3 configurations created")
