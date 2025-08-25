# Training Configuration for Next Run - Eval/Save every 500 steps

from transformers import TrainingArguments

def get_training_args_500_steps(output_dir, model_name="TinyLlama-1.1B-Chat-v1.0"):
    """
    Training arguments with evaluation and saving every 500 steps
    """
    return TrainingArguments(
        output_dir=output_dir,
        
        # Training parameters
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-4,
        warmup_steps=100,
        
        # Evaluation and saving strategy - Every 500 steps
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps", 
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        
        # Logging
        logging_steps=200,
        logging_dir=f"{output_dir}/logs",
        
        # Performance
        fp16=False,  # MPS doesn't support fp16
        dataloader_num_workers=0,
        remove_unused_columns=False,
        
        # Other settings
        seed=42,
        report_to=None,  # Disable wandb/tensorboard
    )

# CLI equivalent command:
"""
--output_dir <OUTPUT_DIR> \\
--num_train_epochs 3 \\
--per_device_train_batch_size 2 \\
--gradient_accumulation_steps 4 \\
--learning_rate 5e-4 \\
--warmup_steps 100 \\
--evaluation_strategy steps --eval_steps 500 \\
--save_strategy steps --save_steps 500 --save_total_limit 2 \\
--load_best_model_at_end --metric_for_best_model loss --greater_is_better false \\
--logging_steps 200 \\
--fp16 false \\
--dataloader_num_workers 0 \\
--remove_unused_columns false \\
--seed 42
"""
