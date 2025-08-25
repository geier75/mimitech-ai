#!/usr/bin/env python3
"""Real Phase 2 Training Script with MPS Acceleration"""

import os, json, random
from pathlib import Path
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

ROOT = Path(os.environ.get("ROOT", "."))
BASE_MODEL = (ROOT/"models/tinyllama").as_posix()
MIXING_JSON = ROOT/"training/configs/mixing_phase2.json"
OUTDIR = (ROOT/"runs/phase2_mps_real").as_posix()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🔥 Using device: {device}")
torch.manual_seed(42)

def format_example(ex, tokenizer):
    prompt = ex.get("problem", "")
    response = ex.get("solution", "") 
    text = f"Problem: {prompt}\nSolution: {response}"
    
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

def load_mixed_dataset(tokenizer):
    """Load and combine AGI type datasets according to mixing config"""
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
    return combined.map(lambda ex: format_example(ex, tokenizer), remove_columns=combined.column_names).shuffle()

def main():
    print("🚀 Starting Phase 2 Real MPS Training")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    
    # Load dataset
    train_dataset = load_mixed_dataset(tokenizer)
    
    # Training arguments optimized for MPS  
    training_args = TrainingArguments(
        output_dir=OUTDIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-4,
        warmup_steps=100,
        logging_steps=50,
        save_strategy="epoch",
        fp16=False,  # MPS doesn't support fp16
        dataloader_pin_memory=False,
        remove_unused_columns=False
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )
    
    # Train
    print("🔥 Starting MPS-accelerated training...")
    trainer.train()
    
    # Save
    model.save_pretrained(OUTDIR)
    tokenizer.save_pretrained(OUTDIR)
    print(f"✅ Training completed! Model saved to {OUTDIR}")

if __name__ == "__main__":
    main()
