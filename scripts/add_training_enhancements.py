#!/usr/bin/env python3
"""Integration Script: Mini-Validation Probes + Rehearsal Replay OHNE Training-Unterbrechung"""

import os
import sys
import json
import shutil
from pathlib import Path
from mini_validation_datasets import save_validation_datasets
from enhanced_training_callbacks import create_enhanced_callbacks

def setup_validation_datasets():
    """Setup Mini-Validation Datasets"""
    root = Path(__file__).parent.parent
    validation_dir = root / "training" / "validation_datasets"
    
    print("🔧 Setting up mini-validation datasets...")
    gsm8k_path, mmlu_path = save_validation_datasets(validation_dir)
    
    return validation_dir

def patch_existing_training_scripts():
    """Patche bestehende Training Scripts für Enhanced Callbacks"""
    
    scripts_dir = Path(__file__).parent
    patches = [
        {
            "file": "train_mps.py",
            "backup": "train_mps_original.py",
            "patches": [
                {
                    "search": "from transformers.trainer_callback import TrainerCallback",
                    "replace": "from transformers.trainer_callback import TrainerCallback\nfrom enhanced_training_callbacks import create_enhanced_callbacks",
                    "required": False
                },
                {
                    "search": "# Create trainer",
                    "replace": "# Create enhanced callbacks\n    enhanced_callbacks = create_enhanced_callbacks(OUTDIR)\n    \n    # Create trainer",
                    "required": False
                },
                {
                    "search": "callbacks=callbacks",
                    "replace": "callbacks=callbacks + enhanced_callbacks",
                    "required": False
                }
            ]
        }
    ]
    
    for patch_info in patches:
        script_path = scripts_dir / patch_info["file"]
        backup_path = scripts_dir / patch_info["backup"]
        
        if not script_path.exists():
            print(f"⚠️  Skipping {patch_info['file']} - not found")
            continue
        
        # Erstelle Backup falls nicht vorhanden
        if not backup_path.exists():
            shutil.copy2(script_path, backup_path)
            print(f"💾 Created backup: {backup_path}")
        
        # Lese Original
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Wende Patches an
        modified = False
        for patch in patch_info["patches"]:
            if patch["search"] in content and patch["replace"] not in content:
                content = content.replace(patch["search"], patch["replace"])
                modified = True
                print(f"✅ Applied patch to {patch_info['file']}: {patch['search'][:30]}...")
        
        if modified:
            with open(script_path, 'w') as f:
                f.write(content)
            print(f"📝 Updated {script_path}")

def create_standalone_enhanced_trainer():
    """Erstelle Standalone Enhanced Training Script"""
    
    enhanced_script = """#!/usr/bin/env python3
\"\"\"Standalone Enhanced Training mit Mini-Validation + Rehearsal Replay\"\"\"

import os, json, random
from pathlib import Path
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from enhanced_training_callbacks import create_enhanced_callbacks

ROOT = Path(os.environ.get("ROOT", "."))
BASE_MODEL = (ROOT/"models/tinyllama").as_posix()
MIXING_JSON = ROOT/"training/configs/mixing_phase2.json"
OUTDIR = (ROOT/"runs/phase2_enhanced_standalone").as_posix()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🔥 Using device: {device}")
torch.manual_seed(42)

def format_example(ex, tokenizer):
    prompt = ex.get("problem", "")
    response = ex.get("solution", "") 
    text = f"Problem: {prompt}\\nSolution: {response}"
    
    tokenized = tokenizer(
        text, 
        truncation=True, 
        padding="max_length", 
        max_length=512,
        return_tensors=None
    )
    
    labels = tokenized["input_ids"].copy()
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }

def load_mixed_dataset(tokenizer):
    with open(MIXING_JSON) as f:
        config = json.load(f)
    
    datasets = []
    for item in config["train"]:
        agi_name = item["name"]
        jsonl_path = Path(item["path"])
        weight = item["weight"]
        
        if not jsonl_path.exists():
            jsonl_path = ROOT / jsonl_path
            
        if jsonl_path.exists():
            dataset = load_dataset("json", data_files=str(jsonl_path), split="train")
            sample_size = int(len(dataset) * weight)
            dataset = dataset.shuffle().select(range(min(sample_size, len(dataset))))
            datasets.append(dataset)
            print(f"✅ Loaded {agi_name}: {len(dataset)} samples (weight: {weight})")
        else:
            print(f"⚠️  Skipping {agi_name}: file not found at {jsonl_path}")
    
    combined = concatenate_datasets(datasets)
    return combined.map(lambda ex: format_example(ex, tokenizer), remove_columns=combined.column_names).shuffle()

def main():
    print("🚀 Starting Enhanced Standalone Training")
    print("   Features: Mini-Validation Probes + Rehearsal Replay")
    
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
    
    # Enhanced training arguments
    training_args = TrainingArguments(
        output_dir=OUTDIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-4,
        warmup_steps=100,
        logging_steps=50,
        save_strategy="steps",
        save_steps=1000,
        evaluation_strategy="steps",
        eval_steps=5000,  # Sync mit mini-validation
        fp16=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Create enhanced callbacks - HIER IST DER SCHLÜSSEL
    enhanced_callbacks = create_enhanced_callbacks(OUTDIR)
    
    # Create trainer mit enhanced callbacks
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=enhanced_callbacks  # Enhanced Callbacks aktiviert
    )
    
    print("🔥 Starting enhanced training...")
    print("   📊 Mini-probes every 5,000 steps")
    print("   🔄 Rehearsal replay: 1.5% ratio")
    print("   🌊 Wave transitions every 10,000 steps")
    
    # Train
    trainer.train()
    
    # Save
    model.save_pretrained(OUTDIR)
    tokenizer.save_pretrained(OUTDIR)
    print(f"✅ Enhanced training completed! Model saved to {OUTDIR}")

if __name__ == "__main__":
    main()
"""
    
    standalone_path = Path(__file__).parent / "train_enhanced_standalone.py"
    with open(standalone_path, 'w') as f:
        f.write(enhanced_script)
    
    # Make executable
    os.chmod(standalone_path, 0o755)
    
    print(f"✅ Created standalone enhanced trainer: {standalone_path}")
    return standalone_path

def create_monitoring_dashboard():
    """Erstelle Mini Dashboard für Enhanced Training Monitoring"""
    
    dashboard_script = """#!/usr/bin/env python3
\"\"\"Enhanced Training Monitor Dashboard\"\"\"

import json
import time
from pathlib import Path
from datetime import datetime

def monitor_training_progress(runs_dir="./runs"):
    \"\"\"Monitor Enhanced Training Progress\"\"\"
    
    runs_path = Path(runs_dir)
    
    print("🔍 Enhanced Training Monitor")
    print("="*50)
    
    for run_dir in runs_path.glob("phase2_*"):
        if not run_dir.is_dir():
            continue
        
        print(f"\\n📁 Run: {run_dir.name}")
        
        # Check for mini-probe results
        probe_files = list(run_dir.glob("mini_probe_results_step_*.json"))
        if probe_files:
            latest_probe = max(probe_files, key=lambda p: int(p.stem.split('_')[-1]))
            
            with open(latest_probe) as f:
                probe_data = json.load(f)
            
            step = probe_data.get('step', 'unknown')
            gsm8k_acc = probe_data.get('gsm8k_accuracy', 0)
            mmlu_acc = probe_data.get('mmlu_accuracy', 0)
            
            print(f"   📊 Latest Probe (Step {step}):")
            print(f"      GSM8K: {gsm8k_acc:.1f}%")
            print(f"      MMLU:  {mmlu_acc:.1f}%")
        
        # Check for rehearsal stats
        rehearsal_stats = run_dir / "rehearsal_replay_stats.json"
        if rehearsal_stats.exists():
            with open(rehearsal_stats) as f:
                stats = json.load(f)
            
            buffer_samples = stats.get('replay_buffer', {}).get('total_samples', 0)
            wave_count = stats.get('total_waves', 0)
            
            print(f"   🔄 Rehearsal Replay:")
            print(f"      Buffer: {buffer_samples} samples")
            print(f"      Waves: {wave_count}")
        
        # Check training progress
        training_summary = run_dir / "training_summary.json"
        if training_summary.exists():
            with open(training_summary) as f:
                summary = json.load(f)
            
            steps = summary.get('total_steps', 0)
            loss = summary.get('training_loss', 0)
            
            print(f"   🎯 Training Progress:")
            print(f"      Steps: {steps}")
            print(f"      Loss: {loss:.4f}")

if __name__ == "__main__":
    import sys
    runs_dir = sys.argv[1] if len(sys.argv) > 1 else "./runs"
    
    while True:
        try:
            monitor_training_progress(runs_dir)
            print(f"\\n⏰ Updated: {datetime.now().strftime('%H:%M:%S')}")
            print("Press Ctrl+C to stop monitoring\\n")
            time.sleep(30)  # Update every 30 seconds
        except KeyboardInterrupt:
            print("\\n👋 Monitoring stopped")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(10)
"""
    
    monitor_path = Path(__file__).parent / "monitor_enhanced_training.py"
    with open(monitor_path, 'w') as f:
        f.write(dashboard_script)
    
    os.chmod(monitor_path, 0o755)
    
    print(f"✅ Created monitoring dashboard: {monitor_path}")
    return monitor_path

def main():
    """Main Integration Function"""
    print("🚀 Adding Training Enhancements WITHOUT Interruption")
    print("="*60)
    
    # 1. Setup validation datasets
    validation_dir = setup_validation_datasets()
    print(f"✅ Step 1: Validation datasets ready at {validation_dir}")
    
    # 2. Create standalone enhanced trainer
    standalone_script = create_standalone_enhanced_trainer()
    print(f"✅ Step 2: Standalone enhanced trainer created")
    
    # 3. Create monitoring dashboard
    monitor_script = create_monitoring_dashboard()
    print(f"✅ Step 3: Monitoring dashboard created")
    
    # 4. Patch existing scripts (optional, non-disruptive)
    try:
        patch_existing_training_scripts()
        print(f"✅ Step 4: Existing scripts patched (optional)")
    except Exception as e:
        print(f"⚠️  Step 4: Script patching skipped: {e}")
    
    print("\\n" + "="*60)
    print("🎉 ENHANCEMENT INTEGRATION COMPLETE")
    print("="*60)
    print("\\n📋 Available Options:")
    print(f"   1. Run standalone enhanced trainer:")
    print(f"      python {standalone_script}")
    print(f"\\n   2. Monitor enhanced training:")
    print(f"      python {monitor_script}")
    print(f"\\n   3. Continue existing training with enhanced callbacks")
    print(f"      (if patches were applied successfully)")
    print("\\n🔍 Features Added:")
    print("   • Mini-Validation Probes (GSM8K-100, MMLU-1k) every 5k steps")
    print("   • Rehearsal Replay (1.5%) at wave transitions every 10k steps")
    print("   • Async probes to avoid training interruption")
    print("   • Real-time monitoring dashboard")
    print("\\n⚠️  Current training will NOT be interrupted!")

if __name__ == "__main__":
    main()
