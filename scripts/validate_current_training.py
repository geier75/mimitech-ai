#!/usr/bin/env python3
"""Real-time validation for ongoing Phase 2 training"""

import os, json, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np

ROOT = Path(os.environ.get("ROOT", "."))
BASE_MODEL = (ROOT/"models/tinyllama").as_posix()
RUNS_DIR = ROOT/"runs/phase2_mps_real"

def find_latest_checkpoint():
    """Find the most recent checkpoint"""
    checkpoints = list(RUNS_DIR.glob("checkpoint-*"))
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda x: int(x.name.split('-')[1]))
    return latest

def load_model_from_checkpoint(checkpoint_path):
    """Load model and tokenizer from checkpoint"""
    print(f"Loading from {checkpoint_path}")
    
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map="mps" if torch.backends.mps.is_available() else "cpu"
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, checkpoint_path)
    
    return model, tokenizer

def test_model_quality(model, tokenizer, test_prompts):
    """Test model with sample prompts"""
    results = []
    model.eval()
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        if torch.backends.mps.is_available():
            inputs = {k: v.to("mps") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({
            "prompt": prompt,
            "response": response[len(prompt):].strip(),
            "full_output": response
        })
    
    return results

def main():
    print("🔍 Validating current training progress...")
    
    # Test prompts covering different AGI types
    test_prompts = [
        "Problem: Solve 2x + 5 = 13 for x.\nSolution:",
        "Problem: What comes next in the sequence: 2, 4, 8, 16, ?\nSolution:",
        "Problem: Write a creative story about a robot learning to paint.\nSolution:",
        "Problem: If it rains, the ground gets wet. The ground is wet. What can we conclude?\nSolution:",
        "Problem: Explain the concept of gravity in simple terms.\nSolution:"
    ]
    
    # Find latest checkpoint
    checkpoint = find_latest_checkpoint()
    if not checkpoint:
        print("❌ No checkpoints found")
        return
    
    print(f"📂 Latest checkpoint: {checkpoint}")
    
    # Load model
    try:
        model, tokenizer = load_model_from_checkpoint(checkpoint)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test model
    print("🧪 Testing model quality...")
    results = test_model_quality(model, tokenizer, test_prompts)
    
    # Display results
    print("\n" + "="*60)
    print("MODEL VALIDATION RESULTS")
    print("="*60)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. PROMPT: {result['prompt']}")
        print(f"   RESPONSE: {result['response']}")
        print("-" * 40)
    
    # Save results
    results_file = checkpoint / "validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Read training state
    trainer_state_file = checkpoint / "trainer_state.json"
    if trainer_state_file.exists():
        with open(trainer_state_file) as f:
            state = json.load(f)
        
        print(f"\n📊 Training Progress:")
        print(f"   Steps: {state.get('global_step', 'N/A')} / {state.get('max_steps', 'N/A')}")
        print(f"   Epoch: {state.get('epoch', 'N/A'):.3f}")
        
        if state.get('log_history'):
            latest_log = state['log_history'][-1]
            print(f"   Latest Loss: {latest_log.get('loss', 'N/A')}")
            print(f"   Learning Rate: {latest_log.get('learning_rate', 'N/A')}")

if __name__ == "__main__":
    main()
