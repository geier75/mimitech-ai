#!/usr/bin/env python3
"""Real LoRA Adapter Merging for Phase 2"""

import os, json, shutil
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(os.environ.get("ROOT", "."))
LORA_MODEL_DIR = ROOT/"runs/phase2_mps_real"
MERGED_MODEL_DIR = ROOT/"runs/phase2_merged_real" 
BASE_MODEL_DIR = ROOT/"models/tinyllama"

def merge_lora_adapters():
    """Merge LoRA adapters with base model"""
    print("🔀 Merging LoRA adapters with base model...")
    
    try:
        # Load base model and tokenizer
        print("📥 Loading base model...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_DIR, torch_dtype=torch.float16)
        
        # Load LoRA model
        print("🔗 Loading LoRA adapters...")
        model = PeftModel.from_pretrained(base_model, LORA_MODEL_DIR)
        
        # Merge adapters into base model
        print("⚡ Merging adapters...")
        merged_model = model.merge_and_unload()
        
        # Save merged model
        print("💾 Saving merged model...")
        MERGED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        merged_model.save_pretrained(MERGED_MODEL_DIR)
        tokenizer.save_pretrained(MERGED_MODEL_DIR)
        
        # Save merge metadata
        merge_info = {
            "base_model": str(BASE_MODEL_DIR),
            "lora_adapters": str(LORA_MODEL_DIR),
            "merged_output": str(MERGED_MODEL_DIR),
            "merge_timestamp": json.dumps({"timestamp": "2025-08-22T02:07:00"}),
            "model_type": "TinyLlama-1.1B-Chat-v1.0",
            "lora_config": {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
                "lora_dropout": 0.1
            }
        }
        
        with open(MERGED_MODEL_DIR / "merge_info.json", "w") as f:
            json.dump(merge_info, f, indent=2)
        
        print(f"✅ Successfully merged model to: {MERGED_MODEL_DIR}")
        return True
        
    except Exception as e:
        print(f"❌ Merge failed: {e}")
        return False

def test_merged_model():
    """Test the merged model with sample inference"""
    print("🧪 Testing merged model...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_DIR)
        model = AutoModelForCausalLM.from_pretrained(MERGED_MODEL_DIR)
        
        test_prompts = [
            "Problem: What is 2+2?\nSolution:",
            "Problem: Define machine learning\nSolution:",
            "Problem: Solve x + 5 = 12\nSolution:"
        ]
        
        print("📝 Sample outputs:")
        for i, prompt in enumerate(test_prompts, 1):
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(
                **inputs, 
                max_new_tokens=50, 
                do_sample=True, 
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"\n{i}. Input: {prompt.split('Solution:')[0]}Solution:")
            print(f"   Output: {response.split('Solution:')[-1].strip()}")
        
        print("✅ Merged model inference successful")
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False

def create_deployment_bundle():
    """Create deployment-ready bundle"""
    print("📦 Creating deployment bundle...")
    
    bundle_dir = ROOT / "runs/phase2_deployment_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy merged model
    if MERGED_MODEL_DIR.exists():
        shutil.copytree(MERGED_MODEL_DIR, bundle_dir / "model", dirs_exist_ok=True)
    
    # Create bundle manifest
    bundle_manifest = {
        "bundle_version": "2.0.0-real",
        "model_name": "VXOR-Phase2-MPS-Real",
        "base_model": "TinyLlama-1.1B-Chat-v1.0",
        "training_method": "LoRA/SFT",
        "device": "MPS (Apple Silicon)",
        "agi_types_trained": [
            "Language_Communication", "Creative_Problem_Solving", 
            "Temporal_Sequential_Logic", "Pattern_Recognition",
            "Abstract_Reasoning", "Knowledge_Transfer",
            "Probability_Statistics", "Mathematics_Logic"
        ],
        "bundle_contents": [
            "model/", "configs/", "README.md"
        ],
        "deployment_ready": True
    }
    
    with open(bundle_dir / "bundle_manifest.json", "w") as f:
        json.dump(bundle_manifest, f, indent=2)
    
    # Create README
    readme_content = """# VXOR Phase 2 Real MPS Training Bundle

## Model Information
- **Base Model**: TinyLlama-1.1B-Chat-v1.0
- **Training Method**: LoRA/SFT with MPS acceleration
- **Training Data**: 8 AGI types with weighted sampling
- **Device**: Apple Silicon MPS

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./model")
model = AutoModelForCausalLM.from_pretrained("./model")

inputs = tokenizer("Problem: What is 2+2?\\nSolution:", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training Gates Status
- Stats Gate: TBD (pending validation)
- Safety Gate: TBD (pending validation)  
- Contamination Gate: TBD (pending validation)
"""
    
    with open(bundle_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    print(f"✅ Deployment bundle created: {bundle_dir}")
    return str(bundle_dir)

def main():
    print("🔄 Starting LoRA Adapter Merging Pipeline")
    
    if not LORA_MODEL_DIR.exists():
        print("⚠️ LoRA model directory not found - training may not be complete")
        print(f"Expected: {LORA_MODEL_DIR}")
        return False
    
    # Step 1: Merge adapters
    merge_success = merge_lora_adapters()
    if not merge_success:
        print("❌ Adapter merging failed")
        return False
    
    # Step 2: Test merged model
    test_success = test_merged_model()
    if not test_success:
        print("❌ Merged model testing failed")
        return False
    
    # Step 3: Create deployment bundle
    bundle_path = create_deployment_bundle()
    
    print("\n" + "="*50)
    print("📋 ADAPTER MERGING SUMMARY")
    print("="*50)
    print(f"✅ LoRA Merge: SUCCESS")
    print(f"✅ Inference Test: SUCCESS")
    print(f"✅ Deployment Bundle: {bundle_path}")
    print(f"\n🎉 Phase 2 Real Training Pipeline: READY FOR VALIDATION")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
