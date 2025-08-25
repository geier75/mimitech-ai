#!/usr/bin/env python3
"""Enhanced Mini-Validation Probes mit Real Model Support"""

import json
import time
import random
import argparse
import os
from pathlib import Path
from datetime import datetime

# Real model imports (nur laden wenn verfügbar)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    REAL_MODE_AVAILABLE = True
except ImportError:
    REAL_MODE_AVAILABLE = False
    print("⚠️  Real mode not available - missing transformers/torch. Using simulation mode.")

class RealModelProbes:
    """Real model validation probes"""
    
    def __init__(self, checkpoint_path: str, datasets: list, output_dir: Path):
        self.checkpoint_path = checkpoint_path
        self.datasets = datasets
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Lade Model und Tokenizer"""
        if not REAL_MODE_AVAILABLE:
            raise RuntimeError("Real mode requires transformers and torch")
            
        print(f"🔄 Loading model from {self.checkpoint_path}")
        
        try:
            # Bestimme base model und adapter paths
            ckpt_path = Path(self.checkpoint_path)
            
            # Handle verschiedene Checkpoint-Strukturen
            if ckpt_path.is_file() and ckpt_path.suffix in ['.safetensors', '.pt', '.bin']:
                # Einzelne Adapter-Datei
                adapter_path = ckpt_path.parent
                base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            elif ckpt_path.is_dir():
                # Checkpoint-Verzeichnis
                adapter_path = ckpt_path
                base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            else:
                raise ValueError(f"Invalid checkpoint path: {self.checkpoint_path}")
            
            # Lade Base Model mit MPS-Unterstützung
            print(f"📥 Loading base model: {base_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Device-spezifische Konfiguration
            device_map = None
            if torch.backends.mps.is_available():
                device_map = "mps"
            elif torch.cuda.is_available():
                device_map = "auto"
            else:
                device_map = "cpu"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if device_map != "cpu" else torch.float32,
                device_map=device_map if device_map != "mps" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Move to MPS if available
            if device_map == "mps":
                self.model = self.model.to("mps")
            
            # Lade LoRA Adapter falls vorhanden
            if adapter_path.exists() and any(adapter_path.glob("*.safetensors")):
                print("🔧 Loading LoRA adapter...")
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
            
            self.model.eval()
            print(f"✅ Model loaded successfully on {device_map}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def run_gsm8k_probe(self, dataset_path: str, sample_size: int = 100) -> dict:
        """Real GSM8K-100 probe"""
        if not self.model:
            self.load_model()
            
        with open(dataset_path) as f:
            data = json.load(f)
        
        correct = 0
        total = min(sample_size, len(data))
        
        print(f"🧮 Running GSM8K probe on {total} samples...")
        
        for i in range(total):
            sample = data[i]
            problem = sample.get("problem", "")
            expected = sample.get("answer", "")
            
            prompt = f"Problem: {problem}\nSolution:"
            
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
                
                # Move inputs to same device as model
                if self.model is not None and hasattr(self.model, 'device'):
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                elif torch.backends.mps.is_available():
                    inputs = {k: v.to("mps") for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=150,
                        do_sample=False,
                        temperature=0.1,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated = response[len(prompt):].strip()
                
                if self._extract_answer_matches(generated, expected):
                    correct += 1
                    
            except Exception as e:
                print(f"Error on sample {i}: {e}")
                continue
        
        accuracy = (correct / total) * 100 if total > 0 else 0.0
        print(f"📊 GSM8K: {correct}/{total} = {accuracy:.1f}%")
        
        return {"task": "GSM8K-100", "acc": accuracy/100, "samples": total, "correct": correct}
    
    def run_mmlu_probe(self, dataset_path: str, sample_size: int = 200) -> dict:
        """Real MMLU probe"""
        if not self.model:
            self.load_model()
            
        with open(dataset_path) as f:
            data = json.load(f)
        
        correct = 0
        total = min(sample_size, len(data))
        
        print(f"🎓 Running MMLU probe on {total} samples...")
        
        for i in range(total):
            sample = data[i]
            question = sample.get("question", "")
            choices = sample.get("choices", [])
            expected = sample.get("answer", "")
            
            prompt = f"Question: {question}\n"
            for idx, choice in enumerate(choices):
                prompt += f"{chr(65+idx)}. {choice}\n"
            prompt += "Answer:"
            
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300)
                
                # Move inputs to same device as model
                if self.model is not None and hasattr(self.model, 'device'):
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                elif torch.backends.mps.is_available():
                    inputs = {k: v.to("mps") for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        temperature=0.1,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated = response[len(prompt):].strip()
                
                predicted = self._extract_choice(generated)
                if predicted == expected:
                    correct += 1
                    
            except Exception as e:
                print(f"Error on sample {i}: {e}")
                continue
        
        accuracy = (correct / total) * 100 if total > 0 else 0.0
        print(f"📊 MMLU: {correct}/{total} = {accuracy:.1f}%")
        
        return {"task": "MMLU-1k", "acc": accuracy/100, "samples": total, "correct": correct}
    
    def _extract_answer_matches(self, generated: str, expected: str) -> bool:
        """Extract numerical answer from GSM8K response"""
        import re
        numbers = re.findall(r'\d+\.?\d*', generated)
        if not numbers:
            return False
        
        try:
            expected_num = float(expected)
            for num_str in numbers:
                if abs(float(num_str) - expected_num) < 0.01:
                    return True
        except ValueError:
            return expected.lower() in generated.lower()
        
        return False
    
    def _extract_choice(self, generated: str):
        """Extract A/B/C/D choice from MMLU response"""
        import re
        match = re.search(r'[ABCD]', generated.upper())
        return match.group(0) if match else None

class SimulationProbes:
    """Simulation mode probes (no real model)"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.step_offset = int(time.time()) // 100  # Mock step progression
    
    def run_gsm8k_probe(self, dataset_path: str, sample_size: int = 100) -> dict:
        base_acc = 45 + (self.step_offset * 0.1)
        noise = random.uniform(-5, 5)
        accuracy = max(20, min(90, base_acc + noise)) / 100
        
        return {"task": "GSM8K-100", "acc": accuracy, "samples": sample_size, "simulation": True}
    
    def run_mmlu_probe(self, dataset_path: str, sample_size: int = 200) -> dict:
        base_acc = 52 + (self.step_offset * 0.05)
        noise = random.uniform(-3, 3)
        accuracy = max(35, min(80, base_acc + noise)) / 100
        
        return {"task": "MMLU-1k", "acc": accuracy, "samples": sample_size, "simulation": True}

def main():
    parser = argparse.ArgumentParser(description='Enhanced Mini-Validation Probes')
    parser.add_argument('--mode', default='simulation', choices=['real', 'simulation'], help='Probe mode')
    parser.add_argument('--ckpt', help='Model checkpoint path (for real mode)')
    parser.add_argument('--datasets', help='Comma-separated dataset paths')
    parser.add_argument('--out', help='Output directory')
    parser.add_argument('--every-steps', type=int, default=5000, help='Steps between probes')
    parser.add_argument('--no-block', action='store_true', help='Non-blocking mode')
    
    # Legacy args for backward compatibility
    parser.add_argument('--run', help='Run directory (legacy)')
    parser.add_argument('--interval', type=int, default=300, help='Check interval (legacy)')
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.out:
        output_dir = Path(args.out)
    elif args.run:
        output_dir = Path(args.run) / "probes" / f"real_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        output_dir = Path("probes_output")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse datasets
    if args.datasets:
        dataset_paths = args.datasets.split(',')
    else:
        # Default datasets
        dataset_paths = [
            "training/validation_datasets/gsm8k_100.json",
            "training/validation_datasets/mmlu_1k.json"
        ]
    
    print(f"🚀 Starting {'real' if args.mode == 'real' else 'simulation'} probes")
    print(f"   Output: {output_dir}")
    print(f"   Datasets: {len(dataset_paths)}")
    if args.mode == 'real' and args.ckpt:
        print(f"   Checkpoint: {args.ckpt}")
    
    # Initialize probe system
    if args.mode == 'real' and args.ckpt:
        if not REAL_MODE_AVAILABLE:
            print("❌ Real mode not available, falling back to simulation")
            probe_system = SimulationProbes(output_dir)
        else:
            probe_system = RealModelProbes(args.ckpt, dataset_paths, output_dir)
    else:
        probe_system = SimulationProbes(output_dir)
    
    # Run probes
    step = args.every_steps  # Start at first probe step
    
    while True:
        try:
            print(f"\n🔍 Running probes at step {step}")
            results = []
            
            # Run GSM8K if dataset available
            gsm8k_datasets = [d for d in dataset_paths if 'gsm8k' in d.lower()]
            if gsm8k_datasets:
                result = probe_system.run_gsm8k_probe(gsm8k_datasets[0])
                result.update({"step": step, "timestamp": time.time()})
                results.append(result)
            
            # Run MMLU if dataset available  
            mmlu_datasets = [d for d in dataset_paths if 'mmlu' in d.lower()]
            if mmlu_datasets:
                result = probe_system.run_mmlu_probe(mmlu_datasets[0])
                result.update({"step": step, "timestamp": time.time()})
                results.append(result)
            
            # Save results
            result_file = output_dir / f"probe_step_{step}.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"✅ Results saved to {result_file}")
            
            # Increment step for next iteration
            step += args.every_steps
            
            # Sleep or exit based on mode
            if args.no_block:
                print(f"⏳ Waiting {args.interval}s for next probe...")
                time.sleep(args.interval)
            else:
                break  # Single run
                
        except KeyboardInterrupt:
            print(f"\n⛔ Probes stopped at step {step}")
            break
        except Exception as e:
            print(f"❌ Probe error: {e}")
            if not args.no_block:
                break
            time.sleep(30)  # Wait before retry

if __name__ == "__main__":
    main()
