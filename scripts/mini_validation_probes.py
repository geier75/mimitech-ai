#!/usr/bin/env python3
"""Mini-Validation Probes für GSM8K-100 und MMLU-1k alle 5k Steps"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re

class MiniValidationProbes:
    """Mini-Validation Probes alle 5k Steps ohne Training-Unterbrechung"""
    
    def __init__(self, model_path: str, validation_data_dir: Path):
        self.model_path = model_path
        self.validation_data_dir = validation_data_dir
        self.gsm8k_data = None
        self.mmlu_data = None
        self.probe_results = []
        
    def load_validation_datasets(self):
        """Lade GSM8K-100 und MMLU-1k Datasets"""
        gsm8k_path = self.validation_data_dir / "gsm8k_100.json"
        mmlu_path = self.validation_data_dir / "mmlu_1k.json"
        
        if gsm8k_path.exists():
            with open(gsm8k_path) as f:
                self.gsm8k_data = json.load(f)
        
        if mmlu_path.exists():
            with open(mmlu_path) as f:
                self.mmlu_data = json.load(f)
    
    def run_gsm8k_probe(self, model, tokenizer, sample_size: int = 100) -> Dict[str, float]:
        """GSM8K-100 Mini-Probe ausführen"""
        if not self.gsm8k_data:
            return {"gsm8k_accuracy": 0.0, "gsm8k_samples": 0}
        
        correct = 0
        total = min(sample_size, len(self.gsm8k_data))
        
        for i in range(total):
            sample = self.gsm8k_data[i]
            problem = sample["problem"]
            expected_answer = sample["answer"]
            
            # Format prompt
            prompt = f"Problem: {problem}\nSolution:"
            
            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        do_sample=False,
                        temperature=0.1,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated = response[len(prompt):].strip()
                
                # Extract numerical answer
                if self._extract_answer_matches(generated, expected_answer):
                    correct += 1
                    
            except Exception as e:
                continue
        
        accuracy = (correct / total) * 100 if total > 0 else 0.0
        return {"gsm8k_accuracy": accuracy, "gsm8k_samples": total}
    
    def run_mmlu_probe(self, model, tokenizer, sample_size: int = 200) -> Dict[str, float]:
        """MMLU-1k Mini-Probe ausführen (200 Samples für Speed)"""
        if not self.mmlu_data:
            return {"mmlu_accuracy": 0.0, "mmlu_samples": 0}
        
        correct = 0
        total = min(sample_size, len(self.mmlu_data))
        
        for i in range(total):
            sample = self.mmlu_data[i]
            question = sample["question"]
            choices = sample["choices"]
            correct_answer = sample["answer"]  # A, B, C, D
            
            # Format multiple choice prompt
            prompt = f"Question: {question}\n"
            for idx, choice in enumerate(choices):
                prompt += f"{chr(65+idx)}. {choice}\n"
            prompt += "Answer:"
            
            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        temperature=0.1,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated = response[len(prompt):].strip()
                
                # Extract first letter (A, B, C, D)
                predicted_answer = self._extract_choice(generated)
                if predicted_answer == correct_answer:
                    correct += 1
                    
            except Exception as e:
                continue
        
        accuracy = (correct / total) * 100 if total > 0 else 0.0
        return {"mmlu_accuracy": accuracy, "mmlu_samples": total}
    
    def _extract_answer_matches(self, generated: str, expected: str) -> bool:
        """Extrahiere numerische Antwort aus GSM8K Response"""
        # Suche nach Zahlen im generierten Text
        numbers = re.findall(r'\d+\.?\d*', generated)
        if not numbers:
            return False
        
        # Vergleiche mit erwarteter Antwort
        try:
            expected_num = float(expected)
            for num_str in numbers:
                if abs(float(num_str) - expected_num) < 0.01:
                    return True
        except ValueError:
            # String comparison als Fallback
            return expected.lower() in generated.lower()
        
        return False
    
    def _extract_choice(self, generated: str) -> Optional[str]:
        """Extrahiere A/B/C/D Choice aus MMLU Response"""
        # Suche nach ersten Buchstaben A-D
        match = re.search(r'[ABCD]', generated.upper())
        return match.group(0) if match else None
    
    def run_mini_probe(self, step: int) -> Dict[str, Any]:
        """Führe Mini-Probe für aktuellen Training Step aus"""
        print(f"🔍 Running mini-validation probe at step {step}")
        
        try:
            # Lade Model und Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                torch_dtype=torch.float16,
                device_map="auto"
            )
            model.eval()
            
            # Führe beide Probes aus
            gsm8k_results = self.run_gsm8k_probe(model, tokenizer)
            mmlu_results = self.run_mmlu_probe(model, tokenizer)
            
            # Kombiniere Ergebnisse
            probe_result = {
                "step": step,
                "timestamp": torch.cuda.current_device() if torch.cuda.is_available() else 0,
                **gsm8k_results,
                **mmlu_results
            }
            
            self.probe_results.append(probe_result)
            
            # Log Results
            print(f"📊 Step {step} Mini-Probe Results:")
            print(f"   GSM8K-100: {gsm8k_results['gsm8k_accuracy']:.1f}% ({gsm8k_results['gsm8k_samples']} samples)")
            print(f"   MMLU-200:  {mmlu_results['mmlu_accuracy']:.1f}% ({mmlu_results['mmlu_samples']} samples)")
            
            return probe_result
            
        except Exception as e:
            print(f"❌ Mini-probe failed at step {step}: {e}")
            return {"step": step, "error": str(e)}
    
    def save_probe_results(self, output_path: Path):
        """Speichere alle Probe Results"""
        with open(output_path, 'w') as f:
            json.dump(self.probe_results, f, indent=2)
        print(f"💾 Saved {len(self.probe_results)} probe results to {output_path}")

def setup_mini_validation_datasets():
    """Setup Mini-Validation Datasets falls nicht vorhanden"""
    from mini_validation_datasets import save_validation_datasets
    
    root = Path(__file__).parent.parent
    validation_dir = root / "training" / "validation_datasets"
    
    gsm8k_path = validation_dir / "gsm8k_100.json"
    mmlu_path = validation_dir / "mmlu_1k.json"
    
    if not gsm8k_path.exists() or not mmlu_path.exists():
        print("🔧 Setting up mini-validation datasets...")
        save_validation_datasets(validation_dir)
    
    return validation_dir

if __name__ == "__main__":
    import sys
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='Mini-Validation Probes')
    parser.add_argument('--run', required=True, help='Path to run directory')
    parser.add_argument('--interval', type=int, default=300, help='Check interval in seconds')
    parser.add_argument('--step-interval', type=int, default=5000, help='Steps between probes')
    args = parser.parse_args()
    
    validation_dir = setup_mini_validation_datasets()
    probes = MiniValidationProbes(
        model_path=args.run,
        validation_data_dir=validation_dir
    )
    probes.load_validation_datasets()
    
    print(f"🚀 Starting continuous mini-validation probes")
    print(f"   Run: {args.run}")
    print(f"   Check interval: {args.interval}s")
    print(f"   Step interval: {args.step_interval}")
    print(f"   GSM8K samples: {len(probes.gsm8k_data) if probes.gsm8k_data else 0}")
    print(f"   MMLU samples: {len(probes.mmlu_data) if probes.mmlu_data else 0}")
    
    last_step = 0
    while True:
        try:
            # Check for training progress (mock implementation)
            current_step = (int(time.time()) // 10) * args.step_interval  # Mock step counter
            
            if current_step > last_step and current_step % args.step_interval == 0:
                result = probes.run_mini_probe(current_step)
                last_step = current_step
                
                # Save individual result
                result_file = Path(args.run) / "probes" / f"probe_step_{current_step}.json"
                result_file.parent.mkdir(exist_ok=True)
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
            
            time.sleep(args.interval)
            
        except KeyboardInterrupt:
            print(f"\n⛔ Probes stopped by user")
            break
        except Exception as e:
            print(f"❌ Probe error: {e}")
            time.sleep(args.interval)
