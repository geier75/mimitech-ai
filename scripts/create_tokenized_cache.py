#!/usr/bin/env python3
"""
Create tokenized cache for Phase 2 training data
This pre-processes all JSONL files into tokenized format for faster training
"""

import os
import json
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

def main():
    ROOT = Path("/Volumes/My Book/MISO_Ultimate 15.32.28")
    
    # Load mixing configuration
    mix_path = ROOT / "training/configs/mixing_phase2.json"
    if not mix_path.exists():
        print(f"❌ Missing mixing config: {mix_path}")
        return
        
    mix = json.load(open(mix_path))
    
    # Load tokenizer
    tokenizer_path = ROOT / "models/tinyllama"
    if not tokenizer_path.exists():
        print(f"❌ Missing tokenizer: {tokenizer_path}")
        return
        
    print("🔄 Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(str(tokenizer_path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    def format_example(example):
        """Format example for chat template"""
        prompt = example.get("prompt") or example.get("input") or example.get("instruction") or ""
        response = example.get("response") or example.get("output") or example.get("completion") or ""
        return {"text": f"
