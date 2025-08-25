#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real LoRA/SFT Training Script for Phase 2 B2C Pipeline
Optimized for Apple Silicon MPS acceleration

Copyright (c) 2025 MISO Tech. All rights reserved.
"""

import os
import json
import math
import random
import logging
from pathlib import Path
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [TRAINING] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MISO.Phase2Training")

# --------- Config ---------
ROOT = Path(os.environ.get("ROOT", "."))
BASE_MODEL = (ROOT/"models/tinyllama").as_posix()
MIXING_JSON = (ROOT/"training/configs/mixing_phase2.json")
OUTDIR = (ROOT/"runs/phase2_sft_lora_real").as_posix()
SEED = 42
MAX_LEN = 512  # Reduced for MPS compatibility
MICRO_BSZ = 1
GRAD_ACC = 4   # Reduced for memory
LR = 2e-4
EPOCHS = 1
SAVE_STEPS = 50
EVAL_STEPS = 50
LOG_STEPS = 10

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")
logger.info(f"Output directory: {OUTDIR}")

torch.manual_seed(SEED)
random.seed(SEED)

def load_jsonl(path: str) -> Dataset:
    """Load JSONL dataset"""
    logger.info(f"Loading dataset: {path}")
    return load_dataset("json", data_files=path, split="train", streaming=False)

def format_example(ex):
    """Format example for training"""
    # Map different field names to standard format
    prompt = ""
    response = ""
    
    # Try different field combinations
    if "problem" in ex and "solution" in ex:
        prompt = ex.get("problem", "")
        response = ex.get("solution", "")
    elif "prompt" in ex and "response" in ex:
        prompt = ex.get("prompt", "")
        response = ex.get("response", "")
    elif "input" in ex and "output" in ex:
        prompt = ex.get("input", "")
        response = ex.get("output", "")
    elif "instruction" in ex and "completion" in ex:
        prompt = ex.get("instruction", "")
        response = ex.get("completion", "")
    else:
        # Fallback: use first available text field
        for field in ["problem", "question", "text", "content"]:
            if field in ex:
                prompt = ex.get(field, "")
                break
        for field in ["solution", "answer", "response", "output"]:
            if field in ex:
                response = ex.get(field, "")
                break
    
    # Create chat format
    text = f"
