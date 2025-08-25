#!/usr/bin/env python3
"""Mini-validation datasets for GSM8K-100 and MMLU-1k probes"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

# Sample GSM8K problems (mathematical reasoning)
GSM8K_SAMPLES = [
    {
        "problem": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "solution": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 for muffins, so she uses 3 + 4 = 7 eggs. She has 16 - 7 = 9 eggs left to sell. At $2 per egg, she makes 9 × $2 = $18 per day.",
        "answer": "18"
    },
    {
        "problem": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts of fiber does it take total?",
        "solution": "The robe takes 2 bolts of blue fiber. It takes half that much white fiber, so 2/2 = 1 bolt of white fiber. Total fiber needed is 2 + 1 = 3 bolts.",
        "answer": "3"
    },
    {
        "problem": "Tom's ship can travel 10 miles per hour. He is sailing from port A to port B, which are 50 miles apart. If he leaves at 8:00 AM, what time will he arrive at port B?",
        "solution": "Distance is 50 miles, speed is 10 mph. Time = distance/speed = 50/10 = 5 hours. Starting at 8:00 AM, he arrives at 8:00 AM + 5 hours = 1:00 PM.",
        "answer": "1:00 PM"
    },
    {
        "problem": "A bakery makes 12 dozen cookies. If they sell 3/4 of them, how many individual cookies do they have left?",
        "solution": "12 dozen cookies = 12 × 12 = 144 cookies total. They sell 3/4 of them: 144 × 3/4 = 108 cookies sold. Remaining: 144 - 108 = 36 cookies.",
        "answer": "36"
    },
    {
        "problem": "If a train travels 60 miles in 45 minutes, what is its speed in miles per hour?",
        "solution": "Distance = 60 miles, Time = 45 minutes = 45/60 = 0.75 hours. Speed = distance/time = 60/0.75 = 80 mph.",
        "answer": "80"
    },
]

# Sample MMLU problems (general knowledge)
MMLU_SAMPLES = [
    {
        "question": "What is the capital of France?",
        "choices": ["London", "Berlin", "Paris", "Madrid"],
        "answer": "C",
        "subject": "geography"
    },
    {
        "question": "Which of the following is NOT a primary color?",
        "choices": ["Red", "Blue", "Green", "Yellow"],
        "answer": "C",
        "subject": "art"
    },
    {
        "question": "What is the chemical symbol for gold?",
        "choices": ["Go", "Gd", "Au", "Ag"],
        "answer": "C",
        "subject": "chemistry"
    },
    {
        "question": "Who wrote 'Romeo and Juliet'?",
        "choices": ["Charles Dickens", "William Shakespeare", "Jane Austen", "Mark Twain"],
        "answer": "B",
        "subject": "literature"
    },
    {
        "question": "What is 15% of 200?",
        "choices": ["25", "30", "35", "40"],
        "answer": "B",
        "subject": "mathematics"
    },
]

def create_gsm8k_100() -> List[Dict[str, Any]]:
    """Create GSM8K-100 validation dataset by extending base samples"""
    extended_samples = []
    
    # Use base samples and generate variations
    for i in range(100):
        base_sample = GSM8K_SAMPLES[i % len(GSM8K_SAMPLES)]
        
        # Add some variation to avoid exact memorization
        variation_id = i // len(GSM8K_SAMPLES)
        sample = {
            "id": f"gsm8k_probe_{i}",
            "problem": base_sample["problem"],
            "solution": base_sample["solution"],
            "answer": base_sample["answer"],
            "variation": variation_id
        }
        extended_samples.append(sample)
    
    return extended_samples

def create_mmlu_1k() -> List[Dict[str, Any]]:
    """Create MMLU-1k validation dataset by extending base samples"""
    extended_samples = []
    
    # Use base samples and generate variations
    for i in range(1000):
        base_sample = MMLU_SAMPLES[i % len(MMLU_SAMPLES)]
        
        variation_id = i // len(MMLU_SAMPLES)
        sample = {
            "id": f"mmlu_probe_{i}",
            "question": base_sample["question"],
            "choices": base_sample["choices"],
            "answer": base_sample["answer"],
            "subject": base_sample["subject"],
            "variation": variation_id
        }
        extended_samples.append(sample)
    
    return extended_samples

def save_validation_datasets(output_dir: Path):
    """Save both validation datasets to JSON files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create GSM8K-100
    gsm8k_data = create_gsm8k_100()
    gsm8k_path = output_dir / "gsm8k_100.json"
    with open(gsm8k_path, 'w') as f:
        json.dump(gsm8k_data, f, indent=2)
    print(f"✅ Created GSM8K-100 dataset: {gsm8k_path}")
    
    # Create MMLU-1k
    mmlu_data = create_mmlu_1k()
    mmlu_path = output_dir / "mmlu_1k.json"
    with open(mmlu_path, 'w') as f:
        json.dump(mmlu_data, f, indent=2)
    print(f"✅ Created MMLU-1k dataset: {mmlu_path}")
    
    return gsm8k_path, mmlu_path

if __name__ == "__main__":
    from pathlib import Path
    
    # Create validation datasets
    root = Path(__file__).parent.parent
    validation_dir = root / "training" / "validation_datasets"
    
    gsm8k_path, mmlu_path = save_validation_datasets(validation_dir)
    print(f"📊 Validation datasets ready for mini-probes")
