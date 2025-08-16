#!/usr/bin/env python3
"""
MISO Dataset Downloader with SHA256 Verification
===============================================

Downloads authentic datasets for VXOR benchmarks with integrity checking.
Ensures reproducible benchmark results with verified data sources.

Usage:
    python scripts/download_datasets.py [dataset_name]
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --verify-only

Environment:
    MISO_DATA_DIR: Override default data directory (default: ./data/authentic)
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dataset registry with authentic sources and checksums
DATASETS = {
    "gsm8k": {
        "name": "GSM8K - Grade School Math Word Problems", 
        "url": "https://raw.githubusercontent.com/openai/grade-school-math/main/grade_school_math/data/train.jsonl",
        "sha256": "b8b2b77e5b4b1c4d6e3f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d",  # Placeholder - update with real hash
        "dst": "data/authentic/gsm8k/train.jsonl",
        "license": "MIT",
        "description": "Mathematical reasoning problems for grade school level",
        "size_mb": 2.1
    },
    "mmlu": {
        "name": "MMLU - Massive Multitask Language Understanding",
        "url": "https://people.eecs.berkeley.edu/~hendrycks/data.tar",
        "sha256": "c4f4e5b0b1a2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8",  # Placeholder
        "dst": "data/authentic/mmlu/data.tar", 
        "license": "MIT",
        "description": "57 subjects covering STEM, humanities, social sciences",
        "size_mb": 165.2,
        "extract": True
    },
    "arc_challenge": {
        "name": "ARC Challenge - AI2 Reasoning Challenge",
        "url": "https://s3-us-west-2.amazonaws.com/ai2-website/data/ARC-V1-Feb2018-2.zip",
        "sha256": "f1e2d3c4b5a6978d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d",  # Placeholder
        "dst": "data/authentic/arc/ARC-V1-Feb2018-2.zip",
        "license": "Apache 2.0", 
        "description": "Science questions requiring reasoning",
        "size_mb": 2.9,
        "extract": True
    },
    "hellaswag": {
        "name": "HellaSwag - Commonsense NLI",
        "url": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
        "sha256": "a1b2c3d4e5f6789a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a",  # Placeholder
        "dst": "data/authentic/hellaswag/hellaswag_train.jsonl", 
        "license": "MIT",
        "description": "Commonsense reasoning about physical situations",
        "size_mb": 156.8
    },
    "humaneval": {
        "name": "HumanEval - Code Generation",
        "url": "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl",
        "sha256": "b2c3d4e5f6a7890b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b",  # Placeholder
        "dst": "data/authentic/humaneval/HumanEval.jsonl",
        "license": "MIT", 
        "description": "Python programming problems",
        "size_mb": 0.2
    },
    "swe_bench": {
        "name": "SWE-bench - Software Engineering Benchmark",
        "url": "https://raw.githubusercontent.com/princeton-nlp/SWE-bench/main/swebench/test.jsonl", 
        "sha256": "c3d4e5f6a7b8901c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c",  # Placeholder
        "dst": "data/authentic/swe_bench/test.jsonl",
        "license": "MIT",
        "description": "Real-world software engineering tasks",
        "size_mb": 45.7
    },
    "medmcqa": {
        "name": "MedMCQA - Medical Multiple Choice QA",
        "url": "https://drive.google.com/uc?id=15VkJdq5eyWIkfb_aoD3oS8i4tScbHYky",  # Note: May need special handling
        "sha256": "d4e5f6a7b8c9012d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d",  # Placeholder
        "dst": "data/authentic/medmcqa/train.json",
        "license": "CC BY 4.0", 
        "description": "Medical entrance exam questions",
        "size_mb": 89.3
    }
}

def sha256_file(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"Error computing hash for {filepath}: {e}")
        return ""

def download_file(url: str, dst_path: Path, expected_size_mb: Optional[float] = None) -> bool:
    """Download file with progress indication."""
    try:
        logger.info(f"Downloading {url} -> {dst_path}")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        with urllib.request.urlopen(url) as response, open(dst_path, 'wb') as out_file:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"\rProgress: {progress:.1f}%", end='', flush=True)
            
        print()  # New line after progress
        
        # Verify file size if expected
        if expected_size_mb:
            actual_size_mb = dst_path.stat().st_size / (1024 * 1024)
            if abs(actual_size_mb - expected_size_mb) > 0.1:  # Allow 0.1MB tolerance
                logger.warning(f"Size mismatch: expected {expected_size_mb}MB, got {actual_size_mb:.1f}MB")
                
        logger.info(f"Downloaded {dst_path} ({dst_path.stat().st_size / (1024*1024):.1f} MB)")
        return True
        
    except Exception as e:
        logger.error(f"Download failed for {url}: {e}")
        if dst_path.exists():
            dst_path.unlink()  # Clean up partial download
        return False

def verify_dataset(dataset_name: str, dataset_info: Dict[str, Any], base_dir: Path) -> bool:
    """Verify dataset integrity."""
    dst_path = base_dir / dataset_info["dst"]
    
    if not dst_path.exists():
        logger.error(f"Dataset file not found: {dst_path}")
        return False
    
    logger.info(f"Verifying {dataset_name}...")
    actual_hash = sha256_file(dst_path)
    expected_hash = dataset_info["sha256"]
    
    if actual_hash == expected_hash:
        logger.info(f"✅ {dataset_name} verification passed")
        return True
    else:
        logger.error(f"❌ {dataset_name} verification failed!")
        logger.error(f"   Expected: {expected_hash}")
        logger.error(f"   Actual:   {actual_hash}")
        return False

def download_dataset(dataset_name: str, dataset_info: Dict[str, Any], base_dir: Path) -> bool:
    """Download and verify a single dataset."""
    dst_path = base_dir / dataset_info["dst"]
    
    # Skip if already exists and verified
    if dst_path.exists() and verify_dataset(dataset_name, dataset_info, base_dir):
        logger.info(f"Dataset {dataset_name} already exists and verified")
        return True
    
    # Download
    success = download_file(
        dataset_info["url"], 
        dst_path, 
        dataset_info.get("size_mb")
    )
    
    if not success:
        return False
    
    # Verify
    if not verify_dataset(dataset_name, dataset_info, base_dir):
        dst_path.unlink()  # Remove corrupted download
        return False
    
    # Extract if needed  
    if dataset_info.get("extract", False):
        logger.info(f"Extracting {dst_path}...")
        try:
            import tarfile
            import zipfile
            
            if dst_path.suffix == '.tar':
                with tarfile.open(dst_path, 'r') as tar:
                    tar.extractall(path=dst_path.parent)
            elif dst_path.suffix == '.zip':
                with zipfile.ZipFile(dst_path, 'r') as zip_ref:
                    zip_ref.extractall(dst_path.parent)
                    
            logger.info(f"Extracted {dst_path}")
        except Exception as e:
            logger.error(f"Extraction failed for {dst_path}: {e}")
            return False
    
    return True

def update_checksums(base_dir: Path):
    """Update checksums for existing files (development helper)."""
    logger.info("Updating checksums for existing files...")
    
    for name, info in DATASETS.items():
        dst_path = base_dir / info["dst"]
        if dst_path.exists():
            actual_hash = sha256_file(dst_path)
            logger.info(f"{name}: {actual_hash}")

def generate_manifest(base_dir: Path) -> Dict[str, Any]:
    """Generate dataset manifest with metadata."""
    manifest = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "miso_version": "15.32.28",
        "datasets": {}
    }
    
    for name, info in DATASETS.items():
        dst_path = base_dir / info["dst"] 
        if dst_path.exists():
            manifest["datasets"][name] = {
                "name": info["name"],
                "description": info["description"],
                "license": info["license"],
                "file_path": str(dst_path.relative_to(base_dir)),
                "sha256": sha256_file(dst_path),
                "size_bytes": dst_path.stat().st_size,
                "downloaded_at": __import__("datetime").datetime.fromtimestamp(dst_path.stat().st_mtime).isoformat()
            }
    
    return manifest

def main():
    parser = argparse.ArgumentParser(description="Download and verify MISO benchmark datasets")
    parser.add_argument("dataset", nargs="?", help="Dataset name to download")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing datasets")
    parser.add_argument("--update-checksums", action="store_true", help="Update checksums for existing files")
    parser.add_argument("--data-dir", help="Override data directory")
    
    args = parser.parse_args()
    
    # Determine base directory
    base_dir = Path(args.data_dir) if args.data_dir else Path(os.getenv("MISO_DATA_DIR", "data/authentic"))
    base_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Using data directory: {base_dir.absolute()}")
    
    if args.list:
        print("\nAvailable datasets:")
        print("=" * 50)
        for name, info in DATASETS.items():
            print(f"{name:15} - {info['name']} ({info['size_mb']}MB)")
            print(f"{'':15}   {info['description']}")
            print(f"{'':15}   License: {info['license']}")
            print()
        return
    
    if args.update_checksums:
        update_checksums(base_dir)
        return
        
    if args.verify_only:
        logger.info("Verifying existing datasets...")
        all_verified = True
        for name, info in DATASETS.items():
            if not verify_dataset(name, info, base_dir):
                all_verified = False
        
        if all_verified:
            logger.info("✅ All datasets verified successfully")
            sys.exit(0)
        else:
            logger.error("❌ Some datasets failed verification")
            sys.exit(1)
    
    # Download datasets
    if args.all:
        datasets_to_download = DATASETS.items()
    elif args.dataset:
        if args.dataset not in DATASETS:
            logger.error(f"Unknown dataset: {args.dataset}")
            logger.error(f"Available: {', '.join(DATASETS.keys())}")
            sys.exit(1)
        datasets_to_download = [(args.dataset, DATASETS[args.dataset])]
    else:
        parser.print_help()
        return
    
    # Download and verify
    success_count = 0
    total_count = len(datasets_to_download)
    
    for name, info in datasets_to_download:
        logger.info(f"Processing {name} ({success_count + 1}/{total_count})")
        if download_dataset(name, info, base_dir):
            success_count += 1
            logger.info(f"✅ {name} completed successfully")
        else:
            logger.error(f"❌ {name} failed")
    
    # Generate manifest
    manifest = generate_manifest(base_dir)
    manifest_path = base_dir / "datasets_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Generated manifest: {manifest_path}")
    logger.info(f"Summary: {success_count}/{total_count} datasets processed successfully")
    
    if success_count == total_count:
        logger.info("🎯 All datasets ready for benchmarking!")
        sys.exit(0)
    else:
        logger.error(f"⚠️ {total_count - success_count} datasets failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
