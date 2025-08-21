"""
Reproducibility utilities for MISO benchmarks
Collects build fingerprint and ensures deterministic execution
"""

import os
import sys
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import random
import logging

logger = logging.getLogger(__name__)

class ReproducibilityCollector:
    """Collects reproducibility information for benchmark reports"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        
    def collect_git_info(self, repo_path: Path = None) -> Dict[str, str]:
        """Collect git commit and tag information"""
        if repo_path is None:
            repo_path = Path(__file__).parent.parent.parent
            
        git_info = {}
        
        try:
            # Get current commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], 
                cwd=repo_path, 
                capture_output=True, 
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                git_info["git_commit"] = result.stdout.strip()
            else:
                git_info["git_commit"] = "unknown"
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            git_info["git_commit"] = "unknown"
            
        try:
            # Get current tag if on a tagged commit
            result = subprocess.run(
                ["git", "describe", "--exact-match", "--tags", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                git_info["git_tag"] = result.stdout.strip()
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # No tag or git not available - this is optional
            pass
            
        return git_info
        
    def collect_env_flags(self) -> Dict[str, str]:
        """Collect environment flags affecting reproducibility"""
        env_flags = {}
        
        # Reproducibility-critical environment variables
        repro_vars = [
            "PYTHONHASHSEED",
            "OMP_NUM_THREADS", 
            "MKL_NUM_THREADS",
            "CUDA_DEVICE_ORDER",
            "CUBLAS_WORKSPACE_CONFIG"
        ]
        
        for var in repro_vars:
            value = os.getenv(var)
            if value is not None:
                env_flags[var] = value
                
        return env_flags
        
    def collect_platform_info(self) -> str:
        """Collect detailed platform information"""
        return f"{platform.system()}-{platform.release()}-{platform.machine()}"
        
    def collect_python_version(self) -> str:
        """Collect Python version information"""
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
    def collect_reproducibility_block(self, compute_mode: str = "full") -> Dict[str, Any]:
        """Collect complete reproducibility information"""
        git_info = self.collect_git_info()
        
        repro_block = {
            "git_commit": git_info.get("git_commit", "unknown"),
            "python_version": self.collect_python_version(),
            "platform": self.collect_platform_info(),
            "env_flags": self.collect_env_flags(),
            "seed": self.seed,
            "compute_mode": compute_mode
        }
        
        # Add git_tag only if available
        if "git_tag" in git_info:
            repro_block["git_tag"] = git_info["git_tag"]
            
        return repro_block

def seed_everything(seed: int = 42) -> None:
    """Set seeds for all random number generators"""
    # Python's built-in random
    random.seed(seed)
    
    # NumPy (if available)
    try:
        import numpy as np
        np.random.seed(seed)
        logger.debug(f"NumPy seed set to {seed}")
    except ImportError:
        logger.debug("NumPy not available, skipping seed")
        
    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug(f"PyTorch seed set to {seed}")
    except ImportError:
        logger.debug("PyTorch not available, skipping seed")
        
    # Set Python hash seed (must be done before Python startup)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    logger.info(f"🎲 All available RNGs seeded with: {seed}")

def ensure_deterministic_env() -> None:
    """Ensure environment is set up for deterministic execution"""
    # Set thread counts for consistent performance
    for var, default in [
        ("OMP_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"), 
        ("NUMEXPR_NUM_THREADS", "1")
    ]:
        if var not in os.environ:
            os.environ[var] = default
            logger.debug(f"Set {var}={default}")
            
    # PyTorch deterministic settings
    try:
        import torch
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
            logger.debug("PyTorch deterministic algorithms enabled")
    except ImportError:
        pass
