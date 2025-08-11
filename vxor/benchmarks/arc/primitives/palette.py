from collections import Counter
from typing import Dict, List
import numpy as np

__all__ = ["extract_palette", "palette_hist"]

def extract_palette(grid: np.ndarray) -> List[int]:
    """Return sorted unique color ids present in the grid."""
    if grid.size == 0:
        return []
    return sorted(np.unique(grid).astype(int).tolist())

def palette_hist(grid: np.ndarray) -> Dict[int, int]:
    """Return histogram of color id -> count."""
    flat = grid.reshape(-1).astype(int).tolist()
    return dict(Counter(flat))
