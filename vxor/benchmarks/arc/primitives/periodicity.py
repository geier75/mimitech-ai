from typing import Optional
import numpy as np

__all__ = ["row_period", "col_period"]

def _fundamental_period(signal: np.ndarray) -> Optional[int]:
    n = signal.shape[0]
    for k in range(1, n + 1):
        if n % k != 0:
            continue
        tile = np.tile(signal[:k], n // k)
        if np.array_equal(tile, signal):
            return k
    return None

def row_period(grid: np.ndarray) -> Optional[int]:
    """Smallest repeating period along rows (vertical tiling)."""
    if grid.size == 0:
        return None
    return _fundamental_period(grid[:, 0])

def col_period(grid: np.ndarray) -> Optional[int]:
    """Smallest repeating period along cols (horizontal tiling)."""
    if grid.size == 0:
        return None
    return _fundamental_period(grid[0, :])
