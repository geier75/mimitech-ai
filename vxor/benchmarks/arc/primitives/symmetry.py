from typing import Dict
import numpy as np

__all__ = ["detect_axes", "reflect"]

def detect_axes(grid: np.ndarray) -> Dict[str, bool]:
    """Detect horizontal/vertical symmetry axes. Returns {"h": bool, "v": bool}."""
    res = {"h": False, "v": False}
    if grid.size == 0:
        return res
    hflip = np.flipud(grid)
    vflip = np.fliplr(grid)
    res["h"] = np.array_equal(grid, hflip)
    res["v"] = np.array_equal(grid, vflip)
    return res

def reflect(grid: np.ndarray, axis: str) -> np.ndarray:
    if axis == "h":
        return np.flipud(grid)
    if axis == "v":
        return np.fliplr(grid)
    raise ValueError(f"unknown axis: {axis}")
