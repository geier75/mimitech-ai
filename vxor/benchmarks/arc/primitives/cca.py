from typing import List, Tuple
import numpy as np

__all__ = ["connected_components"]

_DIRS = [(1,0),(-1,0),(0,1),(0,-1)]  # 4-connectivity

def connected_components(grid: np.ndarray) -> List[np.ndarray]:
    """Return a list of boolean masks, one per connected component (4-neighborhood)."""
    if grid.size == 0:
        return []
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[np.ndarray] = []
    for i in range(h):
        for j in range(w):
            if seen[i, j]:
                continue
            color = grid[i, j]
            stack: List[Tuple[int,int]] = [(i, j)]
            mask = np.zeros((h, w), dtype=bool)
            seen[i, j] = True
            while stack:
                y, x = stack.pop()
                mask[y, x] = True
                for dy, dx in _DIRS:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and grid[ny, nx] == color:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(mask)
    return comps
