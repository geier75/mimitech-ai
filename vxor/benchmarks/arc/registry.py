from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

__all__ = ["register_heuristic", "list_heuristics", "get", "HeuristicSpec"]

@dataclass
class HeuristicSpec:
    name: str
    hemi: str  # "geo" | "topo"
    cost: float
    fn: Callable

_REG: Dict[str, HeuristicSpec] = {}

def register_heuristic(name: str, hemi: str, cost: float):
    def deco(fn: Callable):
        _REG[name] = HeuristicSpec(name=name, hemi=hemi, cost=float(cost), fn=fn)
        return fn
    return deco

def list_heuristics() -> List[HeuristicSpec]:
    return list(_REG.values())

def get(name: str) -> Optional[Callable]:
    spec = _REG.get(name)
    return spec.fn if spec else None
