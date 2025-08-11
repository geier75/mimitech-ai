from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

__all__ = ["register_adapter", "list_adapters", "select", "AdapterSpec"]

@dataclass
class AdapterSpec:
    name: str
    hemi: str  # "lexical" | "semantic"
    fn: Callable

_REG: Dict[str, AdapterSpec] = {}

def register_adapter(name: str, hemi: str):
    def deco(fn: Callable):
        _REG[name] = AdapterSpec(name=name, hemi=hemi, fn=fn)
        return fn
    return deco

def list_adapters() -> List[AdapterSpec]:
    return list(_REG.values())

def select(name: str) -> Optional[Callable]:
    spec = _REG.get(name)
    return spec.fn if spec else None
