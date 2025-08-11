from typing import Dict
import re

__all__ = ["negation_markers", "modal_markers"]

_NEG = re.compile(r"\b(no|not|never|n't|cannot|without)\b", re.I)
_MODAL = re.compile(r"\b(can|could|may|might|must|should|would)\b", re.I)

def negation_markers(text: str) -> Dict[str, bool]:
    return {"has_neg": bool(_NEG.search(text or ""))}

def modal_markers(text: str) -> Dict[str, bool]:
    return {"has_modal": bool(_MODAL.search(text or ""))}
