from typing import Dict
import re

__all__ = ["entailment_cues"]

_ENTAIL = re.compile(r"\b(all|every|therefore|implies|hence)\b", re.I)
_CONTRA  = re.compile(r"\b(but|however|although|contradict|nevertheless)\b", re.I)

def entailment_cues(premise: str, hypothesis: str) -> Dict[str, bool]:
    p, h = premise or "", hypothesis or ""
    return {
        "p_entail": bool(_ENTAIL.search(p)),
        "h_entail": bool(_ENTAIL.search(h)),
        "contra":   bool(_CONTRA.search(p)) or bool(_CONTRA.search(h)),
    }
