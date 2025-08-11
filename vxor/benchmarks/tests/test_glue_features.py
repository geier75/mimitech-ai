from vxor.benchmarks.glue.features.lexical import negation_markers, modal_markers
from vxor.benchmarks.glue.features.entailment import entailment_cues

def test_lexical():
    assert negation_markers("this is not fine")['has_neg'] is True
    assert modal_markers("we might go")['has_modal'] is True

def test_entail():
    cues = entailment_cues("All birds fly", "Therefore they travel")
    assert cues['p_entail'] is True and cues['h_entail'] is True
