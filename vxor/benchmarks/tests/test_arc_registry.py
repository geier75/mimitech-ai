from vxor.benchmarks.arc.registry import register_heuristic, list_heuristics, get

def test_registry():
    @register_heuristic(name="dummy", hemi="geo", cost=1.0)
    def foo(x):
        return x
    assert any(h.name == "dummy" for h in list_heuristics())
    assert get("dummy")(42) == 42
