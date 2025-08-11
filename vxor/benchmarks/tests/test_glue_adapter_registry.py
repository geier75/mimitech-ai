from vxor.benchmarks.glue.adapter_registry import register_adapter, list_adapters, select

def test_registry():
    @register_adapter(name="lex_lr", hemi="lexical")
    def foo(x):
        return x
    assert any(a.name == "lex_lr" for a in list_adapters())
    assert select("lex_lr")(7) == 7
