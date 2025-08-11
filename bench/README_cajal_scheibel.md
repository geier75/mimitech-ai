# Scientist-Principle PR-A: Cajal & Scheibel

This document describes the Cajal micro-feature primitives (ARC) and Scheibel registries (ARC/GLUE) introduced in PR-A, including modules, APIs, and unit tests.

## Modules

- ARC Cajal primitives: `vxor.benchmarks.arc.primitives`
  - `extract_palette(grid) -> List[int]`: sorted unique color ids
  - `palette_hist(grid) -> Dict[int,int]`: color histogram
  - `detect_axes(grid) -> {"h": bool, "v": bool}`: symmetry axes
  - `reflect(grid, axis) -> ndarray`: flip along `"h"` or `"v"`
  - `row_period(grid) -> Optional[int]`: vertical tiling period
  - `col_period(grid) -> Optional[int]`: horizontal tiling period
  - `connected_components(grid) -> List[np.ndarray]`: boolean masks per 4-connected region

- ARC Scheibel registry: `vxor.benchmarks.arc.registry`
  - `@register_heuristic(name, hemi, cost)` → decorator
  - `list_heuristics() -> List[HeuristicSpec]`
  - `get(name) -> Optional[Callable]`
  - Hemisphere labels: `hemi ∈ {"geo", "topo"}`

- GLUE Cajal feature markers: `vxor.benchmarks.glue.features`
  - `negation_markers(text) -> {"has_neg": bool}`
  - `modal_markers(text) -> {"has_modal": bool}`
  - `entailment_cues(premise, hypothesis) -> {"p_entail": bool, "h_entail": bool, "contra": bool}`

- GLUE Scheibel adapter registry: `vxor.benchmarks.glue.adapter_registry`
  - `@register_adapter(name, hemi)` → decorator
  - `list_adapters() -> List[AdapterSpec]`
  - `select(name) -> Optional[Callable]`
  - Hemisphere labels: `hemi ∈ {"lexical", "semantic"}`

## Notes

- Implementations are deterministic and side-effect free; suitable for audit-logged harnesses.
- Packages expose `__all__` for stable imports; see `vxor/benchmarks/*/__init__.py`.

## Tests

Unit tests are provided under `vxor/benchmarks/tests/`:
- `test_arc_primitives.py`
- `test_arc_registry.py`
- `test_glue_features.py`
- `test_glue_adapter_registry.py`

Run just these tests:

```bash
python -m pytest -q \
  vxor/benchmarks/tests/test_arc_primitives.py \
  vxor/benchmarks/tests/test_arc_registry.py \
  vxor/benchmarks/tests/test_glue_features.py \
  vxor/benchmarks/tests/test_glue_adapter_registry.py
```

## Related

- Harness usage and adapter environment: see `bench/README_arc_glue.md` and `bench/README_vxor_adapters.md`.
