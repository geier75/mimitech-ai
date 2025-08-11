import numpy as np
from vxor.benchmarks.arc.primitives.palette import extract_palette, palette_hist
from vxor.benchmarks.arc.primitives.symmetry import detect_axes, reflect
from vxor.benchmarks.arc.primitives.periodicity import row_period, col_period
from vxor.benchmarks.arc.primitives.cca import connected_components

def test_palette():
    g = np.array([[1,2,2],[3,1,0]])
    assert extract_palette(g) == [0,1,2,3]
    h = palette_hist(g)
    assert h[2] == 2 and h[1] == 2

def test_symmetry():
    g = np.array([[1,0],[1,0]])
    axes = detect_axes(g)
    assert axes["v"] is False
    assert axes["h"] is True
    g2 = reflect(g, "v")
    assert (g2 == np.fliplr(g)).all()

def test_periods():
    g = np.array([[1,2,1,2],[1,2,1,2]])
    assert col_period(g) == 2
    assert row_period(g) == 1

def test_cca():
    g = np.array([[1,1,0],[0,0,0],[2,2,2]])
    comps = connected_components(g)
    assert len(comps) >= 3
