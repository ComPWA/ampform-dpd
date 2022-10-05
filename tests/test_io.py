# pyright: reportPrivateUsage=false
from __future__ import annotations

from os.path import abspath, dirname

from ampform_dpd.decay import IsobarNode, Particle
from ampform_dpd.io import aslatex

THIS_DIR = dirname(abspath(__file__))

# https://compwa-org--129.org.readthedocs.build/report/018.html#resonances-and-ls-scheme
dummy_args = dict(mass=0, width=0)
Λc = Particle("Λc", latex=R"\Lambda_c^+", spin=0.5, parity=+1, **dummy_args)
p = Particle("p", latex="p", spin=0.5, parity=+1, **dummy_args)
π = Particle("π+", latex=R"\pi^+", spin=0, parity=-1, **dummy_args)
K = Particle("K-", latex="K^-", spin=0, parity=-1, **dummy_args)
Λ1520 = Particle("Λ(1520)", latex=R"\Lambda(1520)", spin=1.5, parity=-1, **dummy_args)


def test_aslatex_particle():
    latex = aslatex(Λ1520)
    assert latex == Λ1520.latex
    latex = aslatex(Λ1520, only_jp=True)
    assert latex == R"\frac{3}{2}^-"
    latex = aslatex(Λ1520, with_jp=True)
    assert latex == Λ1520.latex + R"\left[\frac{3}{2}^-\right]"


def test_aslatex_isobar_node():
    node = IsobarNode(Λ1520, p, K)
    latex = aslatex(node)
    assert latex == R"\Lambda(1520) \to p K^-"
    latex = aslatex(node, with_jp=True)
    assert (
        latex == R"\Lambda(1520)\left[\frac{3}{2}^-\right] \to"
        R" p\left[\frac{1}{2}^+\right] K^-\left[0^-\right]"
    )

    node = IsobarNode(Λ1520, p, K, interaction=(2, 1))
    latex = aslatex(node)
    assert latex == R"\Lambda(1520) \xrightarrow[S=1]{L=2} p K^-"
