# pyright: reportPrivateUsage=false
from __future__ import annotations

from textwrap import dedent

from attrs import asdict

from ampform_dpd.decay import IsobarNode, Particle, State
from ampform_dpd.io import as_markdown_table, aslatex

# https://compwa-org--129.org.readthedocs.build/report/018.html#resonances-and-ls-scheme
dummy_args = {"mass": 0, "width": 0}
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
    assert latex == R"\left(\Lambda(1520) \to p K^-\right)"
    latex = aslatex(node, with_jp=True)
    expected = R"""
    \left(\Lambda(1520)\left[\frac{3}{2}^-\right] \to p\left[\frac{1}{2}^+\right] K^-\left[0^-\right]\right)
    """.strip()
    assert latex == expected

    node = IsobarNode(Λ1520, p, K, interaction=(2, 1))
    latex = aslatex(node)
    assert latex == R"\left(\Lambda(1520) \xrightarrow[S=1]{L=2} p K^-\right)"


def test_as_markdown_table_particles():
    p_state = State(**asdict(p), index=1)
    k_state = State(**asdict(K), index=2)
    particles = [p_state, k_state, π]
    src = as_markdown_table(particles)
    expected = dedent(R"""
    | index | name | LaTeX | $J^P$ | mass (MeV) | width (MeV) |
    | --- | --- | --- | --- | --- | --- |
    | 1 | `p` | $p$ | $\frac{1}{2}^+$ | 0 | 0 |
    | 2 | `K-` | $K^-$ | $0^-$ | 0 | 0 |
    |   | `π+` | $\pi^+$ | $0^-$ | 0 | 0 |
    """)
    assert src.strip() == expected.strip()
