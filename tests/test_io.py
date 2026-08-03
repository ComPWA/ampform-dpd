from __future__ import annotations

from textwrap import dedent

import sympy as sp
from ampform.dynamics import EnergyDependentWidth
from ampform.dynamics.form_factor import FormFactor, SphericalHankel1
from attrs import asdict

from ampform_dpd.decay import IsobarNode, Particle, State
from ampform_dpd.dynamics import (
    BreitWigner,
    ChannelArguments,
    MultichannelBreitWigner,
    SimpleBreitWigner,
)
from ampform_dpd.io import as_markdown_table, aslatex, unfold_definitions

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
    node = IsobarNode(Λ1520, p, K)  # ty: ignore[invalid-argument-type]
    latex = aslatex(node)
    assert latex == R"\left(\Lambda(1520) \to p K^-\right)"
    latex = aslatex(node, with_jp=True)
    expected = R"""
    \left(\Lambda(1520)\left[\frac{3}{2}^-\right] \to p\left[\frac{1}{2}^+\right] K^-\left[0^-\right]\right)
    """.strip()
    assert latex == expected

    node = IsobarNode(Λ1520, p, K, interaction=(2, 1))  # ty: ignore[invalid-argument-type]
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


def test_unfold_definitions_recursively():
    s, m0, Γ0, m1, m2, L, R = sp.symbols("s m0 Gamma0 m1 m2 L R")
    expression = BreitWigner(s, m0, Γ0, m1, m2, L, R)
    definitions = unfold_definitions(expression)
    classes = [expr.func for expr in definitions]
    assert classes[:4] == [
        BreitWigner,
        SimpleBreitWigner,
        EnergyDependentWidth,
        FormFactor,
    ]
    assert len(classes) == len(set(classes))
    assert all(rhs != lhs for lhs, rhs in definitions.items())


def test_unfold_definitions_generalizes_composite_arguments():
    s, m1, m2, L, R = sp.symbols("s m1 m2 L R")
    definitions = unfold_definitions(FormFactor(s, m1, m2, L, R))
    hankel = next(expr for expr in definitions if isinstance(expr, SphericalHankel1))
    assert hankel.args == (L, sp.Symbol("z"))


def test_unfold_definitions_preserves_indexed_symbols():
    s, m0, R = sp.symbols("s m0 R")
    channels = tuple(
        ChannelArguments(
            s,
            m0,
            coupling_squared=sp.Symbol(f"g_{{{i}}}^2"),  # ty: ignore[unknown-argument]
            m1=sp.Symbol(f"m_{{a,{i}}}"),  # ty: ignore[unknown-argument]
            m2=sp.Symbol(f"m_{{b,{i}}}"),  # ty: ignore[unknown-argument]
            angular_momentum=sp.Symbol(f"L_{{{i}}}"),  # ty: ignore[unknown-argument]
            meson_radius=R,  # ty: ignore[unknown-argument]
        )
        for i in [1, 2]
    )
    definitions = unfold_definitions(MultichannelBreitWigner(s, m0, channels))  # ty: ignore[invalid-argument-type]
    channel = next(expr for expr in definitions if isinstance(expr, ChannelArguments))
    assert channel.args == channels[0].args
    symbol_names = {str(symbol) for expr in definitions for symbol in expr.free_symbols}
    assert "angular_momentum" not in symbol_names
    assert "coupling_squared" not in symbol_names


def test_unfold_definitions_ignores_evaluated_expression():
    assert unfold_definitions(sp.Symbol("x")) == {}
