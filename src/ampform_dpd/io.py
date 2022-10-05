"""Input-output functions for `ampform` and `sympy` objects.

This module extends AmpForm's :func:`~ampform.io.aslatex` function. You can register
more implementations as follows:

>>> from ampform_dpd.io import aslatex
>>> @aslatex.register(int)
... def _(obj: int) -> str:
...     return "my custom rendering"
>>> aslatex(1)
'my custom rendering'
>>> aslatex(3.4 - 2j)
'3.4-2i'

This code originates from `ComPWA/ampform#280
<https://github.com/ComPWA/ampform/pull/280>`_.
"""
from __future__ import annotations

from collections import abc
from typing import Iterable, Mapping, Sequence

import sympy as sp
from ampform.io import aslatex

from ampform_dpd.decay import IsobarNode, Particle, ThreeBodyDecay, ThreeBodyDecayChain


@aslatex.register(complex)
def _(obj: complex, **kwargs) -> str:
    real = __downcast(obj.real)
    imag = __downcast(obj.imag)
    plus = "+" if imag >= 0 else ""
    return f"{real}{plus}{imag}i"


def __downcast(obj: float) -> float | int:
    if obj.is_integer():
        return int(obj)
    return obj


@aslatex.register(sp.Basic)
def _(obj: sp.Basic, **kwargs) -> str:
    return sp.latex(obj)


@aslatex.register(abc.Mapping)
def _(obj: Mapping, **kwargs) -> str:
    if len(obj) == 0:
        raise ValueError("Need at least one dictionary item")
    latex = R"\begin{array}{rcl}" + "\n"
    for lhs, rhs in obj.items():
        latex += Rf"  {aslatex(lhs, **kwargs)} &=& {aslatex(rhs, **kwargs)} \\" + "\n"
    latex += R"\end{array}"
    return latex


@aslatex.register(abc.Iterable)
def _(obj: Iterable, **kwargs) -> str:
    obj = list(obj)
    if len(obj) == 0:
        raise ValueError("Need at least one item to render as LaTeX")
    latex = R"\begin{array}{c}" + "\n"
    for item in obj:
        item_latex = aslatex(item, **kwargs)
        latex += Rf"  {item_latex} \\" + "\n"
    latex += R"\end{array}"
    return latex


@aslatex.register(IsobarNode)
def _(obj: IsobarNode, **kwargs) -> str:
    def render_arrow(node: IsobarNode) -> str:
        if node.interaction is None:
            return R"\to"
        return Rf"\xrightarrow[S={node.interaction.S}]{{L={node.interaction.L}}}"

    parent = aslatex(obj.parent, **kwargs)
    to = render_arrow(obj)
    child1 = aslatex(obj.child1, **kwargs)
    child2 = aslatex(obj.child2, **kwargs)
    return Rf"{parent} {to} {child1} {child2}"


@aslatex.register(ThreeBodyDecay)
def _(obj: ThreeBodyDecay, **kwargs) -> str:
    return aslatex(obj.chains, **kwargs)


@aslatex.register(ThreeBodyDecayChain)
def _(obj: ThreeBodyDecayChain, **kwargs) -> str:
    return aslatex(obj.decay, **kwargs)


@aslatex.register(Particle)
def _(obj: Particle, with_jp: bool = False, only_jp: bool = False, **kwargs) -> str:
    if only_jp:
        return _render_jp(obj)
    if with_jp:
        jp = _render_jp(obj)
        return Rf"{obj.latex}\left[{jp}\right]"
    return obj.latex


def _render_jp(particle: Particle) -> str:
    parity = "-" if particle.parity < 0 else "+"
    if particle.spin.denominator == 1:
        spin = sp.latex(particle.spin)
    else:
        spin = Rf"\frac{{{particle.spin.numerator}}}{{{particle.spin.denominator}}}"
    return f"{spin}^{parity}"


def as_markdown_table(obj: Sequence) -> str:
    """Render objects a `str` suitable for generating a table."""
    item_type = _determine_item_type(obj)
    if item_type is Particle:
        return _as_resonance_markdown_table(obj)
    if item_type is ThreeBodyDecay:
        return _as_decay_markdown_table(obj.chains)
    if item_type is ThreeBodyDecayChain:
        return _as_decay_markdown_table(obj)
    raise NotImplementedError(
        f"Cannot render a sequence with {item_type.__name__} items as a Markdown table"
    )


def _determine_item_type(obj) -> type:
    if not isinstance(obj, abc.Sequence):
        return type(obj)
    if len(obj) < 1:
        raise ValueError(f"Need at least one entry to render a table")
    item_type = type(obj[0])
    if not all(map(lambda i: isinstance(i, item_type), obj)):
        raise ValueError(f"Not all items are of type {item_type.__name__}")
    return item_type


def _as_resonance_markdown_table(items: Sequence[Particle]) -> str:
    column_names = [
        "name",
        "LaTeX",
        "$J^P$",
        "mass (MeV)",
        "width (MeV)",
    ]
    src = _create_markdown_table_header(column_names)
    for particle in items:
        row_items = [
            particle.name,
            f"${particle.latex}$",
            Rf"${aslatex(particle, only_jp=True)}$",
            f"{int(1e3 * particle.mass):,.0f}",
            f"{int(1e3 * particle.width):,.0f}",
        ]
        src += _create_markdown_table_row(row_items)
    return src


def _as_decay_markdown_table(decay_chains: Sequence[ThreeBodyDecayChain]) -> str:
    column_names = [
        "resonance",
        R"$J^P$",
        R"mass (MeV)",
        R"width (MeV)",
        R"$L_\mathrm{dec}^\mathrm{min}$",
        R"$L_\mathrm{prod}^\mathrm{min}$",
    ]
    src = _create_markdown_table_header(column_names)
    for chain in decay_chains:
        child1, child2 = map(aslatex, chain.decay_products)
        row_items = [
            Rf"${chain.resonance.latex} \to" Rf" {child1} {child2}$",
            Rf"${aslatex(chain.resonance, only_jp=True)}$",
            f"{int(1e3 * chain.resonance.mass):,.0f}",
            f"{int(1e3 * chain.resonance.width):,.0f}",
            chain.outgoing_ls.L,
            chain.incoming_ls.L,
        ]
        src += _create_markdown_table_row(row_items)
    return src


def _create_markdown_table_header(column_names: list[str]):
    src = _create_markdown_table_row(column_names)
    src += _create_markdown_table_row(["---" for _ in column_names])
    return src


def _create_markdown_table_row(items: Iterable):
    items = map(lambda i: f"{i}", items)
    return "| " + " | ".join(items) + " |\n"
