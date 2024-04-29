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

import logging
import pickle
from collections import abc
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence, overload

import cloudpickle
import sympy as sp
from ampform.io import aslatex
from ampform.sympy import (
    perform_cached_doit,  # noqa: F401  # pyright:ignore[reportUnusedImport]
)
from tensorwaves.function.sympy import create_function, create_parametrized_function

from ampform_dpd._cache import get_readable_hash, get_system_cache_directory
from ampform_dpd.decay import (
    IsobarNode,
    Particle,
    State,
    ThreeBodyDecay,
    ThreeBodyDecayChain,
)

if TYPE_CHECKING:
    from tensorwaves.function import (
        ParametrizedBackendFunction,
        PositionalArgumentFunction,
    )
    from tensorwaves.interface import Function, ParameterValue, ParametrizedFunction

_LOGGER = logging.getLogger(__name__)


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
        msg = "Need at least one dictionary item"
        raise ValueError(msg)
    latex = R"\begin{array}{rcl}" + "\n"
    for lhs, rhs in obj.items():
        latex += Rf"  {aslatex(lhs, **kwargs)} &=& {aslatex(rhs, **kwargs)} \\" + "\n"
    latex += R"\end{array}"
    return latex


@aslatex.register(abc.Iterable)
def _(obj: Iterable, **kwargs) -> str:
    obj = list(obj)
    if len(obj) == 0:
        msg = "Need at least one item to render as LaTeX"
        raise ValueError(msg)
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
    latex = Rf"{parent} {to} {child1} {child2}"
    if isinstance(obj.parent, State):
        return latex
    return Rf"\left({latex}\right)"


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
    if isinstance(obj, ThreeBodyDecay):
        return _as_decay_markdown_table(obj.chains)
    item_type = _determine_item_type(obj)
    if item_type in {Particle, State}:
        return _as_resonance_markdown_table(obj)
    if item_type is ThreeBodyDecayChain:
        return _as_decay_markdown_table(obj)
    msg = (
        f"Cannot render a sequence with {item_type.__name__} items as a Markdown table"
    )
    raise NotImplementedError(msg)


def _determine_item_type(obj) -> type:
    if not isinstance(obj, abc.Sequence):
        return type(obj)
    if len(obj) < 1:
        msg = "Need at least one entry to render a table"
        raise ValueError(msg)
    item_type = type(obj[0])
    if not all(isinstance(i, item_type) for i in obj):
        msg = f"Not all items are of type {item_type.__name__}"
        raise ValueError(msg)
    return item_type


def _as_resonance_markdown_table(items: Sequence[Particle | State]) -> str:
    column_names = [
        "name",
        "LaTeX",
        "$J^P$",
        "mass (MeV)",
        "width (MeV)",
    ]
    render_index = any(isinstance(i, State) for i in items)
    if render_index:
        column_names.insert(0, "index")
    src = _create_markdown_table_header(column_names)
    for particle in items:
        row_items = [
            f"`{particle.name}`",
            f"${particle.latex}$",
            Rf"${aslatex(particle, only_jp=True)}$",  # type:ignore[call-arg]
            f"{int(1e3 * particle.mass):,.0f}",
            f"{int(1e3 * particle.width):,.0f}",
        ]
        if render_index and isinstance(particle, State):
            row_items.insert(0, particle.index)
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
        row_items: list = [
            Rf"${chain.resonance.latex} \to {child1} {child2}$",
            Rf"${aslatex(chain.resonance, only_jp=True)}$",  # type:ignore[call-arg]
            f"{int(1e3 * chain.resonance.mass):,.0f}",
            f"{int(1e3 * chain.resonance.width):,.0f}",
        ]
        if chain.outgoing_ls is not None:
            row_items.append(chain.outgoing_ls.L)
        if chain.incoming_ls is not None:
            row_items.append(chain.incoming_ls.L)
        src += _create_markdown_table_row(row_items)
    return src


def _create_markdown_table_header(column_names: list[str]):
    src = _create_markdown_table_row(column_names)
    src += _create_markdown_table_row(["---" for _ in column_names])
    return src


def _create_markdown_table_row(items: Iterable):
    return "| " + " | ".join(f"{i}" for i in items) + " |\n"


@overload
def perform_cached_lambdify(
    expr: sp.Expr,
    backend: str = "jax",
    directory: str | None = None,
) -> PositionalArgumentFunction: ...


@overload
def perform_cached_lambdify(
    expr: sp.Expr,
    parameters: Mapping[sp.Symbol, ParameterValue],
    backend: str = "jax",
    directory: str | None = None,
) -> ParametrizedBackendFunction: ...


def perform_cached_lambdify(  # type:ignore[misc]  # pyright:ignore[reportInconsistentOverload]
    expr: sp.Expr,
    parameters: Mapping[sp.Symbol, ParameterValue] | None = None,
    backend: str = "jax",
    cache_directory: Path | str | None = None,
) -> ParametrizedFunction | Function:
    """Lambdify a SymPy `~sympy.core.expr.Expr` and cache the result to disk.

    The cached result is fetched from disk if the hash of the expression is the same as
    the hash embedded in the filename.

    Args:
        expr: A `sympy.Expr <sympy.core.expr.Expr>` on which to call
            :func:`~tensorwaves.function.sympy.create_function` or
            :func:`~tensorwaves.function.sympy.create_parametrized_function`.
        parameters: Specify this argument in order to create a
            `~tensorwaves.function.ParametrizedBackendFunction` instead of a
            `~tensorwaves.function.PositionalArgumentFunction`.
        backend: The choice of backend for the created numerical function. **WARNING**:
            this function has only been tested for :code:`backend="jax"`!
        directory: The directory in which to cache the result. If `None`, the cache
            directory will be put under the home directory, or to the path specified by
            the environment variable :code:`SYMPY_CACHE_DIR`.

    .. tip:: For a faster cache, set `PYTHONHASHSEED
        <https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED>`_ to a
        fixed value.

    .. seealso:: :func:`ampform.sympy.perform_cached_doit`
    """
    if cache_directory is None:
        system_cache_dir = get_system_cache_directory()
        backend_version = version(backend)
        cache_directory = (
            Path(system_cache_dir) / "ampform_dpd" / f"{backend}-v{backend_version}"
        )
    if not isinstance(cache_directory, Path):
        cache_directory = Path(cache_directory)
    cache_directory.mkdir(exist_ok=True, parents=True)
    if parameters is None:
        hash_obj: Any = expr
    else:
        hash_obj = (
            expr,
            tuple((s, parameters[s]) for s in sorted(parameters, key=str)),
        )
    h = get_readable_hash(hash_obj)
    filename = cache_directory / f"{h}.pkl"
    if filename.exists():
        with open(filename, "rb") as f:
            return pickle.load(f)
    _LOGGER.warning(f"Cached function file {filename} not found, lambdifying...")
    func: ParametrizedFunction | Function
    if parameters is None:
        func = create_function(expr, backend)
    else:
        func = create_parametrized_function(expr, parameters, backend)
    with open(filename, "wb") as f:
        cloudpickle.dump(func, f)
    return func


def simplify_latex_rendering() -> None:
    """Improve LaTeX rendering of an `~sympy.tensor.indexed.Indexed` object."""

    def _print_Indexed_latex(self, printer, *args):  # noqa: N802
        base = printer._print(self.base)
        indices = ", ".join(map(printer._print, self.indices))
        return f"{base}_{{{indices}}}"

    sp.Indexed._latex = _print_Indexed_latex  # type:ignore[attr-defined]
