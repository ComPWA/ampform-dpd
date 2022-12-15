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

import hashlib
import logging
import os
import pickle
from collections import abc
from functools import lru_cache
from os.path import abspath, dirname, expanduser
from textwrap import dedent
from typing import Iterable, Mapping, Sequence, overload
from warnings import warn

import cloudpickle
import sympy as sp
from ampform.io import aslatex
from tensorwaves.function import ParametrizedBackendFunction, PositionalArgumentFunction
from tensorwaves.function.sympy import create_function, create_parametrized_function
from tensorwaves.interface import Function, ParameterValue, ParametrizedFunction

from ampform_dpd.decay import IsobarNode, Particle, ThreeBodyDecay, ThreeBodyDecayChain

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


def perform_cached_doit(
    unevaluated_expr: sp.Expr, directory: str | None = None
) -> sp.Expr:
    """Perform :code:`doit()` on an `~sympy.core.expr.Expr` and cache the result to disk.

    The cached result is fetched from disk if the hash of the original expression is the
    same as the hash embedded in the filename.

    Args:
        unevaluated_expr: A `sympy.Expr <sympy.core.expr.Expr>` on which to call
            :code:`doit()`.
        directory: The directory in which to cache the result. If `None`, the cache
            directory will be put under the home directory, or to the path specified by
            the environment variable :code:`SYMPY_CACHE_DIR`.

    .. tip:: For a faster cache, set `PYTHONHASHSEED
        <https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED>`_ to a
        fixed value.

    .. seealso:: :func:`perform_cached_lambdify`

    .. deprecated:: 0.2
        Use :func:`ampform.sympy.perform_cached_doit` instead. See `ComPWA/ampform#24
        <https://github.com/ComPWA/ampform-dpd/issues/24>`_
    """
    warn(
        "Use ampform.sympy.perform_cached_doit from AmpForm-DPD v0.2 onwards. "
        "See https://github.com/ComPWA/ampform-dpd/issues/24",
        category=PendingDeprecationWarning,
    )
    if directory is None:
        main_cache_dir = _get_main_cache_dir()
        directory = abspath(f"{main_cache_dir}/.sympy-cache")
    h = get_readable_hash(unevaluated_expr)
    filename = f"{directory}/{h}.pkl"
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    _LOGGER.warning(
        f"Cached expression file {filename} not found, performing doit()..."
    )
    unfolded_expr = unevaluated_expr.doit()
    os.makedirs(dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(unfolded_expr, f)
    return unfolded_expr


@overload
def perform_cached_lambdify(
    expr: sp.Expr,
    backend: str = "jax",
    directory: str | None = None,
) -> PositionalArgumentFunction:
    ...


@overload
def perform_cached_lambdify(
    expr: sp.Expr,
    parameters: Mapping[sp.Symbol, ParameterValue],
    backend: str = "jax",
    directory: str | None = None,
) -> ParametrizedBackendFunction:
    ...


def perform_cached_lambdify(
    expr: sp.Expr,
    parameters: Mapping[sp.Symbol, ParameterValue] | None = None,
    backend: str = "jax",
    directory: str | None = None,
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
    if directory is None:
        main_cache_dir = _get_main_cache_dir()
        directory = abspath(f"{main_cache_dir}/.sympy-cache-{backend}")
    if parameters is None:
        hash_obj = expr
    else:
        hash_obj = (
            expr,
            tuple((s, parameters[s]) for s in sorted(parameters, key=str)),
        )
    h = get_readable_hash(hash_obj)
    filename = f"{directory}/{h}.pkl"
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    _LOGGER.warning(f"Cached function file {filename} not found, lambdifying...")
    if parameters is None:
        func = create_function(expr, backend)
    else:
        func = create_parametrized_function(expr, parameters, backend)
    os.makedirs(dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        cloudpickle.dump(func, f)
    return func


def _get_main_cache_dir() -> str:
    cache_dir = os.environ.get("SYMPY_CACHE_DIR")
    if cache_dir is None:
        cache_dir = expanduser("~")  # home directory
    return cache_dir


def get_readable_hash(obj) -> str:
    python_hash_seed = _get_python_hash_seed()
    if python_hash_seed is not None:
        return f"pythonhashseed-{python_hash_seed}{hash(obj):+}"
    b = _to_bytes(obj)
    return hashlib.sha256(b).hexdigest()


def _to_bytes(obj) -> bytes:
    if isinstance(obj, sp.Expr):
        # Using the str printer is slower and not necessarily unique,
        # but pickle.dumps() does not always result in the same bytes stream.
        _warn_about_unsafe_hash()
        return str(obj).encode()
    return pickle.dumps(obj)


def _get_python_hash_seed() -> int | None:
    python_hash_seed = os.environ.get("PYTHONHASHSEED", "")
    if python_hash_seed is not None and python_hash_seed.isdigit():
        return int(python_hash_seed)
    return None


@lru_cache(maxsize=None)  # warn once
def _warn_about_unsafe_hash():
    message = """
    PYTHONHASHSEED has not been set. For faster and safer hashing of SymPy expressions,
    set the PYTHONHASHSEED environment variable to a fixed value and rerun the program.
    See https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED
    """
    message = dedent(message).replace("\n", " ").strip()
    _LOGGER.warning(message)
