"""Handy aliases for working with cached SymPy expressions and lambdification."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, overload

import cloudpickle
from ampform.sympy._cache import cache_to_disk  # noqa: PLC2701
from ampform.sympy.cached import doit, simplify, subs, trigsimp, unfold, xreplace
from frozendict import frozendict
from tensorwaves.function.sympy import create_function, create_parametrized_function

if TYPE_CHECKING:
    from collections.abc import Mapping

    import sympy as sp
    from tensorwaves.function import (
        ParametrizedBackendFunction,
        PositionalArgumentFunction,
    )
    from tensorwaves.interface import Function, ParameterValue, ParametrizedFunction

__all__ = [
    "doit",
    "lambdify",
    "simplify",
    "subs",
    "trigsimp",
    "unfold",
    "xreplace",
]


@overload
def lambdify(expr: sp.Expr, *, backend: str = "jax") -> PositionalArgumentFunction: ...
@overload
def lambdify(
    expr: sp.Expr,
    parameters: Mapping[sp.Symbol, ParameterValue],
    *,
    backend: str = "jax",
) -> ParametrizedBackendFunction: ...
def lambdify(
    expr: sp.Expr,
    parameters: Mapping[sp.Symbol, ParameterValue] | None = None,
    *,
    backend: str = "jax",
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
        backend: The choice of backend for the created numerical function.
            **WARNING**: this function has only been tested for :code:`backend="jax"`!

    .. seealso:: :mod:`ampform.sympy.cached`
    """
    if isinstance(parameters, str):
        backend = parameters
        parameters = None
    if parameters is None:
        return _lambdify_impl(expr, backend=backend)
    return _lambdify_impl(expr, frozendict(parameters), backend=backend)


@cache
@cache_to_disk(
    dump_function=cloudpickle.dump,
    function_name="lambdify",
    dependencies=["cloudpickle", "ampform", "jax", "sympy"],
)
def _lambdify_impl(
    expr: sp.Expr,
    parameters: frozendict[sp.Symbol, ParameterValue] | None = None,
    *,
    backend: str = "jax",
):
    if parameters is None:
        return create_function(expr, backend)
    return create_parametrized_function(expr, parameters, backend)
