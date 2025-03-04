"""Handy aliases for working with cached SymPy expressions and lambdification."""

from __future__ import annotations

import logging
import pickle
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, overload

import cloudpickle
from ampform.sympy._cache import (
    get_readable_hash,  # noqa: PLC2701
    get_system_cache_directory,  # noqa: PLC2701
)
from ampform.sympy.cached import (
    doit,  # noqa: F401  # pyright: ignore[reportUnusedImport]
    unfold,  # noqa: F401  # pyright: ignore[reportUnusedImport]
    xreplace,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)
from tensorwaves.function.sympy import create_function, create_parametrized_function

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    import sympy as sp
    from tensorwaves.function import (
        ParametrizedBackendFunction,
        PositionalArgumentFunction,
    )
    from tensorwaves.interface import Function, ParameterValue, ParametrizedFunction

_LOGGER = logging.getLogger(__name__)


@overload
def lambdify(
    expr: sp.Expr,
    backend: str = "jax",
    directory: str | None = None,
) -> PositionalArgumentFunction: ...
@overload
def lambdify(
    expr: sp.Expr,
    parameters: Mapping[sp.Symbol, ParameterValue],
    backend: str = "jax",
    directory: str | None = None,
) -> ParametrizedBackendFunction: ...
def lambdify(  # type:ignore[misc]  # pyright:ignore[reportInconsistentOverload]
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
