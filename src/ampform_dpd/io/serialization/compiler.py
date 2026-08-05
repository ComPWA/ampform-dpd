from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from ampform_dpd.io.serialization.kinematics import formulate_kinematic_map

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    import sympy as sp

    from ampform_dpd import AmplitudeModel, DefinedExpression
    from ampform_dpd.io.serialization.workspace import Workspace


@dataclass(frozen=True)
class CompiledWorkspace:
    """Numerical callables compiled from a symbolic serialization workspace."""

    workspace: Workspace
    backend: str
    functions: Mapping[str, Callable[[Mapping[str, Any]], Any]]
    coordinate_maps: Mapping[str, Mapping[str, Callable[[Mapping[str, Any]], Any]]]


def compile_workspace(
    workspace: Workspace,
    *,
    backend: str,
    targets: Iterable[str] | None = None,
) -> CompiledWorkspace:
    """Compile selected validation targets through TensorWaves."""
    _validate_backend(backend)
    selected_targets = _select_targets(workspace, targets)
    compiled = {
        target: _compile_target(workspace, target, backend)
        for target in selected_targets
    }
    distribution_names = {
        target for target in selected_targets if target in workspace.distributions
    }
    coordinate_maps = {
        name: _compile_coordinate_map(workspace, name, backend)
        for name in distribution_names
    }
    return CompiledWorkspace(
        workspace=workspace,
        backend=backend,
        functions=MappingProxyType(compiled),
        coordinate_maps=MappingProxyType(coordinate_maps),
    )


def _validate_backend(backend: str) -> None:
    module_name = {"numpy": "numpy", "jax": "jax"}.get(backend)
    if module_name is None:
        msg = f"Unsupported numerical backend {backend!r}; choose 'numpy' or 'jax'"
        raise ValueError(msg)
    if importlib.util.find_spec(module_name) is None:
        msg = f"Backend {backend!r} requires the optional {module_name!r} package"
        raise ImportError(msg)


def _select_targets(
    workspace: Workspace, targets: Iterable[str] | None
) -> tuple[str, ...]:
    available = set(workspace.functions) | set(workspace.distributions)
    if targets is None:
        checksum_targets = tuple(
            dict.fromkeys(checksum["distribution"] for checksum in workspace.checksums)
        )
        selected = checksum_targets or tuple(sorted(available))
    else:
        selected = tuple(dict.fromkeys(targets))
    missing = sorted(set(selected) - available)
    if missing:
        msg = f"Unknown compilation targets: {', '.join(missing)}"
        raise KeyError(msg)
    return selected


def _compile_target(
    workspace: Workspace, target: str, backend: str
) -> Callable[[Mapping[str, Any]], Any]:
    if target in workspace.functions:
        expression = _prepare_function(workspace.functions[target])
    else:
        expression = _prepare_distribution(workspace.distributions[target])
    return _lambdify(expression, backend)


def _prepare_function(function: DefinedExpression) -> sp.Expr:
    expression = function.expression.xreplace(function.subexpressions)
    return expression.doit().xreplace(function.parameters)


def _prepare_distribution(model: AmplitudeModel) -> sp.Expr:
    from ampform_dpd.io import cached  # ruff: ignore[import-outside-top-level]

    expression = cached.unfold(cast("Any", model))
    expression = cached.xreplace(expression, model.variables)
    expression = cached.xreplace(expression, model.parameter_defaults)
    return cached.doit(expression)


def _compile_coordinate_map(
    workspace: Workspace, distribution: str, backend: str
) -> Mapping[str, Callable[[Mapping[str, Any]], Any]]:
    symbolic_map = formulate_kinematic_map(workspace, distribution)
    masses = workspace.distributions[distribution].masses
    return MappingProxyType({
        str(symbol): _lambdify(expression.doit().xreplace(masses), backend)
        for symbol, expression in symbolic_map.items()
    })


def _lambdify(expression: sp.Expr, backend: str) -> Callable[[Mapping[str, Any]], Any]:
    from ampform_dpd.io.cached import lambdify  # ruff: ignore[import-outside-top-level]

    return cast(
        "Callable[[Mapping[str, Any]], Any]", lambdify(expression, backend=backend)
    )
