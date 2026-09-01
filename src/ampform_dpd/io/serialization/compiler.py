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
    coordinates: Mapping[str, tuple[str, ...]]


def compile_workspace(
    workspace: Workspace,
    *,
    backend: str,
    targets: Iterable[str] | None = None,
    coordinates: Iterable[str] | None = None,
    parameter_overrides: Mapping[sp.Basic, Any] | None = None,
) -> CompiledWorkspace:
    """Compile selected targets through TensorWaves.

    The ``coordinates`` option selects two independent Mandelstam invariants for
    distributions. By default, they are inferred from the serialized coordinate
    metadata. ``parameter_overrides`` are applied without mutating the symbolic
    workspace.
    """
    _validate_backend(backend)
    selected_targets = _select_targets(workspace, targets)
    distribution_names = {
        target for target in selected_targets if target in workspace.distributions
    }
    selected_coordinates = {
        name: _select_coordinates(workspace, name, coordinates)
        for name in distribution_names
    }
    compiled = {
        target: _compile_target(
            workspace,
            target,
            backend,
            coordinates=selected_coordinates.get(target),
            parameter_overrides=parameter_overrides,
        )
        for target in selected_targets
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
        coordinates=MappingProxyType(selected_coordinates),
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
    workspace: Workspace,
    target: str,
    backend: str,
    coordinates: tuple[str, ...] | None,
    parameter_overrides: Mapping[sp.Basic, Any] | None,
) -> Callable[[Mapping[str, Any]], Any]:
    if target in workspace.functions:
        expression = _prepare_function(workspace.functions[target], parameter_overrides)
    else:
        if coordinates is None:
            msg = f"Missing invariant coordinates for distribution {target!r}"
            raise ValueError(msg)
        expression = _prepare_distribution(
            workspace.distributions[target], coordinates, parameter_overrides
        )
    return _lambdify(expression, backend)


def _prepare_function(
    function: DefinedExpression,
    parameter_overrides: Mapping[sp.Basic, Any] | None,
) -> sp.Expr:
    expression = function.expression.xreplace(function.subexpressions)
    parameters = _apply_parameter_overrides(function.parameters, parameter_overrides)
    return expression.doit().xreplace(parameters)


def _prepare_distribution(
    model: AmplitudeModel,
    coordinates: tuple[str, ...],
    parameter_overrides: Mapping[sp.Basic, Any] | None,
) -> sp.Expr:
    from ampform_dpd.io import cached  # ruff: ignore[import-outside-top-level]

    expression = cached.unfold(cast("Any", model))
    expression = cached.xreplace(expression, model.variables)
    invariants = {str(symbol): symbol for symbol in model.invariants}
    dependent_name, *_ = set(invariants) - set(coordinates)
    dependent = invariants[dependent_name]
    dependent_expression = model.invariants[dependent].xreplace(model.masses).doit()
    expression = cached.xreplace(expression, {dependent: dependent_expression})
    parameters = _apply_parameter_overrides(
        model.parameter_defaults, parameter_overrides
    )
    expression = cached.xreplace(expression, parameters)
    return cached.doit(expression)


def _select_coordinates(
    workspace: Workspace,
    distribution: str,
    coordinates: Iterable[str] | None,
) -> tuple[str, ...]:
    model = workspace.distributions[distribution]
    invariant_names = {str(symbol) for symbol in model.invariants}
    if coordinates is None:
        symbolic_map = formulate_kinematic_map(workspace, distribution)
        dependent = str(next(iter(symbolic_map)))
        return tuple(sorted(invariant_names - {dependent}))
    selected = tuple(dict.fromkeys(map(str, coordinates)))
    n_coordinates = 2
    if len(selected) != n_coordinates or not set(selected) < invariant_names:
        expected = ", ".join(sorted(invariant_names))
        msg = (
            "Distribution coordinates require two distinct invariants from "
            f"{expected}; received {selected}"
        )
        raise ValueError(msg)
    return selected


def _apply_parameter_overrides(
    defaults: Mapping[sp.Basic, Any],
    overrides: Mapping[sp.Basic, Any] | None,
) -> dict[sp.Basic, Any]:
    parameters = dict(defaults)
    if overrides is not None:
        parameters.update({
            symbol: value for symbol, value in overrides.items() if symbol in defaults
        })
    return parameters


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

    return lambdify(expression, backend=backend)  # ty: ignore[invalid-return-type]
