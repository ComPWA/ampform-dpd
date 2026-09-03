from __future__ import annotations

# cspell:ignore errstate
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from ampform_dpd.io.serialization.compiler import CompiledWorkspace, compile_workspace

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from ampform_dpd.io.serialization.workspace import Workspace


@dataclass(frozen=True)
class ChecksumResult:
    """Structured result of evaluating one serialized checksum.

    A failed result has a diagnostic explaining the failure. Its ``value`` is ``None``
    if evaluation did not complete, and its ``reference`` is ``None`` if the serialized
    reference could not be converted to a number.
    """

    target: str
    point_name: str
    point: Mapping[str, float | complex]
    reference: float | complex | None
    value: float | complex | None
    difference: float
    passed: bool
    diagnostic: str | None = None


def validate_checksums(
    workspace_or_compiled: Workspace | CompiledWorkspace,
    *,
    backend: str = "jax",
    atol: float = 1e-10,
    rtol: float = 1e-7,
) -> tuple[ChecksumResult, ...]:
    """Evaluate serialized checksums without aborting on individual failures.

    A checksum passes when ``abs(value - reference) <= atol + rtol * abs(reference)``.
    Compilation, point conversion, evaluation, and non-finite output failures are
    returned as failed records with diagnostics.

    Args:
        workspace_or_compiled: Symbolic workspace to compile on demand, or a
            previously compiled workspace.
        backend: Numerical backend used when compiling a symbolic workspace.
        atol: Absolute tolerance for the checksum comparison.
        rtol: Relative tolerance, scaled by the absolute reference value.

    Returns:
        An immutable result tuple in serialized checksum order.
    """
    if isinstance(workspace_or_compiled, CompiledWorkspace):
        compiled = workspace_or_compiled
        workspace = compiled.workspace
        compilation_diagnostics = {}
    else:
        workspace = workspace_or_compiled
        targets = tuple(
            dict.fromkeys(checksum["distribution"] for checksum in workspace.checksums)
        )
        compiled, compilation_diagnostics = _compile_targets(
            workspace, backend, targets
        )
    points = {
        point["name"]: point["parameters"] for point in workspace.reference_points
    }
    return tuple(
        _validate_checksum(
            checksum,
            points,
            compiled,
            compilation_diagnostics,
            atol=atol,
            rtol=rtol,
        )
        for checksum in workspace.checksums
    )


def _compile_targets(
    workspace: Workspace,
    backend: str,
    targets: tuple[str, ...],
) -> tuple[CompiledWorkspace, dict[str, str]]:
    functions = {}
    coordinate_maps = {}
    coordinates = {}
    diagnostics = {}
    for target in targets:
        try:
            compiled_target = compile_workspace(
                workspace, backend=backend, targets=[target]
            )
        except Exception as exception:  # ruff: ignore[blind-except]
            diagnostics[target] = f"{type(exception).__name__}: {exception}"
            continue
        functions.update(compiled_target.functions)
        coordinate_maps.update(compiled_target.coordinate_maps)
        coordinates.update(compiled_target.coordinates)
    return (
        CompiledWorkspace(
            workspace=workspace,
            backend=backend,
            functions=MappingProxyType(functions),
            coordinate_maps=MappingProxyType(coordinate_maps),
            coordinates=MappingProxyType(coordinates),
        ),
        diagnostics,
    )


def _validate_checksum(
    checksum: Mapping[str, Any],
    points: Mapping[str, Iterable[Mapping[str, Any]]],
    compiled: CompiledWorkspace,
    compilation_diagnostics: Mapping[str, str],
    *,
    atol: float,
    rtol: float,
) -> ChecksumResult:
    target = checksum["distribution"]
    point_name = checksum["point"]
    parameters = points.get(point_name)
    if parameters is None:
        point = {}
        point_diagnostic = f"Reference point {point_name!r} was not found"
    else:
        try:
            point = {
                parameter["name"]: _to_number(parameter["value"])
                for parameter in parameters
            }
        except Exception as exception:  # ruff: ignore[blind-except]
            point = {}
            point_diagnostic = (
                f"Invalid reference point {point_name!r}: "
                f"{type(exception).__name__}: {exception}"
            )
        else:
            point_diagnostic = None
    immutable_point = MappingProxyType(dict(point))
    try:
        reference = _to_number(checksum["value"])
    except Exception as exception:  # ruff: ignore[blind-except]
        return _failed_result(
            target,
            point_name=point_name,
            point=immutable_point,
            reference=None,
            diagnostic=(
                f"Invalid checksum reference: {type(exception).__name__}: {exception}"
            ),
        )
    failure_diagnostic = point_diagnostic
    compilation_diagnostic = compilation_diagnostics.get(target)
    if failure_diagnostic is None and compilation_diagnostic is not None:
        failure_diagnostic = f"Compilation failed: {compilation_diagnostic}"
    if failure_diagnostic is not None:
        return _failed_result(
            target,
            point_name=point_name,
            point=immutable_point,
            reference=reference,
            diagnostic=failure_diagnostic,
        )
    function = compiled.functions.get(target)
    if function is None:
        return _failed_result(
            target,
            point_name=point_name,
            point=immutable_point,
            reference=reference,
            diagnostic=f"Target {target!r} was not compiled",
        )
    try:
        with np.errstate(invalid="ignore"):
            inputs = _transform_point(target, point, compiled)
            value = _to_number(function(inputs))
    except Exception as exception:  # ruff: ignore[blind-except]
        diagnostic = f"{type(exception).__name__}: {exception}"
        return _failed_result(
            target,
            point_name=point_name,
            point=immutable_point,
            reference=reference,
            diagnostic=diagnostic,
        )
    if not _is_finite(value):
        return _failed_result(
            target,
            point_name=point_name,
            point=immutable_point,
            reference=reference,
            diagnostic="Evaluation produced a non-finite value",
            value=value,
        )
    difference = abs(value - reference)
    passed = difference <= atol + rtol * abs(reference)
    diagnostic = None if passed else "Value differs from the reference beyond tolerance"
    return ChecksumResult(
        target=target,
        point_name=point_name,
        point=immutable_point,
        reference=reference,
        value=value,
        difference=difference,
        passed=passed,
        diagnostic=diagnostic,
    )


def _transform_point(
    target: str,
    point: Mapping[str, float | complex],
    compiled: CompiledWorkspace,
) -> dict[str, Any]:
    coordinate_map = compiled.coordinate_maps.get(target)
    if coordinate_map is not None:
        return {
            name: function(dict(point)) for name, function in coordinate_map.items()
        }
    symbolic_function = compiled.workspace.functions.get(target)
    if symbolic_function is not None and len(point) == 1:
        variables = (
            symbolic_function.expression.free_symbols
            - symbolic_function.parameters.keys()
        )
        if len(variables) == 1:
            variable = next(iter(variables))
            return {str(variable): next(iter(point.values()))}
    return {_to_invariant_name(name): value for name, value in point.items()}


def _to_invariant_name(name: str) -> str:
    match = re.fullmatch(r"m_([123])([123])_sq", name)
    if match is None:
        return name
    i, j = map(int, match.groups())
    if i == j:
        return name
    k, *_ = {1, 2, 3} - {i, j}
    return f"sigma{k}"


def _to_number(value: Any) -> float | complex:
    if isinstance(value, str):
        value = complex(value.replace(" ", "").replace("i", "j"))
    number = complex(value)
    if number.imag == 0:
        return number.real
    return number


def _is_finite(value: complex) -> bool:
    number = complex(value)
    return math.isfinite(number.real) and math.isfinite(number.imag)


def _failed_result(
    target: str,
    *,
    point_name: str,
    point: Mapping[str, float | complex],
    reference: complex | None,
    diagnostic: str,
    value: complex | None = None,
) -> ChecksumResult:
    difference = (
        math.inf if value is None or reference is None else abs(value - reference)
    )
    return ChecksumResult(
        target=target,
        point_name=point_name,
        point=point,
        reference=reference,
        value=value,
        difference=difference,
        passed=False,
        diagnostic=diagnostic,
    )
