from __future__ import annotations

import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ampform_dpd.io.serialization.compiler import CompiledWorkspace, compile_workspace

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ampform_dpd.io.serialization.workspace import Workspace


@dataclass(frozen=True)
class ChecksumResult:
    """Structured result of evaluating one serialized checksum."""

    target: str
    point: Mapping[str, float | complex]
    reference: float | complex
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
    """Evaluate serialized checksums with absolute and relative tolerances."""
    if isinstance(workspace_or_compiled, CompiledWorkspace):
        compiled = workspace_or_compiled
        workspace = compiled.workspace
    else:
        workspace = workspace_or_compiled
        targets = tuple(
            dict.fromkeys(checksum["distribution"] for checksum in workspace.checksums)
        )
        compiled = compile_workspace(workspace, backend=backend, targets=targets)
    points = {
        point["name"]: {
            parameter["name"]: _to_number(parameter["value"])
            for parameter in point["parameters"]
        }
        for point in workspace.reference_points
    }
    return tuple(
        _validate_checksum(checksum, points, compiled, atol=atol, rtol=rtol)
        for checksum in workspace.checksums
    )


def _validate_checksum(
    checksum: Mapping[str, Any],
    points: Mapping[str, Mapping[str, float | complex]],
    compiled: CompiledWorkspace,
    *,
    atol: float,
    rtol: float,
) -> ChecksumResult:
    target = checksum["distribution"]
    point_name = checksum["point"]
    reference = _to_number(checksum["value"])
    point = points.get(point_name, {})
    immutable_point = MappingProxyType(dict(point))
    function = compiled.functions.get(target)
    if function is None:
        return _failed_result(
            target, immutable_point, reference, f"Target {target!r} was not compiled"
        )
    try:
        inputs = _transform_point(target, point, compiled)
        value = _to_number(function(inputs))
    except Exception as exception:  # ruff: ignore[blind-except]
        diagnostic = f"{type(exception).__name__}: {exception}"
        return _failed_result(target, immutable_point, reference, diagnostic)
    if not _is_finite(value):
        return _failed_result(
            target,
            immutable_point,
            reference,
            "Evaluation produced a non-finite value",
            value,
        )
    difference = abs(value - reference)
    passed = difference <= atol + rtol * abs(reference)
    diagnostic = None if passed else "Value differs from the reference beyond tolerance"
    return ChecksumResult(
        target=target,
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
    point: Mapping[str, float | complex],
    reference: complex,
    diagnostic: str,
    value: complex | None = None,
) -> ChecksumResult:
    difference = math.inf if value is None else abs(value - reference)
    return ChecksumResult(
        target=target,
        point=point,
        reference=reference,
        value=value,
        difference=difference,
        passed=False,
        diagnostic=diagnostic,
    )
