from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING

import pytest

import ampform_dpd.io.serialization.validation as validation_module
from ampform_dpd.io.serialization import (
    CompiledWorkspace,
    compile_workspace,
    load_workspace,
    validate_checksums,
)

if TYPE_CHECKING:
    from ampform_dpd.io.serialization.format import ModelDefinition


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_validate_real_and_complex_checksums(
    model_definition: ModelDefinition, backend: str
):
    workspace = load_workspace(model_definition)
    checksums = tuple(
        item
        for item in workspace.checksums
        if item["distribution"] in {"default_model", "L1600_BW"}
    )
    workspace = replace(workspace, checksums=checksums)
    distribution_result, function_result = validate_checksums(
        workspace, backend=backend
    )
    assert distribution_result.passed
    assert isinstance(distribution_result.reference, float)
    assert distribution_result.difference == pytest.approx(0, abs=1e-10)
    assert function_result.passed
    assert isinstance(function_result.reference, complex)
    assert function_result.difference == pytest.approx(0, abs=1e-14)


def test_tolerance_boundary(model_definition: ModelDefinition):
    workspace = load_workspace(model_definition)
    checksum = next(
        item for item in workspace.checksums if item["distribution"] == "L1600_BW"
    )
    shifted = dict(checksum)
    shifted["value"] = (
        complex(str(checksum["value"]).replace(" ", "").replace("i", "j")) + 1e-5
    )
    workspace = replace(workspace, checksums=(MappingProxyType(shifted),))
    (failed,) = validate_checksums(workspace, backend="numpy", atol=1e-6, rtol=0)
    (passed,) = validate_checksums(workspace, backend="numpy", atol=1.01e-5, rtol=0)
    assert not failed.passed
    assert passed.passed


def test_named_function_uses_its_symbolic_variable(
    model_definition: ModelDefinition,
):
    workspace = load_workspace(model_definition)
    checksum = next(
        item for item in workspace.checksums if item["distribution"] == "L1600_BW"
    )
    point_name = checksum["point"]
    point = next(
        item for item in workspace.reference_points if item["name"] == point_name
    )
    renamed_point = {
        **point,
        "parameters": ({**point["parameters"][0], "name": "m_12_sq"},),
    }
    workspace = replace(
        workspace,
        checksums=(checksum,),
        reference_points=(MappingProxyType(renamed_point),),
    )
    (result,) = validate_checksums(workspace, backend="jax")
    assert result.passed


def test_missing_target_returns_diagnostic(model_definition: ModelDefinition):
    workspace = load_workspace(model_definition)
    checksum = next(
        item for item in workspace.checksums if item["distribution"] == "L1600_BW"
    )
    workspace = replace(workspace, checksums=(checksum,))
    compiled = CompiledWorkspace(
        workspace=workspace,
        backend="numpy",
        functions=MappingProxyType({}),
        coordinate_maps=MappingProxyType({}),
        coordinates=MappingProxyType({}),
    )
    (result,) = validate_checksums(compiled)
    assert not result.passed
    assert result.diagnostic is not None
    assert "not compiled" in result.diagnostic


def test_invalid_point_does_not_abort_other_checksums(
    model_definition: ModelDefinition,
):
    workspace = load_workspace(model_definition)
    failed_checksum = next(
        item for item in workspace.checksums if item["distribution"] == "L1600_BW"
    )
    passing_checksum = next(
        item for item in workspace.checksums if item["distribution"] == "D1232_BW"
    )
    invalid_point = next(
        item
        for item in workspace.reference_points
        if item["name"] == failed_checksum["point"]
    )
    invalid_point = MappingProxyType({
        **invalid_point,
        "parameters": (
            MappingProxyType({
                **invalid_point["parameters"][0],
                "value": "invalid",
            }),
        ),
    })
    reference_points = tuple(
        invalid_point if point["name"] == invalid_point["name"] else point
        for point in workspace.reference_points
    )
    workspace = replace(
        workspace,
        checksums=(failed_checksum, passing_checksum),
        reference_points=reference_points,
    )
    failed, passed = validate_checksums(workspace, backend="numpy")
    assert not failed.passed
    assert failed.diagnostic is not None
    assert "Invalid reference point" in failed.diagnostic
    assert passed.passed


def test_invalid_reference_returns_diagnostic(model_definition: ModelDefinition):
    workspace = load_workspace(model_definition)
    checksum = next(
        item for item in workspace.checksums if item["distribution"] == "L1600_BW"
    )
    checksum = MappingProxyType({**checksum, "value": "invalid"})
    workspace = replace(workspace, checksums=(checksum,))
    (result,) = validate_checksums(workspace, backend="numpy")
    assert not result.passed
    assert result.reference is None
    assert result.diagnostic is not None
    assert "Invalid checksum reference" in result.diagnostic


def test_non_finite_output_returns_diagnostic(model_definition: ModelDefinition):
    workspace = load_workspace(model_definition)
    checksum = next(
        item for item in workspace.checksums if item["distribution"] == "L1600_BW"
    )
    workspace = replace(workspace, checksums=(checksum,))
    compiled = compile_workspace(workspace, backend="numpy")
    compiled = replace(
        compiled,
        functions=MappingProxyType({"L1600_BW": lambda _: complex("nan")}),
    )
    (result,) = validate_checksums(compiled)
    assert not result.passed
    assert result.diagnostic is not None
    assert "non-finite" in result.diagnostic


def test_invalid_evaluation_returns_diagnostic(model_definition: ModelDefinition):
    workspace = load_workspace(model_definition)
    checksum = next(
        item for item in workspace.checksums if item["distribution"] == "L1600_BW"
    )
    workspace = replace(workspace, checksums=(checksum,))
    compiled = compile_workspace(workspace, backend="numpy")

    def raise_outside_domain(_):
        msg = "point lies outside the function domain"
        raise ValueError(msg)

    compiled = replace(
        compiled,
        functions=MappingProxyType({"L1600_BW": raise_outside_domain}),
    )
    (result,) = validate_checksums(compiled)
    assert not result.passed
    assert result.diagnostic is not None
    assert "outside the function domain" in result.diagnostic


def test_compilation_failure_does_not_abort_other_targets(
    model_definition: ModelDefinition,
    monkeypatch: pytest.MonkeyPatch,
):
    workspace = load_workspace(model_definition)
    failed_checksum = next(
        item for item in workspace.checksums if item["distribution"] == "L1600_BW"
    )
    passing_checksum = next(
        item
        for item in workspace.checksums
        if item["distribution"] in workspace.functions
        and item["distribution"] != "L1600_BW"
    )
    workspace = replace(
        workspace,
        checksums=(failed_checksum, passing_checksum),
    )
    original_compile_workspace = compile_workspace

    def compile_with_failure(workspace, *, backend, targets):
        if tuple(targets) == ("L1600_BW",):
            message = "expression is too deeply nested"
            raise RecursionError(message)
        return original_compile_workspace(
            workspace,
            backend=backend,
            targets=targets,
        )

    monkeypatch.setattr(
        validation_module,
        "compile_workspace",
        compile_with_failure,
    )
    failed, passed = validate_checksums(workspace, backend="jax")
    assert not failed.passed
    assert failed.diagnostic is not None
    assert "Compilation failed: RecursionError" in failed.diagnostic
    assert passed.passed
