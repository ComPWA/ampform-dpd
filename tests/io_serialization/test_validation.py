from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING

import pytest

from ampform_dpd.io.serialization import (
    CompiledWorkspace,
    compile_workspace,
    load_workspace,
    validate_checksums,
)

if TYPE_CHECKING:
    from ampform_dpd.io.serialization.format import ModelDefinition


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_validate_complex_checksum(model_definition: ModelDefinition, backend: str):
    workspace = load_workspace(model_definition)
    checksum = next(
        item for item in workspace.checksums if item["distribution"] == "L1600_BW"
    )
    workspace = replace(workspace, checksums=(checksum,))
    (result,) = validate_checksums(workspace, backend=backend)
    assert result.passed
    assert isinstance(result.reference, complex)
    assert result.difference == pytest.approx(0, abs=1e-14)


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
    )
    (result,) = validate_checksums(compiled)
    assert not result.passed
    assert result.diagnostic is not None
    assert "not compiled" in result.diagnostic


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
