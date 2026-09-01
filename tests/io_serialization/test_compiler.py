from __future__ import annotations

# cspell:ignore errstate
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ampform_dpd.io.serialization import compile_workspace, load_workspace

if TYPE_CHECKING:
    from ampform_dpd.io.serialization.compiler import CompiledWorkspace
    from ampform_dpd.io.serialization.format import ModelDefinition
    from ampform_dpd.io.serialization.workspace import Workspace


def _get_checksums() -> list[dict[str, Any]]:
    doc_directory = Path(__file__).parents[2] / "docs"
    with open(doc_directory / "Lc2ppiK.json") as stream:
        definition = json.load(stream)
    return definition["misc"]["amplitude_model_checksums"]


BACKENDS = ("numpy", "jax")
CHECKSUMS = _get_checksums()
CHECKSUM_IDS = [checksum["distribution"] for checksum in CHECKSUMS]


@pytest.fixture(scope="session")
def workspace(model_definition: ModelDefinition) -> Workspace:
    return load_workspace(model_definition)


@pytest.fixture(scope="session")
def compiled_workspaces(workspace: Workspace) -> dict[str, CompiledWorkspace]:
    return {
        backend: compile_workspace(workspace, backend=backend) for backend in BACKENDS
    }


def test_default_targets_are_checksum_targets(
    compiled_workspaces: dict[str, CompiledWorkspace],
):
    for compiled in compiled_workspaces.values():
        assert set(compiled.functions) == set(CHECKSUM_IDS)
        assert set(compiled.coordinate_maps) == {"default_model"}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("checksum", CHECKSUMS, ids=CHECKSUM_IDS)
def test_reproduce_checksums(
    compiled_workspaces: dict[str, CompiledWorkspace],
    model_definition: ModelDefinition,
    checksum: dict[str, Any],
    backend: str,
):
    compiled = compiled_workspaces[backend]
    value = _evaluate(compiled, model_definition, checksum)
    assert value == pytest.approx(_parse_checksum(checksum["value"]), rel=1e-9)


@pytest.mark.parametrize("checksum", CHECKSUMS, ids=CHECKSUM_IDS)
def test_backends_agree(
    compiled_workspaces: dict[str, CompiledWorkspace],
    model_definition: ModelDefinition,
    checksum: dict[str, Any],
):
    values = {
        backend: _evaluate(compiled, model_definition, checksum)
        for backend, compiled in compiled_workspaces.items()
    }
    assert values["jax"] == pytest.approx(values["numpy"], rel=1e-12)


def _evaluate(
    compiled: CompiledWorkspace,
    model_definition: ModelDefinition,
    checksum: dict[str, Any],
) -> complex:
    target = checksum["distribution"]
    point = _get_parameter_point(model_definition, checksum["point"])
    with np.errstate(invalid="ignore"):  # sub-threshold square roots
        if target not in compiled.coordinate_maps:
            substitutions = {
                _to_invariant_name(name): value for name, value in point.items()
            }
            return complex(compiled.functions[target](substitutions))
        invariants = {
            name: function(point)
            for name, function in compiled.coordinate_maps[target].items()
        }
        independent = compiled.coordinates[target]
        value = compiled.functions[target]({
            name: invariants[name] for name in independent
        })
    return complex(value)


def _get_parameter_point(
    model_definition: ModelDefinition, name: str
) -> dict[str, float]:
    point = next(p for p in model_definition["parameter_points"] if p["name"] == name)
    return {parameter["name"]: parameter["value"] for parameter in point["parameters"]}


def _to_invariant_name(variable_name: str, /) -> str:
    """Convert a serialized variable name to a Mandelstam variable name.

    >>> _to_invariant_name("m_31_sq")
    'sigma2'
    """
    _, subsystem, _ = variable_name.split("_")
    spectator, *_ = {1, 2, 3} - {int(state_id) for state_id in subsystem}
    return f"sigma{spectator}"


def _parse_checksum(value: complex | str, /) -> complex:
    """Convert a serialized checksum value to a complex number.

    >>> _parse_checksum("-0.75 + 0.22i")
    (-0.75+0.22j)
    """
    if isinstance(value, str):
        return complex(value.replace(" ", "").replace("i", "j"))
    return complex(value)


def test_compile_distribution_coordinates(model_definition: ModelDefinition):
    workspace = load_workspace(model_definition)
    compiled = compile_workspace(workspace, backend="numpy", targets=["default_model"])
    coordinates = compiled.coordinate_maps["default_model"]
    point = {"m_31": 1.9101377207489973, "cos_theta_31": -0.2309352648098208}
    assert set(coordinates) == {"sigma1", "sigma2", "sigma3"}
    assert all(float(function(point)) > 0 for function in coordinates.values())


def test_compile_named_function(workspace: Workspace):
    compiled = compile_workspace(workspace, backend="numpy", targets=["L1600_BW"])
    assert compiled.coordinate_maps == {}
    assert compiled.coordinates == {}


def test_select_distribution_coordinates(workspace: Workspace):
    compiled = compile_workspace(
        workspace,
        backend="numpy",
        targets=["default_model"],
        coordinates=["sigma2", "sigma3"],
    )
    assert compiled.coordinates["default_model"] == ("sigma2", "sigma3")


def test_override_distribution_parameters(workspace: Workspace):
    model = workspace.distributions["default_model"]
    coupling_overrides = {
        symbol: 0 for symbol in model.parameter_defaults if str(symbol).startswith("c^")
    }
    compiled = compile_workspace(
        workspace,
        backend="numpy",
        targets=["default_model"],
        parameter_overrides=coupling_overrides,
    )
    point = {"m_31": 1.9101377207489973, "cos_theta_31": -0.2309352648098208}
    invariants = {
        name: function(point)
        for name, function in compiled.coordinate_maps["default_model"].items()
    }
    value = compiled.functions["default_model"](invariants)
    assert float(value) == pytest.approx(0)


def test_rejects_unknown_backend_and_target(workspace: Workspace):
    with pytest.raises(ValueError, match="Unsupported numerical backend"):
        compile_workspace(workspace, backend="unknown")
    with pytest.raises(KeyError, match="missing"):
        compile_workspace(workspace, backend="numpy", targets=["missing"])
    with pytest.raises(ValueError, match="two distinct invariants"):
        compile_workspace(
            workspace,
            backend="numpy",
            targets=["default_model"],
            coordinates=["sigma1"],
        )


def test_reports_missing_backend_package(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("importlib.util.find_spec", lambda _: None)
    with pytest.raises(ImportError, match="requires the optional 'jax' package"):
        compile_workspace(workspace, backend="jax", targets=["L1600_BW"])
