from __future__ import annotations

# cspell:ignore errstate
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ampform_dpd.io.serialization import compile_workspace, load_workspace

if TYPE_CHECKING:
    from ampform_dpd.io.serialization.format import ModelDefinition


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_compile_named_function(model_definition: ModelDefinition, backend: str):
    workspace = load_workspace(model_definition)
    compiled = compile_workspace(workspace, backend=backend, targets=["L1600_BW"])
    value = compiled.functions["L1600_BW"]({"sigma2": 3.2})
    assert complex(value).real == pytest.approx(-0.5711605382444855)
    assert complex(value).imag == pytest.approx(+0.8517294114907099)
    assert compiled.coordinate_maps == {}


def test_compile_distribution_coordinates(model_definition: ModelDefinition):
    workspace = load_workspace(model_definition)
    compiled = compile_workspace(workspace, backend="numpy", targets=["default_model"])
    coordinates = compiled.coordinate_maps["default_model"]
    point = {"m_31": 1.9101377207489973, "cos_theta_31": -0.2309352648098208}
    assert set(coordinates) == {"sigma1", "sigma2", "sigma3"}
    assert all(float(function(point)) > 0 for function in coordinates.values())
    invariants = {name: function(point) for name, function in coordinates.items()}
    independent = compiled.coordinates["default_model"]
    with np.errstate(invalid="ignore"):
        value = compiled.functions["default_model"]({
            name: invariants[name] for name in independent
        })
    assert float(value) > 0


def test_select_distribution_coordinates(model_definition: ModelDefinition):
    workspace = load_workspace(model_definition)
    compiled = compile_workspace(
        workspace,
        backend="numpy",
        targets=["default_model"],
        coordinates=["sigma2", "sigma3"],
    )
    assert compiled.coordinates["default_model"] == ("sigma2", "sigma3")


def test_override_distribution_parameters(model_definition: ModelDefinition):
    workspace = load_workspace(model_definition)
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


def test_rejects_unknown_backend_and_target(model_definition: ModelDefinition):
    workspace = load_workspace(model_definition)
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
