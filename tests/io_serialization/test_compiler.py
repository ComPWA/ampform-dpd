from __future__ import annotations

from typing import TYPE_CHECKING

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


def test_rejects_unknown_backend_and_target(model_definition: ModelDefinition):
    workspace = load_workspace(model_definition)
    with pytest.raises(ValueError, match="Unsupported numerical backend"):
        compile_workspace(workspace, backend="unknown")
    with pytest.raises(KeyError, match="missing"):
        compile_workspace(workspace, backend="numpy", targets=["missing"])
