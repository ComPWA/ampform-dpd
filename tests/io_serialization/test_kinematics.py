from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import pytest
import sympy as sp

from ampform_dpd.angles import formulate_scattering_angle
from ampform_dpd.io.serialization import formulate_kinematic_map, load_workspace

if TYPE_CHECKING:
    from ampform_dpd.io.serialization.format import ModelDefinition


def test_formulate_kinematic_map_round_trip(model_definition: ModelDefinition):
    workspace = load_workspace(model_definition)
    mapping = formulate_kinematic_map(workspace)
    model = workspace.distributions["default_model"]
    point = {
        parameter["name"]: parameter["value"]
        for parameter in model_definition["parameter_points"][0]["parameters"]
    }
    substitutions = {
        **model.parameter_defaults,
        sp.Symbol("m_31", nonnegative=True): point["m_31"],
        sp.Symbol("cos_theta_31", real=True): point["cos_theta_31"],
    }
    invariants: dict[sp.Basic, sp.Expr] = {
        symbol: expression.doit().evalf(subs=substitutions)
        for symbol, expression in mapping.items()
    }
    _, angle = formulate_scattering_angle(3, 1)
    cosine = (
        sp
        .cos(angle)
        .xreplace(invariants)
        .doit()
        .evalf(subs=dict(model.parameter_defaults))
    )
    assert float(cosine) == pytest.approx(point["cos_theta_31"])
    assert len(invariants) == 3


def test_requires_distribution_for_multiple_models(model_definition: ModelDefinition):
    definition = deepcopy(model_definition)
    second = deepcopy(definition["distributions"][0])
    second["name"] = "second"
    definition["distributions"].append(second)
    workspace = load_workspace(definition)
    with pytest.raises(ValueError, match="Select a distribution"):
        formulate_kinematic_map(workspace)
    assert formulate_kinematic_map(workspace, "second")


def test_rejects_unsupported_orientation(model_definition: ModelDefinition):
    definition = deepcopy(model_definition)
    definition["distributions"][0]["variables"][0]["node"] = [1, 3]
    workspace = load_workspace(definition)
    with pytest.raises(ValueError, match=r"orientation \(1, 3\)"):
        formulate_kinematic_map(workspace)
