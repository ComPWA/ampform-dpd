from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import pytest
import sympy as sp

from ampform_dpd.angles import formulate_scattering_angle
from ampform_dpd.io.serialization import formulate_kinematic_map, load_workspace

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ampform.helicity import ParameterValue

    from ampform_dpd.decay import FinalStateID
    from ampform_dpd.io.serialization.format import ModelDefinition


@pytest.mark.parametrize("node", [(1, 2), (2, 3), (3, 1)])
def test_formulate_kinematic_map_round_trip(
    model_definition: ModelDefinition, node: tuple[FinalStateID, FinalStateID]
):
    definition = deepcopy(model_definition)
    i, j = node
    mass_name = f"m_{i}{j}"
    cosine_name = f"cos_theta_{i}{j}"
    definition["distributions"][0]["variables"][0] = {
        "node": list(node),
        "mass_phi_costheta": [mass_name, f"phi_{i}{j}", cosine_name],
    }
    workspace = load_workspace(definition)
    mapping = formulate_kinematic_map(workspace)
    model = workspace.distributions["default_model"]
    phase_space_points = (
        (0.4, 0.23),
        (1e-6, 0.999),
        (1 - 1e-6, -0.999),
    )
    for mass_fraction, expected_cosine in phase_space_points:
        _assert_round_trip(
            mapping,
            model.parameter_defaults,
            node,
            mass_fraction,
            expected_cosine,
        )


def _assert_round_trip(
    mapping: Mapping[sp.Symbol, sp.Expr],
    parameter_defaults: Mapping[sp.Basic, ParameterValue],
    node: tuple[int, int],
    mass_fraction: float,
    expected_cosine: float,
) -> None:
    i, j = node
    expected_mass = _compute_isobar_mass(parameter_defaults, node, mass_fraction)
    substitutions = {
        **parameter_defaults,
        sp.Symbol(f"m_{i}{j}", nonnegative=True): expected_mass,
        sp.Symbol(f"cos_theta_{i}{j}", real=True): expected_cosine,
    }
    invariants: dict[sp.Basic, sp.Expr] = {
        symbol: expression.doit().evalf(subs=substitutions)
        for symbol, expression in mapping.items()
    }
    _, angle = formulate_scattering_angle(i, j)
    cosine = (
        sp.cos(angle).xreplace(invariants).doit().evalf(subs=dict(parameter_defaults))
    )
    assert float(cosine) == pytest.approx(expected_cosine, abs=1e-8)
    spectator_id = next(iter({1, 2, 3} - set(node)))
    assert float(
        sp.sqrt(invariants[sp.Symbol(f"sigma{spectator_id}", nonnegative=True)])
    ) == pytest.approx(expected_mass)
    assert len(invariants) == 3


def _compute_isobar_mass(
    parameter_defaults: Mapping[sp.Basic, ParameterValue],
    node: tuple[int, int],
    mass_fraction: float,
) -> float:
    i, j = node
    k = next(iter({1, 2, 3} - set(node)))
    masses = {
        index: _to_real(parameter_defaults[sp.Symbol(f"m{index}", nonnegative=True)])
        for index in range(4)
    }
    lower_mass = masses[i] + masses[j]
    upper_mass = masses[0] - masses[k]
    return lower_mass + mass_fraction * (upper_mass - lower_mass)


def _to_real(value: ParameterValue) -> float:
    if isinstance(value, complex):
        assert value.imag == 0
        return float(value.real)
    return float(value)


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


def test_rejects_invalid_coordinate_metadata(model_definition: ModelDefinition):
    definition = deepcopy(model_definition)
    distribution: Any = definition["distributions"][0]
    distribution["variables"] = [None]
    workspace = load_workspace(definition)
    with pytest.raises(TypeError, match=r"variables\[0\] must be a mapping"):
        formulate_kinematic_map(workspace)


def test_rejects_invalid_coordinate_names(model_definition: ModelDefinition):
    definition = deepcopy(model_definition)
    definition["distributions"][0]["variables"][0]["mass_phi_costheta"] = [
        "m_31",
        "phi_31",
    ]
    workspace = load_workspace(definition)
    with pytest.raises(ValueError, match=r"variables\[0\].mass_phi_costheta"):
        formulate_kinematic_map(workspace)


def test_rejects_ambiguous_coordinate_metadata(model_definition: ModelDefinition):
    definition = deepcopy(model_definition)
    coordinate = deepcopy(definition["distributions"][0]["variables"][0])
    coordinate["node"] = [1, 2]
    definition["distributions"][0]["variables"].append(coordinate)
    workspace = load_workspace(definition)
    with pytest.raises(ValueError, match="coordinate, found 2"):
        formulate_kinematic_map(workspace)
