from __future__ import annotations

import pytest
import sympy as sp
from ampform.dynamics.form_factor import FormFactor

from ampform_dpd.io.serialization.dynamics import (
    formulate_dynamics,
    formulate_form_factor,
)
from ampform_dpd.io.serialization.format import (
    DecayChain,
    GenericFunctionDefinition,
    ModelDefinition,
    get_decay_chains,
)


def test_formulate_generic_function():
    model = ModelDefinition(
        distributions=[],
        functions=[
            GenericFunctionDefinition(
                name="custom",
                type="generic_function",
                expression="1 / (2^2 - σ - i * exp(-0.5 * σ))",
            )
        ],
        domains=[],
        misc={},
        parameter_points=[],
    )
    chain = DecayChain(
        name="resonance",
        propagators=[{"spin": "0", "node": (2, 3), "parametrization": "custom"}],
        vertices=[],
        topology=[1, [2, 3]],
        kinematics={
            "initial_state": {"index": 0, "name": "X", "spin": "0", "mass": 1.0},
            "final_state": [],
        },
        weight="1",
    )

    dynamics = formulate_dynamics(chain, model)

    σ1 = sp.Symbol("sigma1", nonnegative=True)
    assert dynamics.expression == 1 / (4 - σ1 - sp.I * sp.exp(-sp.Float(0.5) * σ1))
    assert dynamics.parameters == {}


@pytest.mark.parametrize("chain_id", [18, 19, 24, 25])
def test_formulate_bugg_lineshapes(model_definition: ModelDefinition, chain_id: int):
    chain = get_decay_chains(model_definition)[chain_id]

    dynamics = formulate_dynamics(chain, model_definition)

    assert dynamics.expression.free_symbols == {sp.Symbol("sigma1", nonnegative=True)}


def test_formulate_form_factor_uses_serialized_normalization(
    model_definition: ModelDefinition,
):
    vertex = get_decay_chains(model_definition)[2]["vertices"][0]
    form_factor = formulate_form_factor(vertex, model_definition)
    assert sp.sqrt(2) * form_factor.expression == FormFactor(
        s=sp.Symbol("m0", nonnegative=True) ** 2,  # ty: ignore[unknown-argument]
        m1=sp.sqrt(sp.Symbol("sigma2", nonnegative=True)),  # ty: ignore[unknown-argument]
        m2=sp.Symbol("m2", nonnegative=True),  # ty: ignore[unknown-argument]
        angular_momentum=1,  # ty: ignore[unknown-argument]
        meson_radius=sp.Symbol("R_{Lc}", nonnegative=True),  # ty: ignore[unknown-argument]
    )
