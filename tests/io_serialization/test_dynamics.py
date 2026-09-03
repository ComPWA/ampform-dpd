from __future__ import annotations

import pytest
import sympy as sp
from ampform.dynamics.form_factor import BreakupMomentumSquared, FormFactor
from ampform.dynamics.phasespace import PhaseSpaceFactorComplex

from ampform_dpd.io.serialization.dynamics import (
    formulate_dynamics,
    formulate_form_factor,
    formulate_polynomial,
)
from ampform_dpd.io.serialization.format import (
    DecayChain,
    GenericFunctionDefinition,
    LSVertex,
    ModelDefinition,
    MomentumPowerDefinition,
    PolynomialDefinition,
    Propagator,
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


def test_formulate_generic_function_accepts_published_notation():
    model = ModelDefinition(
        distributions=[],
        functions=[
            GenericFunctionDefinition(
                name="custom",
                type="generic_function",
                expression="sqrt(m_12_sq) / (2^2 - m_12_sq - 1im)",
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
    expression = formulate_dynamics(chain, model).expression
    sigma1 = sp.Symbol("sigma1", nonnegative=True)
    assert expression == sp.sqrt(sigma1) / (4 - sigma1 - sp.I)


def test_formulate_polynomial():
    model = _create_polynomial_model(x="m_23_sq")
    propagator = Propagator(spin="0", node=(2, 3), parametrization="polynomial")

    dynamics = formulate_polynomial(propagator, "R", model)

    c0 = sp.Symbol("c_{R,0}", real=True)
    c1 = sp.Symbol("c_{R,1}", real=True)
    sigma1 = sp.Symbol("sigma1", nonnegative=True)
    assert dynamics.expression == c1 * sigma1 + c0
    assert dynamics.parameters == {c0: 1, c1: 2}


def test_formulate_polynomial_variable_mismatch():
    model = _create_polynomial_model(x="m_12_sq")
    propagator = Propagator(spin="0", node=(2, 3), parametrization="polynomial")

    with pytest.raises(
        ValueError, match=r"is sigma3, but node \(2, 3\) implies sigma1"
    ):
        formulate_polynomial(propagator, "R", model)


def _create_polynomial_model(x: str) -> ModelDefinition:
    return ModelDefinition(
        distributions=[],
        functions=[
            PolynomialDefinition(
                name="polynomial",
                type="Polynomial",
                coefficients=[1, 2],
                x=x,
            )
        ],
        domains=[],
        misc={},
        parameter_points=[],
    )


def test_formulate_momentum_power():
    model = ModelDefinition(
        distributions=[],
        functions=[MomentumPowerDefinition(name="momentum", type="MomentumPower", l=3)],
        domains=[],
        misc={},
        parameter_points=[],
    )
    vertex = LSVertex(type="ls", node=(2, 3), formfactor="momentum", l="0", s="0")
    expression = formulate_form_factor(vertex, model).expression
    sigma1 = sp.Symbol("sigma1", nonnegative=True)
    m2, m3 = sp.symbols("m2 m3", nonnegative=True)
    assert expression == BreakupMomentumSquared(sigma1, m2, m3) ** sp.Rational(3, 2)


@pytest.mark.parametrize("chain_id", [18, 19, 24, 25])
def test_formulate_bugg_lineshapes(model_definition: ModelDefinition, chain_id: int):
    chain = get_decay_chains(model_definition)[chain_id]

    dynamics = formulate_dynamics(chain, model_definition)

    assert dynamics.expression.free_symbols == {sp.Symbol("sigma1", nonnegative=True)}


def test_multichannel_uses_complex_phase_space(model_definition: ModelDefinition):
    chain = get_decay_chains(model_definition)[0]
    expression = formulate_dynamics(chain, model_definition).expression
    assert expression.has(PhaseSpaceFactorComplex)


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
