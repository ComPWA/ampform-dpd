# cspell:ignore pksigma
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import qrules
import sympy as sp

from ampform_dpd import AmplitudeModel, DalitzPlotDecompositionBuilder
from ampform_dpd.adapter.qrules import normalize_state_ids, to_three_body_decay
from ampform_dpd.dynamics.builder import formulate_breit_wigner_with_form_factor
from ampform_dpd.polarization import (
    create_final_state_coupling,
    formulate_weak_decay_couplings,
)

if TYPE_CHECKING:
    from qrules.transition import ReactionInfo


class TestDalitzPlotDecompositionBuilder:
    @pytest.mark.parametrize("all_subsystems", [False, True])
    @pytest.mark.parametrize("min_ls", [False, True])
    def test_all_subsystems(
        self, jpsi2pksigma_reaction: ReactionInfo, all_subsystems: bool, min_ls: bool
    ):
        if jpsi2pksigma_reaction.formalism == "helicity" and not min_ls:
            pytest.skip("Helicity formalism with all LS not supported")
        transitions = normalize_state_ids(jpsi2pksigma_reaction.transitions)
        decay = to_three_body_decay(transitions, min_ls=min_ls)
        builder = DalitzPlotDecompositionBuilder(
            decay, min_ls=min_ls, all_subsystems=all_subsystems
        )
        if jpsi2pksigma_reaction.formalism == "canonical-helicity":
            for chain in builder.decay.chains:
                builder.dynamics_choices.register_builder(
                    chain, formulate_breit_wigner_with_form_factor
                )
        if all_subsystems:
            with pytest.warns(
                UserWarning,
                match=r"Decay J/psi\(1S\) → 1: K0, 2: Sigma\+, 3: p~ only has subsystems 2, 3, not 1",
            ):
                model = builder.formulate(reference_subsystem=2)
        else:
            model = builder.formulate(reference_subsystem=2)
        expected_variables = {
            R"\zeta^0_{2(2)}",
            R"\zeta^0_{3(2)}",
            R"\zeta^2_{2(2)}",
            R"\zeta^2_{3(2)}",
            R"\zeta^3_{2(2)}",
            R"\zeta^3_{3(2)}",
            "theta_12",
            "theta_23",
            "theta_31",
        }
        if not all_subsystems:
            expected_variables.remove("theta_23")
        assert {str(s) for s in model.variables} == expected_variables

    @pytest.mark.parametrize("min_ls", [False, True])
    @pytest.mark.parametrize("use_coefficients", [False, True])
    def test_use_coefficients(
        self, jpsi2pksigma_reaction: ReactionInfo, min_ls: bool, use_coefficients: bool
    ):
        if jpsi2pksigma_reaction.formalism == "helicity" and not min_ls:
            pytest.skip("Helicity formalism with all LS not supported")
        transitions = normalize_state_ids(jpsi2pksigma_reaction.transitions)
        decay = to_three_body_decay(transitions, min_ls=min_ls)
        builder = DalitzPlotDecompositionBuilder(decay, min_ls=min_ls)
        model = builder.formulate(
            reference_subsystem=2,
            use_coefficients=use_coefficients,
        )
        amplitudes = _get_physical_amplitudes(model)
        coupling_symbols = _collect_indexed_symbols(amplitudes)

        n_coupling_symbols = len(coupling_symbols)
        coupling_symbols_str = sorted(str(s) for s in coupling_symbols)
        # ----==== COEFFICIENTS ===--- #
        if use_coefficients:
            if min_ls:  # HELICITY BASIS
                assert n_coupling_symbols == 20
                assert coupling_symbols_str == [
                    R"\mathcal{H}^\mathrm{N(1700)^{+}}[-1/2, -1/2, 0, -1/2]",
                    R"\mathcal{H}^\mathrm{N(1700)^{+}}[-1/2, -1/2, 0, 1/2]",
                    R"\mathcal{H}^\mathrm{N(1700)^{+}}[-1/2, 1/2, 0, -1/2]",
                    R"\mathcal{H}^\mathrm{N(1700)^{+}}[-1/2, 1/2, 0, 1/2]",
                    R"\mathcal{H}^\mathrm{N(1700)^{+}}[-3/2, -1/2, 0, -1/2]",
                    R"\mathcal{H}^\mathrm{N(1700)^{+}}[-3/2, -1/2, 0, 1/2]",
                    R"\mathcal{H}^\mathrm{N(1700)^{+}}[1/2, -1/2, 0, -1/2]",
                    R"\mathcal{H}^\mathrm{N(1700)^{+}}[1/2, -1/2, 0, 1/2]",
                    R"\mathcal{H}^\mathrm{N(1700)^{+}}[1/2, 1/2, 0, -1/2]",
                    R"\mathcal{H}^\mathrm{N(1700)^{+}}[1/2, 1/2, 0, 1/2]",
                    R"\mathcal{H}^\mathrm{N(1700)^{+}}[3/2, 1/2, 0, -1/2]",
                    R"\mathcal{H}^\mathrm{N(1700)^{+}}[3/2, 1/2, 0, 1/2]",
                    R"\mathcal{H}^\mathrm{\overline{\Sigma}(1660)^{-}}[-1/2, -1/2, -1/2, 0]",
                    R"\mathcal{H}^\mathrm{\overline{\Sigma}(1660)^{-}}[-1/2, -1/2, 1/2, 0]",
                    R"\mathcal{H}^\mathrm{\overline{\Sigma}(1660)^{-}}[-1/2, 1/2, -1/2, 0]",
                    R"\mathcal{H}^\mathrm{\overline{\Sigma}(1660)^{-}}[-1/2, 1/2, 1/2, 0]",
                    R"\mathcal{H}^\mathrm{\overline{\Sigma}(1660)^{-}}[1/2, -1/2, -1/2, 0]",
                    R"\mathcal{H}^\mathrm{\overline{\Sigma}(1660)^{-}}[1/2, -1/2, 1/2, 0]",
                    R"\mathcal{H}^\mathrm{\overline{\Sigma}(1660)^{-}}[1/2, 1/2, -1/2, 0]",
                    R"\mathcal{H}^\mathrm{\overline{\Sigma}(1660)^{-}}[1/2, 1/2, 1/2, 0]",
                ]
            else:  # CANONICAL BASIS
                assert n_coupling_symbols == 4
                assert coupling_symbols_str == [
                    R"\mathcal{H}^\mathrm{LS,N(1700)^{+}}[1, 1, 2, 1/2]",
                    R"\mathcal{H}^\mathrm{LS,N(1700)^{+}}[1, 2, 2, 1/2]",
                    R"\mathcal{H}^\mathrm{LS,\overline{\Sigma}(1660)^{-}}[0, 1, 1, 1/2]",
                    R"\mathcal{H}^\mathrm{LS,\overline{\Sigma}(1660)^{-}}[2, 1, 1, 1/2]",
                ]
        # ----==== COUPLING ===--- #
        else:
            n_products = len(_collect_products(amplitudes))
            if min_ls:  # HELICITY BASIS
                assert n_coupling_symbols == 14
                assert n_products == 20
                assert coupling_symbols_str == [
                    R"\mathcal{H}^\mathrm{decay}[N(1700)^{+}, 0, -1/2]",
                    R"\mathcal{H}^\mathrm{decay}[N(1700)^{+}, 0, 1/2]",
                    R"\mathcal{H}^\mathrm{decay}[\overline{\Sigma}(1660)^{-}, -1/2, 0]",
                    R"\mathcal{H}^\mathrm{decay}[\overline{\Sigma}(1660)^{-}, 1/2, 0]",
                    R"\mathcal{H}^\mathrm{production}[N(1700)^{+}, -1/2, -1/2]",
                    R"\mathcal{H}^\mathrm{production}[N(1700)^{+}, -1/2, 1/2]",
                    R"\mathcal{H}^\mathrm{production}[N(1700)^{+}, -3/2, -1/2]",
                    R"\mathcal{H}^\mathrm{production}[N(1700)^{+}, 1/2, -1/2]",
                    R"\mathcal{H}^\mathrm{production}[N(1700)^{+}, 1/2, 1/2]",
                    R"\mathcal{H}^\mathrm{production}[N(1700)^{+}, 3/2, 1/2]",
                    R"\mathcal{H}^\mathrm{production}[\overline{\Sigma}(1660)^{-}, -1/2, -1/2]",
                    R"\mathcal{H}^\mathrm{production}[\overline{\Sigma}(1660)^{-}, -1/2, 1/2]",
                    R"\mathcal{H}^\mathrm{production}[\overline{\Sigma}(1660)^{-}, 1/2, -1/2]",
                    R"\mathcal{H}^\mathrm{production}[\overline{\Sigma}(1660)^{-}, 1/2, 1/2]",
                ]
            else:  # CANONICAL BASIS
                assert n_coupling_symbols == 6
                assert n_products == 4
                assert coupling_symbols_str == [
                    R"\mathcal{H}^\mathrm{LS,decay}[N(1700)^{+}, 2, 1/2]",
                    R"\mathcal{H}^\mathrm{LS,decay}[\overline{\Sigma}(1660)^{-}, 1, 1/2]",
                    R"\mathcal{H}^\mathrm{LS,production}[N(1700)^{+}, 1, 1]",
                    R"\mathcal{H}^\mathrm{LS,production}[N(1700)^{+}, 1, 2]",
                    R"\mathcal{H}^\mathrm{LS,production}[\overline{\Sigma}(1660)^{-}, 0, 1]",
                    R"\mathcal{H}^\mathrm{LS,production}[\overline{\Sigma}(1660)^{-}, 2, 1]",
                ]

    @pytest.mark.parametrize("basis", ["canonical", "helicity"])
    @pytest.mark.parametrize("resonance", ["N(1675)", "Sigma(1775)"])
    def test_use_coefficients_combinations(self, basis: str, resonance: str):  # ruff: ignore[too-many-locals]
        reaction = qrules.generate_transitions(
            initial_state=[("J/psi(1S)", [+1])],
            final_state=[("Sigma+", [+0.5]), "K0", ("p~", [+0.5])],
            allowed_interaction_types="strong",
            allowed_intermediate_particles=[resonance],
            formalism="canonical-helicity",
        )
        transitions = normalize_state_ids(reaction.transitions)
        min_ls = basis == "helicity"
        decay = to_three_body_decay(transitions, min_ls)
        builder = DalitzPlotDecompositionBuilder(decay, min_ls)
        # cspell:ignore coeff
        reference_subsystem = 1 if resonance.startswith("Sigma") else 3
        coupling_model = builder.formulate(reference_subsystem)
        coeff_model = builder.formulate(reference_subsystem, use_coefficients=True)
        coupling_amplitudes = _get_physical_amplitudes(coupling_model)
        coeff_amplitudes = _get_physical_amplitudes(coeff_model)

        couplings = _collect_indexed_symbols(coupling_amplitudes)
        coefficients = _collect_indexed_symbols(coeff_amplitudes)
        coupling_products = _collect_products(coupling_amplitudes)

        n_couplings = len(couplings)
        n_decay_couplings = len({s for s in couplings if "decay" in s.name})
        n_production_couplings = len({s for s in couplings if "production" in s.name})
        assert n_couplings == n_decay_couplings + n_production_couplings

        n_coupling_products = len(coupling_products)
        n_coefficients = len(coefficients)
        assert n_coefficients == n_coupling_products
        assert n_coefficients == n_decay_couplings * n_production_couplings


@pytest.fixture(scope="module")
def lc2pkpi_single_resonance_reaction() -> ReactionInfo:
    return qrules.generate_transitions(
        initial_state="Lambda(c)+",
        final_state=["p", "K-", "pi+"],
        allowed_intermediate_particles=["Lambda(1520)"],
        formalism="helicity",
    )


def test_polarized_final_states(  # ruff: ignore[too-many-locals]
    lc2pkpi_single_resonance_reaction: ReactionInfo,
):
    transitions = normalize_state_ids(lc2pkpi_single_resonance_reaction.transitions)
    decay = to_three_body_decay(transitions, min_ls=True)
    builder = DalitzPlotDecompositionBuilder(decay, min_ls=True)
    baseline_model = builder.formulate()
    model = builder.formulate(polarized_final_states=[1])

    proton = decay.final_state[1]
    coupling_symbols = [create_final_state_coupling(proton, h) for h in (-0.5, +0.5)]
    assert all(s in model.parameter_defaults for s in coupling_symbols)

    baseline_intensity = _to_numerical_function(baseline_model)
    intensity = _to_numerical_function(model)
    weak_couplings = {
        symbol: complex(expr)
        for symbol, expr in formulate_weak_decay_couplings(proton, alpha=-0.84).items()
    }
    weak_intensity = _to_numerical_function(model, weak_couplings)

    phase_space_point = _find_phase_space_point(baseline_intensity, baseline_model)
    expected = baseline_intensity(phase_space_point).real
    rng = np.random.default_rng(seed=0)
    for _ in range(3):
        decay_angles = {
            "phi_1": rng.uniform(-np.pi, +np.pi),
            "theta_1": rng.uniform(0, np.pi),
        }
        computed = intensity({**phase_space_point, **decay_angles})
        assert computed.imag == pytest.approx(0)
        assert computed.real == pytest.approx(expected)
        assert weak_intensity({**phase_space_point, **decay_angles}).real != (
            pytest.approx(expected)
        )


@pytest.mark.parametrize(
    ("state_id", "error_message"),
    [
        (2, "spinless"),
        (4, "not one of 1, 2, 3"),
    ],
    ids=["spinless", "invalid-id"],
)
def test_polarized_final_states_invalid(
    lc2pkpi_single_resonance_reaction: ReactionInfo, state_id: int, error_message: str
):
    transitions = normalize_state_ids(lc2pkpi_single_resonance_reaction.transitions)
    decay = to_three_body_decay(transitions, min_ls=True)
    builder = DalitzPlotDecompositionBuilder(decay, min_ls=True)
    with pytest.raises(ValueError, match=error_message):
        builder.formulate(polarized_final_states=[state_id])  # ty:ignore[invalid-argument-type]


def _to_numerical_function(model: AmplitudeModel, substitutions: dict | None = None):
    expression = (
        model.full_expression
        .xreplace(model.variables)
        .xreplace(substitutions or {})
        .xreplace(model.parameter_defaults)
        .doit()
    )
    symbols = sorted(expression.free_symbols, key=str)
    function = sp.lambdify(symbols, expression, "numpy")

    def evaluate(values: dict[str, float]) -> complex:
        with np.errstate(invalid="ignore"):
            return complex(function(*[values[str(s)] for s in symbols]))

    return evaluate


def _find_phase_space_point(intensity, model: AmplitudeModel) -> dict[str, float]:
    """Scan for Mandelstam values that lie on the Dalitz surface."""
    m0, m1, m2, m3 = (float(v) for v in model.masses.values())
    masses_squared = m0**2 + m1**2 + m2**2 + m3**2
    for σ1 in np.linspace((m2 + m3) ** 2, (m0 - m1) ** 2, num=20):
        for σ2 in np.linspace((m1 + m3) ** 2, (m0 - m2) ** 2, num=20):
            values = {
                "sigma1": σ1,
                "sigma2": σ2,
                "sigma3": masses_squared - σ1 - σ2,
            }
            computed = intensity(values)
            if np.isfinite(computed.real) and computed.imag == pytest.approx(0):
                return values
    msg = "No phase space point found"
    raise ValueError(msg)


def _collect_indexed_symbols(amplitudes: list[sp.Expr]) -> set[sp.Indexed]:
    coupling_symbols: set[sp.Indexed] = set()
    for expr in amplitudes:
        symbols = {s for s in expr.free_symbols if isinstance(s, sp.Indexed)}
        coupling_symbols.update(symbols)
    return coupling_symbols


def _collect_products(amplitudes: list[sp.Expr]) -> list[tuple[sp.Indexed, sp.Indexed]]:
    products = set()
    for amp in amplitudes:
        for node in sp.postorder_traversal(amp):
            couplings = {s for s in node.free_symbols if isinstance(s, sp.Indexed)}
            if len(couplings) == 2:
                products.add(tuple(sorted(couplings, key=str)))
    return sorted(products, key=str)  # ty: ignore[invalid-return-type]


def _get_physical_amplitudes(model: AmplitudeModel) -> list[sp.Expr]:
    amplitudes = [expr.doit() for expr in model.amplitudes.values()]
    return [expr for expr in amplitudes if expr]
