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
from ampform_dpd.polarization import EULER_ANGLES, formulate_unpolarized_density

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
        decay = to_three_body_decay(transitions, min_ls=min_ls)  # ty:ignore[invalid-argument-type]
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
        decay = to_three_body_decay(transitions, min_ls=min_ls)  # ty:ignore[invalid-argument-type]
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
    def test_use_coefficients_combinations(self, basis: str, resonance: str):  # noqa: PLR0914
        reaction = qrules.generate_transitions(
            initial_state=[("J/psi(1S)", [+1])],
            final_state=[("Sigma+", [+0.5]), "K0", ("p~", [+0.5])],
            allowed_interaction_types="strong",
            allowed_intermediate_particles=[resonance],
            formalism="canonical-helicity",
        )
        transitions = normalize_state_ids(reaction.transitions)
        min_ls = basis == "helicity"
        decay = to_three_body_decay(transitions, min_ls)  # ty:ignore[invalid-argument-type]
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


def test_polarized_intensity(lc2pkpi_single_resonance_reaction: ReactionInfo):
    transitions = normalize_state_ids(lc2pkpi_single_resonance_reaction.transitions)
    decay = to_three_body_decay(transitions, min_ls=True)  # ty:ignore[invalid-argument-type]
    builder = DalitzPlotDecompositionBuilder(decay, min_ls=True)
    unpolarized_model = builder.formulate()
    polarized_model = builder.formulate(polarized=True)

    ρ = _get_matrix_symbol(polarized_model.intensity)
    assert ρ.shape == (2, 2)
    assert set(EULER_ANGLES) <= polarized_model.intensity.free_symbols

    unpolarized_intensity = _to_numerical_function(unpolarized_model)
    polarized_intensity = _to_numerical_function(
        polarized_model, {ρ: formulate_unpolarized_density(decay.initial_state.spin)}
    )
    phase_space_point = _find_phase_space_point(
        unpolarized_intensity, unpolarized_model
    )
    expected = unpolarized_intensity(phase_space_point)
    assert np.isfinite(expected.real)
    rng = np.random.default_rng(seed=0)
    for _ in range(3):
        euler_angles = {
            "phi": rng.uniform(-np.pi, +np.pi),
            "theta": rng.uniform(0, np.pi),
            "chi": rng.uniform(-np.pi, +np.pi),
        }
        computed = polarized_intensity({**phase_space_point, **euler_angles})
        assert computed.imag == pytest.approx(0)
        assert ρ.shape[0] * computed.real == pytest.approx(expected.real)


def _get_matrix_symbol(expression: sp.Expr) -> sp.MatrixSymbol:
    return next(s for s in expression.free_symbols if isinstance(s, sp.MatrixSymbol))


def _to_numerical_function(
    model: AmplitudeModel,
    substitutions: dict[sp.MatrixSymbol, sp.ImmutableDenseMatrix] | None = None,
):
    expression = (
        model.full_expression
        .xreplace(model.variables)
        .xreplace(model.parameter_defaults)
        .xreplace(substitutions or {})
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
    return sorted(products, key=str)  # ty:ignore[invalid-return-type]


def _get_physical_amplitudes(model: AmplitudeModel) -> list[sp.Expr]:
    amplitudes = [expr.doit() for expr in model.amplitudes.values()]
    return [expr for expr in amplitudes if expr]
