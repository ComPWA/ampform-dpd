# cspell:ignore pksigma
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import sympy as sp

from ampform_dpd import DalitzPlotDecompositionBuilder
from ampform_dpd.adapter.qrules import normalize_state_ids, to_three_body_decay
from ampform_dpd.dynamics.builder import formulate_breit_wigner_with_form_factor

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
        assert {s.name for s in model.variables} == expected_variables

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
        amplitudes = [expr.doit() for expr in model.amplitudes.values()]
        amplitudes = [expr for expr in amplitudes if expr]

        coupling_symbols: set[str] = set()
        for expr in amplitudes:
            symbols = {s for s in expr.free_symbols if isinstance(s, sp.Indexed)}
            coupling_symbols.update(symbols)

        n_coupling_symbols = len(coupling_symbols)
        coupling_symbols_str = sorted(str(s) for s in coupling_symbols)
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


def _collect_products(amplitudes: list[sp.Expr]) -> list[tuple[sp.Indexed, sp.Indexed]]:
    products = set()
    for amp in amplitudes:
        for node in sp.postorder_traversal(amp):
            couplings = {s for s in node.free_symbols if isinstance(s, sp.Indexed)}
            if len(couplings) == 2:
                products.add(tuple(sorted(couplings, key=str)))
    return sorted(products, key=str)
