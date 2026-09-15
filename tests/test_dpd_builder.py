# cspell:ignore gammapipi pksigma
from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from itertools import product
from typing import TYPE_CHECKING

import attrs
import pytest
import qrules
import sympy as sp
from ampform.sympy import PoolSum

from ampform_dpd import (
    AmplitudeModel,
    DalitzPlotDecompositionBuilder,
    DefinedExpression,
    _get_helicity_range,
)
from ampform_dpd.adapter.qrules import normalize_state_ids, to_three_body_decay
from ampform_dpd.decay import (
    IsobarNode,
    Particle,
    State,
    ThreeBodyDecay,
    ThreeBodyDecayChain,
    get_decay_product_ids,
)
from ampform_dpd.dynamics import formulate_breit_wigner_with_form_factor
from ampform_dpd.spin import create_spin_range, generate_ls_couplings

if TYPE_CHECKING:
    from qrules.transition import ReactionInfo

    from ampform_dpd.decay import FinalState, FinalStateID


@pytest.fixture(params=[1, 2, 3])
def jpsi2gammapipi_decay(request):
    photon_id = request.param
    parent = State("J/psi", "J/psi", 1, -1, 3.1, 0, index=0)
    states: dict[FinalStateID, FinalState] = {}
    for i in (1, 2, 3):
        states[i] = (
            State("gamma", "gamma", 1, -1, 0, 0, index=i)
            if i == photon_id
            else State(f"pi{i}", f"pi_{i}", 0, -1, 0.14, 0, index=i)
        )
    chains = []
    for k in (1, 2, 3):
        i, j = get_decay_product_ids(k)
        resonance_spins = (0, 2) if k == photon_id else (1,)
        for spin in resonance_spins:
            resonance = Particle(f"R{k}_{spin}", f"R{k}_{spin}", spin, 1, 1.5, 0.1)
            production_ls = generate_ls_couplings(1, spin, states[k].spin, max_L=4)
            decay_ls = generate_ls_couplings(spin, states[i].spin, states[j].spin)
            for incoming_ls, outgoing_ls in product(production_ls, decay_ls):
                node = IsobarNode(
                    parent=resonance,
                    child1=states[i],
                    child2=states[j],
                    interaction=outgoing_ls,
                )
                chains.append(
                    ThreeBodyDecayChain(
                        IsobarNode(parent, node, states[k], interaction=incoming_ls)
                    )
                )
    return ThreeBodyDecay({0: parent, **states}, chains)  # ty: ignore[invalid-argument-type]


@pytest.fixture
def radiative_decay():
    parent = State("parent", "P", 0.5, 1, 5, 0, index=0)
    first = State("first", "a", 0.5, 1, 1, 0, index=1)
    second = State("second", "b", 0, -1, 1, 0, index=2)
    photon = State("photon", "g", 1, -1, 0, 0, index=3)
    resonance = Particle("resonance", "R", 1.5, -1, 3, 0)
    node = IsobarNode(resonance, first, second, interaction=(2, 0.5))
    chains = [
        ThreeBodyDecayChain(IsobarNode(parent, node, photon, interaction=ls))
        for ls in generate_ls_couplings(parent.spin, resonance.spin, photon.spin)
    ]
    return ThreeBodyDecay({s.index: s for s in (parent, first, second, photon)}, chains)  # ty: ignore[invalid-argument-type]


def describe_DalitzPlotDecompositionBuilder():
    @pytest.mark.parametrize("all_subsystems", [False, True])
    @pytest.mark.parametrize("min_ls", [False, True])
    def it_can_include_all_subsystems(
        jpsi2pksigma_reaction: ReactionInfo, all_subsystems: bool, min_ls: bool
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
    def it_can_use_coefficients(
        jpsi2pksigma_reaction: ReactionInfo, min_ls: bool, use_coefficients: bool
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
    def it_supports_coefficient_combinations(basis: str, resonance: str):  # ruff: ignore[too-many-locals]
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

    def it_makes_ls_amplitudes_independent_of_child_order(
        jpsi2pksigma_reaction: ReactionInfo,
    ):
        """The subsystem amplitude may not depend on how a decay node is stored.

        Which of the two children of an `.IsobarNode` is :attr:`~.IsobarNode.child1` is
        an implementation detail of whoever constructed the `.ThreeBodyDecay`, but the
        Clebsch-Gordan factors of the :math:`LS` basis and the isobar Wigner-:math:`d`
        function both depend on the ordering of the decay products, so the two have to be
        brought into the same ordering first.
        """
        if jpsi2pksigma_reaction.formalism == "helicity":
            pytest.skip("Helicity formalism does not have LS couplings")
        transitions = normalize_state_ids(jpsi2pksigma_reaction.transitions)
        decay = to_three_body_decay(transitions, min_ls=True)
        builders = [
            DalitzPlotDecompositionBuilder(d, min_ls=False)
            for d in (decay, _swap_decay_products(decay))
        ]
        helicities = (
            sp.Integer(1),
            sp.Integer(0),
            sp.Rational(1, 2),
            sp.Rational(-1, 2),
        )
        for subsystem_id in sorted({c.spectator.index for c in decay.chains}):
            expressions = [
                next(
                    iter(
                        b.formulate_subsystem_amplitude(
                            *helicities,
                            subsystem_id,
                        ).amplitudes.values()
                    )
                ).doit()
                for b in builders
            ]
            assert sp.simplify(expressions[0] - expressions[1]) == 0, subsystem_id


def _swap_decay_products(decay: ThreeBodyDecay) -> ThreeBodyDecay:
    """Reverse the order in which every decay node stores its two children."""
    chains = []
    for chain in decay.chains:
        node = attrs.evolve(
            chain.decay_node,
            child1=chain.decay_node.child2,
            child2=chain.decay_node.child1,
        )
        chains.append(attrs.evolve(chain, decay=attrs.evolve(chain.decay, child1=node)))
    return ThreeBodyDecay(decay.states, chains)


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


def describe_jpsi2gammapipi():
    @pytest.mark.parametrize(
        "min_ls",
        [True, False, (True, True), (True, False), (False, True), (False, False)],
    )
    @pytest.mark.parametrize("use_coefficients", [False, True])
    def it_constructs_only_physical_terms(
        jpsi2gammapipi_decay, min_ls, use_coefficients
    ):
        decay = jpsi2gammapipi_decay
        builder = DalitzPlotDecompositionBuilder(decay, min_ls=min_ls)
        model = builder.formulate(use_coefficients=use_coefficients)
        photon = next(state for state in decay.final_state.values() if state.mass == 0)
        assert isinstance(model.intensity, PoolSum)
        assert tuple(model.intensity.indices[photon.index][1]) == (-1, 1)
        assert len(model.amplitudes) == 18
        assert all(key.indices[photon.index] in {-1, 1} for key in model.amplitudes)
        assert all(
            f"zeta^{photon.index}" not in str(symbol) for symbol in model.variables
        )
        assert all(
            not expression.has(PoolSum, sp.KroneckerDelta)
            for expression in model.amplitudes.values()
        )
        coupling_symbols = set().union(
            *(expression.atoms(sp.Indexed) for expression in model.amplitudes.values())
        )
        assert set(model.parameter_defaults) - set(model.masses) == coupling_symbols
        assert any(model.amplitudes.values())

    @pytest.mark.parametrize("min_ls", [True, False, (True, False), (False, True)])
    @pytest.mark.parametrize("subsystem", [1, 2, 3])
    def it_never_constructs_a_longitudinal_photon(
        jpsi2gammapipi_decay, min_ls, subsystem
    ):
        builder = DalitzPlotDecompositionBuilder(jpsi2gammapipi_decay, min_ls=min_ls)

        def unexpected_dynamics(_):
            pytest.fail("Dynamics must not be constructed for a forbidden helicity")

        for chain in jpsi2gammapipi_decay.chains:
            builder.dynamics_choices.register_builder(chain, unexpected_dynamics)
        model = builder.formulate_subsystem_amplitude(
            sp.S.Zero, sp.S.Zero, sp.S.Zero, sp.S.Zero, subsystem
        )
        assert set(model.amplitudes.values()) == {sp.S.Zero}
        assert not model.parameter_defaults

    def it_preserves_distinct_ls_dynamics(jpsi2gammapipi_decay):
        photon = next(
            state
            for state in jpsi2gammapipi_decay.final_state.values()
            if state.mass == 0
        )
        chains = [
            chain
            for chain in jpsi2gammapipi_decay.chains
            if chain.spectator == photon and chain.resonance.spin == 0
        ]
        decay = ThreeBodyDecay(jpsi2gammapipi_decay.states, chains)
        builder = DalitzPlotDecompositionBuilder(decay, min_ls=False)

        def dynamics(chain):
            return DefinedExpression(sp.Symbol(f"D{chain.incoming_ls.L}"))

        for chain in chains:
            builder.dynamics_choices.register_builder(chain, dynamics)
        helicities = [sp.S.NegativeOne, sp.S.Zero, sp.S.Zero, sp.S.Zero]
        helicities[photon.index] = sp.S.One
        model = builder.formulate_subsystem_amplitude(
            λ0=helicities[0],
            λ1=helicities[1],
            λ2=helicities[2],
            λ3=helicities[3],
            subsystem_id=photon.index,
        )
        expression = next(iter(model.amplitudes.values()))
        assert all(expression.has(sp.Symbol(f"D{i}")) for i in (0, 1, 2))
        production = {
            symbol for symbol in model.parameter_defaults if "production" in str(symbol)
        }
        assert len(production) == 3


def describe_massless_final_states():
    @pytest.mark.parametrize("spin", [0, 0.5, 1, 2])
    @pytest.mark.parametrize("mass", [0, 1])
    def it_preserves_scalar_and_massive_spin_ranges(radiative_decay, spin, mass):
        state = attrs.evolve(radiative_decay.final_state[3], spin=spin, mass=mass)
        expected = (
            [-sp.Rational(spin), sp.Rational(spin)]
            if mass == 0 and spin
            else create_spin_range(spin)
        )
        assert _get_helicity_range(state) == expected

    @pytest.mark.parametrize("min_ls", [False, True])
    def it_restricts_both_helicity_sums(radiative_decay, min_ls):
        model = DalitzPlotDecompositionBuilder(
            radiative_decay, min_ls=min_ls
        ).formulate()
        assert isinstance(model.intensity, PoolSum)
        assert tuple(model.intensity.indices[-1][1]) == (-1, 1)
        assert {key.indices[-1] for key in model.amplitudes} == {-1, 1}
        aligned = model.intensity.expression.args[0].args[0]
        assert isinstance(aligned, PoolSum)
        assert tuple(aligned.indices[-1][1]) == (-1, 1)
        assert all("zeta^3" not in str(symbol) for symbol in model.variables)
        assert not model.full_expression.has(sp.nan, sp.zoo)

    @pytest.mark.parametrize("reference", [1, 2, 3])
    def it_conserves_photon_helicity_across_subsystems(radiative_decay, reference):
        builder = DalitzPlotDecompositionBuilder(radiative_decay)
        with (
            pytest.warns(UserWarning, match="only has subsystem")
            if reference != 3
            else does_not_raise()
        ):
            amplitude, angles = builder.formulate_aligned_amplitude(
                λ0=sp.S.Half,
                λ1=-sp.S.Half,
                λ2=sp.S.Zero,
                λ3=sp.S.One,
                reference_subsystem=reference,
            )
        assert all("zeta^3" not in str(symbol) for symbol in angles)
        assert amplitude.doit().atoms(sp.Indexed)
        assert {a.indices[-1] for a in amplitude.doit().atoms(sp.Indexed)} == {1}
