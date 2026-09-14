from __future__ import annotations

import warnings
import zlib
from typing import TYPE_CHECKING

import numpy as np
import pytest
import sympy as sp

from ampform_dpd import DalitzPlotDecompositionBuilder
from ampform_dpd.adapter.qrules import (
    normalize_state_ids,
    permute_equal_final_states,
    to_three_body_decay,
)
from ampform_dpd.decay import State
from ampform_dpd.dynamics.builder import BreitWignerBuilder
from ampform_dpd.symmetrization import (
    _get_decay_coupling_basis,
    get_exchange_phase,
    get_exchange_sign,
    get_exchange_ties,
    get_identical_state_ids,
    get_missing_exchange_partners,
    get_statistics_sign,
    has_identical_particles,
    is_reordered,
    relate_exchange_couplings,
    symmetrize_identical_particles,
)

if TYPE_CHECKING:
    from qrules.transition import ReactionInfo

    from ampform_dpd.decay import ThreeBodyDecay

SIMPLE_BREIT_WIGNER = BreitWignerBuilder(
    energy_dependent_width=False,
    decay_form_factor=False,
    production_form_factor=False,
)
"""Dynamics without form factors, which need LS couplings and are slow to lambdify."""


def _create_intensity_function(decay: ThreeBodyDecay, min_ls=True, **kwargs):
    """Lambdify the Dalitz-plot density over the three Mandelstam variables."""
    builder = DalitzPlotDecompositionBuilder(decay, min_ls=min_ls)
    for chain in decay.chains:
        builder.dynamics_choices.register_builder(chain, SIMPLE_BREIT_WIGNER)
    model = builder.formulate(cleanup_summations=True, **kwargs)
    expression = model.full_expression
    for _ in range(4):
        expression = expression.xreplace(model.variables)
    expression = expression.doit()
    parameters = {p: model.parameter_defaults[p] for p in model.parameter_defaults}
    expression = expression.xreplace({
        symbol: value
        for symbol, value in parameters.items()
        if not isinstance(symbol, sp.Indexed)
    }).xreplace(model.masses)
    couplings = {c: _coupling_value(c) for c in expression.atoms(sp.Indexed)}
    sigmas = sp.symbols("sigma1:4", nonnegative=True)
    return sp.lambdify(sigmas, expression.xreplace(couplings), "numpy")


def _coupling_value(coupling: sp.Indexed) -> complex | float:
    """Give every surviving coupling its own value.

    The intensity is only sensitive to the exchange signs if the decay chains actually
    interfere, so the couplings must not all be the same. Giving the **decay** couplings
    distinct values as well is what makes the test sensitive to the ordering of their
    helicity indices: if a tie maps a coupling onto a symbol that its partner chain does
    not use, the two chains keep independent parameters and the intensity stops being
    symmetric. The value is derived from a checksum of the whole symbol, which -- unlike
    `hash` -- is stable across interpreter runs.
    """
    values = [1.0, 0.8 + 0.6j, -0.5 + 0.9j, 0.3 - 0.7j, -1.1 + 0.2j]
    checksum = zlib.crc32(str(coupling).encode())
    return values[checksum % len(values)]


def _get_exchange(decay: ThreeBodyDecay) -> dict[int, int]:
    """Transposition of the first two identical final-state particles."""
    i, j, *_ = get_identical_state_ids(decay)[0]
    k, *_ = {1, 2, 3} - {i, j}
    return {i: j, j: i, k: k}


def _max_asymmetry(func, decay: ThreeBodyDecay, grid: np.ndarray) -> float:
    """Largest relative deviation of the intensity under the particle exchange.

    The exchange permutes the final-state IDs and therefore the Mandelstam variables, so
    the check is :math:`I(\\sigma_k) = I(\\sigma_{\tau(k)})` on the constraint surface
    :math:`\\sum_k \\sigma_k = m_0^2 + \\sum_k m_k^2`.
    """
    exchange = _get_exchange(decay)
    mandelstam_sum = sum(state.mass**2 for state in decay.states.values())
    deviations = []
    for sigma1 in grid:
        for sigma2 in grid:
            sigmas = {1: sigma1, 2: sigma2, 3: mandelstam_sum - sigma1 - sigma2}
            mirrored_sigmas = [sigmas[exchange[i]] for i in (1, 2, 3)]
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                # points outside the Dalitz-plot region give nan
                warnings.simplefilter("ignore", RuntimeWarning)
                value = func(*sigmas.values())
                mirrored = func(*mirrored_sigmas)
            if not (np.isfinite(value) and np.isfinite(mirrored)):
                continue
            if abs(value) < 1e-12:
                continue
            deviations.append(abs(value - mirrored) / abs(value))
    assert deviations, "No points inside the Dalitz-plot region"
    return max(deviations)


def describe_get_identical_state_ids():
    def it_finds_a_pair(d2pipipi_reaction: ReactionInfo):
        decay = to_three_body_decay(d2pipipi_reaction.transitions)
        assert get_identical_state_ids(decay) == [(1, 2)]
        assert has_identical_particles(decay)

    def it_finds_a_triplet(a2pipipi_reaction: ReactionInfo):
        transitions = normalize_state_ids(a2pipipi_reaction.transitions)
        decay = to_three_body_decay(transitions)
        assert get_identical_state_ids(decay) == [(1, 2, 3)]

    def it_returns_nothing_for_distinct_final_states(jpsi2pksigma_reaction):
        transitions = normalize_state_ids(jpsi2pksigma_reaction.transitions)
        decay = to_three_body_decay(transitions)
        assert get_identical_state_ids(decay) == []
        assert not has_identical_particles(decay)


def describe_get_statistics_sign():
    @pytest.mark.parametrize(
        ("spin", "expected"),
        [("0", +1), ("1/2", -1), ("1", +1), ("3/2", -1), ("2", +1)],
    )
    def it_alternates_with_the_spin(spin: str, expected: int):
        state = State(
            name="X",
            latex="X",
            spin=sp.Rational(spin),
            parity=+1,
            mass=1.0,
            width=0.0,
            index=1,
        )
        assert get_statistics_sign(state) == expected

    def it_is_positive_for_identical_bosons(d2pipipi_reaction: ReactionInfo):
        decay = to_three_body_decay(d2pipipi_reaction.transitions)
        pion = decay.final_state[1]
        assert pion.spin == 0
        assert get_statistics_sign(pion) == +1

    def it_is_negative_for_fermions(xib2pkk_reaction: ReactionInfo):
        transitions = normalize_state_ids(xib2pkk_reaction.transitions)
        decay = to_three_body_decay(transitions)
        proton = next(s for s in decay.final_state.values() if s.name == "p")
        assert get_statistics_sign(proton) == -1


def describe_get_exchange_phase():
    def it_is_minus_one_for_odd_spin_isobars(d2pipipi_reaction: ReactionInfo):
        decay = to_three_body_decay(d2pipipi_reaction.transitions)
        phases = {
            chain.resonance.name: get_exchange_phase(chain, "helicity")
            for chain in decay.chains
        }
        assert phases == {"f(0)(980)": +1, "rho(770)0": -1, "f(2)(1270)": +1}

    def it_rejects_an_unknown_basis(d2pipipi_reaction: ReactionInfo):
        decay = to_three_body_decay(d2pipipi_reaction.transitions)
        with pytest.raises(ValueError, match='either "LS" or "helicity"'):
            get_exchange_phase(decay.chains[0], "canonical")  # ty: ignore[invalid-argument-type]

    def it_needs_ls_couplings_for_the_ls_basis(d2pipipi_reaction: ReactionInfo):
        decay = to_three_body_decay(d2pipipi_reaction.transitions, min_ls=True)
        with pytest.raises(ValueError, match="no LS coupling on its decay node"):
            get_exchange_phase(decay.chains[0], "LS")


def describe_get_exchange_ties():
    def it_ties_the_two_subsystems(d2pipipi_reaction: ReactionInfo):
        decay = to_three_body_decay(d2pipipi_reaction.transitions, min_ls=True)
        ties = get_exchange_ties(decay)
        assert len(ties) == 3  # one per resonance
        for tie in ties:
            assert tie.reference.spectator.index == 1
            assert tie.chain.spectator.index == 2
            assert tie.reference.resonance == tie.chain.resonance
            assert tie.state_map == {1: 2, 2: 1, 3: 3}
            assert is_reordered(tie), "cyclic (23)1 vs (31)2 is a reversed pair"

    def it_signs_odd_spin_isobars_negatively(d2pipipi_reaction: ReactionInfo):
        decay = to_three_body_decay(d2pipipi_reaction.transitions, min_ls=True)
        signs = {
            tie.chain.resonance.name: get_exchange_sign(tie, "helicity")
            for tie in get_exchange_ties(decay)
        }
        assert signs == {"f(0)(980)": +1, "rho(770)0": -1, "f(2)(1270)": +1}

    def it_finds_nothing_without_identical_particles(jpsi2pksigma_reaction):
        transitions = normalize_state_ids(jpsi2pksigma_reaction.transitions)
        decay = to_three_body_decay(transitions, min_ls=True)
        assert get_exchange_ties(decay) == []


def describe_symmetrize_identical_particles():
    def it_keeps_one_coupling_per_resonance(d2pipipi_reaction: ReactionInfo):
        decay = to_three_body_decay(d2pipipi_reaction.transitions, min_ls=True)
        builder = DalitzPlotDecompositionBuilder(decay, min_ls=True)
        asymmetric = builder.formulate(cleanup_summations=True, symmetrize=False)
        symmetric = builder.formulate(cleanup_summations=True)
        subsystems = {
            str(p.indices[0]).rsplit(",", maxsplit=1)[-1]
            for p in asymmetric.parameter_defaults
            if isinstance(p, sp.Indexed)
        }
        assert subsystems == {"1", "2"}, "each subsystem has its own couplings"
        subsystems = {
            str(p.indices[0]).rsplit(",", maxsplit=1)[-1]
            for p in symmetric.parameter_defaults
            if isinstance(p, sp.Indexed)
        }
        assert subsystems == {"1"}, "subsystem 2 is expressed in terms of subsystem 1"

    def it_puts_the_sign_on_the_production_coupling(d2pipipi_reaction: ReactionInfo):
        decay = to_three_body_decay(d2pipipi_reaction.transitions, min_ls=True)
        builder = DalitzPlotDecompositionBuilder(decay, min_ls=True)
        model = builder.formulate(cleanup_summations=True, symmetrize=False)
        substitutions = relate_exchange_couplings(model)
        negated = {
            str(target.args[1].indices[0])  # ty: ignore[unresolved-attribute]
            for source, target in substitutions.items()
            if target.could_extract_minus_sign() and "production" in str(source.base)
        }
        assert negated == {R"\rho(770)^{0},1"}

    def it_is_a_no_op_without_identical_particles(jpsi2pksigma_reaction):
        transitions = normalize_state_ids(jpsi2pksigma_reaction.transitions)
        decay = to_three_body_decay(transitions, min_ls=True)
        builder = DalitzPlotDecompositionBuilder(decay, min_ls=True)
        model = builder.formulate(reference_subsystem=2, cleanup_summations=True)
        assert symmetrize_identical_particles(model) is model


def describe_bose_symmetry():
    @pytest.mark.parametrize("reference_subsystem", [1, 2])
    def it_makes_the_d2pipipi_intensity_symmetric(
        d2pipipi_reaction: ReactionInfo, reference_subsystem: int
    ):
        decay = to_three_body_decay(d2pipipi_reaction.transitions, min_ls=True)
        grid = np.linspace(0.15, 2.8, 25)
        symmetric = _create_intensity_function(
            decay, reference_subsystem=reference_subsystem
        )
        assert _max_asymmetry(symmetric, decay, grid) < 1e-10

    def it_is_violated_without_symmetrization(d2pipipi_reaction: ReactionInfo):
        decay = to_three_body_decay(d2pipipi_reaction.transitions, min_ls=True)
        grid = np.linspace(0.15, 2.8, 25)
        asymmetric = _create_intensity_function(decay, symmetrize=False)
        assert _max_asymmetry(asymmetric, decay, grid) > 1

    @pytest.mark.parametrize("reference_subsystem", [2, 3])
    def it_makes_the_xib2pkk_intensity_symmetric(
        xib2pkk_reaction: ReactionInfo, reference_subsystem: int
    ):
        transitions = normalize_state_ids(xib2pkk_reaction.transitions)
        decay = to_three_body_decay(transitions, min_ls=True)
        grid = np.linspace(2.5, 26, 20)
        func = _create_intensity_function(
            decay, reference_subsystem=reference_subsystem
        )
        assert _max_asymmetry(func, decay, grid) < 1e-10

    @pytest.mark.parametrize("reference_subsystem", [1, 2, 3])
    def it_makes_the_a2pipipi_intensity_symmetric(
        a2pipipi_reaction: ReactionInfo, reference_subsystem: int
    ):
        reaction = permute_equal_final_states(normalize_state_ids(a2pipipi_reaction))
        decay = to_three_body_decay(reaction.transitions, min_ls=True)
        grid = np.linspace(0.08, 1.4, 25)
        func = _create_intensity_function(
            decay, reference_subsystem=reference_subsystem
        )
        assert _max_asymmetry(func, decay, grid) < 1e-10

    @pytest.mark.parametrize("reference_subsystem", [1, 2])
    def it_makes_the_spin1_intensity_symmetric(
        b2rhorhopi_reaction: ReactionInfo, reference_subsystem: int
    ):
        """Identical particles that carry spin.

        The rho(770) mesons have spin 1, so the exchange permutes non-trivial helicities
        and the alignment Wigner-d functions of the exchanged states are not the identity.
        """
        decay = to_three_body_decay(b2rhorhopi_reaction.transitions, min_ls=True)
        assert {s.spin for s in decay.final_state.values()} == {0, 1}
        grid = np.linspace(0.9, 20.0, 16)
        func = _create_intensity_function(
            decay, reference_subsystem=reference_subsystem
        )
        assert _max_asymmetry(func, decay, grid) < 1e-10

    def it_is_violated_for_spin1_without_symmetrization(
        b2rhorhopi_reaction: ReactionInfo,
    ):
        decay = to_three_body_decay(b2rhorhopi_reaction.transitions, min_ls=True)
        grid = np.linspace(0.9, 20.0, 16)
        func = _create_intensity_function(decay, symmetrize=False)
        assert _max_asymmetry(func, decay, grid) > 0.1


def describe_coupling_bases():
    """The exchange sign has to land on the vertex that identifies the chain."""

    @pytest.mark.parametrize(
        ("min_ls", "use_coefficients"),
        [
            (True, False),
            (False, False),
            ((True, False), False),
            ((False, True), False),
            (True, True),
            (False, True),
        ],
    )
    def it_symmetrizes_in_every_basis(
        d2pipipi_canonical_reaction: ReactionInfo,
        min_ls: bool | tuple[bool, bool],
        use_coefficients: bool,
    ):
        decay = to_three_body_decay(
            d2pipipi_canonical_reaction.transitions, min_ls=min_ls
        )
        func = _create_intensity_function(
            decay, min_ls=min_ls, use_coefficients=use_coefficients
        )
        grid = np.linspace(0.15, 2.8, 20)
        assert _max_asymmetry(func, decay, grid) < 1e-10

    @pytest.mark.parametrize(
        ("min_ls", "expected_basis", "expected_carrier"),
        [
            (True, "helicity", R"\mathcal{H}^\mathrm{production}"),
            (False, "LS", R"\mathcal{H}^\mathrm{LS,decay}"),
            ((True, False), "LS", R"\mathcal{H}^\mathrm{LS,decay}"),
            ((False, True), "helicity", R"\mathcal{H}^\mathrm{LS,production}"),
        ],
    )
    def it_signs_the_identifying_vertex(
        d2pipipi_canonical_reaction: ReactionInfo,
        min_ls: bool | tuple[bool, bool],
        expected_basis: str,
        expected_carrier: str,
    ):
        decay = to_three_body_decay(
            d2pipipi_canonical_reaction.transitions, min_ls=min_ls
        )
        builder = DalitzPlotDecompositionBuilder(decay, min_ls=min_ls)
        model = builder.formulate(cleanup_summations=True, symmetrize=False)
        assert _get_decay_coupling_basis(model) == expected_basis
        negated_bases = {
            str(source.base)
            for source, target in relate_exchange_couplings(model).items()
            if target.could_extract_minus_sign()
        }
        assert negated_bases == {expected_carrier}, (
            "only the rho(770) has a negative exchange phase"
        )


def describe_parameter_bookkeeping():
    @pytest.mark.parametrize("min_ls", [True, False])
    @pytest.mark.parametrize("use_coefficients", [False, True])
    def it_declares_every_coupling_it_uses(
        d2pipipi_canonical_reaction: ReactionInfo,
        xib2pkk_canonical_reaction: ReactionInfo,
        min_ls: bool,
        use_coefficients: bool,
    ):
        """Every coupling in the amplitudes has to survive as a parameter.

        A tie that maps a coupling onto a symbol its partner chain does not use leaves the
        expression referring to an undeclared parameter, which is how a silently broken
        symmetrization shows up.
        """
        reactions = [d2pipipi_canonical_reaction, xib2pkk_canonical_reaction]
        for reaction in reactions:
            decay = to_three_body_decay(reaction.transitions, min_ls=min_ls)
            builder = DalitzPlotDecompositionBuilder(decay, min_ls=min_ls)
            model = builder.formulate(
                cleanup_summations=True, use_coefficients=use_coefficients
            )
            declared = {
                p for p in model.parameter_defaults if isinstance(p, sp.Indexed)
            }
            lambda_r = sp.Symbol(R"\lambda_R", rational=True)
            used = set()
            for expression in model.amplitudes.values():
                used |= {
                    node
                    for node in sp.preorder_traversal(expression.doit())
                    if isinstance(node, sp.Indexed)
                    and not str(node.base).startswith("A^")
                    and lambda_r not in node.free_symbols
                }
            assert used - declared == set(), "undeclared couplings"


def describe_warn_about_missing_exchange_partners():
    def it_warns_if_the_permutations_were_not_generated(
        a2pipipi_reaction: ReactionInfo,
    ):
        transitions = normalize_state_ids(a2pipipi_reaction.transitions)
        decay = to_three_body_decay(transitions, min_ls=True)
        assert [i for _, i in get_missing_exchange_partners(decay)] == [2, 3]
        builder = DalitzPlotDecompositionBuilder(decay, min_ls=True)
        with pytest.warns(UserWarning, match="permute_equal_final_states"):
            builder.formulate(cleanup_summations=True)

    def it_stays_silent_if_they_were(d2pipipi_reaction: ReactionInfo):
        decay = to_three_body_decay(d2pipipi_reaction.transitions, min_ls=True)
        assert get_missing_exchange_partners(decay) == []
