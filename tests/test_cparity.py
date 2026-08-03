from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import sympy as sp

from ampform_dpd import DalitzPlotDecompositionBuilder
from ampform_dpd.adapter.qrules import normalize_state_ids, to_three_body_decay
from ampform_dpd.cparity import (
    get_antiparticle_name,
    get_c_forbidden_chains,
    get_conjugate_chain_pairs,
    get_conjugate_coupling_sign,
    get_conjugate_state_map,
    is_c_symmetric,
    relate_conjugate_couplings,
    symmetrize_conjugate_couplings,
)

if TYPE_CHECKING:
    from qrules.transition import ReactionInfo

    from ampform_dpd.decay import ThreeBodyDecay


@pytest.fixture(scope="session")
def jpsi2etappbar_decay(jpsi2etappbar_reaction: ReactionInfo) -> ThreeBodyDecay:
    transitions = normalize_state_ids(jpsi2etappbar_reaction.transitions)
    return to_three_body_decay(transitions, min_ls=True)


@pytest.fixture(scope="session")
def jpsi2pipipi_decay(jpsi2pipipi_reaction: ReactionInfo) -> ThreeBodyDecay:
    transitions = normalize_state_ids(jpsi2pipipi_reaction.transitions)
    return to_three_body_decay(transitions, min_ls=True)


def _signs(decay: ThreeBodyDecay, basis) -> dict[str, int]:
    return {
        chain.resonance.name: get_conjugate_coupling_sign(chain, basis)
        for chain, _ in get_conjugate_chain_pairs(decay)
    }


def _create_intensity_function(model, seed: int = 0):
    """Lambdify the intensity of a model, with random values for all couplings."""
    rng = np.random.default_rng(seed)
    expression = model.full_expression
    variables = {k: v.doit() for k, v in model.variables.items()}
    for _ in range(3):  # angle definitions are nested a few levels deep
        expression = expression.xreplace(variables)
    parameters = {
        symbol: complex(rng.normal(), rng.normal())
        if isinstance(symbol, sp.Indexed)
        else value
        for symbol, value in model.parameter_defaults.items()
    }
    parameters.update({
        sp.Symbol(f"m{i}", nonnegative=True): state.mass
        for i, state in sorted(model.decay.states.items())
    })
    expression = expression.xreplace(parameters)
    remaining = {
        node
        for node in sp.preorder_traversal(expression)
        if isinstance(node, sp.Indexed)
    }
    assert not remaining
    return sp.lambdify(sp.symbols("sigma1:4", nonnegative=True), expression, "numpy")


def _generate_phase_space(decay: ThreeBodyDecay, n_points: int = 5, seed: int = 0):
    """Generate Mandelstam variables that lie **within** the Dalitz plot.

    For a given :math:`\\sigma_3 = m^2(12)`, the boundaries of :math:`\\sigma_1 =
    m^2(23)` follow from the energies and momenta of particles 2 and 3 in the rest frame
    of the :math:`(12)` pair.
    """
    rng = np.random.default_rng(seed)
    m = [state.mass for _, state in sorted(decay.states.items())]
    s_total = sum(x**2 for x in m)
    points = []
    while len(points) < n_points:
        sigma3 = rng.uniform((m[1] + m[2]) ** 2, (m[0] - m[3]) ** 2)
        energy2 = (sigma3 - m[1] ** 2 + m[2] ** 2) / (2 * np.sqrt(sigma3))
        energy3 = (m[0] ** 2 - sigma3 - m[3] ** 2) / (2 * np.sqrt(sigma3))
        momentum2 = np.sqrt(energy2**2 - m[2] ** 2)
        momentum3 = np.sqrt(energy3**2 - m[3] ** 2)
        sigma1_min = (energy2 + energy3) ** 2 - (momentum2 + momentum3) ** 2
        sigma1_max = (energy2 + energy3) ** 2 - (momentum2 - momentum3) ** 2
        margin = 0.02 * (sigma1_max - sigma1_min)
        sigma1 = rng.uniform(sigma1_min + margin, sigma1_max - margin)
        points.append({1: sigma1, 2: s_total - sigma1 - sigma3, 3: sigma3})
    return points


def _compute_mirror_asymmetry(model, state_map, seed: int = 0) -> float:
    """Largest relative difference between the intensity and its C-mirror image."""
    func = _create_intensity_function(model, seed)
    asymmetry = []
    for point in _generate_phase_space(model.decay, seed=seed):
        original = func(*[point[i] for i in (1, 2, 3)])
        mirrored = func(*[point[state_map[i]] for i in (1, 2, 3)])
        if not np.isfinite(original) or not np.isfinite(mirrored):
            continue
        asymmetry.append(abs(original - mirrored) / (abs(original) + abs(mirrored)))
    assert asymmetry
    return max(asymmetry)


class TestGetAntiparticleName:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("N(1535)+", "N(1535)~-"),
            ("N(1535)~-", "N(1535)+"),
            ("p", "p~"),
            ("rho(770)+", "rho(770)-"),
            ("eta", "eta"),
            ("pi0", "pi0"),
            ("rho(770)0", "rho(770)0"),
        ],
    )
    def test_antiparticle(self, name: str, expected: str):
        assert get_antiparticle_name(name) == expected


class TestGetConjugateStateMap:
    def test_transposition(self, jpsi2etappbar_decay: ThreeBodyDecay):
        assert get_conjugate_state_map(jpsi2etappbar_decay) == {1: 1, 2: 3, 3: 2}
        assert is_c_symmetric(jpsi2etappbar_decay)

    def test_map_is_the_same_for_a_single_chain(
        self, jpsi2etappbar_decay: ThreeBodyDecay
    ):
        for chain in jpsi2etappbar_decay.chains:
            assert get_conjugate_state_map(chain) == {1: 1, 2: 3, 3: 2}

    def test_pipipi(self, jpsi2pipipi_decay: ThreeBodyDecay):
        assert get_conjugate_state_map(jpsi2pipipi_decay) == {1: 1, 2: 3, 3: 2}

    def test_identity(self, a2pipipi_reaction: ReactionInfo):
        transitions = normalize_state_ids(a2pipipi_reaction.transitions)
        decay = to_three_body_decay(transitions)
        assert get_conjugate_state_map(decay) == {1: 1, 2: 2, 3: 3}

    def test_gate_fails(self, jpsi2pksigma_reaction: ReactionInfo):
        transitions = normalize_state_ids(jpsi2pksigma_reaction.transitions)
        decay = to_three_body_decay(transitions, min_ls=True)
        assert not is_c_symmetric(decay)
        with pytest.raises(
            ValueError, match="does not map the final state onto itself"
        ):
            get_conjugate_state_map(decay)


class TestGetConjugateCouplingSign:
    def test_ls_sign_is_the_parity_of_the_resonance(
        self, jpsi2etappbar_decay: ThreeBodyDecay
    ):
        """For :math:`J/\\psi\\to p\\bar p\\eta`, the LS sign reduces to :math:`P_{N^*}`."""
        assert _signs(jpsi2etappbar_decay, "LS") == {
            "N(1535)+": -1,  # 1/2^-
            "N(1710)+": +1,  # 1/2^+
        }
        for chain, _ in get_conjugate_chain_pairs(jpsi2etappbar_decay):
            assert get_conjugate_coupling_sign(chain, "LS") == chain.resonance.parity

    def test_helicity_sign_differs_from_the_ls_sign(
        self, jpsi2etappbar_decay: ThreeBodyDecay
    ):
        """:math:`s_\\text{hel} = -(-1)^{J-1/2}` is the same for both :math:`J=1/2`."""
        assert _signs(jpsi2etappbar_decay, "helicity") == {
            "N(1535)+": -1,
            "N(1710)+": -1,
        }

    def test_sign_is_symmetric_within_a_pair(self, jpsi2etappbar_decay: ThreeBodyDecay):
        for basis in ["LS", "helicity"]:
            for chain, conjugate_chain in get_conjugate_chain_pairs(
                jpsi2etappbar_decay
            ):
                sign = get_conjugate_coupling_sign(chain, basis)
                assert get_conjugate_coupling_sign(conjugate_chain, basis) == sign

    def test_rho_pair(self, jpsi2pipipi_decay: ThreeBodyDecay):
        """Cross-check: isospin gives the same :math:`+1` for the :math:`\\rho^\\pm`."""
        assert _signs(jpsi2pipipi_decay, "LS") == {"rho(770)+": +1}
        assert _signs(jpsi2pipipi_decay, "helicity") == {"rho(770)+": +1}

    def test_no_ls_couplings(self, jpsi2etappbar_reaction: ReactionInfo):
        transitions = normalize_state_ids(jpsi2etappbar_reaction.transitions)
        decay = to_three_body_decay(transitions, min_ls=False)
        chain = next(c for c in decay.chains if c.outgoing_ls is not None)
        chain = type(chain)(
            decay=chain.decay.__class__(
                parent=chain.decay.parent,
                child1=chain.decay_node.__class__(
                    parent=chain.resonance,
                    child1=chain.decay_node.child1,
                    child2=chain.decay_node.child2,
                ),
                child2=chain.spectator,
                interaction=chain.incoming_ls,
            )
        )
        with pytest.raises(ValueError, match="no LS coupling on its decay node"):
            get_conjugate_coupling_sign(chain, "LS")
        assert get_conjugate_coupling_sign(chain, "helicity") == -1


class TestGetConjugateChainPairs:
    def test_particle_comes_first(self, jpsi2etappbar_decay: ThreeBodyDecay):
        pairs = get_conjugate_chain_pairs(jpsi2etappbar_decay)
        assert [(a.resonance.name, b.resonance.name) for a, b in pairs] == [
            ("N(1535)+", "N(1535)~-"),
            ("N(1710)+", "N(1710)~-"),
        ]
        for chain, conjugate_chain in pairs:
            assert chain.spectator.index != conjugate_chain.spectator.index
            assert chain.incoming_ls == conjugate_chain.incoming_ls
            assert chain.outgoing_ls == conjugate_chain.outgoing_ls

    def test_self_conjugate_chains_are_not_paired(
        self, jpsi2pipipi_decay: ThreeBodyDecay
    ):
        pairs = get_conjugate_chain_pairs(jpsi2pipipi_decay)
        assert [(a.resonance.name, b.resonance.name) for a, b in pairs] == [
            ("rho(770)+", "rho(770)-")
        ]

    def test_warns_if_a_partner_is_missing(self, jpsi2etappbar_decay: ThreeBodyDecay):
        decay = type(jpsi2etappbar_decay)(
            jpsi2etappbar_decay.states,
            [c for c in jpsi2etappbar_decay.chains if c.spectator.index != 2],
        )
        with pytest.warns(UserWarning, match="has no charge-conjugate partner"):
            assert get_conjugate_chain_pairs(decay) == []


class TestGetCForbiddenChains:
    def test_selection_rule(self, jpsi2pipipi_decay: ThreeBodyDecay):
        """:math:`X\\to\\pi^+\\pi^-` recoiling against a :math:`\\pi^0` needs
        :math:`C_X = C_\\psi C_{\\pi^0} = -1`, which excludes the :math:`f_2(1270)` and
        allows the :math:`\\rho^0`."""
        forbidden = get_c_forbidden_chains(jpsi2pipipi_decay)
        assert [c.resonance.name for c in forbidden] == ["f(2)(1270)"]

    def test_agrees_with_qrules(self, jpsi2pipipi_reaction: ReactionInfo):
        """QRules only generates the :math:`f_2(1270)` chain because the fixture allows
        C-violating interactions; restricting to the strong interaction removes exactly
        the chain that the selection rule forbids."""
        import qrules  # ruff: ignore[import-outside-top-level]

        strong_reaction = qrules.generate_transitions(
            initial_state="J/psi(1S)",
            final_state=["pi0", "pi-", "pi+"],
            allowed_intermediate_particles=["rho(770)", "f(2)(1270)"],
            allowed_interaction_types="strong",
            mass_conservation_factor=0,
            formalism="canonical-helicity",
        )
        transitions = normalize_state_ids(strong_reaction.transitions)
        decay = to_three_body_decay(transitions, min_ls=True)
        assert {c.resonance.name for c in decay.chains} == {
            "rho(770)+",
            "rho(770)-",
            "rho(770)0",
        }
        assert get_c_forbidden_chains(decay) == []

    def test_none_for_a_c_conserving_model(self, jpsi2etappbar_decay: ThreeBodyDecay):
        assert get_c_forbidden_chains(jpsi2etappbar_decay) == []


class TestRelateConjugateCouplings:
    @pytest.mark.parametrize("min_ls", [False, True])
    def test_production_carries_the_sign(
        self, jpsi2etappbar_decay: ThreeBodyDecay, min_ls: bool
    ):
        model = DalitzPlotDecompositionBuilder(
            jpsi2etappbar_decay, min_ls=min_ls
        ).formulate(reference_subsystem=2)
        substitutions = relate_conjugate_couplings(model)
        assert substitutions
        expected_signs = {
            conjugate_chain.resonance.latex: get_conjugate_coupling_sign(
                chain, "helicity" if min_ls else "LS"
            )
            for chain, conjugate_chain in get_conjugate_chain_pairs(jpsi2etappbar_decay)
        }
        assert sorted(expected_signs.values()) == ([-1, -1] if min_ls else [-1, +1])
        for symbol, expression in substitutions.items():
            latex = next(r for r in expected_signs if r in str(symbol))
            sign = -1 if expression.could_extract_minus_sign() else +1
            if "production" in str(symbol.base):
                assert sign == expected_signs[latex], symbol
            else:
                assert sign == +1, symbol

    def test_helicity_indices_of_the_decay_node_are_swapped(
        self, jpsi2etappbar_decay: ThreeBodyDecay
    ):
        model = DalitzPlotDecompositionBuilder(
            jpsi2etappbar_decay, min_ls=True
        ).formulate(reference_subsystem=2)
        substitutions = relate_conjugate_couplings(model)
        decay_couplings = {
            symbol: expression
            for symbol, expression in substitutions.items()
            if "decay" in str(symbol.base) and not symbol.indices[1].free_symbols
        }
        assert decay_couplings
        for symbol, expression in decay_couplings.items():
            target = next(
                node
                for node in sp.preorder_traversal(expression)
                if isinstance(node, sp.Indexed)
            )
            assert target.indices[1:] == symbol.indices[1:][::-1]

    def test_ls_indices_are_not_swapped(self, jpsi2etappbar_decay: ThreeBodyDecay):
        model = DalitzPlotDecompositionBuilder(
            jpsi2etappbar_decay, min_ls=False
        ).formulate(reference_subsystem=2)
        for symbol, expression in relate_conjugate_couplings(model).items():
            target = next(
                node
                for node in sp.preorder_traversal(expression)
                if isinstance(node, sp.Indexed)
            )
            assert target.indices[1:] == symbol.indices[1:]

    @pytest.mark.parametrize("min_ls", [False, True, (True, False), (False, True)])
    @pytest.mark.parametrize("use_coefficients", [False, True])
    def test_all_conjugate_couplings_are_eliminated(
        self,
        jpsi2etappbar_decay: ThreeBodyDecay,
        min_ls: bool | tuple[bool, bool],
        use_coefficients: bool,
    ):
        model = DalitzPlotDecompositionBuilder(
            jpsi2etappbar_decay, min_ls=min_ls
        ).formulate(reference_subsystem=2, use_coefficients=use_coefficients)
        tied = symmetrize_conjugate_couplings(model)
        assert len(tied.parameter_defaults) < len(model.parameter_defaults)
        remaining = {
            str(node)
            for expression in tied.amplitudes.values()
            for node in sp.preorder_traversal(expression.doit())
            if isinstance(node, sp.Indexed)
        }
        assert not [s for s in remaining if R"\overline{N}" in s]
        assert not [
            str(s) for s in tied.parameter_defaults if R"\overline{N}" in str(s)
        ]

    def test_raises_if_the_decay_is_not_c_symmetric(
        self, jpsi2pksigma_reaction: ReactionInfo
    ):
        transitions = normalize_state_ids(jpsi2pksigma_reaction.transitions)
        decay = to_three_body_decay(transitions, min_ls=True)
        model = DalitzPlotDecompositionBuilder(decay, min_ls=True).formulate(
            reference_subsystem=2
        )
        with pytest.raises(
            ValueError, match="does not map the final state onto itself"
        ):
            symmetrize_conjugate_couplings(model)


class TestMirrorSymmetry:
    """Tying the conjugate couplings has to make the intensity mirror-symmetric.

    Charge conjugation relabels the final state without touching any momentum, so the
    intensity of a C-conserving model is invariant under the induced permutation of the
    Mandelstam variables. This does **not** fix the sign of the tie (both signs give a
    mirror-symmetric intensity, they are the two eigenstates of the tie), but it does
    check that the couplings are related to the right partner, with the right helicity
    indices.
    """

    @pytest.mark.parametrize("min_ls", [False, True])
    def test_tying_restores_mirror_symmetry(
        self, jpsi2etappbar_reaction: ReactionInfo, min_ls: bool
    ):
        transitions = normalize_state_ids(jpsi2etappbar_reaction.transitions)
        decay = to_three_body_decay(transitions, min_ls=True)
        decay = type(decay)(
            decay.states,
            [c for c in decay.chains if "N(1535)" in c.resonance.name],
        )
        state_map = get_conjugate_state_map(decay)
        model = DalitzPlotDecompositionBuilder(decay, min_ls=min_ls).formulate(
            reference_subsystem=2
        )
        tied = symmetrize_conjugate_couplings(model)
        assert _compute_mirror_asymmetry(tied, state_map) < 1e-12

    def test_untied_couplings_break_mirror_symmetry(
        self, jpsi2etappbar_reaction: ReactionInfo
    ):
        """Control: without the tie, the two subsystems are independent."""
        transitions = normalize_state_ids(jpsi2etappbar_reaction.transitions)
        decay = to_three_body_decay(transitions, min_ls=True)
        state_map = get_conjugate_state_map(decay)
        model = DalitzPlotDecompositionBuilder(decay, min_ls=True).formulate(
            reference_subsystem=2
        )
        assert _compute_mirror_asymmetry(model, state_map) > 1e-3

    @pytest.mark.xfail(
        reason=(
            "https://github.com/ComPWA/ampform-dpd/issues/202: in the LS basis, the"
            " Clebsch-Gordan factors are built from IsobarNode.child1/child2, which"
            " to_three_body_decay() sorts by final-state ID, while the isobar Wigner-d"
            " function uses the cyclic pair ordering of get_decay_product_ids(). The two"
            " differ for subsystem 2, which gives that subsystem an extra exchange phase."
            " The derived sign is stated in the cyclic ordering, so the tie comes out"
            " with the wrong relative sign between waves of different l. Reordering the"
            " decay nodes cyclically makes this test pass; the helicity basis is"
            " unaffected, see test_mirror_symmetry_is_restored_in_the_helicity_basis."
        ),
        strict=True,
    )
    def test_tying_restores_mirror_symmetry_for_different_waves(
        self, jpsi2etappbar_reaction: ReactionInfo
    ):
        transitions = normalize_state_ids(jpsi2etappbar_reaction.transitions)
        decay = to_three_body_decay(transitions, min_ls=True)
        state_map = get_conjugate_state_map(decay)
        model = DalitzPlotDecompositionBuilder(decay, min_ls=False).formulate(
            reference_subsystem=2
        )
        tied = symmetrize_conjugate_couplings(model)
        assert _compute_mirror_asymmetry(tied, state_map) < 1e-12

    def test_mirror_symmetry_is_restored_in_the_helicity_basis(
        self, jpsi2etappbar_reaction: ReactionInfo
    ):
        """The same model in the helicity basis is unaffected by the ordering problem of
        :meth:`test_tying_restores_mirror_symmetry_for_different_waves`, because helicity
        couplings do not involve Clebsch-Gordan factors."""
        transitions = normalize_state_ids(jpsi2etappbar_reaction.transitions)
        decay = to_three_body_decay(transitions, min_ls=True)
        state_map = get_conjugate_state_map(decay)
        model = DalitzPlotDecompositionBuilder(decay, min_ls=True).formulate(
            reference_subsystem=2
        )
        tied = symmetrize_conjugate_couplings(model)
        assert _compute_mirror_asymmetry(tied, state_map) < 1e-12
