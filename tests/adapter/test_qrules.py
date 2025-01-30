# cspell:ignore pksigma
# pyright: reportPrivateUsage=false
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, SupportsFloat

import pytest

from ampform_dpd.adapter.qrules import (
    _convert_transition,
    _get_equal_final_state_ids,
    convert_transitions,
    filter_min_ls,
    normalize_state_ids,
    permute_equal_final_states,
    to_three_body_decay,
)
from ampform_dpd.decay import LSCoupling, Particle

if TYPE_CHECKING:
    from qrules.topology import FrozenTransition
    from qrules.transition import ReactionInfo, StateTransition


def test_convert_transitions(xib2pkk_reaction: ReactionInfo):
    reaction = normalize_state_ids(xib2pkk_reaction)
    assert reaction.get_intermediate_particles().names == ["Lambda(1520)"]
    assert len(reaction.transitions) == 16
    transitions = convert_transitions(reaction.transitions)
    assert len(transitions) == 2
    decay = to_three_body_decay(transitions, min_ls=True)
    assert len(decay.chains) == 2


def test_filter_min_ls(jpsi2pksigma_reaction: ReactionInfo):
    reaction = jpsi2pksigma_reaction
    transitions = tuple(
        t for t in reaction.transitions if t.states[3].spin_projection == +0.5
    )

    ls_couplings = [_get_couplings(t) for t in transitions]
    if reaction.formalism == "canonical-helicity":
        assert len(ls_couplings) == 3
        assert ls_couplings == [
            (
                {"L": 0, "S": 1},
                {"L": 1, "S": 0.5},
            ),
            (
                {"L": 2, "S": 1},
                {"L": 1, "S": 0.5},
            ),
            (
                {"L": 1, "S": 2},
                {"L": 2, "S": 0.5},
            ),
        ]
    else:
        assert len(ls_couplings) == 2
        for ls_coupling in ls_couplings:
            for ls in ls_coupling:
                assert ls == {"L": None, "S": None}

    min_ls_transitions = filter_min_ls(transitions)
    ls_couplings = [_get_couplings(t) for t in min_ls_transitions]
    assert len(ls_couplings) == 2
    if reaction.formalism == "canonical-helicity":
        assert ls_couplings == [
            (
                {"L": 0, "S": 1},
                {"L": 1, "S": 0.5},
            ),
            (
                {"L": 1, "S": 2},
                {"L": 2, "S": 0.5},
            ),
        ]


@pytest.mark.parametrize("converter", [lambda x: x, _convert_transition])
def test_get_equal_final_state_ids(
    a2pipipi_reaction: ReactionInfo,
    jpsi2pksigma_reaction: ReactionInfo,
    xib2pkk_reaction: ReactionInfo,
    converter: Callable[[FrozenTransition], FrozenTransition],
):
    test_cases = [
        (a2pipipi_reaction, (1, 2, 3)),
        (jpsi2pksigma_reaction, tuple()),
        (xib2pkk_reaction, (2, 3)),
    ]
    for reaction012, expected in test_cases:
        reaction = normalize_state_ids(reaction012)
        transition = converter(reaction.transitions[0])
        equal_ids = _get_equal_final_state_ids(transition)
        assert equal_ids == expected


def test_normalize_state_ids_reaction(jpsi2pksigma_reaction: ReactionInfo):
    reaction012 = jpsi2pksigma_reaction
    reaction123 = normalize_state_ids(reaction012)
    assert set(reaction123.initial_state) == {0}
    assert set(reaction123.final_state) == {1, 2, 3}

    transitions123 = normalize_state_ids(reaction012.transitions)
    for transition012, transition123 in zip(reaction012.transitions, transitions123):
        assert set(transition123.initial_states) == {0}
        assert set(transition123.final_states) == {1, 2, 3}
        assert set(transition123.intermediate_states) == {4}

        topology123 = normalize_state_ids(transition123.topology)
        assert topology123.incoming_edge_ids == {0}
        assert topology123.outgoing_edge_ids == {1, 2, 3}
        assert topology123.intermediate_edge_ids == {4}

        for i in transition012.states:
            assert transition012.states[i] == transition123.states[i + 1]


def test_permute_equal_final_states(
    a2pipipi_reaction: ReactionInfo,
    jpsi2pksigma_reaction: ReactionInfo,
    xib2pkk_reaction: ReactionInfo,
):
    test_cases = [
        (1, jpsi2pksigma_reaction),
        (2, xib2pkk_reaction),
        (3, a2pipipi_reaction),
    ]
    for n_permutations, reaction012 in test_cases:
        reaction = normalize_state_ids(reaction012)
        transition = reaction.transitions[0]
        permutations = permute_equal_final_states(transition)
        assert len(permutations) == n_permutations

        permuted_reaction = permute_equal_final_states(reaction)
        n_transitions = len(permuted_reaction.transitions)
        assert n_transitions == n_permutations * len(reaction.transitions)


@pytest.mark.parametrize("min_ls", [False, True])
def test_to_three_body_decay(jpsi2pksigma_reaction: ReactionInfo, min_ls: bool):
    reaction = normalize_state_ids(jpsi2pksigma_reaction)
    decay = to_three_body_decay(reaction.transitions, min_ls)
    assert decay.initial_state.name == "J/psi(1S)"
    assert {i: p.name for i, p in decay.final_state.items()} == {
        1: "K0",
        2: "Sigma+",
        3: "p~",
    }
    n_chains = len(decay.chains)
    if reaction.formalism == "canonical-helicity":
        production_ls = [c.incoming_ls for c in decay.chains]
        decay_ls = [c.outgoing_ls for c in decay.chains]
        if min_ls:
            assert n_chains == 2
            assert production_ls == [
                LSCoupling(L=1, S=1),
                LSCoupling(L=0, S=1),
            ]
            assert decay_ls == [
                LSCoupling(L=2, S=0.5),
                LSCoupling(L=1, S=0.5),
            ]
        else:
            assert n_chains == 4
            assert production_ls == [
                LSCoupling(L=1, S=1),
                LSCoupling(L=1, S=2),
                LSCoupling(L=0, S=1),
                LSCoupling(L=2, S=1),
            ]
            assert decay_ls == [
                LSCoupling(L=2, S=0.5),
                LSCoupling(L=2, S=0.5),
                LSCoupling(L=1, S=0.5),
                LSCoupling(L=1, S=0.5),
            ]
    elif reaction.formalism == "helicity":
        assert n_chains == 2
        for chain in decay.chains:
            assert chain.incoming_ls is None
            assert chain.outgoing_ls is None
    resonance_names = set()
    for chain in decay.chains:
        assert isinstance(chain.resonance, Particle)
        resonance_names.add(chain.resonance.name)
    assert resonance_names == {
        "N(1700)+",
        "Sigma(1660)~-",
    }


def _get_couplings(transition: StateTransition) -> tuple[dict, dict]:
    return tuple(  # type:ignore[return-value]
        {"L": _to_float(node.l_magnitude), "S": _to_float(node.s_magnitude)}
        for node in transition.interactions.values()
    )


def _to_float(value: SupportsFloat | None) -> float | int | None:
    if value is None:
        return None
    value = float(value)
    if value.is_integer():
        return int(value)
    return value
