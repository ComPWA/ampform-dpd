from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import qrules

from ampform_dpd.adapter.qrules import (
    filter_min_ls,
    normalize_state_ids,
    to_three_body_decay,
)
from ampform_dpd.decay import LSCoupling, Particle

if TYPE_CHECKING:
    from qrules.transition import ReactionInfo, StateTransition


@pytest.fixture(scope="session")
def reaction() -> ReactionInfo:
    return qrules.generate_transitions(
        initial_state=[("J/psi(1S)", [+1])],
        final_state=["K0", ("Sigma+", [+0.5]), ("p~", [+0.5])],
        allowed_interaction_types="strong",
        allowed_intermediate_particles=["Sigma(1660)"],
        formalism="canonical-helicity",
    )


def test_filter_min_ls(reaction: ReactionInfo):
    transitions = tuple(
        t for t in reaction.transitions if t.states[3].spin_projection == +0.5
    )

    ls_couplings = [_get_couplings(t) for t in transitions]
    assert ls_couplings == [
        (
            {"L": 0, "S": 1.0},
            {"L": 1, "S": 0.5},
        ),
        (
            {"L": 2, "S": 1.0},
            {"L": 1, "S": 0.5},
        ),
    ]

    min_ls_transitions = filter_min_ls(transitions)
    ls_couplings = [_get_couplings(t) for t in min_ls_transitions]
    assert ls_couplings == [
        (
            {"L": 0, "S": 1.0},
            {"L": 1, "S": 0.5},
        ),
    ]


def test_normalize_state_ids_reaction(reaction: ReactionInfo):
    reaction012 = reaction
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


@pytest.mark.parametrize("min_ls", [False, True])
def test_to_three_body_decay(reaction: ReactionInfo, min_ls: bool):
    decay = to_three_body_decay(reaction.transitions, min_ls)
    assert decay.initial_state.name == "J/psi(1S)"
    assert {i: p.name for i, p in decay.final_state.items()} == {
        1: "K0",
        2: "Sigma+",
        3: "p~",
    }
    if min_ls:
        assert len(decay.chains) == 1
        assert decay.chains[0].incoming_ls == LSCoupling(L=0, S=1)
        assert decay.chains[0].outgoing_ls == LSCoupling(L=1, S=0.5)
    else:
        assert len(decay.chains) == 2
        assert decay.chains[1].incoming_ls == LSCoupling(L=2, S=1)
        assert decay.chains[1].outgoing_ls == LSCoupling(L=1, S=0.5)
    for chain in decay.chains:
        assert isinstance(chain.resonance, Particle)
        assert chain.resonance.name == "Sigma(1660)~-"


def _get_couplings(transition: StateTransition) -> tuple[dict, dict]:
    return tuple(  # type:ignore[return-value]
        {"L": node.l_magnitude, "S": node.s_magnitude}
        for node in transition.interactions.values()
    )
