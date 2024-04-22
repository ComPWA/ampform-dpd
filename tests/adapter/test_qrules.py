from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import qrules

from ampform_dpd.adapter.qrules import filter_min_ls

if TYPE_CHECKING:
    from qrules.transition import StateTransition


def test_filter_min_ls():
    reaction = qrules.generate_transitions(
        initial_state=[("J/psi(1S)", [+1])],
        final_state=["K0", ("Sigma+", [+0.5]), ("p~", [+0.5])],
        allowed_interaction_types="strong",
        allowed_intermediate_particles=["Sigma(1660)"],
        formalism="canonical-helicity",
    )
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


def _get_couplings(transition: StateTransition) -> tuple[LSCouplings, LSCouplings]:
    return tuple(
        {"L": node.l_magnitude, "S": node.s_magnitude}
        for node in transition.interactions.values()
    )


class LSCouplings(TypedDict):
    L: int
    S: float
