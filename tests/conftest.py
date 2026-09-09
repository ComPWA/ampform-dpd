from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
import pytest
import qrules

if TYPE_CHECKING:
    from _pytest.fixtures import SubRequest
    from qrules.transition import ReactionInfo


@pytest.fixture(scope="session")
def a2pipipi_reaction() -> ReactionInfo:
    return qrules.generate_transitions(
        initial_state="a(1)(1260)0",
        final_state=["pi0", "pi0", "pi0"],
        allowed_intermediate_particles=["a(0)(980)0"],
        formalism="helicity",
    )


@pytest.fixture(scope="session", params=["canonical-helicity", "helicity"])
def jpsi2pksigma_reaction(request: SubRequest) -> ReactionInfo:
    return qrules.generate_transitions(
        initial_state=[("J/psi(1S)", [+1])],
        final_state=["K0", ("Sigma+", [+0.5]), ("p~", [+0.5])],
        allowed_interaction_types="strong",
        allowed_intermediate_particles=["N(1700)+", "Sigma(1660)"],
        formalism=request.param,
    )


@pytest.fixture(scope="session")
def xib2pkk_reaction() -> ReactionInfo:
    reaction = qrules.generate_transitions(
        initial_state="Xi(b)-",
        final_state=["p", "K-", "K-"],
        allowed_intermediate_particles=["Lambda(1520)"],
        formalism="helicity",
    )
    swapped_transitions = tuple(
        attrs.evolve(t, topology=t.topology.swap_edges(1, 2))
        for t in reaction.transitions
    )
    return qrules.transition.ReactionInfo(
        transitions=reaction.transitions + swapped_transitions,
        formalism=reaction.formalism,
    )
