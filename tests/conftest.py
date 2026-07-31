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


@pytest.fixture(scope="session")
def jpsi2etappbar_reaction() -> ReactionInfo:
    """:math:`J/\\psi \\to \\eta\\, p\\, \\bar p` with a :math:`\\tfrac12^-` and a
    :math:`\\tfrac12^+` resonance, so that charge conjugation ties the two :math:`N^*`
    subsystems with opposite signs."""
    return qrules.generate_transitions(
        initial_state="J/psi(1S)",
        final_state=["eta", "p", "p~"],
        allowed_interaction_types="strong",
        allowed_intermediate_particles=["N(1535)", "N(1710)"],
        formalism="canonical-helicity",
        mass_conservation_factor=0,
    )


@pytest.fixture(scope="session")
def jpsi2pipipi_reaction() -> ReactionInfo:
    """:math:`J/\\psi \\to \\pi^0\\pi^-\\pi^+`, allowing C-violating transitions.

    The :math:`\\rho^\\pm` chains form a conjugate pair, while the :math:`\\rho^0` and
    :math:`f_2(1270)` chains are mapped onto themselves and are therefore subject to a
    selection rule instead.
    """
    return qrules.generate_transitions(
        initial_state="J/psi(1S)",
        final_state=["pi0", "pi-", "pi+"],
        allowed_intermediate_particles=["rho(770)", "f(2)(1270)"],
        mass_conservation_factor=0,
        formalism="canonical-helicity",
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
