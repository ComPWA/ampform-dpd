from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from qrules.quantum_numbers import InteractionProperties
from qrules.topology import EdgeType, FrozenTransition, NodeType
from qrules.transition import State

from ampform_dpd.decay import LSCoupling, Particle


def convert_edges_and_nodes(
    transitions: Iterable[FrozenTransition],
) -> tuple[FrozenTransition[Particle, LSCoupling], ...]:
    unique_transitions = {
        transition.convert(
            state_converter=_convert_edge,
            interaction_converter=_convert_node,
        )
        for transition in transitions
    }
    return tuple(sorted(unique_transitions))


def _convert_edge(state: Any) -> Particle:
    if isinstance(state, Particle):
        return state
    if not isinstance(state, State):
        msg = f"Cannot convert state of type {type(state)}"
        raise NotImplementedError(msg)
    particle = state.particle
    if particle.parity is None:
        msg = f"Cannot convert particle {particle.name} with undefined parity"
        raise NotImplementedError(msg)
    return Particle(
        name=particle.name,
        latex=particle.latex,
        spin=particle.spin,
        parity=particle.parity,
        mass=particle.mass,
        width=particle.width,
    )


def _convert_node(node: Any) -> Particle:
    if isinstance(node, LSCoupling):
        return node
    if not isinstance(node, InteractionProperties):
        msg = f"Cannot convert node of type {type(node)}"
        raise NotImplementedError(msg)
    if node.l_magnitude is None or node.s_magnitude is None:
        msg = "Cannot convert node with undefined L or S"
        raise NotImplementedError(msg)
    return LSCoupling(
        L=node.l_magnitude,
        S=node.s_magnitude,
    )


def filter_min_ls(
    transitions: Iterable[FrozenTransition[EdgeType, NodeType]],
) -> tuple[FrozenTransition[EdgeType, NodeType], ...]:
    grouped_transitions = defaultdict(list)
    for transition in transitions:
        resonances = tuple(transition.intermediate_states.values())
        grouped_transitions[resonances].append(transition)
    min_transitions = []
    for group in grouped_transitions.values():
        transition, *_ = group
        min_transition = FrozenTransition(
            topology=transition.topology,
            states=transition.states,
            interactions={
                i: min(t.interactions[i] for t in group)
                for i in transition.interactions
            },
        )
        min_transitions.append(min_transition)
    return tuple(min_transitions)
