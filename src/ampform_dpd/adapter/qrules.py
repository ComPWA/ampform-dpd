from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from qrules.quantum_numbers import InteractionProperties
from qrules.topology import EdgeType, FrozenTransition, NodeType
from qrules.transition import State

from ampform_dpd.decay import (
    IsobarNode,
    LSCoupling,
    Particle,
    ThreeBodyDecay,
    ThreeBodyDecayChain,
)


def to_three_body_decay(
    transitions: Iterable[FrozenTransition[Particle, LSCoupling]],
) -> ThreeBodyDecay:
    transitions = tuple(transitions)
    if not transitions:
        msg = "Need at least one transition object"
        raise ValueError(msg)
    some_transition = transitions[0]
    return ThreeBodyDecay(
        states={**some_transition.initial_states, **some_transition.final_states},
        chains=tuple(sorted(to_decay_chain(t) for t in transitions)),
    )


def to_decay_chain(
    transition: FrozenTransition[Particle, LSCoupling],
) -> ThreeBodyDecayChain:
    if len(transition.initial_states) != 1:
        msg = f"Can only handle one initial state, but got {len(transition.initial_states)}"
        raise ValueError(msg)
    if len(transition.final_states) != 3:  # noqa: PLR2004
        msg = f"Can only handle three final states, but got {len(transition.final_states)}"
        raise ValueError(msg)
    if len(transition.interactions) != 2:  # noqa: PLR2004
        msg = f"There are {len(transition.interactions)} interaction nodes, so this can't be a three-body decay"
        raise ValueError(msg)
    topology = transition.topology
    spectator_id, resonance_id = sorted(topology.get_edge_ids_outgoing_from_node(0))
    resonance_id, *_ = sorted(topology.get_edge_ids_ingoing_to_node(1))
    child1_id, child2_id = sorted(topology.get_edge_ids_outgoing_from_node(1))
    parent, *_ = transition.initial_states.values()
    production_node, decay_node = transition.interactions.values()
    isobar = IsobarNode(
        parent=parent,
        child1=IsobarNode(
            parent=transition.states[resonance_id],
            child1=transition.states[child1_id],
            child2=transition.states[child2_id],
            interaction=decay_node,
        ),
        child2=transition.states[spectator_id],
        interaction=production_node,
    )
    return ThreeBodyDecayChain(decay=isobar)


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
