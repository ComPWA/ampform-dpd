from __future__ import annotations

import logging
from collections import abc, defaultdict
from functools import singledispatch
from pathlib import Path
from typing import Any, Iterable, TypeVar, overload

import attrs
import qrules
from qrules.quantum_numbers import InteractionProperties
from qrules.topology import EdgeType, FrozenTransition, NodeType
from qrules.transition import ReactionInfo, State, StateTransition, Topology

from ampform_dpd.decay import (
    IsobarNode,
    LSCoupling,
    Particle,
    ThreeBodyDecay,
    ThreeBodyDecayChain,
)

_LOGGER = logging.getLogger(__name__)


def to_three_body_decay(
    transitions: Iterable[FrozenTransition],
    min_ls: bool = False,
) -> ThreeBodyDecay:
    transitions = tuple(transitions)
    if not transitions:
        msg = "Need at least one transition object"
        raise ValueError(msg)
    if set(transitions[0].initial_states) != {0} or set(
        transitions[0].final_states
    ) != {1, 2, 3}:
        transitions = normalize_state_ids(transitions)
        _LOGGER.warning("Relabeled initial state to 0 and final states to 1, 2, 3")
    transitions = convert_edges_and_nodes(transitions)
    if min_ls:
        transitions = filter_min_ls(transitions)
    some_transition = transitions[0]
    initial_state, *_ = some_transition.initial_states.values()
    final_states = {
        i: some_transition.final_states[idx]
        for i, idx in enumerate(sorted(some_transition.final_states), 1)
    }
    return ThreeBodyDecay(
        states={0: initial_state, **final_states},  # type:ignore[dict-item]
        chains=tuple(sorted(to_decay_chain(t) for t in transitions)),
    )


def to_decay_chain(
    transition: FrozenTransition[Particle, LSCoupling | None],
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
) -> tuple[FrozenTransition[Particle, LSCoupling | None], ...]:
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
        latex=particle.name if particle.latex is None else particle.latex,
        spin=particle.spin,
        parity=int(particle.parity),  # type:ignore[arg-type]
        mass=particle.mass,
        width=particle.width,
    )


def _convert_node(node: Any) -> LSCoupling | None:
    if node is None:
        return None
    if isinstance(node, LSCoupling):
        return node
    if not isinstance(node, InteractionProperties):
        msg = f"Cannot convert node of type {type(node)}"
        raise NotImplementedError(msg)
    if node.l_magnitude is None or node.s_magnitude is None:
        return None
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
        min_transition: FrozenTransition[EdgeType, NodeType] = FrozenTransition(
            topology=transition.topology,
            states=transition.states,
            interactions={
                i: min(t.interactions[i] for t in group)  # type:ignore[type-var]
                for i in transition.interactions
            },
        )
        min_transitions.append(min_transition)
    return tuple(min_transitions)


def load_particles() -> qrules.particle.ParticleCollection:
    src_dir = Path(__file__).parent.parent
    particle_database = qrules.load_default_particles()
    additional_definitions = qrules.io.load(src_dir / "particle-definitions.yml")  # type:ignore[arg-type]
    particle_database.update(additional_definitions)  # type:ignore[arg-type]
    return particle_database


@overload
def normalize_state_ids(obj: T) -> T: ...
@overload
def normalize_state_ids(obj: Iterable[T]) -> list[T]: ...
def normalize_state_ids(obj):  # pyright:ignore[reportInconsistentOverload]
    """Relabel the state IDs so that they lie in the range :math:`[0, N)`."""
    return _impl_normalize_state_ids(obj)


@singledispatch
def _impl_normalize_state_ids(obj):
    """Relabel the state IDs so that they lie in the range :math:`[0, N)`."""
    msg = f"Cannot relabel edge IDs of a {type(obj).__name__}"
    raise NotImplementedError(msg)


@_impl_normalize_state_ids.register(ReactionInfo)  # type:ignore[attr-defined]
def _(obj: ReactionInfo) -> ReactionInfo:
    return ReactionInfo(
        # no attrs.evolve() in order to call __attrs_post_init__()
        transitions=[_impl_normalize_state_ids(g) for g in obj.transitions],
        formalism=obj.formalism,
    )


@_impl_normalize_state_ids.register(FrozenTransition)  # type:ignore[attr-defined]
def _(obj: StateTransition) -> StateTransition:
    return attrs.evolve(
        obj,
        topology=_impl_normalize_state_ids(obj.topology),
        states={new: obj.states[old] for new, old in enumerate(sorted(obj.states))},
    )


@_impl_normalize_state_ids.register(Topology)  # type:ignore[attr-defined]
def _(obj: Topology) -> Topology:
    mapping = {old: new for new, old in enumerate(sorted(obj.edges))}
    return obj.relabel_edges(mapping)


@_impl_normalize_state_ids.register(abc.Iterable)  # type:ignore[attr-defined]
def _(obj: abc.Iterable[T]) -> list[T]:
    return [_impl_normalize_state_ids(x) for x in obj]


T = TypeVar("T", ReactionInfo, StateTransition, Topology)
"""Type variable for the input and output of :func:`normalize_state_ids`."""
