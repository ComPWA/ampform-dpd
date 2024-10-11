from __future__ import annotations

import itertools
import logging
from collections import abc, defaultdict
from functools import singledispatch
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar, overload

import attrs
import qrules
from qrules.quantum_numbers import InteractionProperties
from qrules.topology import EdgeType, FrozenTransition, NodeType
from qrules.transition import ReactionInfo, StateTransition, Topology

from ampform_dpd.decay import (
    FinalStateID,
    IsobarNode,
    LSCoupling,
    Particle,
    State,
    StateIDTemplate,
    ThreeBodyDecay,
    ThreeBodyDecayChain,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

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
    transitions = convert_transitions(transitions)
    if min_ls:
        transitions = filter_min_ls(transitions)
    some_transition = transitions[0]
    (initial_state_id, initial_state), *_ = some_transition.initial_states.items()
    outer_states = (
        _to_state(initial_state, index=initial_state_id),  # type:ignore[type-var]
        *[
            _to_state(particle, index=idx)  # type:ignore[type-var]
            for idx, particle in some_transition.final_states.items()
        ],
    )
    return ThreeBodyDecay(
        states={state.index: state for state in outer_states},  # type:ignore[misc]
        chains=tuple(sorted(_to_decay_chain(t) for t in transitions)),
    )


def _to_decay_chain(
    transition: FrozenTransition[Particle | State, LSCoupling | None],
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
    parent, *_ = transition.initial_states.values()
    spectator_id, resonance_id = sorted(topology.get_edge_ids_outgoing_from_node(0))
    child1_id, child2_id = sorted(topology.get_edge_ids_outgoing_from_node(1))
    resonance_id, *_ = sorted(topology.get_edge_ids_ingoing_to_node(1))
    production_node, decay_node = transition.interactions.values()
    return ThreeBodyDecayChain(
        decay=IsobarNode(
            parent=parent,
            child1=IsobarNode(
                parent=transition.states[resonance_id],
                child1=transition.states[child1_id],  # type:ignore[arg-type]
                child2=transition.states[child2_id],  # type:ignore[arg-type]
                interaction=decay_node,
            ),
            child2=transition.states[spectator_id],  # type:ignore[arg-type]
            interaction=production_node,
        )
    )


def convert_transitions(
    transitions: Iterable[FrozenTransition],
) -> tuple[FrozenTransition[Particle | State, LSCoupling | None], ...]:
    unique_transitions = {_convert_transition(t) for t in transitions}
    return tuple(sorted(unique_transitions))


def _convert_transition(
    transition: FrozenTransition,
) -> FrozenTransition[Particle | State, LSCoupling | None]:
    return FrozenTransition(
        transition.topology,
        states={
            index: _to_particle(state)
            if index in transition.intermediate_states
            else _to_state(state, index=index)  # type:ignore[type-var]
            for index, state in transition.states.items()
        },
        interactions={
            i: _to_ls_coupling(interaction)
            for i, interaction in transition.interactions.items()
        },
    )


def _to_particle(
    particle: qrules.particle.Particle | qrules.transition.State,
) -> Particle:
    if isinstance(particle, qrules.transition.State):
        particle = particle.particle
    return Particle(
        name=particle.name,
        latex=particle.name if particle.latex is None else particle.latex,
        spin=particle.spin,
        parity=int(particle.parity),  # type:ignore[arg-type]
        mass=particle.mass,
        width=particle.width,
    )


def _to_state(obj: Any, index: StateIDTemplate | None = None):
    if isinstance(obj, qrules.transition.State):
        obj = obj.particle
    if isinstance(obj, State):
        index = obj.index
    if index is None:
        msg = f"Cannot create a {State} from a {type(obj)} without an index"
        raise ValueError(msg)
    if not isinstance(obj, Particle) and not isinstance(obj, qrules.particle.Particle):
        msg = f"Cannot convert object of type {type(obj)} to a {State}"
        raise NotImplementedError(msg)
    return State(
        name=obj.name,
        latex=obj.name if obj.latex is None else obj.latex,  # pyright:ignore[reportUnnecessaryComparison]
        spin=obj.spin,
        parity=int(obj.parity),  # type:ignore[arg-type]
        mass=obj.mass,
        width=obj.width,
        index=index,
    )


def _to_ls_coupling(node: Any) -> LSCoupling | None:
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
        key = tuple(
            (state, _get_decay_product_ids(transition.topology, resonance_id))
            for resonance_id, state in transition.intermediate_states.items()
        )
        grouped_transitions[key].append(transition)
    min_transitions = []
    for group in grouped_transitions.values():
        transition, *_ = group
        min_transition: FrozenTransition[EdgeType, NodeType] = FrozenTransition(
            topology=transition.topology,
            states=transition.states,
            interactions={
                i: None
                if any(t.interactions[i] is None for t in group)
                else min(t.interactions[i] for t in group)  # type:ignore[type-var]
                for i in transition.interactions
            },
        )
        min_transitions.append(min_transition)
    return tuple(min_transitions)


def _get_decay_product_ids(topology: Topology, resonance_id: int) -> tuple[int, ...]:
    node_id = topology.edges[resonance_id].ending_node_id
    if node_id is None:
        msg = f"Resonance graph edge {resonance_id} has no ending node"
        raise ValueError(msg)
    return tuple(sorted(topology.get_originating_final_state_edge_ids(node_id)))


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
def _impl_normalize_state_ids(obj) -> NoReturn:
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


@overload
def permute_equal_final_states(obj: ReactionInfo) -> ReactionInfo: ...
@overload
def permute_equal_final_states(
    obj: Iterable[FrozenTransition[EdgeType, NodeType]],
) -> list[FrozenTransition[EdgeType, NodeType]]: ...
@overload
def permute_equal_final_states(
    obj: FrozenTransition[EdgeType, NodeType],
) -> list[FrozenTransition[EdgeType, NodeType]]: ...
def permute_equal_final_states(obj: T) -> T:  # type:ignore[misc]  # pyright:ignore[reportInconsistentOverload]
    return _impl_permute_equal_final_states(obj)


@singledispatch
def _impl_permute_equal_final_states(obj) -> NoReturn:
    msg = f"Cannot permute equal final states of a {type(obj)}"
    raise NotImplementedError(msg)


@_impl_permute_equal_final_states.register(ReactionInfo)
def _(obj: ReactionInfo) -> ReactionInfo:
    return ReactionInfo(
        transitions=permute_equal_final_states(obj.transitions),
        formalism=obj.formalism,
    )


@_impl_permute_equal_final_states.register(abc.Iterable)
def _(
    obj: Iterable[FrozenTransition[EdgeType, NodeType]],
) -> list[FrozenTransition[EdgeType, NodeType]]:
    permuted_transitions = []
    for transition in obj:
        permuted_transitions.extend(permute_equal_final_states(transition))
    return permuted_transitions


@_impl_permute_equal_final_states.register(FrozenTransition)
def _(
    obj: FrozenTransition[EdgeType, NodeType],
) -> list[FrozenTransition[EdgeType, NodeType]]:
    transition = obj
    equal_state_ids = _get_equal_final_state_ids(transition)
    if not equal_state_ids:
        return [transition]
    unique_permutations = {transition} | {
        attrs.evolve(transition, topology=transition.topology.swap_edges(i, j))
        for i, j in itertools.combinations(equal_state_ids, 2)
    }
    return sorted(unique_permutations)


def _get_equal_final_state_ids(
    transition: FrozenTransition,
) -> (
    tuple[()]
    | tuple[FinalStateID, FinalStateID]
    | tuple[FinalStateID, FinalStateID, FinalStateID]
):
    particle_to_id = defaultdict(list)
    for idx, state in transition.final_states.items():
        key = _uniqueness_repr(state)
        particle_to_id[key].append(idx)
    all_equal_state_ids = [set(ids) for ids in particle_to_id.values() if len(ids) > 1]
    if not all_equal_state_ids:
        return tuple()  # type:ignore[return-value]
    return tuple(sorted(all_equal_state_ids[0]))  # type:ignore[return-value]


def _uniqueness_repr(obj: Any) -> str:
    if isinstance(obj, qrules.transition.State):
        return _uniqueness_repr(obj.particle)
    if isinstance(obj, (Particle, State, qrules.particle.Particle)):
        return obj.name
    msg = f"Cannot create a uniqueness key for {type(obj)}"
    raise NotImplementedError(msg)
