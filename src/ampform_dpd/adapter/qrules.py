from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from qrules.topology import EdgeType, FrozenTransition, NodeType


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
