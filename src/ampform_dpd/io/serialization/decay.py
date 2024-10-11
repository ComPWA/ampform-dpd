from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import sympy as sp

from ampform_dpd.decay import (
    FinalStateID,
    IsobarNode,
    Particle,
    State,
    StateID,
    ThreeBodyDecay,
    ThreeBodyDecayChain,
)
from ampform_dpd.io.serialization.format import (
    Topology,
    get_decay_chains,
    get_distribution_def,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ampform_dpd.io.serialization.format import (
        DecayChain,
        ModelDefinition,
        StateDefinition,
        Vertex,
    )


def to_decay(
    model: ModelDefinition, to_latex: Callable[[str], str] | None = None
) -> ThreeBodyDecay:
    initial_state = get_initial_state(model, to_latex)
    final_state = get_final_state(model, to_latex)
    return ThreeBodyDecay(
        states=get_states(model, to_latex),
        chains=sorted({
            to_decay_chain(chain, initial_state, final_state, to_latex)
            for chain in get_decay_chains(model)
        }),
    )


def to_decay_chain(
    chain_definition: DecayChain,
    initial_state: State,
    final_state: dict[FinalStateID, State],
    to_latex: Callable[[str], str] | None = None,
) -> ThreeBodyDecayChain:
    vertices = chain_definition["vertices"]
    if to_latex is None:
        to_latex = lambda x: x  # noqa:E731
    resonance = Particle(
        name=chain_definition["name"],
        latex=to_latex(chain_definition["name"]),
        spin=sp.Rational(chain_definition["propagators"][0]["spin"]),
        mass=0,
        width=0,
        parity=None,
    )
    child1_id, child2_id = __find_decay_product_ids(vertices)
    spectator_id = __find_spectator_id(vertices)
    return ThreeBodyDecayChain(
        decay=IsobarNode(
            parent=initial_state,
            child1=IsobarNode(
                parent=resonance,
                child1=final_state[child1_id],
                child2=final_state[child2_id],
            ),
            child2=final_state[spectator_id],
        )
    )


def __find_spectator_id(vertices: Iterable[Vertex]) -> FinalStateID:
    for vertex in vertices:
        node = vertex["node"]
        if any(not isinstance(i, int) for i in node):
            for state_id in node:
                if isinstance(state_id, int):
                    return state_id
    msg = "Could not find a production node with a spectator"
    raise ValueError(msg)


def __find_decay_product_ids(
    vertices: Iterable[Vertex],
) -> tuple[FinalStateID, FinalStateID]:
    for vertex in vertices:
        node = vertex["node"]
        if all(isinstance(i, int) for i in node) and len(node) == 2:  # noqa: PLR2004
            return tuple(node)  # type:ignore[return-value]
    msg = "Could not find a node that has two final state items (decay node)"
    raise ValueError(msg)


def get_states(
    model: ModelDefinition, to_latex: Callable[[str], str] | None = None
) -> dict[StateID, State]:
    initial_state = get_initial_state(model, to_latex)
    final_state = get_final_state(model, to_latex)
    return {initial_state.index: initial_state, **final_state}  # type:ignore[dict-item]


def get_initial_state(
    model: ModelDefinition, to_latex: Callable[[str], str] | None = None
) -> State:
    distribution_def = get_distribution_def(model)
    decay_description = distribution_def["decay_description"]
    kinematics = decay_description["kinematics"]
    return _to_particle(kinematics["initial_state"], to_latex)


def get_final_state(
    model: ModelDefinition, to_latex: Callable[[str], str] | None = None
) -> dict[FinalStateID, State]:
    distribution_def = get_distribution_def(model)
    decay_description = distribution_def["decay_description"]
    kinematics = decay_description["kinematics"]
    final_state_def = kinematics["final_state"]
    final_state = [_to_particle(p, to_latex) for p in final_state_def]
    return {p.index: p for p in final_state}


def _to_particle(
    definition: StateDefinition, to_latex: Callable[[str], str] | None = None
) -> State:
    if to_latex is None:
        to_latex = lambda x: x  # noqa:E731
    return State(
        name=definition["name"],
        latex=to_latex(definition["name"]),
        mass=definition["mass"],
        width=0,
        spin=sp.Rational(definition["spin"]),
        parity=None,
        index=definition["index"],
    )


def get_spectator_id(topology: Topology) -> FinalStateID:
    """Get the spectator ID from a reference topology.

    >>> get_spectator_id([1, [2, 3]])
    1
    """
    spectator_candidates = {i for i in topology if isinstance(i, int)}
    if len(spectator_candidates) != 1:
        msg = f"Reference topology {topology} seems not to be a three-body decay"
        raise ValueError(msg)
    return next(iter(spectator_candidates))
