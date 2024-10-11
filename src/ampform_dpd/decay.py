"""Data structures that describe a three-body decay."""

from __future__ import annotations

from functools import cache
from textwrap import dedent
from typing import TYPE_CHECKING, Generic, Literal, TypeVar, overload
from warnings import warn

from attrs import field, frozen
from attrs.validators import instance_of

from ampform_dpd._attrs import assert_spin_value, to_chains, to_ls, to_rational

if TYPE_CHECKING:
    import sympy as sp

InitialStateID = Literal[0]
"""ID for the initial state particle in a three-body decay."""
FinalStateID = Literal[1, 2, 3]
"""ID for a particle in the final state of a three-body decay."""
StateID = Literal[0, 1, 2, 3]
"""ID for any of the initial state or final state particles in a three-body decay."""
StateIDTemplate = TypeVar("StateIDTemplate", InitialStateID, FinalStateID, StateID)
"""Generic template for the ID of a particle in a three-body decay."""


@frozen(order=True)
class Particle:
    name: str
    latex: str
    spin: sp.Rational = field(converter=to_rational, validator=assert_spin_value)
    parity: Literal[-1, 1] | None
    mass: float
    width: float


@frozen(order=True)
class State(Particle, Generic[StateIDTemplate]):
    """Initial or final state `.Particle` in a `ThreeBodyDecay`, carrying an index."""

    index: StateIDTemplate


InitialState = State[InitialStateID]
"""The initial state particle."""
FinalState = State[FinalStateID]
"""One of the final state particles."""
ParentType = TypeVar("ParentType", Particle, InitialState)
"""Type of the parent of an `IsobarNode`."""


@frozen(order=True)
class IsobarNode(Generic[ParentType]):
    parent: ParentType
    child1: IsobarNode[Particle] | FinalState
    child2: IsobarNode[Particle] | FinalState
    interaction: LSCoupling | None = field(default=None, converter=to_ls)

    @property
    def children(self) -> tuple[DecayNode | FinalState, DecayNode | FinalState]:
        return self.child1, self.child2


ProductionNode = IsobarNode[InitialState]
"""The first `IsobarNode` in a `ThreeBodyDecayChain`."""
DecayNode = IsobarNode[Particle]
"""The second `IsobarNode` in a `ThreeBodyDecayChain`."""


@frozen
class ThreeBodyDecay:
    states: dict[StateID, State[StateID]]
    chains: tuple[ThreeBodyDecayChain, ...] = field(converter=to_chains)

    def __attrs_post_init__(self) -> None:
        expected_initial_state = self.initial_state
        expected_final_state = set(self.final_state.values())
        for i, chain in enumerate(self.chains):
            if chain.parent != expected_initial_state:
                msg = dedent(f"""
                    Chain {i} has initial state
                      {chain.parent.index}: {chain.parent.name}
                    but should have
                      {expected_initial_state.index}: {expected_initial_state.name}
                """).strip()
                raise ValueError(msg)
            final_state = {chain.spectator, *chain.decay_products}
            if final_state != expected_final_state:

                def to_str(s: set[FinalState]) -> str:
                    return ", ".join(
                        f"{p.index}: {p.name}" for p in sorted(s, key=lambda x: x.index)
                    )

                msg = dedent(f"""
                    Chain {i} has final state
                       {to_str(final_state)}
                    but should have
                       {to_str(expected_final_state)}
                """).strip()
                raise ValueError(msg)

    @property
    def initial_state(self) -> InitialState:
        return self.states[0]  # type:ignore[return-value]

    @property
    def final_state(self) -> dict[FinalStateID, FinalState]:
        return {s.index: s for s in self.states.values() if s.index != 0}  # type:ignore[misc]

    def find_chain(self, resonance_name: str) -> ThreeBodyDecayChain:
        for chain in self.chains:
            if chain.resonance.name == resonance_name:
                return chain
        msg = f"No decay chain found for resonance {resonance_name}"
        raise KeyError(msg)

    def get_subsystem(self, subsystem_id: FinalStateID) -> ThreeBodyDecay:
        filtered_chains = [c for c in self.chains if c.spectator.index == subsystem_id]
        if not filtered_chains:
            decay_description = _get_decay_description(self)
            subsystems = ", ".join(sorted(str(i) for i in _get_subsystem_ids(self)))
            msg = f"Decay {decay_description} only has subsystems {subsystems}, not {subsystem_id}"
            warn(msg, category=UserWarning)
        return ThreeBodyDecay(self.states, filtered_chains)


def _get_decay_description(decay: ThreeBodyDecay) -> str:
    initial_state = decay.initial_state.name
    final_state = ", ".join(f"{i}: {s.name}" for i, s in decay.final_state.items())
    return f"{initial_state} → {final_state}"


def _get_subsystem_ids(decay: ThreeBodyDecay) -> set[FinalStateID]:
    return {c.spectator.index for c in decay.chains}


def get_decay_product_ids(
    spectator_id: FinalStateID,
) -> tuple[FinalStateID, FinalStateID]:
    if spectator_id == 1:
        return 2, 3
    if spectator_id == 2:  # noqa: PLR2004
        return 3, 1
    if spectator_id == 3:  # noqa: PLR2004
        return 1, 2
    msg = f"Spectator ID has to be one of 1, 2, 3, not {spectator_id}"
    raise ValueError(msg)


@frozen(order=True)
class ThreeBodyDecayChain:
    decay: ProductionNode = field(validator=instance_of(IsobarNode))

    def __attrs_post_init__(self) -> None:
        outer_states: list[State[StateID]] = [self.initial_state, *self.final_state]  # type:ignore[list-item]
        for state in outer_states:
            if not isinstance(state, State):
                msg = f"Not all particles in the initial or final state are not type {State.__name__}"
                raise TypeError(msg)
        if len({state.index for state in outer_states}) != 4:  # noqa: PLR2004
            msg = "The initial and/or final state contains particles with the same ID:"
            for state in outer_states:
                msg += f"\n  {state.index}: {state.name}"
            raise ValueError(msg)

    @property
    def initial_state(self) -> InitialState:
        return self.parent

    @property
    @cache  # noqa: B019
    def final_state(self) -> tuple[FinalState, FinalState, FinalState]:
        final_state = (*self.decay_products, self.spectator)
        return tuple(sorted(final_state, key=lambda x: x.index))  # type:ignore[return-value]

    @property
    def parent(self) -> InitialState:
        return self.decay.parent  # type:ignore[return-value]

    @property
    def resonance(self) -> Particle:
        return to_particle(self.decay_node)

    @property
    def production_node(self) -> ProductionNode:
        return self.decay

    @property
    def decay_node(self) -> DecayNode:
        return self._get_child_of_type(IsobarNode)

    @property
    def decay_products(self) -> tuple[FinalState, FinalState]:
        return (  # type:ignore[return-value]
            to_particle(self.decay_node.child1),
            to_particle(self.decay_node.child2),
        )

    @property
    def spectator(self) -> FinalState:
        return self._get_child_of_type(State)

    @cache  # noqa: B019
    def _get_child_of_type(self, typ: type[T]) -> T:
        for child in self.decay.children:
            if isinstance(child, typ):
                return child
        msg = f"The production node does not have any children that are of type {typ.__name__}"
        raise ValueError(msg)

    @property
    def incoming_ls(self) -> LSCoupling | None:
        return self.decay.interaction

    @property
    def outgoing_ls(self) -> LSCoupling | None:
        return self.decay_node.interaction


T = TypeVar("T", IsobarNode, Particle, InitialState, FinalState)


@frozen(order=True)
class LSCoupling:
    L: int
    S: sp.Rational = field(converter=to_rational, validator=assert_spin_value)


@overload
def to_particle(isobar: IsobarNode[ParentType]) -> ParentType: ...
@overload
def to_particle(isobar: State[StateIDTemplate]) -> State[StateIDTemplate]: ...
@overload
def to_particle(isobar: Particle) -> Particle: ...
def to_particle(isobar):
    if isinstance(isobar, IsobarNode):
        return isobar.parent
    return isobar
