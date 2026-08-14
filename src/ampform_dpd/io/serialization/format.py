from __future__ import annotations

import difflib
import sys
from collections import abc
from typing import TYPE_CHECKING, Literal, TypedDict, TypeGuard, Union
from warnings import warn

from ampform_dpd.decay import FinalStateID

if TYPE_CHECKING:
    from ampform_dpd.decay import StateID

if sys.version_info >= (3, 11):
    from typing import Required
else:
    from typing_extensions import Required


class ModelDefinition(TypedDict):
    distributions: Required[list]
    functions: Required[list[FunctionDefinition]]
    domains: Required[list]
    misc: dict
    parameter_points: list


class Distribution(TypedDict):
    name: Required[str]
    type: Required[str]
    decay_description: Required[DecayDescription]
    variables: Required[list]
    parameters: list


class DecayDescription(TypedDict):
    kinematics: Required[KinematicsDefinition]
    reference_topology: Required[Topology]
    chains: Required[list[DecayChain]]
    appendix: dict


Topology = list[Union[FinalStateID, "Topology"]]
"""Topology definition as a list of final state IDs."""


class DecayChain(TypedDict):
    name: Required[str]
    propagators: Required[list[Propagator]]
    vertices: Required[list[Vertex]]
    topology: Required[Topology]
    kinematics: Required[KinematicsDefinition]
    weight: Required[str]


class Propagator(TypedDict):
    spin: str
    node: Node
    parametrization: str


class Vertex(TypedDict):
    type: Required[Literal["helicity", "ls", "parity"]]
    node: Required[Node]
    formfactor: str


class HelicityVertex(Vertex):
    helicities: Required[tuple[str, str]]


class ParityVertex(HelicityVertex):
    parity_factor: ParityFactor


class LSVertex(Vertex):
    l: str
    s: str


Node = tuple[Union[FinalStateID, "Node"], Union[FinalStateID, "Node"]]
"""Node definition within a `.Topology`."""
ParityFactor = Literal["+", "-", ""]


class KinematicsDefinition(TypedDict):
    initial_state: Required[StateDefinition]
    final_state: Required[list[StateDefinition]]


class StateDefinition(TypedDict):
    index: StateID
    name: str
    spin: str
    mass: float


class FunctionDefinition(TypedDict):
    name: Required[str]
    type: Required[str]


class BlattWeisskopfDefinition(FunctionDefinition):
    l: int
    radius: float


class BreitWignerDefinition(FunctionDefinition):
    mass: float
    width: float
    ma: float
    mb: float
    l: float
    d: float


class GenericFunctionDefinition(FunctionDefinition):
    expression: str


class PolynomialDefinition(FunctionDefinition):
    coefficients: list[float]
    x: str


class MomentumPowerDefinition(FunctionDefinition):
    l: int


class MultichannelBreitWignerDefinition(FunctionDefinition):
    mass: float
    channels: list[ChannelParameters]


class ChannelParameters(TypedDict):
    gsq: float
    ma: float
    mb: float
    l: float
    d: float


def get_decay_chains(model: ModelDefinition, /) -> list[DecayChain]:
    distribution_def = get_distribution_def(model)
    return distribution_def["decay_description"]["chains"]


def get_distribution_def(model: ModelDefinition, /) -> Distribution:
    distribution_defs = model["distributions"]
    n_distributions = len(distribution_defs)
    if n_distributions == 0:
        msg = "The serialized model does not have any distributions"
        raise ValueError(msg)
    if n_distributions > 1:
        msg = f"There are {n_distributions} distributions, but expecting one only"
        warn(msg, category=UserWarning)
    return distribution_defs[0]


def get_function_definition(
    function_name: str, /, model: ModelDefinition
) -> FunctionDefinition:
    function_definitions = model["functions"]
    for function_def in function_definitions:
        if function_def["name"] == function_name:
            return function_def
    existing_names = {f["name"] for f in function_definitions}
    msg = f"Could not find function with name {function_name!r}."
    candidates = difflib.get_close_matches(function_name, existing_names)
    if candidates:
        msg += f" Did you mean any of these? {', '.join(sorted(candidates))}"
    raise KeyError(msg)


def get_reference_topology(model: ModelDefinition, /) -> Topology:
    distribution_def = get_distribution_def(model)
    return distribution_def["decay_description"]["reference_topology"]


def is_decay_node(node: Node, /) -> bool:
    """Whether the node is an isobar decaying into two final-state particles.

    >>> is_decay_node((2, 3))
    True
    >>> is_decay_node((1, (2, 3)))
    False
    """
    return all(isinstance(i, int) for i in node)


def is_production_node(node: Node, /) -> bool:
    """Whether the node is the initial state decaying into an isobar and a spectator.

    >>> is_production_node((1, (2, 3)))
    True
    >>> is_production_node((2, 3))
    False
    """
    return not is_decay_node(node)


def is_isobar(node_item: FinalStateID | Node, /) -> TypeGuard[Node]:
    """Whether an item within a node is a sub-node, not a final-state ID.

    >>> is_isobar((2, 3))
    True
    >>> is_isobar(1)
    False
    """
    return isinstance(node_item, abc.Sequence)
