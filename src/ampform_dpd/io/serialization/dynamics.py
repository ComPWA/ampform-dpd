from __future__ import annotations

from collections import abc
from typing import TYPE_CHECKING, Callable, Protocol, TypeVar, cast

import sympy as sp
from ampform.dynamics.form_factor import FormFactor

from ampform_dpd import DefinedExpression
from ampform_dpd.dynamics import BreitWigner, ChannelArguments, MultichannelBreitWigner
from ampform_dpd.io.serialization.decay import get_initial_state
from ampform_dpd.io.serialization.format import (
    BlattWeisskopfDefinition,
    BreitWignerDefinition,
    DecayChain,
    ModelDefinition,
    MultichannelBreitWignerDefinition,
    Propagator,
    Vertex,
    get_function_definition,
)

if TYPE_CHECKING:
    from ampform_dpd.io.serialization.format import Node

T = TypeVar("T")


def identity_function(x: T) -> T:
    return x


class PropagatorDynamicsBuilder(Protocol):
    def __call__(
        self,
        propagator: Propagator,
        resonance: str,
        model: ModelDefinition,
    ) -> DefinedExpression: ...


def formulate_dynamics(
    chain_definition: DecayChain,
    model: ModelDefinition,
    to_latex: Callable[[str], str] = identity_function,
    additional_definitions: dict[str, PropagatorDynamicsBuilder] | None = None,
) -> DefinedExpression:
    definitions: dict[str, PropagatorDynamicsBuilder] = {
        "BreitWigner": formulate_breit_wigner,
        "MultichannelBreitWigner": formulate_multichannel_breit_wigner,
    }
    if additional_definitions is not None:
        definitions.update(additional_definitions)
    expr = DefinedExpression()
    for propagator in chain_definition["propagators"]:
        parametrization = propagator["parametrization"]
        function_definition = get_function_definition(parametrization, model)
        function_type = function_definition["type"]
        dynamics_builder = definitions.get(function_type)
        if dynamics_builder is None:
            msg = f"No dynamics implementation for function type {function_type!r}"
            raise NotImplementedError(msg)
        expr *= dynamics_builder(
            propagator,
            resonance=to_latex(chain_definition["name"]),
            model=model,
        )
    return expr


def formulate_form_factor(vertex: Vertex, model: ModelDefinition) -> DefinedExpression:
    function_name = vertex.get("formfactor")
    if not function_name:
        return DefinedExpression()
    function_definition = get_function_definition(function_name, model)
    function_definition = cast("BlattWeisskopfDefinition", function_definition)
    function_type = function_definition["type"]
    if function_type == "BlattWeisskopf":
        node = vertex["node"]
        s = to_mandelstam_symbol(node)
        m1, m2 = (to_mass_symbol(i) for i in node)
        if all(isinstance(i, int) for i in node):
            meson_radius = sp.Symbol(R"R_\mathrm{res}", nonnegative=True)
        else:
            initial_state = get_initial_state(model)
            meson_radius = sp.Symbol(f"R_{{{initial_state.latex}}}", nonnegative=True)
        angular_momentum = int(function_definition["l"])
        return DefinedExpression(
            expression=FormFactor(s, m1, m2, angular_momentum, meson_radius),
            definitions={
                meson_radius: function_definition["radius"],
            },
        )
    msg = f"No form factor implementation for {function_name!r}"
    raise NotImplementedError(msg)


def formulate_breit_wigner(
    propagator: Propagator, resonance: str, model: ModelDefinition
) -> DefinedExpression:
    function_definition = get_function_definition(propagator["parametrization"], model)
    function_definition = cast("BreitWignerDefinition", function_definition)
    node = propagator["node"]
    i, j = node
    s = to_mandelstam_symbol(node)
    mass = sp.Symbol(f"m_{{{resonance}}}", nonnegative=True)
    width = sp.Symbol(Rf"\Gamma_{{{resonance}}}", nonnegative=True)
    m1 = to_mass_symbol(i)
    m2 = to_mass_symbol(j)
    angular_momentum = int(function_definition["l"])
    d = sp.Symbol(R"R_\mathrm{res}", nonnegative=True)
    return DefinedExpression(
        expression=BreitWigner(s, mass, width, m1, m2, angular_momentum, d),
        definitions={
            mass: function_definition["mass"],
            width: function_definition["width"],
            m1: function_definition["ma"],
            m2: function_definition["mb"],
            d: function_definition["d"],
        },
    )


def formulate_multichannel_breit_wigner(  # noqa: PLR0914
    propagator: Propagator, resonance: str, model: ModelDefinition
) -> DefinedExpression:
    function_definition = get_function_definition(propagator["parametrization"], model)
    function_definition = cast("MultichannelBreitWignerDefinition", function_definition)
    channel_definitions = function_definition["channels"]
    if len(channel_definitions) < 2:  # noqa: PLR2004
        msg = "Need at least two channels for a multi-channel Breit-Wigner"
        raise NotImplementedError(msg)
    node = propagator["node"]
    i, j = node
    s = to_mandelstam_symbol(node)
    mass = sp.Symbol(f"m_{{{resonance}}}", nonnegative=True)
    width = sp.Symbol(Rf"\Gamma_{{{resonance}}}", nonnegative=True)
    m1 = to_mass_symbol(i)
    m2 = to_mass_symbol(j)
    angular_momentum = int(channel_definitions[0]["l"])
    d = sp.Symbol(f"R_{{{resonance}}}", nonnegative=True)
    channels = [ChannelArguments(s, mass, width, m1, m2, angular_momentum, d)]
    parameter_defaults: dict[sp.Symbol, complex | float] = {
        mass: function_definition["mass"],
        width: channel_definitions[0]["gsq"],
        m1: channel_definitions[0]["ma"],
        m2: channel_definitions[0]["mb"],
        d: channel_definitions[0]["d"],
    }
    for channel_idx, channel_definition in enumerate(channel_definitions[1:], 2):
        Γi = sp.Symbol(
            Rf"\Gamma_{{{resonance}}}^\text{{ch. {channel_idx}}}", nonnegative=True
        )
        mi1 = sp.Symbol(f"m_{{a,{channel_idx}}}", nonnegative=True)
        mi2 = sp.Symbol(f"m_{{b,{channel_idx}}}", nonnegative=True)
        angular_momentum = int(channel_definition["l"])
        channels.append(ChannelArguments(s, mass, Γi, mi1, mi2, angular_momentum, d))
        parameter_defaults.update({
            mi1: channel_definition["ma"],
            mi2: channel_definition["mb"],
            Γi: channel_definition["gsq"],
        })
    return DefinedExpression(
        expression=MultichannelBreitWigner(s, mass, tuple(channels)),
        definitions=parameter_defaults,
    )


def to_mandelstam_symbol(node: Node) -> sp.Symbol:
    """Create a Mandelstam symbol for a node.

    >>> to_mandelstam_symbol([3, 2])
    sigma1
    >>> to_mandelstam_symbol([1, [2, 3]])
    m0
    """
    if all(isinstance(i, int) for i in node):
        return to_mass_symbol(node)
    return to_mass_symbol(0)


def to_mass_symbol(node_item: int | Node) -> sp.Symbol:
    """Create a mass symbol for a node.

    >>> to_mass_symbol(1)
    m1
    >>> to_mass_symbol((1, 2))
    sigma3
    """
    if isinstance(node_item, int):
        return sp.Symbol(f"m{node_item}", nonnegative=True)
    if (
        isinstance(node_item, abc.Sequence)
        and all(isinstance(i, int) for i in node_item)
        and len(node_item) == 2  # noqa: PLR2004
    ):
        k, *_ = {1, 2, 3} - set(node_item)  # type:ignore[arg-type]
        return sp.Symbol(f"sigma{k}", nonnegative=True)
    msg = f"Cannot create mass symbol for node {node_item}"
    raise NotImplementedError(msg)
