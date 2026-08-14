from __future__ import annotations

import re
from collections import abc
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

import sympy as sp
from ampform.dynamics.form_factor import (
    BreakupMomentumSquared,
    FormFactor,
    SphericalHankel1,
)
from ampform.dynamics.phasespace import PhaseSpaceFactorComplex
from sympy.parsing.sympy_parser import parse_expr

from ampform_dpd import DefinedExpression
from ampform_dpd.dynamics import BreitWigner, ChannelArguments
from ampform_dpd.io.serialization.decay import get_initial_state
from ampform_dpd.io.serialization.format import (
    BlattWeisskopfDefinition,
    BreitWignerDefinition,
    DecayChain,
    GenericFunctionDefinition,
    ModelDefinition,
    MomentumPowerDefinition,
    MultichannelBreitWignerDefinition,
    PolynomialDefinition,
    Propagator,
    Vertex,
    get_function_definition,
)

if TYPE_CHECKING:
    from collections.abc import Callable

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
        "generic_function": formulate_generic_function,
        "MultichannelBreitWigner": formulate_multichannel_breit_wigner,
        "Polynomial": formulate_polynomial,
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
    if function_type == "MomentumPower":
        function_definition = cast("MomentumPowerDefinition", function_definition)
        node = vertex["node"]
        m1, m2 = (to_mass_symbol(i) for i in node)
        if _is_initial_state_node(node):
            s = to_mandelstam_symbol(node) ** 2
            m1 = sp.sqrt(m1)
        else:
            s = to_mandelstam_symbol(node)
        power = sp.Rational(function_definition["l"], 2)
        return DefinedExpression(expression=BreakupMomentumSquared(s, m1, m2) ** power)
    if function_type == "BlattWeisskopf":
        node = vertex["node"]
        if _is_initial_state_node(node):
            parent_mass = to_mandelstam_symbol(node)
            isobar_invariant, m2 = (to_mass_symbol(i) for i in node)
            s = parent_mass**2
            m1 = sp.sqrt(isobar_invariant)
            initial_state = get_initial_state(model)
            meson_radius = sp.Symbol(f"R_{{{initial_state.latex}}}", nonnegative=True)
        else:
            s = to_mandelstam_symbol(node)
            m1, m2 = (to_mass_symbol(i) for i in node)
            meson_radius = sp.Symbol(R"R_\mathrm{res}", nonnegative=True)
        angular_momentum = int(function_definition["l"])
        return DefinedExpression(
            expression=FormFactor(s, m1, m2, angular_momentum, meson_radius)  # ty: ignore[invalid-argument-type]
            / _blatt_weisskopf_normalization(angular_momentum),
            parameters={
                meson_radius: function_definition["radius"],
            },
        )
    msg = f"No form factor implementation for {function_name!r}"
    raise NotImplementedError(msg)


def _blatt_weisskopf_normalization(angular_momentum: int) -> sp.Expr:
    r"""Undo the normalization of AmpForm's `~ampform.dynamics.form_factor.FormFactor`.

    AmpForm normalizes its Blatt--Weisskopf factor to one at :math:`z=1`, whereas the
    serialization format uses the unnormalized convention, so the form factor has to be
    divided by this value.

    >>> _blatt_weisskopf_normalization(0)
    1
    >>> _blatt_weisskopf_normalization(1)
    sqrt(2)
    >>> _blatt_weisskopf_normalization(2)
    sqrt(13)
    """
    hankel = SphericalHankel1(sp.Integer(angular_momentum), sp.Integer(1))
    return sp.Abs(hankel.doit())


def formulate_generic_function(
    propagator: Propagator, resonance: str, model: ModelDefinition
) -> DefinedExpression:
    function_definition = get_function_definition(propagator["parametrization"], model)
    function_definition = cast("GenericFunctionDefinition", function_definition)
    expression = function_definition["expression"]
    expression = expression.replace("^", "**").replace("1im", "I")
    expression = expression.replace("m_12_sq", "sigma")
    mandelstam = to_mandelstam_symbol(propagator["node"])
    return DefinedExpression(
        expression=parse_expr(
            expression,
            local_dict={"i": sp.I, "I": sp.I, "sigma": mandelstam, "σ": mandelstam},
        )
    )


def formulate_polynomial(
    propagator: Propagator, resonance: str, model: ModelDefinition
) -> DefinedExpression:
    function_definition = get_function_definition(propagator["parametrization"], model)
    function_definition = cast("PolynomialDefinition", function_definition)
    variable = to_mandelstam_symbol(propagator["node"])
    _assert_variable_matches_node(function_definition["x"], propagator["node"])
    coefficients: dict[sp.Basic, complex | float] = {
        sp.Symbol(Rf"c_{{{resonance},{power}}}", real=True): value
        for power, value in enumerate(function_definition["coefficients"])
    }
    return DefinedExpression(
        expression=sum(
            coefficient * variable**power
            for power, coefficient in enumerate(coefficients)
        ),
        parameters=coefficients,
    )


def _assert_variable_matches_node(variable_name: str, node: Node) -> None:
    """Check that the serialized variable name is the invariant mass of the node.

    >>> _assert_variable_matches_node("m_23_sq", (2, 3))
    >>> _assert_variable_matches_node("m_12_sq", (2, 3))
    Traceback (most recent call last):
        ...
    ValueError: Variable 'm_12_sq' is sigma3, but node (2, 3) implies sigma1
    """
    expected = to_mandelstam_symbol(node)
    variable = _to_mandelstam_symbol_from_name(variable_name)
    if variable != expected:
        msg = f"Variable {variable_name!r} is {variable}, but node {node} implies {expected}"
        raise ValueError(msg)


def _to_mandelstam_symbol_from_name(variable_name: str) -> sp.Symbol:
    """Convert a serialized variable name to a Mandelstam symbol.

    >>> _to_mandelstam_symbol_from_name("m_12_sq")
    sigma3
    """
    matches = re.fullmatch(r"m_([1-3])([1-3])_sq", variable_name)
    if matches is None:
        msg = f"Cannot convert variable name {variable_name!r} to a Mandelstam symbol"
        raise NotImplementedError(msg)
    i, j = (int(index) for index in matches.groups())
    return to_mass_symbol((i, j))


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
        expression=BreitWigner(s, mass, width, m1, m2, angular_momentum, d),  # ty: ignore[invalid-argument-type]
        parameters={
            mass: function_definition["mass"],
            width: function_definition["width"],
            m1: function_definition["ma"],
            m2: function_definition["mb"],
            d: function_definition["d"],
        },
    )


def formulate_multichannel_breit_wigner(  # ruff: ignore[too-many-locals]
    propagator: Propagator, resonance: str, model: ModelDefinition
) -> DefinedExpression:
    function_definition = get_function_definition(propagator["parametrization"], model)
    function_definition = cast("MultichannelBreitWignerDefinition", function_definition)
    channel_definitions = function_definition["channels"]
    if len(channel_definitions) < 2:  # ruff: ignore[magic-value-comparison]
        msg = "Need at least two channels for a multi-channel Breit-Wigner"
        raise NotImplementedError(msg)
    node = propagator["node"]
    i, j = node
    s = to_mandelstam_symbol(node)
    mass = sp.Symbol(f"m_{{{resonance}}}", nonnegative=True)
    g_squared = sp.Symbol(Rf"\Gamma_{{{resonance}}}", nonnegative=True)
    m1 = to_mass_symbol(i)
    m2 = to_mass_symbol(j)
    angular_momentum = int(channel_definitions[0]["l"])
    d = sp.Symbol(f"R_{{{resonance}}}", nonnegative=True)
    channels = [ChannelArguments(s, mass, g_squared, m1, m2, angular_momentum, d)]  # ty: ignore[invalid-argument-type]
    parameter_defaults: dict[sp.Basic, complex | float] = {
        mass: function_definition["mass"],
        g_squared: channel_definitions[0]["gsq"],
        m1: channel_definitions[0]["ma"],
        m2: channel_definitions[0]["mb"],
        d: channel_definitions[0]["d"],
    }
    for channel_idx, channel_definition in enumerate(channel_definitions[1:], 2):
        g_squared_i = sp.Symbol(
            name=Rf"\Gamma_{{{resonance}}}^\text{{ch. {channel_idx}}}",
            nonnegative=True,
        )
        mi1 = sp.Symbol(f"m_{{a,{channel_idx}}}", nonnegative=True)
        mi2 = sp.Symbol(f"m_{{b,{channel_idx}}}", nonnegative=True)
        angular_momentum = int(channel_definition["l"])
        channels.append(
            ChannelArguments(
                s,
                mass,
                g_squared_i,
                mi1,
                mi2,
                angular_momentum,  # ty: ignore[invalid-argument-type]
                meson_radius=d,  # ty: ignore[unknown-argument]
            )
        )
        parameter_defaults.update({
            mi1: channel_definition["ma"],
            mi2: channel_definition["mb"],
            g_squared_i: channel_definition["gsq"],
        })
    channel_terms = (
        channel.coupling_squared
        * PhaseSpaceFactorComplex(channel.s, channel.m1, channel.m2)
        * FormFactor(
            channel.s,
            channel.m1,
            channel.m2,
            channel.angular_momentum,
            channel.meson_radius,
        )
        ** 2
        for channel in channels
    )
    expression = 1 / (mass**2 - s - sp.I * sum(channel_terms))
    return DefinedExpression(expression, parameter_defaults)


def to_mandelstam_symbol(node: Node) -> sp.Symbol:
    """Create a Mandelstam symbol for a node.

    >>> to_mandelstam_symbol([3, 2])
    sigma1
    >>> to_mandelstam_symbol([1, [2, 3]])
    m0
    """
    if _is_initial_state_node(node):
        return to_mass_symbol(0)
    return to_mass_symbol(node)


def _is_initial_state_node(node: Node) -> bool:
    """Whether the decaying particle in this node is the initial state.

    A node that decays into two final-state particles is an isobar, so its invariant mass
    is a Mandelstam variable. If one of the decay products is itself an isobar (a nested
    node), the decaying particle is the initial state.

    >>> _is_initial_state_node([3, 2])
    False
    >>> _is_initial_state_node([1, [2, 3]])
    True
    """
    return not all(isinstance(i, int) for i in node)


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
        and len(node_item) == 2  # ruff: ignore[magic-value-comparison]
    ):
        k, *_ = {1, 2, 3} - set(node_item)
        return sp.Symbol(f"sigma{k}", nonnegative=True)
    msg = f"Cannot create mass symbol for node {node_item}"
    raise NotImplementedError(msg)
