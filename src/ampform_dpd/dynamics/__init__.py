"""Dynamics builders for :meth:`.register_builder`.

.. note:: As opposed to `AmpForm <https://ampform.rtfd.io>`_, AmpForm-DPD defines
    dynamics over the **entire decay chain**, not a single isobar node. The dynamics
    classes and the corresponding builders would have to be extended to implement other
    dynamics lineshapes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import sympy as sp
from ampform.dynamics import BreitWigner, SimpleBreitWigner
from ampform.dynamics.form_factor import FormFactor
from ampform.dynamics.phasespace import PhaseSpaceFactor, PhaseSpaceFactorProtocol
from attrs import define, field
from attrs.validators import in_

from ampform_dpd import DefinedExpression, create_mass_symbol, to_particle
from ampform_dpd.decay import DecayNode, IsobarNode, State, ThreeBodyDecayChain

if TYPE_CHECKING:
    from tensorwaves.interface import ParameterValue


@define
class BreitWignerBuilder:
    """Build chain dynamics with explicit numerator and vertex normalization.

    Production uses the running resonance mass. ``normalize_form_factors`` divides each
    enabled vertex factor by its value at the resonance pole; it does not change the
    running width's pole normalization. ``numerator="mass_width"`` multiplies the
    propagator by the pole mass times the pole width.

    ``blatt_weisskopf_convention`` selects the convention of the vertex factors
    themselves, independently of the pole normalization: ``"normalized"`` keeps
    AmpForm's factor, which is one at :math:`z=1`, while ``"unnormalized"`` uses
    ``FormFactor(..., normalize=False)``, as published amplitude models do. Pole
    normalization cancels the resulting constant, so the two conventions only differ
    when ``normalize_form_factors`` is `False`.

    External masses are fixed parameter defaults, while Mandelstam invariants are event
    variables. Decay radii are per resonance; production radii are per parent.
    ``symbol_mapping`` renames or fixes parameters in expressions and defaults.
    ``parameter_defaults`` overrides values after mapping, allowing shared radii and
    downstream naming conventions without changing the lineshape.
    """

    energy_dependent_width: bool = True
    decay_form_factor: bool = True
    production_form_factor: bool = True
    phsp_factor: PhaseSpaceFactorProtocol = PhaseSpaceFactor  # ty: ignore[invalid-assignment]
    normalize_form_factors: bool = False
    numerator: Literal["unity", "mass_width"] = field(
        default="unity", validator=in_(("unity", "mass_width"))
    )
    blatt_weisskopf_convention: Literal["normalized", "unnormalized"] = field(
        default="normalized", validator=in_(("normalized", "unnormalized"))
    )
    symbol_mapping: dict[sp.Symbol, sp.Expr] = field(factory=dict)
    parameter_defaults: dict[sp.Basic, complex | float] = field(factory=dict)

    def __call__(self, decay_chain: ThreeBodyDecayChain) -> DefinedExpression:
        """Formulate a (relativistic) Breit-Wigner for this resonance."""
        decay_node = decay_chain.decay_node
        s = get_mandelstam_s(decay_node)
        if self.energy_dependent_width:
            expression = _create_breit_wigner(s, decay_node, self.phsp_factor)
        else:
            expression = _create_simple_breit_wigner(s, decay_node)
        mass = create_mass_symbol(decay_chain.resonance)
        if self.numerator == "mass_width":
            width = sp.Symbol(
                Rf"\Gamma_{{{decay_chain.resonance.latex}}}", nonnegative=True
            )
            expression *= mass * width
        if self.decay_form_factor:
            expression *= _create_form_factor(
                s,
                isobar=decay_node,
                pole_mass=mass if self.normalize_form_factors else None,
                convention=self.blatt_weisskopf_convention,
            )
        if self.production_form_factor:
            expression *= _create_form_factor(
                s,
                isobar=decay_chain.production_node,
                pole_mass=mass if self.normalize_form_factors else None,
                convention=self.blatt_weisskopf_convention,
            )
        expression.parameters.update({
            create_mass_symbol(state): state.mass
            for state in (decay_chain.parent, *decay_chain.final_state)
        })
        parameters = {}
        for symbol, value in expression.parameters.items():
            mapped = self.symbol_mapping.get(symbol, symbol)
            if isinstance(mapped, sp.Symbol):
                parameters[mapped] = value
        parameters.update(self.parameter_defaults)
        return DefinedExpression(
            expression.expression.xreplace(self.symbol_mapping), parameters
        )


formulate_breit_wigner_with_form_factor = BreitWignerBuilder()


def _create_form_factor(
    s: sp.Symbol,
    isobar: IsobarNode,
    *,
    pole_mass: sp.Symbol | None = None,
    convention: Literal["normalized", "unnormalized"] = "normalized",
) -> DefinedExpression:
    if _get_angular_momentum(isobar) == 0:
        return DefinedExpression()
    parameter_defaults: dict[sp.Basic, ParameterValue] = {}
    if isinstance(isobar.parent, State):
        parent_mass = create_mass_symbol(isobar.parent)
        invariant_mass_squared = parent_mass**2
        parameter_defaults[parent_mass] = isobar.parent.mass
    else:
        invariant_mass_squared = s
    outgoing_masses = []
    for child in isobar.children:
        if isinstance(child, IsobarNode):
            outgoing_masses.append(sp.sqrt(s))
        else:
            mass = create_mass_symbol(child)
            outgoing_masses.append(mass)
            parameter_defaults[mass] = to_particle(child).mass
    meson_radius = _create_meson_radius_symbol(isobar)
    form_factor = FormFactor(
        s=invariant_mass_squared,  # ty: ignore[unknown-argument]
        m1=outgoing_masses[0],  # ty: ignore[unknown-argument]
        m2=outgoing_masses[1],  # ty: ignore[unknown-argument]
        angular_momentum=_get_angular_momentum(isobar),  # ty: ignore[unknown-argument]
        meson_radius=meson_radius,  # ty: ignore[unknown-argument]
        normalize=convention == "normalized",  # ty: ignore[unknown-argument]
    )
    parameter_defaults[meson_radius] = 1
    if pole_mass is not None:
        form_factor /= form_factor.xreplace({s: pole_mass**2})
    return DefinedExpression(form_factor, parameter_defaults)


def _create_breit_wigner(
    s: sp.Symbol, isobar: DecayNode, phsp_factor: PhaseSpaceFactorProtocol
) -> DefinedExpression:
    outgoing_state_mass1 = create_mass_symbol(isobar.child1)
    outgoing_state_mass2 = create_mass_symbol(isobar.child2)
    angular_momentum = _get_angular_momentum(isobar)
    res_mass = create_mass_symbol(isobar.parent)
    res_width = sp.Symbol(Rf"\Gamma_{{{isobar.parent.latex}}}", nonnegative=True)
    meson_radius = _create_meson_radius_symbol(isobar)
    breit_wigner_expr = BreitWigner(
        s=s,  # ty: ignore[unknown-argument]
        mass=res_mass,  # ty: ignore[unknown-argument]
        width=res_width,  # ty: ignore[unknown-argument]
        m1=outgoing_state_mass1,  # ty: ignore[unknown-argument]
        m2=outgoing_state_mass2,  # ty: ignore[unknown-argument]
        angular_momentum=angular_momentum,  # ty: ignore[unknown-argument]
        meson_radius=meson_radius if angular_momentum else 1,  # ty: ignore[unknown-argument]
        phsp_factor=phsp_factor,  # ty: ignore[unknown-argument]
        numerator="unity",  # ty: ignore[unknown-argument]
    )
    parameter_defaults: dict[sp.Basic, complex | float] = {
        res_mass: isobar.parent.mass,
        res_width: isobar.parent.width,
    }
    if angular_momentum:
        parameter_defaults[meson_radius] = 1
    return DefinedExpression(breit_wigner_expr, parameter_defaults)


def _create_simple_breit_wigner(s: sp.Symbol, isobar: DecayNode) -> DefinedExpression:
    mass = create_mass_symbol(isobar.parent)
    width = sp.Symbol(Rf"\Gamma_{{{isobar.parent.latex}}}", nonnegative=True)
    return DefinedExpression(
        expression=SimpleBreitWigner(
            s,
            mass,
            width,
            numerator="unity",  # ty: ignore[unknown-argument]
        ),
        parameters={
            mass: isobar.parent.mass,
            width: isobar.parent.width,
        },
    )


def _get_angular_momentum(isobar: IsobarNode) -> int:
    if isobar.interaction is None:
        msg = "Need LS couplings to formulate a form factor"
        raise ValueError(msg)
    return isobar.interaction.L


def _create_meson_radius_symbol(isobar: IsobarNode) -> sp.Symbol:
    return sp.Symbol(Rf"R_{{{isobar.parent.latex}}}", nonnegative=True)


def get_mandelstam_s(decay: DecayNode) -> sp.Symbol:
    subsystem_id, *_ = {1, 2, 3} - {
        s.index for s in decay.children if isinstance(s, State)
    }
    return sp.Symbol(f"sigma{subsystem_id}", nonnegative=True)
