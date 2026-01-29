"""Dynamics builder functions for :meth:`.register_builder`.

.. note:: As opposed to `AmpForm <https://ampform.rtfd.io>`_, AmpForm-DPD defines
    dynamics over the **entire decay chain**, not a single isobar node. The dynamics
    classes and the corresponding builders would have to be extended to implement other
    dynamics lineshapes.
"""

from __future__ import annotations

import sympy as sp
from ampform.dynamics.form_factor import FormFactor
from ampform.dynamics.phasespace import PhaseSpaceFactor, PhaseSpaceFactorProtocol
from attrs import define

from ampform_dpd import DefinedExpression, create_mass_symbol, to_particle
from ampform_dpd.decay import DecayNode, IsobarNode, State, ThreeBodyDecayChain
from ampform_dpd.dynamics import RelativisticBreitWigner, SimpleBreitWigner


@define
class BreitWignerBuilder:
    energy_dependent_width: bool = True
    decay_form_factor: bool = True
    production_form_factor: bool = True
    phsp_factor: PhaseSpaceFactorProtocol = PhaseSpaceFactor

    def __call__(self, decay_chain: ThreeBodyDecayChain) -> DefinedExpression:
        """Formulate a (relativistic) Breit-Wigner for this resonance."""
        decay_node = decay_chain.decay_node
        s = get_mandelstam_s(decay_node)
        if self.energy_dependent_width:
            expression = _create_breit_wigner(s, decay_node, self.phsp_factor)
        else:
            expression = _create_simple_breit_wigner(s, decay_node)
        if self.decay_form_factor:
            expression *= _create_form_factor(s, decay_node)
        if self.production_form_factor:
            expression *= _create_form_factor(s, decay_chain.production_node)
        return expression


formulate_breit_wigner_with_form_factor = BreitWignerBuilder()


def _create_form_factor(s: sp.Symbol, isobar: IsobarNode) -> DefinedExpression:
    if isinstance(isobar.parent, State):
        inv_mass = sp.Symbol("m0", nonnegative=True)
    else:
        inv_mass = get_mandelstam_s(isobar)
    outgoing_state_mass1 = create_mass_symbol(isobar.child1)
    outgoing_state_mass2 = create_mass_symbol(isobar.child2)
    meson_radius = _create_meson_radius_symbol(isobar)
    form_factor = FormFactor(
        s=inv_mass**2,  # ty:ignore[unknown-argument]
        m1=outgoing_state_mass1,  # ty:ignore[unknown-argument]
        m2=outgoing_state_mass2,  # ty:ignore[unknown-argument]
        angular_momentum=_get_angular_momentum(isobar),  # ty:ignore[unknown-argument]
        meson_radius=meson_radius,  # ty:ignore[unknown-argument]
    )
    parameter_defaults: dict[sp.Basic, complex | float] = {
        meson_radius: 1,
        outgoing_state_mass1: to_particle(isobar.child1).mass,
        outgoing_state_mass2: to_particle(isobar.child2).mass,
    }
    if not inv_mass.name.startswith("s"):
        parameter_defaults[inv_mass] = to_particle(isobar).mass
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
    breit_wigner_expr = RelativisticBreitWigner(
        s=s,  # ty:ignore[unknown-argument]
        mass0=res_mass,  # ty:ignore[unknown-argument]
        gamma0=res_width,  # ty:ignore[unknown-argument]
        m1=outgoing_state_mass1,  # ty:ignore[unknown-argument]
        m2=outgoing_state_mass2,  # ty:ignore[unknown-argument]
        angular_momentum=angular_momentum,  # ty:ignore[unknown-argument]
        meson_radius=meson_radius,  # ty:ignore[unknown-argument]
        phsp_factor=phsp_factor,  # ty:ignore[unknown-argument]
    )
    parameter_defaults: dict[sp.Basic, complex | float] = {
        res_mass: isobar.parent.mass,
        res_width: isobar.parent.width,
        meson_radius: 1,
    }
    return DefinedExpression(breit_wigner_expr, parameter_defaults)


def _create_simple_breit_wigner(s: sp.Symbol, isobar: DecayNode) -> DefinedExpression:
    mass = create_mass_symbol(isobar.parent)
    width = sp.Symbol(Rf"\Gamma_{{{isobar.parent.latex}}}", nonnegative=True)
    meson_radius = _create_meson_radius_symbol(isobar)
    return DefinedExpression(
        expression=SimpleBreitWigner(s, mass, width),
        parameters={
            mass: isobar.parent.mass,
            width: isobar.parent.width,
            meson_radius: 1,
        },
    )


def _get_angular_momentum(isobar: IsobarNode) -> int:
    if isobar.interaction is None:
        msg = "Need LS couplings to formulate a form factor"
        raise ValueError(msg)
    return isobar.interaction.L


def _create_meson_radius_symbol(isobar: IsobarNode) -> sp.Symbol:
    if isinstance(isobar.parent, State):
        return sp.Symbol(Rf"R_{{{isobar.parent.latex}}}", nonnegative=True)
    return sp.Symbol(R"R_\mathrm{res}", nonnegative=True)


def get_mandelstam_s(decay: DecayNode) -> sp.Symbol:
    subsystem_id, *_ = {1, 2, 3} - {
        s.index for s in decay.children if isinstance(s, State)
    }
    return sp.Symbol(f"sigma{subsystem_id}", nonnegative=True)
