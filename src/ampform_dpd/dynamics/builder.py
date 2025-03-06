"""Dynamics builder functions for :meth:`.register_builder`.

.. note:: As opposed to `AmpForm <https://ampform.rtfd.io>`_, AmpForm-DPD defines
    dynamics over the **entire decay chain**, not a single isobar node. The dynamics
    classes and the corresponding builders would have to be extended to implement other
    dynamics lineshapes.
"""

from __future__ import annotations

import sympy as sp

from ampform_dpd import to_particle
from ampform_dpd.decay import (
    DecayNode,
    IsobarNode,
    Particle,
    State,
    ThreeBodyDecayChain,
)
from ampform_dpd.dynamics import RelativisticBreitWigner


def formulate_breit_wigner_with_form_factor(
    decay_chain: ThreeBodyDecayChain,
) -> tuple[sp.Expr, dict[sp.Symbol, complex | float]]:
    decay_node = decay_chain.decay_node
    s = get_mandelstam_s(decay_node)
    return _create_breit_wigner(s, decay_node)


def _create_breit_wigner(
    s: sp.Symbol, isobar: DecayNode
) -> tuple[sp.Expr, dict[sp.Symbol, complex | float]]:
    outgoing_state_mass1 = create_mass_symbol(isobar.child1)
    outgoing_state_mass2 = create_mass_symbol(isobar.child2)
    angular_momentum = _get_angular_momentum(isobar)
    res_mass = create_mass_symbol(isobar.parent)
    res_width = sp.Symbol(Rf"\Gamma_{{{isobar.parent.latex}}}", nonnegative=True)
    meson_radius = _create_meson_radius_symbol(isobar)

    breit_wigner_expr = RelativisticBreitWigner(
        s=s,
        mass0=res_mass,
        gamma0=res_width,
        m1=outgoing_state_mass1,
        m2=outgoing_state_mass2,
        angular_momentum=angular_momentum,
        meson_radius=meson_radius,
    )
    parameter_defaults: dict[sp.Symbol, complex | float] = {
        res_mass: isobar.parent.mass,
        res_width: isobar.parent.width,
        meson_radius: 1,
    }
    return breit_wigner_expr, parameter_defaults


def _get_angular_momentum(isobar: IsobarNode) -> int:
    if isobar.interaction is None:
        msg = "Need LS couplings to formulate a form factor"
        raise ValueError(msg)
    return isobar.interaction.L


def _create_meson_radius_symbol(isobar: IsobarNode) -> sp.Symbol:
    if isinstance(isobar.parent, State):
        return sp.Symbol(Rf"R_{{{isobar.parent.latex}}}", nonnegative=True)
    return sp.Symbol(R"R_\mathrm{res}", nonnegative=True)


def create_mass_symbol(particle: IsobarNode | Particle) -> sp.Symbol:
    particle = to_particle(particle)
    return sp.Symbol(f"m_{{{particle.latex}}}", nonnegative=True)


def get_mandelstam_s(decay: DecayNode) -> sp.Symbol:
    subsystem_id, *_ = {1, 2, 3} - {s.index for s in decay.children}  # type:ignore[union-attr]
    return sp.Symbol(f"sigma{subsystem_id}", nonnegative=True)
