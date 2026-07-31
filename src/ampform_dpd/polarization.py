"""Symbols for the decay of a polarized final-state particle.

A Dalitz-plot fit does not observe the spin state of the final-state particles. If one
of them decays further, however, the direction of its decay products depends on its spin
state, so that its polarization becomes measurable. This module provides the angle and
coupling symbols for such a subsequent decay. See `formulate_final_state_decay
<.DalitzPlotDecompositionBuilder.formulate_final_state_decay>` for how they enter the
amplitude.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp
from sympy.core.symbol import Str

if TYPE_CHECKING:
    from typing import SupportsFloat

    from ampform_dpd.decay import Particle


def create_final_state_decay_angles(state_id: int) -> tuple[sp.Symbol, sp.Symbol]:
    r"""Create angle symbols for the decay of a polarized final-state particle.

    The symbols :math:`(\phi_k, \theta_k)` represent the azimuthal and polar angles
    of one of the decay products of final-state particle :math:`k`, measured in the
    aligned rest frame of that particle.

    >>> create_final_state_decay_angles(state_id=2)
    (phi_2, theta_2)
    """
    return (
        sp.Symbol(f"phi_{state_id}", real=True),
        sp.Symbol(f"theta_{state_id}", real=True),
    )


def create_final_state_coupling(
    particle: Particle, helicity: SupportsFloat | sp.Basic
) -> sp.Indexed:
    r"""Create a coupling symbol for the decay of a polarized final-state particle.

    The symbol :math:`\mathcal{H}^\mathrm{fs}` is indexed by the particle
    and the helicity of its decay product. See `formulate_weak_decay_couplings` for
    suitable parameter values.
    """
    H = sp.IndexedBase(R"\mathcal{H}^\mathrm{fs}")
    if not isinstance(helicity, sp.Basic):
        helicity = sp.Rational(float(helicity))
    return H[Str(particle.latex), helicity]


def formulate_weak_decay_couplings(
    particle: Particle, alpha
) -> dict[sp.Indexed, sp.Expr]:
    r"""Formulate final-state decay couplings for a weak two-body decay of a spin-½ particle.

    In a parity-violating decay of a spin-½ particle to a spin-½ and a spin-0
    particle (such as :math:`\Sigma^+ \to p\pi^0`), the helicity couplings are
    fixed by the decay-asymmetry parameter :math:`\alpha` as
    :math:`\mathcal{H}_{\pm 1/2} = \sqrt{\left(1\pm\alpha\right)/2}`.
    """
    j = sp.Rational(particle.spin)
    if j != sp.Rational(1, 2):
        msg = f"Particle {particle.name} must have spin 1/2, got {j}"
        raise ValueError(msg)
    return {
        create_final_state_coupling(particle, +0.5): sp.sqrt((1 + alpha) / 2),
        create_final_state_coupling(particle, -0.5): sp.sqrt((1 - alpha) / 2),
    }
