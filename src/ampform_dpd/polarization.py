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
from ampform.helicity.decay import is_opposite_helicity_state
from ampform.helicity.naming import get_helicity_angle_symbols
from ampform.kinematics.angles import compute_helicity_angles
from ampform.kinematics.lorentz import FourMomentumSymbol
from ampform.sympy._array_expressions import (  # ruff: ignore[import-private-name]
    ArraySum,
)
from qrules.topology import create_isobar_topologies
from sympy.core.symbol import Str

from ampform_dpd.decay import get_decay_product_ids

if TYPE_CHECKING:
    from typing import SupportsFloat

    from qrules.topology import Topology

    from ampform_dpd.decay import FinalStateID, Particle


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


def formulate_final_state_decay_angles(  # ruff: ignore[too-many-locals]
    state_id: FinalStateID,
    reference_subsystem: FinalStateID,
) -> dict[sp.Symbol, sp.Expr]:
    r"""Formulate the decay angles of a final-state particle in terms of four-momenta.

    The angles :math:`(\phi_k, \theta_k)` from `create_final_state_decay_angles`
    cannot be expressed in Mandelstam variables, because the decay products of
    final-state particle :math:`k` lie outside the three-body decay plane. This
    function formulates them through a chain of boosts and rotations with
    :func:`ampform.kinematics.angles.compute_helicity_angles`, by extending the
    three-body topology with a node in which particle :math:`k` decays further.

    The expressions are in terms of `~ampform.kinematics.lorentz.FourMomentumSymbol`
    arrays: :code:`p1`, :code:`p2`, :code:`p3` for the three-body final state and
    :code:`q<k>` for the analyzed decay product of particle :math:`k`, all expressed
    in the center-of-momentum frame of the three-body decay. The rest frame of
    particle :math:`k` is reached through the boost chain of the decay chain in
    which the `reference_subsystem
    <.DalitzPlotDecompositionBuilder.formulate_aligned_amplitude>` is the isobar:
    directly from the center-of-momentum frame if particle :math:`k` is the
    spectator of that chain, or through the isobar rest frame otherwise.

    >>> angles = formulate_final_state_decay_angles(2, reference_subsystem=2)
    >>> list(angles)
    [phi_2, theta_2]
    """
    momenta = {i: FourMomentumSymbol(f"p{i}", shape=[]) for i in (1, 2, 3)}
    daughter_momentum = FourMomentumSymbol(f"q{state_id}", shape=[])
    sibling_momentum = ArraySum(momenta[state_id], -daughter_momentum)
    topology, isobar_edges, decay_edges = _create_extended_topology(
        is_spectator=state_id == reference_subsystem
    )
    i, j = get_decay_product_ids(reference_subsystem)
    edge_momenta: dict[int, sp.Expr]
    if state_id == reference_subsystem:
        edge_momenta = dict(zip(isobar_edges, (momenta[i], momenta[j]), strict=True))
    else:
        spectator_edge, other_edge = isobar_edges
        other_id = next(iter({i, j} - {state_id}))
        edge_momenta = {
            spectator_edge: momenta[reference_subsystem],
            other_edge: momenta[other_id],
        }
    analyzed_edge, sibling_edge = decay_edges
    if is_opposite_helicity_state(topology, analyzed_edge):
        analyzed_edge, sibling_edge = sibling_edge, analyzed_edge
    edge_momenta[analyzed_edge] = daughter_momentum
    edge_momenta[sibling_edge] = sibling_momentum
    helicity_angles = compute_helicity_angles(edge_momenta, topology)
    φ_expr, θ_expr = (
        helicity_angles[symbol]
        for symbol in get_helicity_angle_symbols(topology, analyzed_edge)
    )
    φ, θ = create_final_state_decay_angles(state_id)
    return {φ: φ_expr, θ: θ_expr}


def _create_extended_topology(
    *, is_spectator: bool
) -> tuple[Topology, tuple[int, int], tuple[int, int]]:
    """Find the four-body topology that models the decay of a final-state particle.

    Returns the topology along with the IDs of the two edges that represent the
    three-body decay (isobar decay products, or spectator plus remaining isobar
    decay product) and the IDs of the two edges of the subsequent decay.
    """
    topologies = create_isobar_topologies(4)
    root_node = 0
    if is_spectator:
        topology = next(  # 0 → (i j) + k with k → daughters
            t for t in topologies if not _get_final_edges(t, root_node)
        )
        pair_1, pair_2 = (
            _get_final_edges(topology, t.ending_node_id)
            for t in _get_intermediate_edges(topology, root_node)
        )
        return topology, tuple(pair_1), tuple(pair_2)  # ty:ignore[invalid-return-type]
    topology = next(  # 0 → (i j) + r with i → daughters
        t for t in topologies if _get_final_edges(t, root_node)
    )
    (spectator_edge,) = _get_final_edges(topology, root_node)
    (isobar_edge,) = _get_intermediate_edges(topology, root_node)
    isobar_node = isobar_edge.ending_node_id
    (other_edge,) = _get_final_edges(topology, isobar_node)
    (decaying_edge,) = _get_intermediate_edges(topology, isobar_node)
    decay_edges = _get_final_edges(topology, decaying_edge.ending_node_id)
    return topology, (spectator_edge, other_edge), tuple(decay_edges)  # ty:ignore[invalid-return-type]


def _get_final_edges(topology: Topology, node_id: int | None) -> list[int]:
    if node_id is None:
        return []
    return sorted(
        i
        for i in topology.get_edge_ids_outgoing_from_node(node_id)
        if topology.edges[i].ending_node_id is None
    )


def _get_intermediate_edges(topology: Topology, node_id: int | None):
    if node_id is None:
        return []
    return [
        topology.edges[i]
        for i in sorted(topology.get_edge_ids_outgoing_from_node(node_id))
        if topology.edges[i].ending_node_id is not None
    ]


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
