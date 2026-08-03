r"""Charge-conjugation relations between the decay chains of a `.ThreeBodyDecay`.

If charge conjugation maps the final state of a three-body decay onto itself, the strong
interaction relates each decay chain to the chain of its charge-conjugate resonance. The
two chains are then **one** model component with a single set of couplings and a fixed
relative sign :math:`s`,

.. math:: \mathcal{H}\left[\bar R, \dots\right] = s \, \mathcal{H}\left[R, \dots\right],
    :label: conjugate-coupling-relation

with :math:`s=\pm1` determined by the quantum numbers of the decay alone. This module
derives that sign for an **arbitrary** three-body decay (see
:func:`get_conjugate_coupling_sign`) and applies it to an `.AmplitudeModel` (see
:func:`symmetrize_conjugate_couplings`).

The sign is the product of exactly three kinds of factors,

.. math:: s
    = \underbrace{C_0}_\text{initial state}
    \; \underbrace{\prod_a C_a}_\text{self-conjugate final states}
    \; \underbrace{\prod_v \eta_v}_\text{re-ordered vertices},
    :label: conjugate-coupling-sign

where :math:`C_0` is the C-parity of the decaying particle, the second product runs over
the final-state particles that charge conjugation leaves in place, and the third product
runs over the isobar vertices whose two children are written in a different order than
the order in which charge conjugation delivers them. In the cyclic pair ordering of the
`DPD paper <https://doi.org/10.1103/PhysRevD.101.034033>`_ (Eq. 7), the production vertex
:math:`0 \to R\,k` always matches, and the decay vertex :math:`R \to i\,j` is re-ordered
whenever charge conjugation acts non-trivially on :math:`i` or :math:`j`. Its exchange
phase depends on the basis in which the couplings are defined,

.. math::
    \eta^\text{LS} = (-1)^{l+s_i+s_j-S} \, , \qquad
    \eta^\text{helicity} = (-1)^{J_R-s_i-s_j} \, ,
    :label: exchange-phase

with :math:`(l, S)` the LS coupling of :math:`R \to i\,j` and :math:`J_R` the spin of the
resonance. The two bases give different signs (:math:`\eta^\text{LS}` is
:math:`l`-dependent, :math:`\eta^\text{helicity}` is not), so a sign derived in one basis
must never be applied to the couplings of the other.

For :math:`J/\psi \to p\bar p\eta`, Equation :eq:`conjugate-coupling-sign` collapses to
:math:`s = C_\psi C_\eta (-1)^l = P_{N^*}`, that is, :math:`-1` for the
:math:`\tfrac12^-` states and :math:`+1` for the :math:`\tfrac12^+` states.

.. warning:: Equation :eq:`conjugate-coupling-sign` relates the couplings as they are
    defined on two-particle states in the pair ordering of the DPD paper. An
    implementation stacks further conventions on top of that (the :math:`(-1)^{j_2-m_2}`
    phases of Eq. (8), the angle conventions, and the alignment rotations), each of which
    can contribute a further fixed phase that hand algebra does not see. Cross-check the
    sign against the amplitudes themselves before using it in a fit.

    Mirror symmetry of the Dalitz plot is only a partial check. For a model that consists
    of a conjugate **pair** alone, both signs give a mirror-symmetric intensity: they are
    the :math:`C=-1` and :math:`C=+1` eigenstates of the tie, and an overall factor
    :math:`s` drops out of the modulus. As soon as the model also contains a chain that
    charge conjugation maps onto *itself*, that chain interferes with the pair and the
    mirror symmetry does become sensitive to :math:`s`; see the
    :math:`J/\psi \to 3\pi` section of :doc:`/cparity`.

.. warning:: In the :math:`LS` basis, `.DalitzPlotDecompositionBuilder` does not currently
    write subsystem 2 in the pair ordering that Equation :eq:`conjugate-coupling-sign`
    assumes (`ComPWA/ampform-dpd#202
    <https://github.com/ComPWA/ampform-dpd/issues/202>`_), which gives that subsystem an
    extra :math:`\eta^\text{LS}`. Until that is resolved, an :math:`LS` model whose
    conjugate pair involves subsystem 2 is tied with the wrong **relative** sign between
    waves of different :math:`l`. Helicity couplings are unaffected.
"""

from __future__ import annotations

import itertools
from collections import Counter
from functools import lru_cache
from typing import TYPE_CHECKING, Literal
from warnings import warn

import attrs
import sympy as sp
from sympy.core.symbol import Str

from ampform_dpd import _get_coefficient_base, _get_coupling_base
from ampform_dpd.decay import FinalStateID, State, ThreeBodyDecay, ThreeBodyDecayChain

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from qrules.particle import ParticleCollection

    from ampform_dpd import AmplitudeModel

CouplingBasis = Literal["LS", "helicity"]
"""Basis in which the couplings of an isobar vertex are defined."""


def get_conjugate_state_map(
    decay: ThreeBodyDecay | ThreeBodyDecayChain,
    particle_db: ParticleCollection | None = None,
) -> dict[FinalStateID, FinalStateID]:
    """Permutation of the final-state IDs that is induced by charge conjugation.

    Charge conjugation constrains a single amplitude only if it maps the final state onto
    itself as a **set**. If it does, it permutes the final-state IDs and that permutation
    is what all further bookkeeping follows from. Since charge conjugation is an
    involution, the permutation of a three-body final state is either the identity (all
    final-state particles are self-conjugate) or a single transposition. For
    :math:`J/\\psi \\to \\eta\\, p\\, \\bar p`, with the :math:`\\eta` as state 1, it is
    :code:`{1: 1, 2: 3, 3: 2}`.

    A decay whose final state charge conjugation maps onto a *different* final state is
    not constrained at all and raises a `ValueError`.
    """
    particle_db = _get_particle_db(particle_db)
    final_state = _get_final_state(decay)
    antiparticles = {
        i: get_antiparticle_name(state.name, particle_db)
        for i, state in final_state.items()
    }
    if Counter(antiparticles.values()) != Counter(s.name for s in final_state.values()):
        conjugate_final_state = ", ".join(
            f"{i}: {name}" for i, name in sorted(antiparticles.items())
        )
        original_final_state = ", ".join(
            f"{i}: {state.name}" for i, state in sorted(final_state.items())
        )
        msg = (
            "Charge conjugation does not map the final state onto itself:"
            f"\n  {original_final_state}"
            f"\n  ↦ {conjugate_final_state}"
        )
        raise ValueError(msg)
    state_map: dict[FinalStateID, FinalStateID] = {
        i: i for i, state in final_state.items() if antiparticles[i] == state.name
    }
    for i in sorted(final_state):
        if i in state_map:
            continue
        partner = next(
            j
            for j in sorted(final_state)
            if j not in state_map and final_state[j].name == antiparticles[i]
        )
        state_map[i] = partner
        state_map[partner] = i
    return dict(sorted(state_map.items()))


def is_c_symmetric(
    decay: ThreeBodyDecay | ThreeBodyDecayChain,
    particle_db: ParticleCollection | None = None,
) -> bool:
    """Check whether charge conjugation maps the final state of a decay onto itself.

    This is the gate for everything else in this module: if it is `False`, charge
    conjugation relates the decay to a *different* process and there is nothing to
    symmetrize.
    """
    try:
        get_conjugate_state_map(decay, particle_db)
    except ValueError:
        return False
    return True


def get_conjugate_coupling_sign(
    chain: ThreeBodyDecayChain,
    basis: CouplingBasis = "LS",
    particle_db: ParticleCollection | None = None,
) -> int:
    r"""Relative sign between the couplings of a chain and its charge conjugate.

    Computes :math:`s` in Equation :eq:`conjugate-coupling-relation` from Equation
    :eq:`conjugate-coupling-sign` for an arbitrary three-body decay. The sign is the
    same for both chains of a conjugate pair, so it does not matter which of the two is
    passed as :code:`chain`.

    Args:
        chain: The decay chain for which to compute the sign.
        basis: The basis in which the coupling of the **decay** vertex is defined. This
            selects the exchange phase of Equation :eq:`exchange-phase` and generally
            changes the sign.
        particle_db: Particle database from which the C-parities and the antiparticles
            are read. Defaults to :func:`~ampform_dpd.adapter.qrules.load_particles`.

    If the chain is mapped onto itself by charge conjugation (its resonance is
    self-conjugate), the relation becomes a **selection rule** instead: the couplings of
    that chain have to vanish if the sign is :math:`-1`. See
    :func:`get_c_forbidden_chains`.
    """
    particle_db = _get_particle_db(particle_db)
    state_map = get_conjugate_state_map(chain, particle_db)
    sign = _get_c_parity(chain.initial_state, particle_db)
    for state in chain.final_state:
        if state_map[state.index] == state.index:
            sign *= _get_c_parity(state, particle_db)
    child1, child2 = chain.decay_products
    if (
        state_map[child1.index] != child1.index
        or state_map[child2.index] != child2.index
    ):
        sign *= get_exchange_phase(chain, basis)
    return sign


def get_exchange_phase(chain: ThreeBodyDecayChain, basis: CouplingBasis = "LS") -> int:
    r"""Phase for writing the decay products of a chain in the opposite order.

    A two-particle state is a constructed object: one of the two particles is listed
    first and defines the direction of the relative momentum, and the spins are coupled
    in the listed order. Listing the same two particles the other way round describes the
    same state in a different basis, at the cost of the phase of Equation
    :eq:`exchange-phase`.
    """
    child1, child2 = chain.decay_products
    if basis == "helicity":
        exponent = sp.Rational(chain.resonance.spin - child1.spin - child2.spin)
    elif basis == "LS":
        ls = chain.outgoing_ls
        if ls is None:
            msg = (
                f"Chain {chain.resonance.name} has no LS coupling on its decay node, so"
                ' the exchange phase can only be computed in the "helicity" basis'
            )
            raise ValueError(msg)
        exponent = sp.Rational(ls.L + child1.spin + child2.spin - ls.S)
    else:
        msg = f'Basis has to be either "LS" or "helicity", not "{basis}"'
        raise ValueError(msg)
    if exponent.denominator != 1:
        msg = (
            f"Exchange phase of chain {chain.resonance.name} has non-integer exponent"
            f" {exponent}, which means that the decay node is not a valid angular"
            " momentum coupling"
        )
        raise ValueError(msg)
    return (-1) ** int(exponent)


def get_conjugate_chain_pairs(
    decay: ThreeBodyDecay,
    particle_db: ParticleCollection | None = None,
) -> list[tuple[ThreeBodyDecayChain, ThreeBodyDecayChain]]:
    """Pairs of decay chains that charge conjugation ties together.

    Each pair consists of a chain and the chain of the charge-conjugate resonance in the
    charge-conjugate subsystem, with the same LS couplings on both nodes. The chain of
    the *particle* (positive PID) comes first, so that the second chain of each pair is
    the one to be expressed in terms of the first, as in Equation
    :eq:`conjugate-coupling-relation`. Chains that charge conjugation maps onto
    themselves are not part of any pair; see :func:`get_c_forbidden_chains`.
    """
    particle_db = _get_particle_db(particle_db)
    state_map = get_conjugate_state_map(decay, particle_db)
    pairs = []
    collected: set[ThreeBodyDecayChain] = set()
    for chain in decay.chains:
        subsystem_id = chain.spectator.index
        if state_map[subsystem_id] == subsystem_id:
            continue  # charge conjugation maps this chain onto itself
        if chain in collected:
            continue
        conjugate_chain = _find_conjugate_chain(chain, decay, state_map, particle_db)
        if conjugate_chain is None:
            resonance_name = get_antiparticle_name(chain.resonance.name, particle_db)
            msg = (
                f"Decay chain {chain.resonance.name} has no charge-conjugate partner"
                f" ({resonance_name} in subsystem {state_map[subsystem_id]}) with the"
                " same LS couplings. The model therefore violates C conservation."
            )
            warn(msg, category=UserWarning, stacklevel=2)
            continue
        collected.update({chain, conjugate_chain})
        if particle_db[chain.resonance.name].pid < 0:
            chain, conjugate_chain = conjugate_chain, chain  # ruff: ignore[redefined-loop-name]
        pairs.append((chain, conjugate_chain))
    return pairs


def get_c_forbidden_chains(
    decay: ThreeBodyDecay,
    basis: CouplingBasis = "LS",
    particle_db: ParticleCollection | None = None,
) -> list[ThreeBodyDecayChain]:
    r"""Chains whose couplings have to vanish because of C conservation.

    A chain that charge conjugation maps onto itself is not tied to another chain.
    Equation :eq:`conjugate-coupling-relation` then relates its couplings to
    *themselves*, which is a selection rule: the couplings have to vanish unless
    :math:`s=+1`. For :math:`J/\psi\to p\bar p\eta`, this is the statement that a
    resonance :math:`X \to p\bar p` recoiling against the :math:`\eta` needs
    :math:`C_X = C_\psi C_\eta = -1`.
    """
    particle_db = _get_particle_db(particle_db)
    state_map = get_conjugate_state_map(decay, particle_db)
    return [
        chain
        for chain in decay.chains
        if state_map[chain.spectator.index] == chain.spectator.index
        and get_conjugate_coupling_sign(chain, basis, particle_db) < 0
    ]


def relate_conjugate_couplings(
    model: AmplitudeModel,
    particle_db: ParticleCollection | None = None,
) -> dict[sp.Indexed, sp.Expr]:
    """Substitutions that express conjugate couplings in terms of their partners.

    The returned mapping sends each coupling of the second chain of a conjugate pair (see
    :func:`get_conjugate_chain_pairs`) onto :math:`\\pm` the corresponding coupling of the
    first chain. By convention, the sign of Equation :eq:`conjugate-coupling-sign` is
    carried entirely by the **production** coupling: only the product of the couplings
    along a chain is observable, so distributing the sign over the two vertices is a
    choice, and putting it on the production coupling keeps the decay couplings of a
    resonance and its charge conjugate identical.

    Use :func:`symmetrize_conjugate_couplings` to apply these substitutions to a model.
    """
    particle_db = _get_particle_db(particle_db)
    parameters = _collect_couplings(model)
    substitutions: dict[sp.Indexed, sp.Expr] = {}
    for chain, conjugate_chain in get_conjugate_chain_pairs(model.decay, particle_db):
        substitutions.update(
            _relate_chain_couplings(chain, conjugate_chain, parameters, particle_db)
        )
    return substitutions


def symmetrize_conjugate_couplings(
    model: AmplitudeModel,
    particle_db: ParticleCollection | None = None,
) -> AmplitudeModel:
    """Tie the couplings of charge-conjugate decay chains within an `.AmplitudeModel`.

    Substitutes the couplings of each conjugate chain by :math:`\\pm` the couplings of its
    partner (see :func:`relate_conjugate_couplings`) and removes them from the parameter
    defaults, so that the two chains share one set of free parameters. Chains that are
    forbidden by the selection rule of :func:`get_c_forbidden_chains` are reported as a
    `UserWarning`, but are left in the model.

    Raises:
        ValueError: If charge conjugation does not map the final state of the decay onto
            itself, in which case there is nothing to symmetrize. Use
            :func:`is_c_symmetric` to check this up front.
    """
    particle_db = _get_particle_db(particle_db)
    forbidden_chains = get_c_forbidden_chains(
        model.decay, _get_decay_coupling_basis(model), particle_db
    )
    if forbidden_chains:
        resonance_names = ", ".join(c.resonance.name for c in forbidden_chains)
        msg = (
            f"The couplings of decay chain(s) {resonance_names} have to vanish, because"
            " charge conjugation maps them onto themselves with a negative sign"
        )
        warn(msg, category=UserWarning, stacklevel=2)
    substitutions = relate_conjugate_couplings(model, particle_db)
    if not substitutions:
        return model
    return attrs.evolve(
        model,
        intensity=model.intensity.xreplace(substitutions),
        amplitudes={k: v.xreplace(substitutions) for k, v in model.amplitudes.items()},
        parameter_defaults={
            symbol: value
            for symbol, value in model.parameter_defaults.items()
            if symbol not in substitutions
        },
    )


def get_antiparticle_name(
    name: str, particle_db: ParticleCollection | None = None
) -> str:
    """Get the name of the charge conjugate of a particle.

    >>> get_antiparticle_name("N(1535)+")
    'N(1535)~-'
    >>> get_antiparticle_name("eta")
    'eta'
    """
    particle_db = _get_particle_db(particle_db)
    particle = particle_db[name]
    try:
        return particle_db.find(-particle.pid).name
    except KeyError:
        return particle.name  # self-conjugate particles have no negative PID


def _collect_couplings(model: AmplitudeModel) -> list[sp.Indexed]:
    """Collect all coupling symbols that occur in a model.

    The amplitude expressions of an `.AmplitudeModel` contain couplings whose helicity
    indices are still the symbolic summation variable of a `~ampform.sympy.PoolSum`,
    while the parameter defaults contain the same couplings with the summation variable
    substituted by each of its values. Both forms have to be related to their conjugate
    partner, so that the substitutions can be applied before as well as after the sums
    are evaluated.
    """
    couplings = {p for p in model.parameter_defaults if isinstance(p, sp.Indexed)}
    for expression in model.amplitudes.values():
        couplings.update(
            node
            for node in sp.preorder_traversal(expression)
            if isinstance(node, sp.Indexed)
            and not str(node.base).startswith("A^")  # amplitude, not a coupling
        )
    return sorted(couplings, key=str)


def _get_c_parity(state: State, particle_db: ParticleCollection) -> int:
    c_parity = particle_db[state.name].c_parity
    if c_parity is None:
        msg = (
            f"Particle {state.name} has no C-parity defined in the particle database,"
            " so the charge-conjugation sign cannot be derived"
        )
        raise ValueError(msg)
    return int(c_parity)


def _find_conjugate_chain(
    chain: ThreeBodyDecayChain,
    decay: ThreeBodyDecay,
    state_map: Mapping[FinalStateID, FinalStateID],
    particle_db: ParticleCollection,
) -> ThreeBodyDecayChain | None:
    resonance_name = get_antiparticle_name(chain.resonance.name, particle_db)
    subsystem_id = state_map[chain.spectator.index]
    for candidate in decay.chains:
        if (
            candidate.spectator.index == subsystem_id
            and candidate.resonance.name == resonance_name
            and candidate.incoming_ls == chain.incoming_ls
            and candidate.outgoing_ls == chain.outgoing_ls
        ):
            return candidate
    return None


def _relate_chain_couplings(
    chain: ThreeBodyDecayChain,
    conjugate_chain: ThreeBodyDecayChain,
    parameters: Iterable[sp.Indexed],
    particle_db: ParticleCollection,
) -> dict[sp.Indexed, sp.Expr]:
    latex = Str(chain.resonance.latex)
    conjugate_latex = Str(conjugate_chain.resonance.latex)
    couplings = {
        symbol: classification
        for symbol in parameters
        if (classification := _classify_coupling(symbol, conjugate_latex)) is not None
    }
    if not couplings:
        return {}
    sign = get_conjugate_coupling_sign(
        chain, _get_decay_basis(couplings.values()), particle_db
    )
    substitutions: dict[sp.Indexed, sp.Expr] = {}
    for symbol, (node_type, prod_helicity, dec_helicity) in couplings.items():
        factor = sign if node_type in {"production", "chain"} else 1
        if node_type == "chain":
            # the resonance is part of the base of a single coefficient per chain
            indices = _swap_decay_indices(symbol.indices, node_type, dec_helicity)
            base = _get_coefficient_base(latex, prod_helicity, dec_helicity)
            substitutions[symbol] = factor * base[indices]
        else:
            indices = _swap_decay_indices(symbol.indices[1:], node_type, dec_helicity)
            base = _get_coupling_base(prod_helicity, node_type)
            substitutions[symbol] = factor * base[(latex, *indices)]
    return substitutions


_NodeType = Literal["production", "decay", "chain"]
_Classification = tuple[_NodeType, bool, bool]
"""Node type of a coupling symbol, plus whether its vertices use helicity couplings.

A coupling symbol that belongs to a single node only carries the flag of that node, in
both slots, so that the base of the symbol can be reconstructed from the second element
and the ordering of its indices from the third.
"""


def _classify_coupling(symbol: sp.Indexed, latex: Str) -> _Classification | None:
    """Determine to which vertex of which resonance a coupling symbol belongs."""
    for helicity_basis in (False, True):
        for node_type in ("production", "decay"):
            if symbol.base != _get_coupling_base(helicity_basis, node_type):
                continue
            if symbol.indices[0] != latex:
                return None
            return node_type, helicity_basis, helicity_basis
    for production_basis, decay_basis in itertools.product((False, True), repeat=2):
        if symbol.base == _get_coefficient_base(latex, production_basis, decay_basis):
            return "chain", production_basis, decay_basis
    return None


def _swap_decay_indices(
    indices: tuple[sp.Basic, ...], node_type: _NodeType, decay_basis: bool
) -> tuple[sp.Basic, ...]:
    """Swap the two helicity indices of a decay node.

    Charge conjugation delivers the decay products of a chain in the order in which the
    partner chain lists them the other way round (see :func:`get_exchange_phase`), so the
    two helicity indices have to be swapped as well. LS couplings are unaffected, because
    they do not refer to the individual decay products.
    """
    if not decay_basis:
        return indices
    if node_type == "decay":
        return indices[::-1]
    if node_type == "chain":
        production_indices, decay_indices = indices[:2], indices[2:]
        return (*production_indices, *decay_indices[::-1])
    return indices


def _get_decay_basis(classifications: Iterable[_Classification]) -> CouplingBasis:
    helicity_bases = {
        decay_basis
        for node_type, _, decay_basis in classifications
        if node_type != "production"
    }
    if len(helicity_bases) != 1:
        msg = (
            "Cannot determine whether the decay node uses helicity couplings or LS"
            " couplings from the coupling symbols in the model"
        )
        raise ValueError(msg)
    return "helicity" if helicity_bases.pop() else "LS"


def _get_decay_coupling_basis(model: AmplitudeModel) -> CouplingBasis:
    for helicity_basis in (False, True):
        base = _get_coupling_base(helicity_basis, "decay")
        for symbol in model.parameter_defaults:
            if isinstance(symbol, sp.Indexed) and symbol.base == base:
                return "helicity" if helicity_basis else "LS"
    for chain in model.decay.chains:
        latex = Str(chain.resonance.latex)
        for production_basis, decay_basis in itertools.product((False, True), repeat=2):
            base = _get_coefficient_base(latex, production_basis, decay_basis)
            for symbol in model.parameter_defaults:
                if isinstance(symbol, sp.Indexed) and symbol.base == base:
                    return "helicity" if decay_basis else "LS"
    return "LS"


def _get_final_state(
    decay: ThreeBodyDecay | ThreeBodyDecayChain,
) -> dict[FinalStateID, State[FinalStateID]]:
    if isinstance(decay, ThreeBodyDecay):
        return decay.final_state
    if isinstance(decay, ThreeBodyDecayChain):
        return {state.index: state for state in decay.final_state}
    msg = f"Cannot determine the final state of a {type(decay).__name__}"
    raise NotImplementedError(msg)


def _get_particle_db(particle_db: ParticleCollection | None) -> ParticleCollection:
    if particle_db is None:
        return _load_default_particles()
    return particle_db


@lru_cache(maxsize=1)
def _load_default_particles() -> ParticleCollection:
    from ampform_dpd.adapter.qrules import (  # ruff: ignore[import-outside-top-level]
        load_particles,
    )

    return load_particles()
