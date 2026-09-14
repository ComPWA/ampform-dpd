r"""Bose-Einstein symmetrization for decays with identical final-state particles.

If a three-body decay has two (or three) identical particles in its final state, the
amplitude has to be symmetric under an exchange of those particles (antisymmetric if they
are fermions). The exchange permutes the final-state IDs and therefore maps each decay
chain onto the chain of the *same* resonance in another subsystem. The two chains are then
**one** model component with a single set of couplings and a fixed relative sign :math:`s`,

.. math:: \mathcal{H}\left[R,k',\dots\right] = s \, \mathcal{H}\left[R,k,\dots\right].
    :label: exchange-coupling-relation

The sign is the product of two factors,

.. math:: s = \underbrace{(-1)^{2s_a}}_\text{statistics} \; \underbrace{\eta}_\text{re-ordered vertex} \, ,
    :label: exchange-coupling-sign

where :math:`s_a` is the spin of the exchanged particles, so that the first factor is
:math:`+1` for identical bosons (Bose-Einstein) and :math:`-1` for identical fermions
(Fermi-Dirac). The second factor is the exchange phase of the **decay** vertex and appears
whenever the exchange delivers the two decay products in a different order than the one in
which the chain is written. In the cyclic pair ordering :math:`(23)1, (31)2, (12)3` of `the
DPD paper <https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.034033>`_ (Eq. 7),
that is almost always the case: for a final state :math:`1=a, 2=a, 3=b`, subsystem 1 writes
the isobar as :math:`(ab)` and subsystem 2 writes it as :math:`(ba)`. The production vertex
:math:`0 \to R\,k` always matches, because the resonance is always listed first.

The exchange phase depends on the basis in which the couplings are defined,

.. math::
    \eta^\text{LS} = (-1)^{l+s_i+s_j-S} \, , \qquad
    \eta^\text{helicity} = (-1)^{J_R-s_i-s_j} \, ,
    :label: bose-exchange-phase

with :math:`(l, S)` the LS coupling of :math:`R \to i\,j` and :math:`J_R` the spin of the
resonance. For the textbook case of :math:`D^+ \to \pi^+\pi^+\pi^-`, where every particle is
spinless, this collapses to :math:`s = (-1)^{J_R}`: the P-wave :math:`\rho(770)` enters the
two subsystems with a **relative minus sign**, the S-wave :math:`f_0(980)` with a plus sign.
Without it, :math:`\cos\theta_{31}(\sigma_1,\sigma_2) = -\cos\theta_{23}(\sigma_2,\sigma_1)`
and :math:`P_J(-z) = (-1)^J P_J(z)` make the :math:`\rho(770)` come out antisymmetric.

.. warning:: The criterion this module is validated against is the symmetry of the
    **intensity**, :math:`I(\sigma_1,\sigma_2,\sigma_3) = I(\sigma_2,\sigma_1,\sigma_3)`,
    which is the observable Dalitz-plot density. Equation :eq:`exchange-coupling-sign` fixes
    the *relative* signs of the decay chains, which is what that criterion is sensitive to;
    an overall sign of the amplitude is not observable.

    It is tempting to check the **amplitude** instead, but a naive
    :math:`A_{\lambda_0\lambda_1\lambda_2\lambda_3}(\sigma_1,\sigma_2,\sigma_3) =
    A_{\lambda_0\lambda_2\lambda_1\lambda_3}(\sigma_2,\sigma_1,\sigma_3)` does **not** hold
    in the DPD conventions, even for a correct model. The exchange mirrors the decay plane,
    so every alignment angle :math:`\zeta^i_{j(k)}` with :math:`j \neq k` flips sign, and
    the alignment Wigner-:math:`d` functions pick up the corresponding index reversal. The
    effect cancels in the intensity but not in a single helicity amplitude, and it is
    invisible for a decay in which every particle is spinless.

.. warning:: The statistics factor :math:`(-1)^{2s_a}` is imposed from first principles and
    is **not** validated by the intensity criterion above. It multiplies every tied chain by
    the same sign, so flipping it turns the symmetric combination into the antisymmetric one
    -- and both have a mirror-symmetric Dalitz plot. Only the exchange phase
    :math:`\eta`, which varies from resonance to resonance, is pinned by that test. The
    boson case (:math:`+1`) is exercised by the test suite; the fermion case is not, because
    no three-body decay with two identical fermions in the final state is available to test
    it against.

.. seealso:: :func:`~ampform_dpd.adapter.qrules.permute_equal_final_states`, which generates
    the permuted decay chains in the first place -- :mod:`qrules` itself produces only one
    of them.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal
from warnings import warn

import attrs
import sympy as sp
from attrs import frozen

from ampform_dpd import (
    _get_coefficient_base,
    _get_coupling_base,
    create_resonance_label,
)
from ampform_dpd.decay import (
    FinalStateID,
    LSCoupling,
    State,
    ThreeBodyDecay,
    ThreeBodyDecayChain,
    get_decay_product_ids,
)

if TYPE_CHECKING:
    from sympy.core.symbol import Str

    from ampform_dpd import AmplitudeModel

CouplingBasis = Literal["LS", "helicity"]
"""Basis in which the couplings of an isobar vertex are defined."""

StateMap = Mapping[FinalStateID, FinalStateID]
"""Permutation of the final-state IDs that is induced by a particle exchange."""


@frozen
class ExchangeTie:
    """Two decay chains that an exchange of identical particles relates.

    The :attr:`chain` is the one to be expressed in terms of the :attr:`reference`, as in
    Equation :eq:`exchange-coupling-relation`.
    """

    reference: ThreeBodyDecayChain
    chain: ThreeBodyDecayChain
    state_map: StateMap
    """The transposition that maps `reference` onto `chain`."""


def get_identical_state_ids(decay: ThreeBodyDecay) -> list[tuple[FinalStateID, ...]]:
    """Groups of final-state IDs that carry the same particle.

    >>> import qrules
    >>> from ampform_dpd.adapter.qrules import normalize_state_ids, to_three_body_decay
    >>> reaction = qrules.generate_transitions(
    ...     initial_state="D+",
    ...     final_state=["pi+", "pi+", "pi-"],
    ...     allowed_intermediate_particles=["rho(770)0"],
    ...     formalism="helicity",
    ...     mass_conservation_factor=0,
    ... )
    >>> decay = to_three_body_decay(normalize_state_ids(reaction).transitions)
    >>> get_identical_state_ids(decay)
    [(1, 2)]
    """
    state_ids = defaultdict(list)
    for i, state in decay.final_state.items():
        state_ids[state.name].append(i)
    return sorted(tuple(sorted(ids)) for ids in state_ids.values() if len(ids) > 1)


def has_identical_particles(decay: ThreeBodyDecay) -> bool:
    """Whether the final state of the decay contains identical particles."""
    return bool(get_identical_state_ids(decay))


def get_statistics_sign(state: State) -> int:
    r"""Sign :math:`s` that an exchange of two identical `.State` objects imposes.

    This is Equation :eq:`exchange-coupling-sign`: :math:`+1` for bosons (Bose-Einstein
    statistics) and :math:`-1` for fermions (Fermi-Dirac statistics).
    """
    return -1 if int(2 * state.spin) % 2 else 1


def get_exchange_ties(decay: ThreeBodyDecay) -> list[ExchangeTie]:
    """Pairs of decay chains that an exchange of identical particles ties together.

    Chains are grouped by their resonance **and** their LS couplings, so that a tie relates
    the same model component in two subsystems. The chain in the lowest-numbered subsystem
    is the `~ExchangeTie.reference`.
    """
    group_of = {i: g for g in get_identical_state_ids(decay) for i in g}
    ties = []
    for chains in _group_chains(decay):
        if len(chains) < 2:  # ruff: ignore[magic-value-comparison]
            continue
        reference_id = min(chains)
        for subsystem_id in sorted(chains):
            if subsystem_id == reference_id:
                continue
            if subsystem_id not in group_of.get(reference_id, ()):
                resonance = chains[subsystem_id].resonance.name
                msg = (
                    f"Resonance {resonance} occurs in subsystems {reference_id} and"
                    f" {subsystem_id}, but states {reference_id} and {subsystem_id} are"
                    " not identical particles, so the chains cannot be related by an"
                    " exchange of identical particles"
                )
                raise ValueError(msg)
            ties.append(
                ExchangeTie(
                    reference=chains[reference_id],
                    chain=chains[subsystem_id],
                    state_map=_create_transposition(reference_id, subsystem_id),
                )
            )
    return ties


def get_exchange_phase(
    chain: ThreeBodyDecayChain, basis: CouplingBasis = "helicity"
) -> int:
    r"""Phase for writing the decay products of a chain in the opposite order.

    A two-particle state is a constructed object: one of the two particles is listed first
    and defines the direction of the relative momentum, and the spins are coupled in the
    listed order. Listing the same two particles the other way round describes the same
    state in a different basis, at the cost of the phase of Equation
    :eq:`bose-exchange-phase`.

    .. note:: This function is identical to ``cparity.get_exchange_phase`` of `PR #203
        <https://github.com/ComPWA/ampform-dpd/pull/203>`_, which solves the same problem
        for charge conjugation. The two should be merged into a shared location once both
        are on the main branch.
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


def is_reordered(tie: ExchangeTie) -> bool:
    """Whether the exchange delivers the decay products in the opposite order.

    Both chains of an `ExchangeTie` are written in the cyclic pair ordering of the DPD
    paper. This function checks whether the exchange maps the decay products of the
    reference chain onto that ordering (`False`) or onto the reverse of it (`True`), in
    which case the couplings pick up the phase of :func:`get_exchange_phase`.
    """
    i0, j0 = get_decay_product_ids(tie.reference.spectator.index)
    i, j = get_decay_product_ids(tie.chain.spectator.index)
    mapped = (tie.state_map[i0], tie.state_map[j0])
    if mapped == (i, j):
        return False
    if mapped == (j, i):
        return True
    msg = (
        f"Exchange maps decay products {i0}, {j0} onto {mapped[0]}, {mapped[1]}, which is"
        f" not a permutation of the decay products {i}, {j} of subsystem"
        f" {tie.chain.spectator.index}"
    )
    raise ValueError(msg)


def get_exchange_sign(tie: ExchangeTie, basis: CouplingBasis = "helicity") -> int:
    r"""Relative sign :math:`s` between the couplings of two tied decay chains.

    Computes Equation :eq:`exchange-coupling-sign`: the statistics sign of the exchanged
    particles, times the exchange phase of the decay vertex if -- and only if -- the
    exchange re-orders it.
    """
    sign = get_statistics_sign(tie.chain.spectator)
    if is_reordered(tie):
        sign *= get_exchange_phase(tie.chain, basis)
    return sign


def get_missing_exchange_partners(
    decay: ThreeBodyDecay,
) -> list[tuple[ThreeBodyDecayChain, FinalStateID]]:
    """Chains whose exchange partner is absent from the decay.

    Each entry is a chain together with the subsystem in which its partner is missing. A
    non-empty result means that the model cannot be made symmetric under an exchange of the
    identical final-state particles, which usually means that
    :func:`~ampform_dpd.adapter.qrules.permute_equal_final_states` was not applied to the
    `~qrules.transition.ReactionInfo` object.
    """
    missing = []
    for group in get_identical_state_ids(decay):
        for chains in _group_chains(decay):
            present = set(chains) & set(group)
            if not present:
                continue
            reference = chains[min(present)]
            missing.extend(
                (reference, subsystem_id)
                for subsystem_id in sorted(set(group) - present)
            )
    return missing


def warn_about_missing_exchange_partners(decay: ThreeBodyDecay) -> None:
    """Emit a `UserWarning` for each result of :func:`get_missing_exchange_partners`."""
    missing = get_missing_exchange_partners(decay)
    if not missing:
        return
    description = ", ".join(
        f"{chain.resonance.name} in subsystem {subsystem_id}"
        for chain, subsystem_id in missing
    )
    msg = (
        "The final state contains identical particles, but the following decay chains have"
        f" no exchange partner: {description}. The model can therefore not be made"
        " symmetric under an exchange of the identical particles. Did you forget to call"
        " permute_equal_final_states() on the reaction?"
    )
    warn(msg, category=UserWarning, stacklevel=3)


def relate_exchange_couplings(model: AmplitudeModel) -> dict[sp.Indexed, sp.Expr]:
    r"""Substitutions that express tied couplings in terms of their exchange partners.

    The returned mapping sends each coupling of the `~ExchangeTie.chain` of an
    `ExchangeTie` onto :math:`\pm` the corresponding coupling of its
    `~ExchangeTie.reference`, with the sign of Equation :eq:`exchange-coupling-sign`.

    Only the product of the couplings along a chain is observable, so which of the two
    vertices carries the sign is a choice. It is put on the vertex that identifies the
    decay chain uniquely: on the **production** coupling in the helicity basis, where the
    sign depends on nothing but the spins, and on the **decay** coupling in the LS basis,
    where it depends on :math:`(l, S)` and several LS combinations of one resonance can
    share a production coupling.

    Use :func:`symmetrize_identical_particles` to apply these substitutions to a model.
    """
    couplings = _collect_couplings(model)
    basis = _get_decay_coupling_basis(model)
    sign_carrier = "production" if basis == "helicity" else "decay"
    substitutions: dict[sp.Indexed, sp.Expr] = {}
    for tie in get_exchange_ties(model.decay):
        sign = get_exchange_sign(tie, basis)
        reordered = is_reordered(tie)
        label = create_resonance_label(model.decay, tie.chain)
        reference_label = create_resonance_label(model.decay, tie.reference)
        for coupling in couplings:
            match = _match_coupling(coupling, label, tie.chain)
            if match is None:
                continue
            target = _relabel_coupling(
                coupling,
                label,
                reference_label,
                match,
                reverse_decay_indices=reordered and basis == "helicity",
            )
            substituted = (
                sign * target if match in {sign_carrier, "coefficient"} else target
            )
            _register_substitution(substitutions, coupling, substituted)
    return substitutions


def _register_substitution(
    substitutions: dict[sp.Indexed, sp.Expr], coupling: sp.Indexed, value: sp.Expr
) -> None:
    existing = substitutions.get(coupling)
    if existing is not None and existing != value:
        msg = (
            f"Coupling {coupling} is tied to two different expressions, {existing} and"
            f" {value}. The decay chains of this model cannot be symmetrized in this"
            " coupling basis."
        )
        raise ValueError(msg)
    substitutions[coupling] = value


def symmetrize_identical_particles(model: AmplitudeModel) -> AmplitudeModel:
    """Impose the symmetry of the final state on the couplings of an `.AmplitudeModel`.

    Substitutes the couplings of each tied decay chain by :math:`\\pm` the couplings of its
    exchange partner (see :func:`relate_exchange_couplings`) and removes them from the
    parameter defaults, so that the two chains share one set of free parameters. The
    resulting amplitude is symmetric under an exchange of the identical final-state
    particles (antisymmetric if they are fermions).

    .. note:: This is applied by :meth:`.DalitzPlotDecompositionBuilder.formulate` by
        default, so there is usually no need to call it yourself.
    """
    substitutions = relate_exchange_couplings(model)
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


def _group_chains(
    decay: ThreeBodyDecay,
) -> list[dict[FinalStateID, ThreeBodyDecayChain]]:
    """Group the chains of a decay by resonance and LS couplings, keyed by subsystem."""
    groups: dict[tuple, dict[FinalStateID, ThreeBodyDecayChain]] = defaultdict(dict)
    for chain in decay.chains:
        key = (chain.resonance, chain.incoming_ls, chain.outgoing_ls)
        groups[key][chain.spectator.index] = chain
    return [groups[key] for key in sorted(groups, key=str)]


def _create_transposition(i: FinalStateID, j: FinalStateID) -> StateMap:
    k, *_ = {1, 2, 3} - {i, j}
    return {i: j, j: i, k: k}  # ty: ignore[invalid-return-type]


def _collect_couplings(model: AmplitudeModel) -> list[sp.Indexed]:
    """Collect all coupling symbols that occur in a model.

    The amplitude expressions of an `.AmplitudeModel` contain couplings whose helicity
    indices are still the symbolic summation variable of a `~ampform.sympy.PoolSum`, while
    the parameter defaults contain the same couplings with the summation variable
    substituted by each of its values. Both forms have to be related to their exchange
    partner, so that the substitutions can be applied before as well as after the sums are
    evaluated.
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


def _get_decay_coupling_basis(model: AmplitudeModel) -> CouplingBasis:
    """Basis in which the couplings of the decay vertices of a model are defined."""
    ls_base = _get_coupling_base(False, "decay")
    for coupling in _collect_couplings(model):
        if coupling.base == ls_base:
            return "LS"
        for chain in model.decay.chains:
            label = create_resonance_label(model.decay, chain)
            for production in (True, False):
                if coupling.base == _get_coefficient_base(label, production, False):
                    return "LS"
    return "helicity"


def _match_coupling(  # ruff: ignore[too-many-return-statements]
    coupling: sp.Indexed, label: Str, chain: ThreeBodyDecayChain
) -> Literal["production", "decay", "coefficient"] | None:
    """Determine which vertex of ``chain`` a coupling belongs to, if any.

    Returns `None` if the coupling belongs to another chain. An LS coupling is only
    matched if its :math:`(l, S)` indices are those of the corresponding vertex of the
    chain, because several LS combinations of one resonance share a resonance label.
    """
    for typ in ("production", "decay"):
        if coupling.base == _get_coupling_base(True, typ):
            return typ if coupling.indices[0] == label else None
        if coupling.base == _get_coupling_base(False, typ):
            if coupling.indices[0] != label:
                return None
            ls = chain.incoming_ls if typ == "production" else chain.outgoing_ls
            return typ if _matches_ls(coupling.indices[1:], ls) else None
    for production, decay in itertools.product((True, False), repeat=2):
        if coupling.base != _get_coefficient_base(label, production, decay):
            continue
        indices = coupling.indices
        if not production and not _matches_ls(indices[:2], chain.incoming_ls):
            return None
        if not decay and not _matches_ls(indices[2:], chain.outgoing_ls):
            return None
        return "coefficient"
    return None


def _matches_ls(indices: tuple[sp.Basic, ...], ls: LSCoupling | None) -> bool:
    if ls is None:
        return False
    return tuple(indices) == (sp.sympify(ls.L), sp.sympify(ls.S))


def _relabel_coupling(
    coupling: sp.Indexed,
    label: Str,
    reference_label: Str,
    kind: Literal["production", "decay", "coefficient"],
    reverse_decay_indices: bool,
) -> sp.Indexed:
    """Rewrite a coupling of one chain as the same coupling of its exchange partner.

    Both chains of an `ExchangeTie` are written in the cyclic pair ordering of their own
    subsystem, and when the exchange re-orders the decay vertex (see :func:`is_reordered`)
    those two orderings are each other's reverse. The helicity indices of the **decay**
    coupling then have to be reversed as well, because its first slot is the helicity of a
    different particle in each of the two chains. Getting this wrong produces a coupling
    symbol that the partner chain never uses, which silently leaves the two chains with
    independent parameters instead of tying them.

    The indices only have to be reversed if they are helicities. LS couplings are indexed
    by :math:`(l, S)` and are the same either way round.
    """
    if kind == "production":
        return coupling.base[(reference_label, *coupling.indices[1:])]
    if kind == "decay":
        indices = coupling.indices[1:]
        if reverse_decay_indices:
            indices = tuple(reversed(indices))
        return coupling.base[(reference_label, *indices)]
    for production, decay in itertools.product((True, False), repeat=2):
        if coupling.base != _get_coefficient_base(label, production, decay):
            continue
        production_indices = coupling.indices[:2]
        decay_indices = coupling.indices[2:]
        if decay and reverse_decay_indices:
            decay_indices = tuple(reversed(decay_indices))
        base = _get_coefficient_base(reference_label, production, decay)
        return base[(*production_indices, *decay_indices)]
    msg = f"Cannot relabel coupling {coupling}"
    raise NotImplementedError(msg)
