from __future__ import annotations

from collections import abc
from itertools import product
from typing import TYPE_CHECKING, Any, Literal, cast

import sympy as sp
from ampform.sympy import PoolSum, unevaluated
from sympy.functions.special.tensor_functions import (
    KroneckerDelta as δ,  # ruff: ignore[camelcase-imported-as-lowercase, non-ascii-import-name]
)
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.spin import Rotation as Wigner

from ampform_dpd import (
    AmplitudeModel,
    _AlignmentWignerGenerator,
    _generate_amplitude_index_bases,
    create_mass_symbol_mapping,
    formulate_invariants,
)
from ampform_dpd.angles import formulate_scattering_angle
from ampform_dpd.io.serialization.decay import (
    get_final_state,
    get_initial_state,
    get_spectator_id,
    get_states,
    to_decay,
)
from ampform_dpd.io.serialization.dynamics import (
    PropagatorDynamicsBuilder,
    formulate_dynamics,
    formulate_form_factor,
    identity_function,
)
from ampform_dpd.io.serialization.format import (
    DecayChain,
    HelicityVertex,
    LSVertex,
    Node,
    ParityFactor,
    ParityVertex,
    get_decay_chains,
    get_distribution_def,
)
from ampform_dpd.spin import create_spin_range

if TYPE_CHECKING:
    from collections.abc import Callable

    from ampform_dpd.decay import FinalStateID, State, StateID
    from ampform_dpd.io.serialization.format import ModelDefinition

_REFERENCE_SUBSYSTEMS: dict[StateID, FinalStateID] = {0: 1, 1: 1, 2: 2, 3: 3}
"""Reference subsystem to use for the alignment rotation of each state.

Each final state is aligned with respect to its own subsystem and the initial state with
respect to subsystem 1. This is the convention of `ThreeBodyDecays.jl
<https://github.com/mmikhasenko/ThreeBodyDecays.jl>`_, which the serialization format
follows, and it is used instead of the single reference topology that the serialized
model declares.
"""


def formulate(  # ruff: ignore[too-many-locals]
    model: ModelDefinition,
    cleanup_summations: bool = False,
    to_latex: Callable[[str], str] = identity_function,
    additional_builders: dict[str, PropagatorDynamicsBuilder] | None = None,
) -> AmplitudeModel:
    states = get_states(model)
    helicity_symbols = sp.symbols("lambda(:4)", rational=True)
    allowed_helicities = {
        symbol: create_spin_range(states[i].spin)  # ty: ignore[invalid-argument-type]
        for i, symbol in enumerate(helicity_symbols)
    }
    amplitude_definitions = {}
    angle_definitions = {}
    parameter_defaults = {}
    n_chains: int = len(get_decay_chains(model))
    for helicity_values in product(*allowed_helicities.values()):
        for chain_idx in range(n_chains):
            amp_defs = formulate_chain_amplitude(
                *helicity_values,  # ty: ignore[invalid-argument-type]
                model,  # ty: ignore[too-many-positional-arguments]
                chain_idx,
                to_latex,
                additional_builders,
            )
            (amp_symbol, amp_expr), *parameters, (θij, θij_expr) = amp_defs.items()
            if not isinstance(amp_expr, sp.Expr):
                msg = f"Expected an expression, got {amp_expr!r}"
                raise TypeError(msg)
            helicity_substitutions = dict(
                zip(helicity_symbols, helicity_values, strict=True)
            )
            existing_amplitude = amplitude_definitions.get(amp_symbol, sp.Integer(0))
            existing_amplitude += amp_expr.subs(helicity_substitutions)
            amplitude_definitions[amp_symbol] = existing_amplitude
            angle_definitions[θij] = θij_expr
            parameter_defaults.update(dict(parameters))
    aligned_amp, zeta_defs = formulate_aligned_amplitude(model, *helicity_symbols)
    angle_definitions.update(zeta_defs)  # ty: ignore[no-matching-overload]
    decay = to_decay(model)
    masses = create_mass_symbol_mapping(decay)
    parameter_defaults.update(masses)  # ty: ignore[no-matching-overload]
    if cleanup_summations:
        aligned_amp = aligned_amp.cleanup()
    intensity = PoolSum(
        sp.Abs(aligned_amp) ** 2,
        *allowed_helicities.items(),
    )
    if cleanup_summations:
        intensity = intensity.cleanup()
    return AmplitudeModel(
        decay=decay,
        intensity=intensity,
        amplitudes=amplitude_definitions,  # ty: ignore[invalid-argument-type]
        variables=angle_definitions,  # ty: ignore[invalid-argument-type]
        parameter_defaults=parameter_defaults,  # ty: ignore[invalid-argument-type]
        masses=masses,
        invariants=formulate_invariants(decay),
    )


def formulate_chain_amplitude(  # ruff: ignore[too-many-locals, too-many-positional-arguments]
    λ0: sp.Rational,
    λ1: sp.Rational,
    λ2: sp.Rational,
    λ3: sp.Rational,
    model: ModelDefinition,
    chain_idx: int,
    to_latex: Callable[[str], str] = identity_function,
    additional_builders: dict[str, PropagatorDynamicsBuilder] | None = None,
) -> dict[sp.Basic, complex | float | sp.Expr]:
    r"""Formulate the amplitude for one decay chain of a serialized model.

    This is the serialization counterpart of
    `.DalitzPlotDecompositionBuilder.formulate_subsystem_amplitude`: the couplings and
    dynamics are read from the model definition instead of being generated, but the
    phase conventions, the Kronecker delta over the production helicities, and the sum
    over the resonance helicity :math:`\lambda_R` are the same. The two implementations
    have to be kept in sync.
    """
    chain_defs = get_decay_chains(model)
    chain_definition = chain_defs[chain_idx]
    dynamics = formulate_dynamics(
        chain_definition, model, to_latex, additional_builders
    )
    for vertex in chain_definition["vertices"]:
        dynamics *= formulate_form_factor(vertex, model)
    weight, weight_val = _get_weight(chain_definition, to_latex)
    i, j = _get_decay_product_ids(chain_definition)
    θij, θij_expr = formulate_scattering_angle(i, j)
    jR = sp.Rational(chain_definition["propagators"][0]["spin"])  # ruff: ignore[non-lowercase-variable-in-function]
    λR = _get_helicity_symbol(_get_resonance_node(chain_definition))
    A = _generate_amplitude_index_bases()
    spectator_id = get_spectator_id(chain_definition["topology"])
    states = get_states(model)
    helicities = (λ0, λ1, λ2, λ3)
    h_prod = formulate_recoupling(model, chain_idx, vertex_idx=0)
    h_dec = formulate_recoupling(model, chain_idx, vertex_idx=1)
    chain_amplitude = (
        weight
        * sp.sqrt(2 * jR + 1)
        * _formulate_phase_factor(states[spectator_id], helicities[spectator_id])
        * _formulate_phase_factor(states[j], helicities[j])
        * δ(λ0, λR - helicities[spectator_id])
        * h_prod
        * h_dec
        * Wigner.d(jR, λR, helicities[i] - helicities[j], θij)
        * dynamics.expression
    )
    amplitude_expression = PoolSum(chain_amplitude, (λR, create_spin_range(jR)))
    amplitude_symbol = A[spectator_id][λ0, λ1, λ2, λ3]
    return {
        amplitude_symbol: amplitude_expression,
        weight: weight_val,
        **dynamics.parameters,
        θij: θij_expr,
    }


def _formulate_phase_factor(state: State, helicity: sp.Rational | sp.Symbol) -> sp.Expr:
    r"""Formulate the :math:`(-1)^{j-\lambda}` phase factor of a state."""
    return (-1) ** (state.spin - helicity)


def _get_decay_product_ids(
    chain_definition: DecayChain,
) -> tuple[FinalStateID, FinalStateID]:
    """Get the IDs of the two decay products, ignoring their serialized helicities.

    The helicity values from `._get_decay_product_helicities` are not substituted into
    the chain amplitude: it is summed over all allowed helicities instead.
    """
    (i, _), (j, _) = _get_decay_product_helicities(chain_definition)
    return cast("FinalStateID", i), cast("FinalStateID", j)


def _get_resonance_node(
    chain_definition: DecayChain,
) -> tuple[FinalStateID, FinalStateID]:
    """Get the node of the resonance, ignoring its serialized helicity.

    See `._get_decay_product_ids` for why the helicity value is discarded.
    """
    node, _ = _get_resonance_helicity(chain_definition)
    return node


def _get_decay_product_helicities(
    chain_definition: DecayChain,
) -> tuple[tuple[int, sp.Rational], tuple[int, sp.Rational]]:
    vertices = chain_definition["vertices"]
    for vertex in vertices:
        node = vertex["node"]
        if all(isinstance(i, int) for i in node):
            helicities = vertex.get("helicities")
            if helicities is None:
                msg = "Vertex does not contain helicities. Is it an LS vertex?"
                raise ValueError(msg, vertex)
            return tuple(
                (i, sp.Rational(λ)) for i, λ in zip(node, helicities, strict=True)
            )  # ty: ignore[invalid-return-type]
    msg = "Could not fine a helicity for any resonance node"
    raise ValueError(msg)


def formulate_aligned_amplitude(
    model: ModelDefinition,
    λ0: sp.Rational | sp.Symbol,
    λ1: sp.Rational | sp.Symbol,
    λ2: sp.Rational | sp.Symbol,
    λ3: sp.Rational | sp.Symbol,
) -> tuple[PoolSum, dict[sp.Symbol, sp.Expr]]:
    generators = {
        subsystem_id: _AlignmentWignerGenerator(subsystem_id)
        for subsystem_id in sorted(set(_REFERENCE_SUBSYSTEMS.values()))
    }
    wigner_generators = {
        rotated_state: generators[subsystem_id]
        for rotated_state, subsystem_id in _REFERENCE_SUBSYSTEMS.items()
    }
    _λ0, _λ1, _λ2, _λ3 = sp.symbols(R"\lambda_(:4)^{\prime}", rational=True)
    states = get_states(model)
    j0, j1, j2, j3 = (states[i].spin for i in sorted(states))
    A = _generate_amplitude_index_bases()
    amp_expr = PoolSum(
        sum(
            A[k][_λ0, _λ1, _λ2, _λ3]
            * wigner_generators[0](j0, λ0, _λ0, rotated_state=0, aligned_subsystem=k)
            * wigner_generators[1](j1, _λ1, λ1, rotated_state=1, aligned_subsystem=k)
            * wigner_generators[2](j2, _λ2, λ2, rotated_state=2, aligned_subsystem=k)
            * wigner_generators[3](j3, _λ3, λ3, rotated_state=3, aligned_subsystem=k)
            for k in get_existing_subsystem_ids(model)
        ),
        (_λ0, create_spin_range(j0)),
        (_λ1, create_spin_range(j1)),
        (_λ2, create_spin_range(j2)),
        (_λ3, create_spin_range(j3)),
    )
    angle_definitions = {
        symbol: expression
        for generator in generators.values()
        for symbol, expression in generator.angle_definitions.items()
    }
    return amp_expr, angle_definitions


def _get_weight(
    chain_definition: DecayChain, to_latex: Callable[[str], str] = identity_function
) -> tuple[sp.Symbol, complex | float]:
    value: complex | float
    value = complex(str(chain_definition["weight"]).replace(" ", "").replace("i", "j"))
    if not value.imag:
        value = value.real
    resonance_latex = to_latex(chain_definition["name"])
    _, resonance_helicity = _get_resonance_helicity(chain_definition)
    helicities = _get_final_state_helicities(chain_definition).values()
    subscript = ", ".join(sp.latex(λ) for λ in helicities)
    symbol = sp.Symbol(f"c^{{{resonance_latex}[{resonance_helicity}]}}_{{{subscript}}}")
    return symbol, value


def _get_resonance_helicity(
    chain_definition: DecayChain,
) -> tuple[tuple[FinalStateID, FinalStateID], sp.Rational]:
    vertices = chain_definition["vertices"]
    for vertex in vertices:
        node = vertex["node"]
        if all(isinstance(i, int) for i in node):
            continue
        vertex = cast("HelicityVertex", vertex)
        helicities = vertex.get("helicities")
        if helicities is None:
            msg = "Vertex does not contain helicities. Is it an LS vertex?"
            raise ValueError(msg, vertex)
        for helicity, sub_node in zip(helicities, node, strict=True):
            if isinstance(sub_node, abc.Sequence) and len(sub_node) == 2:  # ruff: ignore[magic-value-comparison]
                return tuple(sub_node), sp.Rational(helicity)
    msg = "Could not find a resonance node"
    raise ValueError(msg)


def _get_final_state_helicities(
    chain_definition: DecayChain,
) -> dict[FinalStateID, sp.Rational]:
    vertices = chain_definition["vertices"]
    collected_helicities: dict[FinalStateID, sp.Rational] = {}
    for vertex in vertices:
        vertex = cast("HelicityVertex", vertex)
        helicities = vertex.get("helicities")
        if helicities is None:
            msg = "Vertex does not contain helicities. Is it an LS vertex?"
            raise ValueError(msg, vertex)
        for helicity, node in zip(helicities, vertex["node"], strict=True):
            if not isinstance(node, int):
                continue
            collected_helicities[node] = sp.Rational(helicity)
    return {i: collected_helicities[i] for i in sorted(collected_helicities)}


def formulate_recoupling(  # ruff: ignore[too-many-locals]
    model: ModelDefinition, chain_idx: int, vertex_idx: int
) -> sp.Expr:
    chain_definition = get_decay_chains(model)[chain_idx]
    vertex_definitions = chain_definition["vertices"]
    if len(vertex_definitions) != 2:  # ruff: ignore[magic-value-comparison]
        msg = f"Not a three-body decay: there are {len(vertex_definitions)} vertices"
        raise ValueError(msg)
    if vertex_idx not in {0, 1}:
        msg = f"Vertex index out of range. Can either be 0 or 1, not {vertex_idx}."
        raise ValueError(msg)
    vertex = chain_definition["vertices"][vertex_idx]
    vertex_type = vertex["type"]
    node = vertex["node"]
    λa, λb = map(_get_helicity_symbol, node)
    if vertex_type in {"helicity", "parity"}:
        vertex = cast("HelicityVertex", vertex)
        λa0, λb0 = (sp.Rational(v) for v in vertex["helicities"])
        if vertex_type == "parity":
            vertex = cast("ParityVertex", vertex)
            f = _sign_to_value(vertex.get("parity_factor", "+"))
            return ParityRecoupling(λa, λb, λa0, λb0, f)  # ty: ignore[invalid-argument-type]
        return HelicityRecoupling(λa, λb, λa0, λb0)
    if vertex_type == "ls":
        vertex = cast("LSVertex", vertex)
        l = int(vertex["l"])
        s = sp.Rational(vertex["s"])
        ja, jb = _get_child_spins(model, chain_idx, vertex_idx)
        j = _get_parent_spin(model, chain_idx, vertex_idx)
        return LSRecoupling(λa, λb, l, s, ja, jb, j)  # ty: ignore[invalid-argument-type]
    msg = f"No implementation for vertex of type {vertex_type!r}"
    raise NotImplementedError(msg)


def _sign_to_value(sign: ParityFactor) -> Literal[0, -1, 1]:
    stripped_sign = sign.strip()
    if stripped_sign == "-":
        return -1
    if not stripped_sign:
        return 0
    if stripped_sign == "+":
        return +1
    msg = f"Cannot convert {sign!r} to value"
    raise NotImplementedError(msg)


def _get_parent_spin(
    model: ModelDefinition, chain_idx: int, vertex_idx: int
) -> sp.Rational:
    chain_definition = get_decay_chains(model)[chain_idx]
    vertex = chain_definition["vertices"][vertex_idx]
    if all(isinstance(i, int) for i in vertex["node"]):
        return __get_propagator_spin(chain_definition)
    initial_state = get_initial_state(model)
    return initial_state.spin


def _get_child_spins(
    model: ModelDefinition, chain_idx: int, vertex_idx: int
) -> tuple[sp.Rational, sp.Rational]:
    chain_definition = get_decay_chains(model)[chain_idx]
    vertex = chain_definition["vertices"][vertex_idx]
    node = vertex["node"]
    final_state = get_final_state(model)
    spins = []
    for node_item in node:
        if isinstance(node_item, int):
            spins.append(sp.Rational(final_state[node_item]))
        else:
            spins.append(__get_propagator_spin(chain_definition))
    return tuple(spins)  # ty: ignore[invalid-return-type]


def __get_propagator_spin(chain_definition: DecayChain) -> sp.Rational:
    propagators = chain_definition["propagators"]
    if len(propagators) != 1:
        msg = f"There are {len(propagators)} propagators, not a three-body decay"
        raise ValueError(msg)
    return sp.Rational(propagators[0]["spin"])


def _get_helicity_symbol(node: int | Node) -> sp.Symbol:
    if isinstance(node, int):
        return sp.Symbol(f"lambda{node}", rational=True)
    return sp.Symbol(R"\lambda_R", rational=True)


def get_existing_subsystem_ids(model: ModelDefinition) -> list[FinalStateID]:
    distribution_def = get_distribution_def(model)
    chain_defs = distribution_def["decay_description"]["chains"]
    subsystem_ids = {get_spectator_id(c["topology"]) for c in chain_defs}
    return sorted(subsystem_ids)


@unevaluated
class HelicityRecoupling(sp.Expr):
    λa: sp.Rational | sp.Symbol
    λb: sp.Rational | sp.Symbol
    λa0: sp.Rational | sp.Symbol
    λb0: sp.Rational | sp.Symbol
    _latex_repr_ = (
        R"\mathcal{{H}}^\text{{helicity}}\left({λa},{λb}\middle|{λa0},{λb0}\right)"
    )

    def evaluate(self) -> sp.Expr:
        λa, λb, λa0, λb0 = self.args
        return δ(λa, λa0) * δ(λb, λb0)


@unevaluated
class ParityRecoupling(sp.Expr):
    λa: Any
    λb: Any
    λa0: Any
    λb0: Any
    f: Any
    _latex_repr_ = (
        R"\mathcal{{H}}^\text{{parity}}\left({λa},{λb}\middle|{λa0},{λb0},{f}\right)"
    )

    def evaluate(self) -> sp.Expr:
        λa, λb, λa0, λb0, f = self.args
        if λa0 == 0 and λb0 == 0:
            return δ(λa, λa0) * δ(λb, λb0)
        return δ(λa, λa0) * δ(λb, λb0) + f * δ(λa, -λa0) * δ(λb, -λb0)  # ty: ignore[unsupported-operator]


@unevaluated
class LSRecoupling(sp.Expr):
    λa: Any
    λb: Any
    l: Any
    s: Any
    ja: Any
    jb: Any
    j: Any
    _latex_repr_ = R"\mathcal{{H}}^\text{{parity}}\left({λa},{λb}\middle|{l},{s},{ja},{jb},{j}\right)"

    def evaluate(self) -> sp.Expr:
        λa, λb, l, s, ja, jb, j = self.args
        return (
            sp.sqrt((2 * l + 1) / (2 * j + 1))  # ty: ignore[unsupported-operator]
            * CG(ja, λa, jb, -λb, s, λa - λb)  # ty: ignore[unsupported-operator]
            * CG(l, 0, s, λa - λb, j, λa - λb)  # ty: ignore[unsupported-operator]
        )
