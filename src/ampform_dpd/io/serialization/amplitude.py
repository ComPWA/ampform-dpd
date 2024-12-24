from __future__ import annotations

from collections import abc
from itertools import product
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import sympy as sp
from ampform.sympy import PoolSum, unevaluated
from sympy.functions.special.tensor_functions import (
    KroneckerDelta as δ,  # noqa: N813, PLC2403
)
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.spin import Rotation as Wigner

from ampform_dpd import (
    AmplitudeModel,  # pyright:ignore[reportPrivateUsage]
    _AlignmentWignerGenerator,  # pyright:ignore[reportPrivateUsage]
    _generate_amplitude_index_bases,  # pyright:ignore[reportPrivateUsage]
    create_mass_symbol_mapping,
    formulate_invariants,  # pyright:ignore[reportPrivateUsage]
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
    get_reference_topology,
)
from ampform_dpd.spin import create_spin_range

if TYPE_CHECKING:
    from ampform_dpd.decay import FinalStateID
    from ampform_dpd.io.serialization.format import ModelDefinition


def formulate(  # noqa: PLR0914
    model: ModelDefinition,
    cleanup_summations: bool = False,
    to_latex: Callable[[str], str] = identity_function,
    additional_builders: dict[str, PropagatorDynamicsBuilder] | None = None,
) -> AmplitudeModel:
    states = get_states(model)
    helicity_symbols = sp.symbols("lambda(:4)", rational=True)
    allowed_helicities = {
        symbol: create_spin_range(states[i].spin)  # type:ignore[index]
        for i, symbol in enumerate(helicity_symbols)
    }
    amplitude_definitions = {}  # type:ignore[var-annotated]
    angle_definitions = {}
    parameter_defaults = {}
    n_chains = len(get_decay_chains(model))
    helicity_values: tuple[sp.Rational, sp.Rational, sp.Rational, sp.Rational]
    for helicity_values in product(*allowed_helicities.values()):  # type:ignore[assignment]
        for chain_idx in range(n_chains):
            amp_defs = formulate_chain_amplitude(
                *helicity_values, model, chain_idx, to_latex, additional_builders
            )
            (amp_symbol, amp_expr), *parameters, (θij, θij_expr) = amp_defs.items()
            if not isinstance(amp_expr, sp.Expr):
                msg = f"Expected an expression, got {amp_expr!r}"
                raise TypeError(msg)
            helicity_substitutions = dict(zip(helicity_symbols, helicity_values))
            existing_amplitude = amplitude_definitions.get(amp_symbol, sp.Integer(0))
            existing_amplitude += amp_expr.subs(helicity_substitutions)
            amplitude_definitions[amp_symbol] = existing_amplitude
            angle_definitions[θij] = θij_expr
            parameter_defaults.update(dict(parameters))
    aligned_amp, zeta_defs = formulate_aligned_amplitude(model, *helicity_symbols)
    angle_definitions.update(zeta_defs)
    decay = to_decay(model)
    masses = create_mass_symbol_mapping(decay)
    parameter_defaults.update(masses)
    if cleanup_summations:
        aligned_amp = aligned_amp.cleanup()  # type:ignore[assignment]
    intensity = PoolSum(
        sp.Abs(aligned_amp) ** 2,
        *allowed_helicities.items(),
    )
    if cleanup_summations:
        intensity = intensity.cleanup()  # type:ignore[assignment]
    return AmplitudeModel(
        decay=decay,
        intensity=intensity,
        amplitudes=amplitude_definitions,  # type:ignore[arg-type]
        variables=angle_definitions,  # type:ignore[arg-type]
        parameter_defaults=parameter_defaults,  # type:ignore[arg-type]
        masses=masses,
        invariants=formulate_invariants(decay),
    )


def formulate_chain_amplitude(  # noqa: PLR0914, PLR0917
    λ0: sp.Rational,
    λ1: sp.Rational,
    λ2: sp.Rational,
    λ3: sp.Rational,
    model: ModelDefinition,
    chain_idx: int,
    to_latex: Callable[[str], str] = identity_function,
    additional_builders: dict[str, PropagatorDynamicsBuilder] | None = None,
) -> dict[sp.Symbol, complex | float | sp.Expr]:
    chain_defs = get_decay_chains(model)
    chain_definition = chain_defs[chain_idx]
    # -----------------------
    dynamics = formulate_dynamics(
        chain_definition, model, to_latex, additional_builders
    )
    for vertex in chain_definition["vertices"]:
        dynamics *= formulate_form_factor(vertex, model)
    # -----------------------
    weight, weight_val = _get_weight(chain_definition, to_latex)
    # -----------------------
    (i, λi_val), (j, λj_val) = _get_decay_product_helicities(chain_definition)
    θij, θij_expr = formulate_scattering_angle(i, j)
    jR = sp.Rational(chain_definition["propagators"][0]["spin"])  # noqa: N806
    R_node, λR_val = _get_resonance_helicity(chain_definition)  # noqa: N806
    λR = _get_helicity_symbol(R_node)
    # -----------------------
    A = _generate_amplitude_index_bases()
    subsystem_id = get_spectator_id(chain_definition["topology"])
    h_prod = formulate_recoupling(model, chain_idx, vertex_idx=0)
    h_dec = formulate_recoupling(model, chain_idx, vertex_idx=1)
    amplitude_expression = (
        weight
        * h_prod
        * h_dec
        * Wigner.d(jR, λR, λi_val - λj_val, θij)
        * dynamics.expression
    )
    amplitude_expression = amplitude_expression.subs({λR: λR_val})
    amplitude_symbol = A[subsystem_id][λ0, λ1, λ2, λ3]
    return {
        amplitude_symbol: amplitude_expression,
        weight: weight_val,
        **dynamics.definitions,
        θij: θij_expr,
    }


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
            return tuple((i, sp.Rational(λ)) for i, λ in zip(node, helicities))  # type:ignore[assignment,call-overload,return-value]
    msg = "Could not fine a helicity for any resonance node"
    raise ValueError(msg)


def formulate_aligned_amplitude(
    model: ModelDefinition,
    λ0: sp.Rational | sp.Symbol,
    λ1: sp.Rational | sp.Symbol,
    λ2: sp.Rational | sp.Symbol,
    λ3: sp.Rational | sp.Symbol,
) -> tuple[PoolSum, dict[sp.Symbol, sp.Expr]]:
    reference_topology = get_reference_topology(model)
    reference_subsystem = get_spectator_id(reference_topology)
    wigner_generator = _AlignmentWignerGenerator(reference_subsystem)
    _λ0, _λ1, _λ2, _λ3 = sp.symbols(R"\lambda_(:4)^{\prime}", rational=True)
    states = get_states(model)
    j0, j1, j2, j3 = (states[i].spin for i in sorted(states))
    A = _generate_amplitude_index_bases()
    amp_expr = PoolSum(
        sum(
            A[k][_λ0, _λ1, _λ2, _λ3]
            * wigner_generator(j0, λ0, _λ0, rotated_state=0, aligned_subsystem=k)
            * wigner_generator(j1, _λ1, λ1, rotated_state=1, aligned_subsystem=k)
            * wigner_generator(j2, _λ2, λ2, rotated_state=2, aligned_subsystem=k)
            * wigner_generator(j3, _λ3, λ3, rotated_state=3, aligned_subsystem=k)
            for k in get_existing_subsystem_ids(model)
        ),
        (_λ0, create_spin_range(j0)),
        (_λ1, create_spin_range(j1)),
        (_λ2, create_spin_range(j2)),
        (_λ3, create_spin_range(j3)),
    )
    return amp_expr, wigner_generator.angle_definitions  # type:ignore[return-value]


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
        if helicities is None:  # pyright:ignore[reportUnnecessaryComparison]
            msg = "Vertex does not contain helicities. Is it an LS vertex?"
            raise ValueError(msg, vertex)
        for helicity, sub_node in zip(helicities, node):
            if isinstance(sub_node, abc.Sequence) and len(sub_node) == 2:  # noqa: PLR2004
                return tuple(sub_node), sp.Rational(helicity)  # type:ignore[return-value]
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
        if helicities is None:  # pyright:ignore[reportUnnecessaryComparison]
            msg = "Vertex does not contain helicities. Is it an LS vertex?"
            raise ValueError(msg, vertex)  # type:ignore[index]
        for helicity, node in zip(helicities, vertex["node"]):
            if not isinstance(node, int):
                continue
            collected_helicities[node] = sp.Rational(helicity)
    return {i: collected_helicities[i] for i in sorted(collected_helicities)}


def formulate_recoupling(  # noqa: PLR0914
    model: ModelDefinition, chain_idx: int, vertex_idx: int
) -> sp.Expr:
    chain_definition = get_decay_chains(model)[chain_idx]
    vertex_definitions = chain_definition["vertices"]
    if len(vertex_definitions) != 2:  # noqa: PLR2004
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
            return ParityRecoupling(λa, λb, λa0, λb0, f)
        return HelicityRecoupling(λa, λb, λa0, λb0)
    if vertex_type == "ls":
        vertex = cast("LSVertex", vertex)
        l = int(vertex["l"])
        s = sp.Rational(vertex["s"])
        ja, jb = _get_child_spins(model, chain_idx, vertex_idx)
        j = _get_parent_spin(model, chain_idx, vertex_idx)
        return LSRecoupling(λa, λb, l, s, ja, jb, j)
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
    return tuple(spins)


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
    _latex_repr_ = R"\mathcal{{H}}^\text{{helicity}}\left({λa},{λb}|{λa0},{λb0}\right)"

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
        R"\mathcal{{H}}^\text{{parity}}\left({λa},{λb}|{λa0},{λb0},{f}\right)"
    )

    def evaluate(self) -> sp.Expr:
        λa, λb, λa0, λb0, f = self.args
        return δ(λa, λa0) * δ(λb, λb0) + f * δ(λa, -λa0) * δ(λb, -λb0)  # type:ignore[operator]


@unevaluated
class LSRecoupling(sp.Expr):
    λa: Any
    λb: Any
    l: Any
    s: Any
    ja: Any
    jb: Any
    j: Any
    _latex_repr_ = (
        R"\mathcal{{H}}^\text{{parity}}\left({λa},{λb}|{l},{s},{ja},{jb},{j}\right)"
    )

    def evaluate(self) -> sp.Expr:
        λa, λb, l, s, ja, jb, j = self.args
        return (
            sp.sqrt((2 * l + 1) / (2 * j + 1))  # type:ignore[operator]
            * CG(ja, λa, jb, -λb, s, λa - λb)  # type:ignore[operator]
            * CG(l, 0, s, λa - λb, j, λa - λb)  # type:ignore[operator]
        )
