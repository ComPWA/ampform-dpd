"""Module for formulating the amplitude model for a three-body decay using DPD."""

from __future__ import annotations

import functools
import operator
from collections import abc
from collections.abc import Callable
from functools import cache, wraps
from itertools import product
from typing import TYPE_CHECKING, Any
from warnings import warn

import attrs
import sympy as sp
from ampform.helicity import (
    ParameterValue,
    ParameterValues,
    _to_parameter_values,  # noqa: PLC2701
)
from ampform.kinematics.phasespace import compute_third_mandelstam
from ampform.sympy import PoolSum
from attrs import define, field, frozen
from sympy.core.symbol import Str
from sympy.physics.quantum.spin import CG, WignerD
from sympy.physics.quantum.spin import Rotation as Wigner

from ampform_dpd.angles import formulate_scattering_angle, formulate_zeta_angle
from ampform_dpd.decay import (
    FinalStateID,
    IsobarNode,
    LSCoupling,
    Particle,
    State,
    ThreeBodyDecay,
    ThreeBodyDecayChain,
    _get_decay_description,
    _get_subsystem_ids,
    get_decay_product_ids,
    to_particle,
)
from ampform_dpd.spin import create_spin_range

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any, Literal


@frozen
class AmplitudeModel:
    decay: ThreeBodyDecay
    intensity: sp.Expr = sp.S.One
    amplitudes: dict[sp.Indexed, sp.Expr] = field(factory=dict)
    variables: dict[sp.Basic, sp.Expr] = field(factory=dict)
    parameter_defaults: ParameterValues = field(
        converter=_to_parameter_values,
        factory=dict,
    )
    masses: dict[sp.Symbol, float] = field(factory=dict)
    invariants: dict[sp.Symbol, sp.Expr] = field(factory=dict)

    @property
    def full_expression(self) -> sp.Expr:
        evaluated_amplitudes = {k: v.doit() for k, v in self.amplitudes.items()}
        return self.intensity.doit().xreplace(evaluated_amplitudes)


class DalitzPlotDecompositionBuilder:
    def __init__(
        self,
        decay: ThreeBodyDecay,
        min_ls: bool | tuple[bool, bool] = True,
        all_subsystems: bool = False,
    ) -> None:
        """Amplitude builder for the helicity formalism with Dalitz-plot decomposition.

        Args:
            decay: The `.ThreeBodyDecay` over which to formulate the amplitude model.
            min_ls: Use helicity couplings instead of
                :math:`LS`-couplings. If setting this boolean with a `tuple`, the first
                element of the `tuple` defines whether to use helicity couplings on the
                **production** `.IsobarNode` and the second configures the **decay**
                `.IsobarNode`.
            all_subsystems: Formulate the amplitude model for all allowed subsystems in
                the decay, even if they do not exist in the `.ThreeBodyDecay` object.
        """
        self.decay = decay
        self.dynamics_choices = DynamicsConfigurator(decay)
        if isinstance(min_ls, bool):
            self.use_production_helicity_couplings = min_ls
            self.use_decay_helicity_couplings = min_ls
        elif isinstance(min_ls, tuple) and len(min_ls) == 2:  # noqa: PLR2004
            (
                self.use_production_helicity_couplings,
                self.use_decay_helicity_couplings,
            ) = min_ls
        else:
            msg = f"Cannot configure helicity couplings with a {type(min_ls).__name__}"
            raise NotImplementedError(msg, min_ls)
        self.all_subsystems = all_subsystems

    def formulate(
        self,
        reference_subsystem: FinalStateID | None = None,
        *,
        cleanup_summations: bool = False,
        use_coefficients: bool = False,
    ) -> AmplitudeModel:
        """Formulate the amplitude model given the configuration of this builder.

        Args:
            reference_subsystem: The subsystem to use as reference for the alignment of
                helicities. If `None`, the subsystem with the most resonances is chosen.
            cleanup_summations: Whether to remove helicity indices in the summations if
                their corresponding state is spinless.
            use_coefficients: Whether to use a single complex coefficient per decay
                chain, instead of separate coefficients for each helicity coupling.
        """
        if reference_subsystem is None:
            reference_subsystem = _get_best_reference_subsystems(self.decay)
        else:
            _check_reference_subsystems(self.decay, reference_subsystem)
        helicity_symbols: tuple[sp.Symbol, sp.Symbol, sp.Symbol, sp.Symbol] = (
            sp.symbols("lambda:4", rational=True)
        )
        allowed_helicities = {
            symbol: create_spin_range(self.decay.states[i].spin)  # ty:ignore[invalid-argument-type]
            for i, symbol in enumerate(helicity_symbols)
        }
        amplitude_definitions = {}
        angle_definitions = {}
        parameter_defaults: dict[sp.Basic, ParameterValue] = {}
        if self.all_subsystems:
            subsystem_ids: list[FinalStateID] = [1, 2, 3]
        else:
            subsystem_ids = sorted(_get_subsystem_ids(self.decay))
        for args in product(*allowed_helicities.values()):
            for sub_system in subsystem_ids:
                chain_model = self.formulate_subsystem_amplitude(
                    *args,
                    sub_system,  # ty:ignore[too-many-positional-arguments]
                    use_coefficients=use_coefficients,
                )
                amplitude_definitions.update(chain_model.amplitudes)
                angle_definitions.update(chain_model.variables)
                parameter_defaults.update(chain_model.parameter_defaults)
        aligned_amp, zeta_defs = self.formulate_aligned_amplitude(
            *helicity_symbols,
            reference_subsystem,  # ty:ignore[too-many-positional-arguments]
        )
        angle_definitions.update(zeta_defs)
        masses = create_mass_symbol_mapping(self.decay)
        parameter_defaults.update(masses)  # ty:ignore[no-matching-overload]
        if cleanup_summations:
            aligned_amp = aligned_amp.cleanup()
        intensity = PoolSum(
            sp.Abs(aligned_amp) ** 2,
            *allowed_helicities.items(),
        )
        if cleanup_summations:
            intensity = intensity.cleanup()
        return AmplitudeModel(
            decay=self.decay,
            intensity=PoolSum(
                sp.Abs(aligned_amp) ** 2,
                *allowed_helicities.items(),
            ),
            amplitudes=amplitude_definitions,
            variables=angle_definitions,
            parameter_defaults=parameter_defaults,  # ty:ignore[invalid-argument-type]
            masses=masses,
            invariants=formulate_invariants(self.decay),
        )

    def formulate_subsystem_amplitude(  # noqa: PLR0914
        self,
        λ0: sp.Rational,
        λ1: sp.Rational,
        λ2: sp.Rational,
        λ3: sp.Rational,
        subsystem_id: FinalStateID,
        *,
        use_coefficients: bool = False,
    ) -> AmplitudeModel:
        k = subsystem_id
        i, j = get_decay_product_ids(subsystem_id)
        θij, θij_expr = formulate_scattering_angle(i, j)
        λ = λ0, λ1, λ2, λ3
        spin = (
            self.decay.initial_state.spin,
            self.decay.final_state[1].spin,
            self.decay.final_state[2].spin,
            self.decay.final_state[3].spin,
        )
        λR = sp.Symbol(R"\lambda_R", rational=True)
        amplitude_sum = DefinedExpression(0)  # ty:ignore[invalid-argument-type]
        for chain in self.decay.get_subsystem(subsystem_id).chains:
            formulate_dynamics = self.dynamics_choices.get_builder(chain.resonance.name)
            amplitude = formulate_dynamics(chain)
            resonance_spin = sp.Rational(chain.resonance.spin)
            resonance_helicities = create_spin_range(resonance_spin)
            for λR_val in resonance_helicities:
                if λ[0] != λR_val - λ[k]:  # Kronecker delta
                    continue
                scaling_factors = _create_scaling_factors(
                    chain,
                    (self.use_production_helicity_couplings, λR_val, λ[k]),
                    (self.use_decay_helicity_couplings, λ[i], λ[j]),
                    one_scalar_per_chain=use_coefficients,
                )
                if isinstance(scaling_factors, tuple):
                    h_prod, h_dec = scaling_factors
                    amplitude.parameters[h_prod] = 1 + 0j
                    amplitude.parameters[h_dec] = 1
                else:
                    amplitude.parameters[scaling_factors] = 1 + 0j
            scaling_factors = _create_scaling_factors(
                chain,
                (self.use_production_helicity_couplings, λR, λ[k]),
                (self.use_decay_helicity_couplings, λ[i], λ[j]),
                one_scalar_per_chain=use_coefficients,
            )
            amplitude *= (
                sp.KroneckerDelta(λ[0], λR - λ[k])
                * (-1) ** (spin[k] - λ[k])
                * Wigner.d(resonance_spin, λR, λ[i] - λ[j], θij)
                * _product(scaling_factors)
                * (-1) ** (spin[j] - λ[j])
            )
            if not self.use_decay_helicity_couplings:
                amplitude *= _formulate_clebsch_gordan_factors(
                    chain.decay_node,
                    helicities={
                        self.decay.final_state[i]: λ[i],
                        self.decay.final_state[j]: λ[j],
                    },
                )
            if not self.use_production_helicity_couplings:
                production_isobar = chain.decay
                amplitude *= _formulate_clebsch_gordan_factors(
                    production_isobar,
                    helicities={
                        chain.resonance: λR,
                        self.decay.final_state[k]: λ[k],
                    },
                )
            amplitude_sum += attrs.evolve(
                amplitude,
                expression=PoolSum(amplitude.expression, (λR, resonance_helicities)),
            )
        A = _generate_amplitude_index_bases()
        amp_symbol = A[subsystem_id][λ0, λ1, λ2, λ3]
        return AmplitudeModel(
            decay=self.decay,
            intensity=sp.Abs(amp_symbol) ** 2,
            amplitudes={amp_symbol: amplitude_sum.expression},
            variables=amplitude_sum.subexpressions | {θij: θij_expr},
            parameter_defaults=amplitude_sum.parameters,  # ty:ignore[invalid-argument-type]
        )

    def formulate_aligned_amplitude(
        self,
        λ0: sp.Rational | sp.Symbol,
        λ1: sp.Rational | sp.Symbol,
        λ2: sp.Rational | sp.Symbol,
        λ3: sp.Rational | sp.Symbol,
        reference_subsystem: FinalStateID | None = None,
    ) -> tuple[PoolSum, dict[sp.Symbol, sp.Expr]]:
        if reference_subsystem is None:
            reference_subsystem = _get_best_reference_subsystems(self.decay)
        else:
            _check_reference_subsystems(self.decay, reference_subsystem)
        wigner_generator = _AlignmentWignerGenerator(reference_subsystem)
        _λ0, _λ1, _λ2, _λ3 = sp.symbols(R"\lambda_(0:4)^{\prime}", rational=True)
        j0, j1, j2, j3 = (self.decay.states[i].spin for i in sorted(self.decay.states))
        A = _generate_amplitude_index_bases()
        amp_expr = PoolSum(
            sum(
                A[k][_λ0, _λ1, _λ2, _λ3]
                * wigner_generator(j0, λ0, _λ0, rotated_state=0, aligned_subsystem=k)
                * wigner_generator(j1, _λ1, λ1, rotated_state=1, aligned_subsystem=k)
                * wigner_generator(j2, _λ2, λ2, rotated_state=2, aligned_subsystem=k)
                * wigner_generator(j3, _λ3, λ3, rotated_state=3, aligned_subsystem=k)
                for k in _get_subsystem_ids(self.decay)
            ),
            (_λ0, create_spin_range(j0)),
            (_λ1, create_spin_range(j1)),
            (_λ2, create_spin_range(j2)),
            (_λ3, create_spin_range(j3)),
        )
        return amp_expr, wigner_generator.angle_definitions


def _product(obj: Any | Iterable):
    if isinstance(obj, abc.Iterable):
        return functools.reduce(operator.mul, obj)
    return obj


def _get_best_reference_subsystems(decay: ThreeBodyDecay) -> FinalStateID:
    subsystem_ids = _get_subsystem_ids(decay)
    if not subsystem_ids:
        msg = f"Decay {_get_decay_description(decay)} has no subsystems"
        raise ValueError(msg)
    resonances_per_subsystem = [
        (k, len(decay.get_subsystem(k).chains)) for k in subsystem_ids
    ]
    return max(resonances_per_subsystem, key=operator.itemgetter(1))[0]  # ty:ignore[invalid-return-type]


def _check_reference_subsystems(
    decay: ThreeBodyDecay, reference_subsystem: FinalStateID
) -> None:
    subsystem_ids = _get_subsystem_ids(decay)
    if reference_subsystem not in subsystem_ids:
        decay_description = _get_decay_description(decay)
        subsystems = ", ".join(sorted(str(i) for i in _get_subsystem_ids(decay)))
        msg = (
            f"Decay {decay_description} only has subsystems {subsystems}. Are you"
            f" sure you want to use subsystem {reference_subsystem} as reference?"
        )
        warn(msg, category=UserWarning)


def _create_scaling_factors(
    chain: ThreeBodyDecayChain,
    production_subscripts: tuple[bool, sp.Basic, sp.Basic],
    decay_subscripts: tuple[bool, sp.Basic, sp.Basic],
    one_scalar_per_chain: bool,
):
    prod_helicity_basis, λR, λk = production_subscripts
    dec_helicity_basis, λi, λj = decay_subscripts
    R = Str(chain.resonance.latex)
    h_prod = _create_coupling_symbol(
        prod_helicity_basis,
        resonance=R,
        helicities=(λR, λk),
        interaction=chain.incoming_ls,
        typ="production",
    )
    h_dec = _create_coupling_symbol(
        dec_helicity_basis,
        resonance=R,
        helicities=(λi, λj),
        interaction=chain.outgoing_ls,
        typ="decay",
    )
    if one_scalar_per_chain:
        h = _get_coefficient_base(R, prod_helicity_basis, dec_helicity_basis)
        indices = (*h_prod.indices[1:], *h_dec.indices[1:])
        return h.__getitem__(indices)  # noqa: PLC2801
    return h_prod, h_dec


def _create_coupling_symbol(
    helicity_basis: bool,
    resonance: Str,
    helicities: tuple[sp.Basic, sp.Basic],
    interaction: LSCoupling | None,
    typ: Literal["production", "decay"],
) -> sp.Indexed:
    H = _get_coupling_base(helicity_basis, typ)
    if helicity_basis:
        λi, λj = helicities
        return H[resonance, λi, λj]
    if interaction is None:
        msg = "Cannot formulate LS-coupling without LS combinations"
        raise ValueError(msg)
    return H[resonance, interaction.L, interaction.S]


@cache
def _get_coefficient_base(
    resonance: Str,
    prod_helicity_basis: bool,
    dec_helicity_basis: bool,
) -> sp.IndexedBase:
    if prod_helicity_basis and dec_helicity_basis:
        return sp.IndexedBase(Rf"\mathcal{{H}}^\mathrm{{{resonance}}}")
    if prod_helicity_basis:
        return sp.IndexedBase(Rf"\mathcal{{H}}^\mathrm{{LS,\lambda,{resonance}}}")
    if dec_helicity_basis:
        return sp.IndexedBase(Rf"\mathcal{{H}}^\mathrm{{\lambda,LS,{resonance}}}")
    return sp.IndexedBase(Rf"\mathcal{{H}}^\mathrm{{LS,{resonance}}}")


@cache
def _get_coupling_base(
    helicity_basis: bool, typ: Literal["production", "decay"]
) -> sp.IndexedBase:
    if helicity_basis:
        return sp.IndexedBase(Rf"\mathcal{{H}}^\mathrm{{{typ}}}")
    return sp.IndexedBase(Rf"\mathcal{{H}}^\mathrm{{LS,{typ}}}")


def _formulate_clebsch_gordan_factors(
    isobar: IsobarNode,
    helicities: dict[Particle, sp.Rational | sp.Symbol],
) -> sp.Expr:
    if isobar.interaction is None:
        msg = "Cannot formulate amplitude model in LS-basis if LS-couplings are missing"
        raise ValueError(msg)
    # https://github.com/ComPWA/ampform/blob/65b4efa/src/ampform/helicity/__init__.py#L785-L802
    # and supplementary material p.1 (https://cds.cern.ch/record/2824328/files)
    child1 = to_particle(isobar.child1)
    child2 = to_particle(isobar.child2)
    child1_helicity = helicities[child1]
    child2_helicity = helicities[child2]
    cg_ss = CG(
        j1=child1.spin,
        m1=child1_helicity,
        j2=child2.spin,
        m2=-child2_helicity,
        j3=isobar.interaction.S,
        m3=child1_helicity - child2_helicity,
    )
    cg_ll = CG(
        j1=isobar.interaction.L,
        m1=0,
        j2=isobar.interaction.S,
        m2=child1_helicity - child2_helicity,
        j3=isobar.parent.spin,
        m3=child1_helicity - child2_helicity,
    )
    sqrt_factor = sp.sqrt((2 * isobar.interaction.L + 1) / (2 * isobar.parent.spin + 1))
    return sqrt_factor * cg_ll * cg_ss


@cache
def _generate_amplitude_index_bases() -> dict[FinalStateID, sp.IndexedBase]:
    return dict(enumerate(sp.symbols(R"A^(1:4)", cls=sp.IndexedBase), 1))  # ty:ignore[invalid-return-type]


class _AlignmentWignerGenerator:
    def __init__(self, reference_subsystem: FinalStateID = 1) -> None:
        self.angle_definitions: dict[sp.Symbol, sp.Expr] = {}
        self.reference_subsystem = reference_subsystem

    def __call__(
        self,
        j: sp.Rational,
        m: sp.Rational | sp.Symbol,
        m_prime: sp.Rational | sp.Symbol,
        rotated_state: int,
        aligned_subsystem: int,
    ) -> sp.Rational | WignerD:
        if j == 0:
            return sp.Rational(1)
        zeta, zeta_expr = formulate_zeta_angle(
            rotated_state, aligned_subsystem, self.reference_subsystem
        )
        self.angle_definitions[zeta] = zeta_expr
        return Wigner.d(j, m, m_prime, zeta)


class DynamicsConfigurator:
    def __init__(self, decay: ThreeBodyDecay) -> None:
        self.__decay = decay
        self.__dynamics_builders: dict[ThreeBodyDecayChain, DynamicsBuilder] = {}

    def register_builder(self, identifier, builder: DynamicsBuilder) -> None:
        chain = self.__get_chain(identifier)
        self.__dynamics_builders[chain] = builder

    def get_builder(self, identifier) -> DynamicsBuilder:
        chain = self.__get_chain(identifier)
        return self.__dynamics_builders.get(chain, lambda _: DefinedExpression())

    def __get_chain(self, identifier) -> ThreeBodyDecayChain:
        if isinstance(identifier, ThreeBodyDecayChain):
            chain = identifier
            if chain not in set(self.__decay.chains):
                msg = f"Decay does not have chain with resonance {chain.resonance.name}"
                raise ValueError(msg)
            return chain
        if isinstance(identifier, str):
            return self.__decay.find_chain(identifier)
        msg = f"Cannot get decay chain for identifier type {type(identifier)}"
        raise NotImplementedError(msg)

    @property
    def decay(self) -> ThreeBodyDecay:
        return self.__decay


def _binary_operation(op: Callable[[Any, Any], Any]):
    def decorator(func):
        @wraps(func)
        def wrapper(self: DefinedExpression, other):
            if isinstance(other, DefinedExpression):
                return DefinedExpression(
                    expression=op(self.expression, other.expression),
                    parameters=self.parameters | other.parameters,
                    subexpressions=self.subexpressions | other.subexpressions,
                )
            return DefinedExpression(
                expression=op(self.expression, other),
                parameters=self.parameters,
                subexpressions=self.subexpressions,
            )

        return wrapper

    return decorator


@define
class DefinedExpression:
    expression: sp.Expr = field(converter=sp.sympify, default=sp.S.One)
    parameters: dict[sp.Basic, complex | float] = field(factory=dict)
    subexpressions: dict[sp.Basic, sp.Expr] = field(factory=dict)

    @_binary_operation(operator.mul)
    def __mul__(self, other) -> DefinedExpression: ...  # type:ignore[empty-body]
    @_binary_operation(operator.add)
    def __add__(self, other) -> DefinedExpression: ...  # type:ignore[empty-body]
    @_binary_operation(operator.sub)
    def __sub__(self, other) -> DefinedExpression: ...  # type:ignore[empty-body]
    @_binary_operation(operator.truediv)
    def __truediv__(self, other) -> DefinedExpression: ...  # type:ignore[empty-body]
    @_binary_operation(operator.pow)
    def __pow__(self, other) -> DefinedExpression: ...  # type:ignore[empty-body]


DynamicsBuilder = Callable[[ThreeBodyDecayChain], DefinedExpression]
"""Protocol for functions that formulate dynamics expressions for decay chains."""


def create_mass_symbol_mapping(decay: ThreeBodyDecay) -> dict[sp.Symbol, float]:
    return {
        create_mass_symbol(decay.states[i]): decay.states[i].mass  # ty:ignore[invalid-argument-type]
        for i in sorted(decay.states)  # ensure that dict keys are sorted by state ID
    }


def create_mass_symbol(particle: IsobarNode | Particle | State) -> sp.Symbol:
    particle = to_particle(particle)
    if isinstance(particle, State):
        return sp.Symbol(f"m{particle.index}", nonnegative=True)
    return sp.Symbol(f"m_{{{particle.latex}}}", nonnegative=True)


def formulate_invariants(decay: ThreeBodyDecay) -> dict[sp.Symbol, sp.Expr]:
    s1, s2, s3 = sp.symbols("sigma1:4", nonnegative=True)
    return {
        s1: formulate_third_mandelstam(decay, 2, 3),
        s2: formulate_third_mandelstam(decay, 3, 1),
        s3: formulate_third_mandelstam(decay, 1, 2),
    }


def formulate_third_mandelstam(
    decay: ThreeBodyDecay,
    x_mandelstam: FinalStateID = 1,
    y_mandelstam: FinalStateID = 2,
) -> sp.Add:
    m0, m1, m2, m3 = create_mass_symbol_mapping(decay)
    sigma_x = sp.Symbol(f"sigma{x_mandelstam}", nonnegative=True)
    sigma_y = sp.Symbol(f"sigma{y_mandelstam}", nonnegative=True)
    return compute_third_mandelstam(sigma_x, sigma_y, m0, m1, m2, m3)
