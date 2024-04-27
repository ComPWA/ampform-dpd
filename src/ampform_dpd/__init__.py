"""Module for formulating the amplitude model for a three-body decay using DPD."""

from __future__ import annotations

from functools import lru_cache
from itertools import product
from typing import Literal, Protocol

import sympy as sp
from ampform.kinematics.phasespace import compute_third_mandelstam
from ampform.sympy import PoolSum
from attrs import field, frozen
from sympy.core.symbol import Str
from sympy.physics.quantum.spin import CG, WignerD
from sympy.physics.quantum.spin import Rotation as Wigner

from ampform_dpd.angles import formulate_scattering_angle, formulate_zeta_angle
from ampform_dpd.decay import (
    FinalStateID,
    IsobarNode,
    LSCoupling,
    Particle,
    ThreeBodyDecay,
    ThreeBodyDecayChain,
    get_decay_product_ids,
    to_particle,
)
from ampform_dpd.io import (
    simplify_latex_rendering,  # noqa: F401  # pyright:ignore[reportUnusedImport]
)
from ampform_dpd.spin import create_spin_range


@frozen
class AmplitudeModel:
    decay: ThreeBodyDecay
    intensity: sp.Expr = sp.S.One
    amplitudes: dict[sp.Indexed, sp.Expr] = field(factory=dict)
    variables: dict[sp.Symbol, sp.Expr] = field(factory=dict)
    parameter_defaults: dict[sp.Symbol, float | complex] = field(factory=dict)
    masses: dict[sp.Symbol, float | complex] = field(factory=dict)
    invariants: dict[sp.Symbol, sp.Expr] = field(factory=dict)

    @property
    def full_expression(self) -> sp.Expr:
        return self.intensity.doit().xreplace(self.amplitudes)


class DalitzPlotDecompositionBuilder:
    def __init__(
        self,
        decay: ThreeBodyDecay,
        min_ls: tuple[bool, bool] | bool = True,
    ) -> None:
        """Amplitude builder for the helicity formalism with Dalitz-plot decomposition.

        Args:
            decay: The `.ThreeBodyDecay` over which to formulate the amplitude model.
            min_ls: Use helicity couplings instead of
                :math:`LS`-couplings. If setting this boolean with a `tuple`, the first
                element of the `tuple` defines whether to use helicity couplings on the
                **production** `.IsobarNode` and the second configures the **decay**
                `.IsobarNode`.
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

    def formulate(
        self,
        reference_subsystem: FinalStateID = 1,
        cleanup_summations: bool = False,
    ) -> AmplitudeModel:
        helicity_symbols: tuple[sp.Symbol, sp.Symbol, sp.Symbol, sp.Symbol] = (
            sp.symbols("lambda:4", rational=True)
        )
        allowed_helicities = {
            symbol: create_spin_range(self.decay.states[i].spin)  # type:ignore[index]
            for i, symbol in enumerate(helicity_symbols)
        }
        amplitude_definitions = {}
        angle_definitions = {}
        parameter_defaults = {}
        args: tuple[sp.Rational, sp.Rational, sp.Rational, sp.Rational]
        for args in product(*allowed_helicities.values()):  # type:ignore[assignment]
            for sub_system in (1, 2, 3):
                chain_model = self.formulate_subsystem_amplitude(*args, sub_system)  # type:ignore[arg-type]
                amplitude_definitions.update(chain_model.amplitudes)
                angle_definitions.update(chain_model.variables)
                parameter_defaults.update(chain_model.parameter_defaults)
        aligned_amp, zeta_defs = self.formulate_aligned_amplitude(
            *helicity_symbols, reference_subsystem
        )
        angle_definitions.update(zeta_defs)
        masses = create_mass_symbol_mapping(self.decay)
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
            decay=self.decay,
            intensity=PoolSum(
                sp.Abs(aligned_amp) ** 2,
                *allowed_helicities.items(),
            ),
            amplitudes=amplitude_definitions,
            variables=angle_definitions,
            parameter_defaults=parameter_defaults,
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
        terms = []
        parameter_defaults = {}
        for chain in self.decay.get_subsystem(subsystem_id).chains:
            formulate_dynamics = self.dynamics_choices.get_builder(chain.resonance.name)
            dynamics, new_parameters = formulate_dynamics(chain)
            parameter_defaults.update(new_parameters)
            R = Str(chain.resonance.latex)
            resonance_spin = sp.Rational(chain.resonance.spin)
            resonance_helicities = create_spin_range(resonance_spin)
            for λR_val in resonance_helicities:
                if λ[0] != λR_val - λ[k]:  # Kronecker delta
                    continue
                h_prod = _create_coupling_symbol(
                    self.use_production_helicity_couplings,
                    resonance=R,
                    helicities=(λR_val, λ[k]),
                    interaction=chain.incoming_ls,
                    typ="production",
                )
                h_dec = _create_coupling_symbol(
                    self.use_decay_helicity_couplings,
                    resonance=R,
                    helicities=(λ[i], λ[j]),
                    interaction=chain.outgoing_ls,
                    typ="decay",
                )
                parameter_defaults[h_prod] = 1 + 0j  # type:ignore[index]
                parameter_defaults[h_dec] = 1  # type:ignore[index]
            sub_amp_expr = (
                sp.KroneckerDelta(λ[0], λR - λ[k])
                * (-1) ** (spin[k] - λ[k])
                * dynamics
                * Wigner.d(resonance_spin, λR, λ[i] - λ[j], θij)
                * _create_coupling_symbol(
                    self.use_production_helicity_couplings,
                    resonance=R,
                    helicities=(λR, λ[k]),
                    interaction=chain.incoming_ls,
                    typ="production",
                )
                * _create_coupling_symbol(
                    self.use_decay_helicity_couplings,
                    resonance=R,
                    helicities=(λ[i], λ[j]),
                    interaction=chain.outgoing_ls,
                    typ="decay",
                )
                * (-1) ** (spin[j] - λ[j])
            )
            if not self.use_decay_helicity_couplings:
                sub_amp_expr *= _formulate_clebsch_gordan_factors(
                    chain.decay_node,
                    helicities={
                        self.decay.final_state[i]: λ[i],
                        self.decay.final_state[j]: λ[j],
                    },
                )
            if not self.use_production_helicity_couplings:
                production_isobar = chain.decay
                sub_amp_expr *= _formulate_clebsch_gordan_factors(
                    production_isobar,
                    helicities={
                        chain.resonance: λR,
                        self.decay.final_state[k]: λ[k],
                    },
                )
            sub_amp = PoolSum(
                sub_amp_expr,
                (λR, resonance_helicities),
            )
            terms.append(sub_amp)
        A = _generate_amplitude_index_bases()
        amp_symbol = A[subsystem_id][λ0, λ1, λ2, λ3]
        amp_expr = sp.Add(*terms)
        return AmplitudeModel(
            decay=self.decay,
            intensity=sp.Abs(amp_symbol) ** 2,
            amplitudes={amp_symbol: amp_expr},
            variables={θij: θij_expr},
            parameter_defaults=parameter_defaults,
        )

    def formulate_aligned_amplitude(
        self,
        λ0: sp.Rational | sp.Symbol,
        λ1: sp.Rational | sp.Symbol,
        λ2: sp.Rational | sp.Symbol,
        λ3: sp.Rational | sp.Symbol,
        reference_subsystem: FinalStateID = 1,
    ) -> tuple[PoolSum, dict[sp.Symbol, sp.Expr]]:
        wigner_generator = _AlignmentWignerGenerator(reference_subsystem)
        _λ0, _λ1, _λ2, _λ3 = sp.symbols(R"\lambda_(0:4)^{\prime}", rational=True)
        j0, j1, j2, j3 = (self.decay.states[i].spin for i in sorted(self.decay.states))
        A = _generate_amplitude_index_bases()
        amp_expr = PoolSum(
            A[1][_λ0, _λ1, _λ2, _λ3]
            * wigner_generator(j0, λ0, _λ0, rotated_state=0, aligned_subsystem=1)
            * wigner_generator(j1, _λ1, λ1, rotated_state=1, aligned_subsystem=1)
            * wigner_generator(j2, _λ2, λ2, rotated_state=2, aligned_subsystem=1)
            * wigner_generator(j3, _λ3, λ3, rotated_state=3, aligned_subsystem=1)
            + A[2][_λ0, _λ1, _λ2, _λ3]
            * wigner_generator(j0, λ0, _λ0, rotated_state=0, aligned_subsystem=2)
            * wigner_generator(j1, _λ1, λ1, rotated_state=1, aligned_subsystem=2)
            * wigner_generator(j2, _λ2, λ2, rotated_state=2, aligned_subsystem=2)
            * wigner_generator(j3, _λ3, λ3, rotated_state=3, aligned_subsystem=2)
            + A[3][_λ0, _λ1, _λ2, _λ3]
            * wigner_generator(j0, λ0, _λ0, rotated_state=0, aligned_subsystem=3)
            * wigner_generator(j1, _λ1, λ1, rotated_state=1, aligned_subsystem=3)
            * wigner_generator(j2, _λ2, λ2, rotated_state=2, aligned_subsystem=3)
            * wigner_generator(j3, _λ3, λ3, rotated_state=3, aligned_subsystem=3),
            (_λ0, create_spin_range(j0)),
            (_λ1, create_spin_range(j1)),
            (_λ2, create_spin_range(j2)),
            (_λ3, create_spin_range(j3)),
        )
        return amp_expr, wigner_generator.angle_definitions  # type:ignore[return-value]


def _create_coupling_symbol(
    helicity_coupling: bool,
    resonance: Str,
    helicities: tuple[sp.Basic, sp.Basic],
    interaction: LSCoupling | None,
    typ: Literal["production", "decay"],
) -> sp.Indexed:
    H = _get_coupling_base(helicity_coupling, typ)
    if helicity_coupling:
        λi, λj = helicities
        return H[resonance, λi, λj]
    if interaction is None:
        msg = "Cannot formulate LS-coupling without LS combinations"
        raise ValueError(msg)
    return H[resonance, interaction.L, interaction.S]


@lru_cache(maxsize=None)
def _get_coupling_base(
    helicity_coupling: bool, typ: Literal["production", "decay"]
) -> sp.IndexedBase:
    if helicity_coupling:
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


@lru_cache(maxsize=None)
def _generate_amplitude_index_bases() -> dict[FinalStateID, sp.IndexedBase]:
    return dict(enumerate(sp.symbols(R"A^(1:4)", cls=sp.IndexedBase), 1))  # type:ignore[arg-type]


class _AlignmentWignerGenerator:
    def __init__(self, reference_subsystem: FinalStateID = 1) -> None:
        self.angle_definitions: dict[sp.Symbol, sp.acos] = {}
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
        return self.__dynamics_builders.get(chain, formulate_non_resonant)

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


class DynamicsBuilder(Protocol):
    def __call__(
        self, decay_chain: ThreeBodyDecayChain
    ) -> tuple[sp.Expr, dict[sp.Symbol, float | complex]]: ...


def formulate_non_resonant(
    decay_chain: ThreeBodyDecayChain,
) -> tuple[sp.Expr, dict[sp.Symbol, float | complex]]:
    return sp.Rational(1), {}


def create_mass_symbol_mapping(
    decay: ThreeBodyDecay,
) -> dict[sp.Symbol, float | complex]:
    return {
        sp.Symbol(f"m{i}", nonnegative=True): decay.states[i].mass
        for i in sorted(decay.states)  # ensure that dict keys are sorted by state ID
    }


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
