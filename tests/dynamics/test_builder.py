from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import qrules
import sympy as sp
from ampform.dynamics import BreitWigner, EnergyDependentWidth
from ampform.dynamics.form_factor import FormFactor
from attrs import evolve

from ampform_dpd import create_mass_symbol
from ampform_dpd.adapter.qrules import normalize_state_ids, to_three_body_decay
from ampform_dpd.decay import LSCoupling, ThreeBodyDecayChain
from ampform_dpd.dynamics.builder import BreitWignerBuilder, get_mandelstam_s

if TYPE_CHECKING:
    from qrules.transition import ReactionInfo


@pytest.fixture(scope="module")
def reaction() -> ReactionInfo:
    return qrules.generate_transitions(
        initial_state="D+",
        final_state=["pi+", "pi+", "pi-"],
        allowed_intermediate_particles=["rho(770)0"],
        formalism="canonical-helicity",
        mass_conservation_factor=0,
    )


def describe_BreitWignerBuilder():
    @pytest.mark.parametrize("normalize", [False, True])
    @pytest.mark.parametrize("numerator", ["unity", "mass_width"])
    @pytest.mark.parametrize("running_width", [False, True])
    @pytest.mark.parametrize(
        ("production", "decay"),
        [
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ],
    )
    def it_reproduces_all_normalization_conventions(
        reaction, normalize, numerator, running_width, *, production, decay
    ):
        chain = to_three_body_decay(
            normalize_state_ids(reaction).transitions, min_ls=True
        ).chains[0]
        result = BreitWignerBuilder(
            normalize_form_factors=normalize,
            numerator=numerator,
            energy_dependent_width=running_width,
            production_form_factor=production,
            decay_form_factor=decay,
        )(chain)
        baseline = BreitWignerBuilder()(chain).expression
        (bw,) = baseline.atoms(BreitWigner)
        width = (
            EnergyDependentWidth(
                bw.s,
                bw.mass,
                bw.width,
                bw.m1,
                bw.m2,
                bw.angular_momentum,
                bw.meson_radius,
            )
            if running_width
            else bw.width
        )
        expected = 1 / (bw.mass**2 - bw.s - sp.I * bw.mass * width)
        for ff in baseline.atoms(FormFactor):
            enabled = decay if ff.s == bw.s else production
            if enabled:
                expected *= ff
                if normalize:
                    expected /= ff.xreplace({bw.s: bw.mass**2})
        if numerator == "mass_width":
            expected *= bw.mass * bw.width
        assert result.expression.doit() == expected.doit()
        numeric = (
            result.expression.doit().subs(result.parameters).subs(bw.s, 0.8).evalf()
        )
        assert numeric.is_finite

    def it_normalizes_the_vertex_factors_at_the_pole(reaction):
        chain = to_three_body_decay(
            normalize_state_ids(reaction).transitions, min_ls=True
        ).chains[0]
        result = BreitWignerBuilder(normalize_form_factors=True)(chain)
        (bw,) = result.expression.atoms(BreitWigner)
        vertex_factors = result.expression / bw
        assert vertex_factors.subs(bw.s, bw.mass**2) == 1
        assert (
            sp.simplify(bw.energy_dependent_width().doit().subs(bw.s, bw.mass**2))
            == bw.width
        )

    @pytest.mark.parametrize("normalize", [False, True])
    def it_applies_the_blatt_weisskopf_convention(reaction, normalize):
        chain = to_three_body_decay(
            normalize_state_ids(reaction).transitions, min_ls=True
        ).chains[0]
        normalized = BreitWignerBuilder(normalize_form_factors=normalize)(
            chain
        ).expression
        unnormalized = BreitWignerBuilder(
            normalize_form_factors=normalize, blatt_weisskopf_convention="unnormalized"
        )(chain).expression
        assert {ff.normalize for ff in normalized.atoms(FormFactor)} == {True}
        assert {ff.normalize for ff in unnormalized.atoms(FormFactor)} == {False}
        assert chain.decay_node.interaction is not None
        assert chain.production_node.interaction is not None
        if normalize:
            expected = sp.S.One  # the constant cancels against the pole value
        else:
            normalizations = {0: 1, 1: sp.sqrt(2), 2: sp.sqrt(13)}
            expected = 1 / (
                normalizations[chain.decay_node.interaction.L]
                * normalizations[chain.production_node.interaction.L]
            )
        assert sp.simplify((unnormalized / normalized).doit()) == expected

    def it_rejects_unknown_blatt_weisskopf_conventions():
        with pytest.raises(ValueError, match="blatt_weisskopf_convention"):
            BreitWignerBuilder(blatt_weisskopf_convention="unknown")  # ty: ignore[invalid-argument-type]

    def it_maps_and_fixes_parameters(reaction):
        chain = to_three_body_decay(
            normalize_state_ids(reaction).transitions, min_ls=True
        ).chains[0]
        radius = sp.Symbol(Rf"R_{{{chain.resonance.latex}}}", nonnegative=True)
        shared = sp.Symbol("R_shared")
        parent_radius = sp.Symbol(Rf"R_{{{chain.parent.latex}}}", nonnegative=True)
        result = BreitWignerBuilder(
            symbol_mapping={radius: shared, parent_radius: sp.Integer(5)},
            parameter_defaults={shared: 1.5},
        )(chain)
        assert result.parameters[shared] == 1.5
        assert parent_radius not in result.parameters
        assert not result.expression.has(radius, parent_radius)
        assert result.expression.has(shared)

    @pytest.mark.parametrize("running_width", [False, True])
    def it_omits_unused_s_wave_radii(reaction, running_width):
        chain = to_three_body_decay(
            normalize_state_ids(reaction).transitions, min_ls=True
        ).chains[0]
        node = evolve(chain.decay_node, interaction=LSCoupling(0, 0))
        chain = ThreeBodyDecayChain(
            evolve(chain.production_node, child1=node, interaction=LSCoupling(0, 0))
        )
        result = BreitWignerBuilder(
            energy_dependent_width=running_width, normalize_form_factors=True
        )(chain)
        assert not result.expression.atoms(FormFactor)
        assert not any(str(p).startswith("R_") for p in result.parameters)

    def it_gives_different_resonances_independent_radii(reaction):
        chain = to_three_body_decay(
            normalize_state_ids(reaction).transitions, min_ls=True
        ).chains[0]
        node = evolve(
            chain.decay_node,
            parent=evolve(chain.resonance, name="other", latex="other"),
        )
        other = ThreeBodyDecayChain(evolve(chain.production_node, child1=node))
        first = BreitWignerBuilder()(chain)
        second = BreitWignerBuilder()(other)
        (first_bw,) = first.expression.atoms(BreitWigner)
        (second_bw,) = second.expression.atoms(BreitWigner)
        assert first_bw.meson_radius != second_bw.meson_radius

    def it_rejects_unknown_numerators():
        with pytest.raises(ValueError, match="numerator"):
            BreitWignerBuilder(numerator="unknown")  # ty: ignore[invalid-argument-type]

    @pytest.mark.parametrize("energy_dependent_width", [False, True])
    def it_uses_consistent_form_factor_kinematics(
        reaction: ReactionInfo,
        energy_dependent_width: bool,
    ):
        reaction = normalize_state_ids(reaction)
        decay = to_three_body_decay(reaction.transitions, min_ls=True)
        for chain in decay.chains:
            result = BreitWignerBuilder(
                energy_dependent_width=energy_dependent_width,
            )(chain)
            s = get_mandelstam_s(chain.decay_node)
            m_top = create_mass_symbol(chain.parent)
            m_spec = create_mass_symbol(chain.spectator)
            m1, m2 = map(create_mass_symbol, chain.decay_products)
            assert chain.decay_node.interaction is not None
            assert chain.production_node.interaction is not None
            l_dec = sp.Integer(chain.decay_node.interaction.L)
            l_prod = sp.Integer(chain.production_node.interaction.L)
            r_dec = sp.Symbol(Rf"R_{{{chain.resonance.latex}}}", nonnegative=True)
            r_prod = sp.Symbol(Rf"R_{{{chain.parent.latex}}}", nonnegative=True)
            expected = {
                FormFactor(s, m1, m2, l_dec, r_dec),
                FormFactor(m_top**2, sp.sqrt(s), m_spec, l_prod, r_prod),
            }
            assert result.expression.atoms(FormFactor) == expected
            assert s not in result.parameters
            assert sp.sqrt(s) not in result.parameters
            assert result.parameters[m_top] == chain.parent.mass
            for state in chain.final_state:
                assert result.parameters[create_mass_symbol(state)] == state.mass
            if energy_dependent_width:
                assert {bw.s for bw in result.expression.atoms(BreitWigner)} == {s}
