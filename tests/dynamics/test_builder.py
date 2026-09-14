from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import qrules
import sympy as sp
from ampform.dynamics.form_factor import FormFactor

from ampform_dpd import create_mass_symbol
from ampform_dpd.adapter.qrules import normalize_state_ids, to_three_body_decay
from ampform_dpd.dynamics import BreitWignerMinL, RelativisticBreitWigner
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
            m_res = create_mass_symbol(chain.resonance)
            m_spec = create_mass_symbol(chain.spectator)
            m1, m2 = map(create_mass_symbol, chain.decay_products)
            assert chain.decay_node.interaction is not None
            assert chain.production_node.interaction is not None
            l_dec = sp.Integer(chain.decay_node.interaction.L)
            l_prod = sp.Integer(chain.production_node.interaction.L)
            r_dec = sp.Symbol(R"R_\mathrm{res}", nonnegative=True)
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
                assert {
                    bw.s for bw in result.expression.atoms(RelativisticBreitWigner)
                } == {s}
            reference = BreitWignerMinL(
                s,
                m_top,
                m_spec,
                m_res,
                sp.Symbol("width"),
                m1,
                m2,
                l_dec,
                l_prod,
                r_dec,
                r_prod,
            ).evaluate()
            assert {ff for ff in reference.atoms(FormFactor) if ff.has(s)} == expected
