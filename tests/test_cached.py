from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import qrules
from ampform.sympy._cache import get_readable_hash

from ampform_dpd import DalitzPlotDecompositionBuilder
from ampform_dpd.adapter.qrules import normalize_state_ids, to_three_body_decay
from ampform_dpd.dynamics.builder import formulate_breit_wigner_with_form_factor

if TYPE_CHECKING:
    from qrules.transition import ReactionInfo

    from ampform_dpd import AmplitudeModel


@pytest.fixture(scope="session")
def reaction() -> ReactionInfo:
    return qrules.generate_transitions(
        initial_state=[("J/psi(1S)", [+1])],
        final_state=["K0", ("Sigma+", [+0.5]), ("p~", [+0.5])],
        allowed_interaction_types="strong",
        allowed_intermediate_particles=[
            "N(1650)+",  # largest branching fraction
            "N(1675)+",  # high LS couplings
            "Sigma(1385)",  # largest branching fraction
            "Sigma(1775)",  # high LS couplings
        ],
        formalism="canonical-helicity",
    )


def describe_get_readable_hash():
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "min_ls",
        [pytest.param(True, id="min-ls"), pytest.param(False, id="all-ls")],
    )
    def it_is_reproducible_for_the_full_expression(
        reaction: ReactionInfo, min_ls: bool
    ):
        model = _formulate_model(reaction, min_ls)
        intensity_expr = model.full_expression
        readable_hash = get_readable_hash(intensity_expr)
        assert readable_hash
        assert readable_hash == get_readable_hash(intensity_expr)

    def it_is_unique_for_each_amplitude(reaction: ReactionInfo):
        model = _formulate_model(reaction, min_ls=True)
        hashes = {
            str(k).replace("^", "").replace(" ", ""): get_readable_hash(expr.doit())[:7]
            for k, expr in model.amplitudes.items()
        }
        assert len(hashes) == 24
        assert len(set(hashes.values())) == len(hashes)
        assert {name[:2] for name in hashes} == {"A2", "A3"}
        repeated_hashes = {
            str(key).replace("^", "").replace(" ", ""): get_readable_hash(
                expression.doit()
            )[:7]
            for key, expression in model.amplitudes.items()
        }
        assert hashes == repeated_hashes


def _formulate_model(reaction: ReactionInfo, min_ls: bool) -> AmplitudeModel:
    transitions = normalize_state_ids(reaction.transitions)
    decay = to_three_body_decay(transitions, min_ls=min_ls)
    builder = DalitzPlotDecompositionBuilder(decay, min_ls=min_ls)
    for chain in builder.decay.chains:
        builder.dynamics_choices.register_builder(
            chain, formulate_breit_wigner_with_form_factor
        )
    return builder.formulate(reference_subsystem=2)
