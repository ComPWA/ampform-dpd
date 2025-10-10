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


@pytest.mark.slow
@pytest.mark.parametrize(
    ("min_ls", "expected_hashes"),
    [
        pytest.param(True, {"ba9b5fd"}, id="min-ls"),
        pytest.param(False, {"8039460"}, id="all-ls"),
    ],
)
def test_hashes(reaction: ReactionInfo, min_ls: bool, expected_hashes: set[str]):
    transitions = normalize_state_ids(reaction.transitions)
    decay = to_three_body_decay(transitions, min_ls=min_ls)
    builder = DalitzPlotDecompositionBuilder(decay, min_ls=min_ls)
    for chain in builder.decay.chains:
        builder.dynamics_choices.register_builder(
            chain, formulate_breit_wigner_with_form_factor
        )
    model = builder.formulate(reference_subsystem=2)
    intensity_expr = model.full_expression
    h = get_readable_hash(intensity_expr)[:7]
    assert h in expected_hashes


def test_amplitude_doit_hashes(reaction: ReactionInfo):
    transitions = normalize_state_ids(reaction.transitions)
    decay = to_three_body_decay(transitions, min_ls=True)
    builder = DalitzPlotDecompositionBuilder(decay, min_ls=True)
    for chain in builder.decay.chains:
        builder.dynamics_choices.register_builder(
            chain, formulate_breit_wigner_with_form_factor
        )
    model = builder.formulate(reference_subsystem=2)
    hashes = {
        str(k).replace("^", "").replace(" ", ""): get_readable_hash(expr.doit())[:7]
        for k, expr in model.amplitudes.items()
    }
    print(hashes)
    assert hashes == {
        "A2[-1,0,-1/2,-1/2]": "6770935",
        "A3[-1,0,-1/2,-1/2]": "5b5001a",
        "A2[-1,0,-1/2,1/2]": "716e629",
        "A3[-1,0,-1/2,1/2]": "d338945",
        "A2[-1,0,1/2,-1/2]": "78bf694",
        "A3[-1,0,1/2,-1/2]": "b72a7e2",
        "A2[-1,0,1/2,1/2]": "b1e428d",
        "A3[-1,0,1/2,1/2]": "8730459",
        "A2[0,0,-1/2,-1/2]": "c9cb5e9",
        "A3[0,0,-1/2,-1/2]": "a62c07e",
        "A2[0,0,-1/2,1/2]": "b87152c",
        "A3[0,0,-1/2,1/2]": "8961c3b",
        "A2[0,0,1/2,-1/2]": "73f76f4",
        "A3[0,0,1/2,-1/2]": "82d805d",
        "A2[0,0,1/2,1/2]": "36dd5d0",
        "A3[0,0,1/2,1/2]": "ac73d2d",
        "A2[1,0,-1/2,-1/2]": "7f8513f",
        "A3[1,0,-1/2,-1/2]": "3691bba",
        "A2[1,0,-1/2,1/2]": "165a2ad",
        "A3[1,0,-1/2,1/2]": "7c7b3cf",
        "A2[1,0,1/2,-1/2]": "e3ff1a7",
        "A3[1,0,1/2,-1/2]": "06027b1",
        "A2[1,0,1/2,1/2]": "eb57212",
        "A3[1,0,1/2,1/2]": "3ef2476",
    }


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
