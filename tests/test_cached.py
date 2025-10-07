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
        pytest.param(True, {"11f75df"}, id="min-ls"),
        pytest.param(False, {"56adff3", "7691ace"}, id="all-ls"),
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
        "A2[-1,0,-1/2,-1/2]": "09425eb",
        "A3[-1,0,-1/2,-1/2]": "88ff648",
        "A2[-1,0,-1/2,1/2]": "82bf622",
        "A3[-1,0,-1/2,1/2]": "5454ffb",
        "A2[-1,0,1/2,-1/2]": "2d9c869",
        "A3[-1,0,1/2,-1/2]": "27c7126",
        "A2[-1,0,1/2,1/2]": "32d4010",
        "A3[-1,0,1/2,1/2]": "b5b1795",
        "A2[0,0,-1/2,-1/2]": "a0f421c",
        "A3[0,0,-1/2,-1/2]": "c6e4c24",
        "A2[0,0,-1/2,1/2]": "4be6e73",
        "A3[0,0,-1/2,1/2]": "f9bcde4",
        "A2[0,0,1/2,-1/2]": "f2150a0",
        "A3[0,0,1/2,-1/2]": "e8a4fc4",
        "A2[0,0,1/2,1/2]": "14dd161",
        "A3[0,0,1/2,1/2]": "c35fb8f",
        "A2[1,0,-1/2,-1/2]": "103046f",
        "A3[1,0,-1/2,-1/2]": "687c787",
        "A2[1,0,-1/2,1/2]": "af84c5a",
        "A3[1,0,-1/2,1/2]": "13db5ad",
        "A2[1,0,1/2,-1/2]": "1d27e2a",
        "A3[1,0,1/2,-1/2]": "5573335",
        "A2[1,0,1/2,1/2]": "02fa5b7",
        "A3[1,0,1/2,1/2]": "fb11bee",
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
