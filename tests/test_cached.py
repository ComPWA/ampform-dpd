from __future__ import annotations

from pprint import pprint
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
        pytest.param(True, {"544cb15", "9126846"}, id="min-ls"),
        pytest.param(False, {"44bd846", "c854681"}, id="all-ls"),
    ],
)
def test_hashes(reaction: ReactionInfo, min_ls: bool, expected_hashes: set[str]):
    transitions = normalize_state_ids(reaction.transitions)
    decay = to_three_body_decay(transitions, min_ls=min_ls)  # ty:ignore[invalid-argument-type]
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
    decay = to_three_body_decay(transitions, min_ls=True)  # ty:ignore[invalid-argument-type]
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
    pprint(hashes)
    assert hashes == {
        "A2[-1,0,-1/2,-1/2]": "61d416b",
        "A3[-1,0,-1/2,-1/2]": "86eca04",
        "A2[-1,0,-1/2,1/2]": "78dafd2",
        "A3[-1,0,-1/2,1/2]": "bf9c943",
        "A2[-1,0,1/2,-1/2]": "59dd4af",
        "A3[-1,0,1/2,-1/2]": "1e30a88",
        "A2[-1,0,1/2,1/2]": "8390717",
        "A3[-1,0,1/2,1/2]": "95e4308",
        "A2[0,0,-1/2,-1/2]": "4678a3f",
        "A3[0,0,-1/2,-1/2]": "6490620",
        "A2[0,0,-1/2,1/2]": "288fc74",
        "A3[0,0,-1/2,1/2]": "ede0cd4",
        "A2[0,0,1/2,-1/2]": "3a33edd",
        "A3[0,0,1/2,-1/2]": "f4f1691",
        "A2[0,0,1/2,1/2]": "e625afc",
        "A3[0,0,1/2,1/2]": "c8d871f",
        "A2[1,0,-1/2,-1/2]": "1953f26",
        "A3[1,0,-1/2,-1/2]": "b54d73a",
        "A2[1,0,-1/2,1/2]": "1f95534",
        "A3[1,0,-1/2,1/2]": "a16e368",
        "A2[1,0,1/2,-1/2]": "c659cbb",
        "A3[1,0,1/2,-1/2]": "c09b579",
        "A2[1,0,1/2,1/2]": "7a2e0b4",
        "A3[1,0,1/2,1/2]": "4f8d794",
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
