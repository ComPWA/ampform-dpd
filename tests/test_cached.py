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
        pytest.param(True, {"c313279"}, id="min-ls"),
        pytest.param(False, {"0ec7fb2", "998ea8a"}, id="all-ls"),
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
        "A2[-1,0,-1/2,-1/2]": "2cba34b",
        "A3[-1,0,-1/2,-1/2]": "1777f8b",
        "A2[-1,0,-1/2,1/2]": "8892024",
        "A3[-1,0,-1/2,1/2]": "5dbb5e1",
        "A2[-1,0,1/2,-1/2]": "8043535",
        "A3[-1,0,1/2,-1/2]": "a4a0855",
        "A2[-1,0,1/2,1/2]": "1499699",
        "A3[-1,0,1/2,1/2]": "9cafbcb",
        "A2[0,0,-1/2,-1/2]": "f04fb95",
        "A3[0,0,-1/2,-1/2]": "0cb812f",
        "A2[0,0,-1/2,1/2]": "87a7726",
        "A3[0,0,-1/2,1/2]": "aad72b3",
        "A2[0,0,1/2,-1/2]": "6c0f5f0",
        "A3[0,0,1/2,-1/2]": "2f2834c",
        "A2[0,0,1/2,1/2]": "683d322",
        "A3[0,0,1/2,1/2]": "e8ee998",
        "A2[1,0,-1/2,-1/2]": "002bd60",
        "A3[1,0,-1/2,-1/2]": "5b99e54",
        "A2[1,0,-1/2,1/2]": "6c84f16",
        "A3[1,0,-1/2,1/2]": "10de6f5",
        "A2[1,0,1/2,-1/2]": "d0ec838",
        "A3[1,0,1/2,-1/2]": "4a2ef6a",
        "A2[1,0,1/2,1/2]": "f84b477",
        "A3[1,0,1/2,1/2]": "6ccf5d3",
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
