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
        pytest.param(True, {"f607359"}, id="min-ls"),
        pytest.param(False, {"bffbb77", "d10d72e"}, id="all-ls"),
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
    assert hashes == {
        "A2[-1,0,-1/2,-1/2]": "a1910b8",
        "A2[-1,0,-1/2,1/2]": "b9eaf6a",
        "A2[-1,0,1/2,-1/2]": "1412e07",
        "A2[-1,0,1/2,1/2]": "3a20c48",
        "A2[0,0,-1/2,-1/2]": "479d350",
        "A2[0,0,-1/2,1/2]": "12fb7b5",
        "A2[0,0,1/2,-1/2]": "ff7db6d",
        "A2[0,0,1/2,1/2]": "460a678",
        "A2[1,0,-1/2,-1/2]": "d077581",
        "A2[1,0,-1/2,1/2]": "319ebf8",
        "A2[1,0,1/2,-1/2]": "3913d20",
        "A2[1,0,1/2,1/2]": "e056ca3",
        "A3[-1,0,-1/2,-1/2]": "af5bec4",
        "A3[-1,0,-1/2,1/2]": "2d6eecc",
        "A3[-1,0,1/2,-1/2]": "ffad279",
        "A3[-1,0,1/2,1/2]": "b6397ed",
        "A3[0,0,-1/2,-1/2]": "9edf6dd",
        "A3[0,0,-1/2,1/2]": "0605507",
        "A3[0,0,1/2,-1/2]": "265e0c4",
        "A3[0,0,1/2,1/2]": "5c1ad8f",
        "A3[1,0,-1/2,-1/2]": "219d33c",
        "A3[1,0,-1/2,1/2]": "f8f00f4",
        "A3[1,0,1/2,-1/2]": "4055af1",
        "A3[1,0,1/2,1/2]": "1f28272",
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
