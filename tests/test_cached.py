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
        pytest.param(True, {"544cb15"}, id="min-ls"),
        pytest.param(False, {"01968bc", "7300a10"}, id="all-ls"),
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
        "A2[-1,0,-1/2,-1/2]": "652522d",
        "A3[-1,0,-1/2,-1/2]": "5d8b698",
        "A2[-1,0,-1/2,1/2]": "0a944fe",
        "A3[-1,0,-1/2,1/2]": "a564151",
        "A2[-1,0,1/2,-1/2]": "4732f96",
        "A3[-1,0,1/2,-1/2]": "67bc82d",
        "A2[-1,0,1/2,1/2]": "98370c7",
        "A3[-1,0,1/2,1/2]": "00d5d44",
        "A2[0,0,-1/2,-1/2]": "aef2ca4",
        "A3[0,0,-1/2,-1/2]": "bcb25df",
        "A2[0,0,-1/2,1/2]": "00c9500",
        "A3[0,0,-1/2,1/2]": "7f73b98",
        "A2[0,0,1/2,-1/2]": "f3ade16",
        "A3[0,0,1/2,-1/2]": "96ec036",
        "A2[0,0,1/2,1/2]": "81dcaa6",
        "A3[0,0,1/2,1/2]": "6a547e5",
        "A2[1,0,-1/2,-1/2]": "09ca576",
        "A3[1,0,-1/2,-1/2]": "b3ed335",
        "A2[1,0,-1/2,1/2]": "beed520",
        "A3[1,0,-1/2,1/2]": "115597f",
        "A2[1,0,1/2,-1/2]": "f6e2775",
        "A3[1,0,1/2,-1/2]": "0d898ee",
        "A2[1,0,1/2,1/2]": "9276158",
        "A3[1,0,1/2,1/2]": "f3fc8c5",
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
