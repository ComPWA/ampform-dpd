from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from ampform.sympy._cache import get_readable_hash

from ampform_dpd import DalitzPlotDecompositionBuilder
from ampform_dpd.adapter.qrules import normalize_state_ids, to_three_body_decay
from ampform_dpd.dynamics.builder import formulate_breit_wigner_with_form_factor

if TYPE_CHECKING:
    from qrules.transition import ReactionInfo


@pytest.mark.parametrize(
    ("min_ls", "folded_hash", "unfolded_hash"),
    [
        (True, "dbca20f", "c81d961"),
        (False, "158e632", "a873a4f"),
    ],
    ids=["min_ls", "all_ls_couplings"],
)
def test_hashes(
    jpsi2pksigma_reaction: ReactionInfo,
    min_ls: bool,
    folded_hash: str,
    unfolded_hash: str,
):
    reaction = jpsi2pksigma_reaction
    if reaction.formalism == "helicity":
        pytest.skip("Test for helicity formalism not supported")
    transitions = normalize_state_ids(jpsi2pksigma_reaction.transitions)
    decay = to_three_body_decay(transitions, min_ls=min_ls)
    builder = DalitzPlotDecompositionBuilder(decay, min_ls=min_ls)
    for chain in builder.decay.chains:
        builder.dynamics_choices.register_builder(
            chain, formulate_breit_wigner_with_form_factor
        )
    model = builder.formulate(reference_subsystem=2)
    intensity_expr = model.full_expression
    h = get_readable_hash(intensity_expr)[:7]
    assert h == folded_hash
    h = get_readable_hash(intensity_expr.doit())[:7]
    assert h == unfolded_hash
