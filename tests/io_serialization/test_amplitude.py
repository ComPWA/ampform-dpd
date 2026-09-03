from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import sympy as sp

from ampform_dpd.io.serialization.amplitude import (
    ParityRecoupling,
    _get_decay_product_helicities,
    _get_final_state_helicities,
    _get_resonance_helicity,
    _get_weight,
    get_existing_subsystem_ids,
)
from ampform_dpd.io.serialization.format import get_decay_chains

if TYPE_CHECKING:
    from ampform_dpd.io.serialization.format import ModelDefinition

half = sp.Rational(1, 2)


@pytest.mark.parametrize(
    ("chain_id", "expected"),
    [
        (0, ((3, 0), (1, +half))),
        (15, ((1, +half), (2, 0))),
        (-1, ((2, 0), (3, 0))),
    ],
)
def test_get_decay_product_helicities(
    model_definition: ModelDefinition, chain_id: int, expected: tuple
):
    chain_defs = get_decay_chains(model_definition)
    assert _get_decay_product_helicities(chain_defs[chain_id]) == expected


def test_get_existing_subsystem_ids(model_definition: ModelDefinition):
    assert get_existing_subsystem_ids(model_definition) == [1, 2, 3]


@pytest.mark.parametrize("chain_id", range(26))
def test_get_final_state_helicities(model_definition: ModelDefinition, chain_id: int):
    chain_defs = get_decay_chains(model_definition)
    assert len(chain_defs) == 26
    λp = -half if chain_id in {19, 20, 22, 25} else +half
    assert _get_final_state_helicities(chain_defs[chain_id]) == {1: λp, 2: 0, 3: 0}


@pytest.mark.parametrize(
    ("chain_id", "expected_node", "expected_helicity"),
    [
        (0, (3, 1), +half),
        (1, (3, 1), -half),
        (-1, (2, 3), 0),
    ],
)
def test_get_resonance_helicity(
    model_definition: ModelDefinition,
    chain_id: int,
    expected_node: tuple[int, int],
    expected_helicity: sp.Rational,
):
    chain_defs = get_decay_chains(model_definition)
    node, helicity = _get_resonance_helicity(chain_defs[chain_id])
    assert node == expected_node
    assert helicity == expected_helicity


def test_get_weight(model_definition: ModelDefinition):
    chain_defs = get_decay_chains(model_definition)
    symbol, value = _get_weight(chain_defs[0])
    assert symbol.name == R"c^{L1405[1/2]}_{\frac{1}{2}, 0, 0}"
    assert value == pytest.approx(7.38649400481717 + 1.971018433257411j)


def test_parity_recoupling_does_not_duplicate_zero_helicities():
    recoupling = ParityRecoupling(λa=0, λb=0, λa0=0, λb0=0, f=1)  # ty: ignore[unknown-argument]
    assert recoupling.doit() == 1
