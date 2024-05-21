# pyright:reportPrivateUsage=false
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import sympy as sp

from ampform_dpd.io.serialization.amplitude import (
    _get_decay_product_helicities,
    _get_final_state_helicities,
    _get_resonance_helicity,
    _get_weight,
    get_existing_subsystem_ids,
)
from ampform_dpd.io.serialization.format import get_decay_chains

if TYPE_CHECKING:
    from ampform_dpd.io.serialization.format import ModelDefinition


def test_get_decay_product_helicities(model_definition: ModelDefinition):
    chain_defs = get_decay_chains(model_definition)
    half = sp.Rational(1 / 2)
    assert _get_decay_product_helicities(chain_defs[0]) == ((3, 0), (1, +half))
    assert _get_decay_product_helicities(chain_defs[15]) == ((1, +half), (2, 0))
    assert _get_decay_product_helicities(chain_defs[-1]) == ((2, 0), (3, 0))


def test_get_existing_subsystem_ids(model_definition: ModelDefinition):
    assert get_existing_subsystem_ids(model_definition) == [1, 2, 3]


@pytest.mark.parametrize("chain_id", range(26))
def test_get_final_state_helicities(model_definition: ModelDefinition, chain_id: int):
    chain_defs = get_decay_chains(model_definition)
    assert len(chain_defs) == 26
    if chain_id in {19, 20, 22, 25}:
        λp = -sp.Rational(1 / 2)
    else:
        λp = +sp.Rational(1 / 2)
    assert _get_final_state_helicities(chain_defs[chain_id]) == {1: λp, 2: 0, 3: 0}


def test_get_resonance_helicity(model_definition: ModelDefinition):
    chain_defs = get_decay_chains(model_definition)
    half = sp.Rational(1 / 2)
    node, helicity = _get_resonance_helicity(chain_defs[0])
    assert node == (3, 1)
    assert helicity == +half
    node, helicity = _get_resonance_helicity(chain_defs[1])
    assert node == (3, 1)
    assert helicity == -half
    node, helicity = _get_resonance_helicity(chain_defs[-1])
    assert node == (2, 3)
    assert helicity == 0


def test_get_weight(model_definition: ModelDefinition):
    chain_defs = get_decay_chains(model_definition)
    symbol, value = _get_weight(chain_defs[0])
    assert symbol.name == R"c^{L1405[1/2]}_{\frac{1}{2}, 0, 0}"
    assert value == 7.38649400481717 + 1.971018433257411j
