from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import sympy as sp

from ampform_dpd.io.serialization.amplitude import (
    _get_child_spins,
    _get_decay_product_ids,
    _get_resonance_node,
    _get_weight,
)

if TYPE_CHECKING:
    from ampform_dpd.io.serialization.format import ModelDefinition


def test_ls_vertices_need_no_fake_helicities(model_definition: ModelDefinition):
    model = deepcopy(model_definition)
    chain = model["distributions"][0]["decay_description"]["chains"][0]
    chain["vertices"] = [
        {"type": "ls", "l": "1", "s": "1/2", "node": [[3, 1], 2]},
        {"type": "ls", "l": "0", "s": "1/2", "node": [3, 1]},
    ]
    assert _get_decay_product_ids(chain) == (3, 1)
    assert _get_resonance_node(chain) == (3, 1)
    assert _get_child_spins(model, 0, 1) == (0, sp.Rational(1, 2))
    coupling, _ = _get_weight(chain)
    assert coupling.name.endswith("_{1, 1/2, 0, 1/2}")
