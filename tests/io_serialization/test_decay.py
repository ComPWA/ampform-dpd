from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from ampform_dpd.io.serialization.decay import (
    _get_decay_chains,  # pyright:ignore[reportPrivateUsage]
    get_final_state,
    get_initial_state,
    to_decay_chain,
)

if TYPE_CHECKING:
    from ampform_dpd.io.serialization.format import ModelDefinition


def test_get_final_state(model_definition: ModelDefinition):
    final_state = get_final_state(model_definition)
    assert len(final_state) == 3
    assert set(final_state) == {1, 2, 3}
    assert {p.name for p in final_state.values()} == {"p", "pi", "K"}


def test_get_initial_state(model_definition: ModelDefinition):
    initial_state = get_initial_state(model_definition)
    assert initial_state.index == 0
    assert initial_state.name == "Lc"
    assert initial_state.latex == "Lc"
    assert initial_state.spin is sp.Rational(1 / 2)
    assert initial_state.mass == 2.28646
    assert initial_state.parity is None


def test_to_decay_chain(model_definition: ModelDefinition):
    chain_definitions = _get_decay_chains(model_definition)
    chain = to_decay_chain(
        chain_definitions[0],
        initial_state=get_initial_state(model_definition),
        final_state=get_final_state(model_definition),
    )
    assert chain.resonance.name == "L1405"
    assert chain.resonance.spin is sp.Rational(1 / 2)
    assert tuple(p.name for p in chain.decay_products) == ("K", "p")
    assert chain.spectator.name == "pi"
