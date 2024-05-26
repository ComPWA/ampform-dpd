from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from ampform_dpd.io.serialization.decay import (
    get_decay_chains,  # pyright:ignore[reportPrivateUsage]
    get_final_state,
    get_initial_state,
    get_states,
    to_decay,
    to_decay_chain,
)
from ampform_dpd.io.serialization.format import ModelDefinition

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


def test_get_states(model_definition: ModelDefinition):
    states = get_states(model_definition)
    assert len(states) == 4
    assert {i: s.name for i, s in states.items()} == {0: "Lc", 1: "p", 2: "pi", 3: "K"}


def test_to_decay(model_definition: ModelDefinition):
    decay = to_decay(model_definition)
    assert decay.initial_state.name == "Lc"
    assert {p.name for p in decay.final_state.values()} == {"p", "pi", "K"}
    assert len(decay.chains) == 12
    assert [c.resonance.name for c in decay.chains] == [
        "D1232",
        "D1600",
        "D1700",
        "K1430",
        "K700",
        "K892",
        "L1405",
        "L1520",
        "L1600",
        "L1670",
        "L1690",
        "L2000",
    ]


def test_to_decay_chain(model_definition: ModelDefinition):
    chain_definitions = get_decay_chains(model_definition)
    chain = to_decay_chain(
        chain_definitions[0],
        initial_state=get_initial_state(model_definition),
        final_state=get_final_state(model_definition),
    )
    assert chain.resonance.name == "L1405"
    assert chain.resonance.spin is sp.Rational(1 / 2)
    assert tuple(p.name for p in chain.decay_products) == ("K", "p")
    assert chain.spectator.name == "pi"
