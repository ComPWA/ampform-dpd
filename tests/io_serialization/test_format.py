from __future__ import annotations

import pytest

from ampform_dpd.io.serialization.format import ModelDefinition, get_function_definition


def test_get_function_definition_blatt_weisskopf(model_definition: ModelDefinition):
    func_def = get_function_definition("BlattWeisskopf_resonance_l1", model_definition)
    assert func_def == {
        "name": "BlattWeisskopf_resonance_l1",
        "type": "BlattWeisskopf",
        "radius": 1.5,
        "l": 1,
    }
    func_def = get_function_definition("BlattWeisskopf_resonance_l2", model_definition)
    assert func_def == {
        "name": "BlattWeisskopf_resonance_l2",
        "type": "BlattWeisskopf",
        "radius": 1.5,
        "l": 2,
    }


def test_get_function_definition_missing(model_definition: ModelDefinition):
    # cspell:ignore Weiskopf
    with pytest.raises(
        KeyError,
        match=(
            r" Did you mean any of these\?"
            " BlattWeisskopf_b_decay_l1, BlattWeisskopf_resonance_l1, BlattWeisskopf_resonance_l2"
        ),
    ):
        get_function_definition("BlattWeiskopf", model_definition)
