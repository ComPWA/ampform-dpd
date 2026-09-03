from __future__ import annotations

import pytest

from ampform_dpd.io.serialization.format import ModelDefinition, get_function_definition


def describe_get_function_definition():
    @pytest.mark.parametrize("angular_momentum", [1, 2])
    def it_returns_a_blatt_weisskopf_definition(
        model_definition: ModelDefinition, angular_momentum: int
    ):
        name = f"BlattWeisskopf_resonance_l{angular_momentum}"
        assert get_function_definition(name, model_definition) == {
            "name": name,
            "type": "BlattWeisskopf",
            "radius": 1.5,
            "l": angular_momentum,
        }

    def it_suggests_similar_names_when_missing(model_definition: ModelDefinition):
        # cspell:ignore Weiskopf
        with pytest.raises(
            KeyError,
            match=(
                r" Did you mean any of these\?"
                " BlattWeisskopf_b_decay_l1, BlattWeisskopf_resonance_l1, BlattWeisskopf_resonance_l2"
            ),
        ):
            get_function_definition("BlattWeiskopf", model_definition)
