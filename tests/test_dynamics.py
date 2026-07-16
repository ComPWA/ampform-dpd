from __future__ import annotations

from typing import cast

import pytest
import sympy as sp

from ampform_dpd.dynamics import ChannelArguments, MultichannelBreitWigner


def test_multichannel_breit_wigner_matches_reference_implementation():
    s = 2.0
    mass = 1.4051
    channel_definitions = [
        {
            "gsq": 0.328725260215546,
            "ma": 0.938272046,
            "mb": 0.493677,
            "l": 0,
            "d": 0,
        },
        {
            "gsq": 0.328725260215546,
            "ma": 1.18937,
            "mb": 0.13957018,
            "l": 0,
            "d": 0,
        },
    ]
    channels = tuple(
        ChannelArguments(
            *(cast("sp.Basic", sp.sympify(v)) for v in (s, mass, *channel.values()))
        )
        for channel in channel_definitions
    )

    expression = MultichannelBreitWigner(s, mass, channels)  # ty:ignore[invalid-argument-type]
    actual = complex(sp.N(expression.doit()))
    mass_width = sum(
        channel["gsq"]
        * 2
        * _breakup_momentum(s, channel["ma"], channel["mb"])
        / sp.sqrt(s)
        for channel in channel_definitions
    )
    expected = complex(1 / (mass**2 - s - sp.I * mass_width))

    assert actual == pytest.approx(expected)


def _breakup_momentum(s: float, m1: float, m2: float) -> sp.Expr:
    return sp.sqrt((s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2)) / (2 * sp.sqrt(s))
