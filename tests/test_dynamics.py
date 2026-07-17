from __future__ import annotations

from typing import cast

import pytest
import sympy as sp
from ampform.dynamics.form_factor import FormFactor
from ampform.dynamics.phasespace import BreakupMomentum

from ampform_dpd.dynamics import BreitWigner, ChannelArguments, MultichannelBreitWigner


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


def test_breit_wigner_equals_single_channel_breit_wigner():
    """https://github.com/mmikhasenko/HadronicLineshapes.jl/blob/78f1e04/test/test-lineshapes.jl#L13-L20"""
    s = 2.2
    mass = 1.6
    width = 0.2
    m1 = 0.1
    m2 = 0.2
    angular_momentum = sp.Integer(1)
    meson_radius = 1.5
    p0 = BreakupMomentum(mass**2, m1, m2)  # ty:ignore[invalid-argument-type]
    ff0 = FormFactor(mass**2, m1, m2, angular_momentum, meson_radius)  # ty:ignore[invalid-argument-type]
    coupling_squared = mass**2 * width / (2 * p0 * ff0**2)
    channel = ChannelArguments(
        s,  # ty:ignore[invalid-argument-type]
        mass,  # ty:ignore[invalid-argument-type]
        coupling_squared,
        m1,  # ty:ignore[invalid-argument-type]
        m2,  # ty:ignore[invalid-argument-type]
        angular_momentum,
        meson_radius,  # ty:ignore[invalid-argument-type]
    )

    multichannel_bw = MultichannelBreitWigner(s, mass, (channel,))  # ty:ignore[invalid-argument-type]
    breit_wigner = BreitWigner(s, mass, width, m1, m2, angular_momentum, meson_radius)  # ty:ignore[invalid-argument-type]

    assert _evaluate(multichannel_bw) == pytest.approx(_evaluate(breit_wigner))
    assert _evaluate(multichannel_bw) == pytest.approx(
        1.6503335292467654 + 1.3640597661785752j
    )


def test_two_channel_s_wave_matches_flatte_reference():
    """https://github.com/mmikhasenko/HadronicLineshapes.jl/blob/78f1e04/test/test-lineshapes.jl#L24-L36"""
    expression = _create_two_channel_breit_wigner(sp.Float(2.2))
    assert _evaluate(expression) == pytest.approx(
        0.6268627705629269 + 1.1611754995124888j
    )


@pytest.mark.parametrize(
    ("s", "expected"),
    [
        (0.55**2, 0.4358515993981993 + 0.055692403570256954j),
        (0.54**2, 0.42179982350210526 + 0.05164365276548291j),
    ],
)
def test_multichannel_breit_wigner_below_threshold(s: sp.Expr, expected: complex):
    """https://github.com/mmikhasenko/HadronicLineshapes.jl/blob/78f1e04/test/test-lineshapes.jl#L38-L43"""
    assert _evaluate(_create_two_channel_breit_wigner(s)) == pytest.approx(expected)


def test_multichannel_breit_wigner_below_threshold_with_imaginary_epsilon():
    """https://github.com/mmikhasenko/HadronicLineshapes.jl/blob/78f1e04/test/test-lineshapes.jl#L38-L43"""
    s = sp.Float(0.54**2)
    above_cut = _create_two_channel_breit_wigner(s + sp.I * sp.Float(2**-52))
    on_cut = _create_two_channel_breit_wigner(s)

    assert _evaluate(on_cut) == pytest.approx(_evaluate(above_cut))


def _create_two_channel_breit_wigner(s: sp.Expr) -> MultichannelBreitWigner:
    mass = sp.Float(1.6)
    channels = (
        ChannelArguments(
            s,
            mass,
            width=0.35,  # ty:ignore[unknown-argument]
            m1=0.1,  # ty:ignore[unknown-argument]
            m2=0.2,  # ty:ignore[unknown-argument]
            angular_momentum=0,  # ty:ignore[unknown-argument]
            meson_radius=1,  # ty:ignore[unknown-argument]
        ),
        ChannelArguments(
            s,
            mass,
            width=0.35,  # ty:ignore[unknown-argument]
            m1=0.3,  # ty:ignore[unknown-argument]
            m2=0.25,  # ty:ignore[unknown-argument]
            angular_momentum=0,  # ty:ignore[unknown-argument]
            meson_radius=1.5,  # ty:ignore[unknown-argument]
        ),
    )
    return MultichannelBreitWigner(s, mass, channels)  # ty:ignore[invalid-argument-type]


def _evaluate(expression: sp.Expr) -> complex:
    return complex(sp.N(expression.doit()))


def _breakup_momentum(s: float, m1: float, m2: float) -> sp.Expr:
    return sp.sqrt((s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2)) / (2 * sp.sqrt(s))
