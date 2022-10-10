# pyright: reportPrivateUsage=false
from __future__ import annotations

import logging
import os
from os.path import abspath, dirname
from typing import TYPE_CHECKING

import pytest
import sympy as sp

from ampform_dpd.decay import IsobarNode, Particle
from ampform_dpd.io import _warn_about_unsafe_hash, aslatex, get_readable_hash

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture

THIS_DIR = dirname(abspath(__file__))

# https://compwa-org--129.org.readthedocs.build/report/018.html#resonances-and-ls-scheme
dummy_args = dict(mass=0, width=0)
Λc = Particle("Λc", latex=R"\Lambda_c^+", spin=0.5, parity=+1, **dummy_args)
p = Particle("p", latex="p", spin=0.5, parity=+1, **dummy_args)
π = Particle("π+", latex=R"\pi^+", spin=0, parity=-1, **dummy_args)
K = Particle("K-", latex="K^-", spin=0, parity=-1, **dummy_args)
Λ1520 = Particle("Λ(1520)", latex=R"\Lambda(1520)", spin=1.5, parity=-1, **dummy_args)


def test_aslatex_particle():
    latex = aslatex(Λ1520)
    assert latex == Λ1520.latex
    latex = aslatex(Λ1520, only_jp=True)
    assert latex == R"\frac{3}{2}^-"
    latex = aslatex(Λ1520, with_jp=True)
    assert latex == Λ1520.latex + R"\left[\frac{3}{2}^-\right]"


def test_aslatex_isobar_node():
    node = IsobarNode(Λ1520, p, K)
    latex = aslatex(node)
    assert latex == R"\Lambda(1520) \to p K^-"
    latex = aslatex(node, with_jp=True)
    assert (
        latex == R"\Lambda(1520)\left[\frac{3}{2}^-\right] \to"
        R" p\left[\frac{1}{2}^+\right] K^-\left[0^-\right]"
    )

    node = IsobarNode(Λ1520, p, K, interaction=(2, 1))
    latex = aslatex(node)
    assert latex == R"\Lambda(1520) \xrightarrow[S=1]{L=2} p K^-"


@pytest.mark.parametrize(
    ("assumptions", "expected_hash"),
    [
        (dict(), "pythonhashseed-0+7459658071388516764"),
        (dict(real=True), "pythonhashseed-0+3665410414623666716"),
        (dict(rational=True), "pythonhashseed-0-7926839224244779605"),
    ],
)
def test_get_readable_hash(assumptions, expected_hash, caplog: LogCaptureFixture):
    caplog.set_level(logging.WARNING)
    x, y = sp.symbols("x y", **assumptions)
    expr = x**2 + y
    h = get_readable_hash(expr)
    python_hash_seed = os.environ.get("PYTHONHASHSEED")
    if python_hash_seed is None or not python_hash_seed.isdigit():
        assert h[:7] == "bbc9833"
        if _warn_about_unsafe_hash.cache_info().hits == 0:
            assert "PYTHONHASHSEED has not been set." in caplog.text
            caplog.clear()
    elif python_hash_seed == "0":
        assert h == expected_hash
    else:
        pytest.skip("PYTHONHASHSEED has been set, but is not 0")
    assert caplog.text == ""
