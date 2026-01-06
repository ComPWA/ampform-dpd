from __future__ import annotations

from typing import TYPE_CHECKING

from qrules.transition import ReactionInfo

from ampform_dpd import _get_best_reference_subsystems
from ampform_dpd.adapter.qrules import to_three_body_decay
from ampform_dpd.decay import IsobarNode, Particle

if TYPE_CHECKING:
    from qrules.transition import ReactionInfo

# https://compwa-org--129.org.readthedocs.build/report/018.html#resonances-and-ls-scheme
dummy_args = dict(mass=0, width=0)
Λc = Particle("Λc", latex=R"\Lambda_c^+", spin=0.5, parity=+1, **dummy_args)  # ty:ignore[invalid-argument-type]
p = Particle("p", latex="p", spin=0.5, parity=+1, **dummy_args)  # ty:ignore[invalid-argument-type]
π = Particle("π+", latex=R"\pi^+", spin=0, parity=-1, **dummy_args)  # ty:ignore[invalid-argument-type]
K = Particle("K-", latex="K^-", spin=0, parity=-1, **dummy_args)  # ty:ignore[invalid-argument-type]
Λ1520 = Particle("Λ(1520)", latex=R"\Lambda(1520)", spin=1.5, parity=-1, **dummy_args)  # ty:ignore[invalid-argument-type]


class TestIsobarNode:
    def test_children(self):
        decay = IsobarNode(Λ1520, p, K)  # ty:ignore[invalid-argument-type]
        assert decay.children == (p, K)

    def test_ls(self):
        L, S = 2, 1
        node = IsobarNode(Λ1520, p, K, interaction=(L, S))  # ty:ignore[invalid-argument-type]
        assert node.interaction is not None
        assert node.interaction.L == L
        assert node.interaction.S == S


def test_get_best_reference_subsystems(
    a2pipipi_reaction: ReactionInfo,
    xib2pkk_reaction: ReactionInfo,
    jpsi2pksigma_reaction: ReactionInfo,
):
    decay = to_three_body_decay(a2pipipi_reaction.transitions)
    assert _get_best_reference_subsystems(decay) == 1
    decay = to_three_body_decay(xib2pkk_reaction.transitions)
    assert _get_best_reference_subsystems(decay) == 2
    decay = to_three_body_decay(jpsi2pksigma_reaction.transitions)
    assert _get_best_reference_subsystems(decay) == 2
