from __future__ import annotations

import json
from pathlib import Path

import pytest

from ampform_dpd.io.serialization import load_workspace, validate_checksums

FIXTURE = Path(__file__).with_name("compass.json")
"""Two-chain distribution from the COMPASS model in amplitude-serialization.

Source: https://github.com/RUB-EP1/amplitude-serialization/blob/4bf857c9592a558943c32e42782c5f6cb90224b1/models/x2pipipi-compass-1391643.json
"""


def describe_serialized_couplings():
    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def it_reproduces_compass_intensities(backend: str):
        workspace = load_workspace(FIXTURE)
        results = validate_checksums(workspace, backend=backend)
        assert len(results) == 4
        assert all(result.diagnostic is None for result in results)
        assert all(result.difference < 1e-12 for result in results)

    @pytest.mark.parametrize("relative_sign", [1, -1], ids=["equal", "opposite"])
    def it_preserves_weights_across_subsystems(tmp_path: Path, relative_sign: int):
        definition = json.loads(FIXTURE.read_text())
        chains = definition["distributions"][0]["decay_description"]["chains"]
        chains[0]["weight"] = "1 + 2i"
        chains[1]["weight"] = f"{relative_sign}{2 * relative_sign:+}i"
        path = tmp_path / "model.json"
        path.write_text(json.dumps(definition))
        workspace = load_workspace(path)
        model = next(iter(workspace.distributions.values()))
        couplings = {
            symbol: value
            for symbol, value in model.parameter_defaults.items()
            if str(symbol).startswith("c^")
        }
        assert len(couplings) == 2
        assert sorted(couplings.values(), key=lambda value: value.real) == sorted(
            [1 + 2j, relative_sign * (1 + 2j)], key=lambda value: value.real
        )
