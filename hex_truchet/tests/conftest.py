import sys
from pathlib import Path

import pytest

ENV_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENV_ROOT.parent))

pytest_plugins = ["simulacrum.harness.plugin"]


@pytest.fixture
def harness_config():
    from simulacrum.harness import DiscreteActionSampler, HarnessConfig

    from hex_truchet.fast import HexTruchetBatched
    from hex_truchet.reference import HexTruchetReference

    return HarnessConfig(
        name="hex_truchet",
        root=ENV_ROOT,
        reference_factory=HexTruchetReference,
        batched_factory=lambda n, debug=False: HexTruchetBatched(n, debug=debug),
        action_sampler=DiscreteActionSampler(n_actions=666),  # spec: Actions
        # scripted_policies=[...],  # strongly recommended (see ScriptedPolicy)
        # min_speedup=100.0,        # enforce the vectorization win
    )
