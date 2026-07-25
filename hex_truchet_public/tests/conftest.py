import sys
from pathlib import Path

import pytest

ENV_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENV_ROOT.parent))

pytest_plugins = ["simulacrum.harness.plugin"]


@pytest.fixture
def harness_config():
    from simulacrum.harness import DiscreteActionSampler, HarnessConfig

    from hex_truchet_public.fast import HexTruchetPublicBatched
    from hex_truchet_public.reference import HexTruchetPublicReference

    return HarnessConfig(
        name="hex_truchet_public",
        root=ENV_ROOT,
        reference_factory=HexTruchetPublicReference,
        batched_factory=lambda n, debug=False: HexTruchetPublicBatched(n, debug=debug),
        action_sampler=DiscreteActionSampler(n_actions=2),  # TODO
        # scripted_policies=[...],  # strongly recommended (see ScriptedPolicy)
        # min_speedup=100.0,        # enforce the vectorization win
    )
