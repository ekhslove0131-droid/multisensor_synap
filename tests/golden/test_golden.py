from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from multisensor_synth.orchestration.pipeline import generate_truth_run

HERE = Path(__file__).parent


def test_reviewed_golden_truth_fixture(tmp_path: Path) -> None:
    generated = generate_truth_run(HERE / "golden.yaml", tmp_path)
    manifest = generated.manifest
    latent = pq.read_table(generated.run_dir / "truth" / "latent_timeline.parquet")
    factor_names = (
        "autonomic_arousal",
        "motor_activation",
        "cognitive_load",
        "sleep_pressure",
        "sensory_context",
        "recovery_capacity",
        "social_context",
    )
    table_hashes = {
        path: details["logical_sha256"]
        for path, details in manifest["tables"].items()
    }
    actual = {
        "config_sha256": manifest["config_sha256"],
        "participants_logical_sha256": table_hashes[
            "truth/participants.parquet"
        ],
        "events_logical_sha256": table_hashes["truth/events.parquet"],
        "latent_factor_means": {
            name: round(float(pc.mean(latent[name]).as_py()), 8)
            for name in factor_names
        },
        "truth_logical_sha256": hashlib.sha256(
            json.dumps(table_hashes, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    expected = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))

    assert actual == expected
