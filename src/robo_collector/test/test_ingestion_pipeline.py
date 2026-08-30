from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from robo_collector.ingestion_pipeline import (
    IngestionLedger,
    StageClaimError,
)

from robo_collector import mcap_contract


def _stage_key(name: str) -> dict[str, object]:
    parameters = {}
    if name == "align_rgb":
        parameters["reference_camera_stream"] = "rgb"
    elif name == "encode_video":
        parameters["backend_version"] = "17.1"
    definition = mcap_contract.STAGE_REGISTRY[name]
    return mcap_contract.build_stage_key(
        name,
        config=mcap_contract.build_stage_config(name, **parameters),
        implementation_id="tests@tree",
        input_hashes={
            input_name: hashlib.sha256(input_name.encode()).hexdigest()
            for input_name in definition.input_names
        },
    )


class IngestionLedgerTest(unittest.TestCase):
    def test_claim_reuse_and_expired_owner_fencing(self):
        with (
            tempfile.TemporaryDirectory() as root,
            IngestionLedger(Path(root) / "ledger.sqlite") as ledger,
        ):
            key = _stage_key("validate_landing")
            first = ledger.claim(key, owner="one", lease_duration_ns=10, now_ns=100)
            with self.assertRaises(StageClaimError):
                ledger.claim(key, owner="two", lease_duration_ns=10, now_ns=109)
            second = ledger.claim(key, owner="two", lease_duration_ns=10, now_ns=110)
            self.assertEqual(second.fencing_token, first.fencing_token + 1)
            self.assertEqual(second.generation, 0)
            with self.assertRaises(StageClaimError):
                ledger.complete(first, output_hashes=[], now_ns=110)
            outputs = [
                {"path": path, "sha256": "a" * 64, "size_bytes": "1"}
                for path in mcap_contract.STAGE_REGISTRY[
                    "validate_landing"
                ].output_paths
            ]
            ledger.complete(second, output_hashes=outputs, now_ns=111)
            reused = ledger.claim(key, owner="three", lease_duration_ns=10, now_ns=200)
            self.assertTrue(reused.reused)
            self.assertEqual(reused.generation, 1)

    def test_stage_outputs_are_hash_checked_fsynced_and_immutable(self):
        with (
            tempfile.TemporaryDirectory() as root,
            IngestionLedger(Path(root) / "ledger.sqlite") as ledger,
        ):
            claim = ledger.claim(
                _stage_key("validate_landing"),
                owner="validator",
                lease_duration_ns=100,
                now_ns=1,
            )
            staging = Path(root) / "staging"
            staging.mkdir()
            outputs = []
            for name in mcap_contract.STAGE_REGISTRY["validate_landing"].output_paths:
                payload = f"payload:{name}".encode()
                (staging / name).write_bytes(payload)
                outputs.append(
                    {
                        "path": name,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size_bytes": str(len(payload)),
                    }
                )
            installed = ledger.install_stage_outputs(
                claim,
                staging_dir=staging,
                final_dir=Path(root) / "outputs",
                output_hashes=outputs,
                now_ns=2,
            )
            self.assertFalse(staging.exists())
            self.assertEqual(
                (installed / outputs[0]["path"]).read_bytes(),
                f"payload:{outputs[0]['path']}".encode(),
            )

    def test_complete_dag_writes_canonical_snapshot(self):
        with (
            tempfile.TemporaryDirectory() as root,
            IngestionLedger(Path(root) / "ledger.sqlite") as ledger,
        ):
            for name in mcap_contract.PREPUBLICATION_STAGE_ORDER:
                claim = ledger.claim(
                    _stage_key(name), owner=name, lease_duration_ns=100, now_ns=1
                )
                outputs = [
                    {
                        "path": path,
                        "sha256": hashlib.sha256(path.encode()).hexdigest(),
                        "size_bytes": "1",
                    }
                    for path in mcap_contract.STAGE_REGISTRY[name].output_paths
                ]
                ledger.complete(claim, output_hashes=outputs, now_ns=2)
            target = Path(root) / "stage-ledger.prepublish.json"
            digest = ledger.write_prepublish_snapshot(target)
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), digest)
            self.assertEqual(
                mcap_contract.parse_canonical_json(target.read_bytes())["format"],
                "robo_collector.stage_evidence",
            )

    def test_publication_rejects_nonpublication_authority(self):
        with (
            tempfile.TemporaryDirectory() as root,
            IngestionLedger(Path(root) / "ledger.sqlite") as ledger,
        ):
            claim = ledger.claim(
                _stage_key("validate_landing"),
                owner="validator",
                lease_duration_ns=100,
                now_ns=1,
            )
            with self.assertRaisesRegex(StageClaimError, "publish_canonical"):
                ledger.publish_ready(
                    claim,
                    staging_dir=Path(root) / "staging",
                    canonical_dir=Path(root) / "canonical",
                    manifest={},
                    pointer={},
                    now_ns=2,
                )


if __name__ == "__main__":
    unittest.main()
