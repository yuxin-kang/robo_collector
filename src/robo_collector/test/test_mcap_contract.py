import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import rfc8785
from robo_collector.mcap.v1 import episode_pb2

from robo_collector import mcap_contract

try:
    import pyarrow as pa
except ImportError:  # pragma: no cover - required in the data-collection env
    pa = None


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mcap_v1_contract_golden.json"


def _fixture():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _reference_hash(value):
    return hashlib.sha256(rfc8785.dumps(value)).hexdigest()


def _quality_rule(*, severity, result, rule_id="rule.1"):
    return {
        "evidence_sha256": hashlib.sha256(b"").hexdigest(),
        "metrics": [],
        "result": result,
        "rule_id": rule_id,
        "severity": severity,
    }


def _stage_evidence():
    stages = []
    for index, name in enumerate(mcap_contract.PREPUBLICATION_STAGE_ORDER, start=1):
        digest = f"{index:x}" * 64
        definition = mcap_contract.STAGE_REGISTRY[name]
        stage_key = {
            "config_sha256": digest,
            "implementation_id": "robo-collector-1+tree-deadbeef",
            "input_hashes": [
                {"name": input_name, "sha256": digest}
                for input_name in sorted(definition.input_names)
            ],
            "output_schema_version": "1",
            "stage_name": name,
            "stage_version": "1",
        }
        stages.append(
            dict(
                stage_key,
                output_hashes=[
                    {"path": path, "sha256": digest, "size_bytes": str(index)}
                    for path in sorted(definition.output_paths)
                ],
                stage_key_sha256=mcap_contract.stage_key_hash(stage_key),
            )
        )
    return stages


def _ready_manifest(status="READY"):
    paths = (
        "camera.mcap",
        "provenance.json",
        "quality.json",
        "robot.mcap",
        "stage-ledger.prepublish.json",
        "video-keyframes.parquet",
    )
    members = [
        {"path": path, "sha256": f"{index:x}" * 64, "size_bytes": str(index)}
        for index, path in enumerate(paths, start=1)
    ]
    identity = {
        "alignment": {"policy": "rgb_affine_v2", "policy_version": "2"},
        "codec": {
            "config_sha256": "a" * 64,
            "name": "h264",
            "packetization": "annex_b_access_unit",
            "profile_version": "h264_annexb_au_v1",
        },
        "episode_id": "episode.phase0",
        "format": "robo_collector.canonical_bundle",
        "format_version": 1,
        "members": members,
        "pipeline": {
            "implementation_id": "robo-collector-1+tree-deadbeef",
            "stage_semantics_version": "1",
        },
        "schema": {
            "mcap_profile": "robo_collector.mcap.v1",
            "protobuf_descriptor_sha256": mcap_contract.descriptor_sha256(),
        },
        "source_artifacts": [
            {"name": "landing.mcap", "sha256": "b" * 64, "size_bytes": "1"}
        ],
    }
    inventory_files = []
    for member in members:
        inventory_files.append(
            dict(
                member,
                message_count=(
                    "1" if member["path"] in {"camera.mcap", "robot.mcap"} else None
                ),
            )
        )
    inventory = {
        "checksums_sha256": _reference_hash(
            {
                "algorithm": "sha256",
                "format": "robo_collector.checksums",
                "format_version": 1,
                "members": members,
            }
        ),
        "end_log_time_ns": "101",
        "files": inventory_files,
        "libraries": [
            {"name": "mcap", "version": "1.4.0"},
            {"name": "protobuf", "version": "6.32.0"},
            {"name": "pyarrow", "version": "24.0.0"},
        ],
        "start_log_time_ns": "100",
        "topic_counts": [
            {"count": "1", "topic": "/camera/head/h264"},
            {"count": "1", "topic": "/robot/state/raw"},
        ],
        "total_message_count": "2",
    }
    return {
        "bundle_hash": _reference_hash(identity),
        "canonical_status": status,
        "identity": identity,
        "inventory": inventory,
        "manifest_version": 1,
    }


def _pointer(manifest):
    return {
        "bundle_hash": manifest["bundle_hash"],
        "episode_id": "episode.phase0",
        "manifest_hash": _reference_hash(manifest),
        "publication_generation": "1",
        "publication_profile": "lerobot",
        "publisher_fencing_token": "1",
        "publisher_stage_key": "a" * 64,
    }


class McapContractGoldenTest(unittest.TestCase):
    def test_descriptor_and_protobuf_serialization_goldens(self):
        fixture = _fixture()["proto"]
        descriptor = mcap_contract.descriptor_set_bytes()
        self.assertEqual(hashlib.sha256(descriptor).hexdigest(), fixture["descriptor_sha256"])
        self.assertEqual(str(len(descriptor)), fixture["descriptor_size_bytes"])
        self.assertEqual(mcap_contract.descriptor_sha256(), fixture["descriptor_sha256"])

        for name, vectors in fixture["messages"].items():
            message_type = getattr(episode_pb2, name)
            minimal = message_type()
            self.assertEqual(
                minimal.SerializeToString(deterministic=True).hex(), vectors["minimal_hex"]
            )
            populated = message_type.FromString(bytes.fromhex(vectors["full_hex"]))
            self.assertEqual(
                populated.SerializeToString(deterministic=True).hex(), vectors["full_hex"]
            )

    def test_checkpoint_frame_matches_independent_hash_crc_and_bytes(self):
        fixture = _fixture()["checkpoint"]
        body = rfc8785.dumps(fixture["payload"])
        self.assertEqual(hashlib.sha256(body).hexdigest(), fixture["payload_sha256"])
        self.assertEqual(mcap_contract.encode_checkpoint_frame(fixture["payload"]).hex(), fixture["frame_hex"])
        self.assertEqual(
            mcap_contract.decode_checkpoint_frame(bytes.fromhex(fixture["frame_hex"])),
            fixture["payload"],
        )

        corrupted = bytearray.fromhex(fixture["frame_hex"])
        corrupted[-1] ^= 1
        with self.assertRaises(mcap_contract.CheckpointFrameError):
            mcap_contract.decode_checkpoint_frame(corrupted)

    def test_inprogress_manifest_golden_is_closed_and_canonical(self):
        fixture = _fixture()["inprogress_manifest"]
        encoded = mcap_contract.encode_inprogress_manifest(fixture["payload"])
        self.assertEqual(hashlib.sha256(encoded).hexdigest(), fixture["sha256"])
        self.assertEqual(encoded, rfc8785.dumps(fixture["payload"]))
        self.assertEqual(mcap_contract.decode_inprogress_manifest(encoded), fixture["payload"])

        invalid = dict(fixture["payload"], runtime_host="not-semantic")
        with self.assertRaises(mcap_contract.McapContractError):
            mcap_contract.encode_inprogress_manifest(invalid)

    def test_stage_config_and_stage_key_match_independent_jcs(self):
        fixture = _fixture()["stage_key"]
        self.assertEqual(_reference_hash(fixture["config"]), fixture["config_sha256"])
        self.assertEqual(_reference_hash(fixture["preimage"]), fixture["stage_key_sha256"])

        built = mcap_contract.build_stage_key(
            "validate_landing",
            config=fixture["config"],
            implementation_id=fixture["preimage"]["implementation_id"],
            input_hashes=reversed(fixture["preimage"]["input_hashes"]),
        )
        self.assertEqual(built, fixture["preimage"])
        self.assertEqual(
            mcap_contract.stage_key_hash(built), fixture["stage_key_sha256"]
        )

        encoder = mcap_contract.build_stage_config(
            "encode_video", backend_version="17.1"
        )
        self.assertEqual(encoder["backend"], "pyav-libx264")
        self.assertEqual(encoder["backend_version"], "17.1")

    def test_mcap_dependency_versions_are_pinned(self):
        requirements = (FIXTURE_PATH.parents[4] / "requirements-mcap.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        self.assertIn("mcap==1.4.0", requirements)
        self.assertIn("rfc8785==0.1.4", requirements)
        self.assertIn("av==17.1.0", requirements)

    def test_quality_precedence_is_quarantine_reject_review_ready(self):
        cases = {
            "QUARANTINED": ([_quality_rule(severity="CRITICAL", result="FAIL")], True),
            "REJECT": ([_quality_rule(severity="CRITICAL", result="FAIL")], False),
            "REVIEW": ([_quality_rule(severity="WARNING", result="FAIL")], False),
            "READY": ([], False),
            "READY_INFO_FAIL": ([_quality_rule(severity="INFO", result="FAIL")], False),
        }
        for expected, (rules, quarantined) in cases.items():
            with self.subTest(expected=expected):
                report = mcap_contract.reduce_quality_rules(
                    rules,
                    policy_name="canonical_content_v1",
                    policy_version="1",
                    policy_config={"policy": "canonical_content_v1", "version": "1"},
                    quarantined=quarantined,
                )
                self.assertEqual(
                    report["canonical_status"],
                    "READY" if expected == "READY_INFO_FAIL" else expected,
                )

        self.assertEqual(
            [row["expected"] for row in _fixture()["qc_precedence"]],
            ["QUARANTINED", "REJECT", "REVIEW", "READY", "READY"],
        )

    def test_retry_metadata_cannot_change_semantic_stage_evidence(self):
        semantic = _stage_evidence()
        attempts = []
        for owner, token in (("worker-a", "1"), ("worker-b", "9")):
            attempts.append(
                {
                    "semantic": copy.deepcopy(semantic),
                    "execution": {
                        "attempt_uuid": f"attempt-{token}",
                        "owner": owner,
                        "fencing_token": token,
                        "started_time_ns": str(int(token) * 100),
                    },
                }
            )
        projections = [
            mcap_contract.project_stage_evidence(attempt["semantic"])
            for attempt in attempts
        ]
        self.assertEqual(projections[0], projections[1])
        self.assertEqual(_reference_hash(projections[0]), _reference_hash(projections[1]))

        contaminated = copy.deepcopy(semantic)
        contaminated[0]["owner"] = "worker-a"
        with self.assertRaises(mcap_contract.McapContractError):
            mcap_contract.project_stage_evidence(contaminated)

    def test_stage_version_overrides_and_duplicate_evidence_are_rejected(self):
        fixture = _fixture()["stage_key"]
        with self.assertRaises(mcap_contract.McapContractError):
            mcap_contract.build_stage_key(
                "validate_landing",
                config=fixture["config"],
                implementation_id=fixture["preimage"]["implementation_id"],
                input_hashes=fixture["preimage"]["input_hashes"],
                stage_version="2",
            )

        duplicate = _stage_evidence()
        duplicate[-1] = copy.deepcopy(duplicate[0])
        with self.assertRaises(mcap_contract.McapContractError):
            mcap_contract.project_stage_evidence(duplicate)

    def test_metric_values_and_units_reject_noncanonical_encodings(self):
        for value in ("NaN", "Infinity", "1e3", "01", "-0", "2/4", "1/0", "a,b"):
            with self.subTest(value=value), self.assertRaises(
                mcap_contract.McapContractError
            ):
                mcap_contract.validate_metric_value(value)
        for unit in ("", "m s", "µs", "m,sec"):
            with self.subTest(unit=unit), self.assertRaises(
                mcap_contract.McapContractError
            ):
                mcap_contract.validate_metric_unit(unit)

    def test_channel_metadata_rejects_noncanonical_rates_and_topic_mismatch(self):
        metadata = {
            "robo.calibration_revision": "cal.1",
            "robo.clock_domain": "clock.0",
            "robo.codec": "h264",
            "robo.frame_id": "head",
            "robo.nominal_rate_hz": "30",
            "robo.observed_rate_hz": "30",
            "robo.pipeline_version": "1",
            "robo.schema_version": "1",
            "robo.sensor_id": "camera.0",
            "robo.source_id": "camera.0",
            "robo.stream_id": "head",
        }
        for rate in ("NaN", "Infinity", "30.0", "3e1"):
            invalid = dict(metadata, **{"robo.observed_rate_hz": rate})
            with self.subTest(rate=rate), self.assertRaises(
                mcap_contract.McapContractError
            ):
                mcap_contract.validate_channel_metadata(
                    invalid, family="camera", topic="/camera/head/h264"
                )

        mismatched = dict(metadata, **{"robo.stream_id": "wrist"})
        with self.assertRaises(mcap_contract.McapContractError):
            mcap_contract.validate_channel_metadata(
                mismatched, family="camera", topic="/camera/head/h264"
            )

    def test_checkpoint_progress_and_internal_ranges_cannot_regress(self):
        previous = copy.deepcopy(_fixture()["checkpoint"]["payload"])
        previous.update(
            {
                "accepted_snapshot_count": "2",
                "accepted_snapshot_frontier": "2",
                "checkpoint_sequence": "4",
                "durable_byte_offset": "10",
                "durable_count": "1",
                "durable_frontier": "1",
                "written_count": "2",
                "written_frontier": "2",
            }
        )
        current = copy.deepcopy(previous)
        current["checkpoint_sequence"] = "5"
        current["written_count"] = "1"
        current["written_frontier"] = "1"
        with self.assertRaises(mcap_contract.CheckpointFrameError):
            mcap_contract.validate_checkpoint_payload(current, previous=previous)

        wrong_sequence = copy.deepcopy(previous)
        with self.assertRaises(mcap_contract.CheckpointFrameError):
            mcap_contract.validate_checkpoint_payload(
                wrong_sequence, previous=previous
            )

        invalid_range = copy.deepcopy(previous)
        invalid_range["accepted_snapshot_count"] = "1"
        invalid_range["accepted_snapshot_frontier"] = "1"
        with self.assertRaises(mcap_contract.CheckpointFrameError):
            mcap_contract.validate_checkpoint_payload(invalid_range)

        invalid_fence = copy.deepcopy(_fixture()["checkpoint"]["payload"])
        invalid_fence["source_fences"] = [
            {
                "accepted_count": "1",
                "durable_count": "1",
                "durable_high_watermark": "4",
                "end_sequence_inclusive": "4",
                "session_id": "session.0",
                "source_id": "camera.0",
                "start_sequence_exclusive": "4",
                "written_count": "1",
                "written_high_watermark": "4",
            }
        ]
        with self.assertRaises(mcap_contract.CheckpointFrameError):
            mcap_contract.validate_checkpoint_payload(invalid_fence)

    @unittest.skipUnless(pa is not None, "pyarrow is required for keyframe schema validation")
    def test_keyframe_arrow_schema_is_exact_ordered_and_non_nullable(self):
        expected = _fixture()["keyframe_arrow_schema"]
        schema = mcap_contract.keyframe_arrow_schema()
        actual = [[field.name, str(field.type), field.nullable] for field in schema]
        self.assertEqual(actual, expected)
        self.assertEqual(schema.field("codec_config_sha256").type, pa.binary(32))

    def test_nonready_publication_is_denied_without_pointer_mutation(self):
        fixture = _fixture()["publication"]
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "current.json"
            current.write_bytes(b"existing-pointer")
            for status in fixture["denied_statuses"]:
                manifest = _ready_manifest(status)
                pointer = _pointer(manifest)
                with self.subTest(status=status):
                    with self.assertRaises(mcap_contract.PublicationError):
                        mcap_contract.publish_ready_pointer(directory, manifest, pointer)
                    self.assertEqual(current.read_bytes(), b"existing-pointer")

    def test_ready_publication_validates_inventory_and_replaces_pointer(self):
        manifest = _ready_manifest()
        pointer = _pointer(manifest)
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "current.json"
            current.write_bytes(b"old-pointer")

            published = mcap_contract.publish_ready_pointer(
                directory, manifest, pointer, prevalidated_authority=True
            )
            self.assertEqual(published, current)
            self.assertEqual(json.loads(current.read_bytes()), pointer)
            self.assertEqual(current.read_bytes(), rfc8785.dumps(pointer))

            invalid = copy.deepcopy(manifest)
            invalid["inventory"] = {}
            with self.assertRaises(mcap_contract.PublicationError):
                mcap_contract.assert_ready_manifest(invalid)

    def test_ready_identity_inventory_and_pointer_binding_fail_closed(self):
        valid = _ready_manifest()
        cases = []

        malformed_identity = copy.deepcopy(valid)
        del malformed_identity["identity"]["schema"]
        malformed_identity["bundle_hash"] = _reference_hash(
            malformed_identity["identity"]
        )
        cases.append(("malformed_identity", malformed_identity))

        wrong_mcap = copy.deepcopy(valid)
        wrong_mcap["inventory"]["libraries"][0]["version"] = "1.3.0"
        cases.append(("wrong_mcap", wrong_mcap))

        wrong_checksums = copy.deepcopy(valid)
        wrong_checksums["inventory"]["checksums_sha256"] = "0" * 64
        cases.append(("wrong_checksums", wrong_checksums))

        mismatched_member = copy.deepcopy(valid)
        mismatched_member["inventory"]["files"][0]["sha256"] = "0" * 64
        cases.append(("mismatched_member", mismatched_member))

        for name, manifest in cases:
            with self.subTest(name=name), self.assertRaises(
                mcap_contract.PublicationError
            ):
                mcap_contract.assert_ready_manifest(manifest)

        pointer = _pointer(valid)
        pointer["episode_id"] = "episode.other"
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "current.json"
            current.write_bytes(b"existing-pointer")
            with self.assertRaises(mcap_contract.PublicationError):
                mcap_contract.publish_ready_pointer(
                    directory, valid, pointer, prevalidated_authority=True
                )
                self.assertEqual(current.read_bytes(), b"existing-pointer")

    def test_direct_ready_aliases_reject_root_escape_and_preserve_sentinels(self):
        manifest = _ready_manifest()
        pointer = _pointer(manifest)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            external = root / "external"
            external.mkdir()
            current = external / "current.json"
            current.write_bytes(b"external-sentinel")
            escaped = root / "canonical"
            escaped.symlink_to(external, target_is_directory=True)
            for helper in (
                mcap_contract.publish_ready_pointer,
                mcap_contract.publish_ready_bundle,
            ):
                with self.subTest(helper=helper), self.assertRaises(
                    mcap_contract.PublicationError
                ):
                    helper(escaped, manifest, pointer, prevalidated_authority=True)
            self.assertEqual(current.read_bytes(), b"external-sentinel")

    def test_direct_ready_temp_collision_preserves_pointer_bytes(self):
        manifest = _ready_manifest()
        pointer = _pointer(manifest)
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "current.json"
            current.write_bytes(b"existing-pointer")
            with patch(
                "robo_collector.mcap_contract.tempfile.mkstemp",
                side_effect=FileExistsError("injected collision"),
            ), self.assertRaises(FileExistsError):
                mcap_contract.publish_ready_pointer(
                    directory, manifest, pointer, prevalidated_authority=True
                )
            self.assertEqual(current.read_bytes(), b"existing-pointer")


if __name__ == "__main__":
    unittest.main()
