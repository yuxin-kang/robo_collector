import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mcap.opcode import Opcode
from mcap_phase1_fixtures import (
    corrupt_record_content,
    record_ranges,
    sealed_mcap_bytes,
    truncate_before_record,
)
from robo_collector.mcap_episode import (
    EpisodePublicationError,
    SealedEpisodeError,
    install_ready_bundle,
    install_ready_pointer,
    publish_raw_closed_manifest,
    validate_sealed_mcap,
)
from robo_collector.mcap_landing import LandingWriter

from robo_collector import mcap_contract


class SealedMcapValidationTest(unittest.TestCase):
    def _write(self, directory, payload, name="episode.mcap"):
        path = Path(directory, name)
        path.write_bytes(payload)
        return path

    def test_valid_sealed_inventory_and_expectations(self):
        payload = sealed_mcap_bytes()
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, payload)
            inventory = validate_sealed_mcap(
                path,
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                expected_size_bytes=len(payload),
                expected_total_message_count=2,
                expected_topic_counts={"/episode/event": 2},
                expected_start_log_time_ns=10,
                expected_end_log_time_ns=20,
            )

        self.assertEqual(inventory.sha256, hashlib.sha256(payload).hexdigest())
        self.assertEqual(inventory.size_bytes, len(payload))
        self.assertEqual(len(inventory.schemas), 1)
        self.assertEqual(len(inventory.channels), 1)
        self.assertEqual(inventory.topic_counts, (("/episode/event", 2),))
        self.assertEqual(inventory.message_count, 2)
        self.assertEqual(inventory.start_log_time_ns, 10)
        self.assertEqual(inventory.end_log_time_ns, 20)

    def test_validation_is_read_only_and_source_expectations_fail_closed(self):
        payload = sealed_mcap_bytes()
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, payload)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            partial = path.with_suffix(".partial")
            partial.write_bytes(payload)
            with self.assertRaises(SealedEpisodeError):
                validate_sealed_mcap(partial)
            for kwargs in (
                {"expected_sha256": "0" * 64},
                {"expected_size_bytes": len(payload) + 1},
                {"expected_total_message_count": 3},
                {"expected_topic_counts": {"/episode/event": 1}},
                {"expected_start_log_time_ns": 11},
                {"expected_end_log_time_ns": 21},
            ):
                with self.subTest(kwargs=kwargs), self.assertRaises(SealedEpisodeError):
                    validate_sealed_mcap(path, **kwargs)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
            self.assertEqual(path.read_bytes(), payload)

    def test_missing_footer_and_corrupt_data_crc_are_rejected(self):
        payload = sealed_mcap_bytes()
        cases = {
            "missing_footer": truncate_before_record(payload, Opcode.FOOTER),
            "corrupt_data_crc": corrupt_record_content(payload, Opcode.DATA_END),
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, damaged in cases.items():
                with self.subTest(name=name):
                    path = self._write(directory, damaged, f"{name}.mcap")
                    with self.assertRaises(SealedEpisodeError):
                        validate_sealed_mcap(path)

    def test_summary_statistics_and_record_size_limit_are_required(self):
        cases = {
            "missing_statistics": sealed_mcap_bytes(use_statistics=False),
            "record_too_large": sealed_mcap_bytes(),
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, payload in cases.items():
                with self.subTest(name=name):
                    path = self._write(directory, payload, f"{name}.mcap")
                    kwargs = {"record_size_limit": 4} if name == "record_too_large" else {}
                    with self.assertRaises(SealedEpisodeError):
                        validate_sealed_mcap(path, **kwargs)

    def test_zero_summary_crc_and_wrong_frozen_library_are_rejected_read_only(self):
        payload = sealed_mcap_bytes()
        zero_crc = bytearray(payload)
        footer = next(
            record for record in record_ranges(payload) if record.opcode == int(Opcode.FOOTER)
        )
        zero_crc[footer.content_start + 16 : footer.content_start + 20] = b"\0" * 4
        cases = {
            "zero_summary_crc": bytes(zero_crc),
            "wrong_library": sealed_mcap_bytes(library="mcap-python/1.3.0"),
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, damaged in cases.items():
                with self.subTest(name=name):
                    path = self._write(directory, damaged, f"{name}.mcap")
                    before = path.read_bytes()
                    with self.assertRaises(SealedEpisodeError):
                        validate_sealed_mcap(path)
                    self.assertEqual(path.read_bytes(), before)

    def test_sealed_file_symlink_is_rejected_without_following_or_mutation(self):
        payload = sealed_mcap_bytes()
        with (
            tempfile.TemporaryDirectory() as directory,
            tempfile.TemporaryDirectory() as external_directory,
        ):
            external = Path(external_directory, "external.mcap")
            external.write_bytes(payload)
            linked = Path(directory, "episode.mcap")
            linked.symlink_to(external)

            with self.assertRaises(SealedEpisodeError):
                validate_sealed_mcap(linked)

            self.assertTrue(linked.is_symlink())
            self.assertEqual(external.read_bytes(), payload)

    def test_source_fence_inventory_mismatches_are_rejected(self):
        payload = sealed_mcap_bytes()
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, payload)
            with self.assertRaises(SealedEpisodeError):
                validate_sealed_mcap(
                    path,
                    expected_source_fences=[
                        {
                            "source_id": "camera",
                            "session_id": "session",
                            "start_sequence_exclusive": "0",
                            "end_sequence_inclusive": "1",
                            "accepted_count": "1",
                            "written_count": "1",
                            "durable_count": "1",
                            "accepted_high_watermark": "1",
                            "written_high_watermark": "1",
                            "durable_high_watermark": "1",
                        }
                    ],
                )


class EpisodePublicationTest(unittest.TestCase):
    @staticmethod
    def _empty_checkpoint():
        return {
            "accepted_snapshot_count": "0",
            "accepted_snapshot_frontier": None,
            "channels": [],
            "checkpoint_sequence": "0",
            "durable_byte_offset": "0",
            "durable_count": "0",
            "durable_frontier": None,
            "format": "robo_collector.mcap_checkpoint",
            "format_version": 1,
            "generation": "0",
            "landing_prefix_sha256": hashlib.sha256(b"").hexdigest(),
            "max_unsynced_records": "1",
            "queue_capacity": "64",
            "source_fences": [],
            "written_count": "0",
            "written_frontier": None,
        }

    @staticmethod
    def _source_fence(*, session_id="session-a"):
        return {
            "source_id": "camera",
            "session_id": session_id,
            "start_sequence_exclusive": "0",
            "end_sequence_inclusive": "1",
            "accepted_count": "1",
            "written_count": "1",
            "durable_count": "1",
            "written_high_watermark": "1",
            "durable_high_watermark": "1",
        }

    @classmethod
    def _checkpoint_with_fence(cls, *, session_id="session-a"):
        checkpoint = cls._empty_checkpoint()
        checkpoint_fence = cls._source_fence(session_id=session_id)
        checkpoint.update(
            {
                "accepted_snapshot_count": "2",
                "accepted_snapshot_frontier": "1",
                "durable_count": "2",
                "durable_frontier": "1",
                "source_fences": [checkpoint_fence],
                "written_count": "2",
                "written_frontier": "1",
            }
        )
        return checkpoint

    @staticmethod
    def _ready_fixture(directory):
        from test_mcap_contract import _ready_manifest

        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        manifest = _ready_manifest()
        members = manifest["identity"]["members"]
        inventory = {item["path"]: item for item in manifest["inventory"]["files"]}
        for member in members:
            payload = f"fixture:{member['path']}".encode()
            digest = hashlib.sha256(payload).hexdigest()
            member.update(sha256=digest, size_bytes=str(len(payload)))
            inventory[member["path"]].update(
                sha256=digest, size_bytes=str(len(payload))
            )
            path = root / member["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        manifest["inventory"]["checksums_sha256"] = mcap_contract.canonical_json_hash(
            {
                "algorithm": "sha256",
                "format": "robo_collector.checksums",
                "format_version": 1,
                "members": members,
            }
        )
        manifest["bundle_hash"] = mcap_contract.canonical_json_hash(
            manifest["identity"]
        )
        (root / "manifest.json").write_bytes(
            mcap_contract.canonical_json_bytes(manifest)
        )
        return manifest

    def test_raw_closed_publication_is_atomic_and_immutable(self):
        payload = sealed_mcap_bytes()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sealed = root / "landing" / "episode.mcap"
            sealed.parent.mkdir()
            sealed.write_bytes(payload)
            published = publish_raw_closed_manifest(
                root,
                sealed,
                self._empty_checkpoint(),
                source_complete=True,
            )
            before = published.read_bytes()

            with self.assertRaises(EpisodePublicationError):
                publish_raw_closed_manifest(
                    root,
                    sealed,
                    self._empty_checkpoint(),
                    source_complete=True,
                )

            self.assertEqual(published.read_bytes(), before)
            self.assertEqual(sealed.read_bytes(), payload)

    def test_invalid_seal_and_missing_ready_authority_publish_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            broken = root / "broken.mcap"
            broken.write_bytes(b"not-mcap")
            with self.assertRaises(SealedEpisodeError):
                publish_raw_closed_manifest(
                    root,
                    broken,
                    self._empty_checkpoint(),
                    source_complete=True,
                )
            self.assertFalse((root / "manifest.json").exists())

            canonical = root / "canonical"
            with self.assertRaises(EpisodePublicationError):
                install_ready_pointer(
                    canonical,
                    {},
                    {},
                    prevalidated_authority=False,
                )
            self.assertFalse(canonical.exists())

    def test_raw_closed_publication_binds_exact_latest_source_fences(self):
        sealed_fence = self._source_fence()
        payload = sealed_mcap_bytes(source_fences=[sealed_fence])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sealed = root / "landing" / "episode.mcap"
            sealed.parent.mkdir()
            sealed.write_bytes(payload)
            manifest = publish_raw_closed_manifest(
                root,
                sealed,
                self._checkpoint_with_fence(),
                source_complete=True,
            )
            self.assertTrue(manifest.is_file())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sealed = root / "landing" / "episode.mcap"
            sealed.parent.mkdir()
            sealed.write_bytes(payload)
            pointer = root / "canonical" / "current.json"
            pointer.parent.mkdir()
            pointer.write_bytes(b"existing-pointer")
            pointer_before = pointer.read_bytes()

            with self.assertRaises(SealedEpisodeError):
                publish_raw_closed_manifest(
                    root,
                    sealed,
                    self._checkpoint_with_fence(session_id="session-b"),
                    source_complete=True,
                )

            self.assertFalse((root / "manifest.json").exists())
            self.assertEqual(pointer.read_bytes(), pointer_before)

    def test_raw_closed_rejects_internally_inconsistent_fences_without_mutation(self):
        cases = {
            "checkpoint_end_disagrees_with_written_and_durable": (
                {
                    **self._source_fence(),
                    "written_high_watermark": "2",
                    "durable_high_watermark": "2",
                },
                {
                    **self._source_fence(),
                    "written_high_watermark": "2",
                    "durable_high_watermark": "2",
                },
            ),
            "sealed_explicit_accepted_watermark_disagrees_with_end": (
                self._source_fence(),
                {**self._source_fence(), "accepted_high_watermark": "2"},
            ),
        }
        for name, (checkpoint_fence, sealed_fence) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                sealed = root / "landing" / "episode.mcap"
                sealed.parent.mkdir()
                sealed.write_bytes(sealed_mcap_bytes(source_fences=[sealed_fence]))
                sealed_before = hashlib.sha256(sealed.read_bytes()).hexdigest()
                pointer = root / "canonical" / "current.json"
                pointer.parent.mkdir()
                pointer.write_bytes(b"existing-pointer")
                pointer_before = pointer.read_bytes()
                checkpoint = self._checkpoint_with_fence()
                checkpoint["source_fences"] = [checkpoint_fence]

                with self.assertRaises(SealedEpisodeError):
                    publish_raw_closed_manifest(
                        root,
                        sealed,
                        checkpoint,
                        source_complete=True,
                    )

                self.assertFalse((root / "manifest.json").exists())
                self.assertEqual(pointer.read_bytes(), pointer_before)
                self.assertEqual(
                    hashlib.sha256(sealed.read_bytes()).hexdigest(), sealed_before
                )

    def test_raw_closed_allows_loss_aware_source_sequence_gaps(self):
        gap_fence = {
            "source_id": "camera",
            "session_id": "session-a",
            "start_sequence_exclusive": "5",
            "end_sequence_inclusive": "8",
            "accepted_count": "2",
            "written_count": "2",
            "durable_count": "2",
            "accepted_high_watermark": "8",
            "written_high_watermark": "8",
            "durable_high_watermark": "8",
        }
        checkpoint_fence = dict(gap_fence)
        checkpoint_fence.pop("accepted_high_watermark")
        checkpoint = self._empty_checkpoint()
        checkpoint.update(
            {
                "accepted_snapshot_count": "2",
                "accepted_snapshot_frontier": "1",
                "durable_count": "2",
                "durable_frontier": "1",
                "source_fences": [checkpoint_fence],
                "written_count": "2",
                "written_frontier": "1",
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sealed = root / "landing" / "episode.mcap"
            sealed.parent.mkdir()
            sealed.write_bytes(sealed_mcap_bytes(source_fences=[gap_fence]))

            manifest = publish_raw_closed_manifest(
                root,
                sealed,
                checkpoint,
                source_complete=True,
            )

            self.assertTrue(manifest.is_file())

    def test_raw_closed_rejects_parent_symlink_escape_without_external_mutation(self):
        payload = sealed_mcap_bytes()
        with (
            tempfile.TemporaryDirectory() as directory,
            tempfile.TemporaryDirectory() as external_directory,
        ):
            root = Path(directory)
            external = Path(external_directory)
            sealed = external / "episode.mcap"
            sealed.write_bytes(payload)
            sentinel = external / "sentinel"
            sentinel.write_bytes(b"external-sentinel")
            (root / "landing").symlink_to(external, target_is_directory=True)
            pointer = root / "canonical" / "current.json"
            pointer.parent.mkdir()
            pointer.write_bytes(b"existing-pointer")

            with self.assertRaises(SealedEpisodeError):
                publish_raw_closed_manifest(
                    root,
                    root / "landing" / "episode.mcap",
                    self._empty_checkpoint(),
                    source_complete=True,
                )

            self.assertFalse((root / "manifest.json").exists())
            self.assertEqual(sealed.read_bytes(), payload)
            self.assertEqual(sentinel.read_bytes(), b"external-sentinel")
            self.assertEqual(pointer.read_bytes(), b"existing-pointer")

    def test_ready_version_symlink_escape_preserves_staging_and_external_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            canonical = root / "canonical"
            external = root / "external-version"
            manifest = self._ready_fixture(staging)
            external.mkdir()
            sentinel = external / "sentinel"
            sentinel.write_bytes(b"external-sentinel")
            versions = canonical / "versions"
            versions.mkdir(parents=True)
            (versions / manifest["bundle_hash"]).symlink_to(
                external, target_is_directory=True
            )
            pointer = canonical / "current.json"
            pointer.write_bytes(b"existing-pointer")
            staging_manifest_before = (staging / "manifest.json").read_bytes()

            with self.assertRaises(EpisodePublicationError):
                install_ready_bundle(
                    staging,
                    canonical,
                    manifest,
                    prevalidated_authority=True,
                )

            self.assertEqual(sentinel.read_bytes(), b"external-sentinel")
            self.assertEqual(pointer.read_bytes(), b"existing-pointer")
            self.assertEqual(
                (staging / "manifest.json").read_bytes(), staging_manifest_before
            )

    def test_ready_pointer_rejects_symlinked_canonical_root_without_external_write(self):
        from test_mcap_contract import _pointer

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            external = root / "external-canonical"
            manifest = self._ready_fixture(staging)
            external.mkdir()
            current = external / "current.json"
            current.write_bytes(b"external-current-sentinel")
            canonical = root / "canonical"
            canonical.symlink_to(external, target_is_directory=True)

            with self.assertRaises(EpisodePublicationError):
                install_ready_pointer(
                    canonical,
                    manifest,
                    _pointer(manifest),
                    prevalidated_authority=True,
                )

            self.assertTrue(canonical.is_symlink())
            self.assertEqual(current.read_bytes(), b"external-current-sentinel")
            self.assertEqual(
                (staging / "manifest.json").read_bytes(),
                mcap_contract.canonical_json_bytes(manifest),
            )

    def test_atomic_create_temp_collision_preserves_preexisting_sentinel(self):
        payload = sealed_mcap_bytes()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sealed = root / "landing" / "episode.mcap"
            sealed.parent.mkdir()
            sealed.write_bytes(payload)
            sentinel = root / ".manifest.json.collision.tmp"
            sentinel.write_bytes(b"preexisting-temp-sentinel")
            pointer = root / "canonical" / "current.json"
            pointer.parent.mkdir()
            pointer.write_bytes(b"existing-pointer")

            with (
                patch(
                    "robo_collector.mcap_episode.tempfile.mkstemp",
                    side_effect=FileExistsError("injected temp collision"),
                ),
                self.assertRaises(FileExistsError),
            ):
                publish_raw_closed_manifest(
                    root,
                    sealed,
                    self._empty_checkpoint(),
                    source_complete=True,
                )

            self.assertFalse((root / "manifest.json").exists())
            self.assertEqual(sentinel.read_bytes(), b"preexisting-temp-sentinel")
            self.assertEqual(sealed.read_bytes(), payload)
            self.assertEqual(pointer.read_bytes(), b"existing-pointer")

    def test_writer_seal_validates_and_publishes_raw_closed_end_to_end(self):
        from test_mcap_landing import _channel, _record

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = LandingWriter(
                root,
                episode_id="episode.end-to-end",
                created_time_ns=0,
                max_unsynced_records=8,
            )
            writer.register_channel(_channel())
            writer.start()
            writer.submit(_record(0))
            seal = writer.stop()
            writer.close()
            sealed_before = seal.sealed_path.read_bytes()

            inventory = validate_sealed_mcap(
                seal.sealed_path,
                expected_sha256=hashlib.sha256(sealed_before).hexdigest(),
                expected_size_bytes=len(sealed_before),
                expected_total_message_count=1,
                expected_topic_counts={"/episode/event": 1},
            )
            manifest = publish_raw_closed_manifest(
                root,
                seal.sealed_path,
                seal.last_checkpoint,
                source_complete=seal.source_complete,
            )

            self.assertEqual(inventory.total_message_count, 1)
            self.assertTrue(manifest.is_file())
            self.assertEqual(seal.sealed_path.read_bytes(), sealed_before)


if __name__ == "__main__":
    unittest.main()
