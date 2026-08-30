import hashlib
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from mcap.data_stream import RecordBuilder
from mcap.records import Channel, Header, Schema
from mcap_phase1_fixtures import MAGIC
from robo_collector.mcap.v1 import episode_pb2
from robo_collector.mcap_landing import (
    LandingChannel,
    LandingFaulted,
    LandingQueueFull,
    LandingRecord,
    LandingStateError,
    LandingWriter,
    RecoveryError,
    RecoveryExitCode,
    RequiredSource,
    SourceFence,
    read_checkpoint_journal,
    recover_landing,
    select_durable_prefix,
)

from robo_collector import mcap_contract


def _channel():
    return LandingChannel(
        topic="/episode/event",
        schema_name="EpisodeEventV1",
        schema_data=mcap_contract.descriptor_set_bytes(),
        metadata={
            "robo.robot_id": "fixture_robot",
            "robo.schema_version": "1",
            "robo.pipeline_version": "phase1-test",
        },
    )


def _record(
    record_id,
    *,
    log_time=None,
    source_id="",
    session_id="",
    source_sequence=None,
):
    timestamp = 10 + record_id if log_time is None else log_time
    event = episode_pb2.EpisodeEventV1(
        event_sequence=record_id,
        event_type=episode_pb2.EPISODE_EVENT_STOP,
        lifecycle_attempt=1,
        collector_record_id=record_id,
    )
    event.timestamps.normalized_time_ns = timestamp
    return LandingRecord(
        channel="/episode/event",
        data=event.SerializeToString(deterministic=True),
        log_time=timestamp,
        publish_time=timestamp,
        sequence=record_id,
        collector_record_id=record_id,
        source_id=source_id,
        session_id=session_id,
        source_sequence=source_sequence,
    )


def _source_record(record_id, source_sequence):
    return _record(
        record_id,
        source_id="camera",
        session_id="session-a",
        source_sequence=source_sequence,
    )


class LandingWriterLifecycleTest(unittest.TestCase):
    def test_stop_seals_equal_frontiers_and_forbids_append(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = LandingWriter(
                directory,
                episode_id="episode.phase1",
                created_time_ns=0,
                max_unsynced_records=8,
            )
            writer.register_channel(_channel())
            writer.start()
            writer.submit(_record(0))
            writer.submit(_record(1))
            seal = writer.stop()
            writer.close()

            checkpoint = seal.last_checkpoint
            self.assertEqual(writer.state, "SEALED")
            self.assertTrue(seal.sealed_path.is_file())
            self.assertFalse(writer.partial_path.exists())
            self.assertEqual(checkpoint["accepted_snapshot_count"], "2")
            self.assertEqual(checkpoint["written_count"], "2")
            self.assertEqual(checkpoint["durable_count"], "2")
            self.assertEqual(checkpoint["accepted_snapshot_frontier"], "1")
            self.assertEqual(checkpoint["written_frontier"], "1")
            self.assertEqual(checkpoint["durable_frontier"], "1")
            with self.assertRaises(LandingStateError):
                writer.submit(_record(2))

            replacement = LandingWriter(directory, episode_id="episode.phase1")
            replacement.register_channel(_channel())
            with self.assertRaises(LandingStateError):
                replacement.start()

    def test_fsync_failure_faults_without_sealing(self):
        def fail_landing_fsync(descriptor):
            target = os.readlink(f"/proc/self/fd/{descriptor}")
            if target.endswith("episode.mcap.partial"):
                raise OSError("injected landing fsync failure")
            os.fsync(descriptor)

        with tempfile.TemporaryDirectory() as directory:
            writer = LandingWriter(
                directory,
                episode_id="episode.fsync",
                fsync=fail_landing_fsync,
            )
            writer.register_channel(_channel())
            with self.assertRaises(LandingFaulted):
                writer.start()
            self.assertEqual(writer.state, "FAULTED")
            self.assertFalse(writer.sealed_path.exists())
            manifest = json.loads(writer.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["state"], "FAULTED")

    def test_checkpoint_fsync_order_is_landing_then_journal(self):
        calls = []

        def trace_fsync(descriptor):
            calls.append(os.readlink(f"/proc/self/fd/{descriptor}"))
            os.fsync(descriptor)

        with tempfile.TemporaryDirectory() as directory:
            writer = LandingWriter(
                directory,
                episode_id="episode.fsync-order",
                created_time_ns=0,
                max_unsynced_records=8,
                fsync=trace_fsync,
            )
            writer.register_channel(_channel())
            writer.start()
            writer.submit(_record(0))
            writer.stop()
            writer.close()

        landing_calls = [
            Path(item).name
            for item in calls
            if item.endswith(("episode.mcap.partial", "checkpoints.bin"))
        ]
        journal_indices = [
            index for index, item in enumerate(landing_calls) if item == "checkpoints.bin"
        ]
        self.assertGreaterEqual(len(journal_indices), 2)
        for index in journal_indices:
            self.assertGreater(index, 0)
            self.assertEqual(landing_calls[index - 1], "episode.mcap.partial")

    def test_recovery_exit_codes_are_frozen(self):
        self.assertEqual(
            {item.name: int(item) for item in RecoveryExitCode},
            {
                "RECOVERED": 0,
                "NO_DURABLE_PREFIX": 2,
                "CORRUPT_PREFIX_OR_JOURNAL": 3,
                "SOURCE_INCOMPLETE": 4,
                "IO_OR_INTERNAL_ERROR": 5,
            },
        )

    def test_sealed_target_appearing_after_start_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = LandingWriter(directory, episode_id="episode.raced-seal")
            writer.register_channel(_channel())
            writer.start()
            writer.submit(_record(0))
            raced_bytes = b"concurrent sealed winner"
            writer.sealed_path.write_bytes(raced_bytes)

            with self.assertRaises(LandingFaulted):
                writer.stop()
            writer.close()

            self.assertEqual(writer.sealed_path.read_bytes(), raced_bytes)
            self.assertTrue(writer.partial_path.is_file())
            self.assertNotEqual(writer.partial_path.read_bytes(), raced_bytes)

    def test_concurrent_initial_manifest_owner_wins_exclusively(self):
        paused = threading.Event()
        release = threading.Event()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()

            def pause_before_initial_manifest(descriptor):
                target = Path(os.readlink(f"/proc/self/fd/{descriptor}")).resolve()
                os.fsync(descriptor)
                if target == root and not paused.is_set():
                    paused.set()
                    self.assertTrue(release.wait(1))

            loser = LandingWriter(
                directory,
                episode_id="episode.owner-loser",
                fsync=pause_before_initial_manifest,
            )
            loser.register_channel(_channel())
            loser_error = []
            loser_thread = threading.Thread(
                target=lambda: self._capture_start_error(loser, loser_error)
            )
            loser_thread.start()
            self.assertTrue(paused.wait(1))

            winner = LandingWriter(directory, episode_id="episode.owner-winner")
            winner.register_channel(_channel())
            winner.start()
            manifest_before = winner.manifest_path.read_bytes()
            release.set()
            loser_thread.join(1)

            self.assertFalse(loser_thread.is_alive())
            self.assertEqual(len(loser_error), 1)
            self.assertIsInstance(loser_error[0], FileExistsError)
            self.assertEqual(loser.state, "FAULTED")
            self.assertEqual(winner.manifest_path.read_bytes(), manifest_before)
            winner.fault("test cleanup")
            winner.close()

    @staticmethod
    def _capture_start_error(writer, errors):
        try:
            writer.start()
        except BaseException as error:  # noqa: BLE001 - test captures thread result
            errors.append(error)

    def test_required_source_fences_are_strict_and_exact(self):
        required = (RequiredSource("camera", "session-a", 5),)
        with tempfile.TemporaryDirectory() as directory:
            writer = LandingWriter(
                directory,
                episode_id="episode.source-fence",
                required_sources=required,
            )
            writer.register_channel(_channel())
            writer.start()
            with self.assertRaisesRegex(ValueError, "follow the START fence"):
                writer.submit(_source_record(0, 5))
            writer.submit(_source_record(0, 6))
            with self.assertRaisesRegex(ValueError, "increase monotonically"):
                writer.submit(_source_record(1, 6))

            # Gaps are valid for a loss-aware source; only monotonicity and the
            # exact final watermark are frozen lifecycle requirements.
            writer.submit(_source_record(1, 8))
            checkpoint = writer.checkpoint()
            self.assertIsNone(
                checkpoint["source_fences"][0]["end_sequence_inclusive"]
            )
            with self.assertRaisesRegex(ValueError, "accepted watermark"):
                writer.stop(
                    source_fences=(SourceFence("camera", "session-a", 5, 7),)
                )
            self.assertFalse(writer.sealed_path.exists())

            seal = writer.stop(
                source_fences=(SourceFence("camera", "session-a", 5, 8),)
            )
            writer.close()
            self.assertTrue(seal.source_complete)
            fence = seal.last_checkpoint["source_fences"][0]
            self.assertEqual(fence["end_sequence_inclusive"], "8")
            self.assertEqual(
                fence["accepted_count"],
                fence["written_count"],
            )
            self.assertEqual(fence["written_count"], fence["durable_count"])

    def test_missing_stop_fence_never_seals_source_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = LandingWriter(
                directory,
                episode_id="episode.incomplete-source",
                required_sources=(RequiredSource("camera", "session-a", 5),),
            )
            writer.register_channel(_channel())
            writer.start()
            writer.submit(_source_record(0, 6))
            with self.assertRaises(LandingFaulted):
                writer.stop()
            writer.close()
            self.assertFalse(writer.sealed_path.exists())


class CheckpointRecoveryTest(unittest.TestCase):
    def _faulted_source(self, directory):
        writer = LandingWriter(
            directory,
            episode_id="episode.recovery",
            created_time_ns=0,
            max_unsynced_records=8,
        )
        writer.register_channel(_channel())
        writer.start()
        writer.submit(_record(0))
        writer.checkpoint()
        writer.fault("simulated crash")
        writer.close()
        return writer

    def test_torn_final_frame_is_tolerated_but_crc_corruption_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self._faulted_source(directory)
            journal = writer.checkpoint_path
            original = journal.read_bytes()
            journal.write_bytes(original + b"\x20\x00")
            scan = read_checkpoint_journal(journal)
            self.assertTrue(scan.torn_final)
            self.assertGreaterEqual(len(scan.checkpoints), 1)

            corrupt = bytearray(original)
            corrupt[-1] ^= 1
            journal.write_bytes(corrupt)
            with self.assertRaises(mcap_contract.CheckpointFrameError):
                read_checkpoint_journal(journal)

            first_payload_size = int.from_bytes(original[:8], "little")
            first_frame_size = 8 + first_payload_size + 4
            non_tail_corrupt = bytearray(original)
            non_tail_corrupt[first_frame_size - 1] ^= 1
            journal.write_bytes(non_tail_corrupt)
            with self.assertRaises(mcap_contract.CheckpointFrameError):
                read_checkpoint_journal(journal)

    def test_prefix_digest_and_ahead_offsets_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self._faulted_source(directory)
            source = writer.partial_path
            journal = writer.checkpoint_path
            frames = read_checkpoint_journal(journal).checkpoints
            source_before = source.read_bytes()
            journal_before = journal.read_bytes()

            prefix = select_durable_prefix(source, journal)
            self.assertEqual(
                hashlib.sha256(source_before[: prefix.byte_offset]).hexdigest(),
                prefix.sha256,
            )

            first_frame_size = 8 + int.from_bytes(journal_before[:8], "little") + 4
            journal.write_bytes(journal_before[:first_frame_size])
            stale = select_durable_prefix(source, journal)
            self.assertLess(stale.byte_offset, len(source_before))
            journal.write_bytes(journal_before)

            damaged = bytearray(source_before)
            damaged[0] ^= 1
            source.write_bytes(damaged)
            with self.assertRaises(RecoveryError):
                select_durable_prefix(source, journal)
            source.write_bytes(source_before)

            ahead = dict(frames[-1])
            ahead["durable_byte_offset"] = str(len(source_before) + 1)
            journal.write_bytes(mcap_contract.encode_checkpoint_frame(ahead))
            with self.assertRaises(RecoveryError):
                select_durable_prefix(source, journal)

            self.assertEqual(source.read_bytes(), source_before)
            self.assertNotEqual(journal.read_bytes(), journal_before)

    def test_recovery_writes_new_artifact_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            camera_dir = Path(directory, "camera")
            camera_dir.mkdir()
            raw_spool = camera_dir / "raw-spool.bin"
            raw_manifest = camera_dir / "manifest.json"
            raw_spool.write_bytes(b"camera-crc-spool-evidence")
            raw_manifest.write_text('{"source_scope":"camera_capture"}', encoding="utf-8")
            writer = self._faulted_source(directory)
            source_before = writer.partial_path.read_bytes()
            journal_before = writer.checkpoint_path.read_bytes()
            manifest_before = writer.manifest_path.read_bytes()
            spool_before = raw_spool.read_bytes()
            raw_manifest_before = raw_manifest.read_bytes()

            result = recover_landing(directory, attempt=2)

            self.assertIn(
                result.exit_code,
                {RecoveryExitCode.RECOVERED, RecoveryExitCode.SOURCE_INCOMPLETE},
            )
            self.assertIsNotNone(result.recovered_path)
            self.assertTrue(result.recovered_path.is_file())
            self.assertEqual(writer.partial_path.read_bytes(), source_before)
            self.assertEqual(writer.checkpoint_path.read_bytes(), journal_before)
            self.assertEqual(writer.manifest_path.read_bytes(), manifest_before)
            self.assertEqual(raw_spool.read_bytes(), spool_before)
            self.assertEqual(raw_manifest.read_bytes(), raw_manifest_before)

    def test_missing_inputs_return_no_durable_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            result = recover_landing(directory, attempt=1)
            self.assertEqual(result.exit_code, RecoveryExitCode.NO_DURABLE_PREFIX)
            self.assertFalse(result.ok)

    def test_post_link_directory_fsync_failure_keeps_recoverable_evidence(self):
        failed = False

        def fail_first_post_link_fsync(descriptor):
            nonlocal failed
            target = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
            if (
                target.name == "landing"
                and Path(target, "episode.mcap").exists()
                and Path(target, "episode.mcap.partial").exists()
                and not failed
            ):
                failed = True
                raise OSError("injected post-link directory fsync failure")
            os.fsync(descriptor)

        with tempfile.TemporaryDirectory() as directory:
            writer = LandingWriter(
                directory,
                episode_id="episode.post-link-fsync",
                fsync=fail_first_post_link_fsync,
            )
            writer.register_channel(_channel())
            writer.start()
            writer.submit(_record(0))
            with self.assertRaises(LandingFaulted):
                writer.stop()
            writer.close()

            self.assertTrue(failed)
            self.assertTrue(writer.sealed_path.is_file())
            self.assertTrue(writer.partial_path.is_file())
            self.assertEqual(
                writer.sealed_path.read_bytes(), writer.partial_path.read_bytes()
            )
            recovery = recover_landing(directory, attempt=2)
            self.assertIn(
                recovery.exit_code,
                {RecoveryExitCode.RECOVERED, RecoveryExitCode.SOURCE_INCOMPLETE},
            )

    def test_recovery_post_link_fsync_failure_is_adoptable_and_idempotent(self):
        failed = False

        def fail_recovery_post_link_fsync(descriptor):
            nonlocal failed
            target = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
            if (
                target.name == "landing"
                and Path(target, "recovery-2.mcap").exists()
                and Path(target, "recovery-2.mcap.partial").exists()
                and not failed
            ):
                failed = True
                raise OSError("injected recovery post-link fsync failure")
            os.fsync(descriptor)

        with tempfile.TemporaryDirectory() as directory:
            writer = self._faulted_source(directory)
            immutable_sources = {
                path: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (
                    writer.partial_path,
                    writer.checkpoint_path,
                    writer.manifest_path,
                )
            }

            first = recover_landing(
                directory,
                attempt=2,
                fsync=fail_recovery_post_link_fsync,
            )
            target = Path(directory, "landing", "recovery-2.mcap")
            self.assertTrue(failed)
            self.assertTrue(target.is_file())
            target_before = hashlib.sha256(target.read_bytes()).hexdigest()

            retry = recover_landing(directory, attempt=2)
            adopted = first if first.ok else retry
            self.assertTrue(adopted.ok)
            self.assertEqual(adopted.recovered_path, target)
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), target_before)
            for path, digest in immutable_sources.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_unequal_global_frontiers_are_never_source_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self._faulted_source(directory)
            source_hash = hashlib.sha256(writer.partial_path.read_bytes()).hexdigest()
            journal_hash = hashlib.sha256(writer.checkpoint_path.read_bytes()).hexdigest()
            checkpoint = dict(
                read_checkpoint_journal(writer.checkpoint_path).checkpoints[-1]
            )
            self.assertEqual(
                checkpoint["accepted_snapshot_count"],
                checkpoint["written_count"],
            )
            self.assertEqual(checkpoint["written_count"], checkpoint["durable_count"])
            checkpoint["accepted_snapshot_frontier"] = "1"
            checkpoint["written_frontier"] = "0"
            checkpoint["durable_frontier"] = "0"
            writer.checkpoint_path.write_bytes(
                mcap_contract.encode_checkpoint_frame(checkpoint)
            )
            mismatched_journal_hash = hashlib.sha256(
                writer.checkpoint_path.read_bytes()
            ).hexdigest()

            result = recover_landing(directory, attempt=2)

            self.assertEqual(result.exit_code, RecoveryExitCode.SOURCE_INCOMPLETE)
            self.assertFalse(result.source_complete)
            self.assertFalse(result.ok)
            self.assertEqual(
                hashlib.sha256(writer.partial_path.read_bytes()).hexdigest(), source_hash
            )
            self.assertNotEqual(mismatched_journal_hash, journal_hash)
            self.assertEqual(
                hashlib.sha256(writer.checkpoint_path.read_bytes()).hexdigest(),
                mismatched_journal_hash,
            )

    def test_replay_rejects_wrong_profile_and_divergent_duplicate_ids(self):
        builder = RecordBuilder()
        Header("wrong.profile", "mcap-python/1.4.0").write(builder)
        wrong_profile = MAGIC + builder.end()

        builder = RecordBuilder()
        Header(mcap_contract.MCAP_PROFILE, "mcap-python/1.4.0").write(builder)
        Schema(1, b"a", "protobuf", "SchemaA").write(builder)
        Schema(1, b"b", "protobuf", "SchemaB").write(builder)
        divergent_schema = MAGIC + builder.end()

        builder = RecordBuilder()
        Header(mcap_contract.MCAP_PROFILE, "mcap-python/1.4.0").write(builder)
        Schema(1, b"a", "protobuf", "SchemaA").write(builder)
        Channel(1, "/a", "protobuf", {}, 1).write(builder)
        Channel(1, "/b", "protobuf", {}, 1).write(builder)
        divergent_channel = MAGIC + builder.end()

        for name, payload, expected in (
            ("wrong-profile", wrong_profile, "unexpected MCAP profile"),
            ("divergent-schema", divergent_schema, "divergent duplicate schema id"),
            (
                "divergent-channel",
                divergent_channel,
                "divergent duplicate channel id",
            ),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                self._faulted_source(directory)
                root = Path(directory)
                partial = root / "landing" / "episode.mcap.partial"
                journal = root / "landing" / "checkpoints.bin"
                checkpoint = dict(read_checkpoint_journal(journal).checkpoints[-1])
                partial.write_bytes(payload)
                checkpoint["durable_byte_offset"] = str(len(payload))
                checkpoint["landing_prefix_sha256"] = hashlib.sha256(
                    payload
                ).hexdigest()
                journal.write_bytes(mcap_contract.encode_checkpoint_frame(checkpoint))

                result = recover_landing(directory, attempt=2)
                self.assertEqual(
                    result.exit_code, RecoveryExitCode.IO_OR_INTERNAL_ERROR
                )
                self.assertIn(expected, result.detail)
                self.assertEqual(partial.read_bytes(), payload)


class _BlockingWriter:
    entered = threading.Event()
    release = threading.Event()

    def __init__(self, output, **kwargs):
        self.output = output

    def start(self, **kwargs):
        return None

    def register_schema(self, *args):
        return 1

    def register_channel(self, *args):
        return 1

    def add_message(self, *args, **kwargs):
        self.entered.set()
        self.release.wait(2)

    def finish(self):
        return None


class BoundedAdmissionTest(unittest.TestCase):
    def test_full_queue_rejects_record_and_faults_writer(self):
        _BlockingWriter.entered.clear()
        _BlockingWriter.release.clear()
        with tempfile.TemporaryDirectory() as directory:
            writer = LandingWriter(
                directory,
                episode_id="episode.queue",
                queue_capacity=1,
                max_unsynced_records=8,
                writer_factory=_BlockingWriter,
            )
            writer.register_channel(_channel())
            writer.start()
            writer.submit(_record(0))
            self.assertTrue(_BlockingWriter.entered.wait(1))
            writer.submit(_record(1))
            try:
                with self.assertRaises(LandingQueueFull):
                    writer.submit(_record(2))
                self.assertEqual(writer.state, "FAULTED")
            finally:
                _BlockingWriter.release.set()
                writer.close()

    def test_close_is_bounded_when_writer_thread_is_blocked(self):
        _BlockingWriter.entered.clear()
        _BlockingWriter.release.clear()
        with tempfile.TemporaryDirectory() as directory:
            writer = LandingWriter(
                directory,
                episode_id="episode.bounded-close",
                writer_factory=_BlockingWriter,
                shutdown_timeout_sec=0.05,
            )
            writer.register_channel(_channel())
            writer.start()
            writer.submit(_record(0))
            self.assertTrue(_BlockingWriter.entered.wait(1))
            started = time.monotonic()
            try:
                with self.assertRaisesRegex(LandingFaulted, "shutdown timeout"):
                    writer.close()
                self.assertLess(time.monotonic() - started, 0.5)
            finally:
                _BlockingWriter.release.set()
                writer.close()

    def test_checkpoint_stop_barrier_leaves_no_orphaned_caller(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = LandingWriter(directory, episode_id="episode.barrier")
            writer.register_channel(_channel())
            writer.start()
            writer.submit(_record(0))
            errors = []

            def checkpoint():
                try:
                    writer.checkpoint()
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            caller = threading.Thread(target=checkpoint)
            caller.start()
            seal = writer.stop()
            caller.join(1.0)
            self.assertFalse(caller.is_alive())
            self.assertEqual(writer.state, "SEALED")
            self.assertTrue(seal.sealed_path.is_file())
            writer.close()

    def test_fault_and_close_are_bounded_when_writer_fsync_is_blocked(self):
        entered = threading.Event()
        release = threading.Event()
        partial_fsyncs = 0

        def block_record_fsync(descriptor):
            nonlocal partial_fsyncs
            target = os.readlink(f"/proc/self/fd/{descriptor}")
            if target.endswith("episode.mcap.partial"):
                partial_fsyncs += 1
                if partial_fsyncs >= 2:
                    entered.set()
                    release.wait(2)
                    return
            os.fsync(descriptor)

        with tempfile.TemporaryDirectory() as directory:
            writer = LandingWriter(
                directory,
                episode_id="episode.bounded-fsync",
                max_unsynced_records=1,
                fsync=block_record_fsync,
                shutdown_timeout_sec=0.05,
            )
            writer.register_channel(_channel())
            writer.start()
            writer.submit(_record(0))
            self.assertTrue(entered.wait(1))

            started = time.monotonic()
            writer.fault("test blocked fsync")
            self.assertLess(time.monotonic() - started, 0.5)
            try:
                with self.assertRaisesRegex(LandingFaulted, "shutdown timeout"):
                    writer.close()
            finally:
                release.set()
                writer.close()


if __name__ == "__main__":
    unittest.main()
