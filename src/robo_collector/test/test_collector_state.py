import unittest

from robo_collector.collector_state import (
    CollectorMode,
    CommandFingerprint,
    CommandReceiptLedger,
    RecordCommandType,
    RecordStateMachine,
    recording_safety_reason,
)


class CommandReceiptLedgerTest(unittest.TestCase):
    def setUp(self):
        self.fingerprint = CommandFingerprint(
            command=int(RecordCommandType.START),
            task_prompt="pick up",
            episode_id="episode-a",
            force=False,
        )

    def test_same_command_id_and_payload_replays_terminal_outcome(self):
        ledger = CommandReceiptLedger()
        ledger.remember("command-a", self.fingerprint, "SUCCEEDED")

        replay = ledger.lookup("command-a", self.fingerprint)

        self.assertEqual(replay.disposition, "REPLAY")
        self.assertEqual(replay.outcome, "SUCCEEDED")

    def test_same_command_id_with_different_payload_is_a_conflict(self):
        ledger = CommandReceiptLedger()
        ledger.remember("command-a", self.fingerprint, "SUCCEEDED")
        different = CommandFingerprint(
            command=int(RecordCommandType.STOP),
            task_prompt="pick up",
            episode_id="episode-a",
            force=False,
        )

        replay = ledger.lookup("command-a", different)

        self.assertEqual(replay.disposition, "CONFLICT")

    def test_retriable_failure_executes_again_then_replays_success(self):
        ledger = CommandReceiptLedger()
        ledger.remember(
            "command-a",
            self.fingerprint,
            "FAILED",
            replayable=False,
        )
        self.assertIsNone(ledger.lookup("command-a", self.fingerprint))

        ledger.remember("command-a", self.fingerprint, "SUCCEEDED")

        self.assertEqual(
            ledger.lookup("command-a", self.fingerprint).outcome,
            "SUCCEEDED",
        )


class RecordStateMachineTest(unittest.TestCase):
    def test_idle_start_stop_save_idle_cycle(self):
        machine = RecordStateMachine()

        start = machine.handle_command(
            RecordCommandType.START,
            task_prompt="pick up the cup",
            episode_id="ep-a",
            now_sec=1.0,
        )
        self.assertTrue(start.accepted)
        self.assertTrue(start.should_start)
        self.assertEqual(machine.mode, CollectorMode.RECORDING)
        self.assertEqual(machine.session.task_prompt, "pick up the cup")

        stop = machine.handle_command(RecordCommandType.STOP)
        self.assertTrue(stop.accepted)
        self.assertTrue(stop.should_save)
        self.assertEqual(machine.mode, CollectorMode.NEED_TO_SAVE)

        machine.mark_saved()
        self.assertEqual(machine.mode, CollectorMode.IDLE)
        self.assertIsNone(machine.session)

    def test_start_requires_task_prompt(self):
        machine = RecordStateMachine()

        result = machine.handle_command(RecordCommandType.START, task_prompt=" ")

        self.assertFalse(result.accepted)
        self.assertEqual(result.level, "WARN")
        self.assertEqual(machine.mode, CollectorMode.IDLE)

    def test_stop_without_start_does_not_request_save(self):
        machine = RecordStateMachine()

        result = machine.handle_command(RecordCommandType.STOP)

        self.assertFalse(result.accepted)
        self.assertFalse(result.should_save)
        self.assertEqual(machine.mode, CollectorMode.IDLE)

    def test_discard_from_recording_returns_to_idle_after_mark(self):
        machine = RecordStateMachine()
        machine.handle_command(RecordCommandType.START, task_prompt="walk forward")

        result = machine.handle_command(RecordCommandType.DISCARD)

        self.assertTrue(result.accepted)
        self.assertTrue(result.should_discard)
        self.assertEqual(machine.mode, CollectorMode.DISCARD)

        machine.mark_discarded()
        self.assertEqual(machine.mode, CollectorMode.IDLE)

    def test_failed_discard_can_be_retried(self):
        machine = RecordStateMachine()
        machine.handle_command(
            RecordCommandType.START,
            task_prompt="walk forward",
            episode_id="episode-a",
        )
        first = machine.handle_command(
            RecordCommandType.DISCARD,
            episode_id="episode-a",
        )
        self.assertTrue(first.should_discard)

        machine.mark_discard_failed("temporary unlink failure")
        self.assertEqual(machine.mode, CollectorMode.FAILED)

        retry = machine.handle_command(
            RecordCommandType.DISCARD,
            episode_id="episode-a",
        )
        self.assertTrue(retry.should_discard)
        machine.mark_discarded()
        self.assertEqual(machine.mode, CollectorMode.IDLE)

    def test_start_ignored_while_recording(self):
        machine = RecordStateMachine()
        machine.handle_command(RecordCommandType.START, task_prompt="first")

        result = machine.handle_command(RecordCommandType.START, task_prompt="second")

        self.assertFalse(result.accepted)
        self.assertEqual(machine.session.task_prompt, "first")

    def test_failed_state_requires_discard(self):
        machine = RecordStateMachine()
        machine.handle_command(RecordCommandType.START, task_prompt="record")

        machine.mark_failed("video write failed")

        self.assertEqual(machine.mode, CollectorMode.FAILED)
        self.assertEqual(machine.failure_reason, "video write failed")

        stop = machine.handle_command(RecordCommandType.STOP)
        self.assertFalse(stop.accepted)
        self.assertEqual(machine.mode, CollectorMode.FAILED)

        start = machine.handle_command(RecordCommandType.START, task_prompt="new")
        self.assertFalse(start.accepted)
        self.assertEqual(machine.session.task_prompt, "record")

        discard = machine.handle_command(RecordCommandType.DISCARD)
        self.assertTrue(discard.accepted)
        self.assertTrue(discard.should_discard)
        self.assertEqual(machine.mode, CollectorMode.DISCARD)

        machine.mark_discarded()

        self.assertEqual(machine.mode, CollectorMode.IDLE)
        self.assertEqual(machine.failure_reason, "")
        self.assertIsNone(machine.session)

    def test_stop_and_discard_reject_a_different_episode(self):
        machine = RecordStateMachine()
        machine.handle_command(
            RecordCommandType.START,
            task_prompt="record",
            episode_id="episode-a",
        )

        stop = machine.handle_command(
            RecordCommandType.STOP,
            episode_id="episode-b",
        )
        discard = machine.handle_command(
            RecordCommandType.DISCARD,
            episode_id="episode-b",
        )

        self.assertFalse(stop.accepted)
        self.assertFalse(discard.accepted)
        self.assertEqual(machine.mode, CollectorMode.RECORDING)

    def test_force_can_override_episode_ownership(self):
        machine = RecordStateMachine()
        machine.handle_command(
            RecordCommandType.START,
            task_prompt="record",
            episode_id="episode-a",
        )

        stop = machine.handle_command(
            RecordCommandType.STOP,
            episode_id="episode-b",
            force=True,
        )

        self.assertTrue(stop.accepted)
        self.assertTrue(stop.should_save)

    def test_recording_safety_limits_report_the_first_violation(self):
        self.assertIn(
            "duration",
            recording_safety_reason(
                elapsed_sec=61.0,
                frame_count=10,
                max_duration_sec=60.0,
                max_frames=100,
                free_disk_bytes=10_000,
                min_free_disk_bytes=1_000,
            ),
        )
        self.assertIn(
            "frame",
            recording_safety_reason(
                elapsed_sec=1.0,
                frame_count=100,
                max_duration_sec=60.0,
                max_frames=100,
                free_disk_bytes=10_000,
                min_free_disk_bytes=1_000,
            ),
        )
        self.assertIn(
            "disk",
            recording_safety_reason(
                elapsed_sec=1.0,
                frame_count=10,
                max_duration_sec=60.0,
                max_frames=100,
                free_disk_bytes=999,
                min_free_disk_bytes=1_000,
            ),
        )
        self.assertIsNone(
            recording_safety_reason(
                elapsed_sec=1.0,
                frame_count=10,
                max_duration_sec=60.0,
                max_frames=100,
                free_disk_bytes=10_000,
                min_free_disk_bytes=1_000,
            )
        )


if __name__ == "__main__":
    unittest.main()
