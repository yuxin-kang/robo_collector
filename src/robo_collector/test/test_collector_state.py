import unittest

from robo_collector.collector_state import (
    CollectorMode,
    RecordCommandType,
    RecordStateMachine,
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

    def test_start_ignored_while_recording(self):
        machine = RecordStateMachine()
        machine.handle_command(RecordCommandType.START, task_prompt="first")

        result = machine.handle_command(RecordCommandType.START, task_prompt="second")

        self.assertFalse(result.accepted)
        self.assertEqual(machine.session.task_prompt, "first")


if __name__ == "__main__":
    unittest.main()
