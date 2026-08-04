import unittest
from pathlib import Path

from robo_collector.gesture_logic import AttemptState, GestureTriggerStateMachine
from robo_collector.gesture_metadata import MetadataSnapshot, TrialMetadataStatus
from robo_collector.gesture_plan import gesture_plan_from_payload


class GestureTriggerStateTest(unittest.TestCase):
    def test_bootstrap_start_stop_save_and_complete(self):
        plan = gesture_plan_from_payload(_plan_payload(target_trials=1))
        machine = GestureTriggerStateMachine(plan)
        waiting = _snapshot(
            plan,
            [
                TrialMetadataStatus(
                    task_slug="shake_hand",
                    trial_index=0,
                    complete=False,
                    latest_state="MISSING",
                    latest_attempt_index=0,
                    next_attempt_index=1,
                    message="no metadata row for trial yet",
                )
            ],
        )
        machine.bootstrap(waiting)

        self.assertEqual(machine.attempt_state, AttemptState.WAITING_READY)
        machine.step(0.0, ready_triggered=True)
        self.assertEqual(machine.attempt_state, AttemptState.ARMED)

        actions = machine.step(0.1, start_triggered=True)
        self.assertEqual([action.command for action in actions], ["START"])
        self.assertEqual(machine.attempt_state, AttemptState.WAITING_START_ACK)

        machine.step(
            0.15,
            collector_mode="RECORDING",
            collector_episode_id=machine.current_attempt.episode_id,
        )
        self.assertEqual(machine.attempt_state, AttemptState.RECORDING)

        actions = machine.step(0.2, end_triggered=True)
        self.assertEqual([action.command for action in actions], ["STOP"])
        self.assertEqual(machine.attempt_state, AttemptState.WAITING_SAVE_METADATA)

        saved = _snapshot(
            plan,
            [
                TrialMetadataStatus(
                    task_slug="shake_hand",
                    trial_index=0,
                    complete=True,
                    latest_state="SUCCESS",
                    latest_attempt_index=1,
                    next_attempt_index=2,
                    message="metadata row saved successfully",
                    episode_id=machine.current_attempt.episode_id,
                )
            ],
        )
        machine.step(0.3, metadata_snapshot=saved, collector_mode="NEED_TO_SAVE")
        self.assertEqual(machine.attempt_state, AttemptState.WAITING_SAVE_METADATA)

        machine.step(0.4, metadata_snapshot=saved, collector_mode="IDLE")
        self.assertEqual(machine.attempt_state, AttemptState.SAVED)

        machine.step(0.5, metadata_snapshot=saved, collector_mode="IDLE")
        self.assertEqual(machine.attempt_state, AttemptState.COMPLETE)

    def test_duplicate_metadata_pauses_without_more_commands(self):
        plan = gesture_plan_from_payload(_plan_payload(target_trials=1))
        machine = GestureTriggerStateMachine(plan)
        machine.bootstrap(
            _snapshot(
                plan,
                [
                    TrialMetadataStatus(
                        task_slug="shake_hand",
                        trial_index=0,
                        complete=False,
                        latest_state="MISSING",
                        latest_attempt_index=0,
                        next_attempt_index=1,
                        message="no metadata row for trial yet",
                    )
                ],
            )
        )
        machine.step(0.0, ready_triggered=True)
        machine.step(0.1, start_triggered=True)
        machine.step(
            0.15,
            collector_mode="RECORDING",
            collector_episode_id=machine.current_attempt.episode_id,
        )
        machine.step(0.2, end_triggered=True)

        duplicate = _snapshot(
            plan,
            [
                TrialMetadataStatus(
                    task_slug="shake_hand",
                    trial_index=0,
                    complete=False,
                    latest_state="DUPLICATE",
                    latest_attempt_index=1,
                    next_attempt_index=1,
                    message="duplicate metadata rows for episode_id dup",
                    episode_id="dup",
                )
            ],
        )
        machine.step(0.3, metadata_snapshot=duplicate, collector_mode="IDLE")

        self.assertEqual(machine.attempt_state, AttemptState.PAUSED_AMBIGUOUS_METADATA)
        self.assertEqual(machine.step(0.4, ready_triggered=True), [])

    def test_timeout_pauses_failed(self):
        plan = gesture_plan_from_payload(_plan_payload(target_trials=1))
        machine = GestureTriggerStateMachine(plan)
        missing = _snapshot(
            plan,
            [
                TrialMetadataStatus(
                    task_slug="shake_hand",
                    trial_index=0,
                    complete=False,
                    latest_state="MISSING",
                    latest_attempt_index=0,
                    next_attempt_index=1,
                    message="no metadata row for trial yet",
                )
            ],
        )
        machine.bootstrap(missing)
        machine.step(0.0, ready_triggered=True)
        machine.step(0.1, start_triggered=True)
        machine.step(
            0.15,
            collector_mode="RECORDING",
            collector_episode_id=machine.current_attempt.episode_id,
        )
        machine.step(0.2, end_triggered=True)
        machine.step(30.0, metadata_snapshot=missing, collector_mode="IDLE")

        self.assertEqual(machine.attempt_state, AttemptState.PAUSED_FAILED)
        self.assertIn("timed out", machine.last_error)

    def test_bootstrap_recovery_skips_completed_trials(self):
        plan = gesture_plan_from_payload(_plan_payload(target_trials=3))
        machine = GestureTriggerStateMachine(plan)
        machine.bootstrap(
            _snapshot(
                plan,
                [
                    TrialMetadataStatus(
                        task_slug="shake_hand",
                        trial_index=0,
                        complete=True,
                        latest_state="SUCCESS",
                        latest_attempt_index=1,
                        next_attempt_index=2,
                        message="metadata row saved successfully",
                    ),
                    TrialMetadataStatus(
                        task_slug="shake_hand",
                        trial_index=1,
                        complete=True,
                        latest_state="SUCCESS",
                        latest_attempt_index=1,
                        next_attempt_index=2,
                        message="metadata row saved successfully",
                    ),
                    TrialMetadataStatus(
                        task_slug="shake_hand",
                        trial_index=2,
                        complete=False,
                        latest_state="MISSING",
                        latest_attempt_index=0,
                        next_attempt_index=1,
                        message="no metadata row for trial yet",
                    ),
                ],
            )
        )

        self.assertEqual(machine.completed_count, 2)
        self.assertEqual(machine.current_attempt.trial_index, 2)
        self.assertEqual(machine.current_attempt.attempt_index, 1)

    def test_start_waits_for_matching_ack_and_retries(self):
        plan = gesture_plan_from_payload(_plan_payload(target_trials=1))
        machine = GestureTriggerStateMachine(plan)
        waiting = _snapshot(
            plan,
            [
                TrialMetadataStatus(
                    task_slug="shake_hand",
                    trial_index=0,
                    complete=False,
                    latest_state="MISSING",
                    latest_attempt_index=0,
                    next_attempt_index=1,
                    message="missing",
                )
            ],
        )
        machine.bootstrap(waiting)
        machine.step(0.0, ready_triggered=True)

        first = machine.step(0.1, start_triggered=True)
        self.assertEqual([action.command for action in first], ["START"])
        self.assertEqual(machine.attempt_state, AttemptState.WAITING_START_ACK)

        retry = machine.step(0.7, collector_mode="IDLE")
        self.assertEqual([action.command for action in retry], ["START"])
        self.assertEqual(machine.attempt_state, AttemptState.WAITING_START_ACK)

        machine.step(
            0.8,
            collector_mode="RECORDING",
            collector_episode_id=machine.current_attempt.episode_id,
        )
        self.assertEqual(machine.attempt_state, AttemptState.RECORDING)

    def test_start_ack_for_different_episode_fails_closed(self):
        plan = gesture_plan_from_payload(_plan_payload(target_trials=1))
        machine = GestureTriggerStateMachine(plan)
        machine.bootstrap(
            _snapshot(
                plan,
                [
                    TrialMetadataStatus(
                        task_slug="shake_hand",
                        trial_index=0,
                        complete=False,
                        latest_state="MISSING",
                        latest_attempt_index=0,
                        next_attempt_index=1,
                        message="missing",
                    )
                ],
            )
        )
        machine.step(0.0, ready_triggered=True)
        machine.step(0.1, start_triggered=True)
        machine.step(
            0.2,
            collector_mode="RECORDING",
            collector_episode_id="another-episode",
        )

        self.assertEqual(machine.attempt_state, AttemptState.PAUSED_FAILED)
        self.assertIn("different episode", machine.last_error)

    def test_start_ack_timeout_emits_fail_closed_stop(self):
        plan = gesture_plan_from_payload(_plan_payload(target_trials=1))
        machine = GestureTriggerStateMachine(plan)
        machine.bootstrap(
            _snapshot(
                plan,
                [
                    TrialMetadataStatus(
                        task_slug="shake_hand",
                        trial_index=0,
                        complete=False,
                        latest_state="MISSING",
                        latest_attempt_index=0,
                        next_attempt_index=1,
                        message="missing",
                    )
                ],
            )
        )
        machine.step(0.0, ready_triggered=True)
        machine.step(0.1, start_triggered=True)

        actions = machine.step(6.0, collector_mode="")

        self.assertEqual([action.command for action in actions], ["STOP"])
        self.assertEqual(machine.attempt_state, AttemptState.PAUSED_FAILED)
        self.assertIn("timed out", machine.last_error)

    def test_abort_active_attempt_emits_stop(self):
        plan = gesture_plan_from_payload(_plan_payload(target_trials=1))
        machine = GestureTriggerStateMachine(plan)
        machine.bootstrap(
            _snapshot(
                plan,
                [
                    TrialMetadataStatus(
                        task_slug="shake_hand",
                        trial_index=0,
                        complete=False,
                        latest_state="MISSING",
                        latest_attempt_index=0,
                        next_attempt_index=1,
                        message="missing",
                    )
                ],
            )
        )
        machine.step(0.0, ready_triggered=True)
        machine.step(0.1, start_triggered=True)

        action = machine.abort_active_attempt("progress disk full")

        self.assertEqual(action.command, "STOP")
        self.assertEqual(machine.attempt_state, AttemptState.PAUSED_FAILED)
        self.assertIn("disk full", machine.last_error)


def _snapshot(plan, statuses):
    return MetadataSnapshot(
        dataset_root=Path("outputs/demo"),
        plan_id=plan.plan_id,
        statuses={(status.task_slug, status.trial_index): status for status in statuses},
    )


def _plan_payload(*, target_trials: int) -> dict:
    return {
        "version": 1,
        "plan_id": "handshake_set_a_20260621",
        "collector": {"save_confirm_timeout_sec": 1.0},
        "gesture_source": {
            "topic": "/robo_state/sample",
            "field": "robot_state.joint_pos",
            "indices": [0, 1, 2],
        },
        "references": {
            "ready": {"vector": [0.0, 0.0, 0.0]},
            "start": {"vector": [0.1, 0.1, 0.1]},
            "end": {"vector": [0.2, 0.2, 0.2]},
        },
        "conditions": {
            "task_start_condition": {
                "reference_name": "start",
                "threshold_l2": 0.05,
                "stable_samples": 1,
                "release_threshold_l2": 0.08,
                "cooldown_sec": 0.1,
            },
            "task_end_condition": {
                "reference_name": "end",
                "threshold_l2": 0.05,
                "stable_samples": 1,
            },
            "return_to_ready_condition": {
                "reference_name": "ready",
                "threshold_l2": 0.05,
                "stable_samples": 1,
            },
        },
        "tasks": [
            {
                "task_slug": "shake_hand",
                "task_prompt": "Shake hand with somebody",
                "target_trials": target_trials,
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
