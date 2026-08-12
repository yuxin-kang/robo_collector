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

        _ack_start(machine, actions[0], now_sec=0.15)
        self.assertEqual(machine.attempt_state, AttemptState.RECORDING)

        actions = machine.step(0.2, end_triggered=True)
        self.assertEqual([action.command for action in actions], ["STOP"])
        self.assertEqual(machine.attempt_state, AttemptState.WAITING_STOP_ACK)

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
        _ack_stop(
            machine,
            actions[0],
            now_sec=0.3,
            metadata_snapshot=saved,
        )
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
        start_action = machine.step(0.1, start_triggered=True)[0]
        _ack_start(machine, start_action, now_sec=0.15)
        stop_action = machine.step(0.2, end_triggered=True)[0]

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
        _ack_stop(
            machine,
            stop_action,
            now_sec=0.3,
            collector_mode="IDLE",
            metadata_snapshot=duplicate,
        )

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
        start_action = machine.step(0.1, start_triggered=True)[0]
        _ack_start(machine, start_action, now_sec=0.15)
        stop_action = machine.step(0.2, end_triggered=True)[0]
        _ack_stop(
            machine,
            stop_action,
            now_sec=30.0,
            collector_mode="IDLE",
            metadata_snapshot=missing,
        )
        machine.step(32.0, metadata_snapshot=missing, collector_mode="IDLE")

        self.assertEqual(machine.attempt_state, AttemptState.PAUSED_FAILED)
        self.assertIn("timed out", machine.last_error)

    def test_old_success_cannot_confirm_current_attempt_save(self):
        plan = gesture_plan_from_payload(_plan_payload(target_trials=1))
        machine = GestureTriggerStateMachine(plan)
        retry_needed = _snapshot(
            plan,
            [
                TrialMetadataStatus(
                    task_slug="shake_hand",
                    trial_index=0,
                    complete=False,
                    latest_state="EMPTY",
                    latest_attempt_index=1,
                    next_attempt_index=2,
                    message="attempt one was empty",
                )
            ],
        )
        machine.bootstrap(retry_needed)
        machine.step(0.0, ready_triggered=True)
        start_action = machine.step(0.1, start_triggered=True)[0]
        _ack_start(machine, start_action, now_sec=0.2)
        stop_action = machine.step(0.3, end_triggered=True)[0]
        old_success = _snapshot(
            plan,
            [
                TrialMetadataStatus(
                    task_slug="shake_hand",
                    trial_index=0,
                    complete=True,
                    latest_state="SUCCESS",
                    latest_attempt_index=1,
                    next_attempt_index=2,
                    message="old attempt saved",
                    episode_id="old-attempt-episode",
                )
            ],
        )

        _ack_stop(
            machine,
            stop_action,
            now_sec=0.4,
            collector_mode="IDLE",
            metadata_snapshot=old_success,
        )

        self.assertEqual(machine.attempt_state, AttemptState.WAITING_SAVE_METADATA)
        self.assertIn("different attempt", machine.metadata_match_reason)

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

        _ack_start(machine, retry[0], now_sec=0.8)
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

    def test_start_ack_timeout_emits_episode_scoped_discard(self):
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

        self.assertEqual([action.command for action in actions], ["DISCARD"])
        self.assertEqual(actions[0].episode_id, machine.current_attempt.episode_id)
        _ack_discard(machine, actions[0], now_sec=6.1)
        self.assertEqual(machine.attempt_state, AttemptState.PAUSED_FAILED)
        self.assertIn("timed out", machine.last_error)

    def test_abort_after_start_send_emits_episode_scoped_discard(self):
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

        self.assertEqual(action.command, "DISCARD")
        self.assertEqual(action.episode_id, machine.current_attempt.episode_id)
        _ack_discard(machine, action, now_sec=0.2)
        self.assertEqual(machine.attempt_state, AttemptState.PAUSED_FAILED)
        self.assertIn("disk full", machine.last_error)

    def test_rejected_start_for_same_episode_does_not_discard(self):
        plan = gesture_plan_from_payload(_plan_payload(target_trials=1))
        machine = GestureTriggerStateMachine(plan)
        machine.bootstrap(_missing_snapshot(plan))
        machine.step(0.0, ready_triggered=True)
        action = machine.step(0.1, start_triggered=True)[0]

        emitted = machine.step(
            0.2,
            collector_mode="RECORDING",
            collector_episode_id=action.episode_id,
            collector_last_command_id=action.command_id,
            collector_last_command="START",
            collector_last_command_outcome="REJECTED",
            collector_last_command_episode_id=action.episode_id,
        )

        self.assertEqual(emitted, [])
        self.assertEqual(machine.attempt_state, AttemptState.PAUSED_FAILED)

    def test_stop_waits_for_ack_and_retries_same_episode(self):
        plan = gesture_plan_from_payload(_plan_payload(target_trials=1))
        machine = GestureTriggerStateMachine(plan)
        waiting = _missing_snapshot(plan)
        machine.bootstrap(waiting)
        machine.step(0.0, ready_triggered=True)
        start_action = machine.step(0.1, start_triggered=True)[0]
        _ack_start(machine, start_action, now_sec=0.2)

        first = machine.step(0.3, end_triggered=True)
        retry = machine.step(
            0.9,
            collector_mode="RECORDING",
            collector_episode_id=machine.current_attempt.episode_id,
        )

        self.assertEqual([action.command for action in first], ["STOP"])
        self.assertEqual([action.command for action in retry], ["STOP"])
        self.assertEqual(retry[0].episode_id, machine.current_attempt.episode_id)
        self.assertEqual(machine.attempt_state, AttemptState.WAITING_STOP_ACK)

        _ack_stop(
            machine,
            retry[0],
            now_sec=1.0,
            metadata_snapshot=waiting,
        )
        self.assertEqual(machine.attempt_state, AttemptState.WAITING_SAVE_METADATA)

        machine.step(
            1.5,
            metadata_snapshot=waiting,
            collector_mode="IDLE",
        )
        self.assertEqual(machine.attempt_state, AttemptState.WAITING_SAVE_METADATA)

        machine.step(
            2.1,
            metadata_snapshot=waiting,
            collector_mode="IDLE",
        )
        self.assertEqual(machine.attempt_state, AttemptState.PAUSED_FAILED)
        self.assertIn("save confirmation timed out", machine.last_error)

    def test_stop_ack_can_arrive_after_collector_enters_saving(self):
        plan = gesture_plan_from_payload(_plan_payload(target_trials=1))
        machine = GestureTriggerStateMachine(plan)
        waiting = _missing_snapshot(plan)
        machine.bootstrap(waiting)
        machine.step(0.0, ready_triggered=True)
        start_action = machine.step(0.1, start_triggered=True)[0]
        _ack_start(machine, start_action, now_sec=0.2)
        stop_action = machine.step(0.3, end_triggered=True)[0]

        _ack_stop(
            machine,
            stop_action,
            now_sec=0.4,
            collector_mode="SAVING",
            metadata_snapshot=waiting,
        )

        self.assertEqual(machine.attempt_state, AttemptState.WAITING_SAVE_METADATA)

    def test_saving_state_recovers_when_stop_receipt_is_lost(self):
        plan = gesture_plan_from_payload(_plan_payload(target_trials=1))
        machine = GestureTriggerStateMachine(plan)
        waiting = _missing_snapshot(plan)
        machine.bootstrap(waiting)
        machine.step(0.0, ready_triggered=True)
        start_action = machine.step(0.1, start_triggered=True)[0]
        _ack_start(machine, start_action, now_sec=0.2)
        machine.step(0.3, end_triggered=True)

        emitted = machine.step(
            0.4,
            collector_mode="SAVING",
            collector_episode_id=machine.current_attempt.episode_id,
            metadata_snapshot=waiting,
        )

        self.assertEqual(emitted, [])
        self.assertEqual(machine.attempt_state, AttemptState.WAITING_SAVE_METADATA)

    def test_new_save_progress_renews_save_confirmation_deadline(self):
        plan = gesture_plan_from_payload(_plan_payload(target_trials=1))
        machine = GestureTriggerStateMachine(plan)
        waiting = _missing_snapshot(plan)
        machine.bootstrap(waiting)
        machine.step(0.0, ready_triggered=True)
        start_action = machine.step(0.1, start_triggered=True)[0]
        _ack_start(machine, start_action, now_sec=0.2)
        stop_action = machine.step(0.3, end_triggered=True)[0]
        _ack_stop(
            machine,
            stop_action,
            now_sec=1.0,
            collector_mode="NEED_TO_SAVE",
            metadata_snapshot=waiting,
            collector_save_progress_token="0",
        )

        first = machine.step(
            1.8,
            metadata_snapshot=waiting,
            collector_mode="SAVING",
            collector_episode_id=machine.current_attempt.episode_id,
            collector_save_progress_token="1",
        )
        second = machine.step(
            2.5,
            metadata_snapshot=waiting,
            collector_mode="SAVING",
            collector_episode_id=machine.current_attempt.episode_id,
            collector_save_progress_token="2",
        )

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(machine.attempt_state, AttemptState.WAITING_SAVE_METADATA)

        machine.step(
            3.6,
            metadata_snapshot=waiting,
            collector_mode="",
        )
        self.assertEqual(machine.attempt_state, AttemptState.PAUSED_FAILED)
        self.assertIn("save confirmation timed out", machine.last_error)

    def test_unchanged_save_progress_does_not_renew_deadline(self):
        plan = gesture_plan_from_payload(_plan_payload(target_trials=1))
        machine = GestureTriggerStateMachine(plan)
        waiting = _missing_snapshot(plan)
        machine.bootstrap(waiting)
        machine.step(0.0, ready_triggered=True)
        start_action = machine.step(0.1, start_triggered=True)[0]
        _ack_start(machine, start_action, now_sec=0.2)
        stop_action = machine.step(0.3, end_triggered=True)[0]
        _ack_stop(
            machine,
            stop_action,
            now_sec=1.0,
            collector_mode="SAVING",
            metadata_snapshot=waiting,
            collector_save_progress_token="1",
        )

        machine.step(
            1.8,
            metadata_snapshot=waiting,
            collector_mode="SAVING",
            collector_episode_id=machine.current_attempt.episode_id,
            collector_save_progress_token="1",
        )
        actions = machine.step(
            2.1,
            metadata_snapshot=waiting,
            collector_mode="SAVING",
            collector_episode_id=machine.current_attempt.episode_id,
            collector_save_progress_token="1",
        )

        self.assertEqual(actions, [])
        self.assertEqual(machine.attempt_state, AttemptState.PAUSED_FAILED)
        self.assertIn("save confirmation timed out", machine.last_error)

    def test_save_progress_cannot_extend_maximum_wait(self):
        payload = _plan_payload(target_trials=1)
        payload["collector"]["max_save_wait_sec"] = 2.0
        plan = gesture_plan_from_payload(payload)
        machine = GestureTriggerStateMachine(plan)
        waiting = _missing_snapshot(plan)
        machine.bootstrap(waiting)
        machine.step(0.0, ready_triggered=True)
        start_action = machine.step(0.1, start_triggered=True)[0]
        _ack_start(machine, start_action, now_sec=0.2)
        stop_action = machine.step(0.3, end_triggered=True)[0]
        _ack_stop(
            machine,
            stop_action,
            now_sec=1.0,
            collector_mode="SAVING",
            metadata_snapshot=waiting,
            collector_save_progress_token="0",
        )

        machine.step(
            1.8,
            metadata_snapshot=waiting,
            collector_mode="SAVING",
            collector_episode_id=machine.current_attempt.episode_id,
            collector_save_progress_token="1",
        )
        machine.step(
            2.7,
            metadata_snapshot=waiting,
            collector_mode="SAVING",
            collector_episode_id=machine.current_attempt.episode_id,
            collector_save_progress_token="2",
        )
        actions = machine.step(
            3.0,
            metadata_snapshot=waiting,
            collector_mode="SAVING",
            collector_episode_id=machine.current_attempt.episode_id,
            collector_save_progress_token="3",
        )

        self.assertEqual(actions, [])
        self.assertEqual(machine.attempt_state, AttemptState.PAUSED_FAILED)
        self.assertIn("maximum save reconciliation wait", machine.last_error)

    def test_late_saved_metadata_does_not_bypass_maximum_wait(self):
        payload = _plan_payload(target_trials=1)
        payload["collector"]["max_save_wait_sec"] = 2.0
        plan = gesture_plan_from_payload(payload)
        machine = GestureTriggerStateMachine(plan)
        waiting = _missing_snapshot(plan)
        machine.bootstrap(waiting)
        machine.step(0.0, ready_triggered=True)
        start_action = machine.step(0.1, start_triggered=True)[0]
        _ack_start(machine, start_action, now_sec=0.2)
        stop_action = machine.step(0.3, end_triggered=True)[0]
        _ack_stop(
            machine,
            stop_action,
            now_sec=1.0,
            collector_mode="SAVING",
            metadata_snapshot=waiting,
            collector_save_progress_token="0",
        )
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
                    episode_id=start_action.episode_id,
                )
            ],
        )

        actions = machine.step(
            3.1,
            collector_mode="IDLE",
            metadata_snapshot=saved,
        )

        self.assertEqual(actions, [])
        self.assertEqual(machine.attempt_state, AttemptState.PAUSED_FAILED)
        self.assertIn("maximum save reconciliation wait", machine.last_error)

    def test_saved_metadata_recovers_when_stop_receipt_is_lost(self):
        plan = gesture_plan_from_payload(_plan_payload(target_trials=1))
        machine = GestureTriggerStateMachine(plan)
        machine.bootstrap(_missing_snapshot(plan))
        machine.step(0.0, ready_triggered=True)
        start_action = machine.step(0.1, start_triggered=True)[0]
        _ack_start(machine, start_action, now_sec=0.2)
        machine.step(0.3, end_triggered=True)
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
                    episode_id=start_action.episode_id,
                )
            ],
        )

        actions = machine.step(
            10.0,
            collector_mode="IDLE",
            metadata_snapshot=saved,
        )

        self.assertEqual(actions, [])
        self.assertEqual(machine.attempt_state, AttemptState.SAVED)

    def test_recording_watchdog_discards_incomplete_attempt(self):
        payload = _plan_payload(target_trials=1)
        payload["collector"]["max_recording_duration_sec"] = 1.0
        plan = gesture_plan_from_payload(payload)
        machine = GestureTriggerStateMachine(plan)
        machine.bootstrap(_missing_snapshot(plan))
        machine.step(0.0, ready_triggered=True)
        start_action = machine.step(0.1, start_triggered=True)[0]
        _ack_start(machine, start_action, now_sec=0.2)

        actions = machine.step(
            1.3,
            collector_mode="RECORDING",
            collector_episode_id=machine.current_attempt.episode_id,
        )

        self.assertEqual([action.command for action in actions], ["DISCARD"])
        self.assertEqual(machine.attempt_state, AttemptState.WAITING_DISCARD_ACK)
        _ack_discard(machine, actions[0], now_sec=1.4)
        self.assertEqual(machine.attempt_state, AttemptState.PAUSED_FAILED)
        self.assertIn("maximum recording duration", machine.last_error)

    def test_collector_safety_discard_ends_gesture_recording_immediately(self):
        plan = gesture_plan_from_payload(_plan_payload(target_trials=1))
        machine = GestureTriggerStateMachine(plan)
        machine.bootstrap(_missing_snapshot(plan))
        machine.step(0.0, ready_triggered=True)
        start_action = machine.step(0.1, start_triggered=True)[0]
        _ack_start(machine, start_action, now_sec=0.2)

        actions = machine.step(
            0.3,
            collector_mode="IDLE",
            collector_last_episode_id=start_action.episode_id,
            collector_last_episode_outcome="DISCARDED",
        )

        self.assertEqual(actions, [])
        self.assertEqual(machine.attempt_state, AttemptState.PAUSED_FAILED)
        self.assertIn("discarded", machine.last_error)

    def test_external_stop_reconciles_matching_saved_metadata(self):
        plan = gesture_plan_from_payload(_plan_payload(target_trials=1))
        machine = GestureTriggerStateMachine(plan)
        machine.bootstrap(_missing_snapshot(plan))
        machine.step(0.0, ready_triggered=True)
        start_action = machine.step(0.1, start_triggered=True)[0]
        _ack_start(machine, start_action, now_sec=0.2)
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
                    episode_id=start_action.episode_id,
                )
            ],
        )

        machine.step(
            0.3,
            collector_mode="IDLE",
            metadata_snapshot=saved,
            collector_last_episode_id=start_action.episode_id,
            collector_last_episode_outcome="SAVED",
        )

        self.assertEqual(machine.attempt_state, AttemptState.SAVED)

    def test_stale_gesture_sample_discards_incomplete_attempt(self):
        plan = gesture_plan_from_payload(_plan_payload(target_trials=1))
        machine = GestureTriggerStateMachine(plan)
        machine.bootstrap(_missing_snapshot(plan))
        machine.step(0.0, ready_triggered=True)
        start_action = machine.step(0.1, start_triggered=True)[0]
        _ack_start(machine, start_action, now_sec=0.2)

        actions = machine.step(
            0.3,
            collector_mode="RECORDING",
            collector_episode_id=machine.current_attempt.episode_id,
            recording_sample_fresh=False,
        )

        self.assertEqual([action.command for action in actions], ["DISCARD"])
        self.assertEqual(machine.attempt_state, AttemptState.WAITING_DISCARD_ACK)
        _ack_discard(machine, actions[0], now_sec=0.4)
        self.assertEqual(machine.attempt_state, AttemptState.PAUSED_FAILED)
        self.assertIn("gesture sample", machine.last_error)

    def test_discard_requires_matching_receipt_and_retries_same_command(self):
        payload = _plan_payload(target_trials=1)
        payload["collector"]["discard_confirm_timeout_sec"] = 2.0
        plan = gesture_plan_from_payload(payload)
        machine = GestureTriggerStateMachine(plan)
        machine.bootstrap(_missing_snapshot(plan))
        machine.step(0.0, ready_triggered=True)
        start_action = machine.step(0.1, start_triggered=True)[0]
        _ack_start(machine, start_action, now_sec=0.2)

        first = machine.step(0.3, recording_sample_fresh=False)
        stale_idle = machine.step(0.4, collector_mode="IDLE")
        retry = machine.step(
            0.9,
            collector_mode="RECORDING",
            collector_episode_id=machine.current_attempt.episode_id,
        )

        self.assertEqual(stale_idle, [])
        self.assertEqual(machine.attempt_state, AttemptState.WAITING_DISCARD_ACK)
        self.assertEqual([action.command for action in retry], ["DISCARD"])
        self.assertEqual(retry[0].command_id, first[0].command_id)

        _ack_discard(machine, retry[0], now_sec=1.0)
        self.assertEqual(machine.attempt_state, AttemptState.PAUSED_FAILED)

    def test_saved_episode_while_discard_pending_requires_reconciliation(self):
        plan = gesture_plan_from_payload(_plan_payload(target_trials=1))
        machine = GestureTriggerStateMachine(plan)
        machine.bootstrap(_missing_snapshot(plan))
        machine.step(0.0, ready_triggered=True)
        start_action = machine.step(0.1, start_triggered=True)[0]
        _ack_start(machine, start_action, now_sec=0.2)
        machine.step(0.3, recording_sample_fresh=False)

        machine.step(
            0.4,
            collector_mode="IDLE",
            collector_last_episode_id=machine.current_attempt.episode_id,
            collector_last_episode_outcome="SAVED",
        )

        self.assertEqual(
            machine.attempt_state, AttemptState.PAUSED_AMBIGUOUS_METADATA
        )
        self.assertIn("saved", machine.last_error)

    def test_discard_receipt_for_different_episode_is_not_acknowledged(self):
        plan = gesture_plan_from_payload(_plan_payload(target_trials=1))
        machine = GestureTriggerStateMachine(plan)
        machine.bootstrap(_missing_snapshot(plan))
        machine.step(0.0, ready_triggered=True)
        start_action = machine.step(0.1, start_triggered=True)[0]
        _ack_start(machine, start_action, now_sec=0.2)
        action = machine.step(0.3, recording_sample_fresh=False)[0]

        machine.step(
            0.4,
            collector_mode="IDLE",
            collector_last_command_id=action.command_id,
            collector_last_command="DISCARD",
            collector_last_command_outcome="SUCCEEDED",
            collector_last_command_episode_id="different-episode",
        )

        self.assertEqual(machine.attempt_state, AttemptState.WAITING_DISCARD_ACK)


def _snapshot(plan, statuses):
    return MetadataSnapshot(
        dataset_root=Path("outputs/demo"),
        plan_id=plan.plan_id,
        statuses={(status.task_slug, status.trial_index): status for status in statuses},
    )


def _missing_snapshot(plan):
    return _snapshot(
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


def _ack_discard(machine, action, *, now_sec):
    machine.step(
        now_sec,
        collector_mode="IDLE",
        collector_last_command_id=action.command_id,
        collector_last_command="DISCARD",
        collector_last_command_outcome="SUCCEEDED",
        collector_last_command_episode_id=action.episode_id,
        collector_last_episode_id=action.episode_id,
        collector_last_episode_outcome="DISCARDED",
    )


def _ack_start(machine, action, *, now_sec):
    machine.step(
        now_sec,
        collector_mode="RECORDING",
        collector_episode_id=action.episode_id,
        collector_last_command_id=action.command_id,
        collector_last_command="START",
        collector_last_command_outcome="SUCCEEDED",
        collector_last_command_episode_id=action.episode_id,
    )


def _ack_stop(
    machine,
    action,
    *,
    now_sec,
    collector_mode="NEED_TO_SAVE",
    metadata_snapshot=None,
    collector_save_progress_token="",
):
    machine.step(
        now_sec,
        collector_mode=collector_mode,
        collector_episode_id=(action.episode_id if collector_mode != "IDLE" else ""),
        collector_last_command_id=action.command_id,
        collector_last_command="STOP",
        collector_last_command_outcome="SUCCEEDED",
        collector_last_command_episode_id=action.episode_id,
        collector_save_progress_token=collector_save_progress_token,
        metadata_snapshot=metadata_snapshot,
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
