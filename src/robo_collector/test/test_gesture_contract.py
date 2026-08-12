import unittest

from robo_collector.gesture_episode_id import (
    build_gesture_episode_id,
    parse_gesture_episode_id,
)
from robo_collector.gesture_plan import GesturePlanError, gesture_plan_from_payload


class GestureContractTest(unittest.TestCase):
    def test_episode_id_round_trips(self):
        value = build_gesture_episode_id(
            plan_id="handshake set/a",
            task_slug="shake_hand",
            trial_index=12,
            attempt_index=1,
        )

        parsed = parse_gesture_episode_id(value)

        self.assertEqual(parsed.plan_id, "handshake set/a")
        self.assertEqual(parsed.task_slug, "shake_hand")
        self.assertEqual(parsed.trial_index, 12)
        self.assertEqual(parsed.attempt_index, 1)

    def test_parse_episode_id_rejects_duplicate_components(self):
        with self.assertRaisesRegex(ValueError, "duplicate gesture episode_id component"):
            parse_gesture_episode_id("plan=a__task=t__trial=0000__plan=b")

    def test_episode_id_rejects_negative_or_zero_indexes(self):
        with self.assertRaisesRegex(ValueError, "trial_index must be non-negative"):
            build_gesture_episode_id(
                plan_id="plan",
                task_slug="task",
                trial_index=-1,
                attempt_index=1,
            )
        with self.assertRaisesRegex(ValueError, "attempt_index must be >= 1"):
            build_gesture_episode_id(
                plan_id="plan",
                task_slug="task",
                trial_index=0,
                attempt_index=0,
            )

    def test_episode_id_rejects_separator_collision_in_plan_or_task(self):
        with self.assertRaisesRegex(ValueError, "plan_id must not contain '__'"):
            build_gesture_episode_id(
                plan_id="plan__a",
                task_slug="task",
                trial_index=0,
                attempt_index=1,
            )

    def test_plan_rejects_separator_collision_in_plan_id_and_task_slug(self):
        payload = _plan_payload()
        payload["plan_id"] = "plan__bad"
        with self.assertRaisesRegex(GesturePlanError, "plan_id must not contain '__'"):
            gesture_plan_from_payload(payload)

        payload = _plan_payload()
        payload["tasks"][0]["task_slug"] = "shake__hand"
        with self.assertRaisesRegex(GesturePlanError, "tasks\\[0\\]\\.task_slug must not contain '__'"):
            gesture_plan_from_payload(payload)
        with self.assertRaisesRegex(ValueError, "task_slug must not contain '__'"):
            build_gesture_episode_id(
                plan_id="plan",
                task_slug="task__a",
                trial_index=0,
                attempt_index=1,
            )

    def test_plan_rejects_missing_or_null_plan_id_and_task_slug(self):
        payload = _plan_payload()
        payload["plan_id"] = None
        with self.assertRaisesRegex(GesturePlanError, "plan_id is required"):
            gesture_plan_from_payload(payload)

        payload = _plan_payload()
        del payload["plan_id"]
        with self.assertRaisesRegex(GesturePlanError, "plan_id is required"):
            gesture_plan_from_payload(payload)

        payload = _plan_payload()
        payload["tasks"][0]["task_slug"] = None
        with self.assertRaisesRegex(GesturePlanError, "tasks\\[0\\]\\.task_slug is required"):
            gesture_plan_from_payload(payload)

        payload = _plan_payload()
        del payload["tasks"][0]["task_slug"]
        with self.assertRaisesRegex(GesturePlanError, "tasks\\[0\\]\\.task_slug is required"):
            gesture_plan_from_payload(payload)

    def test_episode_id_rejects_bool_and_fractional_indexes(self):
        with self.assertRaisesRegex(ValueError, "trial_index must be an integer"):
            build_gesture_episode_id(
                plan_id="plan",
                task_slug="task",
                trial_index=1.5,
                attempt_index=1,
            )
        with self.assertRaisesRegex(ValueError, "attempt_index must be an integer"):
            build_gesture_episode_id(
                plan_id="plan",
                task_slug="task",
                trial_index=0,
                attempt_index=True,
            )

    def test_plan_expands_target_trials(self):
        plan = gesture_plan_from_payload(_plan_payload())

        self.assertEqual(plan.total_trials, 50)
        self.assertEqual(plan.planned_trials[0].trial_index, 0)
        self.assertEqual(plan.planned_trials[-1].trial_index, 49)

    def test_plan_rejects_duplicate_task_slug(self):
        payload = _plan_payload()
        payload["tasks"].append(
            {
                "task_slug": "shake_hand",
                "task_prompt": "Shake hand again",
                "target_trials": 1,
            }
        )
        with self.assertRaisesRegex(GesturePlanError, "duplicate task_slug"):
            gesture_plan_from_payload(payload)

    def test_plan_rejects_invalid_target_trials_and_gesture_field(self):
        payload = _plan_payload()
        payload["tasks"][0]["target_trials"] = 0
        with self.assertRaisesRegex(GesturePlanError, "target_trials"):
            gesture_plan_from_payload(payload)

        payload = _plan_payload()
        payload["gesture_source"]["field"] = "robot_state.cmd_joint_pos"
        with self.assertRaisesRegex(GesturePlanError, "unsupported gesture_source.field"):
            gesture_plan_from_payload(payload)

        payload = _plan_payload()
        payload["collector"]["fps"] = 0
        with self.assertRaisesRegex(GesturePlanError, "collector.fps"):
            gesture_plan_from_payload(payload)

        payload = _plan_payload()
        payload["collector"]["max_save_wait_sec"] = 0
        with self.assertRaisesRegex(GesturePlanError, "max_save_wait_sec"):
            gesture_plan_from_payload(payload)

        payload = _plan_payload()
        payload["collector"]["save_confirm_timeout_sec"] = 10
        payload["collector"]["max_save_wait_sec"] = 9
        with self.assertRaisesRegex(GesturePlanError, "max_save_wait_sec"):
            gesture_plan_from_payload(payload)

        payload = _plan_payload()
        payload["collector"]["auto_discard"] = True
        with self.assertRaisesRegex(GesturePlanError, "auto_discard=true is not implemented"):
            gesture_plan_from_payload(payload)

    def test_plan_wraps_malformed_scalar_conversions(self):
        payload = _plan_payload()
        payload["version"] = "not-an-int"
        with self.assertRaisesRegex(GesturePlanError, "version"):
            gesture_plan_from_payload(payload)

        payload = _plan_payload()
        payload["gesture_source"]["indices"] = ["bad"]
        with self.assertRaisesRegex(GesturePlanError, "gesture_source.indices\\[0\\]"):
            gesture_plan_from_payload(payload)

        payload = _plan_payload()
        payload["references"]["ready"]["vector"] = ["bad"]
        with self.assertRaisesRegex(GesturePlanError, "references.ready.vector\\[0\\]"):
            gesture_plan_from_payload(payload)

    def test_plan_rejects_fractional_and_nonfinite_numbers(self):
        payload = _plan_payload()
        payload["version"] = 1.9
        with self.assertRaisesRegex(GesturePlanError, "version"):
            gesture_plan_from_payload(payload)

        payload = _plan_payload()
        payload["gesture_source"]["indices"] = [1.9]
        with self.assertRaisesRegex(GesturePlanError, "gesture_source.indices\\[0\\]"):
            gesture_plan_from_payload(payload)

        payload = _plan_payload()
        payload["references"]["ready"]["vector"] = [float("inf")]
        with self.assertRaisesRegex(GesturePlanError, "finite number"):
            gesture_plan_from_payload(payload)

        payload = _plan_payload()
        payload["conditions"]["task_start_condition"]["threshold_l2"] = float("nan")
        with self.assertRaisesRegex(GesturePlanError, "finite positive number"):
            gesture_plan_from_payload(payload)

    def test_plan_requires_boolean_auto_discard(self):
        payload = _plan_payload()
        payload["collector"]["auto_discard"] = "false"
        with self.assertRaisesRegex(GesturePlanError, "collector.auto_discard must be a boolean"):
            gesture_plan_from_payload(payload)


def _plan_payload():
    return {
        "version": 1,
        "plan_id": "handshake_set_a_20260621",
        "collector": {
            "command_topic": "/robo_collector/record_command",
            "status_topic": "/robo_collector/status",
            "fps": 50.0,
        },
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
                "stable_samples": 3,
                "release_threshold_l2": 0.08,
                "cooldown_sec": 1.0,
            },
            "task_end_condition": {
                "reference_name": "end",
                "threshold_l2": 0.05,
                "stable_samples": 3,
            },
            "return_to_ready_condition": {
                "reference_name": "ready",
                "threshold_l2": 0.05,
                "stable_samples": 3,
            },
        },
        "tasks": [
            {
                "task_slug": "shake_hand",
                "task_prompt": "Shake hand with somebody",
                "target_trials": 50,
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
