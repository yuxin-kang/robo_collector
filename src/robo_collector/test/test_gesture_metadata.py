import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from robo_collector.gesture_episode_id import build_gesture_episode_id
from robo_collector.gesture_metadata import (
    ProgressCurrent,
    ProgressEvent,
    ProgressLog,
    default_progress_path,
    load_progress_log,
    scan_plan_metadata,
    utc_now_iso,
    write_progress_log,
)
from robo_collector.gesture_plan import gesture_plan_from_payload


class GestureMetadataTest(unittest.TestCase):
    def test_matching_metadata_row_marks_trial_success(self):
        plan = gesture_plan_from_payload(_plan_payload(target_trials=1))
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_episodes(
                root,
                [
                    {
                        "episode_id": _episode_id(plan.plan_id, "shake_hand", 0, 1),
                        "tasks": ["Shake hand with somebody"],
                        "length": 12,
                        "episode_index": 99,
                    }
                ],
            )

            snapshot = scan_plan_metadata(root, plan)
            status = snapshot.status_for(plan.planned_trials[0])

            self.assertTrue(status.complete)
            self.assertEqual(snapshot.completed_count, 1)

    def test_missing_empty_mismatch_and_duplicate_do_not_mark_success(self):
        plan = gesture_plan_from_payload(_plan_payload(target_trials=1))
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_episodes(root, [])
            snapshot = scan_plan_metadata(root, plan)
            self.assertEqual(snapshot.status_for(plan.planned_trials[0]).latest_state, "MISSING")

            _write_episodes(
                root,
                [
                    {
                        "episode_id": _episode_id(plan.plan_id, "shake_hand", 0, 1),
                        "tasks": ["Shake hand with somebody"],
                        "length": 0,
                    }
                ],
            )
            snapshot = scan_plan_metadata(root, plan)
            self.assertEqual(snapshot.status_for(plan.planned_trials[0]).latest_state, "EMPTY")

            _write_episodes(
                root,
                [
                    {
                        "episode_id": _episode_id(plan.plan_id, "shake_hand", 0, 1),
                        "tasks": ["Shake hand with somebody"],
                        "length": 1.5,
                    }
                ],
            )
            snapshot = scan_plan_metadata(root, plan)
            self.assertEqual(
                snapshot.status_for(plan.planned_trials[0]).latest_state,
                "EMPTY",
            )

            _write_episodes(
                root,
                [
                    {
                        "episode_id": _episode_id(plan.plan_id, "shake_hand", 0, 1),
                        "tasks": ["Wave hello"],
                        "length": 5,
                    }
                ],
            )
            snapshot = scan_plan_metadata(root, plan)
            self.assertEqual(
                snapshot.status_for(plan.planned_trials[0]).latest_state,
                "TASK_MISMATCH",
            )

            _write_episodes(
                root,
                [
                    {
                        "episode_id": _episode_id(plan.plan_id, "shake_hand", 0, 1),
                        "tasks": ["Shake hand with somebody"],
                        "length": 5,
                    },
                    {
                        "episode_id": _episode_id(plan.plan_id, "shake_hand", 0, 1),
                        "tasks": ["Shake hand with somebody"],
                        "length": 6,
                    },
                ],
            )
            snapshot = scan_plan_metadata(root, plan)
            self.assertTrue(snapshot.status_for(plan.planned_trials[0]).ambiguous)

    def test_restart_recovery_uses_metadata_not_progress_log(self):
        plan = gesture_plan_from_payload(_plan_payload(target_trials=3))
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_episodes(
                root,
                [
                    {
                        "episode_id": _episode_id(plan.plan_id, "shake_hand", 0, 1),
                        "tasks": ["Shake hand with somebody"],
                        "length": 8,
                        "episode_index": 17,
                    },
                    {
                        "episode_id": _episode_id(plan.plan_id, "shake_hand", 1, 1),
                        "tasks": ["Shake hand with somebody"],
                        "length": 9,
                        "episode_index": 42,
                    },
                ],
            )
            progress = ProgressLog(
                schema_version=1,
                plan_id=plan.plan_id,
                dataset_root=str(root),
                updated_at=utc_now_iso(),
                last_state="WAITING_SAVE_METADATA",
                current=ProgressCurrent(
                    task_slug="shake_hand",
                    trial_index=2,
                    attempt_index=1,
                    episode_id=_episode_id(plan.plan_id, "shake_hand", 2, 1),
                ),
                events=(
                    ProgressEvent(
                        ts=utc_now_iso(),
                        event="START_SENT",
                        episode_id=_episode_id(plan.plan_id, "shake_hand", 2, 1),
                    ),
                ),
            )
            write_progress_log(default_progress_path(root), progress)

            snapshot = scan_plan_metadata(root, plan)

            self.assertEqual(snapshot.completed_count, 2)
            self.assertEqual(snapshot.status_for(plan.planned_trials[2]).next_attempt_index, 1)

    def test_progress_log_is_tolerant_to_corruption(self):
        with TemporaryDirectory() as tmp:
            path = default_progress_path(tmp)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{broken", encoding="utf-8")

            self.assertIsNone(load_progress_log(path))

    def test_progress_log_is_tolerant_to_malformed_valid_json(self):
        with TemporaryDirectory() as tmp:
            path = default_progress_path(tmp)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                (
                    '{'
                    '"schema_version": "bad", '
                    '"current": {"trial_index": "bad", "attempt_index": 1}, '
                    '"events": [{"details": []}]'
                    '}'
                ),
                encoding="utf-8",
            )

            self.assertIsNone(load_progress_log(path))

    def test_missing_or_empty_media_does_not_mark_trial_success(self):
        plan = gesture_plan_from_payload(_plan_payload(target_trials=1))
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_episodes(
                root,
                [
                    {
                        "episode_id": _episode_id(plan.plan_id, "shake_hand", 0, 1),
                        "tasks": ["Shake hand with somebody"],
                        "length": 12,
                        "data_path": "data/missing.parquet",
                        "video_paths": {
                            "observation.images.head": "videos/missing.mp4"
                        },
                    }
                ],
                create_media=False,
            )

            status = scan_plan_metadata(root, plan).status_for(
                plan.planned_trials[0]
            )

            self.assertFalse(status.complete)
            self.assertEqual(status.latest_state, "INCOMPLETE_MEDIA")

    def test_parse_error_is_reported(self):
        plan = gesture_plan_from_payload(_plan_payload(target_trials=1))
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            episodes_path = root / "meta/episodes.jsonl"
            episodes_path.parent.mkdir(parents=True, exist_ok=True)
            episodes_path.write_text("{not json}\n", encoding="utf-8")

            snapshot = scan_plan_metadata(root, plan)

            self.assertIn("invalid JSONL", snapshot.parse_error)


def _write_episodes(
    root: Path, rows: list[dict], *, create_media: bool = True
) -> None:
    for index, row in enumerate(rows):
        try:
            has_frames = int(row.get("length", 0)) > 0
        except (TypeError, ValueError):
            has_frames = False
        if not create_media or not has_frames:
            continue
        data_path = str(row.setdefault("data_path", f"data/episode_{index}.parquet"))
        video_paths = row.setdefault(
            "video_paths",
            {
                "observation.images.head": f"videos/head/episode_{index}.mp4",
                "observation.images.ego_view": f"videos/ego/episode_{index}.mp4",
            },
        )
        _write_nonempty(root / data_path)
        for video_path in video_paths.values():
            _write_nonempty(root / str(video_path))

    episodes_path = root / "meta/episodes.jsonl"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    episodes_path.write_text(
        "".join(_json_line(row) for row in rows),
        encoding="utf-8",
    )


def _write_nonempty(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test")


def _json_line(row: dict) -> str:
    import json

    return json.dumps(row, sort_keys=True) + "\n"


def _episode_id(plan_id: str, task_slug: str, trial_index: int, attempt_index: int) -> str:
    return build_gesture_episode_id(
        plan_id=plan_id,
        task_slug=task_slug,
        trial_index=trial_index,
        attempt_index=attempt_index,
    )


def _plan_payload(*, target_trials: int) -> dict:
    return {
        "version": 1,
        "plan_id": "handshake_set_a_20260621",
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
                "target_trials": target_trials,
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
