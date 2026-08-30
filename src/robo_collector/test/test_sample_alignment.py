import random
import unittest
from types import SimpleNamespace

from robo_collector.field_config import FieldSelection, default_field_selection
from robo_collector.sample_alignment import (
    AlignmentConfig,
    AlignmentError,
    ClockNormalizationConfig,
    ClockRecord,
    align_rgb_records,
    message_stamp_sec,
    normalize_clock_records,
    required_sample_inputs,
    selected_missing_inputs,
    selected_source_timestamps_sec,
    source_timestamp_skew_sec,
)


def _record(
    stream,
    sequence,
    timestamp,
    *,
    receive=None,
    source=None,
    session="s",
    collector=None,
    payload=None,
):
    return ClockRecord(
        stream_id=stream,
        source_session_id=session,
        source_sequence=sequence,
        collector_record_id=sequence if collector is None else collector,
        receive_time_ns=timestamp if receive is None else receive,
        source_time_ns=timestamp if source is None else source,
        normalized_time_ns=timestamp,
        payload=payload,
    )


class SampleAlignmentTest(unittest.TestCase):
    def test_default_selection_requires_action_robot_state_and_imu(self):
        required = required_sample_inputs(default_field_selection())

        self.assertIn("action", required)
        self.assertIn("target_joint_pos", required)
        self.assertIn("joint_states", required)
        self.assertIn("imu", required)

    def test_selected_missing_inputs_only_returns_persisted_fields(self):
        selection = FieldSelection(
            target=("aligned_target_pos",),
            state=("relative_ori_6d",),
        )

        self.assertEqual(
            selected_missing_inputs(
                ["action", "target_joint_pos", "aligned_target_pos"],
                selection,
            ),
            ["aligned_target_pos"],
        )

    def test_source_timestamp_skew_requires_all_finite_timestamps(self):
        self.assertAlmostEqual(
            source_timestamp_skew_sec(10.0, [10.02, 9.98]),
            0.04,
        )
        self.assertIsNone(source_timestamp_skew_sec(None, [10.0]))
        self.assertIsNone(source_timestamp_skew_sec(10.0, [float("nan")]))

    def test_selected_source_timestamps_include_selected_action(self):
        selection = default_field_selection()
        msg = SimpleNamespace(
            source_timestamp_names=[
                "joint_states",
                "imu",
                "target_joint_pos",
                "action",
            ],
            source_timestamps_sec=[10.0, 10.01, 10.02, 9.8],
        )

        timestamps = selected_source_timestamps_sec(msg, selection)

        self.assertIsNotNone(timestamps)
        self.assertIn("action", timestamps)
        self.assertAlmostEqual(
            source_timestamp_skew_sec(
                next(iter(timestamps.values())),
                [*list(timestamps.values())[1:], 10.0],
            ),
            0.22,
        )

    def test_selected_source_timestamps_reject_missing_selected_input(self):
        selection = FieldSelection(
            target=("aligned_target_pos",),
            state=("relative_ori_6d",),
        )
        msg = SimpleNamespace(
            source_timestamp_names=["aligned_target_pos"],
            source_timestamps_sec=[10.0],
        )

        self.assertIsNone(selected_source_timestamps_sec(msg, selection))

    def test_message_stamp_rejects_missing_or_zero_stamp(self):
        valid = SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(sec=10, nanosec=20))
        )
        zero = SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0))
        )

        self.assertAlmostEqual(message_stamp_sec(valid), 10.00000002)
        self.assertIsNone(message_stamp_sec(zero))
        self.assertIsNone(message_stamp_sec(SimpleNamespace()))


class ClockNormalizationTest(unittest.TestCase):
    def test_exact_affine_fit_and_session_splits_are_deterministic(self):
        records = [
            _record("rgb", i, 0, source=i * 10_000_000, receive=i * 10_000_000 + 7)
            for i in range(31)
        ]
        records += [
            _record(
                "rgb", 0, 0, session="reconnect", source=5, receive=11, collector=100
            ),
            _record(
                "rgb", 1, 0, session="reconnect", source=15, receive=21, collector=101
            ),
        ]

        result = normalize_clock_records(records)

        self.assertEqual(result.segments[0].normalization_mode, "AFFINE_V2")
        self.assertEqual(result.segments[0].slope_numerator, 1)
        self.assertEqual(result.segments[0].offset_numerator, 7)
        self.assertEqual(result.records[9].normalized_time_ns, 90_000_007)
        self.assertEqual(result.segments[1].clock_session_id, "reconnect.0")
        self.assertEqual(result.segments[1].fallback_reason, "INSUFFICIENT_VALID_EDGES")
        self.assertFalse(result.ready_eligible)

    def test_missing_source_forces_whole_segment_receive_fallback(self):
        records = [
            _record(
                "rgb",
                i,
                0,
                source=None if i == 15 else i * 10_000_000,
                receive=i * 10_000_000,
            )
            for i in range(31)
        ]
        # The fixture helper's default cannot distinguish omitted from explicit None.
        records[15] = ClockRecord("rgb", "s", 15, 15, 150_000_000, None)

        result = normalize_clock_records(records)

        self.assertTrue(result.ready_eligible)
        self.assertEqual(result.segments[0].fallback_reason, "SOURCE_TIME_MISSING")
        self.assertTrue(
            all(record.normalization_mode == "RECEIVE_FALLBACK" for record in result.records)
        )

    def test_uncertainty_boundary_is_inclusive_and_overflow_rejected(self):
        # 31 valid affine edges, with endpoint residual spread exactly at the bound.
        records = [
            _record("rgb", i, 0, source=i * 100_000_000, receive=i * 100_000_000)
            for i in range(31)
        ]
        exact = normalize_clock_records(
            records,
            ClockNormalizationConfig(max_uncertainty_ns=0),
        )
        self.assertEqual(exact.segments[0].normalization_mode, "AFFINE_V2")
        with self.assertRaises(AlignmentError):
            normalize_clock_records(
                [ClockRecord("rgb", "s", 0, 0, 1 << 63, 0)]
            )

    def test_sequence_and_clock_rollbacks_split_sessions(self):
        records = [
            _record("rgb", 4, 0, source=4_000_000, receive=4_000_000, collector=1),
            _record("rgb", 3, 0, source=3_000_000, receive=5_000_000, collector=2),
            _record("rgb", 4, 0, source=1_000_000, receive=6_000_000, collector=3),
        ]
        result = normalize_clock_records(records)
        self.assertEqual([segment.clock_session_id for segment in result.segments], ["s.0", "s.1", "s.2"])

    def test_interleaved_streams_normalize_as_independent_clock_sources(self):
        records = []
        for sequence in range(31):
            records.extend(
                (
                    _record(
                        "left",
                        sequence,
                        0,
                        source=sequence * 10,
                        receive=sequence * 10 + 1,
                        collector=sequence * 2,
                    ),
                    _record(
                        "right",
                        sequence,
                        0,
                        source=sequence * 20,
                        receive=sequence * 20 + 2,
                        collector=sequence * 2 + 1,
                    ),
                )
            )
        result = normalize_clock_records(records)
        self.assertEqual(len(result.segments), 2)
        self.assertTrue(all(segment.normalization_mode == "AFFINE_V2" for segment in result.segments))
        self.assertEqual([record.collector_record_id for record in result.records], list(range(62)))


class RgbAnchoredAlignmentTest(unittest.TestCase):
    def test_explicit_reference_and_independent_rates(self):
        cameras = [_record("left", i, i * 50, collector=100 + i) for i in range(3)]
        cameras += [_record("right", i, i * 100, collector=200 + i) for i in range(2)]
        states = [_record("state", i, i * 25, collector=300 + i) for i in range(5)]
        actions = [_record("action", i, i * 50, collector=400 + i) for i in range(3)]
        result = align_rgb_records(
            cameras,
            states,
            actions,
            AlignmentConfig(
                reference_camera_stream="left",
                max_camera_residual_ns=50,
                max_state_residual_ns=25,
                action_max_age_ns=50,
                configured_rates_hz=(("left", 15.0), ("right", 60.0), ("state", 50.0)),
            ),
        )
        self.assertEqual([row.reference.source_sequence for row in result.rows], [0, 1, 2])
        self.assertEqual(dict(result.configured_rates_hz)["left"], 15.0)
        self.assertNotEqual(dict(result.observed_rates_hz)["left"], 15.0)
        self.assertEqual(len(result.dense_state_records), 5)
        with self.assertRaises(AlignmentError):
            align_rgb_records(cameras, states, actions, AlignmentConfig("missing"))

    def test_v2_nearest_ties_prefer_past_then_sequence_and_collector(self):
        cameras = [_record("rgb", 1, 100, collector=1)]
        states = [
            _record("state", 8, 90, collector=8, session="z"),
            _record("state", 9, 90, collector=2, session="y"),
            _record("state", 9, 90, collector=3, session="x"),
            _record("state", 99, 110, collector=99),
        ]
        actions = [_record("action", 1, 100)]
        result = align_rgb_records(cameras, states, actions, AlignmentConfig("rgb", max_state_residual_ns=10))
        self.assertEqual(result.rows[0].state.collector_record_id, 3)

    def test_legacy_exact_tie_preserves_future_choice(self):
        cameras = [_record("rgb", 1, 100)]
        states = [_record("state", 1, 90), _record("state", 2, 110)]
        actions = [_record("action", 1, 100)]
        result = align_rgb_records(
            cameras,
            states,
            actions,
            AlignmentConfig("rgb", max_state_residual_ns=10, policy="legacy_rgb_v1"),
        )
        self.assertEqual(result.rows[0].state.normalized_time_ns, 110)
        implicit = align_rgb_records(
            cameras,
            states,
            actions,
            AlignmentConfig("", max_state_residual_ns=10, policy="legacy_rgb_v1"),
        )
        self.assertEqual(implicit.rows[0].reference.stream_id, "rgb")

    def test_action_is_causal_and_inclusive_bounds_create_gap_evidence(self):
        cameras = [_record("rgb", 1, 100), _record("rgb", 2, 121)]
        states = [_record("state", 1, 80), _record("state", 2, 101)]
        actions = [_record("action", 1, 80), _record("action", 2, 101)]
        result = align_rgb_records(
            cameras,
            states,
            actions,
            AlignmentConfig("rgb", max_state_residual_ns=20, action_max_age_ns=20),
        )
        self.assertEqual(result.rows[0].action.normalized_time_ns, 80)
        self.assertEqual(result.rows[1].action.normalized_time_ns, 101)
        future_only = align_rgb_records(
            [_record("rgb", 1, 100)],
            [_record("state", 1, 100)],
            [_record("action", 1, 101)],
            AlignmentConfig("rgb"),
        )
        self.assertEqual(future_only.rows, ())
        self.assertEqual(future_only.gaps[0].missing, ("action",))

    def test_duplicates_are_counted_and_divergent_duplicates_quarantine(self):
        camera = _record("rgb", 1, 100, payload=b"same")
        retransmit = _record("rgb", 1, 100, collector=99, payload=b"same")
        result = align_rgb_records(
            [camera, retransmit],
            [_record("state", 1, 100)],
            [_record("action", 1, 100)],
            AlignmentConfig("rgb"),
        )
        self.assertEqual(result.duplicate_count, 1)
        self.assertEqual(len(result.dense_camera_records), 2)
        self.assertEqual(result.rows[0].reference.collector_record_id, 99)
        with self.assertRaises(AlignmentError):
            align_rgb_records(
                [camera, _record("rgb", 1, 100, payload=b"different")],
                [_record("state", 1, 100)],
                [_record("action", 1, 100)],
                AlignmentConfig("rgb"),
            )

    def test_randomized_candidate_order_cannot_change_total_key_selection(self):
        camera = [_record("rgb", 1, 100, collector=1)]
        states = [
            _record("state", 1, 90, collector=2, session="z"),
            _record("state", 2, 90, collector=3, session="y"),
            _record("state", 2, 90, collector=4, session="x"),
            _record("state", 99, 110, collector=5, session="a"),
        ]
        action = [_record("action", 1, 100, collector=6)]
        for seed in range(25):
            shuffled = states[:]
            random.Random(seed).shuffle(shuffled)
            result = align_rgb_records(camera, shuffled, action, AlignmentConfig("rgb"))
            self.assertEqual(result.rows[0].state.collector_record_id, 4)


if __name__ == "__main__":
    unittest.main()
