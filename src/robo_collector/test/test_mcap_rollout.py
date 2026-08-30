import json
import subprocess
import sys
import unittest

from robo_collector.mcap_rollout import (
    ArtifactSummary,
    compare_parity,
    run_harness,
    run_synthetic_harness,
)


def _summary(**overrides):
    value = {
        "episode_id": "episode-1",
        "message_count": 3,
        "source_hash": "abc",
        "timestamps": [10, 20, 30],
        "aligned_rows": 2,
    }
    value.update(overrides)
    return value


class McapRolloutTest(unittest.TestCase):
    def test_parity_passes_when_all_acceptance_evidence_matches(self):
        report = compare_parity(_summary(), _summary())
        self.assertEqual(report.status, "PASS")
        self.assertTrue(all(report.checks.values()))

    def test_parity_tolerates_timestamp_jitter_but_rejects_count_mismatch(self):
        report = compare_parity(_summary(), _summary(timestamps=[11, 20, 29]), timestamp_tolerance_ns=1)
        self.assertEqual(report.status, "PASS")
        report = compare_parity(_summary(), _summary(message_count=4))
        self.assertEqual(report.status, "FAIL")
        self.assertEqual(report.mismatches, ("message_count",))

    def test_parity_marks_missing_optional_evidence_for_review(self):
        report = compare_parity(ArtifactSummary.from_mapping(_summary(aligned_rows=None)), _summary(aligned_rows=None))
        self.assertEqual(report.status, "REVIEW")
        self.assertFalse(report.not_run)

    def test_parity_marks_missing_hash_for_review(self):
        report = compare_parity(_summary(source_hash=None), _summary(source_hash=None))
        self.assertEqual(report.status, "REVIEW")

    def test_synthetic_harness_is_explicitly_not_real_hardware(self):
        for scenario in ("kill", "reconnect", "soak"):
            with self.subTest(scenario=scenario):
                result = run_harness(scenario, iterations=2)
                self.assertEqual(result.status, "PASS")
                self.assertTrue(result.synthetic)
                self.assertTrue(result.not_run)
                self.assertIn("hardware", result.message)

    def test_synthetic_harness_compatibility_name(self):
        result = run_synthetic_harness("reconnect", iterations=2)
        self.assertEqual(result, run_harness("reconnect", iterations=2))

    def test_cli_emits_machine_readable_report(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            raw = f"{directory}/raw.json"
            mcap = f"{directory}/mcap.json"
            with open(raw, "w", encoding="utf-8") as stream:
                json.dump(_summary(), stream)
            with open(mcap, "w", encoding="utf-8") as stream:
                json.dump(_summary(), stream)
            result = subprocess.run(
                [sys.executable, "-m", "robo_collector.mcap_rollout", "--raw", raw, "--mcap", mcap],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0)
            self.assertEqual(json.loads(result.stdout)["status"], "PASS")

    def test_cli_rejects_invalid_harness_arguments(self):
        result = subprocess.run(
            [sys.executable, "-m", "robo_collector.mcap_rollout", "--scenario", "kill", "--iterations", "0"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("iterations must be positive", result.stderr)


if __name__ == "__main__":
    unittest.main()
