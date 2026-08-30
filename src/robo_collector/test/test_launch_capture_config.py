import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LAUNCH = ROOT / "scripts" / "launch_data_collection.sh"


class LaunchCaptureConfigTest(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(LAUNCH), *args, "--print-collector-command"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_default_is_raw_v1_without_legacy_rate_assumption(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("recording_mode:=raw_v1", result.stdout)
        self.assertIn("reference_camera_stream:=head", result.stdout)
        self.assertNotIn("camera_stream_rates_hz:=", result.stdout)
        self.assertNotIn("fps:=", result.stdout)

    def test_explicit_rates_reference_and_dual_write_are_forwarded(self):
        result = self._run(
            "--recording-mode",
            "dual_write",
            "--reference-camera-stream",
            "ego_view",
            "--camera-stream-rate",
            "head=30",
            "--camera-stream-rate",
            "ego_view=15.5",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("recording_mode:=dual_write", result.stdout)
        self.assertIn("reference_camera_stream:=ego_view", result.stdout)
        self.assertIn(
            r"camera_stream_rates_hz:=head=30\,ego_view=15.5", result.stdout
        )

    def test_raw_first_alias_warns_and_legacy_fps_is_opt_in(self):
        result = self._run("--recording-mode", "raw_first", "--fps", "24")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("recording_mode:=raw_first", result.stdout)
        self.assertIn("fps:=24", result.stdout)
        self.assertIn("raw_first' is deprecated", result.stderr)
        self.assertIn("--fps is a legacy fallback", result.stderr)

    def test_invalid_mode_and_duplicate_rate_fail_before_launch(self):
        invalid_mode = self._run("--recording-mode", "legacy")
        self.assertEqual(invalid_mode.returncode, 2)
        self.assertIn("Invalid --recording-mode", invalid_mode.stderr)

        duplicate = self._run(
            "--camera-stream-rate",
            "head=30",
            "--camera-stream-rate",
            "head=60",
        )
        self.assertEqual(duplicate.returncode, 2)
        self.assertIn("Duplicate --camera-stream-rate", duplicate.stderr)

        partial = self._run("--camera-stream-rate", "head=30")
        self.assertEqual(partial.returncode, 2)
        self.assertIn("Missing --camera-stream-rate", partial.stderr)


if __name__ == "__main__":
    unittest.main()
