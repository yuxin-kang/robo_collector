import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from robo_collector.mcap_tool import main


ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = ROOT / "src" / "robo_collector"


class McapToolTest(unittest.TestCase):
    def test_info_missing_file_is_machine_readable_failure(self):
        result = subprocess.run(
            [sys.executable, "-m", "robo_collector.mcap_tool", "info", "/no/such.mcap"],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(PACKAGE_ROOT)},
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(json.loads(result.stdout)["ok"])

    def test_parser_rejects_missing_recover_attempt(self):
        result = subprocess.run(
            [sys.executable, "-m", "robo_collector.mcap_tool", "recover", tempfile.gettempdir()],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(PACKAGE_ROOT)},
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--attempt", result.stderr)

    def test_replay_and_migration_require_ready_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory, "manifest.json")
            manifest.write_text('{"status":"MATERIALIZING"}', encoding="utf-8")
            self.assertEqual(main(["replay", str(manifest)]), 1)
            self.assertEqual(main(["migration", str(manifest)]), 1)

    def test_benchmark_reports_deterministic_input_fields(self):
        with tempfile.NamedTemporaryFile() as source:
            source.write(b"mcap")
            source.flush()
            self.assertEqual(main(["benchmark", source.name, "--iterations", "2"]), 0)

    def test_documented_replay_and_raw_root_flags_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory, "manifest.json")
            manifest.write_text('{"status":"REVIEW"}', encoding="utf-8")
            self.assertEqual(main(["replay", str(manifest), "--format", "json"]), 1)
            self.assertEqual(main(["migration", "--raw-root", directory]), 1)


if __name__ == "__main__":
    unittest.main()
