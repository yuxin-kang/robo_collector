import threading
import time
import unittest

from robo_collector.save_worker import EpisodeSaveWorker


class EpisodeSaveWorkerTest(unittest.TestCase):
    def test_save_runs_in_background_and_reports_progress(self):
        release = threading.Event()
        started = threading.Event()
        worker = EpisodeSaveWorker[str]()

        def save(report_progress):
            report_progress("writing_parquet")
            started.set()
            release.wait(timeout=1.0)
            return "saved"

        try:
            worker.start(save)

            self.assertTrue(started.wait(timeout=1.0))
            self.assertTrue(worker.has_active)
            self.assertFalse(worker.done)
            self.assertEqual(
                [progress.phase for progress in worker.drain_progress()],
                ["writing_parquet"],
            )

            release.set()
            self.assertEqual(worker.take_result(timeout=1.0), "saved")
            self.assertFalse(worker.has_active)
        finally:
            release.set()
            worker.shutdown()

    def test_overlapping_save_is_rejected(self):
        release = threading.Event()
        started = threading.Event()
        worker = EpisodeSaveWorker[str]()

        def save(report_progress):
            started.set()
            release.wait(timeout=1.0)
            return "saved"

        try:
            worker.start(save)
            self.assertTrue(started.wait(timeout=1.0))

            with self.assertRaisesRegex(RuntimeError, "already running"):
                worker.start(save)

            release.set()
            self.assertEqual(worker.take_result(timeout=1.0), "saved")
        finally:
            release.set()
            worker.shutdown()

    def test_failed_save_is_consumed_before_next_save(self):
        worker = EpisodeSaveWorker[str]()

        def fail(report_progress):
            raise RuntimeError("disk failure")

        try:
            worker.start(fail)
            with self.assertRaisesRegex(RuntimeError, "disk failure"):
                worker.take_result(timeout=1.0)
            self.assertFalse(worker.has_active)

            worker.start(lambda report_progress: "recovered")
            self.assertEqual(worker.take_result(timeout=1.0), "recovered")
        finally:
            worker.shutdown()

    def test_shutdown_timeout_does_not_block_on_stuck_save(self):
        release = threading.Event()
        started = threading.Event()
        worker = EpisodeSaveWorker[str]()

        def save(report_progress):
            started.set()
            release.wait()
            return "saved"

        try:
            worker.start(save)
            self.assertTrue(started.wait(timeout=1.0))

            before = time.monotonic()
            self.assertFalse(worker.shutdown(timeout=0.01))
            elapsed = time.monotonic() - before

            self.assertLess(elapsed, 0.25)
            self.assertTrue(worker.has_active)
        finally:
            release.set()
            worker.shutdown(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
