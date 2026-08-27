"""ROS2 node that records validated RoboState samples into LeRobot episodes."""

from __future__ import annotations

import math
import json
import hashlib
import os
import subprocess
import threading
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import rclpy
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from robo_collector_msgs.msg import RecordCommand
from robo_state_msgs.msg import RoboStateSample

from .camera_cache import CameraFrameCache, parse_camera_streams
from .collector_state import (
    CollectorMode,
    CommandFingerprint,
    CommandReceiptLedger,
    RecordStateMachine,
    recording_safety_reason,
)
from .field_config import (
    FieldConfigError,
    default_field_selection,
    load_optional_field_selection,
)
from .lerobot_dataset import LeRobotV21Writer, RobotFrame, SaveResult
from .episode_quality import EpisodeQualityGate, write_quality_report
from .raw_episode import (
    RawEpisodeRecorder,
    _hash_manifest,
    create_materialization_job,
    discard_sealed_episode,
    scan_startup,
    update_materialization_job,
)
from .raw_materializer import MaterializationConfig, MaterializationResult, RawEpisodeMaterializer
from .save_worker import EpisodeSaveWorker
from .sample_alignment import (
    message_stamp_sec,
    selected_missing_inputs,
    selected_source_timestamps_sec,
    source_timestamp_skew_sec,
)

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX deployment
    resource = None

try:
    from robo_collector_camera.raw_spool import read_camera_spool_snapshot
except ImportError:  # pragma: no cover - camera package is a deployment dependency
    read_camera_spool_snapshot = None


@dataclass(frozen=True)
class CachedStateSample:
    msg: RoboStateSample
    received_monotonic_sec: float


class LeRobotCollectorNode(Node):
    """Waits for START/STOP commands and records aligned state + RGB frames."""

    def __init__(self) -> None:
        super().__init__("lerobot_collector_node")

        self.declare_parameter("robo_state_topic", "/robo_state/sample")
        self.declare_parameter("record_command_topic", "/robo_collector/record_command")
        self.declare_parameter("status_topic", "/robo_collector/status")
        self.declare_parameter("camera_host", "192.168.123.164")
        self.declare_parameter("camera_port", 5555)
        self.declare_parameter("camera_stream", "")
        self.declare_parameter("camera_streams", "head,ego_view")
        self.declare_parameter("dataset_name", "")
        self.declare_parameter("root_output_dir", "outputs")
        self.declare_parameter("field_config_path", "")
        self.declare_parameter("fps", 30)
        self.declare_parameter("max_state_age_sec", 0.2)
        self.declare_parameter("max_camera_age_sec", 0.2)
        self.declare_parameter("max_inter_camera_skew_sec", 0.1)
        self.declare_parameter("max_state_camera_skew_sec", 0.1)
        self.declare_parameter("max_camera_clock_mapping_uncertainty_sec", 0.05)
        self.declare_parameter("max_producer_gaps", 0)
        self.declare_parameter("max_publisher_gaps", 0)
        self.declare_parameter("max_transport_gaps", 0)
        self.declare_parameter("max_unattributed_gaps", 0)
        self.declare_parameter("max_selection_gaps", 0)
        self.declare_parameter("max_duplicate_count", 0)
        self.declare_parameter("max_reorder_count", 0)
        self.declare_parameter("max_session_restart_count", 0)
        self.declare_parameter("max_timestamp_anomaly_count", 0)
        self.declare_parameter("max_episode_duration_sec", 600.0)
        self.declare_parameter("max_episode_frames", 18000)
        self.declare_parameter("min_free_disk_bytes", 2147483648)
        self.declare_parameter("save_shutdown_grace_sec", 10.0)
        # Raw-first is the safe final-spec default.  ``legacy`` remains an
        # explicit migration/comparison mode and must not be treated as the
        # authoritative publication path.
        self.declare_parameter("recording_mode", "raw_first")
        self.declare_parameter("raw_recording_enabled", True)
        self.declare_parameter("raw_episode_root", "")
        self.declare_parameter("raw_source_scope", "transport_observed")
        self.declare_parameter("raw_max_records_per_chunk", 256)
        self.declare_parameter("raw_max_record_bytes", 16777216)
        self.declare_parameter("camera_callback_queue_size", 128)
        self.declare_parameter("camera_raw_spool_root", "")

        self._fps = int(self.get_parameter("fps").value)
        self._robo_state_topic = str(self.get_parameter("robo_state_topic").value)
        self._max_state_age_sec = float(
            self.get_parameter("max_state_age_sec").value
        )
        self._max_camera_age_sec = float(
            self.get_parameter("max_camera_age_sec").value
        )
        self._max_inter_camera_skew_sec = float(
            self.get_parameter("max_inter_camera_skew_sec").value
        )
        self._max_state_camera_skew_sec = float(
            self.get_parameter("max_state_camera_skew_sec").value
        )
        self._max_camera_clock_mapping_uncertainty_sec = _optional_configured_float(
            self.get_parameter("max_camera_clock_mapping_uncertainty_sec").value
        )
        self._quality_thresholds = {
            name: _configured_non_negative_int(
                self.get_parameter(name).value, name
            )
            for name in (
                "max_producer_gaps",
                "max_publisher_gaps",
                "max_transport_gaps",
                "max_unattributed_gaps",
                "max_selection_gaps",
                "max_duplicate_count",
                "max_reorder_count",
                "max_session_restart_count",
                "max_timestamp_anomaly_count",
            )
        }
        self._quality_thresholds.update({
            "max_camera_camera_skew_sec": self._max_inter_camera_skew_sec,
            "max_state_camera_skew_sec": self._max_state_camera_skew_sec,
        })
        self._max_episode_duration_sec = float(
            self.get_parameter("max_episode_duration_sec").value
        )
        self._max_episode_frames = int(
            self.get_parameter("max_episode_frames").value
        )
        self._min_free_disk_bytes = int(
            self.get_parameter("min_free_disk_bytes").value
        )
        self._save_shutdown_grace_sec = float(
            self.get_parameter("save_shutdown_grace_sec").value
        )
        self._recording_mode = str(self.get_parameter("recording_mode").value).strip().lower()
        if self._recording_mode not in {"legacy", "raw_first"}:
            raise RuntimeError("recording_mode must be 'legacy' or 'raw_first'")
        self._raw_recording_enabled = bool(
            self.get_parameter("raw_recording_enabled").value
        )
        if self._recording_mode == "raw_first" and not self._raw_recording_enabled:
            raise RuntimeError("raw_first recording_mode requires raw_recording_enabled")
        raw_root_parameter = str(self.get_parameter("raw_episode_root").value).strip()
        output_root_parameter = Path(
            str(self.get_parameter("root_output_dir").value)
        ).expanduser()
        self._raw_episode_root = (
            Path(raw_root_parameter).expanduser()
            if raw_root_parameter
            else output_root_parameter / ".raw_episodes"
        )
        camera_raw_spool_parameter = str(
            self.get_parameter("camera_raw_spool_root").value
        ).strip()
        self._camera_raw_spool_root = (
            Path(camera_raw_spool_parameter).expanduser().resolve()
            if camera_raw_spool_parameter
            else None
        )
        self._raw_source_scope = str(
            self.get_parameter("raw_source_scope").value
        ).strip()
        self._raw_max_records_per_chunk = int(
            self.get_parameter("raw_max_records_per_chunk").value
        )
        self._raw_max_record_bytes = int(
            self.get_parameter("raw_max_record_bytes").value
        )
        self._camera_callback_queue_size = int(
            self.get_parameter("camera_callback_queue_size").value
        )
        for name, value in (
            ("fps", self._fps),
            ("max_state_age_sec", self._max_state_age_sec),
            ("max_camera_age_sec", self._max_camera_age_sec),
            ("max_inter_camera_skew_sec", self._max_inter_camera_skew_sec),
            ("max_state_camera_skew_sec", self._max_state_camera_skew_sec),
            ("max_episode_duration_sec", self._max_episode_duration_sec),
            ("save_shutdown_grace_sec", self._save_shutdown_grace_sec),
        ):
            if not math.isfinite(value) or value <= 0:
                raise RuntimeError(f"{name} must be finite and positive")
        if self._max_camera_clock_mapping_uncertainty_sec is None:
            raise RuntimeError(
                "max_camera_clock_mapping_uncertainty_sec must be finite and non-negative"
            )
        for name, value in self._quality_thresholds.items():
            if name.endswith("_sec"):
                continue
            if value < 0:
                raise RuntimeError(f"{name} must be non-negative")
        if self._max_episode_frames <= 0:
            raise RuntimeError("max_episode_frames must be positive")
        if self._min_free_disk_bytes < 0:
            raise RuntimeError("min_free_disk_bytes must be non-negative")
        if self._raw_source_scope not in {"transport_observed", "camera_capture"}:
            raise RuntimeError(
                "raw_source_scope must be transport_observed or camera_capture"
            )
        self._requested_raw_source_scope = self._raw_source_scope
        if self._raw_source_scope == "camera_capture" and (
            self._camera_raw_spool_root is None or read_camera_spool_snapshot is None
        ):
            self.get_logger().warn(
                "camera-side spool root is not configured; downgrading "
                "raw_source_scope to transport_observed/host_receiver"
            )
            self._raw_source_scope = "transport_observed"
        if self._raw_max_records_per_chunk <= 0 or self._raw_max_record_bytes <= 0:
            raise RuntimeError("raw chunk limits must be positive")
        if self._camera_callback_queue_size <= 0:
            raise RuntimeError("camera_callback_queue_size must be positive")
        legacy_camera_stream = str(self.get_parameter("camera_stream").value).strip()
        if legacy_camera_stream:
            camera_streams = [legacy_camera_stream]
        else:
            camera_streams = parse_camera_streams(
                self.get_parameter("camera_streams").value
            )
        dataset_name = str(self.get_parameter("dataset_name").value).strip() or None
        field_config_path = str(self.get_parameter("field_config_path").value).strip()
        try:
            field_selection = load_optional_field_selection(field_config_path)
        except FieldConfigError as exc:
            message = f"invalid field_config_path: {exc}"
            self.get_logger().error(message)
            raise RuntimeError(message) from exc

        self._field_selection = field_selection or default_field_selection()
        self._state_machine = RecordStateMachine()
        self._writer = LeRobotV21Writer(
            str(self.get_parameter("root_output_dir").value),
            dataset_name=dataset_name,
            fps=self._fps,
            camera_keys=[
                f"observation.images.{stream}" for stream in camera_streams
            ],
            field_selection=self._field_selection,
        )
        self._collector_git_commit = _resolve_git_commit()
        self._capture_config = {
            "schema": "robo_collector.capture_config.v1",
            "topics": {
                "robo_state": self._robo_state_topic,
                "record_command": str(
                    self.get_parameter("record_command_topic").value
                ),
                "status": str(self.get_parameter("status_topic").value),
            },
            "camera": {
                "host": str(self.get_parameter("camera_host").value),
                "port": int(self.get_parameter("camera_port").value),
                "streams": list(camera_streams),
                "callback_queue_size": self._camera_callback_queue_size,
            },
            "alignment": {
                "fps": self._fps,
                "max_state_age_sec": self._max_state_age_sec,
                "max_camera_age_sec": self._max_camera_age_sec,
                "max_inter_camera_skew_sec": self._max_inter_camera_skew_sec,
                "max_state_camera_skew_sec": self._max_state_camera_skew_sec,
                "max_camera_clock_mapping_uncertainty_sec": (
                    self._max_camera_clock_mapping_uncertainty_sec
                ),
            },
            "quality": dict(self._quality_thresholds),
            "limits": {
                "max_episode_duration_sec": self._max_episode_duration_sec,
                "max_episode_frames": self._max_episode_frames,
                "min_free_disk_bytes": self._min_free_disk_bytes,
                "save_shutdown_grace_sec": self._save_shutdown_grace_sec,
            },
            "recording": {
                "mode": self._recording_mode,
                "raw_enabled": self._raw_recording_enabled,
                "raw_root": str(self._raw_episode_root),
                "requested_source_scope": self._requested_raw_source_scope,
                "source_scope": self._raw_source_scope,
                "camera_raw_spool_root": (
                    str(self._camera_raw_spool_root)
                    if self._camera_raw_spool_root is not None
                    else ""
                ),
                "max_records_per_chunk": self._raw_max_records_per_chunk,
                "max_record_bytes": self._raw_max_record_bytes,
            },
            "dataset": {
                "output_root": str(self._writer.root_output_dir),
                "dataset_name": self._writer.dataset_name,
            },
            "field_selection": {
                "target": list(self._field_selection.target),
                "state": list(self._field_selection.state),
                "include_policy_action": self._field_selection.include_policy_action,
            },
        }
        self._collector_config_hash = _canonical_config_hash(self._capture_config)
        self._save_worker = EpisodeSaveWorker[SaveResult]()
        self._save_started_monotonic_sec: float | None = None
        self._save_finished_monotonic_sec: float | None = None
        self._save_progress_monotonic_sec: float | None = None
        self._save_progress_seq = 0
        self._save_phase = ""
        self._saving_episode_index: int | None = None
        self._saving_frame_count = 0
        self._latest_state: CachedStateSample | None = None
        self._last_warn_message = ""
        self._last_warn_monotonic_sec = 0.0
        self._last_status_log_message = ""
        self._last_status_log_monotonic_sec = 0.0
        self._last_command_id = ""
        self._last_command = ""
        self._last_command_outcome = ""
        self._last_command_episode_id = ""
        self._last_episode_id = ""
        self._last_episode_outcome = ""
        self._command_receipts = CommandReceiptLedger()
        self._state_sample_count = 0
        self._last_state_sample_log_monotonic_sec = 0.0
        self._state_camera_skew_samples: list[float] = []
        self._state_camera_skew_sample_cursor = 0
        self._state_age_samples: list[float] = []
        self._state_age_sample_cursor = 0
        self._camera_age_samples: list[float] = []
        self._camera_age_sample_cursor = 0
        self._raw_recorder: RawEpisodeRecorder | None = None
        self._raw_episode_path: Path | None = None
        self._raw_materialization_job: dict[str, Any] | None = None
        self._raw_state_sequence = 0
        self._raw_event_sequence = 0
        self._raw_session_id: str | None = None
        self._raw_capture_start_wall_time: float | None = None
        self._raw_capture_start_monotonic_time: float | None = None
        self._raw_camera_session_ids: set[str] = set()
        self._raw_camera_capture_attached = False
        self._camera_clock_offsets: dict[str, list[float]] = {}
        self._raw_camera_observed_high_watermarks: dict[
            str, dict[str, int]
        ] = {}
        self._raw_camera_packet_high_watermarks: dict[str, int] = {}
        self._raw_camera_binding_checkpoint_signature: tuple[Any, ...] | None = None
        self._raw_recording_failed_reason: str | None = None
        self._raw_camera_last_sequences: dict[str, tuple[str, int]] = {}
        self._raw_camera_last_packet_session_id: str | None = None
        self._raw_camera_last_packet_sequence: int | None = None
        self._raw_frame_count = 0
        self._record_tick_count = 0
        self._timer_deadline_misses = 0
        self._last_record_tick_monotonic_sec: float | None = None
        self._lifecycle_lock = threading.RLock()
        self._command_lock = threading.Lock()
        self._recovery_stop = threading.Event()
        self._recovery_thread: threading.Thread | None = None
        self._recovery_jobs: list[dict[str, Any]] = []
        self._recovery_active = False
        self._recovery_error: str | None = None
        self._frozen_quality: dict[str, Any] | None = None
        self._last_recorded_camera_identity: (
            tuple[str, tuple[tuple[str, int | float | None], ...]] | None
        ) = None
        self._last_stale_camera_identity: (
            tuple[str, tuple[tuple[str, int | float | None], ...]] | None
        ) = None
        self._camera_callback_overflow_at_start = 0

        qos = QoSProfile(depth=10)
        self._status_pub = self.create_publisher(
            DiagnosticStatus, str(self.get_parameter("status_topic").value), qos
        )
        self.create_subscription(
            RoboStateSample,
            self._robo_state_topic,
            self._on_state,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            RecordCommand,
            str(self.get_parameter("record_command_topic").value),
            self._on_record_command,
            qos,
        )

        self._camera_cache = CameraFrameCache(
            str(self.get_parameter("camera_host").value),
            int(self.get_parameter("camera_port").value),
            camera_streams,
            self.get_logger(),
            max_inter_camera_skew_sec=self._max_inter_camera_skew_sec,
            expected_fps=self._fps,
            receive_mode="recording",
            decode_images=self._recording_mode == "legacy",
            callback_queue_size=self._camera_callback_queue_size,
        )
        self._camera_cache.set_packet_callback(self._on_camera_packet)
        self._camera_cache.start()

        if self._raw_recording_enabled:
            try:
                pending_jobs = scan_startup(
                    self._raw_episode_root,
                    default_materialization_config=self._materialization_job_config(),
                    default_output_schema_version="lerobot.v2.1.raw_materialization.v1",
                )
                self._recovery_jobs = pending_jobs
                if pending_jobs:
                    self._recovery_active = True
                    self.get_logger().warn(
                        f"found {len(pending_jobs)} recoverable raw materialization job(s)"
                    )
                    self._recovery_thread = threading.Thread(
                        target=self._run_recovery_jobs,
                        name="robo_collector_raw_recovery",
                        daemon=True,
                    )
                    self._recovery_thread.start()
            except Exception as exc:
                self._recovery_error = f"raw episode startup scan failed: {exc}"
                # A failed scan means the durable writer state is unknown.  Do
                # not accept a new START and risk interleaving a live Episode
                # with stranded materialization work; require an operator
                # restart after the raw root has been repaired.
                self._recovery_active = True
                self.get_logger().error(self._recovery_error)

        self._record_timer = self.create_timer(1.0 / self._fps, self._record_tick)
        self._status_timer = self.create_timer(1.0, self._publish_periodic_status)
        self.get_logger().info(
            "collector ready; waiting for START on "
            f"{self.get_parameter('record_command_topic').value}; "
            f"dataset root will be {self._writer.root}; "
            f"camera streams={','.join(camera_streams)}; "
            f"field config={field_config_path or '<legacy all fields>'}"
        )
        if self._recovery_active:
            self._publish_status(
                DiagnosticStatus.ERROR,
                self._recovery_status_message(),
            )
        else:
            self._publish_status(DiagnosticStatus.OK, "IDLE: waiting for START")

    def destroy_node(self) -> bool:
        self._recovery_stop.set()
        if self._recovery_thread is not None:
            self._recovery_thread.join(timeout=2.0)
            if self._recovery_thread.is_alive():
                self.get_logger().warn(
                    "raw materialization recovery is still running after the "
                    "shutdown grace period; waiting for the active job to be reaped"
                )
                self._recovery_thread.join()
        flush_callbacks = getattr(self._camera_cache, "flush_callbacks", None)
        if callable(flush_callbacks):
            try:
                flushed = flush_callbacks(timeout=self._save_shutdown_grace_sec)
                if flushed is False:
                    self.get_logger().warn(
                        "camera callback queue is still draining after the "
                        "shutdown grace period; waiting for accepted packets"
                    )
                    flush_callbacks()
            except Exception as exc:
                self.get_logger().warn(f"camera callback flush failed during shutdown: {exc}")
                try:
                    flush_callbacks()
                except Exception as final_exc:
                    self.get_logger().error(
                        f"camera callback queue could not be drained during shutdown: {final_exc}"
                    )
        self._camera_cache.set_packet_callback(None)
        self._camera_cache.stop()
        save_finished = self._save_worker.shutdown(
            timeout=self._save_shutdown_grace_sec
        )
        if not save_finished:
            self.get_logger().warn(
                "episode save is still running after the "
                f"{self._save_shutdown_grace_sec:.3f}s shutdown grace period; "
                "waiting for the durable transaction to finish"
            )
            self._save_worker.shutdown()
        if self._state_machine.mode == CollectorMode.SAVING:
            self._poll_save_episode()
        if self._writer.active_episode_index is not None:
            try:
                self._writer.discard_episode()
            except Exception as exc:
                self.get_logger().error(
                    f"failed to discard active episode during shutdown: {exc}"
                )
        if self._raw_recorder is not None:
            try:
                self._freeze_raw_capture(command="SHUTDOWN", reason="process_crash")
            except Exception as exc:
                self.get_logger().error(
                    f"failed to seal raw episode during shutdown: {exc}"
                )
        return super().destroy_node()

    def _on_state(self, msg: RoboStateSample) -> None:
        now = time.monotonic()
        was_unavailable_or_stale = self._latest_state is None or (
            now - self._latest_state.received_monotonic_sec > self._max_state_age_sec
        )
        self._latest_state = CachedStateSample(msg=msg, received_monotonic_sec=now)
        self._state_sample_count += 1
        with self._lifecycle_lock:
            recorder = self._raw_recorder
            recording_raw = (
                self._state_machine.mode == CollectorMode.RECORDING
                and recorder is not None
                and not recorder.closed
            )
            capture_start = self._raw_capture_start_monotonic_time
            if (
                recording_raw
                and capture_start is not None
                and now < capture_start
            ):
                # The state callback may have been queued while START was
                # waiting for the lifecycle lock.  Keep it as the latest
                # preview sample, but never attribute a pre-START message to
                # the new raw Episode.
                recorder = None
        if recording_raw:
            try:
                with self._lifecycle_lock:
                    if self._raw_recorder is recorder and not recorder.closed:
                        self._append_raw_state(msg, now, now)
            except Exception as exc:
                reason = f"raw state append failed: {exc}"
                self.get_logger().error(reason)
                self._isolate_raw_failure(recorder, reason)
                if self._state_machine.mode == CollectorMode.RECORDING:
                    try:
                        self._state_machine.mark_failed(reason)
                    except RuntimeError:
                        pass
        self._log_state_sample_received_throttled(force=was_unavailable_or_stale)

    def _on_record_command(self, msg: RecordCommand) -> None:
        # Flush and lifecycle transitions must be serialized as one command
        # transaction.  A multi-threaded ROS executor can otherwise let STOP
        # and a second START interleave between the callback drain and the
        # state-machine transition.
        with self._command_lock:
            self._on_record_command_serial(msg)

    def _on_record_command_serial(self, msg: RecordCommand) -> None:
        command_value = int(msg.command)
        callback_flush_failed = False
        if command_value in (
            int(RecordCommand.START),
            int(RecordCommand.STOP),
            int(RecordCommand.DISCARD),
        ):
            # The cache callback is intentionally asynchronous so the ZMQ reader
            # cannot be blocked by durable raw I/O. Drain packets already
            # received before moving the lifecycle boundary. START also drains
            # idle packets so they cannot leak into the new Episode.
            flush_callbacks = getattr(self._camera_cache, "flush_callbacks", None)
            if callable(flush_callbacks):
                try:
                    flushed = flush_callbacks(timeout=self._save_shutdown_grace_sec)
                    if not flushed:
                        callback_flush_failed = True
                        self.get_logger().error(
                            "camera callback queue did not drain before command boundary"
                        )
                except Exception as exc:
                    if command_value == int(RecordCommand.STOP):
                        callback_flush_failed = True
                    self.get_logger().warn(f"camera callback flush failed: {exc}")
        with self._lifecycle_lock:
            self._handle_record_command(
                msg, callback_flush_failed=callback_flush_failed
            )

    def _handle_record_command(
        self, msg: RecordCommand, *, callback_flush_failed: bool = False
    ) -> None:
        command_value = int(msg.command)
        command_id = str(getattr(msg, "command_id", "")).strip()
        command_episode_id = str(msg.episode_id).strip()
        command_name = _record_command_name(command_value)
        fingerprint = CommandFingerprint(
            command=command_value,
            task_prompt=str(msg.task_prompt),
            episode_id=command_episode_id,
            force=bool(getattr(msg, "force", False)),
        )
        replay = self._command_receipts.lookup(command_id, fingerprint)
        if replay is not None:
            if replay.disposition == "CONFLICT":
                self._publish_status(
                    DiagnosticStatus.ERROR,
                    "command_id reuse rejected because its payload changed: "
                    f"{command_id}",
                )
                return
            self._set_command_receipt(
                command_id=command_id,
                command=command_name,
                episode_id=command_episode_id,
                outcome=replay.outcome,
            )
            level = (
                DiagnosticStatus.ERROR
                if replay.outcome == "FAILED"
                or self._state_machine.mode == CollectorMode.FAILED
                else DiagnosticStatus.WARN
                if replay.outcome == "REJECTED"
                else DiagnosticStatus.OK
            )
            self._publish_status(
                level,
                f"{command_name} duplicate replayed as {replay.outcome}; "
                f"mode={self._state_machine.mode.value}",
            )
            return
        with self._lifecycle_lock:
            recovery_active = self._recovery_active
            recovery_error = self._recovery_error
        if command_value == int(RecordCommand.START) and recovery_active:
            # Recovery owns the shared writer/dataset lock until every
            # startup job has either completed or been durably failed.  Do
            # not let a live START race that transaction and produce a
            # second writer owner.
            self._set_command_receipt(
                command_id=command_id,
                command=command_name,
                episode_id=command_episode_id,
                outcome="REJECTED",
            )
            self._last_command_outcome = "REJECTED"
            self._command_receipts.remember(command_id, fingerprint, "REJECTED")
            message = "START rejected while raw materialization recovery is in progress"
            if recovery_error:
                message = f"START blocked: {recovery_error}"
            self._publish_status(
                DiagnosticStatus.ERROR if recovery_error else DiagnosticStatus.WARN,
                message,
            )
            return
        if (
            callback_flush_failed
            and command_value == int(RecordCommand.STOP)
            and self._state_machine.mode == CollectorMode.RECORDING
        ):
            reason = "camera callback queue did not drain before STOP"
            self._state_machine.mark_failed(reason)
            self._last_command_outcome = "FAILED"
            self._command_receipts.remember(command_id, fingerprint, "FAILED")
            self._publish_status(
                DiagnosticStatus.ERROR,
                f"save blocked; DISCARD required: {reason}",
            )
            return
        self._set_command_receipt(
            command_id=command_id,
            command=command_name,
            episode_id=command_episode_id,
            outcome="RECEIVED",
        )
        result = self._state_machine.handle_command(
            command_value,
            task_prompt=msg.task_prompt,
            episode_id=command_episode_id,
            force=bool(getattr(msg, "force", False)),
            now_sec=self._now_sec(),
        )
        if result.should_start and result.session is not None:
            try:
                episode_index = None
                if self._recording_mode == "legacy":
                    episode_index = self._writer.start_episode(
                        result.session.task_prompt, result.session.episode_id
                    )
                self._start_raw_episode(result.session.task_prompt, result.session.episode_id)
            except Exception as exc:
                if self._raw_recorder is not None:
                    try:
                        self._raw_recorder.discard(reason=f"start failed: {exc}")
                    except Exception:
                        pass
                    self._raw_recorder = None
                    self._raw_episode_path = None
                self._state_machine = RecordStateMachine()
                self._raw_session_id = None
                if self._writer.active_episode_index is not None:
                    try:
                        self._writer.discard_episode()
                    except Exception:
                        pass
                self._last_command_outcome = "FAILED"
                self._command_receipts.remember(
                    command_id, fingerprint, self._last_command_outcome
                )
                self._publish_status(
                    DiagnosticStatus.ERROR, f"failed to start episode: {exc}"
                )
                return
            self._clear_save_tracking()
            self._last_recorded_camera_identity = None
            self._last_command_outcome = "SUCCEEDED"
            self._command_receipts.remember(
                command_id, fingerprint, self._last_command_outcome
            )
            self._publish_status(
                DiagnosticStatus.OK,
                f"RECORDING episode {episode_index}: {result.session.task_prompt}",
            )
            return

        if result.should_save:
            try:
                self._freeze_raw_capture(command="STOP", reason="user_stop")
            except Exception as exc:
                reason = f"failed to freeze raw capture at STOP: {exc}"
                self._state_machine.mark_failed(reason)
                self._last_command_outcome = "FAILED"
                self._command_receipts.remember(command_id, fingerprint, "FAILED")
                self._publish_status(DiagnosticStatus.ERROR, reason)
                return

        if result.should_discard:
            try:
                self._discard_raw_episode(reason="user_discard")
                if self._writer.active_episode_index is not None:
                    self._writer.discard_episode()
                self._state_machine.mark_discarded()
                self._last_command_outcome = "SUCCEEDED"
                self._last_episode_id = (
                    result.session.episode_id if result.session is not None else ""
                )
                self._last_episode_outcome = "DISCARDED"
                self._last_stale_camera_identity = None
                self._publish_status(DiagnosticStatus.OK, "DISCARD complete; IDLE")
            except Exception as exc:
                reason = f"discard failed: {exc}"
                self._state_machine.mark_discard_failed(reason)
                self._last_command_outcome = "FAILED"
                self._publish_status(DiagnosticStatus.ERROR, reason)
            self._command_receipts.remember(
                command_id,
                fingerprint,
                self._last_command_outcome,
                replayable=self._last_command_outcome != "FAILED",
            )
            return

        self._last_command_outcome = "SUCCEEDED" if result.accepted else "REJECTED"
        self._command_receipts.remember(
            command_id, fingerprint, self._last_command_outcome
        )
        self._publish_status(_diagnostic_level(result.level), result.message)

    def _recovery_status_message(self) -> str:
        with self._lifecycle_lock:
            error = self._recovery_error
            active = self._recovery_active
        if error:
            return f"START blocked: {error}"
        if active:
            return "START rejected while raw materialization recovery is in progress"
        return "raw materialization recovery complete; IDLE: waiting for START"

    def _record_tick(self) -> None:
        tick_now = time.monotonic()
        if (
            self._last_record_tick_monotonic_sec is not None
            and tick_now - self._last_record_tick_monotonic_sec > 1.5 / self._fps
        ):
            self._timer_deadline_misses += 1
        self._last_record_tick_monotonic_sec = tick_now
        self._record_tick_count += 1
        if self._state_machine.mode == CollectorMode.NEED_TO_SAVE:
            self._begin_save_episode()
            return
        if self._state_machine.mode == CollectorMode.SAVING:
            self._poll_save_episode()
            return
        if self._state_machine.mode == CollectorMode.FAILED:
            self._publish_failed_status_throttled()
            return
        if self._state_machine.mode != CollectorMode.RECORDING:
            return

        if self._enforce_recording_safety_limits():
            return

        now = time.monotonic()
        state = self._latest_state
        camera_bundle = self._camera_cache.latest()
        if state is None:
            self._publish_warn_throttled("missing robo_state sample")
            return
        if camera_bundle is None:
            self._publish_warn_throttled("missing complete camera frame bundle")
            return

        state_age = now - state.received_monotonic_sec
        camera_age = now - camera_bundle.received_monotonic_sec
        self._record_capture_age_sample(state_age, camera_age)
        if state_age > self._max_state_age_sec:
            self._publish_warn_throttled(
                f"stale robo_state sample: {state_age:.3f}s old"
            )
            return
        if camera_age > self._max_camera_age_sec:
            self._record_stale_camera_bundle(camera_bundle)
            self._publish_warn_throttled(
                f"stale camera frame: {camera_age:.3f}s old"
            )
            return

        if camera_bundle.identity == self._last_recorded_camera_identity:
            self._publish_warn_throttled("camera frame bundle has already been recorded")
            return

        missing_selected = selected_missing_inputs(
            getattr(state.msg, "missing_optional_fields", ()),
            self._field_selection,
        )
        if missing_selected:
            self._publish_warn_throttled(
                "selected state/action field(s) missing or stale: "
                + ",".join(missing_selected)
            )
            return

        selected_timestamps = selected_source_timestamps_sec(
            state.msg, self._field_selection
        )
        if selected_timestamps is None:
            self._publish_warn_throttled(
                "selected state/action source timestamps unavailable or invalid"
            )
            return
        selected_values = list(selected_timestamps.values())
        selected_skew_sec = source_timestamp_skew_sec(
            selected_values[0], selected_values[1:]
        )
        if selected_skew_sec is None:
            self._publish_warn_throttled(
                "selected state/action source timestamp skew is invalid"
            )
            return
        if selected_skew_sec > self._max_state_camera_skew_sec:
            self._publish_warn_throttled(
                "selected state/action source timestamp skew "
                f"{selected_skew_sec:.3f}s exceeds "
                f"{self._max_state_camera_skew_sec:.3f}s"
            )
            return
        receive_skew_sec = source_timestamp_skew_sec(
            state.received_monotonic_sec,
            [camera_bundle.received_monotonic_sec],
        )
        if receive_skew_sec is None:
            self._publish_warn_throttled(
                "state/camera local receive timestamp unavailable or invalid"
            )
            return
        if receive_skew_sec > self._max_state_camera_skew_sec:
            self._publish_warn_throttled(
                "state/camera local receive-time skew "
                f"{receive_skew_sec:.3f}s exceeds "
                f"{self._max_state_camera_skew_sec:.3f}s"
            )
            return
        self._record_state_camera_skew_sample(receive_skew_sec)

        try:
            if self._recording_mode == "legacy":
                self._writer.add_frame(
                    _robot_frame_from_msg(state.msg),
                    camera_bundle.images,
                    camera_timestamps_sec={
                        stream: frame.camera_timestamp_sec
                        for stream, frame in camera_bundle.frames.items()
                    },
                )
            self._last_recorded_camera_identity = camera_bundle.identity
            self._raw_frame_count += 1
        except Exception as exc:
            reason = self._writer.active_failed_reason or str(exc)
            self._raw_recording_failed_reason = reason
            self._state_machine.mark_failed(reason)
            self._publish_status(
                DiagnosticStatus.ERROR,
                f"recording failed; DISCARD required: {reason}",
            )

    def _enforce_recording_safety_limits(self) -> bool:
        session = self._state_machine.session
        if session is None:
            self._state_machine.mark_failed("RECORDING has no active session")
            self._publish_status(
                DiagnosticStatus.ERROR,
                "recording failed; DISCARD required: RECORDING has no active session",
            )
            return True
        cache_stats = self._camera_cache.stats
        callback_overflow = int(
            cache_stats.get(
                "callback_overflow",
                getattr(self._camera_cache, "callback_overflow", 0),
            )
            or 0
        )
        if callback_overflow > self._camera_callback_overflow_at_start:
            return self._discard_for_safety(
                "camera recording queue overflow: "
                f"{callback_overflow - self._camera_callback_overflow_at_start} packet(s)"
            )
        try:
            free_disk_bytes = _free_disk_bytes(self._writer.root_output_dir)
        except OSError as exc:
            reason = f"cannot inspect output disk: {exc}"
            return self._discard_for_safety(reason)
        reason = recording_safety_reason(
            elapsed_sec=max(0.0, self._now_sec() - session.started_at_sec),
            frame_count=(self._writer.active_frame_count if self._recording_mode == "legacy" else self._raw_frame_count),
            max_duration_sec=self._max_episode_duration_sec,
            max_frames=self._max_episode_frames,
            free_disk_bytes=free_disk_bytes,
            min_free_disk_bytes=self._min_free_disk_bytes,
        )
        if reason is None:
            return False
        return self._discard_for_safety(reason)

    def _discard_for_safety(self, reason: str) -> bool:
        session = self._state_machine.session
        if session is None:
            self._state_machine.mark_failed("RECORDING has no active session")
            self._publish_status(
                DiagnosticStatus.ERROR,
                "recording failed; DISCARD required: RECORDING has no active session",
            )
            return True
        result = self._state_machine.handle_command(
            RecordCommand.DISCARD,
            episode_id=session.episode_id,
            now_sec=self._now_sec(),
        )
        if not result.should_discard:
            failure = f"safety discard command failed: {result.message}"
            self._state_machine.mark_failed(failure)
            self._publish_status(
                DiagnosticStatus.ERROR,
                f"recording failed; DISCARD required: {failure}",
            )
            return True
        try:
            self._discard_raw_episode(reason=reason)
            if self._writer.active_episode_index is not None:
                self._writer.discard_episode()
            self._state_machine.mark_discarded()
        except Exception as exc:
            failure = f"safety discard failed: {exc}"
            self._state_machine.mark_discard_failed(failure)
            self._publish_status(DiagnosticStatus.ERROR, failure)
            return True
        self._last_episode_id = session.episode_id
        self._last_episode_outcome = "DISCARDED"
        self._last_recorded_camera_identity = None
        self._last_stale_camera_identity = None
        self._publish_status(DiagnosticStatus.WARN, f"SAFETY DISCARD: {reason}; IDLE")
        return True

    def _begin_save_episode(self) -> None:
        if self._save_worker.has_active:
            reason = "cannot start save because another save task is active"
            self._state_machine.mark_failed(reason)
            self._publish_status(DiagnosticStatus.ERROR, reason)
            return

        self._saving_episode_index = self._writer.active_episode_index
        self._saving_frame_count = self._writer.active_frame_count if self._recording_mode == "legacy" else self._raw_frame_count
        self._save_started_monotonic_sec = time.monotonic()
        self._save_finished_monotonic_sec = None
        self._save_progress_monotonic_sec = self._save_started_monotonic_sec
        self._save_progress_seq = 0
        self._save_phase = "queued"
        try:
            self._state_machine.mark_saving()
            if self._recording_mode == "raw_first":
                if self._raw_episode_path is None or self._raw_materialization_job is None:
                    self._mark_no_materialized_artifact()
                    raise RuntimeError("raw-first STOP has no durable materialization job")
                config = self._materialization_config(self._raw_materialization_job.get("conversion_config"))
                self._save_worker.start(lambda report_progress: RawEpisodeMaterializer(config).materialize(self._raw_episode_path, progress_callback=report_progress))
            else:
                self._save_worker.start(lambda report_progress: self._writer.save_episode(progress_callback=report_progress))
        except Exception as exc:
            reason = f"failed to start save worker: {exc}"
            if self._state_machine.mode == CollectorMode.SAVING:
                self._state_machine.mark_failed(reason)
            elif self._state_machine.mode == CollectorMode.NEED_TO_SAVE:
                self._state_machine.mark_failed(reason)
            self._mark_raw_materialization_failed(reason)
            self._save_phase = "failed"
            self._save_finished_monotonic_sec = time.monotonic()
            self._save_progress_monotonic_sec = self._save_finished_monotonic_sec
            self._publish_status(
                DiagnosticStatus.ERROR,
                f"save failed; DISCARD required: {reason}",
            )
            return

        self._publish_status(
            DiagnosticStatus.OK,
            self._saving_status_message(),
        )

    def _poll_save_episode(self) -> None:
        self._drain_save_progress()
        if not self._save_worker.done:
            return

        session = self._state_machine.session
        try:
            result = self._save_worker.take_result()
            self._drain_save_progress()
        except Exception as exc:
            reason = self._writer.active_failed_reason or str(exc)
            self._state_machine.mark_failed(reason)
            self._mark_raw_materialization_failed(reason)
            self._save_phase = "failed"
            self._save_finished_monotonic_sec = time.monotonic()
            self._save_progress_monotonic_sec = self._save_finished_monotonic_sec
            self._publish_status(
                DiagnosticStatus.ERROR,
                f"save failed; DISCARD required: {reason}",
            )
            return

        if isinstance(result, MaterializationResult):
            try:
                quality_status = self._mark_raw_materialization_succeeded(result)
            except Exception as exc:
                reason = f"failed to persist materialization result: {exc}"
                self._mark_raw_materialization_failed(reason)
                self._state_machine.mark_failed(reason)
                self._save_phase = "failed"
                self._save_finished_monotonic_sec = time.monotonic()
                self._save_progress_monotonic_sec = self._save_finished_monotonic_sec
                self._last_episode_outcome = "FAILED"
                self._publish_status(
                    DiagnosticStatus.ERROR,
                    f"save failed; DISCARD required: {reason}",
                )
                return
            # REVIEW is a durable, auditable result, but it is not a successful
            # training publication.  Keep the collector in the explicit
            # failure/discard path instead of reporting it as SAVED.
            saved = quality_status == "READY"
        else:
            saved = bool(result.saved)
            if not saved:
                self._mark_no_materialized_artifact()
        if not saved:
            if isinstance(result, MaterializationResult):
                quality_status = str(quality_status).upper()
                reason = (
                    f"materialization QC status={quality_status}; "
                    "READY is required"
                )
                self._last_episode_outcome = quality_status
            else:
                reason = "save returned saved=False"
                self._last_episode_outcome = "REJECTED"
            self._state_machine.mark_failed(reason)
            self._save_phase = "failed"
            self._save_finished_monotonic_sec = time.monotonic()
            level = (
                DiagnosticStatus.WARN
                if isinstance(result, MaterializationResult)
                and str(quality_status).upper() == "REVIEW"
                else DiagnosticStatus.ERROR
            )
            self._publish_status(
                level,
                f"save not published; DISCARD required: {reason}",
            )
            return
        self._state_machine.mark_saved()
        self._save_phase = "complete"
        self._save_finished_monotonic_sec = time.monotonic()
        self._save_progress_monotonic_sec = self._save_finished_monotonic_sec
        self._last_episode_id = session.episode_id if session is not None else ""
        self._last_episode_outcome = "SAVED"
        if isinstance(result, MaterializationResult):
            result_message = f"raw materialization {quality_status.lower()}"
            result_episode = result.episode_id
        else:
            result_message = result.message
            result_episode = str(result.episode_index)
        self._publish_status(
            DiagnosticStatus.OK,
            (
                f"{result_message}: episode={result_episode}, "
                f"frames={result.frame_count}"
            ),
        )

    def _start_raw_episode(self, task_prompt: str, requested_episode_id: str) -> None:
        if not self._raw_recording_enabled:
            return
        self._camera_cache.reset_episode_window()
        self._last_stale_camera_identity = None
        self._camera_callback_overflow_at_start = int(
            getattr(self._camera_cache, "callback_overflow", 0) or 0
        )
        self._state_camera_skew_samples.clear()
        self._state_camera_skew_sample_cursor = 0
        self._state_age_samples.clear()
        self._state_age_sample_cursor = 0
        self._camera_age_samples.clear()
        self._camera_age_sample_cursor = 0
        raw_episode_id = _raw_episode_id(requested_episode_id)
        raw_session_id = f"collector:{uuid4().hex}"
        self._raw_session_id = raw_session_id
        self._raw_camera_last_sequences.clear()
        self._raw_camera_last_packet_session_id = None
        self._raw_camera_last_packet_sequence = None
        self._raw_camera_session_ids.clear()
        self._raw_camera_capture_attached = False
        self._camera_clock_offsets.clear()
        self._raw_camera_observed_high_watermarks.clear()
        self._raw_camera_packet_high_watermarks.clear()
        self._raw_camera_binding_checkpoint_signature = None
        self._raw_recording_failed_reason = None
        common_metadata = {
            "requested_episode_id": requested_episode_id,
            "collector_session_id": raw_session_id,
            "fps": self._fps,
            "camera_streams": list(self._camera_cache.streams),
            "max_alignment_residual_sec": self._max_state_camera_skew_sec,
            "requested_source_scope": self._requested_raw_source_scope,
            "capture_plane": (
                "camera_side+host_receiver"
                if self._raw_source_scope == "camera_capture"
                else "host_receiver"
            ),
            "collector_git_commit": self._collector_git_commit,
            "config_hash": self._collector_config_hash,
            "capture_config": self._capture_config,
            "materialization_role": "primary" if self._recording_mode == "raw_first" else "comparison_shadow",
            "primary_output": "raw_materialization" if self._recording_mode == "raw_first" else "legacy_writer",
            "camera_raw_spool_root": (
                str(self._camera_raw_spool_root)
                if self._camera_raw_spool_root is not None
                else ""
            ),
            "camera_capture_binding": {
                "schema": "robo_collector.camera_capture_binding.v1",
                "status": (
                    "OPEN" if self._raw_source_scope == "camera_capture" else "NOT_REQUESTED"
                ),
                "binding_method": "camera_source_snapshot_window_v1",
                "camera_raw_spool_root": (
                    str(self._camera_raw_spool_root)
                    if self._camera_raw_spool_root is not None
                    else ""
                ),
                "observed_session_ids": [],
                "unbound_observed_session_ids": [],
                "packet_high_watermarks": {},
                "stream_high_watermarks": {},
                "clock_mapping": {},
                "source_snapshots": [],
            },
            "conversion_config": {
                "output_root": str(self._writer.root_output_dir),
                "dataset_name": self._writer.dataset_name,
                **self._raw_conversion_config(),
                "max_alignment_residual_sec": self._max_state_camera_skew_sec,
                "output_schema_version": "lerobot.v2.1.raw_materialization.v1",
                "require_complete_capture": self._raw_source_scope == "camera_capture",
            },
        }
        try:
            recorder = RawEpisodeRecorder(
                self._raw_episode_root,
                raw_episode_id,
                source_scope=self._raw_source_scope,
                task_prompt=task_prompt,
                metadata=common_metadata,
                collector_git_commit=self._collector_git_commit,
                config_hash=self._collector_config_hash,
                max_records_per_chunk=self._raw_max_records_per_chunk,
                max_record_bytes=self._raw_max_record_bytes,
            )
        except FileExistsError:
            recorder = RawEpisodeRecorder(
                self._raw_episode_root,
                f"{raw_episode_id}-{uuid4().hex}",
                source_scope=self._raw_source_scope,
                task_prompt=task_prompt,
                metadata=common_metadata,
                collector_git_commit=self._collector_git_commit,
                config_hash=self._collector_config_hash,
                max_records_per_chunk=self._raw_max_records_per_chunk,
                max_record_bytes=self._raw_max_record_bytes,
            )
        self._raw_recorder = recorder
        self._raw_episode_path = recorder.path
        self._raw_capture_start_wall_time = float(
            recorder.manifest.get("start_wall_time", time.time())
        )
        self._raw_capture_start_monotonic_time = time.monotonic()
        self._raw_materialization_job = None
        self._raw_state_sequence = 0
        self._raw_event_sequence = 0
        self._raw_frame_count = 0
        self._record_tick_count = 0
        self._timer_deadline_misses = 0
        self._last_record_tick_monotonic_sec = None
        self._append_raw_event(
            "START",
            {"task_prompt": task_prompt, "requested_episode_id": requested_episode_id},
        )

    def _on_camera_packet(self, packet: Any) -> None:
        """Persist received encoded payloads while an Episode is active."""
        with self._lifecycle_lock:
            recorder = self._raw_recorder
            recording_raw = self._state_machine.mode == CollectorMode.RECORDING
        session_value = _packet_member(packet, "session_id", "")
        session_id = str(session_value or "")
        frames = _packet_member(packet, "frames", {})
        if recording_raw and self._packet_received_before_capture_start(packet, frames):
            return
        self._observe_camera_clock_mapping(session_id, frames)
        if not recording_raw or recorder is None or recorder.closed:
            return
        if self._raw_source_scope == "camera_capture":
            # The source-side spool is the authoritative camera stream in
            # complete-capture mode.  Keep the receiver callback as a session
            # observation only; importing it here would duplicate sequences
            # when the spool is attached at STOP.
            self._track_camera_capture_observation(
                recorder, session_id, frames, _packet_member(packet, "packet_sequence", None)
            )
            return
        metadata = _packet_member(packet, "metadata", {})
        producer_gaps = _packet_member(packet, "producer_gaps", {})
        publisher_gaps = _packet_member(packet, "publisher_gaps", {})
        packet_sequence = _packet_member(packet, "packet_sequence", None)
        if not isinstance(frames, Mapping) and isinstance(packet, Mapping):
            # A legacy decoded mapping may still carry the original payloads in
            # metadata.frame_provenance.  Rebuild the envelope view without
            # decoding or rewriting those bytes.
            images = packet.get("images", {})
            provenance_map = (
                metadata.get("frame_provenance", {})
                if isinstance(metadata, dict)
                else {}
            )
            if isinstance(images, Mapping) and isinstance(provenance_map, Mapping):
                frames = {
                    stream: {
                        **(provenance_map.get(stream, {}) or {}),
                        "payload": images.get(stream),
                    }
                    for stream in images
                }
            producer_gaps = packet.get("producer_gaps", {})
            publisher_gaps = packet.get("publisher_gaps", {})
            packet_sequence = packet.get("packet_sequence")
        cameras = metadata.get("cameras", {}) if isinstance(metadata, dict) else {}
        if not isinstance(frames, Mapping):
            if recording_raw and recorder is not None and not recorder.closed:
                self._record_raw_rejection(recorder, "camera_packet", "invalid_frames")
            return
        if not isinstance(producer_gaps, Mapping):
            producer_gaps = {}
        if not isinstance(publisher_gaps, Mapping):
            publisher_gaps = {}
        try:
            packet_sequence = (
                None
                if packet_sequence is None
                else _non_negative_int(packet_sequence)
            )
        except ValueError as exc:
            self._record_raw_rejection(recorder, "camera_packet", "invalid_sequence")
            self.get_logger().error(f"raw camera packet sequence is invalid: {exc}")
            return
        same_packet_session = (
            self._raw_camera_last_packet_session_id == session_id
            and session_id != ""
        )
        packet_gap = (
            max(0, packet_sequence - self._raw_camera_last_packet_sequence - 1)
            if same_packet_session
            and packet_sequence is not None
            and self._raw_camera_last_packet_sequence is not None
            else 0
        )
        appended_sequences: dict[str, int] = {}
        try:
            if session_id:
                with self._lifecycle_lock:
                    if self._raw_recorder is recorder:
                        self._raw_camera_session_ids.add(session_id)
            for stream, frame in frames.items():
                payload = _packet_member(frame, "payload")
                if not isinstance(payload, (bytes, bytearray, memoryview)):
                    self._record_raw_rejection(
                        recorder, str(stream), "invalid_payload"
                    )
                    continue
                sequence_value = _packet_member(frame, "sequence")
                sequence = _optional_sequence(sequence_value)
                if sequence is None:
                    self._record_raw_rejection(
                        recorder, str(stream), "invalid_sequence"
                    )
                    continue
                previous = self._raw_camera_last_sequences.get(str(stream))
                missing = (
                    max(0, sequence - previous[1] - 1)
                    if previous is not None
                    and previous[0] == session_id
                    and sequence is not None
                    else 0
                )
                provenance = {
                    "sequence": sequence,
                    "clock_domain": _camera_clock_domain(frame, str(stream)),
                    "timestamp_quality": _packet_member(frame, "timestamp_quality")
                    or "host_after_capture",
                    "device_timestamp": _packet_member(frame, "device_timestamp"),
                    "device_unit": _packet_member(frame, "device_unit"),
                    "timestamp_domain": _packet_member(
                        frame,
                        "timestamp_domain",
                        _packet_member(frame, "device_timestamp_domain_type"),
                    ),
                    "server_wall_timestamp": _packet_member(frame, "server_wall_timestamp"),
                    "server_monotonic_timestamp": _packet_member(frame, "server_monotonic_timestamp"),
                    "receive_wall_timestamp": _packet_member(frame, "receive_wall_timestamp"),
                    "receive_monotonic_timestamp": _packet_member(frame, "receive_monotonic_timestamp"),
                    "record_monotonic_timestamp": time.monotonic(),
                    "session_id": session_id,
                }
                serial = ""
                if isinstance(cameras, Mapping) and isinstance(cameras.get(stream), Mapping):
                    serial = str(cameras[stream].get("serial", ""))
                frame_gap = _packet_member(frame, "producer_gap_count", None)
                producer_gap_count = _non_negative_int(
                    frame_gap if frame_gap is not None else producer_gaps.get(stream, 0)
                )
                producer_gap_count = min(missing, producer_gap_count)
                publisher_gap_count = min(
                    max(0, missing - producer_gap_count),
                    _non_negative_int(publisher_gaps.get(stream, 0)),
                )
                remaining_gap = missing - producer_gap_count - publisher_gap_count
                transport_gap_count = min(remaining_gap, packet_gap)
                unattributed_gap_count = remaining_gap - transport_gap_count
                with self._lifecycle_lock:
                    if (
                        self._state_machine.mode != CollectorMode.RECORDING
                        or self._raw_recorder is not recorder
                        or recorder.closed
                    ):
                        return
                    recorder.append_camera(
                        str(stream),
                        payload,
                        provenance,
                        payload_encoding=str(
                            _packet_member(frame, "payload_encoding") or "image/jpeg"
                        ),
                        serial=serial,
                        producer_gap_count=producer_gap_count,
                        publisher_gap_count=publisher_gap_count,
                        transport_gap_count=transport_gap_count,
                        unattributed_gap_count=unattributed_gap_count,
                    )
                if sequence is not None:
                    appended_sequences[str(stream)] = sequence
            if appended_sequences:
                self._raw_camera_last_sequences.update(
                    {
                        stream: (session_id, sequence)
                        for stream, sequence in appended_sequences.items()
                    }
                )
                self._raw_camera_last_packet_session_id = session_id
                self._raw_camera_last_packet_sequence = packet_sequence
        except Exception as exc:
            self.get_logger().error(f"raw camera append failed: {exc}")
            self._isolate_raw_failure(recorder, f"raw camera append failed: {exc}")
            if self._state_machine.mode == CollectorMode.RECORDING:
                try:
                    self._state_machine.mark_failed(f"raw camera append failed: {exc}")
                except RuntimeError:
                    pass

    def _isolate_raw_failure(self, recorder: RawEpisodeRecorder, reason: str) -> None:
        """Detach a broken recorder so overflow/I/O failures cannot keep appending."""
        with self._lifecycle_lock:
            if self._raw_recorder is not recorder:
                return
            self._raw_recording_failed_reason = str(reason)
            self._raw_recorder = None
            try:
                recorder.quarantine(reason)
            except Exception as exc:
                self.get_logger().error(f"failed to quarantine raw episode: {exc}")

    def _record_raw_rejection(
        self,
        recorder: RawEpisodeRecorder,
        stream: str,
        error_type: str,
        count: int = 1,
    ) -> None:
        """Keep malformed packet counters durable without masking the input error."""
        try:
            recorder.record_rejection(stream, error_type, count=count)
        except Exception as exc:
            self.get_logger().error(f"failed to record raw input rejection: {exc}")

    def _observe_camera_clock_mapping(
        self, session_id: str, frames: Any
    ) -> None:
        """Estimate source-server to collector-wall offset from received frames."""
        if not session_id or not isinstance(frames, Mapping):
            return
        with self._lifecycle_lock:
            samples = self._camera_clock_offsets.setdefault(session_id, [])
            for frame in frames.values():
                server_wall = _packet_member(frame, "server_wall_timestamp")
                receive_wall = _packet_member(frame, "receive_wall_timestamp")
                try:
                    offset = float(receive_wall) - float(server_wall)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(offset):
                    continue
                if len(samples) >= 128:
                    samples.pop(0)
                samples.append(offset)

    def _packet_received_before_capture_start(
        self, packet: Any, frames: Any
    ) -> bool:
        """Reject callback work received before the current START boundary."""
        start = self._raw_capture_start_monotonic_time
        if start is None:
            return False
        values: list[float] = []
        packet_received = _packet_member(packet, "receive_monotonic_timestamp")
        if packet_received is not None:
            try:
                candidate = float(packet_received)
            except (TypeError, ValueError):
                candidate = float("nan")
            if math.isfinite(candidate):
                values.append(candidate)
        if isinstance(frames, Mapping):
            for frame in frames.values():
                received = _packet_member(frame, "receive_monotonic_timestamp")
                if received is None:
                    continue
                try:
                    candidate = float(received)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(candidate):
                    values.append(candidate)
        return bool(values) and min(values) < start

    def _track_camera_capture_observation(
        self,
        recorder: RawEpisodeRecorder,
        session_id: str,
        frames: Any,
        packet_sequence: Any,
    ) -> None:
        """Persist bounded source high-watermarks while a task is recording.

        Camera capture mode imports the source spool at STOP, so the receiver
        does not append callback payloads to the task raw stream.  These
        checkpoints preserve the task-to-source relationship across a host
        crash and make an incomplete source binding visible to QC.
        """
        if not session_id or not isinstance(frames, Mapping):
            return
        try:
            packet_value = (
                None if packet_sequence is None else _non_negative_int(packet_sequence)
            )
        except ValueError:
            packet_value = None
        with self._lifecycle_lock:
            if (
                self._raw_recorder is not recorder
                or recorder.closed
                or self._state_machine.mode != CollectorMode.RECORDING
            ):
                return
            self._raw_camera_session_ids.add(session_id)
            if packet_value is not None:
                previous_packet = self._raw_camera_packet_high_watermarks.get(session_id)
                if previous_packet is None or packet_value > previous_packet:
                    self._raw_camera_packet_high_watermarks[session_id] = packet_value
            stream_watermarks = self._raw_camera_observed_high_watermarks.setdefault(
                session_id, {}
            )
            for stream, frame in frames.items():
                sequence = _optional_sequence(_packet_member(frame, "sequence"))
                if sequence is None:
                    continue
                stream_name = str(stream)
                previous = stream_watermarks.get(stream_name)
                if previous is None or sequence > previous:
                    stream_watermarks[stream_name] = sequence
            try:
                self._persist_camera_binding_checkpoint_locked(recorder)
            except Exception as exc:
                reason = f"camera binding checkpoint failed: {exc}"
                self._raw_recording_failed_reason = reason
                self.get_logger().error(reason)
                self._isolate_raw_failure(recorder, reason)
                if self._state_machine.mode == CollectorMode.RECORDING:
                    try:
                        self._state_machine.mark_failed(reason)
                    except RuntimeError:
                        pass

    def _persist_camera_binding_checkpoint_locked(
        self, recorder: RawEpisodeRecorder, *, force: bool = False
    ) -> None:
        """Durably checkpoint bounded camera observations under the lifecycle lock."""
        sample_counts = tuple(
            (session_id, len(self._camera_clock_offsets.get(session_id, ())))
            for session_id in sorted(self._raw_camera_session_ids)
        )
        packet_watermarks = tuple(
            sorted(self._raw_camera_packet_high_watermarks.items())
        )
        stream_watermarks = tuple(
            (
                session_id,
                tuple(sorted(streams.items())),
            )
            for session_id, streams in sorted(
                self._raw_camera_observed_high_watermarks.items()
            )
        )
        # Do not rewrite the manifest for every received frame.  A new source,
        # the first mapping sample, and each bounded 32-frame bucket are
        # durable checkpoints; STOP still writes the exact final snapshot.
        signature = (
            tuple(sorted(self._raw_camera_session_ids)),
            tuple(
                (session_id, count // 8)
                for session_id, count in sample_counts
            ),
            tuple(
                (session_id, stream, sequence // 32)
                for session_id, streams in stream_watermarks
                for stream, sequence in streams
            ),
            tuple(
                (session_id, sequence // 32)
                for session_id, sequence in packet_watermarks
            ),
        )
        if not force and signature == self._raw_camera_binding_checkpoint_signature:
            return
        clock_mapping = {
            session_id: {
                "method": "minimum_receive_wall_minus_server_wall",
                "samples": [float(value) for value in self._camera_clock_offsets.get(session_id, [])],
                "sample_count": len(self._camera_clock_offsets.get(session_id, [])),
                "offset_sec": (
                    min(self._camera_clock_offsets[session_id])
                    if self._camera_clock_offsets.get(session_id)
                    else None
                ),
                "min_offset_sec": (
                    min(self._camera_clock_offsets[session_id])
                    if self._camera_clock_offsets.get(session_id)
                    else None
                ),
                "max_offset_sec": (
                    max(self._camera_clock_offsets[session_id])
                    if self._camera_clock_offsets.get(session_id)
                    else None
                ),
            }
            for session_id in sorted(self._raw_camera_session_ids)
        }
        binding = {
            "schema": "robo_collector.camera_capture_binding.v1",
            "status": "OPEN",
            "binding_method": "camera_source_snapshot_window_v1",
            "camera_raw_spool_root": (
                str(self._camera_raw_spool_root)
                if self._camera_raw_spool_root is not None
                else ""
            ),
            "observed_session_ids": sorted(self._raw_camera_session_ids),
            "unbound_observed_session_ids": [],
            "packet_high_watermarks": {
                session_id: int(sequence)
                for session_id, sequence in sorted(
                    self._raw_camera_packet_high_watermarks.items()
                )
            },
            "stream_high_watermarks": {
                session_id: {
                    stream: int(sequence)
                    for stream, sequence in sorted(streams.items())
                }
                for session_id, streams in sorted(
                    self._raw_camera_observed_high_watermarks.items()
                )
            },
            "clock_mapping": clock_mapping,
            "source_snapshots": [],
            "last_checkpoint_wall_time": time.time(),
        }
        recorder.update_metadata({"camera_capture_binding": binding})
        self._raw_camera_binding_checkpoint_signature = signature

    def _append_raw_state(
        self,
        msg: RoboStateSample,
        received_monotonic_sec: float,
        record_monotonic_sec: float,
    ) -> None:
        recorder = self._raw_recorder
        if recorder is None or recorder.closed:
            return
        sequence = self._raw_state_sequence
        self._raw_state_sequence += 1
        recorder.append_robot_state(
            _robot_state_mapping(msg),
            {
                "sequence": sequence,
                "clock_domain": "robot_state",
                "timestamp_quality": "ros_message",
                "server_wall_timestamp": time.time(),
                "receive_monotonic_timestamp": received_monotonic_sec,
                "record_monotonic_timestamp": record_monotonic_sec,
                "session_id": self._raw_session_id or f"collector:{recorder.manifest['episode_id']}",
            },
        )

    def _append_raw_event(self, name: str, details: dict[str, Any]) -> None:
        recorder = self._raw_recorder
        if recorder is None or recorder.closed:
            return
        sequence = self._raw_event_sequence
        self._raw_event_sequence += 1
        recorder.append_event(
            {"name": name, **details},
            {
                "sequence": sequence,
                "clock_domain": "collector_monotonic",
                "timestamp_quality": "host_monotonic",
                "record_monotonic_timestamp": time.monotonic(),
                "session_id": self._raw_session_id or f"collector:{recorder.manifest['episode_id']}",
            },
        )

    def _close_raw_episode(self, *, command: str, reason: str) -> None:
        recorder = self._raw_recorder
        self._raw_recorder = None
        if recorder is None:
            return
        try:
            self._append_raw_event_to(recorder, command, {"reason": reason})
        finally:
            recorder.close(command=command, reason=reason)

    def _attach_camera_spool(self, recorder: RawEpisodeRecorder) -> None:
        """Import the source-side camera prefix into this task Episode.

        The camera server owns a longer-lived session than a task Episode, so
        the host binds source records by the shared server-wall capture window
        recorded at START/STOP.  The importer snapshots each spool prefix and
        writes it through the normal RawEpisodeRecorder contract, making the
        resulting task Episode portable and independently checksummable.
        """
        if self._raw_source_scope != "camera_capture":
            return
        if self._camera_raw_spool_root is None or read_camera_spool_snapshot is None:
            raise RuntimeError(
                "camera_capture requires an accessible camera_raw_spool_root"
            )
        root = self._camera_raw_spool_root
        if not root.is_dir():
            raise RuntimeError(f"camera raw spool root is not a directory: {root}")
        start = self._raw_capture_start_wall_time
        if start is None:
            raise RuntimeError("camera capture start wall time is missing")
        start_monotonic = self._raw_capture_start_monotonic_time
        if start_monotonic is None:
            raise RuntimeError("camera capture start monotonic time is missing")
        stop = time.time()
        required_streams = tuple(self._camera_cache.streams)
        records_by_stream: dict[str, int] = {
            stream: 0 for stream in required_streams
        }
        source_entries: list[dict[str, Any]] = []
        observed_sessions = set(self._raw_camera_session_ids)
        unobserved_source_session_ids: set[str] = set()
        candidate_cutoff = start - max(self._max_episode_duration_sec, 60.0) - 5.0

        def append_unbound_source(
            source_id: str,
            session_path: Path,
            source_probe: Mapping[str, Any] | None,
            *,
            reason: str,
            source_snapshot: Mapping[str, Any] | None = None,
            binding_status: str = "UNOBSERVED",
        ) -> None:
            """Keep a source candidate auditable even without packet evidence."""
            probe = dict(source_probe) if isinstance(source_probe, Mapping) else {}
            snapshot = (
                dict(source_snapshot)
                if isinstance(source_snapshot, Mapping)
                else {}
            )
            snapshot_hash = _canonical_config_hash(
                {"manifest": probe, "snapshot": snapshot}
            )
            source_hash = probe.get("raw_manifest_hash", probe.get("manifest_hash"))
            source_entries.append(
                {
                    "session_id": str(source_id),
                    "path": str(session_path),
                    "manifest_hash": source_hash,
                    "manifest_hash_kind": (
                        "camera_source_manifest"
                        if isinstance(source_hash, str) and source_hash
                        else "unavailable_live_manifest"
                    ),
                    "source_manifest": probe,
                    "source_snapshot": snapshot,
                    "source_snapshot_hash": snapshot_hash,
                    "source_snapshot_consistent": False,
                    "observed_stream_high_watermarks": {},
                    "record_counts": {},
                    "selected_sequence_ranges": {},
                    "stream_high_watermarks": dict(
                        snapshot.get("stream_high_watermarks", {})
                        if isinstance(snapshot.get("stream_high_watermarks"), Mapping)
                        else {}
                    ),
                    "record_errors": dict(
                        probe.get("record_errors", {})
                        if isinstance(probe.get("record_errors"), Mapping)
                        else {}
                    ),
                    "status": str(probe.get("status", "UNKNOWN")),
                    "binding_status": binding_status,
                    "binding_error": str(reason),
                    "clock_offset_sec": None,
                    "clock_mapping_samples": 0,
                    "clock_mapping_uncertainty_sec": None,
                    "source_wall_window": snapshot.get("server_wall_window"),
                    "source_stop_acknowledged": False,
                }
            )
            if str(source_id) not in observed_sessions:
                unobserved_source_session_ids.add(str(source_id))

        for session_path in sorted(item for item in root.iterdir() if item.is_dir()):
            manifest_file = session_path / "manifest.json"
            progress_file = session_path / "manifest.inprogress.json"
            if not manifest_file.exists() and not progress_file.exists():
                continue
            source_id_hint = session_path.name
            try:
                source_probe = json.loads(
                    (manifest_file if manifest_file.exists() else progress_file).read_text(
                        encoding="utf-8"
                    )
                )
                if isinstance(source_probe, Mapping):
                    source_id_hint = str(
                        source_probe.get("session_id")
                        or source_probe.get("episode_id")
                        or source_id_hint
                    )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
                source_probe = {}
            try:
                session_mtime = session_path.stat().st_mtime
            except OSError:
                session_mtime = stop
            if source_id_hint not in observed_sessions and session_mtime < candidate_cutoff:
                # Historical spool sessions are unrelated unless their
                # filesystem activity overlaps this task's bounded window.
                continue
            offset_hint = list(self._camera_clock_offsets.get(source_id_hint, []))
            source_window = (
                (start - min(offset_hint), stop - min(offset_hint))
                if offset_hint
                else None
            )
            clock_offset_hint = min(offset_hint) if offset_hint else None
            selected_counts: dict[str, int] = {}
            selected_counts_all: dict[str, int] = {}
            selected_sequences: dict[str, dict[str, int | None]] = {}
            selected_sequences_all: dict[str, dict[str, int | None]] = {}
            selected_record_total = 0
            invalid_record_counts: dict[str, int] = {}
            last_source_sequences: dict[str, tuple[str, int]] = {}

            def consume_source_record(raw_record: dict[str, Any]) -> None:
                nonlocal selected_record_total
                selected_record_total += 1
                physical_stream = str(
                    raw_record.get("_spool_stream")
                    or raw_record.get("stream")
                    or ""
                )
                selected_counts_all[physical_stream] = (
                    selected_counts_all.get(physical_stream, 0) + 1
                )
                error_type = _camera_source_record_error(
                    raw_record,
                    physical_stream,
                    source_id_hint,
                )
                if error_type is not None:
                    invalid_record_counts[error_type] = (
                        invalid_record_counts.get(error_type, 0) + 1
                    )
                    self._record_raw_rejection(
                        recorder,
                        "camera_source",
                        f"source_spool_{error_type}",
                    )
                    return
                sequence = _optional_sequence(raw_record.get("sequence"))
                if sequence is None:
                    # Defensive: _camera_source_record_error already checks
                    # this, but a callback must never append an unvalidated
                    # sequence if that helper changes later.
                    self._record_raw_rejection(
                        recorder,
                        "camera_source",
                        "source_spool_invalid_sequence",
                    )
                    return
                _update_sequence_summary(
                    selected_sequences_all, physical_stream, sequence
                )
                if physical_stream not in records_by_stream:
                    return
                timestamp = raw_record.get("server_wall_timestamp")
                if timestamp is None:
                    timestamp = raw_record.get("server_wall")
                try:
                    timestamp_value = float(timestamp)
                except (TypeError, ValueError):
                    error_type = "invalid_server_wall_timestamp"
                    invalid_record_counts[error_type] = (
                        invalid_record_counts.get(error_type, 0) + 1
                    )
                    self._record_raw_rejection(
                        recorder,
                        "camera_source",
                        f"source_spool_{error_type}",
                    )
                    return
                if clock_offset_hint is None:
                    self._record_raw_rejection(
                        recorder,
                        "camera_source",
                        "source_clock_mapping_unavailable",
                    )
                    return
                mapped_host_timestamp = timestamp_value + clock_offset_hint
                if not math.isfinite(mapped_host_timestamp) or not (
                    start <= mapped_host_timestamp <= stop
                ):
                    return
                mapped_collector_monotonic = (
                    start_monotonic + (mapped_host_timestamp - start)
                )
                if not math.isfinite(mapped_collector_monotonic):
                    error_type = "invalid_mapped_collector_timestamp"
                    invalid_record_counts[error_type] = (
                        invalid_record_counts.get(error_type, 0) + 1
                    )
                    self._record_raw_rejection(
                        recorder,
                        "camera_source",
                        f"source_spool_{error_type}",
                    )
                    return
                record = dict(raw_record)
                record.pop("_spool_stream", None)
                record["stream"] = physical_stream
                record["session_id"] = str(record["session_id"])
                record["server_wall_timestamp"] = timestamp_value
                record["_mapped_receive_wall_timestamp"] = mapped_host_timestamp
                record["_mapped_receive_monotonic_timestamp"] = (
                    mapped_collector_monotonic
                )
                record["_clock_mapping_offset_sec"] = clock_offset_hint
                record["_clock_mapping_samples"] = len(offset_hint)
                records_by_stream[physical_stream] += 1
                selected_counts[physical_stream] = (
                    selected_counts.get(physical_stream, 0) + 1
                )
                _update_sequence_summary(
                    selected_sequences, physical_stream, sequence
                )
                record_session_id = str(raw_record["session_id"])
                previous_sequence = last_source_sequences.get(physical_stream)
                missing_sequence_count = (
                    max(0, sequence - previous_sequence[1] - 1)
                    if previous_sequence is not None
                    and previous_sequence[0] == record_session_id
                    else 0
                )
                last_source_sequences[physical_stream] = (
                    record_session_id,
                    sequence,
                )
                producer_gap_count = min(
                    missing_sequence_count,
                    _non_negative_count_or_none(
                        raw_record.get("producer_gap_count", 0)
                    )
                    or 0,
                )
                publisher_gap_count = min(
                    missing_sequence_count - producer_gap_count,
                    _non_negative_count_or_none(
                        raw_record.get("publisher_gap_count", 0)
                    )
                    or 0,
                )
                transport_gap_count = min(
                    missing_sequence_count
                    - producer_gap_count
                    - publisher_gap_count,
                    _non_negative_count_or_none(
                        raw_record.get("transport_gap_count", 0)
                    )
                    or 0,
                )
                unattributed_gap_count = (
                    missing_sequence_count
                    - producer_gap_count
                    - publisher_gap_count
                    - transport_gap_count
                )
                provenance = {
                    "sequence": sequence,
                    "clock_domain": _camera_clock_domain(record, physical_stream),
                    "timestamp_quality": str(
                        record.get("timestamp_quality") or "host_after_capture"
                    ),
                    "device_timestamp": record.get("device_timestamp"),
                    "device_unit": record.get(
                        "device_unit",
                        "ms" if record.get("device_timestamp") is not None else None,
                    ),
                    "timestamp_domain": record.get(
                        "timestamp_domain",
                        record.get("device_timestamp_domain_type"),
                    ),
                    "server_wall_timestamp": timestamp_value,
                    "server_monotonic_timestamp": record.get(
                        "server_monotonic_timestamp"
                    ),
                    # This is a derived host-clock mapping, not a hardware
                    # timestamp.  The original source timestamps remain in
                    # the raw record and provenance.
                    "receive_wall_timestamp": float(mapped_host_timestamp),
                    "receive_monotonic_timestamp": float(mapped_collector_monotonic),
                    "mapped_collector_monotonic_timestamp": float(
                        mapped_collector_monotonic
                    ),
                    "clock_mapping": {
                        "method": "minimum_receive_wall_minus_server_wall",
                        "offset_sec": float(clock_offset_hint),
                        "samples": len(offset_hint),
                    },
                    "session_id": record_session_id,
                }
                recorder.append_camera(
                    physical_stream,
                    record["payload"],
                    provenance,
                    payload_encoding=str(
                        record.get("payload_encoding") or "image/jpeg"
                    ),
                    serial=str(record.get("serial") or ""),
                    producer_gap_count=producer_gap_count,
                    publisher_gap_count=publisher_gap_count,
                    transport_gap_count=transport_gap_count,
                    unattributed_gap_count=unattributed_gap_count,
                )
            try:
                source_manifest, _ = read_camera_spool_snapshot(
                    session_path,
                    server_wall_window=source_window,
                    include_records=False,
                    record_callback=(consume_source_record if offset_hint else None),
                )
            except Exception as exc:
                # Use the manifest-derived id for the observed-session check;
                # a spool directory name is only a transport-side fallback and
                # may differ from the source session id.
                source_id = source_id_hint
                if source_id in observed_sessions:
                    raise RuntimeError(
                        f"camera spool session {source_id} is unreadable: {exc}"
                    ) from exc
                append_unbound_source(
                    source_id,
                    session_path,
                    source_probe,
                    reason=f"source_snapshot_unreadable: {exc}",
                    binding_status="UNREADABLE",
                )
                continue
            source_snapshot = source_manifest.pop("_snapshot", None)
            if not isinstance(source_snapshot, Mapping):
                raise RuntimeError(
                    f"camera spool session {session_path.name} returned no snapshot descriptor"
                )
            # Keep the source manifest and the exact chunk-prefix descriptor
            # together.  The source-side manifest may be in progress, so the
            # descriptor is the immutable evidence for this particular import.
            source_manifest_snapshot = dict(source_manifest)
            source_snapshot_hash = _canonical_config_hash(
                {
                    "manifest": source_manifest_snapshot,
                    "snapshot": dict(source_snapshot),
                }
            )
            source_id = str(
                source_manifest.get("session_id")
                or source_manifest.get("episode_id")
                or session_path.name
            )
            source_record_errors = source_manifest.get("record_errors")
            source_error_total: int | None = None
            if isinstance(source_record_errors, Mapping):
                source_error_total = _non_negative_count_or_none(
                    source_record_errors.get("total", 0)
                )
                source_error_counted = 0
                source_errors_by_type = source_record_errors.get(
                    "by_error_type", {}
                )
                if isinstance(source_errors_by_type, Mapping):
                    for error_type, count in source_errors_by_type.items():
                        amount = _non_negative_count_or_none(count)
                        if amount is None or amount == 0:
                            continue
                        self._record_raw_rejection(
                            recorder,
                            "camera_source",
                            f"source_spool_rejected_{error_type}",
                            count=amount,
                        )
                        source_error_counted += amount
                if source_error_total is not None and source_error_total > source_error_counted:
                    self._record_raw_rejection(
                        recorder,
                        "camera_source",
                        "source_spool_rejected_record",
                        count=source_error_total - source_error_counted,
                    )
            if source_manifest.get("status") == "QUARANTINED":
                if source_id in observed_sessions:
                    raise RuntimeError(
                        f"camera spool session {source_id} is quarantined"
                    )
                append_unbound_source(
                    source_id,
                    session_path,
                    source_manifest,
                    reason="source_session_quarantined",
                    source_snapshot=source_snapshot,
                    binding_status="QUARANTINED",
                )
                continue
            offset_samples = self._camera_clock_offsets.get(source_id, offset_hint)
            if not offset_samples:
                append_unbound_source(
                    source_id,
                    session_path,
                    source_manifest,
                    reason=(
                        "source_session_not_observed_by_collector"
                        if source_id not in observed_sessions
                        else "source_clock_mapping_unavailable"
                    ),
                    source_snapshot=source_snapshot,
                )
                continue
            # The minimum observed receive-server delta is the least network
            # latency sample and is the least-biased wall-clock offset
            # available without PTP/NTP metadata.
            clock_offset = min(offset_samples)
            snapshot_chunks = source_snapshot.get("chunks")
            snapshot_watermarks = source_snapshot.get("stream_high_watermarks")
            snapshot_record_count = source_snapshot.get(
                "selected_record_count", source_snapshot.get("record_count")
            )
            snapshot_failure_reasons: list[str] = []
            source_streams = source_manifest.get("streams")
            if not isinstance(source_streams, Mapping) or not source_streams:
                snapshot_failure_reasons.append("source_stream_schema_missing")
            else:
                # Only the strict RSP1 record contract can be bound to a
                # camera-capture Episode.  Legacy projections may still be
                # read for diagnostics, but their synthesized identity fields
                # must never make the source appear BOUND.
                for stream_name, stream_info in source_streams.items():
                    if not isinstance(stream_info, Mapping) or stream_info.get(
                        "record_schema"
                    ) != "robo_collector.camera_spool.record.v1":
                        snapshot_failure_reasons.append(
                            f"source_record_schema_missing:{stream_name}"
                        )
            if source_record_errors is not None and source_error_total is None:
                snapshot_failure_reasons.append("source_record_errors_invalid")
            elif source_error_total:
                snapshot_failure_reasons.append(
                    f"source_manifest_invalid_records:{source_error_total}"
                )
            if not isinstance(snapshot_chunks, Mapping) or not snapshot_chunks:
                snapshot_failure_reasons.append("snapshot_chunks_missing")
            if not isinstance(snapshot_watermarks, Mapping) or not snapshot_watermarks:
                snapshot_failure_reasons.append("snapshot_high_watermarks_missing")
            if (
                not isinstance(snapshot_record_count, int)
                or isinstance(snapshot_record_count, bool)
                or snapshot_record_count != selected_record_total
            ):
                snapshot_failure_reasons.append("snapshot_selected_count_mismatch")
            if source_snapshot.get("stable") is not True:
                snapshot_failure_reasons.append("snapshot_unstable")
            snapshot_selected_counts = source_snapshot.get(
                "selected_record_counts"
            )
            if not isinstance(snapshot_selected_counts, Mapping):
                snapshot_failure_reasons.append("snapshot_stream_counts_missing")
            else:
                all_streams = set(selected_counts_all) | {
                    str(stream) for stream in snapshot_selected_counts
                }
                for stream in all_streams:
                    actual_count = selected_counts_all.get(stream, 0)
                    expected_count = _non_negative_count_or_none(
                        snapshot_selected_counts.get(stream, 0)
                    )
                    if expected_count is None or expected_count != actual_count:
                        snapshot_failure_reasons.append(
                            f"snapshot_stream_count_mismatch:{stream}"
                        )
            snapshot_errors = source_snapshot.get("record_errors")
            if isinstance(snapshot_errors, Mapping):
                snapshot_error_total = _non_negative_count_or_none(
                    snapshot_errors.get("total", 0)
                )
                if snapshot_error_total:
                    snapshot_failure_reasons.append(
                        f"snapshot_invalid_records:{snapshot_error_total}"
                    )
            if invalid_record_counts:
                snapshot_failure_reasons.append(
                    "invalid_source_records:"
                    + ",".join(
                        f"{key}={value}"
                        for key, value in sorted(invalid_record_counts.items())
                    )
                )
            if not selected_counts:
                snapshot_failure_reasons.append("no_required_stream_records_in_window")
            for stream, sequence_summary in selected_sequences_all.items():
                watermark = snapshot_watermarks.get(stream)
                last_sequence = (
                    _optional_sequence(watermark.get("last_sequence"))
                    if isinstance(watermark, Mapping)
                    else None
                )
                selected_high_watermark = _optional_sequence(
                    sequence_summary.get("last_sequence")
                )
                if (
                    last_sequence is None
                    or selected_high_watermark is None
                    or int(sequence_summary.get("count", 0) or 0) <= 0
                    or last_sequence < selected_high_watermark
                ):
                    snapshot_failure_reasons.append(
                        f"snapshot_high_watermark_mismatch:{stream}"
                    )
            observed_stream_watermarks = {
                str(stream): int(sequence)
                for stream, sequence in sorted(
                    self._raw_camera_observed_high_watermarks.get(source_id, {}).items()
                )
                if isinstance(sequence, int) and not isinstance(sequence, bool)
            }
            # The receiver saw these source sequences before the lifecycle
            # boundary.  The source-side snapshot must cover that observed
            # prefix or the binding remains explicitly reviewable.
            for stream, observed_sequence in observed_stream_watermarks.items():
                watermark = (
                    snapshot_watermarks.get(stream)
                    if isinstance(snapshot_watermarks, Mapping)
                    else None
                )
                snapshot_sequence = (
                    _optional_sequence(watermark.get("last_sequence"))
                    if isinstance(watermark, Mapping)
                    else None
                )
                if snapshot_sequence is None or snapshot_sequence < observed_sequence:
                    snapshot_failure_reasons.append(
                        f"observed_high_watermark_not_covered:{stream}"
                    )
            source_hash = source_manifest_snapshot.get(
                "raw_manifest_hash", source_manifest.get("manifest_hash")
            )
            source_entries.append(
                {
                    "session_id": source_id,
                    "path": str(session_path),
                    "manifest_hash": source_hash,
                    "manifest_hash_kind": (
                        "camera_source_manifest"
                        if isinstance(source_hash, str) and source_hash
                        else "unavailable_live_manifest"
                    ),
                    "source_manifest": source_manifest_snapshot,
                    "source_snapshot": dict(source_snapshot),
                    "source_snapshot_hash": source_snapshot_hash,
                    "source_snapshot_consistent": not snapshot_failure_reasons,
                    "observed_stream_high_watermarks": observed_stream_watermarks,
                    "record_counts": selected_counts,
                    "selected_sequence_ranges": {
                        stream: {
                            "first_sequence": int(values["first_sequence"]),
                            "last_sequence": int(values["last_sequence"]),
                            "count": int(values["count"]),
                        }
                        for stream, values in selected_sequences.items()
                        if values.get("first_sequence") is not None
                        and values.get("last_sequence") is not None
                    },
                    "stream_high_watermarks": dict(
                        source_snapshot.get("stream_high_watermarks", {})
                    ),
                    "record_errors": dict(source_snapshot.get("record_errors", {}))
                    if isinstance(source_snapshot.get("record_errors"), Mapping)
                    else {},
                    "binding_status": (
                        "BOUND" if not snapshot_failure_reasons else "REVIEW"
                    ),
                    "binding_error": ";".join(snapshot_failure_reasons),
                    "failure_reasons": snapshot_failure_reasons,
                    "status": str(source_manifest.get("status", "UNKNOWN")),
                    "clock_offset_sec": clock_offset,
                    "clock_mapping_samples": len(offset_samples),
                    "clock_mapping_uncertainty_sec": (
                        max(offset_samples) - min(offset_samples)
                        if offset_samples
                        else None
                    ),
                    "source_wall_window": source_snapshot.get(
                        "server_wall_window"
                    ),
                    "source_stop_acknowledged": (
                        source_manifest.get("status") == "RAW_CLOSED"
                        and isinstance(source_manifest.get("termination"), Mapping)
                    ),
                }
            )

        bound_source_ids = {
            str(entry.get("session_id"))
            for entry in source_entries
            if entry.get("session_id")
        }
        unbound_observed_session_ids = sorted(observed_sessions - bound_source_ids)
        missing_streams = [
            stream for stream, values in records_by_stream.items() if not values
        ]
        if missing_streams:
            self.get_logger().warn(
                "camera spool has no records in task window for: "
                + ",".join(missing_streams)
            )
        binding_reasons = []
        if missing_streams:
            binding_reasons.append(
                "missing_streams:" + ",".join(missing_streams)
            )
        if unbound_observed_session_ids:
            binding_reasons.append(
                "unbound_observed_sessions:"
                + ",".join(unbound_observed_session_ids)
            )
        if unobserved_source_session_ids:
            binding_reasons.append(
                "unobserved_source_sessions:"
                + ",".join(sorted(unobserved_source_session_ids))
            )
        for entry in source_entries:
            if entry.get("binding_status") != "BOUND":
                reason = str(entry.get("binding_error") or "source_binding_incomplete")
                binding_reasons.append(
                    f"source_binding_incomplete:{entry.get('session_id', '')}:{reason}"
                )
        binding_status = "ATTACHED" if not binding_reasons else "REVIEW"
        recorder.update_metadata(
            {
                "camera_capture_attached": True,
                "camera_capture_window": {
                    "start_wall_time": start,
                    "end_wall_time": stop,
                },
                "camera_capture_sources": source_entries,
                "camera_capture_binding": {
                    "schema": "robo_collector.camera_capture_binding.v1",
                    "status": binding_status,
                    "binding_method": "camera_source_snapshot_window_v1",
                    "camera_raw_spool_root": str(root),
                    "observed_session_ids": sorted(observed_sessions),
                    "unbound_observed_session_ids": unbound_observed_session_ids,
                    "unobserved_source_session_ids": sorted(
                        unobserved_source_session_ids
                    ),
                    "missing_streams": missing_streams,
                    "failure_reasons": binding_reasons,
                    "packet_high_watermarks": {
                        session_id: int(sequence)
                        for session_id, sequence in sorted(
                            self._raw_camera_packet_high_watermarks.items()
                        )
                    },
                    "stream_high_watermarks": {
                        session_id: {
                            stream: int(sequence)
                            for stream, sequence in sorted(streams.items())
                        }
                        for session_id, streams in sorted(
                            self._raw_camera_observed_high_watermarks.items()
                        )
                    },
                    "clock_mapping": {
                        str(entry["session_id"]): {
                            "offset_sec": entry["clock_offset_sec"],
                            "samples": entry["clock_mapping_samples"],
                            "uncertainty_sec": entry.get(
                                "clock_mapping_uncertainty_sec"
                            ),
                            "method": "minimum_receive_wall_minus_server_wall",
                        }
                        for entry in source_entries
                    },
                    "source_snapshots": [
                        {
                            "session_id": entry["session_id"],
                            "manifest_hash": entry.get("manifest_hash"),
                            "source_snapshot_hash": entry["source_snapshot_hash"],
                            "source_snapshot_consistent": entry.get(
                                "source_snapshot_consistent", False
                            ),
                            "stable": bool(
                                entry.get("source_snapshot", {}).get("stable", False)
                                if isinstance(entry.get("source_snapshot"), Mapping)
                                else False
                            ),
                            "server_wall_window": entry.get("source_wall_window"),
                            "observed_stream_high_watermarks": entry.get(
                                "observed_stream_high_watermarks", {}
                            ),
                            "stream_high_watermarks": entry.get(
                                "stream_high_watermarks", {}
                            ),
                        }
                        for entry in source_entries
                    ],
                    "source_stop_acknowledged": all(
                        bool(entry.get("source_stop_acknowledged"))
                        for entry in source_entries
                    ),
                },
                "camera_capture_record_counts": {
                    stream: count for stream, count in records_by_stream.items()
                },
                "camera_capture_clock_mapping": {
                    str(entry["session_id"]): {
                        "offset_sec": entry["clock_offset_sec"],
                        "samples": entry["clock_mapping_samples"],
                        "uncertainty_sec": entry.get(
                            "clock_mapping_uncertainty_sec"
                        ),
                        "method": "minimum_receive_wall_minus_server_wall",
                    }
                    for entry in source_entries
                },
            }
        )
        self._raw_camera_capture_attached = True

    @staticmethod
    def _raw_gap_statistics(manifest: Mapping[str, Any]) -> dict[str, int]:
        """Aggregate gap counters from the records actually written to Raw."""
        totals = {
            "producer_gaps": 0,
            "publisher_gaps": 0,
            "transport_gaps": 0,
            "unattributed_gaps": 0,
        }
        streams = manifest.get("streams", {})
        if not isinstance(streams, Mapping):
            return totals
        for stream in streams.values():
            if not isinstance(stream, Mapping):
                continue
            for output_name, field_name in (
                ("producer_gaps", "producer_gap_count"),
                ("publisher_gaps", "publisher_gap_count"),
                ("transport_gaps", "transport_gap_count"),
                ("unattributed_gaps", "unattributed_gap_count"),
            ):
                value = stream.get(field_name, 0)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    totals[output_name] += value
        return totals

    def _freeze_raw_capture(self, *, command: str, reason: str) -> None:
        """Freeze per-episode quality, detach callbacks, then seal raw."""
        recorder = self._raw_recorder
        if recorder is not None and self._raw_source_scope == "camera_capture":
            self._attach_camera_spool(recorder)
        self._frozen_quality = {
            "camera_stats": self._camera_cache.stats,
            "camera_quality": self._camera_cache.quality_statistics,
            "state_camera_skew_samples": list(self._state_camera_skew_samples),
            "state_camera_skew": _residual_statistics(self._state_camera_skew_samples),
        }
        self._close_raw_episode(command=command, reason=reason)
        path = self._raw_episode_path
        quality = self._frozen_quality
        if path is None or quality is None or not (path / "manifest.json").exists():
            return
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        stats = quality["camera_stats"]
        raw_gap_stats = {
            "producer_gaps": int(stats.get("producer_gap", 0) or 0),
            "publisher_gaps": int(stats.get("publisher_gap", 0) or 0),
            "transport_gaps": int(stats.get("transport_gap", 0) or 0),
            "unattributed_gaps": int(stats.get("unattributed_gap", 0) or 0),
        }
        if self._raw_source_scope == "camera_capture":
            raw_stats = self._raw_gap_statistics(manifest)
            raw_gap_stats.update(raw_stats)
        camera_quality = quality["camera_quality"]
        recording_failed = bool(
            self._raw_recording_failed_reason
            or (
                camera_quality.get("recording_failed", False)
                if isinstance(camera_quality, Mapping)
                else False
            )
        )
        manifest["quality"] = {"statistics": {
            **(
                {
                    key: value
                    for key, value in camera_quality.items()
                    if key not in {
                        "producer_gaps",
                        "publisher_gaps",
                        "transport_gaps",
                        "unattributed_gaps",
                    }
                }
                if isinstance(camera_quality, Mapping)
                else {}
            ),
            "producer_gaps": raw_gap_stats["producer_gaps"],
            "publisher_gaps": raw_gap_stats["publisher_gaps"],
            "transport_gaps": raw_gap_stats["transport_gaps"],
            "unattributed_gaps": raw_gap_stats["unattributed_gaps"],
            "selection_gaps": stats.get("selection_gap", 0),
            "state_camera_skew_samples": quality["state_camera_skew_samples"],
            "state_camera_skew": quality["state_camera_skew"],
            "recording": {
                "timer_ticks": self._record_tick_count,
                "timer_deadline_misses": self._timer_deadline_misses,
                "raw_frame_count": self._raw_frame_count,
                "process_rss_bytes": _process_rss_bytes(),
                "process_peak_rss_bytes": _process_peak_rss_bytes(),
                "recording_failed": recording_failed,
                "failure_reason": self._raw_recording_failed_reason,
                "state_age_sec": _residual_statistics(self._state_age_samples),
                "camera_age_sec": _residual_statistics(self._camera_age_samples),
                "state_age_max_sec": _residual_statistics(
                    self._state_age_samples
                ).get("max"),
                "camera_age_max_sec": _residual_statistics(
                    self._camera_age_samples
                ).get("max"),
                "max_state_age_sec": self._max_state_age_sec,
                "max_camera_age_sec": self._max_camera_age_sec,
            },
        }}
        manifest.setdefault("metadata", {})["stop_frozen"] = True
        raw_hash = _hash_manifest(manifest)
        manifest["manifest_hash"] = raw_hash
        manifest["raw_manifest_hash"] = raw_hash
        _write_json_atomic(path / "manifest.json", manifest)
        quality_report = EpisodeQualityGate(
            require_complete_capture=self._raw_source_scope == "camera_capture",
            thresholds=self._quality_thresholds,
            max_camera_camera_skew_sec=self._max_inter_camera_skew_sec,
            max_state_camera_skew_sec=self._max_state_camera_skew_sec,
            max_camera_clock_mapping_uncertainty_sec=(
                self._max_camera_clock_mapping_uncertainty_sec
            ),
            max_state_age_sec=self._max_state_age_sec,
            max_camera_age_sec=self._max_camera_age_sec,
        ).evaluate(path / "manifest.json")
        write_quality_report(path, quality_report)
        # A host receiver recording is an auditable shadow only.  It must never
        # enqueue a derived-artifact job claiming complete camera capture.
        if self._recording_mode == "raw_first" and manifest.get("status") != "QUARANTINED":
            self._raw_materialization_job = create_materialization_job(
                path,
                self._materialization_job_config(),
                "lerobot.v2.1.raw_materialization.v1",
            )

    def _materialization_config(self, config: Any) -> MaterializationConfig:
        values = config if isinstance(config, dict) else self._raw_conversion_config()
        field_selection = values.get("field_selection")
        if field_selection is not None and not isinstance(field_selection, Mapping):
            raise ValueError("materialization field_selection must be a mapping")
        quality_thresholds = values.get("quality_thresholds", values.get("quality"))
        if quality_thresholds is not None and not isinstance(quality_thresholds, Mapping):
            raise ValueError("materialization quality_thresholds must be a mapping")
        return MaterializationConfig(
            output_root=Path(str(values.get("output_root", self._writer.root_output_dir))),
            dataset_name=str(values.get("dataset_name", self._writer.dataset_name)),
            fps=int(values.get("fps", self._fps)),
            camera_streams=tuple(values.get("camera_streams", self._camera_cache.streams)),
            alignment_policy=str(values.get("alignment_policy", "strict")),
            max_alignment_residual_sec=float(values.get("max_alignment_residual_sec", self._max_state_camera_skew_sec)),
            max_camera_clock_mapping_uncertainty_sec=_optional_configured_float(
                values.get(
                    "max_camera_clock_mapping_uncertainty_sec",
                    self._max_camera_clock_mapping_uncertainty_sec,
                )
            ),
            output_schema_version=str(values.get("output_schema_version", "lerobot.v2.1.raw_materialization.v1")),
            require_complete_capture=bool(values.get("require_complete_capture", self._raw_source_scope == "camera_capture")),
            max_state_age_sec=_optional_configured_float(
                values.get("max_state_age_sec", self._max_state_age_sec)
            ),
            max_camera_age_sec=_optional_configured_float(
                values.get("max_camera_age_sec", self._max_camera_age_sec)
            ),
            max_timer_deadline_misses=_configured_non_negative_int(
                values.get("max_timer_deadline_misses", 0),
                "max_timer_deadline_misses",
            ),
            field_selection=dict(field_selection) if isinstance(field_selection, Mapping) else None,
            quality_thresholds=(
                dict(quality_thresholds)
                if isinstance(quality_thresholds, Mapping)
                else None
            ),
        )

    def _materialization_job_config(self) -> dict[str, Any]:
        return {
            "output_root": str(self._writer.root_output_dir),
            "dataset_name": self._writer.dataset_name,
            **self._raw_conversion_config(),
            "max_alignment_residual_sec": self._max_state_camera_skew_sec,
            "output_schema_version": "lerobot.v2.1.raw_materialization.v1",
            "require_complete_capture": self._raw_source_scope == "camera_capture",
            "max_camera_clock_mapping_uncertainty_sec": (
                self._max_camera_clock_mapping_uncertainty_sec
            ),
            "max_state_age_sec": self._max_state_age_sec,
            "max_camera_age_sec": self._max_camera_age_sec,
            "max_timer_deadline_misses": 0,
            "field_selection": {
                "target": list(self._field_selection.target),
                "state": list(self._field_selection.state),
                "include_policy_action": self._field_selection.include_policy_action,
            },
            "quality_thresholds": dict(self._quality_thresholds),
        }

    def _append_raw_event_to(
        self, recorder: RawEpisodeRecorder, name: str, details: dict[str, Any]
    ) -> None:
        sequence = self._raw_event_sequence
        self._raw_event_sequence += 1
        recorder.append_event(
            {"name": name, **details},
            {
                "sequence": sequence,
                "clock_domain": "collector_monotonic",
                "timestamp_quality": "host_monotonic",
                "record_monotonic_timestamp": time.monotonic(),
                "session_id": self._raw_session_id or f"collector:{recorder.manifest['episode_id']}",
            },
        )

    def _discard_raw_episode(self, *, reason: str) -> None:
        recorder = self._raw_recorder
        self._raw_recorder = None
        if recorder is not None:
            self._append_raw_event_to(recorder, "DISCARD", {"reason": reason})
            recorder.discard(reason=reason)
        elif self._raw_episode_path is not None and (
            self._raw_episode_path / "manifest.json"
        ).exists():
            manifest = json.loads(
                (self._raw_episode_path / "manifest.json").read_text(encoding="utf-8")
            )
            if manifest.get("status") != "QUARANTINED":
                discard_sealed_episode(self._raw_episode_path, reason=reason)
        self._raw_materialization_job = None

    def _raw_conversion_config(self) -> dict[str, Any]:
        return {
            "fps": self._fps,
            "camera_streams": list(self._camera_cache.streams),
            "alignment_policy": "strict",
            "source_scope": self._raw_source_scope,
            "require_complete_capture": self._raw_source_scope == "camera_capture",
            "max_camera_clock_mapping_uncertainty_sec": (
                self._max_camera_clock_mapping_uncertainty_sec
            ),
            "camera_raw_spool_root": (
                str(self._camera_raw_spool_root)
                if self._camera_raw_spool_root is not None
                else ""
            ),
            "field_selection": {
                "target": list(self._field_selection.target),
                "state": list(self._field_selection.state),
                "include_policy_action": self._field_selection.include_policy_action,
            },
            "quality_thresholds": dict(self._quality_thresholds),
        }

    def _record_stale_camera_bundle(self, bundle: Any) -> None:
        """Count one expired bundle per distinct cached camera identity."""
        identity = getattr(bundle, "identity", None)
        if identity == self._last_stale_camera_identity:
            return
        self._last_stale_camera_identity = identity
        recorder = getattr(self._camera_cache, "record_stale", None)
        if callable(recorder):
            recorder()

    def _mark_no_materialized_artifact(self) -> None:
        path = self._raw_episode_path
        if path is None or not (path / "manifest.json").exists():
            return
        try:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            manifest["status"] = "REJECT"
            _write_json_atomic(path / "manifest.json", manifest)
            write_quality_report(path, {
                "status": "REJECT",
                "reason": ["no_materialized_artifact"],
                "episode_id": manifest.get("episode_id"),
            })
        except Exception as exc:
            self.get_logger().error(f"failed to record no_materialized_artifact: {exc}")

    def _run_recovery_jobs(self) -> None:
        """Retry startup jobs while excluding live START admission."""
        try:
            self._run_recovery_jobs_impl()
        except Exception as exc:
            with self._lifecycle_lock:
                self._recovery_error = f"raw materialization recovery failed: {exc}"
                # Keep START fail-closed if the recovery coordinator itself
                # fails before it can establish a durable terminal state.
                self._recovery_active = True
            self.get_logger().error(self._recovery_error)
            return
        else:
            with self._lifecycle_lock:
                self._recovery_active = False
                self._recovery_error = None

    def _run_recovery_jobs_impl(self) -> None:
        """Retry durable jobs at least once, only while the live writer is idle."""
        failures: list[str] = []
        for job in list(self._recovery_jobs):
            if self._recovery_stop.is_set():
                return
            while not self._recovery_stop.is_set():
                with self._lifecycle_lock:
                    idle = (
                        self._state_machine.mode == CollectorMode.IDLE
                        and self._writer.active_episode_index is None
                        and not self._save_worker.has_active
                    )
                if idle:
                    break
                self._recovery_stop.wait(0.5)
            if self._recovery_stop.is_set():
                return
            try:
                path = Path(str(job["episode"]))
                manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
                saved_config = job.get("conversion_config")
                metadata = manifest.get("metadata", {})
                if not isinstance(saved_config, dict):
                    saved_config = metadata.get("conversion_config")
                config = saved_config if isinstance(saved_config, dict) else self._raw_conversion_config()
                materializer_config = self._materialization_config(config)
                with self._lifecycle_lock:
                    if self._state_machine.mode != CollectorMode.IDLE or self._writer.active_episode_index is not None or self._save_worker.has_active:
                        continue
                    self._save_worker.start(
                        lambda report_progress: RawEpisodeMaterializer(materializer_config).materialize(
                            path, progress_callback=report_progress
                        )
                    )
                try:
                    self._save_worker.take_result(timeout=self._save_shutdown_grace_sec)
                except TimeoutError:
                    # A timed-out wait is only a deadline miss.  The worker is
                    # still the single owner of the job; reap its result once
                    # it finishes so ``has_active`` cannot permanently block
                    # the next recovery or live save.
                    self.get_logger().warn(
                        "raw materialization recovery exceeded shutdown grace; "
                        "waiting to reap the active worker"
                    )
                    self._save_worker.take_result()
            except Exception as exc:
                failure = (
                    f"raw materialization recovery failed for "
                    f"{job.get('episode')}: {exc}"
                )
                failures.append(failure)
                self.get_logger().error(failure)
        if failures:
            raise RuntimeError("; ".join(failures))

    def _mark_raw_materialization_failed(self, reason: str) -> None:
        job = self._raw_materialization_job
        path = self._raw_episode_path
        if path is None:
            return
        try:
            if job is not None:
                update_materialization_job(
                    path,
                    str(job["job_id"]),
                    "FAILED",
                    error=reason,
                )
            manifest = json.loads(
                (path / "manifest.json").read_text(encoding="utf-8")
            )
            report = EpisodeQualityGate(
                require_complete_capture=self._raw_source_scope == "camera_capture",
                thresholds=self._quality_thresholds,
                max_camera_camera_skew_sec=self._max_inter_camera_skew_sec,
                max_state_camera_skew_sec=self._max_state_camera_skew_sec,
                max_camera_clock_mapping_uncertainty_sec=(
                    self._max_camera_clock_mapping_uncertainty_sec
                ),
                max_state_age_sec=self._max_state_age_sec,
                max_camera_age_sec=self._max_camera_age_sec,
            ).evaluate(manifest)
            report["status"] = "REJECT"
            report.setdefault("reason", []).append(
                f"materialization_failed: {reason}"
            )
            write_quality_report(path, report)
            # Keep MATERIALIZATION_FAILED as the durable retry state.  A
            # conversion failure is not a raw-capture rejection; startup
            # recovery must be able to claim the same job again.
        except Exception as exc:
            self.get_logger().error(f"failed to update raw job failure: {exc}")

    def _mark_raw_materialization_succeeded(self, result: SaveResult) -> str:
        job = self._raw_materialization_job
        path = self._raw_episode_path
        if job is None or path is None:
            raise RuntimeError(
                "raw materialization result has no durable episode or job"
            )
        if isinstance(result, MaterializationResult):
            current_manifest = json.loads(
                (path / "manifest.json").read_text(encoding="utf-8")
            )
            existing_artifacts = current_manifest.get("artifacts")
            artifacts = (
                dict(existing_artifacts)
                if isinstance(existing_artifacts, Mapping)
                else {
                    "output_dataset": (
                        str(result.output_dataset) if result.output_dataset else None
                    ),
                    "frame_count": result.frame_count,
                    "dropped_selection_count": result.dropped_selection_count,
                    "output_schema_version": "lerobot.v2.1.raw_materialization.v1",
                }
            )
        else:
            artifacts = {
                "data": str(result.data_path) if result.data_path else None,
                "videos": {key: str(value) for key, value in result.video_paths.items()},
                "frame_count": result.frame_count,
                "output_schema_version": "lerobot.v2.1.legacy_writer.v1",
            }
        try:
            update_materialization_job(
                path,
                str(job["job_id"]),
                "MATERIALIZED",
                artifacts=artifacts,
            )
            frozen_quality = self._frozen_quality or {}
            stats = frozen_quality.get("camera_stats") or self._camera_cache.stats
            camera_quality = (
                frozen_quality.get("camera_quality")
                or self._camera_cache.quality_statistics
            )
            state_quality = _residual_statistics(self._state_camera_skew_samples)
            recorder_manifest = json.loads(
                (path / "manifest.json").read_text(encoding="utf-8")
            )
            existing_quality = recorder_manifest.get("quality")
            existing_stats = (
                existing_quality.get("statistics", existing_quality)
                if isinstance(existing_quality, Mapping)
                else {}
            )
            merged_stats = (
                dict(existing_stats) if isinstance(existing_stats, Mapping) else {}
            )
            raw_gap_stats = self._raw_gap_statistics(recorder_manifest)
            # Capture-side counters are authoritative for producer/transport
            # gaps.  Materialization-side selection gaps and alignment metrics
            # are already present in ``existing_stats``; preserve them instead
            # of replacing them with the receiver's pre-materialization zero.
            # Reading these counters back from the sealed Raw manifest also
            # keeps camera_capture source statistics from being overwritten by
            # the host receiver cache snapshot.
            merged_stats.update(
                {
                    "producer_gaps": raw_gap_stats["producer_gaps"],
                    "publisher_gaps": raw_gap_stats["publisher_gaps"],
                    "transport_gaps": raw_gap_stats["transport_gaps"],
                    "unattributed_gaps": raw_gap_stats["unattributed_gaps"],
                    "camera_camera_skew_sec": camera_quality.get(
                        "camera_camera_skew_sec",
                        merged_stats.get("camera_camera_skew_sec"),
                    ),
                    "camera_camera_skew": camera_quality.get(
                        "camera_camera_skew",
                        merged_stats.get("camera_camera_skew"),
                    ),
                    "state_camera_skew_sec": state_quality["max"],
                    "state_camera_skew": state_quality,
                    "video_frame_counts": (
                        {
                            key: result.frame_count
                            for key in result.video_paths
                        }
                        if hasattr(result, "video_paths")
                        else dict(merged_stats.get("video_frame_counts", {}))
                    ),
                    "parquet_row_count": result.frame_count,
                }
            )
            if "selection_gaps" not in merged_stats:
                merged_stats["selection_gaps"] = stats.get("selection_gap", 0)
            if "selection_gaps_by_stream" not in merged_stats:
                merged_stats["selection_gaps_by_stream"] = {}
            # Queue and per-stream transport details are a capture-time
            # snapshot; merge them without deleting materializer statistics.
            if isinstance(camera_quality, Mapping):
                for key, value in camera_quality.items():
                    if key not in {
                        "camera_camera_skew_sec",
                        "camera_camera_skew",
                        "producer_gaps",
                        "publisher_gaps",
                        "transport_gaps",
                        "unattributed_gaps",
                    }:
                        merged_stats[key] = value
            recorder_manifest["quality"] = {"statistics": merged_stats}
            _write_json_atomic(path / "manifest.json", recorder_manifest)
            gate = EpisodeQualityGate(
                require_complete_capture=self._raw_source_scope == "camera_capture",
                thresholds=self._quality_thresholds,
                max_camera_camera_skew_sec=self._max_inter_camera_skew_sec,
                max_state_camera_skew_sec=self._max_state_camera_skew_sec,
                max_camera_clock_mapping_uncertainty_sec=(
                    self._max_camera_clock_mapping_uncertainty_sec
                ),
                max_state_age_sec=self._max_state_age_sec,
                max_camera_age_sec=self._max_camera_age_sec,
            )
            report = gate.evaluate(path / "manifest.json")
            write_quality_report(path, report)
            recorder_manifest["status"] = "QC"
            _write_json_atomic(path / "manifest.json", recorder_manifest)
            recorder_manifest["status"] = report["status"]
            _write_json_atomic(path / "manifest.json", recorder_manifest)
            return str(report["status"]).strip().upper()
        except Exception as exc:
            self.get_logger().error(f"raw materialization QC update failed: {exc}")
            raise RuntimeError("raw materialization QC update failed") from exc

    def _record_state_camera_skew_sample(self, value: float) -> None:
        if not math.isfinite(value) or value < 0:
            return
        if len(self._state_camera_skew_samples) < 4096:
            self._state_camera_skew_samples.append(value)
        else:
            self._state_camera_skew_samples[self._state_camera_skew_sample_cursor] = value
            self._state_camera_skew_sample_cursor = (
                self._state_camera_skew_sample_cursor + 1
            ) % len(self._state_camera_skew_samples)

    def _record_capture_age_sample(self, state_age: float, camera_age: float) -> None:
        """Keep bounded age evidence for fail-closed post-capture QC."""
        if math.isfinite(state_age) and state_age >= 0:
            if len(self._state_age_samples) < 4096:
                self._state_age_samples.append(state_age)
            else:
                self._state_age_samples[self._state_age_sample_cursor] = state_age
                self._state_age_sample_cursor = (
                    self._state_age_sample_cursor + 1
                ) % len(self._state_age_samples)
        if math.isfinite(camera_age) and camera_age >= 0:
            if len(self._camera_age_samples) < 4096:
                self._camera_age_samples.append(camera_age)
            else:
                self._camera_age_samples[self._camera_age_sample_cursor] = camera_age
                self._camera_age_sample_cursor = (
                    self._camera_age_sample_cursor + 1
                ) % len(self._camera_age_samples)

    def _publish_periodic_status(self) -> None:
        self._drain_save_progress()
        message = self._state_machine.mode.value
        level = DiagnosticStatus.OK
        if self._state_machine.mode == CollectorMode.RECORDING:
            message = (
                f"RECORDING episode={self._writer.active_episode_index} "
                f"frames={self._writer.active_frame_count if self._recording_mode == 'legacy' else self._raw_frame_count}"
            )
        elif self._state_machine.mode == CollectorMode.SAVING:
            message = self._saving_status_message()
        elif self._state_machine.mode == CollectorMode.FAILED:
            message = (
                "FAILED: DISCARD required: "
                f"{self._state_machine.failure_reason}"
            )
            level = DiagnosticStatus.ERROR
        warnings = [
            warning
            for warning in (self._state_warning(), self._camera_warning())
            if warning is not None
        ]
        if warnings:
            if level != DiagnosticStatus.ERROR:
                level = DiagnosticStatus.WARN
            message = f"{message}; {'; '.join(warnings)}"
        self._publish_status(level, message)

    def _drain_save_progress(self) -> None:
        for progress in self._save_worker.drain_progress():
            self._save_phase = progress.phase
            self._save_progress_monotonic_sec = progress.monotonic_sec
            self._save_progress_seq += 1

    def _clear_save_tracking(self) -> None:
        self._save_started_monotonic_sec = None
        self._save_finished_monotonic_sec = None
        self._save_progress_monotonic_sec = None
        self._save_progress_seq = 0
        self._save_phase = ""
        self._saving_episode_index = None
        self._saving_frame_count = 0

    def _saving_status_message(self) -> str:
        episode = (
            ""
            if self._saving_episode_index is None
            else str(self._saving_episode_index)
        )
        return (
            f"SAVING episode={episode} frames={self._saving_frame_count} "
            f"phase={self._save_phase or 'queued'} "
            f"elapsed={self._save_elapsed_sec():.3f}s"
        )

    def _save_elapsed_sec(self) -> float:
        started = self._save_started_monotonic_sec
        if started is None:
            return 0.0
        finished = self._save_finished_monotonic_sec
        end = finished if finished is not None else time.monotonic()
        return max(0.0, end - started)

    def _publish_failed_status_throttled(self) -> None:
        self._publish_status_throttled(
            DiagnosticStatus.ERROR,
            "FAILED: DISCARD required: "
            f"{self._state_machine.failure_reason}",
        )

    def _publish_warn_throttled(self, message: str) -> None:
        self._publish_status_throttled(DiagnosticStatus.WARN, message)

    def _publish_status_throttled(self, level: Any, message: str) -> None:
        now = time.monotonic()
        if message != self._last_warn_message or now - self._last_warn_monotonic_sec > 1.0:
            self._last_warn_message = message
            self._last_warn_monotonic_sec = now
            self._publish_status(level, message)

    def _log_state_sample_received_throttled(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if (
            not force
            and self._state_sample_count != 1
            and now - self._last_state_sample_log_monotonic_sec <= 5.0
        ):
            return
        self._last_state_sample_log_monotonic_sec = now
        self.get_logger().info(
            "receiving robo_state samples on "
            f"{self._robo_state_topic}: count={self._state_sample_count}"
        )

    def _publish_status(self, level: Any, message: str) -> None:
        state_age = ""
        if self._latest_state is not None:
            state_age = (
                f"{time.monotonic() - self._latest_state.received_monotonic_sec:.3f}"
            )
        if self._state_machine.mode == CollectorMode.SAVING:
            active_episode = self._saving_episode_index
            active_frames = self._saving_frame_count
        else:
            active_episode = self._writer.active_episode_index
            active_frames = (
                self._writer.active_frame_count
                if self._recording_mode == "legacy"
                else self._raw_frame_count
            )
        save_progress_age = ""
        if self._save_progress_monotonic_sec is not None:
            save_progress_age = (
                f"{max(0.0, time.monotonic() - self._save_progress_monotonic_sec):.3f}"
            )
        status = DiagnosticStatus()
        status.level = _diagnostic_level_value(level)
        status.name = self.get_name()
        status.hardware_id = "robo_collector"
        status.message = message
        status.values = [
            KeyValue(key="mode", value=self._state_machine.mode.value),
            KeyValue(key="dataset_root", value=str(self._writer.root)),
            KeyValue(
                key="episode_id",
                value=(
                    self._state_machine.session.episode_id
                    if self._state_machine.session is not None
                    else ""
                ),
            ),
            KeyValue(
                key="task_prompt",
                value=(
                    self._state_machine.session.task_prompt
                    if self._state_machine.session is not None
                    else ""
                ),
            ),
            KeyValue(key="last_command_id", value=self._last_command_id),
            KeyValue(key="last_command", value=self._last_command),
            KeyValue(
                key="last_command_outcome", value=self._last_command_outcome
            ),
            KeyValue(
                key="last_command_episode_id",
                value=self._last_command_episode_id,
            ),
            KeyValue(key="last_episode_id", value=self._last_episode_id),
            KeyValue(
                key="last_episode_outcome", value=self._last_episode_outcome
            ),
            KeyValue(key="fps", value=str(self._fps)),
            KeyValue(key="recording_mode", value=self._recording_mode),
            KeyValue(key="raw_source_scope", value=self._raw_source_scope),
            KeyValue(key="raw_requested_source_scope", value=self._requested_raw_source_scope),
            KeyValue(key="camera_callback_queue_size", value=str(self._camera_callback_queue_size)),
            KeyValue(key="active_episode", value=str(active_episode)),
            KeyValue(key="active_frames", value=str(active_frames)),
            KeyValue(key="save_phase", value=self._save_phase),
            KeyValue(key="save_progress_seq", value=str(self._save_progress_seq)),
            KeyValue(key="save_elapsed_sec", value=f"{self._save_elapsed_sec():.3f}"),
            KeyValue(key="save_progress_age_sec", value=save_progress_age),
            KeyValue(
                key="process_rss_bytes",
                value=_metric_string(_process_rss_bytes()),
            ),
            KeyValue(
                key="process_peak_rss_bytes",
                value=_metric_string(_process_peak_rss_bytes()),
            ),
            KeyValue(key="max_state_age_sec", value=str(self._max_state_age_sec)),
            KeyValue(key="max_camera_age_sec", value=str(self._max_camera_age_sec)),
            KeyValue(
                key="robo_state_available",
                value=str(self._latest_state is not None),
            ),
            KeyValue(key="robo_state_age_sec", value=state_age),
            KeyValue(key="camera_streams", value=",".join(self._camera_cache.streams)),
            KeyValue(key="camera_error", value=self._camera_cache.last_error),
        ]
        self._status_pub.publish(status)
        self._log_status_issue_throttled(status.level, message)

    def _set_command_receipt(
        self,
        *,
        command_id: str,
        command: str,
        episode_id: str,
        outcome: str,
    ) -> None:
        self._last_command_id = command_id
        self._last_command = command
        self._last_command_episode_id = episode_id
        self._last_command_outcome = outcome

    def _log_status_issue_throttled(self, level: Any, message: str) -> None:
        level_value = _diagnostic_level_value(level)
        if level_value == _diagnostic_level_value(DiagnosticStatus.OK):
            return

        now = time.monotonic()
        if (
            message == self._last_status_log_message
            and now - self._last_status_log_monotonic_sec <= 5.0
        ):
            return

        self._last_status_log_message = message
        self._last_status_log_monotonic_sec = now
        if level_value == _diagnostic_level_value(DiagnosticStatus.ERROR):
            self.get_logger().error(message)
        else:
            self.get_logger().warn(message)

    def _state_warning(self) -> str | None:
        state = self._latest_state
        if state is None:
            return "robo_state sample unavailable; check robo_state_node status"
        state_age = time.monotonic() - state.received_monotonic_sec
        if state_age > self._max_state_age_sec:
            return f"stale robo_state sample: {state_age:.3f}s old; check robo_state_node"
        return None

    def _camera_warning(self) -> str | None:
        camera_error = self._camera_cache.last_error
        if camera_error:
            return camera_error
        camera_bundle = self._camera_cache.latest()
        if camera_bundle is None:
            return "complete camera frame bundle unavailable"
        camera_age = time.monotonic() - camera_bundle.received_monotonic_sec
        if camera_age > self._max_camera_age_sec:
            return f"stale camera frame bundle: {camera_age:.3f}s old"
        return None

    def _now_sec(self) -> float:
        return time.monotonic()


def _robot_frame_from_msg(msg: RoboStateSample) -> RobotFrame:
    imu = msg.imu
    policy_state = msg.policy_state
    return RobotFrame(
        joint_position=[float(value) for value in msg.robot_state.joint_pos],
        joint_velocity=[float(value) for value in msg.robot_state.joint_vel],
        joint_torque=[float(value) for value in msg.robot_state.joint_torque],
        imu_angular_velocity=[
            float(imu.angular_velocity.x),
            float(imu.angular_velocity.y),
            float(imu.angular_velocity.z),
        ],
        imu_linear_acceleration=[
            float(imu.linear_acceleration.x),
            float(imu.linear_acceleration.y),
            float(imu.linear_acceleration.z),
        ],
        projected_gravity_or_quat=[
            float(imu.orientation.x),
            float(imu.orientation.y),
            float(imu.orientation.z),
            float(imu.orientation.w),
        ],
        target_joint_pos=[float(value) for value in msg.target_joint_pos],
        policy_action=[float(value) for value in msg.action],
        aligned_target_pos=[float(value) for value in msg.aligned_target_pos],
        policy_state={
            "relative_ori_6d": [
                float(value) for value in policy_state.relative_ori_6d
            ],
            "motion_anchor_lin_vel_b": [
                float(value) for value in policy_state.motion_anchor_lin_vel_b
            ],
            "motion_anchor_ang_vel_b": [
                float(value) for value in policy_state.motion_anchor_ang_vel_b
            ],
            "ang_vel_history": [
                float(value) for value in policy_state.ang_vel_history
            ],
            "gravity_history": [
                float(value) for value in policy_state.gravity_history
            ],
            "joint_pos_rel_history": [
                float(value) for value in policy_state.joint_pos_rel_history
            ],
            "joint_vel_history": [
                float(value) for value in policy_state.joint_vel_history
            ],
            "action_history": [
                float(value) for value in policy_state.action_history
            ],
        },
        joint_names=list(msg.robot_state.joint_names),
        state_timestamp_sec=message_stamp_sec(msg),
    )


def _robot_state_mapping(msg: RoboStateSample) -> dict[str, Any]:
    """Convert a ROS state sample to the JSON-safe raw state contract."""
    frame = _robot_frame_from_msg(msg)
    return {
        "joint_position": frame.joint_position,
        "joint_velocity": frame.joint_velocity,
        "joint_torque": frame.joint_torque,
        "imu_angular_velocity": frame.imu_angular_velocity,
        "imu_linear_acceleration": frame.imu_linear_acceleration,
        "projected_gravity_or_quat": frame.projected_gravity_or_quat,
        "target_joint_pos": frame.target_joint_pos,
        "policy_action": frame.policy_action,
        "aligned_target_pos": frame.aligned_target_pos,
        "policy_state": frame.policy_state,
        "joint_names": frame.joint_names,
        "state_timestamp_sec": frame.state_timestamp_sec,
    }


def _raw_episode_id(requested_episode_id: str) -> str:
    normalized = "".join(
        char if (char.isalnum() or char in "_.-") else "-"
        for char in requested_episode_id.strip()
    ).strip(".-")
    return normalized or f"episode-{uuid4().hex}"


def _resolve_git_commit() -> str:
    """Resolve the collector revision without making startup depend on git."""
    override = os.environ.get("ROBO_COLLECTOR_GIT_COMMIT", "").strip()
    if override:
        return override
    repo_root = Path(__file__).resolve().parents[3]
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _canonical_config_hash(value: Mapping[str, Any]) -> str:
    """Hash the JSON-safe capture configuration deterministically."""
    return hashlib.sha256(
        json.dumps(
            dict(value),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _free_disk_bytes(path: Any) -> int:
    candidate = path.resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return int(shutil.disk_usage(candidate).free)


def _process_rss_bytes() -> int | None:
    """Return current process RSS when the host exposes it."""
    try:
        fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
        if len(fields) < 2:
            return None
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        resident_pages = int(fields[1])
        if page_size <= 0 or resident_pages < 0:
            return None
        return resident_pages * page_size
    except (OSError, ValueError, TypeError):
        return None


def _process_peak_rss_bytes() -> int | None:
    """Return the process peak RSS using the platform's native accounting."""
    if resource is None:
        return None
    try:
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    # Linux reports KiB; macOS reports bytes.
    return int(value if sys.platform == "darwin" else value * 1024)


def _metric_string(value: int | None) -> str:
    return "" if value is None else str(value)


def _packet_member(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _camera_clock_domain(value: Any, stream: str) -> str:
    """Preserve source clock metadata and name host fallback explicitly."""
    domain = _packet_member(value, "clock_domain")
    if domain:
        return str(domain)
    quality = str(_packet_member(value, "timestamp_quality", "")).strip().lower()
    if quality == "host_after_capture" or _packet_member(
        value, "device_timestamp"
    ) is None:
        return "server_wall"
    return f"camera:{stream}"


def _non_negative_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("packet gap count must be a non-negative integer")
    if value < 0:
        raise ValueError("packet gap count must be a non-negative integer")
    return value


def _configured_non_negative_int(value: Any, name: str) -> int:
    """Validate integer ROS quality thresholds without silently truncating."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _optional_configured_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("configured age must be a finite non-negative number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError("configured age must be a finite non-negative number")
    return number


def _optional_sequence(value: Any) -> int | None:
    if value is None:
        return None
    return _non_negative_int(value)


def _update_sequence_summary(
    summaries: dict[str, dict[str, int | None]],
    stream: str,
    sequence: int,
) -> None:
    """Track selected sequence ranges without retaining every sequence."""
    summary = summaries.setdefault(
        str(stream),
        {"first_sequence": None, "last_sequence": None, "count": 0},
    )
    first = _optional_sequence(summary.get("first_sequence"))
    last = _optional_sequence(summary.get("last_sequence"))
    summary["first_sequence"] = sequence if first is None else min(first, sequence)
    summary["last_sequence"] = sequence if last is None else max(last, sequence)
    summary["count"] = int(summary.get("count", 0) or 0) + 1


def _non_negative_count_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _camera_source_record_error(
    record: Mapping[str, Any], physical_stream: str, expected_session_id: str
) -> str | None:
    """Validate the source spool fields before they enter host Raw."""
    record_stream = record.get("stream")
    if not isinstance(record_stream, str) or not record_stream.strip():
        return "missing_stream"
    if record_stream != physical_stream:
        return "invalid_stream"
    sequence = record.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        return "invalid_sequence"
    payload = record.get("payload")
    if not isinstance(payload, (bytes, bytearray, memoryview)) or not payload:
        return "invalid_payload"
    record_session_id = record.get("session_id")
    if not isinstance(record_session_id, str) or not record_session_id.strip():
        return "missing_session_id"
    if str(record_session_id) != str(expected_session_id):
        return "invalid_session"
    clock_domain = record.get("clock_domain")
    if not isinstance(clock_domain, str) or not clock_domain.strip():
        return "missing_clock_domain"
    timestamp = record.get("server_wall_timestamp")
    if timestamp is None:
        timestamp = record.get("server_wall")
    try:
        timestamp_value = float(timestamp)
    except (TypeError, ValueError):
        return "invalid_server_wall_timestamp"
    if not math.isfinite(timestamp_value):
        return "invalid_server_wall_timestamp"
    encoding = record.get("payload_encoding")
    if encoding is not None and encoding not in {"image/jpeg", "image/png"}:
        return "invalid_payload_encoding"
    for name in (
        "producer_gap_count",
        "publisher_gap_count",
        "transport_gap_count",
        "unattributed_gap_count",
    ):
        value = record.get(name)
        if value is not None and _non_negative_count_or_none(value) is None:
            return f"invalid_{name}"
    return None


def _residual_statistics(values: list[float]) -> dict[str, float | None]:
    finite = sorted(value for value in values if math.isfinite(value) and value >= 0)
    if not finite:
        return {"p50": None, "p95": None, "p99": None, "max": None}

    def percentile(fraction: float) -> float:
        index = min(len(finite) - 1, int(round((len(finite) - 1) * fraction)))
        return finite[index]

    return {
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": finite[-1],
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _record_command_name(command: int) -> str:
    if command == int(RecordCommand.START):
        return "START"
    if command == int(RecordCommand.STOP):
        return "STOP"
    if command == int(RecordCommand.DISCARD):
        return "DISCARD"
    return f"UNKNOWN({command})"


def _diagnostic_level(level: str) -> bytes:
    if level == "ERROR":
        return _diagnostic_level_value(DiagnosticStatus.ERROR)
    if level == "WARN":
        return _diagnostic_level_value(DiagnosticStatus.WARN)
    return _diagnostic_level_value(DiagnosticStatus.OK)


def _diagnostic_level_value(level: Any) -> bytes:
    if isinstance(level, bytes):
        if len(level) != 1:
            raise ValueError(f"invalid DiagnosticStatus level bytes: {level!r}")
        return level
    if isinstance(level, bytearray):
        if len(level) != 1:
            raise ValueError(f"invalid DiagnosticStatus level bytes: {level!r}")
        return bytes(level)
    if isinstance(level, int):
        return bytes([level])
    return bytes([int(level)])


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: LeRobotCollectorNode | None = None
    try:
        node = LeRobotCollectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
