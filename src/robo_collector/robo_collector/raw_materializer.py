"""Materialize a sealed raw episode into the existing LeRobot writer.

This module is deliberately an adapter around :class:`LeRobotV21Writer`.
Recording remains independent of image decoding and video encoding; those
operations happen only after a durable raw manifest has been closed.
"""

from __future__ import annotations

import base64
import bisect
import hashlib
import json
import math
import os
import random
import sqlite3
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .episode_quality import EpisodeQualityGate, write_quality_report
from .field_config import FieldSelection
from .lerobot_dataset import (
    LeRobotV21Writer,
    RobotFrame,
    SaveResult,
    recover_dataset_metadata,
    video_encoder_identity,
)
from .raw_episode import (
    RawEpisodeReader,
    claim_materialization_job,
    create_materialization_job,
    reset_materialization_job_for_retry,
    update_materialization_job,
)


MATERIALIZER_VERSION = "robo_collector.raw_materializer.v2"
ALIGNMENT_POLICY_VERSION = "rgb_affine_v2"

ImageDecoder = Callable[[bytes, str], Any]
WriterFactory = Callable[[Path, str, int, Sequence[str]], LeRobotV21Writer]


@dataclass(frozen=True)
class MaterializationConfig:
    output_root: Path
    dataset_name: str
    fps: int
    camera_streams: tuple[str, ...]
    alignment_policy: str = "strict"
    max_alignment_residual_sec: float = 0.1
    output_schema_version: str = "lerobot.v2.1.raw_materialization.v2"
    require_complete_capture: bool = False
    max_camera_clock_mapping_uncertainty_sec: float | None = None
    max_state_age_sec: float | None = None
    max_camera_age_sec: float | None = None
    max_timer_deadline_misses: int = 0
    persist_job: bool = True
    publish_non_ready: bool = False
    field_selection: Mapping[str, Any] | FieldSelection | None = None
    quality_thresholds: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_root": str(self.output_root),
            "dataset_name": self.dataset_name,
            "fps": self.fps,
            "camera_streams": list(self.camera_streams),
            "alignment_policy": self.alignment_policy,
            "alignment_policy_version": ALIGNMENT_POLICY_VERSION,
            "materializer_version": MATERIALIZER_VERSION,
            "max_alignment_residual_sec": self.max_alignment_residual_sec,
            "output_schema_version": self.output_schema_version,
            "require_complete_capture": self.require_complete_capture,
            "max_camera_clock_mapping_uncertainty_sec": (
                self.max_camera_clock_mapping_uncertainty_sec
            ),
            "max_state_age_sec": self.max_state_age_sec,
            "max_camera_age_sec": self.max_camera_age_sec,
            "max_timer_deadline_misses": self.max_timer_deadline_misses,
            "persist_job": self.persist_job,
            "publish_non_ready": self.publish_non_ready,
            "field_selection": _field_selection_config(self.field_selection),
            "quality_thresholds": dict(self.quality_thresholds or {}),
        }


@dataclass(frozen=True)
class MaterializationResult:
    episode_id: str
    job_id: str
    output_dataset: Path | None
    frame_count: int
    dropped_selection_count: int
    quality_status: str
    source_manifest_hash: str


class MaterializationError(RuntimeError):
    """Raised when a raw episode cannot be safely materialized."""


class RawEpisodeMaterializer:
    """Replay raw records, align them, and atomically publish LeRobot output."""

    def __init__(
        self,
        config: MaterializationConfig,
        *,
        image_decoder: ImageDecoder | None = None,
        writer_factory: WriterFactory | None = None,
        quality_gate: EpisodeQualityGate | None = None,
        persist_job: bool | None = None,
    ) -> None:
        if config.alignment_policy not in {"strict", "sparse"}:
            raise ValueError("alignment_policy must be 'strict' or 'sparse'")
        if config.fps <= 0:
            raise ValueError("fps must be positive")
        if not config.camera_streams:
            raise ValueError("at least one camera stream is required")
        if not math.isfinite(config.max_alignment_residual_sec) or config.max_alignment_residual_sec < 0:
            raise ValueError("max_alignment_residual_sec must be finite and non-negative")
        if config.max_camera_clock_mapping_uncertainty_sec is not None and (
            not math.isfinite(config.max_camera_clock_mapping_uncertainty_sec)
            or config.max_camera_clock_mapping_uncertainty_sec < 0
        ):
            raise ValueError(
                "max_camera_clock_mapping_uncertainty_sec must be finite and non-negative"
            )
        self.config = config
        self._persist_job = config.persist_job if persist_job is None else bool(persist_job)
        self._image_decoder = image_decoder or _decode_image
        self._writer_factory = writer_factory
        self._quality_gate = quality_gate or EpisodeQualityGate(
            require_complete_capture=config.require_complete_capture,
            materialization_policy=config.alignment_policy,
            thresholds=config.quality_thresholds,
            max_state_age_sec=config.max_state_age_sec,
            max_camera_age_sec=config.max_camera_age_sec,
            max_camera_clock_mapping_uncertainty_sec=(
                config.max_camera_clock_mapping_uncertainty_sec
            ),
            max_timer_deadline_misses=config.max_timer_deadline_misses,
        )

    def materialize(
        self,
        episode: str | os.PathLike[str],
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> MaterializationResult:
        if self.config.alignment_policy == "sparse":
            raise MaterializationError(
                "sparse materialization is diagnostic-only and requires a sparse writer"
            )
        path = Path(episode).resolve()
        try:
            manifest_probe = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            manifest_probe = {}
        if isinstance(manifest_probe, dict) and manifest_probe.get("format") == "robo_collector.mcap_landing":
            from .mcap_episode import McapEpisodeReader
            reader = McapEpisodeReader(path)
        else:
            reader = RawEpisodeReader(path)
        reader.validate()
        manifest = reader.manifest
        if manifest.get("format") == "robo_collector.mcap_landing":
            manifest = dict(manifest)
            identity = manifest.get("identity")
            if isinstance(identity, Mapping):
                manifest.setdefault("episode_id", identity.get("episode_id"))
            manifest.setdefault("episode_id", path.name)
            manifest.setdefault("task_prompt", "MCAP episode")
        if self._persist_job:
            job = create_materialization_job(path, self.config.as_dict(), self.config.output_schema_version)
            completed = _reuse_materialized_job(path, manifest, job, self._quality_gate)
            if completed is not None:
                return completed
            _recover_pending_publication(path, job, self.config)
            manifest = RawEpisodeReader(path).manifest
            completed = _reuse_materialized_job(path, manifest, job, self._quality_gate)
            if completed is not None:
                return completed
            if job.get("status") == "MATERIALIZED":
                # The durable marker survived, but the artifact/provenance
                # evidence did not.  Remove only derived paths under the
                # configured output root and return the same idempotent job to
                # RAW_CLOSED before encoding again.
                _reset_invalid_materialized_job(path, job, self.config)
            claim_materialization_job(path, str(job["job_id"]))
        else:
            job = {"job_id": f"transient-{uuid.uuid4().hex}", "source_manifest_hash": str(manifest.get("raw_manifest_hash", manifest.get("manifest_hash", "")))}
        source_episode_id = str(manifest["episode_id"])
        source_manifest_hash = str(
            manifest.get("raw_manifest_hash", manifest.get("manifest_hash", ""))
        )
        conversion_config_hash = str(
            job.get("conversion_config_hash")
            or hashlib.sha256(
                json.dumps(
                    self.config.as_dict(), sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
        )
        writer: LeRobotV21Writer | None = None
        try:
            _report(progress_callback, "reading_raw")
            # Keep the raw payloads on disk while building the alignment index.
            # A materialization job may be much longer than one bounded raw
            # chunk; retaining decoded/base64 JPEGs here would make RSS grow
            # linearly with episode duration.
            with _DiskRecordIndex() as record_index:
                for stream in self.config.camera_streams:
                    for record in reader.records("camera", stream):
                        record_index.add("camera", stream, record)
                for record in reader.records("robot", "state"):
                    record_index.add("robot", "state", record)
                for record in reader.records("robot", "action"):
                    record_index.add("robot", "action", record)
                if record_index.count("robot", "state") == 0:
                    raise MaterializationError("robot state records are required")

                _report(progress_callback, "encoding_derived_artifacts")
                if self._writer_factory is None:
                    writer = _default_writer_factory(
                        self.config.output_root,
                        self.config.dataset_name,
                        self.config.fps,
                        self.config.camera_streams,
                        field_selection=self.config.field_selection,
                    )
                else:
                    writer = self._writer_factory(
                        self.config.output_root,
                        self.config.dataset_name,
                        self.config.fps,
                        self.config.camera_streams,
                    )
                episode_id = source_episode_id
                writer.start_episode(
                    str(manifest.get("task_prompt", "raw episode")), episode_id
                )
                set_provenance = getattr(writer, "set_episode_provenance", None)
                if callable(set_provenance):
                    set_provenance(
                        {
                            "source_episode_id": source_episode_id,
                            "source_manifest_hash": source_manifest_hash,
                            "converter_version": MATERIALIZER_VERSION,
                            "conversion_config_hash": conversion_config_hash,
                            "output_schema_version": self.config.output_schema_version,
                            "selection_policy": "rgb_reference_nearest_strict",
                                        "reference_camera_stream": _reference_camera_stream(
                                self.config
                            ),
                            "encoder_identity": video_encoder_identity(),
                        }
                    )
                (
                    dropped,
                    selection_gaps_by_stream,
                    metrics,
                    residuals,
                ) = _write_indexed_frames(
                    record_index,
                    writer,
                    config=self.config,
                    image_decoder=self._image_decoder,
                )

            if writer.active_frame_count == 0:
                writer.discard_episode()
                raise MaterializationError("materialization produced no frames")
            encoder_identity = _writer_encoder_identity(writer)
            deferred_publish = callable(
                getattr(writer, "save_episode_unpublished", None)
            )
            if deferred_publish:
                save_result = writer.save_episode_unpublished(
                    progress_callback=progress_callback
                )
            else:
                save_result = writer.save_episode(progress_callback=progress_callback)
            if not save_result.saved:
                raise MaterializationError(
                    "writer did not save materialized artifacts"
                    + (f": {save_result.message}" if save_result.message else "")
                )
            artifacts = _artifact_metadata(
                save_result,
                path,
                self.config,
                encoder_identity=encoder_identity,
            )
            provenance_path = _write_provenance_file(
                save_result,
                source_episode_id=source_episode_id,
                source_manifest_hash=source_manifest_hash,
                config=self.config,
                dropped_selection_count=dropped,
                encoder_identity=encoder_identity,
                residuals=residuals,
            )
            if provenance_path is not None:
                artifacts["provenance"] = str(provenance_path)
                artifacts["evidence"]["provenance"] = _file_evidence(provenance_path, 1, kind="provenance")
            if deferred_publish:
                artifacts["publication"] = "PENDING"
            _report(progress_callback, "quality_check")
            manifest_for_qc = (
                json.loads((path / "manifest.json").read_text(encoding="utf-8"))
                if self._persist_job else dict(manifest)
            )
            # The derived artifacts are committed locally before QC, but the
            # durable lifecycle marker must not advertise MATERIALIZED until
            # the QC report itself has been written.  Evaluate a derived copy
            # as MATERIALIZED while keeping the source manifest in its current
            # recoverable state.
            manifest_for_qc["status"] = "MATERIALIZED"
            manifest_for_qc["artifacts"] = artifacts
            # Raw stream statistics describe records physically present in the
            # immutable chunks.  Selection gaps are a derived alignment metric;
            # expose them to QC without writing them back into the raw stats,
            # otherwise a later RawEpisodeReader.validate() would report a
            # false checksum/statistics mismatch.
            stream_stats = manifest_for_qc.get("streams")
            if isinstance(stream_stats, Mapping):
                stream_stats = {str(name): dict(value) for name, value in stream_stats.items() if isinstance(value, Mapping)}
                for stream, gap_count in selection_gaps_by_stream.items():
                    if stream in stream_stats:
                        stream_stats[stream]["selection_gap_count"] = int(
                            stream_stats[stream].get("selection_gap_count", 0) or 0
                        ) + gap_count
                manifest_for_qc["streams"] = stream_stats
            previous_quality = manifest_for_qc.get("quality")
            previous_stats = previous_quality.get("statistics", previous_quality) if isinstance(previous_quality, Mapping) else {}
            manifest_for_qc["quality"] = {
                "statistics": {
                    **(dict(previous_stats) if isinstance(previous_stats, Mapping) else {}),
                    **metrics,
                    "selection_gaps": dropped,
                    "selection_gaps_by_stream": selection_gaps_by_stream,
                }
            }
            quality = self._quality_gate.evaluate(manifest_for_qc)
            quality["statistics"]["selection_gaps"] = dropped
            quality["statistics"]["selection_gaps_by_stream"] = selection_gaps_by_stream
            quality["statistics"]["alignment_residual_sec"] = _residual_stats(
                residuals
            )

            publish_after_qc = False
            if deferred_publish:
                if quality["status"] == "READY" or self.config.publish_non_ready:
                    publish_after_qc = True
                else:
                    review_root = (
                        self.config.output_root
                        / ".review"
                        / episode_id
                        / str(job["job_id"])
                    )
                    extra_paths: list[Path] = []
                    if provenance_path is not None:
                        extra_paths.append(provenance_path)
                        aggregate_provenance = provenance_path.parent.parent / "raw_provenance.json"
                        if aggregate_provenance.is_file():
                            extra_paths.append(aggregate_provenance)
                    retained = writer.retain_unpublished_episode(
                        review_root,
                        extra_paths=extra_paths,
                    )
                    save_result = retained
                    artifacts = _artifact_metadata(
                        save_result,
                        path,
                        self.config,
                        encoder_identity=encoder_identity,
                    )
                    artifacts["publication"] = "REVIEW"
                    retained_provenance = (
                        review_root
                        / provenance_path.relative_to(provenance_path.parents[2])
                        if provenance_path is not None
                        else review_root / "meta" / "raw_provenance.json"
                    )
                    if retained_provenance.is_file():
                        artifacts["provenance"] = str(retained_provenance)
                        artifacts["evidence"]["provenance"] = _file_evidence(
                            retained_provenance, 1, kind="provenance"
                        )
                    manifest_for_qc["artifacts"] = artifacts
                    quality = self._quality_gate.evaluate(manifest_for_qc)
                    quality["statistics"]["selection_gaps"] = dropped
                    quality["statistics"]["selection_gaps_by_stream"] = selection_gaps_by_stream
            quality_path = write_quality_report(path, quality)
            artifacts["quality"] = str(quality_path)
            artifacts.setdefault("evidence", {})["quality"] = _file_evidence(
                quality_path, 1, kind="quality"
            )
            if self._persist_job:
                _persist_materialization_qc(
                    path,
                    str(job["job_id"]),
                    artifacts,
                    quality,
                )

            if publish_after_qc:
                save_result = writer.publish_unpublished_episode()
                artifacts["publication"] = "PUBLISHED"
                if self._persist_job:
                    # Keep the QC boundary durable while recording the final
                    # publication decision.  A crash here is recoverable from
                    # the staged writer metadata and the persisted QC report.
                    _persist_materialization_qc(
                        path,
                        str(job["job_id"]),
                        artifacts,
                        quality,
                    )
            if self._persist_job:
                _set_quality_status(path, quality["status"])
            _report(progress_callback, "complete")
            return MaterializationResult(
                episode_id=episode_id,
                job_id=str(job["job_id"]),
                output_dataset=(
                    save_result.data_path.parent.parent
                    if save_result.data_path is not None
                    else None
                ),
                frame_count=save_result.frame_count,
                dropped_selection_count=dropped,
                quality_status=str(quality["status"]),
                source_manifest_hash=str(
                    manifest.get("raw_manifest_hash", manifest.get("manifest_hash", ""))
                ),
            )
        except Exception as exc:
            if writer is not None and writer.active_episode_index is not None:
                try:
                    writer.discard_episode()
                except Exception:
                    pass
            if self._persist_job:
                try:
                    update_materialization_job(
                        path, str(job["job_id"]), "FAILED", error=str(exc)
                    )
                except Exception:
                    # Preserve the original materialization failure.  Startup
                    # recovery will quarantine a manifest it cannot update.
                    pass
                try:
                    failed_manifest = json.loads(
                        (path / "manifest.json").read_text(encoding="utf-8")
                    )
                    failure_report = self._quality_gate.evaluate(failed_manifest)
                    reasons = list(failure_report.get("reason", []))
                    reasons.append(f"materialization_failed: {exc}")
                    failure_report["status"] = "REJECT"
                    failure_report["reason"] = reasons
                    write_quality_report(path, failure_report)
                except Exception:
                    # A missing failure report is repaired by scan_startup;
                    # never mask the actual conversion exception here.
                    pass
            raise


def robot_frame_from_mapping(state: Mapping[str, Any]) -> RobotFrame:
    """Build the existing writer frame from a JSON-safe raw state mapping."""

    def values(name: str, fallback: Sequence[float] = ()) -> list[float]:
        value = state.get(name, fallback)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise MaterializationError(f"raw state field {name!r} is not a vector")
        return [float(item) for item in value]

    policy_state = state.get("policy_state", {})
    if not isinstance(policy_state, Mapping):
        raise MaterializationError("raw policy_state must be a mapping")
    return RobotFrame(
        joint_position=values("joint_position"),
        joint_velocity=values("joint_velocity"),
        joint_torque=values("joint_torque"),
        imu_angular_velocity=values("imu_angular_velocity"),
        imu_linear_acceleration=values("imu_linear_acceleration"),
        projected_gravity_or_quat=values("projected_gravity_or_quat"),
        target_joint_pos=values("target_joint_pos"),
        policy_action=values("policy_action"),
        aligned_target_pos=values("aligned_target_pos"),
        policy_state={
            str(name): values_from_mapping(value, name)
            for name, value in policy_state.items()
        },
        joint_names=[str(name) for name in state.get("joint_names", [])],
        state_timestamp_sec=_optional_float(state.get("state_timestamp_sec")),
    )


def values_from_mapping(value: Any, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise MaterializationError(f"raw policy state field {name!r} is not a vector")
    return [float(item) for item in value]


@dataclass(frozen=True)
class _Selection:
    state_record: dict[str, Any]
    camera_by_stream: dict[str, dict[str, Any]]
    residuals: dict[str, float]
    target_time: float
    action_record: dict[str, Any] | None = None


class _DiskRecordIndex:
    """Bounded-memory timestamp index for one materialization pass.

    Raw records are JSON objects whose camera payload is a base64 string.  The
    index deliberately stores that object in SQLite instead of retaining it in
    Python containers.  Alignment queries load only the one state/camera
    candidate needed for the current output frame, so the hot materialization
    loop has an O(number-of-cameras) payload bound.
    """

    def __init__(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="robo-collector-materialize-"
        )
        database_path = Path(self._temporary_directory.name) / "records.sqlite3"
        self._connection = sqlite3.connect(str(database_path))
        self._connection.execute(
            """
            CREATE TABLE records (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                stream TEXT NOT NULL,
                timestamp REAL NOT NULL,
                record_json TEXT NOT NULL
            )
            """
        )
        self._bounds: dict[tuple[str, str], list[float | int]] = {}
        self._pending_rows = 0
        self._ready = False

    def __enter__(self) -> "_DiskRecordIndex":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            self._connection.close()
        finally:
            self._temporary_directory.cleanup()

    def add(self, category: str, stream: str, record: Mapping[str, Any]) -> None:
        if self._ready:
            raise RuntimeError("cannot add records after index finalization")
        if not isinstance(record, Mapping):
            raise MaterializationError("raw record must be a mapping")
        timestamp = _record_time(record)
        try:
            record_json = json.dumps(
                dict(record),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise MaterializationError(
                f"raw {category}/{stream} record is not JSON serializable"
            ) from exc
        self._connection.execute(
            "INSERT INTO records(category, stream, timestamp, record_json) VALUES (?, ?, ?, ?)",
            (str(category), str(stream), float(timestamp), record_json),
        )
        key = (str(category), str(stream))
        bounds = self._bounds.get(key)
        if bounds is None:
            self._bounds[key] = [float(timestamp), float(timestamp), 1]
        else:
            bounds[0] = min(float(bounds[0]), float(timestamp))
            bounds[1] = max(float(bounds[1]), float(timestamp))
            bounds[2] = int(bounds[2]) + 1
        self._pending_rows += 1
        if self._pending_rows >= 1024:
            self._connection.commit()
            self._pending_rows = 0

    def finish(self) -> None:
        if self._ready:
            return
        self._connection.commit()
        self._connection.execute(
            "CREATE INDEX records_by_timestamp ON records(category, stream, timestamp, record_id)"
        )
        self._connection.commit()
        self._ready = True

    def count(self, category: str, stream: str) -> int:
        return int(self._bounds.get((str(category), str(stream)), [0.0, 0.0, 0])[2])

    def target_times(
        self, camera_streams: Sequence[str], fps: int
    ) -> Any:
        """Yield timestamps from the reference RGB stream.

        ``fps`` is retained for callers written against the original fixed-rate
        helper, but it is intentionally not used to synthesize timestamps.  The
        first configured RGB stream is the reference timeline; every accepted
        output row therefore corresponds to one real RGB frame.
        """
        reference_stream = _reference_camera_streams(camera_streams)
        for record in self.reference_camera_records(reference_stream):
            yield _record_time(record)

    def reference_camera_records(self, reference_stream: str) -> Any:
        """Yield reference RGB records in capture-time order.

        Returning the actual record, rather than only its timestamp, prevents a
        duplicate timestamp from selecting the same camera frame twice.
        """
        self.finish()
        key = ("camera", str(reference_stream))
        if key not in self._bounds:
            return
        cursor = self._connection.execute(
            """
            SELECT record_json
            FROM records
            WHERE category = 'camera' AND stream = ?
            ORDER BY timestamp ASC, record_id ASC
            """,
            (str(reference_stream),),
        )
        for row in cursor:
            value = json.loads(str(row[0]))
            if not isinstance(value, dict):
                raise MaterializationError(
                    "indexed reference camera record is not an object"
                )
            yield value

    def nearest(
        self,
        category: str,
        stream: str,
        target: float,
        max_residual: float | None,
    ) -> tuple[dict[str, Any] | None, float | None]:
        self.finish()
        key = (str(category), str(stream))
        if key not in self._bounds:
            return None, None
        before = self._connection.execute(
            """
            SELECT record_id, timestamp, record_json
            FROM records
            WHERE category = ? AND stream = ? AND timestamp <= ?
            ORDER BY timestamp DESC, record_id DESC
            LIMIT 1
            """,
            (str(category), str(stream), float(target)),
        ).fetchone()
        after = self._connection.execute(
            """
            SELECT record_id, timestamp, record_json
            FROM records
            WHERE category = ? AND stream = ? AND timestamp >= ?
            ORDER BY timestamp ASC, record_id ASC
            LIMIT 1
            """,
            (str(category), str(stream), float(target)),
        ).fetchone()
        candidates = [row for row in (before, after) if row is not None]
        if not candidates:
            return None, None
        # Prefer the right-hand candidate on an exact tie, matching the
        # historical bisect-based alignment helper.
        row = min(
            candidates,
            key=lambda item: (
                abs(float(item[1]) - float(target)),
                0 if item is after else 1,
            ),
        )
        residual = abs(float(row[1]) - float(target))
        if max_residual is not None and residual > max_residual:
            return None, None
        value = json.loads(str(row[2]))
        if not isinstance(value, dict):
            raise MaterializationError("indexed raw record is not an object")
        return value, residual

    def latest_at_or_before(
        self, category: str, stream: str, target: float
    ) -> dict[str, Any] | None:
        self.finish()
        row = self._connection.execute(
            """
            SELECT record_json
            FROM records
            WHERE category = ? AND stream = ? AND timestamp <= ?
            ORDER BY timestamp DESC, record_id DESC
            LIMIT 1
            """,
            (str(category), str(stream), float(target)),
        ).fetchone()
        if row is None:
            return None
        value = json.loads(str(row[0]))
        if not isinstance(value, dict):
            raise MaterializationError("indexed raw record is not an object")
        return value


class _BoundedFloatReservoir:
    """Keep exact small-sample metrics and bounded approximate large metrics."""

    def __init__(self, capacity: int = 4096) -> None:
        self._capacity = int(capacity)
        self._values: list[float] = []
        self._count = 0
        self._random = random.Random(0)

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            return
        self._count += 1
        if len(self._values) < self._capacity:
            self._values.append(float(value))
            return
        replacement = self._random.randrange(self._count)
        if replacement < self._capacity:
            self._values[replacement] = float(value)

    def values(self) -> list[float]:
        return list(self._values)


class _AlignmentAccumulator:
    """Accumulate QC/provenance alignment metrics without retaining selections."""

    def __init__(self) -> None:
        self.camera_skews = _BoundedFloatReservoir()
        self.state_skews = _BoundedFloatReservoir()
        self.residuals = _BoundedFloatReservoir()

    def add(self, selection: _Selection) -> None:
        camera_times = [
            _record_time(selection.camera_by_stream[stream])
            for stream in selection.camera_by_stream
        ]
        if len(camera_times) >= 2:
            self.camera_skews.add(max(camera_times) - min(camera_times))
        elif len(camera_times) == 1:
            self.camera_skews.add(0.0)
        state_time = _record_time(selection.state_record)
        for stream in selection.camera_by_stream:
            self.state_skews.add(
                abs(_record_time(selection.camera_by_stream[stream]) - state_time)
            )
        for residual in selection.residuals.values():
            self.residuals.add(residual)

    def metrics(self) -> dict[str, Any]:
        camera = _residual_stats(self.camera_skews.values())
        state = _residual_stats(self.state_skews.values())
        return {
            "camera_camera_skew_sec": camera.get("max"),
            "state_camera_skew_sec": state.get("max"),
            "camera_camera_skew": camera,
            "state_camera_skew": state,
        }


def _write_indexed_frames(
    record_index: _DiskRecordIndex,
    writer: LeRobotV21Writer,
    *,
    config: MaterializationConfig,
    image_decoder: ImageDecoder,
) -> tuple[int, dict[str, int], dict[str, Any], list[float]]:
    """Align and write one frame at a time from the disk-backed index."""
    selection_gaps = {stream: 0 for stream in config.camera_streams}
    selection_gaps["state"] = 0
    dropped = 0
    metrics = _AlignmentAccumulator()
    dataset_frame_index = 0
    reference_stream = _reference_camera_stream(config)
    for reference_record in record_index.reference_camera_records(reference_stream):
        target = _record_time(reference_record)
        state_record, state_residual = record_index.nearest(
            "robot",
            "state",
            target,
            config.max_alignment_residual_sec,
        )
        if state_record is None or state_residual is None:
            selection_gaps["state"] += 1
            # The reference RGB frame is present by construction.  Attribute
            # this RGB target to any non-reference stream that is also outside
            # the matching residual so per-stream gap accounting remains
            # meaningful without inventing an image.
            for stream in config.camera_streams:
                if stream == reference_stream:
                    continue
                candidate, residual = record_index.nearest(
                    "camera",
                    stream,
                    target,
                    config.max_alignment_residual_sec,
                )
                if candidate is None or residual is None:
                    selection_gaps[stream] += 1
            dropped += 1
            continue
        camera_by_stream: dict[str, dict[str, Any]] = {
            reference_stream: reference_record
        }
        residuals: dict[str, float] = {
            "state": state_residual,
            reference_stream: 0.0,
        }
        complete = True
        for stream in config.camera_streams:
            if stream == reference_stream:
                continue
            candidate, residual = record_index.nearest(
                "camera",
                stream,
                target,
                config.max_alignment_residual_sec,
            )
            if candidate is None or residual is None:
                selection_gaps[stream] += 1
                complete = False
                continue
            camera_by_stream[stream] = candidate
            residuals[stream] = residual
        if not complete:
            dropped += 1
            continue
        selection = _Selection(
            state_record=state_record,
            camera_by_stream=camera_by_stream,
            residuals=residuals,
            target_time=target,
            action_record=record_index.latest_at_or_before(
                "robot", "action", target
            ),
        )
        state = state_record.get("state")
        if not isinstance(state, Mapping):
            raise MaterializationError("robot state record has no mapping state")
        # Actions are held from the latest valid sample at or before the target,
        # independently of nearest-neighbour state.
        if selection.action_record is not None:
            action_state = selection.action_record.get("state")
            if isinstance(action_state, Mapping):
                state = dict(state)
                for field in (
                    "target_joint_pos",
                    "policy_action",
                    "aligned_target_pos",
                ):
                    if field in action_state:
                        state[field] = action_state[field]
        frame = robot_frame_from_mapping(state)
        images = {
            stream: image_decoder(
                base64.b64decode(camera_by_stream[stream]["payload"]),
                str(
                    camera_by_stream[stream].get(
                        "payload_encoding", "image/jpeg"
                    )
                ),
            )
            for stream in config.camera_streams
        }
        writer.add_frame(
            frame,
            images,
            camera_timestamps_sec={
                stream: _record_timestamp(camera_by_stream[stream])
                for stream in config.camera_streams
            },
            alignment_metadata={
                "selection_policy": "rgb_reference_nearest_strict",
                            "alignment_selection_policy": "rgb_reference_nearest_bounded",
                "action_policy": "latest_at_or_before_zoh",
                "target_timestamp_sec": selection.target_time,
                "alignment_target_source_timestamp": selection.target_time,
                # This is the timestamp of the dense output row, not the
                # position of the candidate target that may have been dropped.
                # Keep the original target on the alignment source field above.
                "selected_dataset_timestamp": dataset_frame_index / config.fps,
                "state_residual_sec": selection.residuals.get("state"),
                "camera_residuals_sec": {
                    stream: selection.residuals[stream]
                    for stream in config.camera_streams
                },
                "alignment_residual_sec": max(
                    selection.residuals.values(), default=0.0
                ),
                "state_sequence": state_record["sequence"],
                "camera_sequences": {
                    stream: camera_by_stream[stream]["sequence"]
                    for stream in config.camera_streams
                },
                "action_sequence": (
                    selection.action_record.get("sequence")
                    if selection.action_record
                    else None
                ),
            },
        )
        dataset_frame_index += 1
        metrics.add(selection)
    return dropped, selection_gaps, metrics.metrics(), metrics.residuals.values()


def _align_records(
    camera_records: Mapping[str, list[dict[str, Any]]],
    state_records: list[dict[str, Any]],
    config: MaterializationConfig,
) -> list[_Selection | None]:
    ordered_camera_records = {
        stream: sorted(records, key=_record_time)
        for stream, records in camera_records.items()
    }
    ordered_state_records = sorted(state_records, key=_record_time)
    camera_times = {
        stream: [_record_time(record) for record in records]
        for stream, records in ordered_camera_records.items()
    }
    state_times = [_record_time(record) for record in ordered_state_records]
    _validate_clock_domains([*ordered_state_records, *(record for values in ordered_camera_records.values() for record in values)])
    reference_stream = _reference_camera_stream(config)
    selections: list[_Selection | None] = []
    for reference_record in ordered_camera_records.get(reference_stream, []):
        target = _record_time(reference_record)
        state_index = _nearest_index(
            ordered_state_records,
            target,
            0,
            config.max_alignment_residual_sec,
            times=state_times,
        )
        if state_index is None:
            selections.append(None)
            continue
        state_record = ordered_state_records[state_index]
        selected: dict[str, dict[str, Any]] = {reference_stream: reference_record}
        residuals: dict[str, float] = {reference_stream: 0.0}
        complete = True
        for stream in config.camera_streams:
            if stream == reference_stream:
                continue
            records = ordered_camera_records.get(stream, [])
            index = _nearest_index(
                records,
                target,
                0,
                config.max_alignment_residual_sec,
                times=camera_times.get(stream, []),
            )
            if index is None:
                complete = False
                continue
            candidate = records[index]
            residual = abs(camera_times[stream][index] - target)
            selected[stream] = candidate
            residuals[stream] = residual
        if not complete or len(selected) != len(config.camera_streams):
            selections.append(None)
        else:
            residuals["state"] = abs(state_times[state_index] - target)
            action_index = _latest_at_or_before(
                ordered_state_records, target, times=state_times
            )
            selections.append(_Selection(
                state_record, selected, residuals, target,
                ordered_state_records[action_index] if action_index is not None else None,
            ))
    return selections


def _selection_gap_counts(
    camera_records: Mapping[str, list[dict[str, Any]]],
    state_records: list[dict[str, Any]],
    config: MaterializationConfig,
) -> dict[str, int]:
    """Attribute every incomplete RGB-reference target to missing streams."""
    ordered_camera = {stream: sorted(records, key=_record_time) for stream, records in camera_records.items()}
    ordered_state = sorted(state_records, key=_record_time)
    camera_times = {
        stream: [_record_time(record) for record in records]
        for stream, records in ordered_camera.items()
    }
    state_times = [_record_time(record) for record in ordered_state]
    counts = {stream: 0 for stream in config.camera_streams}
    counts["state"] = 0
    for target in _target_times(ordered_camera, ordered_state, config):
        if _nearest_index(
            ordered_state,
            target,
            0,
            config.max_alignment_residual_sec,
            times=state_times,
        ) is None:
            counts["state"] += 1
        for stream in config.camera_streams:
            if _nearest_index(
                ordered_camera.get(stream, []),
                target,
                0,
                config.max_alignment_residual_sec,
                times=camera_times.get(stream, []),
            ) is None:
                counts[stream] += 1
    return counts


def _target_times(
    camera_records: Mapping[str, list[dict[str, Any]]],
    state_records: list[dict[str, Any]],
    config: MaterializationConfig,
) -> list[float]:
    """Build the target axis from the configured reference RGB stream."""

    del state_records
    reference_stream = _reference_camera_stream(config)
    return [
        _record_time(record)
        for record in sorted(
            camera_records.get(reference_stream, []), key=_record_time
        )
    ]


def _reference_camera_stream(config: MaterializationConfig) -> str:
    return _reference_camera_streams(config.camera_streams)


def _reference_camera_streams(camera_streams: Sequence[str]) -> str:
    if not camera_streams:
        raise MaterializationError("at least one reference RGB stream is required")
    return str(camera_streams[0])


def _nearest_index(
    records: list[dict[str, Any]],
    target: float,
    start: int,
    max_residual: float | None = None,
    *,
    times: Sequence[float] | None = None,
) -> int | None:
    if not records:
        return None
    start = max(0, min(start, len(records) - 1))
    time_values: Sequence[float] = (
        times if times is not None else [_record_time(record) for record in records]
    )
    if len(time_values) != len(records):
        raise ValueError("timestamp cache length does not match records")
    insertion = bisect.bisect_left(time_values, target, lo=start)
    candidates = []
    if insertion < len(records):
        candidates.append(insertion)
    if insertion > start:
        candidates.append(insertion - 1)
    if not candidates:
        candidates.append(len(records) - 1)
    index = min(candidates, key=lambda index: abs(time_values[index] - target))
    if max_residual is not None and abs(time_values[index] - target) > max_residual:
        return None
    return index


def _latest_at_or_before(
    records: list[dict[str, Any]],
    target: float,
    *,
    times: Sequence[float] | None = None,
) -> int | None:
    if not records:
        return None
    time_values: Sequence[float] = (
        times if times is not None else [_record_time(record) for record in records]
    )
    if len(time_values) != len(records):
        raise ValueError("timestamp cache length does not match records")
    index = bisect.bisect_right(time_values, target) - 1
    return index if index >= 0 else None


def _validate_clock_domains(records: Sequence[Mapping[str, Any]]) -> None:
    domains = {str(record.get("clock_domain")) for record in records if record.get("clock_domain") is not None}
    if len(domains) <= 1:
        return
    for record in records:
        if _optional_float(record.get("record_monotonic_timestamp")) is None and _optional_float(record.get("receive_monotonic_timestamp")) is None:
            raise MaterializationError("cross clock-domain alignment requires host receive/record monotonic mapping on every record")


def _record_time(record: Mapping[str, Any]) -> float:
    for key in (
        "receive_monotonic_timestamp",
        "record_monotonic_timestamp",
        "receive_wall_timestamp",
        "server_monotonic_timestamp",
        "server_wall_timestamp",
        "device_timestamp",
    ):
        value = _optional_float(record.get(key))
        if value is not None:
            if key == "device_timestamp":
                unit = record.get("device_unit")
                if unit == "ms":
                    value *= 1e-3
                elif unit == "us":
                    value *= 1e-6
                elif unit == "ns":
                    value *= 1e-9
            return value
    return 0.0


def _record_timestamp(record: Mapping[str, Any]) -> float | None:
    value = record.get("server_wall_timestamp")
    if value is not None:
        return _optional_float(value)
    value = _optional_float(record.get("device_timestamp"))
    if value is None:
        return None
    unit = str(record.get("device_unit", "s")).lower()
    if unit == "ms":
        return value * 1e-3
    if unit == "us":
        return value * 1e-6
    if unit == "ns":
        return value * 1e-9
    return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _decode_image(payload: bytes, encoding: str) -> Any:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise MaterializationError("opencv-python and numpy are required to decode raw images") from exc
    array = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED if encoding == "image/png" else cv2.IMREAD_COLOR)
    if image is None:
        raise MaterializationError(f"failed to decode raw {encoding} image")
    if encoding != "image/png" and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _default_writer_factory(
    output_root: Path,
    dataset_name: str,
    fps: int,
    camera_streams: Sequence[str],
    *,
    field_selection: Mapping[str, Any] | FieldSelection | None = None,
) -> LeRobotV21Writer:
    return LeRobotV21Writer(
        output_root,
        dataset_name=dataset_name,
        fps=fps,
        camera_keys=[f"observation.images.{stream}" for stream in camera_streams],
        field_selection=_coerce_field_selection(field_selection),
    )


def _coerce_field_selection(
    value: Mapping[str, Any] | FieldSelection | None,
) -> FieldSelection | None:
    if value is None or isinstance(value, FieldSelection):
        return value
    if not isinstance(value, Mapping):
        raise MaterializationError("field_selection must be a mapping")
    target = value.get("target")
    state = value.get("state")
    include_policy_action = value.get("include_policy_action", False)
    if not isinstance(target, (list, tuple)) or not isinstance(state, (list, tuple)):
        raise MaterializationError(
            "field_selection must contain target and state lists"
        )
    if not isinstance(include_policy_action, bool):
        raise MaterializationError("field_selection.include_policy_action must be boolean")
    try:
        return FieldSelection(
            target=tuple(str(item) for item in target),
            state=tuple(str(item) for item in state),
            include_policy_action=include_policy_action,
        )
    except (TypeError, ValueError) as exc:
        raise MaterializationError(f"invalid field_selection: {exc}") from exc


def _field_selection_config(
    value: Mapping[str, Any] | FieldSelection | None,
) -> dict[str, Any] | None:
    selection = _coerce_field_selection(value)
    if selection is None:
        return None
    return {
        "target": list(selection.target),
        "state": list(selection.state),
        "include_policy_action": selection.include_policy_action,
    }


def _artifact_metadata(
    result: SaveResult,
    episode: Path,
    config: MaterializationConfig,
    *,
    encoder_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    paths: dict[str, Any] = {"source_episode": str(episode), "evidence": {}}
    if result.data_path is not None:
        paths["data"] = str(result.data_path)
        paths["evidence"]["data"] = _file_evidence(result.data_path, result.frame_count, kind="parquet")
    paths["videos"] = {key: str(value) for key, value in result.video_paths.items()}
    paths["evidence"]["videos"] = {
        key: _file_evidence(value, result.frame_count, kind="mp4")
        for key, value in result.video_paths.items()
    }
    paths["frame_count"] = result.frame_count
    paths["output_schema_version"] = config.output_schema_version
    paths["encoder_identity"] = dict(
        encoder_identity or video_encoder_identity()
    )
    return paths


def _writer_encoder_identity(writer: Any) -> dict[str, Any]:
    getter = getattr(writer, "get_encoder_identity", None)
    if callable(getter):
        try:
            value = getter()
        except Exception:
            value = None
        if isinstance(value, Mapping) and value:
            return dict(value)
    return video_encoder_identity()


def _valid_encoder_identity(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    per_camera = value.get("per_camera")
    if per_camera is not None:
        if not isinstance(per_camera, Mapping) or not per_camera:
            return False
        identities = per_camera.values()
    else:
        identities = (value,)
    required = ("library", "library_version", "backend", "codec")
    return all(
        isinstance(identity, Mapping)
        and all(
            isinstance(identity.get(field), str) and identity.get(field).strip()
            for field in required
        )
        for identity in identities
    )


def _file_evidence(path: Path, count: int, *, kind: str | None = None) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise MaterializationError(f"missing or empty materialized artifact: {path}")
    size = path.stat().st_size
    prefix = _read_file_prefix(path, 64)
    suffix = _read_file_suffix(path, 4)
    signature_valid = True
    validation_error: str | None = None
    if kind == "parquet" and (
        size < 8 or prefix[:4] != b"PAR1" or suffix != b"PAR1"
    ):
        signature_valid = False
        validation_error = "format_error:invalid_parquet_signature"
    if kind == "mp4" and b"ftyp" not in prefix:
        signature_valid = False
        validation_error = "format_error:missing_ftyp_box"
    actual_count: int | None = None
    validation = validation_error or "signature_only"
    if signature_valid and kind == "parquet":
        try:
            import pyarrow.parquet as parquet
            actual_count = parquet.ParquetFile(path).metadata.num_rows
            validation = "pyarrow"
        except ImportError:
            validation = "decoder_unavailable:pyarrow"
        except Exception as exc:
            validation = f"decoder_error:pyarrow:{type(exc).__name__}"
    elif signature_valid and kind == "mp4":
        try:
            import cv2
            capture = cv2.VideoCapture(str(path))
            try:
                if not capture.isOpened():
                    validation = "decoder_error:opencv:not_opened"
                else:
                    actual_count = 0
                    while True:
                        ok, _ = capture.read()
                        if not ok:
                            break
                        actual_count += 1
                    validation = "opencv"
            finally:
                capture.release()
        except ImportError:
            validation = "decoder_unavailable:opencv"
        except Exception as exc:
            validation = f"decoder_error:opencv:{type(exc).__name__}"
    elif kind == "provenance":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, Mapping):
                validation = "format_error:provenance_not_object"
            else:
                actual_count = 1
                validation = "json"
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            validation = f"format_error:provenance_json:{type(exc).__name__}"
    elif kind == "quality":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, Mapping):
                validation = "format_error:quality_not_object"
            else:
                actual_count = 1
                validation = "json"
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            validation = f"format_error:quality_json:{type(exc).__name__}"
    decodable = (
        actual_count is not None
        and actual_count == count
        and signature_valid
        and not validation.startswith(("decoder_error:", "format_error:"))
    )
    return {
        "path": str(path),
        "size": size,
        "sha256": _sha256_file(path),
        "row_count": actual_count if kind == "parquet" and actual_count is not None else count,
        "frame_count": actual_count if kind == "mp4" and actual_count is not None else count,
        "expected_count": count,
        "decodable": decodable,
        "validation": validation,
        "format_valid": signature_valid,
    }


def _read_file_prefix(path: Path, size: int) -> bytes:
    with path.open("rb") as stream:
        return stream.read(size)


def _read_file_suffix(path: Path, size: int) -> bytes:
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        file_size = stream.tell()
        stream.seek(max(0, file_size - size), os.SEEK_SET)
        return stream.read(size)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_provenance_file(
    result: SaveResult,
    *,
    source_episode_id: str,
    source_manifest_hash: str,
    config: MaterializationConfig,
    dropped_selection_count: int,
    residuals: list[float],
    encoder_identity: Mapping[str, Any] | None = None,
) -> Path | None:
    if result.data_path is None:
        return None
    dataset_root = result.data_path.parent.parent
    conversion_config_hash = hashlib.sha256(
        json.dumps(
            config.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    episode_metadata = {
        "source_episode_id": source_episode_id,
        "source_manifest_hash": source_manifest_hash,
        "converter_version": MATERIALIZER_VERSION,
        "conversion_config_hash": conversion_config_hash,
        "output_schema_version": config.output_schema_version,
        "encoder_identity": dict(encoder_identity or video_encoder_identity()),
    }
    sidecar = {
        "schema": "robo_collector.raw_materialization_provenance.v1",
        **episode_metadata,
        "source_episode_ids": [source_episode_id],
        "source_manifest_hashes": [source_manifest_hash],
        "alignment_policy": config.alignment_policy,
        "dropped_selection_count": dropped_selection_count,
        "alignment_residual_sec": _residual_stats(residuals),
        "episode_provenance": {source_episode_id: dict(episode_metadata)},
    }
    if not source_episode_id or Path(source_episode_id).name != source_episode_id:
        raise MaterializationError(
            f"source episode id is not safe for provenance path: {source_episode_id!r}"
        )

    meta_dir = dataset_root / "meta"
    sidecar_dir = meta_dir / "raw_provenance"
    sidecar_path = sidecar_dir / f"{source_episode_id}.json"
    existing_sidecar: Mapping[str, Any] | None = None
    if sidecar_path.is_file():
        try:
            value = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MaterializationError(
                f"existing episode provenance is unreadable: {sidecar_path}"
            ) from exc
        if not isinstance(value, Mapping):
            raise MaterializationError(
                f"existing episode provenance is not an object: {sidecar_path}"
            )
        existing_sidecar = value
        if any(
            existing_sidecar.get(key) != sidecar[key]
            for key in (
                "source_episode_id",
                "source_manifest_hash",
                "converter_version",
                "conversion_config_hash",
                "output_schema_version",
                "encoder_identity",
            )
        ):
            raise MaterializationError(
                "existing provenance sidecar has different identity: "
                f"{source_episode_id}"
            )

    # Keep the historical aggregate path as a compatibility/index document, but
    # never use it as the immutable identity of a single episode.  Existing
    # episode entries are merged so materializing a later episode cannot change
    # the provenance bytes referenced by an earlier episode row.
    aggregate_path = meta_dir / "raw_provenance.json"
    aggregate: dict[str, Any] = {}
    if aggregate_path.is_file():
        try:
            value = json.loads(aggregate_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MaterializationError(
                f"existing aggregate provenance is unreadable: {aggregate_path}"
            ) from exc
        if not isinstance(value, Mapping):
            raise MaterializationError(
                f"existing aggregate provenance is not an object: {aggregate_path}"
            )
        aggregate = dict(value)

    entries: dict[str, dict[str, Any]] = {}
    existing_entries = aggregate.get("episode_provenance")
    if isinstance(existing_entries, Mapping):
        entries.update(
            {
                str(key): dict(value)
                for key, value in existing_entries.items()
                if isinstance(value, Mapping)
            }
        )
    elif isinstance(existing_entries, list):
        for value in existing_entries:
            if isinstance(value, Mapping) and value.get("source_episode_id"):
                entries[str(value["source_episode_id"])] = dict(value)
    old_entry = entries.get(source_episode_id)
    if old_entry is not None and any(
        old_entry.get(key) != episode_metadata[key]
        for key in (
            "source_episode_id",
            "source_manifest_hash",
            "converter_version",
            "conversion_config_hash",
            "output_schema_version",
        )
    ):
        raise MaterializationError(
            "existing provenance for source episode has different identity: "
            f"{source_episode_id}"
        )
    entries[source_episode_id] = dict(episode_metadata)

    provenance_files: dict[str, str] = {}
    existing_files = aggregate.get("provenance_files")
    if isinstance(existing_files, Mapping):
        provenance_files.update(
            {str(key): str(value) for key, value in existing_files.items() if value}
        )
    provenance_files[source_episode_id] = str(
        sidecar_path.relative_to(dataset_root).as_posix()
    )
    episode_ids = list(entries)
    manifest_hashes = [str(entries[item]["source_manifest_hash"]) for item in episode_ids]
    aggregate.update(
        {
            "schema": "robo_collector.raw_materialization_provenance.v1.index",
            # These singleton fields are retained for old consumers.  Consumers
            # that support per-episode provenance must use episode_provenance or
            # provenance_files instead.
            "source_episode_id": episode_ids[0],
            "source_episode_ids": episode_ids,
            "source_manifest_hash": manifest_hashes[0],
            "source_manifest_hashes": manifest_hashes,
            "converter_version": entries[episode_ids[0]].get("converter_version"),
            "conversion_config_hash": entries[episode_ids[0]].get("conversion_config_hash"),
            "output_schema_version": entries[episode_ids[0]].get("output_schema_version"),
            "episode_count": len(episode_ids),
            "episode_provenance": entries,
            "provenance_files": provenance_files,
            "latest_source_episode_id": source_episode_id,
            "latest_alignment_policy": config.alignment_policy,
            "latest_dropped_selection_count": dropped_selection_count,
            "latest_alignment_residual_sec": _residual_stats(residuals),
        }
    )
    if existing_sidecar is None:
        _write_json_atomic(sidecar_path, sidecar)
    _write_json_atomic(aggregate_path, aggregate)
    return sidecar_path


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Write one JSON document with a durable replace boundary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
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
        temporary.unlink(missing_ok=True)


def _residual_stats(values: list[float]) -> dict[str, float | None]:
    finite = sorted(value for value in values if math.isfinite(value))
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


def _alignment_metrics(selections: list[_Selection]) -> dict[str, Any]:
    camera_skews: list[float] = []
    state_skews: list[float] = []
    for selection in selections:
        camera_times = [
            _record_time(selection.camera_by_stream[stream])
            for stream in selection.camera_by_stream
        ]
        if len(camera_times) >= 2:
            camera_skews.append(max(camera_times) - min(camera_times))
        elif len(camera_times) == 1:
            # Camera-camera skew is defined as zero when only one camera is
            # configured; an absent metric would incorrectly reject valid
            # single-stream datasets in the quality gate.
            camera_skews.append(0.0)
        state_time = _record_time(selection.state_record)
        state_skews.extend(
            abs(_record_time(selection.camera_by_stream[stream]) - state_time)
            for stream in selection.camera_by_stream
        )
    return {
        "camera_camera_skew_sec": _residual_stats(camera_skews).get("max"),
        "state_camera_skew_sec": _residual_stats(state_skews).get("max"),
        "camera_camera_skew": _residual_stats(camera_skews),
        "state_camera_skew": _residual_stats(state_skews),
    }


def _set_quality_status(episode: Path, status: str) -> None:
    """Commit the QC boundary and then the final quality decision.

    Keeping the intermediate ``QC`` write durable makes a crash between the
    artifact commit and the final decision recoverable: startup sees a
    materialized job with a QC marker and revalidates it.  The second write is
    intentionally separate so the manifest never jumps straight from
    ``MATERIALIZED`` to a publishable status without a persisted QC boundary.
    """
    if status not in {"READY", "REVIEW", "REJECT"}:
        raise ValueError(f"invalid final quality status: {status!r}")
    manifest_path = episode / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "QC"
    _atomic_manifest(manifest_path, manifest)
    manifest["status"] = status
    _atomic_manifest(manifest_path, manifest)


def _persist_materialization_qc(
    episode: Path,
    job_id: str,
    artifacts: Mapping[str, Any],
    quality: Mapping[str, Any],
) -> None:
    """Persist QC evidence and the job marker as one recoverable boundary."""
    manifest_path = episode / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = manifest.get("materialization_jobs")
    if not isinstance(jobs, list):
        raise MaterializationError("manifest materialization_jobs must be a list")
    for job in jobs:
        if isinstance(job, dict) and job.get("job_id") == job_id:
            job["status"] = "MATERIALIZED"
            job["updated_at"] = time.time()
            job["last_error"] = None
            job["artifacts"] = dict(artifacts)
            break
    else:
        raise MaterializationError(f"materialization job not found: {job_id}")
    manifest["artifacts"] = dict(artifacts)
    manifest["quality"] = dict(quality)
    # QC is deliberately persisted before the writer's shared index is
    # published.  Startup can revalidate this exact boundary after a crash.
    manifest["status"] = "QC"
    _atomic_manifest(manifest_path, manifest)


def _reset_invalid_materialized_job(
    episode: Path,
    job: Mapping[str, Any],
    config: MaterializationConfig,
) -> None:
    artifacts = job.get("artifacts")
    if isinstance(artifacts, Mapping):
        _remove_artifact_paths(artifacts, allowed_root=config.output_root)
    reset_materialization_job_for_retry(
        episode,
        str(job["job_id"]),
        error="materialized artifact evidence failed startup revalidation",
    )
    if isinstance(job, dict):
        job["status"] = "PENDING"
        job.pop("artifacts", None)


def _reuse_materialized_job(
    episode: Path,
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    quality_gate: EpisodeQualityGate,
) -> MaterializationResult | None:
    """Avoid appending a second episode when a prior publish already won."""

    if job.get("status") != "MATERIALIZED":
        return None
    artifacts = job.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    if artifacts.get("publication") == "PENDING":
        # The writer may have committed files but not the shared metadata
        # index when the process crashed.  Recovery below removes or
        # reconciles those files before a retry claims the job.
        return None
    data_value = artifacts.get("data")
    if not isinstance(data_value, str) or not data_value:
        return None
    data_path = Path(data_value)
    if not data_path.is_file() or data_path.stat().st_size <= 0:
        return None
    # Never trust a durable MATERIALIZED marker by itself: hashes, sizes,
    # counts, and decodability evidence are revalidated on every reuse.
    evidence = artifacts.get("evidence")
    if not isinstance(evidence, Mapping):
        return None
    if not _valid_encoder_identity(artifacts.get("encoder_identity")):
        return None
    evidence_items = [evidence.get("data"), evidence.get("provenance")]
    if evidence.get("quality") is not None:
        evidence_items.append(evidence.get("quality"))
    if isinstance(evidence.get("videos"), Mapping):
        evidence_items.extend(evidence["videos"].values())
    for item in evidence_items:
        if item is None:
            return None
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            return None
        candidate = Path(item["path"])
        if not candidate.is_file() or candidate.stat().st_size <= 0 or _sha256_file(candidate) != item.get("sha256"):
            return None
        if item is not evidence.get("provenance") and item.get("decodable") is not True and not str(item.get("validation", "")).startswith("decoder_unavailable:"):
            return None
        candidate_prefix = _read_file_prefix(candidate, 64)
        candidate_suffix = _read_file_suffix(candidate, 4)
        if candidate.suffix == ".parquet" and (candidate_prefix[:4] != b"PAR1" or candidate_suffix != b"PAR1"):
            return None
        if candidate.suffix == ".mp4" and b"ftyp" not in candidate_prefix:
            return None
    provenance_path = artifacts.get("provenance")
    if not isinstance(provenance_path, str) or not Path(provenance_path).is_file():
        return None
    try:
        provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    source_hash = str(manifest.get("raw_manifest_hash", manifest.get("manifest_hash", "")))
    if provenance.get("source_manifest_hash") != source_hash or provenance.get("conversion_config_hash") != job.get("conversion_config_hash"):
        return None
    quality = quality_gate.evaluate(episode / "manifest.json")
    quality_path = write_quality_report(episode, quality)
    refreshed_artifacts = dict(artifacts)
    refreshed_evidence = dict(evidence)
    refreshed_artifacts["quality"] = str(quality_path)
    refreshed_evidence["quality"] = _file_evidence(
        quality_path, 1, kind="quality"
    )
    refreshed_artifacts["evidence"] = refreshed_evidence
    update_materialization_job(
        episode,
        str(job["job_id"]),
        "MATERIALIZED",
        artifacts=refreshed_artifacts,
    )
    if isinstance(job, dict):
        job["artifacts"] = refreshed_artifacts
    if isinstance(quality, Mapping) and quality.get("status"):
        _set_quality_status(episode, str(quality["status"]))
    statistics = quality.get("statistics", {})
    dropped = 0
    if isinstance(statistics, Mapping):
        try:
            dropped = int(statistics.get("selection_gaps", 0) or 0)
        except (TypeError, ValueError):
            dropped = 0
    return MaterializationResult(
        episode_id=str(manifest["episode_id"]),
        job_id=str(job["job_id"]),
        output_dataset=data_path.parent.parent,
        frame_count=int(artifacts.get("frame_count", 0)),
        dropped_selection_count=dropped,
        quality_status=str(quality.get("status", manifest.get("status", "REVIEW"))),
        source_manifest_hash=str(
            job.get(
                "source_manifest_hash",
                manifest.get("raw_manifest_hash", manifest.get("manifest_hash", "")),
            )
        ),
    )


def _recover_pending_publication(
    episode: Path,
    job: Mapping[str, Any],
    config: MaterializationConfig,
) -> None:
    """Reconcile a crash between staged artifact commit and index publication."""
    artifacts = job.get("artifacts")
    if job.get("status") != "MATERIALIZED" or not isinstance(artifacts, Mapping):
        return
    if artifacts.get("publication") != "PENDING":
        return
    output_root = Path(config.output_root).resolve() / config.dataset_name
    episode_id = str(_read_manifest_episode_id(episode))
    # Recover the writer's all-files metadata transaction before inspecting
    # any individual document.  In particular, seeing an episode row is not
    # sufficient: a crash may have replaced episodes.jsonl while info.json or
    # modality.json still belongs to the previous generation.
    metadata = recover_dataset_metadata(config.output_root, config.dataset_name)
    if not _metadata_generation_is_consistent(
        metadata,
        fps=config.fps,
        camera_streams=config.camera_streams,
    ):
        phase = metadata.get("recovered_transaction_phase")
        phase_hint = f" after recovering {phase!r} transaction" if phase else ""
        raise MaterializationError(
            "cannot prove a complete committed dataset metadata generation"
            + phase_hint
        )
    episode_rows = [
        row
        for row in metadata.get("episodes", [])
        if isinstance(row, Mapping) and str(row.get("episode_id", "")) == episode_id
    ]
    for row in episode_rows:
        if _published_episode_matches(
            row,
            artifacts=artifacts,
            job=job,
            episode_id=episode_id,
            output_root=output_root,
        ):
            if not _pending_publication_artifacts_are_valid(
                artifacts,
                job=job,
                episode_id=episode_id,
                allow_non_ready=config.publish_non_ready,
            ):
                # A row with this identity already exists.  Preserve the
                # evidence and fail closed; deleting paths here could destroy
                # a partially corrupted but already indexed episode.
                raise MaterializationError(
                    "published metadata exists but derived artifact evidence "
                    "does not validate"
                )
            updated = dict(artifacts)
            updated["publication"] = "PUBLISHED"
            update_materialization_job(
                episode,
                str(job["job_id"]),
                "MATERIALIZED",
                artifacts=updated,
            )
            # ``job`` is the in-memory snapshot used by the caller for the
            # immediate idempotence check.  Keep it in sync with the durable
            # manifest so a crash after publication is recovered without
            # re-encoding the episode.
            if isinstance(job, dict):
                job["artifacts"] = updated
            return
    if episode_rows:
        raise MaterializationError(
            "dataset already contains a conflicting episode publication"
        )
    _remove_artifact_paths(artifacts, allowed_root=config.output_root)


def _pending_publication_artifacts_are_valid(
    artifacts: Mapping[str, Any],
    *,
    job: Mapping[str, Any],
    episode_id: str,
    allow_non_ready: bool,
) -> bool:
    """Revalidate real files before trusting a recovered publication marker."""
    evidence = artifacts.get("evidence")
    data_path = artifacts.get("data")
    videos = artifacts.get("videos")
    if (
        not isinstance(evidence, Mapping)
        or not isinstance(data_path, str)
        or not isinstance(videos, Mapping)
        or not isinstance(evidence.get("data"), Mapping)
        or not isinstance(evidence.get("videos"), Mapping)
        or set(videos) != set(evidence["videos"])
    ):
        return False
    try:
        frame_count = int(artifacts["frame_count"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if frame_count <= 0 or not _valid_encoder_identity(artifacts.get("encoder_identity")):
        return False

    def validate_file(
        path_value: Any,
        item: Any,
        *,
        kind: str,
    ) -> bool:
        if not isinstance(path_value, str) or not isinstance(item, Mapping):
            return False
        if item.get("path") != path_value:
            return False
        path = Path(path_value)
        if not path.is_file():
            return False
        try:
            actual = _file_evidence(path, frame_count if kind in {"parquet", "mp4"} else 1, kind=kind)
        except (OSError, MaterializationError):
            return False
        if actual.get("sha256") != item.get("sha256") or actual.get("size") != item.get("size"):
            return False
        # Recovery is a publication decision, so a decoder that is merely
        # unavailable is not evidence.  The normal materialization path may
        # still report that optional validation was unavailable, but a restart
        # must never turn an expected row/frame count into a trusted one.
        if actual.get("decodable") is not True:
            return False
        if kind == "parquet" and actual.get("row_count") != frame_count:
            return False
        if kind == "mp4" and actual.get("frame_count") != frame_count:
            return False
        return True

    if not validate_file(data_path, evidence["data"], kind="parquet"):
        return False
    for camera_key, path_value in videos.items():
        if not validate_file(
            path_value,
            evidence["videos"].get(camera_key),
            kind="mp4",
        ):
            return False

    source_hash = str(job.get("source_manifest_hash", "")).strip()
    config_hash = str(job.get("conversion_config_hash", "")).strip()
    schema_version = str(
        job.get("output_schema_version")
        or artifacts.get("output_schema_version", "")
    ).strip()
    provenance_path = artifacts.get("provenance")
    provenance_evidence = evidence.get("provenance")
    if (
        not source_hash
        or not config_hash
        or not schema_version
        or not isinstance(provenance_path, str)
        or not validate_file(provenance_path, provenance_evidence, kind="provenance")
    ):
        return False
    try:
        provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("source_episode_id") != episode_id
        or provenance.get("source_manifest_hash") != source_hash
        or provenance.get("conversion_config_hash") != config_hash
        or provenance.get("output_schema_version") != schema_version
    ):
        return False

    quality_path = artifacts.get("quality")
    quality_evidence = evidence.get("quality")
    if not validate_file(quality_path, quality_evidence, kind="quality"):
        return False
    try:
        quality = json.loads(Path(quality_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    allowed_statuses = {"READY", "REVIEW", "REJECT"} if allow_non_ready else {"READY"}
    return isinstance(quality, Mapping) and quality.get("status") in allowed_statuses


def _metadata_generation_is_consistent(
    metadata: Mapping[str, Any],
    *,
    fps: int,
    camera_streams: Sequence[str],
) -> bool:
    """Validate enough of all four metadata documents to prove publication."""
    if metadata.get("present") is not True:
        # An empty dataset is a valid pre-publication state.  The caller will
        # find no matching episode and clean up the unpublished artifacts.
        return not any(
            metadata.get(name) not in (None, [])
            for name in ("tasks", "episodes", "info", "modality")
        )
    tasks = metadata.get("tasks")
    episodes = metadata.get("episodes")
    info = metadata.get("info")
    modality = metadata.get("modality")
    if (
        not isinstance(tasks, list)
        or not isinstance(episodes, list)
        or not isinstance(info, Mapping)
        or not isinstance(modality, Mapping)
    ):
        return False
    try:
        task_indexes = [int(row["task_index"]) for row in tasks]
        if task_indexes != list(range(len(tasks))):
            return False
        if any(not isinstance(row.get("task"), str) or not row["task"].strip() for row in tasks):
            return False
        sorted_episodes = sorted(episodes, key=lambda row: int(row["episode_index"]))
        if [int(row["episode_index"]) for row in sorted_episodes] != list(range(len(episodes))):
            return False
        task_index_set = set(task_indexes)
        expected_from_index = 0
        expected_camera_keys = {
            f"observation.images.{stream}" for stream in camera_streams
        }
        seen_episode_ids: set[str] = set()
        for row in sorted_episodes:
            if not isinstance(row, Mapping):
                return False
            episode_id = str(row.get("episode_id", ""))
            if not episode_id or episode_id in seen_episode_ids:
                return False
            seen_episode_ids.add(episode_id)
            length = int(row["length"])
            from_index = int(row["dataset_from_index"])
            to_index = int(row["dataset_to_index"])
            if (
                length <= 0
                or from_index != expected_from_index
                or to_index != from_index + length
                or int(row["fps"]) != fps
                or int(row["task_index"]) not in task_index_set
            ):
                return False
            video_paths = row.get("video_paths")
            integrity = row.get("integrity")
            if (
                not isinstance(video_paths, Mapping)
                or set(video_paths) != expected_camera_keys
                or not isinstance(integrity, Mapping)
                or integrity.get("algorithm") != "sha256"
            ):
                return False
            video_integrity = integrity.get("videos")
            data_integrity = integrity.get("data")
            if (
                not isinstance(video_integrity, Mapping)
                or set(video_integrity) != expected_camera_keys
                or not isinstance(data_integrity, Mapping)
                or data_integrity.get("rows") != length
            ):
                return False
            for camera_key in expected_camera_keys:
                item = video_integrity.get(camera_key)
                if not isinstance(item, Mapping) or item.get("frames") != length:
                    return False
            expected_from_index = to_index
        expected_counts = {
            "total_episodes": len(episodes),
            "total_frames": expected_from_index,
            "total_tasks": len(tasks),
            "total_videos": len(episodes) * len(expected_camera_keys),
        }
        if any(info.get(key) != value for key, value in expected_counts.items()):
            return False
        features = info.get("features")
        if info.get("fps") != fps or not isinstance(features, Mapping):
            return False
        video_features = {
            str(key): value
            for key, value in features.items()
            if isinstance(key, str) and key.startswith("observation.images.")
        }
        if set(video_features) != expected_camera_keys:
            return False
        observation = modality.get("observation")
        if not isinstance(observation, Mapping):
            return False
        image_modalities = observation.get("images")
        if not isinstance(image_modalities, Mapping) or set(image_modalities) != set(
            camera_streams
        ):
            return False
        for stream in camera_streams:
            feature = video_features.get(f"observation.images.{stream}")
            item = image_modalities.get(stream)
            shape = feature.get("shape") if isinstance(feature, Mapping) else None
            feature_info = feature.get("info") if isinstance(feature, Mapping) else None
            if (
                not isinstance(feature, Mapping)
                or feature.get("dtype") != "video"
                or not isinstance(shape, list)
                or len(shape) != 3
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                    for value in shape
                )
                or not isinstance(feature_info, Mapping)
                or feature_info.get("video.fps") != fps
                or feature_info.get("video.height") != shape[0]
                or feature_info.get("video.width") != shape[1]
                or feature_info.get("video.channels") != shape[2]
                or not isinstance(item, Mapping)
                or item.get("key") != f"observation.images.{stream}"
                or item.get("dtype") != "rgb"
                or item.get("shape") != shape
                or item.get("fps") != fps
            ):
                return False

        def modality_fields_match(section_name: str, prefix: str) -> bool:
            section = observation.get(section_name)
            if not isinstance(section, Mapping):
                return False
            expected_fields = {
                key[len(prefix) :]
                for key, feature in features.items()
                if isinstance(key, str)
                and key.startswith(prefix)
                and isinstance(feature, Mapping)
            }
            if set(section) != expected_fields:
                return False
            for field in expected_fields:
                feature = features[f"{prefix}{field}"]
                item = section.get(field)
                if not isinstance(item, Mapping) or item.get("shape") != feature.get(
                    "shape"
                ):
                    return False
            return True

        if not modality_fields_match("state", "observation.state."):
            return False
        action = modality.get("action")
        if not isinstance(action, Mapping):
            return False
        expected_action_fields = {
            key[len("action.") :]
            for key, feature in features.items()
            if isinstance(key, str)
            and key.startswith("action.")
            and isinstance(feature, Mapping)
        }
        if set(action) != expected_action_fields:
            return False
        for field in expected_action_fields:
            feature = features[f"action.{field}"]
            item = action.get(field)
            if not isinstance(item, Mapping) or item.get("shape") != feature.get(
                "shape"
            ):
                return False
        annotation = modality.get("annotation")
        task_description = (
            annotation.get("human", {}).get("action", {}).get("task_description")
            if isinstance(annotation, Mapping)
            and isinstance(annotation.get("human"), Mapping)
            and isinstance(annotation["human"].get("action"), Mapping)
            else None
        )
        if (
            not isinstance(task_description, Mapping)
            or task_description.get("key")
            != "annotation.human.action.task_description"
            or task_description.get("dtype") != "string"
        ):
            return False
        task_feature = features.get("annotation.human.action.task_description")
        if (
            not isinstance(task_feature, Mapping)
            or task_feature.get("dtype") != "string"
            or task_feature.get("shape") != [1]
        ):
            return False
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return True


def _published_episode_matches(
    row: Any,
    *,
    artifacts: Mapping[str, Any],
    job: Mapping[str, Any],
    episode_id: str,
    output_root: Path,
) -> bool:
    """Require a complete provenance/integrity match before marking PUBLISHED."""
    if not isinstance(row, Mapping):
        return False
    source_hash = str(job.get("source_manifest_hash", "")).strip()
    config_hash = str(job.get("conversion_config_hash", "")).strip()
    schema_version = str(
        job.get("output_schema_version")
        or artifacts.get("output_schema_version", "")
    ).strip()
    if not source_hash or not config_hash or not schema_version:
        return False
    if (
        str(row.get("episode_id", "")) != episode_id
        or str(row.get("source_episode_id", "")) != episode_id
        or row.get("source_manifest_hash") != source_hash
        or row.get("conversion_config_hash") != config_hash
        or row.get("output_schema_version") != schema_version
        or artifacts.get("output_schema_version") != schema_version
    ):
        return False
    provenance = row.get("source_provenance")
    if isinstance(provenance, Mapping) and any(
        provenance.get(key) != value
        for key, value in {
            "source_episode_id": episode_id,
            "source_manifest_hash": source_hash,
            "conversion_config_hash": config_hash,
            "output_schema_version": schema_version,
        }.items()
        if key in provenance
    ):
        return False
    try:
        frame_count = int(artifacts["frame_count"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if frame_count <= 0 or row.get("length") != frame_count:
        return False
    evidence = artifacts.get("evidence")
    data_path = artifacts.get("data")
    videos = artifacts.get("videos")
    if (
        not isinstance(evidence, Mapping)
        or not isinstance(data_path, str)
        or not isinstance(videos, Mapping)
        or not isinstance(evidence.get("data"), Mapping)
        or not isinstance(evidence.get("videos"), Mapping)
    ):
        return False
    if set(videos) != set(evidence["videos"]):
        return False
    if not _published_path_matches(output_root, row.get("data_path"), data_path):
        return False
    row_video_paths = row.get("video_paths")
    if not isinstance(row_video_paths, Mapping) or set(row_video_paths) != set(videos):
        return False
    for camera_key, artifact_path in videos.items():
        if not isinstance(artifact_path, str) or not _published_path_matches(
            output_root, row_video_paths.get(camera_key), artifact_path
        ):
            return False
    integrity = row.get("integrity")
    data_integrity = integrity.get("data") if isinstance(integrity, Mapping) else None
    video_integrity = integrity.get("videos") if isinstance(integrity, Mapping) else None
    data_evidence = evidence["data"]
    if (
        not isinstance(data_integrity, Mapping)
        or not isinstance(video_integrity, Mapping)
        or data_integrity.get("rows") != frame_count
        or data_integrity.get("sha256") != data_evidence.get("sha256")
        or data_integrity.get("size_bytes") != data_evidence.get("size")
    ):
        return False
    if set(video_integrity) != set(videos):
        return False
    for camera_key in videos:
        item = video_integrity.get(camera_key)
        item_evidence = evidence["videos"].get(camera_key)
        if (
            not isinstance(item, Mapping)
            or not isinstance(item_evidence, Mapping)
            or item.get("frames") != frame_count
            or item.get("sha256") != item_evidence.get("sha256")
            or item.get("size_bytes") != item_evidence.get("size")
        ):
            return False
    return True


def _published_path_matches(
    output_root: Path,
    metadata_path: Any,
    artifact_path: str,
) -> bool:
    if not isinstance(metadata_path, str) or not metadata_path:
        return False
    relative = Path(metadata_path)
    if relative.is_absolute():
        return False
    candidate = (output_root / relative).resolve()
    try:
        candidate.relative_to(output_root.resolve())
    except ValueError:
        return False
    return candidate == Path(artifact_path).resolve()


def _read_manifest_episode_id(episode: Path) -> str:
    value = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or not value.get("episode_id"):
        raise MaterializationError("raw manifest is missing episode_id")
    return str(value["episode_id"])


def _remove_artifact_paths(
    artifacts: Mapping[str, Any], *, allowed_root: Path
) -> None:
    paths: list[str] = []
    data = artifacts.get("data")
    if isinstance(data, str):
        paths.append(data)
    videos = artifacts.get("videos")
    if isinstance(videos, Mapping):
        paths.extend(str(path) for path in videos.values() if isinstance(path, str))
    provenance = artifacts.get("provenance")
    if isinstance(provenance, str):
        paths.append(provenance)
    for value in paths:
        path = Path(value).resolve()
        try:
            path.relative_to(Path(allowed_root).resolve())
        except ValueError:
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue


def _atomic_manifest(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def _report(callback: Callable[[str], None] | None, phase: str) -> None:
    if callback is not None:
        try:
            callback(phase)
        except Exception:
            pass


__all__ = [
    "MaterializationConfig",
    "MaterializationError",
    "MaterializationResult",
    "RawEpisodeMaterializer",
    "robot_frame_from_mapping",
]
