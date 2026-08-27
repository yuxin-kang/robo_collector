"""Standard-library quality gate for raw episode manifests.

The gate is deliberately side-effect free: it reads a manifest (or accepts an
already decoded mapping) and returns a JSON-serialisable QC report.
"""

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping


_REJECT_REASON_PREFIXES = (
    "missing_",
    "invalid_",
    "strict_",
    "episode_not_closed",
    "artifact_missing_or_empty",
    "artifact_hash_mismatch",
    "artifact_count_missing",
    "artifact_count_mismatch",
    "artifact_not_valid_",
    "artifact_not_decodable",
    "provenance_missing_or_empty",
    "provenance_hash_mismatch",
    "invalid_provenance_document",
    "raw_integrity_failed",
    "materialization_failed",
    "no_materialized_artifact",
    "camera_capture_",
    "capture_binding_",
    "recording_",
    "invalid_records_rejected",
    "timer_deadline_",
    "state_age_",
    "camera_age_",
)

_MIN_CLOCK_MAPPING_SAMPLES = 2


class EpisodeQualityGate:
    """Evaluate a raw episode without modifying its manifest or artifacts."""

    def __init__(
        self,
        *,
        max_producer_gaps: int = 0,
        max_publisher_gaps: int = 0,
        max_unattributed_gaps: int = 0,
        max_transport_gaps: int = 0,
        max_selection_gaps: int = 0,
        max_duplicates: int = 0,
        max_reorders: int = 0,
        max_session_restarts: int = 0,
        max_timestamp_anomalies: int = 0,
        max_camera_camera_skew_sec: float = 0.1,
        max_state_camera_skew_sec: float = 0.1,
        max_camera_clock_mapping_uncertainty_sec: float | None = None,
        max_state_age_sec: float | None = None,
        max_camera_age_sec: float | None = None,
        max_timer_deadline_misses: int = 0,
        require_complete_capture: bool = False,
        materialization_policy: str = "strict",
        thresholds: Mapping[str, Any] | None = None,
        artifact_evidence: Mapping[str, Any] | None = None,
        **aliases: Any,
    ) -> None:
        # Accept the names used by both the manifest contract and ROS-style
        # parameters while keeping the public API centered on the gap classes
        # and two skew metrics.
        max_producer_gaps = aliases.pop("max_producer_gap", max_producer_gaps)
        max_unattributed_gaps = aliases.pop(
            "max_unattributed_gap", max_unattributed_gaps
        )
        max_publisher_gaps = aliases.pop("max_publisher_gap", max_publisher_gaps)
        max_transport_gaps = aliases.pop("max_transport_gap", max_transport_gaps)
        max_selection_gaps = aliases.pop("max_selection_gap", max_selection_gaps)
        max_duplicates = aliases.pop("max_duplicate_count", max_duplicates)
        max_reorders = aliases.pop("max_reorder_count", max_reorders)
        max_session_restarts = aliases.pop(
            "max_session_restart_count", max_session_restarts
        )
        max_timestamp_anomalies = aliases.pop(
            "max_timestamp_anomaly_count", max_timestamp_anomalies
        )
        max_camera_camera_skew_sec = aliases.pop(
            "max_camera_camera_skew", max_camera_camera_skew_sec
        )
        max_state_camera_skew_sec = aliases.pop(
            "max_state_camera_skew", max_state_camera_skew_sec
        )
        max_camera_clock_mapping_uncertainty_sec = aliases.pop(
            "max_camera_clock_mapping_uncertainty",
            max_camera_clock_mapping_uncertainty_sec,
        )
        max_camera_clock_mapping_uncertainty_sec = aliases.pop(
            "max_camera_clock_mapping_uncertainty_sec",
            max_camera_clock_mapping_uncertainty_sec,
        )
        max_state_age_sec = aliases.pop("max_state_age", max_state_age_sec)
        max_state_age_sec = aliases.pop("max_state_age_sec", max_state_age_sec)
        max_camera_age_sec = aliases.pop("max_camera_age", max_camera_age_sec)
        max_camera_age_sec = aliases.pop("max_camera_age_sec", max_camera_age_sec)
        max_timer_deadline_misses = aliases.pop(
            "max_deadline_misses", max_timer_deadline_misses
        )
        if aliases:
            raise TypeError("unexpected quality gate options: " + ", ".join(sorted(aliases)))
        values = {
            "max_producer_gaps": max_producer_gaps,
            "max_publisher_gaps": max_publisher_gaps,
            "max_unattributed_gaps": max_unattributed_gaps,
            "max_transport_gaps": max_transport_gaps,
            "max_selection_gaps": max_selection_gaps,
            "max_duplicates": max_duplicates,
            "max_reorders": max_reorders,
            "max_session_restarts": max_session_restarts,
            "max_timestamp_anomalies": max_timestamp_anomalies,
            "max_camera_camera_skew_sec": max_camera_camera_skew_sec,
            "max_state_camera_skew_sec": max_state_camera_skew_sec,
            "max_camera_clock_mapping_uncertainty_sec": (
                max_camera_clock_mapping_uncertainty_sec
            ),
        }
        if thresholds:
            for key, value in thresholds.items():
                canonical = {
                    "max_producer_gap": "max_producer_gaps",
                    "max_transport_gap": "max_transport_gaps",
                    "max_selection_gap": "max_selection_gaps",
                    "max_producer_gap_count": "max_producer_gaps",
                    "max_publisher_gap": "max_publisher_gaps",
                    "max_publisher_gap_count": "max_publisher_gaps",
                    "max_unattributed_gap": "max_unattributed_gaps",
                    "max_unattributed_gap_count": "max_unattributed_gaps",
                    "max_transport_gap_count": "max_transport_gaps",
                    "max_selection_gap_count": "max_selection_gaps",
                    "max_duplicate_count": "max_duplicates",
                    "max_duplicates": "max_duplicates",
                    "max_reorder_count": "max_reorders",
                    "max_reorders": "max_reorders",
                    "max_session_restart_count": "max_session_restarts",
                    "max_session_restarts": "max_session_restarts",
                    "max_timestamp_anomaly_count": "max_timestamp_anomalies",
                    "max_timestamp_anomalies": "max_timestamp_anomalies",
                    "max_camera_camera_skew": "max_camera_camera_skew_sec",
                    "max_state_camera_skew": "max_state_camera_skew_sec",
                    "max_camera_clock_mapping_uncertainty": (
                        "max_camera_clock_mapping_uncertainty_sec"
                    ),
                }.get(key, key)
                values[canonical] = value
        self.thresholds = {
            "max_producer_gaps": self._non_negative_int(values["max_producer_gaps"]),
            "max_publisher_gaps": self._non_negative_int(
                values["max_publisher_gaps"]
            ),
            "max_unattributed_gaps": self._non_negative_int(
                values["max_unattributed_gaps"]
            ),
            "max_transport_gaps": self._non_negative_int(values["max_transport_gaps"]),
            "max_selection_gaps": self._non_negative_int(values["max_selection_gaps"]),
            "max_duplicates": self._non_negative_int(values["max_duplicates"]),
            "max_reorders": self._non_negative_int(values["max_reorders"]),
            "max_session_restarts": self._non_negative_int(
                values["max_session_restarts"]
            ),
            "max_timestamp_anomalies": self._non_negative_int(
                values["max_timestamp_anomalies"]
            ),
            "max_camera_camera_skew_sec": self._non_negative_float(
                values["max_camera_camera_skew_sec"]
            ),
            "max_state_camera_skew_sec": self._non_negative_float(
                values["max_state_camera_skew_sec"]
            ),
            "max_camera_clock_mapping_uncertainty_sec": (
                self._optional_non_negative_float(
                    values["max_camera_clock_mapping_uncertainty_sec"]
                )
            ),
        }
        self.max_state_age_sec = self._optional_non_negative_float(max_state_age_sec)
        self.max_camera_age_sec = self._optional_non_negative_float(max_camera_age_sec)
        self.max_camera_clock_mapping_uncertainty_sec = (
            self._optional_non_negative_float(
                max_camera_clock_mapping_uncertainty_sec
            )
        )
        self.max_timer_deadline_misses = self._non_negative_int(
            max_timer_deadline_misses
        )
        if materialization_policy not in {"strict", "sparse"}:
            raise ValueError("materialization_policy must be 'strict' or 'sparse'")
        self.require_complete_capture = bool(require_complete_capture)
        self.materialization_policy = materialization_policy
        self.artifact_evidence = dict(artifact_evidence or {})

    def evaluate(self, manifest_or_path: Mapping[str, Any] | str | Path) -> dict[str, Any]:
        """Return a QC report for a mapping or a manifest JSON path."""
        report: dict[str, Any] = {
            "report_schema": "robo_collector.episode_quality.v1",
            "status": "REJECT",
            "rules": {
                "materialization_policy": self.materialization_policy,
                "require_complete_capture": self.require_complete_capture,
                "thresholds": dict(self.thresholds),
                "max_state_age_sec": self.max_state_age_sec,
                "max_camera_age_sec": self.max_camera_age_sec,
                "max_camera_clock_mapping_uncertainty_sec": (
                    self.max_camera_clock_mapping_uncertainty_sec
                ),
                "max_timer_deadline_misses": self.max_timer_deadline_misses,
            },
            "statistics": {},
            "reason": [],
        }
        episode_path = Path(manifest_or_path).parent if isinstance(manifest_or_path, (str, Path)) else None
        try:
            manifest = self._load(manifest_or_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            report["status"] = "QUARANTINED"
            report["reason"] = [f"manifest_unreadable: {exc}"]
            return report

        report["episode_id"] = manifest.get("episode_id")
        reasons = report["reason"]
        required = ("schema", "episode_id", "status", "source_scope", "streams")
        missing = [key for key in required if key not in manifest]
        if missing or not isinstance(manifest.get("streams"), Mapping):
            report["status"] = "QUARANTINED"
            reasons.append("missing_required_manifest_fields: " + ",".join(missing or ["streams"]))
            return report
        if manifest.get("status") == "DISCARDED":
            report["status"] = "DISCARDED"
            reasons.append("manifest_marked_discarded")
            return report
        status = manifest.get("status")
        if status == "QUARANTINED":
            report["status"] = "QUARANTINED"
            reasons.append("manifest_marked_quarantined")
            return report
        if status not in {"RAW_CLOSED", "MATERIALIZING", "MATERIALIZED", "QC", "READY", "REVIEW", "REJECT"}:
            reasons.append(f"episode_not_closed: status={manifest.get('status')!r}")
        elif status in {"RAW_CLOSED", "MATERIALIZING"}:
            # A valid raw capture is not yet a publishable training episode.
            # Keep its report, but never let a pre-materialization check emit
            # READY merely because the raw chunks passed integrity checks.
            reasons.append(f"not_materialized: status={status}")

        stats = self._statistics(manifest)
        report["statistics"] = stats
        self._check_stream_records(manifest, reasons)
        self._check_record_errors(manifest, reasons)
        self._check_gaps(stats, reasons)
        self._check_anomalies(stats, reasons)
        self._check_skew(stats, reasons)
        self._check_capture_health(manifest, stats, reasons)
        self._check_strict(manifest, reasons)
        self._check_artifacts(manifest, reasons)
        self._check_raw_integrity(episode_path, manifest, reasons)
        if episode_path is not None and manifest.get("status") in {
            "MATERIALIZED", "QC", "READY", "REVIEW", "REJECT"
        } and not (episode_path / "quality.json").is_file():
            reasons.append("missing_quality_report")

        if self.require_complete_capture and manifest.get("source_scope") == "transport_observed":
            reasons.append("transport_observed_cannot_be_ready_for_complete_capture")
        if reasons:
            report["status"] = "REVIEW"
            if any(
                reason.startswith(_REJECT_REASON_PREFIXES)
                for reason in reasons
            ):
                report["status"] = "REJECT"
        elif manifest.get("source_scope") == "transport_observed":
            report["status"] = "REVIEW"
            reasons.append("transport_observed_requires_review")
        else:
            report["status"] = "READY"
        return _json_safe(report)

    check = evaluate

    @staticmethod
    def _load(value: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
        if isinstance(value, Mapping):
            return value
        with Path(value).open("r", encoding="utf-8") as stream:
            loaded = json.load(stream)
        if not isinstance(loaded, Mapping):
            raise ValueError("manifest JSON must be an object")
        return loaded

    def _statistics(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        quality = manifest.get("quality") if isinstance(manifest.get("quality"), Mapping) else {}
        quality = quality.get("statistics", quality) if isinstance(quality, Mapping) else {}
        recording = quality.get("recording", {}) if isinstance(quality, Mapping) else {}
        if not isinstance(recording, Mapping):
            recording = {}
        result: dict[str, Any] = {}
        for name in (
            "producer_gaps",
            "publisher_gaps",
            "unattributed_gaps",
            "transport_gaps",
            "selection_gaps",
            "camera_camera_skew_sec",
            "state_camera_skew_sec",
            "selection_gaps_by_stream",
        ):
            aliases = {name, name.replace("_gaps", "_gap_count"), name.replace("_sec", "_max_sec")}
            if name.endswith("_skew_sec"):
                aliases.update({"max_" + name, name.replace("_skew_sec", "_skew_max_sec")})
            result[name] = next((quality[key] for key in aliases if key in quality), None)
        for name in (
            "recording_failed",
            "timer_ticks",
            "timer_deadline_misses",
            "raw_frame_count",
            "state_age_max_sec",
            "camera_age_max_sec",
            "state_age_sec",
            "camera_age_sec",
        ):
            result[name] = next(
                (
                    value
                    for source in (quality, recording)
                    if isinstance(source, Mapping)
                    for key, value in ((name, source.get(name)),)
                    if key in source
                ),
                None,
            )
        result["recording"] = dict(recording)
        record_errors = manifest.get("record_errors")
        if isinstance(record_errors, Mapping):
            result["record_errors"] = dict(record_errors)
            result["rejected_record_count"] = self._number(
                record_errors.get("total")
            )
        elif record_errors is not None:
            result["record_errors"] = record_errors
            result["rejected_record_count"] = None
        streams: dict[str, Any] = {}
        for stream_name, stream in manifest["streams"].items():
            if isinstance(stream, Mapping):
                streams[str(stream_name)] = {key: stream.get(key, 0) for key in (
                    "frame_count",
                    "producer_gap_count",
                    "publisher_gap_count",
                    "transport_gap_count",
                    "unattributed_gap_count",
                    "selection_gap_count",
                    "duplicate_count",
                    "reorder_count",
                    "session_restart_count",
                    "sequence_gap_count",
                    "timestamp_duplicate_count",
                    "timestamp_reorder_count",
                    "timestamp_monotonic",
                )}
        result["streams"] = streams
        for metric, stream_key in (
            ("duplicates", "duplicate_count"),
            ("reorders", "reorder_count"),
            ("session_restarts", "session_restart_count"),
        ):
            result[metric] = sum(
                self._number(item.get(stream_key)) or 0 for item in streams.values()
            )
        for metric, stream_key in (
            ("producer_gaps", "producer_gap_count"),
            ("publisher_gaps", "publisher_gap_count"),
            ("unattributed_gaps", "unattributed_gap_count"),
            ("transport_gaps", "transport_gap_count"),
            ("selection_gaps", "selection_gap_count"),
        ):
            stream_total = sum(
                self._number(item.get(stream_key)) or 0 for item in streams.values()
            )
            if result[metric] is None:
                result[metric] = stream_total
            else:
                # An explicit aggregate may come from a different capture
                # layer.  Never allow a smaller aggregate to mask counters
                # recomputed from the immutable stream statistics.
                explicit = self._number(result[metric])
                if explicit is not None:
                    result[metric] = max(explicit, stream_total)
        result["timestamp_anomalies"] = sum(
            (self._number(item.get("timestamp_duplicate_count")) or 0)
            + (self._number(item.get("timestamp_reorder_count")) or 0)
            for item in streams.values()
        )
        if result["camera_camera_skew_sec"] is None:
            result["camera_camera_skew_sec"] = _quality_value(
                quality, "camera_camera_skew_sec", "camera_camera_skew_max_sec"
            )
        if result["state_camera_skew_sec"] is None:
            result["state_camera_skew_sec"] = _quality_value(
                quality, "state_camera_skew_sec", "state_camera_skew_max_sec"
            )
        return result

    @staticmethod
    def _check_record_errors(
        manifest: Mapping[str, Any], reasons: list[str]
    ) -> None:
        """Do not publish a prefix that silently dropped malformed records."""
        if "record_errors" not in manifest:
            return
        errors = manifest.get("record_errors")
        if not isinstance(errors, Mapping):
            reasons.append("invalid_record_error_statistics")
            return
        total = errors.get("total")
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            reasons.append("invalid_record_error_statistics")
            return
        by_stream = errors.get("by_stream", {})
        by_type = errors.get("by_error_type", {})
        if not isinstance(by_stream, Mapping) or not isinstance(by_type, Mapping):
            reasons.append("invalid_record_error_statistics")
            return

        def counter_total(counters: Mapping[str, Any]) -> int | None:
            subtotal = 0
            for key, value in counters.items():
                if (
                    not isinstance(key, str)
                    or not key
                    or isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    return None
                subtotal += value
            return subtotal

        stream_total = counter_total(by_stream)
        type_total = counter_total(by_type)
        if stream_total is None or type_total is None:
            reasons.append("invalid_record_error_statistics")
            return
        if stream_total != total or type_total != total:
            reasons.append("invalid_record_error_statistics")
        if total:
            reasons.append(f"invalid_records_rejected: {total}")

    def _check_gaps(self, stats: Mapping[str, Any], reasons: list[str]) -> None:
        for metric, limit in (
            ("producer_gaps", self.thresholds["max_producer_gaps"]),
            ("publisher_gaps", self.thresholds["max_publisher_gaps"]),
            ("unattributed_gaps", self.thresholds["max_unattributed_gaps"]),
            ("transport_gaps", self.thresholds["max_transport_gaps"]),
            ("selection_gaps", self.thresholds["max_selection_gaps"]),
        ):
            value = self._number(stats.get(metric))
            if value is None:
                reasons.append(f"invalid_{metric}")
            elif value > limit:
                reasons.append(f"{metric}_exceed_threshold: {value} > {limit}")

    def _check_anomalies(self, stats: Mapping[str, Any], reasons: list[str]) -> None:
        for metric, limit in (
            ("duplicates", self.thresholds["max_duplicates"]),
            ("reorders", self.thresholds["max_reorders"]),
            ("session_restarts", self.thresholds["max_session_restarts"]),
            ("timestamp_anomalies", self.thresholds["max_timestamp_anomalies"]),
        ):
            value = self._number(stats.get(metric))
            if value is None:
                reasons.append(f"invalid_{metric}")
            elif value > limit:
                reasons.append(f"{metric}_exceed_threshold: {value} > {limit}")

    def _check_skew(self, stats: Mapping[str, Any], reasons: list[str]) -> None:
        for metric, limit in (("camera_camera_skew_sec", self.thresholds["max_camera_camera_skew_sec"]), ("state_camera_skew_sec", self.thresholds["max_state_camera_skew_sec"])):
            value = self._number(stats.get(metric))
            if value is None or value < 0:
                reasons.append(f"invalid_{metric}")
            elif value > limit:
                reasons.append(f"{metric}_exceed_threshold: {value} > {limit}")

    def _check_capture_health(
        self,
        manifest: Mapping[str, Any],
        stats: Mapping[str, Any],
        reasons: list[str],
    ) -> None:
        """Reject complete-capture claims without durable health evidence."""
        recording_failed = stats.get("recording_failed")
        if recording_failed is True:
            reasons.append("recording_failed")
        elif recording_failed is not None and recording_failed is not False:
            reasons.append("recording_health_unreported")

        deadline_misses = self._number(stats.get("timer_deadline_misses"))
        if deadline_misses is not None and deadline_misses < 0:
            reasons.append("invalid_timer_deadline_misses")
        elif deadline_misses is not None and deadline_misses > self.max_timer_deadline_misses:
            reasons.append(
                "timer_deadline_misses_exceed_threshold: "
                f"{deadline_misses} > {self.max_timer_deadline_misses}"
            )

        try:
            state_limit, camera_limit = self._capture_age_limits(manifest)
        except (TypeError, ValueError):
            # Malformed capture metadata must not disable the age gate or
            # crash QC.  Preserve explicit constructor limits and fail closed.
            reasons.append("invalid_capture_age_limit")
            state_limit, camera_limit = (
                self.max_state_age_sec,
                self.max_camera_age_sec,
            )
        for name, limit, reason_prefix in (
            ("state_age_max_sec", state_limit, "state_age"),
            ("camera_age_max_sec", camera_limit, "camera_age"),
        ):
            value = self._age_max(stats, name)
            if value is None and limit is not None:
                reasons.append(f"{reason_prefix}_unreported")
            elif value is not None and value < 0:
                reasons.append(f"invalid_{name}")
            elif value is not None and limit is not None and value > limit:
                reasons.append(f"{reason_prefix}_exceed_threshold: {value} > {limit}")

        if not self.require_complete_capture:
            return

        if manifest.get("source_scope") == "transport_observed":
            # The caller already receives the explicit complete-capture
            # rejection below.  Do not add camera-source binding requirements
            # to a host-observed shadow episode.
            return

        try:
            clock_mapping_limit = self._capture_clock_mapping_limit(manifest)
        except (TypeError, ValueError):
            reasons.append("invalid_camera_clock_mapping_uncertainty_limit")
            clock_mapping_limit = self.max_camera_clock_mapping_uncertainty_sec
        if clock_mapping_limit is None:
            reasons.append("clock_mapping_uncertainty_threshold_unreported")

        metadata = manifest.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        if metadata.get("camera_capture_attached") is not True:
            reasons.append("camera_capture_not_attached")
        sources = metadata.get("camera_capture_sources")
        binding = metadata.get("camera_capture_binding")
        if not isinstance(binding, Mapping) or binding.get("status") != "ATTACHED":
            reasons.append("camera_capture_binding_incomplete")
        if not isinstance(sources, list) or not sources:
            reasons.append("camera_capture_sources_unreported")
        else:
            observed_session_ids = (
                binding.get("observed_session_ids")
                if isinstance(binding, Mapping)
                else None
            )
            observed_ids: set[str] = set()
            observed_sessions_valid = False
            if not isinstance(observed_session_ids, list) or any(
                not isinstance(session_id, str) or not session_id.strip()
                for session_id in observed_session_ids
            ):
                reasons.append("camera_capture_observed_sessions_unreported")
            else:
                observed_sessions_valid = True
                observed_ids = {str(session_id) for session_id in observed_session_ids}
                if len(observed_ids) != len(observed_session_ids):
                    reasons.append("camera_capture_observed_sessions_duplicate")
            unbound_session_ids = (
                binding.get("unbound_observed_session_ids")
                if isinstance(binding, Mapping)
                else None
            )
            unbound_ids: set[str] = set()
            if not isinstance(unbound_session_ids, list) or any(
                not isinstance(session_id, str) or not session_id.strip()
                for session_id in unbound_session_ids
            ):
                reasons.append("camera_capture_unbound_sessions_unreported")
            else:
                unbound_ids = {str(session_id) for session_id in unbound_session_ids}
                if len(unbound_ids) != len(unbound_session_ids):
                    reasons.append("camera_capture_unbound_sessions_duplicate")
            configured_streams = metadata.get("camera_streams")
            configured_streams = (
                tuple(str(stream) for stream in configured_streams)
                if isinstance(configured_streams, (list, tuple))
                else ()
            )
            covered_streams: set[str] = set()
            bound_session_ids: list[str] = []
            for index, source in enumerate(sources):
                if not isinstance(source, Mapping):
                    reasons.append(f"camera_capture_source_invalid: {index}")
                    continue
                binding_status = source.get("binding_status")
                if binding_status != "BOUND":
                    reasons.append(
                        "camera_capture_source_not_bound: "
                        f"{source.get('session_id', index)} ({binding_status})"
                    )
                source_session_id = source.get("session_id")
                if not isinstance(source_session_id, str) or not source_session_id.strip():
                    reasons.append(f"camera_capture_source_session_missing: {index}")
                else:
                    bound_session_ids.append(source_session_id)
                snapshot = source.get("source_snapshot", source.get("snapshot"))
                snapshot_hash = source.get("source_snapshot_hash", source.get("snapshot_hash"))
                watermarks = source.get(
                    "stream_high_watermarks", source.get("high_watermarks")
                )
                if not isinstance(snapshot, Mapping):
                    reasons.append(f"camera_capture_snapshot_missing: {index}")
                if not isinstance(snapshot_hash, str) or not re.fullmatch(
                    r"[0-9a-fA-F]{64}", snapshot_hash
                ):
                    reasons.append(f"camera_capture_snapshot_hash_invalid: {index}")
                if not isinstance(watermarks, Mapping) or not watermarks:
                    reasons.append(f"camera_capture_high_watermark_missing: {index}")
                observed_watermarks = source.get("observed_stream_high_watermarks")
                if not isinstance(observed_watermarks, Mapping) or not observed_watermarks:
                    reasons.append(
                        f"camera_capture_observed_high_watermark_missing: {index}"
                    )
                if source.get("source_snapshot_consistent") is not True:
                    reasons.append(f"camera_capture_snapshot_inconsistent: {index}")
                if not isinstance(snapshot, Mapping) or snapshot.get("stable") is not True:
                    reasons.append(f"camera_capture_snapshot_not_stable: {index}")
                mapping_samples = self._number(source.get("clock_mapping_samples"))
                if (
                    mapping_samples is None
                    or mapping_samples < _MIN_CLOCK_MAPPING_SAMPLES
                    or int(mapping_samples) != mapping_samples
                ):
                    reasons.append(f"clock_mapping_samples_insufficient: {index}")
                uncertainty = source.get("clock_mapping_uncertainty_sec")
                if uncertainty is None and isinstance(binding, Mapping):
                    mapping = binding.get("clock_mapping")
                    source_mapping = (
                        mapping.get(str(source.get("session_id")))
                        if isinstance(mapping, Mapping)
                        else None
                    )
                    if isinstance(source_mapping, Mapping):
                        uncertainty = source_mapping.get("uncertainty_sec")
                if isinstance(uncertainty, bool):
                    uncertainty = None
                try:
                    uncertainty_value = (
                        None if uncertainty is None else float(uncertainty)
                    )
                except (TypeError, ValueError):
                    uncertainty_value = None
                if uncertainty_value is None or not math.isfinite(uncertainty_value):
                    reasons.append(
                        f"clock_mapping_uncertainty_unreported: {index}"
                    )
                elif uncertainty_value < 0:
                    reasons.append(
                        f"invalid_clock_mapping_uncertainty: {index}"
                    )
                elif (
                    clock_mapping_limit is not None
                    and uncertainty_value > clock_mapping_limit
                ):
                    reasons.append(
                        "clock_mapping_uncertainty_exceed_threshold: "
                        f"{index}: {uncertainty_value} > {clock_mapping_limit}"
                    )
                selected = source.get("selected_sequence_ranges")
                if not isinstance(selected, Mapping) or not selected:
                    reasons.append(
                        f"camera_capture_selected_ranges_missing: {index}"
                    )
                    selected = {}
                snapshot_record_counts = None
                if isinstance(snapshot, Mapping):
                    chunks = snapshot.get("chunks")
                    if not isinstance(chunks, Mapping) or not chunks:
                        reasons.append(
                            f"camera_capture_snapshot_chunks_missing: {index}"
                        )
                    snapshot_watermarks = snapshot.get("stream_high_watermarks")
                    if not isinstance(snapshot_watermarks, Mapping) or not snapshot_watermarks:
                        reasons.append(
                            f"camera_capture_snapshot_high_watermark_missing: {index}"
                        )
                    snapshot_record_counts = snapshot.get("selected_record_counts")
                    if not isinstance(snapshot_record_counts, Mapping):
                        reasons.append(
                            f"camera_capture_snapshot_selected_counts_missing: {index}"
                        )
                    source_manifest = source.get("source_manifest")
                    if isinstance(source_manifest, Mapping) and isinstance(
                        snapshot_hash, str
                    ):
                        expected_hash = _canonical_hash(
                            {
                                "manifest": dict(source_manifest),
                                "snapshot": dict(snapshot),
                            }
                        )
                        if snapshot_hash != expected_hash:
                            reasons.append(
                                f"camera_capture_snapshot_hash_mismatch: {index}"
                            )
                snapshot_stream_watermarks = (
                    snapshot.get("stream_high_watermarks")
                    if isinstance(snapshot, Mapping)
                    else None
                )
                if not isinstance(snapshot_stream_watermarks, Mapping):
                    snapshot_stream_watermarks = {}
                record_counts = source.get("record_counts")
                for raw_stream in selected:
                    stream = str(raw_stream)
                    selected_count = self._number(
                        selected.get(raw_stream, {}).get("count")
                        if isinstance(selected.get(raw_stream), Mapping)
                        else None
                    )
                    snapshot_selected_count = (
                        self._number(snapshot_record_counts.get(stream))
                        if isinstance(snapshot_record_counts, Mapping)
                        and stream in snapshot_record_counts
                        else None
                    )
                    if (
                        snapshot_selected_count is None
                        or selected_count is None
                        or snapshot_selected_count != selected_count
                    ):
                        reasons.append(
                            f"camera_capture_snapshot_selected_count_mismatch: {index}:{stream}"
                        )
                    source_hwm = (
                        watermarks.get(stream)
                        if isinstance(watermarks, Mapping)
                        else None
                    )
                    snapshot_hwm = snapshot_stream_watermarks.get(stream)
                    source_last = (
                        self._number(source_hwm.get("last_sequence"))
                        if isinstance(source_hwm, Mapping)
                        else None
                    )
                    snapshot_last = (
                        self._number(snapshot_hwm.get("last_sequence"))
                        if isinstance(snapshot_hwm, Mapping)
                        else None
                    )
                    stream_valid = True
                    if not isinstance(source_hwm, Mapping) or not isinstance(
                        snapshot_hwm, Mapping
                    ):
                        reasons.append(
                            f"camera_capture_stream_high_watermark_missing: {index}:{stream}"
                        )
                        stream_valid = False
                    elif (
                        source_last is None
                        or snapshot_last is None
                        or source_last < 0
                        or snapshot_last < 0
                    ):
                        reasons.append(
                            f"camera_capture_stream_high_watermark_invalid: {index}:{stream}"
                        )
                        stream_valid = False
                    elif source_last > snapshot_last:
                        reasons.append(
                            f"camera_capture_stream_high_watermark_ahead: {index}:{stream}"
                        )
                        stream_valid = False
                    observed_last = (
                        self._number(observed_watermarks.get(stream))
                        if isinstance(observed_watermarks, Mapping)
                        and stream in observed_watermarks
                        else None
                    )
                    if observed_last is not None and (
                        observed_last < 0
                        or snapshot_last is None
                        or observed_last > snapshot_last
                    ):
                        reasons.append(
                            f"camera_capture_observed_high_watermark_ahead: {index}:{stream}"
                        )
                        stream_valid = False
                    selected_range = selected.get(raw_stream)
                    if not isinstance(selected_range, Mapping):
                        reasons.append(
                            f"camera_capture_selected_range_missing: {index}:{stream}"
                        )
                        stream_valid = False
                    else:
                        first = self._number(selected_range.get("first_sequence"))
                        last = self._number(selected_range.get("last_sequence"))
                        count = self._number(selected_range.get("count"))
                        expected_count = (
                            self._number(record_counts.get(stream))
                            if isinstance(record_counts, Mapping)
                            and stream in record_counts
                            else None
                        )
                        if (
                            first is None
                            or last is None
                            or count is None
                            or first < 0
                            or last < first
                            or count <= 0
                            or (snapshot_last is not None and last > snapshot_last)
                            or (expected_count is not None and count != expected_count)
                        ):
                            reasons.append(
                                f"camera_capture_selected_range_invalid: {index}:{stream}"
                            )
                            stream_valid = False
                    if stream_valid:
                        covered_streams.add(stream)

            bound_ids = set(bound_session_ids)
            if len(bound_ids) != len(bound_session_ids):
                reasons.append("camera_capture_source_sessions_duplicate")
            if observed_sessions_valid:
                if bound_ids - observed_ids:
                    reasons.append("camera_capture_source_session_unobserved")
                expected_unbound = observed_ids - bound_ids
                if unbound_ids != expected_unbound:
                    reasons.append("camera_capture_unbound_sessions_mismatch")
                if expected_unbound:
                    reasons.append(
                        "camera_capture_observed_session_unbound: "
                        + ",".join(sorted(expected_unbound))
                    )
                if bound_ids != observed_ids:
                    reasons.append("camera_capture_observed_sessions_not_fully_bound")

            for stream in configured_streams:
                if stream not in covered_streams:
                    reasons.append(
                        f"camera_capture_stream_unreported: {stream}"
                    )

        if recording_failed is None:
            reasons.append("recording_health_unreported")
        if deadline_misses is None:
            reasons.append("timer_deadline_unreported")
        if state_limit is not None and self._age_max(stats, "state_age_max_sec") is None:
            reasons.append("state_age_unreported")
        if camera_limit is not None and self._age_max(stats, "camera_age_max_sec") is None:
            reasons.append("camera_age_unreported")

    def _capture_age_limits(
        self, manifest: Mapping[str, Any]
    ) -> tuple[float | None, float | None]:
        state_limit = self.max_state_age_sec
        camera_limit = self.max_camera_age_sec
        metadata = manifest.get("metadata")
        if isinstance(metadata, Mapping):
            capture_config = metadata.get("capture_config")
            alignment = (
                capture_config.get("alignment")
                if isinstance(capture_config, Mapping)
                else None
            )
            if isinstance(alignment, Mapping):
                if state_limit is None:
                    state_limit = self._optional_non_negative_float(
                        alignment.get("max_state_age_sec")
                    )
                if camera_limit is None:
                    camera_limit = self._optional_non_negative_float(
                        alignment.get("max_camera_age_sec")
                    )
        return state_limit, camera_limit

    def _capture_clock_mapping_limit(
        self, manifest: Mapping[str, Any]
    ) -> float | None:
        limit = self.max_camera_clock_mapping_uncertainty_sec
        if limit is not None:
            return limit
        metadata = manifest.get("metadata")
        if isinstance(metadata, Mapping):
            capture_config = metadata.get("capture_config")
            alignment = (
                capture_config.get("alignment")
                if isinstance(capture_config, Mapping)
                else None
            )
            if isinstance(alignment, Mapping):
                return self._optional_non_negative_float(
                    alignment.get("max_camera_clock_mapping_uncertainty_sec")
                )
        return None

    def _age_max(self, stats: Mapping[str, Any], name: str) -> float | None:
        value = self._number(stats.get(name))
        if value is not None:
            return value
        nested = stats.get(name.replace("_max_sec", "_sec"))
        if isinstance(nested, Mapping):
            return self._number(nested.get("max"))
        return self._number(nested)

    @staticmethod
    def _check_stream_records(manifest: Mapping[str, Any], reasons: list[str]) -> None:
        for stream_name, stream in manifest["streams"].items():
            if not isinstance(stream, Mapping):
                reasons.append(f"invalid_stream: {stream_name}")
            records = stream.get("records") if isinstance(stream, Mapping) else None
            if records is not None:
                for index, record in enumerate(records):
                    if not isinstance(record, Mapping):
                        reasons.append(f"invalid_record: {stream_name}[{index}]")
                        continue
                    missing = [key for key in ("session_id", "sequence", "clock_domain") if key not in record]
                    if not any(key in record for key in ("device_timestamp", "server_wall_timestamp", "receive_monotonic_timestamp", "record_monotonic_timestamp")):
                        missing.append("timestamp")
                    if missing:
                        reasons.append(f"missing_record_fields: {stream_name}[{index}]: {','.join(missing)}")

    def _check_strict(self, manifest: Mapping[str, Any], reasons: list[str]) -> None:
        if self.materialization_policy != "strict":
            return
        materialized = manifest.get("materialization") or manifest.get("quality") or {}
        if not isinstance(materialized, Mapping):
            return
        rows = materialized.get("parquet_row_count", materialized.get("row_count"))
        video_counts = materialized.get("video_frame_counts")
        if rows is not None and isinstance(video_counts, Mapping):
            if any(count != rows for count in video_counts.values()):
                reasons.append("strict_frame_count_mismatch")

    def _check_artifacts(self, manifest: Mapping[str, Any], reasons: list[str]) -> None:
        if manifest.get("status") not in {"MATERIALIZED", "QC", "READY", "REVIEW", "REJECT", "MATERIALIZING"} and "artifacts" not in manifest and "materialization" not in manifest:
            return
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, Mapping) or not _valid_encoder_identity(
            artifacts.get("encoder_identity")
        ):
            reasons.append("missing_encoder_evidence")
        evidence = self.artifact_evidence or (artifacts.get("evidence") if isinstance(artifacts, Mapping) else None)
        if not isinstance(evidence, Mapping):
            reasons.append("missing_artifact_evidence")
            return
        entries = []
        data = evidence.get("data")
        if data is not None: entries.append(("data", data))
        videos = evidence.get("videos")
        if isinstance(videos, Mapping): entries.extend((str(k), v) for k, v in videos.items())
        if data is None:
            reasons.append("missing_data_artifact_evidence")
        if not isinstance(videos, Mapping) or not videos:
            reasons.append("missing_video_artifact_evidence")
        if not entries:
            reasons.append("missing_artifact_evidence")
            return
        expected_rows = None
        for name, item in entries:
            if not isinstance(item, Mapping):
                reasons.append(f"invalid_artifact_evidence: {name}")
                continue
            path = item.get("path")
            if not isinstance(path, str) or not os.path.isfile(path) or os.path.getsize(path) <= 0:
                reasons.append(f"artifact_missing_or_empty: {name}")
                continue
            digest = item.get("sha256")
            if not isinstance(digest, str) or _sha256(Path(path)) != digest:
                reasons.append(f"artifact_hash_mismatch: {name}")
            count = item.get("row_count", item.get("frame_count"))
            if self._number(count) is None or self._number(count) < 0:
                reasons.append(f"artifact_count_missing: {name}")
            if name == "data": expected_rows = count
            elif expected_rows is not None and count != expected_rows:
                reasons.append(f"artifact_count_mismatch: {name}")
            validation = str(item.get("validation", ""))
            if item.get("decodable") is not True and not validation.startswith(
                "decoder_unavailable:"
            ):
                reasons.append(f"artifact_not_decodable: {name}")
            if validation.startswith("decoder_unavailable:"):
                reasons.append(f"artifact_decoder_unavailable: {name}")
            expected_count = item.get("expected_count")
            if expected_count is not None and count != expected_count:
                reasons.append(f"artifact_count_mismatch: {name}")
            suffix = Path(path).suffix.lower()
            try:
                artifact_path = Path(path)
                prefix = _read_file_prefix(artifact_path, 64)
                tail = _read_file_suffix(artifact_path, 4)
            except OSError:
                continue
            file_suffix = Path(path).suffix.lower()
            if name == "data" or file_suffix == ".parquet":
                if (
                    os.path.getsize(path) < 8
                    or prefix[:4] != b"PAR1"
                    or tail != b"PAR1"
                ):
                    reasons.append(f"artifact_not_valid_parquet: {name}")
                else:
                    _check_decoded_artifact(Path(path), "parquet", count, name, reasons)
            elif file_suffix == ".mp4" or name in (manifest.get("artifacts", {}).get("videos", {}) if isinstance(manifest.get("artifacts"), Mapping) else {}):
                if b"ftyp" not in prefix:
                    reasons.append(f"artifact_not_valid_mp4: {name}")
                else:
                    _check_decoded_artifact(Path(path), "mp4", count, name, reasons)
        quality_evidence = evidence.get("quality")
        if quality_evidence is not None:
            quality_path = (
                artifacts.get("quality")
                if isinstance(artifacts, Mapping)
                else None
            )
            if not isinstance(quality_evidence, Mapping):
                reasons.append("invalid_artifact_evidence: quality")
            else:
                evidence_path = quality_evidence.get("path")
                if not isinstance(evidence_path, str):
                    reasons.append("artifact_missing_or_empty: quality")
                elif quality_path is not None and quality_path != evidence_path:
                    reasons.append("artifact_path_mismatch: quality")
                elif not os.path.isfile(evidence_path) or os.path.getsize(evidence_path) <= 0:
                    reasons.append("artifact_missing_or_empty: quality")
                else:
                    if _sha256(Path(evidence_path)) != quality_evidence.get("sha256"):
                        reasons.append("artifact_hash_mismatch: quality")
                    if quality_evidence.get("expected_count") != 1:
                        reasons.append("artifact_count_mismatch: quality")
                    if quality_evidence.get("row_count") != 1:
                        reasons.append("artifact_count_mismatch: quality")
                    if quality_evidence.get("decodable") is not True:
                        reasons.append("artifact_not_decodable: quality")
                    if quality_evidence.get("validation") != "json":
                        reasons.append("artifact_not_valid_json: quality")
                    try:
                        with Path(evidence_path).open("r", encoding="utf-8") as stream:
                            quality_document = json.load(stream)
                        if not isinstance(quality_document, Mapping):
                            reasons.append("artifact_not_valid_json: quality")
                    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                        reasons.append("artifact_not_valid_json: quality")
        provenance = artifacts.get("provenance") if isinstance(artifacts, Mapping) else None
        provenance_evidence = evidence.get("provenance") if isinstance(evidence, Mapping) else None
        if not isinstance(provenance, str) or not isinstance(provenance_evidence, Mapping):
            reasons.append("missing_provenance_evidence")
        elif not os.path.isfile(provenance) or os.path.getsize(provenance) <= 0:
            reasons.append("provenance_missing_or_empty")
        elif _sha256(Path(provenance)) != provenance_evidence.get("sha256"):
            reasons.append("provenance_hash_mismatch")
        else:
            try:
                with Path(provenance).open("r", encoding="utf-8") as stream:
                    provenance_data = json.load(stream)
                if not isinstance(provenance_data, Mapping):
                    reasons.append("invalid_provenance_document")
                else:
                    required_fields = (
                        "source_episode_id",
                        "source_manifest_hash",
                        "converter_version",
                        "conversion_config_hash",
                        "output_schema_version",
                    )
                    for field in required_fields:
                        value = provenance_data.get(field)
                        if not isinstance(value, str) or not value.strip():
                            reasons.append(f"missing_provenance_field: {field}")
                        elif field.endswith("_hash") and not re.fullmatch(
                            r"[0-9a-fA-F]{64}", value
                        ):
                            reasons.append(f"invalid_provenance_field: {field}")
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                reasons.append("invalid_provenance_document")

    @staticmethod
    def _check_raw_integrity(
        episode_path: Path | None,
        manifest: Mapping[str, Any],
        reasons: list[str],
    ) -> None:
        if episode_path is None or not (episode_path / "manifest.json").is_file():
            return
        try:
            from .raw_episode import RawEpisodeReader
            RawEpisodeReader(episode_path).validate()
        except Exception as exc:
            reasons.append(f"raw_integrity_failed: {exc}")

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _non_negative_int(value: Any) -> int:
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError("gap thresholds must be non-negative integers")
        return int(value)

    @staticmethod
    def _non_negative_float(value: Any) -> float:
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError("skew thresholds must be non-negative finite numbers")
        return number

    @staticmethod
    def _optional_non_negative_float(value: Any) -> float | None:
        if value is None:
            return None
        return EpisodeQualityGate._non_negative_float(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _quality_value(quality: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in quality:
            return quality[name]
    return None


def _valid_encoder_identity(value: Any) -> bool:
    """Require enough encoder identity to reproduce a video artifact."""
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


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_file_prefix(path: Path, size: int) -> bytes:
    with path.open("rb") as stream:
        return stream.read(size)


def _read_file_suffix(path: Path, size: int) -> bytes:
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        file_size = stream.tell()
        stream.seek(max(0, file_size - size), os.SEEK_SET)
        return stream.read(size)


def _check_decoded_artifact(
    path: Path, kind: str, expected_count: Any, name: str, reasons: list[str]
) -> None:
    """Require an actual format reader before an artifact can support READY."""
    try:
        if kind == "parquet":
            import pyarrow.parquet as parquet
            actual = parquet.ParquetFile(path).metadata.num_rows
        else:
            import cv2
            capture = cv2.VideoCapture(str(path))
            try:
                if not capture.isOpened():
                    reasons.append(f"artifact_not_decodable: {name}")
                    return
                actual = 0
                while True:
                    ok, _ = capture.read()
                    if not ok:
                        break
                    actual += 1
            finally:
                capture.release()
        if actual != expected_count:
            reasons.append(f"artifact_count_mismatch: {name}")
    except ImportError:
        reasons.append(f"artifact_decoder_unavailable: {name}")
    except Exception:
        reasons.append(f"artifact_not_decodable: {name}")


def write_quality_report(
    episode: str | Path,
    report: Mapping[str, Any],
) -> Path:
    """Atomically write the QC report without changing the manifest."""
    episode_path = Path(episode).resolve()
    target = episode_path / "quality.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(_json_safe(dict(report)), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with temporary.open("rb") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY)
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
    return target
