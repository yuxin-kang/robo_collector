"""Frozen MCAP v1 contract helpers.

This module contains only deterministic, side-effect-light contract machinery.
The collector/ingestion runners own leases, retries, and orchestration; those
mutable execution details must never leak into the hashes produced here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import struct
import tempfile
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.protobuf import descriptor_pb2

from .mcap.v1 import episode_pb2
from .mcap.v1.episode_descriptor import DESCRIPTOR_SET_BYTES

MCAP_PROFILE = "robo_collector.mcap.v1"
PROTO_FILE_NAME = "robo_collector/mcap/v1/episode.proto"
MAX_MCAP_RECORD_BYTES = 64 * 1024 * 1024
MAX_CHECKPOINT_PAYLOAD_BYTES = 1024 * 1024
UINT32_MAX = (1 << 32) - 1
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
BUNDLE_MEMBER_PATHS = (
    "camera.mcap",
    "provenance.json",
    "quality.json",
    "robot.mcap",
    "stage-ledger.prepublish.json",
    "video-keyframes.parquet",
)

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
EPISODE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
DECIMAL_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")
SIGNED_DECIMAL_RE = re.compile(r"(?:0|-?[1-9][0-9]*)\Z")
CANONICAL_NUMBER_RE = re.compile(r"(?:0|-?[1-9][0-9]*)(?:\.[0-9]*[1-9])?\Z")
UCUM_UNIT_RE = re.compile(r"(?:1|[A-Za-z%][A-Za-z0-9%./*^(){}_-]*)\Z")
UINT64_MAX = (1 << 64) - 1


class McapContractError(ValueError):
    """Raised when data violates the frozen MCAP v1 contract."""


class CheckpointFrameError(McapContractError):
    """Raised when a checkpoint journal frame is invalid."""


class PublicationError(McapContractError):
    """Raised when an immutable bundle cannot enter the READY selector."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return RFC-8785 JCS bytes without a BOM or trailing newline."""

    try:
        import rfc8785
    except ImportError as exc:  # pragma: no cover - deployment packaging guard
        raise McapContractError(
            "rfc8785 is required for MCAP contract canonical JSON"
        ) from exc
    try:
        payload = rfc8785.dumps(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise McapContractError(f"value is not RFC-8785 serializable: {exc}") from exc
    if payload.endswith(b"\n") or payload.startswith(b"\xef\xbb\xbf"):
        raise AssertionError("RFC-8785 serializer emitted forbidden framing bytes")
    return payload


def canonical_json_hash(value: Any) -> str:
    """Return lowercase SHA-256 of :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def parse_canonical_json(payload: bytes | bytearray | memoryview) -> Any:
    """Parse JSON and reject bytes that are not their own canonical encoding."""

    raw = bytes(payload)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise McapContractError(f"invalid JSON payload: {exc}") from exc
    if canonical_json_bytes(value) != raw:
        raise McapContractError("JSON payload is not canonical RFC-8785 bytes")
    return value


def descriptor_set_bytes() -> bytes:
    """Return the checked-in protoc ``FileDescriptorSet`` for v1.

    ``FileDescriptor.CopyToProto`` does not preserve every protoc-emitted
    detail (notably explicit JSON names), so runtime reconstruction is not a
    byte-compatible descriptor strategy. The generated Python byte constant is
    the installable source of truth and is regenerated with the binary fixture.
    """

    output = descriptor_pb2.FileDescriptorSet.FromString(DESCRIPTOR_SET_BYTES)
    names = [item.name for item in output.file]
    if names != sorted(names) or names != [PROTO_FILE_NAME]:
        raise AssertionError("checked-in MCAP descriptor set is malformed")
    if any(item.HasField("source_code_info") for item in output.file):
        raise AssertionError("checked-in MCAP descriptor contains source info")
    return DESCRIPTOR_SET_BYTES


def descriptor_sha256() -> str:
    return hashlib.sha256(descriptor_set_bytes()).hexdigest()


MESSAGE_TYPES = {
    "CameraSampleV1": episode_pb2.CameraSampleV1,
    "VideoAccessUnitV1": episode_pb2.VideoAccessUnitV1,
    "RobotStateV1": episode_pb2.RobotStateV1,
    "RobotActionV1": episode_pb2.RobotActionV1,
    "AlignedSampleV1": episode_pb2.AlignedSampleV1,
    "EpisodeEventV1": episode_pb2.EpisodeEventV1,
}

FIXED_TOPICS = {
    "/robot/state/raw": "RobotStateV1",
    "/robot/action/raw": "RobotActionV1",
    "/alignment/sample": "AlignedSampleV1",
    "/episode/event": "EpisodeEventV1",
}

CHANNEL_METADATA_KEYS = frozenset(
    {
        "robo.stream_id",
        "robo.source_id",
        "robo.sensor_id",
        "robo.robot_id",
        "robo.frame_id",
        "robo.calibration_revision",
        "robo.codec",
        "robo.pixel_format",
        "robo.nominal_rate_hz",
        "robo.observed_rate_hz",
        "robo.clock_domain",
        "robo.schema_version",
        "robo.pipeline_version",
    }
)


def validate_identifier(value: str, *, field: str = "identifier") -> str:
    if not isinstance(value, str) or IDENTIFIER_RE.fullmatch(value) is None:
        raise McapContractError(f"invalid {field}: {value!r}")
    return value


def camera_topic(stream_id: str, *, h264: bool = True) -> str:
    validate_identifier(stream_id, field="stream_id")
    return f"/camera/{stream_id}/{'h264' if h264 else 'sample'}"


def message_type_for_topic(topic: str) -> str:
    if topic in FIXED_TOPICS:
        return FIXED_TOPICS[topic]
    match = re.fullmatch(r"/camera/([^/]+)/(sample|h264)", topic)
    if match is None:
        raise McapContractError(f"unregistered MCAP v1 topic: {topic!r}")
    validate_identifier(match.group(1), field="stream_id")
    return "VideoAccessUnitV1" if match.group(2) == "h264" else "CameraSampleV1"


def topic_rank(topic: str, *, camera_streams: Sequence[str] = ()) -> int:
    if topic == "/robot/state/raw":
        return 100
    if topic == "/robot/action/raw":
        return 200
    if topic == "/alignment/sample":
        return 300
    if topic == "/episode/event":
        return 9000
    match = re.fullmatch(r"/camera/([^/]+)/h264", topic)
    if match is None:
        raise McapContractError(f"topic has no canonical writer rank: {topic!r}")
    stream_id = validate_identifier(match.group(1), field="stream_id")
    ordered = sorted(
        (validate_identifier(item, field="stream_id") for item in camera_streams),
        key=lambda item: item.encode("utf-8"),
    )
    if len(set(ordered)) != len(ordered) or stream_id not in ordered:
        raise McapContractError(
            "camera_streams must be unique and contain the topic stream"
        )
    return 1000 + ordered.index(stream_id)


def validate_channel_metadata(
    metadata: Mapping[str, str], *, family: str, topic: str | None = None
) -> dict[str, str]:
    if not isinstance(metadata, Mapping):
        raise McapContractError("channel metadata must be a mapping")
    result = dict(metadata)
    unknown = set(result) - CHANNEL_METADATA_KEYS
    if unknown:
        raise McapContractError(
            "unknown channel metadata keys: " + ",".join(sorted(unknown))
        )
    required = {
        "camera": {
            "robo.stream_id",
            "robo.source_id",
            "robo.sensor_id",
            "robo.frame_id",
            "robo.calibration_revision",
            "robo.nominal_rate_hz",
            "robo.observed_rate_hz",
            "robo.clock_domain",
            "robo.schema_version",
            "robo.pipeline_version",
        },
        "robot": {
            "robo.source_id",
            "robo.robot_id",
            "robo.frame_id",
            "robo.nominal_rate_hz",
            "robo.observed_rate_hz",
            "robo.clock_domain",
            "robo.schema_version",
            "robo.pipeline_version",
        },
        "alignment": {"robo.robot_id", "robo.schema_version", "robo.pipeline_version"},
        "event": {"robo.robot_id", "robo.schema_version", "robo.pipeline_version"},
    }.get(family)
    if required is None:
        raise McapContractError(f"unknown channel family: {family!r}")
    missing = required - set(result)
    if family == "camera":
        representation = {"robo.codec", "robo.pixel_format"} & set(result)
        if len(representation) != 1:
            missing.add("exactly-one-of:robo.codec|robo.pixel_format")
    if missing:
        raise McapContractError(
            "missing channel metadata keys: " + ",".join(sorted(missing))
        )
    if not all(
        isinstance(key, str) and isinstance(value, str) for key, value in result.items()
    ):
        raise McapContractError(
            "channel metadata keys and values must be UTF-8 strings"
        )
    for key in ("robo.nominal_rate_hz", "robo.observed_rate_hz"):
        if key in result:
            _validate_canonical_rate(
                result[key], key, allow_zero=key.endswith("observed_rate_hz")
            )
    for key in (
        "robo.stream_id",
        "robo.source_id",
        "robo.sensor_id",
        "robo.robot_id",
        "robo.frame_id",
    ):
        if key in result:
            validate_identifier(result[key], field=key)
    for key in ("robo.schema_version", "robo.pipeline_version", "robo.clock_domain"):
        if key in result:
            _require_nonempty_string(result[key], key)
    if topic is not None:
        message_name = message_type_for_topic(topic)
        expected_family = (
            "camera"
            if message_name in {"CameraSampleV1", "VideoAccessUnitV1"}
            else "robot"
            if message_name in {"RobotStateV1", "RobotActionV1"}
            else "alignment"
            if message_name == "AlignedSampleV1"
            else "event"
        )
        if family != expected_family:
            raise McapContractError("channel family does not match topic")
        match = re.fullmatch(r"/camera/([^/]+)/(?:sample|h264)", topic)
        if match is not None and result.get("robo.stream_id") != match.group(1):
            raise McapContractError("camera topic stream does not match robo.stream_id")
        if topic.endswith("/h264") and "robo.codec" not in result:
            raise McapContractError("H.264 topic requires robo.codec metadata")
        if topic.endswith("/sample") and "robo.pixel_format" not in result:
            raise McapContractError(
                "camera sample topic requires robo.pixel_format metadata"
            )
    return result


def validate_mcap_sequence(sequence: int) -> int:
    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or not 0 <= sequence <= UINT32_MAX
    ):
        raise McapContractError("MCAP sequence must be an unsigned 32-bit integer")
    return sequence


def mcap_times(timestamp_ns: int) -> tuple[int, int]:
    if (
        isinstance(timestamp_ns, bool)
        or not isinstance(timestamp_ns, int)
        or not 0 <= timestamp_ns <= (1 << 63) - 1
    ):
        raise McapContractError("MCAP log time must be in 0..2^63-1")
    return timestamp_ns, timestamp_ns


def validate_timestamp_set(value: episode_pb2.TimestampSetV1) -> None:
    mode = value.normalization_mode
    reason = value.fallback_reason
    source_present = value.HasField("source_time_ns")
    if (
        value.receive_time_ns < 0
        or value.normalized_time_ns < 0
        or not value.clock_session_id
    ):
        raise McapContractError(
            "receive and normalized timestamps must be non-negative"
        )
    if mode == episode_pb2.NORMALIZATION_MODE_AFFINE_V2:
        valid = (
            source_present
            and reason == episode_pb2.CLOCK_FALLBACK_REASON_NONE
            and value.policy_version == "rgb_affine_v2"
            and value.uncertainty_ns <= 20_000_000
        )
    elif mode == episode_pb2.NORMALIZATION_MODE_RECEIVE_FALLBACK:
        valid_reasons = {
            episode_pb2.CLOCK_FALLBACK_REASON_SOURCE_TIME_MISSING,
            episode_pb2.CLOCK_FALLBACK_REASON_INSUFFICIENT_VALID_EDGES,
            episode_pb2.CLOCK_FALLBACK_REASON_AFFINE_UNCERTAINTY,
        }
        valid = (
            reason in valid_reasons
            and value.policy_version == "rgb_affine_v2"
            and value.normalized_time_ns == value.receive_time_ns
            and (
                source_present
                or reason == episode_pb2.CLOCK_FALLBACK_REASON_SOURCE_TIME_MISSING
            )
        )
    elif mode == episode_pb2.NORMALIZATION_MODE_COLLECTOR_DIRECT:
        valid = (
            source_present
            and value.source_clock_domain
            == episode_pb2.CLOCK_DOMAIN_COLLECTOR_MONOTONIC
            and value.source_time_ns
            == value.receive_time_ns
            == value.normalized_time_ns
            and value.uncertainty_ns == 0
            and reason == episode_pb2.CLOCK_FALLBACK_REASON_NONE
            and value.policy_version == "collector_direct_v1"
        )
    elif mode == episode_pb2.NORMALIZATION_MODE_LEGACY_V1:
        valid = (
            source_present
            and value.source_clock_domain == episode_pb2.CLOCK_DOMAIN_WALL_UTC
            and value.source_time_ns
            == value.receive_time_ns
            == value.normalized_time_ns
            and value.uncertainty_ns == 0
            and reason == episode_pb2.CLOCK_FALLBACK_REASON_NONE
            and value.policy_version == "legacy_rgb_v1"
            and value.clock_session_id.startswith("legacy.")
        )
    else:
        valid = False
    if not valid:
        raise McapContractError("TimestampSetV1 mode/reason/presence policy is invalid")


def validate_video_access_unit(
    value: episode_pb2.VideoAccessUnitV1,
    *,
    previous: episode_pb2.VideoAccessUnitV1 | None = None,
) -> None:
    """Validate one AU and optional pairwise generation transition.

    This is not the full encoder/publication authority: Annex-B NAL parsing,
    closed-GOP enforcement across an entire stream, and lock/ledger fencing
    remain runner/codec responsibilities.
    """

    validate_identifier(value.stream_id, field="stream_id")
    validate_timestamp_set(value.timestamps)
    if value.timestamps.normalized_time_ns > (1 << 63) - 1:
        raise McapContractError("video normalized timestamp exceeds int64")
    if value.timebase_num != 1 or value.timebase_den != 1_000_000_000:
        raise McapContractError("VideoAccessUnitV1 timebase must be 1/1000000000")
    if value.pts != value.timestamps.normalized_time_ns or value.dts != value.pts:
        raise McapContractError("VideoAccessUnitV1 PTS/DTS must equal normalized time")
    if value.codec != episode_pb2.PAYLOAD_ENCODING_H264_ANNEX_B:
        raise McapContractError("VideoAccessUnitV1 codec must be H264 Annex-B")
    if len(value.codec_config_sha256) != 32:
        raise McapContractError("codec_config_sha256 must contain 32 bytes")
    if not value.codec_config_annexb or hashlib.sha256(value.codec_config_annexb).digest() != value.codec_config_sha256:
        raise McapContractError("codec configuration bytes/hash mismatch")
    if not value.source_session_id or not value.profile or not value.level:
        raise McapContractError("video session/profile/level are required")
    if value.width == 0 or value.height == 0:
        raise McapContractError("video width and height must be positive")
    if not value.access_unit_annexb:
        raise McapContractError("VideoAccessUnitV1 access unit must not be empty")
    if previous is None:
        if value.config_generation != 0 or not value.keyframe:
            raise McapContractError("first video AU must be generation zero and keyframe")
        return
    if previous.stream_id != value.stream_id:
        raise McapContractError("pairwise video validation requires one stream")
    if value.config_generation not in {
        previous.config_generation,
        previous.config_generation + 1,
    }:
        raise McapContractError("video config generation must remain or increment by one")
    session_changed = value.source_session_id != previous.source_session_id
    config_changed = value.codec_config_sha256 != previous.codec_config_sha256
    if session_changed or config_changed:
        if value.config_generation != previous.config_generation + 1 or not value.keyframe:
            raise McapContractError("video session/config change requires new keyframe generation")
    elif value.config_generation != previous.config_generation:
        raise McapContractError("generation changed without session/config change")


def keyframe_arrow_schema() -> Any:
    """Return the exact non-null ``video-keyframes.parquet`` Arrow schema."""

    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover - deployment packaging guard
        raise McapContractError("pyarrow is required for the keyframe index") from exc
    return pa.schema(
        [
            pa.field("stream_id", pa.utf8(), nullable=False),
            pa.field("source_session_id", pa.utf8(), nullable=False),
            pa.field("source_sequence", pa.uint64(), nullable=False),
            pa.field("normalized_time_ns", pa.int64(), nullable=False),
            pa.field("mcap_log_time", pa.uint64(), nullable=False),
            pa.field("message_sequence", pa.uint32(), nullable=False),
            pa.field("config_generation", pa.uint32(), nullable=False),
            pa.field("pts", pa.int64(), nullable=False),
            pa.field("timebase_num", pa.uint32(), nullable=False),
            pa.field("timebase_den", pa.uint32(), nullable=False),
            pa.field("codec_config_sha256", pa.binary(32), nullable=False),
        ]
    )


@dataclass(frozen=True)
class StageDefinition:
    name: str
    stage_version: str
    output_schema_version: str
    config_template: Mapping[str, Any]
    required_parameters: frozenset[str]
    input_names: tuple[str, ...]
    output_paths: tuple[str, ...]

    def validate_config(self, config: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(config, Mapping):
            raise McapContractError(f"{self.name} config must be an object")
        result = dict(config)
        expected_keys = set(self.config_template) | set(self.required_parameters)
        if set(result) != expected_keys:
            raise McapContractError(
                f"{self.name} config must contain its exact v1 keys"
            )
        for key, expected in self.config_template.items():
            if result[key] != expected:
                raise McapContractError(f"{self.name}.{key} must equal {expected!r}")
        _reject_non_finite(result)
        _validate_stage_specific_config(self.name, result)
        return result


PREPUBLICATION_STAGE_ORDER = (
    "validate_landing",
    "normalize_clocks",
    "build_index",
    "align_rgb",
    "encode_video",
    "write_canonical_groups",
    "structural_qc",
    "content_qc",
)

STAGE_REGISTRY = {
    "validate_landing": StageDefinition(
        "validate_landing",
        "1",
        "1",
        {
            "format": "robo_collector.stage_config",
            "format_version": 1,
            "max_checkpoint_payload_bytes": "1048576",
            "max_record_content_bytes": "67108864",
            "require_data_crc": True,
            "require_summary": True,
            "stage_name": "validate_landing",
            "validation_profile": "landing_v1",
        },
        frozenset(),
        ("landing_mcap", "checkpoint_journal", "collection_manifest"),
        ("validation-report.json", "source-inventory.json"),
    ),
    "normalize_clocks": StageDefinition(
        "normalize_clocks",
        "1",
        "1",
        {
            "affine_min_edges": "30",
            "affine_ppm_limit": "2000",
            "clock_policy": "rgb_affine_v2",
            "even_sample_limit": "512",
            "fallback_max_interval_ns": "1000000000",
            "format": "robo_collector.stage_config",
            "format_version": 1,
            "max_uncertainty_ns": "20000000",
            "quantile_method": "nearest_rank",
            "stage_name": "normalize_clocks",
        },
        frozenset(),
        ("landing_mcap", "validation_report"),
        ("normalized-timeline.json", "clock-segments.json"),
    ),
    "build_index": StageDefinition(
        "build_index",
        "1",
        "1",
        {
            "format": "robo_collector.stage_config",
            "format_version": 1,
            "index_backend": "sqlite",
            "index_schema": "channel_time_v1",
            "page_size_bytes": "4096",
            "stage_name": "build_index",
            "synchronous": "FULL",
        },
        frozenset(),
        ("landing_mcap", "normalized_timeline"),
        ("channel-time-index.sqlite",),
    ),
    "align_rgb": StageDefinition(
        "align_rgb",
        "1",
        "1",
        {
            "action_max_age_ns": "20000000",
            "format": "robo_collector.stage_config",
            "format_version": 1,
            "max_camera_residual_ns": "20000000",
            "max_state_residual_ns": "20000000",
            "policy": "rgb_affine_v2",
            "policy_version": "2",
            "stage_name": "align_rgb",
        },
        frozenset({"reference_camera_stream"}),
        ("channel_time_index", "normalized_timeline"),
        ("aligned-rows.jcs", "selection-gaps.jcs"),
    ),
    "encode_video": StageDefinition(
        "encode_video",
        "1",
        "1",
        {
            "backend": "pyav-libx264",
            "format": "robo_collector.stage_config",
            "format_version": 1,
            "options": [
                {"name": "annexb", "value": "1"},
                {"name": "aud", "value": "1"},
                {"name": "bframes", "value": "0"},
                {"name": "closed_gop", "value": "1"},
                {"name": "crf", "value": "18"},
                {"name": "gop_duration_ns", "value": "1000000000"},
                {"name": "preset", "value": "medium"},
                {"name": "repeat_headers", "value": "1"},
                {"name": "scenecut", "value": "0"},
                {"name": "tune", "value": "zerolatency"},
            ],
            "stage_name": "encode_video",
        },
        frozenset({"backend_version"}),
        ("landing_mcap", "normalized_timeline"),
        ("encoded-access-units.jcs", "video-keyframes.parquet"),
    ),
    "write_canonical_groups": StageDefinition(
        "write_canonical_groups",
        "1",
        "1",
        {
            "camera_chunk_target_bytes": "16777216",
            "camera_compression": "NONE",
            "chunk_crc": True,
            "format": "robo_collector.stage_config",
            "format_version": 1,
            "index_types": "ALL",
            "profile": MCAP_PROFILE,
            "repeat_channels": True,
            "repeat_schemas": True,
            "robot_chunk_target_bytes": "4194304",
            "robot_compression": "ZSTD",
            "stage_name": "write_canonical_groups",
            "use_chunking": True,
            "use_statistics": True,
            "use_summary_offsets": True,
        },
        frozenset(),
        (
            "aligned_rows",
            "encoded_access_units",
            "landing_mcap",
            "normalized_timeline",
            "selection_gaps",
        ),
        ("camera.mcap", "robot.mcap", "provenance.json"),
    ),
    "structural_qc": StageDefinition(
        "structural_qc",
        "1",
        "1",
        {
            "doctor_policy": "ci_and_rollout",
            "format": "robo_collector.stage_config",
            "format_version": 1,
            "internal_validator": "mcap_v1",
            "require_cli_when_available": True,
            "stage_name": "structural_qc",
        },
        frozenset(),
        ("camera_mcap", "robot_mcap", "source_inventory", "video_keyframes"),
        ("structural-qc-evidence.json",),
    ),
    "content_qc": StageDefinition(
        "content_qc",
        "1",
        "1",
        {
            "format": "robo_collector.stage_config",
            "format_version": 1,
            "policy_name": "canonical_content_v1",
            "policy_version": "1",
            "stage_name": "content_qc",
        },
        frozenset(),
        (
            "aligned_rows",
            "clock_segments",
            "provenance",
            "source_inventory",
            "structural_qc_evidence",
            "camera_mcap",
            "robot_mcap",
        ),
        ("quality.json",),
    ),
}


def build_stage_config(stage_name: str, **parameters: Any) -> dict[str, Any]:
    definition = STAGE_REGISTRY.get(stage_name)
    if definition is None:
        raise McapContractError(f"unregistered stage: {stage_name!r}")
    if set(parameters) != set(definition.required_parameters):
        raise McapContractError(
            f"{stage_name} requires parameters: {','.join(sorted(definition.required_parameters)) or '<none>'}"
        )
    config = dict(definition.config_template)
    config.update(parameters)
    return definition.validate_config(config)


def build_stage_key(
    stage_name: str,
    *,
    config: Mapping[str, Any],
    implementation_id: str,
    input_hashes: Mapping[str, str] | Iterable[Mapping[str, str]],
    stage_version: str | None = None,
    output_schema_version: str | None = None,
) -> dict[str, Any]:
    definition = STAGE_REGISTRY.get(stage_name)
    if definition is None:
        raise McapContractError(f"unregistered stage: {stage_name!r}")
    if stage_version not in {None, definition.stage_version}:
        raise McapContractError("stage_version override is not valid for phase 0")
    if output_schema_version not in {None, definition.output_schema_version}:
        raise McapContractError(
            "output_schema_version override is not valid for phase 0"
        )
    validated = definition.validate_config(config)
    inputs = _normalize_input_hashes(input_hashes)
    if tuple(item["name"] for item in inputs) != tuple(
        sorted(definition.input_names, key=lambda value: value.encode())
    ):
        raise McapContractError(
            f"{stage_name} input hashes must contain exactly: {','.join(definition.input_names)}"
        )
    return {
        "config_sha256": canonical_json_hash(validated),
        "implementation_id": _require_nonempty_string(
            implementation_id, "implementation_id"
        ),
        "input_hashes": inputs,
        "output_schema_version": output_schema_version
        or definition.output_schema_version,
        "stage_name": stage_name,
        "stage_version": stage_version or definition.stage_version,
    }


def stage_key_hash(stage_key: Mapping[str, Any]) -> str:
    if set(stage_key) != {
        "config_sha256",
        "implementation_id",
        "input_hashes",
        "output_schema_version",
        "stage_name",
        "stage_version",
    }:
        raise McapContractError("stage key has unknown or missing v1 keys")
    return canonical_json_hash(dict(stage_key))


def project_stage_evidence(stages: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    stage_list = list(stages)
    names = [str(stage.get("stage_name")) for stage in stage_list]
    if len(names) != len(set(names)):
        raise McapContractError("duplicate stage evidence is forbidden")
    indexed = {name: stage for name, stage in zip(names, stage_list, strict=True)}
    if set(indexed) != set(PREPUBLICATION_STAGE_ORDER):
        raise McapContractError(
            "stage evidence must contain exactly the eight prepublication stages"
        )
    projected = []
    allowed = {
        "config_sha256",
        "implementation_id",
        "input_hashes",
        "output_hashes",
        "output_schema_version",
        "stage_key_sha256",
        "stage_name",
        "stage_version",
    }
    for name in PREPUBLICATION_STAGE_ORDER:
        stage = dict(indexed[name])
        unknown = set(stage) - allowed
        missing = allowed - set(stage)
        if missing or unknown:
            raise McapContractError(f"invalid stage evidence keys for {name}")
        stage["input_hashes"] = _normalize_input_hashes(stage["input_hashes"])
        stage["output_hashes"] = _normalize_output_hashes(stage["output_hashes"])
        if tuple(item["name"] for item in stage["input_hashes"]) != tuple(
            sorted(STAGE_REGISTRY[name].input_names, key=lambda value: value.encode())
        ):
            raise McapContractError(
                f"{name} input hashes do not match its exact projection"
            )
        if tuple(item["path"] for item in stage["output_hashes"]) != tuple(
            sorted(STAGE_REGISTRY[name].output_paths, key=lambda value: value.encode())
        ):
            raise McapContractError(
                f"{name} output hashes do not match its exact projection"
            )
        for key in ("config_sha256", "stage_key_sha256"):
            _require_sha256(stage[key], key)
        if stage["stage_version"] != "1" or stage["output_schema_version"] != "1":
            raise McapContractError(f"{name} must use phase-0 stage/output version 1")
        expected_stage_key_hash = canonical_json_hash(
            {
                key: stage[key]
                for key in (
                    "config_sha256",
                    "implementation_id",
                    "input_hashes",
                    "output_schema_version",
                    "stage_name",
                    "stage_version",
                )
            }
        )
        if stage["stage_key_sha256"] != expected_stage_key_hash:
            raise McapContractError(f"{name} stage_key_sha256 mismatch")
        projected.append(stage)
    return {
        "format": "robo_collector.stage_evidence",
        "format_version": 1,
        "stages": projected,
    }


def reduce_quality_rules(
    rules: Iterable[Mapping[str, Any]],
    *,
    policy_name: str,
    policy_version: str,
    policy_config: Mapping[str, Any],
    quarantined: bool = False,
) -> dict[str, Any]:
    normalized = []
    seen_ids: set[str] = set()
    for item in rules:
        rule = dict(item)
        allowed = {"evidence_sha256", "metrics", "result", "rule_id", "severity"}
        if set(rule) != allowed:
            raise McapContractError("quality rules have a closed v1 schema")
        rule_id = _require_nonempty_string(rule["rule_id"], "rule_id")
        if rule_id in seen_ids:
            raise McapContractError(f"duplicate quality rule_id: {rule_id}")
        seen_ids.add(rule_id)
        _require_sha256(rule["evidence_sha256"], "evidence_sha256")
        if rule["result"] not in {"PASS", "REVIEW", "FAIL"}:
            raise McapContractError("invalid quality rule result")
        if rule["severity"] not in {"INFO", "WARNING", "CRITICAL"}:
            raise McapContractError("invalid quality rule severity")
        metrics = []
        metric_keys: set[tuple[str, str]] = set()
        for metric_value in rule["metrics"]:
            metric = dict(metric_value)
            if set(metric) != {"name", "unit", "value"} or not all(
                isinstance(metric[key], str) for key in metric
            ):
                raise McapContractError(
                    "quality metric must contain string name/unit/value"
                )
            key = (metric["name"], metric["unit"])
            if key in metric_keys:
                raise McapContractError(f"duplicate quality metric: {key!r}")
            metric_keys.add(key)
            validate_metric_value(metric["value"])
            validate_metric_unit(metric["unit"])
            metrics.append(metric)
        rule["metrics"] = sorted(
            metrics, key=lambda value: (value["name"].encode(), value["unit"].encode())
        )
        normalized.append(rule)
    normalized.sort(key=lambda value: value["rule_id"].encode())
    passed = sum(rule["result"] == "PASS" for rule in normalized)
    review = sum(rule["result"] == "REVIEW" for rule in normalized)
    failed = sum(rule["result"] == "FAIL" for rule in normalized)
    if quarantined:
        status = "QUARANTINED"
    elif any(
        rule["result"] == "FAIL" and rule["severity"] == "CRITICAL"
        for rule in normalized
    ):
        status = "REJECT"
    elif review or any(
        rule["result"] == "FAIL" and rule["severity"] == "WARNING"
        for rule in normalized
    ):
        status = "REVIEW"
    else:
        status = "READY"
    return {
        "canonical_status": status,
        "format": "robo_collector.quality",
        "format_version": 1,
        "policy": {
            "config_sha256": canonical_json_hash(policy_config),
            "name": _require_nonempty_string(policy_name, "policy_name"),
            "version": _require_nonempty_string(policy_version, "policy_version"),
        },
        "rules": normalized,
        "summary": {
            "failed": str(failed),
            "passed": str(passed),
            "review": str(review),
        },
    }


def validate_metric_value(value: str) -> str:
    """Validate a deterministic Phase-0 metric string."""

    if not isinstance(value, str) or not value:
        raise McapContractError("metric value must be a non-empty string")
    if value in {"true", "false"} or re.fullmatch(r"[A-Z][A-Z0-9_]*", value):
        return value
    if "/" in value:
        if value.count("/") != 1:
            raise McapContractError("metric rational must contain one slash")
        numerator, denominator = value.split("/", 1)
        if (
            SIGNED_DECIMAL_RE.fullmatch(numerator) is None
            or DECIMAL_RE.fullmatch(denominator) is None
            or denominator == "0"
        ):
            raise McapContractError("metric rational is not canonical")
        if math.gcd(int(numerator), int(denominator)) != 1:
            raise McapContractError("metric rational must be reduced")
        return value
    if CANONICAL_NUMBER_RE.fullmatch(value) is None:
        raise McapContractError("metric value is not a canonical decimal/boolean/enum")
    if value == "-0" or value.startswith("-0."):
        raise McapContractError("negative zero is not canonical")
    return value


def validate_metric_unit(value: str) -> str:
    if not isinstance(value, str) or UCUM_UNIT_RE.fullmatch(value) is None:
        raise McapContractError("metric unit must be a deterministic UCUM ASCII token")
    return value


def build_qc_evidence(
    *,
    rule_id: str,
    rule_version: str,
    artifacts: Iterable[Mapping[str, str]] = (),
    observations: Iterable[Mapping[str, str]] = (),
) -> tuple[dict[str, Any], str]:
    artifact_rows = [dict(item) for item in artifacts]
    seen_artifacts: set[tuple[str, str]] = set()
    for item in artifact_rows:
        if set(item) != {"name", "sha256", "size_bytes"}:
            raise McapContractError("QC evidence artifact has invalid keys")
        key = (
            _require_nonempty_string(item["name"], "artifact name"),
            _require_sha256(item["sha256"], "artifact sha256"),
        )
        if key in seen_artifacts:
            raise McapContractError("duplicate QC evidence artifact")
        seen_artifacts.add(key)
        _require_decimal(item["size_bytes"], "artifact size_bytes")
    artifact_rows.sort(key=lambda item: (item["name"].encode(), item["sha256"]))
    observation_rows = [dict(item) for item in observations]
    seen_observations: set[tuple[str, str, str]] = set()
    for item in observation_rows:
        if set(item) != {"code", "subject", "value"} or not all(
            isinstance(item[key], str) for key in item
        ):
            raise McapContractError("QC evidence observation has invalid keys/types")
        key = (item["code"], item["subject"], item["value"])
        if key in seen_observations:
            raise McapContractError("duplicate QC evidence observation")
        seen_observations.add(key)
    observation_rows.sort(
        key=lambda item: (
            item["code"].encode(),
            item["subject"].encode(),
            item["value"].encode(),
        )
    )
    value = {
        "artifacts": artifact_rows,
        "format": "robo_collector.qc_evidence",
        "format_version": 1,
        "observations": observation_rows,
        "rule_id": _require_nonempty_string(rule_id, "rule_id"),
        "rule_version": _require_nonempty_string(rule_version, "rule_version"),
    }
    return value, canonical_json_hash(value)


def encode_checkpoint_frame(payload: Mapping[str, Any]) -> bytes:
    body = canonical_json_bytes(validate_checkpoint_payload(payload))
    if len(body) > MAX_CHECKPOINT_PAYLOAD_BYTES:
        raise CheckpointFrameError("checkpoint payload exceeds 1 MiB")
    return (
        struct.pack("<Q", len(body))
        + body
        + struct.pack("<I", zlib.crc32(body) & UINT32_MAX)
    )


def decode_checkpoint_frame(frame: bytes | bytearray | memoryview) -> dict[str, Any]:
    raw = bytes(frame)
    if len(raw) < 12:
        raise CheckpointFrameError("checkpoint frame is truncated")
    length = struct.unpack_from("<Q", raw)[0]
    if length > MAX_CHECKPOINT_PAYLOAD_BYTES:
        raise CheckpointFrameError("checkpoint payload exceeds 1 MiB")
    expected_length = 8 + length + 4
    if len(raw) != expected_length:
        raise CheckpointFrameError("checkpoint frame length mismatch")
    body = raw[8 : 8 + length]
    expected_crc = struct.unpack_from("<I", raw, 8 + length)[0]
    if zlib.crc32(body) & UINT32_MAX != expected_crc:
        raise CheckpointFrameError("checkpoint payload CRC mismatch")
    value = parse_canonical_json(body)
    if not isinstance(value, dict):
        raise CheckpointFrameError("checkpoint payload must be a JSON object")
    return validate_checkpoint_payload(value)


def iter_checkpoint_frames(
    journal: bytes | bytearray | memoryview,
    *,
    allow_torn_final: bool = False,
) -> Iterable[dict[str, Any]]:
    raw = bytes(journal)
    offset = 0
    while offset < len(raw):
        remaining = len(raw) - offset
        if remaining < 8:
            if allow_torn_final:
                return
            raise CheckpointFrameError("torn checkpoint length header")
        length = struct.unpack_from("<Q", raw, offset)[0]
        if length > MAX_CHECKPOINT_PAYLOAD_BYTES:
            raise CheckpointFrameError("checkpoint payload exceeds 1 MiB")
        end = offset + 8 + length + 4
        if end > len(raw):
            if allow_torn_final:
                return
            raise CheckpointFrameError("torn final checkpoint frame")
        yield decode_checkpoint_frame(raw[offset:end])
        offset = end


CHECKPOINT_KEYS = frozenset(
    {
        "accepted_snapshot_count",
        "accepted_snapshot_frontier",
        "channels",
        "checkpoint_sequence",
        "durable_byte_offset",
        "durable_count",
        "durable_frontier",
        "format",
        "format_version",
        "generation",
        "landing_prefix_sha256",
        "max_unsynced_records",
        "queue_capacity",
        "source_fences",
        "written_count",
        "written_frontier",
    }
)
CHECKPOINT_CHANNEL_KEYS = frozenset(
    {
        "accepted_count",
        "accepted_high_watermark",
        "channel",
        "durable_count",
        "durable_high_watermark",
        "last_packet_sequence",
        "last_source_sequence",
        "session_id",
        "written_count",
        "written_high_watermark",
    }
)
CHECKPOINT_FENCE_KEYS = frozenset(
    {
        "accepted_count",
        "durable_count",
        "durable_high_watermark",
        "end_sequence_inclusive",
        "session_id",
        "source_id",
        "start_sequence_exclusive",
        "written_count",
        "written_high_watermark",
    }
)


def validate_checkpoint_payload(
    payload: Mapping[str, Any], *, previous: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    result = dict(payload)
    if set(result) != CHECKPOINT_KEYS:
        raise CheckpointFrameError("checkpoint payload has unknown or missing v1 keys")
    if (
        result["format"] != "robo_collector.mcap_checkpoint"
        or result["format_version"] != 1
    ):
        raise CheckpointFrameError("unsupported checkpoint payload format")
    for key in (
        "accepted_snapshot_count",
        "checkpoint_sequence",
        "durable_byte_offset",
        "durable_count",
        "generation",
        "max_unsynced_records",
        "queue_capacity",
        "written_count",
    ):
        _require_decimal(result[key], key)
    for frontier, count in (
        ("accepted_snapshot_frontier", "accepted_snapshot_count"),
        ("durable_frontier", "durable_count"),
        ("written_frontier", "written_count"),
    ):
        _require_nullable_decimal(result[frontier], frontier)
        if (result[frontier] is None) != (result[count] == "0"):
            raise CheckpointFrameError(
                f"{frontier} nullability must match {count}"
            )
    _require_sha256(result["landing_prefix_sha256"], "landing_prefix_sha256")
    if not (
        int(result["accepted_snapshot_count"])
        >= int(result["written_count"])
        >= int(result["durable_count"])
    ):
        raise CheckpointFrameError(
            "checkpoint accepted/written/durable counts are inconsistent"
        )
    for key in (
        "accepted_snapshot_count",
        "accepted_snapshot_frontier",
        "checkpoint_sequence",
        "durable_byte_offset",
        "durable_count",
        "durable_frontier",
        "generation",
        "max_unsynced_records",
        "queue_capacity",
        "written_count",
        "written_frontier",
    ):
        if result[key] is not None and int(result[key]) > UINT64_MAX:
            raise CheckpointFrameError(f"{key} exceeds uint64")
    if result["queue_capacity"] == "0":
        raise CheckpointFrameError("queue_capacity must be positive")
    if not isinstance(result["channels"], list) or not isinstance(
        result["source_fences"], list
    ):
        raise CheckpointFrameError("checkpoint channels/source_fences must be arrays")
    channels = []
    previous_channel: bytes | None = None
    for value in result["channels"]:
        channel = dict(value)
        if set(channel) != CHECKPOINT_CHANNEL_KEYS:
            raise CheckpointFrameError("checkpoint channel has unknown or missing keys")
        channel_name = _require_nonempty_string(channel["channel"], "channel")
        encoded_name = channel_name.encode("utf-8")
        if previous_channel is not None and encoded_name <= previous_channel:
            raise CheckpointFrameError(
                "checkpoint channels must be uniquely sorted by channel"
            )
        previous_channel = encoded_name
        if not isinstance(channel["session_id"], str):
            raise CheckpointFrameError("checkpoint session_id must be a string")
        for key in ("accepted_count", "durable_count", "written_count"):
            _require_decimal(channel[key], key)
        if not (
            int(channel["accepted_count"])
            >= int(channel["written_count"])
            >= int(channel["durable_count"])
        ):
            raise CheckpointFrameError("checkpoint channel counts are inconsistent")
        for key in (
            "accepted_high_watermark",
            "durable_high_watermark",
            "last_packet_sequence",
            "last_source_sequence",
            "written_high_watermark",
        ):
            _require_nullable_decimal(channel[key], key)
        for watermark, count in (
            ("accepted_high_watermark", "accepted_count"),
            ("written_high_watermark", "written_count"),
            ("durable_high_watermark", "durable_count"),
        ):
            if (channel[watermark] is None) != (channel[count] == "0"):
                raise CheckpointFrameError(
                    f"{watermark} nullability must match {count}"
                )
        watermarks = [
            channel["accepted_high_watermark"],
            channel["written_high_watermark"],
            channel["durable_high_watermark"],
        ]
        if all(value is not None for value in watermarks) and not (
            int(watermarks[0]) >= int(watermarks[1]) >= int(watermarks[2])
        ):
            raise CheckpointFrameError("checkpoint channel watermarks are inconsistent")
        message_type_for_topic(channel_name)
        channels.append(channel)
    fences = []
    previous_fence: tuple[bytes, bytes] | None = None
    for value in result["source_fences"]:
        fence = dict(value)
        if set(fence) != CHECKPOINT_FENCE_KEYS:
            raise CheckpointFrameError(
                "checkpoint source fence has unknown or missing keys"
            )
        source_id = validate_identifier(fence["source_id"], field="source_id")
        session_id = _require_nonempty_string(fence["session_id"], "session_id")
        sort_key = (source_id.encode(), session_id.encode())
        if previous_fence is not None and sort_key <= previous_fence:
            raise CheckpointFrameError("source fences must be uniquely sorted")
        previous_fence = sort_key
        for key in (
            "accepted_count",
            "durable_count",
            "start_sequence_exclusive",
            "written_count",
        ):
            _require_decimal(fence[key], key)
        if not (
            int(fence["accepted_count"])
            >= int(fence["written_count"])
            >= int(fence["durable_count"])
        ):
            raise CheckpointFrameError(
                "checkpoint source-fence counts are inconsistent"
            )
        for key in (
            "durable_high_watermark",
            "end_sequence_inclusive",
            "written_high_watermark",
        ):
            _require_nullable_decimal(fence[key], key)
        for watermark, count in (
            ("written_high_watermark", "written_count"),
            ("durable_high_watermark", "durable_count"),
        ):
            if (fence[watermark] is None) != (fence[count] == "0"):
                raise CheckpointFrameError(
                    f"{watermark} nullability must match {count}"
                )
        if (
            fence["written_high_watermark"] is not None
            and fence["durable_high_watermark"] is not None
            and int(fence["written_high_watermark"])
            < int(fence["durable_high_watermark"])
        ):
            raise CheckpointFrameError("source-fence watermarks are inconsistent")
        if (
            fence["end_sequence_inclusive"] is not None
            and int(fence["end_sequence_inclusive"])
            <= int(fence["start_sequence_exclusive"])
        ):
            raise CheckpointFrameError("source fence end must follow its start")
        fences.append(fence)
    result["channels"] = channels
    result["source_fences"] = fences
    if previous is not None:
        _validate_checkpoint_progression(validate_checkpoint_payload(previous), result)
    return result


INPROGRESS_MANIFEST_REQUIRED_KEYS = frozenset(
    {
        "attempt",
        "collection_mode",
        "created_time_ns",
        "episode_id",
        "format",
        "format_version",
        "landing",
        "required_sources",
        "state",
        "writer",
    }
)


def encode_inprogress_manifest(manifest: Mapping[str, Any]) -> bytes:
    normalized = validate_inprogress_manifest(manifest)
    return canonical_json_bytes(normalized)


def decode_inprogress_manifest(
    payload: bytes | bytearray | memoryview,
) -> dict[str, Any]:
    value = parse_canonical_json(payload)
    if not isinstance(value, dict):
        raise McapContractError("in-progress manifest must be a JSON object")
    return validate_inprogress_manifest(value)


def validate_inprogress_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(manifest)
    if set(result) != INPROGRESS_MANIFEST_REQUIRED_KEYS:
        raise McapContractError("in-progress manifest has unknown or missing v1 keys")
    if (
        result["format"] != "robo_collector.mcap_landing"
        or result["format_version"] != 1
    ):
        raise McapContractError("unsupported in-progress manifest format")
    if result["state"] not in {"OPEN", "STOPPING", "FAULTED"}:
        raise McapContractError("invalid in-progress manifest state")
    if result["collection_mode"] not in {"raw_v1", "dual_write", "mcap_first"}:
        raise McapContractError("invalid collection_mode")
    _require_decimal(result["attempt"], "attempt")
    _require_decimal(result["created_time_ns"], "created_time_ns")
    episode_id = result["episode_id"]
    if not isinstance(episode_id, str) or EPISODE_ID_RE.fullmatch(episode_id) is None:
        raise McapContractError("invalid in-progress episode_id")
    landing = result["landing"]
    if not isinstance(landing, Mapping) or dict(landing) != {
        "checkpoint_path": "landing/checkpoints.bin",
        "partial_path": "landing/episode.mcap.partial",
        "writer_profile": MCAP_PROFILE,
    }:
        raise McapContractError("in-progress landing object does not match v1")
    writer = result["writer"]
    if not isinstance(writer, Mapping) or dict(writer) != {
        "mcap_library": "mcap",
        "mcap_version": "1.4.0",
        "profile": MCAP_PROFILE,
    }:
        raise McapContractError("in-progress writer object does not match v1")
    if not isinstance(result["required_sources"], list):
        raise McapContractError("required_sources must be an array")
    required_sources = []
    previous: tuple[bytes, bytes] | None = None
    for source_value in result["required_sources"]:
        source = dict(source_value)
        if set(source) != {"session_id", "source_id", "start_sequence_exclusive"}:
            raise McapContractError("required source has unknown or missing keys")
        source_id = validate_identifier(source["source_id"], field="source_id")
        session_id = _require_nonempty_string(source["session_id"], "session_id")
        _require_decimal(source["start_sequence_exclusive"], "start_sequence_exclusive")
        key = (source_id.encode(), session_id.encode())
        if previous is not None and key <= previous:
            raise McapContractError("required_sources must be uniquely sorted")
        previous = key
        required_sources.append(source)
    result["required_sources"] = required_sources
    _reject_non_finite(result)
    return result


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    return canonical_json_hash(dict(manifest))


def assert_ready_manifest(manifest: Mapping[str, Any]) -> tuple[str, str]:
    value = dict(manifest)
    if set(value) != {
        "bundle_hash",
        "canonical_status",
        "identity",
        "inventory",
        "manifest_version",
    }:
        raise PublicationError("manifest has unknown or missing v1 keys")
    if value["manifest_version"] != 1 or value["canonical_status"] != "READY":
        raise PublicationError(
            "only canonical_status=READY v1 manifests are publishable"
        )
    bundle_hash = _require_sha256(value["bundle_hash"], "bundle_hash")
    if canonical_json_hash(value["identity"]) != bundle_hash:
        raise PublicationError("bundle_hash does not match canonical identity")
    _validate_identity(value["identity"])
    _validate_manifest_inventory(value["identity"], value["inventory"])
    return bundle_hash, manifest_hash(value)


def publish_ready_pointer(
    canonical_dir: str | Path,
    manifest: Mapping[str, Any],
    pointer: Mapping[str, Any],
    *,
    prevalidated_authority: bool = False,
) -> Path:
    """Atomically replace ``current.json`` after a strict READY/hash gate.

    The caller must already hold the publisher file lock and matching ledger
    transaction/fencing token. This helper intentionally does not invent an
    authority model outside the ingestion ledger.
    """

    bundle_hash, expected_manifest_hash = assert_ready_manifest(manifest)
    if not prevalidated_authority:
        raise PublicationError(
            "pointer-only helper requires caller-held lock, ledger transaction, "
            "and fencing validation (prevalidated_authority=True)"
        )
    value = dict(pointer)
    required = {
        "bundle_hash",
        "episode_id",
        "manifest_hash",
        "publication_generation",
        "publication_profile",
        "publisher_fencing_token",
        "publisher_stage_key",
    }
    if set(value) != required:
        raise PublicationError("current pointer has unknown or missing v1 keys")
    if (
        value["bundle_hash"] != bundle_hash
        or value["manifest_hash"] != expected_manifest_hash
    ):
        raise PublicationError("current pointer does not select the supplied manifest")
    identity = manifest["identity"]
    if value["episode_id"] != identity["episode_id"]:
        raise PublicationError("pointer episode_id does not match bundle identity")
    _require_sha256(value["publisher_stage_key"], "publisher_stage_key")
    for key in ("publication_generation", "publisher_fencing_token"):
        if not isinstance(value[key], str) or DECIMAL_RE.fullmatch(value[key]) is None:
            raise PublicationError(f"{key} must be a canonical decimal string")
    root = _safe_publication_path(canonical_dir, "canonical root")
    target = _safe_publication_path(root / "current.json", "READY pointer", root=root)
    _atomic_write_bytes(target, canonical_json_bytes(value))
    return target


publish_ready_bundle = publish_ready_pointer


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = _safe_publication_path(path, "READY pointer")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _safe_publication_path(
    path: str | os.PathLike[str], label: str, *, root: Path | None = None
) -> Path:
    candidate = Path(os.path.abspath(os.fspath(path)))
    boundary = candidate if root is None else Path(os.path.abspath(os.fspath(root)))
    if root is not None:
        try:
            candidate.relative_to(boundary)
        except ValueError as exc:
            raise PublicationError(f"{label} escapes canonical root") from exc
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current /= component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise PublicationError(f"cannot inspect {label}: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise PublicationError(f"{label} contains a symlink component")
    return candidate


def _normalize_input_hashes(
    value: Mapping[str, str] | Iterable[Mapping[str, str]],
) -> list[dict[str, str]]:
    items = (
        [{"name": name, "sha256": digest} for name, digest in value.items()]
        if isinstance(value, Mapping)
        else [dict(item) for item in value]
    )
    if any(set(item) != {"name", "sha256"} for item in items):
        raise McapContractError("input hashes require exactly name and sha256")
    names: set[str] = set()
    for item in items:
        name = _require_nonempty_string(item["name"], "input hash name")
        if name in names:
            raise McapContractError(f"duplicate input hash name: {name}")
        names.add(name)
        _require_sha256(item["sha256"], "input sha256")
    return sorted(items, key=lambda item: (item["name"].encode(), item["sha256"]))


def _normalize_output_hashes(
    value: Iterable[Mapping[str, str]],
) -> list[dict[str, str]]:
    items = [dict(item) for item in value]
    if any(set(item) != {"path", "sha256", "size_bytes"} for item in items):
        raise McapContractError("output hashes require path, sha256, and size_bytes")
    paths: set[str] = set()
    for item in items:
        path = _require_nonempty_string(item["path"], "output path")
        if path in paths:
            raise McapContractError(f"duplicate output hash path: {path}")
        paths.add(path)
        _require_sha256(item["sha256"], "output sha256")
        if (
            not isinstance(item["size_bytes"], str)
            or DECIMAL_RE.fullmatch(item["size_bytes"]) is None
        ):
            raise McapContractError(
                "output size_bytes must be a canonical decimal string"
            )
    return sorted(items, key=lambda item: (item["path"].encode(), item["sha256"]))


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise McapContractError(f"{field} must be 64 lowercase hexadecimal characters")
    return value


def _require_decimal(value: Any, field: str) -> str:
    if not isinstance(value, str) or DECIMAL_RE.fullmatch(value) is None:
        raise McapContractError(f"{field} must be a canonical unsigned decimal string")
    return value


def _require_nullable_decimal(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _require_decimal(value, field)


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise McapContractError(f"{field} must be a non-empty string")
    return value


def _validate_canonical_rate(value: Any, field: str, *, allow_zero: bool) -> str:
    if not isinstance(value, str) or CANONICAL_NUMBER_RE.fullmatch(value) is None:
        raise McapContractError(f"{field} must be a finite canonical decimal string")
    if value.startswith("-") or (not allow_zero and value == "0"):
        raise McapContractError(
            f"{field} must be {'non-negative' if allow_zero else 'positive'}"
        )
    return value


def _reject_non_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise McapContractError("NaN and infinity are forbidden")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise McapContractError("JSON object keys must be strings")
            _reject_non_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_non_finite(item)


def _validate_stage_specific_config(stage_name: str, config: Mapping[str, Any]) -> None:
    if stage_name == "align_rgb":
        validate_identifier(
            config["reference_camera_stream"], field="reference_camera_stream"
        )
    elif stage_name == "encode_video":
        if config["backend_version"] != "17.1":
            raise McapContractError("encode_video.backend_version must equal '17.1'")
        expected_options = STAGE_REGISTRY["encode_video"].config_template["options"]
        if config["options"] != expected_options:
            raise McapContractError(
                "encode_video options must equal the complete sorted v1 registry"
            )


def _validate_checkpoint_progression(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    if int(current["checkpoint_sequence"]) != int(previous["checkpoint_sequence"]) + 1:
        raise CheckpointFrameError("checkpoint_sequence must increment by one")
    if current["generation"] != previous["generation"]:
        raise CheckpointFrameError("checkpoint generation changed within one journal")
    for key in (
        "accepted_snapshot_count",
        "accepted_snapshot_frontier",
        "durable_byte_offset",
        "durable_count",
        "durable_frontier",
        "written_count",
        "written_frontier",
    ):
        before, after = previous[key], current[key]
        if before is not None and (after is None or int(after) < int(before)):
            raise CheckpointFrameError(f"checkpoint field regressed: {key}")
    for group, identity_keys, monotonic_keys in (
        (
            "channels",
            ("channel",),
            (
                "accepted_count",
                "accepted_high_watermark",
                "durable_count",
                "durable_high_watermark",
                "last_packet_sequence",
                "last_source_sequence",
                "written_count",
                "written_high_watermark",
            ),
        ),
        (
            "source_fences",
            ("source_id", "session_id"),
            (
                "accepted_count",
                "durable_count",
                "durable_high_watermark",
                "end_sequence_inclusive",
                "written_count",
                "written_high_watermark",
            ),
        ),
    ):
        old_rows = {
            tuple(row[key] for key in identity_keys): row for row in previous[group]
        }
        new_rows = {
            tuple(row[key] for key in identity_keys): row for row in current[group]
        }
        if not set(old_rows).issubset(new_rows):
            raise CheckpointFrameError(f"checkpoint {group} identity disappeared")
        for identity, old in old_rows.items():
            new = new_rows[identity]
            for key in monotonic_keys:
                before, after = old[key], new[key]
                if before is not None and (after is None or int(after) < int(before)):
                    raise CheckpointFrameError(f"checkpoint {group}.{key} regressed")


def _validate_manifest_inventory(identity: Any, inventory: Any) -> None:
    if not isinstance(identity, Mapping) or not isinstance(inventory, Mapping):
        raise PublicationError("identity and inventory must be objects")
    inventory_value = dict(inventory)
    if set(inventory_value) != {
        "checksums_sha256",
        "end_log_time_ns",
        "files",
        "libraries",
        "start_log_time_ns",
        "topic_counts",
        "total_message_count",
    }:
        raise PublicationError("inventory has unknown or missing v1 keys")
    _require_sha256(inventory_value["checksums_sha256"], "checksums_sha256")
    for key in ("start_log_time_ns", "end_log_time_ns"):
        _require_nullable_decimal(inventory_value[key], key)
    _require_decimal(inventory_value["total_message_count"], "total_message_count")
    identity_members = identity.get("members")
    if not isinstance(identity_members, list):
        raise PublicationError("identity members must be an array")
    expected_members = []
    for item in identity_members:
        member = dict(item)
        if set(member) != {"path", "sha256", "size_bytes"}:
            raise PublicationError("identity member has invalid keys")
        _require_sha256(member["sha256"], "member sha256")
        _require_decimal(member["size_bytes"], "member size_bytes")
        expected_members.append(member)
    if tuple(item["path"] for item in expected_members) != BUNDLE_MEMBER_PATHS:
        raise PublicationError(
            "identity must contain the exact six sorted bundle members"
        )
    expected_checksums = {
        "algorithm": "sha256",
        "format": "robo_collector.checksums",
        "format_version": 1,
        "members": expected_members,
    }
    if inventory_value["checksums_sha256"] != canonical_json_hash(expected_checksums):
        raise PublicationError("checksums_sha256 does not match identity members")
    files = inventory_value["files"]
    if (
        not isinstance(files, list)
        or tuple(item.get("path") for item in files if isinstance(item, Mapping))
        != BUNDLE_MEMBER_PATHS
    ):
        raise PublicationError(
            "inventory files must contain the exact six sorted bundle members"
        )
    total_mcap_messages = 0
    for member, file_value in zip(expected_members, files, strict=True):
        file_row = dict(file_value)
        if set(file_row) != {"message_count", "path", "sha256", "size_bytes"}:
            raise PublicationError("inventory file has invalid keys")
        if any(
            file_row[key] != member[key] for key in ("path", "sha256", "size_bytes")
        ):
            raise PublicationError("inventory file hashes/sizes disagree with identity")
        if file_row["path"] in {"camera.mcap", "robot.mcap"}:
            _require_decimal(file_row["message_count"], "message_count")
            total_mcap_messages += int(file_row["message_count"])
        elif file_row["message_count"] is not None:
            raise PublicationError("non-MCAP inventory message_count must be null")
    topics = inventory_value["topic_counts"]
    if not isinstance(topics, list):
        raise PublicationError("topic_counts must be an array")
    previous_topic: bytes | None = None
    topic_total = 0
    for topic_value in topics:
        topic = dict(topic_value)
        if set(topic) != {"count", "topic"}:
            raise PublicationError("topic count has invalid keys")
        message_type_for_topic(topic["topic"])
        encoded = topic["topic"].encode()
        if previous_topic is not None and encoded <= previous_topic:
            raise PublicationError("topic_counts must be uniquely sorted")
        previous_topic = encoded
        _require_decimal(topic["count"], "topic count")
        topic_total += int(topic["count"])
    if (
        str(total_mcap_messages) != inventory_value["total_message_count"]
        or topic_total != total_mcap_messages
    ):
        raise PublicationError("inventory total_message_count is inconsistent")
    start_time = inventory_value["start_log_time_ns"]
    end_time = inventory_value["end_log_time_ns"]
    if total_mcap_messages == 0:
        if start_time is not None or end_time is not None:
            raise PublicationError("empty inventory must have null time bounds")
        raise PublicationError("an empty canonical candidate cannot be READY")
    if start_time is None or end_time is None or int(start_time) > int(end_time):
        raise PublicationError("non-empty inventory requires ordered time bounds")
    libraries = inventory_value["libraries"]
    if not isinstance(libraries, list) or [
        item.get("name") for item in libraries if isinstance(item, Mapping)
    ] != ["mcap", "protobuf", "pyarrow"]:
        raise PublicationError("inventory libraries must be mcap, protobuf, pyarrow")
    for library in libraries:
        if (
            set(library) != {"name", "version"}
            or not isinstance(library["version"], str)
            or not library["version"]
        ):
            raise PublicationError("inventory library row is invalid")
    if libraries[0]["version"] != "1.4.0":
        raise PublicationError("inventory mcap library version must equal 1.4.0")


def _validate_identity(identity: Any) -> None:
    if not isinstance(identity, Mapping):
        raise PublicationError("identity must be an object")
    value = dict(identity)
    if set(value) != {
        "alignment",
        "codec",
        "episode_id",
        "format",
        "format_version",
        "members",
        "pipeline",
        "schema",
        "source_artifacts",
    }:
        raise PublicationError("identity has unknown or missing v1 keys")
    if (
        value["format"] != "robo_collector.canonical_bundle"
        or value["format_version"] != 1
    ):
        raise PublicationError("unsupported canonical identity format")
    if (
        not isinstance(value["episode_id"], str)
        or EPISODE_ID_RE.fullmatch(value["episode_id"]) is None
    ):
        raise PublicationError("invalid identity episode_id")
    if value["alignment"] != {"policy": "rgb_affine_v2", "policy_version": "2"}:
        raise PublicationError("canonical identity requires rgb_affine_v2")
    codec = value["codec"]
    if not isinstance(codec, Mapping) or set(codec) != {
        "config_sha256",
        "name",
        "packetization",
        "profile_version",
    }:
        raise PublicationError("identity codec has invalid keys")
    _require_sha256(codec["config_sha256"], "codec.config_sha256")
    if {key: codec[key] for key in ("name", "packetization", "profile_version")} != {
        "name": "h264",
        "packetization": "annex_b_access_unit",
        "profile_version": "h264_annexb_au_v1",
    }:
        raise PublicationError("identity codec policy is invalid")
    pipeline = value["pipeline"]
    if not isinstance(pipeline, Mapping) or set(pipeline) != {
        "implementation_id",
        "stage_semantics_version",
    }:
        raise PublicationError("identity pipeline has invalid keys")
    _require_nonempty_string(
        pipeline["implementation_id"], "pipeline.implementation_id"
    )
    if pipeline["stage_semantics_version"] != "1":
        raise PublicationError("stage_semantics_version must equal 1")
    schema = value["schema"]
    if not isinstance(schema, Mapping) or set(schema) != {
        "mcap_profile",
        "protobuf_descriptor_sha256",
    }:
        raise PublicationError("identity schema has invalid keys")
    if schema["mcap_profile"] != MCAP_PROFILE:
        raise PublicationError("identity MCAP profile mismatch")
    if schema["protobuf_descriptor_sha256"] != descriptor_sha256():
        raise PublicationError("identity protobuf descriptor hash mismatch")
    sources = value["source_artifacts"]
    if not isinstance(sources, list) or not sources:
        raise PublicationError("identity source_artifacts must be non-empty")
    previous: tuple[bytes, str] | None = None
    seen_names: set[str] = set()
    for source_value in sources:
        source = dict(source_value)
        if set(source) != {"name", "sha256", "size_bytes"}:
            raise PublicationError("source artifact has invalid keys")
        name = _require_nonempty_string(source["name"], "source artifact name")
        if name in seen_names:
            raise PublicationError("duplicate source artifact name")
        seen_names.add(name)
        _require_sha256(source["sha256"], "source artifact sha256")
        _require_decimal(source["size_bytes"], "source artifact size_bytes")
        key = (name.encode(), source["sha256"])
        if previous is not None and key <= previous:
            raise PublicationError("source artifacts must be uniquely sorted")
        previous = key
