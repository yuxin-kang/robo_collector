"""Deterministic canonical MCAP bundle writer and structural validation.

This module deliberately owns no publication authority.  It produces isolated
``camera.mcap`` and ``robot.mcap`` candidates which must still pass the ledger
and publication gates before becoming current.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.protobuf.message import Message as ProtobufMessage

from . import mcap_contract
from .mcap.v1 import episode_pb2

MCAP_LIBRARY = "mcap-python/1.4.0"
CAMERA_CHUNK_TARGET_BYTES = 16 * 1024 * 1024
ROBOT_CHUNK_TARGET_BYTES = 4 * 1024 * 1024


class CanonicalMcapError(ValueError):
    """A canonical candidate violates the frozen v1 contract."""


@dataclass(frozen=True)
class CanonicalChannel:
    topic: str
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class CanonicalRecord:
    topic: str
    message: ProtobufMessage


@dataclass(frozen=True)
class CanonicalWriteResult:
    camera_path: Path
    robot_path: Path
    camera_message_count: int
    robot_message_count: int
    topic_counts: Mapping[str, int]
    start_log_time_ns: int | None
    end_log_time_ns: int | None
    keyframes: tuple[Mapping[str, Any], ...]


def annexb_nal_types(payload: bytes) -> tuple[int, ...]:
    """Return H.264 NAL unit types from a non-empty Annex-B access unit."""

    data = bytes(payload)
    starts: list[tuple[int, int]] = []
    index = 0
    while index <= len(data) - 3:
        if data[index : index + 4] == b"\x00\x00\x00\x01":
            starts.append((index, 4))
            index += 4
        elif data[index : index + 3] == b"\x00\x00\x01":
            starts.append((index, 3))
            index += 3
        else:
            index += 1
    if not starts or starts[0][0] != 0:
        raise CanonicalMcapError("H.264 access unit is not Annex-B")
    result: list[int] = []
    for position, (_, prefix_size) in enumerate(starts):
        nal_start = starts[position][0] + prefix_size
        nal_end = starts[position + 1][0] if position + 1 < len(starts) else len(data)
        if nal_start >= nal_end:
            raise CanonicalMcapError("H.264 access unit contains an empty NAL")
        result.append(data[nal_start] & 0x1F)
    return tuple(result)


def validate_h264_access_units(
    access_units: Iterable[episode_pb2.VideoAccessUnitV1],
) -> tuple[episode_pb2.VideoAccessUnitV1, ...]:
    """Validate stream-local ordering, Annex-B, generation and random access."""

    values = tuple(access_units)
    previous_by_stream: dict[str, episode_pb2.VideoAccessUnitV1] = {}
    for value in values:
        previous = previous_by_stream.get(value.stream_id)
        try:
            mcap_contract.validate_video_access_unit(value, previous=previous)
        except mcap_contract.McapContractError as exc:
            raise CanonicalMcapError(str(exc)) from exc
        nal_types = annexb_nal_types(value.access_unit_annexb)
        if value.keyframe != (5 in nal_types):
            raise CanonicalMcapError(
                "H.264 keyframe flag must exactly match IDR presence"
            )
        generation_start = previous is None or (
            value.config_generation != previous.config_generation
        )
        if generation_start and not {5, 7, 8}.issubset(nal_types):
            raise CanonicalMcapError(
                "first AU of every config generation must contain SPS, PPS, and IDR"
            )
        previous_by_stream[value.stream_id] = value
    return values


def _message_name(message: ProtobufMessage) -> str:
    return message.DESCRIPTOR.name


def _merge_key(
    record: CanonicalRecord, camera_streams: Sequence[str]
) -> tuple[Any, ...]:
    value = record.message
    timestamp = (
        value.reference_normalized_time_ns
        if isinstance(value, episode_pb2.AlignedSampleV1)
        else value.timestamps.normalized_time_ns
    )
    rank = mcap_contract.topic_rank(record.topic, camera_streams=camera_streams)
    if isinstance(value, episode_pb2.VideoAccessUnitV1):
        tail = (
            value.source_session_id.encode(),
            value.source_sequence,
            value.config_generation,
            value.collector_record_id,
        )
    elif isinstance(value, (episode_pb2.RobotStateV1, episode_pb2.RobotActionV1)):
        tail = (
            value.source_session_id.encode(),
            value.source_sequence,
            value.collector_record_id,
        )
    elif isinstance(value, episode_pb2.AlignedSampleV1):
        tail = (
            value.reference_session_id.encode(),
            value.reference_source_sequence,
            value.aligned_row_index,
        )
    else:
        raise CanonicalMcapError(
            f"unsupported canonical message: {_message_name(value)}"
        )
    return (timestamp, rank, *tail)


def _validate_record(record: CanonicalRecord) -> None:
    expected = mcap_contract.message_type_for_topic(record.topic)
    if _message_name(record.message) != expected:
        raise CanonicalMcapError(
            f"topic {record.topic!r} requires {expected}, got {_message_name(record.message)}"
        )
    if isinstance(record.message, episode_pb2.VideoAccessUnitV1):
        return
    if isinstance(
        record.message, (episode_pb2.RobotStateV1, episode_pb2.RobotActionV1)
    ):
        mcap_contract.validate_timestamp_set(record.message.timestamps)
    elif (
        isinstance(record.message, episode_pb2.AlignedSampleV1)
        and record.message.reference_normalized_time_ns < 0
    ):
        raise CanonicalMcapError("aligned reference time must be non-negative")


def _family(topic: str) -> str:
    if topic.startswith("/camera/"):
        return "camera"
    if topic.startswith("/robot/"):
        return "robot"
    if topic == "/alignment/sample":
        return "alignment"
    raise CanonicalMcapError(f"topic is not part of a canonical group: {topic!r}")


def _write_group(
    path: Path,
    records: Sequence[CanonicalRecord],
    channels: Mapping[str, CanonicalChannel],
    *,
    camera: bool,
) -> None:
    try:
        from mcap.writer import CompressionType, IndexType, Writer
    except ImportError as exc:  # pragma: no cover - deployment guard
        raise CanonicalMcapError("mcap==1.4.0 is required") from exc
    if importlib.metadata.version("mcap") != "1.4.0":
        raise CanonicalMcapError("mcap==1.4.0 is required")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise CanonicalMcapError(f"refusing to replace canonical candidate: {path}")
    with path.open("xb") as output:
        writer = Writer(
            output,
            chunk_size=CAMERA_CHUNK_TARGET_BYTES
            if camera
            else ROBOT_CHUNK_TARGET_BYTES,
            compression=CompressionType.NONE if camera else CompressionType.ZSTD,
            index_types=IndexType.ALL,
            repeat_channels=True,
            repeat_schemas=True,
            use_chunking=True,
            use_statistics=True,
            use_summary_offsets=True,
            enable_crcs=True,
            enable_data_crcs=True,
        )
        writer.start(profile=mcap_contract.MCAP_PROFILE, library=MCAP_LIBRARY)
        schema_ids: dict[str, int] = {}
        channel_ids: dict[str, int] = {}
        for topic in sorted({record.topic for record in records}, key=str.encode):
            name = mcap_contract.message_type_for_topic(topic)
            if name not in schema_ids:
                schema_ids[name] = writer.register_schema(
                    name=name,
                    encoding="protobuf",
                    data=mcap_contract.descriptor_set_bytes(),
                )
            channel = channels.get(topic)
            if channel is None:
                raise CanonicalMcapError(f"missing channel descriptor for {topic}")
            metadata = mcap_contract.validate_channel_metadata(
                channel.metadata, family=_family(topic), topic=topic
            )
            channel_ids[topic] = writer.register_channel(
                topic=topic,
                message_encoding="protobuf",
                schema_id=schema_ids[name],
                metadata=metadata,
            )
        for sequence, record in enumerate(records):
            timestamp = (
                record.message.reference_normalized_time_ns
                if isinstance(record.message, episode_pb2.AlignedSampleV1)
                else record.message.timestamps.normalized_time_ns
            )
            writer.add_message(
                channel_id=channel_ids[record.topic],
                log_time=timestamp,
                publish_time=timestamp,
                sequence=mcap_contract.validate_mcap_sequence(sequence),
                data=record.message.SerializeToString(deterministic=True),
            )
        writer.finish()


def write_canonical_groups(
    output_directory: str | Path,
    records: Iterable[CanonicalRecord],
    channels: Iterable[CanonicalChannel],
) -> CanonicalWriteResult:
    """Write deterministic, isolated camera and robot canonical candidates."""

    output = Path(output_directory)
    channel_values = tuple(channels)
    channel_map = {item.topic: item for item in channel_values}
    if len(channel_map) != len(channel_values):
        raise CanonicalMcapError("duplicate canonical channel topic")
    values = tuple(records)
    for record in values:
        _validate_record(record)
    camera_streams = sorted(
        {
            record.message.stream_id
            for record in values
            if isinstance(record.message, episode_pb2.VideoAccessUnitV1)
        },
        key=str.encode,
    )
    ordered = tuple(sorted(values, key=lambda item: _merge_key(item, camera_streams)))
    validate_h264_access_units(
        record.message
        for record in ordered
        if isinstance(record.message, episode_pb2.VideoAccessUnitV1)
    )
    camera_records = tuple(item for item in ordered if _family(item.topic) == "camera")
    robot_records = tuple(item for item in ordered if _family(item.topic) != "camera")
    camera_path, robot_path = output / "camera.mcap", output / "robot.mcap"
    _write_group(camera_path, camera_records, channel_map, camera=True)
    _write_group(robot_path, robot_records, channel_map, camera=False)
    topic_counts = {
        topic: sum(item.topic == topic for item in ordered)
        for topic in sorted({item.topic for item in ordered}, key=str.encode)
    }
    times = [
        item.message.reference_normalized_time_ns
        if isinstance(item.message, episode_pb2.AlignedSampleV1)
        else item.message.timestamps.normalized_time_ns
        for item in ordered
    ]
    keyframes = tuple(
        {
            "stream_id": item.message.stream_id,
            "source_session_id": item.message.source_session_id,
            "source_sequence": item.message.source_sequence,
            "normalized_time_ns": item.message.timestamps.normalized_time_ns,
            "mcap_log_time": item.message.timestamps.normalized_time_ns,
            "message_sequence": sequence,
            "config_generation": item.message.config_generation,
            "pts": item.message.pts,
            "timebase_num": item.message.timebase_num,
            "timebase_den": item.message.timebase_den,
            "codec_config_sha256": bytes(item.message.codec_config_sha256),
        }
        for sequence, item in enumerate(camera_records)
        if isinstance(item.message, episode_pb2.VideoAccessUnitV1)
        and item.message.keyframe
    )
    return CanonicalWriteResult(
        camera_path,
        robot_path,
        len(camera_records),
        len(robot_records),
        topic_counts,
        min(times) if times else None,
        max(times) if times else None,
        keyframes,
    )


def validate_canonical_mcap(path: str | Path, *, expected_group: str) -> dict[str, Any]:
    """Run mandatory internal structural validation and return stable evidence."""

    if expected_group not in {"camera", "robot"}:
        raise CanonicalMcapError("expected_group must be camera or robot")
    try:
        from mcap.reader import make_reader
    except ImportError as exc:  # pragma: no cover
        raise CanonicalMcapError("mcap==1.4.0 is required") from exc
    source = Path(path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    topic_counts: dict[str, int] = {}
    decoded_records: list[CanonicalRecord] = []
    camera_streams: set[str] = set()
    start: int | None = None
    end: int | None = None
    with source.open("rb") as stream:
        reader = make_reader(stream, validate_crcs=True)
        summary = reader.get_summary()
        if summary is None or summary.statistics is None:
            raise CanonicalMcapError("canonical MCAP requires a summary and statistics")
        if summary.statistics.chunk_count < 1:
            raise CanonicalMcapError("canonical MCAP must be chunked")
        for expected_sequence, (schema, channel, message) in enumerate(
            reader.iter_messages(log_time_order=False)
        ):
            if (
                schema is None
                or schema.encoding != "protobuf"
                or schema.data != mcap_contract.descriptor_set_bytes()
            ):
                raise CanonicalMcapError("canonical schema descriptor mismatch")
            family = _family(channel.topic)
            if (expected_group == "camera") != (family == "camera"):
                raise CanonicalMcapError("topic is stored in the wrong canonical group")
            mcap_contract.validate_channel_metadata(
                channel.metadata, family=family, topic=channel.topic
            )
            message_name = mcap_contract.message_type_for_topic(channel.topic)
            value = mcap_contract.MESSAGE_TYPES[message_name].FromString(message.data)
            _validate_record(CanonicalRecord(channel.topic, value))
            decoded_records.append(CanonicalRecord(channel.topic, value))
            if isinstance(value, episode_pb2.VideoAccessUnitV1):
                camera_streams.add(value.stream_id)
            if message.log_time != message.publish_time:
                raise CanonicalMcapError(
                    "canonical log_time and publish_time must match"
                )
            if message.sequence != expected_sequence:
                raise CanonicalMcapError(
                    "canonical MCAP sequence must be dense deterministic order"
                )
            expected_time = (
                value.reference_normalized_time_ns
                if isinstance(value, episode_pb2.AlignedSampleV1)
                else value.timestamps.normalized_time_ns
            )
            if message.log_time != expected_time:
                raise CanonicalMcapError(
                    "MCAP time does not match protobuf normalized time"
                )
            topic_counts[channel.topic] = topic_counts.get(channel.topic, 0) + 1
            start = message.log_time if start is None else min(start, message.log_time)
            end = message.log_time if end is None else max(end, message.log_time)
    ordered_streams = sorted(camera_streams, key=str.encode)
    if decoded_records != sorted(
        decoded_records, key=lambda item: _merge_key(item, ordered_streams)
    ):
        raise CanonicalMcapError(
            "canonical messages are not in deterministic merge order"
        )
    validate_h264_access_units(
        item.message
        for item in decoded_records
        if isinstance(item.message, episode_pb2.VideoAccessUnitV1)
    )
    return {
        "format": "robo_collector.structural_qc_evidence",
        "format_version": 1,
        "group": expected_group,
        "message_count": str(sum(topic_counts.values())),
        "sha256": digest,
        "size_bytes": str(source.stat().st_size),
        "start_log_time_ns": None if start is None else str(start),
        "end_log_time_ns": None if end is None else str(end),
        "topic_counts": [
            {"topic": key, "count": str(topic_counts[key])}
            for key in sorted(topic_counts, key=str.encode)
        ],
        "valid": True,
    }
