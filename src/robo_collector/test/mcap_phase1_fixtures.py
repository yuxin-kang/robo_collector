"""Small MCAP fixtures and byte-level fault injectors for Phase 1 tests."""

from __future__ import annotations

import io
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from mcap.opcode import Opcode
from mcap.writer import Writer
from robo_collector.mcap.v1 import episode_pb2

from robo_collector import mcap_contract

MAGIC = b"\x89MCAP0\r\n"


@dataclass(frozen=True)
class RecordRange:
    """Location of one complete MCAP record in a fixture."""

    opcode: int
    start: int
    content_start: int
    content_end: int


def sealed_mcap_bytes(
    *,
    enable_data_crcs: bool = True,
    use_statistics: bool = True,
    use_summary_offsets: bool = True,
    source_fences: Iterable[Mapping[str, Any]] = (),
    library: str = "mcap-python/1.4.0",
) -> bytes:
    """Return a minimal valid, unchunked, sealed MCAP made with public APIs."""

    output = io.BytesIO()
    writer = Writer(
        output,
        use_chunking=False,
        use_statistics=use_statistics,
        use_summary_offsets=use_summary_offsets,
        enable_crcs=True,
        enable_data_crcs=enable_data_crcs,
    )
    writer.start(profile="robo_collector.mcap.v1", library=library)
    schema_id = writer.register_schema(
        name="EpisodeEventV1",
        encoding="protobuf",
        data=mcap_contract.descriptor_set_bytes(),
    )
    channel_id = writer.register_channel(
        topic="/episode/event",
        message_encoding="protobuf",
        schema_id=schema_id,
        metadata={
            "robo.robot_id": "fixture_robot",
            "robo.schema_version": "1",
            "robo.pipeline_version": "phase1-test",
        },
    )
    fences = tuple(source_fences)
    for sequence, log_time in enumerate((10, 20)):
        message = episode_pb2.EpisodeEventV1(
            event_sequence=sequence,
            event_type=episode_pb2.EPISODE_EVENT_UNSPECIFIED,
            lifecycle_attempt=1,
            collector_record_id=sequence,
        )
        message.timestamps.normalized_time_ns = log_time
        if sequence == 1:
            for fence in fences:
                row = message.source_fences.add()
                for name, value in fence.items():
                    if value is not None:
                        setattr(row, name, int(value) if name not in {"source_id", "session_id"} else str(value))
        writer.add_message(
            channel_id=channel_id,
            log_time=log_time,
            publish_time=log_time,
            sequence=sequence,
            data=message.SerializeToString(deterministic=True),
        )
    writer.finish()
    return output.getvalue()


def record_ranges(payload: bytes) -> tuple[RecordRange, ...]:
    """Locate complete top-level records without using private mcap internals."""

    if not payload.startswith(MAGIC) or not payload.endswith(MAGIC):
        raise ValueError("fixture is not a complete MCAP")
    records: list[RecordRange] = []
    cursor = len(MAGIC)
    end = len(payload) - len(MAGIC)
    while cursor < end:
        if end - cursor < 9:
            raise ValueError("truncated record header")
        content_length = int.from_bytes(payload[cursor + 1 : cursor + 9], "little")
        content_start = cursor + 9
        content_end = content_start + content_length
        if content_end > end:
            raise ValueError("truncated record content")
        records.append(
            RecordRange(
                opcode=payload[cursor],
                start=cursor,
                content_start=content_start,
                content_end=content_end,
            )
        )
        cursor = content_end
    if cursor != end:
        raise ValueError("record boundary does not end at trailing magic")
    return tuple(records)


def corrupt_record_content(payload: bytes, opcode: Opcode) -> bytes:
    """Flip one byte inside the first matching non-empty record."""

    result = bytearray(payload)
    for record in record_ranges(payload):
        if record.opcode == int(opcode) and record.content_start < record.content_end:
            result[record.content_start] ^= 1
            return bytes(result)
    raise ValueError(f"fixture contains no non-empty {opcode.name} record")


def truncate_before_record(payload: bytes, opcode: Opcode) -> bytes:
    """Return an unsealed prefix ending immediately before ``opcode``."""

    for record in record_ranges(payload):
        if record.opcode == int(opcode):
            return payload[: record.start]
    raise ValueError(f"fixture contains no {opcode.name} record")
