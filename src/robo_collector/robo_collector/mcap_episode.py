"""Strict, read-only validation for sealed MCAP landing files.

The validator deliberately uses only documented :mod:`mcap` 1.4.0 reader and
record classes.  It never opens a source file for writing and it does not try to
repair a partial landing file; recovery must publish a new immutable artifact.
"""

from __future__ import annotations

import errno
import hashlib
import os
import stat
import struct
import tempfile
import zlib
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator

MAX_MCAP_RECORD_BYTES = 64 * 1024 * 1024

from .mcap_contract import decode_inprogress_manifest

_MAGIC = b"\x89MCAP0\r\n"
_MAGIC_SIZE = len(_MAGIC)
_FOOTER_RECORD_SIZE = 1 + 8 + 20
_HEADER_SIZE = 1 + 8
_FOOTER_OPCODE = 0x02
_DATA_END_OPCODE = 0x0F
_SUMMARY_OFFSET_OPCODE = 0x0E
_MCAP_PROFILE = "robo_collector.mcap.v1"
_MCAP_LIBRARY = "mcap-python/1.4.0"


class McapValidationError(ValueError):
    """Raised when an MCAP file is not a complete, contract-valid sealed source."""


SealedEpisodeError = McapValidationError


class EpisodePublicationError(McapValidationError):
    """Raised when an immutable episode publication boundary fails."""


@dataclass(frozen=True)
class SourceFence:
    """Immutable source-boundary evidence decoded from an episode event."""

    source_id: str
    session_id: str
    start_sequence_exclusive: int
    end_sequence_inclusive: int | None
    accepted_count: int
    written_count: int
    durable_count: int
    accepted_high_watermark: int | None
    written_high_watermark: int | None
    durable_high_watermark: int | None


@dataclass(frozen=True)
class SchemaInventory:
    id: int
    name: str
    encoding: str
    data_sha256: str


@dataclass(frozen=True)
class ChannelInventory:
    id: int
    topic: str
    schema_id: int
    message_encoding: str
    metadata: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class McapInventory:
    """The immutable message inventory of one completely sealed MCAP file."""

    source_sha256: str
    source_size_bytes: int
    schemas: tuple[SchemaInventory, ...]
    channels: tuple[ChannelInventory, ...]
    topic_counts: tuple[tuple[str, int], ...]
    total_message_count: int
    start_log_time_ns: int | None
    end_log_time_ns: int | None
    source_fences: tuple[SourceFence, ...]
    event_identities: tuple[tuple[int, int, int | None], ...] = ()

    sha256 = property(lambda self: self.source_sha256)
    size_bytes = property(lambda self: self.source_size_bytes)
    schema_count = property(lambda self: len(self.schemas))
    channel_count = property(lambda self: len(self.channels))
    message_count = property(lambda self: self.total_message_count)


def validate_sealed_mcap(
    path: str | Path,
    *,
    max_record_content_bytes: int = MAX_MCAP_RECORD_BYTES,
    record_size_limit: int | None = None,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
    expected_total_message_count: int | None = None,
    expected_topic_counts: Mapping[str, int] | None = None,
    expected_start_log_time_ns: int | None = None,
    expected_end_log_time_ns: int | None = None,
    expected_source_fences: Iterable[SourceFence | Mapping[str, Any]] | None = None,
    require_data_crc: bool = True,
    require_summary: bool = True,
    require_statistics: bool = True,
) -> McapInventory:
    """Validate a finalized MCAP without mutating it and return its inventory.

    ``StreamReader`` is the public 1.4.0 API used for magic, trailing-footer,
    chunk CRC, and DataEnd CRC validation.  The fixed MCAP footer layout is
    additionally inspected from bytes to validate footer summary offsets and
    the optional summary CRC, which the public Python reader does not validate.
    """

    source = _safe_path(path, label="sealed MCAP", error=McapValidationError)
    if source.name.endswith(".partial"):
        raise McapValidationError("source partial is not a sealed MCAP")
    if record_size_limit is not None:
        max_record_content_bytes = record_size_limit
    if max_record_content_bytes <= 0:
        raise ValueError("max_record_content_bytes must be positive")
    try:
        from mcap.records import (
            Channel,
            DataEnd,
            Footer,
            Header,
            Message,
            Schema,
            Statistics,
        )
        from mcap.stream_reader import StreamReader
    except ImportError as exc:  # pragma: no cover - deployment dependency guard
        raise McapValidationError(
            "strict MCAP validation requires the public mcap==1.4.0 package"
        ) from exc

    header_record: Any = None
    data_schemas: dict[int, Any] = {}
    data_channels: dict[int, Any] = {}
    summary_schemas: dict[int, Any] = {}
    summary_channels: dict[int, Any] = {}
    topic_counts: Counter[str] = Counter()
    channel_counts: Counter[int] = Counter()
    message_count = 0
    start_time: int | None = None
    end_time: int | None = None
    statistics: Any = None
    footer: Any = None
    data_end_seen = False
    source_fence_snapshots: list[tuple[SourceFence, ...]] = []
    event_identities: list[tuple[int, int, int | None]] = []

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise McapValidationError(f"cannot open sealed MCAP {source}: {exc}") from exc
    try:
        stream = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    try:
        source_stat = os.fstat(stream.fileno())
        if not stat.S_ISREG(source_stat.st_mode):
            raise McapValidationError(f"MCAP source is not a regular file: {source}")
        source_size = source_stat.st_size
        digest = _sha256_stream(stream)
        _validate_source_expectations(
            digest=digest,
            size=source_size,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
        )
        layout = _scan_layout(stream, source_size, max_record_content_bytes)
        _validate_footer_summary(stream, source_size, layout, require_summary)
        if layout.has_chunk:
            raise McapValidationError("frozen landing v1 forbids chunked records")
        stream.seek(0)
        reader = StreamReader(
            stream,
            skip_magic=False,
            emit_chunks=False,
            validate_crcs=True,
            record_size_limit=max_record_content_bytes,
        )
        for record in reader.records:
            if isinstance(record, Header):
                if (
                    header_record is not None
                    or data_end_seen
                    or message_count
                    or data_schemas
                    or data_channels
                ):
                    raise McapValidationError("Header record is not first")
                header_record = record
            elif isinstance(record, DataEnd):
                if data_end_seen:
                    raise McapValidationError("multiple DataEnd records")
                if require_data_crc and record.data_section_crc == 0:
                    raise McapValidationError("DataEnd record omits required data CRC")
                data_end_seen = True
            elif isinstance(record, Footer):
                footer = record
            elif isinstance(record, Schema):
                _insert_record(
                    summary_schemas if data_end_seen else data_schemas,
                    record.id,
                    record,
                    "schema",
                )
            elif isinstance(record, Channel):
                _insert_record(
                    summary_channels if data_end_seen else data_channels,
                    record.id,
                    record,
                    "channel",
                )
            elif isinstance(record, Message):
                if data_end_seen:
                    raise McapValidationError("message record occurs after DataEnd")
                channel = data_channels.get(record.channel_id)
                if channel is None:
                    raise McapValidationError(
                        f"message references unknown channel {record.channel_id}"
                    )
                _validate_schema_channel(data_schemas, channel)
                topic = channel.topic
                topic_counts[topic] += 1
                channel_counts[record.channel_id] += 1
                message_count += 1
                start_time = (
                    record.log_time
                    if start_time is None
                    else min(start_time, record.log_time)
                )
                end_time = (
                    record.log_time
                    if end_time is None
                    else max(end_time, record.log_time)
                )
                if topic == "/episode/event":
                    from .mcap.v1 import episode_pb2
                    event = episode_pb2.EpisodeEventV1.FromString(record.data)
                    event_identities.append(
                        (event.event_type, event.lifecycle_attempt,
                         event.collector_record_id if event.HasField("collector_record_id") else None)
                    )
                    source_fence_snapshots.extend(
                        _decode_event_source_fences(record.data)
                    )
            elif isinstance(record, Statistics):
                if not data_end_seen:
                    raise McapValidationError("Statistics record occurs before DataEnd")
                if statistics is not None:
                    raise McapValidationError("multiple Statistics records")
                statistics = record
        final_stat = os.fstat(stream.fileno())
    except McapValidationError:
        raise
    except Exception as exc:
        raise McapValidationError(f"invalid sealed MCAP {source}: {exc}") from exc
    finally:
        stream.close()

    if (
        source_stat.st_dev,
        source_stat.st_ino,
        source_stat.st_size,
        source_stat.st_mtime_ns,
    ) != (
        final_stat.st_dev,
        final_stat.st_ino,
        final_stat.st_size,
        final_stat.st_mtime_ns,
    ):
        raise McapValidationError("sealed MCAP changed during validation")

    if header_record is None:
        raise McapValidationError("MCAP header was not decoded")
    if header_record.profile != _MCAP_PROFILE:
        raise McapValidationError("MCAP header profile does not match frozen v1")
    if header_record.library != _MCAP_LIBRARY:
        raise McapValidationError(
            "MCAP header library does not match frozen mcap 1.4.0"
        )
    if footer is None:
        raise McapValidationError("MCAP footer was not decoded")
    if not data_end_seen:
        raise McapValidationError("MCAP has no DataEnd record")
    if require_summary:
        if not summary_schemas and data_schemas:
            raise McapValidationError("summary omits schema inventory")
        if not summary_channels and data_channels:
            raise McapValidationError("summary omits channel inventory")
        if summary_schemas != data_schemas:
            raise McapValidationError("summary schemas disagree with data schemas")
        if summary_channels != data_channels:
            raise McapValidationError("summary channels disagree with data channels")
    if require_statistics and statistics is None:
        raise McapValidationError("MCAP summary omits required Statistics record")
    if statistics is not None:
        _validate_statistics(
            statistics,
            schemas=data_schemas,
            channels=data_channels,
            channel_counts=channel_counts,
            message_count=message_count,
            start_time=start_time,
            end_time=end_time,
        )

    inventory = McapInventory(
        source_sha256=digest,
        source_size_bytes=source_size,
        schemas=tuple(
            SchemaInventory(
                item.id, item.name, item.encoding, hashlib.sha256(item.data).hexdigest()
            )
            for _, item in sorted(data_schemas.items())
        ),
        channels=tuple(
            ChannelInventory(
                item.id,
                item.topic,
                item.schema_id,
                item.message_encoding,
                tuple(sorted(item.metadata.items())),
            )
            for _, item in sorted(data_channels.items())
        ),
        topic_counts=tuple(sorted(topic_counts.items())),
        total_message_count=message_count,
        start_log_time_ns=start_time,
        end_log_time_ns=end_time,
        source_fences=source_fence_snapshots[-1] if source_fence_snapshots else (),
        event_identities=tuple(event_identities),
    )
    _validate_inventory_expectations(
        inventory,
        expected_total_message_count=expected_total_message_count,
        expected_topic_counts=expected_topic_counts,
        expected_start_log_time_ns=expected_start_log_time_ns,
        expected_end_log_time_ns=expected_end_log_time_ns,
        expected_source_fences=expected_source_fences,
    )
    return inventory


def publish_raw_closed_manifest(
    episode_dir: str | Path,
    sealed_path: str | Path,
    checkpoint: Mapping[str, Any],
    *,
    source_complete: bool,
    max_record_content_bytes: int = MAX_MCAP_RECORD_BYTES,
) -> Path:
    """Atomically publish a new immutable ``RAW_CLOSED`` manifest.

    The landing writer owns partial files and the checkpoint journal.  This
    boundary only reads its already-sealed file, validates it, then creates
    ``manifest.json`` once.  An existing manifest is immutable and is never
    replaced.
    """

    from .mcap_contract import canonical_json_bytes, validate_checkpoint_payload

    root = _safe_path(episode_dir, label="episode root", error=McapValidationError)
    sealed = _safe_path(
        sealed_path,
        root=root,
        label="sealed MCAP",
        error=McapValidationError,
    )
    manifest_path = _safe_path(
        root / "manifest.json",
        root=root,
        label="RAW_CLOSED manifest",
        error=McapValidationError,
    )
    if not sealed.is_file():
        raise McapValidationError(f"sealed MCAP source does not exist: {sealed}")
    relative_path = sealed.relative_to(root).as_posix()
    validated_checkpoint = validate_checkpoint_payload(checkpoint)
    inprogress_path = root / "manifest.inprogress.json"
    inprogress = (
        decode_inprogress_manifest(inprogress_path.read_bytes())
        if inprogress_path.exists()
        else None
    )
    if bool(source_complete) != _checkpoint_sources_complete(validated_checkpoint):
        raise McapValidationError("source_complete disagrees with final source fences")
    inventory = validate_sealed_mcap(
        sealed,
        max_record_content_bytes=max_record_content_bytes,
        expected_source_fences=validated_checkpoint["source_fences"],
    )
    if not all(
        _event_source_fence_complete(fence) for fence in inventory.source_fences
    ):
        raise McapValidationError("sealed MCAP source fences are internally incomplete")
    if source_complete and inprogress is not None:
        if not inventory.event_identities:
            raise McapValidationError("source-complete publication requires a STOP event")
        event_type, lifecycle_attempt, collector_record_id = inventory.event_identities[-1]
        if event_type != 2:
            raise McapValidationError("final lifecycle event is not STOP")
        if lifecycle_attempt != int(inprogress["attempt"]):
            raise McapValidationError("STOP lifecycle attempt does not match landing attempt")
        if collector_record_id != int(validated_checkpoint["durable_frontier"]):
            raise McapValidationError("STOP collector_record_id does not match durable frontier")
    payload = canonical_json_bytes(
        {
            "checkpoint": validated_checkpoint,
            "format": "robo_collector.mcap_landing",
            "format_version": 1,
            "inventory": {
                "channel_count": str(inventory.channel_count),
                "end_log_time_ns": (
                    None
                    if inventory.end_log_time_ns is None
                    else str(inventory.end_log_time_ns)
                ),
                "schema_count": str(inventory.schema_count),
                "source_sha256": inventory.source_sha256,
                "source_size_bytes": str(inventory.source_size_bytes),
                "start_log_time_ns": (
                    None
                    if inventory.start_log_time_ns is None
                    else str(inventory.start_log_time_ns)
                ),
                "topic_counts": [
                    {"count": str(count), "topic": topic}
                    for topic, count in inventory.topic_counts
                ],
                "total_message_count": str(inventory.total_message_count),
            },
            "landing_path": relative_path,
            "source_complete": bool(source_complete),
            "status": "RAW_CLOSED",
        }
    )
    return _atomic_create(manifest_path, payload)


class McapEpisodeReader:
    """Read-only adapter exposing sealed MCAP records in Raw v1 shape."""

    def __init__(self, path: str | Path):
        import json
        self.path = Path(path).resolve()
        try:
            self.manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        except Exception as exc:
            raise McapValidationError(f"cannot read MCAP episode manifest: {self.path}") from exc
        if not isinstance(self.manifest, dict):
            raise McapValidationError("MCAP episode manifest must be an object")
        landing = self.manifest.get("landing_path")
        self._sealed = (self.path / landing) if isinstance(landing, str) and landing else self.path / "landing" / "episode.mcap"
        self._sealed = self._sealed.resolve()
        inventory = self.manifest.get("inventory", {})
        topics = inventory.get("topic_counts", ()) if isinstance(inventory, dict) else ()
        self.camera_streams = tuple(sorted({str(item.get("topic", "")).split("/")[2] for item in topics if isinstance(item, dict) and str(item.get("topic", "")).startswith("/camera/") and str(item.get("topic", "")).endswith("/sample")}))
        if self.path not in self._sealed.parents:
            raise McapValidationError("MCAP landing path escapes episode root")

    def validate(self) -> None:
        if not self._sealed.is_file() or self._sealed.name.endswith(".partial"):
            raise McapValidationError("MCAP episode is not sealed")
        validate_sealed_mcap(self._sealed)

    def records(self, category: str, name: str) -> Iterator[dict[str, Any]]:
        topic = (f"/camera/{name}/sample" if category == "camera" else
                 "/robot/state/raw" if category == "robot" and name == "state" else
                 "/robot/action/raw" if category == "robot" and name == "action" else None)
        if topic is not None:
            yield from self._iter_topic(topic, category, name)

    def _iter_topic(self, topic: str, category: str, name: str) -> Iterator[dict[str, Any]]:
        try:
            from mcap.records import Channel, Message
            from mcap.stream_reader import StreamReader
            from .mcap_contract import MESSAGE_TYPES
        except ImportError as exc:
            raise McapValidationError("MCAP reader requires mcap==1.4.0") from exc
        message_type = MESSAGE_TYPES["CameraSampleV1" if category == "camera" else "RobotStateV1" if name == "state" else "RobotActionV1"]
        with self._sealed.open("rb") as stream:
            channels: dict[int, str] = {}
            for record in StreamReader(stream, validate_crcs=True).records:
                if isinstance(record, Channel):
                    channels[record.id] = record.topic
                elif isinstance(record, Message) and channels.get(record.channel_id) == topic:
                    yield _mcap_message_to_raw(message_type.FromString(record.data), category, name, record.log_time)


def _mcap_message_to_raw(value: Any, category: str, name: str, log_time_ns: int) -> dict[str, Any]:
    timestamps = getattr(value, "timestamps", None)
    normalized = int(getattr(timestamps, "normalized_time_ns", log_time_ns)) if timestamps else int(log_time_ns)
    result: dict[str, Any] = {"stream_name": name, "timestamp": normalized / 1_000_000_000.0, "normalized_time_ns": normalized, "source_sequence": int(getattr(value, "source_sequence", 0))}
    if category == "camera":
        payload = bytes(getattr(value, "image_bytes", b""))
        result.update({"payload": base64.b64encode(payload).decode("ascii"), "payload_encoding": "image/jpeg", "payload_size_bytes": len(payload), "width": int(getattr(value, "width", 0)), "height": int(getattr(value, "height", 0))})
        return result
    state: dict[str, Any] = {}
    for field in getattr(value, "fields", ()):
        if field.field_name:
            state[str(field.field_name)] = [float(item) for item in field.values]
    result["state"] = state
    result["sequence"] = int(getattr(value, "source_sequence", 0))
    if category == "robot" and name == "action":
        result["action_source"] = str(getattr(value, "action_source", ""))
    return result


class SealedMcapReader:
    """Immutable object facade over :func:`validate_sealed_mcap`."""

    def __init__(
        self, path: str | Path, *, record_size_limit: int = MAX_MCAP_RECORD_BYTES
    ):
        self.path = Path(path)
        self.record_size_limit = record_size_limit

    def validate(
        self,
        *,
        expected_source_fences: Iterable[SourceFence | Mapping[str, Any]] | None = None,
    ) -> McapInventory:
        return validate_sealed_mcap(
            self.path,
            record_size_limit=self.record_size_limit,
            expected_source_fences=expected_source_fences,
        )


def install_ready_bundle(
    staging_dir: str | Path,
    canonical_dir: str | Path,
    manifest: Mapping[str, Any],
    *,
    prevalidated_authority: bool = False,
) -> Path:
    """Atomically install a validated READY version without changing its pointer."""

    from . import mcap_contract

    if not prevalidated_authority:
        raise EpisodePublicationError(
            "READY install requires caller-held lock, ledger transaction, and fencing validation"
        )
    bundle_hash, expected_manifest_hash = mcap_contract.assert_ready_manifest(manifest)
    source = _safe_path(
        staging_dir, label="READY staging source", error=EpisodePublicationError
    )
    root = _safe_path(
        canonical_dir, label="canonical root", error=EpisodePublicationError
    )
    versions = _safe_path(
        root / "versions",
        root=root,
        label="READY versions directory",
        error=EpisodePublicationError,
    )
    target = _safe_path(
        versions / bundle_hash,
        root=root,
        label="READY version target",
        error=EpisodePublicationError,
    )
    if not source.is_dir() or source.is_symlink():
        raise EpisodePublicationError(
            "READY staging source must be a regular directory"
        )
    _validate_ready_directory(source, manifest, expected_manifest_hash)
    versions.mkdir(parents=True, exist_ok=True)
    if source.stat().st_dev != versions.stat().st_dev:
        raise EpisodePublicationError(
            "READY staging and versions must share a filesystem"
        )
    if target.exists():
        _validate_ready_directory(target, manifest, expected_manifest_hash)
        return target
    try:
        os.rename(source, target)
    except OSError as exc:
        if exc.errno in {errno.EEXIST, errno.ENOTEMPTY} and target.exists():
            _validate_ready_directory(target, manifest, expected_manifest_hash)
            return target
        raise EpisodePublicationError(
            f"cannot atomically install READY bundle: {exc}"
        ) from exc
    _fsync_directory(versions)
    _fsync_directory(root)
    return target


def install_ready_pointer(
    canonical_dir: str | Path,
    manifest: Mapping[str, Any],
    pointer: Mapping[str, Any],
    *,
    prevalidated_authority: bool = False,
) -> Path:
    """Install a READY selector only through the frozen authority gate.

    This deliberately delegates the authority/token validation to the existing
    frozen contract helper; it does not invent a second publication mechanism.
    """

    if not prevalidated_authority:
        raise EpisodePublicationError(
            "READY pointer install requires caller-held authority and fencing validation"
        )
    from .mcap_contract import PublicationError, publish_ready_pointer

    try:
        root = _safe_path(
            canonical_dir, label="canonical root", error=EpisodePublicationError
        )
        _safe_path(
            root / "current.json",
            root=root,
            label="READY pointer",
            error=EpisodePublicationError,
        )
        _safe_path(
            root / f".current.json.{os.getpid()}.tmp",
            root=root,
            label="READY pointer temporary",
            error=EpisodePublicationError,
        )
        return publish_ready_pointer(
            root,
            manifest,
            pointer,
            prevalidated_authority=True,
        )
    except PublicationError as exc:
        raise EpisodePublicationError(str(exc)) from exc


def _validate_ready_directory(
    directory: Path, manifest: Mapping[str, Any], expected_manifest_hash: str
) -> None:
    from . import mcap_contract

    directory = _safe_path(
        directory, label="READY directory", error=EpisodePublicationError
    )
    path = _safe_path(
        directory / "manifest.json",
        root=directory,
        label="READY manifest",
        error=EpisodePublicationError,
    )
    try:
        payload = path.read_bytes()
        decoded = mcap_contract.parse_canonical_json(payload)
    except (OSError, ValueError) as exc:
        raise EpisodePublicationError(f"invalid READY manifest file: {exc}") from exc
    if (
        decoded != dict(manifest)
        or hashlib.sha256(payload).hexdigest() != expected_manifest_hash
    ):
        raise EpisodePublicationError(
            "READY manifest bytes disagree with supplied manifest"
        )
    for member in manifest["identity"]["members"]:
        member_path = _safe_path(
            directory / member["path"],
            root=directory,
            label=f"READY member {member['path']}",
            error=EpisodePublicationError,
        )
        if not member_path.is_file() or member_path.is_symlink():
            raise EpisodePublicationError(
                f"READY member is missing or unsafe: {member['path']}"
            )
        if (
            str(member_path.stat().st_size) != member["size_bytes"]
            or _sha256_file(member_path) != member["sha256"]
        ):
            raise EpisodePublicationError(
                f"READY member hash/size mismatch: {member['path']}"
            )


def _checkpoint_sources_complete(checkpoint: Mapping[str, Any]) -> bool:
    for fence in checkpoint["source_fences"]:
        if (
            fence["end_sequence_inclusive"] is None
            or fence["accepted_count"] != fence["written_count"]
            or fence["written_count"] != fence["durable_count"]
            or fence["end_sequence_inclusive"] != fence["written_high_watermark"]
            or fence["written_high_watermark"] != fence["durable_high_watermark"]
        ):
            return False
    return (
        checkpoint["accepted_snapshot_count"] == checkpoint["written_count"]
        and checkpoint["written_count"] == checkpoint["durable_count"]
        and checkpoint["accepted_snapshot_frontier"] == checkpoint["written_frontier"]
        and checkpoint["written_frontier"] == checkpoint["durable_frontier"]
    )


def _event_source_fence_complete(fence: SourceFence) -> bool:
    end = fence.end_sequence_inclusive
    if end is None:
        return False
    return (
        fence.accepted_count == fence.written_count == fence.durable_count
        and end == fence.written_high_watermark == fence.durable_high_watermark
        and (
            fence.accepted_high_watermark is None
            or fence.accepted_high_watermark == end
        )
    )


def _atomic_create(path: Path, payload: bytes) -> Path:
    path = _safe_path(
        path,
        root=path.parent,
        label="immutable publication target",
        error=EpisodePublicationError,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise EpisodePublicationError(
                f"immutable manifest already exists: {path}"
            ) from exc
        _fsync_directory(path.parent)
        return path
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _safe_path(
    path: str | os.PathLike[str],
    *,
    label: str,
    error: type[Exception],
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Return an absolute lexical path with no symlink component or root escape."""

    candidate = Path(os.path.abspath(os.fspath(path)))
    if root is not None:
        boundary = Path(os.path.abspath(os.fspath(root)))
        try:
            candidate.relative_to(boundary)
        except ValueError as exc:
            raise error(f"{label} escapes its intended root") from exc
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current /= component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise error(f"cannot inspect {label}: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise error(f"{label} contains a symlink component: {current}")
    return candidate


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class _Layout:
    footer_start: int
    data_end_end: int | None
    first_summary_offset: int | None
    has_chunk: bool


def _scan_layout(stream: BinaryIO, size: int, limit: int) -> _Layout:
    """Bound every top-level record and locate the DataEnd/summary boundary."""

    if size < 2 * _MAGIC_SIZE + _FOOTER_RECORD_SIZE:
        raise McapValidationError("MCAP is too small to contain magic and footer")
    stream.seek(0)
    if stream.read(_MAGIC_SIZE) != _MAGIC:
        raise McapValidationError("MCAP has invalid leading magic")
    stream.seek(-_MAGIC_SIZE, 2)
    if stream.read(_MAGIC_SIZE) != _MAGIC:
        raise McapValidationError("MCAP has invalid trailing magic")
    footer_start = size - _MAGIC_SIZE - _FOOTER_RECORD_SIZE
    stream.seek(footer_start)
    footer_header = stream.read(_HEADER_SIZE)
    if len(footer_header) != _HEADER_SIZE:
        raise McapValidationError("MCAP footer header is truncated")
    footer_opcode, footer_length = struct.unpack("<BQ", footer_header)
    if footer_opcode != _FOOTER_OPCODE or footer_length != 20:
        raise McapValidationError("MCAP trailing record is not a fixed-size Footer")
    offset = _MAGIC_SIZE
    data_end_end: int | None = None
    first_summary_offset: int | None = None
    has_chunk = False
    while offset < footer_start:
        stream.seek(offset)
        header = stream.read(_HEADER_SIZE)
        if len(header) != _HEADER_SIZE:
            raise McapValidationError("truncated MCAP record header")
        opcode, length = struct.unpack("<BQ", header)
        if length > limit:
            raise McapValidationError(
                f"MCAP record content length {length} exceeds configured limit {limit}"
            )
        end = offset + _HEADER_SIZE + length
        if end > footer_start:
            raise McapValidationError("MCAP record extends into footer")
        if offset == _MAGIC_SIZE and opcode != 0x01:
            raise McapValidationError("first MCAP record is not Header")
        if opcode == 0x06:
            has_chunk = True
        if opcode == _DATA_END_OPCODE:
            if data_end_end is not None:
                raise McapValidationError("multiple top-level DataEnd records")
            data_end_end = end
        elif opcode == _SUMMARY_OFFSET_OPCODE:
            if data_end_end is None:
                raise McapValidationError("SummaryOffset occurs before DataEnd")
            if first_summary_offset is None:
                first_summary_offset = offset
        offset = end
    if offset != footer_start:
        raise McapValidationError("MCAP record layout does not terminate at Footer")
    return _Layout(footer_start, data_end_end, first_summary_offset, has_chunk)


def _validate_footer_summary(
    stream: BinaryIO, size: int, layout: _Layout, require_summary: bool
) -> None:
    if layout.data_end_end is None:
        raise McapValidationError("MCAP has no top-level DataEnd before footer")
    stream.seek(layout.footer_start + _HEADER_SIZE)
    payload = stream.read(20)
    summary_start, summary_offset_start, summary_crc = struct.unpack("<QQI", payload)
    if summary_start == 0:
        if require_summary:
            raise McapValidationError("MCAP footer omits required summary section")
        if summary_offset_start != 0 or summary_crc != 0:
            raise McapValidationError("footer has summary fields without summary")
        return
    if summary_start != layout.data_end_end:
        raise McapValidationError("footer summary_start does not follow DataEnd")
    if not (summary_start < layout.footer_start):
        raise McapValidationError("footer summary_start is outside summary range")
    if layout.first_summary_offset is None:
        if summary_offset_start != 0:
            raise McapValidationError("footer declares absent SummaryOffset section")
    elif summary_offset_start != layout.first_summary_offset:
        raise McapValidationError("footer summary_offset_start is inconsistent")
    if summary_crc == 0:
        raise McapValidationError("MCAP footer omits required summary CRC")
    crc_end = layout.footer_start + _HEADER_SIZE + 16
    actual = _crc32_range(stream, summary_start, crc_end)
    if actual != summary_crc:
        raise McapValidationError(
            f"summary CRC mismatch: expected {summary_crc:#x}, got {actual:#x}"
        )


def _crc32_range(stream: BinaryIO, start: int, end: int) -> int:
    checksum = 0
    remaining = end - start
    stream.seek(start)
    while remaining:
        block = stream.read(min(1024 * 1024, remaining))
        if not block:
            raise McapValidationError("truncated range while calculating summary CRC")
        checksum = zlib.crc32(block, checksum)
        remaining -= len(block)
    return checksum & 0xFFFFFFFF


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    stream.seek(0)
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _insert_record(
    target: dict[int, Any], identifier: int, value: Any, kind: str
) -> None:
    existing = target.setdefault(identifier, value)
    if existing != value:
        raise McapValidationError(f"conflicting duplicate {kind} id {identifier}")


def _validate_schema_channel(schemas: Mapping[int, Any], channel: Any) -> None:
    from . import mcap_contract

    schema = schemas.get(channel.schema_id)
    if schema is None:
        raise McapValidationError(f"channel {channel.id} references unknown schema")
    expected_name = mcap_contract.message_type_for_topic(channel.topic)
    if schema.name not in {expected_name, f"robo_collector.mcap.v1.{expected_name}"}:
        raise McapValidationError("channel topic and schema name disagree")
    if schema.encoding != "protobuf" or channel.message_encoding != "protobuf":
        raise McapValidationError("frozen MCAP v1 requires protobuf encoding")
    if schema.data != mcap_contract.descriptor_set_bytes():
        raise McapValidationError("schema descriptor does not match frozen MCAP v1")


def _validate_statistics(
    statistics: Any,
    *,
    schemas: Mapping[int, Any],
    channels: Mapping[int, Any],
    channel_counts: Counter[int],
    message_count: int,
    start_time: int | None,
    end_time: int | None,
) -> None:
    if statistics.schema_count != len(schemas) or statistics.channel_count != len(
        channels
    ):
        raise McapValidationError("Statistics schema/channel counts disagree with data")
    if statistics.message_count != message_count:
        raise McapValidationError("Statistics message_count disagrees with messages")
    expected_start = 0 if start_time is None else start_time
    expected_end = 0 if end_time is None else end_time
    if (
        statistics.message_start_time != expected_start
        or statistics.message_end_time != expected_end
    ):
        raise McapValidationError(
            "Statistics message time bounds disagree with messages"
        )
    if statistics.channel_message_counts and statistics.channel_message_counts != dict(
        channel_counts
    ):
        raise McapValidationError(
            "Statistics channel message counts disagree with messages"
        )


def _decode_event_source_fences(payload: bytes) -> list[tuple[SourceFence, ...]]:
    try:
        from .mcap.v1 import episode_pb2

        event = episode_pb2.EpisodeEventV1.FromString(payload)  # type: ignore[attr-defined]
    except Exception as exc:
        raise McapValidationError(f"invalid /episode/event protobuf: {exc}") from exc
    if not event.source_fences:
        return []
    return [
        tuple(
            SourceFence(
                source_id=fence.source_id,
                session_id=fence.session_id,
                start_sequence_exclusive=fence.start_sequence_exclusive,
                end_sequence_inclusive=(
                    fence.end_sequence_inclusive
                    if fence.HasField("end_sequence_inclusive")
                    else None
                ),
                accepted_count=fence.accepted_count,
                written_count=fence.written_count,
                durable_count=fence.durable_count,
                accepted_high_watermark=(
                    fence.accepted_high_watermark
                    if fence.HasField("accepted_high_watermark")
                    else None
                ),
                written_high_watermark=(
                    fence.written_high_watermark
                    if fence.HasField("written_high_watermark")
                    else None
                ),
                durable_high_watermark=(
                    fence.durable_high_watermark
                    if fence.HasField("durable_high_watermark")
                    else None
                ),
            )
            for fence in event.source_fences
        )
    ]


def _validate_source_expectations(
    *,
    digest: str,
    size: int,
    expected_sha256: str | None,
    expected_size_bytes: int | None,
) -> None:
    if expected_sha256 is not None and digest != expected_sha256:
        raise McapValidationError("MCAP source SHA-256 does not match expected fence")
    if expected_size_bytes is not None and size != expected_size_bytes:
        raise McapValidationError("MCAP source size does not match expected fence")


def _validate_inventory_expectations(
    inventory: McapInventory,
    *,
    expected_total_message_count: int | None,
    expected_topic_counts: Mapping[str, int] | None,
    expected_start_log_time_ns: int | None,
    expected_end_log_time_ns: int | None,
    expected_source_fences: Iterable[SourceFence | Mapping[str, Any]] | None,
) -> None:
    if (
        expected_total_message_count is not None
        and inventory.total_message_count != expected_total_message_count
    ):
        raise McapValidationError(
            "MCAP message count does not match expected inventory"
        )
    if expected_topic_counts is not None and dict(inventory.topic_counts) != dict(
        expected_topic_counts
    ):
        raise McapValidationError("MCAP topic counts do not match expected inventory")
    if (
        expected_start_log_time_ns is not None
        and inventory.start_log_time_ns != expected_start_log_time_ns
    ):
        raise McapValidationError(
            "MCAP start log time does not match expected inventory"
        )
    if (
        expected_end_log_time_ns is not None
        and inventory.end_log_time_ns != expected_end_log_time_ns
    ):
        raise McapValidationError("MCAP end log time does not match expected inventory")
    if expected_source_fences is not None:
        expected = tuple(expected_source_fences)
        normalized = tuple(_coerce_source_fence(item) for item in expected)
        matches = len(inventory.source_fences) == len(normalized) and all(
            actual == wanted
            if isinstance(raw, SourceFence) or "accepted_high_watermark" in raw
            else _checkpoint_source_fence(actual) == _checkpoint_source_fence(wanted)
            for actual, wanted, raw in zip(
                inventory.source_fences, normalized, expected
            )
        )
        if not matches:
            raise McapValidationError(
                "MCAP source fences do not match expected inventory"
            )


def _checkpoint_source_fence(value: SourceFence) -> tuple[Any, ...]:
    """Project an event fence onto fields persisted by checkpoint v1."""

    return (
        value.source_id,
        value.session_id,
        value.start_sequence_exclusive,
        value.end_sequence_inclusive,
        value.accepted_count,
        value.written_count,
        value.durable_count,
        value.written_high_watermark,
        value.durable_high_watermark,
    )


def _coerce_source_fence(value: SourceFence | Mapping[str, Any]) -> SourceFence:
    if isinstance(value, SourceFence):
        return value
    row = dict(value)

    def optional(name: str) -> int | None:
        return None if row.get(name) is None else int(row[name])

    return SourceFence(
        source_id=str(row["source_id"]),
        session_id=str(row["session_id"]),
        start_sequence_exclusive=int(row["start_sequence_exclusive"]),
        end_sequence_inclusive=optional("end_sequence_inclusive"),
        accepted_count=int(row["accepted_count"]),
        written_count=int(row["written_count"]),
        durable_count=int(row["durable_count"]),
        accepted_high_watermark=optional("accepted_high_watermark"),
        written_high_watermark=optional("written_high_watermark"),
        durable_high_watermark=optional("durable_high_watermark"),
    )


__all__ = [
    "MAX_MCAP_RECORD_BYTES",
    "ChannelInventory",
    "EpisodePublicationError",
    "McapInventory",
    "McapValidationError",
    "SchemaInventory",
    "SealedEpisodeError",
    "SealedMcapReader",
    "SourceFence",
    "install_ready_bundle",
    "install_ready_pointer",
    "publish_raw_closed_manifest",
    "validate_sealed_mcap",
]
