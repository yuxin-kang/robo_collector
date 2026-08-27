from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import cv2
import msgpack
import numpy as np
import zmq

CAMERA_PACKET_SCHEMA = "robo_collector_camera.v3"
CAMERA_PACKET_V4_SCHEMA = "robo_collector_camera.v4"
DEFAULT_MAX_PACKET_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_IMAGE_BYTES = 32 * 1024 * 1024
_STREAM_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class CameraPacketError(ValueError):
    """Raised when a camera packet violates the transport contract."""


@dataclass
class NormalizedCameraFrame:
    stream: str
    sequence: int | None
    payload: bytes
    payload_encoding: str
    timestamp_quality: str
    device_timestamp: float | None = None
    device_unit: str | None = None
    timestamp_domain: str | None = None
    server_wall_timestamp: float | None = None
    server_monotonic_timestamp: float | None = None
    receive_wall_timestamp: float | None = None
    receive_monotonic_timestamp: float | None = None
    clock_domain: str | None = None
    decoded_image: Any = None

    @property
    def timestamp_sec(self) -> float | None:
        return _legacy_timestamp_seconds(self)

    def decode(self) -> Any:
        if self.decoded_image is not None:
            return self.decoded_image
        if not self.payload:
            return None
        arr = np.frombuffer(self.payload, dtype=np.uint8)
        if self.payload_encoding == "image/png" or self.stream.endswith("_depth"):
            return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@dataclass
class NormalizedPacket:
    schema: str
    session_id: str
    frames: dict[str, NormalizedCameraFrame]
    metadata: dict[str, Any] = field(default_factory=dict)
    host: str | None = None
    port: int | None = None
    producer_gaps: dict[str, int] | None = None
    publisher_gaps: dict[str, int] | None = None
    packet_sequence: int | None = None

    def decode_images(self) -> dict[str, np.ndarray]:
        images = {}
        for name, frame in self.frames.items():
            image = frame.decode()
            if image is None:
                raise CameraPacketError(f"failed to decode camera image {name!r}")
            images[name] = image
        return images


class CameraClient:
    """Receive latest camera packet from a ZMQ PUB camera server."""

    def __init__(
        self,
        host: str,
        port: int = 5555,
        conflate: bool | None = None,
        max_packet_bytes: int = DEFAULT_MAX_PACKET_BYTES,
        *,
        receive_mode: str = "preview",
        rcv_hwm: int | None = None,
    ):
        self.host = host
        self.port = port
        self.max_packet_bytes = int(max_packet_bytes)
        if self.max_packet_bytes <= 0:
            raise ValueError("max_packet_bytes must be positive")
        if receive_mode not in {"preview", "recording"}:
            raise ValueError("receive_mode must be 'preview' or 'recording'")
        if conflate is None:
            conflate = receive_mode == "preview"
        if receive_mode == "recording" and conflate:
            raise ValueError("recording mode cannot enable CONFLATE")
        hwm = 3 if rcv_hwm is None and receive_mode == "preview" else rcv_hwm
        hwm = 128 if hwm is None else int(hwm)
        if hwm <= 0:
            raise ValueError("rcv_hwm must be positive")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.setsockopt(zmq.RCVHWM, hwm)
        self.socket.setsockopt(zmq.MAXMSGSIZE, self.max_packet_bytes)
        if conflate:
            self.socket.setsockopt(zmq.CONFLATE, True)
        self.socket.connect(f"tcp://{host}:{port}")

    def read(self, timeout_ms: int = 1000) -> dict[str, Any] | None:
        """Read one decoded packet, or return None on timeout."""

        if not self.socket.poll(timeout_ms):
            return None

        packed = self.socket.recv()
        if len(packed) > self.max_packet_bytes:
            raise CameraPacketError(
                f"camera packet is {len(packed)} bytes; limit is {self.max_packet_bytes}"
            )
        try:
            packet = msgpack.unpackb(
                packed,
                raw=False,
                strict_map_key=True,
                max_bin_len=self.max_packet_bytes,
                max_str_len=4096,
                max_array_len=128,
                max_map_len=512,
                max_ext_len=0,
            )
        except (ValueError, TypeError, msgpack.ExtraData) as exc:
            raise CameraPacketError(f"invalid msgpack camera packet: {exc}") from exc
        return decode_packet(
            packet,
            host=self.host,
            port=self.port,
            receive_wall_timestamp=time.time(),
            receive_monotonic_timestamp=time.monotonic(),
        )

    def read_envelope(self, timeout_ms: int = 1000) -> NormalizedPacket | None:
        """Read a packet without decoding its image payloads."""
        if not self.socket.poll(timeout_ms):
            return None
        packed = self.socket.recv()
        if len(packed) > self.max_packet_bytes:
            raise CameraPacketError(
                f"camera packet is {len(packed)} bytes; limit is {self.max_packet_bytes}"
            )
        try:
            packet = msgpack.unpackb(
                packed, raw=False, strict_map_key=True,
                max_bin_len=self.max_packet_bytes, max_str_len=4096,
                max_array_len=128, max_map_len=512, max_ext_len=0,
            )
        except (ValueError, TypeError, msgpack.ExtraData) as exc:
            raise CameraPacketError(f"invalid msgpack camera packet: {exc}") from exc
        return decode_envelope(
            packet, host=self.host, port=self.port,
            receive_wall_timestamp=time.time(),
            receive_monotonic_timestamp=time.monotonic(),
        )

    def close(self):
        self.socket.close(linger=0)
        self.context.term()

def decode_packet(
    packet: dict[str, Any], *, host: str | None = None, port: int | None = None,
    receive_wall_timestamp: float | None = None,
    receive_monotonic_timestamp: float | None = None,
) -> dict[str, Any]:
    """Decode all image blobs in a camera packet."""

    if not isinstance(packet, dict):
        raise CameraPacketError("camera packet must be a mapping")
    envelope = decode_envelope(
        packet,
        host=host,
        port=port,
        receive_wall_timestamp=receive_wall_timestamp,
        receive_monotonic_timestamp=receive_monotonic_timestamp,
    )
    decoded_images = envelope.decode_images()
    decoded_timestamps = {
        name: _legacy_timestamp_seconds(frame)
        for name, frame in envelope.frames.items()
    }
    decoded_sequences = {name: frame.sequence for name, frame in envelope.frames.items()}
    decoded_provenance = {
        name: {
            "timestamp_quality": frame.timestamp_quality,
            "payload_encoding": frame.payload_encoding,
            "device_timestamp": frame.device_timestamp,
            "device_unit": frame.device_unit,
            "timestamp_domain": frame.timestamp_domain,
            "server_wall_timestamp": frame.server_wall_timestamp,
            "server_monotonic": frame.server_monotonic_timestamp,
            "receive_wall_timestamp": frame.receive_wall_timestamp,
            "receive_monotonic_timestamp": frame.receive_monotonic_timestamp,
            "clock_domain": frame.clock_domain,
        }
        for name, frame in envelope.frames.items()
    }
    decoded: dict[str, Any] = {
        "schema": envelope.schema,
        "session_id": envelope.session_id,
        "timestamps": decoded_timestamps,
        "sequences": decoded_sequences,
        "images": decoded_images,
        "raw_payloads": {
            name: frame.payload for name, frame in envelope.frames.items()
        },
        "payload_encodings": {
            name: frame.payload_encoding for name, frame in envelope.frames.items()
        },
        "frame_provenance": decoded_provenance,
        "metadata": envelope.metadata,
        "producer_gaps": dict(envelope.producer_gaps or {}),
        "publisher_gaps": dict(envelope.publisher_gaps or {}),
        "packet_sequence": envelope.packet_sequence,
    }
    if host is not None:
        decoded["host"] = host
    if port is not None:
        decoded["port"] = port
    if receive_wall_timestamp is not None:
        decoded["receive_wall_timestamp"] = receive_wall_timestamp
    if receive_monotonic_timestamp is not None:
        decoded["receive_monotonic_timestamp"] = receive_monotonic_timestamp
    return decoded


def decode_envelope(
    packet: dict[str, Any], *, host: str | None = None, port: int | None = None,
    receive_wall_timestamp: float | None = None,
    receive_monotonic_timestamp: float | None = None,
) -> NormalizedPacket:
    """Normalize v3/v4 wire packets while retaining encoded payloads."""
    if not isinstance(packet, dict):
        raise CameraPacketError("camera packet must be a mapping")
    schema = packet.get("schema")
    if schema not in {CAMERA_PACKET_SCHEMA, CAMERA_PACKET_V4_SCHEMA}:
        raise CameraPacketError(f"unsupported camera packet schema: {schema!r}")
    session_id = _session_id(packet.get("session_id"))
    metadata = _mapping(packet.get("metadata", {}), "metadata")
    frames: dict[str, NormalizedCameraFrame] = {}
    if schema == CAMERA_PACKET_SCHEMA:
        images = _mapping(packet.get("images"), "images")
        timestamps = _mapping(packet.get("timestamps"), "timestamps")
        sequences = _mapping(packet.get("sequences", {}), "sequences")
        provenance = metadata.get("frame_provenance", {})
        if not isinstance(provenance, dict):
            provenance = {}
        for raw_name, payload in images.items():
            name = _frame_name(raw_name)
            _validate_payload(name, payload)
            source = provenance.get(name, {})
            if not isinstance(source, dict):
                source = {}
            device_timestamp = _optional_float(source.get("device"), f"device[{name!r}]")
            quality = source.get("timestamp_quality")
            if device_timestamp is None:
                quality = "host_after_capture"
            if not isinstance(quality, str) or not quality:
                quality = "device" if device_timestamp is not None else "host_after_capture"
            frames[name] = NormalizedCameraFrame(
                name, _non_negative_int(sequences.get(name), f"sequences[{name!r}]"),
                payload, "image/png" if name.endswith("_depth") else "image/jpeg",
                quality,
                device_timestamp=device_timestamp,
                device_unit=_optional_str(source.get("device_unit")),
                timestamp_domain=_optional_str(
                    source.get("device_timestamp_domain_type", source.get("timestamp_domain"))
                ),
                clock_domain=(
                    _optional_str(source.get("device_clock_domain"))
                    or "server_wall"
                ),
                server_wall_timestamp=_finite_float(
                    timestamps.get(name), f"timestamps[{name!r}]"
                ), receive_wall_timestamp=receive_wall_timestamp,
                receive_monotonic_timestamp=receive_monotonic_timestamp,
                server_monotonic_timestamp=_optional_float(source.get("server_monotonic"), f"server_monotonic[{name!r}]"),
            )
    else:
        streams = _mapping(packet.get("streams"), "streams")
        for raw_name, raw_frame in streams.items():
            name = _frame_name(raw_name)
            frame = _mapping(raw_frame, f"streams[{name!r}]")
            payload = frame.get("payload")
            _validate_payload(name, payload)
            timestamps = _mapping(frame.get("timestamps", {}), f"timestamps[{name!r}]")
            quality = frame.get("timestamp_quality")
            if not isinstance(quality, str) or not quality:
                raise CameraPacketError(f"timestamp_quality[{name!r}] is invalid")
            device_timestamp = _optional_float(
                timestamps.get("device"), f"device[{name!r}]"
            )
            if device_timestamp is None:
                quality = "host_after_capture"
            frames[name] = NormalizedCameraFrame(
                name, _non_negative_int(frame.get("sequence"), f"sequence[{name!r}]"),
                payload, _payload_encoding(frame.get("payload_encoding")), quality,
                device_timestamp=device_timestamp,
                device_unit=_optional_str(timestamps.get("device_unit")),
                timestamp_domain=_optional_str(
                    timestamps.get(
                        "device_timestamp_domain_type", timestamps.get("timestamp_domain")
                    )
                ),
                server_wall_timestamp=_optional_float(timestamps.get("server_wall"), f"server_wall[{name!r}]"),
                server_monotonic_timestamp=_optional_float(timestamps.get("server_monotonic"), f"server_monotonic[{name!r}]"),
                receive_wall_timestamp=receive_wall_timestamp,
                receive_monotonic_timestamp=receive_monotonic_timestamp,
                clock_domain=(
                    _optional_str(timestamps.get("device_clock_domain"))
                    or _optional_str(timestamps.get("clock_domain"))
                    or "server_wall"
                ),
            )
    if not frames:
        raise CameraPacketError("camera packet images/streams must not be empty")
    producer_gaps = _gap_mapping(packet.get("producer_gaps", {}), "producer_gaps")
    publisher_gaps = _gap_mapping(packet.get("publisher_gaps", {}), "publisher_gaps")
    packet_sequence = packet.get("packet_sequence")
    if packet_sequence is not None:
        packet_sequence = _non_negative_int(packet_sequence, "packet_sequence")
    return NormalizedPacket(
        schema=schema,
        session_id=session_id,
        frames=frames,
        metadata=metadata,
        host=host,
        port=port,
        producer_gaps=producer_gaps,
        publisher_gaps=publisher_gaps,
        packet_sequence=packet_sequence,
    )


def normalize_packet(
    packet: Any,
    streams: tuple[str, ...] | list[str] | None = None,
    *,
    decode_images: bool = True,
    receive_wall_timestamp: float | None = None,
    receive_monotonic_timestamp: float | None = None,
) -> NormalizedPacket:
    """Normalize wire, legacy, and envelope-like packets at one boundary.

    The collector cache consumes this type only.  Keeping compatibility
    handling here prevents transport-shape parsing from being duplicated in
    the recording path.
    """
    if isinstance(packet, NormalizedPacket):
        return packet
    if isinstance(packet, Mapping):
        schema = packet.get("schema")
        if (
            schema in {CAMERA_PACKET_SCHEMA, CAMERA_PACKET_V4_SCHEMA}
            and "raw_payloads" not in packet
        ):
            return decode_envelope(
                packet,
                receive_wall_timestamp=receive_wall_timestamp,
                receive_monotonic_timestamp=receive_monotonic_timestamp,
            )
        if "images" not in packet:
            raise CameraPacketError("camera packet must contain images or a supported schema")
        return _normalize_legacy_mapping(
            packet,
            streams,
            receive_wall_timestamp=receive_wall_timestamp,
            receive_monotonic_timestamp=receive_monotonic_timestamp,
        )
    session_id = getattr(packet, "session_id", None)
    packet_frames = getattr(packet, "frames", None)
    if session_id is None or not isinstance(packet_frames, Mapping):
        raise CameraPacketError("camera packet must be a mapping or packet object")
    normalized_streams = tuple(str(value) for value in (streams or packet_frames.keys()))
    frames: dict[str, NormalizedCameraFrame] = {}
    for raw_name in normalized_streams:
        if raw_name not in packet_frames:
            continue
        name = _frame_name(raw_name)
        source = packet_frames[raw_name]
        payload = getattr(source, "payload", None)
        if isinstance(payload, (bytearray, memoryview)):
            payload = bytes(payload)
        decoded_image = None
        decoder = getattr(source, "decode", None)
        if decode_images and callable(decoder):
            decoded_image = decoder()
        elif getattr(source, "image", None) is not None:
            decoded_image = getattr(source, "image")
        elif isinstance(payload, bytes):
            # Legacy duck-typed envelopes exposed encoded bytes as the image
            # value when no decoder was supplied.
            decoded_image = payload
        elif payload is not None and not isinstance(payload, bytes):
            decoded_image = payload
            payload = b""
        if payload is None:
            payload = b""
        if not isinstance(payload, bytes):
            if decoded_image is not None:
                payload = b""
            else:
                raise CameraPacketError(f"camera image {name!r} payload is not bytes")
        timestamp_quality = getattr(source, "timestamp_quality", None)
        if not isinstance(timestamp_quality, str) or not timestamp_quality:
            timestamp_quality = "host_after_capture"
        frames[name] = NormalizedCameraFrame(
            stream=name,
            sequence=_optional_non_negative_int(
                getattr(source, "sequence", None), f"sequence[{name!r}]"
            ),
            payload=payload,
            payload_encoding=_payload_encoding(
                getattr(source, "payload_encoding", None) or "image/jpeg"
            ),
            timestamp_quality=timestamp_quality,
            device_timestamp=_optional_float(getattr(source, "device_timestamp", None), f"device[{name!r}]"),
            device_unit=_optional_str(getattr(source, "device_unit", None)),
            timestamp_domain=_optional_str(
                getattr(source, "timestamp_domain", None)
            ),
            server_wall_timestamp=_optional_float(
                getattr(source, "server_wall_timestamp", None), f"server_wall[{name!r}]"
            ),
            server_monotonic_timestamp=_optional_float(
                getattr(source, "server_monotonic_timestamp", None), f"server_monotonic[{name!r}]"
            ),
            receive_wall_timestamp=_optional_float(
                getattr(source, "receive_wall_timestamp", None), f"receive_wall[{name!r}]"
            ),
            receive_monotonic_timestamp=_optional_float(
                getattr(source, "receive_monotonic_timestamp", None), f"receive_monotonic[{name!r}]"
            ),
            clock_domain=_optional_str(getattr(source, "clock_domain", None)) or "server_wall",
            decoded_image=decoded_image,
        )
    if not frames:
        raise CameraPacketError("camera packet frames must not be empty")
    return NormalizedPacket(
        schema=str(getattr(packet, "schema", "")),
        session_id=_session_id(session_id),
        frames=frames,
        metadata=dict(getattr(packet, "metadata", {}) or {}),
        producer_gaps=_gap_mapping(getattr(packet, "producer_gaps", {}) or {}, "producer_gaps"),
        publisher_gaps=_gap_mapping(getattr(packet, "publisher_gaps", {}) or {}, "publisher_gaps"),
        packet_sequence=(
            None
            if getattr(packet, "packet_sequence", None) is None
            else _non_negative_int(getattr(packet, "packet_sequence"), "packet_sequence")
        ),
    )


def _normalize_legacy_mapping(
    packet: Mapping[str, Any],
    streams: tuple[str, ...] | list[str] | None,
    *,
    receive_wall_timestamp: float | None = None,
    receive_monotonic_timestamp: float | None = None,
) -> NormalizedPacket:
    session_id = _session_id(packet.get("session_id"))
    images = packet.get("images")
    timestamps = packet.get("timestamps", {})
    sequences = packet.get("sequences", {})
    if not isinstance(images, Mapping) or not isinstance(timestamps, Mapping) or not isinstance(sequences, Mapping):
        raise CameraPacketError("legacy camera packet images/timestamps/sequences must be mappings")
    metadata = packet.get("metadata", {})
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    provenance = packet.get(
        "frame_provenance", metadata.get("frame_provenance", {})
    )
    provenance = provenance if isinstance(provenance, Mapping) else {}
    raw_payloads = packet.get("raw_payloads", {})
    raw_payloads = raw_payloads if isinstance(raw_payloads, Mapping) else {}
    payload_encodings = packet.get("payload_encodings", {})
    payload_encodings = (
        payload_encodings if isinstance(payload_encodings, Mapping) else {}
    )
    packet_receive_wall = packet.get(
        "receive_wall_timestamp", receive_wall_timestamp
    )
    packet_receive_monotonic = packet.get(
        "receive_monotonic_timestamp", receive_monotonic_timestamp
    )
    names = tuple(str(value) for value in (streams or images.keys()))
    frames: dict[str, NormalizedCameraFrame] = {}
    for raw_name in names:
        if raw_name not in images:
            continue
        name = _frame_name(raw_name)
        image = images[raw_name]
        source = provenance.get(raw_name, {})
        source = source if isinstance(source, Mapping) else {}
        payload = source.get("payload")
        if payload is None:
            payload = raw_payloads.get(raw_name, raw_payloads.get(name))
        if isinstance(payload, (bytearray, memoryview)):
            payload = bytes(payload)
        decoded_image = None
        if isinstance(payload, bytes):
            if not isinstance(image, (bytes, bytearray, memoryview)):
                decoded_image = image
        elif isinstance(image, bytes):
            payload = image
        else:
            payload = b""
            decoded_image = image
        device_timestamp = _optional_float(source.get("device_timestamp", source.get("device")), f"device[{name!r}]")
        quality = source.get("timestamp_quality") or ("device" if device_timestamp is not None else "host_after_capture")
        if not isinstance(quality, str) or not quality:
            quality = "host_after_capture"
        frames[name] = NormalizedCameraFrame(
            stream=name,
            sequence=_optional_non_negative_int(
                sequences.get(name), f"sequences[{name!r}]"
            ),
            payload=payload,
            payload_encoding=_payload_encoding(
                source.get("payload_encoding")
                or payload_encodings.get(raw_name, payload_encodings.get(name))
                or ("image/png" if name.endswith("_depth") else "image/jpeg")
            ),
            timestamp_quality=quality,
            device_timestamp=device_timestamp,
            device_unit=_optional_str(source.get("device_unit")),
            timestamp_domain=_optional_str(
                source.get("device_timestamp_domain_type", source.get("timestamp_domain"))
            ),
            server_wall_timestamp=_optional_float(
                source.get("server_wall_timestamp", timestamps.get(name)),
                f"timestamps[{name!r}]",
            ),
            server_monotonic_timestamp=_optional_float(source.get("server_monotonic"), f"server_monotonic[{name!r}]"),
            receive_wall_timestamp=_optional_float(
                source.get("receive_wall_timestamp", packet_receive_wall),
                f"receive_wall[{name!r}]",
            ),
            receive_monotonic_timestamp=_optional_float(
                source.get("receive_monotonic_timestamp", packet_receive_monotonic),
                f"receive_monotonic[{name!r}]",
            ),
            clock_domain=_optional_str(source.get("clock_domain") or source.get("device_clock_domain")) or "server_wall",
            decoded_image=decoded_image,
        )
    if not frames:
        raise CameraPacketError("legacy camera packet images must not be empty")
    return NormalizedPacket(
        schema=str(packet.get("schema", "legacy")),
        session_id=session_id,
        frames=frames,
        metadata=metadata,
        producer_gaps=_gap_mapping(packet.get("producer_gaps", {}), "producer_gaps"),
        publisher_gaps=_gap_mapping(packet.get("publisher_gaps", {}), "publisher_gaps"),
        packet_sequence=(
            None
            if packet.get("packet_sequence") is None
            else _non_negative_int(packet.get("packet_sequence"), "packet_sequence")
        ),
    )


def _frame_name(value: Any) -> str:
    name = str(value)
    if not _STREAM_PATTERN.fullmatch(name):
        raise CameraPacketError(f"invalid camera stream name: {name!r}")
    return name


def _validate_payload(name: str, payload: Any) -> None:
    if not isinstance(payload, bytes):
        raise CameraPacketError(f"camera image {name!r} must be bytes")
    if not payload or len(payload) > DEFAULT_MAX_IMAGE_BYTES:
        raise CameraPacketError(f"camera image {name!r} has invalid size {len(payload)}")


def _payload_encoding(value: Any) -> str:
    if value not in {"image/jpeg", "image/png"}:
        raise CameraPacketError("payload_encoding must be image/jpeg or image/png")
    return value


def _gap_mapping(value: Any, name: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise CameraPacketError(f"{name} must be a mapping")
    result: dict[str, int] = {}
    for stream, count in value.items():
        result[_frame_name(stream)] = _non_negative_int(count, f"{name}[{stream!r}]")
    return result


def _legacy_timestamp_seconds(frame: NormalizedCameraFrame) -> float | None:
    """Expose one best-effort seconds value for legacy decoded consumers."""
    if frame.timestamp_quality == "device" and frame.device_timestamp is not None:
        value = frame.device_timestamp
        if frame.device_unit == "ms":
            value *= 1e-3
        elif frame.device_unit == "us":
            value *= 1e-6
        elif frame.device_unit == "ns":
            value *= 1e-9
        return value
    if frame.server_wall_timestamp is not None:
        return frame.server_wall_timestamp
    if frame.receive_wall_timestamp is not None:
        return frame.receive_wall_timestamp
    if frame.device_timestamp is not None:
        value = frame.device_timestamp
        if frame.device_unit == "ms":
            value *= 1e-3
        elif frame.device_unit == "us":
            value *= 1e-6
        elif frame.device_unit == "ns":
            value *= 1e-9
        return value
    return None


def _optional_float(value: Any, name: str) -> float | None:
    return None if value is None else _finite_float(value, name)


def _optional_str(value: Any) -> str | None:
    if value is not None and not isinstance(value, str):
        raise CameraPacketError("timestamp string field must be a string")
    return value


def _mapping(value: Any, name: str) -> dict[Any, Any]:
    if not isinstance(value, dict):
        raise CameraPacketError(f"camera packet {name} must be a mapping")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise CameraPacketError(f"{name} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CameraPacketError(f"{name} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise CameraPacketError(f"{name} must be a finite number (non-finite value)")
    return parsed


def _non_negative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CameraPacketError(f"{name} must be a non-negative integer")
    if value < 0:
        raise CameraPacketError(f"{name} must be a non-negative integer")
    return value


def _optional_non_negative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    try:
        return _non_negative_int(value, name)
    except CameraPacketError as exc:
        raise CameraPacketError(f"invalid sequence: {exc}") from exc


def _session_id(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise CameraPacketError("camera packet session_id is invalid")
    return value


def read_once(host: str, port: int = 5555, timeout_ms: int = 3000) -> dict[str, Any] | None:
    client = CameraClient(host, port)
    try:
        return client.read(timeout_ms=timeout_ms)
    finally:
        client.close()
