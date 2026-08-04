from __future__ import annotations

import math
import re
from typing import Any

import cv2
import msgpack
import numpy as np
import zmq

CAMERA_PACKET_SCHEMA = "robo_collector_camera.v2"
DEFAULT_MAX_PACKET_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_IMAGE_BYTES = 32 * 1024 * 1024
_STREAM_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class CameraPacketError(ValueError):
    """Raised when a camera packet violates the transport contract."""


class CameraClient:
    """Receive latest camera packet from a ZMQ PUB camera server."""

    def __init__(
        self,
        host: str,
        port: int = 5555,
        conflate: bool = True,
        max_packet_bytes: int = DEFAULT_MAX_PACKET_BYTES,
    ):
        self.host = host
        self.port = port
        self.max_packet_bytes = int(max_packet_bytes)
        if self.max_packet_bytes <= 0:
            raise ValueError("max_packet_bytes must be positive")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.setsockopt(zmq.RCVHWM, 3)
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
        return decode_packet(packet, host=self.host, port=self.port)

    def close(self):
        self.socket.close(linger=0)
        self.context.term()

    @staticmethod
    def _decode_image(name: str, blob: bytes) -> np.ndarray | None:
        arr = np.frombuffer(blob, dtype=np.uint8)
        if name.endswith("_depth"):
            return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_packet(
    packet: dict[str, Any], *, host: str | None = None, port: int | None = None
) -> dict[str, Any]:
    """Decode all image blobs in a camera packet."""

    if not isinstance(packet, dict):
        raise CameraPacketError("camera packet must be a mapping")
    if packet.get("schema") != CAMERA_PACKET_SCHEMA:
        raise CameraPacketError(
            f"unsupported camera packet schema: {packet.get('schema')!r}"
        )
    images = _mapping(packet.get("images"), "images")
    timestamps = _mapping(packet.get("timestamps"), "timestamps")
    sequences = _mapping(packet.get("sequences", {}), "sequences")
    metadata = _mapping(packet.get("metadata", {}), "metadata")
    if not images:
        raise CameraPacketError("camera packet images must not be empty")

    decoded_images: dict[str, np.ndarray] = {}
    decoded_timestamps: dict[str, float] = {}
    decoded_sequences: dict[str, int] = {}
    for raw_name, blob in images.items():
        name = str(raw_name)
        if not _STREAM_PATTERN.fullmatch(name):
            raise CameraPacketError(f"invalid camera stream name: {name!r}")
        if not isinstance(blob, bytes):
            raise CameraPacketError(f"camera image {name!r} must be bytes")
        if not blob or len(blob) > DEFAULT_MAX_IMAGE_BYTES:
            raise CameraPacketError(
                f"camera image {name!r} has invalid size {len(blob)}"
            )
        timestamp = _finite_float(timestamps.get(name), f"timestamps[{name!r}]")
        decoded_timestamps[name] = timestamp
        decoded_sequences[name] = _non_negative_int(
            sequences.get(name), f"sequences[{name!r}]"
        )
        image = CameraClient._decode_image(name, blob)
        if image is None:
            raise CameraPacketError(f"failed to decode camera image {name!r}")
        decoded_images[name] = image

    decoded: dict[str, Any] = {
        "schema": CAMERA_PACKET_SCHEMA,
        "timestamps": decoded_timestamps,
        "sequences": decoded_sequences,
        "images": decoded_images,
        "metadata": metadata,
    }
    if host is not None:
        decoded["host"] = host
    if port is not None:
        decoded["port"] = port
    return decoded


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
        raise CameraPacketError(f"{name} must be a finite number")
    return parsed


def _non_negative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CameraPacketError(f"{name} must be a non-negative integer")
    if value < 0:
        raise CameraPacketError(f"{name} must be a non-negative integer")
    return value


def read_once(host: str, port: int = 5555, timeout_ms: int = 3000) -> dict[str, Any] | None:
    client = CameraClient(host, port)
    try:
        return client.read(timeout_ms=timeout_ms)
    finally:
        client.close()
