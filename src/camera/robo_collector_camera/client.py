from __future__ import annotations

from typing import Any

import cv2
import msgpack
import numpy as np
import zmq


class CameraClient:
    """Receive latest camera packet from a ZMQ PUB camera server."""

    def __init__(self, host: str, port: int = 5555, conflate: bool = True):
        self.host = host
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.setsockopt(zmq.RCVHWM, 3)
        if conflate:
            self.socket.setsockopt(zmq.CONFLATE, True)
        self.socket.connect(f"tcp://{host}:{port}")

    def read(self, timeout_ms: int = 1000) -> dict[str, Any] | None:
        """Read one decoded packet, or return None on timeout."""

        if not self.socket.poll(timeout_ms):
            return None

        packed = self.socket.recv()
        packet = msgpack.unpackb(packed, raw=False)

        decoded_images: dict[str, np.ndarray] = {}
        for name, blob in packet.get("images", {}).items():
            image = self._decode_image(name, blob)
            if image is not None:
                decoded_images[name] = image

        return {
            "schema": packet.get("schema"),
            "timestamps": packet.get("timestamps", {}),
            "images": decoded_images,
            "metadata": packet.get("metadata", {}),
            "host": self.host,
            "port": self.port,
        }

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


def read_once(host: str, port: int = 5555, timeout_ms: int = 3000) -> dict[str, Any] | None:
    client = CameraClient(host, port)
    try:
        return client.read(timeout_ms=timeout_ms)
    finally:
        client.close()

