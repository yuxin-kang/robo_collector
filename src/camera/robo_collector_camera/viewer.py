from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from robo_collector_camera.client import CameraClient


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="View frames from robo_collector camera server.")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--timeout-ms", type=int, default=1000)
    parser.add_argument("--show-depth", action="store_true")
    return parser


def main():
    args = build_argparser().parse_args()
    client = CameraClient(args.host, args.port)
    status_logger = CameraStatusLogger(args.host, args.port)
    print(f"Connected to tcp://{args.host}:{args.port}")
    print("Press q to quit.")

    try:
        while True:
            packet = client.read(timeout_ms=args.timeout_ms)
            if packet is None:
                status_logger.mark_missing()
                continue

            timestamps = packet.get("timestamps", {})
            images = packet.get("images", {})
            if not images:
                status_logger.mark_missing("no decodable camera images received")
                continue

            status_logger.mark_received(sorted(images))
            tiles = []
            for name in sorted(images):
                image = images[name]
                if name.endswith("_depth"):
                    if not args.show_depth:
                        continue
                    bgr = cv2.convertScaleAbs(image, alpha=0.03)
                    bgr = cv2.applyColorMap(bgr, cv2.COLORMAP_JET)
                else:
                    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                _annotate_stream(bgr, name, timestamps.get(name))
                tiles.append(bgr)

            if tiles:
                cv2.imshow("camera_streams", _tile_images(tiles))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        client.close()
        cv2.destroyAllWindows()


@dataclass
class CameraStatusLogger:
    host: str
    port: int
    log_interval_sec: float = 5.0

    def __post_init__(self) -> None:
        self._state = "unknown"
        self._frame_count = 0
        self._last_info_monotonic_sec = 0.0
        self._last_warn_monotonic_sec = 0.0

    def mark_missing(self, message: str | None = None) -> None:
        now = time.monotonic()
        if (
            self._state == "missing"
            and now - self._last_warn_monotonic_sec <= self.log_interval_sec
        ):
            return

        self._state = "missing"
        self._last_warn_monotonic_sec = now
        if message is None:
            message = f"no camera frame received from tcp://{self.host}:{self.port}"
        _print_warn(message)

    def mark_received(self, streams: list[str]) -> None:
        now = time.monotonic()
        self._frame_count += 1
        recovered = self._state != "receiving"
        if (
            not recovered
            and now - self._last_info_monotonic_sec <= self.log_interval_sec
        ):
            return

        self._state = "receiving"
        self._last_info_monotonic_sec = now
        _print_info(
            "receiving camera frames from "
            f"tcp://{self.host}:{self.port}: "
            f"count={self._frame_count} streams={','.join(streams)}"
        )


def _print_info(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def _print_warn(message: str) -> None:
    print(f"\033[33m[WARN] {message}\033[0m", flush=True)


def _annotate_stream(image_bgr, name: str, timestamp) -> None:
    cv2.putText(
        image_bgr,
        name,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    latency_ms = _latency_ms(timestamp)
    if latency_ms is None:
        return
    cv2.putText(
        image_bgr,
        f"latency {latency_ms:.1f} ms",
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )


def _latency_ms(timestamp) -> float | None:
    if timestamp is None:
        return None
    value = float(timestamp)
    if value > 1e12:
        return (time.time_ns() - int(value)) / 1e6
    return (time.time() - value) * 1000.0


def _tile_images(images: list[np.ndarray]) -> np.ndarray:
    if len(images) == 1:
        return images[0]

    tile_height = max(image.shape[0] for image in images)
    tile_width = max(image.shape[1] for image in images)
    cols = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / cols)
    canvas = np.zeros((rows * tile_height, cols * tile_width, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        row = index // cols
        col = index % cols
        y0 = row * tile_height
        x0 = col * tile_width
        canvas[y0 : y0 + image.shape[0], x0 : x0 + image.shape[1]] = image
    return canvas


if __name__ == "__main__":
    main()
