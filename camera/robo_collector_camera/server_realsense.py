from __future__ import annotations

import argparse
import time
from typing import Any

import cv2
import msgpack
import numpy as np
import pyrealsense2 as rs
import zmq


def encode_jpeg_bgr(image_bgr: np.ndarray, quality: int) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode RGB image as JPEG")
    return buffer.tobytes()


def encode_png(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode depth image as PNG")
    return buffer.tobytes()


def get_device_info(pipeline_profile: rs.pipeline_profile) -> dict[str, str]:
    device = pipeline_profile.get_device()
    info = {}
    for key in [
        rs.camera_info.name,
        rs.camera_info.serial_number,
        rs.camera_info.firmware_version,
    ]:
        if device.supports(key):
            info[str(key).split(".")[-1]] = device.get_info(key)
    return info


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish RealSense frames over ZMQ.")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial", type=str, default=None)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--depth", dest="depth", action="store_true", default=True)
    parser.add_argument("--no-depth", dest="depth", action="store_false")
    parser.add_argument("--print-every", type=int, default=100)
    return parser


def main():
    args = build_argparser().parse_args()

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    if args.depth:
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    device_info = get_device_info(profile)

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 20)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f"tcp://*:{args.port}")

    print(f"RealSense camera server listening on tcp://*:{args.port}")
    print(f"Device info: {device_info}")
    print(
        f"Streams: ego_view {args.width}x{args.height}@{args.fps}, "
        f"depth={'on' if args.depth else 'off'}"
    )

    sent = 0
    last_report = time.monotonic()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            timestamp_ns = time.time_ns()
            color_bgr = np.asanyarray(color_frame.get_data())

            images: dict[str, bytes] = {
                "ego_view": encode_jpeg_bgr(color_bgr, args.jpeg_quality),
            }
            timestamps: dict[str, int] = {
                "ego_view": timestamp_ns,
            }

            if args.depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())
                    images["ego_view_depth"] = encode_png(depth_image)
                    timestamps["ego_view_depth"] = timestamp_ns

            packet: dict[str, Any] = {
                "schema": "robo_collector_camera.v1",
                "timestamps": timestamps,
                "images": images,
                "metadata": {
                    "camera": "realsense",
                    "device_info": device_info,
                    "width": args.width,
                    "height": args.height,
                    "fps": args.fps,
                    "depth": args.depth,
                    "jpeg_quality": args.jpeg_quality,
                },
            }

            socket.send(msgpack.packb(packet, use_bin_type=True))
            sent += 1

            if args.print_every > 0 and sent % args.print_every == 0:
                now = time.monotonic()
                elapsed = max(now - last_report, 1e-6)
                print(f"Image sending FPS: {args.print_every / elapsed:.2f}")
                last_report = now

    except KeyboardInterrupt:
        print("Stopping camera server...")
    finally:
        pipeline.stop()
        socket.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()

