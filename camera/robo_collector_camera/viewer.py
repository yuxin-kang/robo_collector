from __future__ import annotations

import argparse
import time

import cv2

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
    print(f"Connected to tcp://{args.host}:{args.port}")
    print("Press q to quit.")

    try:
        while True:
            packet = client.read(timeout_ms=args.timeout_ms)
            if packet is None:
                print("No frame received")
                continue

            timestamps = packet.get("timestamps", {})
            images = packet.get("images", {})
            if "ego_view" in images:
                rgb = images["ego_view"]
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                ts = timestamps.get("ego_view")
                if ts:
                    latency_ms = (time.time_ns() - int(ts)) / 1e6
                    cv2.putText(
                        bgr,
                        f"latency {latency_ms:.1f} ms",
                        (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                cv2.imshow("ego_view", bgr)

            if args.show_depth and "ego_view_depth" in images:
                depth = images["ego_view_depth"]
                depth_vis = cv2.convertScaleAbs(depth, alpha=0.03)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                cv2.imshow("ego_view_depth", depth_vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        client.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

