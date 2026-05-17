# Robo Collector Camera

Minimal RealSense camera publisher/client for G1 data collection.

This module is intentionally small:

- Robot side: read Intel RealSense color/depth frames and publish over ZMQ.
- Host side: receive latest frame and decode to NumPy arrays.
- Transport: ZMQ PUB/SUB + msgpack + JPEG for RGB + PNG for depth.

## Directory

```text
camera/
  pyproject.toml
  requirements-client.txt
  requirements-realsense.txt
  scripts/
    setup_camera_env.sh
    run_realsense_server.sh
    run_camera_viewer.sh
    test_camera_client.sh
  robo_collector_camera/
    client.py
    server_realsense.py
    viewer.py
```

## Robot Side: RealSense Publisher

On the robot Jetson NX:

```bash
cd /path/to/robo_collector/camera
bash scripts/setup_camera_env.sh --server
source .venv_camera/bin/activate
bash scripts/run_realsense_server.sh --port 5555
```

Default output streams:

- `ego_view`: RGB image, JPEG encoded, decoded as `uint8 [H, W, 3]`.
- `ego_view_depth`: depth image, PNG encoded, decoded as raw depth array.

Disable depth if you only want RGB:

```bash
bash scripts/run_realsense_server.sh --port 5555 --no-depth
```

## Host Side: Camera Client

On the host:

```bash
cd /home/kyx/robot/vla/robo_collector/camera
bash scripts/setup_camera_env.sh --client
source .venv_camera/bin/activate
bash scripts/test_camera_client.sh --host 192.168.123.164 --port 5555
```

Open viewer:

```bash
bash scripts/run_camera_viewer.sh --host 192.168.123.164 --port 5555
```

## Python Use

```python
from robo_collector_camera.client import CameraClient

camera = CameraClient("192.168.123.164", 5555)
packet = camera.read(timeout_ms=10)

if packet is not None:
    image = packet["images"]["ego_view"]
    timestamp_ns = packet["timestamps"]["ego_view"]
```

## Message Format

```python
{
    "schema": "robo_collector_camera.v1",
    "timestamps": {
        "ego_view": 1770000000000000000,
        "ego_view_depth": 1770000000000000000,
    },
    "images": {
        "ego_view": b"...jpg bytes...",
        "ego_view_depth": b"...png bytes...",
    },
    "metadata": {
        "camera": "realsense",
        "serial": "...",
        "width": 640,
        "height": 480,
        "fps": 30,
    },
}
```

