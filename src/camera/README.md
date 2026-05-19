# Robo Collector Camera

Minimal RealSense camera publisher/client for G1 data collection.

This module is intentionally small:

- Robot side: read one or more Intel RealSense RGB streams and publish a composed packet over ZMQ.
- Host side: receive latest frame and decode to NumPy arrays.
- Transport: ZMQ PUB/SUB + msgpack + JPEG for RGB.

## Directory

```text
src/camera/
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
cd /path/to/robo_collector/src/camera
bash scripts/setup_camera_env.sh --server
source .venv_camera/bin/activate
bash scripts/run_realsense_server.sh --list-devices
```

Dual-camera RGB publisher:

```bash
bash scripts/run_realsense_server.sh \
  --camera head:<D405_SERIAL> \
  --camera ego_view:<D435I_SERIAL> \
  --port 5555 \
  --width 640 --height 480 --fps 30 \
  --jpeg-quality 80 \
  --no-depth
```

Output streams:

- `head`: D405 level-view RGB image, JPEG encoded, decoded as `uint8 [H, W, 3]`.
- `ego_view`: D435i overhead RGB image, JPEG encoded, decoded as `uint8 [H, W, 3]`.

Legacy single-camera mode:

```bash
bash scripts/run_realsense_server.sh --serial <SERIAL> --port 5555 --no-depth
```

## Host Side: Camera Client

On the host:

```bash
cd /home/kyx/robot/vla/robo_collector/src/camera
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
    head = packet["images"]["head"]
    ego_view = packet["images"]["ego_view"]
    head_timestamp_sec = packet["timestamps"]["head"]
```

## Message Format

```python
{
    "schema": "robo_collector_camera.v2",
    "timestamps": {
        "head": 1770000000.0,
        "ego_view": 1770000000.0,
    },
    "images": {
        "head": b"...jpg bytes...",
        "ego_view": b"...jpg bytes...",
    },
    "metadata": {
        "cameras": {
            "head": {"serial": "<D405_SERIAL>", "device_info": {...}},
            "ego_view": {"serial": "<D435I_SERIAL>", "device_info": {...}},
        },
        "width": 640,
        "height": 480,
        "fps": 30,
    },
}
```
