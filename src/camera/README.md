# Robo Collector Camera

Minimal RealSense camera publisher/client for G1 data collection.

The deployment-default transport is `robo_collector_camera.v3`; the client can
also read the opt-in `v4` envelope. Upgrade and restart the remote camera
server together with the collector; v2 packets are deliberately rejected
because they do not carry the server session identifier required to detect
sequence resets safely.

This module is intentionally small:

- Robot side: read one or more Intel RealSense RGB streams and publish a composed packet over ZMQ.
- Host side: receive a normalized envelope, then decode to NumPy arrays only
  when the consumer needs images.
- Transport: ZMQ PUB/SUB + msgpack + JPEG for RGB.
- Recording mode: bounded `RCVHWM=128` without `CONFLATE`; preview mode keeps
  `CONFLATE` and a small bounded queue for low latency.

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
    raw_spool.py
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

For camera-side durable capture, pass a spool root. Each server process gets
an isolated session directory; raw JPEG/PNG chunks are written before the
latest-frame publisher and can be recovered after a crash. The spool is a
camera-side source, not automatically a task Episode:

```bash
bash scripts/run_realsense_server.sh \
  --camera head:<D405_SERIAL> \
  --camera ego_view:<D435I_SERIAL> \
  --raw-spool-dir /data/robo_collector/camera_spools \
  --packet-schema v3
```

`--packet-schema v3` is the default. Use `--packet-schema v4` only after the
collector has been validated with the normalized envelope path. A camera-side
spool is the complete-source candidate; a host-side Raw Episode made from ZMQ
packets remains explicitly `source_scope=transport_observed`. To bind the
spool to task Episodes, mount this directory on the collection host and pass
`--camera-raw-spool-root <mounted-path>` with
`--raw-source-scope camera_capture` to the collector. The collector imports
only records in the task's START/STOP wall-time window and stores the source
session/hash in the task manifest. The camera spool uses the `RSP1` framed
msgpack format; the collector converts its verified prefix into the host Raw
Episode `RER1` JSON/base64 record format, preserving the original source hash
and timestamp provenance. The source manifest is explicitly
`robo_collector.camera_spool.v1`, while the host task manifest remains
`robo_collector.raw_episode.v1`; the adapter must validate the boundary rather
than treating the two record encodings as interchangeable. Without this
binding, do not claim complete capture from the receiver copy.

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

camera = CameraClient("192.168.123.164", 5555, receive_mode="preview")
packet = camera.read(timeout_ms=10)

if packet is not None:
    head = packet["images"]["head"]
    ego_view = packet["images"]["ego_view"]
    head_timestamp_sec = packet["timestamps"]["head"]
```

The collector uses `receive_mode="recording"`, which disables `CONFLATE` and
uses the bounded receive policy. For a raw-first collector, use
`read_envelope()`/`decode_envelope()` and retain each frame's payload and
timestamp provenance; do not use the decoded mapping as evidence of frames
that were dropped before reception.

## Message Format

```python
{
    "schema": "robo_collector_camera.v3",
    "session_id": "unique-server-process-id",
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

The v4 equivalent stores each stream under `streams[name]` with `sequence`,
encoded `payload`, `payload_encoding`, device/server timestamps and
`timestamp_quality`. Both versions normalize to the same envelope API. A
missing device timestamp is labeled `host_after_capture`; receive monotonic
time is retained for local alignment and audit.
