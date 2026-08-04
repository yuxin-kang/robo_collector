# Graph Report - robo_collector  (2026-07-30)

## Corpus Check
- 51 files · ~33,944 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 897 nodes · 2454 edges · 40 communities (31 shown, 9 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 199 edges (avg confidence: 0.53)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `dfa2e321`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- pi05_converter.py
- RecordStateMachine
- RoboStateAssembler
- gr00t_converter.py
- CameraFrameCache
- lerobot_dataset.py
- gesture_plan_from_payload
- GestureTriggerNode
- gesture_trigger_node.py
- GestureConditionDetector
- LeRobotV21WriterTest
- test_gesture_trigger_node.py
- GestureTriggerStateMachine
- server_realsense.py
- RealSense Dependency Profile
- Gesture Trigger Plan
- robo_collector
- Robo Collector Workspace
- Collection Field Configuration
- robo_state_msgs
- resolve_ros_setup.sh
- RecordCommand
- launch_data_collection.sh
- setup_data_collection_env.sh
- setup_camera_env.sh
- robo_collector_camera/__init__.py
- run_camera_viewer.sh
- run_realsense_server.sh
- test_camera_client.sh
- robo_state/__init__.py
- robo-collector-camera
- LeRobotV21Writer
- LeRobotCollectorNode
- field_config.py
- collector_node.py
- gesture_metadata.py
- .save_episode
- FieldSelection
- Q: 看一下我的数据采集代码 有没有需要进一步优化的地方 做一个codereview
- CachedStateSample

## God Nodes (most connected - your core abstractions)
1. `LeRobotV21Writer` - 63 edges
2. `FieldSelection` - 54 edges
3. `GestureTriggerNode` - 43 edges
4. `gesture_plan_from_payload()` - 41 edges
5. `RoboStateAssembler` - 31 edges
6. `GestureTriggerStateMachine` - 30 edges
7. `LeRobotV21WriterTest` - 30 edges
8. `RoboStateNode` - 29 edges
9. `ConversionError` - 28 edges
10. `convert_dataset()` - 28 edges

## Surprising Connections (you probably didn't know these)
- `StepIt Robot State Normalization` --semantically_similar_to--> `robo_state_node`  [INFERRED] [semantically similar]
  README.md → src/robo_state/CMakeLists.txt
- `Gesture Triggered Recording` --semantically_similar_to--> `gesture_trigger_node`  [INFERRED] [semantically similar]
  README.md → src/robo_collector/CMakeLists.txt
- `Robo Collector Workspace` --references--> `robo_collector_msgs`  [EXTRACTED]
  README.md → src/robo_collector_msgs/CMakeLists.txt
- `RealSense RGB Alignment` --conceptually_related_to--> `Robo Collector Camera`  [INFERRED]
  README.md → src/camera/README.md
- `Robo Collector Workspace` --references--> `Collection Field Configuration`  [EXTRACTED]
  README.md → configs/collection_fields.yml

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Robo Collector Data Collection Pipeline** — readme_stepit_robot_state_normalization, readme_robo_state_sample, readme_realsense_rgb_alignment, src_camera_readme_realsense_publisher, src_camera_readme_camera_client, readme_lerobot_v2_1_dataset [INFERRED 0.95]
- **Gesture Triggered Episode Control** — readme_gesture_triggered_recording, configs_gesture_trigger_plan_example_gesture_trigger_plan, configs_gesture_trigger_plan_example_gesture_source, configs_gesture_trigger_plan_example_collector_control, src_robo_collector_cmakelists_gesture_trigger_node, src_robo_collector_msgs_cmakelists_recordcommand [INFERRED 0.85]
- **Robo State Interface Family** — src_robo_state_msgs_cmakelists_robo_state_msgs, src_robo_state_msgs_cmakelists_policystate, src_robo_state_msgs_cmakelists_robotlowstate, src_robo_state_msgs_cmakelists_robostatesample [EXTRACTED 1.00]

## Communities (40 total, 9 thin omitted)

### Community 0 - "pi05_converter.py"
Cohesion: 0.06
Nodes (65): FrameReaderFactory, ImageEncoder, build_arg_parser(), _build_info(), _channel_stats_list(), ConversionError, ConversionResult, convert_dataset() (+57 more)

### Community 1 - "RecordStateMachine"
Cohesion: 0.13
Nodes (11): IntEnum, CollectorMode, CommandResult, Enum, str, Pure recording command state machine for the LeRobot collector., Owns legal transitions for START/STOP/DISCARD commands., RecordCommandType (+3 more)

### Community 2 - "RoboStateAssembler"
Cohesion: 0.05
Nodes (44): DiagnosticStatus, Float32MultiArray, Imu, JointState, PolicyState, RobotLowState, _diagnostic_level_number(), _diagnostic_level_value() (+36 more)

### Community 3 - "gr00t_converter.py"
Cohesion: 0.12
Nodes (41): build_arg_parser(), _build_info(), _build_modality(), _camera_keys_from_info(), _camera_stream_from_key(), ConversionError, ConversionResult, convert_dataset() (+33 more)

### Community 4 - "CameraFrameCache"
Cohesion: 0.06
Nodes (41): skipIf, CameraClient, CameraPacketError, decode_packet(), _finite_float(), _mapping(), _non_negative_int(), Any (+33 more)

### Community 5 - "lerobot_dataset.py"
Cohesion: 0.11
Nodes (22): _ActiveEpisode, _camera_stream_from_key(), _default_dataset_name(), _joint_order_feature_keys(), _json_content(), _jsonl_content(), _normalize_camera_keys(), Any (+14 more)

### Community 6 - "gesture_plan_from_payload"
Cohesion: 0.12
Nodes (36): build_gesture_episode_id(), GestureEpisodeId, _non_negative_int(), parse_gesture_episode_id(), _positive_int(), Canonical episode-id helpers for gesture-triggered recording., _strict_int(), validate_episode_component_text() (+28 more)

### Community 7 - "GestureTriggerNode"
Cohesion: 0.16
Nodes (6): collector_status_is_fresh(), _format_l2(), GestureTriggerNode, main(), Node, Publishes collector START/STOP commands from configured gesture detections.

### Community 8 - "gesture_trigger_node.py"
Cohesion: 0.17
Nodes (22): AttemptState, create_detectors(), CurrentAttempt, DetectionResult, _l2_distance(), Enum, str, Pure gesture detection and orchestration state machine. (+14 more)

### Community 9 - "GestureConditionDetector"
Cohesion: 0.25
Nodes (8): extract_gesture_vector(), GestureConditionDetector, Any, GestureCondition, update_detector_safely(), FakeRobotState, FakeSample, GestureDetectionTest

### Community 10 - "LeRobotV21WriterTest"
Cohesion: 0.22
Nodes (7): FakeFrame, LeRobotV21WriterTest, _policy_state_fields(), Path, _robot_frame(), _write_fake_parquet(), _writer()

### Community 11 - "test_gesture_trigger_node.py"
Cohesion: 0.15
Nodes (11): attempt_state_is_armed(), detection_tail_frames(), diagnostic_values_map(), parse_collector_status(), Any, Path, resolve_dataset_root(), resolve_progress_path() (+3 more)

### Community 12 - "GestureTriggerStateMachine"
Cohesion: 0.25
Nodes (5): GestureTriggerStateMachine, TrialMetadataStatus, GestureTriggerStateTest, _plan_payload(), _snapshot()

### Community 13 - "server_realsense.py"
Cohesion: 0.19
Nodes (14): build_argparser(), CameraSpec, encode_jpeg_bgr(), encode_png(), EncodedFrame, get_device_info(), list_devices(), main() (+6 more)

### Community 14 - "RealSense Dependency Profile"
Cohesion: 0.15
Nodes (19): Camera Client, ego_view, head, JPEG RGB Encoding, RealSense Publisher, Robo Collector Camera, robo_collector_camera.v2, ZMQ PUB SUB Transport (+11 more)

### Community 15 - "Gesture Trigger Plan"
Cohesion: 0.21
Nodes (12): end, Gesture Trigger Plan, ready, return_to_ready_condition, shake_hand, start, tail_bounds, task_end_condition (+4 more)

### Community 16 - "robo_collector"
Cohesion: 0.17
Nodes (12): robo_collector, test_camera_cache, test_collector_state, test_field_config, test_gesture_contract, test_gesture_detection, test_gesture_metadata, test_gesture_trigger_node (+4 more)

### Community 17 - "Robo Collector Workspace"
Cohesion: 0.25
Nodes (11): Isaac GR00T Export, LeRobot v2.1 Dataset, RealSense RGB Alignment, Robo Collector Workspace, Robo State Sample, ROS Setup Resolution, StepIt Robot State Normalization, lerobot_collector_node (+3 more)

### Community 18 - "Collection Field Configuration"
Cohesion: 0.22
Nodes (9): action_history, ang_vel_history, Collection Field Configuration, gravity_history, joint_pos_rel_history, joint_vel_history, motion_anchor_ang_vel_b, motion_anchor_lin_vel_b (+1 more)

### Community 19 - "robo_state_msgs"
Cohesion: 0.33
Nodes (6): aligned_target_pos, Gesture Source, PolicyState, robo_state_msgs, RoboStateSample, RobotLowState

### Community 21 - "RecordCommand"
Cohesion: 0.50
Nodes (4): Collector Control, Record Command Protocol, RecordCommand, robo_collector_msgs

### Community 23 - "setup_data_collection_env.sh"
Cohesion: 0.83
Nodes (3): create_venv(), setup_data_collection_env.sh script, validate_venv_target()

### Community 24 - "setup_camera_env.sh"
Cohesion: 0.83
Nodes (3): create_venv(), setup_camera_env.sh script, validate_venv_target()

### Community 31 - "LeRobotV21Writer"
Cohesion: 0.15
Nodes (5): LeRobotV21Writer, _optional_finite_timestamp(), Writes one parquet and one RGB MP4 per saved episode. The writer intentionally…, _rgb_shape(), _validate_monotonic_timestamp()

### Community 32 - "LeRobotCollectorNode"
Cohesion: 0.19
Nodes (8): RecordCommand, _diagnostic_level(), _diagnostic_level_value(), LeRobotCollectorNode, main(), Any, Node, Waits for START/STOP commands and records aligned state + RGB frames.

### Community 33 - "field_config.py"
Cohesion: 0.17
Nodes (12): field_selection_from_payload(), FieldConfigError, load_field_selection(), load_optional_field_selection(), _parse_group(), Any, Path, ValueError (+4 more)

### Community 34 - "collector_node.py"
Cohesion: 0.17
Nodes (12): SimpleNamespace, ROS2 node that records validated RoboState samples into LeRobot episodes., _robot_frame_from_msg(), default_field_selection(), Return the legacy writer field set., message_stamp_sec(), Any, Pure sample-selection rules shared by the ROS collector and tests. (+4 more)

### Community 35 - "gesture_metadata.py"
Cohesion: 0.17
Nodes (26): _classify_trial_attempts(), default_progress_path(), _existing_dataset_file(), load_progress_log(), _parse_progress_current(), _parse_progress_events(), _progress_int(), ProgressCurrent (+18 more)

### Community 36 - ".save_episode"
Cohesion: 0.23
Nodes (4): ParquetWriter, OpenCvVideoSink, SaveResult, write_parquet_pyarrow_stream()

### Community 37 - "FieldSelection"
Cohesion: 0.10
Nodes (14): FieldSelection, Selected robot fields, expressed in user-facing YAML field names., Robo Collector final data recording package., RobotFrame, _validate_len(), _validate_robot_frame(), _validate_selected_robot_value(), FakeFrame (+6 more)

### Community 38 - "Q: 看一下我的数据采集代码 有没有需要进一步优化的地方 做一个codereview"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: 看一下我的数据采集代码 有没有需要进一步优化的地方 做一个codereview, Source Nodes

## Knowledge Gaps
- **39 isolated node(s):** `launch_data_collection.sh script`, `robo-collector-camera`, `run_camera_viewer.sh script`, `run_realsense_server.sh script`, `test_camera_client.sh script` (+34 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `LeRobotCollectorNode` connect `LeRobotCollectorNode` to `RecordStateMachine`, `collector_node.py`, `field_config.py`, `CameraFrameCache`, `FieldSelection`, `CachedStateSample`, `LeRobotV21Writer`?**
  _High betweenness centrality (0.226) - this node is a cross-community bridge._
- **Why does `LeRobotV21Writer` connect `LeRobotV21Writer` to `LeRobotCollectorNode`, `pi05_converter.py`, `collector_node.py`, `gr00t_converter.py`, `.save_episode`, `FieldSelection`, `lerobot_dataset.py`, `CachedStateSample`, `LeRobotV21WriterTest`?**
  _High betweenness centrality (0.186) - this node is a cross-community bridge._
- **Why does `GestureTriggerNode` connect `GestureTriggerNode` to `gesture_metadata.py`, `gesture_plan_from_payload`, `gesture_trigger_node.py`, `test_gesture_trigger_node.py`, `GestureTriggerStateMachine`?**
  _High betweenness centrality (0.155) - this node is a cross-community bridge._
- **Are the 15 inferred relationships involving `LeRobotV21Writer` (e.g. with `CachedStateSample` and `LeRobotCollectorNode`) actually correct?**
  _`LeRobotV21Writer` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 19 inferred relationships involving `FieldSelection` (e.g. with `_ActiveEpisode` and `LeRobotV21Writer`) actually correct?**
  _`FieldSelection` has 19 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `GestureTriggerNode` (e.g. with `AttemptState` and `CurrentAttempt`) actually correct?**
  _`GestureTriggerNode` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `RoboStateAssembler` (e.g. with `RoboStateNode` and `StateBuilderTest`) actually correct?**
  _`RoboStateAssembler` has 2 INFERRED edges - model-reasoned connections that need verification._