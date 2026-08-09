# Graph Report - .  (2026-08-10)

## Corpus Check
- 39 files · ~43,102 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1002 nodes · 2936 edges · 30 communities (20 shown, 10 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 210 edges (avg confidence: 0.53)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Dataset Writer Transactions
- Gesture Trigger Runtime
- Pi0.5 Dataset Conversion
- Robot State Assembly
- Camera Transport Cache
- Collector Control Protocol
- Field Mapping and GR00T
- Dataset Writer Tests
- Gesture State Machine
- Metadata Recovery
- GR00T Dataset Conversion
- Collection Architecture Docs
- Gesture Plan Parsing
- Gesture Episode Contract
- RealSense Camera Server
- Camera Dependencies
- ROS Setup Resolution
- Collection Launcher
- Collector Environment Setup
- Camera Environment Setup
- Robot State Messages
- Robot State Package
- Camera Python Package
- Camera Viewer Launcher
- RealSense Server Launcher
- Camera Client Test Script
- Collector Command Message
- Collector Python Package
- State Python Package
- Camera Distribution

## God Nodes (most connected - your core abstractions)
1. `LeRobotV21Writer` - 75 edges
2. `FieldSelection` - 56 edges
3. `gesture_plan_from_payload()` - 56 edges
4. `GestureTriggerStateMachine` - 47 edges
5. `LeRobotV21WriterTest` - 47 edges
6. `GestureTriggerNode` - 45 edges
7. `FakeFrame` - 44 edges
8. `_robot_frame()` - 43 edges
9. `_writer()` - 40 edges
10. `RoboStateAssembler` - 32 edges

## Surprising Connections (you probably didn't know these)
- `P0-P2 Reliability and Production Roadmap` --semantically_similar_to--> `Fail-Closed Dataset Integrity Policy`  [INFERRED] [semantically similar]
  graphify-out/memory/query_20260809_143618_检查代码_看一下我的任务_我是否还会有修改新增的的意思和想法_有更好的建议修改_和新的设计_全方位看.md → README.md
- `Collector and Gesture Trigger Runtime Entry Points` --conceptually_related_to--> `Gesture-Driven Episode Control`  [INFERRED]
  src/robo_collector/CMakeLists.txt → configs/gesture_trigger_plan.example.yml
- `Gesture-Driven Episode Control` --shares_data_with--> `ROS 2 Teleoperation Data Collection Pipeline`  [INFERRED]
  configs/gesture_trigger_plan.example.yml → README.md
- `robo_collector_camera.v3 ZMQ Message Transport` --shares_data_with--> `ROS 2 Teleoperation Data Collection Pipeline`  [INFERRED]
  src/camera/README.md → README.md
- `Optional Gesture Trigger and Direct Terminal Control` --shares_data_with--> `Gesture-Driven Episode Control`  [INFERRED]
  graphify-out/memory/query_20260730_102933_所以手势是一个备选_可以在终端直接开始或者停止.md → configs/gesture_trigger_plan.example.yml

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Robo Collector End-to-End Data Pipeline** — readme_data_collection_pipeline, configs_collection_fields_collection_field_configuration, src_camera_readme_robo_collector_camera_v3, src_robo_collector_cmakelists_runtime_entry_points [INFERRED 0.95]
- **Gesture Control and Production Reliability** — configs_gesture_trigger_plan_example_gesture_episode_control, graphify_out_memory_query_20260730_102933_optional_gesture_and_terminal_control, graphify_out_memory_query_20260809_143618_reliability_roadmap [INFERRED 0.85]
- **Robo State Interface Family** — src_robo_state_msgs_cmakelists_robo_state_msgs, src_robo_state_msgs_cmakelists_policystate, src_robo_state_msgs_cmakelists_robotlowstate, src_robo_state_msgs_cmakelists_robostatesample [EXTRACTED 1.00]

## Communities (30 total, 10 thin omitted)

### Community 0 - "Dataset Writer Transactions"
Cohesion: 0.06
Nodes (57): ParquetWriter, _ActiveEpisode, _arrow_schema_for_row(), _camera_stream_from_key(), _cleanup_atomic_text_staging(), _cleanup_transaction_files(), _default_dataset_name(), _ensure_directory_durable() (+49 more)

### Community 1 - "Gesture Trigger Runtime"
Cohesion: 0.06
Nodes (48): AttemptState, create_detectors(), CurrentAttempt, DetectionResult, extract_gesture_vector(), GestureConditionDetector, _l2_distance(), Any (+40 more)

### Community 2 - "Pi0.5 Dataset Conversion"
Cohesion: 0.05
Nodes (71): FrameReaderFactory, ImageEncoder, build_arg_parser(), _build_info(), _channel_stats_list(), ConversionError, ConversionResult, convert_dataset() (+63 more)

### Community 3 - "Robot State Assembly"
Cohesion: 0.05
Nodes (44): DiagnosticStatus, Float32MultiArray, Imu, JointState, PolicyState, RobotLowState, _diagnostic_level_number(), _diagnostic_level_value() (+36 more)

### Community 4 - "Camera Transport Cache"
Cohesion: 0.05
Nodes (42): skipIf, CameraClient, CameraPacketError, decode_packet(), _finite_float(), _mapping(), _non_negative_int(), Any (+34 more)

### Community 5 - "Collector Control Protocol"
Cohesion: 0.05
Nodes (31): IntEnum, RecordCommand, CachedStateSample, _diagnostic_level(), _diagnostic_level_value(), _free_disk_bytes(), LeRobotCollectorNode, main() (+23 more)

### Community 6 - "Field Mapping and GR00T"
Cohesion: 0.06
Nodes (39): SimpleNamespace, _robot_frame_from_msg(), default_field_selection(), field_selection_from_payload(), FieldConfigError, FieldSelection, load_field_selection(), load_optional_field_selection() (+31 more)

### Community 7 - "Dataset Writer Tests"
Cohesion: 0.12
Nodes (10): DifferentShapeFrame, FakeFrame, FakeVideoSink, _joint_position_only_selection(), LeRobotV21WriterTest, _policy_state_fields(), Path, _robot_frame() (+2 more)

### Community 8 - "Gesture State Machine"
Cohesion: 0.19
Nodes (13): GestureTriggerStateMachine, TriggerAction, MetadataSnapshot, TrialMetadataStatus, gesture_plan_from_payload(), PlannedTrial, _ack_discard(), _ack_start() (+5 more)

### Community 9 - "Metadata Recovery"
Cohesion: 0.16
Nodes (28): _artifact_integrity_error(), _cached_file_sha256(), _classify_trial_attempts(), default_progress_path(), _existing_dataset_file(), _file_integrity_error(), load_progress_log(), _parse_progress_current() (+20 more)

### Community 10 - "GR00T Dataset Conversion"
Cohesion: 0.15
Nodes (38): build_arg_parser(), _build_info(), _build_modality(), _camera_keys_from_info(), _camera_stream_from_key(), ConversionError, _convert_row(), _converted_rows_for_plan() (+30 more)

### Community 11 - "Collection Architecture Docs"
Cohesion: 0.07
Nodes (30): Collection Field Configuration, Joint, Orientation, Velocity, Gravity, and History State Fields, Aligned Target and Policy Action Fields, Gesture Recording Fail-Closed Watchdog, Gesture-Driven Episode Control, Handshake Gesture Trigger Plan, Data Collection Code Review, Temporal Alignment and Persistence Priorities (+22 more)

### Community 12 - "Gesture Plan Parsing"
Cohesion: 0.23
Nodes (26): CollectorPlanConfig, _episode_component_string(), _float_value(), GesturePlanError, GestureReference, GestureSourceConfig, _non_negative_float(), _non_negative_int() (+18 more)

### Community 13 - "Gesture Episode Contract"
Cohesion: 0.17
Nodes (10): build_gesture_episode_id(), GestureEpisodeId, _non_negative_int(), parse_gesture_episode_id(), _positive_int(), Canonical episode-id helpers for gesture-triggered recording., _strict_int(), validate_episode_component_text() (+2 more)

### Community 14 - "RealSense Camera Server"
Cohesion: 0.19
Nodes (14): build_argparser(), CameraSpec, encode_jpeg_bgr(), encode_png(), EncodedFrame, get_device_info(), list_devices(), main() (+6 more)

### Community 15 - "Camera Dependencies"
Cohesion: 0.28
Nodes (9): Client Dependency Profile, msgpack, numpy, opencv-python, pyzmq, numpy, pyrealsense2, pyzmq (+1 more)

### Community 18 - "Collector Environment Setup"
Cohesion: 0.83
Nodes (3): create_venv(), setup_data_collection_env.sh script, validate_venv_target()

### Community 19 - "Camera Environment Setup"
Cohesion: 0.83
Nodes (3): create_venv(), setup_camera_env.sh script, validate_venv_target()

### Community 20 - "Robot State Messages"
Cohesion: 0.50
Nodes (4): PolicyState, robo_state_msgs, RoboStateSample, RobotLowState

### Community 21 - "Robot State Package"
Cohesion: 0.67
Nodes (3): robo_state, robo_state_node, test_state_builder

## Knowledge Gaps
- **26 isolated node(s):** `robo-collector-camera`, `run_camera_viewer.sh script`, `run_realsense_server.sh script`, `test_camera_client.sh script`, `numpy` (+21 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `LeRobotCollectorNode` connect `Collector Control Protocol` to `Dataset Writer Transactions`, `Camera Transport Cache`, `Field Mapping and GR00T`?**
  _High betweenness centrality (0.228) - this node is a cross-community bridge._
- **Why does `LeRobotV21Writer` connect `Dataset Writer Transactions` to `Pi0.5 Dataset Conversion`, `Camera Transport Cache`, `Collector Control Protocol`, `Field Mapping and GR00T`, `Dataset Writer Tests`?**
  _High betweenness centrality (0.198) - this node is a cross-community bridge._
- **Why does `GestureTriggerNode` connect `Gesture Trigger Runtime` to `Gesture State Machine`, `Gesture Plan Parsing`?**
  _High betweenness centrality (0.153) - this node is a cross-community bridge._
- **Are the 15 inferred relationships involving `LeRobotV21Writer` (e.g. with `CachedStateSample` and `LeRobotCollectorNode`) actually correct?**
  _`LeRobotV21Writer` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 19 inferred relationships involving `FieldSelection` (e.g. with `_ActiveEpisode` and `LeRobotV21Writer`) actually correct?**
  _`FieldSelection` has 19 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `GestureTriggerStateMachine` (e.g. with `MetadataSnapshot` and `GestureCondition`) actually correct?**
  _`GestureTriggerStateMachine` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `LeRobotV21WriterTest` (e.g. with `FieldSelection` and `LeRobotV21Writer`) actually correct?**
  _`LeRobotV21WriterTest` has 3 INFERRED edges - model-reasoned connections that need verification._