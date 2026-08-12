# Graph Report - .  (2026-08-13)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1078 nodes · 3049 edges · 42 communities (28 shown, 14 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 233 edges (avg confidence: 0.6)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8555b0e5`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- LeRobotV21Writer
- pi05_converter.py
- GestureTriggerNode
- Robot State Assembly
- Camera Frame Transport
- gesture_plan_from_payload
- LeRobotV21WriterTest
- gesture_metadata.py
- gr00t_converter.py
- Background Episode Finalization
- gesture_plan.py
- FieldSelection
- CommandReceiptLedger
- field_config.py
- EpisodeSaveWorker
- LeRobotCollectorNode
- convert_dataset
- server_realsense.py
- RecordStateMachine
- collector_node.py
- Validated and Recoverable Collection Pipeline
- RealSense Dependency Profile
- CommandResult
- resolve_ros_setup.sh
- FakeVideoSink
- launch_data_collection.sh
- setup_data_collection_env.sh
- setup_camera_env.sh
- robo_state_msgs
- Collection Field Configuration
- robo_collector_camera.v3 ZMQ Message Transport
- robo_state
- robo_collector_camera/__init__.py
- run_camera_viewer.sh
- run_realsense_server.sh
- test_camera_client.sh
- RecordCommand
- robo_state/__init__.py
- robo-collector-camera
- RoboStateSample
- ValueError
- Protocol

## God Nodes (most connected - your core abstractions)
1. `LeRobotV21Writer` - 75 edges
2. `gesture_plan_from_payload()` - 62 edges
3. `GestureTriggerStateMachine` - 56 edges
4. `LeRobotV21WriterTest` - 51 edges
5. `FakeFrame` - 47 edges
6. `_robot_frame()` - 47 edges
7. `_writer()` - 44 edges
8. `GestureTriggerNode` - 41 edges
9. `FieldSelection` - 37 edges
10. `LeRobotCollectorNode` - 36 edges

## Surprising Connections (you probably didn't know these)
- `Single-Consumer Background Save Worker Recommendation` --semantically_similar_to--> `Background Episode Finalization`  [INFERRED] [semantically similar]
  graphify-out/memory/query_20260812_142509_stop_到_save_episode_的同步保存链会阻塞什么_手势超时是否安全.md → README.md
- `EpisodeSaveWorker` --semantically_similar_to--> `Background Episode Finalization`  [INFERRED] [semantically similar]
  graphify-out/memory/query_20260812_151201_录制停止后_后台保存_进度看门狗和失败恢复如何协作.md → README.md
- `Save Progress Heartbeat Recommendation` --semantically_similar_to--> `Rolling Save Progress Watchdog`  [INFERRED] [semantically similar]
  graphify-out/memory/query_20260812_142625_保存确认超时后延迟完成保存与_discard_是否存在竞态.md → README.md
- `Progress-Sequence Save Watchdog` --semantically_similar_to--> `Rolling Save Progress Watchdog`  [INFERRED] [semantically similar]
  graphify-out/memory/query_20260812_151201_录制停止后_后台保存_进度看门狗和失败恢复如何协作.md → README.md
- `SAVING Discard Rejection and FAILED Recovery` --semantically_similar_to--> `SAVING Discard Rejection and Failed-Save Recovery`  [INFERRED] [semantically similar]
  graphify-out/memory/query_20260812_151201_录制停止后_后台保存_进度看门狗和失败恢复如何协作.md → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Background Save Lifecycle** — readme_background_episode_finalization, readme_saving_status_progress, readme_save_discard_failure_policy, graphify_out_memory_query_20260812_151201_episode_save_worker, graphify_out_memory_query_20260812_151201_saving_failure_recovery, src_robo_collector_cmakelists_test_save_worker [INFERRED 0.95]
- **Bounded Progress-Aware Save Wait** — configs_gesture_trigger_plan_example_save_confirm_timeout_sec, configs_gesture_trigger_plan_example_max_save_wait_sec, readme_rolling_save_progress_watchdog, readme_maximum_metadata_reconciliation_wait, graphify_out_memory_query_20260812_151201_progress_sequence_watchdog [INFERRED 0.95]
- **Durable Shutdown and Recovery** — readme_shutdown_grace_and_staging_recovery, readme_background_episode_finalization, graphify_out_memory_query_20260812_151201_shutdown_wait_and_staging_recovery, graphify_out_memory_query_20260812_141957_lerobotcollectornode_transactional_save_snapshot [INFERRED 0.85]
- **Gesture Control and Production Reliability** — configs_gesture_trigger_plan_example_gesture_episode_control, graphify_out_memory_query_20260730_102933_optional_gesture_and_terminal_control, graphify_out_memory_query_20260809_143618_reliability_roadmap [INFERRED 0.85]
- **Robo State Interface Family** — src_robo_state_msgs_cmakelists_robo_state_msgs, src_robo_state_msgs_cmakelists_policystate, src_robo_state_msgs_cmakelists_robotlowstate, src_robo_state_msgs_cmakelists_robostatesample [EXTRACTED 1.00]

## Communities (42 total, 14 thin omitted)

### Community 0 - "LeRobotV21Writer"
Cohesion: 0.06
Nodes (58): ParquetWriter, Protocol, _ActiveEpisode, _arrow_schema_for_row(), _camera_stream_from_key(), _cleanup_atomic_text_staging(), _cleanup_transaction_files(), _default_dataset_name() (+50 more)

### Community 1 - "pi05_converter.py"
Cohesion: 0.05
Nodes (71): FrameReaderFactory, ImageEncoder, build_arg_parser(), _build_info(), _channel_stats_list(), ConversionError, ConversionResult, convert_dataset() (+63 more)

### Community 2 - "GestureTriggerNode"
Cohesion: 0.06
Nodes (46): ProgressCurrent, AttemptState, create_detectors(), CurrentAttempt, DetectionResult, extract_gesture_vector(), GestureConditionDetector, _l2_distance() (+38 more)

### Community 3 - "Robot State Assembly"
Cohesion: 0.05
Nodes (44): DiagnosticStatus, Float32MultiArray, Imu, JointState, PolicyState, RobotLowState, _diagnostic_level_number(), _diagnostic_level_value() (+36 more)

### Community 4 - "Camera Frame Transport"
Cohesion: 0.05
Nodes (42): skipIf, CameraClient, CameraPacketError, decode_packet(), _finite_float(), _mapping(), _non_negative_int(), Any (+34 more)

### Community 5 - "gesture_plan_from_payload"
Cohesion: 0.11
Nodes (14): GestureTriggerStateMachine, MetadataSnapshot, TriggerAction, gesture_plan_from_payload(), PlannedTrial, GestureContractTest, _plan_payload(), _ack_discard() (+6 more)

### Community 6 - "LeRobotV21WriterTest"
Cohesion: 0.11
Nodes (13): FieldSelection, RobotFrame, DifferentShapeFrame, FailingVideoSink, FakeFrame, FakeVideoSink, _joint_position_only_selection(), LeRobotV21WriterTest (+5 more)

### Community 7 - "gesture_metadata.py"
Cohesion: 0.10
Nodes (42): build_gesture_episode_id(), GestureEpisodeId, _non_negative_int(), parse_gesture_episode_id(), _positive_int(), Canonical episode-id helpers for gesture-triggered recording., _strict_int(), validate_episode_component_text() (+34 more)

### Community 8 - "gr00t_converter.py"
Cohesion: 0.15
Nodes (37): _build_info(), _build_modality(), _camera_keys_from_info(), _camera_stream_from_key(), ConversionError, ConversionResult, _convert_row(), _converted_rows_for_plan() (+29 more)

### Community 9 - "Background Episode Finalization"
Cohesion: 0.07
Nodes (37): Gesture Episode Control, Gesture Trigger Plan, max_save_wait_sec 600 Seconds, save_confirm_timeout_sec 120 Seconds, Collector Modes and Record Commands, Gesture Attempt and Metadata Recovery States, Recording Status Guide, Manual and Gesture Recording Control Guidance (+29 more)

### Community 10 - "gesture_plan.py"
Cohesion: 0.22
Nodes (26): CollectorPlanConfig, _episode_component_string(), _float_value(), GesturePlanError, GestureSourceConfig, load_gesture_trigger_plan(), _non_negative_float(), _non_negative_int() (+18 more)

### Community 11 - "FieldSelection"
Cohesion: 0.16
Nodes (13): SimpleNamespace, default_field_selection(), FieldSelection, Selected robot fields, expressed in user-facing YAML field names., Return the legacy writer field set., message_stamp_sec(), Any, Pure sample-selection rules shared by the ROS collector and tests. (+5 more)

### Community 12 - "CommandReceiptLedger"
Cohesion: 0.16
Nodes (13): IntEnum, CollectorMode, CommandFingerprint, CommandReceiptLedger, CommandReplay, Enum, str, Pure recording command state machine for the LeRobot collector. (+5 more)

### Community 13 - "field_config.py"
Cohesion: 0.18
Nodes (12): field_selection_from_payload(), FieldConfigError, load_field_selection(), load_optional_field_selection(), _parse_group(), Any, Path, ValueError (+4 more)

### Community 14 - "EpisodeSaveWorker"
Cohesion: 0.13
Nodes (7): ProgressReporter, ResultT, EpisodeSaveWorker, Single-owner background execution for durable episode finalization., Runs at most one save at a time and transports progress safely., SaveProgress, EpisodeSaveWorkerTest

### Community 15 - "LeRobotCollectorNode"
Cohesion: 0.18
Nodes (7): _diagnostic_level_value(), _free_disk_bytes(), LeRobotCollectorNode, main(), Any, Node, Waits for START/STOP commands and records aligned state + RGB frames.

### Community 16 - "convert_dataset"
Cohesion: 0.25
Nodes (11): build_arg_parser(), convert_dataset(), main(), ArgumentParser, _create_source_dataset(), FakeFrame, Gr00tConverterTest, _policy_selection() (+3 more)

### Community 17 - "server_realsense.py"
Cohesion: 0.19
Nodes (14): build_argparser(), CameraSpec, encode_jpeg_bgr(), encode_png(), EncodedFrame, get_device_info(), list_devices(), main() (+6 more)

### Community 18 - "RecordStateMachine"
Cohesion: 0.15
Nodes (3): Owns legal transitions for START/STOP/DISCARD commands., RecordStateMachine, RecordStateMachineTest

### Community 19 - "collector_node.py"
Cohesion: 0.16
Nodes (8): RecordCommand, RoboStateSample, CachedStateSample, _diagnostic_level(), ROS2 node that records validated RoboState samples into LeRobot episodes., _record_command_name(), _robot_frame_from_msg(), SaveResult

### Community 20 - "Validated and Recoverable Collection Pipeline"
Cohesion: 0.25
Nodes (9): Data Collection Code Review, Temporal Alignment and Persistence Priorities, Collector, Camera, State, Dataset, Gesture, and Field Components, Collection Pipeline Repair Summary, Validated and Recoverable Collection Pipeline, RoboStateAssembler, LeRobotV21Writer, CameraFrameCache, Collector, Gesture, Field, and Metadata Components, Attended Collection Readiness Assessment, Comprehensive Production Readiness Review (+1 more)

### Community 21 - "RealSense Dependency Profile"
Cohesion: 0.28
Nodes (9): Client Dependency Profile, msgpack, numpy, opencv-python, pyzmq, numpy, pyrealsense2, pyzmq (+1 more)

### Community 26 - "setup_data_collection_env.sh"
Cohesion: 0.83
Nodes (3): create_venv(), setup_data_collection_env.sh script, validate_venv_target()

### Community 27 - "setup_camera_env.sh"
Cohesion: 0.83
Nodes (3): create_venv(), setup_camera_env.sh script, validate_venv_target()

### Community 28 - "robo_state_msgs"
Cohesion: 0.50
Nodes (4): PolicyState, robo_state_msgs, RoboStateSample, RobotLowState

### Community 29 - "Collection Field Configuration"
Cohesion: 0.67
Nodes (3): Collection Field Configuration, Joint, Orientation, Velocity, Gravity, and History State Fields, Aligned Target and Policy Action Fields

### Community 30 - "robo_collector_camera.v3 ZMQ Message Transport"
Cohesion: 0.67
Nodes (3): Robo Collector Camera Module, robo_collector_camera.v3 ZMQ Message Transport, Server Session Identity and Sequence Reset Safety

### Community 31 - "robo_state"
Cohesion: 0.67
Nodes (3): robo_state, robo_state_node, test_state_builder

## Knowledge Gaps
- **25 isolated node(s):** `launch_data_collection.sh script`, `robo-collector-camera`, `run_camera_viewer.sh script`, `run_realsense_server.sh script`, `test_camera_client.sh script` (+20 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **14 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Work-memory lessons

**Preferred sources** — corroborated by past sessions; start here.
- `RecordStateMachine` (5× useful, score=4.408147947) _(code changed — re-verify)_
- `LeRobotCollectorNode` (5× useful, score=4.407426942) _(code changed — re-verify)_
- `GestureTriggerStateMachine` (4× useful, score=3.669989216) _(code changed — re-verify)_
- `LeRobotV21Writer` (4× useful, score=3.669885605) _(code changed — re-verify)_
- `CameraFrameCache` (3× useful, score=2.669963293)
- `.save_episode()` (2× useful, score=1.999864971) _(code changed — re-verify)_
- `_write_files_transactional_locked()` (2× useful, score=1.999761359) _(code changed — re-verify)_
- `GestureTriggerNode` (2× useful, score=1.670824903) _(code changed — re-verify)_
- `RoboStateAssembler` (2× useful, score=1.670124246)
- `CollectorMode` (2× useful, score=1.475238744) _(code changed — re-verify)_

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `LeRobotV21Writer` connect `LeRobotV21Writer` to `pi05_converter.py`, `LeRobotV21WriterTest`, `LeRobotCollectorNode`, `convert_dataset`, `collector_node.py`, `FakeVideoSink`?**
  _High betweenness centrality (0.180) - this node is a cross-community bridge._
- **Why does `_imu()` connect `Robot State Assembly` to `FieldSelection`?**
  _High betweenness centrality (0.167) - this node is a cross-community bridge._
- **Why does `GesturePlanError` connect `gesture_plan.py` to `LeRobotV21Writer`, `GestureTriggerNode`, `gesture_plan_from_payload`?**
  _High betweenness centrality (0.167) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `LeRobotV21Writer` (e.g. with `CachedStateSample` and `LeRobotCollectorNode`) actually correct?**
  _`LeRobotV21Writer` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `GestureTriggerStateMachine` (e.g. with `GestureCondition` and `GestureReference`) actually correct?**
  _`GestureTriggerStateMachine` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `LeRobotV21WriterTest` (e.g. with `LeRobotV21Writer` and `RobotFrame`) actually correct?**
  _`LeRobotV21WriterTest` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `FakeFrame` (e.g. with `LeRobotV21Writer` and `RobotFrame`) actually correct?**
  _`FakeFrame` has 2 INFERRED edges - model-reasoned connections that need verification._