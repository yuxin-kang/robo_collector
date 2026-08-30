# Dyna-style MCAP v1 Phase 0 execution contract

Status: **FROZEN FOR PHASE 0 IMPLEMENTATION**

Contract ID: `robo_collector.dyna_mcap_v1.phase0`

Contract version: `1`

Date: 2026-08-29

This is the committed execution source of truth for the seven contract gates in
the MCAP-first migration PRD. It uses only supported public MCAP behavior. In
particular, canonical camera and robot data are separate physical MCAP files;
this contract does not claim Dyna's unpublished schemas, writer extensions, or
private chunk layout.

Normative terms `MUST`, `MUST NOT`, `REQUIRED`, and `EXACTLY` are testable. JSON
named JCS is RFC 8785 UTF-8 with no BOM or trailing newline. Closed v1 objects
reject unknown keys. Hashes are lowercase SHA-256 hex. Decimal strings match
`0|[1-9][0-9]*`; signed decimals match `0|-?[1-9][0-9]*`. Lists sort by the
named keys using unsigned UTF-8 byte order unless numeric order is stated.

## 1. Stage contract (gate 1)

### 1.1 Common rules

The eight prepublication stages, in dependency order, are:

1. `validate_landing`
2. `normalize_clocks`
3. `build_index`
4. `align_rgb`
5. `encode_video`
6. `write_canonical_groups`
7. `structural_qc`
8. `content_qc`

Every stage has `stage_version="1"`, `output_schema_version="1"`, and a closed
config object with `format="robo_collector.stage_config"`,
`format_version=1`, and its own `stage_name`. Config hashes, stage-key hashes,
logical input hashes, and logical output hashes use the canonical JSON/hash
helpers specified by this repository. Byte artifacts hash their exact bytes;
logical directories hash JCS of sorted `{path,sha256,size_bytes}` member rows.

The exact stage key preimage is:

```json
{"config_sha256":"<sha256>","implementation_id":"<package-version+git-tree-id>","input_hashes":[{"name":"<logical-name>","sha256":"<sha256>"}],"output_schema_version":"1","stage_name":"<stage-name>","stage_version":"1"}
```

`input_hashes` is the dependency projection listed below, sorted by `name`.
Duplicate or missing logical names fail validation. Attempts, owners, fencing
tokens, leases, hosts, PIDs, timestamps, durations, statuses, and metrics never
enter a config, stage key, or semantic output hash.

### 1.2 Exact stage schemas and projections

Defaults shown below are serialized explicitly before hashing; omitted defaults
are invalid in persisted evidence.

#### `validate_landing`

```json
{"format":"robo_collector.stage_config","format_version":1,"max_checkpoint_payload_bytes":"1048576","max_record_content_bytes":"67108864","require_data_crc":true,"require_summary":true,"stage_name":"validate_landing","validation_profile":"landing_v1"}
```

- inputs: `landing_mcap`, `checkpoint_journal`, `collection_manifest`
- outputs: `validation-report.json`, `source-inventory.json`
- dependency projection: all later stages consume `validation_report`; stages
  reading messages also consume the immutable `landing_mcap` source hash.
- a recovered prefix is a new immutable source artifact and therefore has a new
  `landing_mcap` hash. Recovery never rewrites the original partial.

#### `normalize_clocks`

```json
{"affine_min_edges":"30","affine_ppm_limit":"2000","clock_policy":"rgb_affine_v2","even_sample_limit":"512","fallback_max_interval_ns":"1000000000","max_uncertainty_ns":"20000000","quantile_method":"nearest_rank","stage_name":"normalize_clocks","format":"robo_collector.stage_config","format_version":1}
```

- inputs: `landing_mcap`, `validation_report`
- outputs: `normalized-timeline.json`, `clock-segments.json`
- consumers project `normalized_timeline`; QC additionally projects
  `clock_segments`.

#### `build_index`

```json
{"format":"robo_collector.stage_config","format_version":1,"index_backend":"sqlite","index_schema":"channel_time_v1","page_size_bytes":"4096","stage_name":"build_index","synchronous":"FULL"}
```

- inputs: `landing_mcap`, `normalized_timeline`
- output: `channel-time-index.sqlite`
- the SQLite output hash is over deterministic logical row export
  `channel-time-index.rows.jcs`, not database page bytes. The export sorts by
  `(normalized_time_ns,topic_rank,source_session_id,source_sequence,
  collector_record_id)` and is the logical `channel_time_index` input.

#### `align_rgb`

```json
{"action_max_age_ns":"20000000","format":"robo_collector.stage_config","format_version":1,"max_camera_residual_ns":"20000000","max_state_residual_ns":"20000000","policy":"rgb_affine_v2","policy_version":"2","reference_camera_stream":"<required-stream-id>","stage_name":"align_rgb"}
```

- `reference_camera_stream` has no default and MUST be an existing camera stream.
- inputs: `channel_time_index`, `normalized_timeline`
- outputs: `aligned-rows.jcs`, `selection-gaps.jcs`
- `alignment_config_sha256 = SHA256(JCS(the exact config object above))`.
- `legacy_rgb_v1` is not a valid config value for this canonical stage; see §7.

#### `encode_video`

```json
{"backend":"pyav-libx264","backend_version":"17.1","format":"robo_collector.stage_config","format_version":1,"options":[{"name":"annexb","value":"1"},{"name":"aud","value":"1"},{"name":"bframes","value":"0"},{"name":"closed_gop","value":"1"},{"name":"crf","value":"18"},{"name":"gop_duration_ns","value":"1000000000"},{"name":"preset","value":"medium"},{"name":"repeat_headers","value":"1"},{"name":"scenecut","value":"0"},{"name":"tune","value":"zerolatency"}],"stage_name":"encode_video"}
```

- inputs: `landing_mcap`, `normalized_timeline`
- outputs: `encoded-access-units.jcs`, `video-keyframes.parquet`
- the canonical host backend is the PyAV 17.1 contract family, installed from
  the exact `av==17.1.0` package pin, binding to libx264. Options
  are the complete v1 registry in §2.4; unregistered or duplicate names fail
  closed. OpenCV `VideoWriter`, `ffmpeg-python`, and an MCAP protobuf convenience
  writer are not canonical encoder/container authorities.

#### `write_canonical_groups`

```json
{"camera_chunk_target_bytes":"16777216","camera_compression":"NONE","chunk_crc":true,"format":"robo_collector.stage_config","format_version":1,"index_types":"ALL","profile":"robo_collector.mcap.v1","repeat_channels":true,"repeat_schemas":true,"robot_chunk_target_bytes":"4194304","robot_compression":"ZSTD","stage_name":"write_canonical_groups","use_chunking":true,"use_statistics":true,"use_summary_offsets":true}
```

- inputs: `aligned_rows`, `encoded_access_units`, `landing_mcap`,
  `normalized_timeline`, `selection_gaps`
- outputs: `camera.mcap`, `robot.mcap`, `provenance.json`
- uses supported public `mcap==1.4.0` writer settings. No private force-chunk or
  append behavior is permitted.

#### `structural_qc`

```json
{"doctor_policy":"ci_and_rollout","format":"robo_collector.stage_config","format_version":1,"internal_validator":"mcap_v1","require_cli_when_available":true,"stage_name":"structural_qc"}
```

- inputs: `camera_mcap`, `robot_mcap`, `source_inventory`,
  `video_keyframes`
- output: `structural-qc-evidence.json`
- the internal validator is mandatory. CLI `mcap doctor` is mandatory in CI,
  rollout, and deployment preflight; absence at runtime is recorded, not treated
  as successful doctor execution.

#### `content_qc`

```json
{"format":"robo_collector.stage_config","format_version":1,"policy_name":"canonical_content_v1","policy_version":"1","stage_name":"content_qc"}
```

- inputs: `aligned_rows`, `clock_segments`, `provenance`, `source_inventory`,
  `structural_qc_evidence`, `camera_mcap`, `robot_mcap`
- output: `quality.json`
- `quality.json` status and hashes follow §2.

The successful stage-evidence output list is exactly the outputs above, using
repository-relative logical paths. A stage is reusable only when its full stage
key and every recorded output hash/size validate.

## 2. QC, metrics, alignment hash, and encoder registry (gate 2)

### 2.1 Canonical status precedence

Rules are evaluated completely; order never short-circuits evidence generation.
The resulting status uses this strict precedence:

1. `QUARANTINED` if structural integrity is invalid or ambiguous, a trusted
   prefix/hash/manifest/descriptor is corrupt, source binding mismatches, a
   required stream is missing, or a required gap is unclassified.
2. `REJECT` if any content rule has `severity="CRITICAL"` and `result="FAIL"`.
3. `REVIEW` if source completeness is recoverable but lacks a valid STOP or has
   known accepted-not-durable loss, any rule result is `REVIEW`, or any
   `WARNING` rule fails.
4. `READY` only when none of the conditions above applies and all required
   stages are successful with validated outputs.

`QUARANTINED > REJECT > REVIEW > READY`. An operator cannot mutate this value;
overrides are separate audit facts and never select a READY pointer.

### 2.2 Rule evidence hash

Each rule hashes JCS of exactly:

```json
{"artifacts":[{"name":"<logical-name>","sha256":"<sha256>","size_bytes":"<decimal>"}],"format":"robo_collector.qc_evidence","format_version":1,"observations":[{"code":"<stable-code>","subject":"<stable-subject>","value":"<canonical-string>"}],"rule_id":"<rule-id>","rule_version":"<version>"}
```

Artifacts sort by `(name,sha256)`; observations sort by `(code,subject,value)`.
Duplicates fail. `evidence_sha256` is SHA-256 of those bytes. A rule with no
artifact or observation hashes the exact nonempty object with empty arrays; it
MUST NOT substitute a runtime report, timestamp, or host-specific path.

### 2.3 Metric encoding

Metric `value` is always a string. Integer metrics use signed or unsigned
decimal grammar. Rational measurements use reduced `numerator/denominator` with
positive denominator. Decimal measurements use the shortest non-exponent base-10
representation that round-trips the source integer/rational calculation; `-0`
normalizes to `0`. Boolean values are `true|false`; enum values are registered
uppercase ASCII tokens. NaN, infinity, exponent notation, locale separators,
units inside values, and binary-float `repr` output are invalid. Units are UCUM
ASCII tokens or `1` for dimensionless metrics.

### 2.4 Initial H.264 encoder option registry

The only v1 backend ID is `pyav-libx264`, with `backend_version="17.1"` and
package pin `av==17.1.0`. It uses
PyAV's public codec context with the host libx264 encoder. Its exact
byte-affecting option registry and defaults are:

| name | grammar | default |
|---|---|---|
| `annexb` | `0|1` | `1` |
| `aud` | `0|1` | `1` |
| `bframes` | unsigned decimal | `0` |
| `closed_gop` | `0|1` | `1` |
| `crf` | integer `0..51` | `18` |
| `gop_duration_ns` | positive decimal | `1000000000` |
| `preset` | `ultrafast|superfast|veryfast|faster|fast|medium|slow|slower|veryslow` | `medium` |
| `repeat_headers` | `0|1` | `1` |
| `scenecut` | unsigned decimal | `0` |
| `tune` | `zerolatency` | `zerolatency` |

All ten rows are serialized, even at defaults, sorted by option name. V1
requires Annex-B, AUD, zero B-frames, closed GOP, repeated headers, and disabled
scene cuts; conflicting values are rejected. Width, height, pixel format,
nominal rational rate, GOP duration, outer compression, stream IDs, and every
SPS/PPS generation remain in the codec-config object defined by the identity
contract and therefore affect `codec.config_sha256`.

## 3. Timestamp, video, and deterministic merge contract (gate 3)

### 3.1 PTS and timebase

Every `VideoAccessUnitV1` uses `timebase_num=1`,
`timebase_den=1000000000`, `pts=timestamps.normalized_time_ns`, and `dts=pts`.
Normalized time MUST be within `0..2^63-1`; no epoch subtraction, frame-count
derivation, or CFR rounding is permitted. One message contains exactly one
decoded-frame Annex-B access unit.

Config generation is zero-based per stream. Generation resets to zero for a new
episode. It increments by one before the first AU after any source-session
change, encoder restart, resolution/pixel-format/profile/level change, or exact
SPS/PPS byte change. The first AU of every generation is an IDR with the current
SPS/PPS prepended. A normal GOP boundary is an IDR but does not increment the
generation when SPS/PPS and encoder configuration are unchanged.

### 3.2 Complete timestamp modes

- `COLLECTOR_DIRECT`: `source_time_ns`, `receive_time_ns`, and
  `normalized_time_ns` are all present and exactly equal; source domain is
  `CLOCK_DOMAIN_COLLECTOR_MONOTONIC`; `normalization_mode` is
  `NORMALIZATION_MODE_COLLECTOR_DIRECT`; uncertainty is `0`; fallback reason is
  `NONE`; policy version is `collector_direct_v1`; clock session is required.
- `LEGACY_V1`: source time is required and is the Raw v1 recorded timestamp
  converted to integer nanoseconds with ties-to-even; receive and normalized
  time equal source time; source domain is `CLOCK_DOMAIN_WALL_UTC`;
  normalization mode is `NORMALIZATION_MODE_LEGACY_V1`; uncertainty is `0`;
  fallback reason is `NONE`; policy version is `legacy_rgb_v1`; clock session is
  `legacy.<episode_id>`. This mode is compatibility evidence only and cannot
  enter a canonical bundle (§7).
- `AFFINE_V2` and `RECEIVE_FALLBACK` retain the exact rules in the MCAP v1 wire
  contract. All modes require nonnegative normalized time.

### 3.3 Per-family merge keys

Canonical order is ascending on the exact tuples below. String fields compare
as UTF-8 bytes; integers compare numerically. `topic_rank` is the frozen wire
rank. No input iteration order may break a tie.

| family | merge key after `(normalized_time_ns,topic_rank)` |
|---|---|
| camera sample | `(source_session_id,source_sequence,packet_sequence,collector_record_id)` |
| video AU | `(source_session_id,source_sequence,config_generation,collector_record_id)` |
| robot state/action | `(source_session_id,source_sequence,collector_record_id)` |
| aligned sample | `(reference_session_id,reference_source_sequence,aligned_row_index)` |
| landing-origin event | `("0",collector_record_id,event_type,lifecycle_attempt)` |
| generated selection-gap event | `("1",reference_session_id,reference_source_sequence,event_type)` |

The literal origin discriminator makes landing events sort before generated
events at an otherwise identical event time. Generated selection-gap events
sort by reference identity, then their `missing` entries sort by `stream_id`.
Only after the complete merge is stable does the writer assign zero-based,
channel-local `Message.sequence`; `EpisodeEventV1.event_sequence` and
`AlignedSampleV1.aligned_row_index` equal their channel sequence. Duplicate full
keys with identical payloads are deduplicated and counted; divergent payloads
under a full key quarantine the candidate.

## 4. Checkpoint journal and collection manifest (gate 4)

### 4.1 Journal frame

`landing/checkpoints.bin` is a sequence of frames:

```text
uint64_le(payload byte length) || payload JCS bytes || uint32_le(CRC32-ISO-HDLC(payload))
```

The CRC uses polynomial `0xEDB88320`, initial/final XOR `0xffffffff` (the
standard zlib/PNG CRC-32). Payload length is at most 1,048,576. The payload is a
closed object of exactly:

```json
{"accepted_snapshot_count":"<decimal>","accepted_snapshot_frontier":"<decimal-or-null>","channels":[{"accepted_count":"<decimal>","accepted_high_watermark":"<decimal-or-null>","channel":"<topic>","durable_count":"<decimal>","durable_high_watermark":"<decimal-or-null>","last_packet_sequence":"<decimal-or-null>","last_source_sequence":"<decimal-or-null>","session_id":"<session-or-empty>","written_count":"<decimal>","written_high_watermark":"<decimal-or-null>"}],"checkpoint_sequence":"<decimal>","durable_byte_offset":"<decimal>","durable_count":"<decimal>","durable_frontier":"<decimal-or-null>","format":"robo_collector.mcap_checkpoint","format_version":1,"generation":"<decimal>","landing_prefix_sha256":"<sha256>","max_unsynced_records":"<decimal>","queue_capacity":"<decimal>","source_fences":[{"accepted_count":"<decimal>","durable_count":"<decimal>","durable_high_watermark":"<decimal-or-null>","end_sequence_inclusive":"<decimal-or-null>","session_id":"<session>","source_id":"<source>","start_sequence_exclusive":"<decimal>","written_count":"<decimal>","written_high_watermark":"<decimal-or-null>"}],"written_count":"<decimal>","written_frontier":"<decimal-or-null>"}
```

Channels sort by `channel`; fences sort by `(source_id,session_id)`. Frontiers
are global collector-record IDs. A null frontier is allowed only when its count
is zero. Counts/frontiers and per-source watermarks are monotonic. The checkpoint
commit/fsync order and loss bounds remain those in the PRD.

### 4.2 In-progress manifest

`manifest.inprogress.json` is JCS, atomically replaced, and has exactly:

```json
{"attempt":"<decimal>","collection_mode":"raw_v1|dual_write|mcap_first","created_time_ns":"<decimal>","episode_id":"<episode-id>","format":"robo_collector.mcap_landing","format_version":1,"landing":{"checkpoint_path":"landing/checkpoints.bin","partial_path":"landing/episode.mcap.partial","writer_profile":"robo_collector.mcap.v1"},"required_sources":[{"session_id":"<session>","source_id":"<source>","start_sequence_exclusive":"<decimal>"}],"state":"OPEN|STOPPING|FAULTED","writer":{"mcap_library":"mcap","mcap_version":"1.4.0","profile":"robo_collector.mcap.v1"}}
```

Required sources sort by `(source_id,session_id)`. `created_time_ns` is audit
state and therefore not an immutable bundle identity input, but the exact closed
or recovered collection manifest bytes are hashed as the
`validate_landing.collection_manifest` input. `RAW_CLOSED` is represented by a
separate immutable `manifest.json`; it is never an in-progress state.

### 4.3 Golden vectors

The Phase 0 fixture named `empty-open` uses the following exact payloads. The
hash/CRC/frame values are generated from these bytes and committed with the
fixture tests; any byte change is a contract change.

Checkpoint payload:

```json
{"accepted_snapshot_count":"0","accepted_snapshot_frontier":null,"channels":[],"checkpoint_sequence":"0","durable_byte_offset":"0","durable_count":"0","durable_frontier":null,"format":"robo_collector.mcap_checkpoint","format_version":1,"generation":"0","landing_prefix_sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","max_unsynced_records":"0","queue_capacity":"64","source_fences":[],"written_count":"0","written_frontier":null}
```

Manifest payload:

```json
{"attempt":"1","collection_mode":"mcap_first","created_time_ns":"0","episode_id":"episode.phase0","format":"robo_collector.mcap_landing","format_version":1,"landing":{"checkpoint_path":"landing/checkpoints.bin","partial_path":"landing/episode.mcap.partial","writer_profile":"robo_collector.mcap.v1"},"required_sources":[],"state":"OPEN","writer":{"mcap_library":"mcap","mcap_version":"1.4.0","profile":"robo_collector.mcap.v1"}}
```

Expected constants:

- checkpoint payload SHA-256: `08faa5824b3f747e0b0023e89f819c4b2dcaba6e0300ad0249dd7f5e7b66cdf4`
- checkpoint payload CRC32 lowercase hex: `87b2f770`
- checkpoint frame lowercase hex: `c8010000000000007b2261636365707465645f736e617073686f745f636f756e74223a2230222c2261636365707465645f736e617073686f745f66726f6e74696572223a6e756c6c2c226368616e6e656c73223a5b5d2c22636865636b706f696e745f73657175656e6365223a2230222c2264757261626c655f627974655f6f6666736574223a2230222c2264757261626c655f636f756e74223a2230222c2264757261626c655f66726f6e74696572223a6e756c6c2c22666f726d6174223a22726f626f5f636f6c6c6563746f722e6d6361705f636865636b706f696e74222c22666f726d61745f76657273696f6e223a312c2267656e65726174696f6e223a2230222c226c616e64696e675f7072656669785f736861323536223a2265336230633434323938666331633134396166626634633839393666623932343237616534316534363439623933346361343935393931623738353262383535222c226d61785f756e73796e6365645f7265636f726473223a2230222c2271756575655f6361706163697479223a223634222c22736f757263655f66656e636573223a5b5d2c227772697474656e5f636f756e74223a2230222c227772697474656e5f66726f6e74696572223a6e756c6c7d70f7b287`
- in-progress manifest SHA-256: `0ff8b4c92a5f6c920c22a7ddde059fcb26fa4ebfc4f7fd4c10b0f81e943f133e`

## 5. Bundle inventory, checksums, and keyframe index (gate 5)

### 5.1 Identity members and checksums

Bundle identity contains exactly these six sorted members:

`camera.mcap`, `provenance.json`, `quality.json`, `robot.mcap`,
`stage-ledger.prepublish.json`, `video-keyframes.parquet`.

`checksums.json` is outside the bundle-hash preimage and is JCS of exactly:

```json
{"algorithm":"sha256","format":"robo_collector.checksums","format_version":1,"members":[{"path":"<one-of-the-six-paths>","sha256":"<sha256>","size_bytes":"<decimal>"}]}
```

It lists exactly the same six rows, byte-for-byte semantically equal to identity
members and sorted by path. It does not hash itself or `manifest.json`, avoiding
recursion. `checksums_sha256` is recorded in manifest inventory.

### 5.2 Immutable manifest inventory

The immutable manifest top-level keys are exactly `bundle_hash`,
`canonical_status`, `identity`, `inventory`, and `manifest_version`. Inventory is:

```json
{"checksums_sha256":"<sha256>","end_log_time_ns":"<decimal-or-null>","files":[{"message_count":"<decimal-or-null>","path":"<bundle-member-path>","sha256":"<sha256>","size_bytes":"<decimal>"}],"libraries":[{"name":"mcap","version":"1.4.0"},{"name":"protobuf","version":"<exact-version>"},{"name":"pyarrow","version":"<exact-version>"}],"start_log_time_ns":"<decimal-or-null>","topic_counts":[{"count":"<decimal>","topic":"<canonical-topic>"}],"total_message_count":"<decimal>"}
```

Files contain exactly the same six hashes/sizes as identity; message count is a
decimal only for the two MCAPs and `null` otherwise. `total_message_count` equals
the two MCAP message counts and the topic-count sum. Start/end are the min/max
MCAP `log_time`, both null only for an empty non-publishable candidate.
Topic counts sort by topic, files by path, libraries by name. Library versions
are exact installed versions and participate in `manifest_hash`, not
`bundle_hash`. Any disagreement among identity, checksums, decoded MCAP summary,
or inventory fails structural QC.

### 5.3 `video-keyframes.parquet` schema

Parquet has one row per keyframe and exactly these non-null columns in order:

| column | Arrow logical type |
|---|---|
| `stream_id` | `utf8` |
| `source_session_id` | `utf8` |
| `source_sequence` | `uint64` |
| `normalized_time_ns` | `int64` |
| `mcap_log_time` | `uint64` |
| `message_sequence` | `uint32` |
| `config_generation` | `uint32` |
| `pts` | `int64` |
| `timebase_num` | `uint32` |
| `timebase_den` | `uint32` |
| `codec_config_sha256` | `fixed_size_binary[32]` |

Rows sort by `(stream_id,normalized_time_ns,source_session_id,source_sequence,
message_sequence)`. `mcap_log_time == normalized_time_ns`, timebase is `1/1e9`,
and every row resolves exactly one keyframe AU in `camera.mcap`. The file uses
Parquet format `2.6`, ZSTD compression, dictionary encoding only for the two
UTF-8 columns, no statistics for the binary hash, and a row-group size of
65,536. Created-by/library variation is captured by the member hash and manifest
library inventory; logical validators compare schema/rows, not a hard-coded
Parquet byte hash.

## 6. Publication authority (gate 7)

Only `canonical_status="READY"` may enter
`canonical/versions/<bundle_hash>/` and atomically replace
`canonical/current.json`. `REVIEW`, `REJECT`, and `QUARANTINED` candidates may be
retained only under `canonical/nonready/<lowercase-status>/<bundle_hash>/` and
MUST NOT create or change a READY pointer.

Publication performs, in order: validate all required stage keys/outputs; compute
identity/checksums/inventory/manifest; require status READY; fsync token-private
staging files/directories; acquire the publication lock; revalidate fencing token
and stage generation in SQLite; install/reuse the exact version directory with a
same-filesystem atomic rename; fsync `versions/`; atomically replace and fsync
`current.json`; fsync `canonical/`; commit the matching SQLite success row. A
stale token, non-READY status, cross-device root, invalid existing target, or any
fsync/validation failure cannot mutate the pointer.

`current.json` has the exact identity-contract fields and its referenced manifest
MUST decode as READY with matching bundle/manifest hashes. Startup reconciliation
may adopt a pointer only for the exact stage key/token/generation and a fully
validated READY version. Non-READY storage is audit retention, never publication.

## 7. Legacy alignment and bundle identity (gate 6)

`legacy_rgb_v1` is explicitly prohibited from canonical MCAP bundles. It exists
only for Raw v1 compatibility replay and derived-output parity evidence. Such a
run writes under `compatibility/legacy_rgb_v1/`, never writes canonical
`camera.mcap`/`robot.mcap`, never enters bundle identity, and never updates
`canonical/current.json`.

A Raw v1 source may produce a canonical bundle only by running the full
`rgb_affine_v2` normalization/alignment contract, in which case identity records
policy `rgb_affine_v2`, version `2`, its exact alignment config hash, and the Raw
source artifact hash. Claims of legacy equivalence are limited to the separate
compatibility fixture. This removes cross-policy cache reuse and keeps one
meaning for canonical bundle v1.

## 8. Executable evidence and gate closure

| PRD §19 gate | Frozen evidence |
|---|---|
| 1. eight stage schemas/projections | §1 |
| 2. QC/hash/metric/alignment/encoder rules | §2 |
| 3. PTS/timestamps/merge/generated-event order | §3 |
| 4. checkpoint and in-progress manifest bytes | §4 |
| 5. inventory/checksums/keyframe Parquet | §5 |
| 6. legacy identity consistency | §7 (canonical prohibition) |
| 7. READY-only publication authority | §6 |

Phase 0 is complete only when tests independently reproduce the golden constants,
validate closed schemas and ordering, prove semantic retry stability, and prove a
non-READY candidate cannot mutate `current.json`. Later collection, recovery,
codec, integration, shadow-parity, performance, and rollout gates remain required
before enabling MCAP-first by default.
