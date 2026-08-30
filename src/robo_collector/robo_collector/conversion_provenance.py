"""Shared provenance and raw-source helpers for dataset converters."""

from __future__ import annotations

import hashlib
import json
import shutil
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from .raw_materializer import MaterializationConfig, RawEpisodeMaterializer


PROVENANCE_SCHEMA = "robo_collector.conversion_provenance.v1"


def is_raw_episode(path: Path) -> bool:
    """Return whether ``path`` is a sealed Raw v1 Episode."""
    return (
        (path / "manifest.json").is_file()
        and not is_mcap_episode(path)
        and not (path / "meta" / "info.json").is_file()
    )


def is_mcap_episode(path: Path) -> bool:
    """Return whether ``path`` is an explicit sealed MCAP landing manifest."""
    try:
        value = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(value, Mapping) and value.get("format") == "robo_collector.mcap_landing"


def validate_raw_source_ready(source_episode: Path) -> Mapping[str, Any]:
    """Read and enforce the durable raw/QC gate used by every converter path."""
    if is_mcap_episode(source_episode):
        from .mcap_episode import McapEpisodeReader
        reader = McapEpisodeReader(source_episode)
        reader.validate()
        if reader.manifest.get("status") not in {"RAW_CLOSED", "READY"}:
            raise ValueError("MCAP source manifest is not publishable")
        return reader.manifest
    try:
        manifest = json.loads(
            (source_episode / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read raw source manifest: {source_episode}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("raw source manifest must be an object")
    if manifest.get("status") != "READY":
        raise ValueError(
            "cannot convert raw source with status "
            f"{manifest.get('status')!r}; source must be READY"
        )
    if manifest.get("source_scope") != "camera_capture":
        raise ValueError(
            "cannot convert transport_observed raw source; attach a complete "
            "camera_capture source first"
        )
    try:
        quality = json.loads(
            (source_episode / "quality.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("raw source quality report is missing or unreadable") from exc
    if not isinstance(quality, Mapping) or quality.get("status") != "READY":
        quality_status = quality.get("status") if isinstance(quality, Mapping) else None
        raise ValueError(
            "raw source quality status "
            f"{quality_status!r} is not publishable"
        )
    return manifest


def _raw_source_is_publishable(source_episode: Path) -> bool:
    """Check the durable raw/QC gate before considering output reuse.

    Conversion outputs can outlive the source quality decision.  Reuse must
    therefore re-read the source state on every invocation instead of trusting
    only the old output provenance and artifact hashes.
    """
    try:
        validate_raw_source_ready(source_episode)
    except (OSError, TypeError, ValueError):
        return False
    return True


def materialize_raw_source(
    source_episode: Path,
    working_root: Path,
    *,
    default_fps: int | None = None,
    profile: str = "default",
    action_source: str = "aligned_target_pos",
) -> tuple[Path, Path]:
    """Replay a Raw Episode into a temporary LeRobot dataset for conversion."""

    if is_mcap_episode(source_episode):
        from .mcap_episode import McapEpisodeReader
        reader = McapEpisodeReader(source_episode)
        reader.validate()
        manifest = reader.manifest
        metadata = manifest.get("metadata") if isinstance(manifest.get("metadata"), Mapping) else {}
    else:
        manifest = validate_raw_source_ready(source_episode)
        metadata = manifest.get("metadata")

    metadata = metadata if isinstance(metadata, Mapping) else {}
    raw_fps = manifest.get("fps", metadata.get("fps", default_fps))
    try:
        fps = int(raw_fps)
    except (TypeError, ValueError) as exc:
        raise ValueError("raw source is missing a positive fps") from exc
    if fps <= 0:
        raise ValueError("raw source is missing a positive fps")

    if is_mcap_episode(source_episode):
        streams = list(getattr(reader, "camera_streams", ()))
    else:
        streams = [path.name for path in sorted((source_episode / "camera").iterdir()) if path.is_dir() and any(path.glob("*.raw"))] if (source_episode / "camera").is_dir() else []
    if not streams:
        raise ValueError("raw source has no camera streams")

    temporary_root = working_root / f".raw-source-{uuid4().hex}"
    try:
        temporary_root.mkdir(parents=True, exist_ok=False)
        # RawEpisodeMaterializer writes quality/provenance and (when enabled)
        # job state relative to its input. Replay a private copy so even a
        # future materializer change cannot mutate the published raw source.
        ephemeral_source = temporary_root / "source"
        if is_mcap_episode(source_episode):
            ephemeral_source = source_episode
        else:
            shutil.copytree(source_episode, ephemeral_source)
        materialization_root = temporary_root / "materialized"
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    dataset_name = "materialized"
    stored_conversion_config = metadata.get("conversion_config")
    stored_conversion_config = (
        stored_conversion_config
        if isinstance(stored_conversion_config, Mapping)
        else {}
    )
    writer_factory = None
    if profile == "gr00t":
        from .field_config import FieldSelection
        from .lerobot_dataset import LeRobotV21Writer

        if action_source not in {"aligned_target_pos", "policy_action", "joint_position"}:
            raise ValueError(f"unsupported GR00T action source: {action_source}")
        writer_selection = FieldSelection(
            target=("joint_position",) if action_source == "joint_position" else ("aligned_target_pos",),
            state=(
                "relative_ori_6d",
                "motion_anchor_lin_vel_b",
                "motion_anchor_ang_vel_b",
                "ang_vel_history",
                "gravity_history",
                "joint_pos_rel_history",
                "joint_vel_history",
                "action_history",
            ),
            include_policy_action=action_source == "policy_action",
        )

        def writer_factory(output_root, name, output_fps, camera_streams):
            return LeRobotV21Writer(
                output_root,
                dataset_name=name,
                fps=output_fps,
                camera_keys=[f"observation.images.{stream}" for stream in camera_streams],
                field_selection=writer_selection,
            )

    elif profile != "default":
        raise ValueError(f"unsupported raw materialization profile: {profile}")
    try:
        result = RawEpisodeMaterializer(
            MaterializationConfig(
                output_root=materialization_root,
                dataset_name=dataset_name,
                fps=fps,
                camera_streams=tuple(streams),
                alignment_policy="strict",
                max_alignment_residual_sec=float(
                    metadata.get("max_alignment_residual_sec", 0.1)
                ),
                persist_job=False,
                # This dataset is an isolated conversion input, not the
                # collector's shared publication target.  Preserve REVIEW /
                # REJECT artifacts in the temporary tree so the converter can
                # report and inspect them instead of moving them outside the
                # tree that is removed below.
                publish_non_ready=True,
                require_complete_capture=True,
                field_selection=(
                    dict(stored_conversion_config["field_selection"])
                    if isinstance(
                        stored_conversion_config.get("field_selection"), Mapping
                    )
                    else None
                ),
                quality_thresholds=(
                    dict(stored_conversion_config["quality_thresholds"])
                    if isinstance(
                        stored_conversion_config.get("quality_thresholds"), Mapping
                    )
                    else None
                ),
            ),
            writer_factory=writer_factory,
        ).materialize(ephemeral_source, progress_callback=None)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    if result.output_dataset is None:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise ValueError("raw source materialization did not publish a dataset")
    if result.quality_status != "READY":
        status = result.quality_status
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise ValueError(f"raw source QC status {status!r} is not publishable")
    return temporary_root, result.output_dataset


def write_conversion_provenance(
    staging_output: Path,
    source_dataset: Path,
    *,
    converter_version: str,
    output_schema_version: str,
    conversion_config: Mapping[str, Any],
    action_source: str,
    state_source: str,
    selection_policy: str,
) -> Path:
    """Write converter provenance without changing source data."""

    source = _source_provenance(source_dataset)
    config_hash = _canonical_hash(conversion_config)
    metadata = {
        "schema": PROVENANCE_SCHEMA,
        "source_episode_id": source["source_episode_id"],
        "source_episode_ids": source["source_episode_ids"],
        "source_manifest_hash": source["source_manifest_hash"],
        "source_manifest_hashes": source["source_manifest_hashes"],
        "source_manifest_type": source["source_manifest_type"],
        "source_bundle_hash": source.get("source_bundle_hash"),
        "source_dataset": str(source_dataset),
        "converter_version": converter_version,
        "conversion_config_hash": config_hash,
        "output_schema_version": output_schema_version,
        "action_source": action_source,
        "state_source": state_source,
        "selection_policy": selection_policy,
        "source_provenance": source.get("raw_provenance", {}),
        "episode_provenance": source.get("episode_provenance", []),
    }
    target = staging_output / "meta" / "raw_provenance.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata["output_artifacts"] = _artifact_evidence(staging_output)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".raw_provenance.", dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with temporary.open("rb") as stream: os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try: os.fsync(directory_fd)
        finally: os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _artifact_evidence(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "raw_provenance.json":
            continue
        digest = _sha256_file(path)
        result[str(path.relative_to(root))] = {"size": path.stat().st_size, "sha256": digest}
    return result


def _sha256_file(path: Path) -> str:
    """Hash an artifact without materializing the whole file in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_provenance(source_dataset: Path) -> dict[str, Any]:
    manifest_path = source_dataset / "manifest.json"
    if manifest_path.is_file() and not (source_dataset / "meta" / "info.json").is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read raw source manifest: {source_dataset}") from exc
        if not isinstance(manifest, Mapping):
            raise ValueError("raw source manifest must be an object")
        identity = manifest.get("identity")
        episode_id = manifest.get("episode_id") or (identity.get("episode_id") if isinstance(identity, Mapping) else None) or source_dataset.name
        source_hash = manifest.get("raw_manifest_hash", manifest.get("manifest_hash"))
        if not isinstance(source_hash, str) or not source_hash:
            source_hash = hashlib.sha256(
                json.dumps(dict(manifest), sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        return {
            "source_episode_id": str(episode_id),
            "source_episode_ids": [str(episode_id)],
            "source_manifest_hash": source_hash,
            "source_manifest_hashes": [source_hash],
            "source_manifest_type": ("mcap_manifest" if manifest.get("format") == "robo_collector.mcap_landing" else "raw_episode"),
            "source_bundle_hash": (manifest.get("bundle_hash") or manifest.get("identity", {}).get("bundle_hash")) if isinstance(manifest.get("identity"), Mapping) else manifest.get("bundle_hash"),
            "raw_provenance": {},
            "episode_provenance": [{
                "source_episode_id": str(episode_id),
                "source_manifest_hash": source_hash,
            }],
        }
    raw_path = source_dataset / "meta" / "raw_provenance.json"
    raw_provenance: dict[str, Any] = {}
    if raw_path.is_file():
        try:
            value = json.loads(raw_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            value = {}
        if isinstance(value, Mapping):
            raw_provenance = dict(value)

    episode_rows = _episode_rows(source_dataset / "meta" / "episodes.jsonl")
    episode_ids = [
        str(row["source_episode_id"])
        for row in episode_rows
        if row.get("source_episode_id")
    ]
    if not episode_ids:
        episode_ids = [
            str(row["episode_id"])
            for row in episode_rows
            if row.get("episode_id")
        ]
    episode_provenance = [
        item for row in episode_rows
        if (item := episode_provenance_from_row(row))
    ]
    if not episode_provenance:
        episode_provenance = _provenance_entries(
            raw_provenance.get("episode_provenance")
        )
        if not episode_ids:
            episode_ids = [
                str(item["source_episode_id"])
                for item in episode_provenance
                if item.get("source_episode_id")
            ]
    row_hashes = [
        str(item["source_manifest_hash"])
        for item in episode_provenance
        if item.get("source_manifest_hash")
    ]
    inherited_id = raw_provenance.get("source_episode_id")
    if inherited_id and not episode_ids:
        episode_ids = [str(inherited_id)]
    inherited_ids = raw_provenance.get("source_episode_ids")
    if isinstance(inherited_ids, list) and not episode_ids:
        episode_ids = [str(value) for value in inherited_ids if value]
    # Per-episode rows are authoritative.  The aggregate raw_provenance.json
    # is an index kept for compatibility and its singleton fields necessarily
    # become ambiguous once more than one Episode is materialized.
    if row_hashes:
        source_hashes = row_hashes
        source_hash = (
            row_hashes[0]
            if len(row_hashes) == 1
            else _canonical_hash({"source_manifest_hashes": row_hashes})
        )
        source_type = "raw_episode"
    else:
        source_hash = raw_provenance.get("source_manifest_hash")
        if not isinstance(source_hash, str) or not source_hash:
            source_hash = _lerobot_metadata_hash(source_dataset)
            source_type = "lerobot_metadata"
        else:
            source_type = "raw_episode"
        source_hashes = raw_provenance.get("source_manifest_hashes")
        if not isinstance(source_hashes, list) or not all(
            isinstance(value, str) and value for value in source_hashes
        ):
            source_hashes = [source_hash]
    return {
        # Keep the required singular field populated for legacy consumers;
        # source_episode_ids is the authoritative complete set for datasets
        # containing more than one episode.
        "source_episode_id": episode_ids[0] if episode_ids else None,
        "source_episode_ids": episode_ids,
        "source_manifest_hash": source_hash,
        "source_manifest_hashes": [str(value) for value in source_hashes],
        "source_manifest_type": source_type,
        "raw_provenance": raw_provenance,
        "episode_provenance": episode_provenance,
    }


def _provenance_entries(value: Any) -> list[dict[str, Any]]:
    """Normalize legacy list and current per-Episode mapping documents."""
    if isinstance(value, Mapping):
        return [
            dict(item)
            for item in value.values()
            if isinstance(item, Mapping)
        ]
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def reusable_conversion(
    output_dataset: Path,
    source_dataset: Path,
    *,
    converter_version: str,
    output_schema_version: str,
    conversion_config: Mapping[str, Any],
) -> bool:
    """Return true only for an intact output matching this exact conversion."""
    provenance_path = output_dataset / "meta" / "raw_provenance.json"
    try:
        if not output_dataset.is_dir():
            return False
        if is_raw_episode(source_dataset) and not _raw_source_is_publishable(
            source_dataset
        ):
            # A previously generated output is never reusable after its raw
            # source moves to REVIEW/REJECT or loses its QC report.
            return False
        metadata = json.loads(provenance_path.read_text(encoding="utf-8"))
        source = _source_provenance(source_dataset)
        if not isinstance(metadata, Mapping):
            return False
        if metadata.get("converter_version") != converter_version:
            return False
        if metadata.get("output_schema_version") != output_schema_version:
            return False
        if metadata.get("source_episode_id") != source["source_episode_id"]:
            return False
        if metadata.get("source_episode_ids") != source["source_episode_ids"]:
            return False
        if metadata.get("source_manifest_hash") != source["source_manifest_hash"]:
            return False
        if metadata.get("source_manifest_hashes") != source["source_manifest_hashes"]:
            return False
        if metadata.get("conversion_config_hash") != _canonical_hash(conversion_config):
            return False
        evidence = metadata.get("output_artifacts")
        if not isinstance(evidence, Mapping) or not evidence:
            return False
        actual_files = {
            path.relative_to(output_dataset).as_posix()
            for path in output_dataset.rglob("*")
            if path.is_file() and path != provenance_path
        }
        evidence_files: set[str] = set()
        for relative, item in evidence.items():
            safe_relative = _safe_artifact_relative(relative)
            if safe_relative is None or not isinstance(item, Mapping):
                return False
            relative_name = safe_relative.as_posix()
            evidence_files.add(relative_name)
            path = output_dataset / safe_relative
            if not path.is_file():
                return False
            expected_size = item.get("size")
            expected_hash = item.get("sha256")
            if (
                isinstance(expected_size, bool)
                or not isinstance(expected_size, int)
                or expected_size < 0
                or not isinstance(expected_hash, str)
                or len(expected_hash) != 64
            ):
                return False
            if path.stat().st_size != expected_size:
                return False
            if _sha256_file(path) != expected_hash:
                return False
        return evidence_files == actual_files
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return False


def _safe_artifact_relative(value: Any) -> Path | None:
    """Validate a provenance file name before joining it to an output root."""
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        return None
    return path


def _episode_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, Mapping):
            result.append(dict(value))
    return result


def _episode_ids(path: Path) -> list[str]:
    return [
        str(row["episode_id"])
        for row in _episode_rows(path)
        if row.get("episode_id")
    ]


def episode_provenance_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Extract per-episode source provenance without guessing legacy fields."""
    nested = row.get("source_provenance")
    value = dict(nested) if isinstance(nested, Mapping) else {}
    for key in (
        "source_episode_id",
        "source_manifest_hash",
        "conversion_config_hash",
        "output_schema_version",
        "selection_policy",
    ):
        if key in row:
            value[key] = row[key]
    if not value.get("source_episode_id") or not value.get("source_manifest_hash"):
        return {}
    return value


def _lerobot_metadata_hash(dataset: Path) -> str:
    files: dict[str, str] = {}
    for path in sorted((dataset / "meta").glob("*.json*")):
        if path.is_file():
            files[str(path.relative_to(dataset))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return _canonical_hash(files)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "PROVENANCE_SCHEMA",
    "is_raw_episode",
    "materialize_raw_source",
    "reusable_conversion",
    "write_conversion_provenance",
    "episode_provenance_from_row",
]
