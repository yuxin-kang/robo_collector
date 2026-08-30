"""Deterministic, replayable primitives for the canonical MCAP ingestion DAG.

This module deliberately has no ROS or Airflow dependency.  It owns the local
SQLite stage authority and fenced READY publication; orchestrators are thin
callers of these ordinary Python APIs.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Self

from . import mcap_contract

_PREPUBLICATION = frozenset(mcap_contract.PREPUBLICATION_STAGE_ORDER)


class IngestionError(RuntimeError):
    """Base class for deterministic ingestion failures."""


class StageClaimError(IngestionError):
    """Raised when a live owner or fencing mismatch rejects a mutation."""


@dataclass(frozen=True)
class StageClaim:
    stage_key_sha256: str
    stage_name: str
    owner: str
    fencing_token: int
    generation: int
    attempt: int
    lease_expires_ns: int
    reused: bool = False


class IngestionLedger:
    """Single-host SQLite authority for stage claims and fenced completion."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS stages (
              stage_key_sha256 TEXT PRIMARY KEY,
              stage_name TEXT NOT NULL,
              stage_key_json BLOB NOT NULL,
              status TEXT NOT NULL CHECK(status IN ('RUNNING','SUCCEEDED','FAILED')),
              owner TEXT,
              lease_expires_ns INTEGER,
              fencing_token INTEGER NOT NULL,
              generation INTEGER NOT NULL DEFAULT 0,
              attempt INTEGER NOT NULL,
              output_hashes_json BLOB,
              metrics_json BLOB,
              error TEXT
            )
            """
        )

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def claim(
        self,
        stage_key: Mapping[str, Any],
        *,
        owner: str,
        lease_duration_ns: int,
        now_ns: int | None = None,
    ) -> StageClaim:
        if not owner or lease_duration_ns <= 0:
            raise ValueError("owner and positive lease_duration_ns are required")
        key = dict(stage_key)
        key_hash = mcap_contract.stage_key_hash(key)
        stage_name = str(key["stage_name"])
        encoded = mcap_contract.canonical_json_bytes(key)
        now = time.time_ns() if now_ns is None else int(now_ns)
        expires = now + lease_duration_ns
        self._db.execute("BEGIN IMMEDIATE")
        try:
            row = self._db.execute(
                "SELECT * FROM stages WHERE stage_key_sha256=?", (key_hash,)
            ).fetchone()
            if row is None:
                token = attempt = 1
                generation = 0
                self._db.execute(
                    """INSERT INTO stages (
                       stage_key_sha256, stage_name, stage_key_json, status,
                       owner, lease_expires_ns, fencing_token, generation,
                       attempt, output_hashes_json, metrics_json, error
                       ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        key_hash,
                        stage_name,
                        encoded,
                        "RUNNING",
                        owner,
                        expires,
                        token,
                        generation,
                        attempt,
                        None,
                        None,
                        None,
                    ),
                )
                reused = False
            elif bytes(row["stage_key_json"]) != encoded:
                raise StageClaimError("stage key hash collision")
            elif row["status"] == "SUCCEEDED":
                token, attempt, reused = row["fencing_token"], row["attempt"], True
                generation = row["generation"]
                owner = row["owner"] or owner
                expires = row["lease_expires_ns"] or 0
            elif row["status"] == "RUNNING" and row["lease_expires_ns"] > now:
                raise StageClaimError("stage already has an unexpired owner")
            else:
                token = int(row["fencing_token"]) + 1
                generation = int(row["generation"])
                attempt = int(row["attempt"]) + 1
                reused = False
                self._db.execute(
                    """UPDATE stages SET status='RUNNING', owner=?, lease_expires_ns=?,
                       fencing_token=?, attempt=?, output_hashes_json=NULL,
                       metrics_json=NULL, error=NULL WHERE stage_key_sha256=?""",
                    (owner, expires, token, attempt, key_hash),
                )
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        return StageClaim(
            key_hash,
            stage_name,
            owner,
            token,
            generation,
            attempt,
            expires,
            reused,
        )

    def heartbeat(
        self, claim: StageClaim, *, lease_duration_ns: int, now_ns: int | None = None
    ) -> StageClaim:
        now = time.time_ns() if now_ns is None else int(now_ns)
        expires = now + lease_duration_ns
        updated = self._db.execute(
            """UPDATE stages SET lease_expires_ns=? WHERE stage_key_sha256=?
               AND status='RUNNING' AND owner=? AND fencing_token=?
               AND lease_expires_ns>=?""",
            (expires, claim.stage_key_sha256, claim.owner, claim.fencing_token, now),
        ).rowcount
        if updated != 1:
            raise StageClaimError("heartbeat rejected by fencing authority")
        return replace(claim, lease_expires_ns=expires)

    def complete(
        self,
        claim: StageClaim,
        *,
        output_hashes: Iterable[Mapping[str, str]],
        metrics: Mapping[str, Any] | None = None,
        now_ns: int | None = None,
    ) -> None:
        now = time.time_ns() if now_ns is None else int(now_ns)
        outputs = list(output_hashes)
        encoded_outputs = mcap_contract.canonical_json_bytes(outputs)
        encoded_metrics = mcap_contract.canonical_json_bytes(dict(metrics or {}))
        updated = self._db.execute(
            """UPDATE stages SET status='SUCCEEDED', output_hashes_json=?,
               metrics_json=?, error=NULL, generation=generation+1
               WHERE stage_key_sha256=? AND status='RUNNING' AND owner=?
               AND fencing_token=? AND generation=? AND lease_expires_ns>=?""",
            (
                encoded_outputs,
                encoded_metrics,
                claim.stage_key_sha256,
                claim.owner,
                claim.fencing_token,
                claim.generation,
                now,
            ),
        ).rowcount
        if updated != 1:
            raise StageClaimError("completion rejected by fencing authority")

    def fail(self, claim: StageClaim, error: str) -> None:
        updated = self._db.execute(
            """UPDATE stages SET status='FAILED', error=? WHERE stage_key_sha256=?
               AND status='RUNNING' AND owner=? AND fencing_token=?""",
            (str(error), claim.stage_key_sha256, claim.owner, claim.fencing_token),
        ).rowcount
        if updated != 1:
            raise StageClaimError("failure rejected by fencing authority")

    def install_stage_outputs(
        self,
        claim: StageClaim,
        *,
        staging_dir: str | Path,
        final_dir: str | Path,
        output_hashes: Iterable[Mapping[str, str]],
        metrics: Mapping[str, Any] | None = None,
        now_ns: int | None = None,
    ) -> Path:
        """Fsync and atomically install one immutable stage output directory."""

        source = Path(staging_dir)
        target = Path(final_dir)
        outputs = tuple(dict(item) for item in output_hashes)
        expected_paths = tuple(
            sorted(
                mcap_contract.STAGE_REGISTRY[claim.stage_name].output_paths,
                key=str.encode,
            )
        )
        if (
            tuple(sorted((str(item["path"]) for item in outputs), key=str.encode))
            != expected_paths
        ):
            raise IngestionError("stage outputs do not match the registered contract")
        if not source.is_dir() or source.is_symlink():
            raise IngestionError("staging output must be a regular directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.stat().st_dev != target.parent.stat().st_dev:
            raise IngestionError("stage install requires one filesystem")
        for item in outputs:
            path = source / str(item["path"])
            if not path.is_file() or path.is_symlink():
                raise IngestionError(f"missing immutable stage output: {item['path']}")
            digest = _sha256_file(path)
            if (
                digest != item["sha256"]
                or str(path.stat().st_size) != item["size_bytes"]
            ):
                raise IngestionError(f"stage output hash mismatch: {item['path']}")
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        _fsync_directory(source)
        if target.exists():
            _verify_installed_outputs(target, outputs)
        else:
            os.rename(source, target)
            _fsync_directory(target.parent)
        self.complete(claim, output_hashes=outputs, metrics=metrics, now_ns=now_ns)
        return target

    def stage_evidence(self) -> dict[str, Any]:
        rows = self._db.execute(
            "SELECT * FROM stages WHERE status='SUCCEEDED' ORDER BY stage_name"
        ).fetchall()
        evidence = []
        for row in rows:
            if row["stage_name"] not in _PREPUBLICATION:
                continue
            key = json.loads(bytes(row["stage_key_json"]))
            evidence.append(
                {
                    **key,
                    "stage_key_sha256": row["stage_key_sha256"],
                    "output_hashes": json.loads(bytes(row["output_hashes_json"])),
                }
            )
        return mcap_contract.project_stage_evidence(evidence)

    def write_prepublish_snapshot(self, path: str | Path) -> str:
        target = Path(path)
        payload = mcap_contract.canonical_json_bytes(self.stage_evidence())
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        with temp.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
        descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return hashlib.sha256(payload).hexdigest()

    def publish_ready(
        self,
        claim: StageClaim,
        *,
        staging_dir: str | Path,
        canonical_dir: str | Path,
        manifest: Mapping[str, Any],
        pointer: Mapping[str, Any],
        now_ns: int | None = None,
    ) -> Path:
        """Install a READY candidate while holding DB and filesystem authority.

        A failed pointer update can leave only an immutable orphan version.  It
        can never mark the publication stage successful or select a stale
        candidate, matching the frozen crash-reconciliation boundary.
        """

        if claim.stage_name != "publish_canonical":
            raise StageClaimError("only a publish_canonical claim may publish")
        now = time.time_ns() if now_ns is None else int(now_ns)
        root = Path(canonical_dir)
        root.mkdir(parents=True, exist_ok=True)
        lock_path = root.parent / ".publish.lock"
        lock_path.touch(exist_ok=True)
        from .mcap_episode import install_ready_bundle, install_ready_pointer

        with lock_path.open("rb") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    "SELECT * FROM stages WHERE stage_key_sha256=?",
                    (claim.stage_key_sha256,),
                ).fetchone()
                if (
                    row is None
                    or row["status"] != "RUNNING"
                    or row["owner"] != claim.owner
                    or row["fencing_token"] != claim.fencing_token
                    or row["generation"] != claim.generation
                    or row["lease_expires_ns"] < now
                ):
                    raise StageClaimError("publication rejected by fencing authority")
                succeeded = {
                    item["stage_name"]
                    for item in self._db.execute(
                        "SELECT stage_name FROM stages WHERE status='SUCCEEDED'"
                    )
                }
                if not _PREPUBLICATION.issubset(succeeded):
                    raise StageClaimError("publication prerequisites are incomplete")
                if (
                    pointer.get("publisher_stage_key") != claim.stage_key_sha256
                    or pointer.get("publisher_fencing_token")
                    != str(claim.fencing_token)
                    or pointer.get("publication_generation")
                    != str(claim.generation + 1)
                ):
                    raise StageClaimError("pointer authority fields do not match claim")
                target = install_ready_bundle(
                    staging_dir, root, manifest, prevalidated_authority=True
                )
                install_ready_pointer(
                    root, manifest, pointer, prevalidated_authority=True
                )
                updated = self._db.execute(
                    """UPDATE stages SET status='SUCCEEDED', generation=generation+1
                       WHERE stage_key_sha256=? AND status='RUNNING' AND owner=?
                       AND fencing_token=? AND generation=?""",
                    (
                        claim.stage_key_sha256,
                        claim.owner,
                        claim.fencing_token,
                        claim.generation,
                    ),
                ).rowcount
                if updated != 1:
                    raise StageClaimError("publication completion lost authority")
                self._db.execute("COMMIT")
                return target
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_installed_outputs(
    directory: Path, output_hashes: Iterable[Mapping[str, str]]
) -> None:
    for item in output_hashes:
        path = directory / str(item["path"])
        if (
            not path.is_file()
            or path.is_symlink()
            or _sha256_file(path) != item["sha256"]
            or str(path.stat().st_size) != item["size_bytes"]
        ):
            raise IngestionError(f"immutable stage output collision: {item['path']}")
