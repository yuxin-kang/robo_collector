"""Rebuildable catalog and immutable curation snapshots for canonical episodes."""
from __future__ import annotations
import hashlib, json, os, sqlite3, tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

_SCHEMA = """CREATE TABLE IF NOT EXISTS episodes (
 episode_id TEXT NOT NULL, bundle_hash TEXT NOT NULL, canonical_status TEXT NOT NULL,
 manifest_path TEXT NOT NULL, manifest_json TEXT NOT NULL, revision INTEGER NOT NULL,
 PRIMARY KEY (episode_id, bundle_hash));
CREATE TABLE IF NOT EXISTS catalog_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"""

def _hash(value: Any) -> str:
 return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",",":"), ensure_ascii=False).encode()).hexdigest()

class EpisodeCatalog:
 def __init__(self, path: str|Path):
  self.path=Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
  with sqlite3.connect(self.path) as db: db.executescript(_SCHEMA)
 @property
 def revision(self)->int:
  with sqlite3.connect(self.path) as db:
   row=db.execute("SELECT value FROM catalog_meta WHERE key='revision'").fetchone()
  return int(row[0]) if row else 0
 def rebuild(self, manifests: Iterable[str|Path|Mapping[str,Any]])->int:
  rows=[]
  for item in manifests:
   if isinstance(item, Mapping): m=dict(item); p=str(m.get("manifest_path", ""))
   else:
    p=str(item); m=json.loads(Path(item).read_text())
   status=m.get("canonical_status", m.get("status", "")); ident=m.get("identity", {})
   episode_id=str(m.get("episode_id", ident.get("episode_id", "")))
   bundle_hash=str(m.get("bundle_hash", ident.get("bundle_hash", "")))
   if episode_id and bundle_hash and status in {"READY", "REVIEW", "REJECT", "QUARANTINED"}: rows.append((episode_id,bundle_hash,status,p,json.dumps(m,sort_keys=True,separators=(",",":"))))
  rows.sort(key=lambda x:(x[0],x[1]))
  with sqlite3.connect(self.path) as db:
   old=self.revision
   existing=db.execute("SELECT episode_id,bundle_hash,canonical_status,manifest_path,manifest_json FROM episodes ORDER BY episode_id,bundle_hash").fetchall()
   if existing == rows: return old
   new=old+1
   db.execute("DELETE FROM episodes")
   db.executemany("INSERT INTO episodes VALUES (?,?,?,?,?,?)", [(a,b,c,d,e,new) for a,b,c,d,e in rows])
   db.execute("INSERT INTO catalog_meta(key,value) VALUES('revision',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(str(new),)); db.commit()
  return new
 def episodes(self, *, ready_only=False)->list[dict[str,Any]]:
  q="SELECT manifest_json FROM episodes" + (" WHERE canonical_status='READY'" if ready_only else "") + " ORDER BY episode_id,bundle_hash"
  with sqlite3.connect(self.path) as db: return [json.loads(x[0]) for x in db.execute(q)]
 def create_curation_snapshot(self, destination: str|Path, *, curation_id="default", ready_only=True)->Path:
  rows=self.episodes(ready_only=ready_only); destination=Path(destination)
  destination.mkdir(parents=True,exist_ok=True); target=destination/"manifest.parquet"
  metadata={"curation_id":curation_id,"catalog_revision":self.revision,"query":"READY" if ready_only else "ALL","row_count":len(rows),"rows_hash":_hash(rows)}
  if target.exists(): raise FileExistsError(f"immutable curation already exists: {target}")
  fd,tmp=tempfile.mkstemp(prefix=".manifest.",suffix=".partial",dir=destination); os.close(fd)
  try:
   try:
    import pyarrow as pa, pyarrow.parquet as pq
    table=pa.Table.from_pylist([{"episode_id":str(r.get("episode_id",r.get("identity",{}).get("episode_id",""))),"bundle_hash":str(r.get("bundle_hash",r.get("identity",{}).get("bundle_hash",""))),"manifest_json":json.dumps(r,sort_keys=True,separators=(",",":"))} for r in rows])
    table=table.replace_schema_metadata({k:str(v).encode() for k,v in metadata.items()}); pq.write_table(table,tmp)
   except ImportError as exc:
    raise RuntimeError("pyarrow is required for immutable curation Parquet snapshots") from exc
   os.replace(tmp,target)
   meta=target.with_suffix(".json"); meta.write_text(json.dumps(metadata,sort_keys=True,indent=2)+"\n")
   return target
  finally:
   if os.path.exists(tmp): os.unlink(tmp)

__all__=["EpisodeCatalog"]
