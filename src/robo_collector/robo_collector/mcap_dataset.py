"""Deterministic temporal MCAP dataset indexing and worker-local cache."""
from __future__ import annotations
import fcntl, hashlib, os, shutil, tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

def sample_id(curation_id:str, episode_id:str, bundle_hash:str, anchor_row:int)->str:
 if not isinstance(anchor_row,int) or anchor_row < 0: raise ValueError("anchor_row must be non-negative")
 return hashlib.sha256("\0".join((curation_id,episode_id,bundle_hash,str(anchor_row))).encode()).hexdigest()

def shard_samples(samples:Sequence[Any], *, rank=0, world_size=1, worker_id=0, num_workers=1, seed=0, epoch=0):
 if not (0<=rank<world_size and 0<=worker_id<num_workers): raise ValueError("invalid rank/worker")
 target=rank*num_workers+worker_id; total=world_size*num_workers
 keyed=[]
 for sample in samples:
  sid=sample if isinstance(sample,str) else str(sample.get("sample_id",sample))
  h=hashlib.sha256(f"{seed}\0{epoch}\0{sid}".encode()).digest()
  keyed.append((h,sample))
 keyed.sort(key=lambda x:x[0]); return [s for h,s in keyed if int.from_bytes(h[:8],"big")%total==target]

class KeyframeIndex:
 def __init__(self, rows:Iterable[dict[str,Any]]): self.rows=sorted((dict(r) for r in rows),key=lambda r:(str(r.get("stream_id","")),str(r.get("session_id","")),int(r.get("config_generation",0)),int(r.get("timestamp_ns",r.get("time_ns",0))),int(r.get("sequence",0))))
 def seek(self, timestamp_ns:int, *, stream_id=None, session_id=None, config_generation=None, strict=False)->dict[str,Any]|None:
  out=None
  for row in self.rows:
   if stream_id is not None and row.get("stream_id") != stream_id: continue
   if session_id is not None and row.get("session_id") != session_id: continue
   if config_generation is not None and int(row.get("config_generation",0)) != config_generation: continue
   if int(row.get("timestamp_ns",row.get("time_ns",0)))<=timestamp_ns and row.get("keyframe",True): out=row
   if int(row.get("timestamp_ns",row.get("time_ns",0)))>timestamp_ns: break
  if out is None and strict: raise LookupError("no preceding IDR keyframe for requested stream/session/generation")
  return out

class McapSequenceDataset:
 def __init__(self, rows:Sequence[dict[str,Any]], *, curation_id="default", horizon=1, stride=1, seed=0, epoch=0, rank=0, world_size=1, worker_id=0, num_workers=1):
  self.rows=list(rows); self.horizon=int(horizon); self.stride=int(stride)
  if self.horizon <= 0 or self.stride <= 0: raise ValueError("horizon and stride must be positive")
  samples=[]
  for i,row in enumerate(self.rows):
   if not row.get("reference_rgb",row.get("retained",True)): continue
   eid=str(row.get("episode_id","")); bh=str(row.get("bundle_hash","")); samples.append({"sample_id":sample_id(curation_id,eid,bh,i),"episode_id":eid,"bundle_hash":bh,"anchor_row":i})
  self._samples=shard_samples(samples,rank=rank,world_size=world_size,worker_id=worker_id,num_workers=num_workers,seed=seed,epoch=epoch)
 def __len__(self): return len(self._samples)
 def __getitem__(self,index):
  s=self._samples[index]; start=s["anchor_row"]; idx=[start+j*self.stride for j in range(self.horizon) if start+j*self.stride<len(self.rows)]
  return {"sample":s,"rows":[self.rows[i] for i in idx]}
 def __iter__(self): return (self[i] for i in range(len(self)))

class LocalArtifactCache:
 def __init__(self, root:str|Path): self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
 def path(self,digest:str)->Path: return self.root/digest
 def get(self,digest:str, *, size:int|None=None)->Path|None:
  p=self.path(digest)
  if not p.is_file(): return None
  if size is not None and p.stat().st_size!=size: p.unlink(missing_ok=True); return None
  h=hashlib.sha256(p.read_bytes()).hexdigest()
  if h!=digest: p.unlink(missing_ok=True); return None
  return p
 def fetch(self,digest:str, source:str|Path, *, size:int|None=None)->Path:
  if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest): raise ValueError("digest must be lowercase SHA-256")
  existing=self.get(digest,size=size)
  if existing:return existing
  lock=self.root/(digest+".lock")
  with open(lock,"a+") as lock_file:
   fcntl.flock(lock_file, fcntl.LOCK_EX)
   existing=self.get(digest,size=size)
   if existing:return existing
   fd,tmp=tempfile.mkstemp(prefix=digest+".",suffix=".partial",dir=self.root); os.close(fd)
   try:
    shutil.copyfile(source,tmp)
    with open(tmp,"rb") as f: os.fsync(f.fileno())
    if size is not None and os.path.getsize(tmp)!=size: raise ValueError("cache size mismatch")
    if hashlib.sha256(Path(tmp).read_bytes()).hexdigest()!=digest: raise ValueError("cache hash mismatch")
    os.replace(tmp,self.path(digest))
    dir_fd=os.open(self.root, os.O_RDONLY)
    try: os.fsync(dir_fd)
    finally: os.close(dir_fd)
    return self.path(digest)
   finally:
    if os.path.exists(tmp): os.unlink(tmp)

__all__=["sample_id","shard_samples","KeyframeIndex","McapSequenceDataset","LocalArtifactCache"]
