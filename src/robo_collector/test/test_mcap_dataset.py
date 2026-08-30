import hashlib
from robo_collector.mcap_dataset import KeyframeIndex, LocalArtifactCache, McapSequenceDataset, shard_samples

def test_shards_are_exhaustive_and_disjoint():
    values = [str(i) for i in range(31)]
    parts = [shard_samples(values, rank=r, world_size=2, worker_id=w, num_workers=2, seed=4, epoch=2) for r in range(2) for w in range(2)]
    assert set(sum(parts, [])) == set(values)
    assert sum(map(len, parts)) == len(values)

def test_dataset_anchor_window_and_keyframe_seek():
    rows = [{"episode_id":"e", "bundle_hash":"b", "reference_rgb":True, "timestamp_ns":i} for i in range(4)]
    ds = McapSequenceDataset(rows, horizon=2, stride=2)
    assert ds[0]["rows"] == [rows[0], rows[2]]
    assert KeyframeIndex([{"timestamp_ns": 0, "keyframe":True}, {"timestamp_ns": 2, "keyframe":True}]).seek(3)["timestamp_ns"] == 2

def test_cache_verifies_hash(tmp_path):
    source = tmp_path / "source"; source.write_bytes(b"hello")
    digest = hashlib.sha256(b"hello").hexdigest()
    assert LocalArtifactCache(tmp_path / "cache").fetch(digest, source).read_bytes() == b"hello"
