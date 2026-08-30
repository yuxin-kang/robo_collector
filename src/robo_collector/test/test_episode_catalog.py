import json
from robo_collector.episode_catalog import EpisodeCatalog

def test_rebuild_revision_and_ready_snapshot(tmp_path):
    c = EpisodeCatalog(tmp_path / "episode_catalog.sqlite")
    manifests = [
        {"episode_id": "e2", "bundle_hash": "b2", "canonical_status": "REVIEW"},
        {"episode_id": "e1", "bundle_hash": "b1", "canonical_status": "READY"},
    ]
    assert c.rebuild(manifests) == 1
    assert [x["episode_id"] for x in c.episodes(ready_only=True)] == ["e1"]
    out = c.create_curation_snapshot(tmp_path / "curations" / "c1", curation_id="c1")
    assert out.exists()
    assert json.loads(out.with_suffix(".json").read_text())["catalog_revision"] == 1
