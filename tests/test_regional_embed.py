"""The cluster column must survive into the npz, and the npz must be the format
narrowcast reads.

`CLAUDE.md`'s first convention is "cluster, never row" -- row-level intervals have
twice produced effects in this project that failed to replicate. The regional path
is the one place in this repo that has a real cluster to carry, so these tests pin
that it arrives.
"""
import numpy as np
import pandas as pd
import pytest

from plantid.data import regional_embed as re_


@pytest.fixture
def manifest(tmp_path):
    df = pd.DataFrame({
        "species_name": ["Trillium ovatum"] * 3 + ["Rubus ursinus"] * 2,
        "cluster": ["111", "111", "222", "333", "333"],
        "local_path": [f"img/{i}.jpg" for i in range(5)],
        "licence": ["CC_BY"] * 5,
    })
    p = tmp_path / "manifest.parquet"
    df.to_parquet(p, index=False)
    return p


def _fake_embed(monkeypatch, dim=8):
    import plantid.features.pretrained as pre

    monkeypatch.setattr(pre, "load_encoder", lambda v, device=None: (None, None, "cpu"))
    monkeypatch.setattr(pre, "embed_images",
                        lambda paths, m, p, d, batch_size=64, desc="":
                        np.arange(len(paths) * dim, dtype="float32").reshape(len(paths), dim))


def test_cluster_survives_into_the_payload(monkeypatch, manifest, tmp_path):
    _fake_embed(monkeypatch)
    out = re_.embed_manifest(manifest, "mobileclip2_s2", tmp_path)
    assert list(out["cluster"]) == ["111", "111", "222", "333", "333"]
    # three plants, five photographs -- the whole point
    assert len(set(out["cluster"].tolist())) == 3
    assert len(out["label"]) == 5


def test_payload_is_narrowcast_embeddings_format(monkeypatch, manifest, tmp_path):
    """narrowcast.sources.from_embeddings requires `descriptor` and `label`, and
    reads `group`, `cluster`, `origin` when present."""
    _fake_embed(monkeypatch)
    out = re_.embed_manifest(manifest, "mobileclip2_s2", tmp_path)
    assert {"descriptor", "label"} <= set(out)
    assert out["descriptor"].shape[0] == len(out["label"]) == len(out["cluster"])
    assert out["descriptor"].dtype == np.float32


def test_group_is_written_explicitly_not_left_to_the_default(monkeypatch,
                                                             manifest, tmp_path):
    """The caller's group column wins in narrowcast by design; relying on its
    first-whitespace-token default is how the group rank silently died on a
    non-binomial domain."""
    _fake_embed(monkeypatch)
    out = re_.embed_manifest(manifest, "mobileclip2_s2", tmp_path)
    assert list(out["group"]) == ["Trillium"] * 3 + ["Rubus"] * 2


def test_labels_are_curated(monkeypatch, tmp_path):
    df = pd.DataFrame({"species_name": ["Fragaria × ananassa"],
                       "cluster": ["1"], "local_path": ["img/0.jpg"]})
    p = tmp_path / "m.parquet"
    df.to_parquet(p, index=False)
    _fake_embed(monkeypatch)
    out = re_.embed_manifest(p, "mobileclip2_s2", tmp_path)
    assert out["label"][0] == "Fragaria x ananassa"
    assert out["group"][0] == "Fragaria"


def test_a_round_trip_through_narrowcast_keeps_the_clusters(monkeypatch, manifest,
                                                            tmp_path):
    """The end-to-end guarantee: what this writes, narrowcast reads as clustered."""
    pytest.importorskip("narrowcast")
    from narrowcast import sources

    _fake_embed(monkeypatch)
    out = re_.embed_manifest(manifest, "mobileclip2_s2", tmp_path)
    npz = tmp_path / "r.npz"
    np.savez_compressed(npz, **out)
    rows = sources.from_embeddings(npz)
    assert rows.has_clusters is True
    assert len(set(rows.cluster.tolist())) == 3
    assert list(rows.group) == ["Trillium"] * 3 + ["Rubus"] * 2
