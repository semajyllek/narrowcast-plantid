"""The scores path exists so near-OOD can be measured at all.

`load_rows` sets near_ood = 0 unconditionally on the embeddings path, so the
relatives bucket -- the one a regional product lives or dies on -- is invisible
there. These pin that what this writes lets narrowcast see it.
"""
import numpy as np
import pytest

from plantid.data import regional_scores as rs


def _npz(tmp_path, name, labels, groups, clusters, dim=6, seed=0):
    rng = np.random.default_rng(seed)
    p = tmp_path / f"{name}.npz"
    np.savez(p, descriptor=rng.normal(size=(len(labels), dim)).astype("float32"),
             label=np.array(labels), group=np.array(groups),
             cluster=np.array(clusters))
    return p


@pytest.fixture
def sets(tmp_path):
    n = 12
    inl = _npz(tmp_path, "in",
               ["Rubus ursinus"] * n + ["Trillium ovatum"] * n,
               ["Rubus"] * n + ["Trillium"] * n,
               [f"a{i//2}" for i in range(n)] + [f"b{i//2}" for i in range(n)])
    near = _npz(tmp_path, "near", ["Rubus armeniacus"] * 6, ["Rubus"] * 6,
                [f"c{i//2}" for i in range(6)], seed=1)
    far = _npz(tmp_path, "far", ["Pinus ponderosa"] * 6, ["Pinus"] * 6,
               [f"d{i//2}" for i in range(6)], seed=2)
    bg = _npz(tmp_path, "bg", ["Noise sp"] * 20, ["Noise"] * 20,
              [f"e{i}" for i in range(20)], seed=3)
    return inl, near, far, bg


def test_payload_is_narrowcast_scores_format(sets):
    inl, near, far, bg = sets
    p = rs.build_scores(inl, near, far, bg)
    assert {"proba", "classes", "label"} <= set(p)
    assert p["proba"].shape[0] == len(p["label"])
    assert p["proba"].shape[1] == len(p["classes"])


def test_reject_class_is_present(sets):
    """Without __OTHER__ the posterior sums to 1 over in-list labels whatever the
    input, so an out-of-list photograph is confidently wrong and no threshold can
    separate it."""
    inl, near, far, bg = sets
    p = rs.build_scores(inl, near, far, bg)
    assert rs.OTHER in set(p["classes"].tolist())


def test_scored_rows_never_include_a_training_cluster(sets):
    """A photograph of a plant the head trained on would be scored in-sample."""
    inl, near, far, bg = sets
    p = rs.build_scores(inl, near, far, bg, seed=0)
    z = np.load(inl, allow_pickle=True)
    all_in = set(np.asarray(z["cluster"], dtype=str).tolist())
    scored_in = {c for c, l in zip(p["cluster"], p["label"])
                 if l in set(z["label"].astype(str).tolist())}
    # exactly the held-out half of the in-list clusters
    assert scored_in < all_in and len(scored_in) == len(all_in) // 2


def test_narrowcast_buckets_relatives_as_near_ood(sets):
    """The whole point: an unlisted congener must land in near_ood, not distant."""
    pytest.importorskip("narrowcast")
    from narrowcast import build as B, sources

    inl, near, far, bg = sets
    p = rs.build_scores(inl, near, far, bg)
    tmp = inl.parent / "scores.npz"
    np.savez_compressed(tmp, **p)
    rows = sources.from_scores(tmp)
    ds = B.load_scored(rows)
    assert ds.counts["near_ood"] > 0, "unlisted congener did not bucket as near_ood"
    # The far-OOD set is drawn from the same place as everything else, so
    # `build_scores` flags it regional and narrowcast buckets it `regional_ood`
    # rather than `distant_ood` -- the deployment-realistic bucket, and the one
    # the operating point is then anchored to. `distant_ood` is correctly empty:
    # nothing here is global filler.
    assert ds.counts["regional_ood"] > 0
    assert ds.counts["distant_ood"] == 0
    # Rubus armeniacus shares a genus with the listed Rubus ursinus -> near
    near_labels = {str(c) for c, b in zip(rows.label, ds.bucket) if b == "near_ood"}
    assert "Rubus armeniacus" in near_labels
    assert "Pinus ponderosa" not in near_labels


def test_far_ood_is_flagged_regional_and_near_ood_is_not(sets):
    """narrowcast cannot derive geography; this is where the claim is made. Only
    the far-OOD rows are deployment-plausible-but-unlisted — the near-OOD ones are
    congeners, which is a different bucket with a different split key."""
    inl, near, far, bg = sets
    p = rs.build_scores(inl, near, far, bg)
    assert "regional" in p and len(p["regional"]) == len(p["label"])
    flagged = {str(l) for l, r in zip(p["label"], p["regional"]) if r}
    assert "Pinus ponderosa" in flagged
    assert "Rubus armeniacus" not in flagged


def test_inputs_declaring_different_encoders_are_refused(sets, tmp_path):
    """The point where separately embedded pools are combined is the point where a
    Core ML pool would meet a torch one. Declaration is the only check that sees
    it: the geometric one catches 0 of 21 such pairs."""
    inl, near, far, bg = sets
    def tagged(src, enc, name):
        z = dict(np.load(src, allow_pickle=True))
        z["encoder"] = enc
        out = tmp_path / name
        np.savez(out, **z)
        return out
    a = tagged(inl, "plantclef24", "a.npz")
    b = tagged(near, "plantclef24+coreml:pc24_cml4", "b.npz")
    with pytest.raises(SystemExit, match="different encoders"):
        rs.build_scores(a, b, far, bg)
    # and agreeing declarations pass, carrying through to the output
    c = tagged(near, "plantclef24", "c.npz")
    out = rs.build_scores(a, c, far, bg)
    assert out["encoder"] == "plantclef24"


def test_the_same_file_in_two_roles_is_refused(sets):
    """Its rows would be scored twice, doubling that bucket and putting one
    observation on both sides of narrowcast's split."""
    inl, near, far, bg = sets
    with pytest.raises(SystemExit, match="more than one role"):
        rs.build_scores(inl, far, far, bg)
