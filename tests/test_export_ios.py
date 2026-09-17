"""The JSON must be a complete specification of the cascade.

The test that matters is parity: a reimplementation reading **only
`bundle.json`** -- no `narrowcast` import, the way Swift will -- must produce the
same rank and the same answer as `narrowcast.predict.Bundle` on every row. If it
cannot, the JSON is missing a field, and the app would quietly answer differently
from the card that describes it.
"""

import base64
import json

import numpy as np
import pytest

from plantid.deploy.export_ios import build_payload

PRE = {"side": 518, "interpolation": "InterpolationMode.BICUBIC",
       "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}


def _arr(d):
    return np.frombuffer(base64.b64decode(d["b64"]), dtype="<f4").reshape(d["shape"])


def cascade_from_json(payload: dict, X: np.ndarray) -> list[dict]:
    """The Swift port, in Python. Reads nothing but the payload.

    Every line here is a line the app has to have, and the four that are easy to
    get wrong are marked.
    """
    coef, intercept = _arr(payload["coef"]).astype("float64"), \
        _arr(payload["intercept"]).astype("float64")
    classes = np.array(payload["classes"])
    groups = payload["groups"]
    t = payload["thresholds"]
    never = set(payload["never_answer"])

    X = np.asarray(X, dtype="float64")
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    z = X @ coef.T + intercept
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    proba = e / e.sum(axis=1, keepdims=True)      # (1) softmax over ALL classes...

    reject = payload["reject_class"]
    keep = classes != reject if reject else np.ones(len(classes), bool)
    cata = proba[:, keep]                          # (2) ...and only then mask
    names = classes[keep]

    # (3) the group map comes from the bundle, never from label.split()[0]
    gnames = np.array([groups.get(str(c), str(c)) for c in names])
    ug = np.unique(gnames)
    gmat = np.stack([(gnames == g).astype(float) for g in ug])
    gscore = cata @ gmat.T

    label_conf, group_conf = cata.max(1), gscore.max(1)
    novelty = cata.sum(1) / np.clip(proba.sum(1), 1e-12, None)
    pred_label = names[cata.argmax(1)]
    pred_group = ug[gscore.argmax(1)]

    out = []
    for i in range(len(X)):
        rank = "label"
        if label_conf[i] < t["t_label"]:
            rank = "group"
        if group_conf[i] < t["t_group"]:
            rank = "decline"
        # (4) the gate declines, and is applied after the other two
        if t["t_novel"] is not None and novelty[i] < t["t_novel"]:
            rank = "decline"
        if rank == "label" and str(pred_label[i]) in never:
            rank = "decline"
        elif rank == "group" and never:
            members = {str(c) for c in names[gnames == pred_group[i]]}
            if members and members <= never:
                rank = "decline"
        out.append({"rank": rank,
                    "answer": (str(pred_label[i]) if rank == "label"
                               else str(pred_group[i]) if rank == "group" else None)})
    return out


def _bundle(tmp_path, **kw):
    """A real fitted bundle, through the tool's own path."""
    narrowcast = pytest.importorskip("narrowcast")
    from narrowcast import build, sources
    rng = np.random.default_rng(0)
    labs = [f"Genus{i // 3} sp{i}" for i in range(9)]
    d = 32
    cent = {c: rng.normal(size=d) for c in labs}
    def rows(names, n, tag):
        v, l, g, c = [], [], [], []
        for nm in names:
            mu = cent.get(nm, rng.normal(size=d) * 2)
            for o in range(n):
                v.append(mu + rng.normal(scale=0.4, size=d))
                l.append(nm); g.append(nm.split()[0]); c.append(f"{tag}{nm}-{o}")
        return sources._finish(l, descriptor=np.array(v, "float32"), group=g, cluster=c)
    fg = rows(labs, 8, "f")
    bg = rows([f"Bg{i} sp" for i in range(6)], 6, "b")
    ds = build.load_rows(fg, "plantclef24", background=bg)
    clf = build.fit_head(ds)
    metrics = build.fit_and_measure(build.score_frame(clf, ds), p_ood=0.2, **kw)
    gmap = dict(zip(fg.label.tolist(), fg.group.tolist()))
    out = build.save_bundle(tmp_path / "b", clf, fg.labels, "plantclef24", metrics,
                            {}, ds.counts, source="t", groups=gmap,
                            never_answer=kw.get("never_answer"),
                            space=ds.X_train.mean(0))
    return out, ds


def _assert_parity(out, ds):
    from narrowcast import predict as P
    ref = P.Bundle(out).predict(ds.X_eval)
    got = cascade_from_json(build_payload(out, PRE, "plantclef24+coreml:x", "E.mlpackage"),
                            ds.X_eval)
    assert len(ref) == len(got) and len(ref) > 50
    for i, (a, b) in enumerate(zip(ref, got)):
        assert a["rank"] == b["rank"], (i, a, b)
        assert a["answer"] == b["answer"], (i, a, b)


def test_json_alone_reproduces_the_tools_own_decisions(tmp_path):
    out, ds = _bundle(tmp_path)
    _assert_parity(out, ds)


def test_parity_holds_with_the_near_ood_gate(tmp_path):
    out, ds = _bundle(tmp_path, gate=True)
    _assert_parity(out, ds)


def test_parity_holds_with_a_suppressed_label(tmp_path):
    narrowcast = pytest.importorskip("narrowcast")
    from narrowcast import sources
    labs = [f"Genus{i // 3} sp{i}" for i in range(9)]
    out, ds = _bundle(tmp_path, never_answer=[labs[0]], labels=labs)
    payload = build_payload(out, PRE, "plantclef24+coreml:x", "E.mlpackage")
    assert payload["never_answer"] == [labs[0]]
    _assert_parity(out, ds)


def test_an_audit_bundle_is_refused_by_name(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "manifest.json").write_text(json.dumps(
        {"has_head": False, "source": "scores.npz", "metrics": {}}))
    with pytest.raises(SystemExit, match="nothing to export"):
        build_payload(tmp_path / "b", PRE, "e", "E.mlpackage")


def test_a_bundle_without_a_group_map_is_refused(tmp_path):
    out, _ = _bundle(tmp_path)
    m = json.loads((out / "manifest.json").read_text())
    m["groups"] = {}
    (out / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="no group map"):
        build_payload(out, PRE, "e", "E.mlpackage")


def test_the_payload_states_what_it_rests_on(tmp_path):
    out, _ = _bundle(tmp_path)
    p = build_payload(out, PRE, "plantclef24+coreml:pc24_q8", "E.mlpackage",
                      cosine_to_torch=0.9996)
    assert p["encoder"] == "plantclef24+coreml:pc24_q8"
    assert p["rests_on"]["cosine_torch_to_coreml"] == 0.9996
    assert "L2" in p["coreml"]["inside_the_graph"]
    assert p["preprocess"]["side"] == 518
