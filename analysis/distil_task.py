"""Phase 3 of `TINY_PREREG.md` — distil the fitted *task*, not the encoder.

The two closed distillation attempts (`CLAUDE.md`, "do not redo") both asked a
student to reproduce a **general embedding**, and `PRUNE_FINDINGS.md` closes by
noting for the third time that *cosine to a teacher does not predict downstream
accuracy*. So both optimised the quantity that finding calls the wrong one.

Here the teacher is not an encoder. It is `frozen encoder + fitted head`, and its
output is K+1 posteriors over labels the user named. A student reproducing that
does not have to represent plants in general; it has to separate fourteen things
and say *none of these*. `TINY_FINDINGS.md` measures the room this buys: a K=20
congener task lives in ~16-32 dimensions of a 768-dimensional embedding, and a
large encoder gives up only 1-3pp when squeezed into them.

**It does not follow that a 1 MB student can compute those dimensions from
pixels.** Finding a subspace inside a good representation and computing it from
raw input are different problems, and only the first is measured. That is the
whole point of running this.

Two label sets, mirroring narrowcast's published contrast exactly:

    crowded     8 Sedum + 6 Trifolium
    separated   14 distinct genera

Scored through `build.fit_and_measure` unchanged, so student and teacher numbers
are the same numbers a card would print.

Usage:
    PYTHONPATH=. .venv-mps/bin/python -m analysis.distil_task \\
        --arm crowded --variant plantclef24 --epochs 2 --limit 4000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from narrowcast import build as B
from narrowcast.cascade import DECLINE, LABEL, decide
from plantid.config import DATA_PROCESSED
from plantid.data.curation import canonical_name

OTHER = B.OTHER
ORGANS = ("leaf", "flower")
P_OOD = 0.20

# The README's own two arms. Fixed in source rather than drawn, so the teacher's
# numbers can be checked against a published figure instead of a random draw.
CROWDED = ["Sedum acre", "Sedum album", "Sedum dasyphyllum", "Sedum hispanicum",
           "Sedum rupestre", "Sedum sediforme", "Sedum sexangulare",
           "Sedum spathulifolium", "Trifolium arvense", "Trifolium campestre",
           "Trifolium dubium", "Trifolium incarnatum", "Trifolium pratense",
           "Trifolium repens"]


MIN_IMAGES = 120


def separated_set(index, n=14, seed=0):
    """`n` species from `n` distinct genera, drawn at random among eligible ones.

    **Not** the largest by image count, which was the first version and was
    degenerate: taking the best-represented species per genus selects the easiest
    fourteen in the catalogue, and the teacher scored **top-1 1.0000 with headroom
    0.0000**. A comparator at the ceiling cannot show a student losing anything,
    so prediction 1 would have been tested against a wall.

    Random draw among species with at least `MIN_IMAGES`, which is how the
    README's separated arm is built, and averaged over draws so the arm carries
    its own variance rather than borrowing the crowded arm's.
    """
    counts = index.groupby("sp").size()
    eligible = index[index.sp.map(counts) >= MIN_IMAGES]
    rng = np.random.default_rng(seed)
    genera = rng.permutation(sorted(set(eligible.genus)))[:n]
    return sorted(rng.choice(sorted(set(eligible[eligible.genus == g].sp)))
                  for g in genera)


def load_index():
    d = pd.read_parquet(Path(DATA_PROCESSED) / "catalog_index.parquet")
    d = d[d.organ.isin(ORGANS)].copy()
    # 8 of 43,514 rows carry no local path -- images that never downloaded. They
    # still have cached embeddings, so the teacher can use them and the student
    # cannot; dropped here so both sides see the same rows.
    d = d[d.local_path.notna()].copy()
    d["sp"] = [canonical_name(s) for s in d.species_name]
    d["genus"] = d.sp.str.split().str[0]
    d["path"] = [str(Path(DATA_PROCESSED) / p) for p in d.local_path]
    return d


def load_vectors(variant):
    """image_id -> embedding, for the cached catalogue vectors."""
    ids, X = [], []
    for organ in ORGANS:
        p = Path(DATA_PROCESSED) / f"catalog_{organ}_{variant}.npz"
        d = np.load(p, allow_pickle=True)
        ids += list(d["image_id"].astype(str))
        X.append(d["descriptor"])
    X = np.vstack(X).astype("float32")
    X /= np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    return dict(zip(ids, X))


class Precomputed:
    """A `predict_proba` shim so `score_frame` can score a student unchanged.

    `build.score_frame` expects an sklearn-like head over embeddings. The student
    consumes pixels, so its posteriors are computed once and replayed here in
    evaluation-row order. Nothing about the cascade, the thresholds or the
    bootstrap changes -- which is the point: teacher and student must be scored by
    the identical path or the comparison is between two measurement procedures
    rather than two models.
    """

    def __init__(self, classes, proba):
        self.classes_ = np.asarray(classes)
        self._p = np.asarray(proba, dtype=float)

    def predict_proba(self, X):
        assert len(X) == len(self._p), "row order desynchronised from the frame"
        return self._p


def build_frame(classes, proba, truth, bucket, cluster, group):
    ds = B.Dataset(X_train=np.zeros((1, 1)), y_train=np.array([OTHER]),
                   frame=pd.DataFrame(), X_eval=np.zeros((len(truth), 1)),
                   truth=truth, bucket=bucket,
                   counts={"in_catalog": int((bucket == "in_catalog").sum())},
                   cluster=cluster, group=group)
    return B.score_frame(Precomputed(classes, proba), ds)


def inat_eval(variant, labels):
    """Held-out iNaturalist photographs, joined to observation, species and genus.

    **The evaluation is cross-source and that is not a refinement, it is required.**
    Scored on the Pl@ntNet test split, `plantclef24` -- which is fine-tuned on
    Pl@ntNet -- reaches top-1 0.978-1.000 on a 14-label separated set across four
    draws, with headroom 0.000. A teacher at the ceiling gives a student nothing
    to lose against, so prediction 1 would be tested against a wall. Every headline
    in this project is cross-source for the same reason
    (`DOMAIN_SHIFT_FINDINGS.md`).

    The observation is the cluster -- several photographs of one plant -- which is
    the unit `make_splits` and the bootstrap key on. Pl@ntNet has no such column;
    iNaturalist does, so this direction also buys honest intervals.
    """
    d = np.load(Path(DATA_PROCESSED) / f"inat_{variant}.npz", allow_pickle=True)
    # The caches were written before the repo was renamed and still hold absolute
    # paths under `Documents/plantid/`. Those happen to resolve on this machine,
    # which is exactly why they should not be trusted: the student needs pixels,
    # and a silently-missing image would show up as a training-set hole rather
    # than an error. Rebased on the current tree by filename.
    img_dir = Path(DATA_PROCESSED) / "images_inat"
    paths = np.array([str(img_dir / Path(p).name) for p in d["path"].astype(str)])
    obs = np.array([Path(p).stem.rsplit("_", 1)[0] for p in paths])
    meta = pd.read_parquet(Path(DATA_PROCESSED) / "inat_observations.parquet")
    meta["obs_id"] = meta["obs_id"].astype(str)
    meta["sp"] = [canonical_name(s) for s in meta["species_name"].astype(str)]
    m = meta.set_index("obs_id")
    keep = np.isin(obs, m.index.to_numpy())

    X = d["descriptor"].astype("float32")[keep]
    X /= np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    obs, paths = obs[keep], paths[keep]
    sp = m.loc[obs, "sp"].to_numpy()
    genus = m.loc[obs, "genus"].to_numpy().astype(str)

    chosen = set(labels)
    chosen_genera = {s.split()[0] for s in labels}
    in_set = np.isin(sp, list(chosen))
    # A species the list did not take but whose genus it did is the near-OOD case
    # a narrow catalogue actually faces -- the unchosen congener.
    bucket = np.where(in_set, "in_catalog",
                      np.where(np.isin(genus, list(chosen_genera)),
                               "near_ood", "distant_ood"))
    truth = np.where(in_set, sp, OTHER)
    return X, truth, bucket, obs, genus, paths


def teacher(index, vecs, labels, variant, seed=0):
    """Fit narrowcast's head on Pl@ntNet vectors; evaluate on iNaturalist.

    The teacher is the whole fitted object -- encoder, head, and the thresholds
    `fit_and_measure` derives -- not the encoder alone. Fitting on Pl@ntNet is
    what production does (`build_heads`); evaluating on iNaturalist is what every
    headline in this repo does.
    """
    from sklearn.linear_model import LogisticRegression

    sel = index[index.sp.isin(labels) & (index.split == "train")]
    bg = index[~index.sp.isin(labels) & (index.split == "train")].sample(
        n=min(6000, (~index.sp.isin(labels)).sum()), random_state=seed)

    def vec(df):
        keep = [i for i in df.image_id if i in vecs]
        return np.stack([vecs[i] for i in keep]), keep

    Xtr, id_tr = vec(sel)
    ytr = sel.set_index("image_id").loc[id_tr, "sp"].to_numpy()
    Xbg, _ = vec(bg)
    clf = LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced").fit(
        np.vstack([Xtr, Xbg]), np.concatenate([ytr, np.full(len(Xbg), OTHER)]))

    X, truth, bucket, obs, genus, paths = inat_eval(variant, labels)
    return clf, X, list(paths), truth, bucket, obs, genus


def report(name, m):
    print(f"  {name:24s} label_share {m['label_share']:.4f}  "
          f"coverage {m['coverage']:.4f}  precision {m['precision']:.4f}  "
          f"top1 {m['closed_set_top1']:.4f}  headroom {m['headroom']:.4f}", flush=True)
    return m


def main(a):
    index = load_index()
    labels = CROWDED if a.arm == "crowded" else separated_set(index)
    print(f"\narm {a.arm}: {len(labels)} labels — {', '.join(labels[:4])}, ...",
          flush=True)

    vecs = load_vectors(a.variant)
    clf, Xte, ids, truth, bucket, cluster, group = teacher(index, vecs, labels,
                                                                 a.variant)
    classes = np.array(clf.classes_)
    t_frame = build_frame(classes, clf.predict_proba(Xte), truth, bucket, cluster, group)
    t_metrics = report("teacher", B.fit_and_measure(t_frame, p_ood=P_OOD))

    out = {"arm": a.arm, "variant": a.variant, "labels": labels,
           "teacher": {k: t_metrics[k] for k in
                       ("label_share", "coverage", "precision", "closed_set_top1",
                        "headroom", "group_share", "decline_share")}}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {a.out}", flush=True)
    if a.teacher_only:
        return 0

    from analysis.distil_student import train_student
    return train_student(a, index, labels, clf, classes, vecs, ids, truth, bucket,
                         cluster, group, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("crowded", "separated"), default="crowded")
    ap.add_argument("--variant", default="plantclef24")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=0,
                    help="fix total gradient updates; epochs derived from "
                         "transfer-set size, so --limit sweeps vary data "
                         "without varying compute")
    ap.add_argument("--limit", type=int, default=0, help="cap transfer images (smoke)")
    ap.add_argument("--img", type=int, default=128,
                    help="student input resolution; the teacher plantclef24 "
                         "runs at 518, so a low value confounds capacity with "
                         "what the student can physically see")
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--init", choices=("scratch", "imagenet"), default="scratch")
    ap.add_argument("--out", default="analysis/distil_result.json")
    ap.add_argument("--teacher-only", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=4)
    sys.exit(main(ap.parse_args()))
