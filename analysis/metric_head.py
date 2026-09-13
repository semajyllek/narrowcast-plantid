"""Does a learned metric on frozen features do what a bigger encoder cannot?

Pre-registered in `METRIC_PREREG.md`. The one cell nothing has moved is a crowded
label set at few shots: there, encoder quality from 5.7 MB to 152 MB is worth
nothing measurable. Distillation is closed, inference-side levers are closed, and
the space between "frozen encoder + linear head" and "fine-tune the encoder" has
never been touched.

SetFit's actual mechanism is that it adapts the *body* contrastively from the
user's own examples. The equivalent that does not break this project's
architecture is to learn a **projection on top of the frozen embedding** — same
encoder, seconds of CPU, no bytes worth counting. `TINY_FINDINGS.md` §1 gives the
reason to expect the right shape: a K-way task occupies far fewer dimensions than
the embedding carries, and the projection that establishes it (PCA) is
**unsupervised**. A supervised one to the same width is the principled version.

Every arm ends in the same logistic head, so only the representation varies, and
`pca` is present as the control that makes a supervised win mean anything.

Usage:
    PYTHONPATH=. .venv/bin/python -m analysis.metric_head \\
        --variant mobileclip2_s0 --k 20 --shots 1 4 8 32 0
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis

import analysis.fewshot_curve as F
from narrowcast import build as B

P_OOD = 0.20


def _l2(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def _renorm(fn):
    """L2-normalise whatever a projection emits, because `raw` is normalised.

    Caught in the first smoke run, where every projected arm lost heavily to
    `raw`. The catalogue loader L2-normalises the embedding before the head, but
    nothing renormalised *after* a projection -- and PCA, LDA's eigen solver and
    NCA all emit different scales. `fit_head` uses a fixed `C = 10`, so a
    projection that happens to shrink the norm is silently handed far stronger
    regularisation than the baseline. That is a comparison of scales, not of
    representations, and it would have made a null result look like a finding.
    """
    return lambda X: _l2(fn(X))


def _fit_projection(kind, m, Xtr, ytr, seed=0):
    """Learn the map on TRAINING rows only. Returns a callable, or None if the
    arm cannot be fitted at this shot count.

    LDA is capped at K-1 components by construction -- the between-class scatter
    of K classes has rank K-1 -- which is the same bound that makes `d >= K` cells
    uninformative in the capacity floor. Shrinkage is on because at 8 examples per
    label the within-class scatter is estimated from far fewer rows than
    dimensions, and the unshrunk solve is not merely noisy but singular.
    """
    n_classes = len(set(ytr.tolist()))
    if kind == "raw":
        return lambda X: X
    if kind == "pca":
        k = min(m, Xtr.shape[1], len(Xtr) - 1)
        if k < 1:
            return None
        p = PCA(n_components=k, random_state=seed).fit(Xtr)
        return _renorm(p.transform)
    if kind == "lda":
        k = min(m, n_classes - 1, Xtr.shape[1])
        # LDA needs strictly more samples than classes to estimate a within-class
        # scatter at all. At one example per label that is an equality, not an
        # inequality, so the arm simply does not exist there -- which is itself
        # worth reporting rather than papering over: a supervised projection
        # cannot be fitted from one example per class.
        if k < 1 or len(Xtr) <= n_classes:
            return None
        d = LinearDiscriminantAnalysis(n_components=k, solver="eigen",
                                       shrinkage="auto")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d.fit(Xtr, ytr)
        return _renorm(d.transform)
    if kind == "nca":
        k = min(m, Xtr.shape[1], len(Xtr) - 1)
        if k < 1 or len(Xtr) <= n_classes:
            return None
        n = NeighborhoodComponentsAnalysis(n_components=k, max_iter=60,
                                           random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            n.fit(Xtr, ytr)
        return _renorm(n.transform)
    raise ValueError(kind)


def one_draw(Xc, spc, splitc, Xb, inat, chosen, shots, arms, dims, seed):
    """One label draw, every arm scored on the identical rows and split."""
    rng = np.random.default_rng(seed)
    tr = F._train_rows(Xc, spc, splitc, chosen, shots, rng)
    if tr is None:
        return {}
    Xtr, ytr = tr

    cut = rng.permutation(len(Xb))
    half = len(Xb) // 2
    bg = F._negatives(Xb[cut[:half]], rng, shots, len(chosen), 0)
    ev = F._eval_inat(inat, set(chosen), rng)
    if (ev[2] == "in_catalog").sum() < 2 * len(chosen):
        return {}

    # The projection is fitted on the chosen labels' training rows only. The
    # background negatives are deliberately excluded from that fit: __OTHER__ is
    # not a class the user named, and letting a 2,000-row reject pool dominate the
    # scatter would learn a metric for separating "plant" from "not plant" rather
    # than the fourteen things asked about.
    out = {}
    for kind in arms:
        for m in (dims if kind != "raw" else [0]):
            try:
                proj = _fit_projection(kind, m, Xtr, ytr, seed)
            except (ValueError, np.linalg.LinAlgError):
                # One arm being unfittable at a given shot count must not take the
                # other arms' measurements down with it.
                proj = None
            if proj is None:
                continue
            name = kind if kind == "raw" else f"{kind}{m}"
            try:
                d = F._dataset(proj(Xtr), ytr, proj(bg),
                               proj(ev[0]), *ev[1:])
                met = B.fit_and_measure(B.score_frame(B.fit_head(d), d),
                                        p_ood=P_OOD, seed=seed)
            except (ValueError, np.linalg.LinAlgError):
                continue
            out[name] = [met["label_share"], met["coverage"], met["precision"],
                         met["closed_set_top1"]]
    return out


def main(a):
    Xc, spc, splitc = F.load_catalog(a.variant)
    Xb = F.load_bg(a.variant)
    inat = F.load_inat(a.variant)
    pool = sorted(set(inat[2]) & set(spc))
    print(f"variant {a.variant}  ·  {len(pool)} eligible labels  ·  "
          f"{a.draws} draws  ·  dims {a.dims}\n", flush=True)

    for K in a.k:
        for hard in (False, True):
            arm = "HARD (congeners)" if hard else "EASY (separated)"
            draws = F.draws(np.array(pool), K, a.draws, hard,
                            np.random.default_rng(a.seed + K + hard))
            if not draws:
                continue
            print(f"=== K={K}  {arm}  [{a.variant}] ===", flush=True)
            for shots in a.shots:
                acc = {}
                for i, chosen in enumerate(draws):
                    for name, v in one_draw(Xc, spc, splitc, Xb, inat, chosen,
                                            shots, a.arms, a.dims, i).items():
                        acc.setdefault(name, []).append(v)
                if not acc:
                    continue
                if a.dump:
                    import json as _j
                    rows = [{"K": K, "arm": arm, "shots": shots, "method": n,
                             "draw": i, "label_share": v[0], "coverage": v[1],
                             "precision": v[2], "top1": v[3]}
                            for n, vs in acc.items() for i, v in enumerate(vs)]
                    with open(a.dump, "a") as fh:
                        for r in rows:
                            fh.write(_j.dumps(r) + "\n")
                base = np.nanmean([r[0] for r in acc.get("raw", [[np.nan]])])
                print(f"  {shots or 'all':>4} shots"
                      f"   {'arm':10s}{'label_share':>12}{'vs raw':>9}"
                      f"{'coverage':>10}{'top1':>8}", flush=True)
                for name in sorted(acc, key=lambda n: (n != "raw", n)):
                    A = np.array([[np.nan if x is None else x for x in r]
                                  for r in acc[name]], dtype=float)
                    mu = np.nanmean(A, 0)
                    delta = "" if name == "raw" else f"{mu[0] - base:>+9.4f}"
                    print(f"{'':12}   {name:10s}{mu[0]:>12.4f}{delta:>9}"
                          f"{mu[1]:>10.4f}{mu[3]:>8.4f}   (n={len(acc[name])})",
                          flush=True)
                print(flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="mobileclip2_s0")
    ap.add_argument("--k", type=int, nargs="+", default=[20])
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 4, 8, 32, 0])
    ap.add_argument("--arms", nargs="+", default=["raw", "pca", "lda", "nca"])
    ap.add_argument("--dims", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--draws", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump", default="", help="append per-draw rows as JSONL")
    sys.exit(main(ap.parse_args()))
