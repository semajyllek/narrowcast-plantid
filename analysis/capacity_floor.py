"""How much representation does a narrow task actually need?

`EMBEDDED_FINDINGS.md` measured the penalty for a *small encoder* as K shrinks and
found it shrinks too -- 3.4pp at K=10, 7.2pp at K=50, for MobileCLIP2-S0 against
BioCLIP-2. That answers "which of the encoders that exist should I pick". It does
not answer the question a sub-1 MB student poses, which is **how much of a frozen
embedding a K-class task consumes at all**.

That number bounds what a distilled student has to reproduce. If K=12 still needs
256 dimensions at full precision, a tiny student is being asked to carry a general
representation and PRUNE_FINDINGS already says that fails. If K=12 holds at 16
dimensions, the task subspace is small and a student only has to find *it*, which
is a different and much easier problem than matching a teacher's embedding -- the
objective every closed distillation attempt in this repo actually used.

Three training-free knobs over the cached embeddings, so this costs CPU:

  width      PCA to d dims, **fit on training rows only**
  control    random Gaussian projection to the same d -- separates "the task
             subspace is low-dimensional" from "any d dimensions would do"
  precision  per-dimension affine quantisation to int8 / int4, simulating what
             an export actually does to the vector

Both arms, always separately, never averaged. `EMBEDDED_FINDINGS.md` reports EASY
and HARD apart because random draws are mostly cross-genus and flatter the small
encoder, and a user picking eight Sedums is living in the HARD column. A capacity
floor measured on random draws would be the same flattering number.

Usage:
    PYTHONPATH=. .venv/bin/python -m analysis.capacity_floor --k 10 20 \
        --variants bioclip2 mobileclip2_s0 --draws 8
"""

import argparse
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from analysis.subset_frontier import CS, OTHER, load, make_draws

DIMS = [256, 64, 32, 16, 8]
BITS = [8, 4]


def _quantise(Xtr, Xs, bits):
    """Per-dimension affine quantisation, ranges taken from the training rows.

    Test ranges are not available at export time, so taking them from the test
    set would measure a quantiser nobody can ship. Values outside the training
    range clip, which is the real failure mode and is left in deliberately.
    """
    lo, hi = Xtr.min(0), Xtr.max(0)
    scale = np.clip(hi - lo, 1e-12, None) / (2 ** bits - 1)
    return [np.round(np.clip(X - lo, 0, hi - lo) / scale) * scale + lo for X in Xs]


def _subset(cat, bg, species, rng, organs, bg_per_k):
    """Train / val / in-set / near-OOD matrices for one draw.

    Mirrors `subset_frontier.one_draw`'s assembly. It is not imported from there
    because that function fits and scores in one pass, and every reduction here
    has to see the *same* draw -- otherwise draw-to-draw variance, which
    EMBEDDED_FINDINGS puts at sd 0.017-0.026, swamps the effect being measured.
    """
    keep = set(species)
    Xtr, ytr, Xin, yin, Xva, yva, Xnear = [], [], [], [], [], [], []
    for organ in organs:
        E, names, split = cat[organ]
        m = np.array([n in keep for n in names])
        tr = m & (split == "train")
        Xtr.append(E[tr]); ytr.append(names[tr])
        va = m & (split == "val")
        Xva.append(E[va]); yva.append(names[va])
        te = m & (split == "test")
        Xin.append(E[te]); yin.append(names[te])
        Xnear.append(E[(~m) & (split == "test")])

        B = bg[organ]
        cut = rng.permutation(len(B))
        n = min(int(0.6 * len(B)), bg_per_k * len(keep)) if bg_per_k else int(0.6 * len(B))
        Xtr.append(B[cut[:n]]); ytr.append(np.full(n, OTHER))
    return (np.vstack(Xtr), np.concatenate(ytr), np.vstack(Xva), np.concatenate(yva),
            np.vstack(Xin), np.concatenate(yin), np.vstack(Xnear))


def _score(Xtr, ytr, Xva, yva, Xin, yin, Xnear):
    """Closed-set top-1 on the K labels, and near-OOD AUROC.

    Both halves, because discrimination is only half of what narrowcast asks of a
    representation -- ADAPT_FINDINGS had to retract a verdict for measuring the
    first and not the second. C is swept on val rather than fixed: reducing the
    dimension changes the scale of the optimum, so a fixed C would report
    regularisation mismatch as lost capacity.
    """
    def fit(C):
        return LogisticRegression(max_iter=3000, C=C, class_weight="balanced").fit(Xtr, ytr)

    def top1(clf, X, y):
        cls = np.array(clf.classes_)
        sp = np.flatnonzero(cls != OTHER)
        return (cls[sp][clf.predict_proba(X)[:, sp].argmax(1)] == y).mean()

    clf = fit(max(CS, key=lambda C: top1(fit(C), Xva, yva)))
    oi = list(clf.classes_).index(OTHER)
    s_in = 1.0 - clf.predict_proba(Xin)[:, oi]
    s_nr = 1.0 - clf.predict_proba(Xnear)[:, oi]
    auroc = roc_auc_score(np.r_[np.ones(len(s_in)), np.zeros(len(s_nr))], np.r_[s_in, s_nr])
    return top1(clf, Xin, yin), auroc


def arms(Xtr, Xva, Xin, Xnear, dims, bits, seed):
    """(name, transformed matrices) for every reduction, from one shared draw."""
    out = [("full", (Xtr, Xva, Xin, Xnear))]
    d_full = Xtr.shape[1]
    for d in dims:
        if d >= d_full:
            continue
        p = PCA(n_components=d, random_state=0).fit(Xtr)
        out.append((f"pca{d}", tuple(p.transform(X) for X in (Xtr, Xva, Xin, Xnear))))
        R = np.random.default_rng(seed).normal(size=(d_full, d)) / np.sqrt(d)
        out.append((f"rand{d}", tuple(X @ R for X in (Xtr, Xva, Xin, Xnear))))
    for b in bits:
        q = _quantise(Xtr, (Xtr, Xva, Xin, Xnear), b)
        out.append((f"int{b}", tuple(q)))
    # Width and precision are only separately free if they compose. Reported
    # separately they are two ~0 results; multiplied together they are the claim
    # about how many *bits* a narrow task needs, and nothing guarantees a
    # 4-bit quantiser survives having its dimensions rotated into a PCA basis --
    # the components are ordered by variance, so the low ones occupy a much
    # smaller share of their quantisation range than the high ones.
    for d in dims:
        if d >= d_full:
            continue
        p = PCA(n_components=d, random_state=0).fit(Xtr)
        Z = tuple(p.transform(X) for X in (Xtr, Xva, Xin, Xnear))
        for b in bits:
            out.append((f"pca{d}+int{b}", tuple(_quantise(Z[0], Z, b))))
    return out


def main(variants, Ks, draws, organs, bg_per_k, dims, bits):
    ref, _ = load("bioclip2")
    all_species = np.array(sorted(set(ref["leaf"][1]) | set(ref["flower"][1])))
    print(f"{len(all_species)} catalogue species, {draws} draws, organs {organs}",
          flush=True)
    print(f"background train rows: {'%d x K' % bg_per_k if bg_per_k else 'full pool'}\n",
          flush=True)

    loaded = {v: load(v) for v in variants}
    for K in Ks:
        for hard in (False, True):
            arm = "HARD (congeners)" if hard else "EASY (random)"
            ds = make_draws(all_species, K, np.random.default_rng(K + hard), hard)[:draws]
            print(f"=== K={K}  {arm} ===", flush=True)
            for v in variants:
                cat, bg = loaded[v]
                acc = {}
                t0 = time.time()
                for i, sp in enumerate(ds):
                    parts = _subset(cat, bg, sp, np.random.default_rng(i), organs, bg_per_k)
                    Xtr, ytr, Xva, yva, Xin, yin, Xnear = parts
                    for name, (A, B, C_, D) in arms(Xtr, Xva, Xin, Xnear, dims, bits, i):
                        acc.setdefault(name, []).append(
                            _score(A, ytr, B, yva, C_, yin, D))
                print(f"  {v}  ({time.time() - t0:.0f}s)", flush=True)
                base = np.array(acc["full"])[:, 0].mean()
                for name, vals in acc.items():
                    a = np.array(vals)
                    t, u = a[:, 0], a[:, 1]
                    print(f"    {name:8s} top1 {t.mean():.4f} "
                          f"[{np.percentile(t, 2.5):.4f},{np.percentile(t, 97.5):.4f}]"
                          f"  vs full {t.mean() - base:+.4f}"
                          f"   auroc_near {u.mean():.4f}", flush=True)
            print(flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--variants", nargs="+", default=["bioclip2", "mobileclip2_s0"])
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--organs", nargs="+", default=["leaf", "flower"])
    ap.add_argument("--bg-per-k", type=int, default=0,
                    help="cap background train rows at N x K; 0 uses the full pool")
    ap.add_argument("--dims", type=int, nargs="+", default=DIMS)
    ap.add_argument("--bits", type=int, nargs="+", default=BITS)
    a = ap.parse_args()
    sys.exit(main(a.variants, a.k, a.draws, a.organs, a.bg_per_k, a.dims, a.bits))
