"""How many examples per label does a narrow classifier actually need?

Every accuracy number in this repo and in narrowcast is fitted on hundreds of
images per label. The tool's pitch -- "train a classifier on your own classes" --
is aimed at somebody who has a few dozen photographs, and **nobody has measured
what happens down there.** SetFit's whole claim is that 8 examples per class is
enough for text; the equivalent number here is unmeasured in both repos.

This sweeps shots per label and reports narrowcast's own card metrics, by
constructing a `build.Dataset` and running `fit_head` -> `score_frame` ->
`fit_and_measure` unchanged. The numbers are therefore the numbers a card would
print, not a proxy for them -- `label_share` in particular, which is the headline
narrowcast refuses to omit.

Two evaluation modes, reported separately and never averaged:

  --eval plantnet   train and test both from the Pl@ntNet catalogue splits.
                    Clean shot curve, one source. **The catalogue has no
                    observation grouping**, so a "shot" is one photograph and
                    intervals are anticonservative; narrowcast says so on the
                    card and the same caveat applies here.

  --eval inat       train on Pl@ntNet, evaluate on iNaturalist observations.
                    Cross-source, real observation clusters, and the real
                    near-OOD bucket -- what a user actually gets, since they fit
                    on photographs they have and deploy against ones they do not.
                    DOMAIN_SHIFT_FINDINGS' penalty sits on top of the shot
                    effect, so read the *shape* here and the *level* above.

Both arms of `EMBEDDED_FINDINGS.md` throughout: EASY draws K species uniformly,
HARD draws whole genera so congeners stay together. Never averaged -- random
draws are mostly cross-genus and flatter every small-data result, and a user
picking eight Sedums is living in the HARD column.

Usage:
    PYTHONPATH=. .venv/bin/python -m analysis.fewshot_curve \
        --variant mobileclip2_s0 --k 10 20 --eval inat
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from narrowcast import build as B
from plantid.config import DATA_PROCESSED
from plantid.data.curation import canonical_name
from plantid.features.embed_background import catalog_species, load_background

ORGANS = ("leaf", "flower")
SHOTS = [1, 2, 4, 8, 16, 32, 64, 0]      # 0 = every training row available
OTHER = B.OTHER
BG_TRAIN = 800                            # negatives for the reject class, held fixed
P_OOD = 0.20


def _l2(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def load_catalog(variant):
    """Pl@ntNet catalogue rows: vectors, canonical species, split. Training side."""
    X, sp, split = [], [], []
    for organ in ORGANS:
        p = Path(DATA_PROCESSED) / f"catalog_{organ}_{variant}.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        X.append(_l2(d["descriptor"]))
        sp += [canonical_name(n) for n in d["species_name"].astype(str)]
        split += list(d["split"].astype(str))
    if not X:
        raise SystemExit(f"no catalogue cache for variant {variant!r}")
    return np.vstack(X), np.array(sp), np.array(split)


def load_bg(variant):
    return np.vstack([_l2(load_background(o, exclude_species=catalog_species(),
                                          variant=variant)["descriptor"])
                      for o in ORGANS])


def load_inat(variant):
    """iNaturalist photos joined to their observation, species, genus and bucket.

    The cache stores vectors and paths only; `<obs_id>_<n>.jpg` is the join key
    back to `inat_observations.parquet`. The observation is the cluster -- several
    photographs of one plant -- which is the unit narrowcast splits and bootstraps
    on, and the reason this mode can report honest intervals where `plantnet`
    cannot.
    """
    d = np.load(Path(DATA_PROCESSED) / f"inat_{variant}.npz", allow_pickle=True)
    obs = np.array([Path(p).stem.rsplit("_", 1)[0] for p in d["path"].astype(str)])
    meta = pd.read_parquet(Path(DATA_PROCESSED) / "inat_observations.parquet")
    meta["obs_id"] = meta["obs_id"].astype(str)
    meta["species_name"] = [canonical_name(s) for s in meta["species_name"].astype(str)]
    m = meta.set_index("obs_id")
    keep = np.isin(obs, m.index.to_numpy())
    obs = obs[keep]
    return (_l2(d["descriptor"])[keep], obs,
            m.loc[obs, "species_name"].to_numpy(),
            m.loc[obs, "genus"].to_numpy().astype(str),
            m.loc[obs, "bucket"].to_numpy().astype(str))


def draws(species, K, n_draws, hard, rng):
    by_genus = defaultdict(list)
    for s in species:
        by_genus[s.split()[0]].append(s)
    multi = [g for g, v in by_genus.items() if len(v) >= 2]
    out = []
    for _ in range(n_draws):
        if hard:
            picked = []
            for g in rng.permutation(multi):
                picked += by_genus[g]
                if len(picked) >= K:
                    break
            if len(picked) >= K:
                out.append(picked[:K])
        else:
            out.append(list(rng.choice(species, K, replace=False)))
    return out


def _train_rows(Xc, spc, splitc, chosen, shots, rng):
    """`shots` training vectors per chosen label, drawn from the train split."""
    Xs, ys = [], []
    for s in chosen:
        idx = np.flatnonzero((spc == s) & (splitc == "train"))
        if len(idx) == 0:
            return None
        take = idx if shots == 0 else rng.choice(idx, min(shots, len(idx)), replace=False)
        Xs.append(Xc[take]); ys.append(np.full(len(take), s))
    return np.vstack(Xs), np.concatenate(ys)


def _dataset(Xtr, ytr, Xb_train, Xev, truth, bucket, cluster, group):
    """A `build.Dataset` assembled directly, so `fit_and_measure` is untouched.

    `load_rows` is bypassed rather than reimplemented: it owns the 50/50 cluster
    split between train and eval, and this experiment has to control the training
    side exactly. Everything downstream of it -- the head, the cascade, the
    thresholds, the bootstrap -- is the real path.
    """
    return B.Dataset(
        X_train=np.vstack([Xtr, Xb_train]),
        y_train=np.concatenate([ytr, np.full(len(Xb_train), OTHER)]),
        frame=pd.DataFrame(),
        X_eval=Xev, truth=truth, bucket=bucket,
        counts={"in_catalog": int((bucket == "in_catalog").sum())},
        cluster=cluster, group=group)


def _negatives(Xb, rng, shots, K, per_positive):
    """Training negatives for the reject class.

    Held at a fixed count by default, which is what a user with a fixed reject
    pool has. `per_positive` instead scales them with the number of positives,
    holding the ratio constant across the shot sweep -- the control
    `EMBEDDED_FINDINGS.md` ran for the same reason, since at K=20 and one shot a
    fixed 800 negatives outnumber the positives 40:1 and a negative-dominated fit
    would produce a collapsing `label_share` all by itself.
    """
    n = BG_TRAIN if not per_positive else max(K, per_positive * (shots or 64) * K)
    return Xb[rng.permutation(len(Xb))[:min(n, len(Xb))]]


def _eval_inat(inat, chosen, rng):
    """Held-out iNat rows: chosen species in-catalogue, everything else near/far."""
    X, obs, sp, gen, buck = inat
    in_set = np.isin(sp, list(chosen))
    # A chosen species' own rows are in_catalog; every other catalogue species is
    # the near-OOD case that actually bites a narrow list, whatever bucket the
    # original 490-species evaluation assigned it.
    bucket = np.where(in_set, "in_catalog",
                      np.where(buck == "distant_ood", "distant_ood", "near_ood"))
    truth = np.where(in_set, sp, OTHER)
    return X, truth, bucket, obs, gen


def _eval_plantnet(Xc, spc, splitc, chosen, Xb_eval):
    """Held-out catalogue rows, plus catalogue species outside the draw as near-OOD.

    `Xb_eval` must be disjoint from the negatives the head was fitted on -- the
    caller splits the pool once. Scoring `distant_ood` on rows the head trained
    against measures memorisation and reports it as rejection.
    """
    te = splitc == "test"
    in_set = te & np.isin(spc, list(chosen))
    near = te & ~np.isin(spc, list(chosen))
    far = np.arange(len(Xb_eval))
    X = np.vstack([Xc[in_set], Xc[near], Xb_eval])
    truth = np.concatenate([spc[in_set], np.full(near.sum() + len(far), OTHER)])
    bucket = np.array(["in_catalog"] * int(in_set.sum()) + ["near_ood"] * int(near.sum())
                      + ["distant_ood"] * len(far))
    cluster = np.concatenate([spc[in_set], spc[near], np.full(len(far), "__bg__")])
    group = np.array([str(c).split()[0] for c in cluster])
    return X, truth, bucket, cluster, group


def main(variant, Ks, n_draws, mode, shots_list, seed, per_positive):
    Xc, spc, splitc = load_catalog(variant)
    Xb = load_bg(variant)
    inat = load_inat(variant) if mode == "inat" else None
    pool = (sorted(set(inat[2]) & set(spc)) if mode == "inat"
            else sorted(set(spc[splitc == "train"])))
    ratio = (f"{per_positive} per positive" if per_positive else f"{BG_TRAIN} fixed")
    print(f"variant {variant}  mode {mode}  {len(pool)} eligible species  "
          f"{n_draws} draws  negatives {ratio}\n", flush=True)

    for K in Ks:
        for hard in (False, True):
            arm = "HARD (congeners)" if hard else "EASY (random)"
            ds = draws(np.array(pool), K, n_draws, hard,
                       np.random.default_rng(seed + K + hard))
            if not ds:
                continue
            print(f"=== K={K}  {arm}  [{variant}, eval={mode}] ===", flush=True)
            print(f"  {'shots':>6}  {'label_share':>11}  {'coverage':>9}  "
                  f"{'precision':>9}  {'top1':>7}  {'headroom':>8}", flush=True)
            for shots in shots_list:
                rows = []
                for i, chosen in enumerate(ds):
                    rng = np.random.default_rng(1000 * seed + i)
                    tr = _train_rows(Xc, spc, splitc, chosen, shots, rng)
                    if tr is None:
                        continue
                    # The reject pool is split once per draw, so the rows the head
                    # is fitted against and the rows it is scored on cannot overlap.
                    cut = rng.permutation(len(Xb))
                    half = len(Xb) // 2
                    bg_tr = _negatives(Xb[cut[:half]], rng, shots, K, per_positive)
                    if mode == "inat":
                        ev = _eval_inat(inat, set(chosen), rng)
                    else:
                        ev = _eval_plantnet(Xc, spc, splitc, set(chosen),
                                            Xb[cut[half:half + 2000]])
                    if (ev[2] == "in_catalog").sum() < 2 * K:
                        continue
                    d = _dataset(tr[0], tr[1], bg_tr, *ev)
                    try:
                        clf = B.fit_head(d)
                        m = B.fit_and_measure(B.score_frame(clf, d), p_ood=P_OOD,
                                              seed=i)
                    except ValueError:
                        continue
                    rows.append([m["label_share"], m["coverage"], m["precision"],
                                 m["closed_set_top1"], m["headroom"]])
                if not rows:
                    print(f"  {shots or 'all':>6}  (no usable draw)", flush=True)
                    continue
                a = np.array([[np.nan if v is None else v for v in r] for r in rows],
                             dtype=float)
                mu = np.nanmean(a, 0)
                print(f"  {str(shots or 'all'):>6}  {mu[0]:>11.4f}  {mu[1]:>9.4f}  "
                      f"{mu[2]:>9.4f}  {mu[3]:>7.4f}  {mu[4]:>8.4f}"
                      f"   (n={len(rows)})", flush=True)
            print(flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="mobileclip2_s0")
    ap.add_argument("--k", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--draws", type=int, default=12)
    ap.add_argument("--eval", dest="mode", choices=("plantnet", "inat"), default="inat")
    ap.add_argument("--shots", type=int, nargs="+", default=SHOTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bg-per-positive", type=int, default=0, dest="per_positive",
                    help="scale training negatives with shots x K instead of "
                         "holding them at a fixed count; the background-ratio "
                         "control from EMBEDDED_FINDINGS")
    a = ap.parse_args()
    sys.exit(main(a.variant, a.k, a.draws, a.mode, a.shots, a.seed, a.per_positive))
