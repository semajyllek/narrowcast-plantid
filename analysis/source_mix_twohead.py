"""Two heads, and per-class logit centring: can either stop the mix damaging
species that have no in-source data?

Pre-registered in addendum 3 of `SOURCE_MIX_PREREG.md`. Reuses the M2/M3 split
from `analysis/source_mix_middle.py` unchanged -- same reserved species, same
test rows -- so every number is paired with the ones already published.

Usage:
    PYTHONPATH=. .venv/bin/python -m analysis.source_mix_twohead --variant bioclip2
"""

import argparse

import numpy as np
import pandas as pd
from scipy.special import softmax

from analysis.domain_shift import N_BOOT, SEED, fit, per_species
from analysis.source_mix_middle import CAP_K, RESERVE_FRAC, cap_per_species, head, setup
from plantid.config import DATA_PROCESSED
from plantid.data.curation import curated_name
from plantid.eval.inat_fusion import _l2


def reference_pool(variant, cache_dir=DATA_PROCESSED):
    """iNaturalist-source photographs that are *out of catalogue*.

    Unlabelled with respect to the 345 classes and disjoint from the
    in-catalogue test rows, so centring on it leaks nothing.
    """
    obs = pd.read_parquet(cache_dir / "inat_observations.parquet")
    obs = obs[obs["bucket"] != "in_catalog"]
    z = np.load(cache_dir / f"inat_{variant}.npz")
    pos = {p: i for i, p in enumerate(z["path"])}
    rows = [pos[p] for paths in obs["local_paths"] for p in paths if p in pos]
    return _l2(z["descriptor"].astype(np.float32))[rows]


def centred(clf, E, ref):
    """Per-class logits minus that class's mean logit on the reference pool."""
    offset = clf.decision_function(ref).mean(0)
    return lambda X: clf.decision_function(X) - offset


def predict(scorer, classes, E, rows):
    return classes[np.asarray(scorer(E[rows])).argmax(1)]


def acc_by_species(pred, truth, group):
    hit = pd.Series((pred == truth).astype(float))
    return hit.groupby(pd.Series(truth)).mean().reindex(group).to_numpy()


def run(variant, reserve_frac=RESERVE_FRAC, extra=False):
    species, pn_tr, ina, E, off = setup(variant)
    rng = np.random.default_rng(SEED + 1)
    sp = np.array(species)
    rng.shuffle(sp)
    res = set(sp[: int(round(reserve_frac * len(sp)))])
    mixed = [s for s in species if s not in res]
    reserved = [s for s in species if s in res]

    te = ina[ina.q < 0.4]
    tr_all = ina[ina.q >= 0.4]
    tr_mixed = tr_all[tr_all.cn.isin(mixed)]
    capped = cap_per_species(tr_mixed, CAP_K, np.random.default_rng(SEED))
    empty = tr_mixed.iloc[:0]
    ref = reference_pool(variant)

    h_P = head(pn_tr, empty, E, off)                    # Pl@ntNet only, all classes
    h_mix = head(pn_tr, tr_mixed, E, off)               # the M2 mixed head
    h_cap = head(pn_tr, capped, E, off)                 # best arm from M3

    # head i: in-source rows where they exist, Pl@ntNet rows where they do not,
    # so the reserved classes are present in both heads rather than absent from one.
    pn_res = pn_tr[pn_tr.cn.isin(reserved)]
    idx = np.concatenate([tr_mixed["emb_row"].to_numpy() + off, pn_res["emb_row"].to_numpy()])
    y = np.concatenate([tr_mixed["cn"].to_numpy(), pn_res["cn"].to_numpy()])
    h_i = fit(E, idx, y)

    # Control for T1. Averaging two heads is an ensemble, and ensembles improve
    # things for reasons unrelated to source. This head has head i's *structure*
    # exactly -- same class composition, same row count for the mixed classes --
    # but draws those rows from Pl@ntNet instead of iNaturalist. If averaging it
    # with head P reproduces T1's gain, the gain is ensembling, not source.
    rng_c = np.random.default_rng(SEED + 7)
    pn_mixed = pn_tr[pn_tr.cn.isin(mixed)]
    n_take = min(len(tr_mixed), len(pn_mixed))
    take = rng_c.choice(pn_mixed.index.to_numpy(), n_take, replace=False)
    idx_c = np.concatenate([pn_mixed.loc[take, "emb_row"].to_numpy(),
                            pn_res["emb_row"].to_numpy()])
    y_c = np.concatenate([pn_mixed.loc[take, "cn"].to_numpy(), pn_res["cn"].to_numpy()])
    h_ctrl = fit(E, idx_c, y_c)

    cls_P, cls_i = h_P.classes_, h_i.classes_
    assert list(cls_P) == list(cls_i), "heads must span the same label space"

    scorers = {
        "P-full": lambda X: h_P.predict_proba(X),
        "mixed": lambda X: h_mix.predict_proba(X),
        f"mixed-cap-{CAP_K}": lambda X: h_cap.predict_proba(X),
        "T1 two-head avg": lambda X: 0.5 * (h_P.predict_proba(X) + h_i.predict_proba(X)),
        "T1-control (no iNat)": lambda X: 0.5 * (h_P.predict_proba(X) + h_ctrl.predict_proba(X)),
        "T2 mixed + centring": lambda X: softmax(centred(h_mix, E, ref)(X), axis=1),
        "T2 cap + centring": lambda X: softmax(centred(h_cap, E, ref)(X), axis=1),
    }

    rows, keep = [], {}
    for name, group in (("reserved", reserved), ("mixed", mixed)):
        teg = te[te.cn.isin(group)]
        r = teg["emb_row"].to_numpy() + off
        truth = teg["cn"].to_numpy()
        acc = {k: acc_by_species(predict(s, cls_P, E, r), truth, group) for k, s in scorers.items()}
        draws = np.random.default_rng(SEED).integers(0, len(group), (N_BOOT, len(group)))
        base = acc["P-full"]
        for k, v in acc.items():
            if k == "P-full":
                continue
            d = v[draws].mean(1) - base[draws].mean(1)
            lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append({"group": name, "arm": k,
                         "P-full": round(float(np.nanmean(base)), 4),
                         "arm_acc": round(float(np.nanmean(v)), 4),
                         "delta": round(float(np.nanmean(v) - np.nanmean(base)), 4),
                         "lo": round(float(lo), 4), "hi": round(float(hi), 4)})
        # The architecture-matched contrast. T1-control is T1 with the iNaturalist
        # rows swapped for Pl@ntNet ones, so this isolates what the *data* did
        # from what averaging two heads did.
        d = acc["T1 two-head avg"][draws].mean(1) - acc["T1-control (no iNat)"][draws].mean(1)
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append({"group": name, "arm": "T1 vs T1-control (data effect)",
                     "P-full": round(float(np.nanmean(acc["T1-control (no iNat)"])), 4),
                     "arm_acc": round(float(np.nanmean(acc["T1 two-head avg"])), 4),
                     "delta": round(float(np.nanmean(acc["T1 two-head avg"])
                                          - np.nanmean(acc["T1-control (no iNat)"])), 4),
                     "lo": round(float(lo), 4), "hi": round(float(hi), 4)})
        keep[name] = acc
    if extra:
        return pd.DataFrame(rows), keep, {"reserved": reserved, "mixed": mixed}
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="bioclip2")
    a = ap.parse_args()
    t = run(a.variant)
    print(f"\n== two-head and centring arms ({a.variant}), vs the Pl@ntNet-only head")
    print(t.to_string(index=False))
    t.to_csv(DATA_PROCESSED / f"source_mix_twohead_{a.variant}.csv", index=False)
