"""The two middle paths between a clean evaluation and a better head.

Pre-registered in the addendum to `SOURCE_MIX_PREREG.md`.

M1 sweeps the *observation* holdout fraction and reports interval width, which
is what decides whether a smaller evaluation set is worth having. It does not
preserve an out-of-source claim -- a held-out observation is still from a corpus
the head trained on.

M2 reserves whole *species* from the mix, which does preserve one, and asks the
operational question a growing catalogue faces: does mixing in-source data for
the species you have it for damage the species you do not?

Usage:
    PYTHONPATH=. .venv/bin/python -m analysis.source_mix_middle --variant bioclip2
"""

import argparse

import numpy as np
import pandas as pd

from analysis.domain_shift import (
    B, MIN_TEST, N_BOOT, SEED, fit, inat_table, per_species, plantnet_table,
)
from sklearn.linear_model import LogisticRegression

from plantid.config import DATA_PROCESSED

HOLDOUTS = (1.0, 0.8, 0.6, 0.4, 0.2)
CAP_K = 10            # in-source rows per species, for the capped arm
RESERVE_FRAC = 0.20   # of species, for M2


def rank_observations(ina, rng):
    """A stable per-species ordering of observations, so the holdout sets nest:
    the held-out set at h=0.8 contains the one at h=0.6."""
    order = {}
    for cn, g in ina.groupby("cn"):
        ids = np.asarray(g["obs_id"].unique(), dtype=object)
        rng.shuffle(ids)
        order.update({o: i / len(ids) for i, o in enumerate(ids)})
    return ina["obs_id"].map(order).to_numpy()


def setup(variant):
    rng = np.random.default_rng(SEED)
    pn, Epn = plantnet_table(variant)
    ina, Eina = inat_table(variant)
    ina["q"] = rank_observations(ina, rng)

    # Species set fixed across the whole sweep, so nothing moves but the split.
    counts = pd.DataFrame({
        "pn_tr": pn[pn.split == "train"].groupby("cn").size(),
        "in_lo": ina[ina.q < 0.2].groupby("cn").size(),      # smallest holdout
        "in_hi": ina[ina.q >= 0.8].groupby("cn").size(),     # smallest train slice
    }).fillna(0)
    ok = (counts.pn_tr >= 2 * B) & (counts.in_lo >= MIN_TEST) & (counts.in_hi >= 1)
    species = sorted(counts.index[ok])

    pn_tr = pn[(pn.split == "train") & pn.cn.isin(species)]
    ina = ina[ina.cn.isin(species)].reset_index(drop=True)
    E = np.vstack([Epn, Eina])
    return species, pn_tr, ina, E, len(Epn)


def head(pn_tr, ina_train, E, off, balanced=False):
    idx = np.concatenate([pn_tr["emb_row"].to_numpy(), ina_train["emb_row"].to_numpy() + off])
    y = np.concatenate([pn_tr["cn"].to_numpy(), ina_train["cn"].to_numpy()])
    if not balanced:
        return fit(E, idx, y)
    # `class_weight="balanced"` is what the production head uses and what the
    # analysis code inherited from `domain_shift.py` does not.
    return LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced").fit(E[idx], y)


def cap_per_species(df, k, rng):
    """At most `k` rows per species, so every mixed class gets the same in-source
    boost regardless of how many observations it happens to have."""
    keep = []
    for _, g in df.groupby("cn"):
        idx = g.index.to_numpy()
        keep.append(idx if len(idx) <= k else rng.choice(idx, k, replace=False))
    return df.loc[np.concatenate(keep)]


def boot(v, species, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    d = v[rng.integers(0, len(species), (n, len(species)))].mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(np.nanmean(v)), float(lo), float(hi)


def run_m1(variant):
    species, pn_tr, ina, E, off = setup(variant)
    rows = []
    for h in HOLDOUTS:
        te = ina[ina.q < h]
        tr = ina[ina.q >= h]
        clf = head(pn_tr, tr, E, off)
        acc = per_species(clf, E, te.assign(emb_row=te.emb_row + off), species)
        m, lo, hi = boot(acc.to_numpy(), species)
        rows.append({"holdout": h, "n_train_inat": len(tr), "n_test_photos": len(te),
                     "species_top1": round(m, 4), "lo": round(lo, 4), "hi": round(hi, 4),
                     "ci_width": round(hi - lo, 4)})
    return pd.DataFrame(rows)


def run_m2(variant):
    species, pn_tr, ina, E, off = setup(variant)
    rng = np.random.default_rng(SEED + 1)
    sp = np.array(species)
    rng.shuffle(sp)
    reserved = set(sp[: int(round(RESERVE_FRAC * len(sp)))])
    mixed = [s for s in species if s not in reserved]
    reserved = [s for s in species if s in reserved]

    # Same test rows for both heads: the h=0.4 holdout, so the comparison is
    # paired and the training slice matches the rest of this project.
    te = ina[ina.q < 0.4]
    tr_all = ina[ina.q >= 0.4]
    tr_mixed = tr_all[tr_all.cn.isin(mixed)]

    base = fit(E, pn_tr["emb_row"].to_numpy(), pn_tr["cn"].to_numpy())
    mix = head(pn_tr, tr_mixed, E, off)

    rows = []
    for name, group in (("reserved (out-of-source)", reserved), ("mixed (in-source)", mixed)):
        teg = te[te.cn.isin(group)]
        teg = teg.assign(emb_row=teg.emb_row + off)
        a = per_species(base, E, teg, group).to_numpy()
        b = per_species(mix, E, teg, group).to_numpy()
        rng2 = np.random.default_rng(SEED)
        draws = rng2.integers(0, len(group), (N_BOOT, len(group)))
        d = b[draws].mean(1) - a[draws].mean(1)
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append({"group": name, "n_species": len(group), "n_test_photos": len(teg),
                     "P-full": round(float(np.nanmean(a)), 4),
                     "mixed head": round(float(np.nanmean(b)), 4),
                     "delta": round(float(np.nanmean(b) - np.nanmean(a)), 4),
                     "lo": round(float(lo), 4), "hi": round(float(hi), 4)})
    return pd.DataFrame(rows)


def run_m3(variant):
    """Does per-class balancing remove the damage to reserved species?"""
    species, pn_tr, ina, E, off = setup(variant)
    rng = np.random.default_rng(SEED + 1)
    sp = np.array(species)
    rng.shuffle(sp)
    res = set(sp[: int(round(RESERVE_FRAC * len(sp)))])
    mixed = [s for s in species if s not in res]
    reserved = [s for s in species if s in res]

    te = ina[ina.q < 0.4]
    tr_all = ina[ina.q >= 0.4]
    tr_mixed = tr_all[tr_all.cn.isin(mixed)]
    capped = cap_per_species(tr_mixed, CAP_K, np.random.default_rng(SEED))
    empty = tr_mixed.iloc[:0]

    heads = {
        "P-full": head(pn_tr, empty, E, off),
        "mixed": head(pn_tr, tr_mixed, E, off),
        "P-full-bal": head(pn_tr, empty, E, off, balanced=True),
        "mixed-bal": head(pn_tr, tr_mixed, E, off, balanced=True),
        f"mixed-cap-{CAP_K}": head(pn_tr, capped, E, off),
    }
    pairs = [("mixed", "P-full"), ("mixed-bal", "P-full-bal"),
             (f"mixed-cap-{CAP_K}", "P-full")]

    rows = []
    for name, group in (("reserved", reserved), ("mixed", mixed)):
        teg = te[te.cn.isin(group)]
        teg = teg.assign(emb_row=teg.emb_row + off)
        acc = {k: per_species(h, E, teg, group).to_numpy() for k, h in heads.items()}
        draws = np.random.default_rng(SEED).integers(0, len(group), (N_BOOT, len(group)))
        for arm, ref in pairs:
            d = acc[arm][draws].mean(1) - acc[ref][draws].mean(1)
            lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append({"group": name, "arm": arm, "vs": ref,
                         "ref": round(float(np.nanmean(acc[ref])), 4),
                         "arm_acc": round(float(np.nanmean(acc[arm])), 4),
                         "delta": round(float(np.nanmean(acc[arm]) - np.nanmean(acc[ref])), 4),
                         "lo": round(float(lo), 4), "hi": round(float(hi), 4)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="bioclip2")
    a = ap.parse_args()

    m1 = run_m1(a.variant)
    print(f"\n== M1: observation holdout sweep ({a.variant})")
    print("   h=1.0 is the status quo: nothing mixed in, everything held out.")
    print(m1.to_string(index=False))

    m2 = run_m2(a.variant)
    print(f"\n== M2: {int(RESERVE_FRAC * 100)}% of species reserved from the mix entirely")
    print(m2.to_string(index=False))

    m3 = run_m3(a.variant)
    print(f"\n== M3: does per-class balancing remove the damage?")
    print(m3.to_string(index=False))
    m3.to_csv(DATA_PROCESSED / "source_mix_m3.csv", index=False)

    m1.to_csv(DATA_PROCESSED / "source_mix_m1.csv", index=False)
    m2.to_csv(DATA_PROCESSED / "source_mix_m2.csv", index=False)
