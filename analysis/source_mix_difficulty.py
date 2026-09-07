"""Is it the accuracy level, or is it the images?

Pre-registered in addendum 6 of `SOURCE_MIX_PREREG.md`. `narrowcast-derm`'s
K_FINDINGS compared plants at K=345 against dermatology at K=5 because both sit
at baseline 0.80 -- matched accuracy, mismatched everything else. This holds
K = 20, where both domains are measured, and degrades the plant arm by capping
training rows per class until it is as inaccurate as dermatology.

If accuracy is the whole story, a plant build at baseline 0.63 should show
dermatology's damage of -0.148 at the same K.

Usage:
    PYTHONPATH=. .venv/bin/python -m analysis.source_mix_difficulty
"""

import argparse

import numpy as np
import pandas as pd

from analysis.domain_shift import SEED, fit
from analysis.source_mix_middle import setup
from plantid.config import DATA_PROCESSED

K = 20
CAPS = (2, 3, 5, 8, 15, 30, None)     # None = uncapped
DRAWS = 15
RESERVE = 0.10
SOURCE_RATIO = 4                      # out-of-source : in-source, as observed uncapped


def cap(df, k, rng):
    if k is None:
        return df
    keep = []
    for _, g in df.groupby("cn"):
        idx = g.index.to_numpy()
        keep.append(idx if len(idx) <= k else rng.choice(idx, k, replace=False))
    return df.loc[np.concatenate(keep)]


def macro(clf, X, sub, group, off):
    pred = clf.predict(X[sub["emb_row"].to_numpy() + off])
    hit = pd.Series((pred == sub["cn"].to_numpy()).astype(float))
    return float(np.nanmean(hit.groupby(pd.Series(sub["cn"].to_numpy()))
                            .mean().reindex(group).to_numpy()))


def crowded_subset(species, K, rng):
    """K species drawn as whole congener blocks rather than at random.

    Capping training rows cannot make a random 20-species plant set hard -- the
    encoder separates them at 0.93 on two examples each. Relatedness can: this is
    the lever the rest of this project uses to move fine accuracy while holding
    the label count fixed.
    """
    by = {}
    for s in species:
        by.setdefault(s.split()[0], []).append(s)
    blocks = [v for v in by.values() if len(v) >= 2]
    rng.shuffle(blocks)
    out = []
    for b in blocks:
        out += b
        if len(out) >= K:
            return out[:K]
    return None


def one_draw(species, pn_tr, ina, E, off, b_pn, rng, crowded=False):
    sub = crowded_subset(species, K, rng) if crowded else rng.choice(np.array(species), K, replace=False)
    if sub is None:
        return None
    sub = np.array(sub)
    n_res = max(1, int(round(RESERVE * K)))
    reserved, mixed = list(sub[:n_res]), list(sub[n_res:])
    b_in = None if b_pn is None else max(1, round(b_pn / SOURCE_RATIO))

    pn_s = cap(pn_tr[pn_tr.cn.isin(sub)], b_pn, rng)
    te = ina[(ina.q < 0.4) & ina.cn.isin(sub)]
    tr = cap(ina[(ina.q >= 0.4) & ina.cn.isin(mixed)], b_in, rng)
    if te.empty or tr.empty or pn_s.empty:
        return None

    h_L = fit(E, pn_s["emb_row"].to_numpy(), pn_s["cn"].to_numpy())
    idx = np.concatenate([pn_s["emb_row"].to_numpy(), tr["emb_row"].to_numpy() + off])
    y = np.concatenate([pn_s["cn"].to_numpy(), tr["cn"].to_numpy()])
    h_mix = fit(E, idx, y)

    out = {}
    for name, group in (("reserved", reserved), ("mixed", mixed)):
        s = te[te.cn.isin(group)]
        if s.empty:
            return None
        out[name] = {"L": macro(h_L, E, s, group, off), "mix": macro(h_mix, E, s, group, off)}
    return out


def sweep(variant="bioclip2", crowded=False):
    species, pn_tr, ina, E, off = setup(variant)
    rows = []
    for b in CAPS:
        acc = [r for r in (one_draw(species, pn_tr, ina, E, off, b,
                                    np.random.default_rng(SEED + 100 * d + (b or 99)), crowded)
                           for d in range(DRAWS)) if r]
        if not acc:
            continue
        row = {"variant": variant, "shape": "crowded" if crowded else "varied",
               "cap": b if b is not None else "all"}
        for grp in ("reserved", "mixed"):
            v = np.array([a[grp]["mix"] - a[grp]["L"] for a in acc])
            row[f"d_{grp}"] = round(float(v.mean()), 4)
            row[f"sd_{grp}"] = round(float(v.std()), 4)
        row["baseline_top1"] = round(float(np.mean([a["mixed"]["L"] for a in acc])), 4)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=["bioclip2", "mobileclip2_s0"])
    a = ap.parse_args()
    t = pd.concat([sweep(v, c) for v in a.variants for c in (False, True)],
                  ignore_index=True)
    print(f"\n== plants at K = {K}, degraded two ways: fewer rows, and congener-crowded sets")
    print("   dermatology's point at the same K: baseline 0.6319, damage -0.1475")
    print(t.to_string(index=False))
    t.to_csv(DATA_PROCESSED / "source_mix_difficulty.csv", index=False)
