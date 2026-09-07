"""Does the source-mix damage survive the label set the product actually ships?

Pre-registered in addendum 5 of `SOURCE_MIX_PREREG.md`. Every earlier source-mix
number is at K = 345; a user picks 10-50 species. The damage mechanism is
competition in a single argmax, so the number of competitors is the thing most
likely to change the answer.

Usage:
    PYTHONPATH=. .venv/bin/python -m analysis.source_mix_k --variant bioclip2
"""

import argparse

import numpy as np
import pandas as pd

from analysis.domain_shift import SEED, fit
from analysis.source_mix_middle import setup
from plantid.config import DATA_PROCESSED

KS = (10, 20, 50, 100, 345)
DRAWS = 15
RESERVE = 0.10


def acc_by_species(clf, E, rows, truth, group):
    pred = clf.predict(E[rows])
    hit = pd.Series((pred == truth).astype(float))
    return hit.groupby(pd.Series(truth)).mean().reindex(group).to_numpy()


def one_draw(species, pn_tr, ina, E, off, K, rng):
    sub = rng.choice(np.array(species), K, replace=False)
    n_res = max(1, int(round(RESERVE * K)))
    reserved, mixed = list(sub[:n_res]), list(sub[n_res:])

    pn_s = pn_tr[pn_tr.cn.isin(sub)]
    te = ina[(ina.q < 0.4) & ina.cn.isin(sub)]
    tr = ina[(ina.q >= 0.4) & ina.cn.isin(mixed)]
    if te.empty or tr.empty:
        return None

    def build(inat_rows, pn_rows):
        idx = np.concatenate([pn_rows["emb_row"].to_numpy(),
                              inat_rows["emb_row"].to_numpy() + off])
        y = np.concatenate([pn_rows["cn"].to_numpy(), inat_rows["cn"].to_numpy()])
        return fit(E, idx, y)

    h_P = build(tr.iloc[:0], pn_s)
    h_mix = build(tr, pn_s)
    # head i: in-source where it exists, Pl@ntNet where it does not
    h_i = build(tr, pn_s[pn_s.cn.isin(reserved)])
    # control: head i's structure with Pl@ntNet rows in place of the iNaturalist ones
    pn_mix = pn_s[pn_s.cn.isin(mixed)]
    take = rng.choice(pn_mix.index.to_numpy(), min(len(tr), len(pn_mix)), replace=False)
    h_ctrl = build(tr.iloc[:0], pd.concat([pn_mix.loc[take], pn_s[pn_s.cn.isin(reserved)]]))

    classes = h_P.classes_
    out = {}
    for name, group in (("reserved", reserved), ("mixed", mixed)):
        g = te[te.cn.isin(group)]
        if g.empty:
            return None
        r, truth = g["emb_row"].to_numpy() + off, g["cn"].to_numpy()
        p = {k: 0.5 * (h_P.predict_proba(E[r]) + h.predict_proba(E[r]))
             for k, h in (("T1", h_i), ("T1c", h_ctrl))}
        a = {"P": acc_by_species(h_P, E, r, truth, group),
             "mix": acc_by_species(h_mix, E, r, truth, group)}
        for k, probs in p.items():
            pred = classes[probs.argmax(1)]
            hit = pd.Series((pred == truth).astype(float))
            a[k] = hit.groupby(pd.Series(truth)).mean().reindex(group).to_numpy()
        out[name] = {k: float(np.nanmean(v)) for k, v in a.items()}
    return out


def sweep(variant):
    species, pn_tr, ina, E, off = setup(variant)
    rows = []
    for K in KS:
        draws = 1 if K >= len(species) else DRAWS
        acc = []
        for d in range(draws):
            r = one_draw(species, pn_tr, ina, E, off, min(K, len(species)),
                         np.random.default_rng(SEED + 100 * d + K))
            if r:
                acc.append(r)
        if not acc:
            continue
        row = {"K": K, "draws": len(acc)}
        for grp in ("reserved", "mixed"):
            single = np.array([a[grp]["mix"] - a[grp]["P"] for a in acc])
            t1 = np.array([a[grp]["T1"] - a[grp]["P"] for a in acc])
            data = np.array([a[grp]["T1"] - a[grp]["T1c"] for a in acc])
            row[f"single_{grp}"] = round(float(single.mean()), 4)
            row[f"single_{grp}_sd"] = round(float(single.std()), 4)
            row[f"T1_{grp}"] = round(float(t1.mean()), 4)
            row[f"T1data_{grp}"] = round(float(data.mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="bioclip2")
    a = ap.parse_args()
    t = sweep(a.variant)
    print(f"\n== K sweep at r = {RESERVE} ({a.variant}), {DRAWS} draws per K")
    print("   single_* = single mixed head vs P-full;  T1data_* = T1 vs its matched control")
    print(t[["K", "draws", "single_reserved", "single_reserved_sd", "T1_reserved",
             "T1data_reserved", "single_mixed", "T1_mixed"]].to_string(index=False))
    t.to_csv(DATA_PROCESSED / f"source_mix_k_{a.variant}.csv", index=False)
