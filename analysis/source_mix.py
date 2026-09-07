"""Is there accuracy left on the table by fitting the head on one source?

Pre-registered in `SOURCE_MIX_PREREG.md`. Reuses the splits, inclusion rule,
subsampling and bootstrap from `analysis/domain_shift.py` unchanged, so the
numbers sit in the same table as the published source-shift arm.

Every arm is scored on the *same* held-out iNaturalist photographs, from
observations no arm trained on.

Usage:
    PYTHONPATH=. .venv/bin/python -m analysis.source_mix --variants bioclip2
"""

import argparse

import numpy as np
import pandas as pd

from analysis.domain_shift import (
    B, MIN_TEST, N_BOOT, SEED, fit, inat_table, per_species, plantnet_table,
    split_inat, subsample,
)
from plantid.config import DATA_PROCESSED


def run(variant: str) -> dict:
    rng = np.random.default_rng(SEED)
    pn, Epn = plantnet_table(variant)
    ina, Eina = inat_table(variant)
    ina["split"] = split_inat(ina, rng)

    counts = pd.DataFrame({
        "pn_tr": pn[pn.split == "train"].groupby("cn").size(),
        "pn_te": pn[pn.split == "test"].groupby("cn").size(),
        "in_tr": ina[ina.split == "train"].groupby("cn").size(),
        "in_te": ina[ina.split == "test"].groupby("cn").size(),
    }).fillna(0)
    ok = ((counts.pn_tr >= 2 * B) & (counts.in_tr >= B)
          & (counts.pn_te >= MIN_TEST) & (counts.in_te >= MIN_TEST))
    species = sorted(counts.index[ok])

    pn_tr = pn[(pn.split == "train") & pn.cn.isin(species)]
    in_tr = ina[(ina.split == "train") & ina.cn.isin(species)]
    pn_te = pn[(pn.split == "test") & pn.cn.isin(species)].reset_index(drop=True)
    in_te = ina[(ina.split == "test") & ina.cn.isin(species)].reset_index(drop=True)

    # One embedding matrix over both sources, so a single head can be fitted on a
    # mixture. Pl@ntNet rows keep their indices; iNaturalist rows are offset.
    E = np.vstack([Epn, Eina])
    off = len(Epn)

    def rows(df, pick=None, inat=False):
        idx = df.loc[pick, "emb_row"].to_numpy() if pick is not None else df["emb_row"].to_numpy()
        lab = df.loc[pick, "cn"].to_numpy() if pick is not None else df["cn"].to_numpy()
        return (idx + off if inat else idx), lab

    pn_B = subsample(pn_tr, species, rng, B)
    pn_2B = subsample(pn_tr, species, rng, 2 * B)
    in_B = subsample(in_tr, species, rng, B)

    def cat(*parts):
        i = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        return i, y

    arms = {
        "P-full": rows(pn_tr),
        "P-full + i": cat(rows(pn_tr), rows(in_tr, inat=True)),
        "P-2B": rows(pn_tr, pn_2B),
        "P-B + i-B": cat(rows(pn_tr, pn_B), rows(in_tr, in_B, inat=True)),
        "i-only": rows(in_tr, in_B, inat=True),
    }

    cells = {}
    for name, (idx, y) in arms.items():
        clf = fit(E, idx, y)
        cells[name] = {
            "inat": per_species(clf, E, in_te.assign(emb_row=in_te.emb_row + off), species),
            "pn": per_species(clf, E, pn_te, species),
            "inat_genus": _genus(clf, E, in_te.assign(emb_row=in_te.emb_row + off), species),
        }
    return {"variant": variant, "species": species, "cells": cells,
            "n_pn_tr": len(pn_tr), "n_in_tr": len(in_tr),
            "n_in_te": len(in_te), "n_pn_te": len(pn_te)}


def _genus(clf, E, df, species):
    pred = clf.predict(E[df["emb_row"].to_numpy()])
    hit = pd.Series((np.array([p.split()[0] for p in pred])
                     == np.array([t.split()[0] for t in df["cn"]])).astype(float), index=df.index)
    return hit.groupby(df["cn"]).mean().reindex(species)


def report(res: dict) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    species = res["species"]
    draws = rng.integers(0, len(species), (N_BOOT, len(species)))
    base = {k: res["cells"]["P-full"][k].to_numpy() for k in ("inat", "pn", "inat_genus")}

    rows = []
    for name, c in res["cells"].items():
        row = {"variant": res["variant"], "arm": name}
        for metric, label in (("inat", "inat_species"), ("inat_genus", "inat_genus"), ("pn", "pn_species")):
            v = c[metric].to_numpy()
            row[label] = round(float(np.nanmean(v)), 4)
            if name != "P-full":
                d = v[draws].mean(1) - base[metric][draws].mean(1)
                lo, hi = np.percentile(d, [2.5, 97.5])
                row[label + "_d"] = f"{np.nanmean(v) - np.nanmean(base[metric]):+.4f} [{lo:+.4f}, {hi:+.4f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def contrast(res, a, b, metric="inat"):
    """Paired difference between two arms, resampling species. Reported directly
    rather than as a difference of two differences against the baseline."""
    rng = np.random.default_rng(SEED)
    n = len(res["species"])
    x = res["cells"][a][metric].to_numpy()
    y = res["cells"][b][metric].to_numpy()
    draws = rng.integers(0, n, (N_BOOT, n))
    d = x[draws].mean(1) - y[draws].mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return f"{a} - {b}: {np.nanmean(x) - np.nanmean(y):+.4f} [{lo:+.4f}, {hi:+.4f}]"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["bioclip2"])
    ap.add_argument("--out", default=str(DATA_PROCESSED / "source_mix.csv"))
    a = ap.parse_args()
    out = []
    for v in a.variants:
        res = run(v)
        print(f"\n== {v}: {len(res['species'])} species | train {res['n_pn_tr']} Pl@ntNet "
              f"+ {res['n_in_tr']} iNat | test {res['n_in_te']} iNat / {res['n_pn_te']} Pl@ntNet")
        t = report(res)
        print(t.to_string(index=False), flush=True)
        print("\n  fixed-volume contrast, iNat species top-1:")
        print("   ", contrast(res, "P-B + i-B", "P-2B"))
        print("  mix vs best single source:")
        print("   ", contrast(res, "P-full + i", "i-only"))
        out.append(t)
    pd.concat(out).to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")
