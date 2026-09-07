"""Where does the two-head configuration stop costing accuracy and start saving it?

Pre-registered in addendum 4 of `SOURCE_MIX_PREREG.md`. Sweeps the fraction of
species reserved from the mix, holding everything else to the design already
fitted, and locates the crossover between T1 and the single mixed head on
catalogue-mean accuracy.

Usage:
    PYTHONPATH=. .venv/bin/python -m analysis.source_mix_sweep --variant bioclip2
"""

import argparse

import numpy as np
import pandas as pd

from analysis.source_mix_twohead import run
from plantid.config import DATA_PROCESSED

FRACTIONS = (0.05, 0.10, 0.20, 0.35, 0.50)


def sweep(variant):
    rows = []
    for r in FRACTIONS:
        tbl, acc, groups = run(variant, reserve_frac=r, extra=True)
        n_res, n_mix = len(groups["reserved"]), len(groups["mixed"])
        row = {"reserve_frac": r, "n_reserved": n_res, "n_mixed": n_mix}

        for arm in ("mixed", "T1 two-head avg"):
            for grp in ("reserved", "mixed"):
                m = tbl[(tbl.group == grp) & (tbl.arm == arm)]
                row[f"{'single' if arm == 'mixed' else 'T1'}_{grp}"] = float(m.delta.iloc[0])
        d = tbl[(tbl.group == "reserved") & (tbl.arm == "T1 vs T1-control (data effect)")]
        row["T1_data_effect_reserved"] = float(d.delta.iloc[0])
        row["T1_data_lo"], row["T1_data_hi"] = float(d.lo.iloc[0]), float(d.hi.iloc[0])

        # Catalogue-mean accuracy: macro over *all* species, so the two groups are
        # weighted by their actual prevalence. This is what locates the crossover.
        for arm, label in (("P-full", "cat_Pfull"), ("mixed", "cat_single"),
                           ("T1 two-head avg", "cat_T1")):
            allsp = np.concatenate([acc["reserved"][arm], acc["mixed"][arm]])
            row[label] = round(float(np.nanmean(allsp)), 4)
        row["T1_minus_single"] = round(row["cat_T1"] - row["cat_single"], 4)
        rows.append(row)
    return pd.DataFrame(rows).round(4)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="bioclip2")
    a = ap.parse_args()
    t = sweep(a.variant)
    print(f"\n== reserved-fraction sweep ({a.variant})")
    print("\n-- damage and gain, vs the Pl@ntNet-only head")
    print(t[["reserve_frac", "n_reserved", "single_reserved", "single_mixed",
             "T1_reserved", "T1_mixed", "T1_data_effect_reserved",
             "T1_data_lo", "T1_data_hi"]].to_string(index=False))
    print("\n-- catalogue-mean accuracy over all species (locates the crossover)")
    print(t[["reserve_frac", "cat_Pfull", "cat_single", "cat_T1",
             "T1_minus_single"]].to_string(index=False))
    t.to_csv(DATA_PROCESSED / f"source_mix_sweep_{a.variant}.csv", index=False)
