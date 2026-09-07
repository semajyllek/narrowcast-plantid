"""What is the best accuracy achievable at a small byte budget?

Pre-registered in `SMALL_FRONTIER_PREREG.md`. Every encoder experiment in this
repo holds the head fixed and varies the frozen features; this varies the
inference side, which costs few or no bytes and has never been measured here.

Observation-level species and genus top-1 on the standard evaluation set, plus
coverage at the declared 20% out-of-catalogue rate, so the numbers drop straight
into the table in `CLAUDE.md`.

Usage:
    PYTHONPATH=. .venv/bin/python -m analysis.small_frontier
"""

import argparse

import numpy as np
import pandas as pd

from plantid.config import DATA_PROCESSED
from plantid.eval.rejection import (
    IN_CATALOG, OOD_MIX_REGIONAL, build_observations, cluster_bootstrap, decide,
    deployment_weights, fit_thresholds, make_splits, precision_coverage,
)

P_OOD = 0.20
SEED = 0
SIZES = {"mobileclip2_s0": 5.7, "mobileclip2_s2": 17.9, "bioclip1_cml4": 46.0,
         "plantclef24": 43.3, "bioclip2_cml4": 160.0, "bioclip2": 152.0}


def observations(variant, C=10.0):
    """Per-observation posteriors and truth for one encoder, cached on disk."""
    path = DATA_PROCESSED / f"frontier_{variant}_C{C:g}.npz"
    meta = DATA_PROCESSED / f"frontier_{variant}_C{C:g}.parquet"
    if path.exists() and meta.exists():
        z = np.load(path, allow_pickle=True)
        return pd.read_parquet(meta), z["P"], z["classes"], z["mask"], z["gmat"], z["ug"]
    df, _, P, classes, mask, gmat, ug = build_observations(
        DATA_PROCESSED / f"inat_{variant}.npz", variant=variant, keep_posterior=True, C=C)
    np.savez_compressed(path, P=P, classes=classes, mask=mask, gmat=gmat, ug=ug)
    df.to_parquet(meta, index=False)
    return df, P, classes, mask, gmat, ug


def rescore(df, P, classes, mask, gmat, ug):
    """Recompute the cascade inputs from a (possibly combined) posterior."""
    cat = P[:, mask]
    sc = cat.max(1)
    gc = (cat @ gmat.T).max(1)
    pred = classes[mask][cat.argmax(1)]
    gpred = ug[(cat @ gmat.T).argmax(1)]
    out = df.copy()
    out["species_conf"], out["genus_conf"] = sc, gc
    out["species_ok"] = pred == out["species"].to_numpy()
    out["genus_ok"] = gpred == out["genus"].to_numpy()
    return out


def measure(df):
    """Closed-set accuracy on in-catalogue rows, and the fitted three-way answer."""
    inc = df["bucket"] == IN_CATALOG
    fold = make_splits(df, seed=SEED)
    cal, te = df[fold == "calib"], df[fold == "test"]
    w = deployment_weights(cal["bucket"].to_numpy(), p_ood=P_OOD, ood_mix=OOD_MIX_REGIONAL)
    (tg, ts), _ = fit_thresholds(
        cal["species_conf"].to_numpy(), cal["genus_conf"].to_numpy(),
        cal["species_ok"].to_numpy(), cal["genus_ok"].to_numpy(),
        cal["in_catalog"].to_numpy(), sample_weight=w)
    lv = decide(te["species_conf"].to_numpy(), te["genus_conf"].to_numpy(), tg, ts)
    prec, cov = precision_coverage(lv, te["species_ok"].to_numpy(), te["genus_ok"].to_numpy(),
                                   te["bucket"].to_numpy(), p_ood=P_OOD,
                                   ood_mix=OOD_MIX_REGIONAL)
    sp = df.loc[inc, "species_ok"].to_numpy().astype(float)
    gn = df.loc[inc, "genus_ok"].to_numpy().astype(float)
    cl = df.loc[inc, "species"].to_numpy()
    return {"species": round(float(sp.mean()), 4),
            "species_ci": [round(x, 4) for x in cluster_bootstrap(sp, cl, seed=SEED)],
            "genus": round(float(gn.mean()), 4),
            "genus_ci": [round(x, 4) for x in cluster_bootstrap(gn, cl, seed=SEED)],
            "precision": round(prec, 4), "coverage": round(cov, 4)}


def main(arms):
    cache = {}
    rows = []
    for name, spec in arms:
        if isinstance(spec, list):                       # ensemble of variants
            parts = []
            for v in spec:
                if v not in cache:
                    cache[v] = observations(v)
                parts.append(cache[v])
            df, _, classes, mask, gmat, ug = parts[0]
            # Posteriors averaged, not logits: the heads are separately fitted and
            # their logit scales are not comparable.
            P = np.mean([p[1] for p in parts], axis=0)
            size = round(sum(SIZES[v] for v in spec), 1)
        else:
            variant, C = spec
            key = (variant, C)
            if key not in cache:
                cache[key] = observations(variant, C=C)
            df, P, classes, mask, gmat, ug = cache[key]
            size = SIZES[variant]
        m = measure(rescore(df, P, classes, mask, gmat, ug))
        rows.append({"arm": name, "MB": size, **m})
        print(f"  {name:26s} {size:6.1f} MB  species {m['species']:.4f} "
              f"genus {m['genus']:.4f}  cov@20% {m['coverage']:.3f}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    arms = [
        ("mobileclip2_s0", ("mobileclip2_s0", 10.0)),
        ("mobileclip2_s2", ("mobileclip2_s2", 10.0)),
        ("s2, C=1", ("mobileclip2_s2", 1.0)),
        ("s2, C=100", ("mobileclip2_s2", 100.0)),
        ("s0 + s2 ensemble", ["mobileclip2_s0", "mobileclip2_s2"]),
        ("plantclef24", ("plantclef24", 10.0)),
        ("bioclip2_cml4", ("bioclip2_cml4", 10.0)),
    ]
    t = main(arms)
    print()
    print(t.to_string(index=False))
    t.to_csv(DATA_PROCESSED / "small_frontier.csv", index=False)
