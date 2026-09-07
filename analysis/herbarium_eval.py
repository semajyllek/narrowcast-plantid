"""The production head, scored on pressed herbarium specimens.

Pre-registered in `HERBARIUM_PREREG.md`. Same estimator as the product arm of
`analysis/domain_shift_product.py` -- the shipped organ-routed head, photo-level,
`__OTHER__` masked out of the argmax, macro-averaged over species,
species-clustered bootstrap -- so Pl@ntNet, iNaturalist and herbarium sit in one
table.

Embedding needs `.venv-mps`; scoring needs only `.venv`.

Usage:
    PYTHONPATH=. .venv-mps/bin/python -m analysis.herbarium_eval --embed --variants bioclip2
    PYTHONPATH=. .venv/bin/python -m analysis.herbarium_eval --variants bioclip2
"""

import argparse

import numpy as np
import pandas as pd

from plantid.config import DATA_PROCESSED
from plantid.data.curation import curated_name
from plantid.eval.inat_fusion import OTHER, _l2, build_heads, build_router, photo_posteriors

N_BOOT = 2000
SEED = 0


def cache_path(variant):
    return DATA_PROCESSED / f"herbarium_{variant}.npz"


def embed(variant):
    from plantid.features.pretrained import embed_images, load_encoder
    idx = pd.read_parquet(DATA_PROCESSED / "herbarium_index.parquet")
    paths = [str(DATA_PROCESSED / p) for p in idx["local_path"]]
    model, preprocess, device = load_encoder(variant)
    X = embed_images(paths, model, preprocess, device, batch_size=64, desc="herbarium")
    np.savez_compressed(cache_path(variant), descriptor=np.asarray(X, dtype="float32"),
                        path=np.asarray(idx["local_path"], dtype=str))
    print(f"{cache_path(variant).name}: {np.asarray(X).shape}", flush=True)


def score(variant):
    heads, proj, classes = build_heads(variant=variant)
    router, _ = build_router(variant=variant)
    oi = int(np.flatnonzero(classes == OTHER)[0])

    z = np.load(cache_path(variant))
    E = _l2(z["descriptor"].astype(np.float32))
    idx = pd.read_parquet(DATA_PROCESSED / "herbarium_index.parquet").set_index("local_path")
    truth = np.array([curated_name(idx.loc[p, "species_name"]) or idx.loc[p, "species_name"]
                      for p in z["path"]])
    keep = np.isin(truth, classes)
    E, truth = E[keep], truth[keep]

    P, organs, _ = photo_posteriors(E, heads, proj, router, len(classes))
    other = P[:, oi].copy()
    P[:, oi] = -1.0
    pred = classes[P.argmax(1)]

    df = pd.DataFrame({
        "sp": truth,
        "species": (pred == truth).astype(float),
        "genus": (np.array([p.split()[0] for p in pred])
                  == np.array([t.split()[0] for t in truth])).astype(float),
        "other": other,
    })
    per = df.groupby("sp").mean(numeric_only=True)
    species = sorted(per.index)

    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(species), (N_BOOT, len(species)))
    out = {"variant": variant, "n_species": len(species), "n_photos": int(keep.sum())}
    for m in ("species", "genus", "other"):
        v = per[m].to_numpy()
        lo, hi = np.percentile(v[draws].mean(1), [2.5, 97.5])
        out[m] = round(float(np.nanmean(v)), 4)
        out[f"{m}_lo"], out[f"{m}_hi"] = round(float(lo), 4), round(float(hi), 4)
    # The router has no defined answer for a pressed whole plant; what it does
    # with one is part of what is being measured, so it is reported.
    # `organs` is already aligned to the kept rows -- E was filtered before scoring.
    out["routed"] = pd.Series(organs).value_counts(normalize=True).round(3).to_dict()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=["bioclip2"])
    ap.add_argument("--embed", action="store_true")
    a = ap.parse_args()

    rows = []
    for v in a.variants:
        if a.embed:
            embed(v)
            continue
        r = score(v)
        print(f"\n== {v}: {r['n_photos']} sheets over {r['n_species']} species")
        print(f"   species {r['species']:.4f} [{r['species_lo']}, {r['species_hi']}]"
              f" | genus {r['genus']:.4f} [{r['genus_lo']}, {r['genus_hi']}]"
              f" | P(OTHER) {r['other']:.4f}")
        print(f"   routed organ: {r['routed']}")
        rows.append(r)
    if rows:
        pd.DataFrame(rows).to_csv(DATA_PROCESSED / "herbarium_eval.csv", index=False)
        print(f"\nwrote {DATA_PROCESSED / 'herbarium_eval.csv'}")
