"""Build a mixed Pl@ntNet + iNaturalist manifest that `narrowcast fit` can sweep.

This exists to answer one question end to end: **given a metric, a target, and a
dataset, does narrowcast hand back the smallest model that hits the target?**
`analysis/export_for_narrowcast.py` writes *embeddings*, which pins the encoder
and makes `fit` refuse to sweep -- correctly, since scoring one vector file
through N candidates would print the same number N times. A sweep needs pixels,
so this writes a `--manifest` of paths.

Both corpora, with `origin` set, because that is the case the tool was built for:
Pl@ntNet is where the labels are plentiful and iNaturalist is what deployment
looks like. `cluster` is the iNaturalist observation where one exists and the
image id otherwise, so several photographs of one plant cannot straddle a split.

Usage:
    PYTHONPATH=. .venv/bin/python -m analysis.make_mixed_manifest \\
        --k 20 --arm separated --out /tmp/mixed
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from plantid.config import DATA_PROCESSED
from plantid.data.curation import canonical_name

ORGANS = ("leaf", "flower")


def plantnet_rows():
    d = pd.read_parquet(Path(DATA_PROCESSED) / "catalog_index.parquet")
    d = d[d.organ.isin(ORGANS) & d.local_path.notna()].copy()
    d["label"] = [canonical_name(s) for s in d.species_name]
    d["path"] = [str(Path(DATA_PROCESSED) / p) for p in d.local_path]
    d["cluster"] = d.image_id.astype(str)          # one image per plant; no grouping
    d["origin"] = "plantnet"
    return d[["label", "path", "cluster", "origin"]]


def inat_rows():
    m = pd.read_parquet(Path(DATA_PROCESSED) / "inat_observations.parquet")
    m["label"] = [canonical_name(s) for s in m.species_name.astype(str)]
    img = Path(DATA_PROCESSED) / "images_inat"
    out = []
    for r in m.itertuples():
        for p in (r.local_paths if r.local_paths is not None else []):
            f = img / Path(p).name
            if f.exists():
                # The observation is the cluster: several photographs of one plant
                # are not independent rows, and letting them straddle a split is
                # the error this project's first convention exists to prevent.
                out.append((r.label, str(f), str(r.obs_id), "inat"))
    return pd.DataFrame(out, columns=["label", "path", "cluster", "origin"])


def pick(rows, k, arm, seed=0):
    """K labels, either spread across genera or drawn from as few as possible."""
    rows = rows.copy()
    rows["genus"] = rows.label.str.split().str[0]
    per = rows.groupby("label").size()
    ok = rows[rows.label.map(per) >= 60]
    rng = np.random.default_rng(seed)
    if arm == "crowded":
        big = ok.groupby("genus")["label"].nunique().sort_values(ascending=False)
        chosen = []
        for g in big.index:
            chosen += sorted(set(ok[ok.genus == g].label))
            if len(chosen) >= k:
                break
        return chosen[:k]
    genera = rng.permutation(sorted(set(ok.genus)))[:k]
    return sorted(rng.choice(sorted(set(ok[ok.genus == g].label))) for g in genera)


def main(a):
    pn, inat = plantnet_rows(), inat_rows()
    both = pd.concat([pn, inat], ignore_index=True)
    # Only labels present in *both* corpora, so `origin` means something and the
    # deployment-origin question is answerable rather than vacuous.
    shared = set(pn.label) & set(inat.label)
    both = both[both.label.isin(shared)]
    print(f"{len(shared)} labels present in both corpora", flush=True)

    labels = pick(both, a.k, a.arm)
    sel = both[both.label.isin(labels)]
    # Everything else becomes the reject class -- the "narrow subset, rest is OOD"
    # framing. Capped so the negatives do not swamp the fit.
    rest = both[~both.label.isin(labels)]
    bg = rest.sample(n=min(a.background, len(rest)), random_state=0)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sel.to_parquet(out / "data.parquet", index=False)
    bg.to_parquet(out / "background.parquet", index=False)
    print(f"arm {a.arm}, K={len(labels)}: {len(sel)} rows "
          f"({(sel.origin == 'inat').sum()} iNat, {(sel.origin == 'plantnet').sum()} Pl@ntNet)")
    print(f"  background {len(bg)} rows from {rest.label.nunique()} other labels")
    print(f"  labels: {', '.join(labels)}")
    print(f"  wrote {out}/data.parquet and {out}/background.parquet")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--arm", choices=("separated", "crowded"), default="separated")
    ap.add_argument("--background", type=int, default=2500)
    ap.add_argument("--out", default="/tmp/mixed")
    sys.exit(main(ap.parse_args()))
