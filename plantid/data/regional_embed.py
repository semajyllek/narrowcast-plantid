"""Regional photographs -> a narrowcast `--embeddings` file, cluster included.

The catalogue seam (`analysis/export_for_narrowcast.py`) writes `descriptor` and
`label` and deliberately omits `cluster`, because Pl@ntNet has no per-individual
grouping and inventing one would be worse than declaring its absence.

Regional data is the opposite case. `plantid.data.regional_fetch` records the GBIF
occurrence key, so several photographs of one plant are identifiable as such, and
**that column has to survive into the `.npz` or the whole chain of guarantees
breaks**: `narrowcast.cascade.make_splits` would put photographs of one individual
on both sides of a train/test boundary, and `cluster_bootstrap` would resample
photographs rather than plants. `CLAUDE.md`'s first convention -- "cluster, never
row" -- exists because row-level intervals have twice produced effects in this
project that failed to replicate.

There is no separate export step: narrowcast's `--embeddings` format is
`descriptor`, `label`, and optionally `group`, `cluster` and `origin`, so this
writes that directly.

`group` is written explicitly rather than left to narrowcast's default. The
default takes the first whitespace-delimited token, which is the right genus for a
Linnaean binomial -- but the caller's column wins by design, and being explicit
means a region that later groups by family or by edibility changes one line here
rather than depending on a string convention.

Usage:
    PYTHONPATH=. .venv-mps/bin/python -m plantid.data.regional_embed \\
        --manifest data/processed/regions/oregon/manifest.parquet \\
        --variant mobileclip2_s2 --out data/processed/regions/oregon_s2.npz

    PYTHONPATH=. .venv-mps/bin/python -m plantid.data.regional_embed \\
        --manifest .../manifest.parquet --coreml \\
        data/processed/coreml/mobileclip2_s2_q8_per_channel.mlpackage \\
        --variant mobileclip2_s2 --out .../oregon_cq8.npz
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from plantid.data.curation import curated_name


def group_of(label: str) -> str:
    """Genus, as the coarse rank the cascade retreats to."""
    parts = str(label).split()
    return parts[0] if parts else str(label)


def embed_manifest(manifest: Path, variant: str, root: Path,
                   coreml: Path | None = None, batch: int = 64) -> dict:
    """Embed every photograph in a regional manifest.

    Returns the npz payload rather than writing it, so the caller decides where it
    lands and tests can exercise this without a model.
    """
    df = pd.read_parquet(manifest)
    df = df[df["local_path"].notna()].reset_index(drop=True)
    paths = [str(root / p) for p in df["local_path"]]

    if coreml is not None:
        from plantid.deploy.embed_coreml import embed_paths, load_model
        X = embed_paths(load_model(str(coreml)), paths, desc="regional",
                        encoder=variant)
    else:
        from plantid.features.pretrained import embed_images, load_encoder
        model, preprocess, device = load_encoder(variant)
        X = embed_images(paths, model, preprocess, device, batch_size=batch,
                         desc="regional")

    labels = np.array([curated_name(s) or s for s in df["species_name"].astype(str)])
    return {
        "descriptor": np.asarray(X, dtype="float32"),
        "label": labels,
        "group": np.array([group_of(s) for s in labels]),
        # The occurrence, not the photograph. This is the column that makes the
        # intervals honest.
        "cluster": np.asarray(df["cluster"].astype(str)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--variant", required=True,
                    help="torch encoder variant; also selects preprocessing when "
                         "--coreml is given")
    ap.add_argument("--out", required=True)
    ap.add_argument("--coreml", default=None,
                    help="embed through a Core ML artifact instead of torch")
    ap.add_argument("--root", default=".",
                    help="prefix for local_path values in the manifest")
    a = ap.parse_args()

    payload = embed_manifest(Path(a.manifest), a.variant, Path(a.root),
                             coreml=Path(a.coreml) if a.coreml else None)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **payload)

    n_photos = len(payload["label"])
    n_plants = len(set(payload["cluster"].tolist()))
    n_labels = len(set(payload["label"].tolist()))
    print(f"\n{a.out}: {payload['descriptor'].shape}")
    print(f"  {n_labels} labels, {n_plants} plants, {n_photos} photographs "
          f"({n_photos / max(n_plants, 1):.1f} per plant)")
    print(f"  {len(set(payload['group'].tolist()))} groups")
    if n_plants == n_photos:
        print("  note: one photograph per plant, so clustering buys nothing here "
              "-- the intervals will be the same as row-level ones.")


if __name__ == "__main__":
    main()
