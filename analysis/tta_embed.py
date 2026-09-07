"""Test-time augmentation embeddings: the last inference-side lever that is free in bytes.

Views are deterministic -- identity, horizontal flip, and two centre crops with
their flips -- so the result is reproducible rather than seed-dependent. The
per-view L2-normalised embeddings are averaged and re-normalised, which is the
standard form for a linear probe on frozen features.

Costs nothing in bytes and ~6x in latency, which is the trade being measured.

Usage:
    PYTHONPATH=. .venv-mps/bin/python -m analysis.tta_embed --variant mobileclip2_s2
"""

import argparse

import numpy as np
import pandas as pd
from PIL import Image

from plantid.config import DATA_PROCESSED


def views(im):
    out = [im, im.transpose(Image.FLIP_LEFT_RIGHT)]
    w, h = im.size
    for frac in (0.85, 0.70):
        dw, dh = int(w * frac), int(h * frac)
        box = ((w - dw) // 2, (h - dh) // 2, (w - dw) // 2 + dw, (h - dh) // 2 + dh)
        c = im.crop(box)
        out += [c, c.transpose(Image.FLIP_LEFT_RIGHT)]
    return out


def main(variant, batch=64):
    import torch
    from plantid.features.pretrained import load_encoder

    obs = pd.read_parquet(DATA_PROCESSED / "inat_observations.parquet")
    paths = [p for ps in obs["local_paths"] for p in ps]
    model, preprocess, device = load_encoder(variant)

    acc, keep = [], []
    for start in range(0, len(paths), batch):
        chunk, ok = [], []
        for p in paths[start:start + batch]:
            try:
                chunk.append(Image.open(DATA_PROCESSED / p).convert("RGB"))
                ok.append(p)
            except Exception:
                continue
        if not chunk:
            continue
        per_view = []
        for vi in range(6):
            t = torch.stack([preprocess(views(im)[vi]) for im in chunk]).to(device)
            with torch.no_grad():
                # `load_encoder` returns a plain callable (`_ImageTower`), not a
                # CLIP model -- the variants do not share an encode_image API.
                f = model(t).float().cpu().numpy()
            per_view.append(f / np.clip(np.linalg.norm(f, axis=1, keepdims=True), 1e-12, None))
        m = np.mean(per_view, axis=0)
        acc.append(m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None))
        keep += ok
        print(f"  {min(start + batch, len(paths))}/{len(paths)}", end="\r", flush=True)

    X = np.vstack(acc).astype("float32")
    path = DATA_PROCESSED / f"inat_{variant}_tta.npz"
    np.savez_compressed(path, descriptor=X, path=np.asarray(keep, dtype=str))
    print(f"\n{path.name}: {X.shape} ({len(keep)} photos, 6 views each)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="mobileclip2_s2")
    a = ap.parse_args()
    main(a.variant)
