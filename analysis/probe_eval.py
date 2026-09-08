"""Does the adapted tower generalise to plants it never saw?

Pre-registered in the addendum to `ADAPT_PREREG.md`. Linear probe on species
outside Pl@ntNet-300K -- unseen by the fine-tune -- photographed on iNaturalist,
a source the adaptation never touched. Stock and adapted towers see identical
images and an identical split, so the only thing varying is the encoder.

Split is by *observation*, never by photo: two photos of one plant would
otherwise straddle train and test and inflate both arms.

Usage:
    PYTHONPATH=. .venv-mps/bin/python -m analysis.probe_eval --embed
    PYTHONPATH=. .venv/bin/python -m analysis.probe_eval
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

DIR = Path(__file__).parent / "probe_unseen"
VARIANTS = ("mobileclip2_s2", "mobileclip2_s2_ft")
MIN_OBS = 8
N_BOOT = 2000
SEED = 0


def embed(variant):
    from plantid.features.pretrained import embed_images, load_encoder
    df = pd.read_parquet(DIR / "manifest.parquet")
    model, preprocess, device = load_encoder(variant)
    X = embed_images(list(df["local_path"]), model, preprocess, device,
                     batch_size=64, desc=f"probe/{variant}")
    np.savez_compressed(DIR / f"emb_{variant}.npz",
                        descriptor=np.asarray(X, dtype="float32"),
                        local_path=df["local_path"].to_numpy().astype(str))
    print(f"{variant}: {np.asarray(X).shape}", flush=True)


def load(variant):
    z = np.load(DIR / f"emb_{variant}.npz", allow_pickle=True)
    X = z["descriptor"].astype(np.float32)
    X /= np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    return pd.Series(range(len(X)), index=z["local_path"]), X


def probe(X, df, tr_obs, te_obs):
    tr = df["obs_id"].isin(tr_obs).to_numpy()
    te = df["obs_id"].isin(te_obs).to_numpy()
    clf = LogisticRegression(max_iter=3000, C=10.0).fit(X[tr], df["species_name"][tr])
    hit = (clf.predict(X[te]) == df["species_name"][te].to_numpy()).astype(float)
    return hit, df["species_name"][te].to_numpy()


def main():
    df = pd.read_parquet(DIR / "manifest.parquet")
    keep = df.groupby("species_name")["obs_id"].transform("nunique") >= MIN_OBS
    df = df[keep].reset_index(drop=True)
    print(f"{len(df)} photos | {df.species_name.nunique()} species | "
          f"{df.obs_id.nunique()} observations (>= {MIN_OBS} obs/species)")

    rng = np.random.default_rng(SEED)
    tr_obs, te_obs = set(), set()
    for _, g in df.groupby("species_name"):
        o = np.asarray(g["obs_id"].unique(), dtype=object)
        rng.shuffle(o)
        cut = max(1, int(0.6 * len(o)))
        tr_obs |= set(o[:cut]); te_obs |= set(o[cut:])

    out = {}
    for v in VARIANTS:
        pos, X = load(v)
        Xa = X[[pos[p] for p in df["local_path"]]]
        out[v] = probe(Xa, df, tr_obs, te_obs)

    a_hit, truth = out["mobileclip2_s2"]
    b_hit, _ = out["mobileclip2_s2_ft"]
    print(f"\nstock   {a_hit.mean():.4f}")
    print(f"adapted {b_hit.mean():.4f}")

    # paired, resampling species -- photos of one species are not independent
    uniq = np.array(sorted(set(truth)))
    idx = {s: np.flatnonzero(truth == s) for s in uniq}
    r = np.random.RandomState(SEED)
    d = b_hit - a_hit
    draws = [d[np.concatenate([idx[s] for s in r.choice(uniq, len(uniq), replace=True)])].mean()
             for _ in range(N_BOOT)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    print(f"adapted - stock  {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  "
          f"({len(uniq)} species, {len(truth)} test photos)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embed", action="store_true")
    a = ap.parse_args()
    if a.embed:
        for v in VARIANTS:
            embed(v)
    else:
        main()
