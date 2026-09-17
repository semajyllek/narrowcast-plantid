"""Does fitting a head on the user's 20 beat filtering a general classifier to them?

The product thesis is that a user who cares about <=20 plants should get something
*better at those 20* than a general model. That has never been tested directly.
What has been tested is narrow-vs-large at K=490 (COMPETITIVE_FINDINGS, a tie with
iNaturalist's 108k-taxa server model) -- which is a different claim.

The clean test is available because PlantCLEF2024's checkpoint carries its own
7,806-way classifier, which `pretrained.load_encoder` strips to expose features.
Restore it and two systems can be compared on identical photographs, identical
splits, identical embeddings:

  FILTER  general 7,806-way posterior, restricted to the user's labels, argmax
  FIT     logistic head trained on the user's labels over the same frozen features

If FILTER wins, narrowing is a packaging decision and the value of this project is
offline delivery and honest measurement, not accuracy. If FIT wins, the user's
labels carry information the general model does not have.

CLAUDE.md already records a related result -- "our own cascade applied to
Pl@ntNet's returned scores beats us; it is a technique, not a differentiator" --
so the prior is that FILTER does well.
"""
import sys; sys.path.insert(0, '.')
import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from plantid.data.curation import curated_name


def plantclef_head():
    """The 7,806-way classifier weights the feature loader discards."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    p = hf_hub_download('vincent-espitalier/dino-v2-reg4-with-plantclef2024-weights',
                        'vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all.safetensors')
    with safe_open(p, framework='pt') as f:
        return f.get_tensor('head.weight').float(), f.get_tensor('head.bias').float()


def class_names():
    """PlantCLEF class index -> species name, from the shipped label map."""
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download('vincent-espitalier/dino-v2-reg4-with-plantclef2024-weights',
                            'species_id_mapping.txt')
        names = [l.strip() for l in open(p) if l.strip()]
        return names
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--embeddings', default='data/processed/regions/oregon_pc24.npz',
                    help='regional vectors from the SAME encoder the head came from')
    ap.add_argument('--seeds', type=int, default=5)
    a = ap.parse_args()

    z = np.load(a.embeddings, allow_pickle=True)
    X = np.asarray(z['descriptor'], dtype='float32')
    y = np.array([curated_name(s) or s for s in z['label'].astype(str)])
    cl = np.asarray(z['cluster'], dtype=str)
    labels = sorted(set(y.tolist()))
    print(f"{len(X)} photographs, {len(labels)} labels, {len(set(cl))} plants\n")

    W, b = plantclef_head()
    names = class_names()
    if names is None or len(names) != W.shape[0]:
        print("no usable class mapping; FILTER cannot be scored")
        return

    # PlantCLEF ids are numeric strings in some releases; keep whatever maps.
    idx = {}
    for i, n in enumerate(names):
        c = curated_name(n) or n
        idx.setdefault(c, i)
    hit = [l for l in labels if l in idx]
    print(f"{len(hit)} of {len(labels)} labels found in PlantCLEF's 7,806 classes")
    if len(hit) < 5:
        print("too few to compare; PlantCLEF's label space does not cover this region")
        return

    keep = np.isin(y, hit)
    Xk, yk, clk = X[keep], y[keep], cl[keep]
    cols = torch.tensor([idx[l] for l in hit])
    logits_all = torch.tensor(Xk) @ W.T + b
    filt_pred = np.array(hit)[logits_all[:, cols].argmax(1).numpy()]

    rows = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        u = np.array(sorted(set(clk.tolist()))); rng.shuffle(u)
        tr = np.isin(clk, u[:len(u) // 2])
        clf = LogisticRegression(max_iter=3000, C=10.0,
                                 class_weight='balanced').fit(Xk[tr], yk[tr])
        rows.append(dict(seed=seed,
                         fit=float((clf.predict(Xk[~tr]) == yk[~tr]).mean()),
                         filter=float((filt_pred[~tr] == yk[~tr]).mean())))
    d = pd.DataFrame(rows)
    print(f"\n{'':8} {'FIT':>8} {'FILTER':>8}")
    print(f"{'mean':8} {d.fit.mean():>8.3f} {d['filter'].mean():>8.3f}")
    print(f"{'sd':8} {d.fit.std():>8.3f} {d['filter'].std():>8.3f}")
    print(f"\npaired FIT - FILTER: {(d.fit - d['filter']).mean():+.4f}")
    d.to_csv('data/processed/fit_vs_filter.csv', index=False)


if __name__ == '__main__':
    main()
