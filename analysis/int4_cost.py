"""What does int4 cost at K=20? Leaf-only, torch vs the shipped Core ML artifact.

Same species, same draws, same splits; only the embedding source differs. Closed-set
top-1 needs no negatives, so this runs off the leaf caches alone while the flower
embed is still going.
"""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
import analysis.headroom_arms as H
from plantid.data.curation import curated_name
from sklearn.linear_model import LogisticRegression

def leaf(variant):
    z = np.load(f"data/processed/catalog_leaf_{variant}.npz", allow_pickle=True)
    E = z["descriptor"].astype("float32")
    E = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)
    names = np.asarray([curated_name(n) or n for n in z["species_name"].astype(str)])
    return E, names, z["split"].astype(str)

def arms(variant, allsp, K=20, nsets=6):
    E, names, split = leaf(variant)
    out = []
    for crowded in (False, True):
        sets = H.draw_label_sets(allsp, K, nsets,
                                 np.random.default_rng(K + 7*crowded), crowded)
        for si, sp in enumerate(sets):
            m = np.isin(names, sp)
            tr, te = m & (split == "train"), m & (split == "test")
            if te.sum() < 20 or len(set(names[tr])) < K:
                continue
            clf = LogisticRegression(max_iter=3000, C=10.0,
                                     class_weight="balanced").fit(E[tr], names[tr])
            out.append(dict(variant=variant, crowded=crowded, set_id=si,
                            top1=float((clf.predict(E[te]) == names[te]).mean()),
                            n_test=int(te.sum())))
    return out

_, names_ref, _ = leaf("mobileclip2_s2")
allsp = np.array(sorted(set(names_ref)))
print(f"species pool: {len(allsp)}", flush=True)
rows = []
for v in ("mobileclip2_s2", "mobileclip2_s2_cml4"):
    rows += arms(v, allsp); print(f"  {v} done", flush=True)
d = pd.DataFrame(rows); d.to_csv("data/processed/int4_cost_leaf.csv", index=False)
print()
piv = d.pivot_table(index="crowded", columns="variant", values="top1")
piv["delta"] = piv["mobileclip2_s2_cml4"] - piv["mobileclip2_s2"]
print(piv.round(4).to_string()); print()
m = d.pivot_table(index=["crowded","set_id"], columns="variant", values="top1").dropna()
delta = m["mobileclip2_s2_cml4"] - m["mobileclip2_s2"]
print(f"paired over {len(delta)} label sets: mean {delta.mean():+.4f}  "
      f"min {delta.min():+.4f}  max {delta.max():+.4f}")
