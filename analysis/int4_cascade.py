"""What int4 costs the *product* metric, not just top-1.

`analysis/int4_cost.py` measured closed-set top-1 and found -5.5pp. That is not the
number that decides anything: the headline is label share, which needs negatives and
a fitted threshold, and the two can diverge violently. TINY_FINDINGS has a 0.14 MB
student at top-1 0.471 naming *zero* labels while a 1.53 MB student at 0.492 named
17.4%, because capacity buys sharpness as well as accuracy and the threshold reads
sharpness.

Leaf only, because that is the organ embedded through Core ML on both sides at the
time of writing. Same species, same draws, same splits, same declared p_ood; only
the embedding source differs.
"""
import sys; sys.path.insert(0, '.')
import argparse
import numpy as np
import pandas as pd

import analysis.headroom_arms as H

H.ORGANS = ["leaf"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--sets", type=int, default=6)
    ap.add_argument("--p-ood", type=float, default=0.20)
    ap.add_argument("--out", default="data/processed/int4_cascade_leaf.csv")
    a = ap.parse_args()

    ref, _ = H.load("mobileclip2_s2")
    allsp = np.array(sorted(set(ref["leaf"][1])))
    rows = []
    for variant in ("mobileclip2_s2", "mobileclip2_s2_cml4"):
        cat, bg = H.load(variant)
        for crowded in (False, True):
            sets = H.draw_label_sets(allsp, a.k, a.sets,
                                     np.random.default_rng(a.k + 7 * crowded), crowded)
            for si, sp in enumerate(sets):
                arm = H.fit_arm(cat, bg, sp, np.random.default_rng(si))
                gmap = {s: s.split()[0] for s in sp}
                r = H.score(H.frame(arm, gmap),
                            dict(variant=variant, crowded=crowded, set_id=si),
                            p_ood=a.p_ood)
                if r:
                    rows.append(r)
        print(f"  {variant} done", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}: {len(d)} rows\n")
    for m in ("fine", "label_share", "coverage", "precision", "decline_share"):
        piv = d.pivot_table(index="crowded", columns="variant", values=m)
        piv["delta"] = piv["mobileclip2_s2_cml4"] - piv["mobileclip2_s2"]
        print(f"=== {m} ===");  print(piv.round(4).to_string()); print()
    paired = d.pivot_table(index=["crowded", "set_id"], columns="variant",
                           values="label_share").dropna()
    dl = paired["mobileclip2_s2_cml4"] - paired["mobileclip2_s2"]
    print(f"label_share paired over {len(dl)} sets: mean {dl.mean():+.4f} "
          f"min {dl.min():+.4f} max {dl.max():+.4f}")


if __name__ == "__main__":
    main()
