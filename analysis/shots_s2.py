"""How many images per species does the 17.9 MB encoder actually need?

The "0.9345 on two training images per species" figure quoted for this product
comes from a sweep over `bioclip2` and `mobileclip2_s0` (SOURCE_MIX_MIDDLE_FINDINGS
§"Capping alone could not do it"). It has never been measured on MobileCLIP2-S2,
which is the encoder TINY_K_FINDINGS recommends and the one a regional bundle would
ship. That number sets the entire per-region fetch budget, so it is worth its own
measurement rather than an inherited one.

Caps training rows per species and scores through the unmodified cascade at the
declared p_ood, on the same varied/crowded label sets the rest of the project uses.
"""
import sys; sys.path.insert(0, '.')
import argparse
import numpy as np
import pandas as pd

import analysis.headroom_arms as H

SHOTS = [2, 4, 8, 16, 32, None]          # None = every training row available
ENCODERS = ["mobileclip2_s2", "mobileclip2_s2_ft", "bioclip2"]


def capped_arm(cat, bg, species, rng, shots):
    """`fit_arm`, but with at most `shots` training rows per species.

    The cap is applied to the catalogue's own train split before fitting, so the
    evaluation half is untouched and arms at different shot counts are scored on
    identical rows.
    """
    if shots is None:
        return H.fit_arm(cat, bg, species, rng)
    capped = {}
    for organ, (E, names, split) in cat.items():
        keep = np.ones(len(names), bool)
        for sp in species:
            idx = np.flatnonzero((names == sp) & (split == "train"))
            if len(idx) > shots:
                drop = rng.permutation(idx)[shots:]
                keep[drop] = False
        capped[organ] = (E[keep], names[keep], split[keep])
    return H.fit_arm(capped, bg, species, rng)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--sets", type=int, default=3)
    ap.add_argument("--p-ood", type=float, default=0.20)
    ap.add_argument("--out", default="data/processed/shots_s2.csv")
    a = ap.parse_args()

    ref, _ = H.load("bioclip2")
    allsp = np.array(sorted(set(ref["leaf"][1]) | set(ref["flower"][1])))
    rows = []
    for enc in ENCODERS:
        cat, bg = H.load(enc)
        for crowded in (False, True):
            sets = H.draw_label_sets(allsp, a.k, a.sets,
                                     np.random.default_rng(a.k + 7 * crowded), crowded)
            for si, sp in enumerate(sets):
                for shots in SHOTS:
                    arm = capped_arm(cat, bg, sp, np.random.default_rng(si), shots)
                    gmap = {s: s.split()[0] for s in sp}
                    r = H.score(H.frame(arm, gmap),
                                dict(encoder=enc, K=a.k, crowded=crowded,
                                     shots=shots if shots else 9999,
                                     set_id=f"{'c' if crowded else 'v'}{si}"),
                                p_ood=a.p_ood)
                    if r:
                        rows.append(r)
            print(f"  {enc:20s} crowded={crowded} done", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}: {len(d)} rows\n")
    for metric in ("fine", "label_share"):
        print(f"=== {metric} ===")
        print(d.pivot_table(index=["encoder"], columns=["crowded", "shots"],
                            values=metric).round(3).to_string())
        print()


if __name__ == "__main__":
    main()
