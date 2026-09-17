"""Regional embeddings -> a narrowcast `--scores` file, so near-OOD can be measured.

`narrowcast.build.load_rows` -- the `--embeddings` path -- sets
`counts["near_ood"] = 0` unconditionally. Only `load_scored` buckets out-of-list
rows by group membership. So on the embeddings path the *relatives* bucket does
not exist, and every regional number produced so far has been blind to it.

That bucket is the product. A 200-species list drawn from Oregon's 3,893 leaves
3,693 species off it, and most of what someone photographs in the field will be an
unlisted relative of something listed rather than an unrelated plant. It is also
the weakest bucket wherever it has been measured: `NEAR_OOD_FINDINGS.md` puts 22%
of near-OOD observations answered *wrong*, and every posterior-derived score is
closed-set and structurally cannot say "none of these" -- the within-genus ratio
scored AUROC 0.472, worse than chance.

This fits the head here and emits posteriors, so narrowcast does the bucketing it
already knows how to do.

**The head is trained with background negatives and `__OTHER__` is emitted.**
Without a reject class the posterior over in-list labels sums to 1 whatever the
input, so an out-of-list photograph produces a confident wrong answer and no
threshold can separate it. That is the same reason `load_rows` requires
`--background-embeddings` before it will report a model that can decline.

**Scored rows are disjoint from the head's training clusters.** The split is on
`cluster`, so no photograph of a plant used to fit the head is scored. narrowcast
then makes its own calibration/test split from what it is given.

Usage:
    PYTHONPATH=. .venv/bin/python -m plantid.data.regional_scores \\
        --in-list  data/processed/regions/oregon_pc24.npz \\
        --near-ood data/processed/regions/oregon_nearood_pc24.npz \\
        --far-ood  data/processed/regions/oregon_farood_pc24.npz \\
        --background /tmp/bg_pc24.npz \\
        --out data/processed/regions/oregon_scores.npz
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

OTHER = "__OTHER__"


def _load(path):
    z = np.load(path, allow_pickle=True)
    encoder = str(np.asarray(z["encoder"]).ravel()[0]) if "encoder" in z.files else None
    X = np.asarray(z["descriptor"], dtype="float32")
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    n = len(X)
    label = np.asarray(z["label"], dtype=str) if "label" in z.files else \
        np.full(n, OTHER)
    group = (np.asarray(z["group"], dtype=str) if "group" in z.files
             else np.array([str(s).split()[0] if str(s).split() else str(s)
                            for s in label]))
    cluster = (np.asarray(z["cluster"], dtype=str) if "cluster" in z.files
               else np.array([f"{Path(path).stem}-{i}" for i in range(n)]))
    return X, label, group, cluster, encoder


def build_scores(in_list, near_ood=None, far_ood=None, background=None,
                 seed=0, C=10.0, train_frac=0.5):
    """Fit a head on half the in-list clusters; score the rest plus every OOD row."""
    X, y, g, cl, enc = _load(in_list)
    declared = {in_list: enc}
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(cl.tolist())))
    rng.shuffle(uniq)
    train_clusters = set(uniq[: int(train_frac * len(uniq))].tolist())
    tr = np.array([c in train_clusters for c in cl])

    Xtr, ytr = [X[tr]], [y[tr]]
    if background is not None:
        Xb, _, _, _, enc_b = _load(background)
        declared[background] = enc_b
        cut = rng.permutation(len(Xb))
        n_tr = int(0.6 * len(Xb))
        Xtr.append(Xb[cut[:n_tr]])
        ytr.append(np.full(n_tr, OTHER))

    clf = LogisticRegression(max_iter=3000, C=C,
                             class_weight="balanced").fit(np.vstack(Xtr),
                                                          np.concatenate(ytr))
    classes = np.asarray(clf.classes_, dtype=str)

    # Everything the head did not train on: held-out in-list rows, then the OOD
    # sets. Their labels are their *real* species -- narrowcast decides which are
    # out-of-list by testing membership of `classes`, and buckets them near or
    # distant by whether the group appears in-list.
    Xe, ye, ge, ce = [X[~tr]], [y[~tr]], [g[~tr]], [cl[~tr]]
    regional = [np.zeros(int((~tr).sum()), bool)]
    for path in (near_ood, far_ood):
        if path is None:
            continue
        Xo, yo, go, co, enc_o = _load(path)
        declared[path] = enc_o
        Xe.append(Xo); ye.append(yo); ge.append(go); ce.append(co)
        # Far-OOD here is drawn from the SAME PLACE as everything else --
        # `oregon_farood.json` carries `place_name: Oregon, US` -- so these are
        # plants a user in the deployment region could actually photograph, not
        # the global filler `distant_ood` stands for. narrowcast calls that
        # `regional_ood` and anchors the operating point to it; without the flag
        # it buckets them as unrelated inputs and the card says so, which
        # understates the difficulty of exactly the rows that make it honest.
        regional.append(np.full(len(Xo), path == far_ood))

    # One encoder, or refuse. This is the point where separately embedded pools
    # are combined, so it is where a Core ML pool meets a torch one -- the failure
    # that cost three points of label share, silently. Comparing declarations is
    # the only check that sees it (`SPACE_CHECK_FINDINGS.md`: the geometric test
    # catches 0 of 21 such pairs).
    named = {p: e for p, e in declared.items() if e}
    if len(set(named.values())) > 1:
        raise SystemExit(
            "these inputs were embedded with different encoders:\n  "
            + "\n  ".join(f"{e!r}  {p}" for p, e in sorted(named.items(), key=lambda kv: kv[1]))
            + "\nRe-embed them all with one. Mixing an export with its own "
              "original flatters label share and nothing else will catch it.")
    if len(named) < len(declared):
        print(f"  note: {len(declared) - len(named)} input(s) declare no encoder; "
              "re-run regional_embed to record it", flush=True)

    Xe = np.vstack(Xe)
    out = {
        "proba": clf.predict_proba(Xe).astype("float32"),
        "classes": classes,
        "label": np.concatenate(ye),
        "group": np.concatenate(ge),
        "cluster": np.concatenate(ce),
        "regional": np.concatenate(regional),
    }
    if named:
        out["encoder"] = next(iter(named.values()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-list", required=True)
    ap.add_argument("--near-ood")
    ap.add_argument("--far-ood")
    ap.add_argument("--background")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    p = build_scores(a.in_list, a.near_ood, a.far_ood, a.background, seed=a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **p)

    # Use the GROUP COLUMN, not the first token of the label. Taking the genus
    # from the label string is the bug this project has now hit four times --
    # narrowcast fixed it in score_frame, predict and labels.analyse, it is still
    # live at build.py:408, and this summary had it too. With --group-by family it
    # reports zero relatives while narrowcast correctly buckets 588, because
    # "Conium" is not "Apiaceae".
    listed = set(p["classes"].tolist()) - {OTHER}
    lab, grp = p["label"], p["group"]
    listed_groups = {g for l, g in zip(lab, grp) if l in listed}
    in_list = np.isin(lab, list(listed))
    near = ~in_list & np.isin(grp, list(listed_groups))
    print(f"\n{a.out}: proba {p['proba'].shape}, {len(listed)} labels")
    print(f"  in-list      {int(in_list.sum()):>5} rows")
    print(f"  near-OOD     {int(near.sum()):>5} rows  "
          f"({len(set(lab[near].tolist()))} unlisted congeners)")
    print(f"  distant-OOD  {int((~in_list & ~near).sum()):>5} rows")
    print(f"  clusters     {len(set(p['cluster'].tolist())):>5}")


if __name__ == "__main__":
    main()
