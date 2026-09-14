"""Does headroom govern the coarse-answer trap? (HEADROOM_PREREG.md)

The hypothesis in `narrowcast-kws` is that *headroom* -- coarse-rank accuracy
minus fine-rank accuracy -- decides whether a crowded label set retreats to the
group rank. It rests on five arms in which coarse accuracy never leaves
[0.93, 1.00], so headroom is nearly `1 - fine` rescaled and three rival
predictors are collinear. Adding arms of that shape confirms it without testing
it.

The lever here is the **grouping**, which the label head never sees. Holding the
label set, the encoder and the fitted head fixed and changing only the group
assignment holds fine accuracy *exactly* constant while coarse accuracy sweeps
across and below the 0.889 break-even implied by the declared `UTILITY`. One
fitted head therefore yields a whole row of arms that differ in coarse accuracy
and in nothing else.

Predictors are measured on the calibration rows and outcomes on held-out test
rows, so what comes out is an ex-ante rule: measure headroom on the data you fit
thresholds with, before you know what the deployment will answer.

`p_ood`, the assumed out-of-catalogue prevalence, was a module constant when the
published 1,409 arms were scored. That is why the rule fitted to them carries no
term for it, and why `1.8 x headroom` is quotable only with an operating point
attached -- see HEADROOM_FINDINGS.md's third qualification. `--p-ood` makes it an
axis. Scoring each arm at several operating points is nearly free, because
`p_ood` enters only at `deployment_weights`: the embedding, the label set, the
head and the grouping are all built once and re-scored.

Usage:
    PYTHONPATH=. .venv/bin/python -m analysis.headroom_arms --out data/processed/headroom_arms.csv
    PYTHONPATH=. .venv/bin/python -m analysis.headroom_arms --analyse data/processed/headroom_arms.csv

    # the operating-point sweep
    PYTHONPATH=. .venv/bin/python -m analysis.headroom_arms \\
        --p-ood 0.05 0.1 0.2 0.4 0.6 --out data/processed/headroom_ood.csv
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression

from plantid.eval.rejection import (
    DECLINE,
    GENUS,
    SPECIES,
    UTILITY,
    decide,
    deployment_weights,
    fit_thresholds,
    make_splits,
)
from plantid.features.embed_background import catalog_species, load_background

DP = "data/processed"
ORGANS = ["leaf", "flower"]
OTHER = "__OTHER__"
P_OOD = 0.20
OOD_MIX = {"near_ood": 0.32, "distant_ood": 0.68}
BG_TRAIN_FRAC = 0.6
N_SPLITS = 5              # per-arm outcomes are averaged over this many calib/test splits
ENCODERS = ["bioclip2", "bioclip1", "bioclip1_cml4", "mobileclip2_s0", "bioclip_inat"]
K_GRID = [14, 30, 60, 100]
# Extended past 30 so coherent groupings, not only the incoherent `random` ones,
# populate the region below the break-even. Otherwise the threshold test in P3 is
# confounded with group coherence.
KMEANS_GRID = [2, 5, 10, 20, 30, 40, 50, 70, 90]
RANDOM_GRID = [5, 10]

# The group answer's break-even under UTILITY: 0.5*p - 4*(1-p) > 0.
BREAKEVEN = -UTILITY["wrong"] / (UTILITY["genus_correct"] - UTILITY["wrong"])


def _l2(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def load(variant):
    cs = catalog_species()
    cat, bg = {}, {}
    for organ in ORGANS:
        d = np.load(f"{DP}/catalog_{organ}_{variant}.npz", allow_pickle=True)
        cat[organ] = (_l2(d["descriptor"]), d["species_name"].astype(str),
                      d["split"].astype(str))
        b = load_background(organ, exclude_species=cs, variant=variant)
        bg[organ] = (_l2(b["descriptor"]), b["species_name"].astype(str))
    return cat, bg


def load_background_named(bg, organ):
    """(descriptors, real species names). The names are the clustering identity."""
    return bg[organ]


def draw_label_sets(all_species, K, n_sets, rng, crowded):
    """Label sets disjoint in species, so arms do not share difficulty.

    `crowded` takes whole congener blocks -- the all-Sedum catalogue -- and
    `varied` samples species at random. Both draw from a shrinking pool so no
    species appears in two sets of the same family.
    """
    by_genus = defaultdict(list)
    for s in all_species:
        by_genus[s.split()[0]].append(s)
    out = []
    if crowded:
        genera = [g for g in rng.permutation(sorted(by_genus)) if len(by_genus[g]) >= 2]
        i = 0
        for _ in range(n_sets):
            picked = []
            while i < len(genera) and len(picked) < K:
                picked += by_genus[genera[i]]
                i += 1
            if len(picked) < K:
                break
            out.append(picked[:K])
    else:
        pool = list(rng.permutation(all_species))
        for _ in range(n_sets):
            if len(pool) < K:
                break
            out.append(pool[:K])
            pool = pool[K:]
    return out


def fit_arm(cat, bg, species, rng):
    """Fit one head, and return everything the cascade needs *except* the grouping.

    The grouping is deliberately absent: it is the only thing that varies between
    the arms built from this one fit, which is what holds fine accuracy exactly
    constant across them.
    """
    keep = set(species)
    keep_genera = {s.split()[0] for s in keep}

    Xtr, ytr, rows = [], [], []
    for organ in ORGANS:
        E, names, split = cat[organ]
        m = np.array([n in keep for n in names])
        tr = m & (split == "train")
        Xtr.append(E[tr]); ytr.append(names[tr])

        te = m & (split == "test")
        rows.append((E[te], names[te], names[te], "in_catalog"))

        og = np.array([n.split()[0] in keep_genera for n in names])
        out = (~m) & (split == "test") & og
        rows.append((E[out], names[out], names[out], "near_ood"))

        B = load_background_named(bg, organ)
        cut = rng.permutation(len(B[0]))
        n = int(BG_TRAIN_FRAC * len(B[0]))
        Xtr.append(B[0][cut[:n]]); ytr.append(np.full(n, OTHER))
        far, far_names = B[0][cut[n:]], B[1][cut[n:]]
        # Background rows score as __OTHER__ but must cluster on their *real*
        # species for the split. Collapsing them into one __OTHER__ cluster gives
        # `make_splits` a single cluster, puts every negative in test, and leaves
        # calibration with no distant negatives at all -- so the thresholds that
        # produce every outcome here are fitted at the wrong operating point.
        rows.append((far, np.full(len(far), OTHER), far_names, "distant_ood"))

    clf = LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced").fit(
        np.vstack(Xtr), np.concatenate(ytr))
    classes = np.array(clf.classes_)
    mask = classes != OTHER

    X = np.vstack([r[0] for r in rows])
    truth = np.concatenate([r[1] for r in rows])
    cluster = np.concatenate([r[2] for r in rows])
    bucket = np.concatenate([[r[3]] * len(r[0]) for r in rows])
    cata = clf.predict_proba(X)[:, mask]

    # Species centroids, for k-means groupings over this label set.
    cent = {}
    for s in species:
        v = [E[names == s] for organ in ORGANS for E, names, _ in [cat[organ]]]
        v = [x for x in v if len(x)]
        if v:
            cent[s] = np.vstack(v).mean(0)
    return dict(labels=classes[mask], cata=cata, truth=truth, cluster=cluster,
                bucket=bucket, species=list(species), centroids=cent)


def groupings(arm, rng):
    """-> {name: {species -> group}}. The one thing that varies within a fit."""
    species = arm["species"]
    out = {"genus": {s: s.split()[0] for s in species}}
    C = np.array([arm["centroids"][s] for s in species if s in arm["centroids"]])
    have = [s for s in species if s in arm["centroids"]]
    for k in KMEANS_GRID:
        if k < len(have):
            lab = KMeans(k, n_init=10, random_state=0).fit_predict(C)
            out[f"kmeans{k}"] = {s: f"k{l}" for s, l in zip(have, lab)}
    for g in RANDOM_GRID:
        if g < len(species):
            lab = rng.integers(0, g, len(species))
            out[f"random{g}"] = {s: f"r{l}" for s, l in zip(species, lab)}
    return out


def frame(arm, gmap):
    """Per-observation cascade inputs under one grouping.

    Out-of-set congeners keep genus-based bucket membership, so bucket
    composition is identical across groupings, and take the majority group of
    their in-set congeners -- the group a deployment would actually assign them.
    """
    labels, cata, truth, bucket = arm["labels"], arm["cata"], arm["truth"], arm["bucket"]
    ug = sorted(set(gmap.values()))
    gi = {g: i for i, g in enumerate(ug)}
    G = np.zeros((len(ug), len(labels)))
    for j, lab in enumerate(labels):
        G[gi[gmap[lab]], j] = 1.0

    by_genus = defaultdict(list)
    for s, g in gmap.items():
        by_genus[s.split()[0]].append(g)
    near_group = {gen: Counter(gs).most_common(1)[0][0] for gen, gs in by_genus.items()}

    gscore = cata @ G.T
    sp_pred = labels[cata.argmax(1)]
    gn_pred = np.array(ug)[gscore.argmax(1)]
    true_group = np.array([OTHER if t == OTHER else
                           gmap.get(t, near_group.get(t.split()[0], OTHER)) for t in truth])

    return pd.DataFrame({
        "species_conf": cata.max(1),
        "genus_conf": gscore.max(1),
        "species_ok": sp_pred == truth,
        "genus_ok": gn_pred == true_group,
        "in_catalog": bucket == "in_catalog",
        "bucket": bucket,
        # Clustering identities, not scoring identities. `make_splits` keys on
        # these; background rows carry their real species (see `fit_arm`) and
        # near_ood clusters on the *taxonomic* genus rather than the arm's
        # grouping, so the calib/test split is byte-identical across the
        # groupings built from one fit. The grouping is then the only thing that
        # varies between those arms, which is the whole design.
        "species": arm["cluster"],
        "genus": np.array([c.split()[0] for c in arm["cluster"]]),
    })


def measure(df, seed, ood_mix=None, p_ood=P_OOD):
    """Predictors on calib, outcomes on test. -> dict or None if a split is empty.

    `p_ood` is the assumed out-of-catalogue prevalence. It was a module constant
    until the sweep below was added, which meant every published arm shared one
    operating point and the rule fitted to them could not carry a term for it --
    see HEADROOM_FINDINGS.md's third qualification, and narrowcast-derm's
    `ood_sweep.py`, where holding headroom *exactly* fixed and moving only this
    number changes realised retreat by 91x.
    """
    ood_mix = ood_mix or OOD_MIX
    fold = make_splits(df, seed=seed)
    cal, te = df[fold == "calib"], df[fold == "test"]
    if cal.empty or te.empty or not cal["in_catalog"].any() or not te["in_catalog"].any():
        return None

    # `deployment_weights` divides each bucket's share by the sum over the *full*
    # mix, so a bucket that is absent leaves its share unclaimed and the whole
    # weighting renormalises to a lower effective prevalence. Crowded arms take
    # whole genus blocks, so almost no congener is left outside the set and
    # near_ood often has a single cluster and lands entirely in test -- which
    # would fit their thresholds at p_ood 0.145 and score them at 0.20, in
    # exactly the arms whose group share carries the result. Restricting the mix
    # per side puts both at 0.20.
    def _mix(sub):
        m = {b: s for b, s in ood_mix.items() if (sub["bucket"] == b).any()}
        return m or ood_mix

    near_in_calib = bool((cal["bucket"] == "near_ood").any())
    w_cal = deployment_weights(cal["bucket"].to_numpy(), p_ood=p_ood, ood_mix=_mix(cal))
    (tg, ts), _ = fit_thresholds(
        cal["species_conf"].to_numpy(), cal["genus_conf"].to_numpy(),
        cal["species_ok"].to_numpy(), cal["genus_ok"].to_numpy(),
        cal["in_catalog"].to_numpy(), sample_weight=w_cal)

    ci = cal["in_catalog"].to_numpy()
    fine = cal["species_ok"].to_numpy()[ci].mean()
    coarse = cal["genus_ok"].to_numpy()[ci].mean()

    lv = decide(te["species_conf"].to_numpy(), te["genus_conf"].to_numpy(), tg, ts)
    w = deployment_weights(te["bucket"].to_numpy(), p_ood=p_ood, ood_mix=_mix(te))
    answered = lv != DECLINE
    correct = ((lv == SPECIES) & te["species_ok"].to_numpy()) | \
              ((lv == GENUS) & te["genus_ok"].to_numpy())
    inc = te["in_catalog"].to_numpy()

    # Declared diagnostic, not a predictor: at K=14 a species-clustered split
    # leaves ~7 species per side, so calib headroom is a noisy proxy for the test
    # arm's. Noise in a predictor biases a regression towards "adds nothing",
    # which is the outcome pre-declared as fatal -- so the test-side value is
    # recorded to tell a null apart from attenuation.
    fine_te = te["species_ok"].to_numpy()[inc].mean()
    coarse_te = te["genus_ok"].to_numpy()[inc].mean()

    return dict(
        fine=fine, coarse=coarse, headroom=coarse - fine,
        fine_test=fine_te, coarse_test=coarse_te, headroom_test=coarse_te - fine_te,
        t_group=tg, t_label=ts,
        group_share=float((lv[inc] == GENUS).mean()),
        label_share=float((lv[inc] == SPECIES).mean()),
        decline_share=float((lv[inc] == DECLINE).mean()),
        coverage=float(w[answered].sum() / w.sum()),
        precision=float(w[answered & correct].sum() / w[answered].sum())
        if answered.any() else np.nan,
        w_group=float(w[lv == GENUS].sum() / w.sum()),
        n_in_catalog_test=int(inc.sum()),
        near_in_calib=float(near_in_calib),
    )


def score(df, meta, ood_mix=None, p_ood=P_OOD):
    """Average an arm's measurements over N_SPLITS calib/test splits."""
    ms = [m for m in (measure(df, s, ood_mix, p_ood) for s in range(N_SPLITS))
          if m is not None]
    if not ms:
        return None
    row = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    return {**meta, **row, "p_ood": p_ood, "n_splits": len(ms)}


def score_over(df, meta, p_oods, ood_mix=None):
    """The same arm at each operating point.

    The sweep is cheap on purpose. Embedding, drawing the label set, fitting the
    head and building the grouping all happen once per arm; `p_ood` enters only
    at `deployment_weights`, so N operating points cost N cheap re-scorings of a
    frame that is already in memory rather than N builds. Everything upstream of
    the thresholds is held *identical* across the row, which is what makes the
    within-arm comparison clean -- the same property the grouping sweep relies on.
    """
    return [r for r in (score(df, meta, ood_mix, p) for p in p_oods) if r]


def plant_arms(n_sets, p_oods=(P_OOD,)):
    ref, _ = load("bioclip2")
    all_species = np.array(sorted(set(ref["leaf"][1]) | set(ref["flower"][1])))
    rows = []
    for variant in ENCODERS:
        cat, bg = load(variant)
        for K in K_GRID:
            for crowded in (False, True):
                sets = draw_label_sets(all_species, K, n_sets,
                                       np.random.default_rng(K + 7 * crowded), crowded)
                for si, species in enumerate(sets):
                    setid = f"{'crowded' if crowded else 'varied'}-K{K}-{si}"
                    arm = fit_arm(cat, bg, species, np.random.default_rng(si))
                    gs = groupings(arm, np.random.default_rng(si))
                    for gname, gmap in gs.items():
                        rows += score_over(
                            frame(arm, gmap),
                            dict(domain="plants", encoder=variant, K=K,
                                 crowded=crowded, label_set=f"{variant}|{setid}",
                                 set_shape=setid, grouping=gname,
                                 n_groups=len(set(gmap.values()))),
                            p_oods)
                    print(f"  {variant:15s} {setid:16s} {len(gs)} groupings", flush=True)
    return rows


# The out-of-domain arms are what make this a cross-domain result rather than a
# plant one, and they were addressed only under /tmp -- which macOS clears. A
# re-run months later silently skipped audio and birds and produced a
# plants-plus-text table that still described itself as the same experiment.
# Resolved against a persistent directory first, /tmp second.
ARM_INPUTS = Path(f"{DP}/arm_inputs")


def _arm_input(path):
    """Prefer a persistent copy of an arm input over the /tmp original."""
    keep = ARM_INPUTS / Path(path).name
    return str(keep) if keep.exists() else path


OOD_ARMS = [
    ("text", "text-varied", "/tmp/news_varied.npz", "/tmp/news_bg.npz"),
    ("text", "text-crowded", "/tmp/news_crowded.npz", "/tmp/news_bg.npz"),
    ("audio", "kws-sem-varied", "/tmp/kws_varied.npz", "/tmp/kws_bg.npz"),
    ("audio", "kws-sem-crowded", "/tmp/kws_crowded.npz", "/tmp/kws_bg.npz"),
    # kws-ac-* are the *acoustically* grouped Speech Commands arms, and they are
    # NOT reproducible. No script in narrowcast-kws writes kwsA_*.npz; the logic
    # lives in `fewshot_curve.py:acoustic_groups()`, k-means over per-word
    # centroids with `n_groups` a free parameter that headroom moves with. Any
    # value picked now produces arms that are not the published ones, and kws's
    # own finding says to quote the shape of a k-means grouping and never its
    # level. Left addressed at the vanished /tmp paths deliberately: they skip
    # loudly, which is honest, where a reconstruction would look like the
    # original and not be it.
    ("audio", "kws-ac-varied", "/tmp/kwsA_varied.npz", "/tmp/kwsA_bg.npz"),
    ("audio", "kws-ac-crowded", "/tmp/kwsA_crowded.npz", "/tmp/kwsA_bg.npz"),
    # ESC-50 needs a ~600 MB corpus download this machine no longer has.
    ("audio", "esc50-varied", "/tmp/esc50_varied.npz", "/tmp/esc50_bg.npz"),
    ("audio", "esc50-crowded", "/tmp/esc50_crowded.npz", "/tmp/esc50_bg.npz"),
    # Birds needs a fresh iNaturalist fetch (`analysis/bird_fetch.py`).
    ("birds", "birds-crowded", "/tmp/birds_crowded.npz", "/tmp/birds_bg.npz"),
    # Dermatology, from narrowcast-derm: DINOv2 over 2,688 Fitzpatrick17k
    # photographs. These matter out of proportion to their count -- two arms
    # against 1,400 plant ones -- because they are the only *weak* arms here.
    # top-1 ~0.75 against the plants' 0.84-0.97, and it is that regime, not the
    # domain, where narrowcast-derm measured realised retreat moving 91x with
    # p_ood. They are validation, never pooled into the fit; see
    # `check_out_of_domain`.
    ("derm", "derm-varied", f"{DP}/arm_inputs/derm_varied.npz",
     f"{DP}/arm_inputs/derm_background.npz"),
    ("derm", "derm-crowded", f"{DP}/arm_inputs/derm_crowded.npz",
     f"{DP}/arm_inputs/derm_background.npz"),
]


def ood_arms(p_oods=(P_OOD,)):
    """The already-published text/audio/bird arms, re-scored through this code.

    They enter the table so the published points sit on the same axes; they are
    not new corpora, and the findings doc says so.

    The head is fitted through **narrowcast's own** `load_rows` / `score_frame`
    rather than by hand. Hand-rolling it here produced two defects at once: no
    train/eval split of the in-catalogue rows, which made every predictor
    in-sample, and a background cluster taken from the `label` array -- uniformly
    `__OTHER__` in `news_bg` and `birds_bg`, which collapses the distant_ood
    split the same way the plant path did. `sources.from_embeddings` already
    resolves clusters correctly (`cluster` key, else per-row independence) and
    `load_rows` already splits, so the fix is to stop reimplementing them.
    """
    from pathlib import Path

    from narrowcast import build as nbuild
    from narrowcast import sources as nsources

    rows = []
    for domain, name, emb, bg in OOD_ARMS:
        emb, bg = _arm_input(emb), _arm_input(bg)
        if not (Path(emb).exists() and Path(bg).exists()):
            missing = [p for p in (emb, bg) if not Path(p).exists()]
            print(f"  SKIP {name}: {', '.join(missing)} missing -- this arm is "
                  f"absent from the output, and its domain may be too", flush=True)
            continue
        ds = nbuild.load_rows(nsources.load(embeddings=emb), "precomputed",
                              background=nsources.load(embeddings=bg), seed=0)
        nd = nbuild.score_frame(nbuild.fit_head(ds), ds)
        df = pd.DataFrame({
            "species_conf": nd["label_conf"], "genus_conf": nd["group_conf"],
            "species_ok": nd["label_ok"], "genus_ok": nd["group_ok"],
            "in_catalog": nd["in_catalog"], "bucket": nd["bucket"],
            "species": nd["label"], "genus": nd["group"],
        })
        n_groups = int(pd.Series(nd["group"][nd["in_catalog"]]).nunique())
        # These corpora have no near_ood bucket. Left at the plant OOD_MIX,
        # `deployment_weights` skips the missing 0.32 share and renormalises,
        # scoring them at an effective p_ood of 0.145 rather than 0.20 -- a
        # different operating point from every plant arm in the same table.
        K = int(pd.Series(nd["truth"][nd["in_catalog"]]).nunique())
        rs = score_over(df, dict(domain=domain, encoder=name.split("-")[0], K=K,
                                 crowded="crowded" in name, label_set=name,
                                 set_shape=name, grouping="published",
                                 n_groups=n_groups),
                        p_oods, ood_mix={"distant_ood": 1.0})
        rows += rs
        if rs:
            r = rs[0]
            print(f"  {name:18s} fine={r['fine']:.3f} coarse={r['coarse']:.3f} "
                  f"headroom={r['headroom']:+.3f}", flush=True)
    return rows


# ---------------------------------------------------------------- analysis ----

def _cv_r2(d, cols, folds):
    """Grouped cross-validated R2: arms sharing a label set never split folds."""
    y = d["group_share"].to_numpy()
    pred = np.empty_like(y)
    for f in sorted(set(folds)):
        tr, te = folds != f, folds == f
        m = LinearRegression().fit(d.loc[tr, cols], y[tr])
        pred[te] = m.predict(d.loc[te, cols])
    ss = ((y - pred) ** 2).sum()
    return 1 - ss / ((y - y.mean()) ** 2).sum()


def _cv_r2_on(d, cols, folds, y_col="group_share"):
    """Grouped CV R2 for an arbitrary outcome column."""
    y = d[y_col].to_numpy()
    pred = np.empty_like(y, dtype=float)
    for f in sorted(set(folds)):
        tr, te = folds != f, folds == f
        m = LinearRegression().fit(d.loc[tr, cols], y[tr])
        pred[te] = m.predict(d.loc[te, cols])
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def check_out_of_domain(d, fit_domains=("plants",)):
    """Hold out the weak domains and ask whether the fitted rule reaches them.

    Two dermatology arms against 1,400 plant ones would be a rounding error in a
    pooled regression: the fit would come back indistinguishable from plant-only
    and could be misread as "the term generalises". So they are never pooled.
    The rule is fitted on plants alone, then asked to predict arms from a domain
    it has never seen, and the residual is the answer.

    This is the question the sweep exists for. `narrowcast-derm` measured realised
    retreat moving 91x with p_ood where plants move about 2x, and CLAUDE.md's
    reading is that the difference is *encoder strength*, not dermatology -- a
    boundary condition reachable in any domain whose model is weak enough. If
    that is right, the plant-fitted rule should fail on derm in a specific way:
    fine at low p_ood, worsening as p_ood rises, because the term that governs
    the divergence is the one plants barely exercise.
    """
    tr = d[d["domain"].isin(fit_domains)]
    te = d[~d["domain"].isin(fit_domains)]
    if tr.empty or te.empty:
        print(f"\n(no out-of-domain check: fit domains {sorted(set(tr['domain']))}, "
              f"held out {sorted(set(te['domain']))})")
        return

    m = LinearRegression().fit(tr[["headroom", "p_ood"]], tr["group_share"])
    print("\n" + "-" * 72)
    print(f"OUT-OF-DOMAIN: rule fitted on {sorted(set(tr['domain']))} "
          f"({len(tr)} rows), tested on {sorted(set(te['domain']))} ({len(te)} rows)")
    print("-" * 72)
    print(f"  rule: group_share ~ {m.intercept_:+.4f} {m.coef_[0]:+.4f} * headroom "
          f"{m.coef_[1]:+.4f} * p_ood")

    te = te.copy()
    te["pred"] = m.predict(te[["headroom", "p_ood"]])
    te["resid"] = te["group_share"] - te["pred"]

    print(f"\n  {'arm':16s} {'p_ood':>6} {'headroom':>9} {'measured':>9} "
          f"{'predicted':>10} {'residual':>9}")
    for _, r in te.sort_values(["label_set", "p_ood"]).iterrows():
        print(f"  {r['label_set']:16s} {r['p_ood']:>6.2f} {r['headroom']:>9.4f} "
              f"{r['group_share']:>9.4f} {r['pred']:>10.4f} {r['resid']:>+9.4f}")

    print(f"\n  MAE by operating point (does the rule degrade as p_ood rises?)")
    by = te.groupby("p_ood")["resid"].agg(lambda x: x.abs().mean())
    for p, v in by.items():
        print(f"      p_ood {p:<5} MAE {v:.4f}")

    # The comparison that answers "is 2x a plant number": how far does realised
    # retreat move across the swept range, per domain, on arms that retreat at all?
    print(f"\n  span of realised retreat across the p_ood range, per domain")
    lo_p, hi_p = d["p_ood"].min(), d["p_ood"].max()
    for dom, sub in d.groupby("domain"):
        piv = sub[sub["headroom"] > 0.02].pivot_table(
            index="label_set", columns="p_ood", values="group_share", aggfunc="mean")
        if piv.empty or lo_p not in piv or hi_p not in piv:
            continue
        lo, hi = piv[lo_p].mean(), piv[hi_p].mean()
        span = (lo / hi) if hi > 1e-9 else float("inf")
        print(f"      {dom:8s} n={len(piv):4d}  {lo:.4f} -> {hi:.4f}   "
              f"{'x'.join(['', f'{span:.1f}']).lstrip('x') if span != float('inf') else 'to zero'}")
    print("\n      A rule fitted where retreat barely moves cannot be quoted where it does.")


def analyse_operating_point(full, ops):
    """The retreat rule with an operating-point term.

    `HEADROOM_FINDINGS.md` establishes that headroom governs retreat, and then
    records that the `1.8 x headroom` rule of thumb omits `p_ood` and is therefore
    unusable without one -- measured in narrowcast-derm at 91x movement in
    realised retreat while the prediction sat still. That refutation stands. What
    it lacked was a replacement, because every one of the 1,409 arms was fitted at
    a single operating point and a term that never varies cannot be estimated.

    This is that estimate. Headroom is a property of the *calibration* rows and
    does not move with `p_ood` -- verified below rather than assumed, because if
    it did move, the two predictors would be confounded and the whole design
    would be pointless.
    """
    d = full.copy()
    print("\n" + "=" * 72)
    print("OPERATING POINT: the term the 1.8x rule omits")
    print("=" * 72)

    # Confounding check, first and declared. The design rests on headroom being
    # fixed within an arm while p_ood moves. `measure` computes it from
    # species_ok/genus_ok on the calibration half and no deployment weight enters
    # it -- so the spread within an arm should be exactly 0.
    #
    # The arm is (label_set, grouping), NOT set_shape. `set_shape` is the species
    # set alone: it is deliberately shared across all five encoders and every
    # grouping of the same set, which is exactly why the CV folds and the
    # bootstrap key on it. Grouping this check by it would compare different arms
    # to each other and report a spread that is real variation between arms
    # rather than contamination within one.
    d["arm"] = d["label_set"].astype(str) + "|" + d["grouping"].astype(str)
    spread = d.groupby("arm")["headroom"].agg(lambda x: x.max() - x.min())
    worst = float(spread.max())
    verdict = "OK" if worst < 1e-9 else "** CONFOUNDED **"
    print(f"\nheadroom spread within an arm across operating points: "
          f"max {worst:.2e}  {verdict}")
    if worst >= 1e-9:
        print("      headroom must not move with p_ood -- it is computed on the")
        print("      calibration rows with no deployment weight. A non-zero spread")
        print("      means the two predictors are entangled and the fit below is")
        print("      not identified. Fix that before reading anything past here.")

    print(f"\nrealised retreat by operating point "
          f"({d['arm'].nunique()} arms, {d['set_shape'].nunique()} species sets)")
    g = d.groupby("p_ood").agg(headroom=("headroom", "mean"),
                               t_group=("t_group", "mean"),
                               group=("group_share", "mean"),
                               label=("label_share", "mean"),
                               decline=("decline_share", "mean"))
    g["1.8 x headroom"] = 1.8 * g["headroom"]
    print(g.round(4).to_string())

    lo, hi = g["group"].iloc[0], g["group"].iloc[-1]
    print(f"\n      the prediction is pinned at {g['1.8 x headroom'].iloc[0]:.4f} throughout; "
          f"what it predicts moves {lo:.4f} -> {hi:.4f}"
          + (f" ({lo / hi:.0f}x)" if hi > 0 else ""))

    # The two-variable rule. Folds key on the species-set identity, so an arm and
    # all of its operating points stay on one side -- otherwise the model is
    # scored on a p_ood of an arm it has already seen at another p_ood, which is
    # interpolation dressed as prediction.
    folds = pd.factorize(d["set_shape"])[0] % 5
    d["head_x_ood"] = d["headroom"] * d["p_ood"]
    models = {
        "headroom alone (the published rule)": ["headroom"],
        "p_ood alone":                         ["p_ood"],
        "headroom + p_ood":                    ["headroom", "p_ood"],
        "headroom + p_ood + interaction":      ["headroom", "p_ood", "head_x_ood"],
        "headroom + t_group":                  ["headroom", "t_group"],
    }
    print("\ngrouped 5-fold CV R^2 on group-answer share, folds keyed on set_shape")
    for name, cols in models.items():
        print(f"      {name:36s} {_cv_r2_on(d, cols, folds):+.4f}")

    m = LinearRegression().fit(d[["headroom", "p_ood"]], d["group_share"])
    a_h, a_p = m.coef_
    print(f"\n      group_share ~ {m.intercept_:+.4f} {a_h:+.4f} * headroom "
          f"{a_p:+.4f} * p_ood")

    # Cluster bootstrap on the species-set identity, as everywhere else here.
    rng = np.random.default_rng(0)
    sets = d["set_shape"].unique()
    idx = {sh: np.flatnonzero(d["set_shape"].to_numpy() == sh) for sh in sets}
    boots = []
    for _ in range(2000):
        pick = np.concatenate([idx[sh] for sh in rng.choice(sets, len(sets), replace=True)])
        boots.append(LinearRegression().fit(
            d.iloc[pick][["headroom", "p_ood"]], d.iloc[pick]["group_share"]).coef_)
    boots = np.array(boots)
    for nm, j in (("headroom", 0), ("p_ood", 1)):
        blo, bhi = np.percentile(boots[:, j], [2.5, 97.5])
        excl = "excludes 0" if blo * bhi > 0 else "INCLUDES 0"
        print(f"      coef({nm:8s}) = {m.coef_[j]:+.4f}  CI [{blo:+.4f}, {bhi:+.4f}]  {excl}")

    # The mechanism, stated as a number rather than asserted: t_group is fitted
    # mostly as a rejection threshold, climbs with p_ood, and swallows the band
    # between the two thresholds that retreat lives in.
    print(f"\n      corr(p_ood, t_group) = {np.corrcoef(d['p_ood'], d['t_group'])[0,1]:+.3f}"
          f"   corr(t_group, group_share) = "
          f"{np.corrcoef(d['t_group'], d['group_share'])[0,1]:+.3f}")
    print("      t_group is fitted mostly to reject out-of-list inputs, so it rises"
          "\n      with p_ood and squeezes the band between the thresholds that a"
          "\n      group answer has to land in. That is the whole mechanism.")

    # Does the floor survive? The published rule is quoted as a floor, and the
    # derm result put realised retreat *below* it at every operating point.
    d["ratio"] = d["group_share"] / d["headroom"].where(d["headroom"] > 0.02)
    r = d.dropna(subset=["ratio"]).groupby("p_ood")["ratio"].median()
    print("\n      median group_share / headroom by operating point "
          "(1.8 is the published floor)")
    print("      " + r.round(3).to_string().replace("\n", "\n      "))
    print("      A floor that holds only at the p_ood it was fitted at is not a floor.")

    check_out_of_domain(d)


def analyse(path):
    # NB: not `full` -- P2 below binds that name to its two-predictor regression.
    arms_all = pd.read_csv(path)
    if "p_ood" not in arms_all.columns:
        arms_all["p_ood"] = P_OOD        # a CSV from before the sweep existed

    # The pre-registered analysis runs on ONE operating point, always. Pooling an
    # arm's five p_ood rows would quintuple n without adding an independent
    # observation, inflate every interval's apparent precision, and put five
    # copies of the same arm across the CV folds -- which is the exact mistake
    # `set_shape` folding exists to prevent one level up. The sweep is a second
    # question asked of the same arms, not more arms.
    ops = sorted(arms_all["p_ood"].unique())
    d = arms_all[arms_all["p_ood"] == P_OOD].copy()
    if d.empty:
        raise SystemExit(f"no arms at the pre-registered p_ood = {P_OOD}; "
                         f"found {ops}. The declared analysis cannot be run.")
    if len(ops) > 1:
        print(f"{len(arms_all)} rows = {len(d)} arms x {len(ops)} operating points "
              f"{ops}\nPre-registered analysis below uses p_ood = {P_OOD} only "
              f"({len(d)} arms); the sweep follows it.\n")

    print(f"{len(d)} arms, {d['label_set'].nunique()} label sets, "
          f"{d['domain'].nunique()} domains\n")

    below = (d["coarse"] < BREAKEVEN).mean()
    print(f"admissibility: {below:.1%} of arms have coarse < {BREAKEVEN:.3f} "
          f"(need >= 20%) -> {'PASS' if below >= 0.20 else 'FAIL'}")
    print(f"coarse accuracy spans [{d['coarse'].min():.3f}, {d['coarse'].max():.3f}], "
          f"fine [{d['fine'].min():.3f}, {d['fine'].max():.3f}]")
    corr = np.corrcoef(d["fine"], d["headroom"])[0, 1]
    print(f"corr(fine, headroom) = {corr:+.3f}   "
          f"corr(coarse, headroom) = {np.corrcoef(d['coarse'], d['headroom'])[0,1]:+.3f}")

    # If only the incoherent `random` groupings sit below the break-even, P3's
    # threshold test is confounded with group coherence rather than testing it.
    kind = d["grouping"].str.replace(r"\d+", "", regex=True)
    print("\ngrouping x break-even (n arms):")
    print(pd.crosstab(kind, d["coarse"] < BREAKEVEN).rename(
        columns={False: ">= 0.889", True: "< 0.889"}).to_string())

    # Attenuation check: a noisy predictor biases the comparison toward the null.
    print(f"\ncorr(headroom_calib, headroom_test) = "
          f"{np.corrcoef(d['headroom'], d['headroom_test'])[0,1]:+.3f}")
    if below < 0.20:
        print("\nadmissibility failed; the pre-registered analysis is not run.")
        return d

    # P1 -- zero headroom, zero retreat
    z = d[d["headroom"].abs() < 0.01]
    print(f"\nP1  arms with |headroom| < 0.01: n={len(z)}, "
          f"mean group share {z['group_share'].mean():.4f}, max {z['group_share'].max():.4f}")

    # P2 -- does the difference parameterisation survive?
    d = d.copy()
    for c in ("fine", "coarse", "headroom"):
        d[f"z_{c}"] = (d[c] - d[c].mean()) / d[c].std()
    # Folds and the bootstrap key on `set_shape`, the *species-set* identity. The
    # same species sets recur across all five encoders, so keying on `label_set`
    # (which includes the encoder) would treat five correlated arms as five
    # independent draws.
    folds = pd.factorize(d["set_shape"])[0] % 5
    models = {"M_fine": ["z_fine"], "M_coarse": ["z_coarse"],
              "M_head": ["z_headroom"], "M_full": ["z_fine", "z_coarse"]}
    print("\nP2  grouped 5-fold CV R^2 on group-answer share")
    for name, cols in models.items():
        print(f"      {name:9s} {_cv_r2(d, cols, folds):+.4f}")

    # RAW units, deliberately. "Only the difference matters" means the raw
    # coefficients are equal and opposite; on predictors standardised by their own
    # SDs that is b/sigma_fine = -c/sigma_coarse, which reduces to b + c = 0 only
    # when the SDs match. They do not here (0.129 vs 0.087), and the standardised
    # version reports this asymmetry with the *wrong sign*.
    full = LinearRegression().fit(d[["fine", "coarse"]], d["group_share"])
    b, c = full.coef_
    rng = np.random.default_rng(0)
    sets = d["set_shape"].unique()
    idx = {s: np.flatnonzero(d["set_shape"].to_numpy() == s) for s in sets}
    boots = []
    for _ in range(2000):
        pick = np.concatenate([idx[s] for s in rng.choice(sets, len(sets), replace=True)])
        m = LinearRegression().fit(d.iloc[pick][["fine", "coarse"]],
                                   d.iloc[pick]["group_share"])
        boots.append(m.coef_[0] + m.coef_[1])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"      raw b(fine)={b:+.4f}  c(coarse)={c:+.4f}  b+c={b+c:+.4f} "
          f"CI [{lo:+.4f}, {hi:+.4f}]  (0 => headroom parameterisation exact)")
    print(f"      sigma_fine={d['fine'].std():.4f} sigma_coarse={d['coarse'].std():.4f}")

    # P3 -- the break-even threshold
    print(f"\nP3  group-answer share either side of coarse = {BREAKEVEN:.3f}")
    for name, sub in (("coarse <  break-even", d[d["coarse"] < BREAKEVEN]),
                      ("coarse >= break-even", d[d["coarse"] >= BREAKEVEN])):
        hi_head = sub[sub["headroom"] > 0.15]
        print(f"      {name}: n={len(sub):4d}  mean group share {sub['group_share'].mean():.4f}"
              f"   | of those with headroom>0.15: n={len(hi_head):4d}, "
              f"{hi_head['group_share'].mean():.4f}" if len(sub) else f"      {name}: n=0")

    # Immune to the functional form: group share is exactly zero for most arms,
    # so a linear R^2 is fitting a zero-inflated outcome and a median crosstab is
    # the more legible statement of the same thing.
    print("\n    median group-answer share, coarse x fine")
    cb = pd.cut(d["coarse"], [0, 0.7, 0.8, BREAKEVEN, 0.95, 1.01],
                labels=["<.70", ".70-.80", ".80-.889", ".889-.95", ">.95"])
    fb = pd.cut(d["fine"], [0, 0.6, 0.75, 0.9, 1.01],
                labels=["<.60", ".60-.75", ".75-.90", ">.90"])
    print(d.pivot_table(index=cb, columns=fb, values="group_share",
                        aggfunc="median", observed=False).round(3).to_string())

    # P4 -- where do group answers come from?
    print("\nP4  regressing in-catalogue label share on group share")
    for name, sub in (("all arms", d), ("crowded", d[d["crowded"]]), ("varied", d[~d["crowded"]])):
        if len(sub) > 10:
            r = np.corrcoef(sub["group_share"], sub["label_share"])[0, 1]
            rd = np.corrcoef(sub["group_share"], sub["decline_share"])[0, 1]
            print(f"      {name:9s} n={len(sub):4d}  corr(group,label)={r:+.3f}  "
                  f"corr(group,decline)={rd:+.3f}")

    if len(ops) > 1:
        analyse_operating_point(arms_all, ops)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=f"{DP}/headroom_arms.csv")
    # 4, not 3. The published 1,409 arms were produced with 4 sets per cell --
    # 5 encoders x 4 K x 2 crowded x 4 sets = 160 label-set builds, which is where
    # the 160-per-grouping counts in headroom_arms.csv come from. The default said
    # 3, so re-running this script as documented reproduced exactly 75% of the
    # published arms in every cell, which looks like silent data loss and is
    # really just a flag nobody passed.
    ap.add_argument("--sets-per-cell", type=int, default=4)
    ap.add_argument("--analyse", metavar="CSV")
    ap.add_argument("--p-ood", type=float, nargs="+", default=[P_OOD], metavar="P",
                    help="assumed out-of-catalogue prevalence, repeatable. "
                         f"Default {P_OOD}, which reproduces the published arms "
                         "exactly. Give several (e.g. --p-ood 0.05 0.1 0.2 0.4 0.6) "
                         "to score every arm at each, which is what the retreat "
                         "rule needs to carry an operating-point term.")
    a = ap.parse_args()

    if a.analyse:
        analyse(a.analyse)
        return

    p_oods = sorted(set(a.p_ood))
    for p in p_oods:
        if not 0.0 <= p < 1.0:
            raise SystemExit(f"--p-ood {p} is not a prevalence in [0, 1)")
    if P_OOD not in p_oods:
        # The pre-registered analysis is defined at this operating point. A sweep
        # that skips it produces a CSV the declared tests cannot be run on, which
        # is a worse outcome than one extra cheap re-scoring per arm.
        print(f"note: adding the pre-registered p_ood = {P_OOD} to the sweep; "
              "the declared analysis is defined there.")
        p_oods = sorted(set(p_oods) | {P_OOD})

    print(f"break-even for a group answer under UTILITY: {BREAKEVEN:.4f}")
    print(f"operating points: {p_oods}\n")
    print("plant arms:", flush=True)
    rows = plant_arms(a.sets_per_cell, p_oods)
    # Checkpoint before the cheap phase. plant_arms is hours; ood_arms is seconds
    # and reaches for files that may not be there. Writing only at the end once
    # meant an interruption anywhere discarded the expensive half, which nearly
    # happened when the machine slept mid-run.
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"  checkpoint: {len(rows)} plant rows -> {a.out}", flush=True)

    print("\nout-of-domain arms (published, re-scored):", flush=True)
    rows += ood_arms(p_oods)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"\nwrote {a.out}: {len(rows)} rows "
          f"= {len(rows) // max(len(p_oods), 1)} arms x {len(p_oods)} operating points")
    analyse(a.out)


if __name__ == "__main__":
    main()
