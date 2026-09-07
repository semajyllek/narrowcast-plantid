"""Does a geometric novelty score fix near-OOD?

Pre-registered in `NEAR_OOD_PREREG.md`. Three arms on identical clustered
splits: the shipped two-threshold cascade refitted as a paired baseline, a
reject gate, and a retreat gate. Both gate placements were declared in advance
because the declared utility does not obviously prefer one.

The score the cascade lacks is geometric rather than posterior-derived. A
multinomial posterior is closed-set and cannot express "none of these", which is
exactly the near-OOD case -- the observation genuinely resembles a catalogue
genus, so a closed-set score is right to be confident about the genus, and that
confidence is what carries it past `t_genus`.

Usage:
    PYTHONPATH=. .venv/bin/python -m analysis.near_ood
"""

import argparse

import numpy as np
import pandas as pd

from plantid.config import DATA_PROCESSED, ORGANS
from plantid.data.curation import curated_name
from plantid.eval.inat_fusion import _l2
from plantid.eval.rejection import (
    DECLINE, GENUS, IN_CATALOG, OOD_MIX_GLOBAL, SPECIES,
    build_observations, deployment_weights, decide, precision_coverage, utility,
)
from plantid.features.embed_catalog import load_catalog

P_OOD = 0.20
N_GRID = 60
N_BOOT = 2000
SEED = 0


def novelty(df, variant="bioclip2", cache_dir=DATA_PROCESSED):
    """Max cosine to a catalogue species centroid, per observation.

    Centroids are built from catalogue *train* rows only -- the same rows the
    head was fitted on -- so no test information enters the score. Organs are
    pooled, which is declared as a limitation of this design.
    """
    names, embs = [], []
    for organ in ORGANS:
        d = load_catalog(organ, variant=variant, cache_dir=cache_dir)
        keep = d["split"] == "train"
        names.append(pd.Series(d["species_name"][keep]).map(curated_name))
        embs.append(d["descriptor"][keep])
    nm = pd.concat(names, ignore_index=True)
    E = _l2(np.vstack(embs).astype(np.float32))
    ok = nm.notna().to_numpy()
    nm, E = nm[ok].to_numpy(), E[ok]
    species = np.array(sorted(set(nm)))
    centroids = _l2(np.stack([E[nm == s].mean(0) for s in species]))

    z = np.load(cache_dir / f"inat_{variant}.npz")
    Ei = _l2(z["descriptor"].astype(np.float32))
    pos = {p: i for i, p in enumerate(z["path"])}
    obs = pd.read_parquet(cache_dir / "inat_observations.parquet")
    paths = obs.set_index("obs_id")["local_paths"].to_dict()

    V = _l2(np.stack([Ei[[pos[p] for p in paths[o] if p in pos]].mean(0) for o in df["obs_id"]]))
    return (V @ centroids.T).max(1)


def decide_gated(sc, gc, nov, t_genus, t_species, t_novel, arm):
    """The cascade with one extra gate. `arm='none'` reproduces the baseline."""
    out = decide(sc, gc, t_genus, t_species)
    if arm == "reject":
        out[nov < t_novel] = DECLINE
    elif arm == "retreat":
        # applied before the decline gate would have fired, so DECLINE still dominates
        out[(nov < t_novel) & (out == SPECIES)] = GENUS
    return out


def fit(df, nov, arm, weights=None):
    """Grid-search the thresholds this arm has, maximising weighted expected
    utility on the calibration split only."""
    c = df["fold"] == "calib"
    sc, gc = df.loc[c, "species_conf"].to_numpy(), df.loc[c, "genus_conf"].to_numpy()
    nv = nov[c.to_numpy()]
    sok, gok = df.loc[c, "species_ok"].to_numpy(), df.loc[c, "genus_ok"].to_numpy()
    inc = df.loc[c, "in_catalog"].to_numpy()
    sw = deployment_weights(df.loc[c, "bucket"].to_numpy(), p_ood=P_OOD)
    sw = sw / sw.sum()

    g_grid = np.quantile(gc, np.linspace(0, 1, N_GRID))
    s_grid = np.quantile(sc, np.linspace(0, 1, N_GRID))
    n_grid = [0.0] if arm == "none" else list(np.quantile(nv, np.linspace(0, 1, N_GRID)))

    best, best_u = (0.0, 0.0, 0.0), -np.inf
    for tg in g_grid:
        for ts in s_grid:
            for tn in n_grid:
                u = float(np.dot(
                    utility(decide_gated(sc, gc, nv, tg, ts, tn, arm), sok, gok, inc, weights), sw))
                if u > best_u:
                    best, best_u = (float(tg), float(ts), float(tn)), u
    return best, best_u


def evaluate(df, nov, thr, arm):
    """Per-observation utility on test, plus the reported shares."""
    t = df["fold"] == "test"
    d = df[t]
    lv = decide_gated(d["species_conf"].to_numpy(), d["genus_conf"].to_numpy(),
                      nov[t.to_numpy()], *thr, arm)
    sw = deployment_weights(d["bucket"].to_numpy(), p_ood=P_OOD)
    u = utility(lv, d["species_ok"].to_numpy(), d["genus_ok"].to_numpy(),
                d["in_catalog"].to_numpy())
    return d.assign(level=lv, util=u, w=sw)


def shares(res, bucket):
    b = res[res["bucket"] == bucket]
    if b.empty:
        return {}
    wrong = ((b.level == SPECIES) & ~b.species_ok) | ((b.level == GENUS) & ~b.genus_ok)
    return {"n": len(b),
            "species": (b.level == SPECIES).mean(), "genus": (b.level == GENUS).mean(),
            "decline": (b.level == DECLINE).mean(), "wrong": wrong.mean()}


def paired_bootstrap(base, arm, clusters, w, n=N_BOOT, seed=SEED):
    """Resample clusters, recompute both arms on the same draw, difference them.

    The difference is **deployment-weighted**, like every other headline in this
    project. An unweighted mean would report the effect at the evaluation set's
    incidental ~60% out-of-catalogue rate rather than the declared 20%, which
    inflates it -- the gate acts mostly on out-of-catalogue rows, so the buckets
    this reweights *down* are exactly the ones it helps.
    """
    rng = np.random.RandomState(seed)
    uniq = np.array(sorted(set(clusters)))
    index = {c: np.flatnonzero(np.asarray(clusters) == c) for c in uniq}
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([index[c] for c in pick])
        out.append(np.average(arm[idx], weights=w[idx]) - np.average(base[idx], weights=w[idx]))
    return float(np.mean(out)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def run(variant="bioclip2", arms=("none", "reject", "retreat")):
    df, _ = build_observations(DATA_PROCESSED / f"inat_{variant}.npz", variant=variant)
    nov = novelty(df, variant=variant)

    # Post-hoc controls, labelled as such. If a retreat gate on a score the
    # cascade *already has* does as well, the finding is "a second gate on the
    # retreat decision helps" rather than "the geometry helps".
    scores = {"none": nov, "reject": nov, "retreat": nov,
              "retreat_genusconf": df["genus_conf"].to_numpy(),
              "retreat_omo": df["one_minus_other"].to_numpy()}

    # Cluster on the real identity, never on the bucket: the near-OOD decision is
    # genus-level, everything else is species-level.
    cluster = np.where(df["bucket"] == "near_ood", df["genus"], df["species"])

    results, rows = {}, []
    for arm in arms:
        sc_arm = scores[arm]
        kind = "retreat" if arm.startswith("retreat") else arm
        thr, calib_u = fit(df, sc_arm, kind)
        res = evaluate(df, sc_arm, thr, kind)
        results[arm] = res
        rows.append({
            "arm": arm, "t_genus": round(thr[0], 4), "t_species": round(thr[1], 4),
            "t_novel": round(thr[2], 4) if arm != "none" else None,
            "calib_utility": round(calib_u, 4),
            "test_utility": round(float(np.average(res["util"], weights=res["w"])), 4),
        })
    table = pd.DataFrame(rows)

    t = (df["fold"] == "test").to_numpy()
    cl = cluster[t]
    base = results["none"]["util"].to_numpy()
    for arm in [a for a in arms if a != "none"]:
        m, lo, hi = paired_bootstrap(base, results[arm]["util"].to_numpy(), cl,
                                     results["none"]["w"].to_numpy())
        table.loc[table.arm == arm, ["d_util", "lo", "hi"]] = [round(m, 4), round(lo, 4), round(hi, 4)]

    # The pre-specified secondary endpoint, with an interval. The primary is
    # prevalence-weighted and near-OOD is only 6.4% of assumed traffic
    # (20% x 0.32), so the bucket-level rate is where an effect of this size is
    # resolvable at all. Unweighted on purpose: it is a within-bucket rate.
    nb = (results["none"]["bucket"] == "near_ood").to_numpy()
    def wrongv(res):
        b = res[res["bucket"] == "near_ood"]
        return (((b.level == SPECIES) & ~b.species_ok)
                | ((b.level == GENUS) & ~b.genus_ok)).to_numpy().astype(float)
    ones = np.ones(nb.sum())
    for arm in [a for a in arms if a != "none"]:
        m, lo, hi = paired_bootstrap(wrongv(results[arm]), wrongv(results["none"]),
                                     cluster[t][nb], ones)
        table.loc[table.arm == arm, ["d_nearood_wrong", "nw_lo", "nw_hi"]] = [
            round(-m, 4), round(-hi, 4), round(-lo, 4)]

    bucket_rows = []
    for arm, res in results.items():
        for bucket in ("near_ood", IN_CATALOG, "regional_ood", "distant_ood"):
            s = shares(res, bucket)
            if s:
                bucket_rows.append({"arm": arm, "bucket": bucket,
                                    **{k: (round(v, 4) if isinstance(v, float) else v)
                                       for k, v in s.items()}})

    pc_rows = []
    for arm, res in results.items():
        for p in (0.6, 0.4, 0.2, 0.1):
            pr, cov = precision_coverage(res["level"].to_numpy(), res["species_ok"].to_numpy(),
                                         res["genus_ok"].to_numpy(), res["bucket"].to_numpy(),
                                         p_ood=p, ood_mix=OOD_MIX_GLOBAL)
            pc_rows.append({"arm": arm, "p_ood": p,
                            "precision": round(pr, 4), "coverage": round(cov, 4)})

    return table, pd.DataFrame(bucket_rows), pd.DataFrame(pc_rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="bioclip2")
    ap.add_argument("--arms", nargs="+",
                    default=["none", "reject", "retreat", "retreat_genusconf", "retreat_omo"])
    a = ap.parse_args()

    thr, buckets, pc = run(a.variant, tuple(a.arms))
    print("\n== thresholds and utility (paired, genus-clustered bootstrap)")
    print(thr.to_string(index=False))
    print("\n== per-bucket shares")
    print(buckets.to_string(index=False))
    print("\n== precision / coverage")
    print(pc.pivot(index="p_ood", columns="arm", values=["precision", "coverage"]).to_string())
    thr.to_csv(DATA_PROCESSED / "near_ood_arms.csv", index=False)
    buckets.to_csv(DATA_PROCESSED / "near_ood_buckets.csv", index=False)
