"""Export catalogue embeddings in the format `narrowcast --embeddings` expects.

`plantid/tool/` used to build models in-repo. That job moved to narrowcast, which
deliberately does not know how to read this project's catalogue caches: choosing
a corpus, reconciling its taxonomy and licensing its images are domain decisions,
and narrowcast takes a dataset rather than going to find one.

This is the seam between them. It writes the vectors this project already has
into narrowcast's source format, so the workflow is:

    PYTHONPATH=. .venv/bin/python -m analysis.export_for_narrowcast \\
        --variant bioclip2 --species my.txt --out /tmp/cat.npz

    narrowcast audit --embeddings /tmp/cat.npz \\
        --background-embeddings /tmp/bg.npz \\
        --encoder-name bioclip2 --out models/mine

**No `cluster` column is written, and that is deliberate.** The catalogue does not
group its images by individual plant, so there is no honest cluster to declare.
narrowcast will say so on the card and report that its intervals are
anticonservative -- which is true, and better than inventing a grouping.

That is a fact about *this corpus*, not about the tool, and it does not carry to
regional data. `plantid.data.regional_fetch` records the GBIF occurrence key, so
several photographs of one plant are identifiable, and
`plantid.data.regional_embed` writes that `cluster` column into the `.npz`. Use
that path for a regional bundle; this one exports the Pl@ntNet catalogue.
"""

import argparse
from pathlib import Path

import numpy as np

from plantid.config import DATA_PROCESSED
from plantid.data.curation import canonical_name
from plantid.features.embed_background import catalog_species, load_background

ORGANS = ("leaf", "flower")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="bioclip2")
    ap.add_argument("--species", help="file of binomials; omit for the whole catalogue")
    ap.add_argument("--out", required=True)
    ap.add_argument("--background", action="store_true",
                    help="export the reject pool instead of the catalogue")
    a = ap.parse_args()

    vecs, labels, keep, declared = [], [], None, set()
    if a.background:
        cs = catalog_species()
        for organ in ORGANS:
            d = load_background(organ, exclude_species=cs, variant=a.variant)
            vecs.append(d["descriptor"])
            labels += ["__OTHER__"] * len(d["descriptor"])
            if "encoder" in d:
                declared.add(str(np.asarray(d["encoder"]).ravel()[0]))
    else:
        if a.species:
            keep = {canonical_name(x) for x in Path(a.species).read_text().split("\n")
                    if x.strip() and not x.strip().startswith("#")}
        for organ in ORGANS:
            p = Path(DATA_PROCESSED) / f"catalog_{organ}_{a.variant}.npz"
            if not p.exists():
                print(f"  skipping {organ}: no cache for {a.variant}")
                continue
            d = np.load(p, allow_pickle=True)
            names = np.array([canonical_name(n) for n in d["species_name"].astype(str)])
            m = np.ones(len(names), bool) if keep is None else np.isin(names, list(keep))
            vecs.append(d["descriptor"][m])
            labels += list(names[m])
            if "encoder" in d:
                declared.add(str(np.asarray(d["encoder"]).ravel()[0]))

    if not vecs:
        raise SystemExit(f"nothing exported — no caches for variant {a.variant!r}")
    if len(declared) > 1:
        # Concatenating caches is exactly where two encoders would meet, and the
        # export would carry one label over both. Refuse instead.
        raise SystemExit(
            "the caches for this variant declare more than one encoder: "
            + ", ".join(sorted(declared))
            + "\nRe-embed them with one. Mixing an export with its own original "
              "flatters label share and narrowcast's geometric check cannot see "
              "it (0 of 21 such pairs, SPACE_CHECK_FINDINGS.md).")
    if keep is not None:
        missing = sorted(keep - set(labels))
        if missing:
            # A silently dropped label is a model that cannot see a class the
            # user asked for. Say so loudly rather than exporting a short file.
            print(f"warning: {len(missing)} requested label(s) absent from the "
                  f"{a.variant} catalogue and NOT exported: {', '.join(missing[:8])}"
                  + (" ..." if len(missing) > 8 else ""))
    X = np.vstack(vecs)
    # Carried through from the caches, and **omitted entirely when they declare
    # nothing**. `--variant` is only a filename selector here
    # (`catalog_{organ}_{variant}.npz`); nothing in this script verifies the cache
    # was produced by that encoder, so writing it would be a claim this script
    # cannot vouch for. Every other producer writes the identity at the moment of
    # embedding, where the variant is ground truth.
    #
    # A fabricated declaration is worse than none: narrowcast compares two
    # declarations and passes when they agree, so two invented labels would
    # silently disable the geometric check as well -- turning the one mechanism
    # that catches an export mismatch into the thing that hides it. Absent is
    # handled: `sources` notes it and the check falls back to geometry.
    out = {"descriptor": X, "label": np.asarray(labels, dtype=str)}
    if len(declared) == 1:
        out["encoder"] = declared.pop()
    else:
        print("  note: the caches declare no encoder, so neither does this export "
              "— re-embed to record it")
    np.savez_compressed(a.out, **out)
    print(f"{a.out}: {X.shape}, {len(set(labels))} labels")


if __name__ == "__main__":
    main()
