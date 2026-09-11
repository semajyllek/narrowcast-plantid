"""Pack everything the 518 px student run needs into one tarball for Drive.

`notebooks/tiny_student_colab.ipynb` trains at the teacher's own input resolution,
which is the one thing `TINY_PREREG.md`'s amended re-close condition requires and
the one thing MPS cannot deliver in reasonable time. The notebook should not have
to know anything about this project's caches, taxonomy or corpus layout -- so this
writes a bundle that is complete on its own:

    images/           every transfer and evaluation photograph, short side 518
    teacher.npz       per-arm teacher posteriors, truth, buckets, clusters, groups
    manifest.parquet  path -> arm, role, and row index into the npz arrays

**The teacher posteriors are computed here, not there.** They are
`clf.predict_proba` over vectors this project already has cached, so the notebook
never loads PlantCLEF2024 and the GPU is spent entirely on the student. It also
means the teacher the student is scored against is bit-identical to the one
measured locally, rather than a re-fit that might differ.

518 on the short side, not 224: the student must be *able* to see what the teacher
sees, and an image resized down to 224 here would silently reimpose the exact
handicap this run exists to remove. Encoded at quality 90, which is where the
resize stops being visible and the bundle stops growing.

Usage:
    PYTHONPATH=. .venv/bin/python -m analysis.make_distil_bundle --out /tmp/tiny_bundle
"""

import argparse
import json
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from analysis.distil_task import (CROWDED, P_OOD, build_frame, inat_eval,
                                  load_index, load_vectors, separated_set, teacher)
from narrowcast import build as B

SHORT = 518          # the teacher's input; resizing below this reimposes the handicap
QUALITY = 90


def _resize(src, dst):
    """Short side to SHORT, aspect preserved. Images already smaller are copied as-is
    rather than upscaled -- inventing pixels would flatter the student."""
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            s = SHORT / min(w, h)
            if s < 1.0:
                im = im.resize((max(1, round(w * s)), max(1, round(h * s))),
                               Image.LANCZOS)
            im.save(dst, "JPEG", quality=QUALITY)
        return True
    except Exception:
        return False


def main(a):
    out = Path(a.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    index = load_index()
    vecs = load_vectors(a.variant)
    arms = {"crowded": CROWDED, "separated": separated_set(index)}

    payload, rows, seen = {}, [], {}
    summary = {"variant": a.variant, "short_side": SHORT, "p_ood": P_OOD, "arms": {}}

    for arm, labels in arms.items():
        clf, X, eval_paths, truth, bucket, cluster, group = teacher(
            index, vecs, labels, a.variant)
        classes = np.array(clf.classes_)

        # The teacher's own card numbers, carried along so the notebook compares
        # against a measurement rather than against a number typed into a cell.
        t_metrics = B.fit_and_measure(
            build_frame(classes, clf.predict_proba(X), truth, bucket, cluster, group),
            p_ood=P_OOD)

        # Transfer set: catalogue train rows with a cached vector, so every target
        # is known without running the teacher on the GPU.
        tr = index[(index.split == "train") & index.image_id.isin(vecs.keys())]
        if a.limit:
            tr = tr.sample(n=min(a.limit, len(tr)), random_state=0)
        soft = clf.predict_proba(np.stack([vecs[i] for i in tr.image_id])).astype("float32")

        payload[f"{arm}_classes"] = classes.astype(str)
        payload[f"{arm}_soft"] = soft
        payload[f"{arm}_truth"] = truth.astype(str)
        payload[f"{arm}_bucket"] = bucket.astype(str)
        payload[f"{arm}_cluster"] = cluster.astype(str)
        payload[f"{arm}_group"] = group.astype(str)

        for role, paths in (("transfer", list(tr.path)), ("eval", list(eval_paths))):
            for i, p in enumerate(paths):
                name = seen.get(p)
                if name is None:
                    name = f"{len(seen):07d}.jpg"
                    seen[p] = name
                rows.append({"arm": arm, "role": role, "idx": i, "file": name,
                             "src": p})

        summary["arms"][arm] = {
            "labels": list(labels), "n_transfer": len(tr), "n_eval": len(eval_paths),
            "teacher": {k: t_metrics[k] for k in
                        ("label_share", "coverage", "precision", "closed_set_top1",
                         "headroom", "group_share", "decline_share")},
        }
        print(f"  {arm:10s} {len(labels)} labels · transfer {len(tr)} · eval "
              f"{len(eval_paths)} · teacher label_share "
              f"{t_metrics['label_share']:.4f}", flush=True)

    # Declared pass/fail, copied from TINY_PREREG so the notebook cannot drift from it.
    sep, cro = summary["arms"]["separated"], summary["arms"]["crowded"]
    summary["thresholds"] = {
        "separated_pass_at_or_above": round(sep["teacher"]["label_share"] - 0.05, 4),
        "crowded_fail_below": round(cro["teacher"]["label_share"] - 0.15, 4),
    }

    print(f"\nresizing {len(seen)} unique images to short side {SHORT} ...", flush=True)
    t0, bad = time.time(), []
    for src, name in seen.items():
        dst = img_dir / name
        if not dst.exists() and not _resize(src, dst):
            bad.append(src)
    if bad:
        print(f"  WARNING: {len(bad)} unreadable, dropped from the manifest")
    print(f"  {time.time() - t0:.0f}s", flush=True)

    man = pd.DataFrame(rows)
    man = man[~man.src.isin(bad)].drop(columns=["src"])
    man.to_parquet(out / "manifest.parquet", index=False)
    np.savez_compressed(out / "teacher.npz", **payload)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    if a.tar:
        tar = Path(a.tar)
        print(f"\nwriting {tar} ...", flush=True)
        with tarfile.open(tar, "w") as t:
            t.add(out, arcname="tiny_bundle")
        print(f"  {tar.stat().st_size / 1e9:.2f} GB")

    size = sum(f.stat().st_size for f in img_dir.iterdir()) / 1e9
    print(f"\nbundle {out}  ·  {len(seen)} images  ·  {size:.2f} GB")
    print(f"thresholds — separated passes at >= "
          f"{summary['thresholds']['separated_pass_at_or_above']}, "
          f"crowded fails below {summary['thresholds']['crowded_fail_below']}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="plantclef24")
    ap.add_argument("--out", default="/tmp/tiny_bundle")
    ap.add_argument("--tar", default="", help="also write a single tar for Drive")
    ap.add_argument("--limit", type=int, default=0, help="cap transfer images (smoke)")
    sys.exit(main(ap.parse_args()))
