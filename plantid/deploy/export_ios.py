"""A narrowcast bundle + a Core ML encoder -> one directory an iOS app can load.

The encoder is a `.mlpackage`; everything else -- the fitted head, the two (or
three) thresholds, the caller's group map, the suppressed labels -- is one JSON
file. An app then needs ~40 lines of arithmetic, and `cascade.swift` next to this
module is that arithmetic written out.

**Run this under `.venv-mps`.** It reads the preprocessing contract off the
encoder's own transform via `deploy.coreml.preprocess_spec`, which constructs the
torch encoder; the rest of this repo's tooling runs under `.venv`, which has no
torch. It fails rather than defaulting if the spec cannot be read, because a
hardcoded interpolation is the bug that "reports cosine ~1.0 while the embeddings
sit in a different space from every cached npz the accuracy numbers came from".

## What the app has to do, and what it does not

The Core ML graph was built with `ct.ImageType(scale=1/255)` and carries the
channel normalisation **and the final L2** inside it. So the only preprocessing
left outside is **resize and centre-crop to `side`, with the recorded
interpolation**. Hand it an RGB buffer of that size and the output is the unit
vector the head expects.

## What this artifact rests on, declared rather than assumed

The head was fitted on vectors from the **torch** encoder. The app will produce
vectors from the **Core ML int8 export**. `ENCODER_DECISION.md` measures cosine
0.9996 between them, which is why int8 was chosen over int4 -- but 0.9996 is not
1.0, and the thresholds were fitted on the torch side. The JSON records the
encoder identity as a Core ML one (`…+coreml:…`) and states the cosine, so a
reader can see what the numbers rest on instead of inferring it.
"""

import argparse
import base64
import json
import shutil
from pathlib import Path

import numpy as np

SCHEMA = 1
OTHER = "__OTHER__"


def _f32(a) -> dict:
    """A float32 array as base64 plus its shape. Swift reads it with one
    `Data(base64Encoded:)` and a `withUnsafeBytes`; JSON numbers would triple the
    size and lose the exact bits."""
    a = np.ascontiguousarray(np.asarray(a, dtype="<f4"))
    return {"b64": base64.b64encode(a.tobytes()).decode("ascii"),
            "shape": list(a.shape), "dtype": "float32-le"}


def build_payload(bundle: Path, preprocess: dict, encoder: str,
                  coreml_name: str, cosine_to_torch: float | None = None) -> dict:
    """Everything the cascade needs, from a bundle this tool fitted.

    Refuses an audit bundle by name, the way `narrowcast.predict` does: a bundle
    built from `--scores` has no head, so there is nothing to run on a phone.
    """
    manifest = json.loads((bundle / "manifest.json").read_text())
    if not manifest.get("has_head", True):
        raise SystemExit(
            f"{bundle} is an audit of a model this tool did not fit "
            f"(source: {manifest.get('source')!r}). It carries measurements and no "
            "weights, so there is nothing to export. Build a bundle with "
            "`--embeddings` and `--background-embeddings` instead.")

    z = np.load(bundle / "head.npz", allow_pickle=True)
    coef = np.asarray(z["coef"], dtype="float64")
    intercept = np.asarray(z["intercept"], dtype="float64").ravel()
    classes = [str(c) for c in z["classes"]]

    # sklearn's binary parameterisation stores one row; `predict.Bundle` expands it
    # to two at run time. Expanded here instead, so the app has one code path.
    if coef.shape[0] == 1:
        coef = np.vstack([-coef, coef])
        intercept = np.concatenate([-intercept, intercept])

    m = manifest["metrics"]
    groups = manifest.get("groups") or {}
    if not groups:
        raise SystemExit(
            "this bundle stores no group map, so the coarse rank would have to be "
            "re-derived from each label's first whitespace token. That is a "
            "Latin-binomial convention and has been wrong in five places. Rebuild "
            "with the group column the model was measured with.")

    return {
        "schema": SCHEMA,
        "created_from": str(bundle),
        "encoder": encoder,
        "coreml": {
            "package": coreml_name,
            "input": "image",
            "side": int(preprocess["side"]),
            "output": "embedding",
            "dim": int(coef.shape[1]),
            # Stated because it is the whole preprocessing contract: anything not
            # listed here is already inside the graph.
            "outside_the_graph": "resize and centre-crop to `side` with "
                                 "`interpolation`; hand over RGB",
            "inside_the_graph": "1/255 scale, channel mean/std, and the final L2 "
                                "normalisation — do not repeat any of them",
        },
        "preprocess": {
            "side": int(preprocess["side"]),
            "interpolation": str(preprocess["interpolation"]),
            "mean": [float(x) for x in preprocess["mean"]],
            "std": [float(x) for x in preprocess["std"]],
        },
        "classes": classes,
        "reject_class": OTHER if OTHER in classes else None,
        "coef": _f32(coef),
        "intercept": _f32(intercept),
        "groups": {str(k): str(v) for k, v in groups.items()},
        "thresholds": {
            "t_group": float(m["t_group"]),
            "t_label": float(m["t_label"]),
            "t_novel": None if m.get("t_novel") is None else float(m["t_novel"]),
        },
        "never_answer": sorted(manifest.get("never_answer") or []),
        "measured": {
            "coverage": m.get("coverage"), "precision": m.get("precision"),
            "label_share": m.get("label_share"), "p_ood": m.get("p_ood"),
            "n_test": m.get("n_test"),
        },
        "rests_on": {
            "head_fitted_on": "torch embeddings",
            "app_will_use": "the Core ML export named above",
            "cosine_torch_to_coreml": cosine_to_torch,
            "note": "the thresholds were fitted on the torch side; the export is "
                    "a different encoder by this project's own definition, which "
                    "is why the identity above says so",
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, help="a narrowcast bundle directory")
    ap.add_argument("--coreml", required=True, help="the encoder .mlpackage")
    ap.add_argument("--variant", required=True,
                    help="the torch variant the .mlpackage was exported from; its "
                         "own transform is what the preprocessing is read off")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cosine-to-torch", type=float, default=None,
                    help="measured agreement between this export and the torch "
                         "encoder, recorded in the artifact (ENCODER_DECISION.md)")
    a = ap.parse_args()

    from plantid.deploy.coreml import preprocess_spec
    from plantid.features.pretrained import encoder_identity

    pre = preprocess_spec(a.variant)          # raises if it cannot be read
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pkg = Path(a.coreml)
    dest = out / pkg.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(pkg, dest)

    payload = build_payload(Path(a.bundle), pre,
                            encoder_identity(a.variant, a.coreml), pkg.name,
                            cosine_to_torch=a.cosine_to_torch)
    (out / "bundle.json").write_text(json.dumps(payload, indent=2))

    mb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e6
    kb = (out / "bundle.json").stat().st_size / 1e3
    print(f"{out}\n  {pkg.name}  {mb:.0f} MB\n  bundle.json  {kb:.0f} KB  "
          f"({len(payload['classes'])} classes, {payload['coreml']['dim']}-d)")
    print(f"  preprocessing: {pre['side']}px, {pre['interpolation']}")


if __name__ == "__main__":
    main()
