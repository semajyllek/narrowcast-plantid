# Next steps

Written as a handoff. Everything below is committed and pushed; nothing is
running. Four repos, all on feature branches off `main`, none merged.

| repo | branch | what changed |
|---|---|---|
| `narrowcast` | `auditor-cut` | builder → auditor (−833 lines), utility profiles, the forager threat model |
| `narrowcast-plantid` | `regional-bundles` | the regional pipeline, and the measurements that decided it |
| `narrowcast-kws` | `narrowcast-0.2.0` | invocations updated for the 0.2.0 CLI |
| `narrowcast-derm` | `narrowcast-0.2.0` | same |

## Where it got to

**The pipeline runs end to end**: a place name → species survey → fetch →
embed → audit → card, no manual step. `plantid/data/regions.py` →
`regional_fetch.py` → `regional_embed.py` → `regional_scores.py`.

**Settled, with numbers:**

- **Encoder**: PlantCLEF2024 int8, **87 MB**, 0.962 top-1 on real Oregon field
  photographs, no backend pin. BioCLIP-1 int4 at 46 MB / 0.922 if size binds.
  (`ENCODER_DECISION.md`)
- **The original thesis**: not smaller — task-conditional distillation was closed
  three times — but **+27.9 points** better than filtering a general classifier to
  the same species, and able to name the ~75% of a regional flora that a
  7,806-class model has no class for. (`NARROW_THESIS_FINDINGS.md`)
- **Safety**: pooled over 8 splits and 672 hemlock rows, the `forage` utility
  profile takes *Conium maculatum* to **0.00%** named-from-your-list, against
  3.27% under `identify`, for ~17 points of label share.
  (`FORAGER_FINDINGS.md`)

## Do these first

### 1. Stratify declared hazards across the split — narrowcast

`Conium` reached a test half in **6 of 8** splits and `Cicuta` in **2 of 8**,
because `make_splits` shuffles near-OOD species and there are 23 of them. The card
correctly reports the misses as *unmeasured* rather than passed, but a single
audit is a coin-flip on whether your declared hazard was checked at all.

A declared hazard should be forced into both halves. Touch
`cascade.make_splits`, which now takes a `species` column it can key on.
Note `MAX_CLUSTER_SHARE` already exists there for the related problem.

### 2. Per-label cost — narrowcast

`UTILITY["wrong"]` is **one scalar over all labels**, and `fit_thresholds` never
sees the hazard list. `--profile forage` makes the whole model cautious; it cannot
make it cautious *about hemlock specifically*. The cheap version is an explicit
always-decline override at predict time. Per-label thresholds are a larger change
and want a prereg.

### 3. Finish Phase 3 — narrowcast

- **Near-OOD gate.** `1 − P(__OTHER__)` is statistically level with the
  centroid-geometry gate (−0.0901 vs −0.0995) and free: `Bundle.proba` already
  returns that column and discards it. Four-place seam — `cascade.decide` (third
  threshold), `build.fit_and_measure` (grid 2-D → 3-D), `predict.Bundle.predict`,
  the manifest.
- **`regional_ood` bucket.** Vestigial in `cascade.SPLIT_CLUSTER`; plantid has
  `OOD_MIX_REGIONAL` and calls it "the deployment-realistic one".
- **Encoder ↔ bundle binding.** `manifest["encoder"]` is a name string defaulting
  to `"unstated"`. Mixing a Core ML-embedded bundle with a torch-embedded
  background pool silently flattered label share by 3 points during this work.
- **float32 heads.** `head.npz` is 83 KB at K=20/D=512 because sklearn gives
  float64; casting recovers ~40 KB and is numerically free.

## Then

**More regions.** Everything is Oregon. `regions.py` has a `PLACES` registry and a
GBIF path that needs only `--state`.

**FIT vs FILTER at K ≥ 20.** The +27.9 rests on **7 species** — the overlap with
PlantCLEF's classes is the binding constraint. A European region would allow the
test at full size, and the prediction is that FIT's margin *shrinks* where the
general model actually knows the flora.

**Domain shift.** Still the one unmeasured axis, and now the largest. Every number
here comes from iNaturalist/GBIF photographs; nobody has pointed a different
camera at a plant. `DATA_STRATEGY.md` proposes herbarium specimens through GBIF as
a source that ships its own labels. This is data collection, not coding.

## Conventions that will bite a fresh session

- **Cluster, never row.** Row-level intervals have twice produced effects here
  that failed to replicate. `regional_embed` carries the GBIF occurrence key for
  exactly this; without it a regional bundle's intervals come out ~√2 too narrow,
  silently.
- **The caller's group column wins.** Deriving the group from
  `label.split()[0]` is a Latin-binomial convention and has now been wrong in
  **five** places. It is invisible when wrong because it always returns something
  plausible. `build.py` is fixed; check any new code that groups.
- **Declare utilities before fitting.** Selecting a profile from the *label set* is
  legitimate; selecting from measured outcomes is reading payoffs off the test set.
- **Quantization is architecture-specific and fails hard, not gradually.** int4
  costs FastViT 4pp of label share, **destroys** DINOv2 (cosine −0.005), and costs
  CLIP-style ViTs nothing. Validate against `load_encoder`'s own `preprocess`,
  never against `build_traceable` — that compares the export to itself.
- **int4 per-grouped-channel is silently wrong on the Metal GPU backend.** int8
  per-channel is correct everywhere and needs no pin.

## Two things I got wrong that are worth not repeating

**A partial measurement read as a verdict.** Twice: int4 "costs 14pp" (leaf-only;
two organs said 4pp), and "17.9 MB is best at K=20" (curated in-fine-tune images;
real regional data reversed it by 13 points). Both were reasonable to run and
wrong to conclude from.

**Detached vs tracked background jobs.** `nohup … &` inside a tool call detaches to
the OS and the harness tracks only the wrapper, so a finished job goes unnoticed —
once for eight hours. Use the background flag on the real process, and match the
poll interval to how fast the thing actually moves.

## Housekeeping

- Nothing merged to `main` in any repo; review is outstanding.
- `data/processed/` is ~13 GB and gitignored. The one irreplaceable thing in it —
  1,394 cached competitor API responses — is packed at
  `data/processed/competitor_cache/` and deliberately **not** committed, pending a
  redistribution-terms check (`analysis/cache/README.md`).
- The fine-tuned S2 checkpoint is 144 MB, untracked, one machine, sha256 in
  `data/ADAPTED_CHECKPOINT.md`. It needs LFS or an object store if it matters.
- The deep dive at `narrowcast/docs/deep_dive.html` is current and is the public
  artifact; it does **not** yet include any of the regional work.
