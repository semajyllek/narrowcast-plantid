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
- **Safety**: ~~pooled over 8 splits and 672 hemlock rows, the `forage` utility
  profile takes *Conium maculatum* to **0.00%** named-from-your-list, against
  3.27% under `identify`, for ~17 points of label share.~~
  **Do not quote this.** It is in no findings doc and in neither repository's
  history — it was run and reported without being written down, which is the
  failure the "every number traces to a findings doc" rule exists to prevent. The
  pooling was also over the coin flip that `bde128f` removed, so it cannot be
  carried forward either. See the note appended to `FORAGER_FINDINGS.md`;
  re-running it under the new split is the first thing that belongs in that file.

## Done since this was written

Both of the first two are landed in `narrowcast` on `auditor-cut`, with tests.

- **Stratify declared hazards across the split** — `bde128f`. A declared hazard is
  forced into both halves, halved at the cluster, drawn from a generator keyed on
  its own name so declaring one moves nothing else. The predicate was the part
  that could have failed quietly: `hazard_metrics` keys on `truth` and
  `outside_hazard_metrics` on `species` and out-of-list, so stratifying under one
  and measuring under the other would still have printed "unmeasured" with nothing
  to show the miss — `cascade.hazard_rows` is now the union, shared by the split
  and both metrics. Note the headline numbers are *not* comparable across a
  declaration: the hazard's own rows change sides, the calibration composition
  moves with them, and the fitted operating point moves too (up to 8 points of
  label share on the test fixture). Also fixed a crash this made reachable —
  `card._hazard_section` compared an unmeasured hazard's `None` rate to the bar.

- **Per-label cost** — `292b269`. `--never-answer LABEL`: when the cascade would
  name that label, it declines. **Suppress the look-alike, not the hazard** —
  suppressing the hazard is inert, because `hazard_metrics` counts rows whose
  prediction is *not* the hazard, so those rows were never in the numerator. The
  card used to recommend the inert version and now names the look-alikes from
  `named_as`, which was added to the in-list path for this. Applied after the fit,
  never inside it, so the cost stays a printable delta. Read by `predict` too.
  Per-label *thresholds* are still the larger change and still want a prereg.

- **`FORAGER_FINDINGS.md` numbers no longer trace** — `ba38147`. Two reasons, and
  the second is the serious one: the pooled 8-split forage result quoted below is
  in no findings doc and in neither repository's history. Treat it as unrecorded.

## Item 3 is done; what is left is measurement, not code

All four Phase 3 sub-items are landed in `narrowcast` on `auditor-cut`
(`5a9cc41`, `38f474d`, `e9c073e`), and narrowcast is at 134 tests.

- **Near-OOD gate** — `--gate-near-ood`. Declines rather than retreating, and the
  reason is a disagreement between the two repos that is now written up in
  `NEAR_OOD_FINDINGS.md`. Fitted, so a card whose fit turns it off says so.
- **`regional_ood`** — real, and the caller declares it: an optional `regional`
  boolean column on the scores npz. With it the mix becomes `OOD_MIX_REGIONAL`
  and leftover `distant_ood` rows carry weight **zero**.
- **Encoder ↔ bundle binding** — the bundle stores the direction its training
  vectors point in; `audit` and `predict` both compare against it. Different
  vector widths are refused, orthogonal geometry only **warns**. See the open
  task below.
- **float32 heads** — already true. `_vecs` casts to float32 and sklearn keeps
  the dtype, so the 83 KB figure predated that. Pinned by a test.

## Do this first

### 1. Re-measure the findings docs against the current tool

**This is the highest-value task in this repo and it outranks more regions.** Two
findings docs now carry addenda saying their headline numbers predate changes to
the tool, and both describe behaviour narrowcast no longer has:

- `FORAGER_FINDINGS.md` — the split changed under it (declared hazards are
  stratified into both halves, so a rerun sees about half the hemlock test rows
  and a wider interval), and the pooled 8-split `forage` result quoted in this
  file has **no findings entry anywhere**. Treat it as unrecorded.
- `NEAR_OOD_FINDINGS.md` — concluded "not shipped"; narrowcast ships the reject
  arm. The note there explains why, but nothing has been re-measured.

Everything needed exists: the bundles, `--profile`, `--gate-near-ood`,
`--never-answer`. This is running things, not writing them.

### 2. Measure the embedding-space check's false-positive rate

`build.check_same_space` warns instead of refusing, and that is the only thing
keeping it a warning. It rests on the premise that one encoder's embeddings share
a common cone, so a cross-pool cosine near zero means two encoders. Nothing in
narrowcast can load an encoder to test that — **this repo can**. Embed one pool
with several encoders and cross-compare; embed genuinely unrelated subject matter
with a *single* encoder and confirm it does not trip. If the false-positive rate
is negligible, promote it to a refusal, which is what the recorded failure
deserves: a Core ML-embedded bundle against a torch-embedded background pool
flattered label share by three points in silence.

### 3. Feed `regional_ood` from the regional pipeline

`regional_embed.py` already writes the npz; adding a `regional` boolean column is
a small change. It is what makes the Oregon numbers deployment-realistic instead
of flattered by tropical filler, which is this project's own argument for the
bucket.

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
  silently. **One declared exception since `bde128f`**: a hazard named by
  `--hazard` or `--hazard-absent` is stratified into both halves on purpose, so
  the whole-genus rule breaks for that one genus. The cluster itself is still
  never split — a single-cluster hazard goes wholly to test and loses its
  interval.
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
