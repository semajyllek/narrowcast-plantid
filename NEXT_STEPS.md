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
- **Safety**: on 112 *Conium maculatum* rows over 70 clusters, the `forage`
  profile takes poison hemlock to **0% named-from-your-list** against **3.90%**
  under `identify`, for 18.2 points of label share — and `p_ood = 0.4` eliminates
  every breakthrough under every profile. Read "0%" as the rule of three over
  clusters (**≈4.3%** upper bound), not a demonstrated zero.
  (`FORAGER_FINDINGS.md`, re-measured against narrowcast `e9c073e`.)
  *The earlier "0.00% against 3.27%, 672 hemlock rows" is retired: it counted 112
  rows once per split in the 6 of 8 splits that happened to contain them.*

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

~~### 1. Re-measure the findings docs against the current tool~~ — **done**,
`a1956ff`. `FORAGER_FINDINGS.md` carries the re-measurement and supersedes its own
table; `NEAR_OOD_FINDINGS.md` carries the reject arm on real data. Three things
came out of it worth carrying forward:

- **Where 672 came from**: 112 *Conium* rows × the 6 of 8 splits that contained
  them. The false precision and the coin-flip bug were one thing.
- **The cluster bootstrap cannot express a zero-event rate** — it returns
  `[0, 0]`. Use the rule of three over *clusters*. This will recur anywhere a
  safety rate is reported as zero.
- **Spread across seeds is not uncertainty.** Test halves overlap, so it
  understates. Report it as split sensitivity or not at all.

### 1. Measure the embedding-space check's false-positive rate

`build.check_same_space` warns instead of refusing, and that is the only thing
keeping it a warning. It rests on the premise that one encoder's embeddings share
a common cone, so a cross-pool cosine near zero means two encoders. Nothing in
narrowcast can load an encoder to test that — **this repo can**. Embed one pool
with several encoders and cross-compare; embed genuinely unrelated subject matter
with a *single* encoder and confirm it does not trip. If the false-positive rate
is negligible, promote it to a refusal, which is what the recorded failure
deserves: a Core ML-embedded bundle against a torch-embedded background pool
flattered label share by three points in silence.

### 2. Feed `regional_ood` from the regional pipeline

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
