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
  profile takes poison hemlock to **0% named-from-your-list** against **1.70%**
  under `identify`, for 16.2 points of label share — and `p_ood = 0.4` eliminates
  every breakthrough under every profile. Read "0%" as the rule of three over
  clusters (**≈4.3%** upper bound), not a demonstrated zero.
  (`FORAGER_FINDINGS.md`, re-measured against narrowcast `9e36d0c`.)
  *The earlier "0.00% against 3.27%, 672 hemlock rows" is retired; the doc
  explains where 672 came from.*

## Landed since the last handoff

narrowcast `auditor-cut` is at **142 tests**. Twelve commits, `bde128f`…`99cd984`.

| what | commit | note |
|---|---|---|
| declared hazards stratified into both halves | `bde128f` | ends the coin flip; `cascade.hazard_rows` unifies the two threat models' predicates |
| `--never-answer` (per-label cost) | `292b269` | suppresses the **look-alike**, not the hazard |
| the reject class is not a user label | `7b678dd` | `__OTHER__` was counted among `--scores` labels |
| near-OOD gate `--gate-near-ood` | `5a9cc41`, `f9073e0` | declines rather than retreats; fitted, so it can turn itself off |
| `regional_ood`, encoder binding, float32 heads | `38f474d`, `e9c073e` | the last three Phase 3 items |
| encoder declaration beats the flag | `cbca6a3`, `89f67f7` | after measuring the geometry check |
| bucket splits keyed on contents | `9e36d0c`, `99cd984` | a relabelling was moving coverage 12 points |

Here: `FORAGER_FINDINGS.md` re-measured, `NEAR_OOD_FINDINGS.md` extended with the
reject arm on real data, `SPACE_CHECK_FINDINGS.md` written, the regional pipeline
declaring its encoder and flagging its regional rows.

**Three results worth carrying, all of them negative:**

- **The embedding-space check is blind to the failure it was written for.** 0 of
  21 export/quantization pairs caught, including torch BioCLIP-2 against its own
  Core ML int4 at every organ; 2 of 39 false positives. Not promoted to a
  refusal. What catches it is **declaration**, not geometry.
  (`SPACE_CHECK_FINDINGS.md`)
- **The near-OOD gate is marginal.** Adopted in 8 of 8 `identify` seeds and 5 of 8
  under `forage`, buying 1.6 points of near-OOD error for 2.4 of label share.
  Consistent with this repo's original null, not a win over it.
- **Family grouping never once produced a warning.** `warned_at_group` is 0.000
  across all 48 forager arms; every safe outcome comes from declining.

## Do this first

~~Write the `encoder` field from the remaining npz producers~~ — **done**. Every
npz this project writes now declares what produced it:
`plantid/features/pretrained.encoder_identity` is the canonical definition,
`store.save_descriptors`, `embed_catalog`, `embed_background`, `embed_inat`,
`regional_embed` and `export_for_narrowcast` all write it, and the two points
where separately embedded pools are combined — `regional_scores.build_scores`
and `export_for_narrowcast` — refuse inputs whose declarations disagree.
`export_for_narrowcast` carries the caches' own declaration through rather than
taking it from `--variant`, because the point is to describe the vectors.

**The caches on disk predate the field**, so they declare nothing and the checks
fall back to the geometric warning that sees 0 of 21 export pairs. That is the
one loose end: re-embedding is expensive and not obviously worth it on its own,
but anything re-embedded from here carries the declaration, and a pool mixed with
one that does will be caught.

Nothing else is blocked. What follows is data collection.

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

## How to report a measurement here

Learned the hard way this pass; each of these was nearly got wrong.

- **The cluster bootstrap cannot express a zero-event rate.** It returns
  `[0, 0]`, which is not an interval. Use the rule of three over *clusters*
  (3/70 ≈ 4.3% on the forager bundle). This recurs anywhere a safety rate is
  reported as zero.
- **Spread across seeds is not uncertainty.** Test halves overlap, so it
  understates — structurally the same error as counting 672. Report it as *split
  sensitivity*, separately, or not at all.
- **Re-run when the tool moves underneath.** `9e36d0c` changed every threshold in
  the repo and made numbers written one commit earlier stale: *Conium* under
  `identify` went 3.90% → 1.70%. The conclusions held; the figures did not. Two
  findings docs carry the corrected values and say so.
- **Do not call a near miss a reproduction.** 1.70% is not 3.27%. When the
  original was never recorded, the gap cannot be attributed — say that.

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
