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

**There is a working app.** `plantbook` (`~/Documents/plantbook/app`, its own git
repo, own `CLAUDE.md`): 970 Oregon species, 87 MB encoder, ~157 ms per photograph
on an iPad, entirely offline. Two modes — the shipped head, and a user-chosen list
fitted on the phone from a bank of embeddings.

**It has been field-tested**, which closes the axis `DOMAIN_SHIFT_FINDINGS.md`
calls the last one. 45 photographs, 15 species, Dabney State Park, identified
against a printed guide and run through iNaturalist for comparison:

| | top-1 |
|---|---|
| ours | **12 / 14** |
| iNaturalist, vision only | 10 / 14 |
| iNaturalist + coordinates | 13 / 14 |

n = 14, so one species is seven points: this supports *comparable to their vision
model*, not better. (`FIELD_TEST_FINDINGS.md`)

**Settled, with numbers:**

- **Encoder**: PlantCLEF2024 int8, **87 MB**, no backend pin. (`ENCODER_DECISION.md`)
- **The original thesis**: not smaller, but **+27.9 points** over filtering a
  general classifier, and able to name the ~75% of a regional flora it has no
  class for. (`NARROW_THESIS_FINDINGS.md`)
- **Whole bank beats a user-chosen twenty for identification**, ~6× on correct
  identifications per plant photographed, because coverage rises faster than
  accuracy falls. Narrow still wins *on the chosen species* (0.963 vs 0.919), so
  both ship as modes. (`ONDEVICE_CATALOGUE_FINDINGS.md`)
- **Three photographs of one plant** are the largest lever: +5.9 points on GBIF
  material, **+15.4 on real phone photographs** — but they must be of the same
  subject. (`MULTIPHOTO_FINDINGS.md`)
- **Safety**: `forage` takes poison hemlock to 0% named-from-your-list against
  3.90% under `identify`; read 0% as the rule of three over clusters (≈4.3%).
  (`FORAGER_FINDINGS.md`)

## Do this first

### 1. Location

**The largest single improvement available, and nothing else is close.**
iNaturalist goes 67% → 92% on the same field photographs with coordinates; ours
has none and a phone has them for free. `LOCATION_FINDINGS.md` recorded a
geographic prior as worth ~0 at 490 regional species and +4.5pp to their
108k-taxa model — on real field photographs at a known point it is far larger
than either, which is the finding. A coarse regional catalogue is not the same as
knowing where you are standing.

GBIF occurrences carry coordinates and dates, so the prior is buildable from data
already fetched. Start by measuring what a *county-and-month* prior would be
worth on the Dabney photographs before building anything.

### 2. Run `FRAMING_PREREG.md`

Unrun, and it needs ~20 photographs that do not exist: 8–10 species, each shot
naturally and again with one leaf isolated against a hand. Endpoints and all
three possible conclusions are declared in the doc. Cheap, and it decides whether
the app should instruct users before their first photograph.

### 3. Top up the species still thin

The paging fix brought 964 species over four plants, but some common ones are
still light — *Acer macrophyllum* has 11 plants against *Phoradendron leucarpum*'s
19, and lost to it on a real photograph. Anything under ~15 is worth a top-up
pass; the fetch is resumable and the fix is in `regional_fetch.occurrences`.

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
