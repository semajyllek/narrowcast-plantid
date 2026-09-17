# Poison hemlock, named as wild carrot — and a threat model narrowcast cannot express

A forager's Oregon bundle: 26 labels across 17 families, grouped by **family**,
including three edible umbellifers — *Daucus carota*, *Osmorhiza berteroi*,
*Lomatium triternatum*. Out of list: five Apiaceae from `safety_pairs.py`'s
pre-registered pairs, among them *Conium maculatum* and *Cicuta douglasii*.
PlantCLEF2024 int8, cluster-disjoint, real GBIF field photographs.

## Family grouping does what it was meant to

Under genus grouping every hazard bucketed as `distant_ood` — the bucket a
previous run answered on 0.4% of rows — because *Conium* is not *Lomatium*. Under
family grouping all 588 out-of-list rows bucket as **`near_ood`**, and the hazards
are measured in the hard bucket rather than the easy one.

## The measurement

| `p_ood` | *Conium maculatum* | *Cicuta douglasii* |
|---|---|---|
| **0.2** | **4.5% named** — as *Daucus carota* | 0% named, 100% declined |
| 0.4 | 0% named, 100% declined | 0% named, 100% declined |
| 0.6 | 0% named, 100% declined | 0% named, 100% declined |

At a garden-app operating point, **poison hemlock was named as wild carrot** —
the textbook fatal confusion, reproduced from a photograph. Raising the declared
out-of-list rate to 0.4 eliminates every breakthrough.

**State the uncertainty plainly: that is one photograph out of 22, over 14
plants.** The exact binomial interval is **[0.1%, 22.8%]**. The point estimate of
4.5% is four and a half times the 1% bar `card.HAZARD_BAR` declares, and the
interval is wide enough that the honest reading is *"this happens, and this design
controls it"* rather than *"this happens 4.5% of the time."* A real safety claim
needs hundreds of hemlock photographs, not 22.

What is not uncertain is the direction and the lever: `p_ood` is what moves it,
which is exactly the remedy the card already prints when the hazard gate fails.

## narrowcast cannot measure this, and the refusal is correct

```
these --hazard labels are not in the label set: Conium maculatum, Cicuta douglasii
A hazard is a label you already have; naming one you do not measures nothing.
```

`cli._hazard_arg` hard-errors, and `build.hazard_metrics` iterates test rows where
`truth == label` for each declared hazard. Both assume the hazard is **on the
list**. That models: *"my list contains something dangerous; how often is it given
a harmless name?"*

A forager's threat model is the mirror image: *"the dangerous plant is deliberately
**not** on my list — how often is it given the name of something on it?"* Nobody
puts poison hemlock on a list of things they intend to eat.

Both are real. narrowcast supports only the first, and its guard is right to reject
the second rather than silently measure nothing — but the second is the one a
foraging product needs, and it is measurable from exactly the data the `--scores`
path already produces. The gap is a feature, not a bug to patch quietly:

- **in-list hazard** — `hazard_metrics` today: union of harmless names given to a
  dangerous listed species
- **out-of-list hazard** — not implemented: rate at which a dangerous *unlisted*
  species receives any in-list label, with group answers counted as **safe** when
  the named group contains the hazard

That second clause is why family grouping matters twice over. "It is an
umbellifer" is a true statement that correctly warns the person holding the root,
so it should not count against the model — whereas under genus grouping, "it is a
*Lomatium*" is a species-level claim wearing the clothes of caution.

## A fourth sighting of the same bug

The summary print in `regional_scores.py` computed its listed groups as
`{s.split()[0] for s in listed}` — the genus, from the label string, ignoring the
group column the pipeline had just written. It reported **zero** relatives while
narrowcast correctly bucketed 588.

That is the same defect as `narrowcast/build.py:408`, which `labels.py:126` already
calls "the third place this default has broken a non-binomial domain". This was the
fourth, written by someone who had spent the previous hour documenting the other
three. The lesson is not that the rule is hard to remember — it is that a
first-whitespace-token default is *invisible* when it is wrong, because it always
returns something plausible.

## Re-measured against narrowcast at `9e36d0c`

*Supersedes the table above.* All of it is `forager_big_scores.npz` — 26 labels,
27 classes with the reject class, 1,158 out-of-list rows, **112 *Conium
maculatum* rows over 70 clusters** and 172 *Cicuta douglasii* over 70. The
original table used the 588-row file, where *Conium* has only 22 rows.

### Where "672 hemlock rows" came from

**112 × 6 = 672.** *Conium* has 112 rows in this file, and under `make_splits`
**as it stood when that claim was made**, it reached a test half in **6 of 8
seeds** — the count recorded in the handoff at the time, and reproduced exactly
when this was first investigated. The unrecorded claim of "pooled over 8 splits
and 672 hemlock rows" was those 112 rows counted once per split in the six splits
that happened to contain them. It is 112 distinct observations, not 672; the two
splits where the hazard vanished were silently dropped; and "8 splits" describes a
loop that ran eight times rather than evidence from eight. *Cicuta* reached test
in **2 of 8**, so its numbers rested on a quarter of the data with nothing saying
so.

The false precision and the bug are the same thing: `make_splits` shuffled
near-OOD species by genus, each hazard was its own genus, and reaching the test
half was a coin flip. narrowcast `bde128f` stratifies declared hazards into both
halves, and both are now measured in **8 of 8**.

*(Those 6/8 and 2/8 counts are historical. `9e36d0c` changed how each bucket's
shuffle is seeded, so the same unstratified code at HEAD gives 5/8 and 4/8. The
arithmetic explaining 672 is about the code that produced the number, not about
HEAD — which is the point of writing it down.)*

### The measurement

`p_ood` sweep at seed 0, with the cluster bootstrap over the hazard's own
clusters — the interval `outside_hazard_metrics` computes, not the spread across
seeds. Both carry caveats, below.

| profile | `p_ood` | label share | *Conium* named from the list | *Cicuta* |
|---|---|---|---|---|
| `identify` (wrong −2) | 0.2 | 0.762 | **1.79%** [0.0, 5.5] | 0% |
| | 0.4 | 0.642 | 0% | 0% |
| | 0.6 | 0.576 | 0% | 0% |
| `standard` (wrong −4) | 0.2 | 0.728 | 1.79% [0.0, 5.5] | 0% |
| | 0.4 | 0.576 | 0% | 0% |
| | 0.6 | 0.530 | 0% | 0% |
| `forage` (wrong −20) | 0.2 | 0.608 | **0%** | 0% |
| | 0.4 | 0.530 | 0% | 0% |
| | 0.6 | 0.530 | 0% | 0% |

**The original conclusion survives: `p_ood = 0.4` eliminates every breakthrough**,
now on 112 hemlock rows rather than 22 and under every declared profile. So does
the profile result — `forage` takes *Conium* to zero at the garden-app operating
point where `identify` does not.

Averaged over 8 seeds at `p_ood = 0.2`: *Conium* is **1.70%** under `identify`
and **0.00%** under `forage`, for **16.2 points** of label share (0.770 → 0.608).

**The unrecorded claim said 3.27% and ~17 points.** The label-share cost lands
close; the rate does not, and the gap cannot be attributed because the original
was never written down and was computed on a split that dropped two seeds for
*Conium* and six for *Cicuta*. Quote this section.

### Two caveats on the intervals

**The cluster bootstrap cannot express a zero-event rate.** Every `forage` cell
returns `[0.0, 0.0]`, which is not an interval — every resample of 70 clusters
containing no event contains no event. The honest upper bound is the rule of
three over *clusters*: **3/70 ≈ 4.3%**. Read "0%" as "no breakthrough in 70
subjects", not as a demonstrated zero.

**Spread across seeds is not uncertainty.** Each seed's test half holds ~58 of the
same 112 rows, so variation between seeds is variation of overlapping subsamples
and *understates* uncertainty — structurally the same error as counting 672.
Reported separately as split sensitivity at `p_ood = 0.2`:

| profile | *Conium* across 8 seeds | *Cicuta* |
|---|---|---|
| `identify` | 0.00 – 5.08% | 0.00 – 1.11% |
| `standard` | 0.00 – 1.79% | 0.00 – 1.11% |
| `forage` | 0.00 – 0.00% | 0.00 – 0.00% |

A single `identify` audit can report anywhere from 0% to 5.1% — either side of
`card.HAZARD_BAR`. That is the argument for reading the interval rather than the
point, and it is what the stratification made visible rather than what it caused.

### Family grouping never once produced a warning

The section above argues family grouping matters twice over, because "it is an
umbellifer" warns the person holding the root where "it is a *Lomatium*" does
not. **On this bundle the second reason is worth nothing: `warned_at_group` is
0.000 across all 48 profile × seed × hazard arms.** Every safe outcome here comes
from declining. The first reason stands — family grouping is what puts the hazards
in `near_ood` rather than the easy bucket, and that is most of the value.

This bounds the note in `NEAR_OOD_FINDINGS.md`, which records that this project
reads a coarse answer on an out-of-list row as a warning while narrowcast's
`utility` reads it as wrong. The disagreement is real and, on the forager bundle,
empty: the warning never fires. The retreat arm won on the 490-species
genus-grouped catalogue, not on this one.

### A note on re-measuring

Every number in this section was re-run at `9e36d0c` after that commit changed
how `make_splits` seeds each bucket. The first draft of this section, written one
commit earlier, gave 3.90% and 18.2 points where this gives 1.70% and 16.2 — a
useful measure of how much a split choice moves a rate estimated on 70 clusters,
and a reminder to re-run rather than assume when the tool moves underneath.
