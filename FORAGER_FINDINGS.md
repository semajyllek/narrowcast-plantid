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

## Re-measured against narrowcast at `e9c073e`

*Supersedes the table above, and replaces the "treat as unrecorded" note that
stood here.* Everything below is `forager_big_scores.npz` — 26 labels, 27 classes
with the reject class, 1,158 out-of-list rows, **112 *Conium maculatum* rows over
70 clusters** and 172 *Cicuta douglasii* over 70. The original table used the
588-row file, where *Conium* has only 22 rows; the arithmetic in the next section
identifies the big file as the source of the figure that had no home.

### Where "672 hemlock rows" came from

**112 × 6 = 672.** *Conium* has 112 rows in this file, and under the old split it
reached a test half in **6 of 8 seeds**. The unrecorded claim — "pooled over 8
splits and 672 hemlock rows" — was those 112 rows counted once per split in the
six splits that happened to contain them. It is 112 distinct observations, not
672, the two splits where the hazard vanished were silently dropped, and "8
splits" describes a loop that ran eight times rather than evidence from eight.
*Cicuta* reached test in **2 of 8**, so its numbers rested on a quarter of the
data with nothing saying so.

The false precision and the bug are the same thing: `make_splits` shuffled
near-OOD species by genus, each hazard was its own genus, and reaching the test
half was a coin flip. narrowcast `bde128f` stratifies declared hazards into both
halves; both hazards are now measured in **8 of 8** seeds.

### The measurement

`p_ood` sweep at seed 0, with the cluster bootstrap over the hazard's own
clusters — the interval `outside_hazard_metrics` computes, not the spread across
seeds. See the caveat on both below.

| profile | `p_ood` | label share | *Conium* named from the list | *Cicuta* |
|---|---|---|---|---|
| `identify` (wrong −2) | 0.2 | 0.804 | **5.36%** [0.0, 12.1] | 0% |
| | 0.4 | 0.665 | 0% | 0% |
| | 0.6 | 0.646 | 0% | 0% |
| `standard` (wrong −4) | 0.2 | 0.778 | 1.79% [0.0, 5.5] | 0% |
| | 0.4 | 0.646 | 0% | 0% |
| | 0.6 | 0.532 | 0% | 0% |
| `forage` (wrong −20) | 0.2 | 0.608 | **0%** | 0% |
| | 0.4 | 0.532 | 0% | 0% |
| | 0.6 | 0.532 | 0% | 0% |

**The original conclusion survives: `p_ood = 0.4` eliminates every breakthrough**,
now on 112 hemlock rows rather than 22 and under every declared profile. So does
the direction of the profile result — `forage` takes *Conium* to zero at the
garden-app operating point where `identify` does not.

Averaged over 8 seeds at `p_ood = 0.2`: *Conium* is **3.90%** under `identify`
and **0.00%** under `forage`, for **18.2 points** of label share (0.809 → 0.627).

**The unrecorded claim said 3.27% and ~17 points. Do not call this a
reproduction.** The shape and magnitude survive; the exact figures do not match
and the gap cannot be attributed, because the original was never written down. It
was also computed on a split that dropped two seeds for *Conium* and six for
*Cicuta*. Quote the numbers in this section.

### Two caveats on the intervals, both of which the earlier draft would have got wrong

**The cluster bootstrap cannot express a zero-event rate.** Every `forage` cell
above returns `[0.0, 0.0]`, which is not an interval — every resample of 70
clusters containing no event contains no event. The honest upper bound is the
rule of three over *clusters*: **3/70 ≈ 4.3%**. Read "0%" as "no breakthrough in
70 subjects", not as a demonstrated zero.

**Spread across seeds is not uncertainty.** Each seed's test half holds ~58 of
the same 112 rows, so variation between seeds is the variation of overlapping
subsamples and *understates* uncertainty — structurally the same error as
counting 672. Reported separately as split sensitivity at `p_ood = 0.2`:

| profile | *Conium* across 8 seeds | *Cicuta* |
|---|---|---|
| `identify` | 1.56 – 6.35% | 0.00 – 1.16% |
| `standard` | 0.00 – 1.79% | 0.00 – 1.16% |
| `forage` | 0.00 – 0.00% | 0.00 – 0.00% |

That a single audit at `identify` can report anywhere from 1.6% to 6.4% — either
side of `card.HAZARD_BAR` — is the argument for reading the interval and not the
point, and it is what the stratification made visible rather than what it caused.

### Family grouping never once produced a warning

The section above argues family grouping matters twice over, because "it is an
umbellifer" warns the person holding the root where "it is a *Lomatium*" does
not. **On this bundle that second reason is worth nothing: `warned_at_group` is
0.000 across all 48 profile × seed × hazard arms.** Every safe outcome here comes
from declining, not from a coarse warning. The first reason stands — family
grouping is what puts the hazards in `near_ood` rather than the easy bucket, and
that is most of the value.

This bounds the note in `NEAR_OOD_FINDINGS.md`, which records that this project
reads a coarse answer on an out-of-list row as a warning while narrowcast's
`utility` reads it as wrong. That disagreement is real, and on the forager bundle
it is also empty: the warning never fires, so nothing here turns on it. The
retreat arm won on the 490-species genus-grouped catalogue, not on this one.
