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
