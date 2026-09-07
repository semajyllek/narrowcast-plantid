# Pre-registration — is there accuracy left on the table by training on one source?

Written **before** any head in this design was fitted. It extends
`DOMAIN_SHIFT_PREREG.md` / `DOMAIN_SHIFT_FINDINGS.md` and reuses their splits,
inclusion rule and bootstrap unchanged.

## The observation that prompts it

The source-shift 2 × 2 was built to measure a penalty. It also turned up
something nobody went looking for. At a matched 10-photograph budget over 359
shared species, tested on the *same* held-out iNaturalist photographs:

| head fitted on | species top-1 on iNaturalist |
|---|---|
| Pl@ntNet-300K — **the head that ships** | 0.7691 |
| iNaturalist | **0.8615** |

**9.2 percentage points, and the production head has never seen an iNaturalist
photograph.** The training corpus and the deployment corpus differ by historical
accident — the catalogue was built from Pl@ntNet and the evaluation set was
fetched later from iNaturalist — not by any decision. Nobody has tested the mix.

## The claim under test

Adding in-source training data to the shipped head raises species accuracy on
held-out iNaturalist observations. Stated as a null to be rejected: **mixing the
two sources performs no better than Pl@ntNet alone.**

## Design

Closed-set, organ-free, photo-level, macro-averaged over species, paired
species-cluster bootstrap over 2,000 resamples — identical in every respect to
the primary arm of `analysis/domain_shift.py`, so the numbers sit in the same
table as the ones already published. Every arm is scored on the **same** held-out
iNaturalist photographs, from observations no arm trained on.

`B = 10` photographs per species per source, as before. Pl@ntNet supplies a mean
of 62.6 training photographs per species; the iNaturalist training side supplies
about 16 after the observation-level split.

| arm | fitted on | question it answers |
|---|---|---|
| **`P-full`** | every Pl@ntNet training row | the production baseline |
| **`P-full + i`** | that, plus all iNaturalist training photographs | **primary:** does adding in-source data help what ships? |
| `P-2B` | 20 Pl@ntNet photographs/species | fixed-volume control |
| `P-B + i-B` | 10 of each | **secondary:** at *equal volume*, is in-source data worth more? |
| `i-only` | 10 iNaturalist photographs/species | where the ceiling sits |

**The two questions are different and both are declared.** `P-full + i` has more
training data than `P-full`, so a gain there could be volume rather than source.
The fixed-volume pair isolates source composition: same number of rows, half of
them swapped for in-source ones. A gain on the primary with a null on the
secondary would mean "more data helps"; a gain on both would mean "in-source data
is worth more per row."

**Primary endpoint.** Species top-1 on held-out iNaturalist photographs,
`P-full + i` minus `P-full`, paired over species.
**Secondary.** The fixed-volume contrast; genus top-1 for both; and the same
arms scored on held-out **Pl@ntNet** photographs, to check whether mixing costs
anything on the source the head came from.

## Ways this comes out uninformative, declared in advance

1. **The iNaturalist side is small.** About 16 training photographs per species
   against Pl@ntNet's 62.6. Adding 16 rows to 62 may simply be swamped, and a
   null on the primary would then be a statement about *volume available*, not
   about the value of in-source data. The fixed-volume arm exists precisely so
   that this failure mode is still informative.
2. **The test set is smaller than the published one.** Only the held-out 40% of
   iNaturalist observations can be scored here, so intervals are wider than in
   `DOMAIN_SHIFT_FINDINGS.md` and a small effect will not resolve.
3. **Species composition is the same 359** as the source-shift arm, by the same
   declared inclusion rule. Species that cannot supply both a training and a test
   slice on both sides are out, which is not the production label set.

## The cost of a positive result, stated before it is one

This matters more than the measurement and is the reason to report the number
before changing anything that ships.

**iNaturalist is currently a pure held-out source.** That is exactly what makes
`DOMAIN_SHIFT_FINDINGS.md`'s central claim true — the head has never seen an
iNaturalist photograph, so every headline figure in this repo is an
*out-of-source* measurement, which is a much stronger thing to be able to say
than an in-source one.

Spending part of it on training destroys that. The remaining evaluation set gets
smaller and stops being cleanly out-of-source, and the contamination and
source-shift results would both need re-qualifying against a new split.

So a positive result here is **not** an instruction to adopt it. It is a price
tag: *this much accuracy is available, and this is what it costs in evidential
standing.* Which side of that trade is right is a product judgement and belongs
to the repo's owner, not to this experiment.
