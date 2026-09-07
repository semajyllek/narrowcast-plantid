# The mix helps the species it has data for and hurts the ones it doesn't

Pre-registered in the addendum to `SOURCE_MIX_PREREG.md`, written before either
head was fitted.

`SOURCE_MIX_FINDINGS.md` priced the mix at **+7pp of species accuracy** against
the loss of a clean out-of-source evaluation, and proposed a middle path. Two
things came out of measuring it, and the second changes the decision.

1. **The evidential trade-off is much cheaper than assumed** — holding out less
   of the evaluation set costs *no* precision, because the intervals get
   **narrower**, not wider.
2. **But the mix is not a uniform gain.** Species with no in-source training data
   are **damaged** by it: −0.0278 [−0.0423, −0.0154], while species with in-source
   data gain +0.0770. That is the finding, and it is about the product rather
   than about evidence.

## M1 — the interval does not widen. It narrows.

Per species, hold out a fraction `h` of iNaturalist observations, train on
`P-full` plus the rest, score the held-out ones. Nested sets, fixed species set,
species-clustered bootstrap.

| holdout `h` | iNat rows in training | test photos | species top-1 | 95% CI | **width** |
|---|---|---|---|---|---|
| **1.0** — status quo | 0 | 11,818 | 0.7969 | [0.7687, 0.8226] | **0.0540** |
| 0.8 | 1,979 | 9,839 | 0.8310 | [0.8087, 0.8519] | 0.0433 |
| 0.6 | 4,037 | 7,781 | 0.8505 | [0.8306, 0.8690] | 0.0383 |
| **0.4** | 6,609 | 5,209 | **0.8663** | [0.8468, 0.8842] | **0.0375** |
| 0.2 | 8,609 | 3,209 | 0.8743 | [0.8539, 0.8933] | 0.0394 |

Holding out 40% instead of 100% buys **+6.9pp** and the interval gets *tighter* —
0.0375 against 0.0540.

The reason was declared in advance rather than discovered: the bootstrap
resamples **species**, and the species count is constant across the sweep. Only
photographs per species shrink. Precision here is governed by between-species
variance, and a better head has less of it — species that were scoring badly move
up toward the ones that were scoring well. **A smaller evaluation set measuring a
better model is measured more precisely, not less.**

So the framing in `SOURCE_MIX_FINDINGS.md` — "whether the remaining set is large
enough to keep the intervals useful" — was the wrong worry. That is not the
constraint.

> **What M1 does not buy.** It preserves no out-of-source claim. A held-out
> iNaturalist observation is still drawn from a corpus the head trained on, so
> "the head has never seen an iNaturalist photograph" dies for the whole model
> however the observations are split. M1 is an ordinary in-source,
> out-of-observation evaluation. Corrected in the pre-registration addendum
> before measuring, because `SOURCE_MIX_FINDINGS.md` described it wrongly.

## M2 — the result that should decide this

Reserve 20% of species at a fixed seed. **No iNaturalist row of a reserved
species ever enters training.** Train on `P-full` plus the iNaturalist rows of
the other 80%. Both heads scored on identical test rows.

| group | species | test photos | `P-full` | mixed head | **Δ** |
|---|---|---|---|---|---|
| **reserved** — out-of-source | 69 | 1,050 | 0.8033 | **0.7755** | **−0.0278 [−0.0423, −0.0154]** |
| mixed — in-source | 276 | 4,159 | 0.7942 | 0.8712 | +0.0770 [+0.0577, +0.0977] |

**The interval on the reserved species excludes zero.** Mixing in-source data for
80% of the catalogue makes the other 20% *worse* — not merely un-improved, but
2.8 points below where they were before the mix existed.

The mechanism is straightforward once stated. The head is a single multinomial
over all species, so classes compete in one argmax. Adding iNaturalist rows moves
the boundaries of the mixed classes toward iNaturalist statistics. A reserved
species is still represented only by Pl@ntNet photographs, so at test time on an
iNaturalist photograph it competes against neighbours that now fit the test
distribution better than it does. It loses ties it used to win. **The gain and
the damage are the same effect seen from two sides.**

### Replication on the deployable encoder

`bioclip2_cml4`. The damage is slightly *larger* on int4:

| group | `P-full` | mixed head | Δ |
|---|---|---|---|
| reserved — out-of-source | 0.8008 | 0.7647 | **−0.0361 [−0.0538, −0.0211]** |
| mixed — in-source | 0.7921 | 0.8677 | +0.0756 [+0.0569, +0.0956] |

M1 replicates too: 0.7891 at `h = 1.0` with width 0.0540, against 0.8602 at
`h = 0.4` with width 0.0376. Same shape, same conclusion — and the encoder that
would actually ship is the one that loses more on the reserved species.

## Why this matters more than the evidential question

The reserved position is not hypothetical, and it is not a fixed 20%.

- **The catalogue already has a permanent tail there.** 32 catalogue species are
  unevaluated because iNaturalist grades them "casual" — cultivated-only plants
  with little or no research-grade data. Those are exactly the species that
  cannot be given in-source rows, and they would take the −2.8pp.
- **It gets worse as a catalogue grows, which is the tool's whole premise.** The
  product direction is a user choosing their own species list. A newly added
  species arrives with whatever corpus it arrived with, and in-source data for it
  may not exist at all. Under a mixed head, every such species is penalised
  relative to the incumbents — **the mix systematically disadvantages exactly the
  plants a user adds.**

That is a structural argument against adopting the mix that has nothing to do
with protecting the evaluation set, and it survives even if the evidential
objection is waived entirely.

## What this changes about the +7pp

The headline in `SOURCE_MIX_FINDINGS.md` stands as measured, and it was measured
on species that all had in-source data. Read alongside M2 it should be stated
more precisely:

> **+7.7pp for species you have in-source data for, −2.8pp for species you do
> not.** The net depends entirely on the mix of those two populations in the
> catalogue you actually ship, and on how that mix moves as the catalogue grows.

For the current 490-species catalogue with in-source coverage of 465, the net is
strongly positive today. For a catalogue that grows by user request, it is not
obviously positive at all, and it degrades in the direction the product is
heading.

## M3 — per-class balancing does not fix it, as predicted

The named fix, pre-registered in addendum 2 **including the possibility that it
would retract M2 entirely**. The production head fits with
`class_weight="balanced"`; the analysis code inherited from `domain_shift.py`
does not. If the damage were a row-count artifact of unweighted fitting, M2 would
have been measuring the analysis script rather than the shipped configuration.

| group | arm | baseline | arm | Δ |
|---|---|---|---|---|
| **reserved** | unweighted | 0.8033 | 0.7755 | −0.0278 [−0.0423, −0.0154] |
| **reserved** | **`class_weight="balanced"`** | 0.8090 | 0.7878 | **−0.0211 [−0.0375, −0.0068]** |
| **reserved** | in-source capped at 10/species | 0.8033 | 0.7848 | **−0.0186 [−0.0287, −0.0095]** |
| mixed | unweighted | 0.7942 | 0.8712 | +0.0770 [+0.0577, +0.0977] |
| mixed | `balanced` | 0.8112 | 0.8780 | +0.0668 [+0.0482, +0.0866] |
| mixed | capped at 10/species | 0.7942 | 0.8605 | +0.0663 [+0.0489, +0.0852] |

**Both fixes reduce the damage and neither removes it.** Every reserved-species
interval still excludes zero. Replicated on `bioclip2_cml4`: −0.0361 unweighted,
−0.0203 balanced, −0.0185 capped, all excluding zero.

**M2 stands and is not retracted.** The prediction declared in advance was that
balancing would help at the margin and not eliminate the effect, because the
mechanism is not row count — it is that mixed classes have training rows drawn
from the *test* distribution and reserved classes do not. Equal weight on a class
whose rows match the test distribution still wins more argmaxes than equal weight
on a class whose rows do not. That is what the numbers show.

**The cap is the best trade of the three**, and it was the cheaper lever:

| arm | gain on mixed | damage to reserved | ratio |
|---|---|---|---|
| unweighted | +0.0770 | −0.0278 | 2.8 |
| `balanced` | +0.0668 | −0.0211 | 3.2 |
| **capped at 10** | +0.0663 | **−0.0186** | **3.6** |

Ten in-source photographs per species collects 86% of the available gain while
cutting the damage by a third — consistent with `SOURCE_MIX_FINDINGS.md`'s
finding that the gain saturates fast. If the mix is ever adopted, cap it.

## T1 — two heads fix it, and the prediction that said they wouldn't was wrong

Pre-registered in addendum 3, which declared: *"this will not fix it either …
any combination rule that improves mixed classes without improving reserved ones
reproduces it at reduced size."* **That was wrong.**

Two heads, both spanning the full label space — head P on every Pl@ntNet row,
head i on iNaturalist rows for the mixed species and Pl@ntNet rows for the
reserved ones — with the posteriors averaged. Against the Pl@ntNet-only head:

| arm | reserved | mixed |
|---|---|---|
| single mixed head (M2) | −0.0278 [−0.0423, −0.0154] | +0.0770 [+0.0577, +0.0977] |
| cap at 10 (M3, best so far) | −0.0186 [−0.0287, −0.0095] | +0.0663 [+0.0489, +0.0852] |
| **T1, two heads averaged** | **+0.0347 [+0.0140, +0.0572]** | **+0.0442 [+0.0273, +0.0622]** |
| T2, centring | −0.0207 [−0.0380, −0.0031] | +0.0751 [+0.0549, +0.0974] |
| T2, cap + centring | −0.0108 [−0.0277, +0.0065] | +0.0708 [+0.0511, +0.0920] |

**Under T1 no group is worse than the head that ships.** That is the property M2
said was missing and M3 could not recover.

### But most of T1's reserved gain is ensembling, not source handling

Averaging two heads is an ensemble, and ensembles improve things for reasons
having nothing to do with source. The control has head i's structure exactly —
same class composition, same row count for the mixed classes — but draws those
rows from **Pl@ntNet instead of iNaturalist**:

| arm | reserved | mixed |
|---|---|---|
| T1 | +0.0347 | +0.0442 |
| **T1-control, no iNaturalist anywhere** | **+0.0349** | −0.0222 |

`+0.0347` against `+0.0349`. **The entire reserved-species gain is the
architecture.** Reporting T1 as "+3.5pp for the species with no in-source data"
would have been wrong, and only the control shows it.

So the contrast that isolates what the *data* did:

| **T1 − T1-control** | `bioclip2` | `bioclip2_cml4` |
|---|---|---|
| reserved | **−0.0001 [−0.0134, +0.0126]** | −0.0022 [−0.0181, +0.0132] |
| mixed | +0.0664 [+0.0476, +0.0861] | +0.0615 [+0.0447, +0.0788] |

**Adding iNaturalist data under a two-head architecture contributes nothing to
the reserved species and +6.6pp to the mixed ones.** Not −2.8pp. The damage is
gone, on both encoders, and what is left is the gain with no one paying for it.

The control also explains itself: it *helps* reserved species and *hurts* mixed
ones, because weakening the mixed classes hands their competitors the argmax.
Everything in this document is the same competition effect seen from different
sides.

### T2 is out

Per-class logit centring on an out-of-catalogue iNaturalist reference pool
reduces the damage on fp32 and **fails on int4** — `−0.0337 [−0.0541, −0.0158]`
for the mixed head, still excluding zero. It is a mitigation that does not
survive the encoder that would ship.

## What T1 costs

It is not free, and the cost is not compute. Two heads are ~40 KB each against a
152 MB encoder, and the second forward pass is a matrix multiply, so deployment
cost is negligible. The cost is **accuracy on the species that do have in-source
data**:

| | reserved | mixed |
|---|---|---|
| single mixed head | 0.7755 | **0.8712** |
| T1 | **0.8381** | 0.8384 |

T1 gives up **3.3pp on the mixed species** to recover **6.3pp on the reserved**
ones. At the 80/20 split measured here that is a net loss in unweighted mean
accuracy — the single mixed head is better *on average* — and T1 is nonetheless
the right default, because it is the only configuration where **no species is
worse off than under the head that ships today**.

Which one wins on average depends entirely on the reserved fraction, and the
reserved fraction only grows: 32 catalogue species are "casual"-grade already,
and a user-chosen catalogue adds species that may have no in-source data at all.
**T1 is the choice that does not degrade as the product moves in the direction it
is going.**

## What is still untested

1. **Sweep the reserved fraction.** 20% was declared, not derived. It sets where
   the T1-versus-single-head crossover falls, and that crossover is now the
   decision variable rather than the damage itself.
2. **A weighted or routed combination** instead of a flat average. T1 uses 0.5/0.5
   with nothing tuned. A weight fitted on the calibration split might recover part
   of the 3.3pp it gives up on the mixed species, and by this project's
   conventions the weight has to be declared before it is fitted.

## Reproduce

```
PYTHONPATH=. .venv/bin/python -m analysis.source_mix_middle --variant bioclip2
PYTHONPATH=. .venv/bin/python -m analysis.source_mix_middle --variant bioclip2_cml4
```
