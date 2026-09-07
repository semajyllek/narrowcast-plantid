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

## The sweep — the damage is worst when it looks least important

Reserved fraction `r` swept, everything else held fixed. Both damage predictions
in addendum 4 held; the crossover prediction did not.

| `r` | reserved species | **single head: Δ reserved** | Δ mixed | **T1 data effect on reserved** |
|---|---|---|---|---|
| 0.05 | 17 | **−0.0472** | +0.0698 | +0.0135 [−0.0110, +0.0486] |
| 0.10 | 34 | −0.0370 | +0.0744 | +0.0026 [−0.0143, +0.0207] |
| 0.20 | 69 | −0.0278 | +0.0770 | −0.0001 [−0.0134, +0.0126] |
| 0.35 | 121 | −0.0199 | +0.0786 | +0.0012 [−0.0120, +0.0132] |
| 0.50 | 172 | −0.0207 | +0.0754 | +0.0029 [−0.0037, +0.0097] |

**The damage grows as the reserved fraction shrinks**, as predicted, and the
direction is the uncomfortable one. A species without in-source data loses 4.7pp
when only 5% of the catalogue is in its position, against 2.1pp when half of it
is. It competes against however much of the field improved, so **the fewer such
species there are, the worse each one is treated.** On `bioclip2_cml4` the same
shape and steeper: −0.0587 at `r = 0.05` falling to −0.0284 at `r = 0.50`.

That matters most for the case the product creates. **A user adding one new
species to an established catalogue sits at `r → 0`** — off the left end of this
sweep, at its worst point. The newly added species is the maximally damaged one.

**T1's protection holds everywhere.** The architecture-matched data effect on
reserved species includes zero at every `r` on both encoders. Whatever the
composition, adding in-source data under two heads costs the species that lack it
nothing.

### The crossover, and a prediction that was wrong

Addendum 4 predicted break-even near `r ≈ 0.3`. Catalogue-mean accuracy, macro
over all species:

| `r` | `P-full` | single mixed head | T1 | **T1 − single** |
|---|---|---|---|---|
| 0.05 | 0.7960 | 0.8601 | 0.8509 | −0.0092 |
| 0.10 | 0.7960 | 0.8594 | 0.8477 | −0.0117 |
| 0.20 | 0.7960 | 0.8521 | 0.8383 | −0.0138 |
| 0.35 | 0.7960 | 0.8401 | 0.8322 | −0.0079 |
| 0.50 | 0.7960 | 0.8235 | 0.8243 | **+0.0008** |

Break-even is near `r ≈ 0.48`, not 0.3 — and on `bioclip2_cml4`, **the encoder
that would actually ship, it is never reached in range**: −0.0136 at `r = 0.05`
narrowing to −0.0051 at `r = 0.50`, still negative.

> **Retracted in place.** This document previously said T1 "wins as the reserved
> fraction grows, which is where a user-chosen catalogue goes." Wrong twice.
> T1 needs roughly half the catalogue to lack in-source data before it wins on
> mean accuracy, and never wins in range on int4. And the second clause was
> asserted, not measured — a user picking common plants would have a *low*
> reserved fraction, not a high one. Nothing here establishes which way a real
> user catalogue leans.

## Where this leaves the decision

The measurement is finished and the remaining question is not empirical.

- **Optimising catalogue-mean accuracy selects the single mixed head**, across
  the entire realistic range on the deployable encoder. It buys 5–7pp on average.
- **Requiring that no species be worse off than today selects T1**, at a cost of
  **0.5–1.6pp of mean accuracy**, maximal near `r = 0.2` and falling either side.
- The harm the single head does is **concentrated, predictable and identifiable
  in advance**: it falls on exactly the species with no in-source data, it is
  3–6pp, and it is worst when those species are fewest.

That is a judgement about concentrated harm against average gain, and it is not
a question more measurement answers. What can be said is that the harmed set is
knowable before shipping, so a third option exists: **ship the single mixed head
and report per-species which ones sit in the reserved position** — which is what
this project's own card discipline would demand of anyone else.

## The K sweep — the whole trade-off is a large-catalogue phenomenon

Everything above is measured at **K = 345 species**. The product runs at
K = 10–50: a user picks their own list and `build` fits a head on it. The damage
mechanism is competition in one argmax, so the number of competitors is the thing
most likely to change the answer, and it does.

`r = 0.10` throughout — one species in ten with no in-source data — 15 random
species subsets per K.

| K | **single head: Δ reserved** | sd over draws | Δ mixed | T1 data effect on reserved |
|---|---|---|---|---|
| 10 | **0.0000** | 0.0000 | +0.0083 | −0.0067 |
| 20 | −0.0093 | **0.0238** | +0.0138 | 0.0000 |
| 50 | −0.0149 | 0.0175 | +0.0242 | +0.0070 |
| 100 | −0.0275 | 0.0221 | +0.0450 | −0.0100 |
| 345 | −0.0310 | — | +0.0749 | −0.0054 |

**The damage is monotone in K and gone by K = 20.** At K = 10 it is exactly zero;
at K = 20 it is −0.9pp against a spread of 2.4pp across draws, so it is not
distinguishable from noise. Replicated on `bioclip2_cml4`: −0.0155 (sd 0.0331),
−0.0073 (sd 0.0168), −0.0091, −0.0265, −0.0251.

**And so is the gain.** The mix buys +0.8pp at K = 10 and +1.4pp at K = 20,
against +7.5pp at K = 345.

The mechanism is the one this project already documented from the other side: as
K falls, top-1 rises steeply toward ceiling (`EMBEDDED_FINDINGS.md`: 0.939 at
K = 50 to 0.989 at K = 10), which leaves less room for in-source data to help
*or* to hurt. Both effects are headroom, and narrowing spends it.

### What that means for each product

- **The 490-species app.** K = 345 is the right regime, the +7pp is real, and
  the whole reserved-species analysis above applies. Nothing here retracts it.
- **The tool, where a user picks 10–50 species.** The trade-off does not exist at
  that scale. The mix buys 1–2pp and costs about 1pp on species without in-source
  data, both inside draw-to-draw spread. **T1 is not worth building for this
  case** — at K = 20 it buys +0.0123 against the single head's +0.0138.

So for the tool the answer is the simple one: **mix if it is convenient, and tell
the user which of their species had no in-source data** — the same thing `plan`
already does for crowded genera, and for the same reason. Do not build the
two-head architecture to solve a problem that K = 20 dissolves.

### Carried to a second domain, where it does not hold

The K-dissolution was taken to `narrowcast-derm` — Fitzpatrick17k, DINOv2, skin
type as the source variable — with the prediction declared in advance that a
*harder* domain should show the effects persisting to smaller K, because the
mechanism is the accuracy ceiling rather than the label count.

| K | plants: damage | gain | top-1 | dermatology: damage | gain | top-1 |
|---|---|---|---|---|---|---|
| 10 | **0.0000** | +0.0083 | ~0.99 | **−0.1354** | +0.0893 | 0.7177 |
| 20 | **−0.0093** | +0.0138 | ~0.97 | **−0.1475** | +0.1293 | 0.6319 |

**Confirmed by about an order of magnitude.** Dermatology never approaches the
ceiling — top-1 is 0.80 even at K = 5 — and the effects never vanish.

So the conclusion above stands **where it was measured and nowhere else**. "Do
not build the two-head architecture, K = 20 dissolves the problem" is right for
this catalogue and this encoder, and would be wrong advice on that corpus, where
at K = 20 the damage is 14.8 points and a fix is worth having. Any guidance
narrowcast gives has to key on the *measured accuracy of the build*, never on the
label count.

One thing the cross-domain test did **not** establish: a quantitative rule. See
`K_FINDINGS.md` in that repo, and the correction below.

### Is it the accuracy, or is it the images?

The between-domain claim above was first made by comparing **plants at K = 345
against dermatology at K = 5**, on the grounds that both sit at baseline 0.80.
Matched accuracy, mismatched label count, images per class and class balance —
it could not separate "the corpora differ" from "everything differs."

Redone properly: hold **K = 20**, where both domains are measured, and vary only
difficulty — encoder (`bioclip2`, `mobileclip2_s0`), set shape (random species
versus congener-crowded blocks) and a training cap of 2–30 rows per species. 28
arms, class count and roughly the images-per-class regime held fixed throughout.

> **Capping alone could not do it, which is itself worth recording.** A random
> 20-species plant set is separated at **0.9345 on two training images per
> species**. The accuracy is carried by the encoder's representation, not by the
> head's training data, and starving the head cannot make plants hard. Relatedness
> can: congener-crowded blocks take `bioclip2` to 0.65 and `mobileclip2_s0` to
> 0.35.

Across those 28 arms, with everything structural fixed:

> **damage = −0.194 + 0.181 × baseline top-1**, `r = 0.78`, **R² = 0.61**

**So it is mostly the accuracy, and it is not the class distribution.** Holding K,
class count and training regime fixed and moving only how hard the problem is
reproduces most of the effect. The original comparison was confounded and the
confound was doing real work.

**A residual remains, and it is about half what was claimed.** Extrapolating the
plant curve to dermatology's baselines:

| baseline | dermatology | plant curve | residual | ratio |
|---|---|---|---|---|
| 0.799 | −0.1072 | −0.0490 | −0.058 | 2.2× |
| 0.718 | −0.1354 | −0.0637 | −0.072 | 2.1× |
| 0.632 | −0.1475 | −0.0792 | −0.068 | 1.9× |
| 0.553 | −0.1260 | −0.0936 | −0.032 | 1.3× |

Dermatology sits above the plant curve at every point — **~1.9× in the middle of
the range, against the 3.5× the confounded comparison implied.** The declared
prediction in addendum 6 was that the plant curve would sit below dermatology's
point, and it does.

**But treat the residual as suggestive, not established.** Per-arm standard
errors are 0.012–0.031 and dermatology's are ~0.024, so residuals of 0.032–0.072
are one to three standard errors. Four of four dermatology points falling above
the curve is consistent but weak on its own. The honest statement is that
**accuracy explains most of the between-domain gap and something else may explain
a factor of roughly two** — and that "something else" is where the makeup of the
images would live, if it lives anywhere.

**And audio agrees with dermatology.** Speech Commands, 35 words, wav2vec2, with
speaker population as the source variable — chosen because a speaker's voice
pervades every frame of a clip the way skin tone pervades every pixel of a
lesion, and unlike the framing convention that separates two photograph corpora.
Damage is **flat across the sweep and largest at the narrow end**: −0.214 at
K = 5, −0.186 at K = 10, −0.181 at K = 20. Plants are *zero* at K = 10. Against
the plant curve, audio sits at 2.1–2.6× and dermatology at 1.9–2.1× — two domains
sharing no modality, encoder, task or source variable, landing in the same place.
The magnitude is not evidence there: audio's source variable is constructed by
k-means, so its level is chosen rather than measured, and the pre-registration
said so before the run. The *shape* is the result. See `SOURCE_MIX_FINDINGS.md`
in `narrowcast-kws`.

**It is not the encoder.** The dermatology arm ran on one embedding set, so the
residual was confounded with encoder family. Re-embedding those same images with
`mobileclip2_s0` and `bioclip2` — the two the plant curve is fitted on — leaves
the damage at K = 20 essentially unchanged: −0.185 and −0.148 against DINOv2's
−0.148. Compared against the plant curve *fitted on the same encoder*, the
residual is 1.5–2.2×, the same as before, and all twelve dermatology points across
three encoders sit above their matched curve. So: ruled out in order — label-set
size, class distribution, baseline accuracy, encoder family. See `K_FINDINGS.md`.

> **This is the measurement that should have come first**, and the
> pre-registration says so. Four rounds of analysis were spent characterising a
> trade-off at a catalogue size the tool does not use. The finding survived every
> control that was run at K = 345, and the one thing that dissolved it was asking
> whether the label set was the one being shipped.

## What is still untested

**A weighted or routed combination** instead of a flat average. T1 uses 0.5/0.5
with nothing tuned. A weight fitted on the calibration split might recover part
of the mean-accuracy gap and would move the crossover left; by this project's
conventions the weight has to be declared before it is fitted.

## Reproduce

```
PYTHONPATH=. .venv/bin/python -m analysis.source_mix_middle --variant bioclip2
PYTHONPATH=. .venv/bin/python -m analysis.source_mix_middle --variant bioclip2_cml4
```
