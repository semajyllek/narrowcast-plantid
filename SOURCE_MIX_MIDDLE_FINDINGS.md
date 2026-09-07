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

## What would resolve it

Untested and cheap, in rough order of value:

1. **Per-class balancing.** The damage may be an artifact of the mixed classes
   simply having more rows. Weighting classes to equal effective count, or
   capping in-source rows per species, might keep most of the +7.7pp without the
   −2.8pp. This is the obvious first thing to try and it was not part of the
   pre-registered design.
2. **Sweep the reserved fraction.** 20% was declared, not derived. Whether the
   damage scales with how *few* species are reserved is unmeasured, and a
   catalogue where 95% have in-source data may behave differently from one where
   80% do.
3. **Two heads rather than one.** If the competition in a single argmax is the
   mechanism, a per-source head with a routed or averaged posterior would avoid
   it. Considerably more machinery, and it should not be built before (1) is
   tried.

## Reproduce

```
PYTHONPATH=. .venv/bin/python -m analysis.source_mix_middle --variant bioclip2
PYTHONPATH=. .venv/bin/python -m analysis.source_mix_middle --variant bioclip2_cml4
```
