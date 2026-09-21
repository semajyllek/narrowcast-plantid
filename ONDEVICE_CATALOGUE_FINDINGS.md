# The user can pick their own species on the phone — and the catalogue is not what limits it

Scoped, then built and measured on 976 real Oregon species. The question: can the
label set be chosen *by the user at runtime* — twenty plants off a field-guide
page — rather than fixed when the bundle is exported?

**Yes, technically, and well.** What the measurement also shows is that this does
not solve the problem it looked like it would solve.

## Why this shape, not a bigger model

The instinct on seeing a 24-species bundle is to make the model know more of
Oregon. That is the wrong direction and this project already measured why: a
crowded label set buys coverage with coarse answers that narrow nothing. Letting
the user name their twenty gets small K where the accuracy is.

Confirmed here, on a real 976-species bank, over 20 independent random selections:

| user picks | in-list top-1 |
|---|---|
| 10 species | **0.982** |
| 20 species | **0.966** |
| 40 species | 0.951 |

Accuracy falls as K grows, exactly as the crowding finding predicts.

## The head fits on the phone, because it is fitted on embeddings anyway

`build.fit_head` never sees an image — it takes `ds.X_train`, which is already
embeddings. Shipping stored embeddings and fitting on-device is therefore not an
approximation of the offline pipeline; it is **the same computation**.

The bank built here: **961 species, 8 exemplars each, 9.4 MB compressed**, against
an 87 MB encoder. An exemplar is one *plant* — the mean of its photographs — not
one photograph, because several shots of one individual are not independent and
every split in this project keys on that distinction.

Nearest-centroid is within a point of logistic regression at every exemplar count
(0.921 vs 0.920 at 2; 0.967 vs 0.973 at 8), so the on-device fit is a mean and a
dot product rather than a port of L-BFGS.

**The bank supplies its own negatives.** The species the user did *not* pick are
the reject class, and they are better negatives than a background pool because
they are the actual deployment distribution.

## What it does not fix, and this is the finding

Twenty species out of a regional flora of thousands is still twenty. Over 8
selections, scored against ~956 unchosen species acting as out-of-list:

| assumed out-of-list rate | answers | right when it does | names *your* species |
|---|---|---|---|
| 50% | 41.2% | 94.3% | **77.9%** |
| 80% | 12.4% | 93.3% | 57.4% |
| 95% | 3.4% | 82.4% | 52.8% |

Closed-set top-1 is **0.972** throughout: the model is excellent at telling the
chosen twenty apart. What moves is how often it is willing to commit, and that is
governed by how much of what you photograph is out-of-list.

**So the product is only as good as the user's list is well-chosen** — and that is
the argument for this design rather than against it. A curator cannot know which
twenty plants a particular person will actually walk past; the person can. Pick
twenty you genuinely encounter and the effective out-of-list rate is near 50%,
where it names 78% of your plants at 94% precision. Pick twenty at random from a
field guide and it will decline nearly everything, correctly.

The near-OOD gate does not rescue this: fitted at `p_ood` 0.5 and 0.8 it moves
coverage by 0.5 points and 0.0. Consistent with `NEAR_OOD_FINDINGS.md`'s null.

## Retracted in place: "pick twenty" is the wrong default

*Added after measuring the thing the section above assumed.* Everything above is
correct and the conclusion drawn from it was wrong, because it optimised in-list
accuracy — a number nobody experiences — instead of **correct identifications per
plant photographed**, which is the product.

Accuracy does fall with K, gracefully:

| K | species top-1 |
|---|---|
| 10 | 0.971 |
| 20 | 0.973 |
| 40 | 0.931 |
| 160 | 0.881 |
| 416 | 0.821 |
| **961 (whole bank, measured)** | **0.773** |

But coverage rises far faster than accuracy falls, because plant photography is
long-tailed:

| K | share of what you photograph | top-1 | **correct per 100 photographed** |
|---|---|---|---|
| 20 | 11.0% | 0.973 | **10.7** |
| 80 | 27.6% | 0.917 | 25.3 |
| 320 | 53.5% | 0.837 | 44.8 |
| **961** | **78.0%** | **0.773** | **60.3** |

**Shipping the whole bank is roughly six times more useful than the user picking
twenty.** The crowding finding does not forbid this: it is about *composition* —
a set packed with congeners against a varied one at the same K — and a regional
flora at K=961 is the varied case. What was measured here is the ordinary K
effect, and it loses the argument to coverage.

### And the coarse rank comes alive

At K=20 with roughly one species per family, `group_share` was ~0: there was
nothing to retreat to, so the cascade's middle answer was dead weight. At K=961,
**172 of 489 genera hold more than one species**, and:

| answer | accuracy over 6,059 held-out photographs |
|---|---|
| species, top-1 | 0.773 |
| **genus** | **0.846** |
| **species in top 5** | **0.936** |

So the honest product is not one name. It is a name when the model can defend
one, a genus when it cannot, and a short list when it can only narrow — and that
shortlist is right 94% of the time, which is exactly what a person with a field
guide can finish.

### Where user-chosen lists still belong

Not as the default, but as a mode. A forager who wants "is this the edible one or
the lethal one" is asking a different question from "what is this", and for that
question the narrow model is right: it was 0.972 closed-set top-1 and declined
almost everything else. That is the `--never-answer` and `forage` profile
territory this project already built. **Default to the whole bank; offer the
narrow list as a deliberate choice.**

## The whole-bank model, built

901 species over 467 genera, exported and parity-checked. Two things learned in
building it that the scoping section had wrong.

**Fit on photographs, not on averaged exemplars.** The eight-exemplars-per-species
constraint exists for *on-device* fitting; a shipped whole-bank head is fitted
once and can use everything. Training on all photographs of the same eight plants
— 15,713 rows against 6,593 — takes top-1 from 0.773 to **0.812** and, more
importantly, sharpens the posteriors: median max probability 0.048 → 0.759 at
`C = 100`. Diffuse posteriors were making the cascade decline answers it actually
had.

**`identify` is the right declared profile here.** `standard` costs a wrong answer
−4 and names 47.1%; `identify` costs −2 and names 58.3% at 93.5% precision. A
misnamed garden plant costs curiosity, not health — which is exactly the reason
that profile was written down, and the first time anything has selected it.

### What it does, per 100 plants photographed in Oregon

| outcome | per 100 |
|---|---|
| named to the right species | **37** |
| named to the wrong species | 9 |
| answered at genus only | 6 |
| not in the bank at all | 22 |
| declined | ~26 |

Against **11 correct per 100** for a user-chosen twenty. And if the app showed a
five-name shortlist rather than one answer, **74 of 100 shortlists would contain
the right species** — which is the number that should probably drive the
interface, because a person holding a field guide can finish a shortlist and
cannot finish a decline.

### Sizes

`bundle.json` grows from 106 KB to 3.8 MB — 902 classes × 768 dimensions is most
of it — against an unchanged 87 MB encoder. Parity against
`narrowcast.predict.Bundle` holds over 3,000 rows with zero disagreements.

## Honest limits

- **60% of the bank has the full 8 exemplars**; 15 species have only 2. The app
  should show how well-supported a species is before the user selects it, since
  a 2-exemplar class is measurably weaker (0.907 against 0.960).
- **976 species is 79% of what people photograph in Oregon, not 79% of its
  flora.** The tail is real and unfetchable at this quality — 33 of 1,009
  requested species had no open-licence Oregon photographs at all.
- **Everything is still GBIF photographs.** A phone camera in one's own hands
  remains the untested axis.
- The K=10/20/40 accuracies come from species with ≥10 plants (416 of 961), so
  they describe the well-supported part of the bank.
