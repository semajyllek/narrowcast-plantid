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
