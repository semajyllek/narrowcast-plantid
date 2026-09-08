# Domain adaptation moves the 17.9 MB point, and it is the first thing that has

Pre-registered in `ADAPT_PREREG.md`. MobileCLIP2-S2's image tower fine-tuned on
Pl@ntNet-300K, refrozen, and scored through the unchanged evaluation path:
3,435 held-out iNaturalist observations, 490-way, observation-level.

| encoder | MB | ms/img | species | genus | coverage @20% |
|---|---|---|---|---|---|
| `mobileclip2_s2` stock | 17.9 | 5.2 | 0.6236 | 0.8122 | 0.426 |
| **`mobileclip2_s2_ft`** | **17.9** | **5.2** | **0.7258** [0.6966, 0.7516] | **0.8934** | **0.510** |
| `plantclef24` | 43.3 | 38.6 | 0.7671 | 0.9258 | 0.616 |
| `bioclip2_cml4` | 160.0 | 20.4 | 0.8370 | 0.9735 | 0.692 |

Paired over the same 3,435 observations, species-clustered bootstrap:

| contrast | species | genus |
|---|---|---|
| adapted − stock | **+0.1022 [+0.0830, +0.1214]** | **+0.0812 [+0.0665, +0.0972]** |
| adapted − `plantclef24` | −0.0413 [−0.0630, −0.0192] | −0.0323 [−0.0441, −0.0211] |

**The null is rejected decisively.** Adaptation closes **71%** of the 14.4pp gap
to a 2.4× larger encoder, at **no cost in bytes and none in latency** — it is the
same architecture, so still 5.2 ms/image against `plantclef24`'s 38.6.

After four inference-side levers measured and found null
(`SMALL_FRONTIER_FINDINGS.md`), this is the first thing that has moved the small
end of the frontier at all.

## Against the predictions

1. **"0.65–0.72"** — landed at **0.7258**, just above. Under-predicted.
2. **"Genus moves more than species"** — **wrong**. Species moved +0.1022 and
   genus +0.0812. The reasoning was that adaptation teaches plant-shaped structure
   before fine distinctions; the opposite happened, and in hindsight it should
   have been obvious that a 1,081-way species objective optimises species
   separation directly and genus only as a by-product.
3. **"0.74+ changes the size decision"** — **not met on the point estimate**, and
   the interval [0.6966, 0.7516] contains 0.74. Undecided by the threshold I
   declared, which is the honest reading rather than rounding up.

## What this is, precisely

The tower was fine-tuned with a 1,081-way classification objective on Pl@ntNet
photographs and the classifier discarded. Evaluation is iNaturalist — different
photographs, different source — so the accuracy is not measuring memorisation of
the evaluation images.

**But it is task-adapted, not merely domain-adapted.** The 1,081 species include
all 530 catalogue species, so the encoder was explicitly trained to separate the
label set it is then scored on. That is legitimate for the 490-class app and it
is **not** evidence of a general small plant encoder.

### The generalisation claim is not merely unsupported. It is refuted.

The notebook's held-out-species probe reported **stock 0.5278 → adapted 0.7013,
+0.1735**. That number is worthless: those 551 species were *in* the fine-tune and
the probe set included images the tower trained on, so it passes by construction.

A valid probe was run instead, and needed no retrain — **the tower has never seen
a species outside Pl@ntNet's 1,081**, so anything outside that list is unseen.
90 such species, fetched fresh from iNaturalist (a source the adaptation never
touched), 2,233 observations, 3,397 photographs, split by observation so no plant
straddles train and test. Identical images and identical split for both towers;
only the encoder varies.

| tower | 90-way probe accuracy |
|---|---|
| `mobileclip2_s2` stock | **0.8258** |
| `mobileclip2_s2_ft` adapted | **0.7905** |
| **paired difference** | **−0.0353 [−0.0601, −0.0125]** |

**The adapted tower is worse on plants it has not seen, and the interval excludes
zero.** It is broad rather than a few outliers: 41 of 90 species get worse, 22
better, 27 unchanged; macro over species −0.0368. Losses reach −0.36 on *Allium
ursinum* and *Nothofagus cunninghamii*.

So the invalid probe was not just uninformative — **it was inverted**. It reported
+0.17 where the truth is −0.04. Had the design not been questioned, this would
have shipped as a general small plant encoder.

**What the adaptation actually did** is specialise to Pl@ntNet's 1,081-species
label set: +0.1022 inside it, −0.0353 outside it. That is textbook feature
specialisation, and it is invisible to every metric this project reports except
the one built to catch it.

## What it changes

**For the 490-class app**, the size decision has a new point on it: 17.9 MB now
buys 0.726 species where it bought 0.624, and the 43 MB option's advantage is
4.1pp rather than 14.4pp — at 7× the latency and 2.4× the bytes. `plantclef24`
was the middle ground; adapted S2 undercuts it on every axis except accuracy.

### ~~For the tool, do not ship it.~~ Retracted — measured at the wrong K, and half the job was never tested

The paragraph that stood here said a user with species outside Pl@ntNet-300K
"would be handed a model 3.5pp below stock". **Both halves of that were wrong.**

**Wrong scale.** The −0.035 is a K=90 number. narrowcast builds lists of 5–50.
Swept, 12 random draws per point:

| K | stock | adapted | delta | sd across draws |
|---|---|---|---|---|
| 5 | 0.9518 | 0.9508 | **−0.0010** | 0.0257 |
| 10 | 0.9391 | 0.9216 | −0.0175 | 0.0183 |
| 20 | 0.9068 | 0.8911 | −0.0157 | 0.0229 |
| 50 | 0.8516 | 0.8239 | −0.0277 | 0.0173 |
| 90 | 0.7994 | 0.7698 | −0.0296 | — |

At the sizes the tool actually builds, the penalty is 0–1.8pp and sits inside
draw-to-draw spread. It only reaches 3pp at label-set sizes nobody picks.

**And discrimination is only half of what narrowcast asks an encoder for.** The
other half is rejection — saying *not on my list* — and it was never measured.
AUROC of the max posterior separating held-out in-list observations from 7,606
out-of-list photographs:

| K | stock | adapted | delta |
|---|---|---|---|
| 5 | 0.8990 | 0.9119 | **+0.0129** |
| 10 | 0.8784 | 0.8937 | **+0.0153** |
| 20 | 0.8643 | 0.8736 | +0.0092 |
| 50 | 0.8239 | 0.8273 | +0.0034 |

**Adaptation makes rejection better**, by about as much as it makes
discrimination worse. At K=10 that is −0.018 on telling your species apart and
+0.015 on knowing when it is none of them.

**Corrected verdict: at product scale the adapted tower is roughly neutral for
the tool** — it trades a little discrimination for a little rejection — rather
than the clear negative recorded here an hour earlier. It is not a reason to ship
it either; a wash is a wash, and the +0.1022 that justifies it exists only inside
the training label set.

*(The rejection figure uses max posterior over the chosen K, without the
`__OTHER__` class narrowcast fits on background negatives. It is a proxy for the
cascade's decline decision, not the decision itself.)*

The lesson that survives intact: **adapting an encoder to a corpus buys accuracy
inside that corpus's label set and spends it outside.** +0.1022 in, −0.001 to
−0.030 out depending on how many labels you ask about. If narrowcast ever offers
domain-adapted encoders, each needs a published outside-the-set probe beside it —
**and that probe has to be swept over K**, because a single large-K number points
the wrong way.

> **Third time today.** A result measured at large K failed to survive to product
> scale — after `SOURCE_MIX_MIDDLE_FINDINGS.md` recorded exactly that lesson and
> called it "the measurement that should have come first". Sweeping K is not an
> optional robustness check in this project; it is the first thing to do, and I
> did not do it here until it was challenged.

## Reproduce

`notebooks/adapt_s2_colab.ipynb` on an A100 — ~85 min of training, val top-1
0.0003 → 0.8263 over four epochs. Then locally:

```
cp mobileclip2_s2_plantnet.pt data/processed/adapted/
PYTHONPATH=. .venv-mps/bin/python -c "
from plantid.features import embed_catalog, embed_inat, embed_background
for f in (embed_catalog.main, embed_inat.main, embed_background.main): f('mobileclip2_s2_ft')"
PYTHONPATH=. .venv/bin/python -m plantid.eval.rejection --variant mobileclip2_s2_ft
```
