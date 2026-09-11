# Pre-registration — reopening distillation, at the K the tool actually builds

Distillation appears twice on `CLAUDE.md`'s **"Closed — do not redo"** list, and
pruning once. Reopening it needs a reason better than wanting a different answer,
and this document exists to fix the decision rule **before** any of it is run.

## What was closed, precisely

| attempt | objective | student | K it was scored at | result |
|---|---|---|---|---|
| A100 run, 48k transfer images | cosine to teacher **embedding** | general ViT-B | 490 | recovered ~10% of the gap |
| `analysis/prune_distill.py` | cosine to teacher **embedding** | truncated teacher | 20 Oregon spp. | 26–49% recovered, every point dominated |

Both distilled a **general representation**. Both were judged on whether the
student could stand in for the teacher's embedding across a wide label space.
`PRUNE_FINDINGS.md` drew the right conclusion from them — *for large compression
ratios, selecting a natively-small encoder dominates compressing a large one* —
and that conclusion is not under challenge here.

## What has never been tried

**Distilling the fitted task, not the encoder.** narrowcast knows K at build
time. The teacher is not an encoder; it is `frozen encoder + fitted head +
calibrated cascade`, and its output is K+1 posteriors over labels the user named.
A student reproducing *that* does not have to carry a general representation of
plants. It has to separate twelve things and say *none of these*.

No experiment in either repo has that objective, and the difference is not
cosmetic. `PRUNE_FINDINGS.md` closes with the observation that **cosine to a
teacher does not predict downstream accuracy** — "third strike for the same
rule". Every closed attempt optimised the quantity that finding says is the wrong
one.

## Why the K matters, and why that argument is not special pleading

This repo's most-repeated self-criticism is that results measured at large K fail
at product scale. `ADAPT_FINDINGS.md` records it as **"Third time today"**, after
`SOURCE_MIX_MIDDLE_FINDINGS.md` called the same thing "the measurement that
should have come first". In ADAPT the sign of the effect was different at K=90
and at K=5.

`EMBEDDED_FINDINGS.md` already shows the relevant gradient: MobileCLIP2-S0's
deficit against BioCLIP-2 is **7.2pp at K=50, 3.4pp at K=10**, on random draws.
The penalty for small capacity shrinks as the task narrows. Both closed
distillation attempts sit at the far end of that curve.

**But the same table is the reason to expect this to fail.** On congener draws,
the gradient is nearly flat — S0 is 6.7pp behind at K=10 and 8.8pp at K=50, and
EMBEDDED states congener discrimination is "nearly independent of K". A tiny
model gets no discount for narrowness on exactly the label sets users pick
deliberately.

## Gate — this does not run unless Phase 1 clears it

Two training-free measurements (`analysis/capacity_floor.py`,
`analysis/fewshot_curve.py`) run first. Distillation is attempted **only if
both** hold:

- **G1.** PCA to d ≤ 32 costs **< 3pp** of closed-set top-1 at K ∈ {10, 20} on
  the **HARD** arm, for at least one encoder. If a narrow task still needs 256
  dimensions of a frozen embedding, the student is being asked to carry a general
  representation, and `PRUNE_FINDINGS.md` already answered that.

  > **This criterion as written is faulty, and `TINY_FINDINGS.md` records why.**
  > It omits `d < K`. A K-way linear head's decision lives in the span of its K
  > coefficient vectors, so "32 dimensions holds a 20-label task" sits above a
  > bound that holds by construction and is close to arithmetic. Re-read on the
  > `d < K` cells the gate is a **pass on one cell** (bioclip2, K=50, d=32,
  > −1.91pp) and a **near-miss on another** (K=20, d=16, −3.23pp). Read the
  > findings doc's version, not this one.
- **G2.** The random-projection control at the same d is **materially worse than
  PCA**. If random dimensions do as well, the low-dimensional result is about
  head capacity rather than a task subspace, and there is nothing for a student
  to find.

If either fails, distillation stays closed and this document records why.

## The experiment, if the gate clears

Teacher: the best encoder within budget plus its fitted head and thresholds, at
K=14, **two label sets — one crowded, one separated** (the README's own contrast).

> **Amended before running: K=12 → K=14.** No single genus has 12 species with
> enough images, but **8 *Sedum* + 6 *Trifolium*** does — and that is precisely
> the crowded set narrowcast's README publishes (coverage 0.806, label-level
> 0.476, against 0.618 / 0.761 for 14 distinct genera). Matching it means the
> teacher's numbers can be checked against a figure this project already stands
> behind, instead of resting on an arbitrary draw. The separated arm is 14
> distinct genera, as there. Recorded rather than silently changed.
Student: a natively small CNN trained from pixels, target = teacher's K+1
posteriors on the 109k cached images, then INT8. Scored through
`build.fit_and_measure` unchanged.

## Predictions, declared now

1. **The separated set survives and the crowded set does not.** Student
   `label_share` within 5pp of teacher on separated; **more than 15pp** below on
   crowded. Reasoning: crowded sets need fine discrimination between congeners,
   which is precisely the capacity a tiny student lacks, and EMBEDDED shows that
   deficit does not shrink with K.

   **Anchored to the measured teacher, so "within 5pp" cannot drift to whatever
   it scores on the day.** `plantclef24` at K=14, fitted on Pl@ntNet and
   **evaluated cross-source on held-out iNaturalist observations** — the direction
   every headline in this repo uses:

   | arm | `label_share` | coverage | precision | top-1 | headroom |
   |---|---|---|---|---|---|
   | crowded (8 *Sedum* + 6 *Trifolium*) | **0.554** | 0.710 | 0.955 | 0.897 | 0.119 |
   | separated (14 genera, 4 draws) | **0.899** | 0.736 | 0.974 | 0.971 | 0.000 |

   The crowded figure sits beside narrowcast's published **0.476** for the same
   fourteen species on a different encoder — close enough to say the harness is
   sound, far enough to be a real measurement rather than a re-print.

   > **The same-source anchors are withdrawn.** An earlier version recorded
   > crowded 0.500 / separated ~0.95 from the Pl@ntNet test split. That scored
   > `plantclef24` on the corpus it was fine-tuned on, putting the separated arm
   > at top-1 0.978–1.000 with headroom 0.000 — a ceiling a student cannot lose
   > against. Cross-source puts it at 0.951–0.984 and leaves room to measure.

   So the thresholds, fixed now: the student **passes** the separated arm at
   `label_share` ≥ **0.849** (within 5pp of 0.899) and **fails** the crowded arm
   below **0.404** (more than 15pp under 0.554).

   `TINY_FINDINGS.md` sharpens why this is the right contrast: `label_share` on a
   crowded set is the quantity that collapses under *any* reduction in effective
   capacity — fewer shots do it, and a student is a capacity reduction by
   construction.
2. **Sub-1 MB is not reached at useful accuracy.** A student under 1 MB lands
   **more than 10pp** below teacher `label_share` on both arms. If it does not,
   that is the finding.
3. **The cascade absorbs the student's miscalibration.** `fit_thresholds`
   grid-searches quantiles of the actual scores, so a temperature shift should
   move the thresholds and not the outcome. Nesting
   (`max_c P(c) ≤ max_g Σ P(c)`) is structural and must hold by construction —
   if it does not, there is a bug, not a finding.

## What re-closes it

- G1 or G2 fails.
- Student `label_share` is more than 15pp below teacher on **both** arms at every
  size tried. Then task-conditional distillation is beaten by the same rule as
  the general kind, the "wrong K, wrong objective" argument is exhausted, and
  distillation closes for the third and last time.

  > **Amended after the pilot: this condition was met six times over and was
  > *not* applied, because the test was handicapped.** The students ran at
  > 128–224px against a teacher at **518×518**. Resolution is not a nuisance
  > variable here: raising it from 128 to 224 took the crowded arm from
  > `label_share` **0.000 to 0.174** and top-1 from 0.388 to 0.492, on an
  > otherwise identical model. A condition that fires because the student was
  > shown a smaller picture does not test capacity.
  >
  > **The amended condition, which is checkable rather than a licence to keep
  > going:** distillation re-closes when a student more than 15pp below teacher
  > on both arms is either (a) trained at the teacher's own input resolution, or
  > (b) shown to be resolution-saturated — two successive resolution steps buying
  > under 2pp of **fine-rank accuracy**. Until one of those holds, the status is
  > **not demonstrated**, which is different from refuted and must be written
  > that way.
  >
  > **Saturation is tested on top-1, not on `label_share`, and that correction
  > matters.** `label_share` is gated: `cascade.UTILITY` names a label only at
  > p > 0.800, so below roughly 0.45 top-1 on a crowded set it is pinned at
  > exactly 0.000 and *cannot* register an improvement. Two resolution steps
  > buying "under 2pp of `label_share`" would therefore be satisfied trivially by
  > any model under the cliff, which is the opposite of evidence. Measured on
  > `label_share` the sub-1 MB crowded student looks saturated (0.000 → 0.000);
  > measured on top-1 it plainly is not (**0.277 → 0.355, +7.9pp per doubling**).
  >
  > **So as of this pilot the condition is not met and resolution is still live.**
  > Reaching the teacher's 518px is ~2 h/arm on MPS at the observed 26 s/epoch —
  > that is what the borrowed A100 is for, and it is the one remaining question
  > before this can be closed either way.
- ~~The student needs transfer data the tool cannot assume. narrowcast's users
  bring 30 images per label, not 109k. **A method that needs a corpus the tool
  forbids itself from fetching is a research result, not a feature**, and must be
  reported as one even if it works.~~

  **Retracted — this confused "more data" with "fetched data".** The transfer set
  carries **no labels**. Every target is `clf.predict_proba` over the teacher's
  embedding, so what a `narrowcast distil` would ask for is `--unlabeled DIR`:
  the same contract as `--images`, minus the labels, and no more of a fetch than
  any other source. The pilot is already that workflow — those 29,499 Pl@ntNet
  images are unlabelled as far as the student is concerned.

  The asymmetry runs the right way, too. What is scarce in a real deployment is
  *labels*, not pixels: a camera trap has 100k unlabelled frames and 30 labelled
  ones. So the open question is not permission but **quantity** — how much
  unlabelled data a student needs — and that is measurable rather than arguable.

  **The re-close condition is therefore rewritten as a number.** If a student
  needs more unlabelled in-domain images than a plausible user has, it stays a
  research result. `--steps` fixes the gradient-update budget so `--limit` can be
  swept without varying compute; the curve of `label_share` against transfer-set
  size decides it. A student needing ~30k is a paper; one needing ~2k is a
  feature. **This is declared before that sweep is run.**

## What would make this uninformative

The 109k transfer images are Pl@ntNet and iNaturalist photographs of the same
species the teacher was fitted on. A student trained on them is being handed the
deployment distribution, which no user of narrowcast will have. `ADAPT_FINDINGS.md`
is the cautionary case: an invalid probe reported **+0.17 where the truth was
−0.04**, because the evaluation set sat inside the training set. Any student
result must be scored on rows whose *observations* never appear in its transfer
set, asserted rather than assumed.

Sweeping K is not an optional robustness check here. It is the first thing to do.
