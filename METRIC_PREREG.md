# Pre-registration — can a learned metric on frozen features do what a bigger encoder cannot?

> ## Outcome: bar not met, and a real effect outside the window
>
> A shrinkage-LDA projection more than doubles the crowded `label_share` at **32**
> examples per label — 0.0654 → 0.1392, paired Δ **+0.0739 [+0.0269, +0.1203]** —
> with PCA null at matched width, so the gain is **supervision**, not compression.
>
> **The declared bar was >5pp at ≤ 8 shots. The best at ≤ 8 shots is +2.84pp**, on
> an interval whose lower bound is +0.0002 across eight comparisons. The bar is not
> met and the tool does not change. The 32-shot effect is a hypothesis found in the
> data and needs its own pre-registration, not a window moved after the fact.
>
> Predictions: **1 right**, **2 right**, **3 wrong** (LDA does beat `raw` below 8
> shots, by too little to matter). Full result in `METRIC_FINDINGS.md`.

Short, because the question is narrow and the arms are cheap.

## The gap this attacks

`TINY_FINDINGS.md` §2 leaves exactly one cell where nothing has worked. On a
**crowded** label set at **few shots**, encoder quality is worth *nothing*: every
encoder from 5.7 MB to 152 MB lands between 0.004 and 0.013 `label_share` at one
example per label, and the ordering does not even hold. A 27× range of model size
produces no signal. Distillation is now closed (`TINY_PREREG.md`), inference-side
levers are closed (`SMALL_FRONTIER_FINDINGS.md`), and encoder selection —
the lever those left open — is the one that demonstrably fails here.

Everything this project has measured holds the encoder frozen and fits **one
multinomial logistic head** on top. Full fine-tuning sits at the other extreme and
is expensive. **The space between them has never been touched.**

## What SetFit actually does, and the narrowcast-shaped version of it

SetFit's mechanism is not its head; it is that it fine-tunes the *body*
contrastively on pairs drawn from the user's own few examples, then fits a
classifier on the adapted embedding. The adaptation uses no extra corpus — only
the labelled examples already in hand.

Fine-tuning the body breaks this project's architecture and its cost model. The
equivalent that does not is to **learn a projection on top of the frozen
embedding** from the user's own rows: same frozen encoder, same ~40 KB of
personalisation, seconds of CPU. The encoder stays someone else's; what adapts is
the metric.

There is a specific reason to expect it to be the right shape. `TINY_FINDINGS.md`
§1 measures that a K-way task occupies far fewer dimensions than the embedding
carries — PCA to `d < K` costs 1.9–7.7pp on congeners — and **PCA is
unsupervised.** It finds the directions of greatest variance, not the directions
that separate the labels. A supervised projection to the same width is the
principled version of the same compression, and the gap between them is a
measurement nobody here has taken.

## Arms

Frozen embeddings, cached; every arm ends in the same logistic head so only the
representation varies. Both label-set arms throughout, never averaged.

| arm | what it is | supervised |
|---|---|---|
| `raw` | the current tool: logistic head on the L2-normalised embedding | — |
| `pca{m}` | unsupervised projection to *m* dims, then the head | no |
| `lda{m}` | shrinkage LDA projection to *m* dims, then the head | **yes** |
| `nca{m}` | Neighbourhood Components Analysis to *m* dims, then the head | **yes** |

`pca` is the control that makes the comparison mean something: a supervised
projection beating `raw` proves little if an unsupervised one does too.

**The projection is fitted on the training rows only**, the same rows the head
sees, and never on anything scored. Shots are swept as in
`analysis/fewshot_curve.py`, evaluated cross-source on held-out iNaturalist
observations through `build.fit_and_measure` unchanged.

## Predictions, declared now

1. **No gain on separated sets at 8 shots or more.** Those are already saturated —
   BioCLIP-2 reaches 100% of its unlimited-data `label_share` at 8. If an arm
   shows a gain there, suspect the baseline before believing it.
2. **`lda` and `nca` beat `pca` at matched width on the crowded arm**, because the
   task subspace is a discriminative one and PCA cannot see labels. This is the
   prediction I am most confident in and it is also the least interesting: it
   would confirm the projection is doing something, not that it is useful.
3. **Neither beats `raw` on the crowded arm below 8 shots.** With 8 examples per
   label a K=20 scatter estimate rests on 160 rows in 512–768 dimensions; the
   projection has more parameters than the head it feeds. Shrinkage is the
   mitigation, and I do not expect it to be enough.

**So the honest expectation is that this fails**, and the reason it is still worth
running is that prediction 3 is the one measurement that would tell a user
something actionable: *the crowded few-shot cell is not fixable by a better metric
either.* That closes the last cheap lever rather than leaving it as a plausible
untried idea.

## What would change the tool

If any supervised arm beats `raw` by **more than 5pp of `label_share` on the
crowded arm at ≤ 8 shots**, with the interval excluding zero over 12 draws, the
projection belongs in `build` as an option — it costs seconds of CPU and no bytes
worth counting. Below that bar it is a finding and not a feature.

## What would make this uninformative

Fitting the projection on rows that are later scored. The split is the same one
`fewshot_curve` already uses: the projection, the head and the thresholds all see
only the training and calibration halves.

And the usual: one domain, one taxonomy, two label-set shapes. A metric that helps
on congeneric plants has said nothing about acoustically confusable keywords until
it is run there — `narrowcast-kws/fewshot_curve.py` is the instrument and it is
already built.
