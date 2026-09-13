# A supervised projection doubles the crowded label share — outside the window I declared

Pre-registered in `METRIC_PREREG.md`. `analysis/metric_head.py`, K=20,
`mobileclip2_s0`, 12 draws, cross-source evaluation on held-out iNaturalist
observations through `build.fit_and_measure` unchanged. Every arm ends in the same
logistic head, so only the representation varies.

## The result, and the bar it does not clear

**Crowded arm, `label_share` against `raw`, paired bootstrap over the 12 draws:**

| shots | arm | mean Δ | 95% CI | excludes 0 |
|---|---|---|---|---|
| 32 | **`lda16`** | **+0.0739** | [+0.0269, +0.1203] | **yes** |
| 32 | **`lda64`** | **+0.0697** | [+0.0238, +0.1176] | **yes** |
| 32 | `pca16` | +0.0041 | [−0.0448, +0.0548] | no |
| 32 | `pca64` | +0.0144 | [−0.0231, +0.0732] | no |
| 8 | `lda16` | +0.0284 | [+0.0002, +0.0592] | marginal |
| 8 | `lda64` | +0.0196 | [−0.0131, +0.0557] | no |
| 8 | `pca16` | −0.0059 | [−0.0365, +0.0185] | no |
| 8 | `pca64` | −0.0025 | [−0.0383, +0.0328] | no |

At 32 examples per label a shrinkage-LDA projection takes the crowded label share
from **0.0654 to 0.1392** — more than double — and the effect holds at both widths
with intervals well clear of zero.

**The pre-registered bar is not met.** It was declared as *more than 5pp on the
crowded arm at ≤ 8 shots*. The best at ≤ 8 shots is **+2.84pp**, and that arm's
interval has a lower bound of +0.0002 — touching zero — across eight comparisons,
which is precisely the shape a multiple-comparisons artefact takes. **Nothing here
changes the tool**, and the +7.4pp is a hypothesis discovered in the data rather
than a confirmed prediction.

Recording the distinction because the temptation to move a window from "≤ 8" to
"≤ 32" after seeing which side the effect landed on is exactly what declaring it
in advance is for.

## Supervision is the mechanism, not dimensionality reduction

`pca` and `lda` share width, head, splits and draws. PCA is null in every cell;
LDA is not. Since the only difference is that one sees labels and the other does
not, **the gain is supervision** — a discriminative subspace rather than a
high-variance one.

That is the missing half of `TINY_FINDINGS.md` §1, which established that a K-way
task occupies far fewer dimensions than the embedding carries but measured it with
an *unsupervised* projection. The supervised version of the same compression is
better, and by enough to matter — once there is enough data to estimate it.

## The shape of the curve is the actual finding

| shots | `raw` | `lda16` | Δ |
|---|---|---|---|
| 1 | 0.0050 | *cannot be fitted* | — |
| 4 | 0.0037 | 0.0070 | +0.0033 |
| 8 | 0.0301 | 0.0585 | +0.0284 |
| 32 | 0.0654 | 0.1392 | **+0.0739** |
| all | 0.1494 | 0.1424 | −0.0070 |

The benefit **rises with data and then vanishes**. At one example per label a
supervised projection does not exist: LDA needs strictly more samples than classes
to estimate a within-class scatter, and 20 examples for 20 classes is an equality.
At full data `raw` has caught up and the projection is a wash.

So a learned metric helps in a **band** — enough data to estimate a discriminative
subspace, not yet enough for the head to find it unaided. On this arm that band is
roughly 8–64 rows per label. It is not a few-shot rescue, which is what the
pre-registration went looking for.

**This sharpens the mechanism rather than contradicting it.** `TINY_FINDINGS.md`
§2 shows the crowded few-shot cell resists encoder quality across a 27× size
range. It resists a learned metric too, for the same underlying reason: at one to
four examples per label there is not enough signal to estimate *anything* —
neither a better boundary nor a better basis. The failure is informational, not
architectural, and no amount of model changes it.

## Predictions, scored

1. **No gain on separated sets at ≥ 8 shots — right.** `lda64` is +0.008 at full
   data and −0.0005 at 8 shots. A wash, as expected for a regime already
   saturated.
2. **Supervised beats unsupervised at matched width — right**, on the crowded arm,
   and by the whole of the effect.
3. **Neither beats `raw` on the crowded arm below 8 shots — wrong.** LDA does beat
   it at 4 and 8 shots. The prediction was right about the *magnitude* mattering
   and wrong about the *sign*, which is a weaker kind of right than it looks.

## Two corrections made while running this

**A normalisation bug nearly produced a false null.** The catalogue loader
L2-normalises the embedding before the head, but nothing renormalised *after* a
projection — and PCA, LDA's eigen solver and NCA all emit different scales. With
`fit_head` at a fixed `C = 10`, an arm that happened to shrink the norm was handed
far stronger regularisation than the baseline. Fixing it moved `lda16` **+11pp**
on the separated arm. Before the fix every projection lost heavily and the
experiment looked like a clean null; that null was substantially an artefact of
comparing scales rather than representations.

**Two conclusions were drawn from a 3-draw smoke test and both were wrong.** The
first said supervised projections showed no advantage over PCA; at 12 draws they
show the entire effect. Draw variance in this harness is large — `raw` moved 8pp
on the separated arm between 3 and 12 draws, more than most of the between-arm
differences being interpreted. **Nothing in this harness should be read below 12
draws**, and differences under ~2pp should not be read at all.

## What would make this uninformative

12 draws is a small sample for a percentile bootstrap, and eight comparisons were
made. The 32-shot result survives that concern — both widths, both intervals well
clear of zero, and a null control that stays null. The 8-shot result does not, and
is reported as marginal rather than positive.

One encoder, one K, one domain, one taxonomy. Whether the band exists for
acoustically confusable keywords is untested; `narrowcast-kws/fewshot_curve.py` is
the instrument and it is already built.

## What would justify a follow-up

A pre-registration declaring the band, not this document reinterpreting itself. The
question it would ask: **does a supervised projection help wherever the head has
enough data to be fitted but not enough to find the subspace unaided, and is that
band identifiable before building?** If the band can be predicted from rows per
label and measured headroom, it is a `build` option with a rule for when to use
it. If it cannot, it is a curiosity — because a lever that helps in a range nobody
can locate in advance is not a lever a tool can offer.
