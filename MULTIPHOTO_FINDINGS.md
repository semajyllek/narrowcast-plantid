# Three photographs of one plant are worth more than any other lever here

Asked by the person building the app — "doesn't adding several images of the
same plant help?" — and the honest first answer was that it had not been
measured. What *had* been measured is **TTA**: six augmented crops of one
photograph, worth **+0.0070 [−0.0015, +0.0154]** at six times the latency
(`SMALL_FRONTIER_FINDINGS.md`), which is a null.

Several genuinely different photographs of one plant is a different thing, and it
is the single largest effect found in this project's deployment work.

## The measurement

478 held-out Oregon plants with ≥4 photographs each — held out in the sense that
none of their images went into the bank. Embeddings averaged, renormalised,
scored. Five draws per plant per condition.

**Nearest-centroid over the 961-species bank:**

| photographs of the plant | species top-1 | genus top-1 |
|---|---|---|
| 1 | 0.724 | 0.856 |
| 2 | 0.824 | 0.927 |
| 3 | 0.851 | 0.944 |
| 4 | **0.878** | **0.956** |

**The shipped 901-class head:**

| photographs | species top-1 | answers confidently | right when confident |
|---|---|---|---|
| 1 | 0.895 | 85.0% | 0.974 |
| 2 | 0.941 | 91.6% | 0.977 |
| 3 | **0.954** | 93.6% | 0.983 |
| 4 | 0.954 | 93.8% | 0.983 |

**Three photographs buy +12.7 points on the bank and +5.9 on the head, and the
third is where it saturates.** For comparison, everything else on the table in
this project: a 4× larger encoder buys ~2 points, TTA buys 0.7, and matching the
deployment preprocessing exactly buys 1.8.

## Why this is not TTA

TTA resamples one photograph — the same leaf, the same angle, the same light. A
second real photograph is a different organ, a different distance, often a
different part of the plant. The variance it averages out is *acquisition*
variance, which is exactly the variance the encoder is sensitive to and the
augmentation cannot reach. That this project's encoder is unusually sensitive to
resampling (`PREPROCESS_FINDINGS.md`) is the same fact seen from the other side.

## Consequences

**The app should ask for three.** It is the cheapest accuracy available and it
costs the user a few seconds rather than a larger download or a slower model.
Confidence rises with it too — the share of photographs the model will commit to
goes 85.0% → 93.6% — so the user sees fewer declines as well as fewer errors.

**Embeddings average, not posteriors.** Averaging the unit vectors and
renormalising is what was measured. Averaging the softmax outputs is a different
operation and was not.

**It does not fix being out of the catalogue.** A plant the bank has never seen
is not helped by photographing it three times; this improves discrimination among
species the model knows, not coverage of ones it does not.

## What is not measured

The 478 plants come from GBIF contributors, whose several photographs of one
occurrence were taken deliberately to document it. A user taking three quick
phone photographs may capture less variety, and the effect would then be smaller.
The direction is not in doubt; the magnitude on a phone is.
