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

## When averaging hurts: the photographs must be of the same thing

*Added after a field case.* Three photographs of one bigleaf maple, averaged, and
the model answered **mistletoe**. Taken apart:

| photograph | what it shows | top answer |
|---|---|---|
| 1 | a moss-covered trunk filling the frame | *Phoradendron leucarpum* 30% |
| 2 | branches and leaves | *Acer platanoides* 27% |
| 3 | backlit canopy | *Acer platanoides* 31% |
| **averaged** | | ***Phoradendron* 28%**, maple 21% |
| 2 + 3 only | | *Acer platanoides* 31% |

The first photograph is of **bark and moss**, not of a maple — and the
centre-crop keeps only the middle square of a 3024×4032 portrait, which is
trunk. Answering "mistletoe" to a mossy branch is defensible; *Phoradendron* is a
parasite that grows on trees, so its training images are pictures of exactly this.

**Averaging assumes the photographs are of the same subject.** Where one is of
the host rather than the plant, it drags the mean toward whatever that host looks
like. The earlier measurement drew its groups from GBIF occurrences, where a
contributor's several images are of one organism by construction, so the failure
could not appear.

Two things follow for an interface:

- **Ask for the leaf, the flower, the fruit — not the tree.** The measured gain
  assumes three views of the plant, not three views of the scene.
- **Disagreement between photographs is a signal worth surfacing**, and it is
  free: score each one as well as the average, and when the top answers differ,
  say so rather than presenting a confident mean.

The genus rank rescued this case: summed over species, *Acer* took 36% against
*Phoradendron*'s 28%, so the coarse answer was right where the fine one was not.
That is the section of the card that had nothing to do at K=20 and is earning its
place at K=970.
