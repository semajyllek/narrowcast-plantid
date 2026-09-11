# A narrow task needs 16 bytes, and a few-shot card lies about which 16

Phase 1 of `TINY_PREREG.md`, which gates a reopening of distillation on two
training-free measurements. **Both gates clear**, and the two measurements
disagree about what the interesting problem is.

## 1. The capacity floor: 32 dimensions at 4 bits

`analysis/capacity_floor.py`. Cached embeddings, 15 draws per cell, both arms of
`EMBEDDED_FINDINGS.md` reported separately and never averaged. PCA fitted on
training rows only; quantisation ranges taken from training rows only, so values
outside them clip, as they would at export.

### First, a correction to the gate as I wrote it

**G1 named `d ≤ 32` at `K ∈ {10, 20}` without requiring `d < K`, and that makes
the cell it names nearly uninformative.** A K-way linear head's decision depends
only on the span of its K coefficient vectors — at most K directions, K−1 after
the constant. "32 dimensions holds a 20-label task" sits *above* a bound that
holds by construction, so it is close to arithmetic rather than a measurement.
The `rand32` control rules out triviality — an arbitrary 32-dimensional map does
fail — but it cannot make `d ≥ K` informative about capacity.

The gate is therefore re-read on the cells where `d < K`, which were already in
the sweep. Recorded rather than silently re-specified, per this repo's
retract-in-place convention.

**HARD arm (congeners), closed-set top-1 against the unreduced embedding, `d < K`
cells only:**

| encoder | int4 size | K=20, d=16 | K=50, d=32 | K=50, d=16 |
|---|---|---|---|---|
| bioclip2 | 152 MB | **−0.0323** | **−0.0191** | **−0.0437** |
| plantclef24 | 43.3 MB | −0.0564 | −0.0351 | −0.0688 |
| mobileclip2_s2 | 17.9 MB | −0.0682 | −0.0425 | −0.0975 |
| mobileclip2_s0 | 5.7 MB | −0.0771 | −0.0470 | −0.1105 |

against the random-projection control at the same width:

| encoder | K=20, rand16 | K=50, rand32 | K=50, rand16 |
|---|---|---|---|
| bioclip2 | −0.0983 | −0.0632 | −0.1338 |
| plantclef24 | −0.1838 | −0.1126 | −0.2336 |
| mobileclip2_s2 | −0.1678 | −0.1256 | −0.2340 |
| mobileclip2_s0 | −0.2302 | −0.1557 | −0.2698 |

**G1 clears, narrowly and only for the largest encoder** — bioclip2 at K=50 in 32
dimensions costs 1.91pp, inside the declared 3pp; at K=20 in 16 dimensions it
costs 3.23pp, marginally outside it. Read honestly, the gate is a pass on one
cell and a near-miss on another, not the comfortable clearance the `d ≥ K` cells
suggested.

**G2 clears decisively** — the random control costs 4–13pp more than PCA in every
one of the 48 `d < K` cells, and the margin *widens* as d falls. The
low-dimensional structure is a task subspace, not head capacity.

### Quantising the embedding is free, in all 32 cells

Across four encoders, both arms, K ∈ {5, 10, 20, 50}, the largest deviation from
the float embedding at **int8 or int4 is 0.6pp, and its sign is inconsistent** —
it is as often positive as negative. This is noise. It has never been measured
here, and it was worth an afternoon to know.

### And the two compose

Reported apart they are two ~0 results; the claim is the product, and nothing
guaranteed it. PCA orders components by variance, so the low ones occupy a much
smaller share of a per-dimension quantisation range than the high ones, which is
exactly the condition under which a 4-bit quantiser should fall over.

| | K=10 HARD | K=20 HARD |
|---|---|---|
| bioclip2 `pca32` | −0.0145 | −0.0101 |
| bioclip2 `int4` | −0.0015 | −0.0017 |
| **bioclip2 `pca32+int4`** | **−0.0207** | **−0.0163** |

Slightly worse than additive, and far short of a collapse.

**32 dimensions at 4 bits is 16 bytes.** The unreduced BioCLIP-2 embedding is 768
float32s, or 3,072 bytes. A **192× smaller representation costs 2.1pp of
congener top-1 at K=10** and 1.6pp at K=20.

### The asymmetry is the finding, and it points at distillation

Compressibility tracks encoder size in the *opposite* direction to intuition. The
152 MB encoder gives up the least when squeezed to 32 dimensions (−1.45pp); the
5.7 MB encoder gives up the most at K=20 (−4.52pp). A large encoder spends most
of its representation on distinctions a narrow catalogue never asks about, so
almost all of it is discardable. **A natively small encoder has no spare capacity
to shed — it is already near its floor.**

That is the argument for task-conditional distillation stated in measurement
rather than in prose, and it is precisely the axis the two closed attempts did
not vary: they asked a student to reproduce a *general* embedding, and this says
a K=20 task consumes about 4% of one.

**It does not say a 1 MB student can produce those 16 bytes.** Finding a
low-dimensional subspace inside a good representation is not the same problem as
computing it from pixels, and nothing here bounds the second. That is Phase 3.

## 2. The few-shot curve, and it is the more important result

`analysis/fewshot_curve.py`. Training rows drawn from Pl@ntNet at *n* per label,
evaluated on held-out **iNaturalist observations** — cross-source, clustered by
observation, with the real near-OOD bucket. narrowcast's own
`fit_head → score_frame → fit_and_measure` path, unmodified, so these are the
numbers a card would print.

**`mobileclip2_s0`, K=20, HARD arm:**

| shots | coverage | precision | **label_share** | top-1 | headroom |
|---|---|---|---|---|---|
| 1 | 0.283 | **0.957** | **0.006** | 0.280 | 0.520 |
| 2 | 0.329 | 0.956 | 0.018 | 0.333 | 0.468 |
| 4 | 0.380 | **0.969** | 0.003 | 0.400 | 0.426 |
| 8 | 0.404 | 0.967 | 0.030 | 0.461 | 0.405 |
| 16 | 0.470 | 0.966 | 0.052 | 0.504 | 0.352 |
| 32 | 0.474 | 0.968 | 0.080 | 0.531 | 0.328 |
| 64 | 0.490 | 0.946 | 0.161 | 0.554 | 0.312 |
| all | 0.487 | 0.957 | 0.160 | 0.552 | 0.304 |

**From one example per class, a coverage-and-precision card reads: answers 28% of
queries at 96% precision.** That is a number anyone would ship. The model is
naming an actual label on **0.6%** of in-list observations; the rest is "it is a
*Sedum*", which narrows nothing when the list is eight *Sedums*.

**Precision is at its highest where the model is least useful** — 0.969 at four
shots, 0.946 at sixty-four. Coverage and precision both move the *wrong way*
relative to utility across the low end of the curve, and only `label_share`
tracks what the user gets. This is the README's headline trap, and the few-shot
regime is the worst place it has been measured.

### It is not a background-ratio artifact

At K=20 and one shot the head sees 20 positive rows against a fixed 800
negatives — 40:1 — and `class_weight="balanced"` reweights the loss without
guaranteeing the boundary is unmoved. A negative-dominated fit would produce a
collapsing `label_share` all by itself. `EMBEDDED_FINDINGS.md` ran this control
for the same reason; it is run here too, with negatives scaled to four per
positive so the ratio is constant across the sweep.

**K=20 HARD, `mobileclip2_s0`, 12 draws:**

| shots | `label_share` fixed 800 | `label_share` 4-per-positive | coverage fixed | coverage 4:1 |
|---|---|---|---|---|
| 1 | 0.0056 | **0.0052** | 0.283 | 0.285 |
| 8 | 0.0298 | **0.0395** | 0.404 | 0.415 |
| 64 | 0.1611 | **0.1501** | 0.490 | 0.488 |
| all | 0.1601 | **0.1419** | 0.487 | 0.486 |

Indistinguishable. The collapse is a property of the label set and the shot
count, not of how many negatives the reject class was fitted on.

Contrast the same encoder on the EASY arm at K=20, where the group rank carries
no free lunch: `label_share` rises 0.038 → 0.489 monotonically with shots, and
tracks coverage. **The trap is a property of the crowded arm, not of small data —
but small data is where it is most dangerous, because it is where the headline
metrics look most reassuring.**

### Encoder quality is worth 7.8× on separated sets and nothing on crowded ones — at one shot only

The obvious objection to the table above is that `mobileclip2_s0` is the weakest
encoder in the registry. `label_share` at K=10 across the whole size range,
evaluated on iNaturalist:

| encoder | int4 | EASY, 1 shot | EASY, all | HARD, 1 shot | HARD, all |
|---|---|---|---|---|---|
| mobileclip2_s0 | 5.7 MB | 0.092 | 0.580 | 0.013 | 0.168 |
| mobileclip2_s2 | 17.9 MB | 0.147 | 0.644 | 0.004 | 0.187 |
| plantclef24 | 43.3 MB | 0.538 | 0.847 | 0.008 | 0.210 |
| **bioclip2** | **152 MB** | **0.714** | **0.908** | **0.010** | **0.471** |

Read the four cells separately, because they do not say the same thing.

**Separated, one shot: encoder quality is worth 7.8×**, monotone in size, 0.092 →
0.714. One example per label through BioCLIP-2 beats sixty-four through
`mobileclip2_s0` (0.714 against 0.597). That is `CLAUDE.md`'s "the accuracy lives
in the encoder's representation, not the head's training data" at its limit.

**Crowded, one shot: encoder quality is worth nothing.** Every encoder from
5.7 MB to 152 MB lands between 0.004 and 0.013, and the ordering does not hold —
the 17.9 MB encoder is last and the 152 MB one is not first. A 27× range of model
size produces no signal at all.

**Crowded, full data: encoder quality is worth 2.8×** and the ordering returns,
0.168 → 0.471. So a better encoder *does* substantially help a crowded label set.
It simply cannot do so from one example.

> **Correction.** An earlier version of this section, written before the
> `bioclip2` arm finished, said encoder quality "buys nothing at all" on crowded
> sets. That was wrong, and it was wrong because it generalised from
> `plantclef24` — which is fine-tuned on Pl@ntNet and carries the largest
> cross-source deficit of any encoder here (`DOMAIN_SHIFT_FINDINGS.md`). At full
> data BioCLIP-2 more than doubles it on the crowded arm. The claim that survives
> is narrower and about the few-shot regime specifically.

And the full trap is intact at the few-shot end, on the *best* encoder in the
project:

| bioclip2, K=10, 1 shot | coverage | precision | label_share |
|---|---|---|---|
| HARD | **0.723** | **0.993** | **0.010** |
| EASY | 0.595 | 0.968 | 0.714 |

The crowded arm reports **higher coverage at higher precision** than the
separated one and is **69× worse** at naming a label. This is the README's
headline contrast, reproduced from a single example per class, with both
reassuring metrics pointing the wrong way at once, on a 152 MB encoder.

**So the few-shot regime is where encoder shopping stops working.** It is the one
lever `SMALL_FRONTIER_FINDINGS.md` left open, it is worth a great deal on
separated sets and with plenty of data, and on a crowded set with a handful of
examples it is worth nothing measurable.

### It replicates in audio, and harder

Full result in `narrowcast-kws/FEWSHOT_FINDINGS.md`. Speech Commands,
`wav2vec2-base`, K=7, **speaker-disjoint** splits, groups from k-means over
per-word centroids rather than semantics — that repo's hardest-won lesson is that
semantic groups are not groups to the encoder.

| K=7, 4 shots | coverage | precision | `label_share` |
|---|---|---|---|
| CROWDED (one acoustic cluster) | **0.477** | **0.926** | **0.0011** |
| VARIED (seven clusters) | 0.026 | 0.737 | 0.0304 |

**18× the coverage at higher precision, naming a label 28× less often.** Cleaner
than the plant version, where the crowded arm was at least worse at naming
labels; here it is 28× worse while looking 18× better. The coverage ratio between
the arms runs 907× → 18.6× → 1.21× at 1, 4 and unlimited shots, so the distortion
is a function of data volume and disappears with enough of it.

Different modality, different encoder family, a different source of crowding
(acoustic confusability rather than taxonomy), and real clusters. That is the bar
the original coverage trap cleared with birds and 20 Newsgroups.

### The mechanism: few shots manufacture headroom

`HEADROOM_FINDINGS.md` establishes that headroom — coarse-rank minus fine-rank
accuracy — governs retreat to the group rank, CV R² 0.883 over 1,409 arms.
**Nobody measured that headroom is itself a function of the training-set size.**
It is, strongly, and in both domains:

| headroom | 1 shot | 4 shots | all |
|---|---|---|---|
| audio CROWDED | **0.762** | 0.606 | 0.144 |
| plants, `mobileclip2_s0`, K=20 HARD | 0.520 | 0.426 | 0.304 |
| plants, `plantclef24`, K=10 HARD | 0.431 | 0.342 | 0.263 |
| plants, `mobileclip2_s0`, K=20 EASY | 0.047 | 0.031 | 0.021 |
| audio VARIED | 0.000 | 0.000 | 0.000 |

Monotone in every arm. Starving the head collapses **fine**-rank accuracy while
**coarse**-rank accuracy holds, because telling *Sedum acre* from *Sedum album*
needs data and telling a *Sedum* from a *Trifolium* does not. Headroom balloons,
the cascade correctly judges a group answer worth more than a decline under the
declared utility, and coverage inflates while precision holds — on answers that
narrow nothing.

The chain, every link of which is already-established machinery:

> **few shots → fine accuracy collapses → headroom balloons → retreat →
> coverage and precision both look good → the product is useless**

The new part is that **data volume drives the first link**, and nothing in either
repo warned about it. It also explains why the effect is confined to crowded
sets: the separated arms have headroom ~0 by construction — one label per group
means a group answer *is* a label answer — so there is nothing to retreat to and
the cascade declines instead. **Declining is the honest failure.** The crowded
arm's 0.993 precision at one shot is the dishonest one.

### What this says about the "8 examples per class" framing

SetFit's claim is that eight labelled examples per class suffice. **On separated
sets with a good encoder it is not merely roughly right, it is exact** —
`label_share` at 8 shots as a fraction of `label_share` with every available row,
K=10 EASY:

| encoder | 8 shots | all | fraction |
|---|---|---|---|
| bioclip2 | 0.9109 | 0.9082 | **100%** |
| plantclef24 | 0.8684 | 0.8466 | **103%** |
| mobileclip2_s2 | 0.5905 | 0.6441 | 92% |
| mobileclip2_s0 | 0.5073 | 0.5802 | 87% |

At the top of the range eight examples per label is **saturated**: more data adds
nothing measurable, and the whole remaining gap is the encoder. The weaker the
encoder, the more data it still wants — which is the same trade as the 32×
equivalence above, seen from the other end.

**On the HARD arm the claim fails at every encoder.** Eight shots as a fraction of
unlimited, K=10: bioclip2 0.387/0.471 = 82%, plantclef24 0.089/0.210 = 42%,
`mobileclip2_s0` 0.030/0.168 = 18%. The curve is still climbing at 64 shots for
all three, and the *absolute* numbers are what matter — 0.089 is not a product
whatever fraction of its ceiling it represents.

So "eight examples per class is enough" is true, on separated label sets, given an
encoder that already knows the domain. It is exactly the configuration in which
the user needed the least help.

Few-shot works exactly where the label set is separated, and the deficit on
crowded sets is not a data-volume problem that more examples fix cheaply.

## 3. The student: not demonstrated, and not re-closed either

Phase 3 of `TINY_PREREG.md`. Teacher `plantclef24` at K=14 fitted on Pl@ntNet,
everything scored cross-source on held-out iNaturalist observations through
`build.fit_and_measure` unchanged. Transfer set 29,499 Pl@ntNet images, targets
from `clf.predict_proba` over cached embeddings — **no teacher forward pass runs
anywhere**, so the whole pilot is 10–45 minutes on MPS.

| arm | init | int8 | px | teacher | student | Δ | top-1 |
|---|---|---|---|---|---|---|---|
| crowded | scratch | 0.14 MB | 128 | 0.554 | **0.000** | −0.554 | 0.277 |
| crowded | scratch | 0.14 MB | 224 | 0.554 | **0.000** | −0.554 | 0.355 |
| crowded | imagenet | 1.53 MB | 128 | 0.554 | **0.000** | −0.554 | 0.388 |
| crowded | imagenet | 1.53 MB | 224 | 0.554 | **0.174** | −0.380 | 0.492 |
| separated | scratch | 0.14 MB | 128 | 0.898 | **0.112** | −0.786 | 0.439 |
| separated | scratch | 0.14 MB | 224 | 0.898 | **0.158** | −0.740 | 0.505 |
| separated | imagenet | 1.53 MB | 128 | 0.898 | **0.228** | −0.670 | 0.667 |
| separated | imagenet | 1.53 MB | 224 | 0.898 | **0.284** | −0.614 | 0.740 |
| **crowded** | **scratch** | **0.14 MB** | **518** | 0.554 | **0.000** | **−0.554** | **0.471** |
| **separated** | **scratch** | **0.14 MB** | **518** | 0.898 | **0.190** | **−0.709** | **0.628** |

**Prediction 1 was wrong.** It said the separated arm would survive within 5pp
and the crowded arm fail by more than 15pp. Both fail, and *separated fails
harder in absolute terms*. Prediction 2 — sub-1 MB lands more than 10pp below
teacher on both arms — is confirmed at every point tried.

### Settled at the teacher's own resolution: distillation closes

**A sub-1 MB student trained at 518 px — the teacher's own input — fails both
arms.** Separated reaches `label_share` 0.190 against a pass bar of 0.849;
crowded reaches 0.000 against a fail bar of 0.404. Re-close condition (a) of
`TINY_PREREG.md` is satisfied: the resolution handicap is gone and the result did
not move. **Task-conditional distillation is closed**, and unlike the two previous
closures there is no outstanding objection to it.

The resolution lever was genuinely near its ceiling. Extrapolating the 128 → 224
step at ~8pp of top-1 per doubling predicted 0.45 crowded and 0.59 separated
before the run; measured, **0.4711 and 0.6281**. Two further doublings of pixels
bought what one did, and neither came close to the teacher's 0.897 / 0.951.

### The cliff is not a function of top-1 alone — capacity costs sharpness

The most useful thing in the final run is an anomaly. The crowded student reached
top-1 **0.4711**, *above* the 0.492 at which the 1.53 MB student scored
`label_share` 0.174 — and it named **zero** labels.

| student | top-1 | `label_share` |
|---|---|---|
| 1.53 MB @ 224 px | 0.492 | 0.174 |
| **0.14 MB @ 518 px** | **0.471** | **0.000** |

So the `p > 0.800` gate is not reducible to argmax accuracy. The smaller model's
posteriors are **flatter**: it is right about as often and confident far less
often, so it almost never clears the bar even when its best guess is correct.
Capacity buys sharpness as well as accuracy, and the cascade is gated on the
first. "A smaller model" and "a more cautious model" are not separable here, which
is why fine-rank accuracy and the label share have to be read together — neither
predicts the other across a capacity change.

### Latency: the storage-only worry did not materialise

**1.36 ms/img crowded and 1.41 ms/img separated at 518 px, batch 1.** 138k
parameters stay cheap even at the teacher's resolution, so the concern that a
student needing 518 px would be a small *file* and an expensive *computation* is
not borne out.

**This is an A100 figure and is not comparable to `plantclef24`'s 38.6 ms, which
is MPS.** Different hardware, different backend; it bounds nothing about an ANE or
a microcontroller. What it does establish is that resolution alone does not make
a 138k-parameter network expensive — the cost that makes `plantclef24` slow is its
86.6M parameters *at* 518 px, not the pixels by themselves.

### What the retired argument said

`TINY_PREREG.md` says distillation closes for good if the student is more than
15pp below teacher on **both** arms at every size tried. It is, six times. **I am
not closing it, because the test was handicapped in a way I did not control for
when I set the resolution.**

`plantclef24` runs at **518×518**. The student ran at 128 — one sixteenth of the
pixels — so a failure to separate two *Sedum* leaves is partly a statement about
what the student was shown, not only about its capacity. Raising it to 224 moved
the crowded arm from **exactly 0.000 to 0.174** and top-1 from 0.388 to 0.492.
That lever is real and it is nowhere near exhausted at 224 of the teacher's 518.
The training budget is not exhausted either — 30–60 epochs with KL still falling.

Before the 518 px run this section read **not demonstrated**, on the grounds that
the pre-registered condition had been met six times at 128–224 px against a
teacher at 518, and that one resolution step had taken the crowded arm from 0.000
to 0.174. That caution was correct to hold and is now discharged rather than
retracted: the amended condition named the run that would settle it, the run
happened, and the answer did not change. Recording it because a condition that is
only ever invoked to keep a question open is not a condition.

### Two corrections to the size claim

**The `imagenet` arm is 1.53 MB int8, not the 2.54 MB first written here.**
ImageNet's 1,000-way classifier is ~1M of `mobilenet_v3_small`'s 2.54M
parameters, and replacing it with a 15-way layer deletes most of it — measured,
1,533,231 parameters. It sits *just* over the 1 MB budget rather than well past.

**So "no ImageNet-initialised model fits the budget" is not established**, and
nothing here should be read as saying it. It is an artifact of only trying full
width. A narrowed variant would plausibly land under 1 MB with most of the 224px
gain, and that has not been run.

### Every 0.0000 is the declared utility working, not a bug

Three configurations scored `label_share` **exactly** 0.0000, which looks like a
broken run until the utility is read. `cascade.UTILITY` pays `label_correct=+1`
and `wrong=−4`, against `decline_in_catalog=0`, so naming a label beats declining
only when

    p·1 + (1−p)·(−4) > 0   ⟹   **p > 0.800**

The cascade names a label only when it is 80% sure. Sorted by fine-rank accuracy,
the crowded arm shows the resulting cliff:

| top-1 | `label_share` |
|---|---|
| 0.277 | 0.0000 |
| 0.355 | 0.0000 |
| 0.388 | 0.0000 |
| 0.492 | 0.1736 |
| 0.897 *(teacher)* | 0.5537 |

**A model below roughly 0.45 top-1 on a crowded set never reaches 80% confidence
on any single label, so it names none.** The 0.0000 is not a coincidence repeated
three times — it is the threshold refusing to defend an answer it cannot defend,
which is the behaviour the utility was declared for. `TEXT_FINDINGS.md` derived
the group-rank break-even (0.889) the same way.

Two consequences. First, `label_share` is a **gated** quantity, not a smooth
function of accuracy, so it is the wrong instrument for detecting saturation below
the cliff — it is floored at zero and cannot move. Fine-rank accuracy is the
right one. Second, the separated arm has no such floor (0.439 → 0.112,
0.505 → 0.158, 0.667 → 0.228, 0.740 → 0.284, all non-zero and monotone), because
with one label per group the cascade cannot retreat and must either name or
decline.

### What the ladder does establish: headroom is a monotone function of capacity

The crowded arm, ordered by effective capacity, is the cleanest thing in this
document:

| configuration | top-1 | headroom | `label_share` |
|---|---|---|---|
| 0.14 MB @ 128px | 0.277 | **0.500** | 0.000 |
| 1.53 MB @ 128px | 0.388 | **0.422** | 0.000 |
| 1.53 MB @ 224px | 0.492 | **0.291** | 0.174 |
| teacher, 43.3 MB @ 518px | 0.897 | **0.119** | 0.554 |

Monotone in all three columns. As effective capacity rises, fine-rank accuracy
rises, **headroom falls**, and the label share rises with it. Combined with §2's
shot curves this gives the mechanism three independent handles:

| how the fine-rank decision was starved | headroom | `label_share` |
|---|---|---|
| one training example per label | 0.43–0.52 | ~0.01 |
| a 0.14 MB student | 0.500 | 0.000 |
| a 1.53 MB student @ 128px | 0.422 | 0.000 |

**Starve it by data, by parameters, or by pixels and the same thing happens.**
Headroom is not a property of the label set alone, as `HEADROOM_FINDINGS.md`
treated it — it is a property of the *gap between what the task needs and what
the model can deliver*, and any of three levers opens that gap.

### The failure is invisible on exactly one of the two arms

The crowded students scored `label_share` **0.0000** while reporting coverage
0.11–0.13 at precision **0.937–0.972**. A card printing coverage and precision
alone would read *"answers 13% of queries at 97% precision"* for a model that
never once names a label.

The separated students fail just as badly and it is **obvious**: headroom is
0.000 by construction, so there is no group rank to retreat to, the cascade
declines instead, and coverage collapses to 0.09–0.23 where anyone would see it.

Same student, same training, same magnitude of failure, opposite legibility. That
is the entire argument for the label-level share, arrived at from a direction
that had nothing to do with label sets.

## Status of the gate

| gate | result |
|---|---|
| G1 — pca32 within 3pp on HARD, ≥1 encoder | **clears** (bioclip2, plantclef24 at both K) |
| G2 — random projection materially worse | **clears** (4–8pp, all 32 cells) |

Phase 3 is unblocked. Predictions stand as written in `TINY_PREREG.md` and are
not revised in light of the above.

## Reproduce

```
PYTHONPATH=. .venv/bin/python -m analysis.capacity_floor \
    --k 5 10 20 50 --variants bioclip2 plantclef24 mobileclip2_s2 mobileclip2_s0
PYTHONPATH=. .venv/bin/python -m analysis.fewshot_curve \
    --variant mobileclip2_s0 --k 10 20 --eval inat
```

## Bounds

One domain, one modality, one taxonomy. The HARD arm is built from Linnaean
genera, so "congener" here means a real biological grouping and not an arbitrary
hard-negative set; whether the same subspace argument holds for crowded label
sets in audio or text is untested. The capacity floor is measured on **frozen
embeddings that already exist** — it bounds what a student must emit, not what a
student can compute.
