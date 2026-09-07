# Seven points of species accuracy are sitting unclaimed, and the price is the clean evaluation

Pre-registered in `SOURCE_MIX_PREREG.md`, written before any head was fitted.
349 species, 4,676 held-out iNaturalist test photographs from observations no arm
trained on, photo-level top-1 macro-averaged over species, paired
species-cluster bootstrap.

**The primary null is rejected decisively.** Adding iNaturalist training data to
the shipped Pl@ntNet head takes species accuracy on iNaturalist photographs from
**0.7999 to 0.8718**, `+0.0718 [+0.0551, +0.0894]`, and costs nothing on
Pl@ntNet.

## Results, `bioclip2`

| arm | fitted on | iNat species | iNat genus | Pl@ntNet species |
|---|---|---|---|---|
| `P-full` | 22,772 Pl@ntNet rows — **what ships** | 0.7999 | 0.9438 | 0.7941 |
| **`P-full + i`** | that + 7,204 iNaturalist rows | **0.8718** | **0.9650** | 0.7961 |
| `P-2B` | 20 Pl@ntNet photos/species | 0.7873 | 0.9417 | 0.7884 |
| `P-B + i-B` | 10 of each | 0.8666 | 0.9577 | 0.7810 |
| `i-only` | 10 iNaturalist photos/species | 0.8549 | 0.9529 | 0.7170 |

Against the shipped baseline:

| arm | Δ iNat species | Δ iNat genus | Δ Pl@ntNet species |
|---|---|---|---|
| **`P-full + i`** | **+0.0718 [+0.0551, +0.0894]** | +0.0212 [+0.0126, +0.0301] | +0.0019 [−0.0052, +0.0097] |
| `P-B + i-B` | +0.0667 [+0.0449, +0.0907] | +0.0139 [+0.0047, +0.0235] | −0.0131 [−0.0285, +0.0033] |
| `i-only` | +0.0549 [+0.0312, +0.0815] | +0.0091 [−0.0022, +0.0207] | **−0.0771 [−0.1022, −0.0527]** |

## It is the source, not the volume

The pre-registered control. Same number of training rows, half of them swapped
for in-source ones:

> **`P-B + i-B` − `P-2B` = +0.0794 [+0.0584, +0.1020]**

Twenty photographs per species either way. Making half of them iNaturalist buys
almost eight percentage points. **In-source data is worth far more per row**, and
the gain on the primary arm is not a volume effect.

The sharpest form of that: `P-B + i-B` trains on **6,980 rows** and scores
0.8666; `P-full` trains on **22,772 rows** and scores 0.7999. A third of the data,
mixed, beats three times as much of it from one source by 6.7pp. **The gain
saturates fast** — ten in-source photographs per species collects nearly all of
it, which matters because in-source data is the scarce kind.

## Mixing beats either source alone

> **`P-full + i` − `i-only` = +0.0169 [+0.0020, +0.0316]**

So the answer is not "train on iNaturalist instead". The two corpora are
complementary: the mix is ahead of the best single source on iNaturalist, and
`i-only` collapses on Pl@ntNet (−7.7pp) exactly as the source-shift asymmetry
predicts.

## This does not contradict the source-shift finding

At first reading it looks like it should. `DOMAIN_SHIFT_FINDINGS.md` reports that
the Pl@ntNet head pays **nothing** to be tested on iNaturalist — −0.001 — while
this document reports it leaving 7pp on the table there. Both are true, and
reconciling them is the useful part:

- **No source penalty** means `pn→inat` ≈ `pn→pn`: the Pl@ntNet head does as well
  on iNaturalist as on its own held-out photographs.
- **iNaturalist is the easier corpus for this encoder** — `inat→inat` 0.8615
  against `pn→pn` 0.7701, which §15 already established and flagged as the
  unpredicted asymmetry.

So the Pl@ntNet head is not *losing* to the source change. It is **failing to
exploit an easier corpus**. A head that never sees the easy corpus cannot collect
what makes it easy, and no amount of the hard corpus substitutes.

## Replication on the deployable encoder

`bioclip2_cml4`, the 160 MB int4 build. Everything holds, slightly larger:

| contrast | `bioclip2` | `bioclip2_cml4` |
|---|---|---|
| `P-full + i` − `P-full`, iNat species | +0.0718 [+0.0551, +0.0894] | +0.0767 [+0.0604, +0.0947] |
| fixed-volume, `P-B + i-B` − `P-2B` | +0.0794 [+0.0584, +0.1020] | +0.0835 [+0.0624, +0.1060] |
| mix − `i-only` | +0.0169 [+0.0020, +0.0316] | +0.0217 [+0.0064, +0.0364] |
| cost on Pl@ntNet | +0.0019 [−0.0052, +0.0097] | +0.0028 [−0.0053, +0.0110] |

## The price, which was written down before the result

**This is not a recommendation to adopt it.** It is a price tag, and the
pre-registration says so in advance because a positive result is exactly when
that becomes tempting to forget.

iNaturalist is currently a **pure held-out source**. That is what makes the
central claim of `DOMAIN_SHIFT_FINDINGS.md` true — the shipped head has never
seen an iNaturalist photograph — and it is why every headline figure in this repo
is an *out-of-source* measurement. That is a substantially stronger thing to be
able to say than an in-source one, and it is unusual; most published numbers of
this kind are in-source and do not say so.

Adopting the mix spends it. Concretely:

- The evaluation set shrinks to whatever is held out of training.
- The remainder stops being cleanly out-of-source, so `DOMAIN_SHIFT_FINDINGS.md`
  and `CONTAMINATION_FINDINGS.md` both need re-qualifying against the new split.
- The competitive comparison in `COMPETITIVE_FINDINGS.md` — the tie with
  iNaturalist's server model on identical photographs — was measured with a head
  that had never seen iNaturalist data. Retaining that claim in its current form
  requires retaining that property.

**So the trade is: about 7 points of species accuracy against the ability to say
the number was measured out-of-source.** Which side is right is a product
judgement about who the number is for. If it is for users, take the accuracy. If
it is for the research record, the current arrangement is worth more than 7pp.

A middle path exists and is untested: hold out a fixed slice of iNaturalist —
whole observations, chosen once and never trained on — as a permanent
out-of-source evaluation set, and mix the rest into training. That keeps a clean
claim on a smaller set. Whether the remaining set is large enough to keep the
intervals useful is the open question, and it is measurable before committing.

## Reproduce

```
PYTHONPATH=. .venv/bin/python -m analysis.source_mix --variants bioclip2 bioclip2_cml4
```
