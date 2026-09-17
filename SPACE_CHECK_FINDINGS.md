# The embedding-space check cannot see the failure it was written for

narrowcast's `build.check_same_space` warns when two vector pools look like they
came from different encoders. It was written for a recorded failure: a bundle
embedded with a Core ML export was measured against a background pool embedded
with the torch original, the negatives were trivially rejected, and label share
came out **three points** flattered with nothing erroring.

It was shipped as a *warning* rather than a refusal because its premise — that one
encoder's embeddings share a common cone, so a cross-pool cosine near zero means
two encoders — was untestable inside narrowcast, which cannot load an encoder.
This repository can. The answer is that the check should stay a warning, and for
a worse reason than expected.

## Design

13 encoder variants × 3 organs, from the embedding caches in `data/processed/`:
the **same photographs** embedded by every variant, so a pair differs in the
encoder and nothing else. Pairs of different width are excluded, because those are
refused outright and need no geometry.

- **False positives** — one encoder, two different organs (bark/flower/leaf). The
  subject matter changes and the encoder does not, so a trip is a false alarm.
- **True positives** — one organ, two encoders. Split by whether the two are an
  export or quantization of *one* model (`bioclip2` vs `bioclip2_cml4`) or come
  from different model families (`bioclip2` vs `mobileclip2_s2`).

## Results

| case | n | cross-pool cosine | caught |
|---|---|---|---|
| same encoder, different organ | 39 | — | **2 (0.051)** ← false positives |
| export/quantization of one model | 21 | +0.44 … +0.83, mean +0.64 | **0 (0.000)** |
| different model families | 105 | −0.11 … +0.83, mean +0.07 | 74 (0.705) |

**The recorded failure, exactly:**

| organ | torch BioCLIP-2 vs its Core ML int4 export | caught |
|---|---|---|
| bark | cross = **+0.8182** | no |
| flower | cross = **+0.7879** | no |
| leaf | cross = **+0.7865** | no |

## What this means

**The check detects an unrelated encoder family about seven times in ten, a wrong
export never, and it refuses about one legitimate pair in twenty.** The failure it
was written for is the one case it is structurally blind to — a faithful export
lands in nearly the same space, which is exactly what makes it faithful. There is
no threshold that separates the two: same-family pairs reach 0.83 and
different-family pairs also reach 0.83, so the distributions overlap at the top
and a stricter bound buys detection only by spending the false-positive rate,
which is already non-zero.

The two false positives are both `plantclef24`, bark against flower and leaf. It
runs at 518 px and is fine-tuned on 7,806 Pl@ntNet species; whatever the cause,
one encoder's own outputs on two organs were called unrelated. That alone settles
the promotion question — refusing would have blocked a legitimate audit.

**So: do not promote it to a refusal.** narrowcast's warning now names its
measured rates and states what it cannot see, because a warning that overstated
its coverage would be worse than none: a reader seeing no warning would conclude
the pools matched.

## What does catch it

Declaration, not geometry. Both npz files can name the encoder that produced them
(`encoder` field), and narrowcast compares the two and refuses a mismatch. It
cannot verify either claim — it never loads an encoder — but comparing two
declarations is the only mechanism here that sees a Core ML export against its own
torch original, and it costs one string per file.

That puts the obligation on the producing pipeline. `regional_embed.py` and the
other embedding scripts here should write `encoder` into every npz they emit; the
narrowcast side is done.

## A note on what was nearly shipped

The first version of this check refused rather than warned, and its own first test
false-positived on synthetic vectors with independently drawn class centroids —
vectors that share no cone at all. Had that been read as a fixture problem rather
than as evidence, a refusal would have gone out that blocks 5% of legitimate pairs
and catches 0% of the failure it names. This is the third entry in this project's
list of partial measurements read as verdicts.
