# The small end of the frontier does not move on inference tricks

Pre-registered in `SMALL_FRONTIER_PREREG.md`. Three levers that cost few or no
bytes, measured against the 17.9 MB baseline on the standard 5,534-observation
evaluation set, observation-level.

**All three are null. The 14.4pp step from 17.9 MB to 43 MB is an
encoder-quality gap and inference-side levers move it by fractions of a point.**

## The frontier

| arm | MB | species | genus | coverage @20% |
|---|---|---|---|---|
| `mobileclip2_s0` | 5.7 | 0.5814 | 0.7491 | 0.248 |
| **`mobileclip2_s2`** | **17.9** | **0.6236** | 0.8122 | 0.426 |
| `s2`, `C=1` | 17.9 | 0.5537 | 0.5945 | 0.117 |
| `s2`, `C=100` | 17.9 | 0.6242 | **0.8285** | 0.301 |
| `s0 ⊕ s2`, posteriors averaged | 23.6 | 0.6247 | 0.8015 | 0.407 |
| `s2` + TTA, 6 views | 17.9 | **0.6306** | 0.8183 | 0.428 |
| `plantclef24` | 43.3 | 0.7671 | 0.9258 | 0.616 |
| `bioclip2_cml4` | 160.0 | 0.8370 | 0.9735 | 0.692 |

## What each lever bought

**Regularisation: nothing.** `C` has been pinned at 10 since the beginning and
was never swept. At `C=100` species moves **+0.0006**. It does buy +1.6pp of
genus, and costs **12pp of coverage** doing it, so it is not a better product.
`C=1` is much worse across the board. The unexamined default was not leaving
anything on the table, which is worth knowing about a number that has stood
under every result in this repo.

**Ensembling two small encoders: nothing, and it costs genus.** `s0 ⊕ s2` at
23.6 MB gives **+0.0011** species and **−1.1pp** genus against `s2` alone at
17.9 MB. Paying 5.7 MB for a tenth of a point and losing a point of genus is not
a trade.

> Prediction 2 was **wrong**. The pre-registration predicted 1–3pp, with the
> reasoning that two encoders trained on similar web data correlate too much to
> ensemble well. That reasoning was right and the prediction attached to it was
> not: the correlation is high enough to buy nothing at all, not merely less.

**Test-time augmentation: the best of the three, and still null.** Six
deterministic views — identity, horizontal flip, and two centre crops with their
flips — averaged in embedding space. Paired over the same 3,435 in-catalogue
observations, species-clustered:

| | TTA − baseline |
|---|---|
| species | **+0.0070 [−0.0015, +0.0154]** |
| genus | +0.0061 [−0.0011, +0.0130] |

Both intervals include zero, and it costs **6× latency**. Unlike the other two it
at least does no harm — coverage and genus both move the right way — but 0.7pp
with an interval spanning zero is not a reason to sextuple inference cost.

## What this establishes

**A family of cheap ideas is closed.** Regularisation, small-encoder ensembling
and test-time augmentation are the obvious free-win levers at a byte budget, and
on this catalogue none of them is worth taking. Anyone reaching for them next
should read this first.

**The remaining lever is domain adaptation of a small encoder, and it is the
only one left that is not a trick.** `plantclef24` beats `s2` by 14.4pp of
species, and what `plantclef24` *is* is a ViT-B fine-tuned on 7,806 Pl@ntNet
species. The 43 MB advantage is domain adaptation, not size — DINOv2 ViT-B is
otherwise unremarkable here. **Nobody has domain-adapted a 17.9 MB model.**
Fine-tuning MobileCLIP2-S2 on Pl@ntNet-300K is the direct attack on "smallest
usable model", and if it closed even half the gap the 17.9 MB point becomes a
real product option. It is a GPU job, not an afternoon.

## The qualification that matters for the tool

**Every number here is at K = 490.** The tool's case is a user picking 10–50
labels, and `EMBEDDED_FINDINGS.md` already puts `s2` at 0.9493 on a 20-label
Pl@ntNet set and 0.8557 on 20 Oregon species from iNaturalist. Nothing here
contradicts that. **"17.9 MB is not viable" is a statement about the 490-class
app, not about the general-domain tool**, and the two should not be quoted
interchangeably.

## Reproduce

```
PYTHONPATH=. .venv/bin/python -m analysis.small_frontier
PYTHONPATH=. .venv-mps/bin/python -m analysis.tta_embed --variant mobileclip2_s2
```
