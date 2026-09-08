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

> **The generalisation claim is unsupported by this run.** The notebook's
> held-out-species probe reported stock 0.5278 → adapted 0.7013, +0.1735 — but
> those 551 species were *in* the fine-tune, and the probe set included images the
> tower trained on. It passes by construction and means nothing. The corrected
> notebook withholds those species from training entirely so the probe becomes a
> test; that run has not happened, and until it does the claim here is limited to
> this catalogue.

## What it changes

**For the 490-class app**, the size decision has a new point on it: 17.9 MB now
buys 0.726 species where it bought 0.624, and the 43 MB option's advantage is
4.1pp rather than 14.4pp — at 7× the latency and 2.4× the bytes. `plantclef24`
was the middle ground; adapted S2 undercuts it on every axis except accuracy.

**For the tool**, nothing yet, and possibly nothing ever. narrowcast's premise is
one shared frozen encoder plus a ~40 KB per-user head. A *shared* plant-adapted
encoder fits that premise; a per-user fine-tune destroys it. Whether this tower
is shared-usable depends entirely on the transfer probe that has not validly run.

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
