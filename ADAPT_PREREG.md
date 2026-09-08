# Pre-registration — does domain-adapting a 17.9 MB encoder close the gap?

Written before any fine-tuning run.

## The claim under test

`SMALL_FRONTIER_FINDINGS.md` closed the cheap levers: regularisation, ensembling
and test-time augmentation are all null against `mobileclip2_s2`'s 0.6236 species
at K=490, and the step to `plantclef24` at 43 MB is **14.4pp**. That gap is in
the encoder's representation.

**What `plantclef24` *is*, is a DINOv2 ViT-B fine-tuned on 7,806 Pl@ntNet
species.** Stock DINOv2 ViT-B is unremarkable on this task. So the 43 MB
advantage is **domain adaptation, not size** — and nobody has domain-adapted a
17.9 MB model.

> **Null to be rejected:** fine-tuning MobileCLIP2-S2 on a broad plant corpus
> does not narrow its gap to `plantclef24`.

## Design

**Adapt** the MobileCLIP2-S2 image tower on **Pl@ntNet-300K** — 306,146 images,
1,081 species — with a plain classification objective, then **discard the head
and freeze the tower**. That is the same recipe that produced `plantclef24`, one
encoder-scale down and on a smaller corpus.

**Evaluate** by the standard path, unchanged: fit the production logistic head on
the catalogue embeddings, score the 5,534-observation iNaturalist set,
observation-level species and genus top-1 with a species-clustered bootstrap. The
adapted encoder is frozen throughout evaluation.

**Comparisons**: stock `mobileclip2_s2` (0.6236 / 0.8122), `plantclef24` at
43.3 MB (0.7671 / 0.9258), `bioclip2_cml4` at 160 MB (0.8370 / 0.9735).

### What is trained on, and what is deliberately withheld

Adaptation uses the **530 catalogue species only** — 299,832 of Pl@ntNet-300K's
306,146 images. That adds no leakage relative to the current setup: the
production head *already* trains on those exact photographs, and the evaluation
is iNaturalist — different photographs, different source.

**The other 551 species are withheld from training entirely**, and that choice is
load-bearing rather than incidental. It costs almost nothing — the catalogue was
selected by image availability, so those species carry about nine images each —
and it is the only thing that makes the transfer probe below a test rather than a
formality. Fine-tuning on all 1,081 species and then probing 551 of them would
measure in-training-distribution performance and pass by construction.

## Predictions

1. **The gap narrows but does not close.** Predicted adapted-S2 species top-1
   between 0.65 and 0.72, against stock 0.6236 and `plantclef24` 0.7671. Half the
   gap is the modal outcome.
2. **Genus moves more than species**, because adaptation should first teach the
   encoder plant-shaped structure and only then fine distinctions.
3. **If it reaches 0.74+, the 17.9 MB point becomes a real product option** and
   the size decision changes — that is the outcome worth running for.

## What this cannot establish, declared in advance

- **Whether this is a general plant encoder or a catalogue-specific one.** It is
  trained on the catalogue's own species, so a good catalogue score cannot
  separate *learned plants* from *learned this label set*. **The linear probe on
  the 551 withheld species is what separates them**, it runs every epoch beside
  the loss, and it is compared against the stock tower's score on the identical
  set. A species gain with transfer at or below stock is a catalogue-specific
  encoder, and must be reported as such rather than as a small plant encoder.
- **It breaks the tool's economics if used per-user.** narrowcast's premise is one
  shared frozen encoder plus a ~40 KB per-user head. A per-user fine-tune ships a
  per-user encoder and that premise is gone. This is only viable as a **shared**
  adapted encoder, which is exactly what `plantclef24` is and how it must be
  compared.
- **Small encoders lose most to a change of source** (`DOMAIN_SHIFT_FINDINGS.md`:
  −0.179 for S2 against −0.001 for ViT-L). Any gain here is measured on
  iNaturalist and should be re-checked cross-source before it is believed.
