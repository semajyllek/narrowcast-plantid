# The first regional bundle, end to end

Oregon, 24 species drawn from the GBIF survey at ≥500 regional records, 12
occurrences per species, fetched → embedded through the shipped int8 Core ML
artifact → audited by `narrowcast 0.2.0`.

```
python -m plantid.data.regions       --source gbif --state Oregon --min-obs 20 --out .../oregon_gbif.json
python -m plantid.data.regional_fetch --survey .../oregon_gbif.json --out .../oregon --per-species 12 --max-species 24
python -m plantid.data.regional_embed --manifest .../oregon/manifest.parquet --variant mobileclip2_s2 \
                                      --coreml .../mobileclip2_s2_q8_per_channel.mlpackage --out .../oregon_cq8.npz
narrowcast audit --embeddings .../oregon_cq8.npz --background-embeddings .../bg_cq8.npz --ood-rate 0.2
```

**618 photographs, 24 species, 288 distinct plants** (2.1 photographs per plant).

## It works, and it says it needs more data

| | value | 95% CI |
|---|---|---|
| coverage | 35.5% | 27.3–43.2 |
| precision | 96.0% | 91.2–99.0 |
| **label-level share** | **43.2%** | 33.6–53.2 |
| group-level share | 0.6% | 0.0–2.1 |
| closed-set top-1 | **77.2%** | 67.3–86.4 |

Against `TINY_K_FINDINGS`' 0.75 label share for stock S2 at K=20 on catalogue
images, 0.43 looks alarming. The card's own diagnosis is better than that reading:

> This model names a label on only 43.2% of in-list observations; **56.2% are
> declined outright** and only 0.6% are answered at group. This model is mostly
> declining rather than retreating, so coverage is paying the price directly.

> This head was fitted on **12 training rows per label (median), and that is
> thin** … Your list is not retreating to the group rank, which is the case where
> more data helps *most*.

Closed-set top-1 is **77.2%** — the model is not confused about these plants, it
is *unconfident*, and the threshold reads confidence. That is the same mechanism
int4 exposed, arriving from the other direction: too little training data rather
than too coarse a quantization.

So the first regional bundle is not a measurement of how hard Oregon is. It is a
measurement of 12 occurrences per species, and the card said so without being
asked.

**The fix is the cheap axis.** `--per-species 12` was chosen to keep the first
run short. `PHASE0_FINDINGS` puts stock S2 at 16–32 images before saturation, and
this run had 12.

## What the run did establish

- **The chain runs**, from a place name to a card, with no manual step.
- **Clusters survive.** The card reports intervals over **72 clusters** in the test
  half rather than over 311 photographs, and says so in the text. Without
  `regional_embed` carrying the occurrence key, those intervals would have been
  computed over photographs of the same plant and come out roughly √2 too narrow.
- **Real field photographs work at all.** Every prior number in this project came
  from Pl@ntNet's curated, organ-labelled images. These are raw GBIF occurrence
  photos — whole plants at distance, variable light, several species in frame —
  and closed-set top-1 is 77%.
- **The int8 artifact is in the loop.** This is the shipped encoder, not a torch
  stand-in.

## Two things to fix next

**Near-OOD is still zero.** The card reports `unrelated inputs 777` and no
relatives bucket, because `narrowcast.build.load_rows` always sets
`near_ood = 0` on the embeddings path — only `--scores` buckets by group. For a
regional product where most photographed plants are unlisted relatives, that is
the bucket that matters most, and it is currently unmeasurable through this route.

**A mixed-embedding trap, hit and worth recording.** The first audit paired
regional vectors from the Core ML int8 artifact with a background pool embedded in
*torch*. Nothing errors — the two spaces are cosine 0.995 apart — and label share
read 46.9% instead of 43.2%. Matched sources are the honest comparison; the
mismatch flattered the result by about three points. A bundle should record which
artifact produced its vectors, which is the encoder-binding gap the plan already
names.
