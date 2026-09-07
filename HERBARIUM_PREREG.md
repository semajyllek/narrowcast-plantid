# Pre-registration — the Tier 1 acquisition-shift probe

Written before any herbarium image was fetched. The feasibility checks that
preceded it counted GBIF records and nothing else.

## Why this source, after two others were ruled out

`DATA_STRATEGY.md` has listed a clean acquisition test as Tier 1 since the start,
and two candidates died this week:

- **Self-collected field photographs** — retracted in place. The row scored image
  provenance and forgot *label* provenance: photographing a plant does not
  identify it, and the available identifier is the corpus the set was supposed to
  be independent of.
- **Herbarium 2022 (FGVC9)** — ~100 GB as a bulk download, against ~80 GB free.

**Herbarium specimens through the GBIF occurrence API solve both.** The
determination is institutional — a botanist's identification on the sheet, not a
crowd vote — and the API is selective, so it costs a few GB rather than a hundred.

Measured before writing this, over 60 randomly sampled catalogue species:
**56 of 60 have ≥ 12 specimen images**, 54 have ≥ 30, median 1,986.

## What this tests

The shipped head is fitted on **Pl@ntNet-300K photographs**.
`DOMAIN_SHIFT_FINDINGS.md` scored it on held-out Pl@ntNet (0.7633) and on
iNaturalist (0.7601) and found the source change free at ViT-L. Both are
photographs of living plants taken by people in the field.

A herbarium sheet is not: pressed, dried, flattened, colour-shifted, mounted, and
photographed on a copy stand with a label, a scale bar and often a colour chart in
frame. It is the largest acquisition shift reachable without new fieldwork.

**Design.** Fetch up to `N` specimen images per catalogue species, score the
**production organ-routed head** on them photo-level, macro-averaged over
species, species-clustered bootstrap — the same estimator as the product arm of
`DOMAIN_SHIFT_FINDINGS.md`, so the three numbers sit in one table.

**Encoders**: `bioclip2`, `bioclip2_cml4`, `bioclip1_cml4`, `mobileclip2_s2`,
`plantclef24` — the same set the source-shift work used.

## Predictions, declared

1. **Everything drops hard.** BioCLIP-2 species top-1 below 0.40, against 0.76 on
   both photograph corpora. A pressed specimen is a different object.
2. **Genus survives far better than species**, because a sheet preserves gross
   morphology — leaf arrangement, inflorescence structure — while losing the
   colour and texture that separate congeners.
3. **The sharp one: BioCLIP-2's advantage over MobileCLIP2-S2 will *shrink*
   relative to the ~16pp it holds on iNaturalist.** The source-shift finding was
   that shift is an encoder-scale phenomenon — free at ViT-L, −18pp at 17.9 MB.
   That was measured across two corpora of field photographs. Under a shift this
   large I expect the gap to compress, because herbarium sheets are out of
   distribution for all of them.

   **If the gap holds or widens, encoder scale protects even under extreme
   acquisition shift** — a stronger claim than anything currently in this repo,
   and one that would change the size decision.

## Ways this comes out uninformative, declared in advance

1. **This is not contamination-free.** Herbarium sheets are in GBIF, so they are
   plausibly inside BioCLIP-2's training data exactly as iNaturalist is. It is a
   **modality** probe, not the clean test `DATA_STRATEGY.md` asks for, and it does
   not close Tier 1. Contamination would make the measured drop an *underestimate*.
2. **The sheet contains furniture.** Labels, barcodes, scale bars and colour
   charts are in frame and are not the plant. If accuracy is low it will not be
   separable here whether the encoder failed on the specimen or was distracted by
   the stationery, and a cropped arm would be the follow-up rather than a result.
3. **Name matching.** GBIF `scientificName` search can admit synonyms and
   misapplied names. The returned `species` field is checked against the query and
   mismatches dropped, but a taxonomic generation of drift will survive that, as
   `COMPETITIVE_FINDINGS.md` documents for the catalogue generally.
4. **Organ routing is undefined here.** The production head routes leaf/bark/
   flower; a sheet is a whole pressed plant and is none of them. Whatever the
   router does with it is part of what is being measured, and the routed-organ
   distribution will be reported rather than assumed.
