# 89% of the images a regional bundle would train on are non-commercial

Measured before fetching anything, which is the point: 14 Oregon species sampled
at ≥200 regional records, every image-bearing GBIF occurrence, licence recorded.

| licence | photos | share |
|---|---|---|
| **CC BY-NC** | 322 | **89.0%** |
| CC BY | 32 | 8.8% |
| CC0 | 8 | 2.2% |

**Redistributable share: 11.0%.** And **4 of 14 species had zero** CC BY or CC0
photographs at all.

This is unsurprising in hindsight — GBIF's plant photographs are overwhelmingly
iNaturalist-sourced, and iNaturalist's default upload licence is CC BY-NC — but it
had not been looked at, and it is cheaper to know now than after fetching a
hundred thousand images.

## The distinction that decides how much it matters

**A bundle never ships the images.** The pipeline turns photographs into vectors,
fits a head, and ships **~83 KB of weights** plus a 36.8 MB encoder. No
photograph is redistributed.

So the question is not *"may we redistribute these?"* — we don't — but *"does
training a commercial classifier on NC-licensed images count as commercial use?"*
That is a genuine legal question, contested and jurisdiction-dependent, and it is
not one this repository can settle. It is flagged here because it is a **product**
decision that arrives long before a shipping decision, and because the cost of
getting it wrong scales with how much has already been fetched.

## What the options actually cost

| approach | cost |
|---|---|
| **Use NC images** | free technically; the legal question above is open |
| **CC BY / CC0 only** | loses ~29% of species entirely (4 of 14 had none) |
| **Prefer open, fall back to NC** | best coverage; records exactly which species are affected |

The third is what `--any-licence` plus the per-image `licence` column supports:
every image carries its licence into the manifest, so a bundle can report which of
its labels rest on NC data and a later decision can be made per region rather than
globally.

**And the volume result makes this far less painful than it looks.**
`PHASE0_FINDINGS` measures the adapted encoder as saturated at **two training
images per species**. A species with 200 regional records and an 11% open share
has ~22 redistributable photographs — ten times what the head needs. The binding
constraint is not the 89%, it is the **29% of species with none at all**, and that
number is worth measuring per region before committing to a licence policy.

## Recorded, not decided

`plantid/data/regional_fetch.py` defaults to redistributable-only, so the cautious
path is the default and the permissive one is an explicit flag. The licence of
every fetched image is written to the manifest either way.
