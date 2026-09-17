# On real regional photographs the tiny encoder does not hold up — 43 MB does

`TINY_K_FINDINGS.md` concluded that at K=20 the 17.9 MB encoder is the *best* of
four, beating BioCLIP-2 at 152 MB. That was measured on **Pl@ntNet catalogue
images** of species **inside** the adapted tower's fine-tune. Neither condition
holds for a regional bundle, and when both are dropped the ordering reverses.

Same 618 GBIF field photographs of 24 Oregon species, same cluster-disjoint splits,
same head, three seeds; only the encoder varies.

| encoder | MB | top-1 | sd |
|---|---|---|---|
| MobileCLIP2-S2 | 17.9 | 0.830 | 0.025 |
| MobileCLIP2-S2 adapted | 17.9 | 0.839 | 0.023 |
| **PlantCLEF2024** | **43** | **0.960** | **0.005** |
| BioCLIP-1 | 46 | 0.931 | 0.008 |
| BioCLIP-2 | 152 | 0.984 | 0.001 |

**PlantCLEF2024 at 43 MB recovers 12 points over 17.9 MB and sits 2.4 points under
an encoder 3.5× its size.** It is also far more *stable*: sd 0.005 against the
small encoder's 0.025.

This is what `CLAUDE.md` already says and what the regional data now confirms
independently: *"the middle ground between 17.9 MB and 152 MB is not a compressed
BioCLIP-2 — it is `plantclef24`, an off-the-shelf ViT-B at 43 MB."*

## Why the Phase 0 result did not transfer

**Only 2 of the 24 Oregon species are inside Pl@ntNet-300K's 1,081.** The adapted
tower is therefore in its out-of-fine-tune regime, where `ADAPT_FINDINGS.md`
measures it at −0.0353 against stock — and indeed it buys +0.009 here, not the
+0.10 it buys inside its own label set. PlantCLEF2024 was fine-tuned on **7,806**
Pl@ntNet species, which is why it generalises to an Oregon flora it was never
shown.

Two conditions, then, and a regional bundle satisfies neither:

| Phase 0 measured | a real region |
|---|---|
| Pl@ntNet catalogue images, curated and organ-labelled | raw field photographs |
| species inside the 1,081-species fine-tune | 2 of 24 inside it |

## The failures are gross, not subtle

Worst species for the 17.9 MB encoder, with BioCLIP-2 on the identical rows:

| species | S2 | BioCLIP-2 | S2 mostly says |
|---|---|---|---|
| *Cornus sericea* | 0.40 | **1.00** | *Rubus spectabilis* |
| *Saponaria officinalis* | 0.53 | 0.87 | *Prunella vulgaris* |
| *Lonicera involucrata* | 0.59 | 0.94 | *Lotus corniculatus* |
| *Lomatium triternatum* | 0.69 | **1.00** | *Poa bulbosa* — a grass |
| *Prunella vulgaris* | 0.67 | **1.00** | *Ranunculus glaberrimus* |

**Six of 24 species fall below 0.70 on the small encoder; none do on BioCLIP-2.**
These are cross-family errors, not hard congener calls — a dogwood read as a
salmonberry, a desert parsley as a grass. The representation is not there.

(*Fragaria virginiana* → *Fragaria vesca* at 0.60 is the one genuine congener
confusion, and BioCLIP-2 still gets it at 1.00.)

## What this changes

**The encoder recommendation moves from 17.9 MB to 43 MB.** "Tiny and identifies
really well" survives, but at 43 MB rather than 17.9, and the earlier number was
optimistic because it was measured on the easy half of both axes.

**And the safety consequence is direct.** *Lomatium triternatum* — a genus that
appears in four of the eleven pre-registered hazard pairs, as the plant poison
hemlock gets mistaken for — is read as a grass 31% of the time by the small
encoder and never by BioCLIP-2. `OREGON_SAFETY_FINDINGS.md` already recorded that
MobileCLIP2-S2 fails the 1% hazard bar at every operating point; this is the same
verdict from a different direction, on real photographs.

## The gate this opens

`plantclef24` **has never been exported to Core ML**, and `CLAUDE.md` lists it as
open precisely because 518 px is a very different graph on the Neural Engine than
224 px. It also costs 38.6 ms/image on MPS against S2's 5.2.

So the Phase 0 gate has to run again for this encoder: does it convert, what does
it weigh as int8, and what is its ANE latency. The int8 machinery, the
preprocessing fix and the validator are all in place now, so it is a re-run rather
than new work — but the product's encoder is not settled until it passes.
