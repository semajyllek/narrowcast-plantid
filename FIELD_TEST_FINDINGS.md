# A phone camera in one's own hands: the last unmeasured axis, measured

Dabney State Park, Oregon, September 2026. **45 photographs of 15 species**,
taken on an iPad by the person building the app, identified in the field against
a printed guide by two people, and then run through both this project's model and
iNaturalist's server model.

Every number in four repositories came from GBIF or iNaturalist photographs.
`DOMAIN_SHIFT_FINDINGS.md` calls a different camera in different hands the last
untested axis and `DATA_STRATEGY.md` proposes herbarium sheets as a proxy for it.
This is the real thing instead, and it cost an afternoon.

## Headline

On the 12 of 14 species the model was *able* to name — two had been dropped by a
fetch bug this test uncovered, see below — three photographs of each, averaged:

| | top-1 | top-5 |
|---|---|---|
| **ours, 87 MB, offline, 157 ms** | **75%** | **100%** |
| iNaturalist, vision only | 67% | 83% |
| iNaturalist, vision + coordinates | **92%** | 92% |

**n = 12 species. One species is 8 points.** Read the ordering, not the gap: what
this supports is "comparable to iNaturalist's vision model", not "better than".

## Three things it established

**1. The domain holds.** Photographs from a different camera, in different hands,
of plants growing where they grow — not the curated GBIF material every previous
number rests on — and accuracy did not collapse. That was the open risk.

**2. Multiple photographs replicate on real phone photos.** `MULTIPHOTO_FINDINGS.md`
measured +5.9 points on GBIF material and warned the magnitude on a phone was not
in doubt only in direction. Here, over the same 45 photographs:

| | top-1 | top-5 |
|---|---|---|
| one photograph | 48.9% | 75.6% |
| three averaged | **64.3%** | 85.7% |

**+15.4 points**, larger than the GBIF estimate rather than smaller. The
photographs a person takes of one plant differ more than a contributor's do.

**3. Location is worth more than everything else on the table.** iNaturalist with
coordinates goes 67% → 92% on these photographs, **+25 points**, and per
photograph 57.8% → 93.3%. `LOCATION_FINDINGS.md` recorded a geographic prior as
worth ~0 at 490 regional species and +4.5pp to iNaturalist's 108k-taxa model; on
real field photographs at a known point it is far larger than either.

This is not a like-for-like comparison — our model has no location — and that is
the finding. **A coarse regional catalogue is not the same as knowing where you
are standing.** Ours restricts to 901 Oregon species; iNaturalist's geomodel
weights 108,000 taxa by this coordinate and this week. The second is much more
information, and a phone has it for free.

## The bug it found, which is the most useful outcome

Two of the fifteen species — *Verbascum thapsus* (2,219 Oregon observations) and
*Impatiens capensis* (1,129) — were **not in the model at all**. Both had been
dropped for having fewer than four distinct plants in the bank.

`regional_fetch.occurrences` requested one page of GBIF records and kept whatever
survived the licence filter. GBIF returns a page in no useful order, so for a
common species the first sixty records are often one dataset under one licence;
where that licence was non-redistributable, almost nothing survived. The effect
ran backwards and silently:

| species | Oregon observations | distinct plants fetched |
|---|---|---|
| *Polystichum munitum* (sword fern) | 7,652 | **1** |
| *Gaultheria shallon* (salal) | 6,511 | **2** |

Correlation across the region between how often a plant is photographed and how
many plants were fetched: **−0.171**. The bank was thinnest exactly where a user
is most likely to point a phone, and 75 species were excluded from the model
outright. Fixed by paging until enough *distinct usable* occurrences are found;
sword fern and salal both go from 1 and 2 to 20.

**No amount of held-out evaluation would have found this.** Every split, bootstrap
and card was computed over the species that survived the fetch, so the missing
ones were invisible by construction. It took someone photographing a mullein.

## Where our errors were

*Thuja plicata* → *Chamaecyparis lawsoniana* at 76% (western red cedar against
Port Orford cedar), *Rubus parviflorus* → *Ribes bracteosum* (thimbleberry against
stink currant, both large palmate leaves), *Aesculus hippocastanum* →
*Notholithocarpus densiflorus*. The first two are confusions a person makes; the
third is not obviously one. All three had the right answer in the top five, which
is the argument for showing it.

## Limits

- **14 species, 45 photographs, one park, one afternoon, one camera.** Every
  number here has an interval far wider than the differences between the rows.
- The photographs were taken by someone who knew the model existed, which is not
  a neutral sample of what a user would point it at.
- iNaturalist was given 1024px downscales: their endpoint returns
  `500 Error scoring image` on the 6.2 MB originals.
- Ours ran through the macOS CoreGraphics path, measured 0.8 points *below* the
  device (`PREPROCESS_FINDINGS.md`), so the app's own numbers are slightly better
  than these.
