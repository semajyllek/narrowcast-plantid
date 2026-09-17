# Oregon, surveyed: what a regional bundle is actually up against

First real output of `plantid/data/regions.py`. GBIF, Oregon, plants, georeferenced,
species rank, ≥20 records. Reproduce with:

```
PYTHONPATH=. .venv/bin/python -m plantid.data.regions \
    --source gbif --state Oregon --min-obs 20 --out data/processed/regions/oregon_gbif.json
```

## The pool

| threshold | species |
|---|---|
| ≥ 20 records | **3,893** (1,222 genera) |
| ≥ 50 | 2,786 |
| ≥ 100 | **1,972** |
| ≥ 200 | 1,207 |
| ≥ 500 | 528 |

`CLAUDE.md` records an uncommitted iNaturalist query as *"4,570 research-grade
species, 1,175 with ≥100 observations."* GBIF gives **1,972 at ≥100** — about 68%
more, which is what a superset should do, and the first independent check that
figure has ever had. The totals are not comparable (this run's floor truncates the
long tail at 20), but the ≥100 comparison is like-for-like and the ordering is the
expected one.

The top of the list is a sanity check in itself: *Pinus ponderosa*,
*Trillium ovatum*, *Pseudotsuga menziesii*, *Polystichum munitum*,
*Gaultheria shallon*, *Acer circinatum* — species-for-species what
`analysis/common_oregon_fetch.py`'s author typed by hand as "twenty plants someone
in Portland would actually want named."

## Crowding is not a hypothetical here

**263 genera have four or more usable species.** The worst:

| genus | usable species |
|---|---|
| *Carex* | **109** |
| *Astragalus* | 50 |
| *Penstemon* | 50 |
| ***Lomatium*** | **44** |
| *Trifolium* | 42 |
| *Castilleja* | 42 |
| *Lupinus* | 41 |

This is the measured version of the risk the whole project is built around, and it
is worse than the plant catalogue suggested. A user assembling an Oregon list from
this pool will pick siblings *by default* — 109 sedges are not distinguishable to
a 20-label head, and the crowded arms in `TINY_K_FINDINGS` score 0.25 label share
against 0.94 varied.

**The list-checker is therefore not a nicety.** It runs on names in milliseconds,
before any compute, and it is the difference between the two numbers above.

### And *Lomatium* is the safety case, concretely

Four of the eleven pre-registered hazard pairs in `analysis/safety_pairs.py` are
*Conium maculatum* or *Cicuta douglasii* mistaken for a *Lomatium* — poison hemlock
and water hemlock read as a plant foragers dig and eat. Oregon has **44** usable
*Lomatium* species.

So a foraging bundle for this region is exactly the case where `UTILITY["wrong"]`
being one scalar over all labels bites: the threshold that serves *Carex* is not
the threshold that should govern a hemlock/*Lomatium* confusion. That is the gap
`PLAN` Phase 4 names and does not yet close.

## The fetch bill

**77 of 3,893 Oregon species are in the 490-name catalogue already embedded** — 2%.

A real Oregon bundle is therefore a fresh fetch, not a filter of existing caches,
which is what the plan assumed but had not measured. The good news is what
`PHASE0_FINDINGS` established about volume: the adapted encoder saturates at **two
training images per species**, so a 200-species regional list needs ~400 images,
not ~6,400. Stock S2 needs 16–32, so the same list costs 3,200–6,400 — and which
applies depends on whether the species sit inside Pl@ntNet-300K's 1,081.

That single fact is the difference between a regional bundle being an afternoon's
fetch and a week's.

## What this changes

- **Near-OOD dominates by construction.** A 200-species list leaves 3,693 usable
  Oregon species off it, every one of them something a user can photograph. The
  `--ood-rate` declaration is the central product decision, not a config detail.
- **The list-checker moves up.** It is the cheapest large win available.
- **Carex is a product question, not a modelling one.** No encoder distinguishes
  109 sedges at 20 labels; the answer is to exclude them, or to answer at genus
  deliberately and say so.
