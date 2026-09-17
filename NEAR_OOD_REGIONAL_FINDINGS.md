# The relatives bucket, measured — and a hole in how it is defined

First regional bundle with all three buckets. 24 listed Oregon species, **19
unlisted congeners**, 20 unrelated species, PlantCLEF2024 int8, cluster-disjoint
throughout.

| `p_ood` | coverage | precision | label share |
|---|---|---|---|
| 0.2 | 62.6% | 98.6% | **77.2%** |
| 0.4 | 46.1% | 97.3% | 74.7% |
| 0.6 | 31.7% | 94.1% | 74.7% |

At `p_ood = 0.4`:

| bucket | n | answered | correct when answered |
|---|---|---|---|
| on your list | 162 | 74.7% | **100.0%** |
| **relatives you did not choose** | 313 | **8.9%** | **0.0%** |
| unrelated inputs | 263 | 0.4% | 0.0% |

**8.9% of unlisted relatives get a confident wrong name** — better than the 22.4%
`NEAR_OOD_FINDINGS.md` recorded on the plant catalogue, and still about one
photograph in eleven.

## The model only confuses relatives with relatives

Every breakthrough is within-genus. Nothing crosses a genus boundary:

| unlisted relative | answered as | rate |
|---|---|---|
| *Rubus parviflorus* | *Rubus spectabilis* | 35.0% |
| *Lupinus sellulus* | *Lupinus leucophyllus* | 18.8% |
| *Lupinus lepidus* | *Lupinus leucophyllus* | 12.9% |
| *Rubus leucodermis* | *Rubus laciniatus* | 11.5% |
| ***Lomatium nudicaule*** | ***Lomatium triternatum*** | **11.1%** |
| *Lomatium macrocarpum* | *Lomatium triternatum* | 10.5% |

Eight of nineteen never break through at all. This is the cascade working: the
failures are concentrated, taxonomically sensible, and predictable from the label
set before any photograph is taken.

## The hole: near-OOD is defined taxonomically, danger is not

`narrowcast.build.load_scored` buckets an out-of-list row as `near_ood` when its
**group** appears in-list, and the group here is the genus. That is the right
default and it systematically understates hazard, because the classic poisonings
are **cross-genus within a family**.

Every species in `analysis/safety_pairs.py`'s pre-registered pairs is abundant in
Oregon, and every dangerous one is a different *genus* from *Lomatium*:

| species | Oregon records | genus vs *Lomatium* |
|---|---|---|
| *Conium maculatum* (poison hemlock) | 1,329 | different |
| *Cicuta douglasii* (water hemlock) | 269 | different |
| *Daucus carota* | 3,520 | different |
| *Heracleum maximum* | 3,894 | different |
| *Anthriscus caucalis* | 549 | different |
| *Osmorhiza berteroi* | 864 | different |

So on a forager's list containing *Lomatium*, **poison hemlock buckets as
`distant_ood`** — the bucket this run answers on 0.4% of rows — purely because
*Conium* is not *Lomatium*. The measurement would report it as an easy reject. It
is an umbellifer that has killed people who thought it was one.

The 8.9% above is therefore a floor on the risk, not an estimate of it, wherever
the hazard is a cross-genus look-alike.

**The fix is the group map, and the caller already owns it.** `regional_embed`
writes `group` explicitly rather than leaning on narrowcast's first-token default,
precisely so a region can group by something other than genus. A foraging bundle
should group by **family** — "it is an umbellifer, do not eat it" is the answer
that is both true and useful — and then *Conium* becomes `near_ood` against a
*Lomatium* list and gets measured as the hard case it is.

This is the same defect as the unfixed hazard bug at `narrowcast/build.py:408`,
where `hz_groups = {h.split()[0] for h in hz}` derives hazard groups from the
first whitespace token rather than the supplied map. Both assume genus; both are
wrong for exactly the case the safety gate exists to catch.

## Next

Build the Apiaceae bundle and measure it: a forager's list of *Lomatium
triternatum*, *Lomatium dissectum*, *Daucus carota* and *Osmorhiza berteroi*,
grouped by family, against *Conium maculatum* and *Cicuta douglasii* as near-OOD.
That is the first test of the union hazard gate on real regional photographs, and
every species it needs is already in the survey.
