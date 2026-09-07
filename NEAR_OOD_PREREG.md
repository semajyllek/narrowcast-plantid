# Pre-registration — can near-OOD be fixed by a novelty score?

Written **before** any threshold in the design was fitted. The feasibility checks
that preceded it are declared in full below; every one of them touched
*predictors* only — AUROC and correlation between scores — and none touched
utility, coverage, precision or the answer shares.

## The defect

`near_ood` is the weakest bucket in the system and has been through three
successive expansions of the evaluation set. On 299 test observations
(`REJECTION_FINDINGS.md`):

| bucket | n | species | genus | decline | answered wrong |
|---|---|---|---|---|---|
| in-catalogue | 1,736 | 0.521 | 0.348 | 0.131 | 0.018 |
| **near-OOD** | **299** | **0.201** | 0.181 | 0.619 | **0.224** |
| global-OOD | 364 | 0.019 | 0.014 | 0.967 | 0.033 |
| regional-OOD | 373 | 0.013 | 0.008 | 0.979 | 0.021 |

**The 0.224 is almost entirely the 0.201.** A near-OOD observation is a species
from a catalogue genus that is *not itself in the catalogue*, so every species
answer is wrong by construction. Genus answers on this bucket are right 87% of
the time. The defect is not that the model retreats badly — it is that it does
not retreat at all on a fifth of these, and names a species that cannot be right.

It cannot be fixed by fetching: only 172 catalogue genera exist and 120 are
already covered. `REJECTION_FINDINGS.md` closes with "fixing near-OOD needs a
modelling change, not more data."

## The standing hypothesis is wrong, and this is how it died

`CLAUDE.md` has recorded the proposed fix as: *"needs a within-genus margin score
rather than summed genus mass."* The mechanism assumed is that a near-OOD sibling
spreads posterior mass across the catalogue members of its genus, so the top
species holds a smaller *share* of its genus's mass than a true catalogue species
would.

Measured, before writing anything else. Let `s` be the top species posterior and
`own` the mass of the genus containing it; the proposed score is `r = s / own`.

| score | AUROC, in-catalogue vs near-OOD |
|---|---|
| `genus_conf` (already in the cascade) | 0.8051 |
| `species_conf` (already in the cascade) | 0.7455 |
| within-genus top-2 margin | 0.5341 |
| **`r = s / own`** — the proposed fix | **0.4716** |

**`r` is worse than chance, and it is inverted:** median 0.889 on near-OOD
against 0.819 in-catalogue. Near-OOD observations hold a *larger* share of their
genus's mass than in-catalogue ones do.

The reason is structural, and it is a fact about the catalogue rather than about
the encoder. **101 of the 172 catalogue genera contain exactly one species.** In
a singleton genus `r ≡ 1` by construction, and near-OOD observations land on a
singleton genus more often than in-catalogue ones do — 38.7% against 21.2%. The
"spread across siblings" mechanism can only operate in the other 71 genera, and
where it cannot operate it actively reverses the signal.

**This hypothesis is retracted before it was ever fitted**, and `CLAUDE.md` is
corrected in place. Two arms of a design that was about to be built would have
tested a score that carries less than no information.

## The replacement, and why it is a different kind of score

Every score in the current cascade is derived from the head's posterior, and a
multinomial posterior is **closed-set by construction**: it distributes mass over
the 490 classes it knows and has no way to express "none of these". Near-OOD is
precisely the case where that matters — the observation genuinely resembles a
catalogue genus, so a closed-set score is *right* to be confident about the
genus, and that confidence is what carries it past `t_genus`.

A **geometric** score can express what the posterior cannot: how far the
observation sits from anything in the catalogue.

> **novelty** = the maximum cosine similarity between the observation's mean
> L2-normalised photo embedding and the 490 catalogue species centroids, each
> centroid the mean of that species' **training** rows — the same rows the head
> was fitted on, so no test information enters it.

Measured, predictors only:

| score | AUROC in-cat vs near-OOD | AUROC in-cat vs distant-OOD |
|---|---|---|
| `species_conf` | 0.7455 | — |
| `genus_conf` | 0.8051 | — |
| **`novelty`** | **0.8480** | 0.9433 |

Medians: 0.935 in-catalogue, 0.854 near-OOD, 0.824 distant-OOD — ordered the way
the buckets are, which the posterior scores are not.

## Design

Three arms, all fitted by the **same** expected-utility maximisation on the
**same** clustered calibration split at the standard `p_ood = 0.20`, with the
`UTILITY` constants unchanged, and evaluated on the same test split.

- **Baseline** — the shipped two-threshold cascade, refitted, so the comparison
  is paired on identical splits.
- **Arm A, reject gate** — `decline` if `genus_conf < t_genus` **or**
  `novelty < t_novel`.
- **Arm B, retreat gate** — answer `genus` if `species_conf < t_species` **or**
  `novelty < t_novel`.

**Both placements are declared in advance because the utility function does not
obviously prefer one.** For an out-of-catalogue observation the declared payoffs
are `decline_ood = 1.0` against `genus_correct = 0.5`, so declining a near-OOD
scores *higher* than answering its genus correctly — which favours Arm A. But the
same gate in Arm B costs nothing on in-catalogue rows that are merely unusual,
because retreating to genus keeps them answerable. Choosing between them after
seeing the results is exactly the move this file exists to prevent, so both are
fitted and both are reported.

**Primary endpoint.** Paired difference in mean test utility against the
baseline, per arm, at `p_ood = 0.20`, with a **genus-clustered** bootstrap over
2,000 resamples — genus rather than species for the near-OOD bucket, per the
existing `SPLIT_CLUSTER` convention, because the decision being fitted is
genus-level.

**Secondary, all pre-specified.** Near-OOD answered-wrong rate; near-OOD
species/genus/decline shares; in-catalogue species-level share, which is what a
reject gate would be expected to cost; coverage and precision at `p_ood ∈
{0.6, 0.4, 0.2, 0.1}`.

## Ways this comes out uninformative, declared in advance

1. **Redundancy is the biggest risk.** `novelty` correlates **+0.72** with
   `genus_conf` over the in-catalogue and near-OOD rows pooled (+0.62 within
   in-catalogue, +0.72 within near-OOD). Its AUROC advantage — 0.848 against
   0.805 — may be entirely subsumed by a score the cascade already thresholds.
   If so the fit will drive `t_novel` to the floor and reproduce the baseline
   exactly. **That is a null, it will be reported as one, and it is a
   substantive answer**: it would say the posterior already carries what the
   geometry knows, and that near-OOD is hard for a reason no rescoring reaches.
2. **A third fitted parameter on a thin calibration set.** Near-OOD has 120
   genera, roughly 60 of them in calibration. Three thresholds fitted there can
   overfit. Calibration *and* test utility are reported for every arm, and the
   gap between them is the overfitting estimate rather than something to explain
   away afterwards.
3. **The centroids pool organs.** The catalogue is organ-partitioned and the head
   is organ-routed; this score is not. If the arms fail, an organ-matched
   centroid is the obvious follow-up and its absence here is a limitation of this
   design, not evidence against geometric scoring.
4. **The bucket cannot be grown.** 172 catalogue genera exist and 120 are
   covered, so a wide interval stays wide. An effect smaller than roughly 0.1
   utility will not be resolvable, and the design is not powered to find one.

## What a win would and would not mean

A win is a reduction in near-OOD answered-wrong at no material cost to the
in-catalogue species share. It would **not** establish that geometric scoring
belongs in `narrowcast`: the tool is domain-general and a catalogue whose genera
are mostly singletons is a property of *this* catalogue. The structural fact that
killed the ratio hypothesis — 101 of 172 genera hold one species — is a fact
about a corpus selected by image availability, and a differently-built catalogue
would have a different one.
