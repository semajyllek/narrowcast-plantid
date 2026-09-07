# A novelty gate halves near-OOD errors, and near-OOD is too small to move the headline

Pre-registered in `NEAR_OOD_PREREG.md`, written before any threshold was fitted.
Two results, and they point in different directions. Both are reported because
reporting either alone would be misleading.

1. **On the bucket it targets, the gate works.** Near-OOD answered-wrong falls
   from 0.224 to 0.124, a 44% reduction, CI [−0.151, −0.054].
2. **On the declared operating point, it is a null.** Deployment-weighted
   expected utility at `p_ood = 0.20` moves +0.017 with CI [−0.004, +0.040].
   Near-OOD is 6.4% of assumed traffic (20% × 0.32), so an effect this size is
   not resolvable — which the pre-registration declared in advance.

## The defect, restated precisely

A near-OOD observation is a species from a catalogue genus that is not itself in
the catalogue. **Every species answer on it is wrong by construction**, so the
0.224 answered-wrong is almost entirely the 0.201 species answers; genus answers
on this bucket are right 87% of the time. The failure is not bad retreat — it is
*absent* retreat on a fifth of the bucket.

## The hypothesis this repo had recorded was wrong

`CLAUDE.md` proposed "a within-genus margin score rather than summed genus
mass," on the theory that a near-OOD sibling spreads posterior mass across the
catalogue members of its genus. Measured before anything was fitted:

| score | AUROC, in-catalogue vs near-OOD |
|---|---|
| `genus_conf` (already in the cascade) | 0.8051 |
| `species_conf` (already in the cascade) | 0.7455 |
| within-genus top-2 margin | 0.5341 |
| **`s / own`** — the proposed fix | **0.4716** |

Worse than chance, and *inverted*: median 0.889 on near-OOD against 0.819
in-catalogue. The reason is structural and is a fact about the catalogue rather
than the encoder — **101 of the 172 catalogue genera hold exactly one species**,
where the ratio is 1 by construction, and near-OOD lands on a singleton genus
38.7% of the time against in-catalogue's 21.2%. Where the assumed mechanism
cannot operate, it reverses the sign of the signal.

Retracted before it was ever fitted.

## What replaced it

Every score in the cascade is derived from the head's posterior, and a
multinomial posterior is **closed-set by construction** — it distributes mass
over the 490 classes it knows and cannot say "none of these". Near-OOD is exactly
where that bites: the observation really does resemble a catalogue genus, so a
closed-set score is *correct* to be confident about the genus, and that
confidence is what carries it past `t_genus`.

**novelty** = max cosine between the observation's mean L2-normalised photo
embedding and the 490 catalogue species centroids, each built from that species'
*training* rows only. AUROC 0.8480 in-catalogue vs near-OOD, 0.9433 vs
distant-OOD; medians 0.935 / 0.854 / 0.824, ordered the way the buckets are,
which the posterior scores are not.

## Results, `bioclip2`, at `p_ood = 0.20`

Three pre-registered arms plus two post-hoc controls, all fitted by the same
expected-utility maximisation on the same clustered calibration split, paired
genus-clustered bootstrap over 2,000 resamples.

| arm | `t_novel` | calib U | test U | **Δ utility** | **Δ near-OOD wrong** |
|---|---|---|---|---|---|
| baseline, refitted | — | 0.6123 | 0.5875 | — | — |
| **retreat gate** | 0.9077 | 0.6264 | **0.6044** | +0.0170 [−0.0039, +0.0400] | **−0.0995 [−0.1512, −0.0536]** |
| reject gate | 0.8678 | 0.6258 | 0.5971 | +0.0095 [−0.0094, +0.0299] | −0.0695 [−0.1075, −0.0348] |
| *control*: gate on `one_minus_other` | 0.9943 | 0.6272 | 0.6006 | +0.0133 [−0.0031, +0.0330] | −0.0901 [−0.1424, −0.0502] |
| *control*: gate on `genus_conf` | 0.6813 | 0.6164 | 0.5777 | **−0.0097** [−0.0192, +0.0014] | −0.0302 [−0.0546, −0.0096] |

Near-OOD shares, which is where the mechanism is visible:

| arm | species | genus | decline | **wrong** |
|---|---|---|---|---|
| baseline | 0.2007 | 0.1806 | 0.6187 | 0.2241 |
| **retreat gate** | **0.0970** | **0.2843** | 0.6187 | **0.1237** |
| reject gate | 0.1405 | 0.1070 | 0.7525 | 0.1538 |

**The retreat gate does exactly what it was designed to do.** Wrong species
answers convert into correct genus answers — species share halves, genus share
rises by the same amount, decline share is untouched. Coverage is therefore
*identical* to baseline at every assumed OOD rate (0.7238 at 20%), and precision
rises 0.9543 → 0.9619. The reject gate reaches a worse near-OOD rate and pays
5.2pp of coverage for it, because declining is the only thing it can do.

The in-catalogue cost is small: species share 0.5207 → 0.5046, −1.6pp, absorbed
into genus answers rather than declines.

### The calibration-to-test gap does not widen

Pre-declared as the overfitting check for a third fitted parameter on ~60
calibration genera. Baseline 0.6123 → 0.5875, gap 0.0248; retreat gate 0.6264 →
0.6044, gap **0.0220**. The extra threshold did not buy calibration performance
it could not keep.

## The control is the most useful thing here

The pre-registration named redundancy as the biggest risk: `novelty` correlates
**+0.72** with `genus_conf`, so the gain might be nothing but a second threshold
on a score the cascade already had. **It is not.** Gating on `genus_conf` in the
identical position is *negative* on utility (−0.0097) and reaches only −0.030 on
near-OOD wrong, against the novelty gate's −0.0995. A score that correlates at
0.72 does a third of the job.

That is the prereg's central argument surviving its own strongest objection: a
closed-set score cannot gate novelty however well it correlates with one that
can.

**But the geometry is not what matters — expressing novelty is.** The second
control, `one_minus_other` = 1 − P(`__OTHER__`), is statistically
indistinguishable from the geometric score (−0.0901 vs −0.0995, overlapping
intervals). It is the reject-class posterior, **already computed on every
observation and never used in the decision** — it appears in `build_observations`
and no threshold has ever been fitted to it. It needs no centroids, no extra
storage and no extra compute.

The one axis separating them is the in-catalogue cost: the novelty gate takes
−1.6pp of species answers, `one_minus_other` takes −5.0pp (0.5207 → 0.4712). If
the gate ships, that is the reason to prefer the geometric score, and it is a
small reason.

## Replication on the deployable encoder

`bioclip2_cml4`, the 160 MB int4 build. Direction replicates, magnitude roughly
halves:

| arm | Δ utility | Δ near-OOD wrong |
|---|---|---|
| retreat gate | +0.0027 [−0.0138, +0.0196] | −0.0491 [−0.0833, −0.0198] |
| *control*: `one_minus_other` | +0.0012 [−0.0117, +0.0141] | −0.0260 [−0.0495, −0.0037] |
| *control*: `genus_conf` | **−0.0152 [−0.0292, −0.0016]** | −0.0301 [−0.0572, −0.0070] |

On int4 the `genus_conf` control's interval **excludes zero on the negative
side** — gating on a closed-set score actively costs utility. The novelty gate's
near-OOD gain survives at 0.204 → 0.154.

## A correction to this document's own first draft

> The primary endpoint was initially computed as an **unweighted** mean utility
> difference, which gave +0.0394 with CI [+0.0125, +0.0715] — an effect
> comfortably excluding zero. That was wrong. Every other headline in this
> project is anchored to a stated prevalence, and the unweighted version reports
> the effect at the evaluation set's incidental ~60% out-of-catalogue rate. The
> gate acts almost entirely on out-of-catalogue rows, so the buckets an honest
> weighting scales *down* are exactly the ones it helps. Re-weighted to the
> declared `p_ood = 0.20`, the same comparison is +0.0170 [−0.0039, +0.0400] — a
> null.
>
> This is the failure mode `CLAUDE.md` already warns about, found in new code
> rather than avoided by the convention. The convention works only when it is
> applied to the *interval* as well as to the point estimate.

## What to conclude

**Do not ship this yet.** The primary endpoint is a null at the declared
operating point and the honest summary is that the gate fixes a real defect in a
bucket too small for that fix to pay for itself in expected utility. What would
change the calculus:

- **A higher assumed near-OOD share.** `NEAR_OOD_SHARE = 0.32` of
  out-of-catalogue traffic is itself an assumption. A user pointing a phone at
  plants in a garden they curated is plausibly in near-OOD far more often than
  32% of the time, and the arithmetic is linear in that share.
- **The in-catalogue side is not the constraint.** Coverage is untouched and
  precision rises, so the gate is close to free on the bucket that dominates the
  weighting. It is not a trade-off being lost; it is an effect being diluted.

**What is settled regardless of power:** the recorded fix was wrong and is
retracted; closed-set scores cannot gate novelty however well they correlate
with a score that can; and the cascade has been computing a usable novelty
signal (`one_minus_other`) since the beginning without ever thresholding it.

## Reproduce

```
PYTHONPATH=. .venv/bin/python -m analysis.near_ood
PYTHONPATH=. .venv/bin/python -m analysis.near_ood --variant bioclip2_cml4 \
    --arms none retreat retreat_omo retreat_genusconf
```
