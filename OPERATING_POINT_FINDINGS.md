# Retreat is a *fraction of headroom*, and the operating point sets the fraction

`HEADROOM_FINDINGS.md` established that headroom governs retreat and offered
`group_share ≈ 1.8 × headroom` as a rule of thumb. `narrowcast-derm` then showed
the rule omits the assumed out-of-catalogue rate: hold headroom fixed, sweep
`p_ood`, and realised retreat moves by 91× while the prediction sits still. That
refutation stood without a replacement, because `P_OOD` was a module constant in
`analysis/headroom_arms.py` and a term that never varies cannot be estimated.

**10,570 rows: 1,406 published arms and 700 newly swept ones, each at five
operating points.** The replacement is simpler than expected.

```
group_share  ≈  headroom × (2.03 − 1.86 × p_ood)
```

Grouped 5-fold CV R² over all domains, folds keyed on the species-set identity:

| model | CV R² | params |
|---|---|---|
| `headroom` (the published rule) | +0.6472 | 1 |
| `headroom + p_ood` | +0.6928 | 2 |
| **`headroom × (a − b·p_ood)`** | **+0.7433** | **2** |
| `headroom + p_ood + interaction` | +0.7442 | 3 |

The two-term multiplicative form captures everything the free three-term model
does, with one fewer parameter and an interpretation: **retreat is a fraction of
the headroom available, and the operating point sets that fraction.**

| `p_ood` | multiplier |
|---|---|
| 0.05 | 1.93 × headroom |
| **0.20** | **1.65 × headroom** |
| 0.60 | 0.91 × headroom |

## This explains the 1.8× rule rather than discarding it

`1.8 ×` is the multiplier at `p_ood ≈ 0.2`, and **every arm behind the published
rule was fitted at exactly `p_ood = 0.2`.** The rule was never wrong; it was one
slice through a function of an assumption nobody had varied. Quote the multiplier
with its operating point and the original number falls out.

It also fixes a structural defect. The additive form `a + b·headroom + c·p_ood`
predicts **negative** retreat at zero headroom and high `p_ood` — 6 of 30
out-of-domain predictions were negative in the previous run. Both terms here are
proportional to headroom, so zero headroom gives zero retreat by construction,
which is also what P1 measures.

## The domain story dissolves into headroom

Fitted separately, the same two-term form is close across domains that share no
modality, encoder or task:

| domain | arms | mean fine | realised / available |
|---|---|---|---|
| dermatology | 2,770 | 0.697 | `1.62 − 2.17 × p_ood` |
| keyword spotting | 780 | 0.842 | `1.73 − 1.97 × p_ood` |
| plants | 7,000 | 0.808 | `2.12 − 1.75 × p_ood` |

The additive coefficient looked domain-dependent — `b(p_ood)` of −0.300 on
dermatology against −0.177 on plants — and that appearance is a headroom effect.
**Binned by fine accuracy, pooled across domains**, it is monotone:

| fine accuracy | rows | domains in band | mean headroom | `b(p_ood)` |
|---|---|---|---|---|
| < 0.65 | 1,070 | derm + plants | 0.228 | −0.370 |
| 0.65–0.75 | 4,355 | derm + plants | 0.171 | −0.327 |
| 0.75–0.82 | 990 | derm + kws + plants | 0.123 | −0.230 |
| 0.82–0.88 | 1,105 | derm + kws + plants | 0.064 | −0.165 |
| > 0.88 | 3,030 | kws + plants | 0.033 | **+0.028** |

Every band holds more than one domain, so this is not domain identity. But
accuracy and headroom fall together down the table, and the control separates
them: **restricted to headroom ∈ [0.08, 0.14]**, dermatology at fine 0.702 gives
`b(p_ood) = −0.2068` and plants at fine 0.802 gives **−0.1996**. At matched
headroom the two are indistinguishable.

So the governing quantity is headroom, as `HEADROOM_FINDINGS` said. Weak models
simply have more of it, which is why the operating point appears to cost them
more — and why dermatology's 91× was never a fact about dermatology.

## Spans, and why the coefficient is the better statistic

| domain | crowded arms | retreat 0.05 → 0.60 | span |
|---|---|---|---|
| plants | 148 | 0.2840 → 0.1788 | 1.6× |
| keyword spotting | 15 | 0.0876 → 0.0277 | 3.2× |
| dermatology | 49 | 0.2048 → 0.0346 | **5.9×** |

The span is a ratio and inflates wherever the denominator is small, which is why
keyword spotting ranks second here on span and *last* on coefficient. Quote the
multiplier, not the span. Dermatology's 91× was a ratio at an operating point
where the denominator had nearly vanished.

## What this is not

- **Dermatology's crowded sets are one family.** Only `inflammatory` (37 usable
  labels at a 10-row floor) can fill a crowded set at K = 20–30, so nearly every
  crowded derm arm is "some inflammatory conditions". Fine for this measurement —
  headroom varies through the grouping sweep within each fit, not through the
  label sets — and **not** a crowded-versus-varied claim about dermatology.
- **Dermatology's clusters average 1.18 rows.** `make_splits` and the bootstrap
  are therefore effectively row-level there, so its intervals are narrower than
  the plant ones for reasons that have nothing to do with better measurement.
  Declared because "cluster, never row" is load-bearing everywhere else here.
- **Keyword spotting never reached the weak band** — `wav2vec2-base` sits at fine
  0.83 even at K = 20. It contributes the accuracy *spread*, not weak arms.
- **18 derm sets and 15 kws sets.** Better than the one crowded arm per domain
  this replaces, and not a large number of independent label sets.

## Reproducing

```
PYTHONPATH=. .venv/bin/python -m analysis.headroom_arms \
    --p-ood 0.05 0.1 0.2 0.4 0.6 --sweep-weak --out data/processed/headroom_full.csv
```

Swept arms carry `arm_source="sweep"` and are excluded from the pre-registered
analysis, which reads `arm_source == "published"` at `p_ood = 0.2` and reproduces
`HEADROOM_FINDINGS.md` exactly — admissibility 20.7%, `M_head` +0.8829.
