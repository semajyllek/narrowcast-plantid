# The operating-point term is real, small, and does not transfer

`HEADROOM_FINDINGS.md` established that headroom governs retreat (CV R² 0.883,
1,409 arms) and then recorded that the `1.8 × headroom` rule omits the assumed
out-of-catalogue rate — measured in `narrowcast-derm` at **91×** movement in
realised retreat while the prediction sat still. That refutation had no
replacement, because `P_OOD` was a module constant in `analysis/headroom_arms.py`
and a term that never varies cannot be estimated.

It is an axis now: **1,406 arms × 5 operating points**, 7,030 rows.

```
PYTHONPATH=. .venv/bin/python -m analysis.headroom_arms \
    --p-ood 0.05 0.1 0.2 0.4 0.6 --out data/processed/headroom_ood.csv
```

## The design holds

Headroom is computed on the calibration half and no deployment weight enters it,
so it must not move with `p_ood`. Within-arm spread across all 1,406 arms is
**exactly 0.00e+00**. The two predictors are identified.

## On plants, the term is real and small

| `p_ood` | `t_group` | group | label | decline | `1.8 × headroom` |
|---|---|---|---|---|---|
| 0.05 | 0.6985 | **0.2530** | 0.5431 | 0.2040 | 0.2162 |
| 0.20 | 0.8303 | 0.2184 | 0.4880 | 0.2936 | 0.2162 |
| 0.60 | 0.9454 | **0.1526** | 0.2611 | 0.5863 | 0.2162 |

Headroom pinned at 0.1201, so the prediction is pinned at 0.2162. Realised
retreat moves **1.7×**, against dermatology's reported 91×.

| model | CV R² |
|---|---|
| headroom alone (the published rule) | +0.7583 |
| `p_ood` alone | +0.0239 |
| headroom + `p_ood` | +0.7856 |
| **headroom + `p_ood` + interaction** | **+0.8335** |

`group_share ≈ 0.0599 + 1.6448 × headroom − 0.1763 × p_ood`, both coefficients
with cluster-bootstrap intervals excluding zero. **The term buys +0.027 of CV R²
additive, +0.075 with an interaction.** `p_ood` alone explains almost nothing:
the operating point modulates retreat, it does not drive it.

The `1.8×` "floor" is recoverable at `p_ood = 0.20` (median ratio 1.789) and
nowhere else — 2.141 at 0.05, 1.026 at 0.60. It is an operating-point-specific
value, not a floor.

## Out of domain, the rule fails — and not the way predicted

Fitted on plants (7,000 rows), tested on held-out weak domains. Never pooled:
three crowded out-of-domain arms against 1,400 plant ones would be a rounding
error in a joint fit and would come back looking like confirmation.

| arm | headroom | measured @ 0.05 → 0.60 | residual range |
|---|---|---|---|
| `derm-crowded` | 0.128 | 0.1171 → 0.0125 | **−0.144 to −0.195** |
| `kws-sem-crowded` | 0.047 | 0.0674 → 0.0000 | −0.031 to −0.075 |
| `text-crowded` | 0.181 | 0.3973 → 0.1731 | +0.049 to −0.079 |

**The prediction I declared was wrong.** I expected the rule to hold at low
`p_ood` and degrade as it rose, because that is where the omitted term bites.
MAE by operating point is **flat**: 0.068, 0.071, 0.061, 0.052, 0.066. The rule
is *uniformly* wrong on weak domains, not progressively wrong. Whatever it is
missing is not something `p_ood` indexes.

What it over-predicts is the **level** of retreat. On dermatology the plant rule
says 0.16–0.26 and the truth is 0.01–0.12, at every operating point.

### What is actually different is the span

| domain | crowded arms | retreat at 0.05 → 0.60 | span |
|---|---|---|---|
| plants | 148 | 0.2840 → 0.1788 | **1.6×** |
| text | 1 | 0.3973 → 0.1731 | 2.3× |
| dermatology | 1 | 0.1171 → 0.0125 | **9.4×** |
| audio (kws) | 1 | 0.0674 → 0.0000 | **to zero** |

This is the finding. Sensitivity to the operating point is **far larger in weak
domains** — the same result `narrowcast-derm` reported as 91×, reproduced here at
9.4× on a differently-constructed arm — and an additive linear term cannot express
it. Adding the interaction helps and does not rescue it: out-of-domain MAE on
crowded arms goes 0.0923 → 0.0868.

`CLAUDE.md`'s reading is supported: the divergence tracks **encoder strength**,
not dermatology. The derm arm sits at fine 0.796 and kws at 0.895 against plants'
0.84–0.97, and it is the weak arms whose retreat collapses.

### The linear form is also structurally wrong

At headroom 0 it predicts `+0.0514` retreat at `p_ood = 0.05` and **`−0.0457`** at
0.60. Six of thirty out-of-domain predictions are negative. Group share is
bounded below by zero and is exactly zero on 3.5% of plant arms; a linear model
on a zero-inflated bounded outcome will do this. Any published two-variable rule
needs a form that cannot predict negative retreat.

## What this does and does not license

**Does.** The operating-point term exists, its sign is negative, its size on
plants is small, and `1.8 ×` is quotable only at `p_ood = 0.20`.

**Does not.** A general two-variable rule. It rests on **three** crowded
out-of-domain arms — one per domain — which is suggestive and nowhere near
established. The plant-fitted rule demonstrably does not transfer, so the honest
statement is *"here is the term on plants, and here is evidence it is much larger
where the model is weak"*, not *"here is the corrected rule"*.

**The missing measurement is now specific**: many weak arms, not one. Sweeping
label sets within dermatology and keyword spotting the way `plant_arms` does
would give the hundreds of low-accuracy arms needed to fit a rule that covers the
range. That is a script, not a new corpus — `narrowcast-derm` has 2,688 images at
three encoders on disk.

## Traps fixed on the way

- **`--sets-per-cell` defaulted to 3; the published run used 4.** Re-running as
  documented reproduced exactly 75% of the published arms in every cell. Fixed,
  and this run's 1,400 plant arms match the published count.
- **The cross-domain inputs lived in `/tmp`.** They resolve against
  `data/processed/arm_inputs/` first now. kws semantic vectors regenerated from
  `data/speech_commands`; dermatology vectors were on disk in `narrowcast-derm`
  all along.
- **`kws-ac-*` are not reproducible and are not reconstructed.** No script writes
  them; the logic is k-means over word centroids with `n_groups` free, and
  headroom moves with that choice. They skip loudly instead.
- **A CSV checkpoint now lands after the plant phase**, which is hours, before the
  out-of-domain phase, which is seconds.
