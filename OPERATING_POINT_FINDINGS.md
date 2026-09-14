# The operating-point term is real, small on plants, and not 91×

`HEADROOM_FINDINGS.md` established that headroom governs retreat (CV R² 0.883,
1,409 arms) and then recorded that the `1.8 × headroom` rule of thumb omits the
assumed out-of-catalogue rate — measured in `narrowcast-derm` at **91×** movement
in realised retreat while the prediction sat still. That refutation had no
replacement, because `P_OOD` was a module constant in `analysis/headroom_arms.py`
and a term that never varies cannot be estimated.

It is an axis now. `--p-ood 0.05 0.1 0.2 0.4 0.6`, every arm scored at each.
Reproduce with:

```
PYTHONPATH=. .venv/bin/python -m analysis.headroom_arms \
    --p-ood 0.05 0.1 0.2 0.4 0.6 --out data/processed/headroom_ood.csv
```

## The design holds

Headroom is computed on the calibration half from `species_ok`/`genus_ok` and no
deployment weight enters it, so it must not move with `p_ood`. Across all 1,052
arms the within-arm spread is **exactly 0.00e+00**. The two predictors are
identified; everything below rests on that.

## Result

| `p_ood` | `t_group` | group | label | decline | `1.8 × headroom` |
|---|---|---|---|---|---|
| 0.05 | 0.7000 | **0.2524** | 0.5417 | 0.2059 | 0.2191 |
| 0.10 | 0.7624 | 0.2382 | 0.5243 | 0.2375 | 0.2191 |
| 0.20 | 0.8316 | 0.2192 | 0.4845 | 0.2963 | 0.2191 |
| 0.40 | 0.9044 | 0.1879 | 0.3880 | 0.4240 | 0.2191 |
| 0.60 | 0.9458 | **0.1574** | 0.2551 | 0.5875 | 0.2191 |

Headroom is pinned at 0.1217 in every row, so the prediction is pinned at 0.2191.

**Realised retreat moves by 2×, not 91×.** The direction matches dermatology —
`t_group` climbs with `p_ood` and squeezes the band a group answer must land in,
`corr(p_ood, t_group) = +0.783` — but the magnitude does not come close.

### The two-variable rule

Grouped 5-fold CV R² on group-answer share, folds keyed on the species-set
identity so an arm and all five of its operating points stay on one side:

| model | CV R² |
|---|---|
| headroom alone (the published rule) | +0.7619 |
| `p_ood` alone | +0.0127 |
| headroom + `p_ood` | +0.7858 |
| **headroom + `p_ood` + interaction** | **+0.8244** |
| headroom + `t_group` | +0.7844 |

`group_share ≈ 0.0566 + 1.6417 × headroom − 0.1682 × p_ood`, both coefficients
with cluster-bootstrap intervals excluding zero (headroom [+1.4992, +1.8080],
`p_ood` [−0.2540, −0.0691]).

**Adding `p_ood` buys +0.024 of CV R². Adding it with an interaction buys
+0.063.** It is a real term and a small one, and `p_ood` alone explains
essentially nothing — the operating point modulates retreat, it does not drive it.

### The floor is not a floor

| `p_ood` | median `group_share / headroom` |
|---|---|
| 0.05 | 2.105 |
| 0.10 | 1.950 |
| 0.20 | **1.766** |
| 0.40 | 1.446 |
| 0.60 | 1.011 |

`1.8×` is recoverable at `p_ood = 0.20` — the operating point it was fitted at —
and nowhere else. It is an upper-middle value, not a floor.

## What this does *not* settle, and it is the important part

**This run is plants (1,050 arms) plus text (2).** The audio and bird arms were
skipped: `OOD_ARMS` addressed them under `/tmp`, macOS cleared it, and the
re-run produced a plants-plus-text table that would have described itself as the
same cross-domain experiment. Fixed — arm inputs now resolve against
`data/processed/arm_inputs/` first and a missing one is loud — but the audio and
bird vectors themselves are gone and need regenerating from `narrowcast-kws`.

**So the low-accuracy regime is absent, which is exactly where derm found 91×.**
`CLAUDE.md` already predicted this and it is now supported rather than asserted:
*"Realisation depends on where the decline threshold lands, which depends on
`p_ood` and on encoder strength — this arm ran at top-1 ≈ 0.75, well below the
0.84–0.97 of the plant arms the rule was fitted on. It is a boundary condition
reachable here too, not a fact about dermatology."*

The interaction term being the biggest single gain (+0.063 against +0.024 for the
additive form) points the same way: how much the operating point costs you
depends on how much retreat was available in the first place. But it is estimated
over a narrow band of model strength and **must not be extrapolated to the derm
case**. A two-variable rule that covers the range where it matters needs arms
from a hard domain in the same sweep.

Quote `2×` as a plant number, `91×` as a weak-encoder number, and neither as
general.

## Two traps this run walked into, both now fixed

- **`--sets-per-cell` defaulted to 3; the published run used 4.** Re-running the
  script as documented reproduces exactly 75% of the published arms in every
  cell — 160 → 120, 80 → 60, uniformly. That looks like data loss and is a flag
  nobody passed. Default is 4.
- **The cross-domain inputs lived in `/tmp`.** See above. The evidence that makes
  `HEADROOM_FINDINGS` a cross-domain result was one reboot from vanishing.
