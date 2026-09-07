# Pre-registration — can the small end of the frontier be moved?

Short, because the question is narrow and the point is the measurement.

## The gap in what has been tried

Every encoder experiment in this repo holds the head fixed — one multinomial
logistic probe, `C = 10`, `class_weight="balanced"` — and asks *which frozen
features*. Distillation, pruning, `bioclip_inat`, earlier layers, cropping: all
of them vary the encoder.

Nothing has varied the other half. The product question is **the best accuracy
achievable at a byte budget**, and inference-side levers that cost few or no
bytes have never been measured here.

## Arms

Scored through the production pipeline on the standard 5,534-observation
iNaturalist evaluation set, observation-level, so the numbers are directly
comparable to the table in `CLAUDE.md`.

| arm | bytes | costs |
|---|---|---|
| `mobileclip2_s0` | 5.7 MB | reference floor |
| **`mobileclip2_s2`** | **17.9 MB** | **the baseline to beat** |
| `s0 ⊕ s2`, posteriors averaged | 23.6 MB | two forward passes |
| `s2`, `C` swept over {1, 10, 100} | 17.9 MB | nothing |
| `plantclef24` | 43 MB | 2× latency |
| `bioclip2_cml4` | 160 MB | the ceiling |

`C` has been fixed at 10 since the beginning and never swept; that is free to
check and is included so that any ensemble gain is not just a regularisation gain
in disguise.

**Primary endpoint**: species and genus top-1 at the observation level.
**Secondary**: coverage at the declared 20% out-of-catalogue rate.

## Predictions

1. **The `C` sweep buys under 1pp.** If it buys more, several published numbers
   were left on the table by an unexamined default and that is worth knowing.
2. **The ensemble buys 1–3pp of species over `s2` alone** and does not close the
   gap to `plantclef24`. Two weak encoders trained on similar web data should
   correlate too much to ensemble well.
3. **Neither reaches `plantclef24` at 43 MB.** If `s0 ⊕ s2` at 23.6 MB matches
   43 MB, the byte/accuracy frontier has a point on it that nobody has been
   offering, and that is a product result rather than a characterisation.

## What would make this uninformative

The evaluation set is fixed and has been reported on many times, so this is not
an independent test of the encoders — it is a comparison of inference strategies
on a set whose composition is already baked into every number in `CLAUDE.md`. A
gain here is a gain on that set, and the source-shift work says small encoders
lose most to a change of source. Any winner should be re-checked cross-source
before it is believed.
