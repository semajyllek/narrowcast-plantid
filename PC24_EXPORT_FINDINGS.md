# PlantCLEF2024 converts, but int4 destroys it

`REGIONAL_FRONTIER_FINDINGS.md` moved the encoder recommendation to PlantCLEF2024
on the strength of 0.960 top-1 on real Oregon photographs against MobileCLIP2-S2's
0.830. `CLAUDE.md` listed its Core ML export as open, on the grounds that 518 px is
a very different graph on the Neural Engine. It exports. The quantization does not.

| build | cosine vs fp32 torch | verdict |
|---|---|---|
| **fp16** | **0.99998** (min 0.99994) | conversion is exact |
| **int4, per-grouped-channel** | **−0.005** (min −0.103) | **destroyed** |

A cosine of −0.005 is orthogonal — the embeddings are noise. And fp16 at 0.99998
proves the conversion itself is correct, so this is the palettization, not the
graph, not the 518 px input, and not the preprocessing.

## Why this is not the BioCLIP result

`ONDEVICE_FINDINGS.md` has BioCLIP-1 — also a ViT — surviving the *same*
configuration at cosine 0.932, costing about a point of accuracy. So "ViTs
quantize well" was a reasonable prior and is wrong here.

The plausible difference is **LayerScale**. DINOv2 blocks multiply each residual
branch by a learned per-channel gamma initialised around 1e-5 and typically staying
small. A 16-entry k-means palette fitted over a tensor of such values has nothing
useful to spend centroids on, and a LayerScale vector that quantizes to zero
silently removes the block's contribution entirely. BioCLIP's CLIP-style ViT has no
LayerScale. This is a hypothesis consistent with the observation, not something
measured here — the measurement is only that int4 destroys this model and does not
destroy that one.

Worth noting in passing: this is what the *honest* validator is for. Before the
Phase 0b fix, `validate` compared Core ML against `build_traceable` — the same code
path the export is built from. A model whose weights had been zeroed would still
have matched itself.

## What it costs

The "43 MB" figure quoted for PlantCLEF2024 throughout this project is
**int4 arithmetic** — 86.6M parameters × 4 bits. With int4 unavailable the real
options are int8 at ~87 MB or fp16 at ~173 MB, so **43 MB was never on the table
for this encoder** and the size ladder has to be redrawn.

int8 is running. If it holds, the regional frontier becomes roughly:

| encoder | build | MB | Oregon top-1 (fp32) |
|---|---|---|---|
| MobileCLIP2-S2 | int8 | 36.8 | 0.830 |
| **BioCLIP-1** | **int4** | **46** | **0.931** |
| PlantCLEF2024 | int8 | ~87 | 0.960 |
| BioCLIP-2 | int4 | 153 | 0.984 |

**BioCLIP-1 at 46 MB is suddenly the interesting row.** Its int4 artifact already
exists, it is already known to survive that quantization at cosine 0.932, and it
scored 0.931 on the same Oregon photographs — 10 points above the 17.9 MB encoder
for 9 MB more than the S2 int8 build. It has not been measured through its own
Core ML artifact on regional data, which is now the cheapest open question here.
