# The encoder, decided on real regional data

Every earlier encoder comparison in this project used Pl@ntNet catalogue images.
This one uses the artifacts that would actually ship, on 618 GBIF field
photographs of 24 Oregon species, cluster-disjoint splits, three seeds.

| artifact | MB | Oregon top-1 | sd | **usable latency** | backend |
|---|---|---|---|---|---|
| MobileCLIP2-S2 int8 | 36.8 | 0.828 | 0.030 | **2.8 ms** | any |
| **BioCLIP-1 int4** | **46.2** | **0.922** | 0.013 | **19.3 ms** | **ANE only** |
| **PlantCLEF2024 int8** | **87.0** | **0.962** | 0.002 | **36.4 ms** | any |
| BioCLIP-2 int4 | 160.0 | 0.982 | 0.001 | 121.2 ms | ANE only |

## "Usable latency" is doing real work in that table

`ONDEVICE_FINDINGS.md` records that int4 per-grouped-channel is silently wrong on
the Metal GPU backend. Re-measured here on the same 16 photographs:

| artifact | ANE | GPU |
|---|---|---|
| BioCLIP-1 int4 | 0.9333 | **0.1668** |
| BioCLIP-2 int4 | 0.9797 | **0.6158** |
| PlantCLEF2024 int8 | 0.9996 | **0.9996** |

Both int4 builds return a plausible unit-norm embedding that is **garbage** on GPU.
So their fastest measured numbers — BioCLIP-1 at 7.4 ms, BioCLIP-2 at 25.5 ms — are
unusable, and the honest figures are the ANE ones, 19.3 and 121.2.

**int8 is free of the pin.** Linear per-channel quantization is correct on every
backend, for the same reason it was for MobileCLIP2-S2. That is not a small
operational difference: an int4 build requires the app to pin `computeUnits` to
`.cpuAndNeuralEngine` forever, and `.all` is a request rather than a guarantee.

## Consequently PlantCLEF2024 int8 dominates BioCLIP-2 int4

| | PlantCLEF2024 int8 | BioCLIP-2 int4 |
|---|---|---|
| size | **87 MB** | 160 MB |
| usable latency | **36.4 ms** | 121.2 ms |
| backend pin | **none** | ANE only |
| Oregon top-1 | 0.962 | 0.982 |

Half the size, **3.3× faster**, no deployment constraint, two points of accuracy
behind. BioCLIP-2 is no longer the top of a frontier; it is off it on three axes
out of four.

## The recommendation

**BioCLIP-1 int4 at 46 MB** if the download budget is the binding constraint. It is
the knee of the curve — 9 MB and 16 ms buy **9.4 points** over the 17.9 MB encoder
— and the artifact already existed. Cost: an ANE pin forever.

**PlantCLEF2024 int8 at 87 MB** otherwise, and this is the one to ship. It buys
four more points, is five times more stable across splits (sd 0.002 against 0.013),
carries no backend pin, and its extra 41 MB is the cheapest accuracy left on the
table. At 36.4 ms a guided four-photograph capture costs ~146 ms.

**Not MobileCLIP2-S2.** 0.828 on real regional photographs, with six of 24 species
below 0.70 and cross-family errors — a dogwood read as a salmonberry, a *Lomatium*
read as a grass. The last of those is a safety result: *Lomatium* is the genus in
four of the eleven pre-registered hazard pairs.

## What was wrong before, and why

`TINY_K_FINDINGS.md` concluded the 17.9 MB encoder was the *best* of four at K=20.
That measurement used Pl@ntNet catalogue images of species inside the adapted
tower's fine-tune. Only **2 of these 24 Oregon species** are in that fine-tune, and
these are field photographs. Both conditions flipped, and so did the ordering.

Two size figures this project has carried were also wrong:

- **PlantCLEF2024 is not a 43 MB option.** That is 86.6M parameters × 4 bits, and
  **int4 destroys this model** — cosine −0.005, orthogonal, while fp16 converts at
  0.99998. DINOv2's LayerScale is the plausible culprit. Its real floor is int8 at
  87 MB.
- **MobileCLIP2-S2 is 36.8 MB, not 17.9.** Same arithmetic error, and int4 costs it
  4pp of label share where int8 costs nothing.

Quantization, once a precision that works for the architecture is chosen, is free:
0.830→0.828, 0.931→0.922, 0.960→0.962, 0.984→0.982. The failures were binary, not
gradual.
