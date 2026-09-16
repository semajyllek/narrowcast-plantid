# Phase 0 — de-risking the regional-bundle plan

The plan named four things that could invalidate the product before any pipeline
was worth building. Three are answered; the fourth is running.

## 0a. MobileCLIP2-S2 converts to Core ML — **yes**

This was the existential item. `plantid/deploy/coreml.py` had only ever run on
BioCLIP ViT graphs, and S2's tower is `fastvit_mci2`, a hybrid conv/transformer.

It converts.

| | |
|---|---|
| artifact | `data/processed/coreml/mobileclip2_s2_int4_per_grouped_channel.mlpackage` |
| embedding dim | 512 |
| **size on disk** | **21.7 MB** |
| conversion time | 3,462 s (the palettize pass is 41 min of it) |

**The tower needs reparameterizing first, and now does it automatically.** The
checkpoint's tensor names (`conv_kxk`, `conv_scale`, per-branch BN) are the
MobileOne/FastViT signature: parallel branches at training time, algebraically
fused for inference. `build_traceable` now calls `reparameterize_model` when the
architecture supports it. Fusion is exact — **min cosine 1.000000** against the
unfused tower over random input — and yields 35.82M params, which is the
17.9 MB figure the project has been quoting.

> **But the artifact is 21.7 MB, not 17.9 MB.** The 17.9 is `params × 4 bits` and
> ignores everything not palettized — un-quantized ops, the embedding matrix,
> metadata. Quote **21.7 MB** for this build. Still an order of magnitude under
> BioCLIP-2's 153 MB.

### Latency, and it is very fast

| compute units | ms/image |
|---|---|
| CPU_ONLY | 179.3 |
| CPU_AND_GPU | 89.9 |
| **CPU_AND_NE** | **3.1** |
| ALL | 3.8 |

Op dispatch: **708 Neural Engine, 5 GPU**. For comparison, `ONDEVICE_FINDINGS.md`
has BioCLIP-2 int4 at 121 ms and BioCLIP-1 int4 at 19.2 ms on the same machine.
This is ~6× faster than the smallest previously-converted model.

All figures are **M4 Max**. No A-series number exists for any model in this
project.

## 0b. The preprocessing trap — fixed, and generalised

`coreml.py` hardcoded `CLIP_MEAN/CLIP_STD`, `SIDE = 224`, BICUBIC. Verified
against each encoder's own transform:

| variant | side | mean | interpolation |
|---|---|---|---|
| `bioclip1` | 224 | CLIP | bicubic |
| **`mobileclip2_s2`** | **256** | **0 / 1** | **bilinear** |
| `mobileclip2_s2_ft` | 256 | 0 / 1 | bilinear |

Rather than special-casing S2, `preprocess_spec(variant)` now **reads side, mean,
std and interpolation from `load_encoder`'s returned `preprocess`**, so a variant
added to the registry cannot reintroduce the bug. `build_traceable`, `export`,
`_pil_batch` and `benchmark` all take it from there.

### The validator was checking itself

`validate()` compared Core ML against `build_traceable`, which is the same code
path the export is built from and baked the same constants. **A shared wrong
assumption was invisible to it** — it would have reported cosine ≈ 1.0 while the
embeddings sat in a different space from every cached `.npz` in the repo. It now
compares against `load_encoder`'s own `preprocess`, the transform the accuracy
numbers were actually produced with.

The same bug existed one level down: `embed_coreml.embed_paths` called
`_pil_batch` with no variant. Both now thread an explicit `--encoder`.

### What the honest validator reports

**cosine_mean 0.838, cosine_min 0.677** against fp32 torch — worse than BioCLIP-1
int4's 0.932 / 0.831. Reparameterized FastViT weights are known to be harder to
quantize than ViT weights.

**This is not yet a verdict**, per the project's own convention (*"Cosine does not
predict accuracy. Measure the head."*). `ONDEVICE_FINDINGS.md` records 4-bit
BioCLIP at cosine 0.898 with **62% of nearest-neighbour relations destroyed** and
accuracy falling *barely a point*, because a refitted head absorbs a systematic
shift. Regional bundles always refit the head. The catalogue is being re-embedded
through this artifact now; the K=20 accuracy through it is the real answer.

## 0c. The fine-tuned checkpoint — backed up

144,303,123 B, sha256 `9377f0bb754beace119a9211129c03d8a101825be4df6a5d5334d9cf2602946b`,
copied to `~/Documents/narrowcast-artifacts/` with the checksum verified. Provenance,
identity and regeneration recorded in `data/ADAPTED_CHECKPOINT.md`.

Still only a local copy on one machine. Too large for git without LFS; if this
encoder carries a product it needs LFS or an object store.

## 0d. Images per species — **the fine-tune buys data efficiency, not just accuracy**

The "0.9345 on two images" figure came from `bioclip2`/`mobileclip2_s0`, never from
S2. Measured properly (`analysis/shots_s2.py`, K=20, three label sets, capped
training rows, scored through the unmodified cascade at `p_ood = 0.2`):

**Label-level share, varied lists:**

| encoder | 2 | 4 | 8 | 16 | 32 | all |
|---|---|---|---|---|---|---|
| `mobileclip2_s2_ft` | **0.923** | 0.932 | 0.936 | 0.938 | 0.942 | 0.941 |
| `mobileclip2_s2` | 0.541 | 0.644 | 0.726 | 0.791 | 0.813 | 0.783 |
| `bioclip2` | 0.837 | 0.871 | 0.866 | 0.883 | 0.881 | 0.883 |

**This is the most useful number Phase 0 produced.** The adapted encoder is
**saturated at two images per species** — 0.923 against 0.941 with everything.
Stock S2 needs **16–32** to approach its own ceiling.

So the fine-tune is worth far more than the ~2pp of accuracy the encoder table
shows: it is a **~16× reduction in the per-region fetch budget**. For a product
whose main cost is acquiring images for every species in every region, that is the
difference between 40 images per species and 2.

**Which curve applies depends on the region.** The catalogue's 490 species are
inside the fine-tune's 1,081, so these are in-fine-tune numbers. For species
outside Pl@ntNet-300K, `ADAPT_FINDINGS.md` puts adapted at −0.0353 against stock —
plan on the stock curve, and therefore the larger fetch.

Crowded lists remain the dominant risk at any shot count: `s2_ft` scores 0.923
varied against **0.287** crowded at two images. Sparse data and a crowded list is
the combination that fails invisibly.

## Verdict

No blocker. The encoder converts, runs at 3.1 ms on the ANE, fits in 21.7 MB, and
needs as few as two images per species where the region's plants are in
Pl@ntNet-300K. The one open question is what int4's embedding drift costs the
head, and that is measuring now.

Two numbers the project has been quoting need correcting: the artifact is
**21.7 MB, not 17.9**, and the 18.1 ms was MPS — the Core ML figure is **3.1 ms**.
