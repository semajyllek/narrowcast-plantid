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

---

## 0a, continued — what int4 costs the head

`ONDEVICE_FINDINGS.md`'s precedent said a refitted head absorbs quantization
damage almost entirely: 4-bit BioCLIP at cosine 0.898 destroyed 62% of
nearest-neighbour relations and cost *barely a point*. That did **not** transfer.

Measured on leaf only, K=20, identical species and splits, the head refitted on
each embedding source — so this *is* the refitted-head case
(`analysis/int4_cost.py`):

| list | torch S2 | Core ML int4 | delta |
|---|---|---|---|
| varied | 0.9267 | 0.8633 | **−6.3pp** |
| crowded | 0.6573 | 0.6109 | −4.6pp |

Paired over 12 label sets: **mean −0.0549**, min −0.115, max +0.011.

**So cosine was right this time.** The project's standing rule — *"cosine does not
predict accuracy; measure the head"* — is a warning against trusting cosine in
either direction, and here the measurement confirmed it rather than dismissing it.
The mechanism is plausible: reparameterizing MobileOne/FastViT fuses parallel
branches into single convs with a much wider dynamic range than a ViT's weights,
which is exactly what 16 palette entries struggle to cover. BioCLIP's experience
does not carry to this architecture.

### What this does not yet settle

This is **closed-set top-1**, which needs no negatives. The product's headline is
**label share**, and the two can diverge violently: `TINY_FINDINGS` records a
0.14 MB student at top-1 0.471 naming **zero** labels while a 1.53 MB student at
0.492 named 17.4%, because capacity buys *sharpness* as well as accuracy and the
threshold reads sharpness. If int4 flattens the confidence distribution, label
share falls further than top-1 does.

The background pool is embedding now; label share, coverage and rejection follow.

### The options, priced

- **int8, ~40 MB.** `ONDEVICE_FINDINGS` has 8 bits at cosine 0.998 against 4 bits'
  0.898 on BioCLIP, and int8 is far kinder to wide-dynamic-range fused convs.
  Still 4× under BioCLIP-2's 153 MB.
- **int4 at 21.7 MB, accepting −5.5pp.** Possibly fine on the *adapted* encoder,
  which starts at 0.997 and has the headroom to absorb it — unmeasured.
- Note the whole comparison is stock S2. `s2_ft` has never been exported.

---

## Correction: int4 costs ~4pp, not 14pp — and a bug of mine inflated everything

Two things above were wrong, both mine.

### The throughput figure was a bug I introduced

I reported "~0.82 s/image, consistent across organs" as the real cost, and framed
it as a correction to the `benchmark()` figure of 3.1 ms. **The 0.82 s was my
bug.** The 0b fix made `_pil_batch` call `preprocess_spec(variant)`, which
constructs the torch encoder and touches the HF Hub — and `embed_coreml.embed_paths`
calls `_pil_batch` **once per image**. So a 51,000-image run rebuilt MobileCLIP2-S2
about fifty thousand times.

Measured after adding `@lru_cache` to `preprocess_spec`:

| | |
|---|---|
| cold call (builds the encoder) | 4.48 s |
| per image, cached | **2.0 ms** |
| speedup | ~2,200× |

The 12.5-hour run was ~99.97% rebuilding an encoder. It eventually died when the
Hub closed the connection — which is the only reason the bug surfaced at all.
Re-running the one missing cache took **32 seconds** against a 1h30m projection.

The five caches written before the failure are valid: the preprocessing was
correct, just recomputed wastefully.

### Leaf-only overstated the damage by ~3×

The `-14pp label share` verdict was measured on leaf alone, because flower had not
finished embedding. On the two-organ footing `TINY_K_FINDINGS` actually uses:

| metric | varied | crowded |
|---|---|---|
| fine (top-1) | −2.8pp | −3.1pp |
| **label share** | **−6.2pp** | −2.0pp |
| coverage | −5.1pp | −7.9pp |
| precision | −0.8pp | −0.1pp |
| decline share | +6.4pp | +8.7pp |

Paired over 12 label sets: **mean −0.0408**, min −0.147, max +0.110.

So **int4 costs about 4pp of label share, not 14pp.** A second organ gives the head
enough redundancy to absorb much of the quantization noise — which is its own small
finding, and a reason guided multi-photo capture may matter more than the fusion
numbers alone suggest.

The mechanism is unchanged and still visible: precision is flat, declines rise.
int4 makes the model less confident rather than more wrong.

### Revised verdict

~~int4 per-grouped-channel is not acceptable here.~~ **Not established.** At
−4.1pp of label share for 21.7 MB it is a real but arguable cost, where at −11.9pp
it was not. int8 at ~40 MB is still worth measuring, but int4 is no longer
disqualified — and on the *adapted* encoder, which starts at 0.941 label share,
−4pp lands near 0.90.

Both errors point the same way: **a partial measurement read as a verdict.** The
leaf-only run was the right thing to do while waiting, and the wrong thing to
conclude from.

---

## Decision: int8, at 36.8 MB

Three-way, two organs, K=20, identical species and splits, head refitted on each
source (`analysis/int4_cascade.py`, `data/processed/quant_cascade_both.csv`):

| | torch fp32 | int4 (21.7 MB) | **int8 (36.8 MB)** |
|---|---|---|---|
| cosine vs fp32 | — | 0.838 | **0.995** |
| label share, varied | 0.7461 | 0.6841 | **0.7580** |
| label share, crowded | 0.2364 | 0.2167 | **0.2524** |
| decline share, varied | 0.2061 | 0.2705 | **0.1907** |
| **paired label-share delta** | — | **−0.0408** | **+0.0139** |
| worst single set | — | −0.1465 | **−0.0045** |

**int8 is free.** The +1.4pp is inside noise over twelve sets and is not claimed as
a gain; what matters is that the *worst* set moves −0.0045 where int4's moved
−0.147. int4's cost was never uniform — it was a tail.

### Latency, and a backend difference worth knowing

| | int4 | int8 |
|---|---|---|
| CPU_AND_NE | 3.12 ms | **2.75 ms** |
| CPU_AND_GPU | 89.9 ms | 4.89 ms |
| op dispatch | 708 ANE / 5 GPU | **94 ANE / 626 GPU** |

int8 is *faster* and lands mostly on the **GPU**, where int4 per-grouped-channel is
silently wrong (`ONDEVICE_FINDINGS.md`: cosine 0.204). So int8 was checked on every
backend rather than assumed:

| backend | cosine mean | min |
|---|---|---|
| CPU_ONLY | 0.9950 | 0.9922 |
| CPU_AND_GPU | 0.9951 | 0.9924 |
| CPU_AND_NE | 0.9915 | 0.9863 |
| ALL | 0.9951 | 0.9924 |

**Correct everywhere.** Linear per-channel quantization does not have the
per-grouped-channel palette's GPU bug, which means an int8 build does *not* need
the `.cpuAndNeuralEngine` pin that `CLAUDE.md` requires for int4 — it is free to
take whichever backend Core ML prefers.

### The trade

15 MB buys back 4pp of label share, removes a −15pp tail, runs 12% faster, and
drops a deployment constraint. Take it. **36.8 MB** is still 4× under BioCLIP-2's
153 MB, and the encoder is the entire download — the per-region head is ~83 KB.
