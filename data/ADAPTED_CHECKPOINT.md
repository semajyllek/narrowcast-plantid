# `mobileclip2_s2_plantnet.pt` — provenance and backup

The fine-tuned MobileCLIP2-S2 visual tower. `TINY_K_FINDINGS.md` measures it as the
**best of four encoders at K=20** (0.997 top-1, 0.941 label share on a varied
list) — better than BioCLIP-2 at 152 MB. It is the encoder the regional-bundle
product is planned around.

It was trained in Colab (`notebooks/adapt_s2_colab.ipynb`), lives under the
gitignored `data/processed/`, and **exists nowhere in version control.** Losing it
costs a GPU run and, because the notebook samples the adaptation pool, would not
reproduce bit-for-bit.

## Identity

| | |
|---|---|
| path | `data/processed/adapted/mobileclip2_s2_plantnet.pt` |
| size | 144,303,123 bytes |
| sha256 | `9377f0bb754beace119a9211129c03d8a101825be4df6a5d5334d9cf2602946b` |
| contents | flat `state_dict` of the visual tower — 1,572 tensors, **35.9M params**, fp32 |
| loaded by | `plantid/features/pretrained.py`, variant `mobileclip2_s2_ft`, loader `open_clip_ft` |

Verify with:

```bash
shasum -a 256 data/processed/adapted/mobileclip2_s2_plantnet.pt
```

## Backup

`~/Documents/narrowcast-artifacts/mobileclip2_s2_plantnet.pt`, checksum verified
identical. Outside the repo because 144 MB exceeds GitHub's 100 MB per-file limit
without LFS. **This is a local copy on one machine — it is not yet durable against
machine loss.** If this encoder is going to carry a product, it belongs in LFS or
an object store with the checksum above as its identity.

## What the tensor names tell us, and why it matters for Core ML

The keys are `trunk.stem.0.conv_kxk.*`, `conv_scale.*`, and batch-norm statistics
throughout — the signature of **MobileOne / FastViT reparameterizable blocks**.
These carry a multi-branch structure at training time (a k×k conv, a scale conv and
a BN branch in parallel) that is algebraically fused into a single conv for
inference.

Consequence for the export in Phase 0a: the tower almost certainly needs
**reparameterizing before conversion** (timm exposes `reparameterize_model`).
Converting the unfused graph would produce a larger, slower artifact than the
17.9 MB / 18.1 ms figures quoted for this encoder — and those figures are
themselves arithmetic and MPS respectively, never Core ML measurements.

Do not treat a successful conversion of the *unfused* model as proving the size
claim.

## Regenerating

`notebooks/adapt_s2_colab.ipynb` — fine-tunes stock
`hf-hub:timm/MobileCLIP2-S2-OpenCLIP` on Pl@ntNet-300K with a 1,081-way species
objective, then refreezes. `catalogue_species.json` at the repo root is what the
notebook uses to exclude catalogue ids from the adaptation pool.

Note the scope limit recorded in `ADAPT_FINDINGS.md`: the 1,081 training species
**include all 530 catalogue species**, so this tower is task-adapted, not merely
domain-adapted. It is **−0.0353** against stock on species outside that list.
