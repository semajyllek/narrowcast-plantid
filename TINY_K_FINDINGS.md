# At the K a user actually picks, the 160 MB encoder buys nothing

Every encoder comparison in this project is at **K = 490**, and the size decision
it produced — 17.9 MB vs 43 MB vs 152 MB — was argued entirely at that scale.
`SMALL_FRONTIER_FINDINGS.md` flagged the gap and stopped: *"Note this is all
K=490; `s2` at K=20 is a different story."*

This is that story. Same label sets, same cascade, same declared `p_ood = 0.2`,
three draws per cell, K ∈ {10, 20, 30}. Reproduce with `/tmp/tiny_k.py` logic
over the cached `catalog_*_{variant}.npz` embeddings.

## Varied lists — the product a user should be steered to

| encoder | MB | top-1 @20 | label share @20 | label share @30 |
|---|---|---|---|---|
| BioCLIP-2 | 152 | 0.988 | 0.883 | 0.830 |
| PlantCLEF2024 | 43 | 0.992 | 0.923 | 0.840 |
| **MobileCLIP2-S2 fine-tuned** | **17.9** | **0.997** | **0.941** | **0.902** |
| MobileCLIP2-S2 stock | 17.9 | 0.971 | 0.783 | 0.706 |

**The 17.9 MB adapted encoder is the best of the four at every K tested**, on
closed-set accuracy *and* on the label-level share — and it beats the 152 MB
model by 5.8pp of label share at K = 20 and 7.2pp at K = 30. The 14.4pp species
gap that justified the whole size discussion is a K = 490 artifact. Narrowing
spends it.

Even **stock** S2 reaches 0.971 top-1 at K = 20. The floor for a 17.9 MB build on
a varied twenty-plant list is a shippable product.

## The caveat that decides which number applies

`mobileclip2_s2_ft` was fine-tuned on Pl@ntNet-300K, and
`ADAPT_FINDINGS.md` is explicit that its 1,081 training species **include all 530
catalogue species**: it is *task*-adapted, not merely domain-adapted, and is "a
result for the app and **not** evidence of a general small plant encoder."

So the table above is the **in-fine-tune** case. For labels outside Pl@ntNet-300K,
`ADAPT_FINDINGS` measures adapted-minus-stock at **−0.0010 at K=5** and
**−0.0175 at K=10** — a wash, not +10pp. The honest bracket for a 17.9 MB build
at K = 20 on a varied list is therefore:

- **plants inside Pl@ntNet-300K (7,806 species):** top-1 ≈ 0.997, names a species 94% of the time
- **plants outside it:** top-1 ≈ 0.971, names a species 78% of the time

Both ship. Pl@ntNet-300K covers most European and North American garden and wild
flora, so most user-chosen lists land in the first row.

## Crowding still costs far more than size

| K = 20 | varied | crowded |
|---|---|---|
| top-1 (s2_ft) | 0.997 | 0.817 |
| **label share (s2_ft)** | **0.941** | **0.504** |

Choosing three *Sedum* costs 44pp of label share. Choosing a 152 MB encoder over a
17.9 MB one *gains* −5.8pp. **List composition outweighs encoder size by roughly
an order of magnitude at this scale**, and unlike encoder size it is knowable
before any compute from the label strings alone.

## What this changes

`CLAUDE.md`'s open "size decision" — 17.9 / 43 / 152 MB — is a **490-species app**
decision and should not be carried to a narrow user-chosen catalogue. For K ≈
10–30 the answer is 17.9 MB, adapted if the labels are in Pl@ntNet-300K, and the
remaining budget is better spent on list composition warnings and on negatives for
the reject class.

**Unchanged:** source-shift brittleness. A change of acquisition source costs the
152 MB encoder −0.001 and a 17.9 MB one −0.179 (`DOMAIN_SHIFT_FINDINGS.md`), and
that is measured at K = 490 too. Whether it also shrinks at K = 20 is **not
measured**, and it is the live risk in shipping tiny.
