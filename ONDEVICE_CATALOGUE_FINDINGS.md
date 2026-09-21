# The user can pick their own 20 species, on the phone, from a shipped bank

A scoping measurement, not a built thing. The question: can the catalogue be
chosen *by the user, at runtime* — twenty plants off a field-guide page — rather
than fixed when the bundle is exported?

## Why this is the right shape, not a compromise

The instinct is to make the model know more of Oregon. **That is the wrong
direction and this project already measured it.** A crowded label set buys
coverage with coarse answers that narrow nothing; the whole tool exists to warn
about it. A 1,000-species Oregon model would be worse at naming any given plant
than a 20-species one that happens to contain it.

The shipped Oregon bundle shows the other half of the problem. On the device it
is 91.4% correct at precision 1.000 — over the 24 species it knows, out of 4,570
in the state. Its thresholds assume 1 in 5 inputs is out-of-list; on a walk it is
19 in 20, and at that operating point it answers **3.6%** of what it sees.

Letting the user name their twenty resolves both at once: small K where the
accuracy is, and a list whose out-of-list rate is low *because the user chose it*.

## The head can be fitted on the phone, because it is fitted on embeddings anyway

`build.fit_head` never sees an image — it takes `ds.X_train`, which is already
embeddings. So shipping a bank of stored embeddings and fitting on-device is not
an approximation of the offline pipeline; it is **the same computation**.

618 Oregon photographs, 24 species, 12 distinct plants each, cluster-disjoint,
5 seeds. Exemplars are whole plants, not photographs:

| stored exemplars / species | top-1 | fit time | bank for 1,000 species |
|---|---|---|---|
| 1 | 0.799 | 12 ms | 1.5 MB |
| 2 | 0.907 | 9 ms | 3.1 MB |
| 4 | 0.943 | 13 ms | 6.1 MB |
| **8** | **0.960** | 18 ms | **12.3 MB** |
| 10 | 0.966 | 23 ms | 15.4 MB |

**Eight exemplars per species is the knee.** A 1,000-species Oregon bank at
float16 is 12 MB against the encoder's 87 MB, and the fit is ~20 ms for 24
classes on a laptop — trivial for a phone, once, when the user changes their list.

## Nearest-centroid is enough, which decides the implementation

| exemplars | logistic regression | nearest centroid | gap |
|---|---|---|---|
| 2 | 0.920 | 0.921 | −0.002 |
| 4 | 0.954 | 0.949 | +0.005 |
| 8 | 0.973 | 0.967 | +0.007 |

Under a point at every size, and *better* at two. So the on-device fit is a mean
per class and a dot product — about fifteen lines of Swift — rather than a
port of L-BFGS.

The cascade needs posteriors rather than similarities, so the centroids feed a
softmax whose temperature is fitted on the calibration split alongside the two
thresholds. `fit_thresholds` is a 60×60 grid over a few hundred rows: nothing
about it needs a laptop.

## The elegant part: the bank supplies its own negatives

A reject class needs out-of-list examples, and the shipped bundle uses a
background pool for them. Here the **979 species the user did not pick are
exactly the negatives**, and they are better ones — they are the actual
deployment distribution rather than a proxy for it. Both halves of the audit
come out of one bank.

## What is not measured

- **These exemplar counts are from 24 species.** With 1,000 in the bank and 20
  selected, the negatives are far more numerous and far more varied. The
  direction should hold; the numbers should not be quoted.
- **A user can choose a crowded list** — twenty *Ranunculus* — and would get a
  bad model. That is not new, and the card already measures and reports it; it
  would simply need to be shown before the user commits.
- **Building the bank is the real cost.** ~8,000 images fetched and embedded for
  1,000 species, once. The pipeline exists (`regional_fetch` → `regional_embed`)
  and `plantid/data/booklist.py` resolves a guide's names into it.
