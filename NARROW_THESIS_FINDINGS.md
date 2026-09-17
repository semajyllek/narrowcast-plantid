# Was the original intuition right? Half of it, and the half that matters

The thesis this project started from: *a user who already knows they care about
≤ 20 plants should get something (1) smaller and (2) better at those 20 than a
general model.* Both halves are now tested. **(1) is false and was tested three
times. (2) is true and had never been tested directly until now.**

## (1) Smaller — tested, failed, closed

`TINY_FINDINGS.md` §3 is precisely this idea in its strongest form:
**task-conditional distillation.** Teacher = frozen encoder + fitted head; target
= the K+1 posteriors over *the user's own labels*; K = 14; two label sets; scored
cross-source through the unmodified path.

| student | label share, separated | label share, crowded |
|---|---|---|
| 0.14 MB, trained at the teacher's own 518 px | **0.190** (bar 0.849) | **0.000** (bar 0.404) |

Closed with *"do not reopen without a new objective, not a new size"* — the third
of three distillation closures. The reason is the finding the whole project rests
on: **the accuracy lives in the encoder's representation, not in the head.** A
student that matched the task exactly still named almost nothing, because capacity
buys *sharpness* as well as accuracy and the threshold reads sharpness.

So a 20-plant model is **83 KB** — but only on top of a shared encoder that does
not shrink. The right claim is "one encoder, many tiny heads", not "a tiny model
per user".

## (2) Better at those n — tested here, and the answer is yes, twice over

PlantCLEF2024's checkpoint carries its own **7,806-way classifier**, which
`pretrained.load_encoder` discards to expose features. Restoring it allows the
experiment the thesis actually needs, on identical photographs, identical splits
and identical embeddings:

- **FILTER** — the general 7,806-way posterior, masked to the user's labels, argmax
- **FIT** — a logistic head trained on the user's labels over the same frozen features

| | FIT | FILTER |
|---|---|---|
| top-1 | **0.955** | 0.676 |
| sd over 5 seeds | 0.025 | 0.061 |

**Paired: +0.279.** Fitting on the user's own labels beats filtering a general
classifier by **28 points** on the same seven species.

### And the larger effect is that the general model does not have your plants

That comparison runs on **7 of 24** labels, because the other 17 are not in
PlantCLEF's 7,806 classes at all. Across the region:

| Oregon species | in PlantCLEF's 7,806 |
|---|---|
| ≥ 100 records (1,941) | 478 — **24.6%** |
| ≥ 500 records (526) | 144 — 27.4% |
| our 24-species list | 7 — **29%** |

Missing: *Lomatium triternatum*, *Cornus sericea*, *Chrysolepis chrysophylla*,
*Ceanothus prostratus*, *Adenocaulon bicolor*…

**About three quarters of a regional flora is outside a 7,806-class "general"
model's label space**, so for those species filtering does not lose by 28 points —
it cannot answer at all. The general model is general over *European* flora;
Oregon is not that.

### What the 28 points are, honestly

Part of it is that FILTER's head was trained on Pl@ntNet images and tested on GBIF
field photographs, so it pays a source shift FIT does not. That is not a
confound to apologise for — it **is** the argument for the narrow approach, since
fitting locally means fitting on data that resembles deployment. But state it as
what it is: *a head fitted on your labels and your data beats a general head
transferred from elsewhere*, rather than *narrow models are intrinsically better
classifiers*.

Consistent with `CLAUDE.md`'s existing note that the cascade applied to Pl@ntNet's
own returned scores beats this project's model: **the abstention machinery is a
technique, not a moat.** The moat is the fitted head plus regional data plus
running offline.

## The corrected thesis

> A user who knows their ≤ 20 plants gets a **83 KB** head over a shared **87 MB**
> encoder, which is **+28 points better** than masking a general classifier to the
> same species — and which can name the ~75% of their regional flora that the
> general classifier has no class for at all.
>
> It is not a smaller *model*. It is a smaller *per-user artifact*, and that
> distinction was tested to destruction three times before being believed.

## What remains untested

The FIT/FILTER comparison rests on **7 species** — the overlap is the binding
constraint, not the design. A European region (where PlantCLEF coverage is high)
would allow the same test at K = 20+ and is the obvious way to strengthen it. The
prediction, given the above, is that FIT's margin *shrinks* where the general model
actually knows the flora, and that the coverage argument does most of the work
outside Europe.
