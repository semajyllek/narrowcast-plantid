# Does deliberately framing one leaf beat photographing the plant?

Written before the photographs exist, so the prediction cannot drift to fit them.

## The claim

A person identifying a bigleaf maple looks at one leaf and reads its outline —
"those are starry, not the curvy lobes of an oak" — in about a second, from a
photograph the model gets wrong. The hypothesis is that the model can be given
the same view: **deliberately frame a single leaf or branch and block the
background with a hand.**

This is not the same as zooming in. Cropping tighter into a canopy was tried and
made things slightly worse — *Acer macrophyllum* fell from rank 2 to rank 4 at the
tightest crop — so resolution per leaf is not the mechanism. What a hand behind a
leaf does is **remove competing structure**: one outline against a plain field
instead of a texture of hundreds of overlapping leaves.

## Why it is plausible, and why it might fail

**For.** The failing photographs are canopies and trunks. The succeeding ones are
not necessarily uncluttered — *Solanum dulcamara* scored 91.8% in a dense tangle —
but they contain one loud diagnostic feature, in that case red berries. If what
matters is that a diagnostic feature dominates the frame, a hand-isolated leaf
supplies one.

**Against.** The bank is built from GBIF field photographs, which are mostly
*not* hand-isolated. A hand-backed leaf may be as far from the training
distribution as a canopy is, in the other direction, and the skin tone behind it
is a large novel region. The model could do worse.

Both outcomes are informative and neither is the assumed one.

## The test

Eight to ten species, at one site, one visit. For each:

1. **natural** — the photograph you would take anyway
2. **isolated** — one leaf or a small branch, hand behind it, filling the frame

Same plant, same light, minutes apart. About 20 photographs.

Score both through the shipped model. The endpoints, declared now:

- **primary** — top-1 correct on isolated against natural, paired by plant
- **secondary** — rank of the true species, so a near-miss is visible
- **secondary** — confidence in the top answer, since a right answer the model
  will not commit to is still a decline to the user

## What each result would mean

| outcome | what follows |
|---|---|
| isolated clearly better | the app should say so **before** the first photograph, not after a bad answer |
| no difference | drop it; tell users to photograph naturally and take three |
| isolated worse | the bank's distribution is the constraint, and the fix is training data with isolated subjects rather than user instruction |

## What this does not test

Whether a *hand* specifically is the right backdrop, as against a sheet of paper,
the sky, or simply stepping so the background is distant. If isolation helps, that
is the obvious follow-up, and the cheaper one.

## Status

**Not run.** The photographs do not exist and cannot be manufactured: constructing
them by masking an existing canopy image would test an edit, not a way of taking
a picture.
