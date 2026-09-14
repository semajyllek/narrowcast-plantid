# What to do with this work

A disposition memo across the four repos, written 2026-09-13 after reading
`narrowcast-plantid/CLAUDE.md`, `narrowcast/docs/deep_dive.html`, the published
18-section deep dive, `narrowcast-derm/K_FINDINGS.md`,
`narrowcast-kws/FEWSHOT_FINDINGS.md` and `HEADROOM_FINDINGS.md`.

## The short answer

The tool was the wrong deliverable and the **finding** is the right one. Stop
treating narrowcast as a product that builds people models, publish the result as
a written contribution, and keep as code only the part that never failed — the
measurement discipline, which is a few hundred lines rather than a package.

Do **not** delete the three non-plant repos. The headline result's entire claim
to generality lives in them, and plantid cites those numbers without holding
their evidence.

## 1. It is one finding with two consequences, not two findings

The two phenomena named — groupings as a lower-dimensional label space, and an
artificially restricted class space — are not independent. They share one
upstream variable, and the record already says so; what it does *not* yet do is
name the variable and put the two consequences side by side.

**Fine-rank accuracy is the shared upstream variable, and it has two distinct
downstream consequences.** Keeping them apart matters, because the record uses the
word *headroom* for both and they are not the same number:

| quantity | definition | governs | established in |
|---|---|---|---|
| **headroom** | coarse-rank accuracy − fine-rank accuracy | **retreat** to the group rank | `HEADROOM_FINDINGS.md` |
| **remaining accuracy** | 1 − top-1 | **sensitivity** to a training-data intervention | `narrowcast-derm/K_FINDINGS.md` |

They coincide when coarse accuracy pins near 1 — which is the easy, small-K plant
regime — and diverge everywhere else. That is not a footnote: breaking exactly
that collinearity *was* the headroom contribution. The five kws arms that first
proposed headroom were uninterpretable because coarse accuracy never left
[0.93, 1.00], making headroom "nearly `1 − fine` rescaled." The derm K sweep has
**no coarse-rank column at all** — it measures `top1_L` and damage, and its
regressor is baseline accuracy, not coarse minus fine. So the two levers are
genuinely measuring different things.

With that separated, the four levers line up cleanly:

| lever | moves | consequence |
|---|---|---|
| **grouping** | headroom only — fine accuracy is untouched | retreat |
| **narrowing K** | fine accuracy, hence both quantities | less retreat *and* less intervention sensitivity |
| **few shots** | fine accuracy downward, hence both | manufactures retreat; magnifies interventions |
| **declared `p_ood`** | neither — it sets the threshold | decides whether available retreat is *realised* |

The controlled experiment is the thing that makes this publishable. One fitted
head, 60 congeneric species, **fine accuracy pinned at exactly 0.659 in every
row**, only the group column varying: group answers move **0.022 → 0.417** and
coverage **0.200 → 0.500**. Nothing about the task got easier. The grouping alone
moves the product. Then 1,409 arms across four domains: headroom predicts retreat
at **CV R² 0.883**, against 0.362 for fine accuracy alone and 0.113 for coarse
alone — a single number recovering 97% of what a free two-predictor model gets.

**The restricted-class-space story resolves into the upstream variable, and this
is the more interesting half.** The intuition was that small label sets are safe.
They are not — what is safe is *high fine-rank accuracy*, and small K is merely
one common way to get there. K is a proxy, and which consequence you escape
depends on which quantity actually shrank. The 28-arm matched-K regression gives
`damage = −0.194 + 0.181 × baseline, R² = 0.61`, and derm confirms it by about an
order of magnitude: at **K = 10, where plants show exactly 0.000 damage,
dermatology loses 13.5 points**; at K = 5 it still loses 10.7. `K_FINDINGS.md`
states the conclusion in the right form — *"small label sets are safe from this"
is false in general; key any guidance on the build's measured accuracy, never on
K.*

That is a cleaner contribution than two separate stories, and it survives the
obvious reviewer objection — that headroom and `1 − top-1` are the same thing
wearing two hats — because the grouping experiment moves one with the other
pinned, and the derm sweep moves the other with no coarse rank in the design at
all.

**The boundary condition is measured, which is rare and worth leading with.**
The `1.8 × headroom` rule of thumb is refuted in its own record: holding headroom
fixed at 0.124 and sweeping only the declared out-of-list rate, realised retreat
moves from 0.1383 to 0.0015 — a factor of 91 — while the prediction sits pinned
at 0.223 the whole time. Headroom predicts the retreat that is *available*, not the retreat that is
*realised*. A paper that ships its own refutation of its own rule of thumb is
more credible than one that does not.

**And the cross-modal replication is what makes it a finding rather than a plant
result.** Five corpora, two of them audio, one text; and the source-composition
mechanism replicates in derm (subject-borne source variable, skin tone) and in
Speech Commands (speaker identity), landing at 1.9–2.1× and 2.1–2.6× the plant
curve respectively — two domains sharing no modality, encoder, task or source
variable arriving at the same place.

## 2. What failed, and say so plainly

The automation thesis is the part that did not work, and the repo's own history
is the evidence:

- `fit` refuses rather than builds on any label set that does not clear the floor
  — which is the honest behaviour, and also an admission that the general case is
  not servable.
- encoder discovery was demoted from the build path to a maintenance command
  (`narrowcast encoders`) because every discovered candidate failed inside
  `sweep.evaluate` and was silently swallowed.
- distillation was closed **three separate times**, each time by a better
  off-the-shelf model.
- the few-shot rescue (shrinkage LDA on frozen features) missed its declared bar:
  >5pp at ≤8 rows/label was required, +2.84pp was delivered.
- `plan` cannot compute the quantity the whole thing turns on, and CLAUDE.md
  wrongly claimed it could for a while.

That reads as failure only against the goal of a general pipeline. Against the
goal of *establishing what governs this class of problem*, every one of those is
a result — and four of the five are negative results with pre-registrations,
which is the scarcer good.

The instinct that a user could get the same model by iterating with an LLM is
right about the *fitting* and wrong about the *evaluation*. Nothing in that loop
tells them their coverage went up because the model got worse. That asymmetry is
the entire remaining product surface, and it is small.

## 3. Options

**A — Write it up as a research contribution. Recommended.**
The deep dive is already 80% of a paper: 18 sections, real photographs, every
number traced to a findings doc, retractions recorded in place. What it needs is
a spine — reorganised around headroom as the governing quantity with grouping,
K, shots and `p_ood` as the four levers — plus the K result, which the published
version carries and the framing does not yet.

Venue realistically: a public write-up plus arXiv. The audience is people
building narrow classifiers over frozen encoders, which is now a large
population, and the claim "your two headline metrics improve as your model gets
worse, here is the predictor, here is when it fires" is directly actionable for
them. Effort: weeks, not months, because the measurements exist.

**B — Retarget narrowcast from builder to auditor. Recommended alongside A.**
*(Executed 2026-09-13, narrowcast 0.2.0. 833 lines removed, package 2,900 →
1,999, tests 65 → 77. The `--scores` path below was built; `audit` replaces
`fit`/`build`/`plan`/`encoders`. The line estimate in this section was made
before the cut and is checked against the result at the end of it.)*
Not "build me a model" — *"here are my predictions, my labels, my group column
and my assumed out-of-list rate; tell me what I actually have."* No encoders, no
fitting, no sweep, no hub, no config space. The surviving surface:

- three-way reporting; coverage never printed without label-level share
- declared utilities, fixed in source before fitting
- prevalence anchoring to a stated `p_ood`
- cluster-keyed splits and cluster bootstrap
- bootstrap the ratio, not the mean
- hazard accounting as a union, not per pair
- measured headroom and the `label`/`group`/`decline` split

That is the part with no failure in its history. Counted rather than guessed: the
keep-set is `cascade.py` (160 lines), `card.py` (486) and the measurement half of
`build.py` (510 total), plus a much smaller `cli.py` — call it **1,000–1,200 of
the current 2,900**. What drops out entirely is `encode`, `hub`, `encoders`,
`sweep`, `plan`, `projection` and `config`: **833 lines**, and every one of the
failures in §2 lives in them. The near-infinite configuration space disappears
because the caller has already made every configuration decision.

> **What it actually came to.** The 833 was exact. The keep-set estimate was low:
> the package landed at **1,999 lines**, not 1,000–1,200, because `--scores` is a
> genuinely new input path rather than a subtraction, and because `labels.py` and
> `predict.py` survive intact and were not in the estimate. The shape is right and
> the number was optimistic — worth recording, since the same optimism is what
> made the builder look tractable in the first place.

**C — Archive as-is and maintain nothing.** Push a final commit to all four
repos, add an `ARCHIVED` header pointing at the write-up, stop. Acceptable
fallback if A is not going to get written; strictly worse than A because the
findings docs are unreadable without the spine.

**D — Delete everything but plantid.** This is the proposal as stated, and it
is the one option I would argue against. It does not just discard code — it
discards the cross-domain replication, which is the only reason the finding is
about *classification* rather than about *plants*. See §4.

Recommendation: **A + B.** plantid stays as the deployment story it always was.

## 4. Must survive / safe to delete

### Must survive — irreplaceable

- **`narrowcast-kws` and `narrowcast-derm` in full.** plantid's CLAUDE.md quotes
  their numbers but holds neither the evidence nor the scripts. Delete them and
  the headline becomes a single-domain result with a citation to nothing.
  Specifically: `K_FINDINGS.md` + `k_sweep.py` (derm), `FEWSHOT_FINDINGS.md` +
  `fewshot_curve.py` and `SOURCE_MIX_FINDINGS.md` (kws). These are ~12 MB of text
  and code.
- **`data/processed/headtohead/` — 5.4 MB, 1,394 cached API responses.** The only
  asset here that costs external quota to rebuild. Re-scoring
  `COMPETITIVE_FINDINGS.md` is free while it exists and expensive or impossible
  after. Back it up *off* the gitignored tree.
- **All 28 findings/prereg docs**, including the retracted claims. The
  retract-in-place convention is a large part of what makes the record
  trustworthy and it does not survive summarisation.
- **`analysis/headroom_arms.py`** — produces the 1,409 arms behind the headline
  R².
- **`narrowcast/docs/figures.py` and `docs/make_cases.py`** — they generate the
  deep dive's figures from measured data. If option B guts the package, the
  paper's figures become unregenerable. Same class of loss as `headtohead/`.
- **The deep dive, and note it exists twice.** The published artifact has **18
  sections**; the committed `narrowcast/docs/deep_dive.html` has **15** — missing
  source shift, training-source composition and the three-levers section. The
  better copy is private on claude.ai and the public copy is stale *and not
  linked from the README*. Fix that before anything else; it is a five-minute job
  protecting the single most valuable artifact in the project.

### Regenerable, delete freely

- `.venv` + `.venv-mps` (2.0 GB) and derm's `.venv` (402 MB) — **2.4 GB**
- `build/`, `*.egg-info`, `.pytest_cache`, `__pycache__` across all four repos
- `analysis/distil_*.{json,log}` (108 KB) — the line is closed three times over;
  `PRUNE_FINDINGS.md` and `TINY_FINDINGS.md` hold the conclusions
- `notebooks/distil_colab.ipynb`, `tiny_student_colab.ipynb` — closed lines.
  **Keep `adapt_s2_colab.ipynb`**; that one shipped (+0.1022 species).

### Judgement calls, in cost order

`narrowcast-plantid/data/processed` is **17 GB**, not the 12 GB CLAUDE.md claims.

| what | size | refetchable? |
|---|---|---|
| `images/` + `images_inat/` + `images_background/` | 9.6 GB | yes, slowly, from iNat/Pl@ntNet |
| `bundles/` | 4.0 GB | yes, from images + encoders |
| `*.npz` embedding caches | 2.1 GB | yes, needs images **and** the exact encoder builds |
| `coreml/` | 403 MB | yes, needs `.venv-mps` + coremltools |
| `headtohead/` | 5.4 MB | **no** |
| `narrowcast-kws/data` | 5.5 GB | yes — ESC-50 and Speech Commands are public downloads |
| `narrowcast-derm/data` | 205 MB | yes — Fitzpatrick17k |

Cheapest defensible cut: drop `bundles/` and kws `data/` (**9.5 GB**), keep the
embedding caches, because those are what `headroom_arms.py` actually consumes and
regenerating them requires reproducing encoder builds rather than re-downloading a
public corpus.

## 5. If A gets written, the order is

1. Link and refresh `docs/deep_dive.html` from the 18-section published version.
   Stop the divergence.
2. Restructure around the four levers. The K result becomes a section rather than
   a citation.
3. ~~Reconcile one number: the `p_ood` swing is reported as ~70× in plantid's
   CLAUDE.md and 91× in the deep dive.~~ **Resolved, 2026-09-13.** Recomputed
   from `narrowcast-derm/results/ood_sweep.csv`: mean group share is 0.1383 at
   `p_ood = 0.05` and 0.001515 at 0.60, a factor of **91.3**. The deep dive was
   right; CLAUDE.md had divided its own rounded 0.002. Fixed there.
4. Cut narrowcast down to the auditor (option B) and point the paper at it.
5. plantid keeps its own story — 0.7720 against iNaturalist's server model's
   0.7871 on identical photographs, offline, in 160 MB — and it does not need
   the generalisation to be true.
