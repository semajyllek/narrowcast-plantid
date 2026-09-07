# Pre-registration — is there accuracy left on the table by training on one source?

Written **before** any head in this design was fitted. It extends
`DOMAIN_SHIFT_PREREG.md` / `DOMAIN_SHIFT_FINDINGS.md` and reuses their splits,
inclusion rule and bootstrap unchanged.

## The observation that prompts it

The source-shift 2 × 2 was built to measure a penalty. It also turned up
something nobody went looking for. At a matched 10-photograph budget over 359
shared species, tested on the *same* held-out iNaturalist photographs:

| head fitted on | species top-1 on iNaturalist |
|---|---|
| Pl@ntNet-300K — **the head that ships** | 0.7691 |
| iNaturalist | **0.8615** |

**9.2 percentage points, and the production head has never seen an iNaturalist
photograph.** The training corpus and the deployment corpus differ by historical
accident — the catalogue was built from Pl@ntNet and the evaluation set was
fetched later from iNaturalist — not by any decision. Nobody has tested the mix.

## The claim under test

Adding in-source training data to the shipped head raises species accuracy on
held-out iNaturalist observations. Stated as a null to be rejected: **mixing the
two sources performs no better than Pl@ntNet alone.**

## Design

Closed-set, organ-free, photo-level, macro-averaged over species, paired
species-cluster bootstrap over 2,000 resamples — identical in every respect to
the primary arm of `analysis/domain_shift.py`, so the numbers sit in the same
table as the ones already published. Every arm is scored on the **same** held-out
iNaturalist photographs, from observations no arm trained on.

`B = 10` photographs per species per source, as before. Pl@ntNet supplies a mean
of 62.6 training photographs per species; the iNaturalist training side supplies
about 16 after the observation-level split.

| arm | fitted on | question it answers |
|---|---|---|
| **`P-full`** | every Pl@ntNet training row | the production baseline |
| **`P-full + i`** | that, plus all iNaturalist training photographs | **primary:** does adding in-source data help what ships? |
| `P-2B` | 20 Pl@ntNet photographs/species | fixed-volume control |
| `P-B + i-B` | 10 of each | **secondary:** at *equal volume*, is in-source data worth more? |
| `i-only` | 10 iNaturalist photographs/species | where the ceiling sits |

**The two questions are different and both are declared.** `P-full + i` has more
training data than `P-full`, so a gain there could be volume rather than source.
The fixed-volume pair isolates source composition: same number of rows, half of
them swapped for in-source ones. A gain on the primary with a null on the
secondary would mean "more data helps"; a gain on both would mean "in-source data
is worth more per row."

**Primary endpoint.** Species top-1 on held-out iNaturalist photographs,
`P-full + i` minus `P-full`, paired over species.
**Secondary.** The fixed-volume contrast; genus top-1 for both; and the same
arms scored on held-out **Pl@ntNet** photographs, to check whether mixing costs
anything on the source the head came from.

## Ways this comes out uninformative, declared in advance

1. **The iNaturalist side is small.** About 16 training photographs per species
   against Pl@ntNet's 62.6. Adding 16 rows to 62 may simply be swamped, and a
   null on the primary would then be a statement about *volume available*, not
   about the value of in-source data. The fixed-volume arm exists precisely so
   that this failure mode is still informative.
2. **The test set is smaller than the published one.** Only the held-out 40% of
   iNaturalist observations can be scored here, so intervals are wider than in
   `DOMAIN_SHIFT_FINDINGS.md` and a small effect will not resolve.
3. **Species composition is the same 359** as the source-shift arm, by the same
   declared inclusion rule. Species that cannot supply both a training and a test
   slice on both sides are out, which is not the production label set.

## The cost of a positive result, stated before it is one

This matters more than the measurement and is the reason to report the number
before changing anything that ships.

**iNaturalist is currently a pure held-out source.** That is exactly what makes
`DOMAIN_SHIFT_FINDINGS.md`'s central claim true — the head has never seen an
iNaturalist photograph, so every headline figure in this repo is an
*out-of-source* measurement, which is a much stronger thing to be able to say
than an in-source one.

Spending part of it on training destroys that. The remaining evaluation set gets
smaller and stops being cleanly out-of-source, and the contamination and
source-shift results would both need re-qualifying against a new split.

So a positive result here is **not** an instruction to adopt it. It is a price
tag: *this much accuracy is available, and this is what it costs in evidential
standing.* Which side of that trade is right is a product judgement and belongs
to the repo's owner, not to this experiment.

---

# Addendum — pre-registering the middle path

Written before any middle-path head was fitted. `SOURCE_MIX_FINDINGS.md` closed
by proposing one and describing it as keeping "a clean claim on a smaller set."
**That description was wrong, and the error is worth stating before it is
measured.**

Holding out *observations* preserves nothing about source. A held-out
iNaturalist observation is still drawn from a corpus the head trained on, so the
claim "the head has never seen an iNaturalist photograph" dies for the whole
model however the observations are split. What survives is an ordinary
in-source, out-of-observation evaluation — useful, and not the thing that was
being protected.

Only holding out **species** preserves an out-of-source measurement, and only for
those species: if no iNaturalist row of species *X* ever enters training, then
iNaturalist photographs of *X* measure genuine cross-source transfer.

So two arms, and they answer different questions.

## M1 — observation holdout, swept

Per species, hold out a fraction `h` of observations; train on `P-full` plus the
iNaturalist rows of the rest; score the held-out ones. `h = 1.0` is the status
quo. Nested: the held-out set at `h = 0.8` contains the one at `h = 0.6`, so the
sweep is comparable across `h` and the species set is identical throughout.

**Endpoint:** species top-1 on held-out iNaturalist photographs, and the **width
of the 95% interval**, which is what decides whether a smaller evaluation set is
still worth having.

**Declared expectation, so it is not claimed as a discovery afterwards:** the
interval may barely widen. It is a *species-clustered* bootstrap and the number
of species is constant across the sweep; only photographs per species shrink. If
between-species variance dominates, shrinking the set costs little precision —
which would make M1's trade-off far better than the framing in
`SOURCE_MIX_FINDINGS.md` assumed.

## M2 — species holdout

Reserve a random 20% of species at a fixed seed. **No iNaturalist row of a
reserved species ever enters training.** Train on `P-full` plus the iNaturalist
rows of the other 80%. Then score, against the `P-full` baseline on identical
rows:

- **primary:** reserved species, on their iNaturalist test observations — genuine
  out-of-source, and the operational question for a catalogue that grows;
- secondary: mixed species, in-source, as the upper reference.

**The primary can come out negative and that is the point.** Mixing in-source
data for 80% of species moves the head's decision boundaries toward iNaturalist
statistics. Species with no in-source data do not share those statistics and
could be *damaged* by the mix. Nobody has checked, and a catalogue that grows by
adding species will always have some in that position. A negative result here
would be a reason not to adopt the mix that has nothing to do with evidential
standing.

---

# Addendum 2 — pre-registering the per-class balancing fix

Written before any balanced head was fitted.

`SOURCE_MIX_MIDDLE_FINDINGS.md` (M2) found that mixing in-source data for 80% of
species makes the reserved 20% **worse**, −0.0278 [−0.0423, −0.0154]. It named
per-class balancing as the untried fix, on the hypothesis that the damage is a
**row-count artifact**: mixed classes simply have more training rows than
reserved ones, so the head favours them.

**One thing to check first, and it may be the whole story.** The production head
in `eval/inat_fusion.py:build_heads` fits with `class_weight="balanced"`. The
analysis code these results come from — `analysis/domain_shift.py:fit`, inherited
by every source-mix arm — does **not**. If balancing removes the damage, then M2
measured an artifact of the analysis script rather than a property of the
production configuration, and `SOURCE_MIX_MIDDLE_FINDINGS.md` needs retracting in
place.

## Arms

All on the M2 split: 20% of species reserved, no iNaturalist row of theirs in
training, both heads scored on identical test rows.

| arm | fit |
|---|---|
| `P-full` / `mixed` | unweighted — reproduces M2 |
| `P-full-bal` / `mixed-bal` | `class_weight="balanced"` — equal total weight per class |
| `mixed-cap-k` | unweighted, in-source rows capped at `k = 10` per species |

The cap is a different lever from the weighting and is declared separately: it
limits how far the *distribution* of a mixed class moves, where balancing only
equalises how much each class contributes.

**Primary endpoint.** Δ species top-1 on the reserved species, mixed head minus
`P-full`, under matched fitting — `mixed-bal` against `P-full-bal`. Paired over
species, same bootstrap.

## The prediction, declared

**Balancing will reduce the damage but not remove it.** The hypothesised
mechanism in M2 is not row count. It is that mixed classes have training rows
drawn from the *test* distribution and reserved classes do not. Equal weight on a
class whose rows match the test distribution still wins more argmaxes than equal
weight on a class whose rows do not. If that reading is right, balancing helps at
the margin and the reserved species stay below baseline.

If instead balancing removes the damage entirely, the M2 mechanism is wrong, its
finding is an artifact of unweighted fitting, and it gets retracted in place —
which is the outcome this addendum exists to make checkable rather than
deniable.

---

# Addendum 3 — two heads, and the arm the mechanism actually implies

Written before either head was fitted. M3 exhausted the cheap levers: balancing
and capping each cut the damage to reserved species by about a third and neither
removed it.

## T1 — two heads, averaged

Both heads span the **full** label space, so no class is missing from either:

- **head P** — every Pl@ntNet training row, all species.
- **head i** — iNaturalist rows for the mixed species, and Pl@ntNet rows for the
  reserved species, which is the only data they have.

Final posterior is the mean of the two. Rationale: the reserved classes are
represented in both heads rather than competing against a specialist they have no
counterpart in.

**Declared prediction: this will not fix it either.** Averaging still lets the
mixed classes' improved scores compete in one argmax; it changes the weight on
the improvement, not its nature. If the damage is a *relative* effect — mixed
classes score higher on iNaturalist imagery in general, not only when they are
correct — then any combination rule that improves mixed classes without improving
reserved ones reproduces it at reduced size.

## T2 — per-class logit centring, which is what the mechanism implies

If the damage is that mixed classes' scores rise *everywhere* on iNaturalist
imagery rather than only where they are right, the fix is to remove each class's
general propensity to fire rather than to rearrange the heads.

For each class, compute its mean logit over a **reference pool of
iNaturalist-source photographs that are out-of-catalogue** — the existing
`near_ood` / `regional_ood` / `distant_ood` buckets — and subtract it. The pool
is unlabelled with respect to the 345 classes, is not part of the in-catalogue
test set, and is available at build time, so this leaks nothing.

Every class is centred by the same rule, including reserved ones. A class whose
scores inflated on iNaturalist imagery gets centred back by exactly that
inflation.

**Declared risk:** centring may remove the *legitimate* part of the in-source
gain along with the spurious part, so the mixed species' +0.067 could shrink
substantially. An arm that fixes the reserved species by giving back the gain is
not a fix, and the two endpoints will be read together.

**Primary endpoint** for both arms, as in M3: Δ species top-1 on reserved
species against the matched `P-full` baseline, paired over species; with the
mixed-species Δ reported beside it in every case.

---

# Addendum 4 — sweeping the reserved fraction

Written before the sweep was run. `SOURCE_MIX_MIDDLE_FINDINGS.md` names this as
the open item: 20% was declared, not derived, and it sets where the crossover
between T1 and the single mixed head falls. That crossover is now the decision
variable — the damage itself is understood.

**Sweep** `r ∈ {0.05, 0.10, 0.20, 0.35, 0.50}` of species reserved, everything
else held to the design already fitted: same species set, same `q < 0.4` test
rows, same arms (`P-full`, single `mixed`, `T1`, `T1-control`).

**Not a confound, but say it anyway.** As `r` grows the mixed pool shrinks, so
there is less in-source training data in total. That is not an artifact to
correct — a catalogue where half the species have no in-source data really does
have half the in-source data — but it means the sweep varies two things at once
and the mixed-species gain may fall with `r` for that reason alone.

## Endpoints

1. **Δ on reserved species**, per arm, against `P-full`.
2. **The architecture-matched data effect**, `T1 − T1-control`, which is the
   quantity that isolated the fix.
3. **Catalogue-mean accuracy** — macro over *all* species, which weights the two
   groups by their actual prevalence. **This is what locates the crossover**:
   the `r` at which T1 stops costing average accuracy and starts saving it.

## Predictions, declared

- **The damage grows as the reserved fraction shrinks.** A reserved species at
  `r = 0.05` competes against 95% of the field having improved; at `r = 0.50`
  only half the field has. Fewer reserved species should mean *more* damage each,
  not less — which is the uncomfortable direction, because it means the problem
  is worst exactly when it looks least important.
- **T1's reserved-species protection holds across the sweep**, since the
  mechanism it removes does not depend on `r`.
- **The crossover exists and is at a low `r`.** T1 gives up 3.3pp on mixed
  species to save 6.3pp on reserved at `r = 0.20`; those magnitudes imply the
  break-even sits somewhere near `r ≈ 0.3`, and if it lands far from that the
  reasoning above is wrong somewhere.
