# PlantCLEF2024 minds the resampling filter more than it minds the plant

Found while porting the Oregon bundle to iOS. Not looked for, and it changes how
a deployment should be validated.

## The measurement

618 Oregon photographs, the shipped `plantclef24_q8_per_channel.mlpackage`,
identical resize target (short side 518, centre-crop 518) and identical crop
arithmetic. **Only the resampling filter differs.**

| pair | mean pixel Δ (0–255) | embedding cosine |
|---|---|---|
| PIL bicubic vs PIL **bilinear** | 2.69 | 0.9959 |
| PIL bicubic vs PIL **hamming** | 3.55 | 0.9524 |
| PIL bicubic vs PIL **lanczos** | 1.76 | **0.9218** |
| PIL bicubic vs **CoreGraphics** `.high` | 3.27 | **0.8753** |

A 1.76/255 mean pixel difference — invisible to a person — moves the embedding to
cosine 0.92. For comparison, the same model on the **same pixels** across Core ML
backends (CPU vs ANE) sits at 0.9984.

**So the scaler matters ~50× more than the compute backend.** That ordering is
the opposite of what `ONDEVICE_FINDINGS.md` and `ENCODER_DECISION.md` prepared us
for: those record backend hazards (int4 on the Metal GPU) and validate an export
by cosine against torch. Neither varies the resampling.

## What it costs, end to end

Running the full cascade over all 618, with the head fitted on torchvision-bicubic
vectors:

| preprocessing | names a species | correct when it does | overall correct |
|---|---|---|---|
| torchvision bicubic (what the head saw) | 93.4% | 99.8% | **93.2%** |
| CoreGraphics `.high` (what the app does) | 90.6% | **100.0%** | 90.6% |

**−2.6 points of correct-species rate, +0.2 of precision.** The mismatch makes the
model more cautious rather than more wrong, which is the benign direction — but it
is a real loss and it is invisible without this measurement.

## Why it is probably the resolution

PlantCLEF2024 is a ViT-B/14 at **518 px**: 1,369 patch tokens. At that resolution
a patch is 14 px of a 518 px image, so high-frequency detail — exactly what
separates one interpolation kernel from another — lands inside individual tokens
rather than being averaged away. Untested prediction: MobileCLIP2-S2 at 256 px
should be markedly less sensitive, and that is checkable with the caches already
on disk.

## Consequences for deployment

1. **An embedding reference cannot be generated on a different machine.** iOS and
   macOS CoreGraphics differ enough that a *correct* iOS pipeline scores cosine
   0.984 and 0.926 against a Mac-generated reference for two of six photographs.
   The iOS app's parity test was rewritten around synthetic flat-colour images,
   where orientation and cropping are exactly checkable and no tolerance is needed.
2. **Validate geometry, not vectors, across platforms.** A vertically flipped
   photograph scores cosine 0.976 against its correct embedding on one of these
   images — *higher* than a correct pipeline on the wrong platform. Any
   cosine gate tuned to pass real cross-platform variation is blind to a flip.
   This is not hypothetical: the iOS port shipped with a vertical flip, and the
   classification test passed throughout.
3. **The principled fix is to fit the head on deployment-preprocessed vectors.**
   That recovers the 2.6 points and is this project's own argument for narrow
   models — fitting locally means fitting on data that resembles deployment
   (`NARROW_THESIS_FINDINGS.md`). Not done; the vectors exist.

## Measured on the device

*Added after running all 618 photographs on an iPad (A16), 157 ms each,
108 seconds total.*

| preprocessing | names a species | correct when it does | overall correct |
|---|---|---|---|
| torchvision bicubic (head fitted on this) | 93.4% | 99.8% | **93.2%** |
| macOS CoreGraphics | 90.6% | 100.0% | 90.6% |
| **iOS CoreGraphics — the deployed pipeline** | **91.4%** | **100.0%** | **91.4%** |

**The deployed cost is 1.8 points, not the 2.6 the Mac predicted.** iOS's scaler
lands slightly closer to torchvision's bicubic than macOS's does, so the earlier
figure was pessimistic — which is the right direction for a lower bound to be
wrong in, and the reason it was labelled one.

**Precision on the device is 100.0%: 565 species answers, none of them wrong.**
The preprocessing mismatch is spent entirely on declining more (8.6% against
6.6%), never on answering wrongly. Nothing was answered at the family rank at
all, which matches the bundle's measured `group_share` of ~0 — this label set has
one species per family almost everywhere, so there is no coarse rank to retreat
to.

That settles the open question in favour of leaving it alone: re-fitting the head
on device-preprocessed vectors would recover under two points of coverage and
cannot improve a precision that is already 1.000.

## What is still not measured

The 618 photographs are GBIF field images, which is the corpus everything here is
drawn from. **A phone camera in one's own hands remains untested** — the axis
`DOMAIN_SHIFT_FINDINGS.md` calls the last one. The app is now the cheapest
instrument for measuring it.
