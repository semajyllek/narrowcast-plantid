"""The student: pixels in, K+1 posteriors out, small enough to be the whole model.

Trained against the **teacher's posteriors over the user's labels**, not against
its embedding. That distinction is the reason `TINY_PREREG.md` reopens a question
`CLAUDE.md` closed twice: both closed attempts minimised cosine to a general
embedding, and `PRUNE_FINDINGS.md` records for the third time that cosine to a
teacher does not predict downstream accuracy.

The teacher's posteriors on the transfer set are **free**. They are
`clf.predict_proba` over embeddings this project already has cached, so no
teacher forward pass runs anywhere in this file. The GPU cost is entirely the
student's own training.

Architecture is a depthwise-separable stack -- the MobileNet idea, without the
parts that exist to serve 1,000 ImageNet classes. Measured at K=14:

    width 0.5    37,823 params    0.15 MB fp32    0.04 MB int8
    width 1.0   138,351 params    0.55 MB fp32    0.14 MB int8
    width 2.0   527,567 params    2.11 MB fp32    0.53 MB int8

against an encoder of 43.3 MB. The size claim this project cares about is the
int8 one, since that is what would ship. **Every width here is under the 1 MB
target**, so the pilot is not a search for an architecture that fits -- it is a
test of whether anything that fits also works.

Two things are deliberately *not* done, and both would flatter the result:

  - No teacher embeddings are given to the student as an auxiliary target. That
    is the closed experiment.
  - The transfer set is Pl@ntNet, the corpus the teacher's head was fitted on;
    evaluation is iNaturalist. Training the student on evaluation-source images
    would hand it the deployment distribution, which is the error
    `ADAPT_FINDINGS.md` caught reporting +0.17 where the truth was -0.04.
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from narrowcast import build as B

IMG = 128     # module default; overridden per-run by --img
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _sep(cin, cout, stride):
    """Depthwise 3x3 then pointwise 1x1 -- the separable block, ~8x cheaper than
    a dense 3x3 at these widths."""
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
        nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Student(nn.Module):
    def __init__(self, n_classes, width=1.0):
        super().__init__()
        c = [max(8, int(w * width)) for w in (16, 32, 64, 128, 256)]
        self.net = nn.Sequential(
            nn.Conv2d(3, c[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(c[0]), nn.ReLU(inplace=True),
            _sep(c[0], c[1], 2), _sep(c[1], c[2], 2),
            _sep(c[2], c[3], 2), _sep(c[3], c[3], 1),
            _sep(c[3], c[4], 2), _sep(c[4], c[4], 1),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Linear(c[4], n_classes)

    def forward(self, x):
        return self.fc(self.net(x))


class Frames(Dataset):
    """Images plus their teacher posteriors. Augmentation on the training side
    only -- the evaluation rows must be the same pixels the teacher saw."""

    def __init__(self, paths, targets=None, train=True, img=IMG):
        self.paths, self.targets, self.train = list(paths), targets, train
        self.img = img

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
            if self.train:
                # Scale/flip only. Colour jitter is left out on purpose: several
                # of these species are separated by flower colour, and jittering
                # it would teach the student to ignore the feature the task turns
                # on.
                w, h = im.size
                s = np.random.uniform(0.7, 1.0)
                nw, nh = int(w * s), int(h * s)
                x0 = np.random.randint(0, max(1, w - nw + 1))
                y0 = np.random.randint(0, max(1, h - nh + 1))
                im = im.crop((x0, y0, x0 + nw, y0 + nh))
                if np.random.rand() < 0.5:
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
            im = im.resize((self.img, self.img), Image.BILINEAR)
            x = torch.from_numpy(np.asarray(im, dtype=np.float32).copy() / 255.0)
        x = ((x.permute(2, 0, 1) - MEAN) / STD)
        if self.targets is None:
            return x
        return x, torch.from_numpy(self.targets[i])


def build_model(n_classes, width, init):
    """`scratch` -> the separable stack above; `imagenet` -> MobileNetV3-Small.

    Two arms answering different questions, reported separately and never merged.

    **scratch** is the strict test the prereg declares: can 138k parameters carry
    the *task alone*, given nothing but the teacher's posteriors? Nothing generic
    is inherited, so a pass is a statement about task-conditional distillation.

    **imagenet** is what a practitioner would actually build, and it is the arm
    `CNN_FINDINGS.md` used -- `mobilenet_v3_small` from ImageNet init reached
    0.623 top-1 on 87 species there, which is the pessimistic prior the prereg
    cites.

    **Measured at K=14 it is 1,533,231 parameters -- 1.53 MB int8, not the 2.54 MB
    this docstring first claimed.** ImageNet's 1,000-way classifier is ~1M of
    `mobilenet_v3_small`'s 2.54M, and replacing it with a 15-way layer deletes
    most of that. So the arm sits *just* over the 1 MB budget rather than
    comfortably past it.

    That matters for what can be concluded: **"no ImageNet-initialised model fits
    the budget" is not established here**, it is an artifact of only trying full
    width. `width_mult=0.35` has no torchvision weights, but a narrowed variant
    or a distilled-then-pruned one would plausibly land under 1 MB while keeping
    most of the gain. Treat this arm as a near-budget reference, not as proof of
    a floor.

    If the strict arm fails and this one passes, the deficit is initialisation
    rather than capacity -- a different finding from "a tiny model cannot do this".
    """
    if init == "scratch":
        return Student(n_classes, width)
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
    return m


def _device():
    return "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")


def size_report(model):
    n = sum(p.numel() for p in model.parameters())
    return n, n * 4 / 1e6, n / 1e6      # params, fp32 MB, int8 MB


def train_student(a, index, labels, clf, classes, vecs, eval_paths, truth, bucket,
                  cluster, group, out):
    from analysis.distil_task import P_OOD, build_frame

    dev = _device()
    # Transfer set: every catalogue image with a cached vector, so the teacher's
    # posterior is already known. Background images included -- the student has to
    # learn to say __OTHER__, and a transfer set of only in-list species would
    # never show it one.
    tr = index[index.split == "train"]
    tr = tr[[i in vecs for i in tr.image_id]]
    if a.limit:
        tr = tr.sample(n=min(a.limit, len(tr)), random_state=0)
    X = np.stack([vecs[i] for i in tr.image_id])
    soft = clf.predict_proba(X).astype("float32")
    print(f"  transfer set {len(tr)} images, teacher posteriors precomputed "
          f"({soft.shape[1]} classes)", flush=True)

    model = build_model(len(classes), a.width, a.init).to(dev)
    n, mb32, mb8 = size_report(model)
    over = "  [OVER 1 MB BUDGET]" if mb8 > 1.0 else ""
    print(f"  student [{a.init}] {n:,} params — {mb32:.2f} MB fp32, "
          f"{mb8:.2f} MB int8{over}", flush=True)

    dl = DataLoader(Frames(tr.path.to_numpy(), soft, train=True, img=a.img),
                    batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    drop_last=True, persistent_workers=a.workers > 0)

    # `--steps` fixes the number of gradient updates and derives the epoch count
    # from the transfer-set size. Sweeping `--limit` at fixed *epochs* would vary
    # data and compute together -- 30k images for 60 epochs is 15x the updates of
    # 2k for 60 -- so any curve it produced would answer neither question. The
    # quantity of interest is how much unlabelled data a student needs, holding
    # training effort constant.
    epochs = a.epochs
    if a.steps:
        epochs = max(1, -(-a.steps // max(1, len(dl))))
        print(f"  {a.steps} gradient steps over {len(tr)} images "
              f"-> {epochs} epochs", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=max(1, epochs * len(dl)))

    model.train()
    for ep in range(epochs):
        t0, tot, seen = time.time(), 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            # KL to the teacher's posterior, temperature-scaled. T**2 keeps the
            # gradient magnitude comparable across temperatures (Hinton et al.);
            # without it, raising T quietly lowers the learning rate.
            logp = F.log_softmax(model(xb) / a.temp, dim=1)
            tgt = F.softmax(torch.log(yb.clamp_min(1e-8)) / a.temp, dim=1)
            loss = F.kl_div(logp, tgt, reduction="batchmean") * (a.temp ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(xb); seen += len(xb)
        print(f"  epoch {ep + 1}/{epochs}  kl {tot / max(seen, 1):.4f}  "
              f"{time.time() - t0:.0f}s", flush=True)

    # Score on the identical evaluation rows the teacher was scored on, through
    # the identical path. `Precomputed` replays these posteriors into
    # `score_frame`, so nothing about the cascade or the bootstrap differs.
    model.eval()
    ev = DataLoader(Frames(eval_paths, None, train=False, img=a.img), batch_size=a.batch,
                    num_workers=a.workers)
    probs = []
    with torch.no_grad():
        for xb in ev:
            probs.append(F.softmax(model(xb.to(dev)), dim=1).float().cpu().numpy())
    probs = np.vstack(probs)

    frame = build_frame(classes, probs, truth, bucket, cluster, group)
    m = B.fit_and_measure(frame, p_ood=P_OOD)
    # A student that declines everything has **no** precision -- the mean over
    # zero answered rows -- and `fit_and_measure` correctly returns None rather
    # than 0.0. That is a real outcome for an undertrained student, not an error,
    # so it prints as "—" instead of crashing the run that produced it.
    def _f(v):
        return "—" if v is None else f"{v:.4f}"
    print(f"  {'student':24s} label_share {_f(m['label_share'])}  "
          f"coverage {_f(m['coverage'])}  precision {_f(m['precision'])}  "
          f"top1 {_f(m['closed_set_top1'])}  headroom {_f(m['headroom'])}",
          flush=True)
    if m["precision"] is None:
        print("  (student answered nothing — the cascade declined every row)",
              flush=True)

    t = out["teacher"]
    delta = ("—" if m["label_share"] is None or t["label_share"] is None
             else f"{m['label_share'] - t['label_share']:+.4f}")
    print(f"\n  label_share  teacher {_f(t['label_share'])} -> "
          f"student {_f(m['label_share'])}  ({delta})", flush=True)
    out["student"] = {k: m[k] for k in ("label_share", "coverage", "precision",
                                        "closed_set_top1", "headroom",
                                        "group_share", "decline_share")}
    out["student"].update(params=n, mb_fp32=round(mb32, 3), mb_int8=round(mb8, 3),
                          epochs=epochs, width=a.width, img=a.img,
                          init=a.init, transfer_images=len(tr))
    Path(a.out).write_text(__import__("json").dumps(out, indent=2))
    print(f"  wrote {a.out}", flush=True)
    return 0
