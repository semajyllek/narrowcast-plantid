# Cached competitor responses — local only, deliberately

`COMPETITIVE_FINDINGS.md` is the only measurement in these repos that compares
this work against incumbent systems **on identical photographs** —
`bioclip2_cml4` at 0.7720 species top-1 against iNaturalist's 108k-taxa server
model at 0.7871, paired −0.015 [−0.060, +0.030], and ahead at genus 0.974 against
0.914. It rests on 1,394 cached API responses: 465 Pl@ntNet, 465 iNaturalist,
464 iNaturalist+geo, each a ranked list of `{name, rank, score}`.

**Those responses are not in this repository.** They were committed here briefly
and removed: they are a third party's model outputs, this repo is public, and
whether Pl@ntNet's and iNaturalist's terms permit redistributing them has not
been checked. Publishing them is easy to do later and hard to undo, so the
default is not to. They contain no credentials, no image data and no personal
data — the question is redistribution rights, not sensitivity.

## Where they live

| path | what | tracked? |
|---|---|---|
| `data/processed/headtohead/` | the 1,394 JSON responses, live | no (gitignored) |
| `data/processed/headtohead.parquet` | the joined table the analysis reads | no |
| `data/processed/competitor_cache/` | packed backup of both, 340 KB | no |

`data/processed/competitor_cache/` exists because the live cache sits under a
gitignored tree with 13 GB of regenerable data around it, and one careless
cleanup would take the only irreplaceable thing in it. Restore with:

```bash
tar xzf data/processed/competitor_cache/headtohead_responses.tar.gz -C data/processed
cp data/processed/competitor_cache/headtohead.parquet data/processed/
```

Then re-scoring costs **no API quota**:

```bash
PYTHONPATH=. .venv/bin/python -m analysis.compare_h2h
PYTHONPATH=. .venv/bin/python -m analysis.score_bc2_cml4
```

## If you do want them published

Check the terms first. If they permit it, the packed archive is 248 KB and can be
committed as-is — the reproducibility argument for doing so is real, since a
re-fetch scores whatever those services run *today*, which is a different model
than the one `COMPETITIVE_FINDINGS.md` describes.
